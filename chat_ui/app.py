import os
import sys
import time
import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai import types

# Додаємо шлях до кореня проєкту
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from vector_store import VectorStore

load_dotenv()

# Налаштування сторінки
st.set_page_config(
    page_title="Lviv Real Estate AI Assistant",
    page_icon="🏢",
    layout="centered"
)

# ---- КАСТОМНИЙ ПРЕМІАЛЬНИЙ ЛАКШЕРІ CSS ----
st.markdown("""
<style>
    /* Загальний фон та шрифти */
    .stApp {
        background-color: #0b0d10;
        color: #e2d9ce;
        font-family: 'Helvetica Neue', sans-serif;
    }
    
    /* Бічна панель (Sidebar) у глибокому темному лакшері стилі */
    [data-testid="stSidebar"] {
        background-color: #12161c;
        border-right: 1px solid #222933;
    }
    
    /* Брендинг у сайдбарі */
    .sidebar-brand {
        padding: 1.5rem 0rem 1rem 0rem;
        border-bottom: 1px solid #222933;
        margin-bottom: 1.5rem;
    }
    .brand-title {
        font-family: 'Georgia', serif;
        font-size: 1.3rem;
        font-weight: 500;
        letter-spacing: 1px;
        color: #f3e9dc;
    }
    .brand-subtitle {
        font-size: 0.75rem;
        letter-spacing: 2px;
        color: #c5a059;
        text-transform: uppercase;
        margin-top: 4px;
    }
    
    /* Преміальна картка статусу в сайдбарі */
    .luxury-status-card {
        background: linear-gradient(135deg, #171c24 0%, #1f2630 100%);
        border: 1px solid #c5a05933;
        border-radius: 8px;
        padding: 14px;
        margin-top: 1rem;
    }
    .status-indicator {
        display: inline-block;
        width: 7px;
        height: 7px;
        background-color: #c5a059;
        border-radius: 50%;
        margin-right: 6px;
        box-shadow: 0 0 8px #c5a05988;
    }
    
    /* Повідомлення чату (бульбашки) */
    .user-bubble {
        background: #191f28;
        border: 1px solid #2d3748;
        padding: 14px 18px;
        border-radius: 12px 12px 2px 12px;
        margin: 10px 0;
        color: #f3e9dc;
        font-size: 0.95rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }
    
    .bot-bubble {
        background: linear-gradient(135deg, #161922 0%, #1c212c 100%);
        border: 1px solid #c5a05944;
        padding: 16px 20px;
        border-radius: 12px 12px 12px 2px;
        margin: 10px 0;
        color: #e2d9ce;
        font-size: 0.95rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.4);
    }
    
    /* Поле вводу */
    [data-testid="stChatInput"] textarea {
        background-color: #12161c !important;
        color: #f3e9dc !important;
        border: 1px solid #c5a05966 !important;
        border-radius: 8px !important;
    }
    [data-testid="stChatInput"] textarea:focus {
        border-color: #c5a059 !important;
        box-shadow: 0 0 10px #c5a05922 !important;
    }
    
    h1, h2, h3 {
        font-family: 'Georgia', serif !important;
        color: #f3e9dc !important;
    }
</style>
""", unsafe_allow_html=True)

# ---- ЕЛЕГАНТНИЙ САЙДБАР (SIDEBAR) ----
with st.sidebar:
    st.markdown("""
        <div class="sidebar-brand">
            <div class="brand-title">Lviv Estate</div>
            <div class="brand-subtitle">Excellence & Aesthetics</div>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<p style='color: #94a3b8; font-size: 0.85rem; line-height: 1.5;'>Елітний цифровий консьєрж-сервіс для підбору та аналізу новобудов Львова.</p>", unsafe_allow_html=True)
    
    # Статус бази у лакшері стилі
    try:
        store = VectorStore(name="real_estate_corpus")
        chunk_count = store.count()
        st.markdown(f"""
            <div class="luxury-status-card">
                <div style="font-size: 0.8rem; color: #c5a059; margin-bottom: 4px; font-weight: 500;">
                    <span class="status-indicator"></span>БАЗА ЗНАНЬ АКТИВНА
                </div>
                <div style="font-size: 0.85rem; color: #94a3b8;">Індексовано чанків: <b style="color: #f3e9dc;">{chunk_count}</b></div>
            </div>
        """, unsafe_allow_html=True)
    except Exception:
        st.markdown("<p style='color: #e53e3e; font-size: 0.85rem;'>Помилка підключення до бази</p>", unsafe_allow_html=True)
        
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("<p style='color: #64748b; font-size: 0.7rem; letter-spacing: 1px;'>LUXURY REAL ESTATE RAG v2.0</p>", unsafe_allow_html=True)

# ---- ОСНОВНИЙ ЕКРАН ----
st.markdown("<h1 style='font-weight: 400; margin-bottom: 0px;'>Персональний помічник</h1>", unsafe_allow_html=True)
st.markdown("<p style='color: #c5a059; font-size: 0.95rem; letter-spacing: 0.5px; margin-bottom: 2rem;'>Елітний підбір нерухомості на базі штучного інтелекту</p>", unsafe_allow_html=True)

# Ініціалізація клієнта Gemini
@st.cache_resource
def init_rag():
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    store = VectorStore(name="real_estate_corpus")
    return client, store

client, store = init_rag()

SYSTEM_INSTRUCTION = (
    "Ти — розумний преміальний помічник із нерухомості у Львові. "
    "Використовуй НАДАНІ ФРАГМЕНТИ ДЖЕРЕЛ для відповіді на запитання користувача. "
    "Кожна відповідь повинна містити посилання на джерело (назву файлу). "
    "Якщо в наданих фрагментах немає повної інформації для відповіді, чітко повідом: "
    "'У наданих джерелах немає інформації для відповіді на це питання.' "
    "Не вигадуй жодних фактів від себе."
)

# Історія чату
if "messages" not in st.session_state:
    st.session_state.messages = []

# Виведення історії у вигляді гарних блоків
for message in st.session_state.messages:
    if message["role"] == "user":
        st.markdown(f"""
        <div style="display: flex; justify-content: flex-end;">
            <div style="max-width: 80%;">
                <div style="font-size: 0.7rem; color: #64748b; text-align: right; margin-bottom: 2px; letter-spacing: 0.5px;">ГІСТЬ</div>
                <div class="user-bubble">{message["content"]}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="display: flex; justify-content: flex-start;">
            <div style="max-width: 85%;">
                <div style="font-size: 0.7rem; color: #c5a059; margin-bottom: 2px; letter-spacing: 0.5px;">LVIV ESTATE CONCIERGE</div>
                <div class="bot-bubble">{message["content"]}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ---- ПОЛЕ ВВЕДЕННЯ ----
if prompt := st.chat_input("Задайте запитання щодо нерухомості..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.markdown(f"""
    <div style="display: flex; justify-content: flex-end;">
        <div style="max-width: 80%;">
            <div style="font-size: 0.7rem; color: #64748b; text-align: right; margin-bottom: 2px; letter-spacing: 0.5px;">ГІСТЬ</div>
            <div class="user-bubble">{prompt}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    with st.spinner("Аналізуємо базу даних..."):
        try:
            # Збільшено n_results до 5 для гарантованого захоплення Excel-чанків
            search_results = store.search(prompt, n_results=5)
            sources_text = ""
            retrieved_sources = set()
            
            for idx, row in enumerate(search_results, 1):
                meta = row["metadata"]
                source_name = meta.get("source", "невідомо")
                retrieved_sources.add(source_name)
                sources_text += f"[{idx}] (Джерело: {source_name})\n{row['chunk']}\n\n"
                
            rag_prompt = f"ДЖЕРЕЛА:\n{sources_text}\nПИТАННЯ КОРИСТУВАЧА:\n{prompt}"
            
            # --- БЕЗПЕЧНИЙ ВИКЛИК З АВТОМАТИЧНИМ ПОВТОРОМ ПРИ ПОМИЛКАХ 503/429 ---
            response = None
            for attempt in range(1, 6):
                try:
                    response = client.models.generate_content(
                        model="gemini-3.5-flash",
                        contents=rag_prompt,
                        config=types.GenerateContentConfig(
                            system_instruction=SYSTEM_INSTRUCTION,
                            temperature=0.1
                        )
                    )
                    break
                except Exception as api_err:
                    err_str = str(api_err)
                    if "503" in err_str or "429" in err_str or "Unavailable" in err_str:
                        if attempt == 3:
                            raise api_err
                        time.sleep(attempt * 3)
                    else:
                        raise api_err
            # -------------------------------------------------------------
            
            answer = response.text.strip()
            sources_str = f"<br><br><span style='font-size: 0.75rem; color: #94a3b8;'>📁 <i>Джерела: {list(retrieved_sources)}</i></span>"
            full_response = answer + sources_str
            
            st.markdown(f"""
            <div style="display: flex; justify-content: flex-start;">
                <div style="max-width: 85%;">
                    <div style="font-size: 0.7rem; color: #c5a059; margin-bottom: 2px; letter-spacing: 0.5px;">LVIV ESTATE CONCIERGE</div>
                    <div class="bot-bubble">{full_response}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            
        except Exception as e:
            st.error(f"Сервери Google наразі перевантажені або сталася помилка. Спробуйте повторити запит за кілька секунд. (Деталі: {e})")