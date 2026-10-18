import os
import time
from dotenv import load_dotenv
from google import genai
from google.genai import errors, types
from vector_store import VectorStore

load_dotenv()

# Ініціалізуємо клієнт Gemini та наше векторне сховище
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
store = VectorStore(name="real_estate_corpus")

# Тестовий набір запитів для перевірки RAG-системи
TEST_QUERIES = [
    # Позитивні тести (інформація є в базі — тексти або Excel)
    ("Яка ціна за квадратний метр у ЖК Семицвіт?", True),
    ("Які пропозиції площ є в оголошеннях?", True),
    ("Який район у ЖК Avalon?", True),
    ("Скільки коштує квартира в доларах у ЖК Пасічний?", True),
    
    # Негативні тести (інформації немає в базі)
    ("Яка ціна квартири у ЖК Парус у Києві?", False),
    ("Чи є басейн на даху у ЖК Семицвіт?", False),
]

# Системна інструкція для RAG
SYSTEM_INSTRUCTION = (
    "Ти — розумний помічник із нерухомості у Львові. "
    "Використовуй НАДАНІ ФРАГМЕНТИ ДЖЕРЕЛ для відповіді на запитання користувача. "
    "Кожна відповідь повинна містити посилання на джерело. "
    "Якщо в наданих фрагментах немає повної інформації для відповіді, ти зобов'язаний чітко повідомити: "
    "'У наданих джерелах немає інформації для відповіді на це питання.' "
    "Не вигадуй жодних фактів від себе."
)

def run_rag_evaluation():
    print(f"Запуск тестування RAG-системи (у сховищі {store.count()} чанків)...\n")
    print("=" * 80)

    for i, (query, is_positive) in enumerate(TEST_QUERIES, 1):
        print(f"\n[Запит {i}] ({'Позитивний' if is_positive else 'Негативний'} тест)")
        print(f"Питання: {query}")

        # 1. Пошук релевантних фрагментів у базі ChromaDB
        search_results = store.search(query, n_results=5)

        # Формуємо контекст із знайдених фрагментів
        sources_text = ""
        retrieved_sources = set()
        for idx, row in enumerate(search_results, 1):
            meta = row["metadata"]
            source_name = meta.get("source", "невідомо")
            retrieved_sources.add(source_name)
            sources_text += f"[{idx}] (Джерело: {source_name})\n{row['chunk']}\n\n"

        # 2. Формуємо промпт для моделі
        prompt = (
            f"ДЖЕРЕЛА:\n{sources_text}\n"
            f"ПИТАННЯ КОРИСТУВАЧА:\n{query}"
        )

        # 3. Генерація відповіді через Gemini з обробкою ліміту (429)
        for attempt in range(1, 4):
            try:
                response = client.models.generate_content(
                    model="gemini-3.5-flash",
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM_INSTRUCTION,
                        temperature=0.1
                    )
                )
                print(f"Відповідь моделі:\n{response.text.strip()}")
                print(f"Використані джерела: {list(retrieved_sources)}")
                break
            except errors.ClientError as exc:
                if getattr(exc, "code", None) == 429:
                    print(f"  [Ліміт запитів вичерпано, чекаємо 15 секунд перед повтором...]")
                    time.sleep(15)
                else:
                    raise

        print("-" * 80)
        time.sleep(5)

if __name__ == "__main__":
    run_rag_evaluation()

