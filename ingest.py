import os
import glob
import pandas as pd
from dotenv import load_dotenv
import chromadb
from embedding_client import EmbeddingClient

load_dotenv()

# Ініціалізація
emb_client = EmbeddingClient()
chroma_client = chromadb.PersistentClient(path="./chroma")
collection = chroma_client.get_or_create_collection(name="real_estate_corpus")

def chunk_text(text: str, *, chunk_size: int = 200, overlap: int = 40) -> list[str]:
    """Розбиття тексту на чанки з перекриттям по словах"""
    if overlap >= chunk_size:
        raise ValueError(f"overlap ({overlap}) must be smaller than chunk_size ({chunk_size})")
    words = text.split()
    if not words:
        return []
    chunks = []
    start = 0
    step = chunk_size - overlap
    while start < len(words):
        chunks.append(" ".join(words[start : start + chunk_size]))
        if start + chunk_size >= len(words):
            break
        start += step
    return chunks

print("=== Початок індексації даних (Excel + Text Corpus) ===")

# 1. Індексація текстових файлів із папки corpus/
files = glob.glob(os.path.join("corpus", "*.txt"))
for filepath in files:
    filename = os.path.basename(filepath)
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()

    chunks = chunk_text(text, chunk_size=150, overlap=30)
    for i, chunk in enumerate(chunks):
        chunk_id = f"txt_{filename}_chunk_{i}"
        vector = emb_client.embed(chunk, task_type="RETRIEVAL_DOCUMENT")
        collection.upsert(
            ids=[chunk_id],
            embeddings=[vector],
            documents=[chunk],
            metadatas=[{"source": filename, "type": "corpus_text", "chunk_index": i}]
        )
    print(f"Проіндексовано текст: {filename} ({len(chunks)} чанків)")

# 2. Індексація структурованого Excel-файлу (Complexes + Listings)
# Оскільки файл у папці corpus, вказуємо правильний шлях:
excel_path = os.path.join("corpus", "lviv_real_estate.xlsx")

if os.path.exists(excel_path):
    print(f"Знайдено Excel-файл у папці corpus. Обробка...")
    df_complexes = pd.read_excel(excel_path, sheet_name="Complexes")
    df_listings = pd.read_excel(excel_path, sheet_name="Listings")
    
    # Об'єднуємо таблиці через реляційний зв'язок complex_id
    merged_df = pd.merge(df_listings, df_complexes, on="complex_id", how="left")
    
    for _, row in merged_df.iterrows():
        record_text = (
            f"Оголошення {row['listing_id']}: Продаж {row['rooms']}-кімнатної квартири "
            f"площею {row['area_m2']} кв.м, поверх {row['floor']}, адреса: {row['address']}. "
            f"Ціна: {row['price_usd']} USD ({row['price_uah']} грн). "
            f"ЖК: {row['complex_name']} ({row['class']} клас, забудовник {row['developer']}). "
            f"Район: {row['district']}. Опис: {row['description']}"
        )
        
        chunks = chunk_text(record_text, chunk_size=200, overlap=20)
        for i, chunk in enumerate(chunks):
            chunk_id = f"excel_{row['listing_id']}_c{i}"
            vector = emb_client.embed(chunk, task_type="RETRIEVAL_DOCUMENT")
            collection.upsert(
                ids=[chunk_id],
                embeddings=[vector],
                documents=[chunk],
                metadatas={
                    "source": "lviv_real_estate.xlsx",
                    "type": "apartment_listing",
                    "district": str(row['district']),
                    "complex": str(row['complex_name']),
                    "price_usd": int(row['price_usd'])
                }
            )
    print(f"Проіндексовано Excel-датасет: {len(merged_df)} об'єктів зв'язано та закинуто в базу.")
else:
    print(f"ПОМИЛКА: Файл не знайдено за шляхом {excel_path}!")

print(f"\nУспішно! Загальна кількість чанків у ChromaDB: {collection.count()}")