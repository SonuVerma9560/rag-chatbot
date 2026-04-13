from pypdf import PdfReader
import faiss
import numpy as np
from groq import Groq
import os

# 🔑 PUT YOUR KEY HERE (IMPORTANT)



client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# -----------------------------
# 1. LOAD PDF
# -----------------------------
def load_pdf(file_path):
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text()
    return text


# -----------------------------
# 2. SPLIT TEXT
# -----------------------------
def split_text(text, chunk_size=500, overlap=50):
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunks.append(text[i:i + chunk_size])
    return chunks


# -----------------------------
# 3. REAL EMBEDDINGS (FREE)
# -----------------------------
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

def get_embeddings(texts):
    return model.encode(texts).tolist()


# -----------------------------
# 4. FAISS INDEX
# -----------------------------
def create_faiss_index(embeddings):
    dimension = len(embeddings[0])
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings).astype("float32"))
    return index


# -----------------------------
# 5. SEARCH
# -----------------------------
def search(query, index, texts, k=3):
    query_embedding = model.encode([query])
    query_vector = np.array(query_embedding).astype("float32")

    distances, indices = index.search(query_vector, k)
    return [texts[i] for i in indices[0]]


# -----------------------------
# 6. LLM (GROQ)
# -----------------------------
def ask_llm(context, question):
    prompt = f"""
Answer ONLY from the context below.

Context:
{context}

Question:
{question}
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content