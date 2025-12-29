import json
import faiss
from sentence_transformers import SentenceTransformer


def search(query, k=3):
    index = faiss.read_index("rag/index.faiss")

    with open("rag/chunks.jsonl", "r", encoding="utf-8") as f:
        docs = [json.loads(l)["text"] for l in f if l.strip()]

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    qvec = model.encode([query])
    distances, indices = index.search(qvec, k)

    results = []
    for idx in indices[0]:
        if idx < len(docs):
            results.append(docs[idx])

    return results
