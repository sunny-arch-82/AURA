import os
import json
import faiss
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader


def load_docs(doc_dir="data/docs"):
    corpus = []
    print("Scanning:", os.path.abspath(doc_dir))

    for f in os.listdir(doc_dir):
        path = os.path.join(doc_dir, f)
        print("Reading:", f)

        if f.lower().endswith(".txt"):
            text = open(path, "r", encoding="utf-8").read()
            if text.strip():
                corpus.append(text)

        elif f.lower().endswith(".pdf"):
            text = ""
            pdf = PdfReader(path)
            for page in pdf.pages:
                extracted = page.extract_text() or ""
                text += extracted + "\n"

            if text.strip():
                corpus.append(text)
            else:
                print("⚠️ PDF has no extractable text:", f)

        else:
            print("Skipping:", f)

    print("Documents loaded:", len(corpus))
    return corpus


def build_index():
    docs = load_docs()

    if not docs:
        print("❌ No valid documents found.")
        return

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    vectors = model.encode(docs)

    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(vectors)
    faiss.write_index(index, "rag/index.faiss")

    # Write JSON lines
    with open("rag/chunks.jsonl", "w", encoding="utf-8") as f:
        for d in docs:
            json.dump({"text": d}, f)
            f.write("\n")

    print("✅ Index and chunks saved.")


if __name__ == "__main__":
    build_index()
