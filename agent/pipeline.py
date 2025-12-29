from rag.search import search
from ml.train import train_model
from vision.caption import caption_image
from transformers import pipeline

llm = pipeline("text2text-generation", model="google/flan-t5-base")

def run_agent(query, dataset=None, image=None):
    context = ""

    if dataset:
        model, cols, acc, f1 = train_model(dataset)
        context += f"ML model trained. Accuracy={acc}, F1={f1}\n"

    if image:
        cap = caption_image(image)
        context += f"Image caption: {cap}\n"

    docs = search(query, k=2)
    for d in docs:
        context += f"\nDOC: {d['path']} -> {d['text'][:300]}..."

    prompt = f"""
    {context}

    User question: {query}
    Provide a concise, analytical answer:
    """

    answer = llm(prompt, max_new_tokens=200)[0]["generated_text"]
    return answer, docs
