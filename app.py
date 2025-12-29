import streamlit as st
from agent.pipeline import run_agent

st.title("AURA – Multimodal AI Analyst")

query = st.text_input("Ask AURA:", "Why does my model perform poorly?")

dataset = st.file_uploader("Upload CSV for ML (optional)", type=["csv"])
image = st.file_uploader("Upload Image (optional)", type=["png","jpg","jpeg"])

if st.button("Run"):
    ds_path = None
    img_path = None

    if dataset:
        ds_path = "temp.csv"
        with open(ds_path,"wb") as f: f.write(dataset.getbuffer())

    if image:
        img_path = "temp.jpg"
        with open(img_path,"wb") as f: f.write(image.getbuffer())

    answer, docs = run_agent(query, dataset=ds_path, image=img_path)

    st.subheader("AURA's Answer:")
    st.write(answer)

    st.subheader("Top Documents Retrieved:")
    for d in docs:
        st.write(d["path"])
        st.write(d["text"][:500] + "...")
