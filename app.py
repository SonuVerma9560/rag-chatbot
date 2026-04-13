import streamlit as st
from rag import *
import tempfile

st.title("📄 AI RAG Chatbot (Pro Version)")
st.write("Upload a PDF and ask questions from it.")

# Upload
uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

if uploaded_file is not None:
    # Save temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        file_path = tmp_file.name

    st.success("PDF uploaded successfully!")

    # Process
    text = load_pdf(file_path)
    chunks = split_text(text)
    embeddings = get_embeddings(chunks)
    index = create_faiss_index(embeddings)

    # Query
    query = st.text_input("Ask your question:")

    if query:
        results = search(query, index, chunks)
        context = "\n".join(results)

        answer = ask_llm(context, query)
        st.write("### Answer:")
        st.write(answer)