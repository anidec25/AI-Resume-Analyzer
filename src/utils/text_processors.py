from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS

def create_faiss_index(text, embeddings):
    """Creates a FAISS index from resume text."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_text(text)
    vector_store = FAISS.from_texts(chunks, embeddings)
    return vector_store 