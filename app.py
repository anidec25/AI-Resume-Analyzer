import os
import streamlit as st
from dotenv import load_dotenv  # Import dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
import PyPDF2
import docx
import base64
import gc

# Set page title
st.set_page_config(page_title="AI Resume Assistant", page_icon="📄")

# Title of the app
st.title("📄 AI Resume Assistant")


resume_text = ""
faiss_index = None


def display_pdf(file_bytes: bytes, file_name: str):
    """Displays the uploaded PDF in an iframe."""
    base64_pdf = base64.b64encode(file_bytes).decode("utf-8")
    pdf_display = f"""
    <iframe 
        src="data:application/pdf;base64,{base64_pdf}" 
        width="100%" 
        height="300px" 
        type="application/pdf"
    >
    </iframe>
    """
    st.markdown(f"### Preview of {file_name}")
    st.markdown(pdf_display, unsafe_allow_html=True)

def reset_chat():
    st.session_state.messages = []
    gc.collect()

# Function to extract text from resume
def extract_text_from_resume(file):
    """Extracts text from a resume file."""
    if file.type == "application/pdf":
        reader = PyPDF2.PdfReader(file)
        text = "".join(page.extract_text() or "" for page in reader.pages)
    elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        doc = docx.Document(file)
        text = "\n".join([para.text for para in doc.paragraphs])
    elif file.type == "text/plain":
        text = file.read().decode("utf-8")
    else:
        return "Unsupported file type."
    return text

# Function to create FAISS index
def create_faiss_index(text):
    """Creates a FAISS index from resume text."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_text(text)
    vector_store = FAISS.from_texts(chunks, embeddings)  # Use embeddings here
    return vector_store

with st.sidebar:

    # Sidebar for API key and resume upload
    st.sidebar.title("📄 Upload Your Resume")
    api_key = st.sidebar.text_input("Enter your OpenAI API Key:", type="password")  # New input for API key
    uploaded_file = st.sidebar.file_uploader("Upload your resume:", type=["pdf", "docx", "txt"], help="Supported formats: PDF, DOCX, TXT")

    # Check if the API key is loaded correctly
    if api_key:
        chat_model = ChatOpenAI(openai_api_key=api_key)
        embeddings = OpenAIEmbeddings(openai_api_key=api_key)  # Initialize embeddings here
    else:
        st.error("API key not found. Please enter your API key in the sidebar.")


    # Load resume text and create FAISS index
    if uploaded_file:
        resume_text = extract_text_from_resume(uploaded_file)
        faiss_index = create_faiss_index(resume_text)
        faiss_index.save_local("faiss_index")
        st.sidebar.success("✅ Resume uploaded and processed successfully!")

        # Optionally display the PDF in the sidebar
        display_pdf(uploaded_file.getvalue(), uploaded_file.name)

    
    st.button("Clear Chat", on_click=reset_chat)



# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Layout
tab1, tab2, tab3 = st.tabs(["💬 Chat", "📊 Analysis", "🚀 Career Guidance"])

# Chat Tab with RAG
with tab1:
    st.subheader("💬 Chat with AI about Your Resume")
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    def process_input():
        user_input = st.session_state.get("user_input", "").strip()
        if user_input and faiss_index:
            relevant_docs = faiss_index.similarity_search(user_input, k=3)
            retrieved_text = "".join([doc.page_content for doc in relevant_docs])
            prompt = f"Based on the following resume content:\n{retrieved_text}\n\nAnswer the following question: {user_input}"
            response = chat_model.predict(prompt)
            st.session_state.messages.append({"role": "user", "content": user_input})
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.session_state.user_input = ""  # Clear input field

    user_input = st.text_input("🔍 Ask something about your resume:", key="user_input", help="Example: 'What skills should I improve?' or 'What job roles fit my experience?'", on_change=process_input)


generate_response = st.button("💡 Generate Response")

if generate_response:
    if user_input and faiss_index:
        relevant_docs = faiss_index.similarity_search(user_input, k=3)
        retrieved_text = "\n".join([doc.page_content for doc in relevant_docs])
        prompt = f"Based on the following resume content:\n{retrieved_text}\n\nAnswer the following question: {user_input}"
        response = chat_model.predict(prompt)
        st.session_state.messages.append({"role": "user", "content": user_input})
        st.session_state.messages.append({"role": "assistant", "content": response})
    elif not uploaded_file:
        st.warning("⚠️ Please upload a resume first.")
    else:
        st.warning("⚠️ Please enter a question.")

# Analysis Tab
with tab2:
    st.subheader("📊 Resume Analysis")
    if st.button("🔍 Analyze Resume"):
        if faiss_index:
            analysis_prompt = f"Analyze the following resume in detail:\n{resume_text}\n\nProvide a structured analysis covering: \n- Strengths of the resume.\n- Areas for improvement (clarity, formatting, skills, experience).\n- Missing key elements that recruiters look for.\n- Suggestions for better wording and structure."
            analysis_response = chat_model.predict(analysis_prompt)
            st.write("### Detailed Resume Analysis:")
            st.write(analysis_response)
        else:
            st.warning("⚠️ Please upload a resume first.")

# Career Guidance Tab
with tab3:
    st.subheader("🚀 Career Guidance")
    if st.button("📈 Get Career Advice"):
        if faiss_index:
            career_prompt = f"Based on the following resume, provide career guidance:\n{resume_text}\n\nInclude:\n- Recommended career paths based on the skills and experience listed.\n- Skills or certifications that could enhance job opportunities.\n- Suggestions for potential job roles and industries that fit the candidate's profile.\n- Networking and job-hunting strategies relevant to the candidate's field."
            career_response = chat_model.predict(career_prompt)
            st.write("### Personalized Career Guidance:")
            st.write(career_response)
        else:
            st.warning("⚠️ Please upload a resume first.")