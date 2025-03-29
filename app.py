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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json
import numpy as np
import requests
from bs4 import BeautifulSoup
import re

from src.utils.file_handlers import display_pdf, extract_text_from_resume
from src.utils.text_processors import create_faiss_index
from src.components import chat, analysis, career_guidance

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "job_url" not in st.session_state:
    st.session_state.job_url = ""
if "job_description" not in st.session_state:
    st.session_state.job_description = ""

def reset_chat():
    st.session_state.messages = []
    gc.collect()

def main():
    st.set_page_config(page_title="Nemo", page_icon="🐠")
    st.title("🐠 Nemo, Welcomes you!")
    st.subheader("Finding your best resume")

    # Initialize these variables with None
    chat_model = None
    embeddings = None
    resume_text = ""
    faiss_index = None

    with st.sidebar:
        st.sidebar.title("📄 Upload Your Resume")
        api_key = st.sidebar.text_input("Enter your OpenAI API Key:", type="password")
        uploaded_file = st.sidebar.file_uploader(
            "Upload your resume:",
            type=["pdf", "docx", "txt"],
            help="Supported formats: PDF, DOCX, TXT"
        )

        # Add a divider in sidebar
        if uploaded_file:
            st.sidebar.subheader("Resume Preview")
            # Display PDF in sidebar with custom styling for smaller view
            with st.sidebar:
                display_pdf(uploaded_file.getvalue(), uploaded_file.name)

        # Move clear chat button to bottom of sidebar
        st.sidebar.markdown("---")
        st.sidebar.button("Clear Chat", on_click=reset_chat)

    # Show welcome message and instructions before API key is entered
    if not api_key:
        st.markdown("""
        #### 👋 Welcome to Nemo, Your AI Powered Resume Assistant!
        
        #### To get started:
        1. Enter your OpenAI API key in the sidebar
        2. Upload your resume
        3. Use our tools to analyze and improve your resume
        
        #### Features:
        - 💬 Chat with AI about your resume
        - 📊 Get detailed resume analysis
        - 🎯 Compare with job descriptions
        - 🚀 Receive career guidance
        
        > **Note**: An OpenAI API key is required to use this application. You can get one at [OpenAI's website](https://platform.openai.com/api-keys).
        """)
        st.sidebar.info("👆 Please enter your OpenAI API key to get started!")
        return

    # Validate API key
    api_valid = False
    with st.spinner("Validating API key..."):
        try:
            # First try to create a simple chat completion to validate the API key
            chat_model = ChatOpenAI(openai_api_key=api_key, temperature=0)
            # Test the API key with a simple request
            chat_model.invoke("test")
            embeddings = OpenAIEmbeddings(openai_api_key=api_key)
            api_valid = True
        except Exception as e:
            error_message = str(e)
            st.sidebar.error("Invalid API Key", icon="⚠️")
            st.markdown("""
            ### ⚠️ API Key Error
            
            Your API key appears to be invalid. Please check:
            - You've copied the entire API key correctly
            - There are no extra spaces
            - You're using a valid OpenAI API key
            
            Need a valid API key? [Get one here](https://platform.openai.com/api-keys)
            """)
            return

    if not api_valid:
        return

    if uploaded_file and embeddings:
        resume_text = extract_text_from_resume(uploaded_file)
        faiss_index = create_faiss_index(resume_text, embeddings)
        faiss_index.save_local("faiss_index")
        st.sidebar.success("✅ Resume uploaded and processed successfully!")

    # Render tabs with larger font size using custom CSS
    st.markdown("""
        <style>
        .stTab {
            font-size: 1.2rem !important;
        }
        button[data-baseweb="tab"] {
            font-size: 1.2rem !important;
        }
        button[data-baseweb="tab"] p {
            font-size: 1.2rem !important;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Render tabs with emojis and text
    tab1, tab2, tab3 = st.tabs([
        "💬  Chat", 
        "📊  Analysis", 
        "🚀  Career Guidance"
    ])

    with tab1:
        chat.render_chat_tab(faiss_index, chat_model, resume_text)
    
    with tab2:
        analysis.render_analysis_tab(resume_text, uploaded_file)
    
    with tab3:
        career_guidance.render_career_guidance_tab(faiss_index, chat_model, resume_text)

if __name__ == "__main__":
    main()