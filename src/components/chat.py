import streamlit as st
from src.utils.text_processors import get_latest_experience

def render_chat_tab(faiss_index, chat_model, resume_text):
    """Render the chat tab interface."""
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
            st.session_state.user_input = ""

    st.text_input(
        "🔍 Ask something about your resume:",
        key="user_input",
        help="Example: 'What skills should I improve?' or 'What job roles fit my experience?'",
        on_change=process_input
    )

def process_experience_query(query, resume_text):
    """Process experience-related queries with improved accuracy."""
    if "latest" in query.lower() and "experience" in query.lower():
        return get_latest_experience(resume_text)
    # Add more experience-related query handlers as needed 