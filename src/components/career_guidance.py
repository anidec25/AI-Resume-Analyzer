import streamlit as st

def render_career_guidance_tab(faiss_index, chat_model, resume_text):
    """Render the career guidance tab interface."""
    st.subheader("🚀 Career Guidance")
    
    if st.button("📈 Get Career Advice"):
        if faiss_index:
            with st.spinner("Analyzing your career path..."):
                career_prompt = f"""Based on the following resume, provide career guidance:
                {resume_text}

                Include:
                - Recommended career paths based on the skills and experience listed
                - Skills or certifications that could enhance job opportunities
                - Suggestions for potential job roles and industries that fit the candidate's profile
                - Networking and job-hunting strategies relevant to the candidate's field
                """
                career_response = chat_model.predict(career_prompt)
                st.write("### Personalized Career Guidance:")
                st.write(career_response)
        else:
            st.warning("⚠️ Please upload a resume first.") 