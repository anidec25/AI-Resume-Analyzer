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

# First, move all these functions to the top of the file, right after the imports and before any Streamlit code
def get_industry_keywords():
    """Returns industry-specific keywords."""
    return {
        "technology": [
            # Programming Languages
            "python", "java", "javascript", "typescript", "c++", "c#", "ruby", "go", "rust", "swift",
            # Web Development
            "react", "angular", "vue.js", "node.js", "html5", "css3", "rest api", "graphql", "webpack",
            # Cloud & DevOps
            "aws", "azure", "gcp", "docker", "kubernetes", "terraform", "jenkins", "ci/cd", "microservices",
            # Data & AI
            "machine learning", "deep learning", "tensorflow", "pytorch", "data science", "big data",
            "sql", "nosql", "mongodb", "postgresql", "data analytics", "artificial intelligence",
            # Methodologies & Tools
            "agile", "scrum", "jira", "git", "github", "devops", "test-driven development", "clean code",
            # Cybersecurity
            "cybersecurity", "penetration testing", "security+", "encryption", "oauth", "jwt"
        ],
        
        "finance": [
            # Analysis & Modeling
            "financial analysis", "financial modeling", "valuation", "forecasting", "budgeting",
            "risk management", "portfolio management", "investment analysis", "market analysis",
            # Tools & Software
            "bloomberg terminal", "excel", "vba", "power bi", "tableau", "hyperion", "quickbooks",
            # Technical Skills
            "financial statements", "balance sheet", "income statement", "cash flow", "ratio analysis",
            "derivatives", "options trading", "hedge funds", "private equity", "venture capital",
            # Certifications & Knowledge
            "cfa", "frm", "series 7", "series 63", "gaap", "ifrs", "basel iii", "dodd-frank",
            # Banking & Finance Areas
            "investment banking", "corporate finance", "mergers acquisitions", "credit analysis",
            "asset management", "wealth management", "cryptocurrency", "blockchain"
        ],
        
        "marketing": [
            # Digital Marketing
            "seo", "sem", "google analytics", "google ads", "social media marketing", "content marketing",
            "email marketing", "marketing automation", "a/b testing", "conversion optimization",
            # Social Media
            "facebook ads", "instagram marketing", "linkedin marketing", "tiktok marketing",
            "social media strategy", "community management", "influencer marketing",
            # Analytics & Tools
            "google tag manager", "hotjar", "mailchimp", "hubspot", "salesforce", "adobe analytics",
            "data visualization", "marketing metrics", "roi analysis", "attribution modeling",
            # Content & Strategy
            "content strategy", "brand management", "market research", "competitive analysis",
            "customer journey", "marketing funnel", "storytelling", "copywriting", "ab testing",
            # Traditional Marketing
            "brand development", "product marketing", "market segmentation", "customer acquisition",
            "public relations", "event marketing", "marketing communications"
        ],
        
        "healthcare": [
            # Clinical
            "patient care", "clinical documentation", "medical terminology", "hipaa", "electronic health records",
            "telehealth", "patient assessment", "medical coding", "icd-10", "cpt codes",
            # Technology
            "epic", "cerner", "meditech", "healthcare informatics", "medical devices", "emr systems",
            # Administrative
            "healthcare management", "medical billing", "revenue cycle", "population health",
            "quality assurance", "regulatory compliance", "joint commission", "case management",
            # Specializations
            "clinical research", "pharmaceutical", "biotechnology", "medical devices",
            "healthcare analytics", "public health", "mental health", "patient safety"
        ],

        "data_science": [
            # Programming & Tools
            "python", "r", "sql", "jupyter", "pandas", "numpy", "scipy", "scikit-learn",
            "tensorflow", "pytorch", "keras", "spark", "hadoop", "tableau", "power bi",
            # Techniques
            "machine learning", "deep learning", "natural language processing", "computer vision",
            "statistical analysis", "predictive modeling", "time series analysis", "a/b testing",
            # Big Data
            "big data", "data warehousing", "etl", "data pipeline", "data visualization",
            "data mining", "data cleaning", "feature engineering", "database design",
            # Mathematics & Statistics
            "statistics", "probability", "linear algebra", "calculus", "hypothesis testing",
            "regression analysis", "clustering", "classification", "neural networks"
        ],

        "project_management": [
            # Methodologies
            "agile", "scrum", "waterfall", "prince2", "lean", "six sigma", "kanban",
            "pmp", "itil", "critical path method",
            # Tools
            "jira", "trello", "asana", "microsoft project", "basecamp", "confluence",
            "smartsheet", "monday.com", "slack", "microsoft teams",
            # Skills
            "risk management", "stakeholder management", "budget management",
            "resource allocation", "change management", "quality management",
            "project planning", "team leadership", "scope management",
            # Documentation
            "requirements gathering", "project documentation", "status reporting",
            "business case", "project charter", "lessons learned"
        ]
    }

def calculate_ats_score(resume_text):
    """Calculate ATS compatibility score."""
    ats_factors = {
        'has_contact_info': any(keyword in resume_text.lower() for keyword in ['email', 'phone', 'address']),
        'has_education': any(keyword in resume_text.lower() for keyword in ['education', 'university', 'degree']),
        'has_experience': any(keyword in resume_text.lower() for keyword in ['experience', 'work', 'employment']),
        'has_skills': any(keyword in resume_text.lower() for keyword in ['skills', 'proficient', 'expertise']),
        'proper_sections': all(section in resume_text.lower() for section in ['summary', 'experience', 'education']),
        'no_images': 'base64' not in resume_text,
        'reasonable_length': 300 <= len(resume_text.split()) <= 1000
    }
    
    score = (sum(ats_factors.values()) / len(ats_factors)) * 100
    return score, ats_factors

def analyze_keywords(resume_text):
    """Analyze keyword optimization."""
    industry_keywords = get_industry_keywords()
    found_keywords = {}
    total_score = 0
    
    for industry, keywords in industry_keywords.items():
        found = [keyword for keyword in keywords if keyword.lower() in resume_text.lower()]
        score = (len(found) / len(keywords)) * 100
        found_keywords[industry] = {
            'found': found,
            'missing': list(set(keywords) - set(found)),
            'score': score
        }
        total_score += score
    
    avg_score = total_score / len(industry_keywords)
    return avg_score, found_keywords

def analyze_sections(resume_text):
    """Analyze completeness of resume sections."""
    essential_sections = ['summary', 'experience', 'education', 'skills']
    found_sections = [section for section in essential_sections 
                     if section.lower() in resume_text.lower()]
    return (len(found_sections) / len(essential_sections)) * 100

def calculate_resume_score(resume_text):
    """Calculate overall resume score based on various parameters."""
    scores = {
        'ats_compatibility': calculate_ats_score(resume_text)[0],
        'keyword_optimization': analyze_keywords(resume_text)[0],
        'content_length': min(100, (len(resume_text.split()) / 800) * 100),
        'section_completeness': analyze_sections(resume_text)
    }
    
    total_score = sum(scores.values()) / len(scores)
    return total_score, scores

def compare_with_job_description(resume_text, job_description):
    """Compare resume with job description."""
    vectorizer = TfidfVectorizer(stop_words='english')
    vectors = vectorizer.fit_transform([resume_text, job_description])
    similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0] * 100
    
    # Extract key terms from job description
    job_terms = set(job_description.lower().split()) - set(vectorizer.get_stop_words())
    resume_terms = set(resume_text.lower().split())
    matching_terms = job_terms.intersection(resume_terms)
    missing_terms = job_terms - resume_terms
    
    return similarity, list(matching_terms), list(missing_terms)

def extract_job_description_from_url(url):
    """Extract job description from common job posting websites."""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
            
        # Extract text from common job description containers
        job_description = ""
        
        # LinkedIn
        if "linkedin.com" in url:
            job_section = soup.find("div", {"class": "description__text"})
            if job_section:
                job_description = job_section.get_text()
                
        # Indeed
        elif "indeed.com" in url:
            job_section = soup.find("div", {"id": "jobDescriptionText"})
            if job_section:
                job_description = job_section.get_text()
                
        # Glassdoor
        elif "glassdoor.com" in url:
            job_section = soup.find("div", {"class": "jobDescriptionContent"})
            if job_section:
                job_description = job_section.get_text()
        
        # If no specific site matcher, try to get all text
        if not job_description:
            job_description = soup.get_text()
            
        # Clean the text
        job_description = re.sub(r'\s+', ' ', job_description).strip()
        
        return job_description
    except Exception as e:
        return f"Error extracting job description: {str(e)}"

# Set page title
st.set_page_config(page_title="AI Resume Assistant", page_icon="📄")

# Title of the app
st.title("📄 AI Resume Assistant")


resume_text = ""
faiss_index = None

# Add this near the top of your file where other session state initializations are done
if "job_url" not in st.session_state:
    st.session_state.job_url = ""
if "job_description" not in st.session_state:
    st.session_state.job_description = ""

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

# Analysis Tab
with tab2:
    st.subheader("📊 Resume Analysis Dashboard")
    
    if uploaded_file and resume_text:
        # Create two expandable sections
        with st.expander("Resume Scores", expanded=False):
            # Create three columns for the main metrics
            metric_col1, metric_col2, metric_col3 = st.columns(3)
            
            if st.button("🔍 Analyze Resume", type="primary", use_container_width=True):
                with st.spinner("Analyzing your resume..."):
                    # Calculate all scores
                    total_score, score_breakdown = calculate_resume_score(resume_text)
                    ats_score, ats_factors = calculate_ats_score(resume_text)
                    keyword_score, keyword_analysis = analyze_keywords(resume_text)
                    
                    # Display main metrics
                    with metric_col1:
                        st.metric(
                            label="Overall Resume Score",
                            value=f"{total_score:.1f}%",
                            delta="Target: 85%+"
                        )
                    
                    with metric_col2:
                        st.metric(
                            label="ATS Compatibility",
                            value=f"{ats_score:.1f}%",
                            delta="Target: 90%+"
                        )
                    
                    with metric_col3:
                        st.metric(
                            label="Keyword Optimization",
                            value=f"{keyword_score:.1f}%",
                            delta="Target: 80%+"
                        )
                    
                    # Create tabs for detailed analysis
                    analysis_tab1, analysis_tab2, analysis_tab3 = st.tabs([
                        "📊 Score Analysis KPIs", 
                        "🎯 ATS Analysis", 
                        "🔑 Keyword Analysis"
                    ])
                    
                    # Score Breakdown Tab
                    with analysis_tab1:
                        st.subheader("Score Analysis KPIs")
                        
                        # Create a modern KPI layout
                        kpi_metrics = {
                            'ATS Compatibility': {
                                'score': score_breakdown['ats_compatibility'],
                                'icon': '🎯',
                                'target': 90,
                                'description': 'Measures how well your resume works with ATS'
                            },
                            'Keyword Optimization': {
                                'score': score_breakdown['keyword_optimization'],
                                'icon': '🔍',
                                'target': 80,
                                'description': 'Analyzes presence of industry-relevant keywords'
                            },
                            'Content Length': {
                                'score': score_breakdown['content_length'],
                                'icon': '📝',
                                'target': 85,
                                'description': 'Evaluates if your resume has optimal length'
                            },
                            'Section Completeness': {
                                'score': score_breakdown['section_completeness'],
                                'icon': '📋',
                                'target': 95,
                                'description': 'Checks if all essential sections are present'
                            }
                        }

                        # Create a 2x2 grid for KPIs
                        col1, col2 = st.columns(2)
                        col3, col4 = st.columns(2)
                        cols = [col1, col2, col3, col4]

                        for i, (metric, data) in enumerate(kpi_metrics.items()):
                            with cols[i]:
                                # Calculate percentage of target achieved
                                target_percentage = min(100, (data['score'] / data['target']) * 100)
                                
                                # Determine status color
                                if data['score'] >= data['target']:
                                    color = "#28a745"  # green
                                    status = "Excellent"
                                elif data['score'] >= data['target'] * 0.8:
                                    color = "#ffc107"  # orange
                                    status = "Good"
                                else:
                                    color = "#dc3545"  # red
                                    status = "Needs Improvement"

                                # Create KPI card HTML
                                kpi_html = (
                                    '<div style="padding: 1rem; border-radius: 0.7rem; background: #f8f9fa; '
                                    'box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 1rem;">'
                                    '<div style="display: flex; justify-content: space-between; align-items: center;">'
                                    f'<span style="font-size: 1.8rem;">{data["icon"]}</span>'
                                    f'<span style="color: {color}; font-weight: bold; font-size: 1.5rem;">'
                                    f'{data["score"]:.1f}%</span></div>'
                                    f'<h3 style="margin: 0.5rem 0; font-size: 1.1rem; color: #2c3e50;">{metric}</h3>'
                                    '<div style="background: #e9ecef; border-radius: 0.5rem; height: 0.5rem; '
                                    'margin: 0.5rem 0;">'
                                    f'<div style="width: {target_percentage}%; height: 100%; background: {color}; '
                                    'border-radius: 0.5rem;"></div></div>'
                                    '<div style="display: flex; justify-content: space-between; '
                                    'font-size: 0.8rem; color: #495057;">'
                                    f'<span>Target: {data["target"]}%</span>'
                                    f'<span style="color: {color};">{status}</span>'
                                    '</div></div>'
                                )
                                
                                st.markdown(kpi_html, unsafe_allow_html=True)

                        # Overall Score Card
                        overall_color = "#28a745" if total_score >= 85 else "#ffc107" if total_score >= 70 else "#dc3545"
                        overall_status = ("Excellent! Your resume is well-optimized." if total_score >= 85 
                                         else "Good progress, but there's room for improvement." if total_score >= 70 
                                         else "Your resume needs significant improvements.")
                        
                        overall_html = (
                            '<div style="padding: 1.5rem; border-radius: 0.7rem; background: #f8f9fa; '
                            'box-shadow: 0 2px 4px rgba(0,0,0,0.1); text-align: center; margin: 1rem 0;">'
                            f'<h1 style="font-size: 3rem; color: {overall_color}; margin: 0;">'
                            f'{total_score:.1f}%</h1>'
                            f'<p style="font-size: 1.2rem; margin: 0.5rem 0; color: #2c3e50;">{overall_status}</p>'
                            '</div>'
                        )
                        
                        st.markdown("### Overall Resume Score")
                        st.markdown(overall_html, unsafe_allow_html=True)
                    
                    # ATS Analysis Tab
                    with analysis_tab2:
                        st.subheader("ATS Compatibility Check")
                        
                        # Create two columns
                        ats_col1, ats_col2 = st.columns(2)
                        
                        with ats_col1:
                            st.write("### Required Elements")
                            for factor, passed in ats_factors.items():
                                if passed:
                                    st.success(f"✅ {factor.replace('_', ' ').title()}")
                                else:
                                    st.error(f"❌ {factor.replace('_', ' ').title()}")
                        
                        with ats_col2:
                            st.write("### Recommendations")
                            for factor, passed in ats_factors.items():
                                if not passed:
                                    st.warning(f"Add {factor.replace('_', ' ').lower()} to improve ATS compatibility")
                    
                    # Keyword Analysis Tab
                    with analysis_tab3:
                        st.subheader("Industry Keyword Analysis")
                        
                        # Create tabs for each industry
                        industry_tabs = st.tabs([industry.title() for industry in keyword_analysis.keys()])
                        
                        for tab, (industry, analysis) in zip(industry_tabs, keyword_analysis.items()):
                            with tab:
                                st.write(f"### {industry.title()} Industry Match")
                                st.progress(analysis['score']/100)
                                st.write(f"Industry Score: {analysis['score']:.1f}%")
                                
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.write("#### ✅ Found Keywords")
                                    for keyword in analysis['found']:
                                        st.success(keyword)
                                
                                with col2:
                                    st.write("#### ❌ Missing Keywords")
                                    for keyword in analysis['missing']:
                                        st.error(keyword)
                    
        # Move Job Description section outside and make it a separate expander
        with st.expander("📋 Job Description Matcher", expanded=True):
            # Initialize job_description in session state if not exists
            if "job_description" not in st.session_state:
                st.session_state.job_description = ""

            # Create tabs for different input methods
            jd_tab1, jd_tab2 = st.tabs(["📝 Paste Description", "🔗 URL Input"])

            with jd_tab1:
                manual_input = st.text_area(
                    "Paste a job description to compare with your resume:",
                    height=150,
                    key="manual_jd_input",
                    placeholder="Paste the job description here to see how well your resume matches..."
                )
                if manual_input:
                    st.session_state.job_description = manual_input

            with jd_tab2:
                job_url = st.text_input(
                    "Enter job posting URL:",
                    placeholder="https://www.linkedin.com/jobs/view/...",
                    key="url_input"
                )
                
                if st.button("Extract Job Description", key="extract_button"):
                    if job_url:
                        with st.spinner("Extracting job description..."):
                            try:
                                extracted_description = extract_job_description_from_url(job_url)
                                if extracted_description.startswith("Error"):
                                    st.error(extracted_description)
                                else:
                                    st.session_state.job_description = extracted_description
                                    st.success("✅ Job description extracted successfully! Click 'Compare with Job Description' to analyze.")
                            except Exception as e:
                                st.error(f"❌ Error extracting job description: {str(e)}")
                    else:
                        st.warning("⚠️ Please enter a job URL first.")

            # Compare button outside both tabs
            if st.session_state.job_description:
                if st.button("Compare with Job Description", type="primary", key="compare_button"):
                    with st.spinner("Analyzing compatibility..."):
                        match_score, matching_terms, missing_terms = compare_with_job_description(
                            resume_text, st.session_state.job_description
                        )
                        
                        # Create columns for the analysis
                        score_col, recommendation_col = st.columns([1, 2])
                        
                        with score_col:
                            st.metric(
                                label="Job Match Score",
                                value=f"{match_score:.1f}%",
                                delta=f"{'Good Match!' if match_score > 70 else 'Needs Improvement'}"
                            )
                        
                        with recommendation_col:
                            if match_score >= 80:
                                st.success("🌟 Excellent match! Your resume aligns well with this position.")
                            elif match_score >= 60:
                                st.warning("👍 Good match, but there's room for improvement.")
                            else:
                                st.error("⚠️ Consider updating your resume to better match this role.")
                        
                        # # Detailed Analysis - Replace expander with container
                        st.markdown("### Detailed Analysis")
                        # st.markdown("""
                        #     <div style="padding: 1rem; border-radius: 0.5rem; 
                        #     background: #2c3e50; color: white; margin-bottom: 1rem;">
                        #     <h3 style="margin: 0; color: white;">Detailed Analysis</h3>
                        #     </div>
                        # """, unsafe_allow_html=True)
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write("#### ✅ Matching Keywords")
                            for term in matching_terms[:10]:  # Show top 10 matches
                                st.success(term)
                        
                        with col2:
                            st.write("#### 📝 Recommended Additions")
                            for term in missing_terms[:10]:  # Show top 10 missing terms
                                st.warning(term)
                        
                        # Recommendations
                        st.markdown("### 💡 Recommendations")
                        recommendations = [
                            f"Add relevant missing keywords: {', '.join(missing_terms[:5])}..." if missing_terms else "Your keyword coverage is good!",
                            f"Your resume matches {match_score:.1f}% of the job requirements.",
                            "Consider tailoring your experience descriptions to better match the job requirements.",
                            "Ensure your most relevant experience for this role is prominently featured."
                        ]
                        
                        for rec in recommendations:
                            st.info(rec)
    else:
        st.info("👆 Please upload your resume to get started with the analysis!")

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