from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS

def create_faiss_index(text, embeddings):
    """Creates a FAISS index from resume text."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_text(text)
    vector_store = FAISS.from_texts(chunks, embeddings)
    return vector_store 

def extract_work_experience(resume_text):
    """Extract work experiences with improved date parsing and sorting."""
    import re
    from datetime import datetime
    
    # Pattern to match work experiences with dates
    experience_pattern = r"(?i)(.*?)\s*(?:at|@)\s*(.*?)\s*(?:from|-)?\s*(\w+\s*\d{4})\s*(?:to|-)?\s*(\w+\s*\d{4}|present|current)"
    
    experiences = []
    matches = re.finditer(experience_pattern, resume_text)
    
    for match in matches:
        role, company, start_date, end_date = match.groups()
        
        # Clean up extracted data
        role = role.strip()
        company = company.strip()
        start_date = start_date.strip()
        end_date = end_date.strip().lower()
        
        # Convert dates to datetime objects for sorting
        try:
            start = datetime.strptime(start_date, '%B %Y')
            end = datetime.now() if end_date in ['present', 'current'] else datetime.strptime(end_date, '%B %Y')
            
            experiences.append({
                'role': role,
                'company': company,
                'start_date': start,
                'end_date': end,
                'is_current': end_date in ['present', 'current']
            })
        except ValueError:
            continue
    
    # Sort by end_date in descending order
    experiences.sort(key=lambda x: x['end_date'], reverse=True)
    return experiences

def get_latest_experience(resume_text):
    """Get the most recent work experience."""
    experiences = extract_work_experience(resume_text)
    
    if experiences:
        latest = experiences[0]
        if latest['is_current']:
            end_date = "Present"
        else:
            end_date = latest['end_date'].strftime('%B %Y')
            
        return f"Your latest experience is as a {latest['role']} at {latest['company']} from {latest['start_date'].strftime('%B %Y')} to {end_date}."
    
    return "Could not find work experience information in your resume." 