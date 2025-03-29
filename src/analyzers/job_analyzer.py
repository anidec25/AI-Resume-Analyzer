from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import requests
from bs4 import BeautifulSoup
import re

def compare_with_job_description(resume_text, job_description):
    """Compare resume with job description."""
    vectorizer = TfidfVectorizer(stop_words='english')
    vectors = vectorizer.fit_transform([resume_text, job_description])
    similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0] * 100
    
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
        
        for script in soup(["script", "style"]):
            script.decompose()
            
        job_description = ""
        
        if "linkedin.com" in url:
            job_section = soup.find("div", {"class": "description__text"})
            if job_section:
                job_description = job_section.get_text()
                
        elif "indeed.com" in url:
            job_section = soup.find("div", {"id": "jobDescriptionText"})
            if job_section:
                job_description = job_section.get_text()
                
        elif "glassdoor.com" in url:
            job_section = soup.find("div", {"class": "jobDescriptionContent"})
            if job_section:
                job_description = job_section.get_text()
        
        if not job_description:
            job_description = soup.get_text()
            
        job_description = re.sub(r'\s+', ' ', job_description).strip()
        
        return job_description
    except Exception as e:
        return f"Error extracting job description: {str(e)}" 