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
    from src.config.keywords import get_industry_keywords
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