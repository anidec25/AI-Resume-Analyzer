def calculate_ats_score(resume_text):
    """Calculate ATS compatibility score with more robust checks."""
    # Normalize text for consistent checking
    text_lower = resume_text.lower()
    words = text_lower.split()
    
    # More comprehensive keyword sets
    contact_keywords = {
        'email', '@', 'phone', 'tel', 'mobile', 'address', 'location', 
        'linkedin', 'github', 'portfolio'
    }
    
    education_keywords = {
        'education', 'university', 'college', 'degree', 'bachelor', 
        'master', 'phd', 'diploma', 'certification', 'academic'
    }
    
    experience_keywords = {
        'experience', 'work', 'employment', 'career', 'job', 'position',
        'role', 'professional', 'achievements', 'responsibilities'
    }
    
    skills_keywords = {
        'skills', 'proficient', 'expertise', 'competencies', 'technologies',
        'tools', 'languages', 'frameworks', 'methodologies', 'abilities'
    }
    
    required_sections = {
        'summary': {'summary', 'profile', 'objective', 'about'},
        'experience': {'experience', 'employment', 'work history', 'professional background'},
        'education': {'education', 'academic', 'qualifications', 'training'}
    }

    # Enhanced scoring factors
    ats_factors = {
        'has_contact_info': {
            'score': any(keyword in text_lower for keyword in contact_keywords),
            'message': "Include contact information (email, phone, location)"
        },
        'has_education': {
            'score': any(keyword in text_lower for keyword in education_keywords),
            'message': "Add education details (degree, university, certifications)"
        },
        'has_experience': {
            'score': any(keyword in text_lower for keyword in experience_keywords),
            'message': "Include work experience details"
        },
        'has_skills': {
            'score': any(keyword in text_lower for keyword in skills_keywords),
            'message': "Add a skills section highlighting your expertise"
        },
        'proper_sections': {
            'score': all(
                any(keyword in text_lower for keyword in section_keywords)
                for section_keywords in required_sections.values()
            ),
            'message': "Ensure all key sections (Summary, Experience, Education) are present"
        },
        'no_images': {
            'score': 'base64' not in resume_text,
            'message': "Avoid embedding images in your resume"
        },
        'reasonable_length': {
            'score': 300 <= len(words) <= 1000,
            'message': f"Current length: {len(words)} words. Aim for 300-1000 words"
        },
        'keyword_density': {
            'score': 0.1 <= len(set(words)) / len(words) <= 0.5,
            'message': "Maintain a good balance of keyword variety"
        }
    }
    
    # Calculate weighted score
    weights = {
        'has_contact_info': 1.0,
        'has_education': 1.0,
        'has_experience': 1.5,  # Higher weight for experience
        'has_skills': 1.2,      # Higher weight for skills
        'proper_sections': 1.0,
        'no_images': 0.8,
        'reasonable_length': 1.0,
        'keyword_density': 0.5
    }
    
    total_weight = sum(weights.values())
    weighted_score = sum(
        ats_factors[factor]['score'] * weights[factor]
        for factor in ats_factors
    ) / total_weight * 100

    return weighted_score, ats_factors

def get_ats_improvement_suggestions(ats_factors):
    """Get specific suggestions for improving ATS compatibility."""
    suggestions = []
    for factor, details in ats_factors.items():
        if not details['score']:
            suggestions.append(details['message'])
    return suggestions

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