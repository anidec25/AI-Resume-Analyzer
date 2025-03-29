import streamlit as st
from src.analyzers.resume_analyzer import calculate_resume_score, calculate_ats_score, analyze_keywords
from src.analyzers.job_analyzer import compare_with_job_description, extract_job_description_from_url

def render_manual_input():
    """Render the manual job description input section."""
    manual_input = st.text_area(
        "Paste a job description to compare with your resume:",
        height=150,
        key="manual_jd_input",
        placeholder="Paste the job description here to see how well your resume matches..."
    )
    if manual_input:
        st.session_state.job_description = manual_input

def render_url_input():
    """Render the URL input section for job description extraction."""
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

def render_comparison_results(resume_text, job_description):
    """Render the comparison results between resume and job description."""
    if st.button("Compare with Job Description", type="primary", key="compare_button"):
        with st.spinner("Analyzing compatibility..."):
            match_score, matching_terms, missing_terms = compare_with_job_description(
                resume_text, job_description
            )
            
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
            
            st.markdown("### Detailed Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("#### ✅ Matching Keywords")
                for term in matching_terms[:10]:
                    st.success(term)
            
            with col2:
                st.write("#### 📝 Recommended Additions")
                for term in missing_terms[:10]:
                    st.warning(term)
            
            st.markdown("### 💡 Recommendations")
            recommendations = [
                f"Add relevant missing keywords: {', '.join(missing_terms[:5])}..." if missing_terms else "Your keyword coverage is good!",
                f"Your resume matches {match_score:.1f}% of the job requirements.",
                "Consider tailoring your experience descriptions to better match the job requirements.",
                "Ensure your most relevant experience for this role is prominently featured."
            ]
            
            for rec in recommendations:
                st.info(rec)

def render_analysis_tab(resume_text, uploaded_file):
    """Render the analysis tab interface."""
    st.subheader("📊 Resume Analysis Dashboard")
    
    if uploaded_file and resume_text:
        # Create two expandable sections
        with st.expander("📋 Resume Scores", expanded=False):
            render_resume_scores(resume_text)
        
        with st.expander("📋 Job Description Matcher", expanded=False):
            # Initialize job_description in session state if not exists
            if "job_description" not in st.session_state:
                st.session_state.job_description = ""

            # Create tabs for different input methods
            jd_tab1, jd_tab2 = st.tabs(["📝 Paste Description", "🔗 URL Input"])

            with jd_tab1:
                render_manual_input()

            with jd_tab2:
                render_url_input()

            # Compare button outside both tabs
            if st.session_state.job_description:
                render_comparison_results(resume_text, st.session_state.job_description)
    else:
        st.info("👆 Please upload your resume to get started with the analysis!")

def render_resume_scores(resume_text):
    """Render the resume scores section."""
    metric_col1, metric_col2, metric_col3 = st.columns(3)
    
    # Remove the Analyze Resume button and directly analyze
    with st.spinner("Analyzing your resume..."):
        total_score, score_breakdown = calculate_resume_score(resume_text)
        ats_score, ats_factors = calculate_ats_score(resume_text)
        keyword_score, keyword_analysis = analyze_keywords(resume_text)
        
        # Display metrics
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
        
        # Add detailed analysis sections here
        render_detailed_analysis(score_breakdown, ats_factors, keyword_analysis)

def render_detailed_analysis(score_breakdown, ats_factors, keyword_analysis):
    """Render detailed analysis of resume scores."""
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
                    for keyword in analysis['found'][:10]:  # Show top 10 matches
                        st.success(keyword)
                
                with col2:
                    st.write("#### ❌ Missing Keywords")
                    for keyword in analysis['missing'][:10]:  # Show top 10 missing
                        st.error(keyword) 