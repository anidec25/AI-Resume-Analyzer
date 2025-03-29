## Overview

AI-Resume-Analyzer is an intelligent application designed to help job seekers improve their resumes and get personalized career advice. By combining natural language processing and machine learning techniques, the system analyzes resume content and provides tailored feedback, suggestions, and answers to career-related questions.

## How It Works

1. **Resume Upload & Processing**: The system extracts text from various resume formats using specialized parsers.
2. **Vector Embedding**: The resume content is converted into vector embeddings using OpenAI's embedding model.
3. **Retrieval-Augmented Generation (RAG)**: When a user asks a question, the system:
   - Retrieves the most relevant sections from the resume
   - Combines this context with the user's question
   - Generates an accurate, personalized response using OpenAI's language model

This approach ensures that responses are grounded in the actual content of your resume rather than generic advice.

## Dashboard (Streamlit App)
#### Link: https://airesume-analyzer.streamlit.app/
![alt text](<images/dashboard.png>)


## Features

- 📄 **Resume Upload & Analysis**
  - Supports multiple file formats (PDF, DOCX, and TXT)
  - Extracts and processes text content automatically
  - Creates searchable vector embeddings for intelligent retrieval

- 🔍 **Interactive AI Chat**
  - Ask specific questions about your resume (e.g., "What skills should I highlight?", "Which companies have I worked for?")
  - Get accurate information about your work history and experience
  - Receive tailored feedback on your experience and qualifications
  - Smart company extraction to identify all previous employers
  - Maintains context for follow-up questions

- 📊 **Comprehensive Resume Analysis**
  - Strengths assessment
  - Areas for improvement identification
  - Missing key elements detection
  - Formatting and structure recommendations
  - Modern KPI-based scoring system

- 🚀 **Personalized Career Guidance**
  - Career path recommendations based on your skills and experience
  - Suggestions for skill development opportunities
  - Industry-specific advice and insights
  - Job hunting strategies tailored to your profile

- 🧠 **Chat Memory & Context Retention**
  - Maintains conversation history for more coherent interactions
  - References previous questions and answers for context

- 🎨 **User-Friendly Interface**
  - Clean, intuitive design for easy navigation
  - Responsive layout that works on desktop and mobile devices
  - Tab-based organization for different functionalities

## Installation

### Prerequisites
- Python 3.8 or higher
- An OpenAI API key (sign up at [OpenAI](https://platform.openai.com) if you don't have one)

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/anidec25/AI-Resume-Analyzer.git
cd AI-Resume-Analyzer
```

### 2️⃣ Set Up a Virtual Environment (Recommended)
```bash
# For Windows
python -m venv venv
venv\Scripts\activate

# For macOS/Linux
python -m venv venv
source venv/bin/activate
```

### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 4️⃣ Set Up OpenAI API Key
There are two ways to set up your API key:

**Option 1**: Set as an environment variable:
```bash
# For Windows
set OPENAI_API_KEY=your-api-key

# For macOS/Linux
export OPENAI_API_KEY="your-api-key"
```

**Option 2**: Enter directly in the application (now available in the user interface)

### 5️⃣ Run the Application
```bash
streamlit run app.py
```

## Detailed Usage Guide

### Starting the Application
1. Launch the application using `streamlit run app.py`
2. Open your web browser and navigate to the URL displayed in the terminal (typically `http://localhost:8501`)

### Using the Application
1. **Enter your OpenAI API Key**: 
   - Input your API key in the sidebar text field
   - This is required before any other functionality becomes available

2. **Upload your Resume**: 
   - Click the "Browse files" button in the sidebar
   - Select your resume file (PDF, DOCX, or TXT format)
   - Wait for the "Resume uploaded and processed successfully!" confirmation

3. **Ask Questions and Get Insights**:
   - Use the "Chat" tab to ask specific questions about your resume
   - Ask about work history (e.g., "Which companies have I worked for?")
   - Type your question in the input field and press Enter
   - View AI-generated responses based on your resume content
   - Follow up with related questions for more details

4. **Get a Comprehensive Resume Analysis**:
   - Navigate to the "Analysis" tab
   - Click the "Analyze Resume" button
   - Review the detailed analysis with modern KPI cards
   - Check company-specific insights and experience details
   - See recommendations for improvements

5. **Receive Career Guidance**:
   - Go to the "Career Guidance" tab
   - Click "Get Career Advice"
   - Explore personalized career path recommendations, skill development opportunities, and job hunting strategies

### Best Practices
- Ask specific questions for more targeted responses
- Upload an up-to-date, complete resume for the best results
- Clear the chat occasionally to start fresh conversations
- Try different questions to explore various aspects of your resume

## Technologies Used
- **Python** 🐍 - Core programming language
- **Streamlit** 🎨 - Web application framework for creating the interactive UI
- **OpenAI GPT** 🧠 - Large language model for generating human-like responses
- **OpenAI Embeddings** 🔢 - Creates vector representations of text for semantic search
- **FAISS** 📚 - Facebook AI Similarity Search for efficient vector similarity search
- **PyPDF2 & python-docx** 📄 - Libraries for parsing PDF and DOCX documents
- **Langchain** ⛓️ - Framework for developing applications powered by language models

## Future Enhancements 🚀
- 📌 **Advanced Resume Parsing**
  - Support for more resume formats (JSON, HTML, etc.)
  - Better handling of tables, graphics, and complex layouts
  - Improved extraction of contact information and metadata

- 🏆 **AI-powered Resume Scoring**
  - Quantitative assessment of resume quality
  - Industry-specific scoring benchmarks
  - Section-by-section rating with visualization

- 📊 **Enhanced Analytics**
  - Visual representation of skills and experience
  - Comparison with industry standards and job requirements
  - Keyword optimization suggestions

- 📢 **Job Role Recommendations**
  - Integration with job posting APIs
  - Matching resume to suitable positions
  - Tailored application advice for specific job listings

- 🌐 **Multi-language Support**
  - Resume analysis in multiple languages
  - Translation capabilities for international job seekers

- 💾 **User Accounts & Resume Storage**
  - Secure login system
  - Save multiple resume versions
  - Track improvements over time

## Contributing
Contributions are welcome and appreciated! Here's how you can contribute:

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Commit your changes**: `git commit -m 'Add some amazing feature'`
4. **Push to the branch**: `git push origin feature/amazing-feature`
5. **Open a Pull Request**

You can also contribute by:
- Reporting bugs
- Suggesting enhancements
- Improving documentation
- Sharing the project

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements
- OpenAI for providing the API
- Streamlit for the excellent web app framework
- The open-source community for various libraries used in this project
