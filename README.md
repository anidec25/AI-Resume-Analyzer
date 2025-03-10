# AI-Resume-Analyzer

This is an AI-powered Resume Chatbot that allows users to upload their resume and ask career-related questions. The system leverages **Retrieval-Augmented Generation (RAG)** to provide more accurate responses based on the uploaded resume.

## UI
![alt text](<Screenshot 2025-03-10 at 9.04.46 PM.png>)


## Features
- 📄 **Upload Resume** (Supports PDF, DOCX, and TXT)
- 🔍 **Ask Questions** about your resume (e.g., "What skills should I improve?")
- 📊 **Resume Analysis & Career Guidance**
- 🧠 **Memory Retention** (Keeps previous chat history)
- 🎨 **Minimal & Professional UI**
- ⚡ **Press Enter to Generate Response**
- 🗑 **Clear Chat Button** (Next to Generate Response)

## Installation

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/anidec25/AI-Resume-Analyzer.git
cd resume-chatbot
```

### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ Set Up OpenAI API Key
Set your OpenAI API key as an environment variable:
```bash
export OPENAI_API_KEY="your-api-key"
```

### 4️⃣ Run the Application
```bash
streamlit run app.py
```

## Usage
1. Upload your resume 📄
2. Ask career-related questions 🔍
3. Get personalized insights and recommendations ✨

## Technologies Used
- **Python** 🐍
- **Streamlit** 🎨 (UI Framework)
- **OpenAI GPT** 🧠 (Chat Model)
- **FAISS** 📚 (For RAG-based retrieval)
- **PyPDF2 & python-docx** 📄 (Resume Parsing)

## Future Enhancements 🚀
- 📌 Support for more resume formats
- 🏆 AI-powered resume scoring
- 📢 Job role recommendations

## Contributing
Feel free to contribute by submitting a pull request or opening an issue!

## License
This project is licensed under the MIT License.
