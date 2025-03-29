import PyPDF2
import docx
import base64
import streamlit as st

def display_pdf(file_content, filename):
    """Display PDF in a more compact format suitable for sidebar."""
    # For PDFs
    if filename.lower().endswith('.pdf'):
        base64_pdf = base64.b64encode(file_content).decode('utf-8')
        # Adjust the width and height to fit sidebar
        pdf_display = f"""
            <iframe
                src="data:application/pdf;base64,{base64_pdf}"
                width="100%"
                height="300px"
                type="application/pdf"
                style="border: 1px solid #ccc; border-radius: 5px;"
            >
            </iframe>
        """
        st.markdown(pdf_display, unsafe_allow_html=True)
    # For DOCX and TXT
    else:
        try:
            text = extract_text_from_resume(file_content)
            # Display text in a scrollable container
            st.markdown(
                f"""
                <div style="
                    border: 1px solid #ccc;
                    border-radius: 5px;
                    padding: 10px;
                    height: 400px;
                    overflow-y: auto;
                    background-color: white;
                    font-size: 0.8em;
                ">
                    {text.replace(chr(10), '<br>')}
                </div>
                """,
                unsafe_allow_html=True
            )
        except Exception as e:
            st.error(f"Error displaying file: {str(e)}")

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