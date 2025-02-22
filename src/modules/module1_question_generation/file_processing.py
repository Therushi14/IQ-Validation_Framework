import PyPDF2
from docx import Document

def extract_text_from_file(uploaded_file):
    """Handle PDF and DOCX file parsing"""
    text = ""
    
    if uploaded_file.type == "application/pdf":
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        text = "\n".join([page.extract_text() for page in pdf_reader.pages])
        
    elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        doc = Document(uploaded_file)
        text = "\n".join([para.text for para in doc.paragraphs])
        
    return text