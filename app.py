import streamlit as st
from src.modules.module1_question_generation.file_processing import extract_text_from_file
from src.modules.module1_question_generation.groq_client import GroqClient

def main():
    st.title("AI Interview Question Generator")
    
    # Job details input
    job_role = st.text_input("Enter Job Role")
    jd_file = st.file_uploader("Upload Job Description (PDF/DOCX)", type=["pdf", "docx"])
    
    if jd_file and job_role:
        with st.spinner("Analyzing Job Description..."):
            jd_text = extract_text_from_file(jd_file)
            
            # Generate questions directly
            client = GroqClient()
            questions = client.generate_questions(job_role, jd_text)
            
            st.subheader("Generated Questions")
            st.write(questions)
            
            # Add export functionality
            st.download_button(
                "Download Questions",
                questions,
                file_name=f"{job_role.replace(' ', '_')}_interview_questions.txt"
            )

if __name__ == "__main__":
    main()