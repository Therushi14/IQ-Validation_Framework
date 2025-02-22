import streamlit as st
import os
import sys

# Adjust the system path to find project modules
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
sys.path.append(project_root)

from src.modules.module2_relevancy.relevance_analyzer import EnhancedRelevanceAnalyzer
from groq_client import GroqClient
from file_processing import extract_text_from_file


def main():
    st.title("AI Interview Question Generator & Analyzer")
    
    # Initialize modules
    client = GroqClient()
    analyzer = EnhancedRelevanceAnalyzer()
    
    # Job details input
    job_role = st.text_input("Enter Job Role")
    jd_file = st.file_uploader("Upload Job Description (PDF/DOCX)", type=["pdf", "docx"])
    
    if jd_file and job_role:
        with st.spinner("Analyzing Job Description..."):
            jd_text = extract_text_from_file(jd_file)
            
            # Generate questions
            questions = client.generate_questions(job_role, jd_text)
            
            question_lines = [q.strip() for q in questions.split('\n') if q.strip()]
            if question_lines and not question_lines[0][0].isdigit():
                question_lines = question_lines[1:]
            question_list = question_lines
            
            # Analyze relevance
            scores = analyzer.calculate_question_scores(jd_text, question_list)
            avg_score = sum(scores) / len(scores)
            
            half_avg = avg_score /1.25
            count_above_half = sum(1 for s in scores if s > half_avg)
            overall_relevance = (count_above_half / len(scores)) * 100
            
            # Display metrics (without individual question scores or plots)
            st.subheader("Analysis Results")
            st.metric("Overall Relevance", f"{overall_relevance:.1f}%")
            st.metric("Total Questions", len(question_list))
            
            # Export data for reference if needed
            export_data = "\n".join(
                [f"Q{i+1}\t{score}\t{question}" 
                 for i, (question, score) in enumerate(zip(question_list, scores))]
            )
            st.download_button(
                "Download Questions with Scores",
                f"Job Role: {job_role}\nOverall Relevance: {overall_relevance:.1f}%\n\n{export_data}",
                file_name=f"{job_role.replace(' ', '_')}_questions_with_scores.tsv",
                mime="text/tsv"
            )

if __name__ == "__main__":
    main()
