import streamlit as st
import os
import sys
import numpy as np
# Adjust the system path to find project modules
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
sys.path.append(project_root)

from src.modules.module2_relevancy.relevance_analyzer import EnhancedRelevanceAnalyzer
from groq_client import GroqClient
from file_processing import extract_text_from_file
from src.modules.module3_compare.model import QuestionSimilarityModel  # Import the model

def main():
    st.title("AI Interview Question Generator & Analyzer")
    
    # Initialize modules
    client = GroqClient()
    analyzer = EnhancedRelevanceAnalyzer()
    similarity_model = QuestionSimilarityModel('dataset/leetcode_dataset.csv')  # Initialize similarity model
    
    # Job details input
    job_role = st.text_input("Enter Job Role")
    jd_file = st.file_uploader("Upload Job Description (PDF/DOCX)", type=["pdf", "docx"])
    
    if jd_file and job_role:
        with st.spinner("Analyzing Job Description..."):
            jd_text = extract_text_from_file(jd_file)
            
            with st.spinner("Validating Job Description..."):
                if not analyzer.check_title_jd_match(job_role, jd_text):
                    st.error("⚠️ Job description doesn't match the job title! Upload a relevant JD.")
                    st.stop()
            
            # Generate questions
            questions = client.generate_questions(job_role, jd_text)
            
            question_lines = [q.strip() for q in questions.split('\n') if q.strip()]
            if question_lines and not question_lines[0][0].isdigit():
                question_lines = question_lines[1:]
            
            # Separate first 3 questions and remaining questions
            first_three_questions = question_lines[:3]
            remaining_questions = question_lines[3:]
            
            # Get similarity analysis for first 3 questions
            similarity_results = similarity_model.check_similarity(first_three_questions)
            
            # Display questions with similarity analysis
            st.subheader("First 3 Questions with Similarity Analysis")
            for i, (question, result) in enumerate(zip(first_three_questions, similarity_results), 1):
                st.write(f"{i}. {question}")
                with st.expander(f"Similarity Analysis for Question {i}"):
                    st.write(f"Similarity Score: {result['relevance_score']:.2f}")
                    st.write(f"Best Match: {result['best_match']['title']}")
                    st.write(f"Difficulty: {result['best_match']['difficulty']}")
                    if result['matched_sources']:
                        st.write("\nSimilar Questions:")
                        for source in result['matched_sources']:
                            st.write(f"- {source['title']} (Difficulty: {source['difficulty']})")
                
            st.subheader("Remaining Questions")
            for i, q in enumerate(remaining_questions, 4):
                st.write(f"{i}. {q}")
            
            # Analyze relevance for all questions combined
            scores = analyzer.calculate_question_scores(jd_text, question_lines)
            avg_score = sum(scores) / len(scores)
            
            half_avg = avg_score /1.25
            count_above_half = sum(1 for s in scores if s > half_avg)
            overall_relevance = (count_above_half / len(scores)) * 100
            
            st.subheader("Analysis Results")
            st.metric("Overall Relevance", f"{overall_relevance:.1f}%")
            
            # Export data with both scores and similarity analysis
            export_data = []
            # First 3 questions with similarity analysis
            for i, (question, sim_result, score) in enumerate(zip(first_three_questions, similarity_results, scores[:3]), 1):
                export_data.append(f"Q{i}. {question}")
                export_data.append(f"Overall Score: {score}")
                export_data.append(f"Similarity Score: {sim_result['relevance_score']:.2f}")
                export_data.append(f"Best Match: {sim_result['best_match']['title']}")
                export_data.append("")
            
            # Remaining questions with only scores
            for i, (question, score) in enumerate(zip(remaining_questions, scores[3:]), 4):
                export_data.append(f"Q{i}. {question}")
                export_data.append(f"Overall Score: {score}")
                export_data.append("")
            
            st.download_button(
                "Download Questions with Analysis",
                f"Job Role: {job_role}\nOverall Relevance: {overall_relevance:.1f}%\n\n" + "\n".join(export_data),
                file_name=f"{job_role.replace(' ', '_')}_questions_analysis.txt",
                mime="text/plain"
            )

if __name__ == "__main__":
    main()