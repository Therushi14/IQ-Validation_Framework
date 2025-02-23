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
from src.modules.module4_bias.bias import screen_questions  # Import bias screening function

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
            first_five_questions = question_lines[:5]
            remaining_questions = question_lines[5:15]
            
            # Get similarity analysis for first 3 questions
            similarity_results = similarity_model.check_similarity(first_five_questions)
            
            # Display questions with similarity analysis
            st.subheader("First 3 Questions with Similarity Analysis")
            for i, (question, result) in enumerate(zip(first_five_questions, similarity_results), 1):
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
            for i, q in enumerate(remaining_questions, 5):
                st.write(f"{i}. {q}")
            
            # Analyze relevance for all questions combined
            scores = analyzer.calculate_question_scores(jd_text, question_lines)
            avg_score = sum(scores) / len(scores)
            
            half_avg = avg_score / 1.25
            count_above_half = sum(1 for s in scores if s > half_avg)
            overall_relevance = (count_above_half / len(scores)) * 100
            
            st.subheader("Analysis Results")
            st.metric("Overall Relevance", f"{overall_relevance:.1f}%")
            
            # ---- Bias Analysis Integration ----
            # Extract the last 4 questions from the full list (or all if fewer than 4)
            last_four_questions = question_lines[-5:] if len(question_lines) >= 5 else question_lines
            
            valid_bias_questions, invalid_bias_questions, bias_accuracy = screen_questions(last_four_questions)
            
            st.subheader("Bias Analysis for Last 5 Questions")
            st.write("**Last 5 Generated Questions:**")
            for q in last_four_questions:
                st.write(f"- {q}")
            
            st.metric("Bias Accuracy", f"{bias_accuracy * 100:.1f}%")
            # ---- End Bias Analysis ----
            
            # Export data with both scores and similarity analysis
            export_data = []
            # First 5 questions with similarity analysis
            for i, (question, sim_result, score) in enumerate(zip(first_five_questions, similarity_results, scores[:3]), 1):
                export_data.append(f"Q{i}. {question}")
                export_data.append(f"Overall Score: {score}")
                export_data.append(f"Best Match: {sim_result['best_match']['title']}")
                export_data.append("")
            
            # Remaining questions with only scores
            for i, (question, score) in enumerate(zip(remaining_questions, scores[5:15]), 5):
                export_data.append(f"Q{i}. {question}")
                # export_data.append(f"Overall Score: {score}")
                export_data.append("")
            
            st.download_button(
                "Download Questions with Analysis",
                f"Job Role: {job_role}\nOverall Relevance: {overall_relevance:.1f}%\n\n" + "\n".join(export_data),
                file_name=f"{job_role.replace(' ', '_')}_questions_analysis.txt",
                mime="text/plain"
            )

if __name__ == "__main__":
    main()
