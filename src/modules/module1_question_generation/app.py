import streamlit as st
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import datetime

# Adjust the system path to find project modules
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
sys.path.append(project_root)

from src.modules.module2_relevancy.relevance_analyzer import EnhancedRelevanceAnalyzer
from groq_client import GroqClient
from file_processing import extract_text_from_file
from src.modules.module3_compare.model import QuestionSimilarityModel
from src.modules.module4_bias.bias import screen_questions
from src.modules.module1_question_generation.project_controller import Project

project_control = Project()
if 'page' not in st.session_state:
    st.session_state.page = 'main'
if ('accuracy_history' not in st.session_state):
    st.session_state['accuracy_history'] = {
                "DSA" : [],
                "Technical" : [],
                "Behaviour": []
            }

def main():
    
    if st.session_state.page == 'main':
        sidebar()
        if ('current_project' in st.session_state):
            if (st.session_state['current_project']['project_name'] == 'default'):
                st.title("AI Interview Question Generator & Analyzer")
            main_page()
        else:
            st.subheader('No project selected')
    elif st.session_state.page == 'configure':
        configure_page()

def sidebar():
    st.sidebar.title("Project Options")
    project_action = st.sidebar.selectbox("Select Action", ["Create New Project", "Open Existing Project"])
    if project_action == "Create New Project":
        new_project_name = st.sidebar.text_input("Enter Project Name")
        print('Title: ', new_project_name)
        if st.sidebar.button("Create Project") and new_project_name:
            if new_project_name in project_control.list_projects():
                st.sidebar.error("Project with this name already exists.")
            else:
                project_data = project_control.initialize_project(new_project_name)
                st.session_state["current_project"] = project_data
                st.success(f"Project '{new_project_name}' created successfully!")

    elif project_action == "Open Existing Project":
        existing_projects = project_control.list_projects()
        selected_project = st.sidebar.selectbox("Select Project", existing_projects)
        if st.sidebar.button("Open Project") and selected_project:
            project_data = project_control.load_project(selected_project)
            if project_data:
                st.session_state["current_project"] = project_data
            else:
                st.sidebar.error("Failed to load project_control.")

def main_page():
    client = GroqClient()
    analyzer = EnhancedRelevanceAnalyzer()
    similarity_model = QuestionSimilarityModel('dataset/leetcode_dataset.csv')
    project = st.session_state["current_project"]
    
    st.subheader('Project: ', project['project_name'])

    job_role = st.text_input("Enter Job Role")
    question_type = st.selectbox("Type of questions", ["DSA", "Technical", "Behaviour"])
    jd_file = st.file_uploader("Upload Job Description (PDF/DOCX)", type=["pdf", "docx"])
    

    if jd_file and job_role and question_type and st.button('Get questions') :
        with st.spinner("Analyzing Job Description..."):
            jd_text = extract_text_from_file(jd_file)

            if not analyzer.check_title_jd_match(job_role, jd_text):
                st.error("⚠️ Job description doesn't match the job title! Upload a relevant JD.")
                st.stop()

            questions = client.generate_questions(job_role, jd_text, question_type)
            question_lines = [q.strip() for q in questions.split('\n') if q.strip()]
            if question_lines and not question_lines[0][0].isdigit():
                question_lines = question_lines[1:]

            # first_five_questions = question_lines[:10]
            # remaining_questions = question_lines[5:15]
            scores = []
            if (question_type == "DSA"): 
                similarity_results = similarity_model.check_similarity(question_lines)
                scores = similarity_results
                st.subheader("DSA questions with similarity analysis")
                score = 0
                for i, (question, result) in enumerate(zip(question_lines, similarity_results), 1):
                    st.write(f"{i}. {question}")
                    score += result["relevance_score"]
                    with st.expander(f"Similarity Analysis for Question {i}"):
                        st.write(f"Similarity Score: {result['relevance_score']:.2f}")
                        st.write(f"Best Match: {result['best_match']['title']}")
                        st.write(f"Difficulty: {result['best_match']['difficulty']}")
                        if result['matched_sources']:
                            st.write("\nSimilar Questions:")
                            for source in result['matched_sources']:
                                st.write(f"- {source['title']} (Difficulty: {source['difficulty']})")
                overall_similarity = score / len(question_lines)
                st.metric("Overall Relevance", f"{overall_similarity*100:.1f}%")
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                project['accuracy_history'][question_type].append((timestamp, overall_similarity))

            if (question_type == "Technical" or question_type == "Behaviour"):
                
                for q in question_lines:
                    st.write(f"- {q}")

            if (question_type == "Technical"):
                scores = analyzer.calculate_question_scores(jd_text, question_lines)
                avg_score = sum(scores) / len(scores)

                half_avg = avg_score / 1.25
                count_above_half = sum(1 for s in scores if s > half_avg)
                overall_relevance = (count_above_half / len(scores)) * 100

                st.subheader("Analysis Results")
                st.metric("Overall Relevance", f"{overall_relevance:.1f}%")

                # Store accuracy with timestamp
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                project['accuracy_history'][question_type].append((timestamp, overall_relevance))

            if question_type == "Behaviour": 
                valid_bias_questions, invalid_bias_questions, bias_accuracy = screen_questions(question_lines)
                st.metric("Bias Accuracy", f"{bias_accuracy * 100:.1f}%")

            # Plot accuracy history
            if project['accuracy_history']:
                st.subheader("Accuracy History")
                timestamps, accuracies = zip(*project['accuracy_history'][question_type])
                fig, ax = plt.subplots()
                ax.plot(timestamps, accuracies, marker='o')
                ax.set_xlabel("Timestamp")
                ax.set_ylabel("Overall Relevance (%)")
                ax.set_title("Relevance Over Time")
                plt.xticks(rotation=45)
                st.pyplot(fig)

            export_data = []
            for i, (question, score) in enumerate(zip(question_lines, scores), 1):
                export_data.append(f"Q{i}. {question}")
                if (question_type == "DSA"):
                    export_data.append(f"Overall Score: {score['relevance_score']}")
                    export_data.append(f"Best Match: {score['best_match']['title']}")
                else:
                    export_data.append(f"Overall Score: {score}")
                export_data.append("")

            # for i, (question, score) in enumerate(zip(remaining_questions, scores[5:15]), 5):
            #     export_data.append(f"Q{i}. {question}")
            #     export_data.append("")

            st.download_button(
                "Download Questions with Analysis",
                f"Job Role: {job_role}\nOverall Relevance: {overall_relevance:.1f}%\n\n" + "\n".join(export_data),
                file_name=f"{job_role.replace(' ', '_')}_questions_analysis.txt",
                mime="text/plain"
            )

def configure_page():
    st.title("Project Configuration")

    assertion_type = st.selectbox("Select Assertion Type", ["deterministic", "factual", "misc"])
    check_type = st.text_input("Check Type (e.g., contains, regex, json)")
    value = st.text_input("Assertion Value")

    if st.button("Save Assertion"):
        st.success(f"Assertion saved: {assertion_type} - {check_type}: {value}")

    if st.button("Go Back"):
        st.session_state.page = 'main'

if __name__ == "__main__":
    main()
