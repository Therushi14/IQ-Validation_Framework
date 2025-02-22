import streamlit as st
import json
import os
from datetime import datetime
import pandas as pd

PROJECTS_DIR = "projects"
DATASET_DIR = "dataset"

# Ensure projects directory exists
if not os.path.exists(PROJECTS_DIR):
    os.makedirs(PROJECTS_DIR)

# Helper Functions
def list_projects():
    return [f.replace(".json", "") for f in os.listdir(PROJECTS_DIR) if f.endswith(".json")]

def load_project(project_name):
    file_path = os.path.join(PROJECTS_DIR, f"{project_name}.json")
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            return json.load(f)
    return None

def save_project(project_name, data):
    file_path = os.path.join(PROJECTS_DIR, f"{project_name}.json")
    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)

def initialize_project(project_name):
    data = {
        "project_name": project_name,
        "assertions": {"deterministic": [], "misc": [], "factual": "", "contains-sql": False},
        "log_history": [],
        "accuracy_history": []
    }
    save_project(project_name, data)
    return data

# Streamlit UI
st.set_page_config(page_title="ValidLM Project Manager", layout="wide")
st.sidebar.title("📁 Project Manager")

# Sidebar - Project Management
project_action = st.sidebar.selectbox("Select Action", ["Create New Project", "Open Existing Project"])

if project_action == "Create New Project":
    new_project_name = st.sidebar.text_input("Enter Project Name")
    if st.sidebar.button("Create Project") and new_project_name:
        if new_project_name in list_projects():
            st.sidebar.error("Project with this name already exists.")
        else:
            project_data = initialize_project(new_project_name)
            st.session_state["current_project"] = project_data
            st.success(f"Project '{new_project_name}' created successfully!")

elif project_action == "Open Existing Project":
    existing_projects = list_projects()
    selected_project = st.sidebar.selectbox("Select Project", existing_projects)
    if st.sidebar.button("Open Project") and selected_project:
        project_data = load_project(selected_project)
        if project_data:
            st.session_state["current_project"] = project_data
        else:
            st.sidebar.error("Failed to load project.")

# Main Content
if "current_project" in st.session_state:
    project = st.session_state["current_project"]

    st.title(f"📊 Project: {project['project_name']}")

    # Assertions Section
    st.header("Add new assertions")
    assertion_type = st.selectbox("Assertion Type", ["deterministic", 'factual', "misc"])

    if assertion_type == "deterministic":
        check_type = st.selectbox("Select Deterministic Check Type", ["regex", "json_format", "contains", "not-contains"])
        check_value = st.text_area("Enter pattern")
        if st.button("Add Deterministic Assertion") and check_value:
            assertion_data = {
                "check_type": check_type,
                "value": check_value,
            }
            project["assertions"]["deterministic"].append(assertion_data)
            save_project(project["project_name"], project)
            st.success("Deterministic Assertion added.")

    elif assertion_type == "factual":
        fact = st.file_uploader("Provide knowledgebase for factual assertion", type=["pdf", "docx"])
        if st.button("Add") and fact:
            project_id = project["project_name"]
            file_extension = os.path.splitext(fact.name)[1]
            # current working dir
            saved_path = os.path.join(os.getcwd(), DATASET_DIR, f"{project_id}{file_extension}")
            with open(saved_path, "wb") as f:
                f.write(fact.getbuffer())
            project["assertions"]["knowledgebase"] = saved_path
            st.success("Factual Assertion added and file saved.")

    elif assertion_type == "misc":
        new_assertion = st.text_input("Add Miscellaneous Assertion")
        if st.button("Add Miscellaneous Assertion") and new_assertion:
            project["assertions"]["misc"].append(new_assertion)
            save_project(project["project_name"], project)
            st.success("Miscellaneous Assertion added.")

    st.subheader("Current Assertions")
    for a_type, assertions in project["assertions"].items():
        if (a_type == 'factual' or a_type == "contains-sql"):
            st.write(f"**{a_type.capitalize()}: {assertions}**")
            continue
        st.write(f"**{a_type.capitalize()} Assertions:**" if len(assertions) > 0 else "") 
        for assertion in assertions:
            st.write(f"- {assertion}")

    # Log History
    st.header("📝 Application Log History")
    if project["log_history"]:
        log_df = pd.DataFrame(project["log_history"], columns=["Timestamp", "Event"])
        st.dataframe(log_df)
    else:
        st.write("No logs available.")

    # Accuracy History
    st.header("📈 Accuracy History")
    if project["accuracy_history"]:
        acc_df = pd.DataFrame(project["accuracy_history"], columns=["Timestamp", "Accuracy"])
        st.line_chart(acc_df.set_index("Timestamp"))
    else:
        st.write("No accuracy data available.")

    # Simulate Log & Accuracy Updates
    if st.button("Simulate Log Entry"):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        project["log_history"].append([timestamp, "Sample log event."])
        save_project(project["project_name"], project)
        st.experimental_rerun()

    if st.button("Simulate Accuracy Update"):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        accuracy = round(50 + 50 * (os.urandom(1)[0] / 255), 2)
        project["accuracy_history"].append([timestamp, accuracy])
        save_project(project["project_name"], project)
        st.experimental_rerun()
else:
    st.title("🔍 No Project Selected")
    st.write("Please create or open a project from the sidebar.")
