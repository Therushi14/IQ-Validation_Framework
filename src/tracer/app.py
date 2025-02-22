import streamlit as st
import json
import os
from datetime import datetime
import pandas as pd



PROJECTS_DIR = "projects"

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
        "assertions": {"deterministic": [], "factual": [], "misc": []},
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
    st.header("✅ Assertions")
    assertion_type = st.selectbox("Assertion Type", ["deterministic", "factual", "misc"])
    new_assertion = st.text_input("Add New Assertion")
    if st.button("Add Assertion") and new_assertion:
        project["assertions"][assertion_type].append(new_assertion)
        save_project(project["project_name"], project)
        st.success("Assertion added.")

    st.subheader("Current Assertions")
    for a_type, assertions in project["assertions"].items():
        st.write(f"**{a_type.capitalize()} Assertions:**")
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
        accuracy = round(50 + 50 * (os.urandom(1)[0] / 255), 2)  # Random accuracy
        project["accuracy_history"].append([timestamp, accuracy])
        save_project(project["project_name"], project)
        st.experimental_rerun()
else:
    st.title("🔍 No Project Selected")
    st.write("Please create or open a project from the sidebar.")
