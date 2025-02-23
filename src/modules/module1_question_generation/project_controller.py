
PROJECTS_DIR = "projects"
DATASET_DIR = "dataset"
import json
import os

class Project:
    def __init__(self):
        pass
    def list_projects(self):
        return [f.replace(".json", "") for f in os.listdir(PROJECTS_DIR) if f.endswith(".json")]

    def load_project(self,project_name):
        file_path = os.path.join(PROJECTS_DIR, f"{project_name}.json")
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                return json.load(f)
        return None

    def save_project(self,project_name, data):
        file_path = os.path.join(PROJECTS_DIR, f"{project_name}.json")
        with open(file_path, "w") as f:
            json.dump(data, f, indent=4)

    def initialize_project(self,project_name):
        data = {
            "project_name": project_name,
            "assertions": {"deterministic": [], "misc": [], "factual": "", "contains-sql": False},
            "log_history": [],
            "accuracy_history": {
                "DSA" : [],
                "Technical" : [],
                "Behaviour": []
            },
         
        }
        self.save_project(project_name, data)
        return data