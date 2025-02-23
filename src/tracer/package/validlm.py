import json
import os
import logging
import re
import subprocess
from functools import wraps

from tools.tools import verify_sql_query
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class ValidLM:
    """Validation & Logging System for LLM Applications"""

    PROJECTS_DIR = "projects"  # Define the directory for project files

    def __init__(self, project_name="default_project"):
        self.project_name = project_name
        self.project_file = os.path.join(self.PROJECTS_DIR, f"{project_name}.json")
        self.knowledge_base = None  # Could be a link, PDF, or CSV
        self._initialize_project()
        # self._start_streamlit_ui

    def _initialize_project(self):
        """Create an empty project file if it doesn't exist"""
        if not os.path.exists(self.project_file):
            initial_data = {
                "project_name": self.project_name,
                "assertions": {
                    "deterministic": [],
                    "misc": [],
                    "factual": False,
                    "sql-only": False,
                    "knowledgebase": None
                },
                "log_history": [],
                "accuracy_history": []
            }
            with open(self.project_file, "w") as f:
                json.dump(initial_data, f, indent=4)

    def _load_project(self):
        """Load the project data from the JSON file"""
        with open(self.project_file, "r") as f:
            return json.load(f)

    def _save_project(self, data):
        """Save the project data to the JSON file"""
        with open(self.project_file, "w") as f:
            json.dump(data, f, indent=4)

    def _start_streamlit_ui(self):
        """Start Streamlit UI in the background"""
        app_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "app.py"))

        # Start Streamlit without blocking the main thread
        subprocess.Popen(
            ["streamlit", "run", app_path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        print(f"✅ Streamlit UI started for project '{self.project_name}'")


    def add_assertion(self, assertion_type, assertion):
        """Add an assertion to the project file"""
        valid_types = {"deterministic", "factual", "misc", "sql-only", "knowledgebase"}
        if assertion_type not in valid_types:
            raise ValueError(f"Invalid assertion type. Choose from {valid_types}")

        project_data = self._load_project()
        if assertion_type in {"factual", "sql-only"}:
            project_data["assertions"][assertion_type] = assertion
        elif assertion_type == "knowledgebase":
            project_data["assertions"]["knowledgebase"] = assertion
        else:
            project_data["assertions"][assertion_type].append(assertion)

        self._save_project(project_data)
        logging.info(f"Added {assertion_type} assertion: {assertion}")

    def generate_clarifying_questions(self, user_input):
        """Generate clarifying questions using ChatGroq in JSON mode."""
        llm = ChatGroq(temperature=0, response_format="json")

        prompt = ChatPromptTemplate.from_template("""
        Given the user prompt: "{user_input}", generate clarifying multiple-choice questions
        to define constraints, preferences, and requirements.

        Example Output:
        [
            {
                "question": "What is the preferred programming language?",
                "options": ["Python", "Java", "C++"]
            },
            {
                "question": "Should the solution be optimized for speed?",
                "options": ["Yes", "No"]
            }
        ]

        Return ONLY valid JSON as per the format above.
        """)

        response = llm.predict(prompt.format(user_input=user_input))

        try:
            clarifying_questions = json.loads(response)
            self.clarifying_questions = clarifying_questions
            return clarifying_questions
        except json.JSONDecodeError:
            logging.error("Invalid JSON response from LLM.")
            self.clarifying_questions = []
            return []

    def verify_assertions(self, user_input, llm_output):


        """Run checks against stored assertions"""
        # 1. Deterministic
        # 2. Fact correction
        # 3. Misc check via llm
        # 4. Behaviour check

        project_data = self._load_project()
        assertions = project_data["assertions"]
        results = {"deterministic": [], "factual": [], "misc": []}

        # 🔵 Deterministic Assertions
        for assertion in assertions["deterministic"]:
            pattern = assertion.get("value")
            check_type = assertion.get("check_type")

            if check_type == "regex":
                match = re.search(pattern, llm_output) is not None
            elif check_type == "contains":
                match = pattern in llm_output
            elif check_type == "not-contains":
                match = pattern not in llm_output
            elif check_type == "json_format":
                try:
                    json.loads(llm_output)
                    match = True
                except json.JSONDecodeError:
                    match = False
            elif check_type == "sql_format":
                match = verify_sql_query(llm_output)
            else:
                match = False

            results["deterministic"].append((assertion, match))

        # 🟡 Factual Assertions ############################# use module 3
        if assertions["factual"] and assertions["knowledgebase"]:
            # Load and parse the knowledge base (PDF, etc.) here for comparison
            kb_path = assertions["knowledgebase"]
            # Placeholder for actual factual verification
            for fact in ["sample fact"]:
                match = fact in llm_output
                results["factual"].append((fact, match))
        else:
            results["factual"].append(("Knowledge Base Missing or Disabled", False))

        # 🟢 Miscellaneous Assertions
        for assertion in assertions["misc"]:    #########################
            validation = "complex check passed"  # Placeholder for complex checks
            results["misc"].append((assertion, validation))

        return results

    # def trace(self, func):
    #     """Decorator for tracing function calls and verifying LLM responses"""
    #     @wraps(func)
    #     def wrapper(*args, **kwargs):
    #         user_input = args[0] if args else None
    #         logging.info(f"Executing {func.__name__} with input: {user_input}")

    #         result = func(*args, **kwargs)
    #         logging.info(f"Received Output: {result}")

    #         verification_results = self.verify_assertions(user_input, result)
    #         logging.info(f"Verification Results: {verification_results}")

    #         # Update accuracy history
    #         project_data = self._load_project()
    #         project_data["accuracy_history"].append(verification_results)
    #         self._save_project(project_data)

    #         return result
    #     return wrapper
