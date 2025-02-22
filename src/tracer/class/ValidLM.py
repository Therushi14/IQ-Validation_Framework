# un-tested

import json
import os
import logging
import re
import subprocess
from functools import wraps

from langchain.chat_models import ChatGroq

from langchain.prompts import ChatPromptTemplate

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class ValidLM:
    """Validation & Logging System for LLM Applications"""

    def __init__(self, project_name="default_project"):
        self.project_name = project_name
        self.assertion_file = f"{project_name}_assertions.json"
        self.knowledge_base = None  # Could be a link, PDF, or CSV
        self._initialize_assertions()
        self._start_streamlit_ui()

    def _initialize_assertions(self):
        """Create an empty assertions file if not exists"""
        if not os.path.exists(self.assertion_file):
            with open(self.assertion_file, "w") as f:
                json.dump({"deterministic": [], "factual": [], "misc": []}, f)

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

    def _load_assertions(self):
        """Load assertions from the JSON file"""
        with open(self.assertion_file, "r") as f:
            return json.load(f)

    def _save_assertions(self, assertions):
        """Save assertions to the JSON file"""
        with open(self.assertion_file, "w") as f:
            json.dump(assertions, f, indent=4)

    def add_assertion(self, assertion_type, assertion):
        """Add an assertion to the JSON file"""
        valid_types = {"deterministic", "factual", "misc"}
        if assertion_type not in valid_types:
            raise ValueError(f"Invalid assertion type. Choose from {valid_types}")

        assertions = self._load_assertions()
        assertions[assertion_type].append(assertion)
        self._save_assertions(assertions)
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
    
        # Parse JSON response
        try:
            clarifying_questions = json.loads(response)
            self.clarifying_questions = clarifying_questions
            return clarifying_questions
        except json.JSONDecodeError:
            logging.error("Invalid JSON response from LLM.")
            self.clarifying_questions = []
            return []


    def predefined_output_format_questions():
        """Predefined questions about the output format for deterministic validation"""
        return [
            "Specify JSON format of the output",
            "Specify required keyword in the output",
            "Should the output be in SQL format?",
            "Specify the regex format of the output"
        ]
    
    def verify_assertions(self, user_input, llm_output):
        """Run checks against stored assertions"""
        assertions = self._load_assertions()
        results = self.clarifying_questions

        # 🔵 Deterministic Assertions (Regex, format-based)
        for assertion in assertions["deterministic"]:
            pattern = assertion.get("pattern")
            check_type = assertion.get("type")  # regex, contains, etc.

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
                match = llm_output.strip().lower().startswith("select")
            else:
                match = False

            results["deterministic"].append((assertion, match))

            # 🟡 Factual Assertions (Knowledge Base Parsing)
            if self.knowledge_base:
                for assertion in assertions["factual"]:
                    fact = assertion.get("fact")
                    match = fact in self.knowledge_base  # Simplified check
                    results["factual"].append((fact, match))
            else:
                for assertion in assertions["factual"]:
                    results["factual"].append((assertion, "Knowledge Base Missing"))

            # 🟢 Miscellaneous Assertions (LLM Verification)
            for assertion in assertions["misc"]:
                # Example: Using LLM for context validation (placeholder)
                validation = "complex check passed"  # Replace with LLM call if needed
                results["misc"].append((assertion, validation))

            return results
    
    def trace(self, func):
        """Decorator for tracing function calls and verifying LLM responses"""
        @wraps(func)
        def wrapper(*args, **kwargs):
            user_input = args[0] if args else None
            logging.info(f"Executing {func.__name__} with input: {user_input}")

            result = func(*args, **kwargs)  # Run LLM function
            logging.info(f"Received Output: {result}")

            # Verify assertions
            verification_results = self.verify_assertions(user_input, result)
            logging.info(f"Verification Results: {verification_results}")

            return result
        return wrapper

