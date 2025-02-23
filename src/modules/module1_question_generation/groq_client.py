from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()

class GroqClient:
    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY")  
        if not api_key:
            raise ValueError("API key not found. Please set GROQ_API_KEY in the .env file.")

        self.client = Groq(api_key=api_key)

    def generate_questions(self, job_role, job_description, type):
        prompt = self._build_prompt(job_role, job_description, type)
       
        response = self.client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        
        return response.choices[0].message.content

    def _build_prompt(self, job_role, job_description, type):
        prompt = ""
        if type == "DSA":
            prompt = f"""Generate 10 comprehensive interview questions for a {job_role} position.
                These questions must focus only on DSA and comprise of various difficulty levels
            """
        elif type == "Technical":
            prompt = f"""Generate 10 comprehensive interview questions for a {job_role} position.
            These questions must focus on technical skills of the job role of {job_role} and comprise of various difficulty levels
            Focus on key aspects from the below job description: {job_description}
            """
        elif type == "Behaviour":
            prompt = f"""Generate 10 comprehensive interview questions for a {job_role} position.
            These questions must focus on behavioural skills of the job role of {job_role} and comprise of
            various difficulty levels
            Focus on key aspects from the below job description: {job_description}
            """
        return prompt + """
        Format requirements:
        1. Each question must be numbered starting with 'Q1'
        2. Put each question on a new line
        3. First list technical questions, then behavioral
        4. Do not include any section headers"""
       