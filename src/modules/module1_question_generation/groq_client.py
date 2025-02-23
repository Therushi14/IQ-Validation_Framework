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

    def generate_questions(self, job_role, job_description):
        prompt = self._build_prompt(job_role, job_description)
        
        response = self.client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        
        return response.choices[0].message.content

    def _build_prompt(self, job_role, job_description):
        return f"""
        Generate 20 comprehensive interview questions for a {job_role} position.
        First 5 questions must be based on DSA concepts
        Next 10 questions must only and only be technical based on {job_role}
        Last 5 questions must be behavioural questions only. They must be behavioural.
        Include both technical and behavioral questions.
        Focus on these key aspects from the job description:
        {job_description}

        For technical questions:
        - Analyze the role and job description to determine the relevant technical domains and skills required.
        - Generate questions that assess role-specific technical competencies, such as coding challenges, data structures, algorithms, system design, analytical reasoning, or other domain-specific problem solving.
        - Ensure these questions reflect real-world scenarios and practical challenges pertinent to the position.

        For behavioral questions:
        - Include questions that evaluate soft skills, teamwork, communication, problem-solving approaches, and cultural fit.

        Format requirements:
        1. Each question must be numbered starting with 'Q1'
        2. Put each question on a new line
        3. First list technical questions, then behavioral
        4. Do not include any section headers
        """