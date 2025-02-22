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
        Generate 10-15 comprehensive interview questions for a {job_role} position. 
        Include both technical and behavioral questions.
        Focus on these key aspects from the job description:
        {job_description}
        
        Format requirements:
        - Each question should be numbered
        - Separate technical and behavioral sections
        - Avoid any markdown formatting
        """