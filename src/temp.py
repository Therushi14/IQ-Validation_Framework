from modules.module2_relevancy.relevance_analyzer import EnhancedRelevanceAnalyzer
if __name__ == "__main__":
    jd_text = """Looking for a Python developer with experience in Django, REST APIs,
    and cloud services like AWS or GCP. Familiarity with CI/CD pipelines and
    containerization using Docker is a plus."""

    questions = [
        "What is your experience with Django and REST APIs?",
        "Can you explain how Docker containers work?",
        "How do you implement CI/CD pipelines in cloud environments?",
        "What is your favorite programming language?"
    ]

    analyzer = EnhancedRelevanceAnalyzer()
    results = analyzer.calculate_question_scores(jd_text, questions)
    print(results)