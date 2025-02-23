import re
from keybert import KeyBERT
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from module2_relevancy.relevance_analyzer import EnhancedRelevanceAnalyzer

class MultiMethodAnalyzer:
    """Analyzer using KeyBERT, BM25, TF-IDF, Jaccard, and SBERT for relevance ranking."""

    def __init__(self):
        self.keyword_extractor = KeyBERT()
        self.sbert_model = SentenceTransformer('all-MiniLM-L6-v2')

    def extract_keywords(self, text, max_keywords=20):
        """Extract and tokenize top keywords using KeyBERT."""
        keywords = self.keyword_extractor.extract_keywords(
            text,
            keyphrase_ngram_range=(1, 2),
            stop_words='english',
            top_n=max_keywords
        )
        return [kw[0].lower().split() for kw in keywords]

    def calculate_bm25_score(self, jd_keywords, question_keywords):
        jd_flat = [word for phrase in jd_keywords for word in phrase]
        question_flat = [word for phrase in question_keywords for word in phrase]

        if not jd_flat or not question_flat:
            return 0.0

        bm25 = BM25Okapi([jd_flat])
        score = bm25.get_scores(question_flat)[0]
        return round(max(score, 0) * 10, 2)

    def calculate_tfidf_score(self, jd_text, question_text):
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform([jd_text, question_text])
        score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        return round(score * 100, 2)

    def calculate_jaccard_score(self, jd_keywords, question_keywords):
        jd_set = set(word for phrase in jd_keywords for word in phrase)
        question_set = set(word for phrase in question_keywords for word in phrase)
        intersection = jd_set.intersection(question_set)
        union = jd_set.union(question_set)
        score = len(intersection) / len(union) if union else 0
        return round(score * 100, 2)

    def calculate_sbert_score(self, jd_text, question_text):
        jd_embedding = self.sbert_model.encode(jd_text, convert_to_tensor=True)
        question_embedding = self.sbert_model.encode(question_text, convert_to_tensor=True)
        score = util.pytorch_cos_sim(jd_embedding, question_embedding).item()
        return round(score * 100, 2)

    def analyze_questions(self, job_description, questions):
        jd_keywords = self.extract_keywords(job_description)
        results = []

        for question in questions:
            question_keywords = self.extract_keywords(question)
            bm25_score = self.calculate_bm25_score(jd_keywords, question_keywords)
            tfidf_score = self.calculate_tfidf_score(job_description, question)
            jaccard_score = self.calculate_jaccard_score(jd_keywords, question_keywords)
            sbert_score = self.calculate_sbert_score(job_description, question)
            results.append({
                'question': question,
                'BM25': bm25_score,
                'TF-IDF': tfidf_score,
                'Jaccard': jaccard_score,
                'SBERT': sbert_score
            })

        return results

# Example usage
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
    # for i, res in enumerate(results):
    #     print(f"Question {i+1}: {res['question']}")
    #     print(f"  BM25 Score: {res['BM25']}%")
    #     print(f"  TF-IDF Score: {res['TF-IDF']}%")
    #     print(f"  Jaccard Score: {res['Jaccard']}%")
    #     print(f"  SBERT Score: {res['SBERT']}%\n")
