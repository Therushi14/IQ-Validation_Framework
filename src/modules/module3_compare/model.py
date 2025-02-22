import pandas as pd
import numpy as np
import os
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import nltk
from nltk.tokenize import word_tokenize


class QuestionSimilarityModel:
    def __init__(self, dataset_path, cache_path='embeddings_cache.pkl'):
        self.dataset_path = dataset_path
        self.cache_path = cache_path
        self.dataset = pd.read_csv(dataset_path)
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embeddings = self._load_or_generate_embeddings()

    def _generate_embeddings(self, questions):
        combined_text = questions.apply(lambda x: f"{x['title']} Difficulty: {x['difficulty']}", axis=1)
        return self.model.encode(combined_text.tolist(), convert_to_tensor=True)

    def _load_or_generate_embeddings(self):
        if os.path.exists(self.cache_path):
            with open(self.cache_path, 'rb') as f:
                print("Loading cached embeddings...")
                return pickle.load(f)
        else:
            print("Generating new embeddings...")
            embeddings = self._generate_embeddings(self.dataset)
            with open(self.cache_path, 'wb') as f:
                pickle.dump(embeddings, f)
            return embeddings

    def _preprocess(self, text):
        tokens = word_tokenize(text.lower())
        return ' '.join(tokens)

    def check_similarity(self, new_questions):
        results = []
        for question in new_questions:
            preprocessed = self._preprocess(question)
            new_embedding = self.model.encode(preprocessed, convert_to_tensor=True)
            similarities = cosine_similarity([new_embedding], self.embeddings)[0]
            max_score = np.max(similarities)
            max_index = np.argmax(similarities)
            matched_indices = np.where(similarities >= 0.7)[0]  # Threshold for strong match
            matched_sources = self.dataset.iloc[matched_indices][['title', 'difficulty']].to_dict('records')
            best_match = self.dataset.iloc[max_index]
            results.append({
                'input_question': question,
                'relevance_score': float(max_score),
                'matched_sources': matched_sources,
                'best_match': {
                    'index': int(max_index),
                    'title': best_match['title'],
                    'difficulty': best_match['difficulty']
                }
            })
        return results

# Example usage:
model = QuestionSimilarityModel('dataset/leetcode_dataset.csv')
new_questions = ["If john throws a ball at the speed of 5m/s vertically upwards, find the time it first reaches ground", "Implement a binary search algorithm"]
result = model.check_similarity(new_questions)
print(result)
