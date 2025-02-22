import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from rake_nltk import Rake
import importlib.util
import sys
import subprocess
import logging
import re

class EnhancedRelevanceAnalyzer:
    """
    A class for analyzing the relevance of interview questions against job descriptions
    using multiple NLP techniques and scoring mechanisms.
    """
    
    def __init__(self):
        """Initialize the analyzer with necessary models and vectorizers."""
        self.tfidf = TfidfVectorizer(
            stop_words='english', 
            ngram_range=(1, 3),
            max_features=5000
        )
        self.semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.keyword_extractor = Rake()
        
        # Initialize spaCy with proper error handling
        self.nlp = self._initialize_spacy()
        
    def _initialize_spacy(self):
        """Initialize spaCy with proper error handling and installation if needed."""
        try:
            import spacy
            try:
                return spacy.load('en_core_web_sm')
            except OSError:
                print("Downloading required spaCy model...")
                subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"], check=True)
                return spacy.load('en_core_web_sm')
        except ImportError:
            print("Installing required dependencies...")
            subprocess.run([sys.executable, "-m", "pip", "install", "spacy"], check=True)
            import spacy
            subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"], check=True)
            return spacy.load('en_core_web_sm')
        except Exception as e:
            print(f"Warning: Could not initialize spaCy ({str(e)}). Falling back to basic analysis.")
            return None

    def calculate_question_scores(self, job_description, questions):
        """
        Calculate relevance scores for a list of questions against a job description.
        
        Args:
            job_description (str): The job description text
            questions (list): List of question strings to analyze
            
        Returns:
            list: List of relevance scores (0-100) for each question
        """
        # Extract key phrases using RAKE
        self.keyword_extractor.extract_keywords_from_text(job_description)
        jd_keywords = set(self.keyword_extractor.get_ranked_phrases()[:20])
        
        # Extract entities if spaCy is available
        jd_entities = set()
        if self.nlp:
            jd_doc = self.nlp(job_description)
            jd_entities = set([ent.text.lower() for ent in jd_doc.ents])
        
        # Clean and prepare texts
        jd_clean = self._clean_text(job_description)
        questions_clean = [self._clean_text(q) for q in questions]
        
        # Calculate scores for each question
        scores = []
        for i, question in enumerate(questions):
            # Calculate base scores
            tfidf_score = self._calculate_tfidf_score(jd_clean, questions_clean[i])
            semantic_score = self._calculate_semantic_score(jd_clean, questions_clean[i])
            keyword_score = self._calculate_keyword_score(jd_keywords, question)
            
            # Calculate additional scores if spaCy is available
            if self.nlp:
                entity_score = self._calculate_entity_score(jd_entities, question)
                context_score = self._calculate_context_score(job_description, question)
                
                # Combine all scores with weights
                weighted_score = (
                    tfidf_score * 0.15 +      # Term frequency importance
                    semantic_score * 0.35 +    # Semantic meaning importance
                    keyword_score * 0.20 +     # Keyword matching importance
                    entity_score * 0.15 +      # Named entity importance
                    context_score * 0.15       # Contextual relevance importance
                )
            else:
                # Fallback scoring without spaCy-dependent components
                weighted_score = (
                    tfidf_score * 0.25 +
                    semantic_score * 0.45 +
                    keyword_score * 0.30
                )
            
            # Normalize and boost the final score
            final_score = self._normalize_and_boost_score(weighted_score)
            scores.append(final_score)
            
        return [round(score * 100, 2) for score in scores]
    
    def _calculate_tfidf_score(self, jd_text, question):
        """Calculate TF-IDF based similarity score."""
        tfidf_matrix = self.tfidf.fit_transform([jd_text, question])
        return cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    
    def _calculate_semantic_score(self, jd_text, question):
        """Calculate semantic similarity using sentence transformers."""
        jd_embedding = self.semantic_model.encode([jd_text], convert_to_tensor=True)
        question_embedding = self.semantic_model.encode([question], convert_to_tensor=True)
        return cosine_similarity(jd_embedding, question_embedding)[0][0]
    
    def _calculate_keyword_score(self, jd_keywords, question):
        """Calculate keyword overlap score."""
        question_words = set(self._clean_text(question).split())
        overlap = len(jd_keywords & question_words)
        # Adjusted threshold for better scoring
        return min(1.0, overlap / max(len(jd_keywords) * 0.3, 1))
    
    def _calculate_entity_score(self, jd_entities, question):
        """Calculate named entity overlap score."""
        if not self.nlp:
            return 0.0
        question_doc = self.nlp(question)
        question_entities = set([ent.text.lower() for ent in question_doc.ents])
        overlap = len(jd_entities & question_entities)
        return min(1.0, overlap / max(len(jd_entities) * 0.2, 1))
    
    def _calculate_context_score(self, job_description, question):
        """Calculate contextual relevance score using noun phrases."""
        if not self.nlp:
            return 0.0
        jd_doc = self.nlp(job_description)
        question_doc = self.nlp(question)
        
        # Extract noun phrases
        jd_phrases = set([chunk.text.lower() for chunk in jd_doc.noun_chunks])
        question_phrases = set([chunk.text.lower() for chunk in question_doc.noun_chunks])
        
        # Calculate phrase overlap with boosting
        phrase_overlap = len(jd_phrases & question_phrases) / max(len(jd_phrases), 1)
        return min(1.0, phrase_overlap * 1.5)
    
    def _normalize_and_boost_score(self, score):
        """Apply normalization and boosting to the final score."""
        # Sigmoid normalization to compress extreme values
        normalized = 1 / (1 + np.exp(-5 * (score - 0.5)))
        
        # Apply boosting for scores above threshold
        if score > 0.3:
            boost_factor = 1.2
            normalized = min(1.0, normalized * boost_factor)
        
        return normalized
    
    def _clean_text(self, text):
        """Clean and normalize text with technical term handling."""
        # Basic cleaning
        text = re.sub(r'[^\w\s-]', '', text.lower())
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Handle common technical terms and abbreviations
        tech_mappings = {
            'js': 'javascript',
            'py': 'python',
            'ml': 'machine learning',
            'ai': 'artificial intelligence',
            'dl': 'deep learning',
            'nlp': 'natural language processing',
            'db': 'database',
            'ui': 'user interface',
            'ux': 'user experience',
            'api': 'application programming interface',
            'oop': 'object oriented programming',
            'ci': 'continuous integration',
            'cd': 'continuous deployment',
            'aws': 'amazon web services',
            'azure': 'microsoft azure',
            'gcp': 'google cloud platform'
        }
        
        words = text.split()
        cleaned_words = [tech_mappings.get(word, word) for word in words]
        return ' '.join(cleaned_words)