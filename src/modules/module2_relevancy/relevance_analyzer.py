import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from rake_nltk import Rake
import nltk
import importlib.util
import sys
import subprocess
import logging
import re
import os

class NLTKResourceManager:
    """Manages NLTK resource initialization and verification"""
    
    REQUIRED_RESOURCES = [
        ('tokenizers/punkt', 'punkt'),
        ('corpora/stopwords', 'stopwords'),
        ('tokenizers/punkt_tab', 'punkt_tab'),
        ('taggers/averaged_perceptron_tagger_eng', 'averaged_perceptron_tagger_eng')
    ]
    
    @staticmethod
    def initialize_nltk_resources() -> None:
        """Initialize all required NLTK resources with proper error handling"""
        
        def verify_resource(resource_path: str) -> bool:
            try:
                nltk.data.find(resource_path)
                return True
            except LookupError:
                return False
        
        # Create nltk_data directory in user's home if it doesn't exist
        nltk_data_dir = os.path.expanduser('~/nltk_data')
        os.makedirs(nltk_data_dir, exist_ok=True)
        
        # Ensure NLTK uses the correct data directory
        nltk.data.path.append(nltk_data_dir)
        
        # Download missing resources
        for resource_path, resource_name in NLTKResourceManager.REQUIRED_RESOURCES:
            if not verify_resource(resource_path):
                print(f"Downloading {resource_name}...")
                nltk.download(resource_name, quiet=True)
                
                # Verify successful download
                if not verify_resource(resource_path):
                    raise RuntimeError(f"Failed to download NLTK resource: {resource_name}")
                
        print("All NLTK resources successfully initialized")

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
        NLTKResourceManager.initialize_nltk_resources()
        self.semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.keyword_extractor = Rake(min_length=1, max_length=1)
        
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
        
    def check_title_jd_match(self, job_title, jd_text, threshold=0.45):
        """Check semantic match between job title and JD using sentence transformers"""
        title_embed = self.semantic_model.encode([job_title], convert_to_tensor=True)
        jd_embed = self.semantic_model.encode([jd_text[:5000]], convert_to_tensor=True)  # Use first 5000 chars for efficiency
        similarity = cosine_similarity(title_embed, jd_embed)[0][0]
        return similarity >= threshold
    
    def extract_single_keywords(self, text):
        """Extract single keywords using RAKE + NLTK POS tagging."""
        # Step 1: RAKE for initial extraction
        self.keyword_extractor.extract_keywords_from_text(text)
        rake_phrases = set(self.keyword_extractor.get_ranked_phrases()[:20])

        # Step 2: Use NLTK POS tagging for single-word nouns and proper nouns
        words = nltk.word_tokenize(text)
        pos_tags = nltk.pos_tag(words)

        # Select nouns, proper nouns, and technical terms
        single_keywords = {word.lower() for word, tag in pos_tags if tag in ['NN', 'NNS', 'NNP', 'NNPS']}

        # Step 3: Combine RAKE keywords and POS nouns
        combined_keywords = single_keywords.union(set(rake_phrases))

        return combined_keywords
    
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
        # self.keyword_extractor.extract_keywords_from_text(job_description)
        # jd_keywords = set(self.keyword_extractor.get_ranked_phrases()[:20])
        jd_keywords = self.extract_single_keywords(job_description)
        print('HEYY')
        print(jd_keywords)
        # Extract entities if spaCy is available
        jd_entities = set()
        if self.nlp:
            jd_doc = self.nlp(job_description)
            jd_entities = set([ent.text.lower() for ent in jd_doc.ents])
        
        # Clean and prepare texts
        jd_clean = self._clean_text(job_description)
        questions_clean = [self._clean_text(q) for q in questions]
        
        # Calculate scores for each question
        # scores = []
        results = []
        for i, question in enumerate(questions):
            # Calculate base scores
            tfidf_score = self._calculate_tfidf_score(jd_clean, questions_clean[i])
            semantic_score = self._calculate_semantic_score(jd_clean, questions_clean[i])
            keyword_score = self._calculate_keyword_score(jd_keywords, question)
            
            question_words = set(self._clean_text(question).split())
            keyword_overlap = len(jd_keywords & question_words)
            matched_keywords = jd_keywords & question_words
            print('Matched- ', matched_keywords)
            # Calculate additional scores if spaCy is available
            if self.nlp:
                entity_score = self._calculate_entity_score(jd_entities, question)
                context_score = self._calculate_context_score(job_description, question)
                
                # Combine all scores with weights
                weighted_score = (
                    tfidf_score * 0.20 +      # Term frequency importance
                    semantic_score * 0.30 +    # Semantic meaning importance
                    keyword_score * 0.40 +     # Keyword matching importance
                    entity_score * 0.05 +      # Named entity importance
                    context_score * 0.05       # Contextual relevance importance
                )
            else:
                # Fallback scoring without spaCy-dependent components
                weighted_score = (
                    tfidf_score * 0.15 +
                    semantic_score * 0.40 +
                    keyword_score * 0.45
                )
            
            # Normalize and boost the final score
            final_score = self._normalize_and_boost_score(weighted_score, keyword_overlap)
            # scores.append(final_score)
            results.append((round(final_score * 100, 2), list(matched_keywords)))
        return results
    
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
        """Enhanced keyword scoring with threshold-based boosting"""
        question_words = set(self._clean_text(question).split())
        overlap = len(jd_keywords & question_words)
        
        # Base score calculation
        base_score = min(1.0, overlap / max(len(jd_keywords)*0.25, 1))
        
        # Threshold-based boosting
        if overlap >= 3:  # Absolute threshold
            base_score = min(1.0, base_score * 1.25)
        if len(question_words) > 0 and (overlap/len(question_words)) >= 0.25:  # Relative threshold
            base_score = min(1.0, base_score * 1.15)
        return base_score
    
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
    
    def _normalize_and_boost_score(self, score,keyword_overlap):
        """Enhanced normalization with keyword-based boosting"""
        # Sigmoid normalization
        normalized = 1 / (1 + np.exp(-6 * (score - 0.5)))
        
        # Additional boost based on keyword overlap
        if keyword_overlap >= 2:
            normalized = min(1.0, normalized * 1.1)
        if keyword_overlap >= 4:
            normalized = min(1.0, normalized * 1.15)
        
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