# LLM-Generated Interview Question Validator

## Overview
This project presents an automated framework to ensure the relevancy and appropriateness of AI-generated interview questions. It leverages open-source language models to generate interview questions based on a provided job role and job description. The framework then validates these questions through multiple automated analysis modules that assess relevance, technical (DSA) concept coverage, and potential bias or unethical language. The final results are presented via an interactive dashboard

## Features
- Validates LLM-generated interview questions.
- Ensures content relevancy and syllabus coverage.
- Screens for bias and offensive content.

## Approach
  Modules:
  Module 1 – Input & Question Generation:
    Frontend: Built using Streamlit for job role input and file upload (PDF/DOCX job description).
    LLM Integration: An open-source LLM (via GroqClient) generates 10–15 interview questions based on a crafted prompt.


  Module 2 – Technical Relevance Analysis:
    #1. Overview
  This framework evaluates how well interview questions align with a job description (JD) using 5 NLP-driven metrics and a threshold-based final score. Below is a structured breakdown of each component.
  
  #2. Core Components
  2.1 TF-IDF Score
  Process	Reason	Formula
  - Convert JD & question into TF-IDF vectors
  - Compute cosine similarity	Measures lexical overlap of key terms/phrases (including multi-word phrases)	TF(t,d) = (Term count in doc) / (Total terms)
  IDF(t) = log(Total docs / Docs containing t)
  TF-IDF = TF × IDF
  Example:
  JD: "Python developer with AWS experience"
  
  Question: "How to deploy Python apps on AWS?"
  
  High TF-IDF due to "Python" and "AWS" overlap.
  
  2.2 Semantic Score
  Process	Reason	Formula
  - Encode texts using SentenceTransformer
  (all-MiniLM-L6-v2)
  - Compute cosine similarity	Evaluates contextual meaning (e.g., paraphrases like "cloud" ≈ "AWS")	SemScore = cos(SBERT(JD), SBERT(Question))
  Example:
  JD: "Optimize cloud-based workflows"
  
  Question: "Improve AWS pipeline efficiency"
  
  High Semantic Score despite no keyword overlap.
  
  2.3 Keyword Score
  Process	Reason	Formula
  - Extract top 20 JD keywords (RAKE)
  - Calculate overlap + apply boosts	Ensures explicit term matching for critical JD terminology	Base = min(1, Overlap / max(0.25×JD_Keywords,1))
  Final = Base × 1.25 (if Overlap ≥3) × 1.15 (if Overlap ≥25% of Q)
  Example:
  JD Keywords: ["Python", "AWS", "ML"]
  
  Question: "Debug Python code on AWS"
  
  Score: 3 overlaps → 0.6 × 1.25 × 1.15 = 0.86.
  
  2.4 Entity Score
  Process	Reason	Formula
  - Extract entities (spaCy NER)
  - Compute entity overlap	Validates alignment with specific tools/technologies (e.g., "React.js", "PMP")	EntityScore = min(1, Overlap / (0.2 × JD_Entities))
  Example:
  JD Entities: ["Python", "AWS", "Agile"]
  
  Question: "Python testing frameworks?"
  
  Score: 1 overlap → 1 / (0.2×3) = 1.0 (capped).
  
  2.5 Context Score
  Process	Reason	Formula
  - Extract noun phrases (spaCy)
  - Calculate phrase overlap	Assesses thematic relevance (e.g., "cloud systems" ≈ "AWS")	PhraseOverlap = (Matching Phrases) / JD_Phrases
  ContextScore = min(1, PhraseOverlap × 1.5)
  Example:
  JD Phrases: ["machine learning models", "cloud infrastructure"]
  
  Question: "ML pipeline optimization"
  
  Score: 1 overlap → (1/2) × 1.5 = 0.75.
  
  2.6 Overall Relevance
  Process	Reason Formula
  1. Compute weighted average of all scores
  2. Apply dynamic threshold (μ/1.25)
  3. Calculate % questions above threshold	Evaluates cohort consistency (avoids skewed averages)	FinalScore = 0.15×TF-IDF + 0.35×Semantic + 0.20×Keyword + 0.15×(Entity + Context)
  Threshold = Avg(FinalScores) / 1.25
  Relevance = (% Questions > Threshold) × 100
  Example:
  Avg Score: 70 → Threshold = 56
  
  8/10 questions > 56 → 80% Relevance.
  
  #3. Summary Table
  Component	Weight	Purpose	Key Advantage
  TF-IDF	15%	Term/phrase overlap	Filters generic terms
  Semantic	35%	Meaning alignment	Handles paraphrases
  Keyword	20%	Explicit term matching	Sanity check
  Entity	15%	Tool/tech validation	Avoids vagueness
  Context	15%	Thematic relevance	Captures big-picture
  #4. Why This Matters
  Hiring Efficiency: Filters irrelevant questions automatically.

  Bias Reduction: Objective scoring vs. subjective judgment.

  JD Compliance: Ensures questions map to job requirements.
  
  Module 3 – DSA Coverage Analysis:
    - Compares the LLM-generated technical (DSA) questions against a Leetcode dataset (2200+ questions).
    - Uses SentenceTransformer embeddings and cosine similarity to calculate a coverage score and identify covered DSA concepts.
  
  Module 4 – Bias Screening:
    This module ensures that the interview questions are free from bias and offensive content.     The questions are free from language that may disadvantage or offend any group based on factors such as gender, race, age, ability, religion, and more.
  Biased terms and phrases:
    - It is a collection of biased, offensive terms and phrases which is used for sentimental analysis.
  Bias Screening:
    - Uses NLP (Spacy) to detect and remove biased terms.
    - It used NER (Named Entity Recognition) to classify named entities such as people, organization, location, etc.
    - Converts each sentence into tokens and tags each token with its part of speech such as adjective, verb, etc.
    Offensive Language Screening:
    - Uses TextBlob to apply sentiment analysis to filter out inappropriate language.
    - Assigns polarization value to each sentiment. Threshold polarization value is taken as -0.5
