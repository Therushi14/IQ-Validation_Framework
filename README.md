# LLM-Generated Interview Question Validator

## Overview
This project presents an automated framework to ensure the relevancy and appropriateness of AI-generated interview questions. It leverages open-source language models to generate interview questions based on a provided job role and job description. The framework then validates these questions through multiple analysis modules that assess relevance, technical (DSA) concept coverage, and potential bias or unethical language. The final results are presented via an interactive dashboard.

## Features
- **LLM-Generated Questions:** Generates 10–15 interview questions based on a job role and job description.
- **Validation:**  
  - Ensures content relevancy using multiple NLP-driven metrics.  
  - Checks technical syllabus/DSA concept coverage.  
  - Screens for bias and offensive content.
- **Dashboard Reporting:** Aggregates metrics and enables report export.

## Approach

### Modules

#### Module 1 – Input & Question Generation
- **Frontend:** Built using Streamlit to collect job role input and job description file uploads (PDF/DOCX).
- **LLM Integration:** Uses an open-source LLM (via GroqClient) with a crafted prompt to generate 10–15 interview questions.

#### Module 2 – Technical Relevance Analysis
- **Overview:** Evaluates how well the generated interview questions align with the job description using five NLP-driven metrics and a threshold-based final score.
- **Core Components:**
  - **TF-IDF Score:**  
    - *Process:* Convert JD & question into TF-IDF vectors and compute cosine similarity.  
    - *Purpose:* Measures lexical overlap of key terms/phrases.
  - **Semantic Score:**  
    - *Process:* Encode texts using SentenceTransformer (`all-MiniLM-L6-v2`) and compute cosine similarity.  
    - *Purpose:* Evaluates contextual meaning, even if direct keyword overlap is missing.
  - **Keyword Score:**  
    - *Process:* Extract top 20 JD keywords (using RAKE) and calculate overlap with boosting factors.  
    - *Purpose:* Ensures explicit matching of critical JD terminology.
  - **Entity Score:**  
    - *Process:* Extract entities using spaCy NER and compute entity overlap.  
    - *Purpose:* Validates alignment with specific tools/technologies.
  - **Context Score:**  
    - *Process:* Extract noun phrases using spaCy and calculate phrase overlap.  
    - *Purpose:* Assesses thematic relevance.
- **Overall Relevance:**  
  - Compute a weighted average of all five scores.
  - Apply a dynamic threshold and calculate the percentage of questions exceeding the threshold.

#### Module 3 – DSA Coverage Analysis
- **Purpose:** Compares the LLM-generated technical (DSA) questions against a Leetcode dataset (2200+ questions).
- **Approach:**  
  - Uses SentenceTransformer embeddings and cosine similarity to calculate a coverage score.
  - Identifies which DSA concepts are covered based on similarity to dataset entries.
- **Outcome:** Provides a DSA concept coverage score and lists the matched DSA concepts.

#### Module 4 – Bias Screening
- **Purpose:** Screens interview questions for bias and offensive content.
- **Approach:**  
  - **Biased Terms:** Uses a predefined lexicon of biased/offensive terms.
  - **NLP Techniques:** Applies spaCy NER and tokenization for bias detection.
  - **Sentiment Analysis:** Uses TextBlob to filter out inappropriate language (with a polarization threshold, e.g., -0.5).
- **Outcome:** Flags questions with potentially discriminatory or unethical language.

## Dashboard & Reporting
- **Integrated Dashboard:** Built with Streamlit to aggregate outputs from all modules.
- **Metrics Displayed:**  
  - Overall relevance score  
  - DSA coverage score with a list of covered concepts  
  - Bias screening results
- **Export:** Users can download detailed reports for further analysis.

## Conclusion
This automated framework ensures that AI-generated interview questions are not only relevant and technically comprehensive but also ethically sound. By combining multiple NLP techniques and leveraging real-world datasets, the solution significantly improves the quality and fairness of interview processes.
