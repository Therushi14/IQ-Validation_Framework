import spacy
from textblob import TextBlob

nlp = spacy.load('en_core_web_md')  

# Define biased terms
biased_terms = [
    "motherhood", "fatherhood", "stay-at-home parent", "single parent", "working mom", "working dad",
    "manpower", "man-hours", "man-made", "young", "old", "youthful", "elderly", "fresh", "experienced",
    "race", "ethnicity", "color", "origin", "black", "white", "Asian", "Hispanic", "minority", "majority",
    "rich", "poor", "wealthy", "impoverished", "disabled", "handicapped", "deaf", "blind", "religion",
    "Christian", "Muslim", "Hindu", "Jewish", "atheist", "LGBT", "gay", "lesbian", "transgender",
    "married", "single", "divorced", "widowed", "children", "family", "dumb", "intelligent", "beautiful", "ugly"
]

# Preprocess biased terms as spaCy docs
biased_docs = [nlp(term) for term in biased_terms]

def screen_for_bias(question, threshold=0.85):
    """
    Checks if a question contains biased terms directly or has high similarity.
    """
    doc = nlp(question)
    max_similarity = 0
    for token in doc:
        for biased_doc in biased_docs:
            similarity = token.similarity(biased_doc)
            if similarity > max_similarity:
                max_similarity = similarity
            if similarity >= threshold:
                print(f"⚠️ Biased term detected: '{token.text}' similmmar to '{biased_doc.text}' ({similarity:.2f})")
                return False, max_similarity  # Mark as biased
    return True, max_similarity  # Unbiased with similarity score

def screen_for_offensive_language(question):
    """
    Checks for offensive sentiment using TextBlob.
    """
    sentiment = TextBlob(question).sentiment
    if sentiment.polarity < -0.5:  # Negative sentiment threshold
        print(f"❌ Offensive sentiment detected: Polarity {sentiment.polarity}")
        return False, sentiment.polarity
    return True, sentiment.polarity

def combine_scores(score1, score2, bias_weight=0.7, sentiment_weight=0.3):
    """
    Combines bias similarity and sentiment polarity into a single score.
    """
    # Normalize sentiment score: (-1 to 1) → (0 to 1)
    normalized_score2 = (1 - score2) / 2  # Positive → 0, Negative → 1

    # Weighted average
    combined_score = (bias_weight * score1) + (sentiment_weight * normalized_score2)
    return combined_score

def screen_questions(questions):
    """
    Screens a list of questions for bias and offensive language.
    Returns combined scores for each question.
    """
    valid_questions = []
    invalid_questions = []
    combined_scores = []

    for question in questions:
        is_unbiased, score1 = screen_for_bias(question)
        is_non_offensive, score2 = screen_for_offensive_language(question)

        combined_score = combine_scores(score1, score2)
        combined_scores.append(combined_score)

        if combined_score < 0.85:  # Threshold for validity
            valid_questions.append(question)
        else:
            invalid_questions.append(question)

    accuracy = len(valid_questions) / len(questions) if questions else 0
    return valid_questions, invalid_questions, accuracy, combined_scores

