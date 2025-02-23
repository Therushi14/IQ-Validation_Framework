import spacy
from groq import Groq
from textblob import TextBlob

nlp = spacy.load('en_core_web_sm')

# Define comprehensive biased terms/phrases
biased_terms = [
    "motherhood", "fatherhood", "stay-at-home parent", "single parent", "working mom", "working dad",
    "manpower", "man-hours", "man-made",
    "young", "old", "youthful", "elderly", "fresh", "experienced", "seasoned", "retirement", "pensioner",
    "generation gap", "junior", "senior",
    "race", "ethnicity", "color", "origin", "black", "white", "Asian", "Hispanic", "minority", "majority", "ethnic", "racial", "caucasian", "African-American", "Latino", "foreigner", "native", "immigrant",
    "rich", "poor", "wealthy", "impoverished", "affluent", "destitute", "low-income", "high-income", "upper class", "lower class", "social status", "blue-collar", "white-collar",
    "able-bodied", "disabled", "handicapped", "impaired", "crippled", "invalid", "wheelchair-bound", "mentally challenged", "deaf", "blind",
    "religion", "faith", "belief", "Christian", "Muslim", "Hindu", "Jewish", "atheist", "agnostic", "god", "divine", "holy", "sacred",
    "gay", "lesbian", "bisexual", "heterosexual", "LGBT", "LGBTQIA", "coming out", "partner", "same-sex", "straight", "homosexual", "transgender",
    "married", "single", "divorced", "widowed", "husband", "wife", "spouse", "children", "kids", "family",
    "dumb", "homemaker", "breadwinner", "caretaker", "guardian", "dependent",
    "accomplished", "inexperienced", "intermediate", "novice", "beginner", "skilled", "talented", "gifted",
    "active", "energetic", "lively", "vigorous", "enthusiastic", "spirited", "dynamic",
    "passive", "inactive", "lethargic", "sluggish", "apathetic", "unmotivated",
    "introvert", "extrovert", "ambivert", "shy", "outgoing", "sociable", "reserved", "gregarious",
    "optimistic", "pessimistic", "realistic", "pragmatic", "idealistic", "dreamer",
    "curious", "inquisitive", "interested", "uninterested", "indifferent", "apathetic",
    "brave", "courageous", "fearless", "bold", "daring", "audacious", "intrepid",
    "scared", "frightened", "afraid", "timid", "cowardly", "nervous", "anxious",
    "happy", "joyful", "cheerful", "content", "delighted", "pleased", "ecstatic",
    "sad", "unhappy", "sorrowful", "depressed", "miserable", "melancholic",
    "angry", "furious", "irate", "enraged", "mad", "upset", "annoyed", "frustrated",
    "calm", "peaceful", "serene", "tranquil", "relaxed", "composed", "collected",
    "confident", "assured", "self-assured", "self-confident", "assertive", "bold",
    "insecure", "self-doubting", "unconfident", "hesitant", "tentative",
    "loyal", "faithful", "trustworthy", "reliable", "dependable",
    "disloyal", "unfaithful", "untrustworthy", "unreliable",
    "generous", "kind", "benevolent", "charitable", "philanthropic", "magnanimous",
    "selfish", "greedy", "stingy", "miserly", "self-centered", "egotistical",
    "intelligent", "smart", "clever", "wise", "knowledgeable", "brilliant",
    "dumb", "stupid", "foolish", "ignorant", "unintelligent",
    "beautiful", "attractive", "handsome", "pretty", "gorgeous",
    "ugly", "unattractive", "plain", "homely", "unsightly"
]


def screen_for_bias(question):
    doc = nlp(question)
    for token in doc:
        if token.text.lower() in biased_terms:
            return False  # Question is biased
    return True  # Question is unbiased

def screen_for_offensive_language(question):
    sentiment = TextBlob(question).sentiment
    if sentiment.polarity < -0.5:  # Threshold for negative sentiment
        return False  # Question is offensive
    return True  # Question is not offensive

def screen_questions(questions):
    valid_questions = []
    invalid_questions = []
    for question in questions:
        if screen_for_bias(question) and screen_for_offensive_language(question):
            valid_questions.append(question)
        else:
            invalid_questions.append(question)

    print("Invalid question")
    for question in invalid_questions:
      print(question)
    return valid_questions

# Use the generated_questions from the first cell
valid_questions = screen_questions(generated_questions)

print("\nValid Questions:")
for question in valid_questions:
    print(question)

count = len(valid_questions)
accuracy = count/int(questionNumber)
print('accuracy is ',accuracy*100)