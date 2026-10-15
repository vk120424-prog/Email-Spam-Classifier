"""
Text Preprocessing and Feature Extraction Utilities
for Email Spam Classifier.
"""

import re
import string

# Built-in comprehensive English stopwords to ensure offline reliability
DEFAULT_STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", 
    "aren't", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", 
    "but", "by", "can't", "cannot", "could", "couldn't", "did", "didn't", "do", "does", "doesn't", 
    "doing", "don't", "down", "during", "each", "few", "for", "from", "further", "had", "hadn't", 
    "has", "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here", 
    "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", 
    "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's", "its", "itself", "let's", 
    "me", "more", "most", "mustn't", "my", "myself", "no", "nor", "not", "of", "off", "on", 
    "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", 
    "same", "shan't", "she", "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", 
    "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then", 
    "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this", 
    "those", "through", "to", "too", "under", "until", "up", "very", "was", "wasn't", "we", 
    "we'd", "we'll", "we're", "we've", "were", "weren't", "what", "what's", "when", "when's", 
    "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with", 
    "won't", "would", "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", 
    "yours", "yourself", "yourselves"
}

# Try importing NLTK components with graceful fallback
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem.porter import PorterStemmer
    
    # Download stopwords if missing
    try:
        nltk_stopwords = set(stopwords.words("english"))
    except LookupError:
        nltk.download("stopwords", quiet=True)
        try:
            nltk_stopwords = set(stopwords.words("english"))
        except Exception:
            nltk_stopwords = DEFAULT_STOPWORDS
            
    stemmer = PorterStemmer()
except Exception:
    nltk_stopwords = DEFAULT_STOPWORDS
    stemmer = None

# Merged stopwords set
STOPWORDS = DEFAULT_STOPWORDS.union(nltk_stopwords)

# High-risk spam trigger phrases for explainability and heuristic risk evaluation
SPAM_TRIGGER_KEYWORDS = {
    "Prize & Lottery": [
        "free", "win", "winner", "won", "prize", "cash", "lottery", "claim", 
        "reward", "award", "jackpot", "bonus", "selected", "guaranteed"
    ],
    "Financial & Offers": [
        "urgent credit", "loan", "investment", "earn", "income", "million", 
        "dollars", "pounds", "cheap", "discount", "buy direct", "refinance", "rates"
    ],
    "Call To Action": [
        "call now", "click here", "subscribe", "text to", "reply now", 
        "stop to unsubscribe", "urgent", "immediately", "expire", "act now", "apply now"
    ],
    "Account & Security Fraud": [
        "account suspended", "verify your account", "security alert", "password reset", 
        "confirm identity", "billing problem", "unauthorized access"
    ]
}


def clean_text_for_model(text: str) -> str:
    """
    Standard preprocessing pipeline applied consistently during
    training and inference.
    
    Steps:
    1. Lowercase text
    2. Remove URLs, email addresses, and HTML tags
    3. Remove punctuation and special symbols
    4. Tokenize by whitespace
    5. Remove stopwords
    6. Apply Porter stemming
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    # Convert to lowercase
    text = text.lower()

    # Remove HTML tags
    text = re.sub(r"<.*?>", " ", text)

    # Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)

    # Remove email addresses
    text = re.sub(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", " ", text)

    # Remove non-alphabetical characters (keeps letters and spaces)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    # Tokenize
    tokens = text.split()

    # Filter stopwords and short tokens (< 2 chars)
    filtered_tokens = [
        word for word in tokens 
        if word not in STOPWORDS and len(word) > 1
    ]

    # Stem tokens if stemmer is available
    if stemmer:
        try:
            filtered_tokens = [stemmer.stem(word) for word in filtered_tokens]
        except Exception:
            pass

    return " ".join(filtered_tokens)


def extract_message_features(raw_text: str) -> dict:
    """
    Extracts statistical and heuristic features from raw message text
    for visualization and interactive dashboard analysis.
    """
    if not isinstance(raw_text, str):
        raw_text = ""

    cleaned_text = clean_text_for_model(raw_text)
    words = raw_text.split()
    word_count = len(words)
    char_count = len(raw_text)

    # Uppercase words count (SHOUTING)
    upper_words = [w for w in words if w.isupper() and len(w) > 1]
    upper_ratio = (len(upper_words) / word_count * 100) if word_count > 0 else 0.0

    # Digits and currency counts
    digits_count = len(re.findall(r"\d", raw_text))
    currency_count = len(re.findall(r"[\$£€¥₹]", raw_text))
    exclamation_count = raw_text.count("!")
    question_count = raw_text.count("?")

    # Detect triggered spam words
    lower_text = raw_text.lower()
    matched_triggers = []
    category_matches = {}

    for category, keywords in SPAM_TRIGGER_KEYWORDS.items():
        found = [kw for kw in keywords if re.search(r"\b" + re.escape(kw) + r"\b", lower_text)]
        if found:
            matched_triggers.extend(found)
            category_matches[category] = found

    return {
        "char_count": char_count,
        "word_count": word_count,
        "clean_word_count": len(cleaned_text.split()),
        "upper_words_count": len(upper_words),
        "upper_ratio": round(upper_ratio, 1),
        "digits_count": digits_count,
        "currency_count": currency_count,
        "exclamation_count": exclamation_count,
        "question_count": question_count,
        "matched_triggers": list(set(matched_triggers)),
        "category_matches": category_matches,
        "cleaned_text": cleaned_text
    }
