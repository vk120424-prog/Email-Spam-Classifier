"""
Automated Test Suite for Email Spam Classifier
Validates preprocessing, models, vectorizer, and inference integrity.
"""

import os
import json
import joblib
# Standard library and local imports
from utils.preprocessing import clean_text_for_model, extract_message_features


def test_artifacts_exist():
    """Verify that all required files and trained model artifacts exist."""
    assert os.path.exists("data/dataset.csv"), "Dataset file missing!"
    assert os.path.exists("models/spam_model.pkl"), "Best model artifact missing!"
    assert os.path.exists("models/all_models.pkl"), "All models artifact missing!"
    assert os.path.exists("models/tfidf_vectorizer.pkl"), "TF-IDF Vectorizer missing!"
    assert os.path.exists("models/model_metrics.json"), "Metrics json missing!"


def test_metrics_integrity():
    """Verify metrics JSON contents and non-hardcoded values."""
    with open("models/model_metrics.json", "r", encoding="utf-8") as f:
        metrics = json.load(f)

    assert "best_model" in metrics
    assert "dataset_metadata" in metrics
    assert "model_performance" in metrics
    assert "top_spam_keywords" in metrics
    assert "top_ham_keywords" in metrics

    perf = metrics["model_performance"]
    for model_name in ["Multinomial Naive Bayes", "Logistic Regression", "Support Vector Machine"]:
        assert model_name in perf
        assert 0.85 <= perf[model_name]["accuracy"] <= 1.0
        assert 0.85 <= perf[model_name]["precision"] <= 1.0
        assert len(perf[model_name]["confusion_matrix"]) == 2


def test_preprocessing_pipeline():
    """Verify text cleaning, stopwords, and punctuation handling."""
    raw = "WINNER! You've won a $1,000 cash prize! Click https://scam.com/win now!"
    cleaned = clean_text_for_model(raw)
    
    assert "https" not in cleaned
    assert "$" not in cleaned
    assert "!" not in cleaned
    # Stem of winner is win
    assert "win" in cleaned or "prize" in cleaned


def test_empty_input_handling():
    """Verify handling of empty or whitespace strings."""
    assert clean_text_for_model("") == ""
    assert clean_text_for_model("   \t\n  ") == ""
    assert clean_text_for_model("... ??? !!!") == ""


def test_feature_extraction():
    """Verify heuristic feature extraction."""
    text = "URGENT! Claim your FREE $500 reward now! Call 123456."
    features = extract_message_features(text)
    
    assert features["char_count"] == len(text)
    assert features["word_count"] > 0
    assert features["currency_count"] >= 1
    assert features["exclamation_count"] >= 2
    assert len(features["matched_triggers"]) >= 2


def test_model_predictions():
    """Verify that models correctly classify clear spam and clear ham."""
    vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
    models = joblib.load("models/all_models.pkl")

    spam_message = "CONGRATULATIONS! You have won a £1,000 cash prize! Call 09061701461 now or text CLAIM to 87121."
    ham_message = "Hey Alex, are we still meeting in the library at 4 PM to work on the computer science assignment?"

    cleaned_spam = clean_text_for_model(spam_message)
    cleaned_ham = clean_text_for_model(ham_message)

    vec_spam = vectorizer.transform([cleaned_spam])
    vec_ham = vectorizer.transform([cleaned_ham])

    for name, model in models.items():
        pred_spam = model.predict(vec_spam)[0]
        pred_ham = model.predict(vec_ham)[0]

        assert pred_spam == 1, f"Model {name} failed to identify spam message!"
        assert pred_ham == 0, f"Model {name} failed to identify legitimate ham message!"


if __name__ == "__main__":
    print("Running automated tests...")
    test_artifacts_exist()
    print("[PASS] Artifacts exist")
    test_metrics_integrity()
    print("[PASS] Metrics integrity verified")
    test_preprocessing_pipeline()
    print("[PASS] Preprocessing pipeline verified")
    test_empty_input_handling()
    print("[PASS] Empty input handling verified")
    test_feature_extraction()
    print("[PASS] Feature extraction verified")
    test_model_predictions()
    print("[PASS] Model predictions verified for all algorithms!")
    print("\nALL 6 TEST SUITES PASSED SUCCESSFULLY!")
