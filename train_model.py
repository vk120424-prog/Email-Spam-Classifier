"""
Model Training Script for Email Spam Classifier.

This script:
1. Loads the labeled SMS/Email spam dataset
2. Cleans and preprocesses the text data
3. Splits into train and test sets (stratified)
4. Vectorizes using TF-IDF (fitted strictly on train data to avoid data leakage)
5. Trains 3 classification algorithms:
   - Multinomial Naive Bayes
   - Logistic Regression
   - Support Vector Machine (Linear Support Vector Classifier with Calibration)
6. Evaluates each model: Accuracy, Precision, Recall, F1-Score, Confusion Matrix
7. Selects the best performing model
8. Saves trained models, vectorizer, and evaluation metrics for Streamlit app
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# Import custom preprocessing pipeline
from utils.preprocessing import clean_text_for_model


def load_and_prepare_dataset(csv_path: str):
    """Loads dataset and standardizes column names to 'label' and 'text'."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found at: {csv_path}")

    # Read with latin-1 or utf-8 fallback
    try:
        df = pd.read_csv(csv_path, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(csv_path, encoding="latin-1")

    # Rename Kaggle / UCI columns (v1 = label, v2 = text) if present
    if "v1" in df.columns and "v2" in df.columns:
        df = df[["v1", "v2"]].copy()
        df.columns = ["label", "text"]
    elif "Category" in df.columns and "Message" in df.columns:
        df = df[["Category", "Message"]].copy()
        df.columns = ["label", "text"]
    elif "label" in df.columns and "text" in df.columns:
        df = df[["label", "text"]].copy()
    else:
        # Fallback: take first two columns
        df = df.iloc[:, :2].copy()
        df.columns = ["label", "text"]

    # Drop missing values
    df = df.dropna(subset=["label", "text"])

    # Standardize label to lowercase: 'ham' or 'spam'
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df = df[df["label"].isin(["ham", "spam"])].copy()

    # Drop duplicates
    initial_count = len(df)
    df = df.drop_duplicates(subset=["text"]).copy()
    print(f"[Dataset] Total records: {initial_count}, Unique messages: {len(df)}")
    print(f"[Dataset] Ham: {(df['label'] == 'ham').sum()} | Spam: {(df['label'] == 'spam').sum()}")

    return df


def train_and_evaluate(data_path="data/dataset.csv", output_dir="models"):
    """Full ML training and evaluation workflow."""
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("STEP 1: Loading Dataset...")
    df = load_and_prepare_dataset(data_path)

    total_records = len(df)
    ham_count = int((df["label"] == "ham").sum())
    spam_count = int((df["label"] == "spam").sum())

    print("\nSTEP 2: Cleaning and Preprocessing Text...")
    df["clean_text"] = df["text"].apply(clean_text_for_model)

    # Filter out empty cleaned strings if any
    valid_mask = df["clean_text"].str.strip().str.len() > 0
    df = df[valid_mask].copy()

    # Map labels: ham -> 0, spam -> 1
    y = df["label"].map({"ham": 0, "spam": 1}).values
    X_raw = df["clean_text"].values

    print("\nSTEP 3: Splitting Dataset (80% Train, 20% Test, Stratified)...")
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw, y, test_size=0.20, random_state=42, stratify=y
    )

    print(f"Training samples: {len(X_train_raw)}, Testing samples: {len(X_test_raw)}")

    print("\nSTEP 4: TF-IDF Feature Extraction (Preventing Data Leakage)...")
    vectorizer = TfidfVectorizer(
        max_features=3500,
        ngram_range=(1, 2),
        sublinear_tf=True
    )
    # Fit ONLY on training data
    X_train = vectorizer.fit_transform(X_train_raw)
    X_test = vectorizer.transform(X_test_raw)

    print(f"TF-IDF Vocabulary Size: {len(vectorizer.vocabulary_)} features")

    print("\nSTEP 5: Training Algorithms...")
    # Define models with probability output support
    models = {
        "Multinomial Naive Bayes": MultinomialNB(alpha=0.2),
        "Logistic Regression": LogisticRegression(max_iter=1000, C=1.5, random_state=42),
        "Support Vector Machine": CalibratedClassifierCV(
            LinearSVC(random_state=42, dual="auto", C=1.0)
        )
    }

    results = {}
    fitted_models = {}

    for name, model in models.items():
        print(f"-> Training {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        acc = float(accuracy_score(y_test, y_pred))
        prec = float(precision_score(y_test, y_pred, pos_label=1, zero_division=0))
        rec = float(recall_score(y_test, y_pred, pos_label=1, zero_division=0))
        f1 = float(f1_score(y_test, y_pred, pos_label=1, zero_division=0))
        cm = confusion_matrix(y_test, y_pred).tolist()

        results[name] = {
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "confusion_matrix": cm,
            "test_samples": len(y_test)
        }
        fitted_models[name] = model

        print(f"   Accuracy : {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f}")

    # Determine best model based on F1-Score & Precision balance
    best_model_name = max(
        results.keys(),
        key=lambda k: (results[k]["f1_score"] * 0.6 + results[k]["precision"] * 0.4)
    )
    print(f"\n[Selection] Best Model Selected: {best_model_name}")

    print("\nSTEP 6: Extracting Top Spam & Ham Keyword Indicators...")
    feature_names = np.array(vectorizer.get_feature_names_out())
    
    # Extract from Logistic Regression coefficients
    lr_model = fitted_models["Logistic Regression"]
    lr_coefs = lr_model.coef_[0]
    top_spam_indices = np.argsort(lr_coefs)[-20:][::-1]
    top_ham_indices = np.argsort(lr_coefs)[:20]

    top_spam_keywords = [
        {"word": str(feature_names[i]), "weight": round(float(lr_coefs[i]), 3)}
        for i in top_spam_indices
    ]
    top_ham_keywords = [
        {"word": str(feature_names[i]), "weight": round(float(abs(lr_coefs[i])), 3)}
        for i in top_ham_indices
    ]

    # Save artifacts
    print("\nSTEP 7: Saving Models, Vectorizer, and Evaluation Metrics...")
    joblib.dump(fitted_models[best_model_name], os.path.join(output_dir, "spam_model.pkl"))
    joblib.dump(fitted_models, os.path.join(output_dir, "all_models.pkl"))
    joblib.dump(vectorizer, os.path.join(output_dir, "tfidf_vectorizer.pkl"))

    metrics_payload = {
        "best_model": best_model_name,
        "dataset_metadata": {
            "total_records": total_records,
            "ham_count": ham_count,
            "spam_count": spam_count,
            "ham_percentage": round((ham_count / total_records) * 100, 2),
            "spam_percentage": round((spam_count / total_records) * 100, 2),
            "vocabulary_size": len(vectorizer.vocabulary_)
        },
        "model_performance": results,
        "top_spam_keywords": top_spam_keywords,
        "top_ham_keywords": top_ham_keywords
    }

    metrics_file = os.path.join(output_dir, "model_metrics.json")
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=4)

    print(f"Saved best model -> {os.path.join(output_dir, 'spam_model.pkl')}")
    print(f"Saved all models -> {os.path.join(output_dir, 'all_models.pkl')}")
    print(f"Saved vectorizer -> {os.path.join(output_dir, 'tfidf_vectorizer.pkl')}")
    print(f"Saved metrics json -> {metrics_file}")
    print("\nTraining workflow completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    train_and_evaluate()
