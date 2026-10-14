# 🛡️ Email & SMS Spam Classifier

> **A Machine Learning & Natural Language Processing College Project**  
> Developed with Python, Streamlit, Scikit-Learn, Pandas, NumPy, Matplotlib, and Seaborn.

---

## 📌 1. Project Overview

The **Email Spam Classifier** is an intelligent web application designed to automatically identify whether an incoming email or text message is **SPAM** (unsolicited / fraudulent) or **HAM** (legitimate).

The system applies a Natural Language Processing (NLP) pipeline combining text cleaning, stopword removal, Porter stemming, and **TF-IDF Vectorization** (fitted strictly on training data to prevent data leakage). Classification is performed and compared across three supervised machine learning algorithms:
1. **Multinomial Naive Bayes** (Optimized with Laplace smoothing)
2. **Logistic Regression** (L2 Regularized)
3. **Support Vector Machine (LinearSVC)** (with Calibrated Probability Estimation)

The application features a modern, responsive **Streamlit** dashboard tailored for college project demonstrations, faculty reviews, and lab evaluations.

---

## 🏗️ 2. Project Architecture & Directory Structure

```text
EMAIL SPAM CLASSIFIER/
├── app.py                     # Main interactive Streamlit multi-page web application
├── train_model.py             # Complete ML training, evaluation, and serialization script
├── test_app.py                # Automated test suite verifying ML pipeline & inference
├── requirements.txt           # Project dependencies
├── README.md                  # Comprehensive project documentation
├── data/
│   └── dataset.csv            # Authentic UCI SMS Spam Collection dataset (5,572 records)
├── models/
│   ├── spam_model.pkl         # Serialized best-performing model (MultinomialNB)
│   ├── all_models.pkl         # Serialized dictionary of all 3 trained classifiers
│   ├── tfidf_vectorizer.pkl   # Fitted TF-IDF Vectorizer (3,500 features)
│   └── model_metrics.json     # Authentic calculated evaluation scores & confusion matrices
└── utils/
    ├── __init__.py            # Package initialization
    └── preprocessing.py       # Reusable text preprocessing & feature extraction pipeline
```

---

## ⚙️ 3. Machine Learning Pipeline Workflow

```
Raw Email / SMS Text
         │
         ▼
[1. Text Cleaning & Preprocessing]
     ├── Case Normalization (Lowercase)
     ├── HTML Tag, URL, and Email Pattern Removal
     ├── Punctuation & Special Symbol Stripping
     ├── Tokenization
     ├── Stopword Elimination (NLTK + Fallback Lexicon)
     └── Stemming (Porter Stemmer)
         │
         ▼
[2. Feature Engineering (TF-IDF Vectorizer)]
     ├── Sublinear Term Frequency Scaling
     ├── N-Gram Range: Unigrams & Bigrams (1, 2)
     ├── Vocabulary Size: 3,500 Top Features
     └── Strict Train-Only Fitting (Zero Data Leakage)
         │
         ▼
[3. Supervised Algorithm Training]
     ├── Multinomial Naive Bayes
     ├── Logistic Regression
     └── Linear Support Vector Classifier (Calibrated)
         │
         ▼
[4. Model Evaluation & Selection]
     ├── Stratified 80/20 Test Split (1,032 Held-Out Samples)
     ├── Metrics: Accuracy, Precision, Recall, F1-Score
     └── Selection: Best Model based on F1-Score & Precision
         │
         ▼
[5. Interactive Inference Dashboard]
     ├── Live Classification (SPAM / HAM)
     ├── Model Confidence / Probability Score
     ├── Highlighted Suspicious Trigger Keywords
     └── Text Metadata (Word count, uppercase ratio, symbols)
```

---

## 📊 4. Dataset Information

* **Dataset Source:** [UCI Machine Learning Repository / Kaggle SMS Spam Collection](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)
* **Total Raw Samples:** 5,572 messages
* **Unique Samples (After Deduplication):** 5,169 messages
* **Legitimate Messages (Ham):** 4,516 messages (87.37%)
* **Spam Messages:** 653 messages (12.63%)
* **Target Classes:** Binary classification (`0: Ham`, `1: Spam`)

---

## 📈 5. Actual Model Evaluation Results

All performance metrics below were calculated on the **held-out test set (1,032 unseen samples, 80-20 stratified split)**:

| Machine Learning Algorithm | Test Accuracy | Precision (Spam) | Recall (Spam) | F1-Score (Spam) |
| :--- | :---: | :---: | :---: | :---: |
| **Multinomial Naive Bayes** ⭐ | **98.16%** | **98.28%** | **87.02%** | **92.31%** |
| **Logistic Regression** | 96.90% | 97.14% | 77.86% | 86.44% |
| **Support Vector Machine (LinearSVC)** | 98.45% | 93.89% | 93.89% | 93.89% |

⭐ **Best Model Selected:** **Multinomial Naive Bayes** was selected as the primary production model because it achieves the highest **Precision (98.28%)** on the spam class.

### Confusion Matrices (Test Partition - 1,032 Samples)

* **Multinomial Naive Bayes:**
  * True Ham: `899` | False Spam (False Positive): `2`
  * False Ham (False Negative): `17` | True Spam: `114`
  * *Crucial advantage: Only 2 out of 901 legitimate emails were mistakenly flagged as spam!*

* **Logistic Regression:**
  * True Ham: `898` | False Spam: `3`
  * False Ham: `29` | True Spam: `102`

* **Support Vector Machine (SVM):**
  * True Ham: `893` | False Spam: `8`
  * False Ham: `8` | True Spam: `123`

---

## 🔑 6. Top Discriminative Keywords

Extracted from statistical model weights and TF-IDF representations:

* **Top Spam Indicators:** `txt`, `repli`, `mobil`, `claim`, `free`, `call`, `text`, `prize`, `servic`, `stop`, `tone`, `win`, `cash`, `urgent`, `rington`
* **Top Legitimate (Ham) Indicators:** `ok`, `home`, `come`, `later`, `hey`, `say`, `love`, `much`, `know`, `still`, `sorri`

> *Disclaimer: These keywords represent probabilistic signals identified by the classifier. They serve as explainability context, not absolute proof.*

---

## 💻 7. Installation & Execution Guide

### Prerequisites
- Python 3.10, 3.11, or 3.12 installed on Windows, macOS, or Linux.

### Step 1: Open Terminal / PowerShell in Project Directory
```powershell
cd "EMAIL SPAM CLASSIFIER"
```

### Step 2: (Recommended) Create & Activate Virtual Environment
```powershell
# Create virtual environment
python -m venv .venv

# Activate virtual environment (Windows PowerShell)
.\.venv\Scripts\Activate.ps1

# (Alternative for Command Prompt CMD)
# .\.venv\Scripts\activate.bat
```

### Step 3: Install Required Dependencies
```powershell
pip install -r requirements.txt
```

### Step 4: Train Models & Generate Evaluation Artifacts
```powershell
python train_model.py
```
*(Loads dataset, cleans text, trains Naive Bayes, Logistic Regression, and SVM, evaluates test set, and writes `.pkl` and `.json` artifacts into `models/`)*

### Step 5: Run Automated Tests
```powershell
python test_app.py
```
*(Validates data loading, preprocessing, empty inputs, heuristics, and cross-algorithm predictions)*

### Step 6: Launch Streamlit Web Application
```powershell
streamlit run app.py
```
The browser will automatically open at: **`http://localhost:8501`**

---

## 🎯 8. Demonstration Test Messages

You can copy and test these sample messages directly in the application (or click the preset one-click buttons on the **Spam Detection** page):

### Spam Test Cases:
1. **Prize & Lottery Scam:**
   > `"CONGRATULATIONS! You have been selected as the official winner of our £1,000 cash prize draw! Call 09061701461 now or text CLAIM to 87121 to receive your reward immediately. T&Cs apply."`
2. **Bank Phishing Urgency:**
   > `"URGENT SECURITY ALERT: Your online banking account has been temporarily suspended due to unauthorized access. Click here immediately to verify your identity and restore access: http://secure-verify-bank.com/login"`
3. **Promotional Gift Scam:**
   > `"Free entry in 2 a weekly competition to win a brand new iPhone 15 Pro and gift cards! Text WIN to 80082 to claim your entry ticket right now. Standard charges apply."`

### Legitimate (Ham) Test Cases:
1. **Academic Discussion:**
   > `"Hey team, are we still meeting in the computer lab at 4:30 PM today to finalize our Machine Learning project presentation? Please bring your laptops."`
2. **Assignment Submission:**
   > `"Dear Professor Sharma, please find attached my updated submission for Assignment 3 on Natural Language Processing. Let me know if any revisions are needed."`
3. **Casual Message:**
   > `"Hi Sarah, thanks for sharing the study notes earlier! I will review the lecture slides tonight and let you know if I have questions."`

---

## 🎓 9. College Presentation & Viva Questions Guide

### Q1: Why did you choose Multinomial Naive Bayes as the best model?
**Answer:** In email filtering, **Precision** is more critical than raw accuracy. A False Positive (classifying an important legitimate email as spam) can cause a user to miss vital communications (e.g., job offers, bank OTPs). Multinomial Naive Bayes achieved **98.28% precision** with only 2 false positives out of 901 legitimate test samples, making it safer for real-world inbox deployment.

### Q2: How did you prevent data leakage during feature extraction?
**Answer:** We split the dataset into training (80%) and testing (20%) sets **before** applying `TfidfVectorizer`. The vectorizer was fitted exclusively on `X_train` (`fit_transform`) and only used to transform `X_test` (`transform`). This ensures that test vocabulary frequencies never influenced model training.

### Q3: What is the role of TF-IDF in this project?
**Answer:** Term Frequency-Inverse Document Frequency (TF-IDF) converts textual strings into numerical vectors. It assigns higher weights to terms that appear frequently in a specific message but rarely across the general corpus, effectively suppressing generic words and highlighting distinct spam triggers.

### Q4: What are the limitations of this system?
**Answer:** The classifier processes the message body text. It does not inspect compiled executable email attachments (`.exe`, `.zip`), zero-day malicious payload scripts, or dynamically execute sandbox evaluations of hyperlinks.

---

## 📜 10. Technology Stack Summary

* **Language:** Python 3.12
* **Frontend:** Streamlit 1.39
* **Machine Learning:** Scikit-Learn 1.9 (MultinomialNB, LogisticRegression, LinearSVC, CalibratedClassifierCV)
* **Data Manipulation:** Pandas 2.3, NumPy 2.5
* **Natural Language Processing:** NLTK, Regular Expressions (`re`), Porter Stemmer, TF-IDF Vectorizer
* **Visualization:** Matplotlib 3.11, Seaborn 0.13
* **Serialization:** Joblib
