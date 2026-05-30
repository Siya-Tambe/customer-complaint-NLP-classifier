# 🔍 Financial Complaint NLP Classifier

> An end-to-end NLP project that automatically classifies real financial customer complaints into categories using Machine Learning — trained on 40,000+ real CFPB complaints.

## 🚀 Live Demo
### 👉 [financial-complaint-nlp.streamlit.app](https://financial-complaint-nlp.streamlit.app)

---

## 📌 Project Overview

Financial institutions receive thousands of customer complaints daily. Manually routing each complaint to the right department is slow and error-prone.

This project builds an **automated complaint classification system** that:
- Takes raw complaint text as input
- Cleans and vectorizes it using TF-IDF
- Predicts which of 5 financial categories it belongs to
- Shows confidence scores across all categories

---

## 📊 Model Results

| Model | Accuracy |
|-------|----------|
| Naive Bayes + TF-IDF | 82.5% |
| **Logistic Regression + TF-IDF** | **87.0%** ✅ |

### Per-Category Performance (Logistic Regression)
| Category | Precision | Recall | F1 |
|----------|-----------|--------|----|
| Bank Accounts and Services | 0.88 | 0.91 | 0.90 |
| Credit Card Services | 0.85 | 0.84 | 0.85 |
| Credit Reporting | 0.87 | 0.88 | 0.87 |
| Debt Collection | 0.86 | 0.85 | 0.85 |
| Loans | 0.89 | 0.88 | 0.88 |

---

## 🗂️ Categories Classified

| Icon | Category | Example Keywords |
|------|----------|-----------------|
| 🏦 | Bank Accounts and Services | bank, account, transfer, deposit |
| 💳 | Credit Card Services | card, charge, transaction, statement |
| 📊 | Credit Reporting | Equifax, Experian, TransUnion, report |
| 📞 | Debt Collection | debt, collector, owe, collections |
| 🏠 | Loans | mortgage, loan, interest, payment |

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python | Core language |
| pandas / numpy | Data manipulation |
| NLTK | Text preprocessing, stopword removal |
| scikit-learn | TF-IDF vectorization, model training |
| matplotlib / seaborn | EDA charts, confusion matrix |
| WordCloud | Word frequency visualization |
| Streamlit | Web app deployment |
| Google Colab | Model training environment |

---

## 📁 Project Structure

customer-complaint-NLP-classifier/
│
├── app.py                          # Streamlit web app
├── complaint_classifier.pkl        # Trained LR + TF-IDF pipeline
├── label_encoder.pkl               # Category label encoder
├── requirements.txt                # Dependencies
├── customer_complaint_nlp_classifier.ipynb  # Full training notebook
└── .gitignore

---

## 🔄 Project Pipeline

Raw Text → Clean Text → TF-IDF Vectors → Logistic Regression → Category + Confidence

1. **Data Loading** — 150k rows from CFPB dataset, sampled to 40k balanced
2. **Text Cleaning** — lowercase, remove stopwords, remove redacted tokens (XXXX)
3. **Class Balancing** — 8,000 samples per category (undersampling)
4. **TF-IDF Vectorization** — 50,000 features, unigrams + bigrams
5. **Model Training** — Naive Bayes baseline → Logistic Regression final
6. **Deployment** — Streamlit Cloud

---

## 📈 Visualizations

- Category distribution (balanced vs original)
- Complaint length distribution
- Word clouds per category
- Confusion matrix
- Top predictive words per category

---

## 📦 Dataset

[Consumer Complaint Database](https://www.kaggle.com/datasets/anoopjohny/consumer-complaint-database) from CFPB via Kaggle.
2 million real complaints — we used a balanced sample of 40,000.

> ⚠️ Dataset not included in repo due to size (2.5GB). Download from Kaggle link above.

---

## 🔗 Connect

**Siya Tambe** — [GitHub](https://github.com/Siya-Tambe) · [LinkedIn](#)

*This project is part of a connected finance analytics portfolio:*
- **Project 2** — EMI Loan Default Predictor (who will default?)
- **Project 3** — Financial Complaint Classifier (what will they complain about?) ← you are here

📓 [View Notebook on nbviewer](https://nbviewer.org/github/Siya-Tambe/customer-complaint-NLP-classifier/blob/main/customer_complaint_NLP_classifier.ipynb)
