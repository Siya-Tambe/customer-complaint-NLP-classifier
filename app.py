import streamlit as st
import pickle
import re
from nltk.corpus import stopwords
import nltk

nltk.download('stopwords', quiet=True)

# ── Load model ────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open('complaint_classifier.pkl', 'rb') as f:
        pipeline = pickle.load(f)
    with open('label_encoder.pkl', 'rb') as f:
        le = pickle.load(f)
    return pipeline, le

pipeline, le = load_model()

# ── Same clean_text function used during training ─────────────────
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'x+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    words = text.split()
    words = [w for w in words if w not in stop_words and len(w) > 2]
    return ' '.join(words)

# ── Category info ─────────────────────────────────────────────────
CATEGORY_INFO = {
    "Bank Accounts and Services": {
        "icon": "🏦",
        "description": "Issues with bank accounts, transfers, deposits, or banking services",
        "examples": "Unauthorized transactions, account freezes, wire transfer problems"
    },
    "Credit Card Services": {
        "icon": "💳",
        "description": "Credit card billing, charges, disputes, or card services",
        "examples": "Double charges, fraudulent transactions, late fees, card cancellation"
    },
    "Credit Reporting": {
        "icon": "📊",
        "description": "Credit report errors, bureau disputes, score issues",
        "examples": "Wrong entries, Equifax/Experian/TransUnion disputes, identity theft"
    },
    "Debt Collection": {
        "icon": "📞",
        "description": "Debt collector behaviour, harassment, or invalid debt claims",
        "examples": "Repeated calls, collecting wrong amounts, unverified debts"
    },
    "Loans": {
        "icon": "🏠",
        "description": "Mortgage, auto, personal, or payday loan issues",
        "examples": "Interest disputes, payment processing, loan modification problems"
    }
}

# ── UI ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Complaint Classifier",
    page_icon="🔍",
    layout="centered"
)

st.title("🔍 Financial Complaint Classifier")
st.markdown("Type a customer complaint and the model will predict which category it belongs to.")
st.markdown("---")

# Input
complaint = st.text_area(
    "Enter complaint text:",
    placeholder="e.g. My credit score dropped because Equifax has a wrong account listed that doesn't belong to me...",
    height=150
)

col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    predict_btn = st.button("🔍 Classify Complaint", use_container_width=True)

# Prediction
if predict_btn:
    if not complaint.strip():
        st.warning("Please enter a complaint first.")
    elif len(complaint.split()) < 5:
        st.warning("Please enter a more detailed complaint (at least 5 words).")
    else:
        cleaned = clean_text(complaint)
        pred_label = pipeline.predict([cleaned])[0]
        pred_proba = pipeline.predict_proba([cleaned])[0]
        predicted_category = le.inverse_transform([pred_label])[0]
        confidence = pred_proba[pred_label] * 100

        info = CATEGORY_INFO[predicted_category]

        st.markdown("---")
        st.markdown(f"## {info['icon']} {predicted_category}")

        # Confidence bar
        conf_color = "green" if confidence >= 75 else "orange" if confidence >= 50 else "red"
        st.markdown(f"**Confidence: :{conf_color}[{confidence:.1f}%]**")
        st.progress(confidence / 100)

        st.info(f"**What this covers:** {info['description']}")
        st.caption(f"Common examples: {info['examples']}")

        # All probabilities
        st.markdown("---")
        st.markdown("**Probabilities across all categories:**")
        probs = zip(le.classes_, pred_proba)
        for cat, prob in sorted(probs, key=lambda x: x[1], reverse=True):
            bar_val = prob
            icon = CATEGORY_INFO[cat]["icon"]
            st.markdown(f"{icon} **{cat}**")
            st.progress(float(bar_val), text=f"{prob*100:.1f}%")

# Sidebar
with st.sidebar:
    st.markdown("## About")
    st.markdown("""
    This app classifies financial complaints into 5 categories using **NLP + Machine Learning**.
    
    **Model:** Logistic Regression + TF-IDF  
    **Accuracy:** 87%  
    **Trained on:** 40,000 real CFPB complaints
    """)
    st.markdown("---")
    st.markdown("**Categories:**")
    for cat, info in CATEGORY_INFO.items():
        st.markdown(f"{info['icon']} {cat}")
    st.markdown("---")
    st.caption("Built by Siya Tambe · [GitHub](https://github.com/Siya-Tambe/customer-complaint-nlp-classifier)")