import streamlit as st
import re
import pickle
from tqdm import tqdm
import time

# Function to clean the text data
def clean_text(text):
    text = text.lower()
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = text.strip()
    return text

# Load multiple models and vectorizers
def load_models():
    with open('fake_news_model.pkl', 'rb') as f:
        models = pickle.load(f)  # expects a dict like { "logistic": (vec, model), "xgboost": (vec, model) }
    return models

# Streamlit app title
st.title("📰 Fake News Detection System (Multiple Models)")

# Input field
news_title = st.text_input("Enter a news title to classify:")

if news_title:
    cleaned_title = clean_text(news_title)

    with st.spinner("Classifying news article..."):
        for _ in tqdm(range(100)):
            time.sleep(0.005)

        models = load_models()
        print(type(models))  # Check if it's a dict or tuple

        st.subheader("🧠 Predictions by Model:")

        for model_name, (vectorizer, model) in models.items():
            X = vectorizer.transform([cleaned_title])
            if hasattr(X, "toarray") and "xgboost" in str(type(model)).lower():
                X = X.toarray()

            prediction = model.predict(X)[0]
            prob = model.predict_proba(X)[0]

            st.markdown(f"**Model:** `{model_name}`")
            if prediction == 1:
                st.error(f"🟥 This news is **Fake** ({prob[1]*100:.2f}% confidence)")
            else:
                st.success(f"🟩 This news is **True** ({prob[0]*100:.2f}% confidence)")
            st.markdown("---")

# App instructions
st.write("""
### ℹ️ Instructions:
1. Type a news headline in the input field above.
2. This app will classify the news using multiple models (Logistic Regression, XGBoost, etc.).
3. You’ll see results side-by-side for comparison.
""")
