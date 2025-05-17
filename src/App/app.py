import streamlit as st
import re
import pickle
import torch
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
from tqdm import tqdm
import time
import plotly.express as px

# Text cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = text.strip()
    return text

# Load all models
@st.cache_resource
def load_models():
    # Load BERT model and tokenizer
    bert_model = BertForSequenceClassification.from_pretrained("models/bert_pipeline")
    tokenizer = BertTokenizer.from_pretrained("models/bert_pipeline")

    # Load logistic regression pipeline
    with open("models/logistic_pipeline.pkl", "rb") as f:
        logistic_pipeline = pickle.load(f)

    # Load XGBoost vectorizer + model dict
    with open("models/xgb_pipeline.pkl", "rb") as f:
        xgb_data = pickle.load(f)

    return {
        "BERT": (tokenizer, bert_model),
        "Logistic Regression": logistic_pipeline,
        "XGBoost": xgb_data
    }

# BERT prediction function
def predict_with_bert(tokenizer, model, text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1).squeeze().numpy()
        prediction = probs.argmax()
    return prediction, probs

# Function to plot prediction probabilities
def plot_probabilities(probs, model_name):
    df = pd.DataFrame({
        "Class": ["True", "Fake"],
        "Confidence": [probs[0] * 100, probs[1] * 100]
    })
    fig = px.bar(df, x="Class", y="Confidence", color="Class",
                 title=f"{model_name} Prediction Confidence",
                 color_discrete_map={"True": "green", "Fake": "red"},
                 labels={"Confidence": "% Confidence"},
                 text_auto=".2f")
    fig.update_layout(yaxis_range=[0, 100])
    st.plotly_chart(fig, use_container_width=True)

# App UI
st.title("📰 Fake News Detection System (Multiple Models)")

news_title = st.text_input("Enter a news title to classify:")

if news_title:
    cleaned_title = clean_text(news_title)

    with st.spinner("Classifying news article..."):
        for _ in tqdm(range(100)):
            time.sleep(0.003)

        models = load_models()
        st.subheader("🧠 Predictions by Model:")

        for name, model_obj in models.items():
            st.markdown(f"**Model:** `{name}`")

            if name == "BERT":
                tokenizer, bert_model = model_obj
                pred, probs = predict_with_bert(tokenizer, bert_model, cleaned_title)
            elif isinstance(model_obj, dict) and "vectorizer" in model_obj and "model" in model_obj:
                vectorizer = model_obj["vectorizer"]
                model = model_obj["model"]
                transformed = vectorizer.transform([cleaned_title])
                pred = model.predict(transformed)[0]
                probs = model.predict_proba(transformed)[0]
            else:
                pred = model_obj.predict([cleaned_title])[0]
                probs = model_obj.predict_proba([cleaned_title])[0]

            if pred == 1:
                st.error(f"🟥 This news is **Fake** ({probs[1]*100:.2f}% confidence)")
            else:
                st.success(f"🟩 This news is **True** ({probs[0]*100:.2f}% confidence)")

            plot_probabilities(probs, name)
            st.markdown("---")

# Instructions
st.write("""
### ℹ️ Instructions:
1. Enter a news headline above.
2. The system will use three models:
   - BERT (Deep Learning)
   - Logistic Regression
   - XGBoost
3. You'll see each model's prediction, confidence, and a bar chart of probabilities.
""")
