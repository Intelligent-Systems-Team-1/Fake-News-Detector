import streamlit as st
import re
import pickle
from tqdm import tqdm
import time

# Function to clean the text data
def clean_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r"\d+", "", text)  # Remove numbers
    text = re.sub(r"http\S+|www\S+", "", text)  # Remove URLs
    text = re.sub(r"[^\w\s]", "", text)  # Remove punctuation
    text = text.strip()  # Remove extra spaces
    return text

# Load the trained model (make sure you've saved it before)
def load_model():
    with open('fake_news_model.pkl', 'rb') as f:
        vectorizer, model = pickle.load(f)
    return vectorizer, model

# Title of the Streamlit app
st.title("Fake News Detection System")

# Input field for news title
news_title = st.text_input("Enter a news title to classify:")

if news_title:
    # Clean the input news title
    cleaned_title = clean_text(news_title)
    
    # Display a progress bar while processing the data
    with st.spinner("Classifying news article..."):
        # Simulate loading with tqdm progress bar
        for _ in tqdm(range(100)):
            time.sleep(0.01)  # Simulating some computation time
        
        # Load the model and vectorizer
        vectorizer, model = load_model()
        X = vectorizer.transform([cleaned_title])
        prediction = model.predict(X)[0]
        prob = model.predict_proba(X)[0]

 # Display result
        if prediction == 1:
            st.error(f" This news is **Fake** ({prob[1]*100:.2f}% confidence).")
        else:
            st.success(f" This news is **True** ({prob[0]*100:.2f}% confidence).")

# Add some app instructions
st.write("""
    **Instructions:**
    1. Type a news headline in the input field above.
    2. The app will classify it as either **Fake** or **True** based on the model.
""")
