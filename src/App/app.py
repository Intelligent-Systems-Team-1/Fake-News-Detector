import streamlit as st
import re
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
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
        model = pickle.load(f)
    return model

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
        model = load_model()
        vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
        
        # Create a simple pipeline for text vectorization and classification
        pipeline = make_pipeline(vectorizer, model)
        
        # Predict whether the news is Fake or True
        prediction = pipeline.predict([cleaned_title])[0]
        
        # Show the result
        if prediction == 1:
            st.error("This news is **Fake**.")
        else:
            st.success("This news is **True**.")

# Add some app instructions
st.write("""
    **Instructions:**
    1. Type a news headline in the input field above.
    2. The app will classify it as either **Fake** or **True** based on the model.
""")
