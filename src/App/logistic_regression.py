import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report
import pickle

os.chdir(os.path.dirname(os.path.abspath(__file__)))

df = pd.read_csv(r"../../cleaned_datasets/cleaned_news.csv")

X = df['text']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

logistic_model = LogisticRegression()
logistic_model.fit(X_train_vec, y_train)

y_pred = logistic_model.predict(X_test_vec)
print("Accuracy: ", accuracy_score(y_test, y_pred))
print("F1 Score: ", f1_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

with open("fake_news_model.pkl", "wb") as f:
    pickle.dump((vectorizer, logistic_model), f)

print("Model and vectorizer saved.")