# 📰 Fake News Detection using Machine Learning

This project explores the use of machine learning models to detect fake news articles. We compare traditional models like Logistic Regression and XGBoost with a transformer-based deep learning model (BERT), using both binary and multiclass classification setups.


<summary>📁 Project Structure</summary>

- `datasets/` – Original, untouched datasets from Kaggle  
- `cleaned_datasets/` – Preprocessed datasets used for modeling  
- `notebooks/` – Jupyter notebooks for data exploration, training, and evaluation  
- `models/` – Saved models (`.pkl`) ready for reuse or deployment  
- `App/` – Streamlit app for interactive news classification  
- `requirements.txt` – All project dependencies for easy setup  


## 🔍 Problem Statement

Fake news is a widespread issue affecting public opinion, especially in areas like politics, health, and finance. This project aims to build intelligent models that automatically classify news as *real* or *fake*, and explore the added complexity of multiclass truthfulness detection using the LIAR dataset.


## 📊 Datasets Used

- **Fake News Dataset** – Headlines and article bodies for fake news only.

- **Real News Dataset** – Headlines and article bodies for real news only.

- **LIAR Dataset** – Short political statements labeled with 6 levels of truthfulness (e.g., true, mostly-true, half-true, barely-true, pants-fire, false).


## 🤖 Models Implemented

| Model                | Feature Type     | Binary | Multiclass | Multiclass (Oversampled) |
|---------------------|------------------|--------|------------|---------------------------|
| Logistic Regression | TF-IDF           | ✅      | ✅          | ✅                         |
| XGBoost             | TF-IDF           | ✅      | ✅          | ✅                         |
| BERT                | Transformer Text | ✅      | ✅          | ✅                         |



## ⚙️ How to Run

1. Clone the repo  
   ```bash
   git clone https://github.com/Intelligent-Systems-Team-1/Fake-News-Detector
   cd fake-news-detector
   ```

2. Install Requirements 
    ```bash
    pip install -r requirements.txt
    ```

3. Run the Streamlit App
    ```bash
    cd App
    streamlit run app.py
    ```

## 🧪 Evaluation Metrics
We use the following metrics to assess model performance:
- Accuracy

- Weighted F1-score

- Macro F1-score

- Confusion Matrix

Each model was evaluated in both binary and multiclass settings, with and without oversampling for class imbalance.


## 📌 Key Findings
- Logistic Regression performs well in binary settings but struggles with nuanced multiclass tasks.

- XGBoost handles class imbalance slightly better, especially with oversampling.

- BERT offers the most promise in contextual understanding (pending multiclass results).


## ✍️ Contributors
- Jaheem Edwards

- Anees Wahid

- Saleem Wahid