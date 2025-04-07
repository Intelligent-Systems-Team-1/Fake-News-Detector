import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# EDA function
def perform_eda(df):
    # Basic statistics and info
    print("Data Info:")
    print(df.info())
    print("\nSummary Statistics:")
    print(df.describe())
    
    # Check for missing values
    print("\nMissing Values:")
    print(df.isnull().sum())
    
    # Distribution of the labels
    print("\nLabel Distribution:")
    print(df['label'].value_counts())
    
    # Plot label distribution
    plt.figure(figsize=(6, 4))
    sns.countplot(x='label', data=df)
    plt.title("Label Distribution (True vs Fake News)")
    plt.xlabel('Label')
    plt.ylabel('Count')
    plt.show()

    # Plot text length distribution
    df['text_length'] = df['text'].apply(lambda x: len(str(x)))
    plt.figure(figsize=(6, 4))
    sns.histplot(df['text_length'], kde=True, color='blue', bins=30)
    plt.title("Text Length Distribution")
    plt.xlabel('Text Length')
    plt.ylabel('Count')
    plt.show()

    # Word cloud for the 'fake' news
    fake_news_text = " ".join(df[df['label'] == 0]['text'])
    plt.figure(figsize=(8, 6))
    wordcloud_fake = WordCloud(width=800, height=400, background_color='white').generate(fake_news_text)
    plt.imshow(wordcloud_fake, interpolation='bilinear')
    plt.title("Word Cloud for Fake News")
    plt.axis('off')
    plt.show()

    # Word cloud for the 'true' news
    true_news_text = " ".join(df[df['label'] == 1]['text'])
    plt.figure(figsize=(8, 6))
    wordcloud_true = WordCloud(width=800, height=400, background_color='white').generate(true_news_text)
    plt.imshow(wordcloud_true, interpolation='bilinear')
    plt.title("Word Cloud for True News")
    plt.axis('off')
    plt.show()