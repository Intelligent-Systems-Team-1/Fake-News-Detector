import pandas as pd
import re

def clean_text(text):
    text = text.lower()  # Lowercase text
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with one
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    return text

def clean_fake_news_data():
    # Load the Fake and True datasets
    fake_df = pd.read_csv(r"Fake-News-Detector\src\data\Fake.csv")
    true_df = pd.read_csv(r"Fake-News-Detector\src\data\True.csv")

    # Add labels (Fake = 0, True = 1)
    fake_df['label'] = 0
    true_df['label'] = 1

    # Concatenate both datasets and select the relevant columns (title, text, subject, and label)
    df = pd.concat([fake_df[['title', 'text', 'subject', 'label']], true_df[['title', 'text', 'subject', 'label']]], ignore_index=True)

    # Drop irrelevant columns
    df = df[['title', 'text', 'subject', 'label']]

    # Handle missing values (drop rows where title or text is missing)
    df.dropna(subset=['title', 'text'], inplace=True)

    # Rename columns for clarity
    df.rename(columns={'title': 'Title', 'text': 'Text', 'subject': 'Subject', 'label': 'Label'}, inplace=True)

    # Apply the cleaning function to the 'Text' column
    df['Text'] = df['Text'].apply(clean_text)

    # Return cleaned DataFrame
    return df

def clean_liar_data():
    # Load the Liar dataset
    liar_df = pd.read_csv(r"Fake-News-Detector\src\data\Liar.csv")

    # Clean column names (strip spaces and unwanted characters)
    liar_df.columns = liar_df.columns.str.strip()
    
    # Select relevant columns: 'statement' and 'label'
    liar_df = liar_df[['statement', 'label']]
    
    # Handle missing values (drop rows where 'statement' or 'label' is missing)
    liar_df.dropna(subset=['statement', 'label'], inplace=True)

    # Standardize the 'label' column (optional: convert to numeric labels)
    liar_df['label'] = liar_df['label'].replace({
        'TRUE': 1,
        'FALSE': 0,
        'half-true': 0.5,
        'pants-fire': -1
    })

    # Clean the 'statement' column
    liar_df['statement'] = liar_df['statement'].apply(clean_text)

    # Return cleaned DataFrame
    return liar_df

def combine_datasets(fake_news_df, liar_df):
    # Standardize column names to lowercase for consistency
    fake_news_df.columns = [col.lower() for col in fake_news_df.columns]
    liar_df.columns = [col.lower() for col in liar_df.columns]
    
    # Add a column to indicate the source of the data
    fake_news_df['source'] = 'fake_news'
    liar_df['source'] = 'political_statement'
    
    # Select relevant columns and standardize labels in both datasets
    fake_news_df = fake_news_df[['title', 'text', 'label', 'source']]
    
    # Standardize the 'label' column in the Liar dataset to match the fake_news_df labels (True/False)
    liar_df['label'] = liar_df['label'].map({1: 'True', 0: 'False', 0.5: 'Half-True', -1: 'False'})
    
    # Rename 'statement' column to 'text' for consistency
    liar_df.rename(columns={'statement': 'text'}, inplace=True)
    
    # Select relevant columns from liar_df
    liar_df = liar_df[['text', 'label', 'source']]
    
    # Combine the datasets
    combined_df = pd.concat([fake_news_df[['title', 'text', 'label', 'source']], 
                             liar_df[['text', 'label', 'source']]], 
                            ignore_index=True)
    
    # Rename 'title' column in fake_news_df to 'text' for consistency
    combined_df.rename(columns={'title': 'text'}, inplace=True)

    # Reset index after combining
    combined_df.reset_index(drop=True, inplace=True)
    
    return combined_df

def clean_data():
    return combine_datasets(clean_fake_news_data(), clean_liar_data())

# Example usage
#combined_df = clean_data()
#print(combined_df.head())
#print(combined_df.info())