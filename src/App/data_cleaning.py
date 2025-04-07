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

    # Handle missing values (drop rows where title or text is missing)
    df.dropna(subset=['title', 'text'], inplace=True)

    # Concatenate the title and text into a new column called 'text_combined'
    df['text_combined'] = df['title'] + " " + df['text']

    # Apply the cleaning function to the 'text_combined' column
    df['text_combined'] = df['text_combined'].apply(clean_text)

    # Drop irrelevant columns
    df = df[['text_combined', 'label']]  # Keep only the 'text_combined' and 'label' columns

    # Rename columns for clarity
    df.rename(columns={'text_combined': 'text'}, inplace=True)

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

    # Suppress downcasting and chained assignment warnings
    pd.set_option('future.no_silent_downcasting', True)
    pd.options.mode.chained_assignment = None  # Disable chained assignment warning

    print(liar_df['label'].dtype)
    liar_df['label'] = liar_df['label'].apply(str)
    print(liar_df['label'].dtype)
    liar_df['label'] = liar_df['label'].replace({
        'TRUE': 1,
        'FALSE': 0,
        'mostly-true': 1,
        'half-true': 1,
        'barely-true': 1,
        'pants-fire': 0
    })
    print(liar_df['label'].dtype)
    liar_df['label'] = pd.to_numeric(liar_df['label'], errors='coerce')
    print(liar_df['label'].dtype)
    liar_df['label'] = liar_df['label'].fillna(0)  # or another appropriate value
    
    # Clean the 'statement' column
    liar_df['statement'] = liar_df['statement'].apply(clean_text)

    # Rename 'statement' to 'text' for consistency
    liar_df.rename(columns={'statement': 'text'}, inplace=True)

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
    fake_news_df = fake_news_df[['text', 'label', 'source']]
    
    # Standardize the 'label' column in the Liar dataset to match the fake_news_df labels (True/False)
    liar_df['label'] = liar_df['label'].map({1: 'True', 0: 'False', 0.5: 'Half-True', -1: 'False'})
    
    # Select relevant columns from liar_df
    liar_df = liar_df[['text', 'label', 'source']]
    
    # Combine the datasets
    combined_df = pd.concat([fake_news_df[['text', 'label', 'source']], 
                             liar_df[['text', 'label', 'source']]], 
                            ignore_index=True)
    
    # Reset index after combining
    combined_df.reset_index(drop=True, inplace=True)
    
    return combined_df

def clean_data():
    return combine_datasets(clean_fake_news_data(), clean_liar_data())
