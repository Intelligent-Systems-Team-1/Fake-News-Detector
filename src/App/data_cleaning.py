import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def explore_dataframe(file_path):
    """
    Loads a CSV file into a pandas DataFrame and provides an overview of its structure,
    including missing values, duplicate rows, summary statistics, and visualizations.
    
    Parameters:
    file_path (str): Path to the CSV file.

    Returns:
    pd.DataFrame: The loaded DataFrame.
    """
    # Load CSV
    df = pd.read_csv(file_path)

    # Display first 5 rows
    print("\nFirst 5 Rows:")
    print(df.head())

    # Display column names
    print("\nColumns:", df.columns.tolist())

    # Show basic info (data types, non-null counts, memory usage)
    print("\nDataFrame Info:")
    print(df.info())

    # Show summary statistics (only for numeric columns)
    print("\nSummary Statistics:")
    print(df.describe())

    # Check for missing values
    print("\nMissing Values:")
    print(df.isnull().sum())

    # Check for duplicate rows
    print("\nDuplicate Rows:", df.duplicated().sum())

    # Show unique values per column
    print("\nUnique Values Per Column:")
    for col in df.columns:
        print(f"{col}: {df[col].nunique()} unique values")

    # Display data types
    print("\nData Types:")
    print(df.dtypes)

    return df  # Return the DataFrame for further analysis

# Example usage:
# df = explore_dataframe("your_file.csv")


fake_df = explore_dataframe("Fake-News-Detector\src\data\Fake.csv")
true_df = explore_dataframe("Fake-News-Detector\src\data\Liar.csv")
liar_df = explore_dataframe("Fake-News-Detector\src\data\True.csv")

fake_df["label"] = 1
true_df["label"] = 0

merged_df = pd.concat([fake_df, true_df], ignore_index=True)

print(merged_df.head())
