from data_cleaning import *
from eda import *
from text_preprocessing import *


combined_df = clean_data()
print(combined_df.isnull().sum())
# Check for missing values
print("\nMissing Values:")
print(combined_df.isnull().sum())
    
    # Distribution of the labels
print("\nLabel Distribution:")
print(combined_df['label'].value_counts())
    
    # Plot label distribution
plt.figure(figsize=(6, 4))
sns.countplot(x='label', data=combined_df)
plt.title("Label Distribution (True vs Fake News)")
plt.xlabel('Label')
plt.ylabel('Count')
plt.show()

#perform_eda()