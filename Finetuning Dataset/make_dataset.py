import os
import pandas as pd
from nltk import tokenize
import re
from sklearn.model_selection import train_test_split
from datasets import load_dataset, Dataset, DatasetDict

# Replace this with the actual path to your directory containing the subdirectories with text files
root_directory = './data'

# Initialize a list to store the sentence groups and source values
sentence_groups = []

# Iterate over each subdirectory in the root directory
for subdir in os.listdir(root_directory):
    subdir_path = os.path.join(root_directory, subdir)
    
    # Check if it's indeed a directory
    if os.path.isdir(subdir_path):
        # Now, iterate over all files in this subdirectory
        for filename in os.listdir(subdir_path):
            if filename.endswith('.txt'):
                # Construct the full path to the text file
                file_path = os.path.join(subdir_path, filename)

                # Open and read the file
                with open(file_path, 'r', encoding='utf-8') as file:
                    content = file.read()

                    # Split the content into sentences
                    sentences = tokenize.sent_tokenize(content)

                    # Group the sentences into groups of 3
                    for i in range(0, len(sentences), 6):
                        # Ensure that we have at least 3 sentences to form a group
                        if i + 5 < len(sentences):
                            # Combine the group of three sentences into one string
                            group = ' '.join(sentences[i:i+6])
                            # Clean the group by removing newlines and excess whitespace
                            group = re.sub(r'\s+', ' ', group).strip()
                            # Add the group of sentences and the source (subdirectory name) to the list
                            sentence_groups.append({'sentences': group, 'source': subdir})

# Convert the list of sentence groups to a DataFrame
df = pd.DataFrame(sentence_groups)

# Split the DataFrame into training and validation sets
train_df, val_df = train_test_split(df, test_size=0.1)  # Here, we are reserving 10% of the data for validation

# Save the DataFrames to CSV files
train_df.to_csv('train_dataset.csv', index=False)
val_df.to_csv('validation_dataset.csv', index=False)

# Load your datasets from CSV files
train_dataset = Dataset.from_csv('train_dataset.csv')
validation_dataset = Dataset.from_csv('validation_dataset.csv')

# Create a DatasetDict
dataset_dict = DatasetDict({
    'train': train_dataset,
    'validation': validation_dataset
})

print(dataset_dict)
# The CSV files 'train_dataset.csv' and 'validation_dataset.csv' can now be uploaded to Hugging Face or used as needed.
dataset_dict.push_to_hub('kyueran/cond-mat')