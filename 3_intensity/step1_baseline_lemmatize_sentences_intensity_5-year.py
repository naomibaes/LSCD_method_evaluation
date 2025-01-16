# Authors: Naomi Baes and Chat GPT

import spacy
import pandas as pd
import os
from tqdm import tqdm

# Load the SpaCy model for English with disabled components for speed
nlp = spacy.load('en_core_web_sm', disable=['ner', 'parser'])  # Use the transformer-based model

# Define the log file path and function to log messages
log_file_path = 'warnings.log.tokenize'

def log_message(message):
    with open(log_file_path, "a", encoding="utf-8") as log_file:
        log_file.write(message + "\n")

def normalize_string(text):
    # Process the text using the SpaCy pipeline
    doc = nlp(text)
    
    # Filter tokens: remove stopwords, punctuation, symbols, spaces, and numbers
    tokens = [
        token.lemma_.lower() for token in doc
        if not token.is_stop and not token.is_punct and not token.is_space and not token.like_num
    ]
    
    # Return the lemmatized string
    return ' '.join(tokens)

def lemmatize_sentences(file_path):
    try:
        # Manually define column names since the files do not contain headers
        column_names = ['sentence', 'year', 'label']
        
        # Load the dataset
        df = pd.read_csv(file_path, header=None, names=column_names, on_bad_lines='skip', delimiter='\t')
        
        # Check if 'sentence' column exists
        if 'sentence' not in df.columns:
            raise ValueError(f"No 'sentence' column found in the file: {file_path}")
        
        # Process in batches for efficiency
        texts = df['sentence'].astype(str).tolist()  # Ensure all data is in string format
        lemmatized_sentences = [normalize_string(text) for text in nlp.pipe(texts, batch_size=50)]
        
        # Replace the original sentences with the lemmatized sentences in column 1
        df['sentence'] = lemmatized_sentences

        # Save the DataFrame to a new file without column labels
        output_file_path = file_path.replace('.tsv', '_lemmatized.tsv')
        df.to_csv(output_file_path, sep='\t', index=False, header=False)
    
    except Exception as e:
        error_msg = f"Error processing {file_path}: {str(e)}"
        log_message(error_msg)

# Paths to the directories containing CSV files
directory_paths = [
    'output/5-year/high',
    'output/5-year/low'
]

# Process each file in the directories
for directory_path in directory_paths:
    files = [f for f in os.listdir(directory_path) if f.endswith('.tsv')]
    for file_name in tqdm(files, desc="Lemmatizing files"):
        file_path = os.path.join(directory_path, file_name)
        lemmatize_sentences(file_path)
