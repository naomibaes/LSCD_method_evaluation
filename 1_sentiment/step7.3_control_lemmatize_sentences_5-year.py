# Authors: Naomi Baes and Chat GPT

import pandas as pd
import spacy
import os
from tqdm.notebook import tqdm

# Load the SpaCy model
nlp = spacy.load('en_core_web_sm', disable=['ner', 'parser'])

# Define the log file path
log_file_path = 'output/5-year/control/warnings.log.tokenize'

def log_message(message):
    """Log messages to the log file."""
    with open(log_file_path, "a", encoding="utf-8") as log_file:
        log_file.write(message + "\n")

def normalize_string(text):
    """Normalize a string by lemmatizing and removing stop words and punctuation."""
    try:
        doc = nlp(text)
        tokens = [token.lemma_.lower() for token in doc if not token.is_stop and not token.is_punct and not token.is_space]
        return ' '.join(tokens)
    except Exception as e:
        log_message(f"Error normalizing text: {text[:50]}... Exception: {str(e)}")
        return text  # Return original text in case of failure

def lemmatize_sentences(file_path):
    """Process a file, lemmatize its sentences, and save the output."""
    try:
        df = pd.read_csv(file_path, sep='\t', header=None)
        df[0] = df[0].apply(normalize_string)  # Assuming the text to lemmatize is in the first column

        output_file_path = file_path.replace('.tsv', '_lemmatized.tsv')
        df.to_csv(output_file_path, sep='\t', header=False, index=False)  # No headers, no index
    except Exception as e:
        log_message(f"Error processing {file_path}: {str(e)}")

def process_directory(directory_path):
    """Process all TSV files in a directory sequentially."""
    try:
        log_message(f"Processing directory: {directory_path}")
        files = [os.path.join(directory_path, f) for f in os.listdir(directory_path) if f.endswith('.tsv')]
        if not files:
            log_message(f"No TSV files found in directory: {directory_path}")
            return
        
        for file_path in tqdm(files, desc=f"Processing {directory_path}"):
            lemmatize_sentences(file_path)
    except Exception as e:
        log_message(f"Error processing directory {directory_path}: {str(e)}")

# Run the script
directory_paths = [
    'output/5-year/control/positive',
    'output/5-year/control/negative'
]

for directory_path in directory_paths:
    process_directory(directory_path)
