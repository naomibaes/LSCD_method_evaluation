import pandas as pd
import spacy
import os
from tqdm import tqdm

# Load the SpaCy model
nlp = spacy.load('en_core_web_sm', disable=['ner', 'parser'])

# Define the log file path
log_file_path = 'output/all-year/warnings.log.tokenize'

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

def lemmatize_sentences(file_path, output_dir):
    """Process a file, lemmatize its sentences, and save the output."""
    try:
        # Construct the output file path
        output_file_path = os.path.join(output_dir, os.path.basename(file_path).replace('.tsv', '_lemmatized.tsv'))
        
        # Skip processing if the output file already exists
        if os.path.exists(output_file_path):
            log_message(f"Skipping {file_path}: Already processed.")
            return

        # Read and process the input file
        df = pd.read_csv(file_path, sep='\t', header=None)
        df[0] = df[0].apply(normalize_string)  # Assuming the text to lemmatize is in the first column

        # Save the processed file
        df.to_csv(output_file_path, sep='\t', header=False, index=False)
        log_message(f"Processed and saved: {output_file_path}")
    except Exception as e:
        log_message(f"Error processing {file_path}: {str(e)}")

def process_directory(input_dir, output_dir):
    """Process all TSV files in a directory sequentially."""
    try:
        log_message(f"Processing directory: {input_dir}")
        files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.tsv')]
        if not files:
            log_message(f"No TSV files found in directory: {input_dir}")
            return
        
        for file_path in tqdm(files, desc=f"Processing {input_dir}"):
            lemmatize_sentences(file_path, output_dir)
    except Exception as e:
        log_message(f"Error processing directory {input_dir}: {str(e)}")

# Run the script
directories = [
    ('output/all-year/positive', 'output/all-year/positive'),
    ('output/all-year/negative', 'output/all-year/negative')
]

for input_dir, output_dir in directories:
    process_directory(input_dir, output_dir)
