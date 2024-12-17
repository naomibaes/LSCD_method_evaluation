import spacy
import pandas as pd
import os

# Load the SpaCy model for English
nlp = spacy.load('en_core_web_sm')  # Change 'en_core_web_sm' to your specific model if needed

def lemmatize_sentences(file_path):
    # Load the dataset
    df = pd.read_csv(file_path)
    lemmatized_sentences = []

    # Lemmatize each sentence
    for sentence in df['sentence']:
        doc = nlp(sentence)
        lemmas = ' '.join([token.lemma_ for token in doc])
        lemmatized_sentences.append(lemmas)

    # Create a new DataFrame with lemmatized sentences
    lemmatized_df = pd.DataFrame(lemmatized_sentences, columns=['lemmatized_sentence'])
    lemmatized_df.to_csv(file_path.replace('.csv', '_lemmatized.csv'), index=False)
    print(f"Lemmatized sentences saved to {file_path.replace('.csv', '_lemmatized.csv')}")

# Path to the directory containing CSV files
directory_path = 'path/to/your/csv/files'

# Process each file in the directory
for file_name in os.listdir(directory_path):
    if file_name.endswith('.csv'):
        lemmatize_sentences(os.path.join(directory_path, file_name))
