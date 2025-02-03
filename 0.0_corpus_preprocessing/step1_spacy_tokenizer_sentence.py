import spacy
from tqdm import tqdm  # Progress bar for large corpora
import re

# Load SpaCy model and add sentencizer for sentence boundary detection
nlp = spacy.load("en_core_web_sm", disable=["ner", "parser", "tagger"])
nlp.add_pipe("sentencizer")  # Add sentencizer to detect sentence boundaries

# Custom function to add periods where they may be missing
def add_missing_punctuation(text):
    # Use regex to find likely sentence boundaries that are missing a period
    # This looks for sentences ending without punctuation, followed by an uppercase letter (new sentence start)
    text_with_periods = re.sub(r'(?<!\.\!\?)([a-z0-9\"\'])\s+(?=[A-Z])', r'\1. ', text)
    return text_with_periods

# Step 1: Parse the corpus and tokenize the text into sentences using SpaCy
def parse_and_tokenize_corpus(corpus):
    sentence_corpus = []
    
    # Skip the first line, which is the header
    for doc in tqdm(nlp.pipe(corpus[1:], batch_size=500), total=len(corpus) - 1):
        # Split the document by the metadata delimiter (|||||)
        parts = doc.text.split('|||||')

        # Ensure we have the necessary parts and that the text is valid
        if len(parts) >= 4:
            text = parts[0].strip()  # The actual content (abstract)
            text = add_missing_punctuation(text)  # Add periods where missing
            year = parts[1].strip()  # The publication year
            journal_title = parts[3].strip()  # The journal title

            # Filter out metadata and rows with invalid text
            if text and year.isdigit() and journal_title:
                # Append each sentence in the desired format with tab delimiters
                for sentence in doc.sents:
                    sentence_corpus.append(f"{sentence.text.strip()}\t{year}\t{journal_title}")  # Tab-separated format
    
    return sentence_corpus

# Step 2: Read the file and split by newlines (assuming each line is a document)
def read_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.readlines()  # Read each line as a document

# Step 3: Write the tokenized sentences to an output file with the desired header
def write_to_file(output_file_path, sentence_corpus):
    with open(output_file_path, 'w', encoding='utf-8') as f:
        f.write("sentences\tpublication_year\tjournal_title\n")  # Write the new header
        f.write("\n".join(sentence_corpus))  # Write the tokenized sentences

# Step 4: Print the first 10 tokenized rows for debugging purposes
def print_first_10_rows(sentence_corpus):
    print("First 10 tokenized rows:")
    for i, row in enumerate(sentence_corpus[:10]):
        print(f"{i+1}: {row}")

# Input file path
input_file_path = "C:/Users/naomi/OneDrive/COMP80004_PhDResearch/RESEARCH/DATA/CORPORA/Psychology/abstract_year_journal.csv.mental"

# Output file path
output_file_path = "C:/Users/naomi/OneDrive/COMP80004_PhDResearch/RESEARCH/DATA/CORPORA/Psychology/abstract_year_journal.csv.mental.sentence.tsv"

# Step 5: Read the file
corpus = read_file(input_file_path)

# Step 6: Parse and tokenize the corpus (skipping the header)
sentence_corpus = parse_and_tokenize_corpus(corpus)

# Step 7: Print the first 10 tokenized rows
print_first_10_rows(sentence_corpus)

# Step 8: Write the tokenized sentences to the output file with the new header
write_to_file(output_file_path, sentence_corpus)

# Step 9: Notify user of completion
print(f"Tokenized sentences written to {output_file_path}")
