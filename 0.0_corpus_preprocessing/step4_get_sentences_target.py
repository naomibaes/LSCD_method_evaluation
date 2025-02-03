import os
import nltk
from nltk.tokenize import word_tokenize

# Ensure NLTK data is downloaded (for tokenization)
nltk.download('punkt')

# List of input files for the natural corpus
input_file = [
    "C:/Users/naomi/OneDrive/COMP80004_PhDResearch/RESEARCH/DATA/CORPORA/Psychology/abstract_year_journal.csv.mental.sentence-CLEAN.tsv"
]

# Target terms to look for
target_terms = ["mental_health", "mental_illness", "trauma", "abuse", "anxiety", "depression"]

# Function to filter and save lines for each target term in the natural corpus
def filter_lines(input_file, target_term, corpus, output_suffix=""):
    output_dir = "output/natural_lines_targets"
    
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Construct the output file path with the correct naming convention
    output_file = os.path.join(output_dir, f"{target_term}.lines{output_suffix}.psych")

    # Check if the output file already exists and skip processing if it does
    if os.path.exists(output_file):
        print(f"Output file already exists, skipping: {output_file}")
        return 0  # Return 0 if no new lines are written

    print(f"Processing {input_file} for target term: {target_term} in natural corpus")

    # Initialize counter for the number of lines written
    lines_written = 0

    # Open input and output files
    with open(input_file, "r", encoding="utf-8") as infile, open(output_file, "w", encoding="utf-8") as outfile:
        # Iterate over each line in the input file
        for line in infile:
            # Split the line into its components
            parts = line.strip().split("\t")
            
            # Handle two-column format: sentence and year
            if len(parts) == 2:
                sentence, year = parts
                tokens = word_tokenize(sentence.lower())  # Tokenize the sentence
                if target_term in tokens:  # Check for exact match
                    outfile.write(f"{sentence}\t{year}\n")
                    lines_written += 1

            # Handle three-column format: sentence, year, and label (e.g., journal_title or donor/natural label)
            elif len(parts) == 3:
                sentence, year, label = parts
                tokens = word_tokenize(sentence.lower())  # Tokenize the sentence
                if target_term in tokens:  # Check for exact match
                    outfile.write(f"{sentence}\t{year}\t{label}\n")
                    lines_written += 1
        
        # Print the summary of lines written for the current target term
        print(f"Lines containing '{target_term}' written to: {output_file} ({lines_written} sentences)")
    
    return lines_written  # Return the number of lines written

# Process natural corpus for each target term
for input_file in input_file:
    if 'Psychology' in input_file:
        corpus = 'psych'
        
        # Call function for the natural corpus
        for term in target_terms:
            filter_lines(input_file, term, corpus, output_suffix="")

print("Processing complete.")
