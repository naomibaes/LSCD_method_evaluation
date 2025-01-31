import os
import pandas as pd
import csv
import re
from nltk.tokenize import PunktSentenceTokenizer
import nltk

# Download Punkt models if not already downloaded
nltk.download('punkt')

# Initialize Punkt tokenizer
tokenizer = PunktSentenceTokenizer()

# Input file for the dataset
input_file = "C:/Users/naomi/OneDrive/COMP80004_PhDResearch/RESEARCH/DATA/CORPORA/Psychology/abstract_year_journal.csv.mental.sentence-CLEAN.tsv"

# Output directory and files
output_directory = "output"
output_file = "year_counts_sentences.csv"
stats_file = "sentence_stats.txt"

# Terms to filter (check frequencies of candidate terms in the corpus)
targets = ["mental_health", "mental_illness", "mental_disorder", "trauma", "addiction", 
           "bullying", "harassment", "abuse", "anger", "distress", "empathy", "grief", 
           "anxiety", "depression", "stress", "suffering", "worry", "automobile", "bank", "camera", "car", "deer", 
           "export", "founder", "horse", "mail", "mirror", "music", "newspaper", 
           "paper", "phone", "ship", "symptom", "telephone", "train", "travel"]

# Create the output directory if it doesn't exist
os.makedirs(output_directory, exist_ok=True)

# Initialize list to hold dataframes for each term and corpus
dfs = []

# Initialize variables to track total sentence count and target-specific sentence counts
total_sentence_count = 0
target_sentence_counts = {target: 0 for target in targets}

# Check if the input file exists
if not os.path.exists(input_file):
    print(f"File {input_file} does not exist.")
else:
    # Read the input file and extract counts
    with open(input_file, "r", encoding="utf-8") as infile:
        reader = csv.reader(infile, delimiter='\t')  # Use tab as the delimiter
        header = next(reader)  # Read header
        print(f"Header: {header}")  # Print header for debugging
        
        # Ensure the file has the correct number of columns (two columns expected)
        if len(header) != 2:
            print(f"Expected 2 columns but found {len(header)}. Check input file structure.")
        else:
            # Process each line in the input file
            for line in reader:
                if len(line) != 2:  # Ensure there are exactly two columns
                    continue
                
                text, year = line[0].strip(), line[1].strip()

                # Use Punkt tokenizer to split text into sentences
                sentences = tokenizer.tokenize(text)

                # Increment the total sentence count for each valid line
                total_sentence_count += len(sentences)
                
                # Convert text to lowercase for case-insensitive matching
                for sentence in sentences:
                    text_lower = sentence.lower()

                    # Check for each target term in the text and increment its count
                    for target in targets:
                        pattern = rf"\b{re.escape(target)}\b"  # Regular expression for exact match
                        if re.search(pattern, text_lower):
                            target_sentence_counts[target] += 1  # Increment target sentence count
                            if year.isdigit() and 1970 <= int(year) <= 2019:
                                # Append counts to the list
                                dfs.append({'target': target, 'corpus': 'psych', 'year': year, 'count': 1})

    # Create a DataFrame from the list of counts
    if dfs:
        aggregated_df = pd.DataFrame(dfs)
        aggregated_df = aggregated_df.groupby(['target', 'corpus', 'year']).count().reset_index()

        # Write the aggregated dataframe to a CSV file
        output_path = os.path.join(output_directory, output_file)
        aggregated_df.to_csv(output_path, index=False)

        print(f"Data written to {output_path}")
    else:
        print("No data found for the specified target and corpora.")
    
    # Write the total sentence count and target term counts to the stats file
    stats_output_path = os.path.join(output_directory, stats_file)
    with open(stats_output_path, 'w', encoding='utf-8') as stats_file:
        stats_file.write(f"Total sentence count across the corpus: {total_sentence_count}\n")
        for target, count in target_sentence_counts.items():
            stats_file.write(f"Total sentence count with '{target}': {count}\n")

    print(f"Sentence statistics written to {stats_output_path}")
