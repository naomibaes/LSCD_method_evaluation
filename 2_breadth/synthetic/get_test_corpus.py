# Create small test set of corpus

import pandas as pd

# File paths
input_file_path = "C:/Users/naomi/OneDrive/COMP80004_PhDResearch/RESEARCH/DATA/CORPORA/Psychology/abstract_year_journal.csv.mental.sentence-CLEAN.tsv"
output_file_path = "C:/Users/naomi/OneDrive/COMP80004_PhDResearch/RESEARCH/DATA/CORPORA/Psychology/abstract_year_journal.csv.mental.sentence-CLEAN.TEST.tsv"

# Number of rows to sample for the test set
sample_size = 100000  # Adjust this number based on the size you want for the test set

# Load the input file
df = pd.read_csv(input_file_path, sep='\t')

# Randomly sample rows
test_set = df.sample(n=sample_size, random_state=42)

# Save the sampled test set to a new file
test_set.to_csv(output_file_path, sep='\t', index=False)

print(f"Test corpus saved to: {output_file_path}")