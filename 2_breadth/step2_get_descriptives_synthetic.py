import os
import pandas as pd
import glob
import re
from collections import defaultdict

# Setup
input_dir = "synthetic/output/unique_all-year"
output_dir = "synthetic/output/unique_all-year"
os.makedirs(output_dir, exist_ok=True)

# Targets are optional here, as the script will process all relevant files in the input directory
targets = ["trauma", "anxiety", "depression", "mental_health", "mental_illness", "abuse"]

# Function to load data
def load_data(file):
    # Assuming comma-separated values (CSV) with a header
    return pd.read_csv(file, sep=",", header=0, names=["sentence", "type", "year"])

# Function to analyze sibling counts
def analyze_siblings():
    sibling_counts = defaultdict(lambda: defaultdict(int))

    # Process each target
    for file_target in targets:
        # Look for files containing the target name
        target_path = os.path.join(input_dir, f"{file_target}_synthetic_sentences.csv")
        files = glob.glob(target_path)

        for file in files:
            data = load_data(file)

            # Ensure the 'type' column is string
            data['type'] = data['type'].astype(str)

            # Extract sibling information from the 'type' column (e.g., "synthetic_anxiety")
            synthetic_sentences = data[data['type'].str.contains("synthetic")]
            synthetic_sentences['sibling'] = synthetic_sentences['type'].str.extract(r"synthetic_(\w+)")

            # Count sibling occurrences
            for _, row in synthetic_sentences.iterrows():
                sibling_counts[file_target][row['sibling']] += 1

    # Prepare data for output
    all_data = []
    for target, counts in sibling_counts.items():
        sorted_siblings = sorted(counts.items(), key=lambda x: x[1], reverse=True)
        for sibling, count in sorted_siblings:
            all_data.append({'Target': target, 'Sibling': sibling, 'Count': count})

    # Convert to DataFrame and save
    df = pd.DataFrame(all_data)
    output_file_path = os.path.join(output_dir, 'z_sibling_frequencies_1500.csv')
    df.to_csv(output_file_path, index=False)
    print(f"Sibling frequencies saved to {output_file_path}")

if __name__ == "__main__":
    analyze_siblings()
