import os
import pandas as pd
import glob
import re
from collections import defaultdict

# Setup
input_dir = "output/5-year.cosine"
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

targets = ["trauma", "anxiety", "depression", "mental_health", "mental_illness", "abuse"]

# Function to analyze sibling counts
def analyze_siblings():
    sibling_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

    # Process each target
    for target in targets:
        # Adjust target name for file matching
        file_target = target.replace(' ', '_')  # Ensure the target name matches file convention
        target_path = os.path.join(input_dir, f"{file_target}_*synthetic_*")
        files = glob.glob(target_path)

        # Analyze each file
        for file in files:
            filename = os.path.basename(file)
            # Use regex to extract the period more accurately
            match = re.match(rf"{file_target}_(\d{{4}}-\d{{4}})_synthetic_(\d+\.\d+)", filename)
            if match:
                period = match.group(1)
                injection_ratio = match.group(2)

                # Load data
                data = pd.read_csv(file, sep="\t", header=None, names=["sentence", "year", "type"])
                data['type'] = data['type'].astype(str)  # Convert 'type' to string

                # Filter only synthetic sentences
                synthetic_sentences = data[data['type'].str.contains("synthetic")]

                # Count siblings in the synthetic labels
                synthetic_sentences['sibling'] = synthetic_sentences['type'].str.extract(r"synthetic_(\S+)")
                for _, row in synthetic_sentences.iterrows():
                    sibling_counts[period][target][row['sibling']] += 1

    # Prepare and save the counts to a CSV
    all_data = []
    for period, target_dict in sibling_counts.items():
        for target, counts in target_dict.items():
            sorted_siblings = sorted(counts.items(), key=lambda x: x[1], reverse=True)
            for sibling, count in sorted_siblings:
                all_data.append({'Period': period, 'Target': target, 'Sibling': sibling, 'Count': count})

    # Convert to DataFrame and save
    df = pd.DataFrame(all_data)
    df.to_csv(os.path.join(output_dir, 'sibling_frequencies_5-year.cosine.csv'), index=False)

# Execute
if __name__ == "__main__":
    analyze_siblings()
