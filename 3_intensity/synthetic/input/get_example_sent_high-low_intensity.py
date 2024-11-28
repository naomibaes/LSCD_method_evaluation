import pandas as pd
import os
import re

# Define file paths
input_dir = r"C:\Users\naomi\OneDrive\COMP80004_PhDResearch\RESEARCH\PROJECTS\3_evaluation+validation - ACL 2025\ICL\0.0_corpus_preprocessing\output\natural_lines_targets"
vad_file_path = os.path.abspath(os.path.join("NRC-VAD-Lexicon.txt"))
output_folder = os.path.abspath(os.path.join("example_inspiration"))  # Baselines subdirectory

# Ensure the baselines folder exists
os.makedirs(output_folder, exist_ok=True)

# Target terms
targets = ["abuse", "anxiety", "depression", "mental_health", "mental_illness", "trauma"]

# Check if the input directory exists
if not os.path.exists(input_dir):
    raise FileNotFoundError(f"Input directory not found: {input_dir}")

# Load VAD ratings into a DataFrame
if not os.path.exists(vad_file_path):
    raise FileNotFoundError(f"VAD ratings file not found: {vad_file_path}")

vad_ratings = pd.read_csv(vad_file_path, sep="\t", header=0)
vad_ratings.columns = ["word", "valence", "arousal", "dominance"]

# Map words to arousal for lookup
arousal_dict = {row["word"]: row["arousal"] for _, row in vad_ratings.iterrows()}

# Function to calculate sentence-level arousal
def calculate_sentence_arousal(sentence):
    words = re.findall(r'\w+', sentence.lower())
    arousal_sum, count = 0.0, 0
    for word in words:
        if word in arousal_dict:
            arousal_sum += arousal_dict[word]
            count += 1
    if count > 0:
        return arousal_sum / count
    return None

# Process each target
for target in targets:
    sentences_with_scores = []

    # Process each file that matches the target
    target_file = f"{target}.lines.psych"
    file_path = os.path.join(input_dir, target_file)
    if os.path.exists(file_path):
        print(f"Processing file: {file_path}")
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    sentence, year = parts[0], int(parts[1])
                    arousal = calculate_sentence_arousal(sentence)

                    # Save all sentences with calculated arousal
                    if arousal is not None:
                        sentences_with_scores.append((sentence, year, arousal))

        # Convert to DataFrame for sorting
        sentence_df = pd.DataFrame(sentences_with_scores, columns=["sentence", "year", "mean_arousal"])

        # Sort by arousal and select top/bottom 50
        top_50 = sentence_df.nlargest(50, "mean_arousal").copy()
        top_50["category"] = "high"

        bottom_50 = sentence_df.nsmallest(50, "mean_arousal").copy()
        bottom_50["category"] = "low"

        # Combine and save to a single CSV
        combined_df = pd.concat([top_50, bottom_50])
        output_file = os.path.join(output_folder, f"{target}_high_low_arousal.csv")
        combined_df.to_csv(output_file, index=False)

        # Print the number of sentences written
        print(f"Saved high/low arousal sentences for {target} to {output_file}.")
        print(f"Total sentences processed for {target}: {len(sentences_with_scores)}")
    else:
        print(f"File not found for target: {target}. Expected at: {file_path}")

print("All processing complete.")
