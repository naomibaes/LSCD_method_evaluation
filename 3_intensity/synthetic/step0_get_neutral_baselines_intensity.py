import pandas as pd
import os
import re

# Define file paths (adjusted for arousal only)
input_dir = os.path.abspath(r"..\..\0.0_corpus_preprocessing\output\natural_lines_targets")
vad_file_path = os.path.abspath(r"input\NRC-VAD-Lexicon.txt")
output_folder = os.path.abspath(r"input\baselines")  # Baselines subdirectory

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

# Sort by arousal and calculate the dynamic neutral range
arousal_sorted = vad_ratings.sort_values(by="arousal")


def get_dynamic_range(df, target_count=1000):
    mid_start = (len(df) // 2) - (target_count // 2)
    mid_range = df.iloc[mid_start: mid_start + target_count]["arousal"]
    return mid_range.min(), mid_range.max()


arousal_range = get_dynamic_range(arousal_sorted, target_count=1000)
print(f"Dynamic neutral arousal range: {arousal_range}")

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
    neutral_sentences = []

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

                    # Check if the sentence falls into the neutral arousal range
                    if arousal is not None:
                        if arousal_range[0] <= arousal <= arousal_range[1]:
                            neutral_sentences.append((sentence, year, arousal))

        # Save all neutral sentences for the target
        output_file = os.path.join(output_folder, f"{target}_neutral_baselines_intensity.csv")
        neutral_df = pd.DataFrame(neutral_sentences, columns=["sentence", "year", "mean_arousal"])
        neutral_df.to_csv(output_file, index=False)

        # Print the number of sentences written
        print(f"Saved neutral arousal sentences for {target} to {output_file}.")
        print(f"Total neutral sentences for {target}: {len(neutral_sentences)}")
    else:
        print(f"File not found for target: {target}. Expected at: {file_path}")

print("All processing complete.")
