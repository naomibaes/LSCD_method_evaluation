import pandas as pd
import os
import re

# Define file paths
input_dir = os.path.abspath(os.path.join("..", "..", "0.0_corpus_preprocessing", "output", "natural_lines_targets"))
vad_file_path = os.path.abspath(os.path.join("input", "NRC-VAD-Lexicon.txt"))
output_folder = os.path.abspath(os.path.join("input", "baselines"))  # Baselines subdirectory

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

# Sort by sentiment (valence) and calculate the dynamic neutral range
sentiment_sorted = vad_ratings.sort_values(by="valence")


def get_dynamic_range(df, target_count=1000):
    """Get the dynamic neutral range for sentiment."""
    mid_start = (len(df) // 2) - (target_count // 2)
    mid_range = df.iloc[mid_start: mid_start + target_count]["valence"]
    return mid_range.min(), mid_range.max()


sentiment_range = get_dynamic_range(sentiment_sorted, target_count=1000)
print(f"Dynamic neutral sentiment range: {sentiment_range}")

# Map words to sentiment for lookup
sentiment_dict = {row["word"]: row["valence"] for _, row in vad_ratings.iterrows()}

# Function to calculate sentence-level sentiment
def calculate_sentence_sentiment(sentence):
    words = re.findall(r'\w+', sentence.lower())
    sentiment_sum, count = 0.0, 0
    for word in words:
        if word in sentiment_dict:
            sentiment_sum += sentiment_dict[word]
            count += 1
    if count > 0:
        return sentiment_sum / count
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
                    sentiment = calculate_sentence_sentiment(sentence)

                    # Check if the sentence falls into the neutral sentiment range
                    if sentiment is not None:
                        if sentiment_range[0] <= sentiment <= sentiment_range[1]:
                            neutral_sentences.append((sentence, year, sentiment))

        # Save all neutral sentences for the target
        output_file = os.path.join(output_folder, f"{target}_neutral_baselines_sentiment.csv")
        neutral_df = pd.DataFrame(neutral_sentences, columns=["sentence", "year", "mean_sentiment"])
        neutral_df.to_csv(output_file, index=False)

        # Print the number of sentences written
        print(f"Saved neutral sentiment sentences for {target} to {output_file}.")
        print(f"Total neutral sentences for {target}: {len(neutral_sentences)}")
    else:
        print(f"File not found for target: {target}. Expected at: {file_path}")

print("All processing complete.")
