# Author: Naomi Baes and Chat GPT

import pandas as pd
import os
import numpy as np
import nltk
from nltk.tokenize import word_tokenize

# Ensure NLTK tokenizers are downloaded
nltk.download('punkt')

# Define file paths
input_dir = os.path.abspath(os.path.join("..", "..", "0.0_corpus_preprocessing", "output", "natural_lines_targets"))
vad_file_path = os.path.abspath(os.path.join("input", "NRC-VAD-Lexicon.txt"))
output_folder = os.path.abspath(os.path.join("input", "baselines"))  # Baselines subdirectory
output_folder2 = os.path.abspath("input/baselines/output")  # Define a generic output directory

# Ensure the baselines folder exists
os.makedirs(output_folder, exist_ok=True)
os.makedirs(output_folder2, exist_ok=True)

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

# Map words to sentiment for lookup
sentiment_dict = {row["word"]: row["valence"] for _, row in vad_ratings.iterrows()}

# Function to calculate sentence-level sentiment using NLTK tokenization
def calculate_sentence_sentiment(sentence):
    words = word_tokenize(sentence.lower())  # Tokenize sentence using NLTK
    sentiment_sum, count = 0.0, 0
    for word in words:
        if word in sentiment_dict:
            sentiment_sum += sentiment_dict[word]
            count += 1
    if count > 0:
        return sentiment_sum / count
    return None

# Initialize a summary list for neutral ranges
neutral_range_summary = []

# Process each target
for target in targets:
    neutral_sentences = []

    # Process each file that matches the target
    target_file = f"{target}.lines.psych"
    file_path = os.path.join(input_dir, target_file)
    if os.path.exists(file_path):
        print(f"Processing file: {file_path}")

        # Read sentences and calculate sentiments
        sentences = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    sentence, year = parts[0], int(parts[1])
                    if 1970 <= year <= 2019:
                        sentiment = calculate_sentence_sentiment(sentence)
                        if sentiment is not None:
                            sentences.append((sentence, year, sentiment))

        # Create a DataFrame for the sentences
        sentences_df = pd.DataFrame(sentences, columns=["sentence", "year", "sentiment"])

        # Calculate quartiles and median for the target
        median = sentences_df["sentiment"].median()
        q1 = sentences_df["sentiment"].quantile(0.25)
        q3 = sentences_df["sentiment"].quantile(0.75)

        # Group by 5-year epochs
        sentences_df["epoch"] = (sentences_df["year"] // 5) * 5
        grouped = sentences_df.groupby("epoch")

        for epoch, group in grouped:
            # Initialize the range with the median
            lower, upper = median, median
            selected_sentences = group[(group["sentiment"] >= lower) & (group["sentiment"] <= upper)]

            # Expand range dynamically to hit at least 500 sentences, up to 1500 max
            while len(selected_sentences) < 500 and (lower > q1 or upper < q3):
                if lower > q1:
                    lower -= 0.01
                if upper < q3:
                    upper += 0.01
                selected_sentences = group[(group["sentiment"] >= lower) & (group["sentiment"] <= upper)]

            # Cap at 1500 sentences if possible
            if len(selected_sentences) > 1500:
                selected_sentences = selected_sentences.sample(n=1500, random_state=42)

            # Save selected sentences for each epoch and target
            epoch_file = os.path.join(output_folder, f"{target}_{epoch}-{epoch+4}.baseline_1500_sentences.csv")
            selected_sentences.to_csv(epoch_file, index=False)

            # Append to neutral range summary
            neutral_range_summary.append({
                "target": target,
                "epoch": f"{epoch}-{epoch+4}",
                "lower_bound": lower,
                "upper_bound": upper,
                "num_sentences": len(selected_sentences)
            })

    else:
        print(f"File not found for target: {target}. Expected at: {file_path}")

# Save the neutral range summary
summary_file = os.path.join(output_folder2, "0_neutral_range_summary.csv")
summary_df = pd.DataFrame(neutral_range_summary)
summary_df.to_csv(summary_file, index=False)
print(f"Neutral range summary saved to {summary_file}.")
