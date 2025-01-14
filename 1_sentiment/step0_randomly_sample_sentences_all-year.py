# Authors: Naomi Baes and Chat GPT

import pandas as pd
import os
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import numpy as np

# Constants
RANDOM_SEED = 93
SENTENCE_SAMPLE_SIZE = 50
NUM_REPEATS = 100
MAX_SENTENCE_REPETITION = 3
DATASET_SIZE_THRESHOLD = 97478

# Paths setup
natural_corpus_dir = "../0.0_corpus_preprocessing/output/natural_lines_targets"
synthetic_corpus_dir = "synthetic/output/all-year"
output_dir_positive = "output/all-year/positive"
output_dir_negative = "output/all-year/negative"

os.makedirs(output_dir_positive, exist_ok=True)
os.makedirs(output_dir_negative, exist_ok=True)
log_file_dir = "output/all-year"
os.makedirs(log_file_dir, exist_ok=True)
log_file_path = os.path.join(log_file_dir, "warnings.log")

# Logging function
def log_message(message):
    with open(log_file_path, "a") as log_file:
        log_file.write(message + "\n")

# Load data functions
def load_corpus(file_path):
    return pd.read_csv(file_path, sep="\t", names=["sentence", "year"])

def load_synthetic_file(file_path):
    return pd.read_csv(file_path)

# Sampling function with frequency cap
def bootstrap_sample(df, sample_size, num_repeats, max_repetition, apply_cap=False):
    all_samples = []
    sentence_counts = {}

    if apply_cap:
        # Dynamically calculate max repetition based on dataset size to prevent over-representation
        total_samples = sample_size * num_repeats
        max_repetition = max(3, int(0.05 * total_samples / len(df)))  # Ensure at least 3 or adjusted cap

    for _ in range(num_repeats):
        iteration_samples = []
        while len(iteration_samples) < sample_size:
            sample = df.sample(n=1, replace=True, random_state=np.random.randint(0, 10000))
            sentence = sample.iloc[0]['sentence'] if isinstance(sample, pd.DataFrame) else sample.iloc[0]

            if apply_cap:
                if sentence_counts.get(sentence, 0) < max_repetition:
                    iteration_samples.append(sentence)
                    sentence_counts[sentence] = sentence_counts.get(sentence, 0) + 1
            else:
                iteration_samples.append(sentence)
        all_samples.append(iteration_samples)
    return all_samples

# Process targets
def process_target(target):
    natural_file_path = os.path.join(natural_corpus_dir, f"{target}.lines.psych")
    synthetic_file_path = os.path.join(synthetic_corpus_dir, f"{target}_synthetic_sentences.csv")

    natural_corpus = load_corpus(natural_file_path)
    synthetic_variations = load_synthetic_file(synthetic_file_path)

    if natural_corpus.empty:
        log_message(f"Natural corpus is empty for target {target}. Skipping.")
        return
    if synthetic_variations.empty:
        log_message(f"Synthetic variations are empty for target {target}. Skipping.")
        return

    apply_cap = len(natural_corpus) < DATASET_SIZE_THRESHOLD

    for ratio in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        num_synthetic = int(SENTENCE_SAMPLE_SIZE * ratio)
        num_natural = SENTENCE_SAMPLE_SIZE - num_synthetic

        natural_samples = bootstrap_sample(natural_corpus, num_natural, NUM_REPEATS, MAX_SENTENCE_REPETITION, apply_cap)
        positive_samples = bootstrap_sample(synthetic_variations['positive_variation'], num_synthetic, NUM_REPEATS, MAX_SENTENCE_REPETITION, apply_cap)
        negative_samples = bootstrap_sample(synthetic_variations['negative_variation'], num_synthetic, NUM_REPEATS, MAX_SENTENCE_REPETITION, apply_cap)

        for repeat, (natural_sample, positive_sample, negative_sample) in enumerate(zip(natural_samples, positive_samples, negative_samples), start=1):
            positive_output_path = os.path.join(output_dir_positive, f"{target}_synthetic_sentiment_{int(ratio * 100)}.{repeat}.tsv")
            negative_output_path = os.path.join(output_dir_negative, f"{target}_synthetic_sentiment_{int(ratio * 100)}.{repeat}.tsv")

            # Save positive samples
            positive_combined_sample = pd.DataFrame({
                "sentence": natural_sample + positive_sample,
                "source": ["natural"] * len(natural_sample) + ["synthetic_positive"] * len(positive_sample)
            })
            positive_combined_sample.to_csv(positive_output_path, sep="\t", index=False, header=False)

            # Save negative samples
            negative_combined_sample = pd.DataFrame({
                "sentence": natural_sample + negative_sample,
                "source": ["natural"] * len(natural_sample) + ["synthetic_negative"] * len(negative_sample)
            })
            negative_combined_sample.to_csv(negative_output_path, sep="\t", index=False, header=False)

            log_message(f"Saved samples for {target} at ratio {ratio}, repeat {repeat}.")

if __name__ == "__main__":
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        list(tqdm(executor.map(process_target, ["trauma", "anxiety", "depression", "mental_health", "mental_illness", "abuse"]), total=6))
    log_message("Sampling completed")
