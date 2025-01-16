import pandas as pd
import os
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import numpy as np

# Constants
RANDOM_SEED = 93
SENTENCE_SAMPLE_SIZE = 50
NUM_REPEATS = 100

# Paths setup
natural_corpus_dir = "../0.0_corpus_preprocessing/output/natural_lines_targets"
synthetic_corpus_dir = "synthetic/output/all-year"
output_dir = "output/all-year"

os.makedirs(output_dir, exist_ok=True)
log_file_dir = output_dir
os.makedirs(log_file_dir, exist_ok=True)
log_file_path = os.path.join(log_file_dir, "warnings.log")

# Logging function
def log_message(message):
    with open(log_file_path, "a") as log_file:
        log_file.write(f"{message}\n")

# Load data functions
def load_corpus(file_path):
    data = pd.read_csv(file_path, sep="\t", names=["sentence", "year"])
    log_message(f"Loaded natural corpus from {file_path} with {len(data)} entries.")
    return data

def load_synthetic_file(file_path, variation_type='positive'):
    df = pd.read_csv(file_path)
    column_name = 'positive_variation' if variation_type == 'positive' else 'negative_variation'
    if column_name in df.columns and not df[column_name].empty:
        log_message(f"Loaded {len(df[column_name])} {variation_type} entries from {file_path}.")
        return pd.DataFrame(df[column_name], columns=['sentence'])
    else:
        log_message(f"No data found in column {column_name} at {file_path}.")
        return pd.DataFrame()

# Sampling function with dynamic frequency cap
def bootstrap_sample(df, sample_size, num_repeats):
    if df.empty:
        log_message("Dataframe is empty, skipping sampling.")
        return []
    max_repetition = calculate_dynamic_cap(df, sample_size, num_repeats)
    all_samples = []
    sentence_counts = {}

    for _ in range(num_repeats):
        iteration_samples = []
        while len(iteration_samples) < sample_size:
            sample = df.sample(n=1, replace=True)
            sentence = sample.iloc[0]['sentence']

            if sentence_counts.get(sentence, 0) < max_repetition:
                iteration_samples.append(sentence)
                sentence_counts[sentence] = sentence_counts.get(sentence, 0) + 1

        all_samples.append(iteration_samples)
    return all_samples

# Process targets
def process_target(target, variation_type='positive'):
    natural_file_path = os.path.join(natural_corpus_dir, f"{target}.lines.psych")
    synthetic_file_path = os.path.join(synthetic_corpus_dir, f"{target}_synthetic_sentences.csv")

    natural_corpus = load_corpus(natural_file_path)
    synthetic_variations = load_synthetic_file(synthetic_file_path, variation_type)

    if natural_corpus.empty or synthetic_variations.empty:
        log_message(f"Skipping {target} due to insufficient data.")
        return

    for ratio in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        num_synthetic = int(SENTENCE_SAMPLE_SIZE * ratio)
        num_natural = SENTENCE_SAMPLE_SIZE - num_synthetic

        natural_samples = bootstrap_sample(natural_corpus, num_natural, NUM_REPEATS)
        synthetic_samples = bootstrap_sample(synthetic_variations, num_synthetic, NUM_REPEATS)

        for repeat, samples in enumerate(zip(natural_samples, synthetic_samples), start=1):
            natural_sample, synthetic_sample = samples
            output_path = os.path.join(output_dir, f"{target}_{variation_type}_synthetic_{int(ratio * 100)}.{repeat}.tsv")

            combined_sample = pd.DataFrame({
                "sentence": natural_sample + synthetic_sample,
                "source": ["natural"] * len(natural_sample) + ["synthetic"] * len(synthetic_sample)
            })
            combined_sample.to_csv(output_path, sep="\t", index=False, header=False)

            log_message(f"Saved samples for {target} at ratio {ratio}, repeat {repeat}.")

if __name__ == "__main__":
    variation_type = 'positive'  # or 'negative', dynamically set based on your requirement
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        list(tqdm(executor.map(lambda target: process_target(target, variation_type), ["trauma", "anxiety", "depression", "mental_health", "mental_illness", "abuse"]), total=6))
    log_message("Sampling completed")
