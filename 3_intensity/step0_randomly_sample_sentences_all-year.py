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

# Paths setup
natural_corpus_dir = "../0.0_corpus_preprocessing/output/natural_lines_targets"
synthetic_corpus_dir = "synthetic/output/all-year"
output_dir_high = "output/all-year/high"
output_dir_low = "output/all-year/low"

# Ensure directories exist
os.makedirs(output_dir_high, exist_ok=True)
os.makedirs(output_dir_low, exist_ok=True)
log_file_dir = "output/all-year"
os.makedirs(log_file_dir, exist_ok=True)
log_file_path = os.path.join(log_file_dir, "warnings.log")

# Logging function
def log_message(message):
    with open(log_file_path, "a") as log_file:
        log_file.write(message + "\n")

# Load natural corpus data
def load_corpus(file_path):
    return pd.read_csv(file_path, sep="\t", names=["sentence", "year"])

# Load synthetic file and handle different intensity levels
def load_synthetic_file(file_path):
    df = pd.read_csv(file_path)
    df.columns = ['baseline', 'high_intensity', 'low_intensity']
    return df

# Sampling function without frequency cap
def bootstrap_sample(df, sample_size, num_repeats, column_name='sentence'):
    all_samples = []
    for _ in range(num_repeats):
        samples = df[column_name].sample(n=sample_size, replace=True, random_state=np.random.randint(0, 10000)).tolist()
        all_samples.append(samples)
    return all_samples

# Process targets function
def process_target(target):
    natural_file_path = os.path.join(natural_corpus_dir, f"{target}.lines.psych")
    synthetic_file_path = os.path.join(synthetic_corpus_dir, f"{target}_synthetic_sentences.csv")

    natural_corpus = load_corpus(natural_file_path)
    synthetic_variations = load_synthetic_file(synthetic_file_path)

    for ratio in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        num_synthetic = int(SENTENCE_SAMPLE_SIZE * ratio)
        num_natural = SENTENCE_SAMPLE_SIZE - num_synthetic

        natural_samples = bootstrap_sample(natural_corpus, num_natural, NUM_REPEATS, 'sentence')
        high_samples = bootstrap_sample(synthetic_variations, num_synthetic, NUM_REPEATS, 'high_intensity')
        low_samples = bootstrap_sample(synthetic_variations, num_synthetic, NUM_REPEATS, 'low_intensity')

        for repeat, (natural_sample, high_sample, low_sample) in enumerate(zip(natural_samples, high_samples, low_samples), start=1):
            output_path_high = os.path.join(output_dir_high, f"{target}_synthetic_intensity_{int(ratio * 100)}.{repeat}.tsv")
            output_path_low = os.path.join(output_dir_low, f"{target}_synthetic_intensity_{int(ratio * 100)}.{repeat}.tsv")
            
            # Save high samples without headers
            with open(output_path_high, 'w') as file:
                for line in natural_sample + high_sample:
                    file.write(f"{line}\tnatural\n" if line in natural_sample else f"{line}\tsynthetic_intensity\n")
            
            # Save low samples without headers
            with open(output_path_low, 'w') as file:
                for line in natural_sample + low_sample:
                    file.write(f"{line}\tnatural\n" if line in natural_sample else f"{line}\tsynthetic_intensity\n")

            log_message(f"Saved samples for {target} at ratio {ratio}, repeat {repeat}.")

if __name__ == "__main__":
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        list(tqdm(executor.map(process_target, ["trauma", "anxiety", "depression", "mental_health", "mental_illness", "abuse"]), total=6))
    log_message("Sampling completed")
