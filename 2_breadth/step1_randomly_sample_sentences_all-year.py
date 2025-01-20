import os
import random
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

# Constants
RANDOM_SEED = 93
SENTENCE_SAMPLE_SIZE = 50
NUM_REPEATS = 100

# Paths setup
natural_corpus_dir = "../0.0_corpus_preprocessing/output/natural_lines_targets"
synthetic_corpus_dir = "synthetic/output/unique_all-year"
output_dir = "output/all-year.cosine"
os.makedirs(output_dir, exist_ok=True)
log_file_path = os.path.join(output_dir, "warnings.log")

def log_message(message):
    with open(log_file_path, "a") as log_file:
        log_file.write(f"{message}\n")

def load_corpus(file_path):
    df = pd.read_csv(file_path, sep="\t", names=["sentence", "year"])
    df['type'] = 'natural'
    return df[['sentence', 'type']]

def load_synthetic_file(file_path):
    df = pd.read_csv(file_path)
    df.columns = ['sentence', 'type', 'year']
    return df[['sentence', 'type']]

def bootstrap_sample(df, sample_size):
    if df.empty:
        return []
    return df.sample(n=sample_size, replace=True, random_state=random.randint(0, 100000)).values.tolist()

def process_target(target):
    natural_corpus = load_corpus(os.path.join(natural_corpus_dir, f"{target}.lines.psych"))
    synthetic_corpus = load_synthetic_file(os.path.join(synthetic_corpus_dir, f"{target}_synthetic_sentences.csv"))

    for ratio in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        num_synthetic = int(SENTENCE_SAMPLE_SIZE * ratio)
        num_natural = SENTENCE_SAMPLE_SIZE - num_synthetic

        for repeat in range(1, NUM_REPEATS + 1):
            natural_samples = bootstrap_sample(natural_corpus, num_natural)
            synthetic_samples = bootstrap_sample(synthetic_corpus, num_synthetic)

            combined_samples = natural_samples + synthetic_samples
            combined_sample = pd.DataFrame(combined_samples)
            output_file_path = os.path.join(output_dir, f"{target}_synthetic_breadth_{int(ratio * 100)}_{repeat}.csv")
            combined_sample.to_csv(output_file_path, index=False, header=False)
            log_message(f"Saved {len(combined_samples)} samples for {target} at ratio {ratio}, repeat {repeat}.")

if __name__ == "__main__":
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        list(tqdm(executor.map(process_target, ["trauma", "anxiety", "depression", "mental_health", "mental_illness", "abuse"])))
    log_message("Sampling completed")
