import pandas as pd
import os
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import numpy as np

# Constants
RANDOM_SEED = 93
SENTENCE_SAMPLE_SIZE = 50
NUM_ROUNDS = 6
NUM_REPEATS = 100
INJECTION_RATIO = 0.5  # Fixed injection ratio

# Paths setup
base_dir = "output/all-year/control"
input_dirs = {
    "high": os.path.join(base_dir, "input", "high"),
    "low": os.path.join(base_dir, "input", "low")
}
output_dirs = {
    "high": os.path.join(base_dir, "high"),
    "low": os.path.join(base_dir, "low")
}

# Ensure output directories exist
for path in output_dirs.values():
    os.makedirs(path, exist_ok=True)

log_file_path = os.path.join(base_dir, "warnings.log")

def log_message(message):
    """Logs messages to a file."""
    with open(log_file_path, "a") as log_file:
        log_file.write(message + "\n")

def load_corpus(file_path):
    """Loads corpus data from a file, ensuring correct column parsing."""
    try:
        df = pd.read_csv(file_path, sep="\t", names=["sentence", "label"], dtype=str)
        if df.empty:
            log_message(f"Warning: {file_path} is empty.")
            return None
        print(f"Loaded corpus from {file_path}, first 5 rows:\n{df.head()}")  # Debugging print
        return df
    except Exception as e:
        log_message(f"Error loading {file_path}: {str(e)}")
        return None

def bootstrap_sample(df, sample_size, num_repeats):
    """Performs bootstrap sampling with a fixed random seed."""
    if df is None or df.empty:
        log_message("Warning: Attempting to sample from an empty dataset.")
        return [[] for _ in range(num_repeats)]
    
    rng = np.random.RandomState(RANDOM_SEED)  # Ensures reproducibility
    all_samples = [
        df.sample(n=sample_size, replace=True, random_state=rng)["sentence"].tolist()
        for _ in range(num_repeats)
    ]
    return all_samples

def process_target(target):
    """Processes each target term across rounds and conditions."""
    for condition in ["high", "low"]:
        natural_path = os.path.join(input_dirs[condition], "natural", f"{target}_all-year_intensity_natural.tsv")
        synthetic_path = os.path.join(input_dirs[condition], "synthetic", f"{target}_all-year_intensity_synthetic.tsv")

        if not os.path.exists(natural_path) or not os.path.exists(synthetic_path):
            log_message(f"Skipping {target} ({condition}) - missing files.")
            continue

        natural_corpus = load_corpus(natural_path)
        synthetic_corpus = load_corpus(synthetic_path)

        if natural_corpus is None or synthetic_corpus is None:
            log_message(f"Skipping {target} ({condition}) - empty corpus.")
            continue

        num_synthetic = int(SENTENCE_SAMPLE_SIZE * INJECTION_RATIO)
        num_natural = SENTENCE_SAMPLE_SIZE - num_synthetic

        for round_num in range(1, NUM_ROUNDS + 1):  # 6 rounds
            natural_samples = bootstrap_sample(natural_corpus, num_natural, NUM_REPEATS)
            synthetic_samples = bootstrap_sample(synthetic_corpus, num_synthetic, NUM_REPEATS)

            for repeat, (natural_sample, synthetic_sample) in enumerate(zip(natural_samples, synthetic_samples), start=1):
                output_path = os.path.join(output_dirs[condition], f"{target}_synthetic_intensity_50_round{round_num}.{repeat}.tsv")

                # Ensure directory exists before writing
                os.makedirs(os.path.dirname(output_path), exist_ok=True)

                with open(output_path, 'w') as file:
                    for line in natural_sample:
                        file.write(f"{line}\tnatural\n")
                    for line in synthetic_sample:
                        file.write(f"{line}\tsynthetic_{condition}\n")

                log_message(f"Saved samples for {target} ({condition}) - Round {round_num}, Repeat {repeat}.")

if __name__ == "__main__":
    targets = ["trauma", "anxiety", "depression", "mental_health", "mental_illness", "abuse"]
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        list(tqdm(executor.map(process_target, targets), total=len(targets)))
    log_message("Sampling completed.")
