import os
import pandas as pd
import csv  # For improved CSV handling
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
base_dir = "output/all-year.cosine/control"
input_dir = os.path.join(base_dir, "input")
output_dir = base_dir  # Output remains in the control directory

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

log_file_path = os.path.join(base_dir, "warnings.log")

def log_message(message):
    """Logs messages to a file."""
    with open(log_file_path, "a") as log_file:
        log_file.write(message + "\n")

def load_corpus(file_path):
    """Loads corpus data from a CSV file, handling commas properly."""
    try:
        df = pd.read_csv(file_path, sep=",", names=["sentence", "label"], dtype=str, quoting=csv.QUOTE_MINIMAL, engine="python")
        df = df.dropna()  # Remove any empty rows
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
        df.sample(n=sample_size, replace=True, random_state=rng).values.tolist()
        for _ in range(num_repeats)
    ]
    return all_samples

def process_target(target):
    """Processes each target term across rounds and saves output as TSV."""
    natural_path = os.path.join(input_dir, "natural", f"{target}_all-year_breadth_natural.csv")
    synthetic_path = os.path.join(input_dir, "synthetic", f"{target}_all-year_breadth_synthetic.csv")

    if not os.path.exists(natural_path) or not os.path.exists(synthetic_path):
        log_message(f"Skipping {target} - missing files.")
        return

    natural_corpus = load_corpus(natural_path)
    synthetic_corpus = load_corpus(synthetic_path)

    if natural_corpus is None or synthetic_corpus is None:
        log_message(f"Skipping {target} - empty corpus.")
        return

    num_synthetic = int(SENTENCE_SAMPLE_SIZE * INJECTION_RATIO)
    num_natural = SENTENCE_SAMPLE_SIZE - num_synthetic

    for round_num in range(1, NUM_ROUNDS + 1):  # 6 rounds
        natural_samples = bootstrap_sample(natural_corpus, num_natural, NUM_REPEATS)
        synthetic_samples = bootstrap_sample(synthetic_corpus, num_synthetic, NUM_REPEATS)

        for repeat, (natural_sample, synthetic_sample) in enumerate(zip(natural_samples, synthetic_samples), start=1):
            output_path = os.path.join(output_dir, f"{target}_synthetic_breadth_50_round{round_num}.{repeat}.tsv")

            # Ensure directory exists before writing
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            # Convert to DataFrame and write as TSV (Tab-Separated Values)
            output_df = pd.DataFrame(natural_sample + synthetic_sample, columns=["sentence", "label"])
            output_df.to_csv(output_path, sep="\t", index=False, header=False, quoting=csv.QUOTE_NONE)

            log_message(f"Saved samples for {target} - Round {round_num}, Repeat {repeat}.")

if __name__ == "__main__":
    targets = ["trauma", "anxiety", "depression", "mental_health", "mental_illness", "abuse"]
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        list(tqdm(executor.map(process_target, targets), total=len(targets)))
    log_message("Sampling completed.")
