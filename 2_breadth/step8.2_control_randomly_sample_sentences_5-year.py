# Authors: Naomi Baes and ChatGPT

import os
import pandas as pd
import random
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import cpu_count

# Constants
SENTENCE_SAMPLE_SIZE = 50
NUM_REPEATS = 10
output_dir = "output/5-year.cosine/control"

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Define the log file path
log_file_path = os.path.join(output_dir, "warnings.log")

# Input directory
base_input_dir = "output/5-year.cosine/control/input"

# Define the targets
targets = ['abuse', 'anxiety', 'depression', 'mental_health', 'mental_illness', 'trauma']
injection_ratios = [0.5] * 6  # All ratios are 0.5, but we differentiate rounds

# Logging helper
def log_message(message, log_buffer=[], max_lines=1000):
    log_buffer.append(message)
    if len(log_buffer) >= 50:
        with open(log_file_path, "a", encoding="utf-8") as log_file:
            log_file.write("\n".join(log_buffer) + "\n")
        log_buffer.clear()

# Create intervals
def create_intervals(start_year, end_year, step=5):
    return [(i, i + step - 1) for i in range(start_year, end_year, step)]

# Process target and interval
def process_target_and_interval(args):
    target, interval = args
    start, end = interval
    interval_label = f"{start}-{end}"
    log_message(f"Processing target '{target}' for interval {interval_label}...")

    # Load natural sentences
    natural_file_path = os.path.join(base_input_dir, "natural", f"{target}_{interval_label}_5-year_breadth_natural.tsv")
    if not os.path.exists(natural_file_path):
        log_message(f"Natural file not found: {natural_file_path}")
        return

    natural_corpus = pd.read_csv(natural_file_path, sep="\t", header=None, names=["sentence", "year", "type"])
    interval_natural = natural_corpus.query(f"{start} <= year <= {end}")

    # Load synthetic sentences
    synthetic_file_path = os.path.join(base_input_dir, "synthetic", f"{target}_{interval_label}_5-year_breadth_synthetic.tsv")
    if not os.path.exists(synthetic_file_path):
        log_message(f"Synthetic file not found: {synthetic_file_path}")
        return

    synthetic_corpus = pd.read_csv(synthetic_file_path, sep="\t", header=None, names=["sentence", "year", "type"])

    for repeat in range(1, NUM_REPEATS + 1):
        for ratio_idx, ratio in enumerate(injection_ratios):
            num_synthetic = int(SENTENCE_SAMPLE_SIZE * ratio)
            num_natural = SENTENCE_SAMPLE_SIZE - num_synthetic

            # Adjust sample sizes to avoid errors
            if len(interval_natural) < num_natural:
                log_message(f"Insufficient natural sentences for {target} during {interval_label}: expected {num_natural}, available {len(interval_natural)}")
                num_natural = len(interval_natural)

            if len(synthetic_corpus) < num_synthetic:
                log_message(f"Insufficient synthetic sentences for {target} during {interval_label}: expected {num_synthetic}, available {len(synthetic_corpus)}")
                num_synthetic = len(synthetic_corpus)

            natural_sample = interval_natural.sample(n=num_natural, random_state=repeat, replace=False)
            synthetic_sample = synthetic_corpus.sample(n=num_synthetic, random_state=repeat, replace=False)

            # Combine and shuffle samples
            combined_sample = pd.concat([natural_sample, synthetic_sample])
            combined_sample = combined_sample.sample(frac=1, random_state=repeat)

            # Define the output directory
            os.makedirs(output_dir, exist_ok=True)

            # Save to output directory with the injection ratio round in the file name
            output_file_path = os.path.join(output_dir, f"{target}_{interval_label}_synthetic_50_round{ratio_idx + 1}.{repeat}.tsv")
            combined_sample.to_csv(output_file_path, index=False, header=False, sep='\t')  # No column labels

    log_message(f"Completed target '{target}' for interval {interval_label}.")

# Main execution
if __name__ == "__main__":
    intervals = create_intervals(1970, 2020)
    tasks = [(target, interval) for target in targets for interval in intervals]
    with ThreadPoolExecutor(max_workers=cpu_count()) as executor:
        list(tqdm(executor.map(process_target_and_interval, tasks), total=len(tasks)))
    log_message("Sampling and file generation completed.")
