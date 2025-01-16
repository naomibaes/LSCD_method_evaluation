# Authors: Naomi Baes and Chat GPT

import pandas as pd
from tqdm import tqdm
import os
from concurrent.futures import ThreadPoolExecutor
import random

# Constants
random_seed = 93
SENTENCE_SAMPLE_SIZE = 50
NUM_REPEATS = 10

# Output directories for high and low variations
output_dir_high = "output/5-year/high"
output_dir_low = "output/5-year/low"
os.makedirs(output_dir_high, exist_ok=True)
os.makedirs(output_dir_low, exist_ok=True)

# Define the log file path
log_file_path = os.path.join("output/5-year", "warnings.log")

# Input files
natural_corpus_dir = "../0.0_corpus_preprocessing/output/natural_lines_targets"
synthetic_corpus_dir = "synthetic/output/5-year"
targets = ["trauma", "anxiety", "depression", "mental_health", "mental_illness", "abuse"]
injection_ratios = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

# Logging helper
def log_message(message, log_buffer=[]):
    log_buffer.append(message)
    if len(log_buffer) >= 10:
        with open(log_file_path, "a", encoding="utf-8") as log_file:
            log_file.write("\n".join(log_buffer) + "\n")
        log_buffer.clear()

# Create intervals
def create_intervals(start_year, end_year, step=5):
    return [(i, i + step - 1) for i in range(start_year, end_year, step)]

intervals = create_intervals(1970, 2020)

# Process target and interval
def process_target_and_interval(args):
    target, interval = args
    start, end = interval
    interval_label = f"{start}-{end}"
    log_message(f"Processing target '{target}' for interval {interval_label}...")

    # Load natural sentences
    natural_file_path = os.path.join(natural_corpus_dir, f"{target}.lines.psych")
    natural_corpus = pd.read_csv(natural_file_path, sep="\t", header=None, names=["sentence", "year"])
    natural_corpus["type"] = "natural"
    interval_natural = natural_corpus.query(f"{start} <= year <= {end}")

    # Load synthetic sentences
    synthetic_file_path = os.path.join(synthetic_corpus_dir, f"{target}_{start}-{end}.synthetic_sentences.csv")
    synthetic_corpus = pd.read_csv(synthetic_file_path)

    # Separate high and low synthetic sentences
    synthetic_high = synthetic_corpus[['high_intensity']].rename(columns={'high_intensity': 'sentence'})
    synthetic_low = synthetic_corpus[['low_intensity']].rename(columns={'low_intensity': 'sentence'})
    synthetic_high['type'] = 'synthetic_high'
    synthetic_low['type'] = 'synthetic_low'

    for repeat in range(1, NUM_REPEATS + 1):
        for ratio in injection_ratios:
            num_synthetic = int(SENTENCE_SAMPLE_SIZE * ratio)
            num_natural = SENTENCE_SAMPLE_SIZE - num_synthetic

            # Adjust sample sizes based on availability, without replacement within the same sample
            if len(interval_natural) < num_natural:
                log_message(f"Insufficient natural sentences for {target} during {interval_label}: expected {num_natural}, available {len(interval_natural)}")
                num_natural = len(interval_natural)
            if len(synthetic_high) < num_synthetic or len(synthetic_low) < num_synthetic:
                adjusted_synthetic = min(len(synthetic_high), len(synthetic_low))
                log_message(f"Insufficient synthetic sentences for {target} during {interval_label}: expected {num_synthetic}, available {adjusted_synthetic}")
                num_synthetic = adjusted_synthetic

            natural_sample = interval_natural.sample(n=num_natural, random_state=random_seed + repeat, replace=False)
            high_sample = synthetic_high.sample(n=num_synthetic, random_state=random_seed + repeat, replace=False)
            low_sample = synthetic_low.sample(n=num_synthetic, random_state=random_seed + repeat, replace=False)

            # Combine, shuffle samples, and add year and type
            combined_high = pd.concat([natural_sample, high_sample.assign(year=start, type='synthetic_high')])
            combined_low = pd.concat([natural_sample, low_sample.assign(year=start, type='synthetic_low')])

            combined_high = combined_high.sample(frac=1, random_state=random_seed + repeat)
            combined_low = combined_low.sample(frac=1, random_state=random_seed + repeat)

            # Save to respective directories
            combined_high.to_csv(os.path.join(output_dir_high, f"{target}_{interval_label}_synthetic_intensity_{int(ratio * 100)}.{repeat}.tsv"), index=False, sep='\t', header=False)
            combined_low.to_csv(os.path.join(output_dir_low, f"{target}_{interval_label}_synthetic_intensity_{int(ratio * 100)}.{repeat}.tsv"), index=False, sep='\t', header=False)

            # Log the creation of undersampled files
            log_message(f"File created with adjusted samples for {target} at {interval_label}: {int(ratio * 100)}% synthetic at repeat {repeat}")

    log_message(f"Completed target '{target}' for interval {interval_label}.")

# Main execution
if __name__ == "__main__":
    tasks = [(target, interval) for target in targets for interval in intervals]
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        list(tqdm(executor.map(process_target_and_interval, tasks), total=len(tasks)))
    log_message("Sampling and file generation completed.", [])
