import os
import random
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import cpu_count

# Constants
SENTENCE_SAMPLE_SIZE = 50
NUM_REPEATS = 10
output_dir = "output/5-year.cosine"

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Define the log file path
log_file_path = os.path.join(output_dir, "warnings.log")

# Input files
natural_corpus_dir = "../0.0_corpus_preprocessing/output/natural_lines_targets"
synthetic_corpus_dir = "synthetic/output/unique_5-year"
targets = ["abuse", "trauma", "anxiety", "depression", "mental_health", "mental_illness"]
injection_ratios = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

# Logging helper
def log_message(message, log_buffer=[]):
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
    natural_file_path = os.path.join(natural_corpus_dir, f"{target}.lines.psych")
    natural_corpus = pd.read_csv(natural_file_path, sep="\t", header=None, names=["sentence", "year"])
    natural_corpus["type"] = "natural"
    interval_natural = natural_corpus.query(f"{start} <= year <= {end}")

    # Load synthetic sentences
    synthetic_file_path = os.path.join(synthetic_corpus_dir, f"{target}_{start}-{end}.synthetic_1500_sentences.tsv")
    synthetic_corpus = pd.read_csv(synthetic_file_path, sep="\t", header=None, names=["sentence", "type", "year"])

    # Update synthetic rows: move sibling_label values to the type column
    synthetic_corpus["type"] = synthetic_corpus["type"]

    for repeat in range(1, NUM_REPEATS + 1):
        for ratio in injection_ratios:
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

            # Save to output directory without headers
            output_file_path = os.path.join(output_dir, f"{target}_{interval_label}_synthetic_{int(ratio * 100)}_{repeat}")
            combined_sample.to_csv(output_file_path, index=False, header=False, sep='\t')

    log_message(f"Completed target '{target}' for interval {interval_label}.")

# Main execution
if __name__ == "__main__":
    intervals = create_intervals(1970, 2020)
    tasks = [(target, interval) for target in targets for interval in intervals]
    with ThreadPoolExecutor(max_workers=cpu_count()) as executor:
        list(tqdm(executor.map(process_target_and_interval, tasks), total=len(tasks)))
    log_message("Sampling and file generation completed.")
