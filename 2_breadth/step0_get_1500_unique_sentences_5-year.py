import os
import re
import random
import pandas as pd
from tqdm import tqdm
import spacy

# Load the SpaCy English model for lightweight tokenization
nlp = spacy.blank("en")  # Use a blank pipeline for lightweight tokenization

# Constants
random_seed = 93
SENTENCE_SAMPLE_SIZE = 1500  # Target number of synthetic sentences
BATCH_SIZE = 50  # Number of sentences to fetch per sibling at a time
output_dir = "synthetic/output/unique_5-year"
summary_file_path = "synthetic/output/unique_5-year/summary.csv"  # Summary file path
special_siblings = {"elation", "mental_health", "mental_illness", "mental_disorder", "state_of_mind", "cognitive_state"}  # Terms needing SpaCy tokenization

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Input files
synthetic_corpus_file = "C:/Users/naomi/OneDrive/COMP80004_PhDResearch/RESEARCH/DATA/CORPORA/Psychology/abstract_year_journal.csv.mental.sentence-CLEAN.tsv"
siblings_file = "synthetic/output/donor_terms/siblings_eligible.csv"
targets = ["trauma", "anxiety", "depression", "mental_health", "mental_illness", "abuse"]

# Logging helper
def log_message(message):
    print(f"[INFO] {message}")

# Load datasets
def load_datasets():
    synthetic_corpus = pd.read_csv(synthetic_corpus_file, sep="\t", header=None, names=["sentence", "year", "label"])
    synthetic_corpus = synthetic_corpus.dropna(subset=["sentence"])
    synthetic_corpus["year"] = pd.to_numeric(synthetic_corpus["year"], errors="coerce").dropna().astype(int)

    siblings_df = pd.read_csv(siblings_file)
    siblings_df["sibling_normalized"] = (
        siblings_df["sibling"]
        .str.split(".")
        .str[0]
        .str.replace("_", " ", regex=False)
        .str.lower()
    )
    return synthetic_corpus, siblings_df

# Create intervals
def create_intervals(start_year, end_year, step=5):
    return [(i, i + step - 1) for i in range(start_year, end_year, step)]

intervals = create_intervals(1970, 2020)

# Determine if a sibling requires SpaCy tokenization
def requires_spacy(sibling):
    return len(sibling.split()) > 1 or sibling in special_siblings

# Tokenization helper
def tokenize_with_spacy(sentence):
    return [token.text.lower() for token in nlp(sentence)]

# Function to prepare a regular expression for matching the sibling, considering word boundaries
def prepare_sibling_regex(sibling):
    if "_" in sibling:  # Handle multi-word phrases like "mental_health"
        sibling = sibling.replace("_", " ")  # Replace underscores with spaces
    return rf'\b{sibling}\b'

# Prefilter sentences in batches with conditional tokenization
def prefilter_sentences(interval_synthetic, sibling_list, target, num_valid_sentences):
    valid_synthetic_sentences = []
    used_sentences = set()  # Reset for each target and interval
    siblings_skipped = set()  # Track siblings that have been skipped

    # Progress bar for sentence collection
    with tqdm(total=num_valid_sentences, desc=f"Processing {target}", position=0, ncols=100) as pbar:
        while len(valid_synthetic_sentences) < num_valid_sentences:
            # Create a flag to track if we processed any valid sentences in this round
            sentences_collected_this_round = False

            for sibling in sibling_list:
                # Skip siblings that have already been fully processed
                if sibling in siblings_skipped:
                    continue

                sibling_matches = interval_synthetic[
                    interval_synthetic["sentence"].apply(
                        lambda sentence: re.search(rf'\b{sibling}\b', sentence, flags=re.IGNORECASE) is not None
                    )
                ]

                if sibling_matches.empty:
                    # If no valid sentences are found for this sibling, mark it as skipped
                    siblings_skipped.add(sibling)
                    continue  # Skip this sibling and move on to the next one

                sampled_rows = sibling_matches.sample(n=min(BATCH_SIZE, len(sibling_matches)))
                for _, row in sampled_rows.iterrows():
                    sentence = row["sentence"]
                    if sentence not in used_sentences and len(valid_synthetic_sentences) < num_valid_sentences:
                        modified_sentence = re.sub(rf'\b{sibling}\b', target, sentence, flags=re.IGNORECASE)
                        synthetic_label = f"synthetic_{sibling.replace(' ', '_')}"
                        valid_synthetic_sentences.append((modified_sentence, synthetic_label))
                        used_sentences.add(sentence)  # Track sentence usage within the target and interval context
                        pbar.update(1)  # Update the progress bar
                        sentences_collected_this_round = True

                if len(valid_synthetic_sentences) >= num_valid_sentences:
                    break  # Stop if enough sentences are collected

            # If no valid sentences were collected in this round, skip revisiting siblings
            if not sentences_collected_this_round:
                log_message(f"No valid sentences found in this round for target '{target}'. Skipping further siblings.")
                break  # Stop the loop if no valid sentences are collected

    return valid_synthetic_sentences




# Process target and interval
def process_target_and_interval(task):
    target, interval, synthetic_corpus, siblings_df = task
    start, end = interval
    interval_label = f"{start}-{end}"
    log_message(f"Processing target '{target}' for interval {interval_label}.")

    sibling_list = siblings_df[siblings_df["target"] == target]["sibling_normalized"].tolist()
    if not sibling_list:
        log_message(f"No siblings found for target '{target}'. Skipping interval {interval_label}.")
        return []

    interval_synthetic = synthetic_corpus.query(f"{start} <= year <= {end}")
    valid_synthetic_sentences = prefilter_sentences(
        interval_synthetic, sibling_list, target, SENTENCE_SAMPLE_SIZE
    )

    if not valid_synthetic_sentences:
        log_message(f"No valid synthetic sentences collected for target '{target}' in interval {interval_label}.")
    
    output_file_path = os.path.join(output_dir, f"{target}_{interval_label}.synthetic_1500_sentences.tsv")
    
    # Ensure file is written correctly
    if valid_synthetic_sentences:
        with open(output_file_path, "w", encoding="utf-8") as output_file:
            for sentence, label in valid_synthetic_sentences:
                output_file.write(f"{sentence}\t{label}\t{random.randint(start, end)}\n")

        log_message(f"Output file saved at: {output_file_path}")
    else:
        log_message(f"No sentences to write to {output_file_path}")
    
    return [interval_label, target, len(valid_synthetic_sentences), SENTENCE_SAMPLE_SIZE - len(valid_synthetic_sentences)]



# Save summary to CSV after processing
def save_summary(summary_data):
    summary_df = pd.DataFrame(summary_data, columns=["epoch", "target", "count", "difference"])
    summary_df.to_csv(summary_file_path, index=False)
    log_message(f"Summary file saved at: {summary_file_path}")

# Main execution
if __name__ == "__main__":
    random.seed(random_seed)
    synthetic_corpus, siblings_df = load_datasets()
    tasks = [
        (target, interval, synthetic_corpus, siblings_df)
        for target in targets for interval in intervals
    ]

    results = []
    for task in tqdm(tasks, desc="Processing"):
        results.append(process_target_and_interval(task))

    summary_data = [result for result in results if result]
    save_summary(summary_data)
    log_message("Sampling and file generation completed.")
