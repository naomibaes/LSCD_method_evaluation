import os
import re
import numpy as np
import pandas as pd

# New target terms
target_terms = ["trauma", "anxiety", "depression", "mental_health", "mental_illness", "abuse"]

# Function to process each file and extract necessary data
def process_file(file_path):
    # Extract the relevant information from the file name
    file_name = os.path.basename(file_path)
    # Updated regex to correctly parse the term, injection ratio, and ignore the iteration
    match = re.match(r"(\w+)_synthetic_breadth_(\d+)_(\d+).csv_cds_lexeme", file_name)
    if not match:
        print(f"Skipping file due to unexpected format: {file_name}")
        return None

    term = match.group(1)
    inj_ratio = match.group(2)  # Now correctly assigning the injection ratio
    epoch = "all-year"  # Assuming a constant epoch for all files
    corpus = "synthetic_breadth"

    print(f"File Name: {file_name}, Epoch: {epoch}, Term: {term}, Corpus: {corpus}, Inj Ratio: {inj_ratio}")

    try:
        cosine_scores = np.loadtxt(file_path)
        avg_cosine_dissim = np.mean(cosine_scores)
        std_cosine_dissim = np.std(cosine_scores)
        std_error_cosine_dissim = std_cosine_dissim / np.sqrt(len(cosine_scores)) if len(cosine_scores) > 0 else np.nan
        print(f"Successfully processed file: {file_name}, Avg cosine dissimilarity: {avg_cosine_dissim}")
    except Exception as e:
        print(f"Error reading or processing file {file_path}: {e}")
        return None

    return epoch, avg_cosine_dissim, std_cosine_dissim, std_error_cosine_dissim, term, corpus, inj_ratio

# Main function to process the files
def main(folder_path):
    epoch_scores = {}
    all_files = os.listdir(folder_path)

    for file_name in all_files:
        file_path = os.path.join(folder_path, file_name)
        if os.path.isfile(file_path) and "cds_lexeme" in file_name:
            result = process_file(file_path)
            if result is None:
                continue

            epoch, avg_cosine_dissim, std_cosine_dissim, std_error_cosine_dissim, term, corpus, inj_ratio = result
            if inj_ratio not in epoch_scores:
                epoch_scores[inj_ratio] = {}
            if epoch not in epoch_scores[inj_ratio]:
                epoch_scores[inj_ratio][epoch] = {t: [] for t in target_terms}
            epoch_scores[inj_ratio][epoch][term].append((avg_cosine_dissim, std_cosine_dissim, std_error_cosine_dissim, corpus))

    os.makedirs("output", exist_ok=True)
    combined_output_filename = "output/baseline_final_combined.all-year.cds_lexeme.csv"
    combine_results(epoch_scores, target_terms, combined_output_filename)

def combine_results(epoch_scores, target_terms, output_filename):
    final_combined_scores = []
    for inj_ratio, epochs in epoch_scores.items():
        for epoch, scores in epochs.items():
            for term in target_terms:
                values = scores[term]
                if values:
                    avg_cosine_dissim = np.mean([v[0] for v in values])
                    std_cosine_dissim = np.mean([v[1] for v in values])
                    std_error_cosine_dissim = np.mean([v[2] for v in values])
                    corpus = values[0][3] if inj_ratio != '0' else "natural"
                else:
                    avg_cosine_dissim, std_cosine_dissim, std_error_cosine_dissim, corpus = np.nan, np.nan, np.nan, None
                final_combined_scores.append((epoch, avg_cosine_dissim, std_cosine_dissim, std_error_cosine_dissim, term, inj_ratio, corpus))

    combined_df = pd.DataFrame(final_combined_scores, columns=["epoch", "cosine_dissim_mean", "cosine_dissim_sd", "cosine_dissim_se", "term", "inj_ratio", "corpus"])
    combined_df.to_csv(output_filename, index=False)
    print(f"Saved the final combined DataFrame to: {output_filename}")

folder_path = "output/all-year.cosine"
main(folder_path)
print("Processing completed.")
