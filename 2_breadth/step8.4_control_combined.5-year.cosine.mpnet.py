import os
import re
import numpy as np
import pandas as pd

# Define target terms
target_terms = ["trauma", "anxiety", "depression", "mental_health", "mental_illness", "abuse"]

# Function to process each file and extract necessary data
def process_file(file_path):
    """Processes an individual file to extract cosine dissimilarity metrics."""
    file_name = os.path.basename(file_path)
    
    # Updated regex pattern to match new file naming format
    pattern = re.compile(r"(" + '|'.join(target_terms) + r")_(\d{4}-\d{4})_synthetic_(\d+)_round(\d+)\.(\d+)\.tsv_cds_mpnet")
    match = pattern.search(file_name)
    
    if not match:
        print(f"Skipping file due to unexpected format: {file_name}")
        return None

    term, epoch, inj_ratio, round_number, repeat_number = match.groups()
    round_number = int(round_number)  # Convert round number to integer
    repeat_number = int(repeat_number)  # Convert repeat number to integer

    try:
        cosine_scores = np.loadtxt(file_path)
        avg_cosine_dissim = np.mean(cosine_scores)
        std_cosine_dissim = np.std(cosine_scores)
        std_error_cosine_dissim = std_cosine_dissim / np.sqrt(len(cosine_scores)) if len(cosine_scores) > 0 else np.nan
        
        print(f"Processed file: {file_name} | Epoch: {epoch} | Term: {term} | Round: {round_number} | Repeat: {repeat_number}")
    except Exception as e:
        print(f"Error reading or processing file {file_path}: {e}")
        return None

    return epoch, avg_cosine_dissim, std_cosine_dissim, std_error_cosine_dissim, term, inj_ratio, round_number, repeat_number

# Main function to process all files in the folder
def main(folder_path):
    round_scores = {}
    all_files = os.listdir(folder_path)
    
    for file_name in all_files:
        if file_name.endswith(".tsv_cds_mpnet"):
            file_path = os.path.join(folder_path, file_name)
            result = process_file(file_path)
            if result is None:
                continue

            epoch, avg_cosine_dissim, std_cosine_dissim, std_error_cosine_dissim, term, inj_ratio, round_number, repeat_number = result
            
            if term not in round_scores:
                round_scores[term] = {}
            if epoch not in round_scores[term]:
                round_scores[term][epoch] = {}
            if round_number not in round_scores[term][epoch]:
                round_scores[term][epoch][round_number] = []
            
            # Store the results per repeat within each round
            round_scores[term][epoch][round_number].append((avg_cosine_dissim, std_cosine_dissim, std_error_cosine_dissim))
    
    # Ensure output directory exists
    os.makedirs("output", exist_ok=True)
    output_filename = "output/control_final_combined.5-year.cds_mpnet.csv"
    combine_results(round_scores, output_filename)

def combine_results(round_scores, output_filename):
    """Aggregates results across repeats within each round, then saves the final results."""
    final_combined_scores = []
    
    for term, epochs in round_scores.items():
        for epoch, rounds in epochs.items():
            for round_number, values in rounds.items():
                if values:
                    avg_cosine_dissim = np.mean([v[0] for v in values])
                    std_cosine_dissim = np.mean([v[1] for v in values])
                    std_error_cosine_dissim = np.mean([v[2] for v in values])
                    
                    final_combined_scores.append((epoch, term, round_number, avg_cosine_dissim, std_cosine_dissim, std_error_cosine_dissim))
                else:
                    final_combined_scores.append((epoch, term, round_number, np.nan, np.nan, np.nan))
    
    combined_df = pd.DataFrame(final_combined_scores, columns=["epoch", "term", "round_number", "cosine_dissim_mean", "cosine_dissim_sd", "cosine_dissim_se"])
    combined_df.to_csv(output_filename, index=False)
    print(f"Saved final combined results to: {output_filename}")

# Run the processing
folder_path = "output/5-year.cosine/control"
main(folder_path)
print("Processing completed.")
