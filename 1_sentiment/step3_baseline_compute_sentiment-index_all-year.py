# Authors: Naomi Baes and Chat GPT

import pandas as pd
import os
import numpy as np
from tqdm.notebook import tqdm

def load_warriner_ratings(filename):
    """Load Warriner et al.'s ratings from a CSV file, normalize the word field."""
    data = pd.read_csv(filename)
    data['word'] = data['word'].str.lower().str.strip()  # Normalize words for matching
    return data

def extract_file_details(filename, targets):
    """Extract target and injection ratio from filename by iterating through filename components."""
    target = None
    injection_ratio = None
    parts = filename.replace('.csv', '').split('_')
    potential_target = ""

    for i, part in enumerate(parts):
        if potential_target:
            new_potential_target = potential_target + "_" + part
        else:
            new_potential_target = part

        if any(new_potential_target.lower() == t.lower() for t in targets):
            target = new_potential_target
            potential_target = new_potential_target
        elif potential_target and part.split('.')[0].isdigit():
            injection_ratio = part.split('.')[0]
            break
        else:
            potential_target = new_potential_target  # Continue building the potential target

    print(f"File: {filename}, Target: {target}, Injection Ratio: {injection_ratio}")  # Debug output
    return target, injection_ratio

def compute_valence_indices(collocates_df, warriner_ratings):
    """Compute valence indices after normalizing and merging collocate data with Warriner ratings."""
    if collocates_df.empty:
        return 0, 0  # Ensure no computation on empty data frames
    merged_data = pd.merge(collocates_df, warriner_ratings[['word', 'V.Mean.Sum']], left_on='collocate', right_on='word', how='inner')
    if not merged_data.empty:
        merged_data['weighted_valence'] = merged_data['count'] * merged_data['V.Mean.Sum']
        total_count = merged_data['count'].sum()
        total_weighted_valence = merged_data['weighted_valence'].sum()
        valence_index = total_weighted_valence / total_count if total_count > 0 else 0
        variance = np.var(merged_data['V.Mean.Sum'], ddof=1) * total_count
        se = np.sqrt(variance / total_count) if total_count > 0 else 0
        return valence_index, se
    return 0, 0

def process_collocates(input_dir, output_dir, warriner_ratings, targets):
    results = []
    valence_indices = {}  # Dictionary to store valence indices for SEM calculation

    all_files = [os.path.join(root, file) for root, dirs, files in os.walk(input_dir) for file in files if 'lemmatized_collocates.csv' in file]
    for file in tqdm(all_files, desc='Processing files'):
        collocates_df = pd.read_csv(file, dtype={'collocate': str}, on_bad_lines='skip')
        subfolder = os.path.basename(os.path.dirname(file)).lower()
        target, injection_ratio = extract_file_details(os.path.basename(file), targets)
        if target and injection_ratio:
            valence_index, _ = compute_valence_indices(collocates_df, warriner_ratings)  # Ignore SEM here
            results.append({
                'target': target,
                'injection_ratio': injection_ratio,
                f'avg_valence_index_{subfolder}': valence_index
            })
            key = (target, injection_ratio, subfolder)
            if key not in valence_indices:
                valence_indices[key] = []
            valence_indices[key].append(valence_index)

    # Calculate SEM after all files have been processed
    for key, indices in valence_indices.items():
        target, injection_ratio, subfolder = key
        sem = np.std(indices, ddof=1) / np.sqrt(len(indices)) if len(indices) > 1 else 0
        for result in results:
            if result['target'] == target and result['injection_ratio'] == injection_ratio:
                result[f'se_valence_index_{subfolder}'] = sem

    if results:
        results_df = pd.DataFrame(results)
        final_df = results_df.groupby(['target', 'injection_ratio']).agg({
            'avg_valence_index_positive': 'mean',
            'avg_valence_index_negative': 'mean',
            'se_valence_index_positive': 'mean',
            'se_valence_index_negative': 'mean'
        }).reset_index()
        final_df.to_csv(os.path.join(output_dir, 'baseline_averaged_valence_index_all-year.csv'), index=False)
        print(f"Output written to: {os.path.join(output_dir, 'baseline_averaged_valence_index_all-year.csv')}")
    else:
        print("No data processed, check input files and parameters.")

# Main execution
targets = ["trauma", "anxiety", "depression", "mental_health", "mental_illness", "abuse"]
warriner_ratings = load_warriner_ratings('input/warriner_rat.csv')
input_base_dir = 'output/all-year/collocates'
output_base_dir = 'output'

# Process collocate files from both negative and positive subdirectories and compute indices
process_collocates(input_base_dir, output_base_dir, warriner_ratings, targets)
