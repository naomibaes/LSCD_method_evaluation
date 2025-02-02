import pandas as pd
import os
import numpy as np
import re
from tqdm.notebook import tqdm

def load_warriner_ratings(filename):
    """Load Warriner et al.'s ratings from a CSV file."""
    return pd.read_csv(filename)

def extract_file_details(filename):
    """Extract target, injection ratio, and round number from filename."""
    parts = filename.split('_')

    # Extract target (everything before 'synthetic' or 'natural')
    if 'synthetic' in parts:
        target_end_idx = parts.index('synthetic')
    elif 'natural' in parts:
        target_end_idx = parts.index('natural')
    else:
        print(f"⚠️ Warning: Unexpected filename format - {filename}")
        return None, None, None

    target = '_'.join(parts[:target_end_idx])  # Join the target term
    
    # Extract injection ratio (numbers after 'intensity_')
    match_inj = re.search(r"intensity_(\d+)", filename)
    injection_ratio = int(match_inj.group(1)) if match_inj else None  # Convert to integer

    # Extract round number (integer part of 'roundX.Y')
    match_round = re.search(r"round(\d+)\.\d+", filename)  # Extracts only X (integer part)
    round_number = int(match_round.group(1)) if match_round else None  # Convert to integer

    if injection_ratio is None:
        print(f"⚠️ Warning: Injection ratio not found in filename: {filename}")
    
    if round_number is None:
        print(f"⚠️ Warning: Round number not found in filename: {filename}")

    return target, injection_ratio, round_number

def normalize_values(value, min_val, max_val):
    """Normalize values to a scale from 0 to 1."""
    return (value - min_val) / (max_val)

def compute_arousal_indices(collocates_df, warriner_ratings):
    """Merge collocates with Warriner ratings on words and calculate normalized arousal indices."""
    merged_data = pd.merge(collocates_df, warriner_ratings[['word', 'V.Mean.Sum']], left_on='collocate', right_on='word', how='inner')
    if not merged_data.empty:
        merged_data['weighted_arousal'] = merged_data['count'] * merged_data['V.Mean.Sum']
        total_count = merged_data['count'].sum()
        total_weighted_arousal = merged_data['weighted_arousal'].sum()
        arousal_index = total_weighted_arousal / total_count if total_count > 0 else 0
        normalized_index = normalize_values(arousal_index, 1, 9)  # Assuming V.Mean.Sum is scaled from 1 to 9
        
        # Compute standard error
        merged_data['squared_diff'] = ((merged_data['V.Mean.Sum'] - arousal_index) ** 2) * merged_data['count']
        variance = merged_data['squared_diff'].sum() / total_count if total_count > 0 else 0
        se = np.sqrt(variance / total_count) if total_count > 0 else 0
        normalized_se = se / (9 - 1)  # Normalizing SE in the same range as arousal_index
        return normalized_index, normalized_se
    return 0, 0

def process_collocates(input_dir, output_dir, warriner_ratings):
    results = []
    all_files = [os.path.join(root, file) for root, dirs, files in os.walk(input_dir) for file in files if 'lemmatized_collocates.csv' in file]
    
    for file in tqdm(all_files, desc='Processing files'):
        collocates_df = pd.read_csv(file, dtype={'collocate': str}, on_bad_lines='skip')
        target, injection_ratio, round_number = extract_file_details(os.path.basename(file))

        if target and injection_ratio is not None and round_number is not None:
            arousal_index, se = compute_arousal_indices(collocates_df, warriner_ratings)
            subdirectory = os.path.basename(os.path.dirname(file)).lower()
            index_type = 'high' if 'high' in subdirectory else 'low'
            results.append({
                'target': target,
                'injection_ratio': injection_ratio,
                'round_number': round_number,  # Now correctly extracted as an integer (1-6)
                f'avg_arousal_index_{index_type}': arousal_index,
                f'se_arousal_index_{index_type}': se
            })

    # Aggregate results to ensure values are grouped by target, injection_ratio, and round_number
    if results:
        results_df = pd.DataFrame(results)

        # Pivot to reshape the data and group by target, injection_ratio, and round number
        pivot_df = results_df.pivot_table(
            index=['target', 'injection_ratio', 'round_number'],
            values=[
                'avg_arousal_index_high', 'se_arousal_index_high',
                'avg_arousal_index_low', 'se_arousal_index_low'
            ],
            aggfunc='mean'  # Aggregates across multiple files within the same injection ratio and round
        ).reset_index()

        # Save the processed results
        pivot_df.to_csv(os.path.join(output_dir, 'control_averaged_arousal_index_all-year_normalized.csv'), index=False)
    else:
        print("No valid data to process.")

# Main execution
warriner_ratings = load_warriner_ratings('input/warriner_rat.csv')
input_base_dir = 'output/all-year/control/collocates'  # Adjusted path without epoch
output_base_dir = 'output'
process_collocates(input_base_dir, output_base_dir, warriner_ratings)
