import pandas as pd
import os
import numpy as np
import re
from tqdm.notebook import tqdm

def load_warriner_ratings(filename):
    """Load Warriner et al.'s ratings from a CSV file."""
    return pd.read_csv(filename)

def extract_file_details(filename):
    """Extract target, epoch, and injection ratio from filename."""
    parts = filename.split('_')

    # Find the epoch (5-year range)
    epoch = None
    for part in parts:
        if re.match(r'^\d{4}-\d{4}$', part):  # Detects patterns like '1970-1974'
            epoch = part
            break

    if not epoch:
        print(f"⚠️ Warning: Epoch not found in filename: {filename}")
        return None, None, None

    # Extract target (everything before the epoch)
    epoch_index = parts.index(epoch)
    target = '_'.join(parts[:epoch_index])

    # Find the injection ratio (extract only the integer round number)
    injection_ratio = None
    for part in parts:
        if part.startswith('round'):
            match = re.match(r'round(\d+)', part)  # Extract only the integer round number
            if match:
                injection_ratio = match.group(1)  # Extracts '1', '2', etc.
            break

    if not injection_ratio:
        print(f"⚠️ Warning: Injection ratio not found in filename: {filename}")
        return target, epoch, None

    return target, epoch, int(injection_ratio)  # Convert to integer

def normalize_values(value, min_val, max_val):
    """Normalize values to a scale from 0 to 1."""
    return (value - min_val) / (max_val - min_val)

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
        target, epoch, injection_ratio = extract_file_details(os.path.basename(file))

        if target and epoch and injection_ratio:
            arousal_index, se = compute_arousal_indices(collocates_df, warriner_ratings)
            subdirectory = os.path.basename(os.path.dirname(file)).lower()
            index_type = 'high' if 'high' in subdirectory else 'low'
            results.append({
                'target': target,
                'epoch': epoch,
                'injection_ratio': injection_ratio,  # Now only {1, 2, 3, 4, 5, 6}
                f'avg_arousal_index_{index_type}': arousal_index,
                f'se_arousal_index_{index_type}': se
            })

    # Aggregate results to ensure only `injection_ratio` 1-6 exist
    if results:
        results_df = pd.DataFrame(results)

        # Pivot to reshape the data and group by target, epoch, and injection ratio (round number only)
        pivot_df = results_df.pivot_table(
            index=['target', 'epoch', 'injection_ratio'],
            values=[
                'avg_arousal_index_high', 'se_arousal_index_high',
                'avg_arousal_index_low', 'se_arousal_index_low'
            ],
            aggfunc='mean'  # Aggregates across multiple `roundX.Y` within the same injection ratio
        ).reset_index()

        # Save the processed results
        pivot_df.to_csv(os.path.join(output_dir, 'control_averaged_arousal_index_5-year_normalized.csv'), index=False)
    else:
        print("No valid data to process.")

# Main execution
warriner_ratings = load_warriner_ratings('input/warriner_rat.csv')
input_base_dir = 'output/5-year/control/collocates'
output_base_dir = 'output'
process_collocates(input_base_dir, output_base_dir, warriner_ratings)
