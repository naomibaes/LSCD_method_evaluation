# Authors: Naomi Baes and Chat GPT

import pandas as pd
import os
import numpy as np
from tqdm.notebook import tqdm  # Import tqdm for Jupyter Notebook

def load_warriner_ratings(filename):
    """Load Warriner et al.'s ratings from a CSV file."""
    return pd.read_csv(filename)

def extract_file_details(filename):
    """Extract target, epoch, and injection ratio from filename."""
    parts = filename.split('_')
    try:
        target = parts[0]
        # Loop to handle compound targets with underscores
        for i in range(1, len(parts)):
            if parts[i].isdigit() or '-' in parts[i]:  # Assuming the epoch always contains digits or a dash
                break
            target += '_' + parts[i]

        epoch = parts[i]
        injection_part = parts[i + 3]  # Adjusting based on the expected position after the epoch
        
        # Extracting the first part of '80.10' for injection ratio
        injection_ratio = injection_part.split('.')[0]  # This gets '80' from '80.10'
    except IndexError as e:
        print(f"Error processing filename {filename}: {str(e)}")
        return None, None, None

    return target, epoch, injection_ratio

def compute_arousal_indices(collocates_df, warriner_ratings):
    """Merge collocates with Warriner ratings on words and calculate arousal indices."""
    merged_data = pd.merge(collocates_df, warriner_ratings[['word', 'A.Mean.Sum']], left_on='collocate', right_on='word', how='inner')
    if not merged_data.empty:
        merged_data['weighted_arousal'] = merged_data['count'] * merged_data['A.Mean.Sum']
        total_count = merged_data['count'].sum()
        total_weighted_arousal = merged_data['weighted_arousal'].sum()
        arousal_index = total_weighted_arousal / total_count if total_count > 0 else 0
        
        # Compute standard error
        merged_data['squared_diff'] = ((merged_data['A.Mean.Sum'] - arousal_index) ** 2) * merged_data['count']
        variance = merged_data['squared_diff'].sum() / total_count if total_count > 0 else 0
        se = np.sqrt(variance / total_count) if total_count > 0 else 0
        return arousal_index, se
    return 0, 0

def process_collocates(input_dir, output_dir, warriner_ratings):
    results = []
    all_files = [os.path.join(root, file) for root, dirs, files in os.walk(input_dir) for file in files if 'lemmatized_collocates.csv' in file]
    for file in tqdm(all_files, desc='Processing files'):  # Use tqdm to show progress
        try:
            # Force 'collocate' column to be read as string
            collocates_df = pd.read_csv(file, dtype={'collocate': str}, on_bad_lines='skip')
            target, epoch, injection_ratio = extract_file_details(os.path.basename(file))
            if target and epoch and injection_ratio:
                arousal_index, se = compute_arousal_indices(collocates_df, warriner_ratings)
                
                # Determine if the file is from the high or low folder
                subdirectory = os.path.basename(os.path.dirname(file))
                if 'high' in subdirectory.lower():
                    index_type = 'high'
                elif 'low' in subdirectory.lower():
                    index_type = 'low'
                else:
                    index_type = 'unknown'

                results.append({
                    'target': target,
                    'epoch': epoch,
                    'injection_ratio': injection_ratio,
                    f'avg_arousal_index_{index_type}': arousal_index,
                    f'se_arousal_index_{index_type}': se
                })
        except Exception as e:
            print(f"Failed to process {file} due to {e}")

    # Convert results to DataFrame and save to combined CSV
    if results:
        results_df = pd.DataFrame(results)

        # Pivot to separate high and low indices into separate columns
        pivot_df = results_df.pivot_table(
            index=['target', 'epoch', 'injection_ratio'],
            values=[
                'avg_arousal_index_high', 'se_arousal_index_high',
                'avg_arousal_index_low', 'se_arousal_index_low'
            ],
            aggfunc='mean'
        ).reset_index()

        # Save to CSV
        pivot_df.to_csv(os.path.join(output_dir, 'baseline_averaged_arousal_index_5-year.csv'), index=False)
    else:
        print("No valid data to process.")

# Main execution
warriner_ratings = load_warriner_ratings('input/warriner_rat.csv')
input_base_dir = 'output/5-year/collocates'
output_base_dir = 'output'

# Process collocate files from both low and high subdirectories and compute indices
process_collocates(input_base_dir, output_base_dir, warriner_ratings)
