import pandas as pd
import os
import numpy as np
from tqdm.notebook import tqdm

def load_warriner_ratings(filename):
    """Load Warriner et al.'s ratings from a CSV file."""
    return pd.read_csv(filename)

def extract_file_details(filename):
    """Extract target and injection ratio from filename, removing the '_synthetic_intensity' suffix."""
    parts = filename.split('_')
    try:
        target = parts[0]
        # Continue adding to the target if the next part is not a digit, assuming digits start injection ratios or irrelevant parts
        for i in range(1, len(parts)):
            if parts[i][0].isdigit():
                break
            target += '_' + parts[i]

        # Remove the '_synthetic_sentiment' suffix
        target = target.replace('_synthetic_intensity', '')

        # The part that is likely to start with digits is considered the start of the injection ratio part
        injection_ratio_part = parts[i]
        # Splitting by '.' to ignore file extensions or decimals in filenames if they exist
        injection_ratio = injection_ratio_part.split('.')[0]
    except IndexError as e:
        print(f"Error processing filename {filename}: {str(e)}")
        return None, None

    return target, injection_ratio

def normalize_values(value, min_val, max_val):
    """Normalize values to a scale from 0 to 1."""
    return (value - min_val) / (max_val - min_val)

def compute_arousal_indices(collocates_df, warriner_ratings):
    """Merge collocates with Warriner ratings on words and calculate normalized arousal indices."""
    merged_data = pd.merge(collocates_df, warriner_ratings[['word', 'A.Mean.Sum']], left_on='collocate', right_on='word', how='inner')
    if not merged_data.empty:
        merged_data['weighted_arousal'] = merged_data['count'] * merged_data['A.Mean.Sum']
        total_count = merged_data['count'].sum()
        total_weighted_arousal = merged_data['weighted_arousal'].sum()
        arousal_index = total_weighted_arousal / total_count if total_count > 0 else 0
        normalized_index = normalize_values(arousal_index, 1, 9)  # Assuming A.Mean.Sum is scaled from 1 to 9
        
        # Compute standard error
        merged_data['squared_diff'] = ((merged_data['A.Mean.Sum'] - arousal_index) ** 2) * merged_data['count']
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
        target, injection_ratio = extract_file_details(os.path.basename(file))
        if target and injection_ratio:
            arousal_index, se = compute_arousal_indices(collocates_df, warriner_ratings)
            subdirectory = os.path.basename(os.path.dirname(file)).lower()
            index_type = 'high' if 'high' in subdirectory else 'low'
            results.append({
                'target': target,
                'injection_ratio': injection_ratio,
                f'avg_arousal_index_{index_type}': arousal_index,
                f'se_arousal_index_{index_type}': se
            })

    # Save results to CSV
    if results:
        results_df = pd.DataFrame(results)
        pivot_df = results_df.pivot_table(
            index=['target', 'injection_ratio'],
            values=[
                'avg_arousal_index_high', 'se_arousal_index_high',
                'avg_arousal_index_low', 'se_arousal_index_low'
            ],
            aggfunc='mean'
        ).reset_index()
        pivot_df.to_csv(os.path.join(output_dir, 'baseline_averaged_arousal_index_all-year_normalized.csv'), index=False)
    else:
        print("No valid data to process.")

# Main execution
warriner_ratings = load_warriner_ratings('input/warriner_rat.csv')
input_base_dir = 'output/all-year/collocates'
output_base_dir = 'output'
process_collocates(input_base_dir, output_base_dir, warriner_ratings)
