import pandas as pd
import os
import numpy as np
from tqdm.notebook import tqdm

def load_warriner_ratings(filename):
    """Load Warriner et al.'s ratings from a CSV file."""
    return pd.read_csv(filename)

def extract_file_details(filename, root):
    """Extract target, injection ratio, and sentiment from filename."""
    parts = filename.split('_')
    target = parts[0]
    for i in range(1, len(parts)):
        if parts[i][0].isdigit():
            break
        target += '_' + parts[i]

    target = target.replace('_synthetic_sentiment', '')
    injection_ratio = parts[i].split('.')[0]
    condition = 'positive' if 'positive' in root else 'negative'
    return target, injection_ratio, condition

def normalize_values(value, min_val, max_val):
    """Normalize values to a scale from 0 to 1."""
    return (value - min_val) / (max_val - min_val)

def process_collocates(input_dir, output_dir, warriner_ratings):
    all_data = []
    # Load all files and extract necessary details
    for root, dirs, files in os.walk(input_dir):
        for file in tqdm(files, desc='Loading files'):
            if 'lemmatized_collocates.csv' in file:
                df = pd.read_csv(os.path.join(root, file), dtype={'collocate': str}, on_bad_lines='skip')
                target, injection_ratio, condition = extract_file_details(file, root)
                df['target'] = target
                df['injection_ratio'] = injection_ratio
                df['condition'] = condition
                all_data.append(df)
    
    if all_data:
        # Combine all data into a single DataFrame
        full_data = pd.concat(all_data)
        # Merge with Warriner's ratings
        merged_data = pd.merge(full_data, warriner_ratings[['word', 'V.Mean.Sum']], left_on='collocate', right_on='word', how='inner')
        merged_data['weighted_valence'] = merged_data['count'] * merged_data['V.Mean.Sum']

        # Group by target, injection_ratio, and condition
        grouped = merged_data.groupby(['target', 'injection_ratio', 'condition'])
        result = grouped.apply(lambda x: pd.Series({
            'avg_valence_index': normalize_values((x['weighted_valence'].sum() / x['count'].sum()), 1, 9),
            'SE': np.sqrt((x['weighted_valence'].sum() ** 2 / x['count'].sum()) / x['count'].sum())
        })).reset_index()

        result.to_csv(os.path.join(output_dir, 'baseline_averaged_valence_index_all-year_normalized.csv'), index=False)
        print("Results saved to CSV.")
    else:
        print("No valid data to process.")

# Main execution
warriner_ratings = load_warriner_ratings('input/warriner_rat.csv')
input_base_dir = 'output/all-year/collocates'
output_base_dir = 'output'
process_collocates(input_base_dir, output_base_dir, warriner_ratings)
