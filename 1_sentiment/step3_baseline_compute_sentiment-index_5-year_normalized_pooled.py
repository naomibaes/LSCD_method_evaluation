import pandas as pd
import os
import numpy as np
from tqdm.notebook import tqdm

def load_warriner_ratings(filename):
    """Load Warriner et al.'s ratings from a CSV file."""
    return pd.read_csv(filename)

def extract_file_details(filename, root):
    """Extract target, epoch, injection ratio, and folder grouping (sentiment) from filename."""
    sentiment = 'positive' if 'positive' in root else 'negative'
    parts = filename.split('_')
    try:
        target = parts[0]
        for i in range(1, len(parts)):
            if parts[i].isdigit() or '-' in parts[i]:  
                break
            target += '_' + parts[i]
        epoch = parts[i]
        injection_part = parts[i + 3]
        injection_ratio = injection_part.split('.')[0]
    except IndexError as e:
        print(f"Error processing filename {filename}: {str(e)}")
        return None, None, None, None
    return target, epoch, injection_ratio, sentiment

def normalize_values(value, min_val, max_val):
    """Normalize values to a scale from 0 to 1."""
    return (value - min_val) / (max_val - min_val)

def compute_grouped_indices(full_data):
    """Aggregate data by injection_ratio and sentiment, and compute mean valence index and SE."""
    full_data['weighted_valence'] = full_data['count'] * full_data['V.Mean.Sum']
    full_data['squared_diff'] = (full_data['V.Mean.Sum'] - full_data['weighted_valence'].mean()) ** 2 * full_data['count']

    grouped = full_data.groupby(['target', 'epoch', 'injection_ratio', 'sentiment'])
    result = grouped.apply(lambda x: pd.Series({
        'avg_valence_index': normalize_values(x['weighted_valence'].sum() / x['count'].sum(), 1, 9),
        'SE': np.sqrt((x['squared_diff'].sum()) / x['count'].sum()) / x['count'].sum()
    })).unstack().fillna(0)

    result.columns = [f"{col[0]}_{col[1]}" for col in result.columns]  # Flatten MultiIndex columns
    return result.reset_index()

def process_collocates(input_dir, output_dir, warriner_ratings):
    all_data = []
    all_files = [(root, file) for root, dirs, files in os.walk(input_dir) for file in files if 'lemmatized_collocates.csv' in file]
    for root, file in tqdm(all_files, desc='Processing files'):
        collocates_df = pd.read_csv(os.path.join(root, file), dtype={'collocate': str}, on_bad_lines='skip')
        target, epoch, injection_ratio, sentiment = extract_file_details(os.path.basename(file), root)
        collocates_df['target'], collocates_df['epoch'], collocates_df['injection_ratio'], collocates_df['sentiment'] = target, epoch, injection_ratio, sentiment
        all_data.append(collocates_df)

    if all_data:
        full_data = pd.concat(all_data)
        full_data = pd.merge(full_data, warriner_ratings[['word', 'V.Mean.Sum']], left_on='collocate', right_on='word', how='inner')
        results = compute_grouped_indices(full_data)
        results.to_csv(os.path.join(output_dir, 'baseline_averaged_valence_index_5-year_normalized_pooled.csv'), index=False)
    else:
        print("No valid data to process.")

# Main execution
warriner_ratings = load_warriner_ratings('input/warriner_rat.csv')
input_base_dir = 'output/5-year/collocates'
output_base_dir = 'output'
process_collocates(input_base_dir, output_base_dir, warriner_ratings)
