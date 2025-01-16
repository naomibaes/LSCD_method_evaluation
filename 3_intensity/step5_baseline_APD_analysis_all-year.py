# Authors: Naomi Baes and Chat GPT

import pandas as pd
from itertools import combinations

# Load the data from the CSV file
try:
    df = pd.read_csv("output/baseline_averaged_arousal_index_all-year.csv")
    print("Successfully read data from CSV file.")
except FileNotFoundError:
    print("Error: CSV file not found. Please check the file path.")
    raise SystemExit  # Exit the script if the file is not found

# Ensure `injection_ratio` is numeric
df['injection_ratio'] = pd.to_numeric(df['injection_ratio'], errors='coerce').fillna(0).astype(int)

# List of target terms
target_terms = ["trauma", "anxiety", "depression", "mental_health", "mental_illness", "abuse"]

# Dictionaries to store APDs for high and low arousal indices
high_apd = {}
low_apd = {}

# Loop through each term to calculate pairwise distances for high and low indices
for term in target_terms:
    # Filter data for the current term
    term_data = df[df['target'] == term]
    
    # Get unique injection ratios
    inj_ratios = sorted(term_data['injection_ratio'].unique())
    
    # Compute average arousal indices for high
    avg_arousal_high = {
        inj_ratio: term_data[term_data['injection_ratio'] == inj_ratio]['avg_arousal_index_high'].mean()
        for inj_ratio in inj_ratios
        if not term_data[term_data['injection_ratio'] == inj_ratio]['avg_arousal_index_high'].isnull().all()
    }
    
    # Compute average arousal indices for low
    avg_arousal_low = {
        inj_ratio: term_data[term_data['injection_ratio'] == inj_ratio]['avg_arousal_index_low'].mean()
        for inj_ratio in inj_ratios
        if not term_data[term_data['injection_ratio'] == inj_ratio]['avg_arousal_index_low'].isnull().all()
    }
    
    # Calculate pairwise differences for high
    distances_high = [
        abs(value1 - value2)
        for (inj_ratio1, value1), (inj_ratio2, value2) in combinations(avg_arousal_high.items(), 2)
    ]
    avg_distance_high = sum(distances_high) / len(distances_high) if distances_high else 0
    high_apd[term] = avg_distance_high

    # Calculate pairwise differences for low
    distances_low = [
        abs(value1 - value2)
        for (inj_ratio1, value1), (inj_ratio2, value2) in combinations(avg_arousal_low.items(), 2)
    ]
    avg_distance_low = sum(distances_low) / len(distances_low) if distances_low else 0
    low_apd[term] = avg_distance_low

# Combine results into a DataFrame
apd_df = pd.DataFrame({
    "Term": target_terms,
    "APD_High": [high_apd[term] for term in target_terms],
    "APD_Low": [low_apd[term] for term in target_terms]
}).sort_values(by="APD_High", ascending=False)

# Print the resulting DataFrame
print(apd_df)

# Optional: Save the results to a CSV file
apd_df.to_csv("output/baseline_average_pairwise_distances_intensity_all-year.csv", index=False)
