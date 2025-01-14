import pandas as pd
from itertools import combinations

# Load the data from the CSV file
try:
    df = pd.read_csv("output/averaged_valence_index_all-year.csv")
    print("Successfully read data from CSV file.")
except FileNotFoundError:
    print("Error: CSV file not found. Please check the file path.")
    raise SystemExit  # Exit the script if the file is not found

# Ensure `injection_ratio` is numeric
df['injection_ratio'] = pd.to_numeric(df['injection_ratio'], errors='coerce').fillna(0).astype(int)

# List of target terms
target_terms = ["trauma", "anxiety", "depression", "mental_health", "mental_illness", "abuse"]

# Dictionaries to store APDs for positive and negative valence indices
positive_apd = {}
negative_apd = {}

# Loop through each term to calculate pairwise distances for positive and negative indices
for term in target_terms:
    # Filter data for the current term
    term_data = df[df['target'] == term]
    
    # Get unique injection ratios
    inj_ratios = sorted(term_data['injection_ratio'].unique())
    
    # Compute average valence indices for positive
    avg_valence_positive = {
        inj_ratio: term_data[term_data['injection_ratio'] == inj_ratio]['avg_valence_index_positive'].mean()
        for inj_ratio in inj_ratios
        if not term_data[term_data['injection_ratio'] == inj_ratio]['avg_valence_index_positive'].isnull().all()
    }
    
    # Compute average valence indices for negative
    avg_valence_negative = {
        inj_ratio: term_data[term_data['injection_ratio'] == inj_ratio]['avg_valence_index_negative'].mean()
        for inj_ratio in inj_ratios
        if not term_data[term_data['injection_ratio'] == inj_ratio]['avg_valence_index_negative'].isnull().all()
    }
    
    # Calculate pairwise differences for positive
    distances_positive = [
        abs(value1 - value2)
        for (inj_ratio1, value1), (inj_ratio2, value2) in combinations(avg_valence_positive.items(), 2)
    ]
    avg_distance_positive = sum(distances_positive) / len(distances_positive) if distances_positive else 0
    positive_apd[term] = avg_distance_positive

    # Calculate pairwise differences for negative
    distances_negative = [
        abs(value1 - value2)
        for (inj_ratio1, value1), (inj_ratio2, value2) in combinations(avg_valence_negative.items(), 2)
    ]
    avg_distance_negative = sum(distances_negative) / len(distances_negative) if distances_negative else 0
    negative_apd[term] = avg_distance_negative

# Combine results into a DataFrame
apd_df = pd.DataFrame({
    "Term": target_terms,
    "APD_Positive": [positive_apd[term] for term in target_terms],
    "APD_Negative": [negative_apd[term] for term in target_terms]
}).sort_values(by="APD_Positive", ascending=False)

# Print the resulting DataFrame
print(apd_df)

# Optional: Save the results to a CSV file
apd_df.to_csv("output/average_pairwise_distances_sentiment_all-year.csv", index=False)
