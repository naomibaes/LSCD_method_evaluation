import pandas as pd
from itertools import combinations

# Load the data from the CSV file
try:
    df = pd.read_csv("output/baseline_final_combined.all-year.cds_mpnet.csv")
    print("Successfully read data from CSV file.")
except FileNotFoundError:
    print("Error: CSV file not found. Please check the file path.")

# List of target terms
target_terms = ["trauma", "anxiety", "depression", "mental_health", "mental_illness", "abuse"]

# Dictionary to store average pairwise distances for each term
pairwise_distances = {}

# Loop through each term to calculate pairwise distance between bins
for term in target_terms:
    # Filter data for the current term
    term_data = df[df['term'] == term]
    
    # Get unique injection ratios and mean cosine dissimilarities
    inj_ratios = sorted(term_data['inj_ratio'].unique())
    avg_cosine_dissim = {
        inj_ratio: term_data[term_data['inj_ratio'] == inj_ratio]['cosine_dissim_mean'].mean()
        for inj_ratio in inj_ratios if not term_data[term_data['inj_ratio'] == inj_ratio]['cosine_dissim_mean'].isnull().all()
    }
    
    # Calculate pairwise differences between all bins
    distances = []
    for (inj_ratio1, value1), (inj_ratio2, value2) in combinations(avg_cosine_dissim.items(), 2):
        distance = abs(value1 - value2)
        distances.append(distance)
    
    # Calculate the average pairwise distance for the term
    avg_distance = sum(distances) / len(distances) if distances else 0
    pairwise_distances[term] = avg_distance

# Display the average pairwise distances for each target term, sorted from highest to lowest
pairwise_distances_df = pd.DataFrame(pairwise_distances.items(), columns=['Term', 'Average Pairwise Distance'])
sorted_pairwise_distances_df = pairwise_distances_df.sort_values(by='Average Pairwise Distance', ascending=False)
print(sorted_pairwise_distances_df)
