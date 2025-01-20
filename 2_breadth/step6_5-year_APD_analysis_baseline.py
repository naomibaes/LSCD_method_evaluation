import pandas as pd
from itertools import combinations

# Load the data from the CSV file
try:
    df = pd.read_csv("output/baseline_final_combined.5-year.cds_mpnet.csv")
    print("Successfully read data from CSV file.")
except FileNotFoundError:
    print("Error: CSV file not found. Please check the file path.")

# List of target terms
target_terms = ["trauma", "anxiety", "depression", "mental_health", "mental_illness", "abuse"]

# Dictionary to store average pairwise distances for each term and injection ratio
pairwise_distances = {}

# Loop through each term and injection ratio to calculate pairwise distance between epochs
for term in target_terms:
    # Filter data for the current term
    term_data = df[df['term'] == term]
    
    # Get unique injection ratios
    inj_ratios = sorted(term_data['inj_ratio'].unique())
    pairwise_distances[term] = {}

    # Loop through each injection ratio
    for inj_ratio in inj_ratios:
        # Filter data for the current injection ratio
        ratio_data = term_data[term_data['inj_ratio'] == inj_ratio]
        
        # Get the mean cosine dissimilarity for each epoch
        avg_cosine_dissim = {
            epoch: ratio_data[ratio_data['epoch'] == epoch]['cosine_dissim_mean'].mean()
            for epoch in ratio_data['epoch'].unique() if not ratio_data[ratio_data['epoch'] == epoch]['cosine_dissim_mean'].isnull().all()
        }
        
        # Calculate pairwise differences between all epochs
        distances = []
        for (epoch1, value1), (epoch2, value2) in combinations(avg_cosine_dissim.items(), 2):
            distance = abs(value1 - value2)
            distances.append(distance)
        
        # Calculate the average pairwise distance for the injection ratio
        avg_distance = sum(distances) / len(distances) if distances else 0
        pairwise_distances[term][inj_ratio] = avg_distance

# Display the average pairwise distances for each term and injection ratio
pairwise_distances_df = pd.DataFrame([
    {"Term": term, "Injection Ratio": inj_ratio, "Average Pairwise Distance": avg_distance}
    for term, ratios in pairwise_distances.items()
    for inj_ratio, avg_distance in ratios.items()
])

# Sort and print the DataFrame
sorted_pairwise_distances_df = pairwise_distances_df.sort_values(by=['Term', 'Injection Ratio'])
print(sorted_pairwise_distances_df)
