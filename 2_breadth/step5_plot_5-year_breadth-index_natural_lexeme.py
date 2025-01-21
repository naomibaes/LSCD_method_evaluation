import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr

# Load the data from the CSV file
try:
    df = pd.read_csv("output/baseline_final_combined.5-year.cds_lexeme.csv")
    print("Successfully read data from CSV file.")
except FileNotFoundError:
    print("Error: CSV file not found. Please check the file path.")

target_terms = ["trauma", "anxiety", "depression", "mental_health", "mental_illness", "abuse"]

# Adjusted colors for better differentiation
colors = {
    'abuse': '#8B0000',            # Dark Red
    'anxiety': '#FF6347',          # Tomato
    'depression': '#4B0082',       # Indigo
    'mental_health': '#008080',    # Teal
    'mental_illness': '#800080',   # Dark Purple
    'trauma': '#DC143C',           # Crimson
}

# Set the figure size
plt.figure(figsize=(12, 7))

# Filter to include only data where injection ratio is 0
filtered_df = df[df['inj_ratio'] == 0]

# Get the unique epochs to ensure they are all shown
epochs = sorted(filtered_df['epoch'].unique())

# Sort target terms alphabetically for plotting and legend ordering
target_terms.sort()

# Iterate over each term and plot only the 'psych' corpus
plot_labels = []  # To store plot labels for sorting in legend
for term in target_terms:
    # Filter the data for the specific term
    term_data = filtered_df[(filtered_df['term'] == term)]
    
    # Check if there's enough data to plot
    if not term_data.empty and np.std(term_data['cosine_dissim_mean']) > 0:  # Check for variance in the data
        x_values = [int(e.split('-')[0]) for e in term_data['epoch']]  # Convert epochs to numeric (start year of each range)
        y_values = term_data['cosine_dissim_mean']
        y_errors = term_data['cosine_dissim_se']  # Standard errors
        
        # Calculate the correlation coefficient (Pearson) and p-value
        if len(x_values) > 1:  # Ensure there is enough data for correlation
            correlation, p_value = pearsonr(x_values, y_values)
            p_value_str = f"{p_value:.3f}" if p_value >= 0.001 else "<0.001"  # Format p-value
            label = f'{term} (r={correlation:.2f}, p={p_value_str})'
        else:
            label = term  # Basic label if not enough data for correlation
        
        # Plot the data with error bars
        plot_labels.append((label, {
            'x': term_data['epoch'],
            'y': y_values,
            'yerr': y_errors,
            'color': colors.get(term, 'black'),
            'marker': 'o',
            'capsize': 5
        }))

# Plot with labels sorted alphabetically
for label, kwargs in sorted(plot_labels):
    plt.errorbar(**kwargs, label=label)

# Set labels and title
plt.xlabel('Epoch')
plt.ylabel('Cosine Dissimilarity Mean')
plt.title('Cosine Dissimilarity Across Epochs by Term (Injection Ratio = 0)')

# Customize x-axis ticks to include all unique epochs and rotate them for better readability
plt.xticks(ticks=epochs, rotation=45, ha='right')

# Add gridlines
plt.grid(True)

# Add a legend
plt.legend()

# Improve layout
plt.tight_layout()

# Save the plot to the output folder
#plt.savefig("output/plot0_inj_ratio_0.png", dpi=300, bbox_inches='tight')

# Display the plot
plt.show()
