import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.ticker import AutoLocator, AutoMinorLocator

# Specify the path to the CSV file
csv_file = "output/year_counts_sentences.csv"

# Read the CSV file into a DataFrame
df = pd.read_csv(csv_file)

# Define targets and their colors
targets = ["abuse", "anxiety", "depression", "mental_health", "mental_illness", "trauma"]
colors = {
    'abuse': '#8B0000',
    'anxiety': '#FF6347',
    'depression': '#4B0082',
    'mental_health': '#008080',
    'mental_illness': '#800080',
    'trauma': '#DC143C'
}

# Determine the layout for 2-column grid
n_plots = len(targets)
n_cols = 2  # 2 columns for each row
n_rows = (n_plots + n_cols - 1) // n_cols  # Calculate rows needed

# Create subplots in 2-column layout
fig, axs = plt.subplots(n_rows, n_cols, figsize=(10, n_rows * 3), constrained_layout=True)
axs = axs.flatten()  # Flatten axs for easy indexing

# Loop through targets for plotting
for i, target in enumerate(targets):
    filtered_df = df[df['target'] == target].copy()

    # Convert 'year' to numeric and group by 5-year intervals
    filtered_df['year'] = pd.to_numeric(filtered_df['year'], errors='coerce')
    filtered_df['year_interval'] = (filtered_df['year'] // 5) * 5
    grouped_df = filtered_df.groupby('year_interval')['count'].sum().reset_index()

    # Plot data
    ax = axs[i]
    ax.bar(grouped_df['year_interval'], grouped_df['count'], color=colors[target], edgecolor="black", width=4.5)
    ax.set_title(target.capitalize(), fontsize=20)
    ax.set_ylabel('Count', fontsize=18)

    # Dynamic y-axis interval based on the maximum count
    max_count = grouped_df['count'].max()
    interval = max(100, int(max_count / 10))  # Adjust the 10 to control the granularity
    ax.yaxis.set_major_locator(AutoLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))

    # Only set x-tick labels on the bottom plots
    if i >= n_plots - n_cols:
        ax.set_xticks(grouped_df['year_interval'])
        ax.set_xticklabels([f"{year}-{year+4}" for year in grouped_df['year_interval']], rotation=45, ha="right", fontsize=16)
    else:
        ax.set_xticks([])  # Hide x-axis labels
    ax.tick_params(axis='y', labelsize=16)  # Adjust '16' to your desired font size
    
    #ax.axhline(y=500, color='black', linestyle='--', linewidth=1.5)  # Threshold line
    ax.grid(False)  # Turn off the grid
    ax.set_facecolor('white')  # Set background to white

# Hide unused subplots
for j in range(len(targets), len(axs)):
    fig.delaxes(axs[j])

# Save the figure
output_file_path = "../figures/plot_appendixA.png"
plt.savefig(output_file_path, dpi=300, bbox_inches='tight')
plt.show()

print(f"Plots saved to {output_file_path}")
