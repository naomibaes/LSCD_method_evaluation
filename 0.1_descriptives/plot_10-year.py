# Setup dependencies
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Specify the path to the CSV file
csv_file = "output/year_counts_sentences.csv"

# Read the CSV file into a DataFrame
df = pd.read_csv(csv_file)

# Define targets and their colors
targets = ["mental_health", "mental_illness", "mental_disorder", "trauma", "addiction", 
           "bullying", "harassment", "abuse", "anger", "distress", "empathy", "grief", 
           "anxiety", "depression", "stress", "worry"]

colors = {
    'abuse': '#8B0000',            # Dark Red
    'addiction': '#2E8B57',        # Sea Green
    'anger': '#FF4500',            # Orange Red
    'anxiety': '#FF6347',          # Tomato
    'depression': '#4B0082',       # Indigo
    'distress': '#8B4513',         # Saddle Brown
    'grief': '#4682B4',            # Steel Blue
    'mental_disorder': '#6A5ACD',  # Slate Blue
    'mental_health': '#008080',    # Teal
    'mental_illness': '#800080',   # Dark Purple
    'stress': '#800000',           # Maroon
    'trauma': '#DC143C',           # Crimson
    'worry': '#708090'             # Slate Gray
}

corpora = ["psych"]

# Define the year range and padding
start_year = 1970
end_year = 2019
padding = 0.05
interval = 10

# Calculate x-axis limits with padding
x_min = start_year - (end_year - start_year) * padding
x_max = end_year + (end_year - start_year) * padding

# Collect valid axes for plotting
valid_axes = []
for target in targets:
    for corpus in corpora:
        filtered_df = df[(df['target'] == target) & (df['corpus'] == corpus)]
        if not filtered_df.empty:
            valid_axes.append((target, corpus))

# Determine the layout for 2-column grid
n_plots = len(valid_axes)
n_cols = 2  # 2 columns for each row
n_rows = (n_plots + n_cols - 1) // n_cols  # Calculate rows needed

# Create subplots in 2-column layout
fig, axs = plt.subplots(n_rows, n_cols, figsize=(12, n_rows * 3), constrained_layout=True)

# Flatten axs for easy indexing when plotting
axs = axs.flatten()

# Loop through valid targets and corpora for plotting
for index, (target, corpus) in enumerate(valid_axes):
    ax = axs[index]
    filtered_df = df[(df['target'] == target) & (df['corpus'] == corpus)].copy()

    # Convert 'year' to numeric and group by 10-year intervals
    filtered_df.loc[:, 'year'] = pd.to_numeric(filtered_df['year'], errors='coerce')
    filtered_df['year_interval'] = (filtered_df['year'] // interval) * interval
    grouped_df = filtered_df.groupby('year_interval')['count'].sum().reset_index()

    # Prepare bin labels for the x-axis
    bin_labels = [f"{int(year)}-{int(year + interval - 1)}" for year in grouped_df['year_interval']]
    
    # Plot the data with the color assigned to the target
    color = colors.get(target, 'black')  # Default to black if target color is not specified
    ax.bar(grouped_df['year_interval'], grouped_df['count'], width=interval, color=color, edgecolor='darkgrey')
    
    # Simplified title with term label only
    ax.set_title(f"{target.replace('_', ' ').capitalize()}", fontsize=10)
    ax.set_xlabel('Epoch', color='black', fontsize=8)
    ax.set_ylabel('Sentence Count', color='black', fontsize=8)

    # Set x-axis ticks and labels to show intervals, rotate labels for better readability
    ax.set_xticks(grouped_df['year_interval'])
    ax.set_xticklabels(bin_labels, rotation=30, ha="right", fontsize=7)

    # Style adjustments for classic theme
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('black')
    ax.spines['bottom'].set_color('black')
    ax.tick_params(axis='x', colors='black', labelsize=7)
    ax.tick_params(axis='y', colors='black', labelsize=7)
    ax.grid(False)
    ax.set_facecolor('white')

    # Horizontal line at count = 500 with increased width for visibility
    ax.axhline(y=500, color='black', linestyle='--', linewidth=1.2)

# Hide any unused subplots
for j in range(index + 1, len(axs)):
    fig.delaxes(axs[j])

# Save the figure as a PNG file
output_directory = "output"
output_file_path = f"{output_directory}/10-year_counts_sentences.png"
plt.savefig(output_file_path, bbox_inches='tight', dpi=300)

# Show the plots
plt.show()

print(f"Plots saved to {output_file_path}")
