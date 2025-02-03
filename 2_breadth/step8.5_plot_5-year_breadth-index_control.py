import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors
from matplotlib.ticker import FuncFormatter

# Load the data from the CSV file
try:
    df = pd.read_csv("output/control_final_combined.5-year.cds_mpnet.csv")
    print("Successfully read data from CSV file.")
except FileNotFoundError:
    print("Error: CSV file not found. Please check the file path.")
    raise

# Define colors for each target term
colors = {
    'abuse': '#8B0000',            # Dark Red
    'anxiety': '#FF6347',          # Tomato
    'depression': '#4B0082',       # Indigo
    'mental_health': '#008080',    # Teal
    'mental_illness': '#800080',   # Dark Purple
    'trauma': '#DC143C',           # Crimson
}

# Function to format y-axis ticks to 2 decimal places
def format_y_ticks(x, pos):
    return f'{x:.2f}'

# Create subplots for each term
fig, axes = plt.subplots(6, 1, figsize=(8, 12), sharex=True)
axes = axes.flatten()

# List of target terms
target_terms = ['abuse', 'anxiety', 'depression', 'mental_health', 'mental_illness', 'trauma']

# Iterate over each term and create a subplot for each
for i, term in enumerate(target_terms):
    ax = axes[i]

    # Filter data for the term
    term_data = df[df['term'] == term]

    # Plot data for each round, keeping color consistent for the term
    for round_num in sorted(term_data['round_number'].unique()):
        round_data = term_data[term_data['round_number'] == round_num]
        x_values = round_data['epoch']
        y_values = round_data['cosine_dissim_mean']
        yerr = round_data['cosine_dissim_se']

        # Keep term color, vary linestyle to differentiate rounds
        linestyle = '-' if round_num % 2 == 0 else '--'
        
        ax.errorbar(x_values, y_values, yerr=yerr, fmt='o', linestyle=linestyle, label=f'Round {round_num}', 
                    color=colors[term], markersize=4, linewidth=1.5, capsize=3)

    # Set axis labels
    ax.set_ylabel('Breadth', fontsize=14)
    if i == len(target_terms) - 1:
        ax.set_xlabel('Epoch', fontsize=14)
        ax.set_xticks(range(len(df['epoch'].unique())))  # Set correct x-axis ticks
        ax.set_xticklabels(df['epoch'].unique(), rotation=30)  # Rotate for readability
    else:
        ax.set_xticklabels([])

    # Remove grid lines and format y-axis ticks
    ax.grid(False)
    ax.yaxis.set_major_formatter(FuncFormatter(format_y_ticks))
    ax.tick_params(axis='x', labelrotation=30, labelsize=12)
    ax.tick_params(axis='y', labelsize=12)

    # Add term label to subplot
    ax.text(0.02, 0.9, term.capitalize(), transform=ax.transAxes, fontsize=14,
            ha="left", va="top", bbox=dict(facecolor='white', edgecolor='none', alpha=0.0, pad=1), fontweight='bold')

# Adjust subplot spacing
plt.subplots_adjust(hspace=0.2)

# Define legend for rounds
fig.legend(title="Rounds", loc='upper center', bbox_to_anchor=(0.5, 0.07), ncol=3, fontsize=12, frameon=False)

# Adjust layout and save the plot
plt.tight_layout(rect=[0, 0.05, 1, 0.95])
plt.savefig("../figures/plot_5-year_breadth_control.png", dpi=600, bbox_inches='tight')
plt.show()
