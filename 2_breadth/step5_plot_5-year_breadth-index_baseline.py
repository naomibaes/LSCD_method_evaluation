import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors
import colorsys
from matplotlib.ticker import FuncFormatter

# Load the data from the CSV file
try:
    df = pd.read_csv("output/baseline_final_combined.5-year.cds_mpnet.csv")
    print("Successfully read data from CSV file.")
except FileNotFoundError:
    print("Error: CSV file not found. Please check the file path.")
    raise

# Adjusted colors for better differentiation
colors = {
    'abuse': '#8B0000',            # Dark Red
    'anxiety': '#FF6347',          # Tomato
    'depression': '#4B0082',       # Indigo
    'mental_health': '#008080',    # Teal
    'mental_illness': '#800080',   # Dark Purple
    'trauma': '#DC143C',           # Crimson
}

# Function to adjust the color with more contrast based on the injection ratio
def adjust_color_shade(base_color, ratio):
    ratio = int(ratio)
    rgb_color = mcolors.to_rgb(base_color)
    hls_color = colorsys.rgb_to_hls(*rgb_color)
    lightness = max(0.3, min(1.0, hls_color[1] + (ratio / 100 - 0.5) * 0.5))
    saturation = max(0.3, min(1.0, hls_color[2] + (ratio / 100 - 0.5) * 0.5))
    modified_rgb = colorsys.hls_to_rgb(hls_color[0], lightness, saturation)
    return modified_rgb

# Function to format y-axis ticks to 2 decimal places
def format_y_ticks(x, pos):
    return f'{x:.2f}'

# Create subplots for each term
fig, axes = plt.subplots(6, 1, figsize=(8, 12))
axes = axes.flatten()

# List of target terms
target_terms = ['abuse', 'anxiety', 'depression', 'mental_health', 'mental_illness', 'trauma']

# Define optimal positions for each term label
label_positions = {
    "trauma": (0.04, 0.965),
    "anxiety": (0.04, 0.965),
    "depression": (0.04, 0.965),
    "mental_health": (0.04, 0.965),
    "mental_illness": (0.04, 0.965),
    "abuse": (0.04, 0.965)
}

# Iterate over each term and create a subplot for each
for i, term in enumerate(target_terms):
    ax = axes[i]

    # Plot psych corpus data (renamed to "natural" in legend)
    psych_data = df[(df['term'] == term) & (df['corpus'] == 'natural')]
    if not psych_data.empty:
        x_values = psych_data['epoch']
        y_values = psych_data['cosine_dissim_mean']
        yerr = psych_data['cosine_dissim_se']
        ax.errorbar(x_values, y_values, yerr=yerr, fmt='o-', label='Natural', color=colors[term], 
                    markersize=4, linewidth=1.5, capsize=3)

    # Plot synthetic breadth.0 data with varying shades and line styles
    for inj_ratio in df['inj_ratio'].unique():
        data = df[(df['term'] == term) & (df['corpus'] == 'synthetic_breadth') & (df['inj_ratio'] == inj_ratio)]
        if not data.empty:
            x_values = data['epoch']
            y_values = data['cosine_dissim_mean']
            yerr = data['cosine_dissim_se']
            adjusted_color = adjust_color_shade(colors[term], int(inj_ratio))
            linestyle = '--' if inj_ratio != 0 else '-'
            ax.errorbar(x_values, y_values, yerr=yerr, fmt='o'+linestyle, color=adjusted_color, 
                        markersize=4, linewidth=1.5, capsize=3)

    # Position the term label with specific coordinates and background transparency
    label_x, label_y = label_positions[term]
    ax.text(label_x, label_y, term.capitalize(), transform=ax.transAxes, fontsize=14,
            ha="left", va="top", bbox=dict(facecolor='white', edgecolor='none', alpha=0.0, pad=1), fontweight='bold')

    # Set y-axis label, only set x-axis label for the last plot
    ax.set_ylabel('Breadth', fontsize=16)
    if i == len(target_terms) - 1:
        ax.set_xlabel('Epoch', fontsize=16)
    else:
        ax.set_xticklabels([])

    # Remove grid lines
    ax.grid(False)

    # Apply y-axis tick formatting and set tick locations for at least three labels
    y_min, y_max = ax.get_ylim()
    ax.set_yticks(np.linspace(y_min, y_max, 4))  # Ensure at least three y-axis labels
    ax.yaxis.set_major_formatter(FuncFormatter(format_y_ticks))
    ax.tick_params(axis='x', labelrotation=30, labelsize=14)
    ax.tick_params(axis='y', labelsize=14)  # Set y-axis tick label size to 14

# Minimize space between plots
plt.subplots_adjust(hspace=0.1)

# Define the legend
legend_lines = [
    plt.Line2D([0], [0], color='black', linestyle='-', linewidth=1.5),
    plt.Line2D([0], [0], color='black', linestyle='--', linewidth=1.5)
]
legend_labels = ['Natural', 'Synthetic']
fig.legend(legend_lines, legend_labels, loc='upper center', bbox_to_anchor=(0.5, 0.07), ncol=2, fontsize=16, frameon=False)

# Adjust layout and remove extra x-axis label
plt.tight_layout(rect=[0, 0.05, 1, 0.95])  # Adjusted bottom spacing to align with other plots

# Save and display the plot
plt.savefig("../figures/plot_5-year_breadth_baseline.png", dpi=600, bbox_inches='tight')
plt.show()
