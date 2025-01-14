import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors
import colorsys
from matplotlib.ticker import FuncFormatter

# Load the data from the provided input
try:
    df = pd.read_csv("output/averaged_valence_index_5-year.csv")
    print("Successfully read data from CSV file.")
except FileNotFoundError:
    print("Error: CSV file not found. Please check the file path.")

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

# Define y-axis formatter
def format_y_ticks(x, pos):
    return f'{x:.2f}'

# Create subplots for each term
fig, axes = plt.subplots(6, 1, figsize=(8, 12), sharex=True)
axes = axes.flatten()

# Target terms and label positions
target_terms = [ 'abuse', 'anxiety', 'depression', 'mental_health', 'mental_illness', 'trauma']

label_positions = {
    "trauma": (0.95, 0.85),
    "anxiety": (0.95, 0.2),
    "depression": (0.95, 0.96),
    "mental_health": (0.95, 0.3),
    "mental_illness": (0.95, 0.2),
    "abuse": (0.95, 0.25)
}

# Set labels and intervals for epochs
epoch_labels = sorted(set(df['epoch']))
epoch_positions = [int(label.split('-')[0]) for label in epoch_labels]

# Iterate over each term and create a subplot for each
for i, term in enumerate(target_terms):
    ax = axes[i]

    # Plot data for each injection ratio
    for injection_ratio in sorted(df['injection_ratio'].unique(), reverse=True):
        data = df[(df['target'] == term) & (df['injection_ratio'] == injection_ratio)]
        linestyle = '--' if injection_ratio != 0 else '-'  # Dashed for Synthetic, Solid for Natural
        if not data.empty:
            x_values = [int(epoch.split('-')[0]) for epoch in data['epoch']]  # Use start year as x value
            y_values = data['avg_valence_index_negative']
            y_errors = data['se_valence_index_negative']
            color = adjust_color_shade(colors[term], injection_ratio)

            # Plot with error bars
            ax.errorbar(x_values, y_values, yerr=y_errors, linestyle=linestyle, marker='o', color=color, capsize=3)

    # Position the term label with specific coordinates and background
    label_x, label_y = label_positions[term]
    ax.text(label_x, label_y, term.capitalize(), transform=ax.transAxes, fontsize=14,
            ha="right", va="top", backgroundcolor="white", bbox=dict(facecolor='white', edgecolor='none', pad=1))

    # Set y-axis label
    ax.set_ylabel('Val (±SE)', fontsize=16)

    # Set x-axis ticks and labels only for the last plot
    ax.set_xticks(epoch_positions)
    ax.set_xticklabels(epoch_labels, rotation=45, fontsize=16 if i == len(target_terms) - 1 else 0)
    ax.tick_params(axis='x', which='both', length=0)  # Hide tick marks for all but the last plot

    # Apply y-axis tick formatting
    y_min, y_max = ax.get_ylim()
    ax.set_yticks(np.linspace(y_min, y_max, 4))
    ax.yaxis.set_major_formatter(FuncFormatter(format_y_ticks))
    ax.tick_params(axis='y', labelsize=14)

# Define the legend for Natural and Synthetic with black lines
legend_lines = [
    plt.Line2D([0], [0], color='black', linestyle='-', linewidth=1.5),
    plt.Line2D([0], [0], color='black', linestyle='--', linewidth=1.5)
]
legend_labels = ['Natural', 'Synthetic']
fig.legend(legend_lines, legend_labels, loc='lower center', bbox_to_anchor=(0.5, -0.02), ncol=2, fontsize=16, frameon=False)

# Adjust layout
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

# Save and display the plot
plt.savefig("../figures/plot_5-year_sentiment_NEG.png", dpi=600, bbox_inches='tight')
plt.show()