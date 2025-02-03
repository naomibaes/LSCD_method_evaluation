import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors
import colorsys
from matplotlib.ticker import FuncFormatter

# Load the dataset
try:
    df = pd.read_csv("output/control_averaged_valence_index_5-year_normalized.csv")
    print("Successfully read data from CSV file.")
except FileNotFoundError:
    print("Error: CSV file not found. Please check the file path.")
    raise

# Ensure injection_ratio is treated as an integer
df['injection_ratio'] = df['injection_ratio'].astype(int)

# Adjusted colors for better differentiation
colors = {
    'abuse': '#8B0000',            # Dark Red
    'anxiety': '#FF6347',          # Tomato
    'depression': '#4B0082',       # Indigo
    'mental_health': '#008080',    # Teal
    'mental_illness': '#800080',   # Dark Purple
    'trauma': '#DC143C',           # Crimson
}

# Function to adjust the color shade based on injection ratio
def adjust_color_shade(base_color, ratio):
    """Modify color contrast based on injection ratio"""
    rgb_color = mcolors.to_rgb(base_color)
    hls_color = colorsys.rgb_to_hls(*rgb_color)
    lightness = max(0.3, min(1.0, hls_color[1] + (ratio / 10 - 0.5) * 0.5))
    modified_rgb = colorsys.hls_to_rgb(hls_color[0], lightness, hls_color[2])
    return modified_rgb

# Define y-axis formatter
def format_y_ticks(x, pos):
    return f'{x:.2f}'

# Create subplots for each term
fig, axes = plt.subplots(6, 1, figsize=(8, 12), sharex=True)
axes = axes.flatten()

# Target terms and human-readable labels
target_terms = ['abuse', 'anxiety', 'depression', 'mental_health', 'mental_illness', 'trauma']
readable_labels = {
    'mental_health': 'Mental Health',
    'mental_illness': 'Mental Illness',
    'abuse': 'Abuse',
    'anxiety': 'Anxiety',
    'depression': 'Depression',
    'trauma': 'Trauma'
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
        
        if not data.empty:
            x_values = [int(epoch.split('-')[0]) for epoch in data['epoch']]  # Use start year as x-axis value
            
            # Positive Sentiment: Always Dashed (`--`)
            y_values_pos = data['avg_valence_index_positive']
            color = adjust_color_shade(colors[term], injection_ratio)
            ax.plot(x_values, y_values_pos, linestyle='--', marker='o', color=color, label=f'Positive (Ratio {injection_ratio})')

            # Negative Sentiment: Always Dotted (`:`)
            y_values_neg = data['avg_valence_index_negative']
            ax.plot(x_values, y_values_neg, linestyle=':', marker='x', color=color, label=f'Negative (Ratio {injection_ratio})')

            # Annotate injection ratio labels on the **right-hand side**
            ax.annotate(f'{injection_ratio}', (x_values[-1], y_values_pos.iloc[-1]), 
                        textcoords="offset points", xytext=(5,0), ha='left', fontsize=9, color=color)

            ax.annotate(f'{injection_ratio}', (x_values[-1], y_values_neg.iloc[-1]), 
                        textcoords="offset points", xytext=(5,0), ha='left', fontsize=9, color=color)

    # Set y-axis label
    ax.set_ylabel('Sentiment', fontsize=14)

    # Label the target term with the human-readable label
    y_min, y_max = ax.get_ylim()
    ax.text(1970, y_max - (y_max - y_min) / 7, readable_labels[term], fontsize=14, color='black', weight='bold')

    # Set x-axis ticks and labels explicitly for all subplots
    ax.set_xticks(epoch_positions)  
    if i == len(target_terms) - 1:
        ax.set_xticklabels(epoch_labels, rotation=45, fontsize=12)  # Show epochs on the last subplot
    else:
        ax.set_xticklabels([])  # Hide x-axis labels for other subplots

    # Apply y-axis tick formatting and set tick locations
    ax.set_yticks(np.linspace(y_min, y_max, 4))  
    ax.yaxis.set_major_formatter(FuncFormatter(format_y_ticks))
    ax.tick_params(axis='x', labelrotation=30, labelsize=12)
    ax.tick_params(axis='y', labelsize=12)

# Define the legend
legend_lines = [
    plt.Line2D([0], [0], color='black', linestyle='--', linewidth=1.5, label='Synthetic (Positive)'),
    plt.Line2D([0], [0], color='black', linestyle=':', linewidth=1.5, label='Synthetic (Negative)')
]

# Adjust legend positioning
fig.legend(
    legend_lines,
    ['Synthetic (Positive)', 'Synthetic (Negative)'],
    loc='lower center',
    bbox_to_anchor=(0.5, -0.03),
    ncol=3,
    fontsize=14,
    frameon=False
)

# Add x-axis label "Epoch" to the shared x-axis
fig.text(0.5, 0.02, 'Epoch', ha='center', fontsize=14)

# Adjust layout
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

# Save and display the plot
plt.savefig("../figures/plot_5-year_sentiment_BOTH_control_normalized.png", dpi=600, bbox_inches='tight')
plt.show()
