# Authors: Naomi Baes and Chat GPT

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np

# Load the data from the CSV file
df = pd.read_csv("output/baseline_averaged_arousal_index_all-year.csv")

# Ensure 'injection_ratio' is an integer for accurate comparisons
df['injection_ratio'] = pd.to_numeric(df['injection_ratio'], errors='coerce')

# Adjusted colors for better differentiation
colors = {
    'abuse': '#8B0000',            # Dark Red
    'anxiety': '#FF6347',          # Tomato
    'depression': '#4B0082',       # Indigo
    'mental_health': '#008080',    # Teal
    'mental_illness': '#800080',   # Dark Purple
    'trauma': '#DC143C',           # Crimson
}

# List of target terms sorted alphabetically
target_terms = sorted(["trauma", "anxiety", "depression", "mental_health", "mental_illness", "abuse"])

# Create subplots with shared x-axis and closer spacing
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 14), sharex=True, gridspec_kw={'hspace': 0.1})

# Function to format y-axis ticks to 2 decimal places
def format_y_ticks(x, pos):
    return f'{x:.2f}'

# Increase font sizes
font_label = 32  # Axis labels
font_tick = 28   # Axis tick numerals
font_legend = 26  # Legend text

# Determine dynamic y-range based on data to ensure ticks cover all data points
min_vals, max_vals = [], []

for term in target_terms:
    term_data = df[df['target'] == term]
    if not term_data.empty:
        min_vals.append(term_data[['avg_arousal_index_high', 'avg_arousal_index_low']].min().min())
        max_vals.append(term_data[['avg_arousal_index_high', 'avg_arousal_index_low']].max().max())

# Adjust y-axis limits and tick marks dynamically
y_min = np.floor(min(min_vals) * 10) / 10
y_max = np.ceil(max(max_vals) * 10) / 10

# Set tick range with 0.10 intervals
y_ticks = np.arange(y_min, y_max + 0.1, 0.10)

# Plot High Arousal Index (Top Panel)
for term in target_terms:
    term_data = df[df['target'] == term]
    if not term_data.empty:
        term_data = term_data.sort_values(by='injection_ratio')
        ax1.errorbar(term_data['injection_ratio'], term_data['avg_arousal_index_high'],
                     yerr=term_data['se_arousal_index_high'], fmt='o', color=colors[term],
                     capsize=4, markersize=6)
        ax1.plot(term_data['injection_ratio'], term_data['avg_arousal_index_high'],
                 color=colors[term], linestyle='-')

ax1.set_ylim(y_min, y_max)
ax1.set_yticks(y_ticks)

# Plot Low Arousal Index (Bottom Panel) with circle markers
for term in target_terms:
    term_data = df[df['target'] == term]
    if not term_data.empty:
        term_data = term_data.sort_values(by='injection_ratio')
        ax2.errorbar(term_data['injection_ratio'], term_data['avg_arousal_index_low'],
                     yerr=term_data['se_arousal_index_low'], fmt='o', color=colors[term],
                     capsize=4, markersize=6)
        ax2.plot(term_data['injection_ratio'], term_data['avg_arousal_index_low'],
                 color=colors[term], linestyle='--')

ax2.set_ylim(y_min, y_max)
ax2.set_yticks(y_ticks)

# Add grey shading for injections > 0% and darker shading beyond 90%
ax1.axvspan(10, 90, color='grey', alpha=0.2)  # Lighter shading for 0-90%
ax2.axvspan(10, 90, color='grey', alpha=0.2)
ax1.axvspan(90, 105, color='darkgrey', alpha=0.5)  # Darker shading for 90-105%
ax2.axvspan(90, 105, color='darkgrey', alpha=0.5)

# Set x-axis and y-axis labels with increased font size
ax2.set_xlabel('Synthetic Sentence Injections (%)', fontsize=font_label)
ax1.set_ylabel('Arousal Score', fontsize=font_label)
ax2.set_ylabel('Arousal Score', fontsize=font_label)

# Set x-ticks and y-ticks formatting with larger font
ax2.set_xticks([0, 20, 40, 60, 80, 100])
ax2.set_xticklabels(['0%', '20%', '40%', '60%', '80%', '100%'], fontsize=font_tick)
for ax in [ax1, ax2]:
    ax.set_xlim(-5, 105)  # Explicitly set x-axis limits to extend slightly beyond 100%
    ax.yaxis.set_major_formatter(FuncFormatter(format_y_ticks))
    ax.tick_params(axis='both', labelsize=font_tick)

# Add legends for target terms
handles, labels = [], []
for term in target_terms:
    handles.append(plt.Line2D([0], [0], color=colors[term], label=term.capitalize()))
legend = fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.04), fontsize=font_legend, ncol=3, frameon=False, title="Target Terms")

# Adjust the title font size of the legend
plt.setp(legend.get_title(), fontsize=font_legend)

# Adjust layout to ensure no overlap and optimize space
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

# Save the plot with adjustments
plt.savefig("../figures/plot_all-year_intensity_BOTH_baseline.png", dpi=600, bbox_inches='tight')
plt.show()
