import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np

# Load the data
df = pd.read_csv("output/control_averaged_valence_index_all-year_normalized.csv")
df['injection_ratio'] = pd.to_numeric(df['injection_ratio'], errors='coerce')
df['round_number'] = pd.to_numeric(df['round_number'], errors='coerce')

# Define colors, line styles, and marker styles
colors = {
    'abuse': '#8B0000',
    'anxiety': '#FF4500',
    'depression': '#483D8B',
    'mental_health': '#008080',
    'mental_illness': '#6A0DAD',
    'trauma': '#DC143C',
}
line_styles = {
    'abuse': '-',
    'anxiety': '--',
    'depression': '-.',
    'mental_health': ':',
    'mental_illness': (0, (3, 10, 1, 10)),
    'trauma': (0, (5, 1))
}
marker_style = {
    'abuse': 'o',
    'anxiety': 's',
    'depression': 'D',
    'mental_health': '^',
    'mental_illness': 'v',
    'trauma': 'P'
}

# Sorted list of target terms
target_terms = sorted(["trauma", "anxiety", "depression", "mental_health", "mental_illness", "abuse"])

# Map for legend labels
legend_labels = {
    'trauma': 'Trauma',
    'anxiety': 'Anxiety',
    'depression': 'Depression',
    'mental_health': 'Mental Health',
    'mental_illness': 'Mental Illness',
    'abuse': 'Abuse'
}

# Create subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 14), sharex=True, gridspec_kw={'hspace': 0.1})

# Plotting with error bars
for term in target_terms:
    term_data = df[df['target'] == term]
    if not term_data.empty:
        sorted_data = term_data.sort_values(by=['injection_ratio', 'round_number'])
        ax1.errorbar(sorted_data['round_number'], sorted_data['avg_valence_index_positive'],
                     yerr=sorted_data['se_valence_index_positive'], fmt='o-', color=colors[term],
                     linestyle=line_styles[term], marker=marker_style[term], markersize=10, linewidth=2, label=legend_labels[term])
        ax2.errorbar(sorted_data['round_number'], sorted_data['avg_valence_index_negative'],
                     yerr=sorted_data['se_valence_index_negative'], fmt='o-', color=colors[term],
                     linestyle=line_styles[term], marker=marker_style[term], markersize=10, linewidth=2)

# Add labels to the upper left corner
ax1.text(0.03, 0.95, 'Positive Sentiment Injection', transform=ax1.transAxes, fontsize=18, verticalalignment='top')
ax2.text(0.03, 0.95, 'Negative Sentiment Injection', transform=ax2.transAxes, fontsize=18, verticalalignment='top')

# Formatting function for y-axis ticks
def format_y_ticks(x, pos):
    return f'{x:.2f}'

# Set y-axis limits and ticks
ax1.set_ylim(0.45, 0.55)
ax2.set_ylim(0.45, 0.55)
ax1.set_ylabel('Valence Index (±SE)', fontsize=14)
ax2.set_ylabel('Valence Index (±SE)', fontsize=14)
ax2.set_xlabel('Synthetic Injection Level Rounds (50%)', fontsize=14)

# Format x-axis
ax2.set_xticks([1, 2, 3, 4, 5, 6])
ax2.set_xticklabels(['1', '2', '3', '4', '5', '6'], fontsize=12)
for ax in [ax1, ax2]:
    ax.tick_params(axis='both', labelsize=12)
    ax.yaxis.set_major_formatter(FuncFormatter(format_y_ticks))

# Legend
handles = [plt.Line2D([0], [0], color=colors[term], linestyle=line_styles[term], marker=marker_style[term],
                      markersize=10, linewidth=2, label=legend_labels[term]) for term in target_terms]
legend = fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.02), fontsize=14, ncol=3,
                    frameon=False, title="Target Terms")
plt.setp(legend.get_title(), fontsize=14)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
#plt.savefig("valence_index_plot.png", dpi=300, bbox_inches='tight')
plt.show()
