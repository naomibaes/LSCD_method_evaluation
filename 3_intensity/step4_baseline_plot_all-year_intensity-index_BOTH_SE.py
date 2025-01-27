import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np

# Load the data
df = pd.read_csv("output/baseline_averaged_arousal_index_all-year.csv")
df['injection_ratio'] = pd.to_numeric(df['injection_ratio'], errors='coerce')

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
marker_styles = {
    'abuse': 'o',
    'anxiety': 's',
    'depression': 'D',
    'mental_health': '^',
    'mental_illness': 'v',
    'trauma': 'P'
}

# Adjusted target terms for capitalization in the legend
legend_labels = {
    'abuse': 'Abuse',
    'anxiety': 'Anxiety',
    'depression': 'Depression',
    'mental_health': 'Mental Health',
    'mental_illness': 'Mental Illness',
    'trauma': 'Trauma'
}

# Create subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 14), sharex=True, gridspec_kw={'hspace': 0.1})

# Plotting with error bars
for term in colors:
    term_data = df[df['target'] == term]
    if not term_data.empty:
        sorted_data = term_data.sort_values(by='injection_ratio')
        ax1.errorbar(sorted_data['injection_ratio'], sorted_data['avg_arousal_index_high'],
                     yerr=sorted_data['se_arousal_index_high'], fmt='o-', color=colors[term],
                     linestyle=line_styles[term], marker=marker_styles[term], markersize=14, linewidth=3)
        ax2.errorbar(sorted_data['injection_ratio'], sorted_data['avg_arousal_index_low'],
                     yerr=sorted_data['se_arousal_index_low'], fmt='o-', color=colors[term],
                     linestyle=line_styles[term], marker=marker_styles[term], markersize=14, linewidth=3)

# Add labels to the upper left corner
ax1.text(0.03, 0.95, 'High Intensity Injection', transform=ax1.transAxes, fontsize=28, verticalalignment='top')
ax2.text(0.03, 0.95, 'Low Intensity Injection', transform=ax2.transAxes, fontsize=28, verticalalignment='top')

# Add grey shading for certain injection levels
for ax in [ax1, ax2]:
    ax.axvspan(10, 90, color='darkgrey', alpha=0.7)
    ax.axvspan(90, 105, color='darkgrey', alpha=1)

# Formatting function for y-axis ticks
def format_y_ticks(x, pos):
    """ Format y-axis tick labels to two decimal places """
    return f'{x:.2f}'

# Set y-axis limits and ticks
ax1.set_ylim(3.80, 4.60)
ax2.set_ylim(3.80, 4.60)
y_ticks = np.arange(3.80, 4.60, 0.10)
ax1.set_yticks(y_ticks)
ax2.set_yticks(y_ticks)

# Axis labels and ticks
ax2.set_xlabel('Synthetic Injection Levels (%)', fontsize=32)
ax1.set_ylabel('Arousal Index (±SE)', fontsize=32)
ax2.set_ylabel('Arousal Index (±SE)', fontsize=32)
ax2.set_xticks([0, 20, 40, 60, 80, 100])
ax2.set_xticklabels(['0%', '20%', '40%', '60%', '80%', '100%'], fontsize=28)
for ax in [ax1, ax2]:
    ax.set_xlim(-5, 105)
    ax.yaxis.set_major_formatter(FuncFormatter(format_y_ticks))
    ax.tick_params(axis='both', labelsize=28)

# Legend with corrected capitalization
handles = [plt.Line2D([0], [0], color=colors[term], linestyle=line_styles[term], marker=marker_styles[term], markersize=14, linewidth=3, label=legend_labels[term]) for term in colors]
legend = fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.04), fontsize=30, ncol=3, frameon=False, title="Target Terms")
plt.setp(legend.get_title(), fontsize=30)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("../figures/plot_all-year_intensity_BOTH_baseline_with_SE.png", dpi=600, bbox_inches='tight')
plt.show()
