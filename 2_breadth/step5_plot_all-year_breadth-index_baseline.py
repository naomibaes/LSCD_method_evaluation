import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np

# Load the data from the CSV file
try:
    df = pd.read_csv("output/baseline_final_combined.all-year.cds_mpnet.csv")
    print("Successfully read data from CSV file.")
except FileNotFoundError:
    print("Error: CSV file not found. Please check the file path.")

# Define colors, line styles, and marker styles
colors = {
    'Abuse': '#8B0000',
    'Anxiety': '#FF6347',
    'Depression': '#4B0082',
    'Mental Health': '#008080',
    'Mental Illness': '#800080',
    'Trauma': '#DC143C',
}
line_styles = {
    'Abuse': '-',
    'Anxiety': '--',
    'Depression': '-.',
    'Mental Health': ':',
    'Mental Illness': (0, (3, 10, 1, 10)),
    'Trauma': (0, (5, 1))
}
marker_style = {
    'Abuse': 'o',
    'Anxiety': 's',
    'Depression': 'D',
    'Mental Health': '^',
    'Mental Illness': 'v',
    'Trauma': 'P'
}

# List of target terms
sorted_target_terms = sorted(['Abuse', 'Anxiety', 'Depression', 'Mental Health', 'Mental Illness', 'Trauma'])

# Create the plot
fig, ax = plt.subplots(figsize=(14, 10.1))

# Function to format y-axis ticks to 2 decimal places
def format_y_ticks(x, pos):
    return f'{x:.2f}'

# Loop through each term and plot them
for term in sorted_target_terms:
    term_data = df[df['term'].str.replace('_', ' ').str.title() == term]
    if not term_data.empty:
        sorted_data = term_data.sort_values(by='inj_ratio')
        ax.errorbar(sorted_data['inj_ratio'], sorted_data['cosine_dissim_mean'],
                    yerr=sorted_data['cosine_dissim_se'], fmt='o-', color=colors[term],
                    linestyle=line_styles[term], marker=marker_style[term], markersize=14, linewidth=3)

# Add grey shading for certain injection levels
ax.axvspan(10, 90, color='grey', alpha=0.2)
ax.axvspan(90, 105, color='darkgrey', alpha=0.5)

# Set y-axis limits and ticks
y_min = 0.66
y_max = 0.77  # Fixed at 0.76
ax.set_ylim(y_min, y_max)
y_ticks = np.arange(y_min, y_max + 0.0, 0.02)  # Adjusted increment to 0.02
ax.set_yticks(y_ticks)

ax.text(0.03, 0.95, 'Diverse Contexts Injection', transform=ax.transAxes, fontsize=28, verticalalignment='top')

# Axis labels and ticks
ax.set_xlabel('Synthetic Injection Levels (%)', fontsize=32)
ax.set_ylabel('Breadth Index (±SE)', fontsize=32)
ax.set_xticks([0, 20, 40, 60, 80, 100])
ax.set_xticklabels(['0%', '20%', '40%', '60%', '80%', '100%'], fontsize=28)
ax.set_xlim(-5, 105)
ax.yaxis.set_major_formatter(FuncFormatter(format_y_ticks))
ax.tick_params(axis='both', labelsize=28)

# Legend
handles = [plt.Line2D([0], [0], color=colors[term], linestyle=line_styles[term], marker=marker_style[term], markersize=14, linewidth=3, label=term) for term in sorted_target_terms]
legend = ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.18), fontsize=30, ncol=3, frameon=False, title="Target Terms")
plt.setp(legend.get_title(), fontsize=30)

# Save and show the plot
plt.tight_layout(rect=[0, 0.08, 1, 0.95])  # Adjusted rect to add more space for the legend
plt.savefig("../figures/plot_all-year_breadth_baseline_with_SE.png", dpi=600, bbox_inches='tight')
plt.show()
