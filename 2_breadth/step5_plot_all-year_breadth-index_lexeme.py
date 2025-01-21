import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np

# Load the data from the CSV file
try:
    df = pd.read_csv("output/baseline_final_combined.all-year.cds_lexeme.csv")
    print("Successfully read data from CSV file.")
except FileNotFoundError:
    print("Error: CSV file not found. Please check the file path.")

# Define colors, line styles, and marker styles
colors = {
    'abuse': '#8B0000',
    'anxiety': '#FF6347',
    'depression': '#4B0082',
    'mental_health': '#008080',
    'mental_illness': '#800080',
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

# List of target terms
sorted_target_terms = sorted(['abuse', 'anxiety', 'depression', 'mental_health', 'mental_illness', 'trauma'])

# Create the plot
fig, ax = plt.subplots(figsize=(14, 10.1))

# Function to format y-axis ticks to 2 decimal places
def format_y_ticks(x, pos):
    return f'{x:.2f}'

# Loop through each term and plot them
for term in sorted_target_terms:
    term_data = df[df['term'] == term]
    if not term_data.empty:
        sorted_data = term_data.sort_values(by='inj_ratio')
        ax.errorbar(sorted_data['inj_ratio'], sorted_data['cosine_dissim_mean'],
                    yerr=sorted_data['cosine_dissim_se'], fmt='o-', color=colors[term],
                    linestyle=line_styles[term], marker=marker_style[term], markersize=14, linewidth=3)

# Add grey shading for certain injection levels
ax.axvspan(10, 90, color='grey', alpha=0.2)
ax.axvspan(90, 105, color='darkgrey', alpha=0.5)

# Set y-axis limits and ticks
y_min = 0
y_max = 0.35  # Fixed at 0.76
ax.set_ylim(y_min, y_max)
y_ticks = np.arange(y_min, y_max + 0.05, 0.05)  # Adjusted increment to 0.02
ax.set_yticks(y_ticks)

# Axis labels and ticks
ax.set_xlabel('Synthetic Injection Levels (%)', fontsize=32)
ax.set_ylabel('Breadth Index (±SE)', fontsize=32)
ax.set_xticks([0, 20, 40, 60, 80, 100])
ax.set_xticklabels(['0%', '20%', '40%', '60%', '80%', '100%'], fontsize=28)
ax.set_xlim(-5, 105)
ax.yaxis.set_major_formatter(FuncFormatter(format_y_ticks))
ax.tick_params(axis='both', labelsize=28)

# Legend
handles = [plt.Line2D([0], [0], color=colors[term], linestyle=line_styles[term], marker=marker_style[term], markersize=14, linewidth=3, label=f'{term.capitalize()}') for term in sorted_target_terms]
legend = ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.18), fontsize=30, ncol=3, frameon=False, title="Target Terms")
plt.setp(legend.get_title(), fontsize=30)

# Save and show the plot
plt.tight_layout(rect=[0, 0.08, 1, 0.95])  # Adjusted rect to add more space for the legend
plt.savefig("../figures/plot_all-year_breadth_baseline_with_SE_lexeme.png", dpi=600, bbox_inches='tight')
plt.show()
