import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np

# Load the data from the CSV file
try:
    df = pd.read_csv("output/control_final_combined.all-year.cds_mpnet.csv")
    print("Successfully read data from CSV file.")
except FileNotFoundError:
    print("Error: CSV file not found. Please check the file path.")
    raise

# Define colors, line styles, and marker styles
colors = {
    'abuse': '#8B0000',            # Dark Red
    'anxiety': '#FF6347',          # Tomato
    'depression': '#483D8B',       # Dark Slate Blue
    'mental_health': '#008080',    # Teal
    'mental_illness': '#6A0DAD',   # Purple
    'trauma': '#D7191C',           # Red
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

# Convert term names to lowercase to match the dictionary
df['term'] = df['term'].str.lower()

# List of target terms in sorted order
sorted_target_terms = sorted(colors.keys())

# Create the plot
fig, ax = plt.subplots(figsize=(14, 10.1))

# Function to format y-axis ticks to 2 decimal places
def format_y_ticks(x, pos):
    return f'{x:.2f}'

# Loop through each term and plot them
for term in sorted_target_terms:
    term_data = df[df['term'] == term]
    if not term_data.empty:
        sorted_data = term_data.sort_values(by='round_number')
        ax.errorbar(sorted_data['round_number'], sorted_data['cosine_dissim_mean'],
                    yerr=sorted_data['cosine_dissim_se'], fmt='o-', color=colors[term],
                    linestyle=line_styles[term], marker=marker_style[term], markersize=14, linewidth=3)

# Set y-axis limits and ticks
y_min = df['cosine_dissim_mean'].min() - 0.02
y_max = df['cosine_dissim_mean'].max() + 0.02
ax.set_ylim(y_min, y_max)
y_ticks = np.arange(y_min, y_max + 0.0, 0.02)
ax.set_yticks(y_ticks)

# Title and Axis Labels
ax.set_xlabel('Synthetic Injection Level (50%)', fontsize=32)
ax.set_ylabel('Cosine Distance (±SE)', fontsize=32)

# Set X-axis
ax.set_xticks(range(1, df['round_number'].max() + 1))  # Ensure all round numbers appear
ax.tick_params(axis='both', labelsize=28)
ax.yaxis.set_major_formatter(FuncFormatter(format_y_ticks))

# Legend
handles = [plt.Line2D([0], [0], color=colors[term], linestyle=line_styles[term], marker=marker_style[term], markersize=14, linewidth=3, label=term.capitalize()) for term in sorted_target_terms]
legend = ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.18), fontsize=30, ncol=3, frameon=False, title="Target Terms")
plt.setp(legend.get_title(), fontsize=30)

# Save and show the plot
plt.tight_layout(rect=[0, 0.08, 1, 0.95])  # Adjusted rect to add more space for the legend
plt.savefig("../figures/plot_all-year_breadth_rounds_control.png", dpi=600, bbox_inches='tight')
plt.show()
