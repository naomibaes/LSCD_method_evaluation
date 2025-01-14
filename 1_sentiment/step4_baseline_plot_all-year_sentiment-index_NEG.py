import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# Load the data from the CSV file
try:
    df = pd.read_csv("output/averaged_valence_index_all-year.csv")
    print("Successfully read data from CSV file.")
except FileNotFoundError:
    print("Error: CSV file not found. Please check the file path.")
    raise SystemExit  # Exit script if no file is found

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

# Create a single figure
fig, ax = plt.subplots(figsize=(10, 6))  # Adjust width and height as needed for layout

# Function to format y-axis ticks to 2 decimal places
def format_y_ticks(x, pos):
    return f'{x:.2f}'

# Calculate offset for each term to avoid overlap, based on alphabetical order
offsets = {term: i * 1.8 for i, term in enumerate(target_terms)}  # Adjust 0.08 as needed to reduce or increase spacing

# Loop through each target term and plot them with offsets
for term in target_terms:
    term_data = df[df['target'] == term]
    if not term_data.empty:
        # Apply offset to injection ratio for plotting
        adjusted_ratios = term_data['injection_ratio'] + offsets[term]
        ax.errorbar(adjusted_ratios, term_data['avg_valence_index_negative'],
                    yerr=term_data['se_valence_index_negative'], fmt='o', color=colors[term],
                    label=term.capitalize(), capsize=3, markersize=10)

# Set labels, title, and adjust axes
ax.set_xlabel('Negative Synthetic Sentence Injections (%)', fontsize=16)
ax.set_ylabel('Valence Index (±SE)', fontsize=16)
ax.set_xticks([0, 20, 40, 60, 80, 100])
ax.set_xticklabels(['0%', '20%', '40%', '60%', '80%', '100%'], fontsize=14)

# Apply y-axis tick formatting
ax.yaxis.set_major_formatter(FuncFormatter(format_y_ticks))
ax.tick_params(axis='both', labelsize=12)

# Add grey shading to all areas except the 0% injection ratio
ax.axvspan(10, 100, color='grey', alpha=0.2)  # Shade from just after 0% to the end

# Add a legend below the figure
ax.legend(title="Target Terms", fontsize=12, title_fontsize=14, loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3)

# Save and show the plot
plt.tight_layout()
plt.savefig("../figures/plot_all-year_sentiment_NEG.png", dpi=300, bbox_inches='tight')
plt.show()
