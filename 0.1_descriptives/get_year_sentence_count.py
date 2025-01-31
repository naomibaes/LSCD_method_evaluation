# Setup dependencies
import matplotlib.pyplot as plt
import pandas as pd

# Specify the path to the input file
input_file = "C:/Users/naomi/OneDrive/COMP80004_PhDResearch/RESEARCH/DATA/CORPORA/Psychology/abstract_year_journal.csv.mental.sentence-CLEAN.tsv"

# Read the TSV file into a DataFrame
# Manually specify column names if they are not present in the file
column_names = ['sentence', 'year', 'corpus']
df = pd.read_csv(input_file, sep='\t', header=None, names=column_names)

# Set the style to mimic ggplot2 classic
plt.style.use('ggplot')

# Define the year range and padding
start_year = 1970
end_year = 2019
padding = 0.05  # Adjust this value for the desired amount of padding

# Calculate the adjusted x-axis limits with padding
x_min = start_year - (end_year - start_year) * padding
x_max = end_year + (end_year - start_year) * padding

# Group data by year and count the number of sentences per year
annual_counts = df.groupby('year').size().reset_index(name='count')

# Convert 'year' column to numeric (to avoid any non-numeric issues)
annual_counts['year'] = pd.to_numeric(annual_counts['year'], errors='coerce')

# Drop rows with NaN years (if any)
annual_counts = annual_counts.dropna(subset=['year'])

# Convert year to integer type for better control of x-axis labels
annual_counts['year'] = annual_counts['year'].astype(int)

# Combine duplicate years by summing their counts
annual_counts = annual_counts.groupby('year', as_index=False).sum()

# Debug: Check if duplicates are resolved
print("Post-aggregation check for duplicates:")
duplicates = annual_counts[annual_counts.duplicated(subset='year', keep=False)]
if not duplicates.empty:
    print("Duplicate year entries found:")
    print(duplicates)
else:
    print("No duplicate year entries found.")

# Create the plot
fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)

# Plot bar plot of count distribution
ax.bar(annual_counts['year'], annual_counts['count'], color='darkgrey', edgecolor='black')
ax.set_title("Annual Sentence Counts")
ax.set_xlabel('Year')
ax.set_ylabel('Sentence Count')
ax.grid(False)  # Turn off grid

# Set x-axis ticks and labels explicitly as integers
years = sorted(annual_counts['year'].unique())
ax.set_xticks(years)
ax.set_xticklabels([str(year) for year in years], rotation=45)  # Format labels as strings without decimals

# Add horizontal line at a specified count, e.g., 500
ax.axhline(y=500, color='red', linestyle='--', linewidth=2)  # Customize the horizontal line

# Set adjusted x-axis limits with padding
ax.set_xlim(x_min, x_max)

# Show the plot
plt.show()
