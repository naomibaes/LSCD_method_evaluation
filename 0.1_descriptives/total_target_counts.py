import os 
import pandas as pd

# Path to the year_counts_sentences.csv file
year_counts_file = "output/year_counts_sentences.csv"

# Define the year range
start_year = 1970
end_year = 2019

# Check if the file exists before proceeding
if os.path.exists(year_counts_file):
    # Print the input file being used
    print(f"Reading data from: {year_counts_file}")
    
    # Load the CSV file into a DataFrame
    df = pd.read_csv(year_counts_file)
    
    # Convert 'year' column to numeric (in case it's not already)
    df['year'] = pd.to_numeric(df['year'], errors='coerce')

    # Group by the 'target' and 'corpus' columns and sum the counts for each target term
    target_counts_total = df.groupby(['target', 'corpus'])['count'].sum().reset_index()

    # Group by 'target' and 'corpus' and filter for the year range 1970-2019, then sum the counts
    df_filtered = df[(df['year'] >= start_year) & (df['year'] <= end_year)]
    target_counts_filtered = df_filtered.groupby(['target', 'corpus'])['count'].sum().reset_index()

    # Get year ranges for each target term in each corpus (entire corpus)
    year_ranges = df.groupby(['target', 'corpus']).agg(min_year=('year', 'min'), max_year=('year', 'max')).reset_index()

    # Print the total occurrences for each target term in the entire corpus
    print("Total sentence counts and year ranges for each target term in each corpus (entire corpus):")
    for index, row in target_counts_total.iterrows():
        target = row['target']
        corpus = row['corpus']
        count = row['count']
        min_year = year_ranges.loc[(year_ranges['target'] == target) & (year_ranges['corpus'] == corpus), 'min_year'].values[0]
        max_year = year_ranges.loc[(year_ranges['target'] == target) & (year_ranges['corpus'] == corpus), 'max_year'].values[0]
        print(f"  - '{target}' in corpus '{corpus}': {count} (Year range: {min_year}-{max_year})")

    # Print the total occurrences for each target term within the year range 1970-2019
    print(f"\nTotal sentence counts for each target term in each corpus (within {start_year}-{end_year}):")
    for index, row in target_counts_filtered.iterrows():
        target = row['target']
        corpus = row['corpus']
        count = row['count']
        print(f"  - '{target}' in corpus '{corpus}': {count}")
else:
    print(f"File {year_counts_file} does not exist.")
