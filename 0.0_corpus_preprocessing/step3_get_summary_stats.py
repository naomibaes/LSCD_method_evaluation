import csv
from collections import defaultdict

# Path to the file you want to read
input_file = 'C:/Users/naomi/OneDrive/COMP80004_PhDResearch/RESEARCH/DATA/CORPORA/Psychology/abstract_year_journal.csv.mental.sentence-CLEAN.tsv'
output_file_path = 'output/sentence_stats.txt'   

def read_first_100_lines_and_get_statistics(file_path, output_path):
    min_year = float('inf')
    max_year = float('-inf')
    total_lines = 0
    journal_counts = defaultdict(int)  # Dictionary to count occurrences of each journal

    # Variables to track stats for 1970-2016
    min_year_1970_2016 = float('inf')
    max_year_1970_2016 = float('-inf')
    total_lines_1970_2016 = 0

    # Open the file and process the rows using csv.reader
    with open(file_path, 'r', encoding='utf-8') as file:
        reader = csv.reader(file, delimiter='\t')  # Use tab as the delimiter

        # Read and print the header
        header = next(reader)  # Get the header row
        print(f"Header: {header}")

        print("\nFirst 100 lines of the file (printed as they appear):\n")

        for index, row in enumerate(reader):
            if len(row) == 2:
                # If the line only has 2 fields, assume it's a file without the journal field
                sentence, year = row[0], row[1]
                journal_title = "No Journal"  # Set default value for journal
            elif len(row) == 3:
                # If the line has 3 fields, treat it as sentence, year, and journal
                sentence, year, journal_title = row[0], row[1], row[2]
                journal_counts[journal_title] += 1  # Increment the journal count
            else:
                print(f"Skipping malformed line {index + 1}: {row}")
                continue

            try:
                year = int(year)  # Try to convert the year to an integer
            except ValueError:
                print(f"Skipping line {index + 1}: invalid year '{year}'")  # Debugging line
                continue  # If year is not a valid number, skip this line

            # Track min and max year
            min_year = min(min_year, year)
            max_year = max(max_year, year)

            # Check if the year falls within 1970-2016 range
            if 1970 <= year <= 2016:
                min_year_1970_2016 = min(min_year_1970_2016, year)
                max_year_1970_2016 = max(max_year_1970_2016, year)
                total_lines_1970_2016 += 1

            # Increment total line count
            total_lines += 1

            # Print the first 100 lines exactly as they appear in the corpus
            if index < 100:
                print("\t".join(row))  # Print the row as it appears with tabs

    # Calculate percentage of each journal in the corpus
    journal_percentages = {journal: (count / total_lines) * 100 for journal, count in journal_counts.items()}

    # Write statistics and journal percentages to the output file
    with open(output_path, 'w', encoding='utf-8') as out_file:
        out_file.write(f"Total lines: {total_lines}\n")
        out_file.write(f"Minimum year: {min_year if min_year != float('inf') else 'N/A'}\n")
        out_file.write(f"Maximum year: {max_year if max_year != float('-inf') else 'N/A'}\n")
        out_file.write(f"\nStatistics for data between 1970 and 2016:\n")
        out_file.write(f"Total lines: {total_lines_1970_2016}\n")
        out_file.write(f"Minimum year (1970-2016): {min_year_1970_2016 if min_year_1970_2016 != float('inf') else 'N/A'}\n")
        out_file.write(f"Maximum year (1970-2016): {max_year_1970_2016 if max_year_1970_2016 != float('-inf') else 'N/A'}\n")

        # Write journal percentages
        out_file.write("\nJournal Percentages:\n")
        for journal, percentage in sorted(journal_percentages.items(), key=lambda x: x[1], reverse=True):
            out_file.write(f"{journal}: {percentage:.2f}%\n")

    print(f"\nStatistics and journal percentages written to {output_path}")

# Call the function to process the file and output stats
read_first_100_lines_and_get_statistics(input_file, output_file_path)