import csv

# Step 1: Read the file and filter out rows containing '|||||' anywhere in the line and empty lines
def clean_corpus(input_file_path):
    cleaned_corpus = []
    removed_lines = []  # List to track removed lines
    
    with open(input_file_path, 'r', encoding='utf-8') as f:
        corpus = f.readlines()  # Read all lines
        
        first_line = True
        for line_num, line in enumerate(corpus, start=1):
            # Skip the first line (old header)
            if first_line:
                first_line = False
                continue
            
            # Filter out rows that contain '|||||' anywhere in the line and empty lines
            if '|||||' not in line and line.strip():
                # Expecting lines to be tab-separated with three fields: sentence, publication_year, journal_title
                parts = [part.strip() for part in line.strip().split('\t')]  # Strip whitespace around each part
                
                # Ensure there are exactly three parts (sentence, publication_year, journal_title)
                if len(parts) == 3:
                    # Try to convert publication_year to integer to ensure consistency
                    try:
                        parts[1] = int(parts[1])  # Convert year to integer
                        cleaned_corpus.append(parts)  # Add the split line as a tuple/list
                    except ValueError:
                        removed_lines.append(f"Line {line_num}: Invalid year '{parts[1]}'")  # Track lines with invalid year
                else:
                    removed_lines.append(f"Line {line_num}: {line.strip()}")  # Debugging for malformed lines
            else:
                removed_lines.append(f"Line {line_num}: {line.strip()}")  # Track removed lines
    
    print(f"Total lines removed: {len(removed_lines)}")
    return cleaned_corpus, removed_lines

# Step 2a: Write the cleaned corpus to an output file, including the journal_title
def write_with_journal_title(output_file_path, cleaned_corpus):
    with open(output_file_path, 'w', encoding='utf-8') as f:
        # Write the new header
        f.write("sentence\tpublication_year\tjournal_title\n")
        
        # Write the rest of the cleaned corpus
        for entry in cleaned_corpus:
            # Ensure integer formatting for publication_year and strip whitespace
            f.write(f"{entry[0].strip()}\t{entry[1]}\t{entry[2].strip()}\n")

# Step 2b: Write the cleaned corpus to an output file, excluding the journal_title
def write_without_journal_title(output_file_path, cleaned_corpus):
    with open(output_file_path, 'w', encoding='utf-8') as f:
        # Write the new header (without journal_title)
        f.write("sentence\tpublication_year\n")
        
        # Write the rest of the cleaned corpus (only sentence and year)
        for entry in cleaned_corpus:
            # Add an extra check to ensure that there are exactly two parts before writing
            if len(entry) >= 2:
                f.write(f"{entry[0].strip()}\t{entry[1]}\n")
            else:
                print(f"Skipping entry with missing fields: {entry}")  # Debugging for incomplete entries

# Step 2c: Write the removed lines to a separate file
def write_removed_lines(output_file_path, removed_lines):
    with open(output_file_path, 'w', encoding='utf-8') as f:
        f.write("Lines removed during cleaning:\n")
        for line in removed_lines:
            f.write(f"{line}\n")

# Input file path
input_file_path = "C:/Users/naomi/OneDrive/COMP80004_PhDResearch/RESEARCH/DATA/CORPORA/Psychology/abstract_year_journal.csv.mental.sentence.tsv"

# Output file paths (with and without journal_title, and for removed lines)
output_file_with_journal = "C:/Users/naomi/OneDrive/COMP80004_PhDResearch/RESEARCH/DATA/CORPORA/Psychology/abstract_year_journal.csv.mental.sentence-CLEAN-journals.tsv"
output_file_without_journal = "C:/Users/naomi/OneDrive/COMP80004_PhDResearch/RESEARCH/DATA/CORPORA/Psychology/abstract_year_journal.csv.mental.sentence-CLEAN.tsv"
removed_lines_file = "output/removed_lines_log.txt"

# Step 3: Clean the corpus by removing rows that contain '|||||' and skipping the old header
cleaned_corpus, removed_lines = clean_corpus(input_file_path)

# Step 4a: Write the cleaned data to a file with the journal_title included
write_with_journal_title(output_file_with_journal, cleaned_corpus)

# Step 4b: Write the cleaned data to a file without the journal_title
write_without_journal_title(output_file_without_journal, cleaned_corpus)

# Step 4c: Write the removed lines to a separate file
write_removed_lines(removed_lines_file, removed_lines)

# Step 5: Notify user of completion
print(f"Cleaned corpus with journal_title written to {output_file_with_journal}")
print(f"Cleaned corpus without journal_title written to {output_file_without_journal}")
print(f"Removed lines written to {removed_lines_file}")
