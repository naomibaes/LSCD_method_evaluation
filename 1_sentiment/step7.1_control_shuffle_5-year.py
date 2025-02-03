import os
import pandas as pd

# Define the base directory for input and output
base_input_dir = 'output/5-year'
base_output_dir = 'output/5-year/control'

# Check and create the output directories if not exists
os.makedirs(base_output_dir, exist_ok=True)

# Define the epochs to be processed
epochs = [
    '1970-1974', '1975-1979', '1980-1984', '1985-1989',
    '1990-1994', '1995-1999', '2000-2004', '2005-2009',
    '2010-2014', '2015-2019'
]

# Define the target conditions and their corresponding folders
conditions = {
    'positive': 'positive',  # Files in the 'positive' folder
    'negative': 'negative'   # Files in the 'negative' folder
}

# Define the targets
targets = ['abuse', 'anxiety', 'depression', 'mental_health', 'mental_illness', 'trauma']

# Iterate over each target, condition, and epoch
for target in targets:
    for condition, folder in conditions.items():
        for epoch in epochs:
            # Initialize an empty DataFrame to store all sentences
            master_df = pd.DataFrame()

            # Define the input directory for the condition
            input_dir = os.path.join(base_input_dir, folder)

            # Iterate over each file in the input directory
            for filename in os.listdir(input_dir):
                # Check if the file matches the target and epoch
                if target in filename and epoch in filename:
                    # Process files based on the condition (folder name)
                    if condition == 'negative' and 'lemmatized.tsv' not in filename:
                        file_path = os.path.join(input_dir, filename)
                        # Read the file
                        temp_df = pd.read_csv(file_path, sep='\t', header=None, dtype=str)  # Read as string to avoid issues
                        if temp_df.shape[1] >= 3:  # Ensure at least 3 columns exist
                            # Append to the master DataFrame
                            master_df = pd.concat([master_df, temp_df], ignore_index=True)
                    elif condition == 'positive' and not filename.endswith('lemmatized.tsv'):
                        file_path = os.path.join(input_dir, filename)
                        # Read the file
                        temp_df = pd.read_csv(file_path, sep='\t', header=None, dtype=str)  # Read as string to avoid issues
                        if temp_df.shape[1] >= 3:  # Ensure at least 3 columns exist
                            # Append to the master DataFrame
                            master_df = pd.concat([master_df, temp_df], ignore_index=True)

            if not master_df.empty:
                # Ensure columns exist before filtering
                if master_df.shape[1] >= 3:
                    # Split the data based on 'natural' or 'synthetic' in the third column
                    natural_df = master_df[master_df.iloc[:, 2].str.contains(r'^\s*natural\s*$', case=False, regex=True, na=False)]
                    synthetic_df = master_df[master_df.iloc[:, 2].str.contains(r'^\s*synthetic.*$', case=False, regex=True, na=False)]

                    # Define the output paths for natural and synthetic
                    natural_output_dir = os.path.join(base_output_dir, condition, 'natural')
                    synthetic_output_dir = os.path.join(base_output_dir, condition, 'synthetic')

                    # Ensure directories exist
                    os.makedirs(natural_output_dir, exist_ok=True)
                    os.makedirs(synthetic_output_dir, exist_ok=True)

                    # Define output file paths
                    natural_output_path = os.path.join(natural_output_dir, f"{target}_{epoch}_5-year_sentiment_natural.tsv")
                    synthetic_output_path = os.path.join(synthetic_output_dir, f"{target}_{epoch}_5-year_sentiment_synthetic.tsv")

                    # Write the DataFrames to files
                    natural_df.to_csv(natural_output_path, sep='\t', index=False, header=False)
                    synthetic_df.to_csv(synthetic_output_path, sep='\t', index=False, header=False)

print("Files have been successfully grouped, split by natural and synthetic, and saved.")