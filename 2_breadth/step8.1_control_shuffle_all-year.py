import os
import pandas as pd

# Define the base directory for input and output
base_input_dir = 'output/all-year.cosine'
base_output_dir = 'output/all-year.cosine/control/input'

# Check and create the output directory if not exists
os.makedirs(base_output_dir, exist_ok=True)

# Define the targets
targets = ['abuse', 'anxiety', 'depression', 'mental_health', 'mental_illness', 'trauma']

# Iterate over each target
for target in targets:
    # Initialize an empty DataFrame to store all sentences
    master_df = pd.DataFrame()

    # Iterate over each file in the input directory
    for filename in os.listdir(base_input_dir):
        # Check if the file matches the target
        if target in filename and filename.endswith(".csv"):
            file_path = os.path.join(base_input_dir, filename)
            
            # Read CSV instead of TSV
            temp_df = pd.read_csv(file_path, sep=',', header=None, dtype=str, names=["sentence", "label"])  
            
            if temp_df.shape[1] >= 2:  # Ensure at least 2 columns exist
                # Append to the master DataFrame
                master_df = pd.concat([master_df, temp_df], ignore_index=True)

    if not master_df.empty:
        # Ensure columns exist before filtering
        if master_df.shape[1] >= 2:
            # Identify 'natural' rows and 'synthetic' rows correctly
            natural_df = master_df[master_df['label'].str.strip().str.lower() == 'natural']
            synthetic_df = master_df[master_df['label'].str.startswith('synthetic_', na=False)]

            # Define the output paths for natural and synthetic
            natural_output_dir = os.path.join(base_output_dir, 'natural')
            synthetic_output_dir = os.path.join(base_output_dir, 'synthetic')

            # Ensure directories exist
            os.makedirs(natural_output_dir, exist_ok=True)
            os.makedirs(synthetic_output_dir, exist_ok=True)

            # Define output file paths
            natural_output_path = os.path.join(natural_output_dir, f"{target}_all-year_breadth_natural.csv")
            synthetic_output_path = os.path.join(synthetic_output_dir, f"{target}_all-year_breadth_synthetic.csv")

            # Write the DataFrames to files
            natural_df.to_csv(natural_output_path, sep=',', index=False, header=False)
            synthetic_df.to_csv(synthetic_output_path, sep=',', index=False, header=False)

print("Files have been successfully grouped, split by natural and synthetic, and saved.")
