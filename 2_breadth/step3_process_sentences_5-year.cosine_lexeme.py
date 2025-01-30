import os
import sys
import pickle
import re
import numpy as np
from dataclasses import dataclass
from typing import List
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
import codecs
from WordTransformer import WordTransformer, InputExample

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # needed as lexeme_utils.py is in the parent dir
from lexeme_utils import calculate_dissimilarity_scores, targets, File, FileManager

targets = ["trauma", "anxiety", "depression", "mental_health", "mental_illness", "abuse"]  # Specify the target list here (set to None to process all targets)


def main(data_folder, output_folder, targets=None):
    # Iterate through each breadth subdirectory
    print(f"Processing files in directory: {data_folder}")
    file_manager = FileManager(data_folder)
    inputs_to_embeddings = file_manager.get_inputs_to_embeddings(pickle_filename='inputs_to_embeddings_5-year.pkl')
    processed_files = [f for f in os.listdir(output_folder) if "_cds_lexeme" in f]

    for file in file_manager.files:
        file_target = file.target
        file_name = file.file_name
        
        # If specific targets are provided, process only files matching those targets
        if targets and file_target not in targets:
            print(f"Skipping file: {file_name} (target '{file_target}' does not match specified targets {targets})")
            continue

        output_filename = os.path.join(output_folder, f"{file_name}_cds_lexeme")

        # Check if the file has already been processed
        if any(file_name in f for f in processed_files):
            print(f"Skipping file: {file_name} (already processed)")
            continue

        if os.path.isfile(file.path):
            print(f"Processing file: {file.path}")
            inputs = file.inputs

            embeddings = [inputs_to_embeddings[input] for input in inputs]

            # Calculate dissimilarity scores directly from embeddings
            dissimilarity_scores = calculate_dissimilarity_scores(embeddings)

            if dissimilarity_scores is not None:
                # Save the dissimilarity scores
                np.savetxt(output_filename, dissimilarity_scores, fmt='%.6f')
                print(f"Saved the dissimilarity scores to: {output_filename}")

            else:
                print(f"Skipping file {file_name} due to error in dissimilarity score calculation")


# Specify input/output directories and targets
data_folder = "output/5-year.cosine"
output_folder = "output/5-year.cosine"

# Call the main function with the loaded model
main(data_folder, output_folder, targets)

print("Processing completed.")
