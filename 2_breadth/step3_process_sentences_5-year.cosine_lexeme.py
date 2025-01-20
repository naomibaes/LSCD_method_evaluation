import os
import re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import codecs
from WordTransformer import WordTransformer, InputExample

# Initialize the WordTransformer model
model = WordTransformer('pierluigic/xl-lexeme')  # Load the XL-LEXEME model

def calculate_dissimilarity_scores(embeddings):
    similarity_matrix = cosine_similarity(embeddings)
    dissimilarity_scores = 1 - similarity_matrix
    
    # Get the shape of the dissimilarity matrix
    n = dissimilarity_scores.shape[0]
    
    # Create a mask to select the upper triangular part (excluding diagonal)
    upper_triangular_mask = np.triu(np.ones((n, n)), k=1)
    
    # Apply the mask to get the upper triangular part of the dissimilarity matrix
    upper_triangular_dissimilarity = dissimilarity_scores[upper_triangular_mask == 1]
    
    return upper_triangular_dissimilarity

def find_target_position(sentence, target_word):
    """Find the start and end position of the target word in the sentence."""
    match = re.search(rf'\b{re.escape(target_word)}\b', sentence)
    if match:
        start = match.start()
        end = match.end()
        return (start, end)
    else:
        return None

def main(data_folder, output_folder, model, targets=None):
    # Iterate through each breadth subdirectory
    for breadth_dir in os.listdir(data_folder):
        breadth_path = os.path.join(data_folder, breadth_dir)

        if os.path.isdir(breadth_path):
            print(f"Processing directory: {breadth_path}")  # Log the directory being processed
            
            # Create an output directory for the current breadth directory
            output_breadth_path = os.path.join(output_folder, breadth_dir)
            os.makedirs(output_breadth_path, exist_ok=True)  # Create the directory if it doesn't exist
            
            for file_name in os.listdir(breadth_path):
                file_target = file_name.split('.')[0]
                
                # If specific targets are provided, process only files matching those targets
                if targets and file_target not in targets:
                    print(f"Skipping file: {file_name} (target '{file_target}' does not match specified targets {targets})")
                    continue

                file_path = os.path.join(breadth_path, file_name)
                output_filename = os.path.join(output_breadth_path, f"{file_name}_cds_lexeme")

                # Get list of all processed files in the output folder
                processed_files = [f for f in os.listdir(output_breadth_path) if "_cds_lexeme" in f]

                # Check if the file has already been processed
                if any(file_name in f for f in processed_files):
                    print(f"Skipping file: {file_name} (already processed)")
                    continue

                if os.path.isfile(file_path):
                    print(f"Processing file: {file_path}")
                    
                    try:
                        sentences = []
                        with codecs.open(file_path, "r", encoding="utf-8") as f:
                            for line in f:
                                parts = line.strip().split("\t")
                                if len(parts) == 2:  # Two-column file (sentence and year/metadata)
                                    sentence = parts[0].strip()  # Extract the sentence
                                    sentences.append(sentence)
                                elif len(parts) == 1:  # One-column file (only sentence)
                                    sentence = parts[0].strip()
                                    sentences.append(sentence)
                                else:
                                    print(f"Skipping malformed line in {file_name}: {line}")

                        embeddings = []

                        # For each sentence, find the position of the target word and generate embeddings
                        for sentence in sentences:
                            target_position = find_target_position(sentence, file_target)
                            
                            if target_position:
                                example = InputExample(texts=sentence, positions=[target_position[0], target_position[1]])
                                embedding = model.encode(example)
                                embeddings.append(embedding)
                            else:
                                print(f"Target word '{file_target}' not found in sentence: {sentence}")

                        print(f"Number of embeddings generated: {len(embeddings)}")

                        # Calculate dissimilarity scores directly from embeddings
                        dissimilarity_scores = calculate_dissimilarity_scores(embeddings)

                        if dissimilarity_scores is not None:
                            # Save the dissimilarity scores
                            np.savetxt(output_filename, dissimilarity_scores, fmt='%.6f')
                            print(f"Saved the dissimilarity scores to: {output_filename}")

                        else:
                            print(f"Skipping file {file_name} due to error in dissimilarity score calculation")

                    except Exception as e:
                        print(f"Error processing file {file_name}: {e}")

# Specify input/output directories and targets
data_folder = "output/5-year.cosine"
output_folder = "output/5-year.cosine"
targets = ["trauma", "anxiety", "depression", "mental_health", "mental_illness", "abuse"]  # Specify the target list here (set to None to process all targets)

# Call the main function with the loaded model
main(data_folder, output_folder, model, targets)

print("Processing completed.")
