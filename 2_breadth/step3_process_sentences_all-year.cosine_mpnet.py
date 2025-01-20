# Author: Naomi Baes and Chat GPT

from tqdm import tqdm  # For progress visualization
from sentence_transformers import SentenceTransformer
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import codecs  # Import the 'codecs' module for file I/O with specific encodings

# Initialize a Sentence Transformer model
model = SentenceTransformer("all-mpnet-base-v2")  # Transformer model with pooling layer

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

def generate_embeddings(sentences, model, batch_size=32):
    """
    Generate embeddings for a list of sentences in batches to reduce memory usage.
    """
    embeddings = []
    for i in tqdm(range(0, len(sentences), batch_size), desc="Generating embeddings"):
        batch = sentences[i:i + batch_size]
        batch_embeddings = model.encode(batch)
        embeddings.extend(batch_embeddings)
    return np.array(embeddings)

def main(data_folders, model, targets=None):
    # Iterate through each specified data folder
    for data_folder in data_folders:
        print(f"Processing directory: {data_folder}")  # Log the directory being processed
        
        # Check if the path is a directory
        if os.path.isdir(data_folder):
            for file_name in os.listdir(data_folder):
                # Skip files ending with 'cds_mpnet'
                if file_name.endswith('cds_mpnet'):
                    print(f"Skipping file: {file_name} (ends with 'cds_mpnet')")
                    continue

                file_path = os.path.join(data_folder, file_name)
                output_filename = file_path + '_cds_mpnet'
                
                if os.path.isfile(file_path):
                    print(f"Processing file: {file_path}")
                    
                    try:
                        sentences = []
                        with codecs.open(file_path, "r", encoding="utf-8") as f:
                            for line in f:
                                # Take only the first part of the line as the sentence, ignoring other parts
                                sentence = line.strip().split(",")[0]
                                sentences.append(sentence)

                        # Generate embeddings for the sentences in batches
                        embeddings = generate_embeddings(sentences, model)
                        print(f"Number of embeddings generated: {len(embeddings)}")

                        # Calculate dissimilarity scores directly from embeddings
                        dissimilarity_scores = calculate_dissimilarity_scores(embeddings)

                        if dissimilarity_scores is not None:
                            # Save the dissimilarity scores in the same folder as the input file
                            np.savetxt(output_filename, dissimilarity_scores, fmt='%.6f')
                            print(f"Saved the dissimilarity scores to: {output_filename}")

                        else:
                            print(f"Skipping file {file_name} due to error in dissimilarity score calculation")

                    except Exception as e:
                        print(f"Error processing file {file_name}: {e}")

# Specify input directories and targets
data_folders = [
    "output/all-year.cosine"
]
targets = ["abuse", "anxiety", "depression", "mental_health", "mental_illness", "trauma"]  # Specify the target list here (set to None to process all targets)

# Call the main function with the loaded model (from the top of the script)
main(data_folders, model, targets)

print("Processing completed.")
