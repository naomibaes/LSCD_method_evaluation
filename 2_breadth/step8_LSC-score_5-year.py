import os
import pickle
import re
import numpy as np
from dataclasses import dataclass
from typing import List
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
import codecs
from WordTransformer import WordTransformer, InputExample

targets = ["trauma", "anxiety", "depression", "mental_health", "mental_illness", "abuse"]  # Specify the target list here (set to None to process all targets)

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


@dataclass(frozen=True)
class Input:
    text: str
    target: str

    @property
    def target_position(self):
        """Find the start and end position of the target word in the sentence."""
        match = re.search(rf'\b{re.escape(self.target)}\b', self.text.lower())
        if match:
            start = match.start()
            end = match.end()
            return (start, end)
        else:
            raise ValueError(f"Target word '{self.target}' not found in sentence: {self.text}")
    
    def as_input_example(self) -> InputExample:
        if self.target_position:
            return InputExample(texts=self.text, positions=[self.target_position[0], self.target_position[1]])
        else:
            return None


@dataclass
class File:
    path: str

    def __post_init__(self):
        try:
            self.target = next(t for t in targets if t in os.path.basename(self.path))
        except StopIteration:
            raise ValueError(f"Target not found in file name: {self.path}")
    
    @property
    def sentences(self) -> List[str]:
        with open(self.path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            return [line.strip().split("\t")[0] for line in lines]
    
    @property
    def inputs(self) -> List[Input]:
        return [Input(text=sentence, target=self.target) for sentence in self.sentences]


def get_all_inputs(data_folder) -> List[Input]:
    """Get all unique inputs from the data folder.
    
    Around 45k unique inputs are expected from 7,200 files.
    """
    inputs = set()
    file_names = os.listdir(data_folder)
    file_names = [f for f in file_names if '_cds_' not in f]
    for file_name in file_names:
        file_path = os.path.join(data_folder, file_name)
        if os.path.isfile(file_path):
            for input in File(file_path).inputs:
                inputs.add(input)
    print(f"Number of unique inputs: {len(inputs)}")
    return list(inputs)


def generate_embeddings(inputs: List[Input], model, batch_size=32):
    embeddings = []
    for i in tqdm(range(0, len(inputs), batch_size), desc="Generating embeddings", leave=False):
        batch = inputs[i:i + batch_size]
        batch_embeddings = model.encode([input.as_input_example() for input in batch])
        embeddings.extend(batch_embeddings)
    return np.array(embeddings)


def get_inputs_to_embeddings(data_folder, model, pickle_filename = 'inputs_to_embeddings.pkl'):
    """ Generate embeddings for all inputs in the data folder.
    If embeddings have been generated before, load them from the pickle file.
    """
    if os.path.exists(pickle_filename):
        print(f"Loading embeddings from file: {pickle_filename}")
        with open(pickle_filename, 'rb') as f:
            inputs_to_embeddings = pickle.load(f)
            return inputs_to_embeddings
    all_inputs = get_all_inputs(data_folder)
    inputs_to_embeddings = dict(zip(all_inputs, generate_embeddings(all_inputs, model)))
    with open(pickle_filename, 'wb') as f:
        pickle.dump(inputs_to_embeddings, f)
    return inputs_to_embeddings


def main(data_folder, output_folder, model, targets=None):
    # Iterate through each breadth subdirectory
    print(f"Processing files in directory: {data_folder}")
    inputs_to_embeddings = get_inputs_to_embeddings(data_folder, model)
    processed_files = [f for f in os.listdir(output_folder) if "_cds_lexeme" in f]

    for file_name in os.listdir(data_folder):
        if '_cds_' in file_name:
            continue
        file_path = os.path.join(data_folder, file_name)
        file = File(file_path)
        file_target = file.target
        
        # If specific targets are provided, process only files matching those targets
        if targets and file_target not in targets:
            print(f"Skipping file: {file_name} (target '{file_target}' does not match specified targets {targets})")
            continue

        output_filename = os.path.join(output_folder, f"{file_name}_cds_lexeme")

        # Check if the file has already been processed
        if any(file_name in f for f in processed_files):
            print(f"Skipping file: {file_name} (already processed)")
            continue

        if os.path.isfile(file_path):
            print(f"Processing file: {file_path}")
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
main(data_folder, output_folder, model, targets)

print("Processing completed.")
