import os
import pickle
import re
import csv
import numpy as np
import logging
from functools import cached_property
from dataclasses import dataclass
from typing import List
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from WordTransformer import WordTransformer, InputExample

targets = ["trauma", "anxiety", "depression", "mental_health", "mental_illness", "abuse"]  # Specify the target list here (set to None to process all targets)


def calculate_dissimilarity_scores(embeddings1, embeddings2=None) -> np.ndarray:
    if embeddings2 is None:
        # Original behavior - comparing within one matrix
        similarity_matrix = cosine_similarity(embeddings1)
        dissimilarity_scores = 1 - similarity_matrix
        
        # Get the shape of the dissimilarity matrix
        n = dissimilarity_scores.shape[0]
        
        # Create a mask to select the upper triangular part (excluding diagonal)
        upper_triangular_mask = np.triu(np.ones((n, n)), k=1)
        
        # Apply the mask to get the upper triangular part of the dissimilarity matrix
        upper_triangular_dissimilarity = dissimilarity_scores[upper_triangular_mask == 1]
        
        return upper_triangular_dissimilarity
    else:
        # Compare between two different matrices
        similarity_matrix = cosine_similarity(embeddings1, embeddings2)
        dissimilarity_scores = 1 - similarity_matrix
        # Return full matrix since there's no redundancy when comparing different sets
        return dissimilarity_scores.flatten()


@dataclass(frozen=True)
class Input:
    text: str
    target: str

    def __post_init__(self):
        if self.target not in self.text.lower():
            raise ValueError(f"Target word '{self.target}' not found in sentence: {self.text}")

    @property
    def target_position(self):
        """Find the start and end position of the target word in the sentence."""
        match = re.search(rf'\b{re.escape(self.target)}', self.text.lower())
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

    @cached_property
    def file_name(self) -> str:
        return os.path.basename(self.path)
    
    @cached_property
    def injection_level(self) -> int:
        """Extract the injection level from the file name.

        e.g. abuse_1980-1984_synthetic_20_1 -> 20
        e.g. abuse_synthetic_breadth_0_46.csv -> 0
        """
        return [int(x) for x in re.findall(r'_(\d+)', self.file_name)][0]
    
    @cached_property
    def iteration(self) -> int:
        """ Extract the iteration number from the file name.

        e.g. abuse_1980-1984_synthetic_20_1 -> 1
        e.g. abuse_synthetic_breadth_0_46.csv -> 46
        """
        try:
            # pattern for the 2_breadth files
            return [int(x) for x in re.findall(r'_(\d+)', self.file_name)][1]
        except IndexError:
            # pattern for the 1_sentiment files
            return [int(x) for x in re.findall(r'\.(\d+)', self.file_name)][0]

    def __post_init__(self):
        try:
            self.target = next(t for t in targets if t in self.file_name)
        except StopIteration:
            raise ValueError(f"Target not found in file name: {self.path}")
    
    @property
    def sentences(self) -> List[str]:
        with open(self.path, "r", encoding="utf-8") as f:
            # type is csv if path ends with .csv, otherwise assume tsv
            reader = csv.reader(f, delimiter="," if self.path.endswith(".csv") else "\t")
            return [line[0] for line in reader]
    
    @property
    def inputs(self) -> List[Input]:
        inputs = []
        for sentence in self.sentences:
            try:
                inputs.append(Input(text=sentence, target=self.target))
            except ValueError:
                logging.warning(f"Target word '{self.target}' not found in sentence: {sentence} -- ignoring")
                continue
        return inputs


class FileManager:
    def __init__(self, data_folder):
        self.data_folder = data_folder
        self.files = self.get_files()
    
    @cached_property
    def model(self):
        return WordTransformer('pierluigic/xl-lexeme')

    def get_files(self):
        files = []
        for file_name in os.listdir(self.data_folder):
            if '_cds_' in file_name or 'warnings.log' in file_name or '_lemmatized' in file_name:
                continue
            file_path = os.path.join(self.data_folder, file_name)
            if os.path.isfile(file_path):
                files.append(File(file_path))
        return files
    
    def get_inputs(self) -> List[Input]:
        """Get all unique inputs from the data folder.
        
        Around 45k unique inputs are expected from 7,200 files.
        """
        inputs = set()
        for file in self.files:
            for input in file.inputs:
                inputs.add(input)
        print(f"Number of unique inputs: {len(inputs)}")
        return list(inputs)

    def generate_embeddings(self, inputs: List[Input], batch_size=32):
        embeddings = []
        for i in tqdm(range(0, len(inputs), batch_size), desc="Generating embeddings", leave=False):
            batch = inputs[i:i + batch_size]
            batch_embeddings = self.model.encode([input.as_input_example() for input in batch])
            embeddings.extend(batch_embeddings)
        return np.array(embeddings)


    def get_inputs_to_embeddings(self, pickle_filename = 'inputs_to_embeddings_5-year.pkl'):
        """ Generate embeddings for all inputs in the data folder.
        If embeddings have been generated before, load them from the pickle file.
        """
        try:
            if os.path.exists(pickle_filename):
                print(f"Loading embeddings from file: {pickle_filename}")
                with open(pickle_filename, 'rb') as f:
                    inputs_to_embeddings = pickle.load(f)
                    return inputs_to_embeddings
        except (AttributeError, pickle.UnpicklingError):
            print(f"Error loading embeddings from file: {pickle_filename}; regenerating embeddings...")
        all_inputs = self.get_inputs()
        inputs_to_embeddings = dict(zip(all_inputs, self.generate_embeddings(all_inputs)))
        with open(pickle_filename, 'wb') as f:
            pickle.dump(inputs_to_embeddings, f)
        return inputs_to_embeddings

    def get_files_for(self, injection_level, target, iteration=None) -> List[File]:
        files = [f for f in self.files if f.injection_level == injection_level and f.target == target]
        if iteration is not None:
            files = [f for f in files if f.iteration == iteration]
        return files