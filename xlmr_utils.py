import os
import pickle
import re
import csv
import numpy as np
import logging
from dotenv import load_dotenv
load_dotenv()
import torch
from transformers import XLMRobertaTokenizer, XLMRobertaModel, DebertaV2Model, AutoTokenizer
from functools import cached_property
from dataclasses import dataclass
from typing import List, Dict, Tuple
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity

targets = ["trauma", "anxiety", "depression", "abuse"]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def calculate_dissimilarity_scores(embeddings1, embeddings2=None) -> np.ndarray:
    """ Return the dissimilarity scores between all possible pairs of embeddings. """
    if embeddings2 is None:
        # Comparing within one matrix
        similarity_matrix = cosine_similarity(embeddings1)
        dissimilarity_scores = 1 - similarity_matrix

        # Create a mask to select the upper triangular part (excluding diagonal)
        n = dissimilarity_scores.shape[0]
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


def save_to_csv(results: List[Dict], filename: str):
    with open(filename, 'w') as csvfile:
        fieldnames = results[0].keys()
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(result)
    print(f"Saved {len(results)} results to {filename}")

@dataclass(frozen=True)
class Input:
    text: str
    target: str


@dataclass
class File:
    path: str

    @cached_property
    def file_name(self) -> str:
        return os.path.basename(self.path)

    @cached_property
    def is_5_year_file(self) -> bool:
        # year would be of the form 1980-1984, check if there is any using regex
        return bool(re.search(r'\d{4}-\d{4}', self.file_name))

    @cached_property
    def injection_level(self) -> int:
        """Extract the injection level from the file name.

        e.g. abuse_1980-1984_synthetic_20_1 -> 20
        e.g. abuse_synthetic_breadth_0_46.csv -> 0
        """
        if self.is_5_year_file:
            return [int(x) for x in re.findall(r'_(\d+)', self.file_name)][1]
        else:
            return [int(x) for x in re.findall(r'_(\d+)', self.file_name)][0]
    
    @cached_property
    def iteration(self) -> int:
        """ Extract the iteration number from the file name.

        e.g. abuse_1980-1984_synthetic_20_1 -> 1
        e.g. abuse_synthetic_breadth_0_46.csv -> 46
        """
        return [int(x) for x in re.findall(r'[_\.](\d+)', self.file_name)][-1]

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
            sentences = [line[0] for line in reader]
        # replace upper-case target words with lower-case
        sentences = [re.sub(self.target.title(), self.target, sentence) for sentence in sentences]
        sentences = [re.sub(self.target[:2].upper() + self.target[2:], self.target, sentence) for sentence in sentences]
        sentences = [re.sub(self.target.upper(), self.target, sentence) for sentence in sentences]
        # replace "(target" with "( target"
        sentences = [re.sub(r'([^\w\s])' + self.target, r'\1 ' + self.target, sentence) for sentence in sentences]
        # replace "traumatic" with "trauma tic"
        sentences = [sentence.replace("traumatic", "trauma tic") for sentence in sentences]
        sentences = [sentence.replace("abused", "abuse d") for sentence in sentences]
        sentences = [sentence.replace("abuses", "abuse s") for sentence in sentences]
        sentences = [sentence.replace("abuser", "abuse r") for sentence in sentences]
        return sentences
    
    @property
    def inputs(self) -> List[Input]:
        return [Input(text=sentence, target=self.target) for sentence in self.sentences]


class FileManager:
    def __init__(self, data_folder):
        self.data_folder = data_folder
        self.files: List[File] = self.get_files()
        # self.model = XLMRobertaModel.from_pretrained('xlm-roberta-base')
        self.model = DebertaV2Model.from_pretrained("microsoft/deberta-v3-base")
        self.model.to(device)
        self.model.eval()
        # self.tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")
    
    def get_files(self):
        files = []
        for file_name in os.listdir(self.data_folder):
            if '_cds_' in file_name or 'warnings.log' in file_name or '_lemmatized' in file_name:
                continue
            file_path = os.path.join(self.data_folder, file_name)
            if os.path.isfile(file_path):
                try:
                    files.append(File(file_path))
                except ValueError:
                    continue
        return files
    
    @cached_property
    def target_word_to_token(self):
        return {target: self.tokenizer.tokenize(target)[0] for target in targets}
    
    def check_inputs_have_target(self, inputs: List[Input]):
        """ Check that the tokenized target word is present in the tokenized input text. """
        tokenized_inputs = self.tokenizer([input.text for input in inputs], return_tensors="pt", padding=True, truncation=True)
        for input, tokenized_input in zip(inputs, tokenized_inputs):
            tokens = self.tokenizer.convert_ids_to_tokens(tokenized_input['input_ids'])
            input_target_token = self.target_word_to_token[input.target]
            try:
                target_position = tokens.index(input_target_token)
            except ValueError:
                raise ValueError(f"Target word {input.target} not found in input: {input.text}")
            if input_target_token not in tokens:
                raise ValueError(f"Target word {input.target} not found in input: {input.text}")
        print("All inputs have target word present.")

    
    def generate_embeddings(self, inputs: List[Input], batch_size=16) -> List[np.ndarray]:
        embeddings = []
        for i in tqdm(range(0, len(inputs), batch_size), desc="Generating embeddings", leave=False):
            batch = inputs[i:i + batch_size]
            batch_sents = [i.text for i in batch]
            tokenized_batch = self.tokenizer(batch_sents, return_tensors="pt", padding=True, truncation=True)
            tokenized_batch.to(device)
            with torch.no_grad():
                outputs = self.model(**tokenized_batch)  # shape: (batch_size, seq_len, hidden_size)
            for j, input in enumerate(batch):
                hidden_states = outputs.last_hidden_state[j]  # shape: (seq_len, hidden_size)
                tokens = self.tokenizer.convert_ids_to_tokens(tokenized_batch['input_ids'][j])
                try:
                    target_position = tokens.index(self.target_word_to_token[input.target])
                except:
                    raise ValueError(f"Target word {input.target} not found in input: {input.text}")
                hidden_state_for_target = hidden_states[target_position]  # shape: (hidden_size,)
                embeddings.append(hidden_state_for_target.cpu().numpy())
        return embeddings
    
    def get_inputs(self) -> List[Input]:
        return [input for f in self.files for input in f.inputs]

    def get_inputs_to_embeddings(self, pickle_filename = 'xlmr_sentences_to_embeddings_5-year.pkl') -> Dict[Input, np.ndarray]:
        """ Generate embeddings for all sentences in the data folder.
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
        all_inputs = list(set(all_inputs))
        inputs_to_embeddings = dict(zip(all_inputs, self.generate_embeddings(all_inputs)))
        with open(pickle_filename, 'wb') as f:
            pickle.dump(inputs_to_embeddings, f)
        return inputs_to_embeddings

    def get_files_for(self, injection_level, target, iteration=None, year=None) -> List[File]:
        files = [f for f in self.files if f.injection_level == injection_level and f.target == target]
        if iteration is not None:
            files = [f for f in files if f.iteration == iteration]
        if year is not None:
            files = [f for f in files if year in f.file_name]
        return files