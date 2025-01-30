import os
import sys
import numpy as np
from typing import List, Dict

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # needed as lexeme_utils.py is in the parent dir
from lexeme_utils import calculate_dissimilarity_scores, targets, File, FileManager


ITERATION_LEVELS = range(1, 101)
INJECTION_LEVELS = [20, 40, 60, 80, 100]


def get_embeddings(files: List[File], inputs_to_embeddings):
    """ Get embeddings for all inputs in the list of files """
    all_inputs = []
    for f in files:
        all_inputs.extend(f.inputs)
    embeddings = [inputs_to_embeddings[input] for input in all_inputs]
    return embeddings


def main(data_folder):
    print(f"Processing files in directory: {data_folder}")
    file_manager = FileManager(data_folder)
    inputs_to_embeddings = file_manager.get_inputs_to_embeddings(pickle_filename='inputs_to_embeddings_all-year.pkl')

    for target in targets:
        print(f"\nCalculating dissimilarity scores for target: {target}")
        for injection_level in INJECTION_LEVELS:
            dissimilarity_scores_per_injection = []
            for iteration in ITERATION_LEVELS:
                files_no_injection = file_manager.get_files_for(injection_level=0, target=target, iteration=iteration)
                embeddings_no_injection = get_embeddings(files_no_injection, inputs_to_embeddings)

                files_for_bin = file_manager.get_files_for(injection_level, target, iteration)
                embeddings_for_bin = get_embeddings(files_for_bin, inputs_to_embeddings)

                dissimilarity_scores = calculate_dissimilarity_scores(embeddings_no_injection, embeddings_for_bin)
                avg_dissimilarity = np.mean(dissimilarity_scores)

                dissimilarity_scores_per_injection.append(avg_dissimilarity)
            avg_dissimilarity = np.mean(dissimilarity_scores_per_injection)
            print(f"Average dissimilarity for bin {injection_level} with bin 0: {avg_dissimilarity:.6f}")


data_folder = "output/all-year.cosine"

main(data_folder)
