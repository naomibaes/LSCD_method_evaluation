# Author: Raphael Merx
# Aim: Compute LSC score for all-year and 5-year files (bootstrap and stratified random sampling stratigies)

import os
import sys
import argparse
import numpy as np
from typing import List, Dict

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # needed as lexeme_utils.py is in the parent dir
from xlmr_utils import calculate_dissimilarity_scores, save_to_csv, targets, File, FileManager


parser = argparse.ArgumentParser(description="Calculate dissimilarity scores for all-year data")
parser.add_argument("--year-mode", type=str, choices=["all-year", "5-year"], help="Year mode to use")
parser.add_argument("--variation", type=str, choices=["high", "low"], help="Variation to calculate dissimilarity scores for")
args = parser.parse_args()

ITERATION_LEVELS = range(1, 11) if args.year_mode == '5-year' else range(1, 101)
INJECTION_LEVELS = [20, 40, 60, 80, 100]
YEARS = [f"{i}-{i+4}" for i in range(1970, 2020, 5)]


def get_embeddings(files: List[File], inputs_to_embeddings):
    """ Get embeddings for all inputs in the list of files """
    all_inputs = [input for f in files for input in f.inputs]
    embeddings = [inputs_to_embeddings[input] for input in all_inputs]
    return embeddings


def main(data_folder):
    print(f"Processing files in directory: {data_folder}")
    file_manager = FileManager(data_folder)
    inputs_to_embeddings = file_manager.get_inputs_to_embeddings(
        pickle_filename=f'inputs_to_xlmr_embeddings_{args.year_mode}_{args.variation}.pkl'
    )
    print(f"Number of inputs to embeddings: {len(inputs_to_embeddings.keys())}")

    results = []

    for target in targets:
        print(f"\n### {target}")
        if args.year_mode == '5-year':
            for year in YEARS:
                for injection_level in INJECTION_LEVELS:
                    dissimilarity_scores = []
                    for iteration in ITERATION_LEVELS:
                        files_no_injection = file_manager.get_files_for(0, target, year=year, iteration=iteration)
                        embeddings_no_injection = get_embeddings(files_no_injection, inputs_to_embeddings)

                        files_for_bin = file_manager.get_files_for(injection_level, target, year=year, iteration=iteration)
                        embeddings_for_bin = get_embeddings(files_for_bin, inputs_to_embeddings)

                        dissimilarity_scores_for_it = calculate_dissimilarity_scores(embeddings_no_injection, embeddings_for_bin)
                        dissimilarity_scores.extend(dissimilarity_scores_for_it)

                    avg_dissimilarity = np.mean(dissimilarity_scores)
                    std_dissimilarity = np.std(dissimilarity_scores)

                    results.append({
                        "dimension": "intensity",
                        "variation": args.variation,
                        "target": target,
                        "year": year,
                        "injection_level": f"0_{injection_level}",
                        "avg_dissimilarity": avg_dissimilarity,
                        "std_dissimilarity": std_dissimilarity,
                    })
        elif args.year_mode == 'all-year':
            for injection_level in INJECTION_LEVELS:
                dissimilarity_scores = []
                for iteration in ITERATION_LEVELS:
                    files_no_injection = file_manager.get_files_for(0, target, iteration=iteration)
                    embeddings_no_injection = get_embeddings(files_no_injection, inputs_to_embeddings)

                    files_for_bin = file_manager.get_files_for(injection_level, target, iteration=iteration)
                    embeddings_for_bin = get_embeddings(files_for_bin, inputs_to_embeddings)

                    dissimilarity_scores_for_it = calculate_dissimilarity_scores(embeddings_no_injection, embeddings_for_bin)
                    dissimilarity_scores.extend(dissimilarity_scores_for_it)

                avg_dissimilarity = np.mean(dissimilarity_scores)
                std_dissimilarity = np.std(dissimilarity_scores)

                results.append({
                    "dimension": "intensity",
                    "variation": args.variation,
                    "target": target,
                    "year": "all",
                    "injection_level": f"0_{injection_level}",
                    "avg_dissimilarity": avg_dissimilarity,
                    "std_dissimilarity": std_dissimilarity,
                })
    filename = f'deberta_dissimilarity_across_{args.year_mode}_{args.variation}_intensity.csv'
    save_to_csv(results, filename)



if __name__ == "__main__":
    main(f"output/{args.year_mode}/{args.variation}/")