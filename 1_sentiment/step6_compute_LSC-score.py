# Author: Raphael Merx
# Aim: Compute LSC score for all-year and 5-year files (bootstrap and stratified random sampling stratigies)

import os
import sys
import argparse
import numpy as np
from typing import List, Dict

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # needed as lexeme_utils.py is in the parent dir
from lexeme_utils import calculate_dissimilarity_scores, save_to_csv, targets, File, FileManager


parser = argparse.ArgumentParser(description="Calculate dissimilarity scores for all-year data")
parser.add_argument("--year-mode", type=str, choices=["all-year", "5-year"], help="Year mode to use")
parser.add_argument("--variation", type=str, choices=["positive", "negative"], help="Variation to calculate dissimilarity scores for")
args = parser.parse_args()

ITERATION_LEVELS = range(1, 11) if args.year_mode == '5-year' else range(1, 101)
INJECTION_LEVELS = [20, 40, 60, 80, 100]
YEARS = [f"{i}-{i+4}" for i in range(1970, 2020, 5)]


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
    inputs_to_embeddings = file_manager.get_inputs_to_embeddings(
        pickle_filename=f'inputs_to_lexeme_embeddings_{args.year_mode}_{args.variation}.pkl'
    )

    results = []

    for target in targets:
        print(f"\n### {target}")
        for year in YEARS:
            for injection_level in INJECTION_LEVELS:
                dissimilarity_scores_for_injection = []
                for iteration in ITERATION_LEVELS:
                    files_no_injection = file_manager.get_files_for(injection_level=0, target=target, iteration=iteration, year=year)
                    assert len(files_no_injection) == 1, f"Expected 1 file for iteration={iteration}, target={target}, injection_level=0, year {year}, got {len(files_no_injection)}"
                    embeddings_no_injection = get_embeddings(files_no_injection, inputs_to_embeddings)

                    files_for_bin = file_manager.get_files_for(injection_level, target, iteration, year=year)
                    assert len(files_for_bin) == 1, f"Expected 1 file for iteration={iteration}, target={target}, injection_level={injection_level}, year {year}, got {len(files_for_bin)}"
                    embeddings_for_bin = get_embeddings(files_for_bin, inputs_to_embeddings)

                    dissimilarity_scores = calculate_dissimilarity_scores(embeddings_no_injection, embeddings_for_bin)
                    avg_dissimilarity = np.mean(dissimilarity_scores)

                    dissimilarity_scores_for_injection.append(avg_dissimilarity)
                avg_dissimilarity = np.mean(dissimilarity_scores_for_injection)
                results.append({
                    "dimension": "sentiment",
                    "variation": args.variation,
                    "target": target,
                    "year": year,
                    "injection_level": f"0_{injection_level}",
                    "avg_dissimilarity": avg_dissimilarity
                })
    filename = f'sentiment_lexeme_dissimilarity_{args.year_mode}_{args.variation}.csv'
    save_to_csv(results, filename)



if __name__ == "__main__":
    main(f"output/{args.year_mode}/{args.variation}/")