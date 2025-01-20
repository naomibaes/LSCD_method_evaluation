# Author: Raphael Merx

import re
import os
import csv
from glob import glob
from typing import Literal, List
from functools import cached_property

from tqdm import tqdm
from dataclasses import dataclass
from dotenv import load_dotenv
load_dotenv()
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline
import torch
import torch.nn.functional as F
import numpy as np

# Load the ABSA model and tokenizer
model_name = "yangheng/deberta-v3-base-absa-v1.1"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)


@dataclass
class File:
    path: str

    @property
    def target(self):
        return self.file_details[0]
    
    @property
    def injection_ratio(self):
        return self.file_details[1]
    
    @property
    def index_type(self):
        if 'positive' in self.path.lower():
            return 'positive'
        elif 'negative' in self.path.lower():
            return 'negative'
        else:
            return 'unknown'
    
    @property
    def sentences(self):
        with open(self.path, 'r') as f:
            lines = f.readlines()
        return [line.split('\t')[0] for line in lines]

    @cached_property
    def file_details(self):
        parts = re.findall(r'([a-zA-Z]+)_synthetic_sentiment_([0-9]+)', self.path)
        return parts[0]
    
    def calculate_sentiment_score(self) -> List[float]:
        scores = get_sentiment_score(self.sentences, self.target)
        self.sentiment_scores = scores
        return scores


class FileManager(list):
    def __init__(self, files: List[File]):
        super().__init__(files)
    
    def get_all_target_injection_ratio_combinations(self):
        combinations = set([(file.target, file.injection_ratio) for file in self])
        # sort them by (target, injection_ratio)
        return sorted(combinations, key=lambda x: (x[0], int(x[1])))
    
    def get_files_for(self, target: str, injection_ratio: str, index_type: str):
        return [file for file in self if file.target == target and file.injection_ratio == injection_ratio and file.index_type == index_type]
    
    def get_results(self):
        """ Get the average sentiment score and standard errors for each target, injection_ratio combination """
        results = []
        for target, injection_ratio in self.get_all_target_injection_ratio_combinations():
            scores_per_index = {}
            for index_type in ['positive', 'negative']:
                files = self.get_files_for(target, injection_ratio, index_type)
                all_scores = [score for file in files for score in file.sentiment_scores]
                avg_score = sum(all_scores) / len(all_scores) if all_scores else 0
                se_score = (np.std(all_scores, ddof=1) / np.sqrt(len(all_scores))) if len(all_scores) > 1 else None
                scores_per_index[index_type] = (avg_score, se_score)
            results.append({
                'target': target,
                'injection_ratio': injection_ratio,
                'avg_valence_index_positive': scores_per_index['positive'][0],
                'se_valence_index_positive': scores_per_index['positive'][1],
                'avg_valence_index_negative': scores_per_index['negative'][0],
                'se_valence_index_negative': scores_per_index['negative'][1]
            })
        return results
    
def get_sentiment_score(texts: List[str], aspect: str, batch_size: int = 16) -> List[float]:
    scores = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        inputs = tokenizer(
            batch_texts, 
            [aspect] * len(batch_texts),
            padding=True,
            return_tensors="pt"
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits

        probs = F.softmax(logits, dim=1)
        # Convert to continuous scores (0 to 1)
        # labels are [negative, neutral, positive] -- see model config.json
        batch_scores = (probs[:, 1] * 0.5 + probs[:, 2] * 1.0).cpu().tolist()
        scores.extend(batch_scores)
    
    return scores

def calculate_synthetic_files_sentiment():
    all_files = glob('output/all-year/positive/*.tsv')
    all_files.extend(glob('output/all-year/negative/*.tsv'))
    all_files = [f for f in all_files if not f.endswith('lemmatized.tsv')]
    file_manager = FileManager([File(f) for f in all_files])
    for file in tqdm(file_manager, desc='Calculating sentiment scores'):
        file.calculate_sentiment_score()
    results = file_manager.get_results()

    output_folder = 'output'
    output_file = 'absa_averaged_sentiment_index_all-year_with_se.csv'
    os.makedirs(output_folder, exist_ok=True)  # Ensures the output folder exists
    output_path = os.path.join(output_folder, output_file)

    # save to CSV
    with open(output_path, 'w') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print(f"File saved to {output_path}")

calculate_synthetic_files_sentiment()