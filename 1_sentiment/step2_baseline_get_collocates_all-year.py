import glob
import os
import pandas as pd
from collections import Counter
import re

def extract_contexts(sentence, target, window_size=5):
    words = sentence.split()
    target_indices = [i for i, word in enumerate(words) if target in word]
    contexts = []
    for index in target_indices:
        start = max(index - window_size, 0)
        end = min(index + window_size + 1, len(words))
        context_words = [words[i] for i in range(start, end) if i != index]
        contexts.extend(context_words)
    return contexts

def process_files(input_dir, output_base_dir):
    output_dir = os.path.join(output_base_dir, os.path.basename(input_dir))
    os.makedirs(output_dir, exist_ok=True)
    
    log_file_path = os.path.join(output_dir, 'warnings.log')
    
    # Only process files that contain 'lemmatized' in their name
    for file_path in glob.glob(f"{input_dir}/*lemmatized.tsv"):
        file_base = os.path.basename(file_path).replace('.tsv', '')
        target = file_base.split('_')[0]  # Assumes target is the first element in the filename
        df = pd.read_csv(file_path, delimiter='\t', header=None, names=["sentence", "year", "type"])

        context_counter = Counter()
        for _, row in df.iterrows():
            contexts = extract_contexts(row["sentence"], target)
            context_counter.update(contexts)

        if context_counter:
            output_file = f"{output_dir}/{file_base}_collocates.csv"
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write("collocate,count\n")
                for word, count in context_counter.items():
                    f.write(f"{word},{count}\n")
            print(f"Output written to {output_file}")
        else:
            with open(log_file_path, 'a', encoding='utf-8') as log_file:
                log_file.write(f"No contexts found for {target} in file {file_path}\n")

# Specify the directories
input_base_dir = "output/all-year"
output_base_dir = "output/all-year/collocates"

# Process the files in both 'positive' and 'negative' subdirectories
for sentiment in ["positive", "negative"]:
    input_dir = os.path.join(input_base_dir, sentiment)
    process_files(input_dir, output_base_dir)
