import os
import math
from nltk.corpus import wordnet as wn
from nltk.tokenize import word_tokenize
from collections import defaultdict
from tqdm import tqdm
import nltk

# Ensure required NLTK data is downloaded
nltk.download("punkt")
nltk.download("wordnet")
nltk.download("averaged_perceptron_tagger")

# Function to map NLTK POS tags to WordNet POS tags
def get_wordnet_pos(nltk_pos_tag):
    if nltk_pos_tag.startswith('N'): # nouns
        return 'n'
    elif nltk_pos_tag.startswith('V'): # verbs
        return 'v'
    elif nltk_pos_tag.startswith('J'): # adjectives 
        return 'a'
    elif nltk_pos_tag.startswith('R'): # adverbs
        return 'r'
    else:
        return None

# Function to get synsets, their siblings, and common hypernyms for each target term
def get_relevant_synsets(target_terms):
    relevant_synsets = set()
    common_hypernyms_set = set()
    
    for term in target_terms:
        synsets = wn.synsets(term)
        for synset in synsets:
            # Add the synset itself
            relevant_synsets.add(synset)
            
            # Get siblings: synsets sharing the same hypernyms
            for hypernym in synset.hypernyms():
                siblings = hypernym.hyponyms()
                for sibling in siblings:
                    # Add all sibling synsets
                    relevant_synsets.add(sibling)
                    
                    # Add common hypernyms for each synset-sibling pair
                    common_hypernyms = synset.common_hypernyms(sibling)
                    common_hypernyms_set.update(common_hypernyms)
                    
    # Combine relevant synsets and common hypernyms
    relevant_synsets.update(common_hypernyms_set)
    return relevant_synsets

# Tokenize and preprocess text, handling terms with underscores
def preprocess_text(text):
    multi_word_terms = ["mental health", "mental illness", "state of mind", "bipolar disorder"]
    for term in multi_word_terms:
        text = text.replace(term, term.replace(" ", "_"))
    tokens = word_tokenize(text.lower())
    return tokens

# Map words to relevant synsets only, handling both underscore and space-separated terms, considering POS
def map_to_relevant_synsets(word, pos, relevant_synsets):
    possible_forms = {word, word.replace(" ", "_")}
    matched_synsets = []
    for form in possible_forms:
        synsets = wn.synsets(form, pos=pos)
        matched_synsets.extend([s for s in synsets if s in relevant_synsets])
    return matched_synsets

# Calculate frequency of relevant synsets in the corpus with POS consideration
def calculate_relevant_synset_frequencies(input_file, relevant_synsets):
    synset_freq = defaultdict(int)
    total_words = 0
    found_synsets = set()

    with open(input_file, 'r', encoding='utf-8') as f:
        next(f)  # Skip the header line
        for line in tqdm(f, desc="Processing input file"):
            parts = line.strip().split("\t")
            if len(parts) != 2:
                continue  # Skip lines that don't match expected format

            sentence = parts[0]  # Extract the sentence from the first column
            tokens = preprocess_text(sentence)
            pos_tags = nltk.pos_tag(tokens)

            for token, nltk_pos in pos_tags:
                wordnet_pos = get_wordnet_pos(nltk_pos)
                if wordnet_pos:  # Only proceed if there's a WordNet-compatible POS
                    synsets = map_to_relevant_synsets(token, wordnet_pos, relevant_synsets)
                    if synsets:
                        for synset in synsets:
                            synset_freq[synset.name()] += 1
                            found_synsets.add(synset.name())
                            total_words += 1

    # Identify synsets that were relevant but not found in the corpus
    not_found_synsets = {s.name() for s in relevant_synsets} - found_synsets

    # Save matched and unmatched synsets to a single file
    with open("output/donor_terms/matched_and_unmatched_synsets.txt", "w", encoding="utf-8") as f:
        f.write("Unmatched Synsets:\n")
        f.write(", ".join(not_found_synsets) + "\n\n")
        f.write("Matched Synsets:\n")
        f.write(", ".join(found_synsets) + "\n")

    return synset_freq, total_words

# Compute Information Content (IC) for relevant synsets
def compute_ic(synset_freq, total_words):
    ic_values = {}
    for synset, freq in synset_freq.items():
        prob = freq / total_words
        ic = -math.log(prob)
        ic_values[synset] = ic
    return ic_values

# Save IC to file using numeric offsets instead of synset names
def save_ic_file(ic_values, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        for synset_name, ic in ic_values.items():
            synset = wn.synset(synset_name)
            offset = synset.offset()
            pos = synset.pos()
            f.write(f"{offset}.{pos} {ic}\n")

# Main function to handle all steps
def create_psychology_specific_ic(input_file, target_terms, output_ic_file):
    # Step 1: Get relevant synsets including common hypernyms for Lin similarity
    relevant_synsets = get_relevant_synsets(target_terms)

    # Step 2: Calculate frequencies of these synsets in the corpus with POS tagging
    synset_freq, total_words = calculate_relevant_synset_frequencies(input_file, relevant_synsets)

    # Step 3: Compute Information Content (IC)
    ic_values = compute_ic(synset_freq, total_words)

    # Step 4: Save the IC values to a file with numeric offsets
    save_ic_file(ic_values, output_ic_file)
    print(f"IC file saved at {output_ic_file}")

# Example Usage
if __name__ == "__main__":
    input_file = r"C:\Users\naomi\OneDrive\COMP80004_PhDResearch\RESEARCH\DATA\CORPORA\Psychology\abstract_year_journal.csv.mental.sentence-CLEAN.tsv"
    target_terms = ["trauma", "anxiety", "depression", "distress", "mental_health", "mental_illness", "addiction", "abuse"]
    output_ic_file = "output/donor_terms/ic-psychology.dat"
    
    create_psychology_specific_ic(input_file, target_terms, output_ic_file)
