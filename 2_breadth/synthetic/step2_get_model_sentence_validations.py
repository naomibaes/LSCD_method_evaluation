import os
import random
import pandas as pd
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch
import re

# Constants
random_seed = 93
SENTENCE_SAMPLE_SIZE = 50  # Total number of sentences to sample per target
COSINE_THRESHOLD = 0.85  # Cosine similarity threshold for validation
MLM_RATIO_THRESHOLD = 0.5  # Maximum MLM ratio for validation
output_dir = "output/annotate"
os.makedirs(output_dir, exist_ok=True)

# Input files
natural_corpus_file = "C:/Users/naomi/OneDrive/COMP80004_PhDResearch/RESEARCH/DATA/CORPORA/Psychology/abstract_year_journal.csv.mental.sentence-CLEAN.tsv"
siblings_file = os.path.join(output_dir, "donor_terms/siblings_eligible.csv")
#targets = ["trauma"] # for testing
targets = ["trauma", "anxiety", "depression", "mental_health", "mental_illness", "abuse"]

# Models for Validation
cosine_models = {
    "MiniLM-L12-v2": SentenceTransformer('all-MiniLM-L12-v2'),
    "DistilRoBERTa-v1": SentenceTransformer('all-distilroberta-v1'),
    "Sentence-T5": SentenceTransformer('sentence-t5-base')
}
mlm_models = {}
mlm_tokenizers = {}

def load_model_and_tokenizer(name, model_id):
    model = AutoModelForMaskedLM.from_pretrained(model_id).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    mlm_models[name] = model
    mlm_tokenizers[name] = tokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
load_model_and_tokenizer("RoBERTa-large", "roberta-large")
load_model_and_tokenizer("DeBERTa-v3-large", "microsoft/deberta-v3-large")
load_model_and_tokenizer("BioBERT", "dmis-lab/biobert-base-cased-v1.1")

random.seed(random_seed)
torch.manual_seed(random_seed)

# Load datasets
natural_corpus = pd.read_csv(natural_corpus_file, sep="\t", header=None, names=["sentence", "year", "label"])
natural_corpus = natural_corpus.dropna(subset=["sentence"])
siblings_df = pd.read_csv(siblings_file)
siblings_df["sibling_normalized"] = (
    siblings_df["sibling"]
    .str.split(".")
    .str[0]
    .str.replace("_", " ", regex=False)
    .str.lower()
)

# Step 1: MLM Validation
def validate_with_mlm(sentence, donor, target, model, tokenizer):
    try:
        donor_normalized = donor.lower().strip()
        target_normalized = target.lower().strip()
        sentence_normalized = sentence.lower().strip()

        if not re.search(fr"\b{re.escape(donor_normalized)}\b", sentence_normalized):
            return None

        masked_sentence = re.sub(fr"\b{re.escape(donor_normalized)}\b", tokenizer.mask_token, sentence_normalized, flags=re.IGNORECASE)
        inputs = tokenizer(masked_sentence, return_tensors="pt").to(device)
        if tokenizer.mask_token_id not in inputs.input_ids:
            return None

        logits = model(**inputs).logits
        mask_indices = (inputs.input_ids == tokenizer.mask_token_id).nonzero(as_tuple=True)
        if len(mask_indices[1]) == 0:
            return None

        mask_index = mask_indices[1][0]
        donor_id = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(donor_normalized)[0])
        target_id = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(target_normalized)[0])
        donor_prob = torch.softmax(logits[0, mask_index, :], dim=-1)[donor_id].item()
        target_prob = torch.softmax(logits[0, mask_index, :], dim=-1)[target_id].item()

        return target_prob / donor_prob if donor_prob > 0 else None
    except Exception as e:
        return None

# Step 2: Cosine Similarity Validation
def compute_cosine_similarity(sentence, donor, target, models):
    similarities = {}
    for model_name, model in models.items():
        try:
            embeddings = model.encode(
                [sentence.replace(donor, ""), sentence.replace(donor, target)], convert_to_tensor=True
            )
            similarity = torch.nn.functional.cosine_similarity(embeddings[0], embeddings[1], dim=0).item()
            similarities[model_name] = similarity
        except Exception as e:
            similarities[model_name] = None
    return similarities

# Sampling and Validation Loop
results = []
skipped = []  # To track skipped sentences with detailed information
for target in targets:
    sibling_rows = siblings_df[siblings_df["target"] == target]
    siblings = sibling_rows["sibling_normalized"].tolist()
    sampled_sentences = []

    sibling_sentences_dict = {}  # To store all sentences for each sibling
    
    while len(sampled_sentences) < SENTENCE_SAMPLE_SIZE:
        if not siblings:
            skipped.append({"target": target, "sibling": None, "sentence": None, "reason": "No more siblings available"})
            break
        
        sibling = random.choice(siblings)  # Randomly pick a sibling
        
        if sibling not in sibling_sentences_dict:  # Check if we've processed sentences for this sibling
            sibling_sentences = natural_corpus[natural_corpus["sentence"].str.contains(fr"\b{sibling}\b", case=False, na=False, regex=True)]
            sibling_sentences_dict[sibling] = sibling_sentences.sample(frac=1, random_state=random_seed).reset_index(drop=True)  # Shuffle sentences
        
        # Ensure there are available sentences
        if sibling_sentences_dict[sibling].empty:
            siblings.remove(sibling)  # Remove sibling if no sentences are available
            skipped.append({"target": target, "sibling": sibling, "sentence": None, "reason": "No sentences available"})
            continue
        
        # Randomly sample a sentence for this sibling
        sentence_row = sibling_sentences_dict[sibling].iloc[0]  # Pick the first shuffled sentence
        sentence = sentence_row["sentence"]
        sibling_sentences_dict[sibling] = sibling_sentences_dict[sibling].iloc[1:]  # Remove the selected sentence
        
        result = {"target": target, "sibling": sibling, "sentence": sentence}

        # Step 1: MLM Validation
        mlm_ratios = {}
        for name, model in mlm_models.items():
            tokenizer = mlm_tokenizers[name]
            ratio = validate_with_mlm(sentence, sibling, target, model, tokenizer)
            mlm_ratios[f"mlm_ratio_{name}"] = ratio
        result.update(mlm_ratios)

        # Step 2: Cosine Similarity Validation
        cosine_scores = compute_cosine_similarity(sentence, sibling, target, cosine_models)
        result.update(cosine_scores)

        # Step 3: Validation Column
        valid_cosine = all(sim >= COSINE_THRESHOLD for sim in cosine_scores.values() if sim is not None)
        valid_mlm = any(ratio is not None and ratio <= MLM_RATIO_THRESHOLD for ratio in mlm_ratios.values())
        result["validated"] = valid_cosine and valid_mlm

        sampled_sentences.append(result)

    results.extend(sampled_sentences)

# Save Results
results_df = pd.DataFrame(results)
results_df.to_csv(os.path.join(output_dir, "validated_sentences.csv"), index=False)
print("Saved validated_sentences.csv.")

# Create the ANNOTATE file with an extra column for ground_truth
annotate_df = results_df.copy()
annotate_df["ground_truth"] = ""  # Add an empty 'ground_truth' column
annotate_df.to_csv(os.path.join(output_dir, "validated_sentences_ANNOTATE.csv"), index=False)
print("Saved validated_sentences_ANNOTATE.csv with an additional 'ground_truth' column for annotation.")

# Check if there are skipped sentences
if skipped:
    skipped_df = pd.DataFrame(skipped, columns=["target", "sibling", "sentence", "reason"])
    skipped_df.to_csv(os.path.join(output_dir, "skipped_sentences.csv"), index=False)
    print("Skipped sentences detected. Saved skipped_sentences.csv.")
else:
    print("No skipped sentences detected.")
