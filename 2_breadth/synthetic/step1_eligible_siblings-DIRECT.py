import os
import csv
from nltk.corpus import wordnet as wn
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import torch

# Define thresholds
LIN_SIMILARITY_THRESHOLD = 0.5  # Set as needed
COSINE_SIMILARITY_THRESHOLD = 0.7  # Set as needed

# Define keywords to identify psychology-related synsets
PSYCHOLOGY_KEYWORDS = {"abnormality", "abnormally", "emotional", "feeling", "feelings", "harm", "hurt", 
                       "mental", "mind", "psychological", "psychology", 
                       "psychiatry", "syndrome", "therapy", "treatment"}

# Load custom IC values
def load_custom_ic(file_path):
    ic = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            offset_pos, ic_value = line.strip().split()
            offset, pos = offset_pos.split('.')
            try:
                offset_int = int(offset.lstrip("0"))
                synset = wn.synset_from_pos_and_offset(pos, offset_int)
                ic[synset] = float(ic_value)
            except Exception as e:
                print(f"Error loading IC for {offset_pos}: {e}")
    return ic

# Load IC values
absolute_path = os.path.abspath('output/donor_terms/ic-psychology.dat')
ic = load_custom_ic(absolute_path)

# Initialize BERT model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.1")
model = AutoModel.from_pretrained("dmis-lab/biobert-base-cased-v1.1")

# Check if a synset gloss is psychology-related
def is_psychology_related(synset):
    result = any(keyword in synset.definition().lower() for keyword in PSYCHOLOGY_KEYWORDS)
    print(f"Checking if {synset.name()} is psychology-related: {result}")
    return result

# Calculate Lin similarity
def calculate_custom_lin_similarity(synset1, synset2):
    ic1, ic2 = ic.get(synset1), ic.get(synset2)
    if ic1 is None or ic2 is None:
        print(f"Missing IC for: {synset1} or {synset2}")
        return "NA"
    lcs = find_lcs(synset1, synset2)
    if not lcs or lcs not in ic:
        print(f"No LCS with IC for: {synset1} and {synset2}")
        return "NA"
    ic_lcs = ic[lcs]
    return (2 * ic_lcs) / (ic1 + ic2)

# Find LCS with the highest IC value for Lin similarity
def find_lcs(synset1, synset2):
    common_hypernyms = synset1.common_hypernyms(synset2)
    ic_hypernyms = [s for s in common_hypernyms if s in ic]
    return max(ic_hypernyms, key=lambda s: ic[s], default=None)

# Calculate cosine similarity using BERT embeddings
def calculate_cosine_similarity(synset1, synset2):
    def embed_text(text):
        inputs = tokenizer(text, return_tensors="pt")
        outputs = model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).detach().numpy()

    embedding1 = embed_text(synset1.definition())
    embedding2 = embed_text(synset2.definition())
    return cosine_similarity(embedding1, embedding2)[0][0]

# Get non-antonym lemmas for a sibling synset that are not antonyms of the target synset
def get_non_antonym_lemmas(sibling, target_synset):
    target_lemmas = {lemma for lemma in target_synset.lemmas()}
    non_antonym_lemmas = []

    # Add lemmas to non_antonym_lemmas if they aren't antonyms of any lemmas in the target synset
    for lemma in sibling.lemmas():
        if not any(antonym in target_lemmas for antonym in lemma.antonyms()):
            non_antonym_lemmas.append(lemma.name())
    return non_antonym_lemmas

# Get siblings (co-hyponyms) by retrieving all hyponyms under shared higher-level hypernyms
def get_siblings(synset):
    siblings = set()
    
    # Get the immediate hypernyms (direct parents) of the target synset
    for hypernym in synset.hypernyms():
        # Add all hyponyms of each hypernym as potential siblings
        for sibling in hypernym.hyponyms():
            # Exclude the target synset itself
            if sibling != synset:
                siblings.add(sibling)
                
    return siblings

# Write siblings to CSV with detailed logging
def write_siblings_to_csv(siblings_data, output_file):
    with open(output_file, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["target", "synset", "definition", "sibling", "sibling_definition", "lin_similarity", "cosine_similarity", "note"])
        for entry in siblings_data:
            writer.writerow([
                entry["target"], entry["synset"], entry["definition"], 
                entry["sibling"], entry["sibling_definition"], 
                entry.get("lin_similarity", "NA"), entry.get("cosine_similarity", "NA"), 
                entry.get("note", "")
            ])

# Process each target term, filter synsets based on gloss, and get eligible/ineligible siblings
def process_target_terms(target_terms):
    all_filtered_data = []
    eligible_siblings, ineligible_siblings = [], []

    for target_term in target_terms:
        print(f"\nProcessing term: {target_term}")
        
        for synset in wn.synsets(target_term, pos='n'):
            # Check if the synset is psychology-related
            if not is_psychology_related(synset):
                all_filtered_data.append({
                    "target": target_term,
                    "synset": synset.name(),
                    "definition": synset.definition(),
                    "note": "Synset (gloss) not psychology-related"
                })
                continue

            # Retrieve all siblings for debugging
            siblings = get_siblings(synset)
            print(f"Retrieved siblings for {synset.name()}: {[sibling.name() for sibling in siblings]}")

            # Evaluate each sibling synset
            for sibling in siblings:
                print(f"  Evaluating sibling: {sibling.name()} - {sibling.definition()}")

                # Get non-antonym lemmas for the sibling synset
                non_antonym_lemmas = get_non_antonym_lemmas(sibling, synset)
                if not non_antonym_lemmas:
                    ineligible_siblings.append({
                        "target": target_term,
                        "synset": synset.name(),
                        "definition": synset.definition(),
                        "sibling": sibling.name(),
                        "sibling_definition": sibling.definition(),
                        "lin_similarity": "NA",
                        "cosine_similarity": "NA",
                        "note": "Contains only antonyms of target synset"
                    })
                    print(f"    {sibling.name()} has only antonymic relationship, marked ineligible.")
                    continue

                # Calculate similarities
                lin_sim = calculate_custom_lin_similarity(synset, sibling)
                cos_sim = calculate_cosine_similarity(synset, sibling)
                print(f"    Similarities for {sibling.name()} - Lin: {lin_sim}, Cosine: {cos_sim}")

                sibling_entry = {
                    "target": target_term,
                    "synset": synset.name(),
                    "definition": synset.definition(),
                    "sibling": sibling.name(),
                    "sibling_definition": sibling.definition(),
                    "lin_similarity": lin_sim,
                    "cosine_similarity": cos_sim
                }

                # Check thresholds and categorize
                if lin_sim != "NA" and lin_sim >= LIN_SIMILARITY_THRESHOLD and cos_sim >= COSINE_SIMILARITY_THRESHOLD:
                    print(f"    {sibling.name()} meets thresholds. Adding to eligible.")
                    eligible_siblings.append(sibling_entry)
                else:
                    reason = "Below threshold(s)" if lin_sim != "NA" else "Missing IC"
                    sibling_entry["note"] = reason
                    ineligible_siblings.append(sibling_entry)
                    print(f"    {sibling.name()} does NOT meet thresholds. Reason: {reason}")

    # Write ineligible siblings with detailed reasons to siblings_ineligible.csv
    write_siblings_to_csv(ineligible_siblings, "output/donor_terms/siblings_ineligible.csv")
    # Write eligible siblings to eligible_siblings.csv
    write_siblings_to_csv(eligible_siblings, "output/donor_terms/siblings_eligible.csv")

# List of target terms
target_terms = ["trauma", "anxiety", "depression", "mental_health", "mental_illness", "abuse"]

# Run the processing
process_target_terms(target_terms)
