from nltk.corpus import wordnet as wn

target_list = ["trauma", "anxiety", "depression", "mental_health", "mental_illness", "abuse", "car", "symptom"]  # Add more terms as needed

# Function to get siblings of a synset
def get_siblings(synset):
    siblings = set()
    
    # Get the hypernyms of the current synset
    hypernyms = synset.hypernyms()
    
    for hypernym in hypernyms:
        # Get all hyponyms (children) of the hypernym, which includes the siblings
        hyponyms = hypernym.hyponyms()
        
        for hyponym in hyponyms:
            if hyponym != synset:  # Exclude the original synset to get only siblings
                siblings.update(hyponym.lemma_names())
    
    return siblings

# Loop through each target term and get the synsets and related information
for target_term in target_list:
    print(f"\nProcessing term: {target_term}\n")
    for synset in wn.synsets(target_term, "n"):  # Only considering noun synsets here
        print(f"Synset: {synset.name()}")
        print(f"Definition: {synset.definition()}")
        print(f"Lemma names: {synset.lemma_names()}")
        print(f"Examples: {synset.examples()}")
        print(f"Hypernyms: {synset.hypernyms()}")
        print(f"Hyponyms: {synset.hyponyms()}")

        # Get siblings using the defined function
        siblings = get_siblings(synset)
        print(f"Siblings: {siblings}")
        print("-------")
