# Author: Raphael Merx

from dotenv import load_dotenv
load_dotenv()
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline
import torch
import torch.nn.functional as F

# Load the ABSA model and tokenizer
model_name = "yangheng/deberta-v3-base-absa-v1.1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

def get_sentiment_score(text: str, aspect: str) -> float:
    inputs = tokenizer(text, text_pair=aspect, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    
    probs = F.softmax(logits, dim=1)

    # Convert to continuous score (0 to 1)
    # labels are [negative, neutral, positive] -- see model config.json
    score = (probs[0][1] * 0.5 + probs[0][2] * 1.0).item()
    return score

target_word = 'depression'

sentences = {
    "positive_var": "In a majority of the preliminary open studies selected for this review, VNS was associated with a noticeable improvement of the depressive conditions (primary outcome: Hamilton Depression Rating Scale, HDRS) in the short and long term.",
    "negative_var": "In a majority of the preliminary open studies selected for this review, VNS was correlated with a troubling persistence of the depressive symptoms (primary outcome: Hamilton Depression Rating Scale, HDRS) in the short and long term.",
    "baseline": "In a majority of the preliminary open studies selected for this review, VNS was associated with a significant reduction of the depressive symptoms (primary outcome: Hamilton Depression Rating Scale, HDRS) in the short and long term."
}

for key, sentence in sentences.items():
    score = get_sentiment_score(sentence, target_word)
    print(f"{key}: {score:.3f}")

# prints:
# positive_var: 0.483
# negative_var: 0.159
# baseline: 0.499