import cramming
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("pbelcak/UltraFastBERT-1x11-long")
model = AutoModelForMaskedLM.from_pretrained("pbelcak/UltraFastBERT-1x11-long")

text = "My name is Sylvain and I am a <mask> engineer."
encoded_input = tokenizer(text, return_tensors='pt')
mask_token_index = torch.where(encoded_input["input_ids"] == tokenizer.mask_token_id)[1]

# Get the model's output
output = model(**encoded_input)

# Extract the logits from the 'outputs' key
logits = output['outputs']

logits_at_mask = logits[mask_token_index, :]

# Convert logits to probabilities
probabilities = torch.softmax(logits_at_mask, dim=-1)

# Get top 5 likely tokens
top_5_tokens = torch.topk(probabilities, 5, dim=-1)
top_5_prob, top_5_indices = top_5_tokens.values, top_5_tokens.indices

# Decode the top 5 tokens back to words
top_5_words = tokenizer.decode(top_5_indices[0]).split()

# Printing the results
print("Top 5 predictions for the masked word:")
for i, word in enumerate(top_5_words):
    print(f"{i+1}: {word} (Probability: {top_5_prob[0][i].item():.4f})")
