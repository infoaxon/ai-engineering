from transformers import pipeline
import pandas as pd

# Set up pipeline for masked language modeling
unmasker = pipeline('fill-mask', model='bert-base-uncased')

# Get predictions for masked sentence
results = unmasker("The capital of France is [MASK].")

# Pretty-print results in DataFrame
df_results = pd.DataFrame(results)
print(df_results[['sequence', 'score', 'token_str']])
