# save as encoder_demo_no_torchtext.py
import streamlit as st
import torch
import torch.nn as nn
import pandas as pd

# ----------------------------
# Basic Tokenizer and Vocabulary
# ----------------------------
corpus = [
    "i am happy today",
    "you are going to the market",
    "we will learn ai",
    "this is an encoder demo"
]

# Build vocabulary


def build_vocab(corpus):
    vocab = {"<pad>": 0, "<unk>": 1}
    idx = 2
    for sentence in corpus:
        for word in sentence.split():
            if word not in vocab:
                vocab[word] = idx
                idx += 1
    return vocab


vocab = build_vocab(corpus)
inv_vocab = {i: w for w, i in vocab.items()}

# ----------------------------
# Encoder Definition
# ----------------------------


class Encoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)

    def forward(self, input_seq):
        embedded = self.embedding(input_seq)
        outputs, (hidden, cell) = self.lstm(embedded)
        return embedded, outputs, hidden, cell


embedding_dim = 32
hidden_dim = 64
encoder = Encoder(len(vocab), embedding_dim, hidden_dim)

# ----------------------------
# Streamlit Interface
# ----------------------------
st.title("🧠 Encoder Demo (No torchtext)")

sentence = st.text_input("Enter a sentence", "i am happy today")

if sentence:
    tokens = sentence.lower().split()
    token_ids = [vocab.get(tok, vocab["<unk>"]) for tok in tokens]
    input_tensor = torch.tensor(token_ids).unsqueeze(
        1)  # [seq_len, batch_size]

    embedded, outputs, hidden, cell = encoder(input_tensor)

    st.markdown("### 🔤 Tokenized Input")
    st.write(f"Tokens: {tokens}")
    st.write(f"Token IDs: {token_ids}")

    st.markdown("### 🧬 Word Embeddings")
    embed_table = pd.DataFrame(
        embedded.squeeze(1).detach().numpy(),
        index=tokens,
        columns=[f"dim_{i+1}" for i in range(embedding_dim)]
    )
    st.dataframe(embed_table.style.set_precision(4), height=300)

    st.markdown("### 🧠 Final Hidden State (Summary Vector)")
    hidden_vector = hidden.squeeze(0).detach().numpy()
    hidden_df = pd.DataFrame(hidden_vector.reshape(
        1, -1), columns=[f"h{i+1}" for i in range(hidden_dim)])
    st.dataframe(hidden_df.style.set_precision(4))

    st.success("Encoding complete! Try a different sentence.")
