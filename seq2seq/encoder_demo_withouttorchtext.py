import torch
import torch.nn as nn

# ----------------------------
# Step 1: Build Vocabulary
# ----------------------------
corpus = [
    "i am happy today",
    "you are going to the market",
    "we will learn ai",
    "this is an encoder demo"
]

def build_vocab(corpus):
    vocab = {"<pad>": 0, "<unk>": 1}
    idx = 2
    for sentence in corpus:
        for word in sentence.lower().split():
            if word not in vocab:
                vocab[word] = idx
                idx += 1
    return vocab

vocab = build_vocab(corpus)
inv_vocab = {i: w for w, i in vocab.items()}

# ----------------------------
# Step 2: Encoder Class
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

# Model setup
embedding_dim = 32
hidden_dim = 64
encoder = Encoder(len(vocab), embedding_dim, hidden_dim)

# ----------------------------
# Step 3: Sentence Encoding Function
# ----------------------------
def encode_sentence(sentence):
    tokens = sentence.lower().split()
    token_ids = [vocab.get(tok, vocab["<unk>"]) for tok in tokens]
    tensor = torch.tensor(token_ids).unsqueeze(1)  # shape: [seq_len, batch_size]
    return tokens, token_ids, tensor

# ----------------------------
# Step 4: Demo Run
# ----------------------------
def run_demo(sentence):
    print(f"\n📝 Input sentence: {sentence}")
    tokens, token_ids, input_tensor = encode_sentence(sentence)

    embedded, outputs, hidden, cell = encoder(input_tensor)

    print(f"\n🔡 Tokens: {tokens}")
    print(f"🔢 Token IDs: {token_ids}")

    print("\n📦 Embedded Vectors:")
    for i, vec in enumerate(embedded.squeeze(1)):
        print(f"{tokens[i]} → {vec.detach().numpy()}")

    print("\n🧠 Final Hidden State (Summary Vector):")
    print(hidden.squeeze().detach().numpy())

# ----------------------------
# Entry Point
# ----------------------------
if __name__ == "__main__":
    user_input = input("Enter a sentence to encode: ")
    run_demo(user_input)

