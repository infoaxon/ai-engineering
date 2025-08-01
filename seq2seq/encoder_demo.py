import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtext.vocab import build_vocab_from_iterator
from torchtext.data.utils import get_tokenizer

# Step 1: Tokenizer and Vocabulary
tokenizer = get_tokenizer('basic_english')


def yield_tokens(sentences):
    for sentence in sentences:
        yield tokenizer(sentence)


# Sample corpus to build small vocab
corpus = [
    "I am happy today",
    "You are going to the market",
    "We will learn AI",
    "This is an encoder demo"
]

vocab = build_vocab_from_iterator(
    yield_tokens(corpus), specials=["<pad>", "<unk>"])
vocab.set_default_index(vocab["<unk>"])

# Step 2: Encoder Class


class Encoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)

    def forward(self, input_seq):
        embedded = self.embedding(input_seq)
        outputs, (hidden, cell) = self.lstm(embedded)
        return embedded, outputs, hidden, cell


# Hyperparameters
embedding_dim = 32
hidden_dim = 64
vocab_size = len(vocab)

encoder = Encoder(vocab_size, embedding_dim, hidden_dim)

# Step 3: Input Sentence to Tensor


def encode_sentence(sentence):
    tokens = tokenizer(sentence)
    token_ids = vocab(tokens)
    tensor = torch.tensor(token_ids).unsqueeze(
        1)  # shape: [seq_len, batch_size]
    return tensor, tokens

# Step 4: Run a Demo


def run_demo(sentence):
    print(f"\n📝 Input sentence: {sentence}")
    input_tensor, tokens = encode_sentence(sentence)
    embedded, outputs, hidden, cell = encoder(input_tensor)

    print(f"\n🔡 Tokens: {tokens}")
    print(f"\n🔢 Token IDs: {input_tensor.squeeze().tolist()}")

    print("\n📦 Embedded Vectors (per word):")
    for idx, vector in enumerate(embedded.squeeze(1)):
        print(f"{tokens[idx]} → {vector.detach().numpy()}")

    print("\n🧠 Final Hidden State (summary vector):")
    print(hidden.squeeze().detach().numpy())


# Example use
if __name__ == "__main__":
    sentence = input("Enter a sentence: ")
    run_demo(sentence)
