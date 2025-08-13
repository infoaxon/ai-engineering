import torch
import torch.nn as nn
import random

# ----------------------------
# Vocabulary Setup
# ----------------------------
corpus = [
    "i am happy today",
    "you are going to the market",
    "we will learn ai",
    "this is an encoder demo",
]


def build_vocab(corpus):
    vocab = {"<pad>": 0, "<unk>": 1, "<sos>": 2, "<eos>": 3}
    idx = 4
    for sentence in corpus:
        for word in sentence.lower().split():
            if word not in vocab:
                vocab[word] = idx
                idx += 1
    return vocab


vocab = build_vocab(corpus)
inv_vocab = {i: w for w, i in vocab.items()}
vocab_size = len(vocab)

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
        return outputs, hidden, cell


# ----------------------------
# Decoder Definition
# ----------------------------


class Decoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_token, hidden, cell):
        # input_token: [1, batch_size]
        embedded = self.embedding(input_token)
        output, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        prediction = self.fc(output.squeeze(0))  # [batch_size, vocab_size]
        return prediction, hidden, cell


# ----------------------------
# Helper Function: Encode + Decode
# ----------------------------


def run_seq2seq_demo(input_sentence, max_len=10):
    tokens = input_sentence.lower().split()
    token_ids = [vocab.get(tok, vocab["<unk>"]) for tok in tokens]
    input_tensor = torch.tensor(token_ids).unsqueeze(1)  # [seq_len, batch_size]

    # Encode
    _, hidden, cell = encoder(input_tensor)

    # Decode
    decoder_input = torch.tensor([[vocab["<sos>"]]])  # start token
    decoded_tokens = []

    for _ in range(max_len):
        output, hidden, cell = decoder(decoder_input, hidden, cell)
        predicted_id = output.argmax(1).item()

        if predicted_id == vocab["<eos>"]:
            break

        decoded_tokens.append(inv_vocab.get(predicted_id, "<unk>"))
        decoder_input = torch.tensor([[predicted_id]])

    return tokens, decoded_tokens


# ----------------------------
# Initialize and Run
# ----------------------------
embedding_dim = 32
hidden_dim = 64
encoder = Encoder(vocab_size, embedding_dim, hidden_dim)
decoder = Decoder(vocab_size, embedding_dim, hidden_dim)

if __name__ == "__main__":
    sentence = input("Enter a sentence: ")
    input_tokens, output_tokens = run_seq2seq_demo(sentence)
    print(f"\n🔡 Input Tokens: {input_tokens}")
    print(f"🗣️ Generated Output Tokens: {output_tokens}")
