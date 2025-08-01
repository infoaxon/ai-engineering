import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
import seaborn as sns
import numpy as np

# ----------------------------
# 1. Vocabulary Preparation
# ----------------------------


def build_vocab(sentences):
    vocab = {"<pad>": 0, "<sos>": 1, "<eos>": 2, "<unk>": 3}
    idx = 4
    for sentence in sentences:
        for word in sentence.split():
            if word not in vocab:
                vocab[word] = idx
                idx += 1
    return vocab


def sentence_to_tensor(sentence, vocab):
    tokens = sentence.split()
    ids = [vocab.get(tok, vocab["<unk>"]) for tok in tokens]
    return torch.tensor([vocab["<sos>"]] + ids + [vocab["<eos>"]]).unsqueeze(1)

# ----------------------------
# 2. Model Definitions
# ----------------------------


class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim * 2, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        src_len = encoder_outputs.shape[0]
        hidden = hidden.repeat(src_len, 1, 1)
        energy = torch.tanh(
            self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        attention = self.v(energy).squeeze(2)
        return torch.softmax(attention, dim=0)


class Encoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.lstm = nn.LSTM(emb_dim, hidden_dim)

    def forward(self, src):
        embedded = self.embedding(src)
        outputs, (hidden, cell) = self.lstm(embedded)
        return outputs, hidden, cell


class AttnDecoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim, attention):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.attention = attention
        self.lstm = nn.LSTM(hidden_dim + emb_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_token, hidden, cell, encoder_outputs):
        embedded = self.embedding(input_token).unsqueeze(0)
        attn_weights = self.attention(
            hidden, encoder_outputs).unsqueeze(1).permute(1, 2, 0)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        context = torch.bmm(attn_weights, encoder_outputs).permute(1, 0, 2)
        rnn_input = torch.cat((embedded, context), dim=2)
        output, (hidden, cell) = self.lstm(rnn_input, (hidden, cell))
        prediction = self.fc(output.squeeze(0))
        return prediction, hidden, cell

# ----------------------------
# 3. Training Function
# ----------------------------


def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion):
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    encoder_outputs, hidden, cell = encoder(input_tensor)
    target_len = target_tensor.shape[0]
    loss = 0

    decoder_input = target_tensor[0]  # <sos>
    for t in range(1, target_len):
        output, hidden, cell = decoder(
            decoder_input, hidden, cell, encoder_outputs)
        loss += criterion(output, target_tensor[t])
        decoder_input = target_tensor[t]  # teacher forcing

    loss.backward()
    encoder_optimizer.step()
    decoder_optimizer.step()
    return loss.item() / (target_len - 1)


# ----------------------------
# 4. Prepare Data
# ----------------------------
pairs = [
    ("i am happy", "mai khush hoon"),
    ("you are learning", "tum seekh rahe ho"),
    ("this is fun", "yeh mazedar hai"),
    ("we will go", "hum jaayenge"),
    ("they are here", "ve yahaan hain")
]

src_sentences = [s[0] for s in pairs]
tgt_sentences = [s[1] for s in pairs]
src_vocab = build_vocab(src_sentences)
tgt_vocab = build_vocab(tgt_sentences)

# ----------------------------
# 5. Initialize Model and Optimizer
# ----------------------------
embedding_dim = 32
hidden_dim = 64
encoder = Encoder(len(src_vocab), embedding_dim, hidden_dim)
attention = Attention(hidden_dim)
decoder = AttnDecoder(len(tgt_vocab), embedding_dim, hidden_dim, attention)

encoder_opt = optim.Adam(encoder.parameters(), lr=0.01)
decoder_opt = optim.Adam(decoder.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

# ----------------------------
# 6. Train the Model
# ----------------------------
epochs = 100
losses = []
for epoch in range(epochs):
    total_loss = 0
    for src, tgt in pairs:
        src_tensor = sentence_to_tensor(src, src_vocab)
        tgt_tensor = sentence_to_tensor(tgt, tgt_vocab)
        loss = train(src_tensor, tgt_tensor, encoder, decoder,
                     encoder_opt, decoder_opt, criterion)
        total_loss += loss
    avg_loss = total_loss / len(pairs)
    losses.append(avg_loss)
    if epoch % 10 == 0:
        print(f"Epoch {epoch}: Loss = {avg_loss:.4f}")


# Create inverse vocab for target and source
inv_src_vocab = {v: k for k, v in src_vocab.items()}
inv_tgt_vocab = {v: k for k, v in tgt_vocab.items()}


def predict_with_attention(input_sentence, encoder, decoder, src_vocab, tgt_vocab, inv_tgt_vocab, max_len=10):
    input_tensor = sentence_to_tensor(input_sentence, src_vocab)
    encoder_outputs, hidden, cell = encoder(input_tensor)

    decoder_input = torch.tensor([tgt_vocab["<sos>"]])
    decoded_tokens = []
    attentions = []

    for _ in range(max_len):
        output, hidden, cell = decoder(
            decoder_input, hidden, cell, encoder_outputs)
        prob = F.softmax(output, dim=1)
        pred_token = prob.argmax(1).item()
        decoded_tokens.append(pred_token)

        attn_weights = decoder.attention(
            hidden, encoder_outputs).squeeze(1).detach().numpy()
        attentions.append(attn_weights)

        if pred_token == tgt_vocab["<eos>"]:
            break

        decoder_input = torch.tensor([pred_token])

    decoded_words = [inv_tgt_vocab.get(tok, "<unk>") for tok in decoded_tokens]
    return decoded_words, np.array(attentions), input_tensor


def plot_attention(attn_weights, input_tensor, input_vocab, output_tokens):
    input_words = [inv_src_vocab.get(tok.item(), "<unk>")
                   for tok in input_tensor.squeeze()]
    input_words = input_words[1:-1]  # strip <sos>, <eos>
    output_words = output_tokens

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(attn_weights, xticklabels=input_words,
                yticklabels=output_words, cmap="viridis", annot=True, fmt=".2f")
    ax.set_xlabel("Input Sequence")
    ax.set_ylabel("Output Sequence")
    plt.title("Attention Weights")
    plt.tight_layout()
    plt.show()


#  Example sentence to visualize
input_text = "i am happy"
output_words, attn_matrix, input_tensor = predict_with_attention(
    input_text, encoder, decoder, src_vocab, tgt_vocab, inv_tgt_vocab)
plot_attention(attn_matrix, input_tensor, src_vocab, output_words)


# ----------------------------
# 7. Plot Training Loss
# ----------------------------
plt.plot(losses)
plt.title("Training Loss over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.tight_layout()
plt.show()
