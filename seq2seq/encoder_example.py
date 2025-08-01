import torch
import torch.nn as nn

# Suppose we have a vocabulary size of 10,000 and embedding dim of 300
vocab_size = 10000
embedding_dim = 300
hidden_dim = 512


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)

    def forward(self, input_seq):
        # input_seq: [seq_len, batch_size]
        embedded = self.embedding(input_seq)
        # embedded: [seq_len, batch_size, embedding_dim]

        outputs, (hidden, cell) = self.lstm(embedded)
        # outputs: [seq_len, batch_size, hidden_dim]
        # hidden, cell: [1, batch_size, hidden_dim]

        return hidden, cell


# Assume we have the input: "I am happy today" → token IDs
# shape: [4, 1] (seq_len, batch_size)
sentence_ids = torch.tensor([[23, 45, 67, 89]]).T

encoder = Encoder()
hidden, cell = encoder(sentence_ids)

print(hidden.shape)  # [1, 1, 512]
