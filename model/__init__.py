from torch import nn

__all__ = ["RNN"]


class RNN(nn.Module):
    def __init__(self,
                 num_layers: int,
                 hidden_dim: int,
                 embedding_dim: int,
                 dictionary_size: int):
        super(RNN, self).__init__()
        self.rnn = nn.LSTM(
            embedding_dim, hidden_dim, num_layers, batch_first=True
        )
        self.out = nn.Linear(hidden_dim, dictionary_size)
        self.embedding = nn.Embedding(dictionary_size, embedding_dim)

    def forward(self, sequence, hidden=None):
        batch, length = sequence.shape
        output, hidden = self.rnn(self.embedding(sequence), hidden)
        return self.out(output.reshape((batch * length, -1))), hidden
