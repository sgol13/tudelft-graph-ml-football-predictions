import torch
import torch.nn as nn


class SimpleRNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(SimpleRNNModel, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,  # input shape: (batch, seq, features)
        )

        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """
        x: [batch, seq_len, input_size]
        returns: [batch, output_size]
        """

        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, hn = self.rnn(x, h0)  # out: [batch, seq_len, hidden_size]

        out = out[:, -1, :]  # [batch, hidden_size]
        out = self.fc(out)  # [batch, output_size]

        return out
