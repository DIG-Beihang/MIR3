import torch.nn as nn
import torch.nn.functional as F


class RNNBase(nn.Module):
    def __init__(self, input_shape, hidden_dim, output_shape):
        super(RNNBase, self).__init__()
        self.hidden_dim = hidden_dim

        self.fc1 = nn.Linear(input_shape, hidden_dim)
        self.rnn = nn.GRUCell(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_shape)

    def init_hidden(self):
        return self.fc1.weight.new(1, self.hidden_dim)

    def forward(self, inputs, hidden_state):
        assert len(inputs.shape) == len(hidden_state.shape)
        ori_shape = inputs.shape[:-1]

        x = F.relu(self.fc1(inputs.reshape(-1, inputs.shape[-1])))
        h = self.rnn(x, hidden_state.reshape(-1, self.hidden_dim))
        out = self.fc3(h)

        return out.view(*ori_shape, -1), h.view(*ori_shape, self.hidden_dim)
    
    def representation(self, inputs, hidden_state=None):
        assert len(inputs.shape) == len(hidden_state.shape)
        ori_shape = inputs.shape[:-1]

        x = F.relu(self.fc1(inputs.reshape(-1, inputs.shape[-1])))
        h = self.rnn(x, hidden_state.reshape(-1, self.hidden_dim))
        return h.view(*ori_shape, self.hidden_dim)
