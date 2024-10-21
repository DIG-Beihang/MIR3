import torch.nn as nn
import torch.nn.functional as F


class MLPBase(nn.Module):
    def __init__(self, input_shape, hidden_dim, output_shape):
        super(MLPBase, self).__init__()

        self.fc1 = nn.Linear(input_shape, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_shape)

    def init_hidden(self):
        return None

    def forward(self, inputs, hidden_state=None):
        x = F.relu(self.fc1(inputs))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x, None

    def representation(self, inputs, hidden_state=None):
        x = F.relu(self.fc1(inputs))
        x = F.relu(self.fc2(x))
        return x