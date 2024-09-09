from torch import nn
import torch.nn.functional as F
import torch
import sys

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=128):
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.fc1 = nn.Linear(self.input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, self.output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        y = self.fc3(x)
        y = torch.sigmoid(y)
        #return x, y
        return y