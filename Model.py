import torch
import torch.nn as nn
import torch.nn.functional as F


class IrisModel(nn.Module):

    def __init__(self, in_features=0, out_feature=0, hidden_features_1=0, hidden_features_2=0):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features_1)
        self.fc2 = nn.Linear(hidden_features_1, hidden_features_2)
        self.out = nn.Linear(hidden_features_2, out_feature)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)
        return x