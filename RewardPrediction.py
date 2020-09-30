import phyre
import torch.nn as nn
import torch.nn.functional as F


class RewardPrediction(nn.Module):

    def __init__(self):
        super(RewardPrediction, self).__init__()
        self.lstm1 = nn.LSTMCell(1024, 1)

    def forward(self, x):
        x = self.lstm1(x)
        return x



