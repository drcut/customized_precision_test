import torch
import torch.nn as nn
import torch.nn.functional as F
from CPDtorch.quant import float_quantize

class Precision_cast(nn.Module):
    def __init__(self, use_APS=False):
        super(Precision_cast, self).__init__()
        self.use_APS = use_APS
    def forward(self,x):
        if not self.use_APS:
            return float_quantize(x, 5,2)
        else:
            shift_factor = 15 - torch.log2(torch.abs(x).max()).ceil().detach().cpu().numpy()
            return float_quantize(x*(2**shift_factor), 5,2)

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, 5, padding=2),
            Precision_cast(True),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 5,padding=2),
            Precision_cast(True),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.fc1 = nn.Sequential(
            nn.Linear(3136, 1024),
            Precision_cast(True),
            nn.ReLU(),
            nn.Dropout(0.4)
        )
        self.fc2 = nn.Linear(1024, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x
