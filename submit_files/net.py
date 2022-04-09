from turtle import forward
import torch
import torch.utils.data as data_utils
import torch.nn as nn
class net(torch.nn.Module):
    def __init__(self):
        super(net, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(10, 16), nn.ReLU())
        self.layer2 = nn.Sequential(nn.Linear(16, 16), nn.ReLU())
        self.layer3 = nn.Sequential(nn.Linear(16, 7),nn.Sigmoid())

    def forward(self, x):
        x = x.to(torch.float32)
        x = self.layer1(x)
        # x = self.layer2(x)
        x = self.layer3(x)

        return x
    