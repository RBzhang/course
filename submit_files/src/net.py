from turtle import forward
import torch
import torch.utils.data as data_utils
import torch.nn as nn
class net(torch.nn.Module):
    def __init__(self, T):
        super(net, self).__init__()
        first = 0
        if T == 1:
            first = 5
        else:
            first =  10
        self.layer1 = nn.Sequential(nn.Linear(first, 16), nn.ReLU())
        self.layer2 = nn.Sequential(nn.Linear(16, 16), nn.ReLU())
        self.layer3 = nn.Sequential(nn.Linear(16, 7),nn.Sigmoid())

    def forward(self, x):
        x = x.to(torch.float32)
        x = self.layer1(x)
        # x = self.layer2(x)
        x = self.layer3(x)

        return x
if __name__ == '__main__':
    x = torch.ones(5,dtype=torch.float32)
    y = torch.ones(6,dtype=torch.float32)
    tt = [x,y]
    ee = []
    ee.append(torch.cat(tt,dim=0))
    ee.append(torch.cat(tt,dim=0))
    print(torch.stack(ee,dim=1))