from random import shuffle
import torch
import torch.nn as nn
import torch.optim as optim
from loader import loader
from net import net
import torch.utils.data as data_utils

cla = 1
first = 1
critetion = nn.BCELoss()
def train(model,epoch):
    optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum = 0.9,weight_decay=0.0005)
    data = data_utils.DataLoader(loader('train',True,cla),batch_size=32,shuffle=False)
    loss_train = 0
    correct = 0
    for idx in range(10):
        for _, (inputs, targets) in enumerate(data,0):
            y_pred = model(inputs)
            # print(torch.max(targets,dim=1)[1].data)
            # print(torch.max(y_pred,dim=1)[1].data)
            optimizer.zero_grad()
            loss = critetion(y_pred, targets)
            loss_train += loss.item()
            
            loss.backward()
            optimizer.step()
    print('[%d], train loss is %2.5f'% (epoch ,loss_train / (10 * 5980 * 7)))
def test(model):
    data = data_utils.DataLoader(loader('test',False,cla),batch_size=20,shuffle=False)
    loss_test = 0
    correct = 0
    for _, (inputs, targets) in enumerate(data, 0):
        y_pred = model(inputs)
        # print(y_pred)
        pridicted = torch.max(y_pred, dim=1)[1].data
        # print(pridicted)
        # print(targets.shape)
        correct += (pridicted == torch.max(targets,dim=1)[1]).sum().item()
    print('Accuracy on testt set: %d %%'  %  (100 * correct / (2530 * 7)))

if __name__=='__main__':
    model = net(first)
    for epoch in range(10):
        train(model,epoch)
        test(model)