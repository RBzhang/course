from random import shuffle
from re import A
from traceback import print_tb
import torch
import torch.nn as nn
import torch.optim as optim
from loader import loader
from loader import loader_e
from net import net
import torch.utils.data as data_utils
import numpy as np
import os
critetion = nn.BCELoss()
def train(model,epoch,cla):
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
def test(model,cla):
    data = data_utils.DataLoader(loader('test',False,cla),batch_size=20,shuffle=False)
    loss_test = 0
    correct = 0
    kap = torch.zeros((7,7),dtype=torch.float32)
    for _, (inputs, targets) in enumerate(data, 0):
        y_pred = model(inputs)
        # print(y_pred)
        pridicted = torch.max(y_pred, dim=1)[1].data
        tar = torch.max(targets,dim=1)[1].data
        for i in range(tar.shape[0]):
            kap[pridicted[i],tar[i]] += 1
        # print(pridicted)
        # print(targets.shape)
        correct += (pridicted == tar).sum().item()
    a = 0
    for i in range(7):
        a += kap[i,:].sum().item() * kap[:,i].sum().item()
    b = kap.sum().item()
    pc = a / (b * b)
    print('kappa %2.5f' % ((correct / (2530 * 7) - pc / (2530 * 7)) / (1 - pc / (2530 * 7))))
    print('Accuracy on testt set: %d %%'  %  (100 * correct / (2530 * 7)))

if __name__=='__main__':
    result = []
    model = net(1)
    for epoch in range(7):
        train(model,epoch,1)
        test(model,1)
    exam = data_utils.DataLoader(loader_e(1),batch_size=20,shuffle=False)
    intos = []
    for _, (inputs, targets) in enumerate(exam, 0):
        y_pred = model(inputs)
        pridicted = torch.max(y_pred, dim=1)[1].data
        intos.append(pridicted)
    result.append(torch.cat(intos,dim=0))
    model = net(1)
    for epoch in range(7):
        train(model,epoch,2)
        test(model,2)
    exam = data_utils.DataLoader(loader_e(2),batch_size=20,shuffle=False)
    intos = []
    for _, (inputs, targets) in enumerate(exam, 0):
        y_pred = model(inputs)
        pridicted = torch.max(y_pred, dim=1)[1].data
        intos.append(pridicted)
    result.append(torch.cat(intos,dim=0))
    model = net(2)
    for epoch in range(7):
        train(model,epoch,3)
        test(model,3)
    exam = data_utils.DataLoader(loader_e(3),batch_size=20,shuffle=False)
    intos = []
    for _, (inputs, targets) in enumerate(exam, 0):
        y_pred = model(inputs)
        pridicted = torch.max(y_pred, dim=1)[1].data
        intos.append(pridicted)
    result.append(torch.cat(intos,dim=0))
    result = torch.stack(result,dim=1).numpy()
    print(result.shape)
    file = os.path.dirname(os.path.dirname(__file__)) + '/D202180803.npy'
    np.save(file,result)