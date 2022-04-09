import numpy as np
import os
import random
import torch.utils.data as data_utils
import torch
import matplotlib.pyplot as plt
import seaborn
import pandas as pd
class loader(data_utils.Dataset):
    def __init__(self,filename,train, cla) -> None:
        file = os.path.dirname(os.path.dirname(__file__))
        print(file)
        file = file + '/data/'+filename
        x_data = []
        y_data = []
        mean_d = []
        var_s = []
        S = []
        for i in range(1,8):
            filepath = file + '/' + str(i) + '.npy'
            data = np.load(filepath)
            if cla == 1:
                data = data[:,0,:]
            elif cla == 2:
                data = data[:,1,:]
            else:
                data = np.hstack((data[:,0,:], data[:,1,:]))
            label = np.zeros((data.shape[0], 7), dtype=np.float32)
            mean_d.append(np.mean(data,axis=(0)))
            # count = np.dot((data- np.mean(data,axis=(0))).transpose() , (data- np.mean(data,axis=(0)))) / (data.shape[0])
            # S.append(count)
            var_s.append(np.mean(abs(data - np.mean(data,axis=(0))** 2), axis=0))
            # print(np.mean(data,axis=(0)).shape)
            label[:,[i-1]] = 1.
            x_data.append(data.copy())
            y_data.append(label.copy())
        # Sw = sum(S) / 7
        # print(Sw)
        mean_d = np.array(mean_d)
        var_s = np.array(var_s)
        x_data = np.concatenate(x_data, axis=0)
        y_data = np.concatenate(y_data, axis= 0)
        # mean = np.mean(x_data,axis=(0))
        # Sb = np.dot((mean - mean_d).transpose() , (mean - mean_d)) / 7

        # print(Sb)
        # print(x_data.shape)
        x_data = (x_data - x_data.min(0)) / (x_data.max(0) - x_data.min(0))
        t = np.split(x_data, 7, axis=0)
        for x in range(7):
            S.append(np.dot((t[x] - mean_d[x]).transpose(), (t[x] - mean_d[x])) / t[x].shape[0])
        Sw = sum(S) / 7
        mean = np.mean(mean_d,axis=(0))
        Sb = np.dot((mean - mean_d).transpose() , (mean - mean_d)) / 7
        S = np.dot(np.linalg.inv(Sw),Sb)
        eigenvalue, featurevector = np.linalg.eig(S)
        W = featurevector[0:4,:]
        # print(eigenvalue.shape)
        # print(featurevector.shape)
        # # print(Sw.shape)
        # print(x_data.shape)
        if train:
            data = list(zip(x_data, y_data))
            random.shuffle(data)
            x_data, y_data = zip(*data)
        # x_data = np.dot(W, np.array(x_data).transpose()).transpose()
        x_data = np.array(x_data)
        y_data = np.array(y_data)
        # fig = plt.figure()
        # t = pd.DataFrame(x_data)
        # plt.subplot2grid((2, 3), (0, 0))
        # t[0].value_counts().plot(kind='bar')
        # plt.show()
        self.x_data = torch.from_numpy(x_data)
        self.y_data = torch.from_numpy(y_data)
        # print(self.x_data.shape)
        # print(self.y_data.shape)
    def __len__(self):
        return self.x_data.shape[0]
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]
if __name__ == "__main__":
    t = loader('test',False,3)

