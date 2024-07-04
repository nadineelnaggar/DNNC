import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset,DataLoader
from Dyck_Generator_Suzgun_Batch import DyckLanguage

vocab2 = ['(',')']

def encode_sentence_onehot(sentence, dataset='short'):
    max_length = 50
    # seq_lengths = [len(sentence) for sentence in sentences]
    # print(seq_lengths)
    # max_length = max(seq_lengths)
    # print(max_length)
    rep = torch.zeros(1,max_length,len(vocab2))
    lengths = []


    for index, char in enumerate(sentence):
        pos = vocab2.index(char)
        rep[0][index][pos] = 1

    rep.requires_grad_(True)
    return rep


Dyck = DyckLanguage(1,0.5, 0.25)

class SemiDyck1TrainDataset(Dataset):
    def __init__(self):

        self.x = []
        self.y = []
        self.lengths = []
        # self.n_samples = xy.shape[0]
        with open('SemiDyck1_Dataset_train.txt', 'r') as f:
            for line in f:
                line = line.split(",")
                sentence = line[0].strip()
                label = line[1].strip()
                self.x.append(sentence)
                self.y.append(label)
                self.lengths.append(len(sentence))

        # self.x = self.x[:10000]
        # self.y = self.y[:10000]
        #
        # self.x_tensor = []
        # self.y_tensor = []
        # for i in range(len(self.x)):
        #     self.x_tensor.append(encode_sentence_onehot(self.x[i]))
        #     self.y_tensor.append(Dyck.lineToTensorSigmoid(self.y[i], max_len=50))

        self.lengths = self.lengths[:10000]
        self.n_samples = len(self.x)



    def __getitem__(self, index):
        # return self.x[index], self.y[index]
        return {'x': self.x[index], 'y': self.y[index], 'length': self.lengths[index]}
        # return {'x':encode_sentence_onehot(self.x[index]), 'y': Dyck.lineToTensorSigmoid(str(self.y[index]), max_len=50), 'length': self.lengths[index]}
        # return {'x':self.x[index], 'y':self.y[index]}

    def __len__(self):
        return self.n_samples

# dataset = NextTokenPredictionTrainDataset()
# print(len(dataset))


class SemiDyck1ShortTestDataset(Dataset):
    def __init__(self):
        # xy = np.loadtxt('Dyck1_Dataset_Suzgun_train_.txt', delimiter=",")
        # self.x = torch.from_numpy(xy[:,0])
        # self.y = torch.from_numpy(xy[:, [1]])
        self.x = []
        self.y = []
        self.lengths = []
        # self.n_samples = xy.shape[0]
        with open('SemiDyck1_Dataset_short_test.txt', 'r') as f:
            for line in f:
                line = line.split(",")
                sentence = line[0].strip()
                label = line[1].strip()
                self.x.append(sentence)
                self.y.append(label)
                self.lengths.append(len(sentence))

        # self.x = self.x[10000:15000]
        # self.y = self.y[10000:15000]
        # self.lengths = self.lengths[10000:15000]
        self.n_samples = len(self.x)



    def __getitem__(self, index):
        # return self.x[index], self.y[index]
        return {'x': self.x[index], 'y': self.y[index], 'length': self.lengths[index]}

    def __len__(self):
        return self.n_samples
# dataset_test = NextTokenPrediction_Short_Test_Dataset()
# print(len(dataset_test))


class SemiDyck1ValidationDataset(Dataset):
    def __init__(self):
        # xy = np.loadtxt('Dyck1_Dataset_Suzgun_train_.txt', delimiter=",")
        # self.x = torch.from_numpy(xy[:,0])
        # self.y = torch.from_numpy(xy[:, [1]])
        self.x = []
        self.y = []
        self.lengths = []
        # self.n_samples = xy.shape[0]
        with open('SemiDyck1_Dataset_val.txt', 'r') as f:
            for line in f:
                line = line.split(",")
                sentence = line[0].strip()
                label = line[1].strip()
                self.x.append(sentence)
                self.y.append(label)
                self.lengths.append(sentence)

        # self.x = self.x[15000:]
        # self.y = self.y[15000:]
        # self.lengths = self.lengths[15000:]
        self.n_samples = len(self.x)



    def __getitem__(self, index):
        # return self.x[index], self.y[index]
        return {'x': self.x[index], 'y': self.y[index], 'length': self.lengths[index]}

    def __len__(self):
        return self.n_samples

# dataset_val = NextTokenPrediction_Validation_Dataset()
# print(len(dataset_val))


class SemiDyck1TestDataset(Dataset):
    def __init__(self):
        # xy = np.loadtxt('Dyck1_Dataset_Suzgun_train_.txt', delimiter=",")
        # self.x = torch.from_numpy(xy[:,0])
        # self.y = torch.from_numpy(xy[:, [1]])
        self.x = []
        self.y = []
        self.lengths = []
        # self.n_samples = xy.shape[0]
        with open('SemiDyck1_Dataset_test.txt', 'r') as f:
            for line in f:
                line = line.split(",")
                sentence = line[0].strip()
                label = line[1].strip()
                self.x.append(sentence)
                self.y.append(label)
                self.lengths.append(len(sentence))

        # self.x = self.x[:5000]
        # self.y = self.y[:5000]
        # self.lengths = self.lengths[:5000]
        self.n_samples = len(self.x)



    def __getitem__(self, index):
        # return self.x[index], self.y[index]
        return {'x':self.x[index], 'y':self.y[index], 'length':self.lengths[index]}

    def __len__(self):
        return self.n_samples




class SemiDyck1Dataset2000tokens_zigzag(Dataset):
    def __init__(self):

        self.x = []
        self.y = []
        self.max_depths= []


        self.lengths = []
        # self.n_samples = xy.shape[0]
        with open('SemiDyck1_Dataset_zigzag.txt', 'r') as f:
            for line in f:
                line = line.split(",")
                sentence = line[0].strip()
                label = line[1].strip()
                length = line[2].strip()

                self.x.append(sentence)
                self.y.append(label)
                self.lengths.append(length)





        self.n_samples = len(self.x)



    def __getitem__(self, index):
        return {'x': self.x[index], 'y': self.y[index], 'length': self.lengths[index]}

    def __len__(self):
        return self.n_samples



class SemiDyck1Dataset1000tokens(Dataset):
    def __init__(self):
        # xy = np.loadtxt('Dyck1_Dataset_Suzgun_train_.txt', delimiter=",")
        # self.x = torch.from_numpy(xy[:,0])
        # self.y = torch.from_numpy(xy[:, [1]])
        self.x = []
        self.y = []
        self.max_depths= []
        x_new = []
        y_new = []

        self.lengths = []
        # self.n_samples = xy.shape[0]
        with open('SemiDyck1_Dataset_long_test.txt', 'r') as f:
            for line in f:
                line = line.split(",")
                sentence = line[0].strip()
                label = line[1].strip()
                # if len(sentence)==1000:
                #     self.x.append(sentence)
                #     self.y.append(label)
                #     self.lengths.append(len(sentence))

                self.x.append(sentence)
                self.y.append(label)
                self.lengths.append(len(sentence))


        max_depths = []
        timestep_depths = []

        # self.x = self.x[:100]
        # self.y = self.y[:100]
        # self.lengths = self.lengths[:100]



        max_depth, timestep_depth = self.get_timestep_depths(self.x)
        max_depths.append(max_depth)
        timestep_depths.append(timestep_depth)

        self.max_depths = max_depths
        self.timestep_depths = timestep_depths


        self.n_samples = len(self.x)



    def __getitem__(self, index):

        return {'x': self.x[index], 'y': self.y[index], 'length': self.lengths[index]}

    def __len__(self):
        return self.n_samples

    def get_max_depth(self,x):
        max_depth = 0
        # stack = []
        current_depth = 0
        for i in range(len(x)):

            if x[i]=='(':
                current_depth+=1
                if current_depth>max_depth:
                    max_depth=current_depth
            elif x[i]==')':
                current_depth-=1
        return max_depth

    def get_timestep_depths(self,x):
        max_depth = 0
        current_depth = 0
        timestep_depths = []
        for i in range(len(x)):

            if x[i] == '(':
                current_depth += 1
                timestep_depths.append(current_depth)
                if current_depth > max_depth:
                    max_depth = current_depth
            elif x[i] == ')':
                current_depth -= 1
                timestep_depths.append(current_depth)
        return max_depth, timestep_depths

