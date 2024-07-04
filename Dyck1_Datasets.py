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

class NextTokenPredictionTrainDataset(Dataset):
    def __init__(self):
        # xy = np.loadtxt('Dyck1_Dataset_Suzgun_train_.txt', delimiter=",")
        # self.x = torch.from_numpy(xy[:,0])
        # self.y = torch.from_numpy(xy[:, [1]])
        self.x = []
        self.y = []
        self.lengths = []
        # self.n_samples = xy.shape[0]
        with open('Dyck1_Dataset_Suzgun_train_.txt', 'r') as f:
            for line in f:
                line = line.split(",")
                sentence = line[0].strip()
                label = line[1].strip()
                self.x.append(sentence)
                self.y.append(label)
                self.lengths.append(len(sentence))

        self.x = self.x[:10000]
        self.y = self.y[:10000]

        self.x_tensor = []
        self.y_tensor = []
        for i in range(len(self.x)):
            self.x_tensor.append(encode_sentence_onehot(self.x[i]))
            self.y_tensor.append(Dyck.lineToTensorSigmoid(self.y[i], max_len=50))

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


class NextTokenPredictionShortTestDataset(Dataset):
    def __init__(self):
        # xy = np.loadtxt('Dyck1_Dataset_Suzgun_train_.txt', delimiter=",")
        # self.x = torch.from_numpy(xy[:,0])
        # self.y = torch.from_numpy(xy[:, [1]])
        self.x = []
        self.y = []
        self.lengths = []
        # self.n_samples = xy.shape[0]
        with open('Dyck1_Dataset_Suzgun_train_.txt', 'r') as f:
            for line in f:
                line = line.split(",")
                sentence = line[0].strip()
                label = line[1].strip()
                self.x.append(sentence)
                self.y.append(label)
                self.lengths.append(len(sentence))

        self.x = self.x[10000:15000]
        self.y = self.y[10000:15000]
        self.lengths = self.lengths[10000:15000]
        self.n_samples = len(self.x)



    def __getitem__(self, index):
        # return self.x[index], self.y[index]
        return {'x': self.x[index], 'y': self.y[index], 'length': self.lengths[index]}

    def __len__(self):
        return self.n_samples
# dataset_test = NextTokenPrediction_Short_Test_Dataset()
# print(len(dataset_test))


class NextTokenPredictionValidationDataset(Dataset):
    def __init__(self):
        # xy = np.loadtxt('Dyck1_Dataset_Suzgun_train_.txt', delimiter=",")
        # self.x = torch.from_numpy(xy[:,0])
        # self.y = torch.from_numpy(xy[:, [1]])
        self.x = []
        self.y = []
        self.lengths = []
        # self.n_samples = xy.shape[0]
        with open('Dyck1_Dataset_Suzgun_train_.txt', 'r') as f:
            for line in f:
                line = line.split(",")
                sentence = line[0].strip()
                label = line[1].strip()
                self.x.append(sentence)
                self.y.append(label)
                self.lengths.append(sentence)

        self.x = self.x[15000:]
        self.y = self.y[15000:]
        self.lengths = self.lengths[15000:]
        self.n_samples = len(self.x)



    def __getitem__(self, index):
        # return self.x[index], self.y[index]
        return {'x': self.x[index], 'y': self.y[index], 'length': self.lengths[index]}

    def __len__(self):
        return self.n_samples

# dataset_val = NextTokenPrediction_Validation_Dataset()
# print(len(dataset_val))


class NextTokenPredictionLongTestDataset(Dataset):
    def __init__(self):
        # xy = np.loadtxt('Dyck1_Dataset_Suzgun_train_.txt', delimiter=",")
        # self.x = torch.from_numpy(xy[:,0])
        # self.y = torch.from_numpy(xy[:, [1]])
        self.x = []
        self.y = []
        self.lengths = []
        # self.n_samples = xy.shape[0]
        with open('Dyck1_Dataset_Suzgun_test_.txt', 'r') as f:
            for line in f:
                line = line.split(",")
                sentence = line[0].strip()
                label = line[1].strip()
                self.x.append(sentence)
                self.y.append(label)
                self.lengths.append(len(sentence))

        self.x = self.x[:5000]
        self.y = self.y[:5000]
        self.lengths = self.lengths[:5000]
        self.n_samples = len(self.x)



    def __getitem__(self, index):
        # return self.x[index], self.y[index]
        return {'x':self.x[index], 'y':self.y[index], 'length':self.lengths[index]}

    def __len__(self):
        return self.n_samples


class NextTokenPredictionDataset102to500tokens(Dataset):
    def __init__(self):
        # xy = np.loadtxt('Dyck1_Dataset_Suzgun_train_.txt', delimiter=",")
        # self.x = torch.from_numpy(xy[:,0])
        # self.y = torch.from_numpy(xy[:, [1]])
        self.x = []
        self.y = []
        self.lengths = []
        # self.n_samples = xy.shape[0]
        with open('Dyck1_Dataset_Suzgun_102to500tokens.txt', 'r') as f:
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


class NextTokenPredictionDataset502to1000tokens(Dataset):
    def __init__(self):
        # xy = np.loadtxt('Dyck1_Dataset_Suzgun_train_.txt', delimiter=",")
        # self.x = torch.from_numpy(xy[:,0])
        # self.y = torch.from_numpy(xy[:, [1]])
        self.x = []
        self.y = []
        self.lengths = []
        # self.n_samples = xy.shape[0]
        with open('Dyck1_Dataset_Suzgun_502to1000tokens.txt', 'r') as f:
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



class NextTokenPredictionDataset990to1000tokens(Dataset):
    def __init__(self):
        # xy = np.loadtxt('Dyck1_Dataset_Suzgun_train_.txt', delimiter=",")
        # self.x = torch.from_numpy(xy[:,0])
        # self.y = torch.from_numpy(xy[:, [1]])
        self.x = []
        self.y = []
        self.lengths = []
        # self.n_samples = xy.shape[0]
        with open('Dyck1_Dataset_Suzgun_502to1000tokens.txt', 'r') as f:
            for line in f:
                line = line.split(",")
                sentence = line[0].strip()
                label = line[1].strip()
                if len(sentence)>=990:
                    self.x.append(sentence)
                    self.y.append(label)
                    self.lengths.append(len(sentence))

        # self.x = self.x[:5000]
        # self.y = self.y[:5000]
        # self.lengths = self.lengths[:5000]

        # self.x = self.x[:600]
        # self.y = self.y[:600]
        # self.lengths = self.lengths[:600]

        # self.x = self.x[:1000]
        # self.y = self.y[:1000]
        # self.lengths = self.lengths[:1000]

        self.n_samples = len(self.x)



    def __getitem__(self, index):
        # return self.x[index], self.y[index]
        return {'x':self.x[index], 'y':self.y[index], 'length':self.lengths[index]}

    def __len__(self):
        return self.n_samples

# dataset_long = NextTokenPredictionLongTestDataset()
# print(len(dataset_long))
# print(dataset_long[20])
# print(dataset_long[20]['x'])


class NextTokenPredictionDataset2000tokens(Dataset):
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
        with open('Dyck1_Dataset_Suzgun_502to1000tokens.txt', 'r') as f:
            for line in f:
                line = line.split(",")
                sentence = line[0].strip()
                label = line[1].strip()
                if len(sentence)==1000:
                    self.x.append(sentence)
                    self.y.append(label)
                    # self.lengths.append(len(sentence))

        # self.x = self.x[:5000]
        # self.y = self.y[:5000]
        # self.lengths = self.lengths[:5000]

        # self.x = self.x[:600]
        # self.y = self.y[:600]
        # self.lengths = self.lengths[:600]

        # self.x = self.x[:1000]
        # self.y = self.y[:1000]
        # self.lengths = self.lengths[:1000]

        max_depths = []
        timestep_depths = []

        for i in range(len(self.x)):
            for j in range(len(self.x)):
                x_val = self.x[i]+self.x[j]
                y_val = self.y[i]+self.y[j]
                # x_new.append(self.x[i]+self.x[i+1])
                # y_new.append(self.y[i]+self.y[i+1])
                x_new.append(x_val)
                y_new.append(y_val)
                self.lengths.append(len(x_val))
                # x_max_depth = self.get_max_depth(x_val)
                # max_depth.append(x_max_depth)

                max_depth, timestep_depth = self.get_timestep_depths(self.x)
                max_depths.append(max_depth)
                timestep_depths.append(timestep_depth)

            # x_val2 = self.x[i+1]+self.x[i]
            # y_val2 = self.y[i+1]+self.y[i]
            # # x_new.append(self.x[i+1]+self.x[i])
            # # y_new.append(self.y[i+1]+self.y[i])
            # x_max_depth2 = self.get_max_depth(x_val2)
            # max_depth.append(x_max_depth2)




        self.x= x_new
        self.y = y_new
        # self.max_depth = max_depth
        self.max_depths = max_depths
        self.timestep_depths = timestep_depths


        self.n_samples = len(self.x)



    def __getitem__(self, index):
        # return self.x[index], self.y[index]
        # return {'x':self.x[index], 'y':self.y[index], 'length':self.lengths[index], 'max_depth':self.max_depths[index], 'timestep_depths':self.timestep_depths[index]}
        # return {'x':self.x[index], 'y':self.y[index], 'length':self.lengths[index], 'max_depth':self.max_depths[index]}
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


class NextTokenPredictionDataset2000tokens_nested(Dataset):
    def __init__(self):

        self.x = []
        self.y = []
        self.max_depths= []


        self.lengths = []
        # self.n_samples = xy.shape[0]
        with open('Dyck1_Dataset_Suzgun_2000tokens_nested.txt', 'r') as f:
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


class NextTokenPredictionDataset2000tokens_zigzag(Dataset):
    def __init__(self):

        self.x = []
        self.y = []
        self.max_depths= []


        self.lengths = []
        # self.n_samples = xy.shape[0]
        with open('Dyck1_Dataset_Suzgun_2000tokens_zigzag.txt', 'r') as f:
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


class NextTokenPredictionTrainDataset_SAMPLE(Dataset):
    def __init__(self):
        # xy = np.loadtxt('Dyck1_Dataset_Suzgun_train_.txt', delimiter=",")
        # self.x = torch.from_numpy(xy[:,0])
        # self.y = torch.from_numpy(xy[:, [1]])
        self.x = []
        self.y = []
        self.lengths = []
        # self.n_samples = xy.shape[0]
        with open('Dyck1_Dataset_Suzgun_train_Shuffle.txt', 'r') as f:
            for line in f:
                line = line.split(",")
                sentence = line[0].strip()
                label = line[1].strip()
                self.x.append(sentence)
                self.y.append(label)
                self.lengths.append(len(sentence))

        self.x = self.x[:1000]
        self.y = self.y[:1000]

        self.x_tensor = []
        self.y_tensor = []
        for i in range(len(self.x)):
            self.x_tensor.append(encode_sentence_onehot(self.x[i]))
            self.y_tensor.append(Dyck.lineToTensorSigmoid(self.y[i], max_len=50))

        self.lengths = self.lengths[:1000]
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


class NextTokenPredictionShortTestDataset_SAMPLE(Dataset):
    def __init__(self):
        # xy = np.loadtxt('Dyck1_Dataset_Suzgun_train_.txt', delimiter=",")
        # self.x = torch.from_numpy(xy[:,0])
        # self.y = torch.from_numpy(xy[:, [1]])
        self.x = []
        self.y = []
        self.lengths = []
        # self.n_samples = xy.shape[0]
        with open('Dyck1_Dataset_Suzgun_train_Shuffle.txt', 'r') as f:
            for line in f:
                line = line.split(",")
                sentence = line[0].strip()
                label = line[1].strip()
                self.x.append(sentence)
                self.y.append(label)
                self.lengths.append(len(sentence))

        self.x = self.x[10000:10500]
        self.y = self.y[10000:10500]
        self.lengths = self.lengths[10000:10500]
        self.n_samples = len(self.x)



    def __getitem__(self, index):
        # return self.x[index], self.y[index]
        return {'x': self.x[index], 'y': self.y[index], 'length': self.lengths[index]}

    def __len__(self):
        return self.n_samples
# dataset_test = NextTokenPrediction_Short_Test_Dataset()
# print(len(dataset_test))


class NextTokenPredictionValidationDataset_SAMPLE(Dataset):
    def __init__(self):
        # xy = np.loadtxt('Dyck1_Dataset_Suzgun_train_.txt', delimiter=",")
        # self.x = torch.from_numpy(xy[:,0])
        # self.y = torch.from_numpy(xy[:, [1]])
        self.x = []
        self.y = []
        self.lengths = []
        # self.n_samples = xy.shape[0]
        with open('Dyck1_Dataset_Suzgun_train_Shuffle.txt', 'r') as f:
            for line in f:
                line = line.split(",")
                sentence = line[0].strip()
                label = line[1].strip()
                self.x.append(sentence)
                self.y.append(label)
                self.lengths.append(sentence)

        self.x = self.x[15000:15500]
        self.y = self.y[15000:15500]
        self.lengths = self.lengths[15000:15500]
        self.n_samples = len(self.x)



    def __getitem__(self, index):
        # return self.x[index], self.y[index]
        return {'x': self.x[index], 'y': self.y[index], 'length': self.lengths[index]}

    def __len__(self):
        return self.n_samples

# dataset_val = NextTokenPrediction_Validation_Dataset()
# print(len(dataset_val))


class NextTokenPredictionLongTestDataset_SAMPLE(Dataset):
    def __init__(self):
        # xy = np.loadtxt('Dyck1_Dataset_Suzgun_train_.txt', delimiter=",")
        # self.x = torch.from_numpy(xy[:,0])
        # self.y = torch.from_numpy(xy[:, [1]])
        self.x = []
        self.y = []
        self.lengths = []
        # self.n_samples = xy.shape[0]
        with open('Dyck1_Dataset_Suzgun_test_.txt', 'r') as f:
            for line in f:
                line = line.split(",")
                sentence = line[0].strip()
                label = line[1].strip()
                self.x.append(sentence)
                self.y.append(label)
                self.lengths.append(len(sentence))

        self.x = self.x[:500]
        self.y = self.y[:500]
        self.lengths = self.lengths[:500]
        self.n_samples = len(self.x)



    def __getitem__(self, index):
        # return self.x[index], self.y[index]
        return {'x':self.x[index], 'y':self.y[index], 'length':self.lengths[index]}

    def __len__(self):
        return self.n_samples


class NextTokenPredictionDataset1000tokens(Dataset):
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
        with open('Dyck1_Dataset_Suzgun_502to1000tokens.txt', 'r') as f:
            for line in f:
                line = line.split(",")
                sentence = line[0].strip()
                label = line[1].strip()
                if len(sentence)==1000:
                    self.x.append(sentence)
                    self.y.append(label)
                    self.lengths.append(len(sentence))

        # self.x = self.x[:5000]
        # self.y = self.y[:5000]
        # self.lengths = self.lengths[:5000]

        # self.x = self.x[:600]
        # self.y = self.y[:600]
        # self.lengths = self.lengths[:600]

        # self.x = self.x[:1000]
        # self.y = self.y[:1000]
        # self.lengths = self.lengths[:1000]
        with open('Dyck1_Dataset_Suzgun_1000tokens.txt', 'r') as f:
            for line in f:
                line = line.split(",")
                sentence = line[0].strip()
                label = line[1].strip()
                if len(sentence)==1000:
                    if sentence not in self.x:
                        self.x.append(sentence)
                        self.y.append(label)
                        self.lengths.append(len(sentence))

        with open('Dyck1_Dataset_Suzgun_1000tokens_2.txt', 'r') as f:
            for line in f:
                line = line.split(",")
                sentence = line[0].strip()
                label = line[1].strip()
                if len(sentence)==1000:
                    if sentence not in self.x:
                        self.x.append(sentence)
                        self.y.append(label)
                        self.lengths.append(len(sentence))

        with open('Dyck1_Dataset_Suzgun_1000tokens_3.txt', 'r') as f:
            for line in f:
                line = line.split(",")
                sentence = line[0].strip()
                label = line[1].strip()
                if len(sentence)==1000:
                    if sentence not in self.x:
                        self.x.append(sentence)
                        self.y.append(label)
                        self.lengths.append(len(sentence))

        with open('Dyck1_Dataset_Suzgun_1000tokens_4.txt', 'r') as f:
            for line in f:
                line = line.split(",")
                sentence = line[0].strip()
                label = line[1].strip()
                if len(sentence)==1000:
                    if sentence not in self.x:
                        self.x.append(sentence)
                        self.y.append(label)
                        self.lengths.append(len(sentence))

        with open('Dyck1_Dataset_Suzgun_1000tokens_5.txt', 'r') as f:
            for line in f:
                line = line.split(",")
                sentence = line[0].strip()
                label = line[1].strip()
                if len(sentence)==1000:
                    if sentence not in self.x:
                        self.x.append(sentence)
                        self.y.append(label)
                        self.lengths.append(len(sentence))

        with open('Dyck1_Dataset_Suzgun_1000tokens_5 (1).txt', 'r') as f:
            for line in f:
                line = line.split(",")
                sentence = line[0].strip()
                label = line[1].strip()
                if len(sentence)==1000:
                    if sentence not in self.x:
                        self.x.append(sentence)
                        self.y.append(label)
                        self.lengths.append(len(sentence))


        with open('Dyck1_Dataset_Suzgun_1000tokens_6.txt', 'r') as f:
            for line in f:
                line = line.split(",")
                sentence = line[0].strip()
                label = line[1].strip()
                if len(sentence)==1000:
                    if sentence not in self.x:
                        self.x.append(sentence)
                        self.y.append(label)
                        self.lengths.append(len(sentence))

        max_depths = []
        timestep_depths = []

        self.x = self.x[:100]
        self.y = self.y[:100]
        self.lengths = self.lengths[:100]



        max_depth, timestep_depth = self.get_timestep_depths(self.x)
        max_depths.append(max_depth)
        timestep_depths.append(timestep_depth)

            # x_val2 = self.x[i+1]+self.x[i]
            # y_val2 = self.y[i+1]+self.y[i]
            # # x_new.append(self.x[i+1]+self.x[i])
            # # y_new.append(self.y[i+1]+self.y[i])
            # x_max_depth2 = self.get_max_depth(x_val2)
            # max_depth.append(x_max_depth2)





        # self.max_depth = max_depth
        self.max_depths = max_depths
        self.timestep_depths = timestep_depths


        self.n_samples = len(self.x)



    def __getitem__(self, index):
        # return self.x[index], self.y[index]
        # return {'x':self.x[index], 'y':self.y[index], 'length':self.lengths[index], 'max_depth':self.max_depths[index], 'timestep_depths':self.timestep_depths[index]}
        # return {'x':self.x[index], 'y':self.y[index], 'length':self.lengths[index], 'max_depth':self.max_depths[index]}
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


class Dyck1RegressionDataset_train(Dataset):
    def __int__(self):
        self.x = []
        self.y = []
        self.lengths = []
        # pass

        with open('Dyck1_Dataset_Suzgun_train_.txt', 'r') as f:
            for line in f:
                line = line.split(",")
                sentence = line[0].strip()
                label = line[1].strip()
                self.x.append(sentence)
                # y.append(label)
                self.lengths.append(len(self.x))

        for j in range(len(self.x)):
            elem = self.x[j]
            depth = 0
            print(elem)
            for i in range(len(elem)):
                print(elem[i])
                if elem[i] == '(':
                    depth += 1
                elif elem[i] == ')':
                    depth -= 1
                print(depth)
    def __getitem__(self, item):
        pass

    def __len__(self):
        pass



