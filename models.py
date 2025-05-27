import torch
import torch.nn as nn
from DNNC import DNNC, DNNCNN, DNNCNNNoFalsePop,DFRCNN
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


if torch.cuda.is_available():
    device=torch.device("cuda")
elif torch.backends.mps.is_available():
    device=torch.device("mps")
else:
    device=torch.device("cpu")

print(device)

class VanillaLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, batch_size, output_size, output_activation='Sigmoid'):
        super(VanillaLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_size=batch_size
        self.output_size = output_size
        self.output_activation = output_activation
        self.model_name = 'VanillaLSTM'

        self.vocab = {'<PAD>': 0, '(':1, ')':2}
        self.tags = {'<PAD>':0, '0':1, '1':2}
        self.nb_tags = len(self.vocab)-1
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, length):
        x = pack_padded_sequence(x, length, batch_first=True)
        h0 = self.init_hidden()

        x, h0 = self.lstm(x, h0)

        x, _ = pad_packed_sequence(x, batch_first=True)

        x = x.contiguous()

        x = x.view(-1, x.shape[2])

        x = self.fc2(x)

        x = self.sigmoid(x).view(-1, self.output_size)

        return x

    def init_hidden(self):
        return (torch.zeros(self.num_layers, self.batch_size, self.hidden_size).to(device),
                torch.zeros(self.num_layers, self.batch_size, self.hidden_size).to(device))


    def mask(self, Y_hat, Y, X_lengths):

        Y_hat_out = torch.zeros(Y_hat.shape)
        max_batch_length = max(X_lengths)

        for i in range(self.batch_size):
            Y_hat_out[i*max_batch_length:(i*max_batch_length+X_lengths[i])] = Y_hat[i*max_batch_length:(i*max_batch_length+X_lengths[i])]

        return Y_hat_out.to(device)




class VanillaRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, batch_size, output_size, output_activation='Sigmoid'):
        super(VanillaRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_size=batch_size
        self.output_size = output_size
        self.output_activation = output_activation
        self.model_name = 'VanillaRNN'

        self.vocab = {'<PAD>': 0, '(':1, ')':2}
        self.tags = {'<PAD>':0, '0':1, '1':2}
        self.nb_tags = len(self.vocab)-1
        self.rnn = nn.RNN(input_size, hidden_size, num_layers=num_layers)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, length):
        x = pack_padded_sequence(x, length, batch_first=True)
        h0 = self.init_hidden()

        x, h0 = self.rnn(x, h0)

        x, _ = pad_packed_sequence(x, batch_first=True)

        x = x.contiguous()

        x = x.view(-1, x.shape[2])

        x = self.fc2(x)

        x = self.sigmoid(x).view(-1, self.output_size)

        return x

    def init_hidden(self):
        return torch.zeros(self.num_layers, self.batch_size, self.hidden_size).to(device)




    def mask(self, Y_hat, Y, X_lengths):

        Y_hat_out = torch.zeros(Y_hat.shape)
        max_batch_length = max(X_lengths)


        for i in range(self.batch_size):
            Y_hat_out[i*max_batch_length:(i*max_batch_length+X_lengths[i])] = Y_hat[i*max_batch_length:(i*max_batch_length+X_lengths[i])]

        return Y_hat_out.to(device)



class VanillaReLURNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, batch_size, output_size, output_activation='Sigmoid'):
        super(VanillaReLURNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_size=batch_size
        self.output_size = output_size
        self.output_activation = output_activation
        self.model_name = 'VanillaReLURNN'

        self.vocab = {'<PAD>': 0, '(':1, ')':2}
        self.tags = {'<PAD>':0, '0':1, '1':2}
        self.nb_tags = len(self.vocab)-1
        self.rnn = nn.RNN(input_size, hidden_size, num_layers=num_layers, nonlinearity='relu')
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, length):
        x = pack_padded_sequence(x, length, batch_first=True)
        h0 = self.init_hidden()

        x, h0 = self.rnn(x, h0)

        x, _ = pad_packed_sequence(x, batch_first=True)

        x = x.contiguous()

        x = x.view(-1, x.shape[2])

        x = self.fc2(x)

        x = self.sigmoid(x).view(-1, self.output_size)

        return x

    def init_hidden(self):
        return torch.zeros(self.num_layers, self.batch_size, self.hidden_size).to(device)



    def mask(self, Y_hat, Y, X_lengths):

        Y_hat_out = torch.zeros(Y_hat.shape)
        max_batch_length = max(X_lengths)



        for i in range(self.batch_size):
            Y_hat_out[i*max_batch_length:(i*max_batch_length+X_lengths[i])] = Y_hat[i*max_batch_length:(i*max_batch_length+X_lengths[i])]

        return Y_hat_out.to(device)



class VanillaGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, batch_size, output_size, output_activation='Sigmoid'):
        super(VanillaGRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_size=batch_size
        self.output_size = output_size
        self.output_activation = output_activation
        self.model_name = 'VanillaGRU'

        self.vocab = {'<PAD>': 0, '(':1, ')':2}
        self.tags = {'<PAD>':0, '0':1, '1':2}
        self.nb_tags = len(self.vocab)-1
        self.gru = nn.GRU(input_size, hidden_size, num_layers=num_layers)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, length):
        x = pack_padded_sequence(x, length, batch_first=True)
        h0 = self.init_hidden()

        x, h0 = self.gru(x, h0)

        x, _ = pad_packed_sequence(x, batch_first=True)

        x = x.contiguous()

        x = x.view(-1, x.shape[2])

        x = self.fc2(x)

        x = self.sigmoid(x).view(-1, self.output_size)

        return x

    def init_hidden(self):
        return torch.zeros(self.num_layers, self.batch_size, self.hidden_size).to(device)



    def mask(self, Y_hat, Y, X_lengths):

        Y_hat_out = torch.zeros(Y_hat.shape)
        max_batch_length = max(X_lengths)


        for i in range(self.batch_size):
            Y_hat_out[i*max_batch_length:(i*max_batch_length+X_lengths[i])] = Y_hat[i*max_batch_length:(i*max_batch_length+X_lengths[i])]


        return Y_hat_out.to(device)





class VanillaReLURNN_NoBias(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, batch_size, output_size, output_activation='Sigmoid'):
        super(VanillaReLURNN_NoBias, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_size=batch_size
        self.output_size = output_size
        self.output_activation = output_activation
        self.model_name = 'VanillaReLURNN_NoBias'

        self.vocab = {'<PAD>': 0, '(':1, ')':2}
        self.tags = {'<PAD>':0, '0':1, '1':2}
        self.nb_tags = len(self.vocab)-1
        self.rnn = nn.RNN(input_size, hidden_size, num_layers=num_layers, nonlinearity='relu', bias=False)

        self.fc2 = nn.Linear(hidden_size, output_size) #output size 1 for this layer to work without bias
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, length):
        x = pack_padded_sequence(x, length, batch_first=True)
        h0 = self.init_hidden()

        x, h0 = self.rnn(x, h0)

        x, _ = pad_packed_sequence(x, batch_first=True)

        x = x.contiguous()

        x = x.view(-1, x.shape[2])

        x = self.fc2(x)

        x = self.sigmoid(x).view(-1, self.output_size)

        return x

    def init_hidden(self):
        return torch.zeros(self.num_layers, self.batch_size, self.hidden_size).to(device)




    def mask(self, Y_hat, Y, X_lengths):

        Y_hat_out = torch.zeros(Y_hat.shape)
        max_batch_length = max(X_lengths)


        for i in range(self.batch_size):
            Y_hat_out[i*max_batch_length:(i*max_batch_length+X_lengths[i])] = Y_hat[i*max_batch_length:(i*max_batch_length+X_lengths[i])]



        return Y_hat_out.to(device)


class VanillaReLURNN_Tanh(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, batch_size, output_size, output_activation='tanh'):
        super(VanillaReLURNN_Tanh, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_size=batch_size
        self.output_size = output_size
        self.output_activation = output_activation
        self.model_name = 'VanillaReLURNN_Tanh'

        self.vocab = {'<PAD>': 0, '(':1, ')':2}
        self.tags = {'<PAD>':0, '0':1, '1':2}
        self.nb_tags = len(self.vocab)-1
        self.rnn = nn.RNN(input_size, hidden_size, num_layers=num_layers, nonlinearity='relu', bias=False)
        self.fc2 = nn.Linear(hidden_size, output_size, bias=False) #output size 1 for this layer to work without bias
        self.tanh = nn.Tanh()

    def forward(self, x, length):
        x = pack_padded_sequence(x, length, batch_first=True)
        h0 = self.init_hidden()

        x, h0 = self.rnn(x, h0)

        x, _ = pad_packed_sequence(x, batch_first=True)

        x = x.contiguous()

        x = x.view(-1, x.shape[2])

        x = self.fc2(x)



        x = self.tanh(x).view(-1, self.output_size)

        return x

    def init_hidden(self):
        return torch.zeros(self.num_layers, self.batch_size, self.hidden_size).to(device)



    def mask(self, Y_hat, Y, X_lengths):

        Y_hat_out = torch.zeros(Y_hat.shape)
        max_batch_length = max(X_lengths)


        for i in range(self.batch_size):
            Y_hat_out[i*max_batch_length:(i*max_batch_length+X_lengths[i])] = Y_hat[i*max_batch_length:(i*max_batch_length+X_lengths[i])]

        return Y_hat_out.to(device)



class VanillaReLURNNCorrectInitialisation(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, batch_size, output_size, output_activation='Sigmoid', rnn_input_weight=[1,-1], rnn_hidden_weight=[1]):
        super(VanillaReLURNNCorrectInitialisation, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_size=batch_size
        self.output_size = output_size
        self.output_activation = output_activation
        self.model_name = 'VanillaReLURNN'

        self.vocab = {'<PAD>': 0, '(':1, ')':2}
        self.tags = {'<PAD>':0, '0':1, '1':2}
        self.nb_tags = len(self.vocab)-1
        self.rnn = nn.RNN(input_size, hidden_size, num_layers=num_layers, nonlinearity='relu', bias=False)
        self.rnn.weight_ih_l0=nn.Parameter(torch.tensor([rnn_input_weight], dtype=torch.float32))
        self.rnn.weight_hh_l0=nn.Parameter(torch.tensor([rnn_hidden_weight], dtype=torch.float32))
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.fc2.weight=nn.Parameter(torch.tensor([[1],[1]],dtype=torch.float32))
        self.fc2.bias = nn.Parameter(torch.tensor([1,-0.5], dtype=torch.float32))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, length):
        x = pack_padded_sequence(x, length, batch_first=True)
        h0 = self.init_hidden()

        x, h0 = self.rnn(x, h0)
        x, _ = pad_packed_sequence(x, batch_first=True)

        x = x.contiguous()

        x = x.view(-1, x.shape[2])

        x = self.fc2(x)

        x = self.sigmoid(x).view(-1, self.output_size)

        return x

    def init_hidden(self):
        return torch.zeros(self.num_layers, self.batch_size, self.hidden_size).to(device)



    def mask(self, Y_hat, Y, X_lengths):

        Y_hat_out = torch.zeros(Y_hat.shape)

        max_batch_length = max(X_lengths)


        for i in range(self.batch_size):

            Y_hat_out[i*max_batch_length:(i*max_batch_length+X_lengths[i])] = Y_hat[i*max_batch_length:(i*max_batch_length+X_lengths[i])]


        return Y_hat_out.to(device)


class VanillaReLURNNCorrectInitialisationWithBias(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, batch_size, output_size, output_activation='Sigmoid', rnn_input_weight=[1,-1], rnn_hidden_weight=[1]):
        super(VanillaReLURNNCorrectInitialisationWithBias, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_size=batch_size
        self.output_size = output_size
        self.output_activation = output_activation
        self.model_name = 'VanillaReLURNN'

        self.vocab = {'<PAD>': 0, '(':1, ')':2}
        self.tags = {'<PAD>':0, '0':1, '1':2}
        self.nb_tags = len(self.vocab)-1
        self.rnn = nn.RNN(input_size, hidden_size, num_layers=num_layers, nonlinearity='relu')

        self.rnn.weight_ih_l0=nn.Parameter(torch.tensor([rnn_input_weight], dtype=torch.float32))
        self.rnn.weight_hh_l0=nn.Parameter(torch.tensor([rnn_hidden_weight], dtype=torch.float32))
        self.rnn.bias_ih=nn.Parameter(torch.tensor([0], dtype=torch.float32))
        self.rnn.bias_hh = nn.Parameter(torch.tensor([0], dtype=torch.float32))
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.fc2.weight=nn.Parameter(torch.tensor([[1],[1]],dtype=torch.float32))
        self.fc2.bias = nn.Parameter(torch.tensor([1,-0.5], dtype=torch.float32))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, length):
        x = pack_padded_sequence(x, length, batch_first=True)
        h0 = self.init_hidden()

        x, h0 = self.rnn(x, h0)
        x, _ = pad_packed_sequence(x, batch_first=True)

        x = x.contiguous()

        x = x.view(-1, x.shape[2])

        x = self.fc2(x)

        x = self.sigmoid(x).view(-1, self.output_size)

        return x

    def init_hidden(self):
        return torch.zeros(self.num_layers, self.batch_size, self.hidden_size).to(device)



    def mask(self, Y_hat, Y, X_lengths):

        Y_hat_out = torch.zeros(Y_hat.shape)

        max_batch_length = max(X_lengths)


        for i in range(self.batch_size):

            Y_hat_out[i*max_batch_length:(i*max_batch_length+X_lengths[i])] = Y_hat[i*max_batch_length:(i*max_batch_length+X_lengths[i])]


        return Y_hat_out.to(device)



class RecurrentDNNC(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, batch_size, output_size, output_activation='Sigmoid'):
        super(RecurrentDNNC,self).__init__()
        self.input_size=input_size
        self.hidden_size = hidden_size
        self.num_layers=num_layers
        self.batch_size=batch_size
        self.output_size=output_size
        self.output_activation=output_activation

        self.fc1 = nn.Linear(input_size,hidden_size)
        # self.dnnc = DNNC()
        self.dnnc = DNNCNN()
        self.fc2 = nn.Linear(hidden_size,output_size)
        self.sigmoid = nn.Sigmoid()
        self.model_name='RecurrentDNNC'


    # def forward(self,x,state):
    #     x = self.fc1(x)

    def forward(self,x,length):
        # x = pack_padded_sequence(x, length, batch_first=True)
        # h0 = self.init_hidden()

        # x, h0 = self.fc1
        # print('length = ',length)
        # print('x before linear layer ',x)
        x = self.fc1(x)
        # print('x after linear layer ',x)
        # x, state = self.dnnc(x,)
        # x = pack_padded_sequence(x, length, batch_first=True)
        # print('x packed sequence ',x)
        # for i in range(length.item()):
        # print('x.shape = ',x.shape)
        # print('x after linear layer ',x)
        x1 = x.clone()
        y = torch.tensor([0,0], dtype=torch.float32)
        # print('y before dnnc = ',y)
        for i in range(x.size()[1]):
            # print('x[0][',i,'] before DNNC = ', x[0][i])
            x1[0][i] = self.dnnc(x[0][i], y)
            # print('x1[0][',i,'] = ',x1[0][i])
        #     y = x1[0][i].clone().detach()
            y = x1[0][i].clone()
            # print('y after DNNC = ',y)
        # print('x after DNNC ',x1)

        # x, _ = pad_packed_sequence(x, batch_first=True)
        #
        # x = x.contiguous()
        #
        # x = x.view(-1, x.shape[2])
        # x = x.clone()
        x = x1.clone()
        x = self.fc2(x)

        x = self.sigmoid(x).view(-1, self.output_size)

        return x

    def init_hidden(self):
        return torch.zeros(self.num_layers, self.batch_size, self.hidden_size).to(device)

    def mask(self, Y_hat, Y, X_lengths):
        Y_hat_out = torch.zeros(Y_hat.shape)

        max_batch_length = max(X_lengths)

        for i in range(self.batch_size):
            Y_hat_out[i * max_batch_length:(i * max_batch_length + X_lengths[i])] = Y_hat[i * max_batch_length:(
                        i * max_batch_length + X_lengths[i])]

        return Y_hat_out.to(device)

class RecurrentDNNCClipping(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, batch_size, output_size, output_activation='Sigmoid'):
        super(RecurrentDNNCClipping,self).__init__()
        self.input_size=input_size
        self.hidden_size = hidden_size
        self.num_layers=num_layers
        self.batch_size=batch_size
        self.output_size=output_size
        self.output_activation=output_activation

        self.fc1 = nn.Linear(input_size,hidden_size)
        # self.dnnc = DNNC()
        self.dnnc = DNNCNN()
        self.fc2 = nn.Linear(hidden_size,output_size)
        self.sigmoid = nn.Sigmoid()
        self.model_name='RecurrentDNNCClipping'


    # def forward(self,x,state):
    #     x = self.fc1(x)

    def forward(self,x,length):
        # x = pack_padded_sequence(x, length, batch_first=True)
        # h0 = self.init_hidden()

        # x, h0 = self.fc1
        # print('length = ',length)
        # print('x before linear layer ',x)
        x = self.fc1(x)
        # print('x after linear layer ',x)
        # x, state = self.dnnc(x,)
        # x = pack_padded_sequence(x, length, batch_first=True)
        # print('x packed sequence ',x)
        # for i in range(length.item()):
        # print('x.shape = ',x.shape)
        # print('x after linear layer ',x)
        x1 = x.clone()
        y = torch.tensor([0,0], dtype=torch.float32)
        # print('y before dnnc = ',y)
        for i in range(x.size()[1]):
            # print('x[0][',i,'] before DNNC = ', x[0][i])
            x1[0][i] = self.dnnc(x[0][i], y)
            # print('x1[0][',i,'] = ',x1[0][i])
        #     y = x1[0][i].clone().detach()
            y = x1[0][i].clone()
            # print('y after DNNC = ',y)
        # print('x after DNNC ',x1)

        # x, _ = pad_packed_sequence(x, batch_first=True)
        #
        # x = x.contiguous()
        #
        # x = x.view(-1, x.shape[2])
        # x = x.clone()
        x = x1.clone()
        x = self.fc2(x)

        # x = self.sigmoid(x).view(-1, self.output_size)
        x = torch.clamp(x,min=0,max=1).view(-1, self.output_size)
        return x

    def init_hidden(self):
        return torch.zeros(self.num_layers, self.batch_size, self.hidden_size).to(device)

    def mask(self, Y_hat, Y, X_lengths):
        Y_hat_out = torch.zeros(Y_hat.shape)

        max_batch_length = max(X_lengths)

        for i in range(self.batch_size):
            Y_hat_out[i * max_batch_length:(i * max_batch_length + X_lengths[i])] = Y_hat[i * max_batch_length:(
                        i * max_batch_length + X_lengths[i])]

        return Y_hat_out.to(device)




class RecurrentDNNCFrozenInputLayer(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, batch_size, output_size, output_activation='Sigmoid'):
        super(RecurrentDNNCFrozenInputLayer,self).__init__()
        self.input_size=input_size
        self.hidden_size = hidden_size
        self.num_layers=num_layers
        self.batch_size=batch_size
        self.output_size=output_size
        self.output_activation=output_activation

        # self.fc1 = nn.Linear(input_size,hidden_size)
        # self.dnnc = DNNC()
        self.dnnc = DNNCNN()
        self.fc2 = nn.Linear(hidden_size,output_size)
        self.sigmoid = nn.Sigmoid()
        self.model_name='RecurrentDNNCFrozenInputLayer'


    # def forward(self,x,state):
    #     x = self.fc1(x)

    def forward(self,x,length):
        # x = pack_padded_sequence(x, length, batch_first=True)
        # h0 = self.init_hidden()

        # x, h0 = self.fc1
        # print('length = ',length)
        # print('x before linear layer ',x)
        # x = self.fc1(x)
        # print('x after linear layer ',x)
        # x, state = self.dnnc(x,)
        # x = pack_padded_sequence(x, length, batch_first=True)
        # print('x packed sequence ',x)
        # for i in range(length.item()):
        # print('x.shape = ',x.shape)
        x1 = x.clone()
        y = torch.tensor([0,0], dtype=torch.float32)
        # print('y before dnnc = ',y)
        for i in range(x.size()[1]):
            x1[0][i] = self.dnnc(x[0][i], y)
        #     print('x1[0][',i,'] = ',x1[0][i])
        #     y = x1[0][i].clone().detach()
            y = x1[0][i].clone()
            # print('y after DNNC = ',y)
        # print('x after DNNC ',x1)

        # x, _ = pad_packed_sequence(x, batch_first=True)
        #
        # x = x.contiguous()
        #
        # x = x.view(-1, x.shape[2])
        # x = x.clone()
        x = x1.clone()
        x = self.fc2(x)

        x = self.sigmoid(x).view(-1, self.output_size)

        return x

    def init_hidden(self):
        return torch.zeros(self.num_layers, self.batch_size, self.hidden_size).to(device)

    def mask(self, Y_hat, Y, X_lengths):
        Y_hat_out = torch.zeros(Y_hat.shape)

        max_batch_length = max(X_lengths)

        for i in range(self.batch_size):
            Y_hat_out[i * max_batch_length:(i * max_batch_length + X_lengths[i])] = Y_hat[i * max_batch_length:(
                        i * max_batch_length + X_lengths[i])]

        return Y_hat_out.to(device)


# this is like a neural implementation of the DNNC, it needs to be integrated into a model to perform classification
class NonZeroReLUCounter(nn.Module):
    def __init__(self,counter_input_size, counter_output_size, output_size, initialisation='random',output_activation='Sigmoid'):
        super(NonZeroReLUCounter, self).__init__()
        #FIX INPUT AND OUTPUT PARAMETERS IN INIT AND FORWARD FUNCTIONS HERE IN ORDER TO MAKE THIS COMPATIBLE WITH THE MAIN CODE
        self.model_name = 'NonZeroReLUCounter'
        self.open_bracket_filter = nn.Linear(in_features=2,out_features=1,bias=False)
        # self.close_bracket_filter = nn.ReLU(nn.Linear(in_features=2,out_features=1,bias=False))
        self.close_bracket_filter = nn.Linear(in_features=2, out_features=1, bias=False)
        self.open_bracket_counter = nn.Linear(in_features=2,out_features=1,bias=False)
        self.close_bracket_counter = nn.Linear(in_features=2,out_features=1,bias=False)
        self.open_minus_close = nn.Linear(in_features=2,out_features=1,bias=False)
        self.close_minus_open = nn.Linear(in_features=2,out_features=1,bias=False)
        self.open_minus_close_copy = nn.Linear(in_features=1,out_features=1,bias=False)
        self.surplus_close_count = nn.Linear(in_features=2,out_features=1,bias=False)
        self.out = nn.Linear(in_features=2,out_features=1,bias=False)
        self.sigmoid = nn.Sigmoid()
        self.output_activation=output_activation
        self.ReLU = nn.ReLU()
        if initialisation=='correct':
            self.open_bracket_filter.weight = nn.Parameter(torch.tensor([[1,0]],dtype=torch.float32),requires_grad=False)
            self.close_bracket_filter.weight = nn.Parameter(torch.tensor([[0,1]],dtype=torch.float32),requires_grad=False)
            self.open_bracket_counter.weight = nn.Parameter(torch.tensor([[1,1]],dtype=torch.float32),requires_grad=False)
            self.close_bracket_counter.weight = nn.Parameter(torch.tensor([[1,1]],dtype=torch.float32),requires_grad=False)
            self.open_minus_close.weight = nn.Parameter(torch.tensor([[1,-1]],dtype=torch.float32),requires_grad=False)
            self.close_minus_open.weight = nn.Parameter(torch.tensor([[-1,1]],dtype=torch.float32),requires_grad=False)
            self.open_minus_close_copy.weight = nn.Parameter(torch.tensor([[1]],dtype=torch.float32),requires_grad=False)
            self.surplus_close_count.weight = nn.Parameter(torch.tensor([[1,1]],dtype=torch.float32),requires_grad=False)
            self.out.weight = nn.Parameter(torch.tensor([[1,1]],dtype=torch.float32))

    def forward(self, x, opening_brackets, closing_brackets, excess_closing_brackets):
        closing = self.close_bracket_filter(x.unsqueeze(dim=0))
        closing = self.ReLU(closing)


        closing = torch.cat((closing, closing_brackets.unsqueeze(dim=0)))
        closing = self.close_bracket_counter(closing.squeeze())
        closing = self.ReLU(closing)
        closing_brackets = closing

        opening = self.open_bracket_filter(x.unsqueeze(dim=0))
        opening = self.ReLU(opening)

        opening = torch.cat((opening, opening_brackets.unsqueeze(dim=0)))
        opening = self.open_bracket_counter(opening.squeeze())
        opening = self.ReLU(opening)
        opening_brackets = opening


        closing_minus_opening = torch.cat((opening, closing))

        opening_minus_closing = torch.cat((opening, closing))
        closing_minus_opening = self.close_minus_open(closing_minus_opening)
        closing_minus_opening = self.ReLU(closing_minus_opening)

        opening_minus_closing = self.open_minus_close(opening_minus_closing)
        opening_minus_closing = self.ReLU(opening_minus_closing)

        opening_minus_closing = self.open_minus_close_copy(opening_minus_closing)
        opening_minus_closing = self.ReLU(opening_minus_closing)

        surplus_closing_brackets = torch.cat(
            (closing_minus_opening, excess_closing_brackets))
        surplus_closing_brackets = self.surplus_close_count(surplus_closing_brackets)
        surplus_closing_brackets = self.ReLU(surplus_closing_brackets)
        output = torch.cat((opening_minus_closing, surplus_closing_brackets))

        return output, opening_brackets, closing_brackets, surplus_closing_brackets
        # return output


# Integrating the NonZeroReLUCounter into a model
class RecurrentNonZeroReLUCounter(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, batch_size, output_size, output_activation='Sigmoid'):
        super(RecurrentNonZeroReLUCounter,self).__init__()
        self.input_size=input_size
        self.hidden_size = hidden_size
        self.num_layers=num_layers
        self.batch_size=batch_size
        self.output_size=output_size
        self.output_activation=output_activation

        self.fc1 = nn.Linear(input_size,hidden_size)
        self.counter = NonZeroReLUCounter(counter_input_size=input_size, counter_output_size=output_size, output_size=output_size,output_activation='Sigmoid', initialisation='correct') #THESE PARAMETERS ARENT IN THE NONZERORELUCOUNTER SO WE JUST SET THEM TO AVOID ERRORS
        self.fc2 = nn.Linear(hidden_size,output_size)
        self.sigmoid = nn.Sigmoid()
        self.model_name='RecurrentNonZeroReLUCounter'



    def forward(self,x,length):
        opening_brackets = torch.tensor([0.], dtype=torch.float32)
        closing_brackets = torch.tensor([0.], dtype=torch.float32)
        excess_closing_brackets = torch.tensor([0.], dtype=torch.float32)
        outputs = []
        x = self.fc1(x)
        x1 = x.clone()
        y = torch.tensor([0,0], dtype=torch.float32)
        for i in range(x.size()[1]):
            x1[0][i], opening_brackets,closing_brackets, excess_closing_brackets = self.counter(x[0][i], opening_brackets,closing_brackets,excess_closing_brackets)
            # x1[0][i] = self.counter(x[0][i], y)
            # y = x1[0][i].clone()

        x = x1.clone()
        x = self.fc2(x)

        x = self.sigmoid(x).view(-1, self.output_size)

        return x
    def init_hidden(self):
        return torch.zeros(self.num_layers, self.batch_size, self.hidden_size).to(device)

    def mask(self, Y_hat, Y, X_lengths):
        Y_hat_out = torch.zeros(Y_hat.shape)

        max_batch_length = max(X_lengths)

        for i in range(self.batch_size):
            Y_hat_out[i * max_batch_length:(i * max_batch_length + X_lengths[i])] = Y_hat[i * max_batch_length:(
                        i * max_batch_length + X_lengths[i])]

        return Y_hat_out.to(device)


class RecurrentDNNCNoFalsePop(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, batch_size, output_size, output_activation='Sigmoid'):
        super(RecurrentDNNCNoFalsePop,self).__init__()
        self.input_size=input_size
        self.hidden_size = hidden_size
        self.num_layers=num_layers
        self.batch_size=batch_size
        self.output_size=output_size
        self.output_activation=output_activation

        self.fc1 = nn.Linear(input_size,hidden_size)
        # self.dnnc = DNNC()
        self.dnnc = DNNCNNNoFalsePop()
        self.fc2 = nn.Linear(1,output_size)
        # self.fc2 = nn.Linear(1, output_size)
        self.sigmoid = nn.Sigmoid()
        self.model_name='RecurrentDNNCNoFalsePop'


    # def forward(self,x,state):
    #     x = self.fc1(x)

    def forward(self,x,length):

        print('x before linear layer ',x)
        x = self.fc1(x)
        print('x after linear layer ',x)

        x1 = x.clone()
        y = torch.tensor([0], dtype=torch.float32)
        # print('y before dnnc = ',y)
        for i in range(x.size()[1]):
            # print('x[0][',i,'] before DNNC = ', x[0][i])
            x1[0][i] = self.dnnc(x[0][i], y)
            # print('x1[0][',i,'] = ',x1[0][i])
        #     y = x1[0][i].clone().detach()
            y = x1[0][i][0].clone()
        #     print('y after DNNC = ',y)
        print('x after DNNC ',x1)
        print('y after DNNC ',y)

        # x, _ = pad_packed_sequence(x, batch_first=True)
        #
        # x = x.contiguous()
        #
        # x = x.view(-1, x.shape[2])
        # x = x.clone()
        x = x1.clone()
        x = self.fc2(x)
        print('x after fc2 ',x)

        x = self.sigmoid(x).view(-1, self.output_size)
        print('x after sigmoid ',x)

        return x
    def init_hidden(self):
        return torch.zeros(self.num_layers, self.batch_size, self.hidden_size).to(device)

    def mask(self, Y_hat, Y, X_lengths):
        Y_hat_out = torch.zeros(Y_hat.shape)

        max_batch_length = max(X_lengths)

        for i in range(self.batch_size):
            Y_hat_out[i * max_batch_length:(i * max_batch_length + X_lengths[i])] = Y_hat[i * max_batch_length:(
                        i * max_batch_length + X_lengths[i])]

        return Y_hat_out.to(device)

class RecurrentDNNCNoFalsePopNoBiases(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, batch_size, output_size, output_activation='Sigmoid'):
        super(RecurrentDNNCNoFalsePopNoBiases,self).__init__()
        self.input_size=input_size
        self.hidden_size = hidden_size
        self.num_layers=num_layers
        self.batch_size=batch_size
        self.output_size=output_size
        self.output_activation=output_activation

        self.fc1 = nn.Linear(input_size,hidden_size,bias=False)
        # self.dnnc = DNNC()
        self.dnnc = DNNCNNNoFalsePop()
        self.fc2 = nn.Linear(hidden_size,output_size,bias=False)
        # self.fc2 = nn.Linear(1, output_size)
        self.sigmoid = nn.Sigmoid()
        self.model_name='RecurrentDNNCNoFalsePopNoBiases'


    # def forward(self,x,state):
    #     x = self.fc1(x)

    def forward(self,x,length):

        # print('x before linear layer ',x)
        x = self.fc1(x)
        # print('x after linear layer ',x)

        x1 = x.clone()
        y = torch.tensor([0], dtype=torch.float32)
        # print('y before dnnc = ',y)
        for i in range(x.size()[1]):
            # print('x[0][',i,'] before DNNC = ', x[0][i])
            x1[0][i] = self.dnnc(x[0][i], y)
            # print('x1[0][',i,'] = ',x1[0][i])
        #     y = x1[0][i].clone().detach()
            y = x1[0][i][0].clone()
        #     print('y after DNNC = ',y)
        # print('x after DNNC ',x1)

        # x, _ = pad_packed_sequence(x, batch_first=True)
        #
        # x = x.contiguous()
        #
        # x = x.view(-1, x.shape[2])
        # x = x.clone()
        x = x1.clone()
        x = self.fc2(x)

        x = self.sigmoid(x).view(-1, self.output_size)

        return x
    def init_hidden(self):
        return torch.zeros(self.num_layers, self.batch_size, self.hidden_size).to(device)

    def mask(self, Y_hat, Y, X_lengths):
        Y_hat_out = torch.zeros(Y_hat.shape)

        max_batch_length = max(X_lengths)

        for i in range(self.batch_size):
            Y_hat_out[i * max_batch_length:(i * max_batch_length + X_lengths[i])] = Y_hat[i * max_batch_length:(
                        i * max_batch_length + X_lengths[i])]

        return Y_hat_out.to(device)