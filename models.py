import torch
import torch.nn as nn
from DNNC import DNNC, DNNCNN
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
        x1 = x.clone()
        y = torch.tensor([[0,0]], dtype=torch.float32)
        for i in range(x.size()[1]):
            x1[0][i] = self.dnnc(x[0][i], y)
        #     print('x1[0][',i,'] = ',x1[0][i])
            y = x1[0][i].clone()
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



