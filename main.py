import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
import numpy as np
import argparse
import matplotlib.pyplot as plt
from models import VanillaLSTM, VanillaRNN, VanillaGRU, VanillaReLURNN, VanillaReLURNN_NoBias, VanillaReLURNNCorrectInitialisation, VanillaReLURNNCorrectInitialisationWithBias, RecurrentDNNC
from Dyck_Generator_Suzgun_Batch import DyckLanguage
import random
from torch.utils.tensorboard import SummaryWriter
# from tensorboardX import SummaryWriter
from torch.utils.data import Dataset, DataLoader
from Dyck1_Datasets import NextTokenPredictionLongTestDataset, NextTokenPredictionShortTestDataset, NextTokenPredictionTrainDataset, NextTokenPredictionValidationDataset, NextTokenPredictionLongTestDataset_SAMPLE, NextTokenPredictionShortTestDataset_SAMPLE, NextTokenPredictionTrainDataset_SAMPLE, NextTokenPredictionValidationDataset_SAMPLE
from torch.optim.lr_scheduler import StepLR
import math
import time

from Semi_Dyck1_Datasets import SemiDyck1TrainDataset, SemiDyck1ValidationDataset, SemiDyck1TestDataset, SemiDyck1ShortTestDataset, SemiDyck1Dataset1000tokens, SemiDyck1Dataset2000tokens_zigzag



"""
PLOT THE LOSS FOR EACH TRAINING RUN
PLOT THE LEARNING RATE FROM THE SCHEDULER THROUGHOUT TRAINING
"""

seed = 10
torch.manual_seed(seed)
np.random.seed(seed)


parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, help='input model name (VanillaLSTM, VanillaRNN, VanillaGRU)')
parser.add_argument('--task', type=str, help='NextTokenPrediction, BinaryClassification, TernaryClassification, NextTokenPredictionCrossEntropy, SemiDyck1MSE, SemiDyck1BCE')
parser.add_argument('--feedback', type=str, help='EveryTimeStep, EndofSequence')
parser.add_argument('--hidden_size', type=int, help='hidden size')
parser.add_argument('--num_layers', type=int, help='number of layers', default=1)
parser.add_argument('--batch_size', type=int, help='batch size', default=1)
parser.add_argument('--learning_rate', type=float, help='learning rate')
parser.add_argument('--lr_scheduler_step',type=int, help='number of epochs before reducing', default=100)
parser.add_argument('--lr_scheduler_gamma',type=float, help='multiplication factor for lr scheduler', default=1.0)
parser.add_argument('--num_epochs', type=int, help='number of training epochs')
parser.add_argument('--num_runs', type=int, help='number of training runs')
parser.add_argument('--checkpoint_step', type=int, help='checkpoint step', default=0)
parser.add_argument('--shuffle_dataset',type=bool,default=False)
parser.add_argument('--output_size',type=int,default=2,help='how many output neurons, 1 or 2?')
# parser.add_argument('--loss_function', type=str, default='MSELoss', help='MSELoss or BCELoss')
parser.add_argument('--runtime',type=str,default='colab',help='colab or local or linux')



args = parser.parse_args()

model_name = args.model_name
task = args.task
feedback = args.feedback
hidden_size = args.hidden_size
num_layers = args.num_layers
learning_rate = args.learning_rate
num_epochs = args.num_epochs
num_runs = args.num_runs
batch_size = args.batch_size
# load_model = args.load_model
lr_scheduler_step = args.lr_scheduler_step
lr_scheduler_gamma = args.lr_scheduler_gamma
output_size = args.output_size
# loss_function=args.loss_function
runtime = args.runtime


checkpoint_step = int(num_epochs/4)
if args.checkpoint_step!=0:
    checkpoint_step = args.checkpoint_step

shuffle_dataset = args.shuffle_dataset




use_optimiser='Adam'

num_bracket_pairs = 25

length_bracket_pairs = 50


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    device=torch.device("cuda")
elif torch.backends.mps.is_available():
    device=torch.device("mps")
else:
    device=torch.device("cpu")


vocab = ['(', ')']
tags = {'':0, '0':1, '1':2}
n_letters= len(vocab)
n_tags = len(tags)-1
num_bracket_pairs = 25
length_bracket_pairs = 50


pad_token=0

NUM_PAR = 1
MIN_SIZE = 2
MAX_SIZE = 50
P_VAL = 0.5
Q_VAL = 0.25


epsilon=0.5

# train_size = 10000
# test_size = 5000
# long_size = 5000
train_size = 1000
test_size = 500
long_size = 500

Dyck = DyckLanguage(NUM_PAR, P_VAL, Q_VAL)

if runtime=='colab':
    path = "/content/drive/MyDrive/PhD/EXPT_LOGS/Dyck1_"+str(task)+"/Minibatch_Training/"+model_name+"/"\
           +str(batch_size)+"_batch_size/"+str(learning_rate)+"_learning_rate/"+str(num_epochs)+"_epochs/"\
           +str(lr_scheduler_step)+"_lr_scheduler_step/"+str(lr_scheduler_gamma)+"_lr_scheduler_gamma/"\
           +str(hidden_size)+"_hidden_units/"+str(num_runs)+"_runs/shuffle_"+str(shuffle_dataset)+"/"
elif runtime=='local':
    path = "/Users/nadineelnaggar/Google Drive/PhD/EXPT_LOGS/Dyck1_"+str(task)+"/Minibatch_Training/"+model_name+"/"\
       +str(batch_size)+"_batch_size/"+str(learning_rate)+"_learning_rate/"+str(num_epochs)+"_epochs/"\
       +str(lr_scheduler_step)+"_lr_scheduler_step/"+str(lr_scheduler_gamma)+"_lr_scheduler_gamma/"\
       +str(hidden_size)+"_hidden_units/"+str(num_runs)+"_runs/shuffle_"+str(shuffle_dataset)+"/"
elif runtime=='linux':
    path = "EXPT_LOGS/Dyck1_"+str(task)+"/Minibatch_Training/"+model_name+"/"\
       +str(batch_size)+"_batch_size/"+str(learning_rate)+"_learning_rate/"+str(num_epochs)+"_epochs/"\
       +str(lr_scheduler_step)+"_lr_scheduler_step/"+str(lr_scheduler_gamma)+"_lr_scheduler_gamma/"\
       +str(hidden_size)+"_hidden_units/"+str(num_runs)+"_runs/shuffle_"+str(shuffle_dataset)+"/"

print('model_name = ',model_name)
print('task = ',task)
print('feedback = ',feedback)
print('hidden_size = ',hidden_size)
print('batch_size = ',batch_size)
print('num_layers = ',num_layers)
print('learning_rate = ',learning_rate)
print('num_epochs = ',num_epochs)
print('num_runs = ',num_runs)


file_name = path+ 'Dyck1_' + task + '_' + str(
        num_bracket_pairs) + '_bracket_pairs_' + model_name + '_Feedback_' + feedback + '_' +str(batch_size) +'_batch_size_'+'_' + str(
        hidden_size) + 'hidden_units_' + use_optimiser + '_lr=' + str(learning_rate) + '_' + str(
        num_epochs) + 'epochs_'+str(lr_scheduler_step)+"lr_scheduler_step_"+str(lr_scheduler_gamma)+"lr_scheduler_gamma_"+ str(num_runs)+'runs' + '.txt'
excel_name = path+ 'Dyck1_' + task + '_' + str(
        num_bracket_pairs) + '_bracket_pairs_' + model_name + '_Feedback_' + feedback + '_' +str(batch_size) +'_batch_size_'+'_' + str(
        hidden_size) + 'hidden_units_' + use_optimiser + '_lr=' + str(learning_rate) + '_' + str(
        num_epochs) + 'epochs_'+str(lr_scheduler_step)+"lr_scheduler_step_"+str(lr_scheduler_gamma)+"lr_scheduler_gamma_"+ str(num_runs)+'runs' + '.xlsx'

modelname = path+ 'Dyck1_' + task + '_' + str(
        num_bracket_pairs) + '_bracket_pairs_' + model_name + '_Feedback_' + feedback + '_' +str(batch_size) +'_batch_size_'+'_' + str(
        hidden_size) + 'hidden_units_' + use_optimiser + '_lr=' + str(learning_rate) + '_' + str(
        num_epochs) + 'epochs_'+ str(num_runs)+'runs' + '_MODEL_'

optimname = path+ 'Dyck1_' + task + '_' + str(
        num_bracket_pairs) + '_bracket_pairs_' + model_name + '_Feedback_' + feedback + '_' +str(batch_size) +'_batch_size_'+'_' + str(
        hidden_size) + 'hidden_units_' + use_optimiser + '_lr=' + str(learning_rate) + '_' + str(
        num_epochs) + 'epochs_'+str(lr_scheduler_step)+"lr_scheduler_step_"+str(lr_scheduler_gamma)+"lr_scheduler_gamma_"+ str(num_runs)+'runs' + '_OPTIMISER_'
train_log = path+ 'Dyck1_' + task + '_' + str(
        num_bracket_pairs) + '_bracket_pairs_' + model_name + '_Feedback_' + feedback + '_' +str(batch_size) +'_batch_size_'+'_' + str(
        hidden_size) + 'hidden_units_' + use_optimiser + '_lr=' + str(learning_rate) + '_' + str(
        num_epochs) + 'epochs_'+str(lr_scheduler_step)+"lr_scheduler_step_"+str(lr_scheduler_gamma)+"lr_scheduler_gamma_"+ str(num_runs)+'runs' + '_TRAIN_LOG.txt'
train_log_raw = path+ 'Dyck1_' + task + '_' + str(
        num_bracket_pairs) + '_bracket_pairs_' + model_name + '_Feedback_' + feedback + '_' +str(batch_size) +'_batch_size_'+'_' + str(
        hidden_size) + 'hidden_units_' + use_optimiser + '_lr=' + str(learning_rate) + '_' + str(
        num_epochs) + 'epochs_'+str(lr_scheduler_step)+"lr_scheduler_step_"+str(lr_scheduler_gamma)+"lr_scheduler_gamma_"+ str(num_runs)+'runs' + '_TRAIN_LOG_RAW.txt'
validation_log = path+ 'Dyck1_' + task + '_' + str(
        num_bracket_pairs) + '_bracket_pairs_' + model_name + '_Feedback_' + feedback + '_' +str(batch_size) +'_batch_size_'+'_' + str(
        hidden_size) + 'hidden_units_' + use_optimiser + '_lr=' + str(learning_rate) + '_' + str(
        num_epochs) + 'epochs_'+str(lr_scheduler_step)+"lr_scheduler_step_"+str(lr_scheduler_gamma)+"lr_scheduler_gamma_"+ str(num_runs)+'runs' + '_VALIDATION_LOG.txt'
train_validation_log = path+ 'Dyck1_' + task + '_' + str(
        num_bracket_pairs) + '_bracket_pairs_' + model_name + '_Feedback_' + feedback + '_' +str(batch_size) +'_batch_size_'+'_' + str(
        hidden_size) + 'hidden_units_' + use_optimiser + '_lr=' + str(learning_rate) + '_' + str(
        num_epochs) + 'epochs_'+str(lr_scheduler_step)+"lr_scheduler_step_"+str(lr_scheduler_gamma)+"lr_scheduler_gamma_"+ str(num_runs)+'runs' + '_TRAINING_SET_VALIDATION_LOG.txt'
long_validation_log = path+ 'Dyck1_' + task + '_' + str(
        num_bracket_pairs) + '_bracket_pairs_' + model_name + '_Feedback_' + feedback + '_' +str(batch_size) +'_batch_size_'+'_' + str(
        hidden_size) + 'hidden_units_' + use_optimiser + '_lr=' + str(learning_rate) + '_' + str(
        num_epochs) + 'epochs_'+str(lr_scheduler_step)+"lr_scheduler_step_"+str(lr_scheduler_gamma)+"lr_scheduler_gamma_"+ str(num_runs)+'runs' + '_LONG_VALIDATION_LOG.txt'
test_log = path+'Dyck1_' + task + '_' + str(
        num_bracket_pairs) + '_bracket_pairs_' + model_name + '_Feedback_' + feedback + '_' +str(batch_size) +'_batch_size_'+'_' + str(
        hidden_size) + 'hidden_units_' + use_optimiser + '_lr=' + str(learning_rate) + '_' + str(
        num_epochs) + 'epochs_'+str(lr_scheduler_step)+"lr_scheduler_step_"+str(lr_scheduler_gamma)+"lr_scheduler_gamma_"+ str(num_runs)+'runs' + '_TEST_LOG.txt'
long_test_log = path+'Dyck1_' + task + '_' + str(
        num_bracket_pairs) + '_bracket_pairs_' + model_name + '_Feedback_' + feedback + '_' +str(batch_size) +'_batch_size_'+'_' + str(
        hidden_size) + 'hidden_units_' + use_optimiser + '_lr=' + str(learning_rate) + '_' + str(
        num_epochs) + 'epochs_'+str(lr_scheduler_step)+"lr_scheduler_step_"+str(lr_scheduler_gamma)+"lr_scheduler_gamma_"+ str(num_runs)+'runs' + '_LONG_TEST_LOG.txt'
plot_name = path+'Dyck1_' + task + '_' + str(
        num_bracket_pairs) + '_bracket_pairs_' + model_name + '_Feedback_' + feedback + '_' +str(batch_size) +'_batch_size_'+'_' + str(
        hidden_size) + 'hidden_units_' + use_optimiser + '_lr=' + str(learning_rate) + '_' + str(
        num_epochs) + 'epochs_'+str(lr_scheduler_step)+"lr_scheduler_step_"+str(lr_scheduler_gamma)+"lr_scheduler_gamma_"+ str(num_runs)+'runs' + '_PLOT.png'

plt_name = path+'Dyck1_' + task + '_' + str(
        num_bracket_pairs) + '_bracket_pairs_' + model_name + '_Feedback_' + feedback + '_' +str(batch_size) +'_batch_size_'+'_' + str(
        hidden_size) + 'hidden_units_' + use_optimiser + '_lr=' + str(learning_rate) + '_' + str(
        num_epochs) + 'epochs_'+str(lr_scheduler_step)+"lr_scheduler_step_"+str(lr_scheduler_gamma)+"lr_scheduler_gamma_"+ str(num_runs)+'runs'

checkpoint = path+ 'Dyck1_' + task + '_' + str(
        num_bracket_pairs) + '_bracket_pairs_' + model_name + '_Feedback_' + feedback + '_' +str(batch_size) +'_batch_size_'+'_' + str(
        hidden_size) + 'hidden_units_' + use_optimiser + '_lr=' + str(learning_rate) + '_' + str(
        num_epochs) + 'epochs_'+str(lr_scheduler_step)+"lr_scheduler_step_"+str(lr_scheduler_gamma)+"lr_scheduler_gamma_"+ str(num_runs)+'runs' + '_CHECKPOINT_'

with open(file_name, 'w') as f:
    f.write('\n')

with open(train_log, 'w') as f:
    f.write('\n')

with open(test_log, 'w') as f:
    f.write('\n')
with open(long_test_log, 'w') as f:
    f.write('\n')
with open(validation_log, 'w') as f:
    f.write('\n')

def encode_batch(sentences, labels, lengths, batch_size):

    max_length = max(lengths)
    sentence_tensor = torch.zeros(batch_size,max_length,len(vocab))

    labels_tensor = torch.tensor([])
    for i in range(batch_size):

        sentence = sentences[i]
        labels_tensor = torch.cat((labels_tensor, Dyck.lineToTensorSigmoid(labels[i],max_len=max_length)))
        if len(sentence)<max_length:
            for index, char in enumerate(sentence):
                pos = vocab.index(char)
                sentence_tensor[i][index][pos] = 1
        else:
            for index, char in enumerate(sentence):
                pos = vocab.index(char)
                sentence_tensor[i][index][pos]=1
    sentence_tensor.requires_grad_(True)
    lengths_tensor = torch.tensor(lengths, dtype=torch.int64).cpu()
    return sentence_tensor, labels_tensor, lengths_tensor

def encode_batch_semiDyck1(sentences, labels, lengths, batch_size):

    max_length = max(lengths)
    sentence_tensor = torch.zeros(batch_size,max_length,len(vocab))

    labels_tensor = torch.tensor([])
    for i in range(batch_size):

        sentence = sentences[i]
        label = labels[i]
        label_tensor = torch.tensor([])
        for j in range(len(label)):
            if label[j]=='1':
                timestep_label_tensor = torch.tensor([[1,1]], dtype=torch.float32)
            elif label[j]=='0':
                timestep_label_tensor = torch.tensor([[1,0]],dtype=torch.float32)
            label_tensor=torch.cat((label_tensor, timestep_label_tensor))
        labels_tensor = torch.cat((labels_tensor, label_tensor))
        if len(sentence)<max_length:
            for index, char in enumerate(sentence):
                pos = vocab.index(char)
                sentence_tensor[i][index][pos] = 1
        else:
            for index, char in enumerate(sentence):
                pos = vocab.index(char)
                sentence_tensor[i][index][pos]=1
    sentence_tensor.requires_grad_(True)
    lengths_tensor = torch.tensor(lengths, dtype=torch.int64).cpu()
    return sentence_tensor, labels_tensor, lengths_tensor


def collate_fn(batch):

    sentences = [batch[i]['x'] for i in range(len(batch))]
    labels = [batch[i]['y'] for i in range(len(batch))]
    lengths = [len(sentence) for sentence in sentences]

    sentences.sort(key=len, reverse=True)
    labels.sort(key=len,reverse=True)
    lengths.sort(reverse=True)

    if task =='SemiDyck1MSE' or task=='SemiDyck1BCE':
        seq_tensor, labels_tensor, lengths_tensor = encode_batch_semiDyck1(sentences, labels, lengths, batch_size=batch_size)
    else:
        seq_tensor, labels_tensor, lengths_tensor = encode_batch(sentences, labels, lengths, batch_size=batch_size)


    return sentences, labels, seq_tensor.to(device), labels_tensor.to(device), lengths_tensor

if task=='SemiDyck1MSE' or task=='SemiDyck1BCE':
    train_dataset = SemiDyck1TrainDataset()
    test_dataset = SemiDyck1ShortTestDataset()
    long_dataset = SemiDyck1TestDataset()
    validation_dataset = SemiDyck1ValidationDataset()
else:
    train_dataset = NextTokenPredictionTrainDataset()
    test_dataset = NextTokenPredictionShortTestDataset()
    long_dataset = NextTokenPredictionLongTestDataset()
    validation_dataset = NextTokenPredictionValidationDataset()




train_loader = DataLoader(train_dataset,batch_size=batch_size, shuffle=shuffle_dataset, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset,batch_size=batch_size, shuffle=shuffle_dataset, collate_fn=collate_fn)
long_loader = DataLoader(long_dataset, batch_size=batch_size, shuffle=shuffle_dataset, collate_fn=collate_fn)
validation_loader = DataLoader(validation_dataset,batch_size=batch_size, shuffle=shuffle_dataset, collate_fn=collate_fn)





def select_model(model_name, input_size, hidden_size, num_layers,batch_size, num_classes, output_activation):
    if model_name=='VanillaLSTM':
        model = VanillaLSTM(input_size,hidden_size, num_layers, batch_size, num_classes, output_activation=output_activation)
    elif model_name=='VanillaRNN':
        model = VanillaRNN(input_size, hidden_size, num_layers, batch_size, num_classes, output_activation=output_activation)
    elif model_name=='VanillaGRU':
        model = VanillaGRU(input_size,hidden_size, num_layers, batch_size, num_classes, output_activation=output_activation)
    elif model_name=='VanillaReLURNN':
        model = VanillaReLURNN(input_size, hidden_size, num_layers, batch_size, num_classes,
                           output_activation=output_activation)
    elif model_name=='VanillaReLURNN_NoBias':
        model = VanillaReLURNN_NoBias(input_size, hidden_size, num_layers, batch_size, num_classes,
                           output_activation=output_activation)
    elif model_name=='VanillaReLURNNCorrectInitialisation':
        model=VanillaReLURNNCorrectInitialisation(input_size, hidden_size, num_layers, batch_size, num_classes, output_activation=output_activation)
    elif model_name == 'VanillaReLURNNCorrectInitialisationWithBias':
        model = VanillaReLURNNCorrectInitialisationWithBias(input_size, hidden_size, num_layers, batch_size, num_classes,
                                                    output_activation=output_activation)
    elif model_name=='RecurrentDNNC':
        model = RecurrentDNNC(input_size, hidden_size, num_layers, batch_size, num_classes, output_activation=output_activation)



    return model.to(device)









def main():


    output_activation = 'Sigmoid'

    if task == 'TernaryClassification':
        num_classes = 3
        output_activation = 'Softmax'
    elif task == 'BinaryClassification' or task == 'NextTokenPrediction' or task == 'NextTokenPredictionCrossEntropy' or task=='SemiDyck1MSE' or task=='SemiDyck1BCE':
        num_classes = 2
        output_activation = 'Sigmoid'




    input_size = n_letters







    with open(file_name, 'a') as f:
        f.write('Output activation = ' + output_activation + '\n')
        f.write('Optimiser used = ' + use_optimiser + '\n')
        f.write('Learning rate = ' + str(learning_rate) + '\n')
        f.write('Batch size = '+str(batch_size)+'\n')
        f.write('Number of runs = ' + str(num_runs) + '\n')
        f.write('Number of epochs in each run = ' + str(num_epochs) + '\n')
        f.write('LR scheduler step = '+str(lr_scheduler_step)+'\n')
        f.write('LR Scheduler Gamma = '+str(lr_scheduler_gamma)+'\n')
        f.write('Checkpoint step = '+str(checkpoint_step)+'\n')
        f.write('Saved model name prefix = ' + modelname + '\n')
        f.write('Saved optimiser name prefix = ' + optimname + '\n')
        f.write('Excel name = ' + excel_name + '\n')
        f.write('Train log name = ' + train_log + '\n')
        f.write('Raw train log name = '+train_log_raw+'\n')
        f.write('Validation log name = '+validation_log+'\n')
        f.write('Long Validation log name = ' + long_validation_log + '\n')
        f.write('Test log name = ' + test_log + '\n')
        f.write('Long test log name = ' + long_test_log + '\n')
        f.write('Plot name prefix = '+plt_name+'\n')
        f.write('Checkpoint name prefix = '+checkpoint+'\n')
        f.write('Checkpoint step = '+str(checkpoint_step)+'\n')

        f.write('///////////////////////////////////////////////////////////////\n')
        f.write('\n')

    train_accuracies = []
    test_accuracies = []
    long_test_accuracies = []
    train_dataframes = []
    runs = []
    for i in range(num_runs):
        seed = num_runs+i
        # seed = i
        torch.manual_seed(seed)
        np.random.seed(seed)
        with open(train_log, 'a') as f:
            f.write('random seed for run '+str(i)+' = '+str(seed)+'\n')
        model = select_model(model_name, input_size, hidden_size, num_layers, batch_size, num_classes, output_activation='Sigmoid')
        model.to(device)


        log_dir = path + "logs/run" + str(i)
        sum_writer = SummaryWriter(log_dir)



        runs.append('run'+str(i))
        print('****************************************************************************\n')
        print('random seed = ',seed)
        train_accuracy, df = train(model, train_loader, sum_writer, i)
        train_accuracies.append(train_accuracy)
        train_dataframes.append(df)
        test_accuracy = test_model(model, test_loader, 'short')
        test_accuracies.append(test_accuracy)
        long_test_accuracy = test_model(model, long_loader, 'long')
        long_test_accuracies.append(long_test_accuracy)

        df.plot(x='epoch',y=['Average training losses', 'Average validation losses', 'Average long validation losses'])
        plt.savefig(plt_name + '_losses_run'+str(i)+'.png')
        df.plot(x='epoch', y=['Training accuracies', 'Validation accuracies', 'Long validation accuracies'])
        plt.savefig(plt_name + '_accuracies_run' + str(i) + '.png')
        df.plot(x='epoch', y = 'learning rates')
        plt.savefig(plt_name + '_learning_rates_run' + str(i) + '.png')
        # plt.savefig(plt_name+'_run')

        with open(file_name, "a") as f:
            # f.write('Saved model name for run '+str(i)+' = ' + modelname + '\n')
            f.write('train accuracy for run ' + str(i) + ' = ' + str(train_accuracy) + '%\n')
            f.write('test accuracy for run ' + str(i) + ' = ' + str(test_accuracy) + '%\n')
            f.write('long test accuracy for run '+str(i)+' = '+str(long_test_accuracy)+'%\n')

    dfs = dict(zip(runs, train_dataframes))
    writer = pd.ExcelWriter(excel_name, engine='xlsxwriter')

    for sheet_name in dfs.keys():
        dfs[sheet_name].to_excel(writer, sheet_name=sheet_name, index=False)

    writer.save()

    max_train_accuracy = max(train_accuracies)
    min_train_accuracy = min(train_accuracies)
    avg_train_accuracy = sum(train_accuracies) / len(train_accuracies)
    std_train_accuracy = np.std(train_accuracies)

    max_test_accuracy = max(test_accuracies)
    min_test_accuracy = min(test_accuracies)
    avg_test_accuracy = sum(test_accuracies) / len(test_accuracies)
    std_test_accuracy = np.std(test_accuracies)

    max_long_test_accuracy = max(long_test_accuracies)
    min_long_test_accuracy = min(long_test_accuracies)
    avg_long_test_accuracy = sum(long_test_accuracies) / len(test_accuracies)
    std_long_test_accuracy = np.std(long_test_accuracies)

    with open(file_name, "a") as f:
        f.write('/////////////////////////////////////////////////////////////////\n')
        f.write('Maximum train accuracy = ' + str(max_train_accuracy) + '%\n')
        f.write('Minimum train accuracy = ' + str(min_train_accuracy) + '%\n')
        f.write('Average train accuracy = ' + str(avg_train_accuracy) + '%\n')
        f.write('Standard Deviation for train accuracy = ' + str(std_train_accuracy) + '\n')
        f.write('/////////////////////////////////////////////////////////////////\n')
        f.write('Maximum test accuracy = ' + str(max_test_accuracy) + '%\n')
        f.write('Minimum test accuracy = ' + str(min_test_accuracy) + '%\n')
        f.write('Average test accuracy = ' + str(avg_test_accuracy) + '%\n')
        f.write('Standard Deviation for test accuracy = ' + str(std_test_accuracy) + '\n')

        f.write('/////////////////////////////////////////////////////////////////\n')
        f.write('Maximum long test accuracy = ' + str(max_long_test_accuracy) + '%\n')
        f.write('Minimum long test accuracy = ' + str(min_long_test_accuracy) + '%\n')
        f.write('Average long test accuracy = ' + str(avg_long_test_accuracy) + '%\n')
        f.write('Standard Deviation for long test accuracy = ' + str(std_long_test_accuracy) + '\n')









def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60

    return m, s


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s

    return asMinutes(s), asMinutes(rs)

def train(model, loader, sum_writer, run=0):


    start = time.time()


    if task=='NextTokenPredictionCrossEntropy' or task=='SemiDyck1BCE':
        criterion=nn.BCELoss()
    else:
        criterion = nn.MSELoss()

    optimiser = optim.Adam(model.parameters(), lr=learning_rate)
    optimiser.zero_grad()
    losses = []
    correct_arr = []
    accuracies = []
    epochs = []
    all_epoch_incorrect_guesses = []
    df1 = pd.DataFrame()
    print_flag = False

    train_validation_accuracies = []
    train_validation_losses = []

    validation_losses = []
    validation_accuracies = []
    lrs = []

    long_validation_losses = []
    long_validation_accuracies = []

    error_indices = []
    error_seq_lengths = []

    # global_step=0

    print(model)
    print('num_train_samples = ',len(loader.dataset))
    print('device = ',device)
    with open(train_log, 'a') as f:
        f.write('device = '+str(device)+'\n')


    scheduler = StepLR(optimiser, step_size=lr_scheduler_step, gamma=lr_scheduler_gamma)

    for epoch in range(num_epochs):

        num_correct = 0
        num_correct_timesteps = 0
        total_loss = 0
        epoch_incorrect_guesses = []
        epoch_correct_guesses = []
        epochs.append(epoch)

        epoch_error_indices = []
        epoch_error_seq_lengths = []



        if epoch==num_epochs-1:
            print_flag=True
        if print_flag == True:
            with open(train_log_raw, 'a') as f:
                f.write('\nEPOCH ' + str(epoch) + '\n')


        for i, (sentences, labels, input_seq, target_seq, length) in enumerate(loader):
            model.zero_grad()

            output_seq = model(input_seq.to(device), length)

            if print_flag == True:
                with open(train_log_raw, 'a') as f:
                    f.write('////////////////////////////////////////\n')

                    f.write('input batch = ' + str(sentences) + '\n')
                    f.write('encoded batch = '+str(input_seq)+'\n')


            output_seq=model.mask(output_seq, target_seq, length)
            loss = criterion(output_seq, target_seq)

            total_loss += loss.item()
            loss.backward()
            optimiser.step()


            if print_flag == True:
                with open(train_log_raw, 'a') as f:
                    f.write('actual output in train function = ' + str(output_seq) + '\n')


            output_seq = output_seq.view(batch_size, length[0], n_letters)
            target_seq = target_seq.view(batch_size, length[0], n_letters)


            out_seq = output_seq.clone().detach()>=epsilon
            out_seq = out_seq.float()


            if print_flag == True:
                with open(train_log_raw, 'a') as f:
                    f.write('rounded output in train function = ' + str(out_seq) + '\n')
                    f.write('target in train function = ' + str(target_seq) + '\n')




            for j in range(batch_size):

                if torch.equal(out_seq[j],target_seq[j]):



                    num_correct += 1

                    epoch_correct_guesses.append(sentences[j])

                    if print_flag == True:
                        with open(train_log_raw, 'a') as f:
                            f.write('CORRECT' + '\n')
                else:
                    epoch_incorrect_guesses.append(sentences[j])


                    for k in range(length[j]):
                        if torch.equal(out_seq[j][k], target_seq[j][k]) !=True:
                            epoch_error_indices.append(k)
                            epoch_error_seq_lengths.append(length[j])
                            break
                    if print_flag == True:
                        with open(train_log_raw, 'a') as f:
                            f.write('INCORRECT' + '\n')



        error_indices.append(epoch_error_indices)
        error_seq_lengths.append(epoch_error_seq_lengths)
        lrs.append(optimiser.param_groups[0]["lr"])
        accuracy = num_correct/len(train_dataset)*100

        # break
        accuracies.append(accuracy)
        losses.append(total_loss/len(train_dataset))
        train_val_acc, train_val_loss = validate_model(model, train_loader, train_dataset, run, epoch)
        validation_acc, validation_loss = validate_model(model, validation_loader,validation_dataset, run, epoch)
        long_validation_acc, long_validation_loss = validate_model_long(model, long_loader, long_dataset, run, epoch)
        time_mins, time_secs = timeSince(start, epoch+1/num_epochs*100)

        with open(train_log,'a') as f:
            f.write('Accuracy for epoch '+ str(epoch)+ '='+ str(round(accuracy,2))+ '%, avg train loss = '+
              str(total_loss / len(train_dataset))+
              ' num_correct = '+ str(num_correct)+', train val loss = '+str(train_val_loss)+', train val acc = '+ str(round(train_val_acc,2))+'%, val loss = '+ str(validation_loss) + ', val accuracy = '+ str(round(validation_acc,2))+ '%, long val loss = '+str(long_validation_loss)+', long val acc = '+str(round(long_validation_acc,4))+'%, time = '+str(time_mins[0])+'m '+str(round(time_mins[1],2))+'s \n')

        print('Accuracy for epoch ', epoch, '=', round(accuracy,2), '%, avg train loss = ',
              total_loss / len(train_dataset),
              ' num_correct = ', num_correct,', train val loss = ', train_val_loss, ', train val accuracy = ', round(train_val_acc,2),'%, val loss = ', validation_loss, ', val accuracy = ', round(validation_acc,2), '%, long val loss = ',long_validation_loss, ', long val acc = ',round(long_validation_acc,4), '%, time = ',time_mins[0],'m ',round(time_mins[1],2),'s')


        scheduler.step()
        validation_losses.append(validation_loss)
        validation_accuracies.append(validation_acc)
        train_validation_losses.append(train_val_loss)
        train_validation_accuracies.append(train_val_acc)
        long_validation_losses.append(long_validation_loss)
        long_validation_accuracies.append(long_validation_acc)
        sum_writer.add_scalar('epoch_losses', total_loss/len(train_dataset),global_step=epoch)
        sum_writer.add_scalar('accuracy', accuracy, global_step=epoch)
        sum_writer.add_scalar('train validation losses', train_val_loss, global_step=epoch)
        sum_writer.add_scalar('train validation accuracy', train_val_acc, global_step=epoch)
        sum_writer.add_scalar('validation losses',validation_loss, global_step=epoch)
        sum_writer.add_scalar('validation_accuracy',validation_acc, global_step=epoch)
        sum_writer.add_scalar('long validation losses',long_validation_loss, global_step=epoch)
        sum_writer.add_scalar('long validation_accuracy',long_validation_acc, global_step=epoch)
        sum_writer.add_scalar('learning_rates', optimiser.param_groups[0]["lr"], global_step=epoch)
        # global_step+=1
        sum_writer.close()
        all_epoch_incorrect_guesses.append(epoch_incorrect_guesses)
        correct_arr.append(epoch_correct_guesses)
        if epoch == num_epochs - 1:
            # print('\n////////////////////////////////////////////////////////////////////////////////////////\n')
            print('num_correct = ',num_correct)
            print('Final training accuracy = ', num_correct / len(train_dataset) * 100, '%')
            # print('**************************************************************************\n')

        if epoch%checkpoint_step==0:
            checkpoint_path = checkpoint+'run'+str(run)+"_epoch"+str(epoch)+".pth"
            torch.save({'run':run,
                        'epoch':epoch,
                        'model_state_dict':model.state_dict(),
                        'optimiser_state_dict':optimiser.state_dict(),
                        'loss':loss},checkpoint_path)
            checkpoint_loss_plot = modelname+'run'+str(run)+'_epoch'+str(epoch)+'_losses.png'
            checkpoint_accuracy_plot = modelname+'run'+str(run)+'_epoch'+str(epoch)+'_accuracies.png'
            checkpoint_lr_plot = modelname+'run'+str(run)+'_epoch'+str(epoch)+'_lrs.png'
            fig_loss, ax_loss = plt.subplots()
            plt.plot(epochs,losses, label='avg train loss')
            plt.plot(epochs, train_validation_losses, label='avg train validation loss')
            plt.plot(epochs,validation_losses, label='avg validation loss')
            plt.plot(long_validation_losses, label='avg long validation loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.savefig(checkpoint_loss_plot)
            plt.close()
            fig_acc, ax_acc = plt.subplots()
            plt.plot(epochs, accuracies, label='train accuracies')
            plt.plot(epochs, train_validation_accuracies, label='train validation accuracies')
            plt.plot(epochs, validation_accuracies,label='validation accuracies')
            plt.plot(epochs, long_validation_accuracies,label='long validation accuracies')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()
            plt.savefig(checkpoint_accuracy_plot)
            plt.close()
            fig_lr, ax_lr = plt.subplots()
            plt.plot(epochs, lrs, label='learning rate')
            plt.xlabel('Epoch')
            plt.ylabel('Learning rate')
            plt.legend()
            plt.savefig(checkpoint_lr_plot)
            plt.close()



    df1['epoch'] = epochs
    df1['Training accuracies'] = accuracies
    df1['Average training losses'] = losses
    df1['Train validation accuracies'] = train_validation_accuracies
    df1['Average train validation losses'] = train_validation_losses
    df1['Average validation losses'] = validation_losses
    df1['Validation accuracies'] = validation_accuracies
    df1['Average long validation losses'] = long_validation_losses
    df1['Long validation accuracies'] = long_validation_accuracies
    df1['learning rates'] = lrs
    df1['epoch correct guesses'] = correct_arr
    df1['epoch incorrect guesses'] = all_epoch_incorrect_guesses
    df1['epoch error indices'] = error_indices
    df1['epoch error seq lengths'] = error_seq_lengths


    sum_writer.add_hparams({'model_name':model.model_name,'dataset_size': len(train_dataset), 'num_epochs': num_epochs,
                            'learning_rate': learning_rate, 'batch_size':batch_size,
                            'optimiser': use_optimiser}, {'Training accuracy': accuracy, 'Training loss': total_loss/len(train_dataset)})


    sum_writer.close()


    optm = optimname+'run'+str(run)+'.pth'
    mdl = modelname+'run'+str(run)+'.pth'
    torch.save(model.state_dict(), mdl)
    torch.save(optimiser.state_dict(), optm)


    return accuracy, df1

def validate_model(model, loader, dataset, run, epoch):

    num_correct = 0

    log_file=''


    if loader==validation_loader:
        log_file = validation_log
        dataset='Validation Set'
        ds = validation_dataset
    elif loader==train_loader:
        log_file = train_validation_log
        dataset = 'Train Set'
        ds = train_dataset
    if task=='NextTokenPredictionCrossEntropy':
        criterion=nn.BCELoss()
    else:
        criterion = nn.MSELoss()

    total_loss = 0


    with open(log_file,'a') as f:
        f.write('////////////////////////////////////////\n')
        f.write('TEST '+dataset+'\n')


    for i, (sentences, labels, input_seq, target_seq, length) in enumerate(loader):
        output_seq = model(input_seq.to(device), length)


        output_seq = model.mask(output_seq, target_seq, length)
        loss = criterion(output_seq,target_seq)
        total_loss+=loss.item()



        output_seq = output_seq.view(batch_size, length[0], n_letters)
        target_seq = target_seq.view(batch_size, length[0], n_letters)


        out_seq = output_seq.clone().detach() >= epsilon
        out_seq = out_seq.float()





        for j in range(batch_size):

            if torch.equal(out_seq[j], target_seq[j]):
                num_correct += 1





    accuracy = num_correct / len(ds) * 100
    with open(log_file, 'a') as f:
        if loader==validation_loader:

            f.write('val accuracy for run'+str(run)+' epoch '+str(epoch)+' = ' + str(accuracy)+'%, val loss = '+str(loss.item()/len(ds)) + '\n')
        elif loader==train_loader:
            f.write('train val accuracy for run' + str(run) + ' epoch ' + str(epoch) + ' = ' + str(
                accuracy) + '%, train val loss = ' + str(loss.item() / len(ds)) + '\n')


    return accuracy, loss.item()/len(ds)


def validate_model_long(model, loader, dataset, run, epoch):

    num_correct = 0

    log_file=''


    log_file = long_validation_log
    dataset='Long Validation Set'
    ds = long_dataset

    if task=='NextTokenPredictionCrossEntropy' or task =='SemiDyck1BCE':
        criterion=nn.BCELoss()
    else:
        criterion = nn.MSELoss()

    total_loss = 0


    with open(log_file,'a') as f:
        f.write('////////////////////////////////////////\n')
        f.write('TEST '+dataset+'\n')


    for i, (sentences, labels, input_seq, target_seq, length) in enumerate(loader):
        output_seq = model(input_seq.to(device), length)


        output_seq = model.mask(output_seq, target_seq, length)
        loss = criterion(output_seq,target_seq)
        total_loss+=loss.item()



        output_seq = output_seq.view(batch_size, length[0], n_letters)
        target_seq = target_seq.view(batch_size, length[0], n_letters)


        out_seq = output_seq.clone().detach() >= epsilon
        out_seq = out_seq.float()


        for j in range(batch_size):


            if torch.equal(out_seq[j], target_seq[j]):
                num_correct += 1





    accuracy = num_correct / len(ds) * 100

    with open(log_file, 'a') as f:
        f.write('val accuracy for run'+str(run)+' epoch '+str(epoch)+' = ' + str(accuracy)+'%, val loss = '+str(loss.item()/len(ds)) + '\n')


    return accuracy, loss.item()/len(ds)

def test_model(model, loader, dataset):
    model.eval()
    num_correct = 0
    log_file=''

    if dataset=='short':
        log_file=test_log
        ds = test_dataset
    elif dataset=='long':
        log_file=long_test_log
        ds = long_dataset


    with open(log_file,'a') as f:
        f.write('////////////////////////////////////////\n')
        f.write('TEST '+dataset+'\n')


    for i, (sentences, labels, input_seq, target_seq, length) in enumerate(loader):
        output_seq = model(input_seq.to(device), length)


        output_seq = model.mask(output_seq, target_seq, length)



        output_seq = output_seq.view(batch_size, length[0], n_letters)
        target_seq = target_seq.view(batch_size, length[0], n_letters)


        out_seq = output_seq.clone().detach() >= epsilon
        out_seq = out_seq.float()




        for j in range(batch_size):
            if torch.equal(out_seq[j], target_seq[j]):
                num_correct += 1





    accuracy = num_correct / len(ds) * 100
    with open(log_file, 'a') as f:
        f.write('accuracy = ' + str(accuracy)+'%' + '\n')
    print(''+dataset+' test accuracy = '+ str(accuracy)+'%')


    return accuracy







if __name__=='__main__':
    main()



