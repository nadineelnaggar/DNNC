import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
import numpy as np
import argparse
import matplotlib.pyplot as plt
from models import VanillaLSTM, VanillaRNN, VanillaGRU, VanillaReLURNN, VanillaReLURNN_NoBias, VanillaReLURNNCorrectInitialisation, VanillaReLURNNCorrectInitialisationWithBias, RecurrentDNNC, RecurrentDNNCFrozenInputLayer, RecurrentDNNCClipping, RecurrentNonZeroReLUCounter
from Dyck_Generator_Suzgun_Batch import DyckLanguage
import random
from torch.utils.tensorboard import SummaryWriter
# from tensorboardX import SummaryWriter
from torch.utils.data import Dataset, DataLoader
from Dyck1_Datasets import NextTokenPredictionLongTestDataset, NextTokenPredictionShortTestDataset, NextTokenPredictionTrainDataset, NextTokenPredictionValidationDataset, NextTokenPredictionLongTestDataset_SAMPLE, NextTokenPredictionShortTestDataset_SAMPLE, NextTokenPredictionTrainDataset_SAMPLE, NextTokenPredictionValidationDataset_SAMPLE, NextTokenPrediction2TokenDataset
from torch.optim.lr_scheduler import StepLR
import math
import time

from Semi_Dyck1_Datasets import SemiDyck1TrainDataset, SemiDyck1ValidationDataset, SemiDyck1TestDataset, SemiDyck1ShortTestDataset, SemiDyck1Dataset1000tokens, SemiDyck1Dataset2000tokens_zigzag

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, help='input model name (VanillaLSTM, VanillaRNN, VanillaGRU)')
parser.add_argument('--task', type=str, help='NextTokenPrediction, BinaryClassification, TernaryClassification, NextTokenPredictionCrossEntropy, SemiDyck1MSE, SemiDyck1BCE, NextTokenPredictionSanityCheck')
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
parser.add_argument('--num_checkpoints', type=int,default=100, help='number of checkpoints we want to include if we dont need all of them (e.g., first 5 checkpoints only), stop after n checkpoints')



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
num_checkpoints = args.num_checkpoints

# checkpoint_step = int(num_epochs/4)
if args.checkpoint_step!=0:
    checkpoint_step = args.checkpoint_step
else:
    checkpoint_step = int(num_epochs / 4)

shuffle_dataset = args.shuffle_dataset
use_optimiser='Adam'

num_bracket_pairs = 25

length_bracket_pairs = 50
input_size=2


if runtime=='colab':
    path = "/content/drive/MyDrive/DNNC/EXPT_LOGS/Dyck1_"+str(task)+"/Minibatch_Training/"+model_name+"/"\
           +str(batch_size)+"_batch_size/"+str(learning_rate)+"_learning_rate/"+str(num_epochs)+"_epochs/"\
           +str(lr_scheduler_step)+"_lr_scheduler_step/"+str(lr_scheduler_gamma)+"_lr_scheduler_gamma/"\
           +str(hidden_size)+"_hidden_units/"+str(num_runs)+"_runs/shuffle_"+str(shuffle_dataset)+"/"
elif runtime=='local':
    path = "/Users/nadineelnaggar/Google Drive/DNNC/EXPT_LOGS/Dyck1_"+str(task)+"/Minibatch_Training/"+model_name+"/"\
       +str(batch_size)+"_batch_size/"+str(learning_rate)+"_learning_rate/"+str(num_epochs)+"_epochs/"\
       +str(lr_scheduler_step)+"_lr_scheduler_step/"+str(lr_scheduler_gamma)+"_lr_scheduler_gamma/"\
       +str(hidden_size)+"_hidden_units/"+str(num_runs)+"_runs/shuffle_"+str(shuffle_dataset)+"/"
elif runtime=='linux':
    path = "EXPT_LOGS/Dyck1_"+str(task)+"/Minibatch_Training/"+model_name+"/"\
       +str(batch_size)+"_batch_size/"+str(learning_rate)+"_learning_rate/"+str(num_epochs)+"_epochs/"\
       +str(lr_scheduler_step)+"_lr_scheduler_step/"+str(lr_scheduler_gamma)+"_lr_scheduler_gamma/"\
       +str(hidden_size)+"_hidden_units/"+str(num_runs)+"_runs/shuffle_"+str(shuffle_dataset)+"/"

# import os
# if not os.path.isdir(path):
#         os.makedirs(path)
#         print('FOLDER ',path, 'CREATED')
# else:
#     print('FOLDER',path,'EXISTS')


print('model_name = ',model_name)
print('task = ',task)
print('feedback = ',feedback)
print('hidden_size = ',hidden_size)
print('batch_size = ',batch_size)
print('num_layers = ',num_layers)
print('learning_rate = ',learning_rate)
print('num_epochs = ',num_epochs)
print('num_runs = ',num_runs)


if torch.cuda.is_available():
    device=torch.device("cuda")
elif torch.backends.mps.is_available():
    device=torch.device("mps")
else:
    device=torch.device("cpu")

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

checkpoint = path+ 'Dyck1_' + task + '_' + str(
        num_bracket_pairs) + '_bracket_pairs_' + model_name + '_Feedback_' + feedback + '_' +str(batch_size) +'_batch_size_'+'_' + str(
        hidden_size) + 'hidden_units_' + use_optimiser + '_lr=' + str(learning_rate) + '_' + str(
        num_epochs) + 'epochs_'+str(lr_scheduler_step)+"lr_scheduler_step_"+str(lr_scheduler_gamma)+"lr_scheduler_gamma_"+ str(num_runs)+'runs' + '_CHECKPOINT_'


output_activation = 'Sigmoid'

if task == 'TernaryClassification':
    num_classes = 3
    output_activation = 'Softmax'
elif task == 'BinaryClassification' or task == 'NextTokenPrediction' or task == 'NextTokenPredictionCrossEntropy' or task=='SemiDyck1MSE' or task=='SemiDyck1BCE' or task=='NextTokenPredictionSanityCheck':
    num_classes = 2
    output_activation = 'Sigmoid'

def select_model(model_name, input_size, hidden_size, num_layers,batch_size, num_classes, output_activation):
    if model_name=='VanillaLSTM':
        selected_model = VanillaLSTM(input_size,hidden_size, num_layers, batch_size, num_classes, output_activation=output_activation)
    elif model_name=='VanillaRNN':
        selected_model = VanillaRNN(input_size, hidden_size, num_layers, batch_size, num_classes, output_activation=output_activation)
    elif model_name=='VanillaGRU':
        selected_model = VanillaGRU(input_size,hidden_size, num_layers, batch_size, num_classes, output_activation=output_activation)
    elif model_name == 'VanillaReLURNN':
        selected_model = VanillaReLURNN(input_size, hidden_size, num_layers, batch_size, num_classes, output_activation=output_activation)
    elif model_name=='VanillaReLURNNCorrectInitialisation':
        selected_model = VanillaReLURNNCorrectInitialisation(input_size, hidden_size, num_layers, batch_size, num_classes, output_activation=output_activation)
    elif model_name=='VanillaReLURNNCorrectInitialisationWithBias':
        selected_model = VanillaReLURNNCorrectInitialisationWithBias(input_size, hidden_size, num_layers, batch_size, num_classes, output_activation=output_activation)
    elif model_name=='RecurrentDNNC':
        model = RecurrentDNNC(input_size, hidden_size, num_layers, batch_size, num_classes, output_activation=output_activation)
    elif model_name=='RecurrentDNNCFrozenInputLayer':
        model = RecurrentDNNCFrozenInputLayer(input_size, hidden_size, num_layers, batch_size, num_classes, output_activation=output_activation)
    elif model_name == 'RecurrentDNNCClipping':
        model = RecurrentDNNCClipping(input_size, hidden_size, num_layers, batch_size, num_classes,
                              output_activation=output_activation)
    elif model_name=='RecurrentNonZeroReLUCounter':
        model = RecurrentNonZeroReLUCounter(input_size,hidden_size,num_layers,batch_size,num_classes,output_activation=output_activation)



    return selected_model.to(device)

for run in range(num_runs):
   num_checkpoints = num_epochs
   checkpoint_count = 0
   for epoch in range(num_epochs):
        if epoch%checkpoint_step==0 and checkpoint_count<=num_checkpoints:
            checkpoint_count+=1

            checkpoint_model = select_model(model_name,input_size,hidden_size,num_layers,batch_size,num_classes,output_activation)
            # checkpoint_model.to(device)
            checkpoint_path = checkpoint+'run'+str(run)+"_epoch"+str(epoch)+".pth"

            checkpt = torch.load(checkpoint_path)
            checkpoint_model.load_state_dict(checkpt['model_state_dict'])
            checkpoint_model.to(device)

            for name, param in checkpoint_model.named_parameters():
                if param.grad is not None:
                    print(f"Layer: {name}")
                    print(f"Gradient: {param.grad}")
                else:
                    print(f"Layer: {name} has no gradient.")


