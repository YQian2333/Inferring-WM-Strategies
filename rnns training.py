#%%
# interactive window settings
%reload_ext autoreload
%autoreload 2

import numpy as np
from scipy import ndimage
import pandas as pd

import matplotlib.pyplot as plt
import re, seaborn as sns

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning) # turn off warning messages

import gc
from itertools import permutations

import torch
import torch.nn as nn
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # if available, use GPU

# import custom functions
import myRNNs
import f_simulation
import f_trainRNN
import f_evaluateRNN
import f_stats
#%% task parameters
data_path = 'D:/data' # change to your own data path

locs = [0,1,2,3] # location conditions
ttypes = [1,2] # ttype conditions
locCombs = list(permutations(locs,2))

dt = 50 # The simulation timestep.
tau = dt # time constant, here set to be the same as the timestep
tRange = np.arange(-300,2700,dt) # simulation time
tLength = len(tRange) # The trial length.

N_trials = 1000 # number of trials
accFracs = (1, 0, 0) # fraction of correct, random incorrect (non-displayed-irrelavent), non-random incorrect (displayed-irrelavent) trials

#%% generate trial information
trialInfo = f_simulation.generate_trials_balance(N_trials, locs, ttypes, accFracs) # generate trials with balanced number per condition
N_trials = len(trialInfo) # recalculate N_trials based on generated trials

#%%

###############
# 8ch version #
###############

#%% simulate input time series

# input signal parameters
gocue = True # include go cue
vloc = 1.0 # input strength in location channels
vttype = 1.0 # input strength in type(color) channel, non used
noise = 0.1 # background random noise level
vcue = 1 # input strength in go cue
vtgt = 1 # target input strength
vdis = 0.25 # non used
taskVersion = 'seqSingle' # task version, just a note

# define the event markers
if tRange.min() < 0:
    trialEvents = {'bsl':[-300,0], 's1':[0,300],'d1':[300,1300],'s2':[1300,1600],'d2':[1600,2600],'go':[2600,3000]} # event markers # 
else:
    trialEvents = {'s1':[0,300],'d1':[300,1300],'s2':[1300,1600],'d2':[1600,2600],'go':[2600,3000]} # event markers # 

# generate input time series
X = f_simulation.generate_X_8ch(trialInfo, trialEvents, tRange, dt, vloc=vloc, vttype=vttype, noise=noise, gocue=gocue, vcue = vcue)
torch.cuda.empty_cache()
X_ = torch.tensor(X, dtype=torch.float32).to(device) # convert to pyTorch tensor

#%% RNN parameters

hidden_noise = 0.1 # hidden noise level
ext = 0 # external input to the hidden layer, non used
ceiling = None #[0,10] # value range of hidden unit activity

F_hidden = 'relu' # activation function of hidden layer
F_out = 'softmax' #'sigmoid' # activation function of output layer

N_in = X.shape[-1] # number of input units
N_out = len(locs) # number of output units
N_hidden = 64 # number of hidden units

#%% genereate expected output during baseline

expected0 = 1/N_out if F_out == 'softmax' else 0.5 # always force baseline output to stay at chance level
Y0_Bonds = (-300,0)
Y0 = f_simulation.generate_Y(N_out, trialInfo.choice.values, Y0_Bonds[0], Y0_Bonds[1], dt, expected0)
Y0_ = torch.tensor(Y0, dtype=torch.float32).to(device)

#%% define fitting windows for different strategies
fitIntervals = {'R@R':((0,1300),(1600,2600),), 'R&U':((300,1300), (1600,2600),), } 

#%%

##############
# train RNNs #
##############

#%% train RNNs
nModels = 100 #number of models to be trained per strategy
withBsl = True if tRange.min() <0 else False # include baseline to fitting or not

expectedFull = 1 # expected correct output at the target time

modelDicts = {} # save trained models

for nfi, kfi in enumerate(fitIntervals):
    
    print(kfi)
    modelDicts[kfi] = {i:{} for i in range(nModels)}
    
    expected1 = expectedFull if kfi == 'R&U' else expected0 # expected output during first interval, varied by strategy
    expected2 = expectedFull # expected output during second interval, assumed always at full level
    
    fi1 = fitIntervals[kfi][0]
    #Y1_Bonds = fi1
    
    if len(fitIntervals[kfi]) == 1:
        # if R@R strategy, use full expected output and fit to the final choice location
        Y1 = f_simulation.generate_Y(N_out, trialInfo.choice.values, fi1[0], fi1[1], dt, expectedFull)
    else:
        # if R&U strategy, use expected output during first interval
        Y1 = f_simulation.generate_Y(N_out, trialInfo.loc1.values, fi1[0], fi1[1], dt, expected1)
        
    Y1_ = torch.tensor(Y1, dtype=torch.float32).to(device)
    
    if len(fitIntervals[kfi])>1:
        
        fi2 = fitIntervals[kfi][1] # fit interval 2, only applicable to R&U
        #Y2_Bonds = fi2
        Y2 = f_simulation.generate_Y(N_out, trialInfo.choice.values, fi2[0], fi2[1], dt, expected2)
        Y2_ = torch.tensor(Y2, dtype=torch.float32).to(device)
    
    Ys_ = ((Y0_,Y0_Bonds,1,), ) if withBsl else ()
    
    regWeight1 = 3 if kfi == 'R&U' else 5 #0.5 # regularization weight of first interval
    regWeight2 = 3 if kfi == 'R&U' else 3 #0.5 # regularization weight of second interval
    
    # wrap Ys_ in a tuple
    if len(fitIntervals[kfi])>1:
        Ys_ += ((Y1_,fi1,regWeight1,),(Y2_,fi2,regWeight2,),) 
    else: 
        Ys_ += ((Y1_,fi1,regWeight1,),) 
    
    
    for i in range(nModels):
        print(f'model n={i}')
        
        # initialize RNN
        modelD = myRNNs.decayRNN(N_in, N_hidden, N_out, dt = dt, tau = tau, external = ext, hidden_noise = hidden_noise, ceiling = ceiling, 
                                 F_hidden = F_hidden, F_out = F_out, device= device, init_hidden='orthogonal_', useLinear_hidden = False, seed = i).to(device)
        
        # train RNN
        losses = f_trainRNN.train_model(modelD, trialInfo, X_, Ys_, tRange, frac = 0.2, criterion = nn.MSELoss(), 
                                        learning_rate = 0.0001, n_iter = 1000*5, loss_cutoff = 0.001, lr_cutoff = 1e-7, l2reg=False)
        
        # save model and losses
        modelDicts[kfi][i]['rnn'] = modelD
        modelDicts[kfi][i]['losses'] = losses
                
        del modelD
        torch.cuda.empty_cache()
        
        gc.collect()

#%% save trained models
np.save(f'{data_path}/modelDicts.npy', modelDicts, allow_pickle=True)

