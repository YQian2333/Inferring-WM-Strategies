# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 14:23:51 2024

@author: aka2333
"""
# In[]
%reload_ext autoreload
%autoreload 2

import numpy as np
import scipy
from scipy import stats
from scipy.stats import vonmises # circular distribution
from scipy import ndimage

import pandas as pd

# basic plot functions
import matplotlib as mpl
import matplotlib.pyplot as plt
import re, seaborn as sns

# turn off warning messages
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# system paths
import os
import sys
import gc
sys.path.append(r'C:\Users\aka2333\OneDrive\phd\project')
sys.path.append(r'C:\Users\wudon\OneDrive\phd\project')

import time # timer

from itertools import permutations, combinations, product # itertools


import matplotlib.ticker as ticker
from matplotlib.pyplot import MultipleLocator #从pyplot导入MultipleLocator类，这个类用于设置刻度间隔
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec  # 导入专门用于网格分割及子图绘制的扩展包Gridspec

# In[] import pyTorch
import torch
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # define device, may need to change with Mac OS

# In[]
import myRNNs
import f_simulation
import f_trainRNN
import f_evaluateRNN
# In[] testing analysis
import f_subspace
import f_stats
import f_decoding
import f_plotting

# In[]
# check paths
if os.path.exists('E:/NUS/PhD/data'):
    data_path = 'E:/NUS/PhD/data' # for desktop
else:
    data_path = 'D:/NUS/PhD/data' # for laptop

if os.path.exists('E:/NUS/PhD/data/pseudo_ww'):
    save_path = 'E:/NUS/PhD/data/pseudo_ww' # for desktop
else:
    save_path = 'D:/NUS/PhD/data/pseudo_ww' # for laptop


if os.path.exists('E:/NUS/PhD'):
    phd_path = 'E:/NUS/PhD' # for desktop
else:
    phd_path = 'D:/NUS/PhD' # for laptop
# In[]
locs = [0,1,2,3] # location conditions
ttypes = [1,2] # ttype conditions

locCombs = list(permutations(locs,2))
# In[]
dt = 50 # The simulation timestep.
# time axis
tRange = np.arange(-300,2700,dt)
tLength = len(tRange) # The trial length.

# In[]
N_batch = 1000 # trial pool size
accFracs = (1, 0, 0) # fraction of correct, random incorrect (non-displayed-irrelavent), non-random incorrect (displayed-irrelavent) trials

#trialInfo = f_simulation.generate_trials(N_batch, locs, ttypes, accFracs)
trialInfo = f_simulation.generate_trials_balance(N_batch, locs, ttypes, accFracs)
N_batch = len(trialInfo)

# In[]
# simulate input values for each trial across time

gocue = True
vloc = 1.0 # input strength in location channels
vttype = 1.0 # input strength in type(color) channel
noise = 0.1 # background random noise level

vcue = 1

vtgt, vdis = 1, 0.25
# In[] setting parameters for RNN
torch.cuda.empty_cache() # empty GPU
tau = 50

hidden_noise = 0.1
ext = 0
ceiling = [0,10]#None #
F_hidden = 'relu'
F_out = 'softmax'#'sigmoid'


# In[]

###############
# 8ch version #
###############

# In[]
# task version 1: sequntial input, distractor & retarget, our version of task
taskVersion = 'seqSingle'

if tRange.min() < 0:
    trialEvents = {'bsl':[-300,0], 's1':[0,300],'d1':[300,1300],'s2':[1300,1600],'d2':[1600,2600],'go':[2600,3000]} # event markers # 
else:
    trialEvents = {'s1':[0,300],'d1':[300,1300],'s2':[1300,1600],'d2':[1600,2600],'go':[2600,3000]} # event markers # 
X = f_simulation.generate_X_8ch(trialInfo, trialEvents, tRange, dt, vloc=vloc, vttype=vttype, noise=noise, gocue=gocue, vcue = vcue)

X_ = torch.tensor(X, dtype=torch.float32).to(device)


# In[]
# task version 2: sequentially displayed, 4 loc channels, cued recall
#taskVersion = 'seqMulti'
#trialEvents = {'bsl':[-300,0], 's1':[0,300],'d1':[300,1300],'s2':[1300,1600],'d2':[1600,2600],'cue':[2600,2900], 'd3':[2900,3900],'go':[3900,4500]} # event markers
#X = f_simulation.generate_X_6ch_seqMulti(trialInfo, trialEvents, tRange, dt, vloc=vloc, vttype=vttype, noise=noise, gocue=gocue)
#X_ = torch.tensor(X, dtype=torch.float32).to(device)

# In[]
# task version 3: simultaneous displayed, 8 loc channels, cued selection
#taskVersion = 'simMulti'
#trialEvents = {'bsl':[-300,0], 's1':[0,500],'d1':[500,1500],'cue':[1500,1800],'d2':[1800,2800],'go':[2800,3300]} # event markers
#X = f_simulation.generate_X_8ch_simMulti(trialInfo, trialEvents, tRange, dt, vloc=vloc, vttype=vttype, noise=noise, gocue=gocue)
#X_ = torch.tensor(X, dtype=torch.float32).to(device)

# In[]
kappa = 3
# In[]
#N_in = len(locs) + len(ttypes) + int(gocue) # The number of network inputs.
#N_in = len(locs) * len(ttypes) + int(gocue) # The number of network inputs.
N_in = X.shape[-1]
N_out = len(locs)
N_hidden = 64

# In[] genereate expected output values at different time windows
# always force baseline output to stay at chance level
expected0 = 1/N_out
Y0_Bonds = (-300,0)
Y0 = f_simulation.generate_Y(N_out, trialInfo.choice.values, Y0_Bonds[0], Y0_Bonds[1], dt, expected0)

Y0_ = torch.tensor(Y0, dtype=torch.float32).to(device)


# In[]
fitIntervals = {'ed2':((0,1300),(1600,2600),), 'ed12':((300,1300), (1600,2600),), } #{'go':((2600,2700),)}#'sd2':((0,1300),(1300,2600),), 
# {'sd12':((0,1300), (1300,2600),), }
# 'end':((2990,3000),), 'go':((300,1300), (2600,3000),), 'ld2':((300,1300), (2100,2600),), 
# intervals to fit the model outputs 
#fitIntervals = {'ed12':((300,1300),(1600,2600),), 'ld12':((700,1300),(2000,2600),), } #'s2':((1300,2600),), 'ed2':((1600,2600),), 'ld2':((2100,2600),), 
#%%

##############
# train RNNs #
##############

# In[]
nModels = 5#100 # number of models to be trained per interval
withBsl = True if tRange.min() <0 else False

expectedFull = 1 #0.75 #
# In[]
modelDicts = {} # save trained models

for nfi, kfi in enumerate(fitIntervals):
    
    print(kfi)
    modelDicts[kfi] = {i:{} for i in range(nModels)}
    
    fi1 = fitIntervals[kfi][0]
    
    expected1 = expectedFull if kfi[-2:] == '12' else 0.25 #0.5 #
    expected2 = expectedFull #0.5 #
    
    
    
    Y1_Bonds = fi1
    
    if len(fitIntervals[kfi]) == 1:
        Y1 = f_simulation.generate_Y(N_out, trialInfo.choice.values, Y1_Bonds[0], Y1_Bonds[1], dt, expectedFull)
    else:
        Y1 = f_simulation.generate_Y(N_out, trialInfo.loc1.values, Y1_Bonds[0], Y1_Bonds[1], dt, expected1)
        
    #Y1 = f_simulation.generate_Y_circular(N_out, trialInfo.loc1.values, Y1_Bonds[0], Y1_Bonds[1], dt, expected1, kappa)
    Y1_ = torch.tensor(Y1, dtype=torch.float32).to(device)
    
    if len(fitIntervals[kfi])>1:
        fi2 = fitIntervals[kfi][1]
        
        Y2_Bonds = fi2
        Y2 = f_simulation.generate_Y(N_out, trialInfo.choice.values, Y2_Bonds[0], Y2_Bonds[1], dt, expected2)
        #Y2 = f_simulation.generate_Y_circular(N_out, trialInfo.choice.values, Y2_Bonds[0], Y2_Bonds[1], dt, expected2, kappa)
        Y2_ = torch.tensor(Y2, dtype=torch.float32).to(device)
    
    Ys_ = ((Y0_,Y0_Bonds,1,), ) if withBsl else ()
    
    regWeight1 = 3 if kfi[-2:] == '12' else 5 #0.5 #
    regWeight2 = 3 if kfi[-2:] == '12' else 3 #0.5 #
    
    if len(fitIntervals[kfi])>1:
        Ys_ += ((Y1_,Y1_Bonds,regWeight1,),(Y2_,Y2_Bonds,regWeight2,),) 
    else: 
        Ys_ += ((Y1_,Y1_Bonds,regWeight1,),) #(Y0_,Y0_Bonds,1,), 
    
    
    for i in range(nModels):
        print(f'model n={i}')
        
        # if decay RNN
        modelD = myRNNs.decayRNN(N_in, N_hidden, N_out, dt = dt, tau = tau, external = ext, hidden_noise = hidden_noise, ceiling = ceiling, 
                                 F_hidden = F_hidden, F_out = F_out, device= device, init_hidden='orthogonal_', useLinear_hidden = False, seed = i).to(device)
        losses = f_trainRNN.train_model(modelD, trialInfo, X_, Ys_, tRange, frac = 0.2, criterion = nn.MSELoss(), 
                                        learning_rate = 0.0001, n_iter = 1000*5, loss_cutoff = 0.001, lr_cutoff = 1e-7, l2reg=False)
        #losses = f_trainRNN.train_model_circular(modelD, trialInfo, X_, Ys_, tRange, frac = 0.2, learning_rate = 0.0001, n_iter = 1000*2, loss_cutoff = 0.001, lr_cutoff = 1e-7)
        
        modelDicts[kfi][i]['rnn'] = modelD
        modelDicts[kfi][i]['losses'] = losses
        
        _, _, _, test_setID, test_X, _ = f_simulation.split_dataset(X_, Ys_[0][0], frac = 0.8, ranseed=114514)
        test_label = trialInfo.loc[test_setID,'choice'].astype('int').values
        acc_memo = f_evaluateRNN.evaluate_acc(modelD, test_X, test_label, toPrint=True)
        
        del modelD
        torch.cuda.empty_cache()
        
        gc.collect()

    # save/load models
    #Ybounds = [''.join([str(int(j[1][0]/100)), str(int(j[1][1]/100))]) for j in Ys_[1:]]
    #modelName = taskVersion + '_' + '_'.join([str(i) for i in [N_in, N_hidden]]+Ybounds)#, tau, dt, int(ext), ceiling, F_hidden, F_out
# In[]
np.save(f'{save_path}/modelDicts8_basic_35_rvd_sd.npy', modelDicts, allow_pickle=True) #_{expected2}
#%%
modelDicts = np.load(f'{save_path}/modelDicts8_basic_35_rvd_ed.npy', allow_pickle=True).item()
#%%

##############
# adds on... #
##############

# In[]
#nModels = 100 # number of models to be trained per interval
#withBsl = True if tRange.min() <0 else False
#expectedFull = 1 #0.75 #
# In[]
#modelDicts = np.load(f'{save_path}/modelDicts8_basic_35_rvd_ed.npy', allow_pickle=True).item()
#%%
#for nfi, kfi in enumerate(fitIntervals):
#    
#    nexisting = len(modelDicts[kfi])
#    
#    print(kfi)
    #modelDicts[kfi] = {i:{} for i in range(nexisting, nModels)}
    
#    fi1 = fitIntervals[kfi][0]
    
#    expected1 = expectedFull if kfi[-2:] == '12' else 0.25 #0.5 #
#    expected2 = expectedFull #0.5 #
    
    
    
#    Y1_Bonds = fi1
    
#    if len(fitIntervals[kfi]) == 1:
#        Y1 = f_simulation.generate_Y(N_out, trialInfo.choice.values, Y1_Bonds[0], Y1_Bonds[1], dt, expected1)
#    else:
#        Y1 = f_simulation.generate_Y(N_out, trialInfo.loc1.values, Y1_Bonds[0], Y1_Bonds[1], dt, expected1)
        
    #Y1 = f_simulation.generate_Y_circular(N_out, trialInfo.loc1.values, Y1_Bonds[0], Y1_Bonds[1], dt, expected1, kappa)
#    Y1_ = torch.tensor(Y1, dtype=torch.float32).to(device)
    
#    if len(fitIntervals[kfi])>1:
#        fi2 = fitIntervals[kfi][1]
        
#        Y2_Bonds = fi2
#        Y2 = f_simulation.generate_Y(N_out, trialInfo.choice.values, Y2_Bonds[0], Y2_Bonds[1], dt, expected2)
#        #Y2 = f_simulation.generate_Y_circular(N_out, trialInfo.choice.values, Y2_Bonds[0], Y2_Bonds[1], dt, expected2, kappa)
#        Y2_ = torch.tensor(Y2, dtype=torch.float32).to(device)
    
#    Ys_ = ((Y0_,Y0_Bonds,1,), ) if withBsl else ()
    
#    regWeight1 = 3 if kfi[-2:] == '12' else 5 #0.5 #
#    regWeight2 = 3 if kfi[-2:] == '12' else 3 #0.5 #
    
#    if len(fitIntervals[kfi])>1:
#        Ys_ += ((Y1_,Y1_Bonds,regWeight1,),(Y2_,Y2_Bonds,regWeight2,),) 
#    else: 
#        Ys_ += ((Y1_,Y1_Bonds,regWeight1,),) #(Y0_,Y0_Bonds,1,), 
    
    
#    for i in range(nexisting, nModels):
#        modelDicts[kfi][i] = {}
#        print(f'model n={i}')
        
        # if decay RNN
#        modelD = myRNNs.decayRNN(N_in, N_hidden, N_out, dt = dt, tau = tau, external = ext, hidden_noise = hidden_noise, ceiling = ceiling, 
#                                 F_hidden = F_hidden, F_out = F_out, device= device, init_hidden='orthogonal_', useLinear_hidden = False, seed = i).to(device)
#        losses = f_trainRNN.train_model(modelD, trialInfo, X_, Ys_, tRange, frac = 0.2, criterion = nn.MSELoss(), 
#                                        learning_rate = 0.0001, n_iter = 1000*5, loss_cutoff = 0.001, lr_cutoff = 1e-7, l2reg=False)
#        #losses = f_trainRNN.train_model_circular(modelD, trialInfo, X_, Ys_, tRange, frac = 0.2, learning_rate = 0.0001, n_iter = 1000*2, loss_cutoff = 0.001, lr_cutoff = 1e-7)
        
#        modelDicts[kfi][i]['rnn'] = modelD
#        modelDicts[kfi][i]['losses'] = losses
        
#        _, _, _, test_setID, test_X, _ = f_simulation.split_dataset(X_, Ys_[0][0], frac = 0.8, ranseed=114514)
#        test_label = trialInfo.loc[test_setID,'choice'].astype('int').values
#        acc_memo = f_evaluateRNN.evaluate_acc(modelD, test_X, test_label, toPrint=True)
        
#        del modelD
#        torch.cuda.empty_cache()
        
#        gc.collect()
#%%
#np.save(f'{save_path}/modelDicts8_basic_35_rvd_ed.npy', modelDicts, allow_pickle=True) #_{expected2}














#%%

#################
# evaluate RNNs #
#################

# In[]

#_, _, _, test_setID, test_X, _ = f_simulation.split_dataset(X_, Y0_, frac = 0.5, ranseed=114514)
_, _, _, test_setID, test_X, _ = f_simulation.split_dataset_balance(X_, Y0_, trialInfo, frac = 0.5, ranseed=114514)
test_Info = trialInfo.loc[test_setID,:].reset_index(drop = True)
test_label = trialInfo.loc[test_setID,'choice'].astype('int').values


bins = 50
tslice = (tRange.min(),tRange.max()+dt)
tbins = np.arange(tslice[0], tslice[1], bins)

#%% model performance, plot states
plot_samples = (0,1,2)

#performances1, performances2 = {},{}
performancesX1, performancesX2 = {},{}
performancesX1_shuff, performancesX2_shuff = {},{}
performancesXtt, performancesXtt_shuff = {},{}
EVRs = {}
for nfi, kfi in enumerate(fitIntervals):
    print(kfi)
    models_dict = modelDicts[kfi]
    
    label = f'fit interval:{kfi}'
    strategyLabel = 'Rehearse & Update' if kfi[-2:] == '12' else 'Retrieve at Recall'
    
    #pfm = {'Retarget':[], 'Distractor':[]}
    
    pfmsX1 = {tt:[] for tt in ttypes}
    pfmsX2 = {tt:[] for tt in ttypes}
    
    pfmsX1_shuff = {tt:[] for tt in ttypes}
    pfmsX2_shuff = {tt:[] for tt in ttypes}
    
    pfmsXtt = []
    pfmsXtt_shuff = []
    
    evrs = []

    ii=0
    
    for i in range(len(modelDicts[kfi])):
        print(i)
        modelD = models_dict[i]['rnn']
        acc_memo = f_evaluateRNN.evaluate_acc(modelD, test_X, test_label, toPrint=True, 
                                              checkpointX = np.where(tRange <= fitIntervals[kfi][-1][-1])[0].max()) #
        
        if acc_memo >=75:
            ii+=1
            if (ii in plot_samples):
                f_evaluateRNN.plot_states(modelD, test_Info, test_X, tRange, trialEvents, dt=dt, locs = (0,1,2,3), ttypes = (1,2), 
                                          lcX = np.arange(0,1,1), cues=False, cseq = None, label = strategyLabel, 
                                          withhidden=False, withannote=False, save_path=save_path, savefig=False)
                #f_evaluateRNN.plot_weights(modelD,gen_in=False)



#%% model performance, plot states
accuracies_rnns = {}

for nfi, kfi in enumerate(fitIntervals):
    print(kfi)
    models_dict = modelDicts[kfi]
    
    label = f'fit interval:{kfi}'
    strategyLabel = 'Rehearse & Update' if kfi[-2:] == '12' else 'Retrieve at Recall'
    
    #pfm = {'Retarget':[], 'Distractor':[]}
    
    accuraciesT = []

    for i in range(len(modelDicts[kfi])):
        modelD = models_dict[i]['rnn']
        acc_memo = f_evaluateRNN.evaluate_acc(modelD, test_X, test_label, toPrint=False, 
                                              checkpointX = np.where(tRange <= fitIntervals[kfi][-1][-1])[0].max()) #
        accuraciesT += [acc_memo]
    
    accuraciesT = np.array(accuraciesT)
    accuracies_rnns[strategyLabel] = accuraciesT
    print(f"{strategyLabel}: M(SD) = {accuraciesT.mean():.3f}({accuraciesT.std():.3f})")
#%%

###########################
# Full Space Decodability #
###########################

# In[] compute decodability

plot_samples = (0,1,2,3,4)

#performances1, performances2 = {},{}
performancesX1, performancesX2 = {},{}
performancesX1_shuff, performancesX2_shuff = {},{}
performancesXtt, performancesXtt_shuff = {},{}
EVRs = {}
for nfi, kfi in enumerate(fitIntervals):
    print(kfi)
    models_dict = modelDicts[kfi]
    
    label = f'fit interval:{kfi}'
    strategyLabel = 'Rehearse & Update' if kfi[-2:] == '12' else 'Retrieve at Recall'
    
    #pfm = {'Retarget':[], 'Distractor':[]}
    
    pfmsX1 = {tt:[] for tt in ttypes}
    pfmsX2 = {tt:[] for tt in ttypes}
    
    pfmsX1_shuff = {tt:[] for tt in ttypes}
    pfmsX2_shuff = {tt:[] for tt in ttypes}
    
    pfmsXtt = []
    pfmsXtt_shuff = []
    
    evrs = []

    ii=0
    
    for i in range(len(modelDicts[kfi])):
        print(i)
        modelD = models_dict[i]['rnn']
        acc_memo = f_evaluateRNN.evaluate_acc(modelD, test_X, test_label, toPrint=True, 
                                              checkpointX = np.where(tRange <= fitIntervals[kfi][-1][-1])[0].max()) #
        
        evrsT = f_evaluateRNN.rnns_EVRs(modelD, trialInfo, X_, Y0_, tRange, dt = 50, bins = 50, 
                                        nPCs = (0,15), nIters = 1, nBoots = 10, pca_tWins = ((300,1300), (1600,2600),), 
                                        pDummy = True, label=f'{strategyLabel} ', toPlot = False, shuff_excludeInv = False)
        
        if acc_memo >=75:
            ii+=1
            if (ii in plot_samples):
                f_evaluateRNN.plot_states(modelD, test_Info, test_X, tRange, trialEvents, dt=dt, locs = (0,1,2,3), ttypes = (1,2), 
                                          lcX = np.arange(0,1,1), cues=False, cseq = None, label = strategyLabel, 
                                          withhidden=False, withannote=False, save_path=save_path, savefig=False)
                #f_evaluateRNN.plot_weights_mixed(modelD)
            
            if ii <=100:
                #pfmT,_ = f_evaluateRNN.plot_crossTemp_lda(modelD, trialInfo, X_, Y0_, tRange, trialEvents, nIters=5, dt=dt,bins=50, pca_tWins=((300,1300),(1600,2600),), label=f'{kfi} ', toPlot = False) # (100,800),
                pfmsX12T, pfmsX12_shuffT, evrsT = f_evaluateRNN.rnns_lda12X(modelD, trialInfo, X_, Y0_, tRange, dt = 50, bins = 50, 
                                                                nPCs = (0,15), nIters = 1, nBoots = 10, pca_tWins = ((300,1300), (1600,2600),), 
                                                                pDummy = True, label=f'{strategyLabel} ', toPlot = False, shuff_excludeInv = False) # (100,800),
                
                pfmsXttT, pfmsXtt_shuffT, _ = f_evaluateRNN.rnns_ldattX(modelD, trialInfo, X_, Y0_, tRange, dt = 50, bins = 50, 
                                                                nPCs = (0,15), nIters = 1, nBoots = 10, pca_tWins = ((300,1300), (1600,2600),), 
                                                                pDummy = True, label=f'{strategyLabel} ', toPlot = False, shuff_excludeInv = False) # (100,800),
                #pfmT,_ = f_evaluateRNN.plot_lda(modelD, trialInfo, X_, Y0_, tRange, trialEvents, nIters=5, dt = dt,bins=50, targetItem='locX', pca_tWins=((800,1300),(2100,2600),), label=f'{kfi}, locX', toPlot = False) #(100,800),
                
                pfmsX1[1] += [np.array(pfmsX12T[0][1])] #.mean(0)
                pfmsX1[2] += [np.array(pfmsX12T[0][2])] #.mean(0)
                
                pfmsX2[1] += [np.array(pfmsX12T[1][1])] #.mean(0)
                pfmsX2[2] += [np.array(pfmsX12T[1][2])] #.mean(0)
                
                pfmsX1_shuff[1] += [np.array(pfmsX12_shuffT[0][1])] #.mean(0)
                pfmsX1_shuff[2] += [np.array(pfmsX12_shuffT[0][2])] #.mean(0)
                
                pfmsX2_shuff[1] += [np.array(pfmsX12_shuffT[1][1])] #.mean(0)
                pfmsX2_shuff[2] += [np.array(pfmsX12_shuffT[1][2])] #.mean(0)
                
                evrs += [np.array(evrsT)]
                
                pfmsXtt += [np.array(pfmsXttT)]
                pfmsXtt_shuff += [np.array(pfmsXtt_shuffT)]
                
                del modelD
                torch.cuda.empty_cache()
        
                gc.collect()
            
    #performances1[kfi] = pfm1
    #performances2[kfi] = pfm2
    performancesX1[kfi] = pfmsX1
    performancesX2[kfi] = pfmsX2
    performancesX1_shuff[kfi] = pfmsX1_shuff
    performancesX2_shuff[kfi] = pfmsX2_shuff
    EVRs[kfi] = np.array(evrs)
    
    performancesXtt[kfi] = pfmsXtt
    performancesXtt_shuff[kfi] = pfmsXtt_shuff

# In[] store decodability
np.save(f'{phd_path}/outputs/rnns/' + 'performanceX1_full_rnn.npy', performancesX1, allow_pickle=True)
np.save(f'{phd_path}/outputs/rnns/' + 'performanceX2_full_rnn.npy', performancesX2, allow_pickle=True)
np.save(f'{phd_path}/outputs/rnns/' + 'performanceX1_full_shuff_rnn.npy', performancesX1_shuff, allow_pickle=True)
np.save(f'{phd_path}/outputs/rnns/' + 'performanceX2_full_shuff_rnn.npy', performancesX2_shuff, allow_pickle=True)

np.save(f'{phd_path}/outputs/rnns/' + 'performanceXtt_full_rnn.npy', performancesXtt, allow_pickle=True)
np.save(f'{phd_path}/outputs/rnns/' + 'performanceXtt_full_shuff_rnn.npy', performancesXtt_shuff, allow_pickle=True)
# In[] plot item decodability full space
for nfi, kfi in enumerate(fitIntervals):
    print(kfi)
    strategyLabel = 'Rehearse & Update' if kfi[-2:] == '12' else 'Retrieve at Recall'

    pfmX1 = performancesX1[kfi]
    pfmX2 = performancesX2[kfi]
    pfmX1_shuff = performancesX1_shuff[kfi]
    pfmX2_shuff = performancesX2_shuff[kfi]
    
    # decodability with/without permutation P value
    bins = 50
    tslice = (tRange.min(), tRange.max()+dt)
    #tRange = np.arange(-300,3000,dt)
    
    tbins = np.arange(tslice[0], tslice[1], bins) #
    
    
    if len(pfmX1[1])>0:
        conditions = (('ttype', 1), ('ttype', 2))
        fig = plt.figure(figsize=(28, 24), dpi=100)
        
        for condition in conditions:
            
            ttypeT = 'Retarget' if condition[-1]==1 else 'Distractor'
            ttypeT_ = 'Retarget' if condition[-1]==1 else 'Distraction'
            tt = 1 if condition[-1]==1 else 2
            
                                
            pfmTX1 = np.array(pfmX1[tt]).squeeze().mean(1)
            pfmTX2 = np.array(pfmX2[tt]).squeeze().mean(1)
            pfmTX1_shuff = np.array(pfmX1_shuff[tt]).squeeze().mean(1)#np.swapaxes(np.swapaxes(np.concatenate(pfmX1_shuff[ttypeT],-1),1,2),0,1)
            pfmTX2_shuff = np.array(pfmX2_shuff[tt]).squeeze().mean(1)#np.swapaxes(np.swapaxes(np.concatenate(pfmX2_shuff[ttypeT],-1),1,2),0,1)
            
            pPerms_pfm1 = np.ones((len(tbins), len(tbins)))
            pPerms_pfm2 = np.ones((len(tbins), len(tbins)))

            for t in range(len(tbins)):
                for t_ in range(len(tbins)):
                    #pPerms_pfm1[t, t_] = f_stats.permutation_p(pfmTX1.mean(0)[t,t_], pfmTX1_shuff[:,t,t_], tail='greater')
                    #pPerms_pfm2[t, t_] = f_stats.permutation_p(pfmTX2.mean(0)[t,t_], pfmTX2_shuff[:,t,t_], tail='greater')
                    pPerms_pfm1[t, t_] = f_stats.permutation_pCI(pfmTX1[:,t,t_], pfmTX1_shuff[:,t,t_], tail='greater', alpha=5)
                    pPerms_pfm2[t, t_] = f_stats.permutation_pCI(pfmTX2[:,t,t_], pfmTX2_shuff[:,t,t_], tail='greater', alpha=5)
                    


            vmax = 1
            plt.subplot(2,2,tt)
            ax = plt.gca()
            sns.heatmap(pd.DataFrame(pfmTX1.mean(0), index=tbins,columns=tbins), cmap = 'magma', vmin = 0.0, vmax = vmax, ax = ax)#, vmax = 0.6, xticklabels = 100, yticklabels = 100, vmin = cbar_min, vmax = cbar_max
            
            #from scipy import ndimage
            smooth_scale = 10
            z = ndimage.zoom(pPerms_pfm1, smooth_scale)
            ax.contour(np.linspace(0, len(tbins), len(tbins) * smooth_scale),
                    np.linspace(0, len(tbins), len(tbins) * smooth_scale),
                    z, levels=([0.05]), colors='white', alpha = 1, linewidths = 3)
            
            ax.invert_yaxis()
            
            
            # event lines
            for i in [0, 1300, 2600]:
                ax.plot(tbins, np.full_like(tbins,list(tbins).index(i)), 'w-.', linewidth=4)
                ax.plot(np.full_like(tbins,list(tbins).index(i)), tbins, 'w-.', linewidth=4)
            
            ax.set_xticks([list(tbins).index(i) for i in [0, 1300, 2600]])
            ax.set_xticklabels(['S1','S2','Go Cue'], rotation=0, fontsize = 20)
            ax.set_xlabel('Test Timebin (ms)', labelpad = 10, fontsize = 25)
            ax.set_yticks([list(tbins).index(i) for i in [0, 1300, 2600]])
            ax.set_yticklabels(['S1','S2','Go Cue'], fontsize = 20)
            ax.set_ylabel('Train Timebin (ms)', labelpad = 10, fontsize = 25)
            
            cbar = ax.collections[0].colorbar
            # here set the labelsize by 20
            cbar.ax.tick_params(labelsize=20)
            
            ax.set_title(f'{ttypeT_}, Item1', fontsize = 30, pad = 20)
            
            
            # item2
            plt.subplot(2,2,tt+2)
            ax = plt.gca()
            sns.heatmap(pd.DataFrame(pfmTX2.mean(0), index=tbins,columns=tbins), cmap = 'magma', vmin = 0.0, vmax = vmax, ax = ax)#, vmax = 0.6, xticklabels = 100, yticklabels = 100, vmin = cbar_min, vmax = cbar_max
            
            #from scipy import ndimage
            smooth_scale = 10
            z = ndimage.zoom(pPerms_pfm2, smooth_scale)
            ax.contour(np.linspace(0, len(tbins), len(tbins) * smooth_scale),
                    np.linspace(0, len(tbins), len(tbins) * smooth_scale),
                    z, levels=([0.05]), colors='white', alpha = 1, linewidths = 3)
            
            ax.invert_yaxis()
            
            
            # event lines
            for i in [0, 1300, 2600]:
                ax.plot(tbins, np.full_like(tbins,list(tbins).index(i)), 'w-.', linewidth=4)
                ax.plot(np.full_like(tbins,list(tbins).index(i)), tbins, 'w-.', linewidth=4)
            
            ax.set_xticks([list(tbins).index(i) for i in [0, 1300, 2600]])
            ax.set_xticklabels(['S1','S2','Go Cue'], rotation=0, fontsize = 20)
            ax.set_xlabel('Test Timebin (ms)', labelpad = 10, fontsize = 25)
            ax.set_yticks([list(tbins).index(i) for i in [0, 1300, 2600]])
            ax.set_yticklabels(['S1','S2','Go Cue'], fontsize = 20)
            ax.set_ylabel('Train Timebin (ms)', labelpad = 10, fontsize = 25)
            
            cbar = ax.collections[0].colorbar
            # here set the labelsize by 20
            cbar.ax.tick_params(labelsize=20)
            
            ax.set_title(f'{ttypeT_}, Item2', fontsize = 30, pad = 20)
            
        plt.tight_layout()
        plt.subplots_adjust(top = 0.95)
        plt.suptitle(f'{strategyLabel}, Full Space', fontsize = 35, y=1) #, Arti_Noise = {arti_noise_level}
        plt.show()

        fig.savefig(f'{save_path}/decodabilityX_full_{strategyLabel}.tif', bbox_inches='tight')
        
#%% plot ttype decodability full space        
for nfi, kfi in enumerate(fitIntervals):
    print(kfi)
    strategyLabel = 'Rehearse & Update' if kfi[-2:] == '12' else 'Retrieve at Recall'

    pfmX1 = performancesXtt[kfi]
    pfmX1_shuff = performancesXtt_shuff[kfi]
    
    # decodability with/without permutation P value
    bins = 50
    tslice = (tRange.min(), tRange.max()+dt)
    #tRange = np.arange(-300,3000,dt)
    
    tbins = np.arange(tslice[0], tslice[1], bins) #
    
    
    if len(pfmX1[1])>0:
        
        fig = plt.figure(figsize=(7, 6), dpi=300)
                            
        pfmTX1 = np.array(pfmX1).squeeze().mean(1)
        pfmTX1_shuff = np.array(pfmX1_shuff).squeeze().mean(1)#np.swapaxes(np.swapaxes(np.concatenate(pfmX1_shuff[ttypeT],-1),1,2),0,1)
        
        pPerms_pfm1 = np.ones((len(tbins), len(tbins)))
        
        for t in range(len(tbins)):
            for t_ in range(len(tbins)):
                #pPerms_pfm1[t, t_] = f_stats.permutation_p(pfmTX1.mean(0)[t,t_], pfmTX1_shuff[:,t,t_], tail='greater')
                #pPerms_pfm2[t, t_] = f_stats.permutation_p(pfmTX2.mean(0)[t,t_], pfmTX2_shuff[:,t,t_], tail='greater')
                pPerms_pfm1[t, t_] = f_stats.permutation_pCI(pfmTX1[:,t,t_], pfmTX1_shuff[:,t,t_], tail='greater', alpha=5)
                
        vmax = 1
        plt.subplot(1,1,1)
        ax = plt.gca()
        sns.heatmap(pd.DataFrame(pfmTX1.mean(0), index=tbins,columns=tbins), cmap = 'magma', vmin = 0.25, vmax = vmax, ax = ax)#, vmax = 0.6, xticklabels = 100, yticklabels = 100, vmin = cbar_min, vmax = cbar_max
        
        #from scipy import ndimage
        smooth_scale = 10
        z = ndimage.zoom(pPerms_pfm1, smooth_scale)
        ax.contour(np.linspace(0, len(tbins), len(tbins) * smooth_scale),
                np.linspace(0, len(tbins), len(tbins) * smooth_scale),
                z, levels=([0.05]), colors='white', alpha = 1, linewidths = 3)
        
        ax.invert_yaxis()
        
        
        # event lines
        for i in [0, 1300, 2600]:
            ax.plot(tbins, np.full_like(tbins,list(tbins).index(i)), 'w-.', linewidth=4)
            ax.plot(np.full_like(tbins,list(tbins).index(i)), tbins, 'w-.', linewidth=4)
        
        ax.set_xticks([list(tbins).index(i) for i in [0, 1300, 2600]])
        ax.set_xticklabels(['S1','S2','Go Cue'], rotation=0, fontsize = 15)
        ax.set_xlabel('Test Timebin (ms)', labelpad = 10, fontsize = 20)
        ax.set_yticks([list(tbins).index(i) for i in [0, 1300, 2600]])
        ax.set_yticklabels(['S1','S2','Go Cue'], rotation=90, fontsize = 15)
        ax.set_ylabel('Train Timebin (ms)', labelpad = 10, fontsize = 20)
        
        cbar = ax.collections[0].colorbar
        # here set the labelsize by 20
        cbar.ax.tick_params(labelsize=15)
        
        ax.set_title(f'Trial Type', fontsize = 25, pad = 20)
        
            
        plt.tight_layout()
        plt.subplots_adjust(top = 1)
        plt.suptitle(f'{strategyLabel}, Full Space', fontsize = 25, y=1.25) #, Arti_Noise = {arti_noise_level}
        plt.show()

        fig.savefig(f'{save_path}/decodability_ttX_full_{strategyLabel}.tif', bbox_inches='tight')


#%%

################
# euDist drift #
################

#%% calculate euclidean distance between centroids
nPerms = 100
nBoots = 1
infoMethod='lda'

bins = 50
tslice = (tRange.min(),tRange.max()+dt)
tbins = np.arange(tslice[0], tslice[1], bins)
end_D1s, end_D2s = np.arange(800,1300+bins,bins), np.arange(2100,2600+bins,bins)

#euDists = {}
#euDists_centroids = {}
euDists_centroids2 = {}

#euDists_shuff = {}
#euDists_centroids_shuff = {}
euDists_centroids2_shuff = {}

for nfi, kfi in enumerate(fitIntervals):
    print(kfi)
    
    label = f'fit interval:{kfi}'
    strategyLabel = 'Rehearse & Update' if kfi[-2:] == '12' else 'Retrieve at Recall'
    
    models_dict = modelDicts[kfi]
    
    #euDists[kfi] = {tt:[] for tt in ttypes}
    #euDists_centroids[kfi] = {tt:[] for tt in ttypes}
    euDists_centroids2[kfi] = {tt:[] for tt in ttypes}
    
    #euDists_shuff[kfi] = {tt:[] for tt in ttypes}
    #euDists_centroids_shuff[kfi] = {tt:[] for tt in ttypes}
    euDists_centroids2_shuff[kfi] = {tt:[] for tt in ttypes}

    vecs_C, projs_C, projsAll_C, trialInfos_C, data_3pc_C, _, evrs_C, vecs_C_shuff, projs_C_shuff, projsAll_C_shuff, trialInfos_C_shuff, data_3pc_C_shuff, _ = f_evaluateRNN.generate_choiceVectors(models_dict, trialInfo, X_, Y0_, tRange, nBoots=nBoots, nPerms=nPerms, pca_tWins=((300,1300),(1600,2600),), 
                                                                                                                                                                                                    dt=dt, toPlot = False, label=f'{strategyLabel}',plot3d=False)
    
    
    for i in range(len(modelDicts[kfi])):
        
        for tt in ttypes:
            #euDists[kfi][tt].append([])
            #euDists_centroids[kfi][tt].append([])
            euDists_centroids2[kfi][tt].append([])
            
            #euDists_shuff[kfi][tt].append([])
            #euDists_centroids_shuff[kfi][tt].append([])
            euDists_centroids2_shuff[kfi][tt].append([])
        
        
        for nbt in range(nBoots):
            geomsC_valid = (vecs_C[i][nbt], projs_C[i][nbt], projsAll_C[i][nbt], trialInfos_C[i][nbt], data_3pc_C[i][nbt])
            
            #euDistT = f_evaluateRNN.get_euDist(geomsC_valid, locs = locs, ttypes = ttypes, bins = bins, dt = dt, tslice = tslice, end_D1s = end_D1s, end_D2s = end_D2s, zscore = False, normalizeMinMax=(-1,1), hideLocs=())
            #euDist_centroidsT = f_evaluateRNN.get_euDist_centroids(geomsC_valid, locs = locs, ttypes = ttypes, bins = bins, dt = dt, tslice = tslice, end_D1s = end_D1s, end_D2s = end_D2s, zscore = False)
            euDist_centroids2T = f_evaluateRNN.get_euDist_centroids2(geomsC_valid, locs = locs, ttypes = ttypes, bins = bins, dt = dt, tslice = tslice, end_D1s = end_D1s, end_D2s = end_D2s, zscore = False, normalizeMinMax=(-1,1), hideLocs=(0,2))
            
            for tt in ttypes:
                #euDists[kfi][tt][i] += [euDistT[tt]]
                #euDists_centroids[kfi][tt][i] += [euDist_centroidsT[tt]]
                euDists_centroids2[kfi][tt][i] += [euDist_centroids2T[tt]]
            
        for npm in range(nPerms): # do nperms shuffles
            #geomsC_shuff = (vecs_C_shuff[i][0][npm], projs_C_shuff[i][0][npm], projsAll_C_shuff[i][0][npm], trialInfos_C_shuff[i][0][npm], data_3pc_C_shuff[i][0][npm])
            #shuff
            #euDistT_shuff = f_evaluateRNN.get_euDist(geomsC_shuff, locs = locs, ttypes = ttypes, bins = bins, dt = dt, tslice = tslice, end_D1s = end_D1s, end_D2s = end_D2s, zscore = False, normalizeMinMax=(-1,1), hideLocs=())
            #euDist_centroidsT_shuff = f_evaluateRNN.get_euDist_centroids(geomsC_shuff, locs = locs, ttypes = ttypes, bins = bins, dt = dt, tslice = tslice, end_D1s = end_D1s, end_D2s = end_D2s, zscore = False)
            euDist_centroids2T_shuff = f_evaluateRNN.get_euDist_centroids2(geomsC_valid, locs = locs, ttypes = ttypes, bins = bins, dt = dt, tslice = tslice, end_D1s = end_D1s, end_D2s = end_D2s, zscore = False, normalizeMinMax=(-1,1), hideLocs=(0,2), shuffleBsl=True)
        
            for tt in ttypes:
                #euDists_shuff[kfi][tt][i] += [euDistT_shuff[tt]]
                #euDists_centroids_shuff[kfi][tt][i] += [euDist_centroidsT_shuff[tt]]
                euDists_centroids2_shuff[kfi][tt][i] += [euDist_centroids2T_shuff[tt]]

#%%
#np.save(f'{phd_path}/outputs/rnns/' + 'euDists_rnns_normalized_full.npy', euDists, allow_pickle=True)
#np.save(f'{phd_path}/outputs/rnns/' + 'euDists_rnns_centroids.npy', euDists_centroids, allow_pickle=True)
np.save(f'{phd_path}/outputs/rnns/' + 'euDists_rnns_centroids2_normalized_hide02.npy', euDists_centroids2, allow_pickle=True)            

#np.save(f'{phd_path}/outputs/rnns/' + 'euDists_rnns_shuff_normalized_full.npy', euDists_shuff, allow_pickle=True)
#np.save(f'{phd_path}/outputs/rnns/' + 'euDists_rnns_centroids_shuff.npy', euDists_centroids_shuff, allow_pickle=True)
np.save(f'{phd_path}/outputs/rnns/' + 'euDists_rnns_centroids2_shuff_normalized_hide02.npy', euDists_centroids2_shuff, allow_pickle=True)
#%%


####################################################
# item info by subspaces (item-specific & readout) #
####################################################


# In[] info decoding by subspace projection
plot_samples = (0,1,)
checkpoints = [150, 550, 1050, 1450, 1850, 2350]#
checkpointsLabels = ['S1','ED1','LD1','S2','ED2','LD2'] #, 2800
avgInterval = {150:150, 550:250, 1050:250, 1450:150, 1850:250, 2350:250} #, 2800:200
#nIters = 1
nPerms = 10
nBoots = 1
infoMethod='lda'

bins = 50
tslice = (tRange.min(),tRange.max()+dt)
tbins = np.arange(tslice[0], tslice[1], bins)

evrs = {}

info3ds_1, info3ds_2 = {},{}
info3ds_1_shuff, info3ds_2_shuff = {},{}

infos_C1, infos_C2 = {},{}
infos_C1_shuff, infos_C2_shuff = {},{}

infos_C1X, infos_C2X = {},{}
infos_C1X_shuff, infos_C2X_shuff = {},{}

for nfi, kfi in enumerate(fitIntervals):
    print(kfi)
    
    label = f'fit interval:{kfi}'
    strategyLabel = 'Rehearse & Update' if kfi[-2:] == '12' else 'Retrieve at Recall'
    
    models_dict = modelDicts[kfi]
    
    evrsThreshold = 0.9#5 if kfi[-2:] == '12' else 0.9
    evrs[kfi] = []
    

    info3d_1, info3d_2 = [],[]
    info_C1, info_C2 = [],[]
    info_C1X, info_C2X = [],[]

    info3d_1_shuff, info3d_2_shuff = [],[]
    info_C1_shuff, info_C2_shuff = [],[]
    info_C1X_shuff, info_C2X_shuff = [],[]
    
    
    vecs_C, projs_C, projsAll_C, trialInfos_C, data_3pc_C, pcas_C, evrs_C, vecs_C_shuff, projs_C_shuff, projsAll_C_shuff, trialInfos_C_shuff, data_3pc_C_shuff, _ = f_evaluateRNN.generate_choiceVectors(models_dict, trialInfo, X_, Y0_, tRange, nBoots=nBoots, pca_tWins=((300,1300),(1600,2600),), 
                                                                                                                                                                                                    dt=dt, toPlot = False, label=f'{strategyLabel}',plot3d=False) #,{i}
    
    vecs, projs, projsAll, trialInfos, _, _, vecs_shuff, projs_shuff, projsAll_shuff, trialInfos_shuff, _ = f_evaluateRNN.generate_itemVectors(models_dict, trialInfo, X_, Y0_, tRange, checkpoints, avgInterval, 
                                                                                                                                               nBoots=nBoots, pca_tWins=((300,1300),(1600,2600),), dt=dt, adaptPCA=pcas_C, adaptEVR=evrs_C) #
    
    
    
    for i in range(len(modelDicts[kfi])):
        print(i)
        geoms_valid, geoms_shuff = (vecs[i], projs[i], projsAll[i], trialInfos[i]), (vecs_shuff[i], projs_shuff[i], projsAll_shuff[i], trialInfos_shuff[i])
        geomsC_valid, geomsC_shuff = (vecs_C[i], projs_C[i], projsAll_C[i], trialInfos_C[i], data_3pc_C[i]), (vecs_C_shuff[i], projs_C_shuff[i], projsAll_C_shuff[i], trialInfos_C_shuff[i], data_3pc_C_shuff[i])
                
        #print(evrs_C.sum(1).mean())
        
        #if evrs_C.sum(1).mean() > evrsThreshold:
        
        
        info3d, info3d_shuff = f_evaluateRNN.itemInfo_by_plane(geoms_valid, checkpoints, nPerms = nPerms, infoMethod=infoMethod, shuff_excludeInv = False)
        info_CIX, info_CIX_shuff = f_evaluateRNN.itemInfo_by_planeCX(geomsC_valid, nPerms = nPerms, bins = bins, tslice = tslice, shuff_excludeInv = False)
        
        #evrs[kfi] += [evrs_C]
        
        
        info3d_1 += [info3d[0]]
        info3d_2 += [info3d[1]]

        info_C1X += [info_CIX[0]]
        info_C2X += [info_CIX[1]]
        
        info3d_1_shuff += [info3d_shuff[0]]
        info3d_2_shuff += [info3d_shuff[1]]

        
        info_C1X_shuff += [info_CIX_shuff[0]]
        info_C2X_shuff += [info_CIX_shuff[1]]
        

        

    # store
    info3ds_1[kfi], info3ds_2[kfi] = info3d_1,info3d_2
    infos_C1X[kfi], infos_C2X[kfi] = info_C1X,info_C2X
    
    info3ds_1_shuff[kfi], info3ds_2_shuff[kfi] = info3d_1_shuff,info3d_2_shuff
    infos_C1X_shuff[kfi], infos_C2X_shuff[kfi] = info_C1X_shuff,info_C2X_shuff
# In[] save decodability 
np.save(f'{phd_path}/outputs/rnns/' + 'performance1_item_rnn.npy', info3ds_1, allow_pickle=True)
np.save(f'{phd_path}/outputs/rnns/' + 'performance2_item_rnn.npy', info3ds_2, allow_pickle=True)
np.save(f'{phd_path}/outputs/rnns/' + 'performanceX1_readout_rnn.npy', infos_C1X, allow_pickle=True)
np.save(f'{phd_path}/outputs/rnns/' + 'performanceX2_readout_rnn.npy', infos_C2X, allow_pickle=True)

np.save(f'{phd_path}/outputs/rnns/' + 'performance1_item_shuff_rnn.npy', info3ds_1_shuff, allow_pickle=True)
np.save(f'{phd_path}/outputs/rnns/' + 'performance2_item_shuff_rnn.npy', info3ds_2_shuff, allow_pickle=True)
np.save(f'{phd_path}/outputs/rnns/' + 'performanceX1_readout_shuff_rnn.npy', infos_C1X_shuff, allow_pickle=True)
np.save(f'{phd_path}/outputs/rnns/' + 'performanceX2_readout_shuff_rnn.npy', infos_C2X_shuff, allow_pickle=True)
    
# In[]
#performancesX1 = np.load(f'{save_path}/performanceX1_rnn.npy', allow_pickle=True).item()
#performancesX2 = np.load(f'{save_path}/performanceX2_rnn.npy', allow_pickle=True).item()
#performancesX1_shuff = np.load(f'{save_path}/performanceX1_shuff_rnn.npy', allow_pickle=True).item()
#performancesX2_shuff = np.load(f'{save_path}/performanceX2_shuff_rnn.npy', allow_pickle=True).item()
# In[] load 

#infos_C1X = np.load(f'{save_path}/infos_C1X.npy', allow_pickle=True).item()
#infos_C2X = np.load(f'{save_path}/infos_C2X.npy', allow_pickle=True).item()
#infos_C1X_shuff = np.load(f'{save_path}/infos_C1X_shuff.npy', allow_pickle=True).item()
#infos_C2X_shuff = np.load(f'{save_path}/infos_C2X_shuff.npy', allow_pickle=True).item()
# In[] plot item plane decodability
for nfi, kfi in enumerate(fitIntervals):
    print(kfi)
    info3d_1,info3d_2 = info3ds_1[kfi], info3ds_2[kfi]
    #info_C1,info_C2 = infos_C1[kfi], infos_C2[kfi]
    info_C1X,info_C2X = infos_C1X[kfi], infos_C2X[kfi] 

    info3d_1_shuff,info3d_2_shuff = info3ds_1_shuff[kfi], info3ds_2_shuff[kfi]
    #info_C1_shuff,info_C2_shuff = infos_C1_shuff[kfi], infos_C2_shuff[kfi]
    info_C1X_shuff,info_C2X_shuff = infos_C1X_shuff[kfi], infos_C2X_shuff[kfi] 


    label = f'fit interval:{kfi}'
    strategyLabel = 'Rehearse & Update' if kfi[-2:] == '12' else 'Retrieve at Recall'
        
    
    ########################
    # plot info in omega^2 #
    ########################

    decode_proj1_3d = {1:np.array([info3d_1[i][1].mean(0).mean(0) for i in range(len(info3d_1))]),
                       2:np.array([info3d_1[j][2].mean(0).mean(0) for j in range(len(info3d_1))])}
    decode_proj2_3d = {1:np.array([info3d_2[i][1].mean(0).mean(0) for i in range(len(info3d_2))]),
                       2:np.array([info3d_2[j][2].mean(0).mean(0) for j in range(len(info3d_2))])}
    
    decode_proj1_3d_shuff = {1:np.array([info3d_1_shuff[i][1].mean(0).mean(0) for i in range(len(info3d_1_shuff))]),
                             2:np.array([info3d_1_shuff[j][2].mean(0).mean(0) for j in range(len(info3d_1_shuff))])
                             }
    decode_proj2_3d_shuff = {1:np.array([info3d_2_shuff[i][1].mean(0).mean(0) for i in range(len(info3d_2_shuff))]),
                             2:np.array([info3d_2_shuff[j][2].mean(0).mean(0) for j in range(len(info3d_2_shuff))])
                             }


    if len(decode_proj1_3d[1])>0:
        fig = plt.figure(figsize=(12, 4), dpi=100)
        for tt in ttypes:
    
            #colorT = 'b' if tt == 1 else 'm'
            condT = 'Retarget' if tt == 1 else 'Distraction'
    
            infoLabel = 'Accuracy' if infoMethod=='lda' else 'PEV'
            
            plt.subplot(1,2,tt)
            ax = plt.gca()
            ax.errorbar(np.arange(0, len(checkpoints), 1), decode_proj1_3d[tt].mean(0), yerr = decode_proj1_3d[tt].std(0), marker = 'o', color = 'b', label = 'Item1', capsize=4)
            
            pPerms_decode1_3d = np.array([f_stats.permutation_pCI(decode_proj1_3d[tt][:,t], decode_proj1_3d_shuff[tt][:,t],alpha=5,tail='greater') for t in range(decode_proj1_3d[tt].shape[1])])
            
            trans = ax.get_xaxis_transform()
            for nc, cp in enumerate(checkpoints):
                if 0.05 < pPerms_decode1_3d[nc] <= 0.1:
                    ax.annotate('+', xy=(nc, 0.05), xycoords = trans, annotation_clip = False, color = 'b', ha = 'center', va = 'center', fontsize=10)
                elif 0.01 < pPerms_decode1_3d[nc] <= 0.05:
                    ax.annotate('*', xy=(nc, 0.05), xycoords = trans, annotation_clip = False, color = 'b', ha = 'center', va = 'center', fontsize=10)
                elif pPerms_decode1_3d[nc] <= 0.01:
                    ax.annotate('**', xy=(nc, 0.05), xycoords = trans, annotation_clip = False, color = 'b', ha = 'center', va = 'center', fontsize=10)
    
            pPerms_decode2_3d = np.array([f_stats.permutation_pCI(decode_proj2_3d[tt][:,t], decode_proj2_3d_shuff[tt][:,t],alpha=5,tail='greater') for t in range(decode_proj2_3d[tt].shape[1])])
            
            ax.errorbar(np.arange(0, len(checkpoints), 1), decode_proj2_3d[tt].mean(0), yerr = decode_proj2_3d[tt].std(0), marker = 'o', color = 'm', label = 'Item2', capsize=4)
            trans = ax.get_xaxis_transform()
            for nc, cp in enumerate(checkpoints):
                if 0.05 < pPerms_decode2_3d[nc] <= 0.1:
                    ax.annotate('+', xy=(nc, 0.1), xycoords = trans, annotation_clip = False, color = 'm', ha = 'center', va = 'center', fontsize=10)
                elif 0.01 < pPerms_decode2_3d[nc] <= 0.05:
                    ax.annotate('*', xy=(nc, 0.1), xycoords = trans, annotation_clip = False, color = 'm', ha = 'center', va = 'center', fontsize=10)
                elif pPerms_decode2_3d[nc] <= 0.01:
                    ax.annotate('**', xy=(nc, 0.1), xycoords = trans, annotation_clip = False, color = 'm', ha = 'center', va = 'center', fontsize=10)
    
            ax.set_title(f'{condT}', fontsize = 15, pad = 20)
            ax.set_xticks([n for n in range(len(checkpoints))])
            ax.set_xticklabels(checkpointsLabels, fontsize = 15)
            ax.set_xlabel('Time', fontsize = 15)
            ax.set_ylim((-0.1,1.1))
            #ax.set_yticklabels(checkpoints, fontsize = 10)
            ax.set_ylabel(f'{infoLabel}', fontsize = 15)
            
            if tt == 2:
                ax.legend(bbox_to_anchor = (1.05,0.65), fontsize=15)#loc='lower right'
    
        plt.subplots_adjust(top = 0.8)
        plt.suptitle(f'Mean Information, {strategyLabel}', fontsize = 20, y=0.95)
        plt.tight_layout()
        plt.show()

        fig.savefig(f'{save_path}/decodability_items_rnn_{strategyLabel}.tif')
    
    
    #############################
    # plot info on choice plane #
    #############################

    #decode_projC1_3d = {1:np.array([info_C1[i][1].mean(0) for i in range(len(info_C1))]),2:np.array([info_C1[j][2].mean(0) for j in range(len(info_C1))])}
    #decode_projC2_3d = {1:np.array([info_C2[i][1].mean(0) for i in range(len(info_C2))]),2:np.array([info_C2[j][2].mean(0) for j in range(len(info_C2))])}
    
    #if len(decode_projC1_3d[1])>0:
    #    plt.figure(figsize=(12, 4), dpi=100)
    #    for tt in ttypes:
    #
    #        #colorT = 'b' if tt == 1 else 'm'
    #        condT = 'Retarget' if tt == 1 else 'Distraction'
    #
    #        infoLabel = 'Accuracy' if infoMethod=='lda' else 'PEV'
    #        
    #        plt.subplot(1,2,tt)
    #        ax = plt.gca()
    #        ax.plot(np.arange(0, decode_projC1_3d[tt].shape[1], 1), decode_projC1_3d[tt].mean(axis=-1).mean(axis=0), color = 'b', label = 'Item1')
    #        ax.plot(np.arange(0, decode_projC2_3d[tt].shape[1], 1), decode_projC2_3d[tt].mean(axis=-1).mean(axis=0), color = 'm', label = 'Item2')
    #        ax.fill_between(np.arange(0, decode_projC1_3d[tt].shape[1], 1), (decode_projC1_3d[tt].mean(axis=-1).mean(axis=0)-decode_projC1_3d[tt].mean(axis=-1).std(axis=0)), (decode_projC1_3d[tt].mean(axis=-1).mean(axis=0)+decode_projC1_3d[tt].mean(axis=-1).std(axis=0)), alpha = 0.1, color = 'b')
    #        ax.fill_between(np.arange(0, decode_projC2_3d[tt].shape[1], 1), (decode_projC2_3d[tt].mean(axis=-1).mean(axis=0)-decode_projC2_3d[tt].mean(axis=-1).std(axis=0)), (decode_projC2_3d[tt].mean(axis=-1).mean(axis=0)+decode_projC2_3d[tt].mean(axis=-1).std(axis=0)), alpha = 0.1, color = 'm')
            
    #        # event lines
    #        for i in [0, 300, 1300, 1600, 2600]:
                
    #            #ax.plot(tbins, np.full_like(tbins,list(tbins).index(i)), 'k-.', linewidth=4, alpha = 0.25)
    #            ax.plot(np.full_like(tbins,list(tbins).index(i)), tbins, 'k-.', linewidth=2, alpha = 0.25)
            
    #       #ax.set_title(f'{condT}, 3d', pad = 10)
    #        ax.set_title(f'{condT}', fontsize = 20, pad = 20)
    #        ax.set_xticks([list(tbins).index(i) for i in [0, 300, 1300, 1600, 2600]])
    #        ax.set_xticklabels([0, 300, 1300, 1600, 2600], fontsize = 10)
    #        ax.set_xlabel('Time', fontsize = 15)
    #        #ax.set_xlim((list(tbins).index(0),list(tbins).index(2600))) #(0,)
    #        ax.set_ylim((0,1.1))
    #        #ax.set_yticklabels(checkpoints, fontsize = 10)
    #        ax.set_ylabel(f'{infoLabel}', fontsize = 15)
    #        ax.legend(loc='upper right')
    
    #    plt.subplots_adjust(top = 0.8)
    #    plt.suptitle(f'Mean Information, {strategyLabel}', fontsize = 20, y=0.95)
    #    plt.tight_layout()
    #    plt.show()
# In[] plot readout decodability
for nfi, kfi in enumerate(fitIntervals):
    print(kfi)
    info3d_1,info3d_2 = info3ds_1[kfi], info3ds_2[kfi]
    #info_C1,info_C2 = infos_C1[kfi], infos_C2[kfi]
    info_C1X,info_C2X = infos_C1X[kfi], infos_C2X[kfi] 

    info3d_1_shuff,info3d_2_shuff = info3ds_1_shuff[kfi], info3ds_2_shuff[kfi]
    #info_C1_shuff,info_C2_shuff = infos_C1_shuff[kfi], infos_C2_shuff[kfi]
    info_C1X_shuff,info_C2X_shuff = infos_C1X_shuff[kfi], infos_C2X_shuff[kfi] 


    label = f'fit interval:{kfi}'
    strategyLabel = 'Rehearse & Update' if kfi[-2:] == '12' else 'Retrieve at Recall'
        
            
    ########################################
    # plot info on choice plane cross temp #
    ########################################
    decode_projC1X_3d = {1:np.array([info_C1X[i][1].mean(0).mean(0) for i in range(len(info_C1X))]),
                         2:np.array([info_C1X[j][2].mean(0).mean(0) for j in range(len(info_C1X))])}
    decode_projC2X_3d = {1:np.array([info_C2X[i][1].mean(0).mean(0) for i in range(len(info_C2X))]),
                         2:np.array([info_C2X[j][2].mean(0).mean(0) for j in range(len(info_C2X))])}
    
    decode_projC1X_3d_shuff = {1:np.array([info_C1X_shuff[i][1].mean(0).mean(0) for i in range(len(info_C1X_shuff))]),
                         2:np.array([info_C1X_shuff[j][2].mean(0).mean(0) for j in range(len(info_C1X_shuff))])}
    decode_projC2X_3d_shuff = {1:np.array([info_C2X_shuff[i][1].mean(0).mean(0) for i in range(len(info_C2X_shuff))]),
                         2:np.array([info_C2X_shuff[j][2].mean(0).mean(0) for j in range(len(info_C2X_shuff))])}
    

    infoLabel = 'Accuracy' if infoMethod=='lda' else 'PEV'
    #plt.figure(figsize=(12, 8), dpi=100)
    
    if len(decode_projC1X_3d[1])>0:
        fig = plt.figure(figsize=(28, 24), dpi=100)
        
        for tt in ttypes:
            
            condT = 'Retarget' if tt == 1 else 'Distraction'
            h0 = 0 if infoMethod == 'omega2' else 1/len(locs)
            
            pfm1, pfm2 = decode_projC1X_3d[tt], decode_projC2X_3d[tt]
            
            pPerms_decode1_3d = np.ones((len(tbins), len(tbins)))
            pPerms_decode2_3d = np.ones((len(tbins), len(tbins)))
            
            for t in range(len(tbins)):
                for t_ in range(len(tbins)):
                    pPerms_decode1_3d[t, t_] = f_stats.permutation_pCI(decode_projC1X_3d[tt][:,t,t_], decode_projC1X_3d_shuff[tt][:,t,t_], alpha=5, tail='greater')
                    pPerms_decode2_3d[t, t_] = f_stats.permutation_pCI(decode_projC2X_3d[tt][:,t,t_], decode_projC2X_3d_shuff[tt][:,t,t_], alpha=5, tail='greater')
                    
            
            
            # item1
            plt.subplot(2,2,tt)
            ax = plt.gca()
            sns.heatmap(pd.DataFrame(pfm1.mean(0), index=tbins,columns=tbins), cmap = 'magma', vmin = 0.0, vmax = 1, ax = ax)#, vmax = 0.6, xticklabels = 100, yticklabels = 100, vmin = cbar_min, vmax = cbar_max
            
            #from scipy import ndimage
            smooth_scale = 10
            z = ndimage.zoom(pPerms_decode1_3d, smooth_scale)
            ax.contour(np.linspace(0, len(tbins), len(tbins) * smooth_scale),
                    np.linspace(0, len(tbins), len(tbins) * smooth_scale),
                    z, levels=([0.01]), colors='white', alpha = 1, linewidths = 3)
            
            ax.invert_yaxis()
            
            
            # event lines
            for i in [0, 1300, 2600]:
                ax.plot(tbins, np.full_like(tbins,list(tbins).index(i)), 'w-.', linewidth=4)
                ax.plot(np.full_like(tbins,list(tbins).index(i)), tbins, 'w-.', linewidth=4)
            
            ax.set_xticks([list(tbins).index(i) for i in [0, 1300, 2600]])
            ax.set_xticklabels(['S1','S2','Go Cue'], rotation=0, fontsize = 20)
            ax.set_xlabel('Test Timebin (ms)', labelpad = 10, fontsize = 25)
            ax.set_yticks([list(tbins).index(i) for i in [0, 1300, 2600]])
            ax.set_yticklabels(['S1','S2','Go Cue'], fontsize = 20)
            ax.set_ylabel('Train Timebin (ms)', labelpad = 10, fontsize = 25)
            
            cbar = ax.collections[0].colorbar
            # here set the labelsize by 20
            cbar.ax.tick_params(labelsize=20)
            
            ax.set_title(f'{condT}, Item1', fontsize = 30, pad = 20)
            
            # item2
            plt.subplot(2,2,tt+2)
            ax = plt.gca()
            sns.heatmap(pd.DataFrame(pfm2.mean(0), index=tbins,columns=tbins), cmap = 'magma', vmin = 0.0, vmax = 1, ax = ax)#, vmax = 0.6, xticklabels = 100, yticklabels = 100, vmin = cbar_min, vmax = cbar_max
            
            #from scipy import ndimage
            smooth_scale = 10
            z = ndimage.zoom(pPerms_decode2_3d, smooth_scale)
            ax.contour(np.linspace(0, len(tbins), len(tbins) * smooth_scale),
                    np.linspace(0, len(tbins), len(tbins) * smooth_scale),
                    z, levels=([0.01]), colors='white', alpha = 1, linewidths = 3)
            
            ax.invert_yaxis()
            
            
            # event lines
            for i in [0, 1300, 2600]:
                ax.plot(tbins, np.full_like(tbins,list(tbins).index(i)), 'w-.', linewidth=4)
                ax.plot(np.full_like(tbins,list(tbins).index(i)), tbins, 'w-.', linewidth=4)
            
            ax.set_xticks([list(tbins).index(i) for i in [0, 1300, 2600]])
            ax.set_xticklabels(['S1','S2','Go Cue'], rotation=0, fontsize = 20)
            ax.set_xlabel('Test Timebin (ms)', labelpad = 10, fontsize = 25)
            ax.set_yticks([list(tbins).index(i) for i in [0, 1300, 2600]])
            ax.set_yticklabels(['S1','S2','Go Cue'], fontsize = 20)
            ax.set_ylabel('Train Timebin (ms)', labelpad = 10, fontsize = 25)
            
            cbar = ax.collections[0].colorbar
            # here set the labelsize by 20
            cbar.ax.tick_params(labelsize=20)
            
            ax.set_title(f'{condT}, Item2', fontsize = 30, pad = 20)
        
        plt.tight_layout()
        plt.subplots_adjust(top = 0.95)
        plt.suptitle(f'{strategyLabel}, Readout Subspace', fontsize = 35, y=1) #, Arti_Noise = {arti_noise_level}
        plt.show()
        fig.savefig(f'{save_path}/decodabilityX_readout_rnns_{strategyLabel}.tif')

#%%


#####################################
# ttype info by subspaces (readout) #
#####################################


# In[] trial type decoding by readout subspace projection
plot_samples = (0,1,)
checkpoints = [150, 550, 1050, 1450, 1850, 2350]#
checkpointsLabels = ['S1','ED1','LD1','S2','ED2','LD2'] #, 2800
avgInterval = {150:150, 550:250, 1050:250, 1450:150, 1850:250, 2350:250} #, 2800:200
#nIters = 1
nPerms = 10
nBoots = 1
infoMethod='lda'

bins = 50
tslice = (tRange.min(),tRange.max()+dt)
tbins = np.arange(tslice[0], tslice[1], bins)

infos_CttX, infos_CttX_shuff = {},{}

for nfi, kfi in enumerate(fitIntervals):
    print(kfi)
    
    label = f'fit interval:{kfi}'
    strategyLabel = 'Rehearse & Update' if kfi[-2:] == '12' else 'Retrieve at Recall'
    
    models_dict = modelDicts[kfi]
    
    info_CttX, info_CttX_shuff = [],[]
    
    vecs_C, projs_C, projsAll_C, trialInfos_C, data_3pc_C, _, evrs_C, vecs_C_shuff, projs_C_shuff, projsAll_C_shuff, trialInfos_C_shuff, data_3pc_C_shuff, _ = f_evaluateRNN.generate_choiceVectors(models_dict, trialInfo, X_, Y0_, tRange, nBoots=nBoots, pca_tWins=((300,1300),(1600,2600),), 
                                                                                                                                                                                                    dt=dt, toPlot = False, label=f'{strategyLabel}',plot3d=False) #,{i}
    
    
    
    for i in range(len(modelDicts[kfi])):
        print(i)
        geomsC_valid, geomsC_shuff = (vecs_C[i], projs_C[i], projsAll_C[i], trialInfos_C[i], data_3pc_C[i]), (vecs_C_shuff[i], projs_C_shuff[i], projsAll_C_shuff[i], trialInfos_C_shuff[i], data_3pc_C_shuff[i])
                
        info_CIX, info_CIX_shuff = f_evaluateRNN.ttypeInfo_by_planeCX(geomsC_valid, nPerms = nPerms, bins = bins, tslice = tslice)
        
        info_CttX += [info_CIX]
        
        info_CttX_shuff += [info_CIX_shuff]
        
    # store
    infos_CttX[kfi], infos_CttX_shuff[kfi] = info_CttX, info_CttX_shuff
    
# In[] save trial type decodability
np.save(f'{phd_path}/outputs/rnns/' + 'performance_ttX_readout_rnn.npy', infos_CttX, allow_pickle=True)
np.save(f'{phd_path}/outputs/rnns/' + 'performance_ttX_readout_shuff_rnn.npy', infos_CttX_shuff, allow_pickle=True)
#%% plot trial type decodability
# 

for nfi, kfi in enumerate(fitIntervals):
    label = f'fit interval:{kfi}'
    strategyLabel = 'Rehearse & Update' if kfi[-2:] == '12' else 'Retrieve at Recall'
    strategyLabel_ = 'R&U' if kfi[-2:] == '12' else 'R@R'
    
    fig = plt.figure(figsize=(7, 6), dpi=300)

    performanceT1 = infos_CttX[kfi]
    performanceT1_shuff = infos_CttX_shuff[kfi]

    pfm1 = np.array(performanceT1).mean(1).mean(1)
    pfm1_shuff = np.array(performanceT1_shuff).mean(1).mean(1)

    pvalues1 = np.ones((len(tbins), len(tbins)))
    for t in range(len(tbins)):
        for t_ in range(len(tbins)):
            pvalues1[t,t_] = f_stats.permutation_pCI(pfm1[:,t,t_], pfm1_shuff[:,t,t_,], tail = 'greater', alpha=0.05)
                

    vmax = 1

    plt.subplot(1,1,1)
    ax = plt.gca()
    sns.heatmap(pd.DataFrame(pfm1.mean(0), index=tbins,columns=tbins), cmap = 'magma', vmin = 0.25, ax = ax)#, vmax = vmax, vmax = 0.6, xticklabels = 100, yticklabels = 100, vmin = cbar_min, vmax = cbar_max

    #from scipy import ndimage
    smooth_scale = 10
    z = ndimage.zoom(pvalues1, smooth_scale)
    ax.contour(np.linspace(0, len(tbins), len(tbins) * smooth_scale),
                np.linspace(0, len(tbins), len(tbins) * smooth_scale),
                z, levels=([0.05]), colors='white', alpha = 1)

    ax.invert_yaxis()


    # event lines
    for i in [0, 1300, 2600]:
        ax.plot(tbins, np.full_like(tbins,list(tbins).index(i)), 'w-.', linewidth=4)
        ax.plot(np.full_like(tbins,list(tbins).index(i)), tbins, 'w-.', linewidth=4)

    ax.set_xticks([list(tbins).index(i) for i in [0, 1300, 2600]])
    ax.set_xticklabels(['S1', 'S2', 'Go-Cue'], rotation=0, fontsize = 15)
    ax.set_xlabel('Test Timebin (ms)', labelpad = 10, fontsize = 20)
    ax.set_yticks([list(tbins).index(i) for i in [0, 1300, 2600]])
    ax.set_yticklabels(['S1', 'S2', 'Go-Cue'], rotation=90, fontsize = 15)
    ax.set_ylabel('Train Timebin (ms)', labelpad = 10, fontsize = 20)

    cbar = ax.collections[0].colorbar
    # here set the labelsize by 20
    cbar.ax.tick_params(labelsize=20)

    ax.set_title(f'Trial Type', fontsize = 20, pad = 5)
        
        
    plt.tight_layout()
    plt.subplots_adjust(top = 1)
    plt.suptitle(f'{strategyLabel}, Readout Subspace', fontsize = 20, y=1.1) #, Arti_Noise = {arti_noise_level}
    plt.show()
    fig.savefig(f'{phd_path}/outputs/rnns/decodability_ttX_readout_{strategyLabel_}.tif', bbox_inches='tight')
#%%


##########################
# subspace relationships #
##########################


# In[] calculate geoms
plot_samples = (0,) #1,2,3,4,5,6,7,8,9,10
checkpoints = [150, 550, 1050, 1450, 1850, 2350]#
checkpointsLabels = ['S1','ED1','LD1','S2','ED2','LD2'] #, 2800
avgInterval = {150:150, 550:250, 1050:250, 1450:150, 1850:250, 2350:250} #, 2800:200
nPerms = 100
nBoots = 1
infoMethod='lda'
#%%
# choice items
cosThetas_choice, cosPsis_choice = {},{}
cosThetas_nonchoice, cosPsis_nonchoice = {},{}

cosThetas_choice_shuff, cosPsis_choice_shuff = {},{}
cosThetas_nonchoice_shuff, cosPsis_nonchoice_shuff = {},{}

# items v item
cosThetas_11, cosThetas_22, cosThetas_12 = {},{},{}
cosPsis_11, cosPsis_22, cosPsis_12 = {},{},{}

cosThetas_11_shuff, cosThetas_22_shuff, cosThetas_12_shuff = {},{},{}
cosPsis_11_shuff, cosPsis_22_shuff, cosPsis_12_shuff = {},{},{}

cosThetas_11_bsl, cosThetas_22_bsl, cosThetas_12_bsl = {},{},{}
cosPsis_11_bsl, cosPsis_22_bsl, cosPsis_12_bsl = {},{},{}

# item v readout
cosThetas_1C, cosThetas_2C = {},{}
cosPsis_1C, cosPsis_2C = {},{}

cosThetas_1C_shuff, cosThetas_2C_shuff = {},{}
cosPsis_1C_shuff, cosPsis_2C_shuff = {},{}

# code transferability items
performanceX_12, performanceX_21 = {},{}
performanceX_12_shuff, performanceX_21_shuff = {},{}


# code transferability choices/nonchoices
performanceX_rdc, performanceX_drc = {},{}
performanceX_rdnc, performanceX_drnc = {},{}
performanceX_rdc_shuff, performanceX_drc_shuff = {},{}
performanceX_rdnc_shuff, performanceX_drnc_shuff = {},{}

# item info by item subspace
info3ds_1, info3ds_2 = {},{}
info3ds_1_shuff, info3ds_2_shuff = {},{}

#infos_C1, infos_C2 = {},{}
#infos_C1_shuff, infos_C2_shuff = {},{}

#infos_C1X, infos_C2X = {},{}
#infos_C1X_shuff, infos_C2X_shuff = {},{}
EVRs_C1 = {}
EVRs_C2 = {}
#EVRs_M1 = {}
#EVRs_M2 = {}

#%%
# calculate geoms
for nfi, kfi in enumerate(fitIntervals):
    print(kfi)
    
    label = f'fit interval:{kfi}'
    strategyLabel = 'Rehearse & Update' if kfi[-2:] == '12' else 'Retrieve at Recall'
    
    models_dict = modelDicts[kfi]
    
    vecs_C, projs_C, projsAll_C, trialInfos_C, data_3pc_C, pcas_C, evrs_C, evrs2nd_C, vecs_C_shuff, projs_C_shuff, projsAll_C_shuff, trialInfos_C_shuff, data_3pc_C_shuff, _ = f_evaluateRNN.generate_choiceVectors(models_dict, trialInfo, X_, Y0_, tRange, nBoots=nBoots, nPerms=nPerms, pca_tWins=((300,1300),(1600,2600),), 
                                                                                                                                                                                                    dt=dt, toPlot = True, label=f'{strategyLabel}',plot3d=False,
                                                                                                                                                                                                    hideLocs = (0,2), savefig=False, save_path = f'{phd_path}/outputs/rnns/', normalizeMinMax=(-1,1), separatePlot=False) #
    
    #_, _, _, _, _, _, evrs1st_M, evrs2nd_M, _, _, _, _, _, _ = f_evaluateRNN.generate_memoryVectors(models_dict, trialInfo, X_, Y0_, tRange, nBoots=nBoots, nPerms=nPerms, pca_tWins=((300,1300),(1600,2600),),adaptPCA=pcas_C, adaptEVR=evrs_C, 
    #                                                                                                                                                                                                dt=dt, toPlot = False, label=f'{strategyLabel}',plot3d=False,
    #                                                                                                                                                                                                hideLocs = (), savefig=False, save_path = f'{phd_path}/outputs/rnns/', normalizeMinMax=(-1,1), separatePlot=False) #
    EVRs_C1[kfi] = evrs_C
    EVRs_C2[kfi] = evrs2nd_C
    #EVRs_M1[kfi] = evrs1st_M
    #EVRs_M2[kfi] = evrs2nd_M
    
    vecs, projs, projsAll, trialInfos, _, _, vecs_shuff, projs_shuff, projsAll_shuff, trialInfos_shuff, _ = f_evaluateRNN.generate_itemVectors(models_dict, trialInfo, X_, Y0_, tRange, checkpoints, avgInterval, adaptPCA=pcas_C, adaptEVR=evrs_C, 
                                                                                                                                               nBoots=nBoots, nPerms=nPerms, pca_tWins=((300,1300),(1600,2600),), dt=dt,) #
    
    vecs_bsl_train, projs_bsl_train, projsAll_bsl_train, trialInfos_bsl_train, _, _, vecs_bsl_test, projs_bsl_test, projsAll_bsl_test, trialInfos_bsl_test, _, _ = f_evaluateRNN.generate_bslVectors(models_dict, trialInfo, X_, Y0_, tRange, checkpoints, avgInterval, adaptPCA=pcas_C, adaptEVR=evrs_C, 
                                                                                     nBoots=nBoots, nPerms=nPerms, pca_tWins=((300,1300),(1600,2600),), dt=dt, fracBoots=0.5) #
    # choice items    
    cosThetas_choice[kfi], cosPsis_choice[kfi], cosThetas_nonchoice[kfi], cosPsis_nonchoice[kfi] = [],[],[],[]
    cosThetas_choice_shuff[kfi], cosThetas_nonchoice_shuff[kfi] = [],[]
    cosPsis_choice_shuff[kfi], cosPsis_nonchoice_shuff[kfi] = [],[]
    
    # item v item
    cosThetas_11[kfi], cosThetas_22[kfi], cosThetas_12[kfi] = [],[],[]
    cosPsis_11[kfi], cosPsis_22[kfi], cosPsis_12[kfi] = [],[],[]
    
    cosThetas_11_shuff[kfi], cosThetas_22_shuff[kfi], cosThetas_12_shuff[kfi] = [],[],[]
    cosPsis_11_shuff[kfi], cosPsis_22_shuff[kfi], cosPsis_12_shuff[kfi] = [],[],[]
    
    cosThetas_11_bsl[kfi], cosThetas_22_bsl[kfi], cosThetas_12_bsl[kfi] = [],[],[]
    cosPsis_11_bsl[kfi], cosPsis_22_bsl[kfi], cosPsis_12_bsl[kfi] = [],[],[]
    
    # angle alignment items v readout
    cosThetas_1C[kfi], cosThetas_2C[kfi] = [],[]
    cosPsis_1C[kfi], cosPsis_2C[kfi] = [], []
    
    cosThetas_1C_shuff[kfi], cosThetas_2C_shuff[kfi] = [],[]
    cosPsis_1C_shuff[kfi], cosPsis_2C_shuff[kfi] = [], []
    
    # code transferability item 12
    performanceX_12[kfi], performanceX_21[kfi] = [],[]
    performanceX_12_shuff[kfi], performanceX_21_shuff[kfi] = [],[]
    
    # code transferability choice nonchoice
    performanceX_rdc[kfi], performanceX_drc[kfi] = [],[]
    performanceX_rdnc[kfi], performanceX_drnc[kfi] = [],[]
    
    performanceX_rdc_shuff[kfi], performanceX_drc_shuff[kfi] = [],[]
    performanceX_rdnc_shuff[kfi], performanceX_drnc_shuff[kfi] = [],[]
    
    # item info by item subspace
    info3ds_1[kfi], info3ds_2[kfi] = [],[]
    info3ds_1_shuff[kfi], info3ds_2_shuff[kfi] = [],[]
    
    #infos_C1[kfi], infos_C2[kfi] = [],[]
    #infos_C1_shuff[kfi], infos_C2_shuff[kfi] = [],[]
    
    #infos_C1X[kfi], infos_C2X[kfi] = [],[]
    #infos_C1X_shuff[kfi], infos_C2X_shuff[kfi] = [],[]
    
    for i in range(len(modelDicts[kfi])):
        print(i)
        geoms_valid, geoms_shuff = (vecs[i], projs[i], projsAll[i], trialInfos[i]), (vecs_shuff[i], projs_shuff[i], projsAll_shuff[i], trialInfos_shuff[i])
        geomsC_valid, geomsC_shuff = (vecs_C[i], projs_C[i], projsAll_C[i], trialInfos_C[i]), (vecs_C_shuff[i], projs_C_shuff[i], projsAll_C_shuff[i], trialInfos_C_shuff[i])
        
        geoms_bsl_train, geoms_bsl_test = (vecs_bsl_train[i], projs_bsl_train[i], projsAll_bsl_train[i], trialInfos_bsl_train[i]), (vecs_bsl_test[i], projs_bsl_test[i], projsAll_bsl_test[i], trialInfos_bsl_test[i])
        
        cosThetas, cosThetas_shuff, cosPsis, cosPsis_shuff = f_evaluateRNN.get_angleAlignment_itemPairs(geoms_valid, geoms_shuff, checkpoints)
        cosThetas_C, cosThetas_C_shuff, cosPsis_C, cosPsis_C_shuff = f_evaluateRNN.get_angleAlignment_choicePairs(geoms_valid, geoms_shuff, checkpoints)
        
        cosThetas_bsl, cosPsis_bsl = f_evaluateRNN.get_angleAlignment_itemPairs_bsl(geoms_bsl_train, geoms_bsl_test, checkpoints)
        
        cosThetas_ivr, cosThetas_ivr_shuff, cosPsis_ivr, cosPsis_ivr_shuff = f_evaluateRNN.get_angleAlignment_itemRead(geoms_valid, geoms_shuff, geomsC_valid, geomsC_shuff, checkpoints)
        
        pfmTrans, pfmTrans_shuff = f_evaluateRNN.itemInfo_by_plane_Trans(geoms_valid, checkpoints, nPerms=nPerms, shuff_excludeInv = False)
        pfmTransc, pfmTransc_shuff, pfmTransnc, pfmTransnc_shuff = f_evaluateRNN.chioceInfo_by_plane_Trans(geoms_valid, checkpoints, nPerms=nPerms, shuff_excludeInv = False)
        
        
        info3d, info3d_shuff = f_evaluateRNN.itemInfo_by_plane(geoms_valid, checkpoints, nPerms = nPerms, infoMethod=infoMethod, shuff_excludeInv = False)
        #info_CIX, info_CIX_shuff = f_evaluateRNN.itemInfo_by_planeCX(geomsC_valid, nPerms = nPerms, bins = bins, tslice = tslice, shuff_excludeInv = False)
        
        # choice item
        cosThetas_choice[kfi] += [cosThetas_C[0]]
        cosPsis_choice[kfi] += [cosPsis_C[0]]
        
        cosThetas_nonchoice[kfi] += [cosThetas_C[1]]
        cosPsis_nonchoice[kfi] += [cosPsis_C[1]]

        cosThetas_choice_shuff[kfi] += [cosThetas_C_shuff[0]]
        cosPsis_choice_shuff[kfi] += [cosPsis_C_shuff[0]]
        
        cosThetas_nonchoice_shuff[kfi] += [cosThetas_C_shuff[1]]
        cosPsis_nonchoice_shuff[kfi] += [cosPsis_C_shuff[1]]

        # item v item
        cosThetas_11[kfi] += [cosThetas[0]]
        cosPsis_11[kfi] += [cosPsis[0]]
        
        cosThetas_12[kfi] += [cosThetas[1]]
        cosPsis_12[kfi] += [cosPsis[1]]
        
        cosThetas_22[kfi] += [cosThetas[2]]
        cosPsis_22[kfi] += [cosPsis[2]]
        
        cosThetas_11_shuff[kfi] += [cosThetas_shuff[0]]
        cosPsis_11_shuff[kfi] += [cosPsis_shuff[0]]
        
        cosThetas_12_shuff[kfi] += [cosThetas_shuff[1]]
        cosPsis_12_shuff[kfi] += [cosPsis_shuff[1]]
        
        cosThetas_22_shuff[kfi] += [cosThetas_shuff[2]]
        cosPsis_22_shuff[kfi] += [cosPsis_shuff[2]]
        
        cosThetas_11_bsl[kfi] += [cosThetas_bsl[0]]
        cosPsis_11_bsl[kfi] += [cosPsis_bsl[0]]
        
        cosThetas_12_bsl[kfi] += [cosThetas_bsl[1]]
        cosPsis_12_bsl[kfi] += [cosPsis_bsl[1]]
        
        cosThetas_22_bsl[kfi] += [cosThetas_bsl[2]]
        cosPsis_22_bsl[kfi] += [cosPsis_bsl[2]]
        
        
        # item v readout
        cosThetas_1C[kfi] += [cosThetas_ivr[0]]
        cosPsis_1C[kfi] += [cosPsis_ivr[0]]
        
        cosThetas_2C[kfi] += [cosThetas_ivr[1]]
        cosPsis_2C[kfi] += [cosPsis_ivr[1]]
        
        cosThetas_1C_shuff[kfi] += [cosThetas_ivr_shuff[0]]
        cosPsis_1C_shuff[kfi] += [cosPsis_ivr_shuff[0]]
        
        cosThetas_2C_shuff[kfi] += [cosThetas_ivr_shuff[1]]
        cosPsis_2C_shuff[kfi] += [cosPsis_ivr_shuff[1]]
        
        # code transferability 
        performanceX_12[kfi] += [pfmTrans[0]]
        performanceX_21[kfi] += [pfmTrans[1]]
        
        performanceX_12_shuff[kfi] += [pfmTrans_shuff[0]]
        performanceX_21_shuff[kfi] += [pfmTrans_shuff[1]]
        
        # code transferability choice nonchoice
        performanceX_rdc[kfi] += [pfmTransc[0]]
        performanceX_drc[kfi] += [pfmTransc[1]]
        
        performanceX_rdnc[kfi] += [pfmTransnc[0]]
        performanceX_drnc[kfi] += [pfmTransnc[1]]
        
        performanceX_rdc_shuff[kfi] += [pfmTransc_shuff[0]]
        performanceX_drc_shuff[kfi] += [pfmTransc_shuff[1]]
        
        performanceX_rdnc_shuff[kfi] += [pfmTransnc_shuff[0]]
        performanceX_drnc_shuff[kfi] += [pfmTransnc_shuff[1]]

        # item info by item subspace
        info3ds_1[kfi] += [info3d[0]]
        info3ds_2[kfi] += [info3d[1]]
        
        info3ds_1_shuff[kfi] += [info3d_shuff[0]]
        info3ds_2_shuff[kfi] += [info3d_shuff[1]]
        
        #infos_C1[kfi] += [infoC[0]]
        #infos_C2[kfi] += [infoC[1]]
        
        #infos_C1_shuff[kfi] += [infoC_shuff[0]]
        #infos_C2_shuff[kfi] += [infoC_shuff[1]]
        
        #infos_C1X[kfi] += [infoCX[0]]
        #infos_C2X[kfi] += [infoCX[1]]
        
        #infos_C1X_shuff[kfi] += [infoCX_shuff[0]]
        #infos_C2X_shuff[kfi] += [infoCX_shuff[1]]
    
# In[] store results

np.save(f'{phd_path}/outputs/rnns/' + 'cosTheta_choice_rnn.npy', cosThetas_choice, allow_pickle=True)
np.save(f'{phd_path}/outputs/rnns/' + 'cosPsi_choice_rnn.npy', cosPsis_choice, allow_pickle=True)
np.save(f'{phd_path}/outputs/rnns/' + 'cosTheta_nonchoice_rnn.npy', cosThetas_nonchoice, allow_pickle=True)
np.save(f'{phd_path}/outputs/rnns/' + 'cosPsi_nonchoice_rnn.npy', cosPsis_nonchoice, allow_pickle=True)

np.save(f'{phd_path}/outputs/rnns/' + 'cosTheta_choice_shuff_rnn.npy', cosThetas_choice_shuff, allow_pickle=True)
np.save(f'{phd_path}/outputs/rnns/' + 'cosPsi_choice_shuff_rnn.npy', cosPsis_choice_shuff, allow_pickle=True)
np.save(f'{phd_path}/outputs/rnns/' + 'cosTheta_nonchoice_shuff_rnn.npy', cosThetas_nonchoice_shuff, allow_pickle=True)
np.save(f'{phd_path}/outputs/rnns/' + 'cosPsi_nonchoice_shuff_rnn.npy', cosPsis_nonchoice_shuff, allow_pickle=True)

np.save(f'{phd_path}/outputs/rnns/' + 'cosTheta_11_rnn.npy', cosThetas_11, allow_pickle=True)
np.save(f'{phd_path}/outputs/rnns/' + 'cosTheta_22_rnn.npy', cosThetas_22, allow_pickle=True)
np.save(f'{phd_path}/outputs/rnns/' + 'cosTheta_12_rnn.npy', cosThetas_12, allow_pickle=True)
np.save(f'{phd_path}/outputs/rnns/' + 'cosPsi_11_rnn.npy', cosPsis_11, allow_pickle=True)
np.save(f'{phd_path}/outputs/rnns/' + 'cosPsi_22_rnn.npy', cosPsis_22, allow_pickle=True)
np.save(f'{phd_path}/outputs/rnns/' + 'cosPsi_12_rnn.npy', cosPsis_12, allow_pickle=True)

np.save(f'{phd_path}/outputs/rnns/' + 'cosTheta_11_shuff_rnn.npy', cosThetas_11_shuff, allow_pickle=True)
np.save(f'{phd_path}/outputs/rnns/' + 'cosTheta_22_shuff_rnn.npy', cosThetas_22_shuff, allow_pickle=True)
np.save(f'{phd_path}/outputs/rnns/' + 'cosTheta_12_shuff_rnn.npy', cosThetas_12_shuff, allow_pickle=True)
np.save(f'{phd_path}/outputs/rnns/' + 'cosPsi_11_shuff_rnn.npy', cosPsis_11_shuff, allow_pickle=True)
np.save(f'{phd_path}/outputs/rnns/' + 'cosPsi_22_shuff_rnn.npy', cosPsis_22_shuff, allow_pickle=True)
np.save(f'{phd_path}/outputs/rnns/' + 'cosPsi_12_shuff_rnn.npy', cosPsis_12_shuff, allow_pickle=True)

np.save(f'{phd_path}/outputs/rnns/' + 'cosTheta_11_bsl_rnn.npy', cosThetas_11_bsl, allow_pickle=True)
np.save(f'{phd_path}/outputs/rnns/' + 'cosTheta_22_bsl_rnn.npy', cosThetas_22_bsl, allow_pickle=True)
np.save(f'{phd_path}/outputs/rnns/' + 'cosTheta_12_bsl_rnn.npy', cosThetas_12_bsl, allow_pickle=True)
np.save(f'{phd_path}/outputs/rnns/' + 'cosPsi_11_bsl_rnn.npy', cosPsis_11_bsl, allow_pickle=True)
np.save(f'{phd_path}/outputs/rnns/' + 'cosPsi_22_bsl_rnn.npy', cosPsis_22_bsl, allow_pickle=True)
np.save(f'{phd_path}/outputs/rnns/' + 'cosPsi_12_bsl_rnn.npy', cosPsis_12_bsl, allow_pickle=True)

np.save(f'{phd_path}/outputs/rnns/' + 'cosTheta_1C_rnn.npy', cosThetas_1C, allow_pickle=True)
np.save(f'{phd_path}/outputs/rnns/' + 'cosTheta_2C_rnn.npy', cosThetas_2C, allow_pickle=True)
np.save(f'{phd_path}/outputs/rnns/' + 'cosPsi_1C_rnn.npy', cosPsis_1C, allow_pickle=True)
np.save(f'{phd_path}/outputs/rnns/' + 'cosPsi_2C_rnn.npy', cosPsis_2C, allow_pickle=True)

np.save(f'{phd_path}/outputs/rnns/' + 'cosTheta_1C_shuff_rnn.npy', cosThetas_1C_shuff, allow_pickle=True)
np.save(f'{phd_path}/outputs/rnns/' + 'cosTheta_2C_shuff_rnn.npy', cosThetas_2C_shuff, allow_pickle=True)
np.save(f'{phd_path}/outputs/rnns/' + 'cosPsi_1C_shuff_rnn.npy', cosPsis_1C_shuff, allow_pickle=True)
np.save(f'{phd_path}/outputs/rnns/' + 'cosPsi_2C_shuff_rnn.npy', cosPsis_2C_shuff, allow_pickle=True)
#%%
np.save(f'{phd_path}/outputs/rnns/' + 'performanceX_12_rnn.npy', performanceX_12, allow_pickle=True)
np.save(f'{phd_path}/outputs/rnns/' + 'performanceX_12_shuff_rnn.npy', performanceX_12_shuff, allow_pickle=True)
np.save(f'{phd_path}/outputs/rnns/' + 'performanceX_21_rnn.npy', performanceX_21, allow_pickle=True)
np.save(f'{phd_path}/outputs/rnns/' + 'performanceX_21_shuff_rnn.npy', performanceX_21_shuff, allow_pickle=True)

np.save(f'{phd_path}/outputs/rnns/' + 'performanceX_rdc_rnn.npy', performanceX_rdc, allow_pickle=True)
np.save(f'{phd_path}/outputs/rnns/' + 'performanceX_rdc_shuff_rnn.npy', performanceX_rdc_shuff, allow_pickle=True)
np.save(f'{phd_path}/outputs/rnns/' + 'performanceX_drc_rnn.npy', performanceX_drc, allow_pickle=True)
np.save(f'{phd_path}/outputs/rnns/' + 'performanceX_drc_shuff_rnn.npy', performanceX_drc_shuff, allow_pickle=True)

np.save(f'{phd_path}/outputs/rnns/' + 'performanceX_rdnc_rnn.npy', performanceX_rdnc, allow_pickle=True)
np.save(f'{phd_path}/outputs/rnns/' + 'performanceX_rdnc_shuff_rnn.npy', performanceX_rdnc_shuff, allow_pickle=True)
np.save(f'{phd_path}/outputs/rnns/' + 'performanceX_drnc_rnn.npy', performanceX_drnc, allow_pickle=True)
np.save(f'{phd_path}/outputs/rnns/' + 'performanceX_drnc_shuff_rnn.npy', performanceX_drnc_shuff, allow_pickle=True)

np.save(f'{phd_path}/outputs/rnns/' + 'performance1_item_rnn.npy', info3ds_1, allow_pickle=True)
np.save(f'{phd_path}/outputs/rnns/' + 'performance2_item_rnn.npy', info3ds_2, allow_pickle=True)
#np.save(f'{phd_path}/outputs/rnns/' + 'performanceX1_readout_rnn.npy', infos_C1X, allow_pickle=True)
#np.save(f'{phd_path}/outputs/rnns/' + 'performanceX2_readout_rnn.npy', infos_C2X, allow_pickle=True)

np.save(f'{phd_path}/outputs/rnns/' + 'performance1_item_shuff_rnn.npy', info3ds_1_shuff, allow_pickle=True)
np.save(f'{phd_path}/outputs/rnns/' + 'performance2_item_shuff_rnn.npy', info3ds_2_shuff, allow_pickle=True)
#np.save(f'{phd_path}/outputs/rnns/' + 'performanceX1_readout_shuff_rnn.npy', infos_C1X_shuff, allow_pickle=True)
#np.save(f'{phd_path}/outputs/rnns/' + 'performanceX2_readout_shuff_rnn.npy', infos_C2X_shuff, allow_pickle=True)

#%% load
performanceX_12 = np.load(f'{phd_path}/outputs/rnns/' + 'performanceX_12_rnn.npy', allow_pickle=True).item()
performanceX_12_shuff = np.load(f'{phd_path}/outputs/rnns/' + 'performanceX_12_shuff_rnn.npy', allow_pickle=True).item()
performanceX_21 = np.load(f'{phd_path}/outputs/rnns/' + 'performanceX_21_rnn.npy', allow_pickle=True).item()
performanceX_21_shuff = np.load(f'{phd_path}/outputs/rnns/' + 'performanceX_21_shuff_rnn.npy', allow_pickle=True).item()

performanceX_rdc = np.load(f'{phd_path}/outputs/rnns/' + 'performanceX_rdc_rnn.npy', allow_pickle=True).item()
performanceX_rdc_shuff = np.load(f'{phd_path}/outputs/rnns/' + 'performanceX_rdc_shuff_rnn.npy', allow_pickle=True).item()
performanceX_drc = np.load(f'{phd_path}/outputs/rnns/' + 'performanceX_drc_rnn.npy', allow_pickle=True).item()
performanceX_drc_shuff = np.load(f'{phd_path}/outputs/rnns/' + 'performanceX_drc_shuff_rnn.npy', allow_pickle=True).item()

performanceX_rdnc = np.load(f'{phd_path}/outputs/rnns/' + 'performanceX_rdnc_rnn.npy', allow_pickle=True).item()
performanceX_rdnc_shuff = np.load(f'{phd_path}/outputs/rnns/' + 'performanceX_rdnc_shuff_rnn.npy', allow_pickle=True).item()
performanceX_drnc = np.load(f'{phd_path}/outputs/rnns/' + 'performanceX_drnc_rnn.npy', allow_pickle=True).item()
performanceX_drnc_shuff = np.load(f'{phd_path}/outputs/rnns/' + 'performanceX_drnc_shuff_rnn.npy', allow_pickle=True).item()

#%% plot item v item
for nfi, kfi in enumerate(fitIntervals):
    label = f'fit interval:{kfi}'
    strategyLabel = 'Rehearse & Update' if kfi[-2:] == '12' else 'Retrieve at Recall'
    strategyLabel_ = 'R&U' if kfi[-2:] == '12' else 'R@R'
    
    angleCheckPoints = np.linspace(0,np.pi,13).round(5)
    cmap = plt.get_cmap('coolwarm')
    norm = mcolors.BoundaryNorm(np.flip(np.cos(angleCheckPoints).round(5)),cmap.N, clip=True)
    
    cosTheta_11T = {tt:np.array([cosThetas_11[kfi][n][tt] for n in range(100)]).mean(1) for tt in ttypes}
    cosTheta_22T = {tt:np.array([cosThetas_22[kfi][n][tt] for n in range(100)]).mean(1) for tt in ttypes}
    cosTheta_12T = {tt:np.array([cosThetas_12[kfi][n][tt] for n in range(100)]).mean(1) for tt in ttypes}

    cosPsi_11T = {tt:np.array([cosPsis_11[kfi][n][tt] for n in range(100)]).mean(1) for tt in ttypes}
    cosPsi_22T = {tt:np.array([cosPsis_22[kfi][n][tt] for n in range(100)]).mean(1) for tt in ttypes}
    cosPsi_12T = {tt:np.array([cosPsis_12[kfi][n][tt] for n in range(100)]).mean(1) for tt in ttypes}
    
    ### cosTheta
    for tt in ttypes:
        ttype = 'Retarget' if tt==1 else 'Distraction'    
        
        plt.figure(figsize=(16, 6), dpi=100)
        plt.subplot(1,3,1)
        ax = plt.gca()
        im = ax.imshow(cosTheta_11T[tt].mean(0), cmap=cmap, norm=norm, aspect='auto')
#        for i in range(len(checkpoints)):
#            for j in range(len(checkpoints)):
#                if 0.05 < pcosTheta_11T[i,j] <= 0.1:
#                    text = ax.text(j, i, '+', ha="center", va="center", fontsize=6)
#                elif 0.01 < pcosTheta_11T[i,j] <= 0.05:
#                    text = ax.text(j, i, '*', ha="center", va="center", fontsize=6)
#                elif pcosTheta_11T[i,j] <= 0.01:
#                    text = ax.text(j, i, '**', ha="center", va="center", fontsize=6)
                    
        ax.set_title('Item1-Item1', fontsize = 15, pad = 15)
        ax.set_xticks([n for n in range(len(checkpoints))])
        ax.set_xticklabels(checkpointsLabels, fontsize = 10)
        ax.set_yticks([n for n in range(len(checkpoints))])
        ax.set_yticklabels(checkpointsLabels, fontsize = 10, rotation=90)
        ax.set_xlabel('Item 1', fontsize = 15)
        ax.set_ylabel('Item 1', fontsize = 15)
        ax.set_frame_on(False)
        
        
        plt.subplot(1,3,2)
        ax = plt.gca()
        im = ax.imshow(cosTheta_22T[tt].mean(0), cmap=cmap, norm=norm, aspect='auto')
#        for i in range(len(checkpoints)):
#            for j in range(len(checkpoints)):
#                if 0.05 < pcosTheta_22T[i,j] <= 0.1:
#                    text = ax.text(j, i, '+', ha="center", va="center", fontsize=6)
#                elif 0.01 < pcosTheta_22T[i,j] <= 0.05:
#                    text = ax.text(j, i, '*', ha="center", va="center", fontsize=6)
#                elif pcosTheta_22T[i,j] <= 0.01:
#                    text = ax.text(j, i, '**', ha="center", va="center", fontsize=6)
                    
        ax.set_title('Item2-Item2', fontsize = 15, pad = 15)
        ax.set_xticks([n for n in range(len(checkpoints))])
        ax.set_xticklabels(checkpointsLabels, fontsize = 10)
        ax.set_yticks([n for n in range(len(checkpoints))])
        ax.set_yticklabels(checkpointsLabels, fontsize = 10, rotation=90)
        ax.set_xlabel('Item 2', fontsize = 15)
        ax.set_ylabel('Item 2', fontsize = 15)
        ax.set_frame_on(False)
        
        
        
        plt.subplot(1,3,3)
        ax = plt.gca()
        im = ax.imshow(cosTheta_12T[tt].mean(0), cmap=cmap, norm=norm, aspect='auto')
#        for i in range(len(checkpoints)):
#            for j in range(len(checkpoints)):
#                if 0.05 < pcosTheta_12T[i,j] <= 0.1:
#                    text = ax.text(j, i, '+', ha="center", va="center", fontsize=6)
#                elif 0.01 < pcosTheta_12T[i,j] <= 0.05:
#                    text = ax.text(j, i, '*', ha="center", va="center", fontsize=6)
#                elif pcosTheta_12T[i,j] <= 0.01:
#                    text = ax.text(j, i, '**', ha="center", va="center", fontsize=6)
        ax.set_title('Item1-Item2', fontsize = 15, pad = 15)
        ax.set_xticks([n for n in range(len(checkpoints))])
        ax.set_xticklabels(checkpointsLabels, fontsize = 10)
        ax.set_yticks([n for n in range(len(checkpoints))])
        ax.set_yticklabels(checkpointsLabels, fontsize = 10, rotation=90)
        ax.set_xlabel('Item 2', fontsize = 15)
        ax.set_ylabel('Item 1', fontsize = 15)
        ax.set_frame_on(False)
        
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_ticklabels(np.flip(np.degrees(angleCheckPoints).round(1).astype(int)))
        cbar.set_label('cos(θ)', fontsize = 15, rotation = 270, labelpad=20)
        
        plt.subplots_adjust(top = 0.8)
        plt.suptitle(f'Principal Angle (θ), {strategyLabel_}, {ttype}', fontsize = 20, y=1)
        plt.tight_layout()
        plt.show()
        
        
        
        
        ### cosPsi
        
        plt.figure(figsize=(16, 6), dpi=100)
        plt.subplot(1,3,1)
        ax = plt.gca()
        im = ax.imshow(cosPsi_11T[tt].mean(0), cmap=cmap, norm=norm, aspect='auto')
#        for i in range(len(checkpoints)):
#            for j in range(len(checkpoints)):
#                if 0.05 < pcosPsi_11T[i,j] <= 0.1:
#                    text = ax.text(j, i, '+', ha="center", va="center", fontsize=6)
#                elif 0.01 < pcosPsi_11T[i,j] <= 0.05:
#                    text = ax.text(j, i, '*', ha="center", va="center", fontsize=6)
#                elif pcosPsi_11T[i,j] <= 0.01:
#                    text = ax.text(j, i, '**', ha="center", va="center", fontsize=6)
                    
        ax.set_title('Item1-Item1', fontsize = 15, pad = 15)
        ax.set_xticks([n for n in range(len(checkpoints))])
        ax.set_xticklabels(checkpointsLabels, fontsize = 10)
        ax.set_yticks([n for n in range(len(checkpoints))])
        ax.set_yticklabels(checkpointsLabels, fontsize = 10, rotation=90)
        ax.set_xlabel('Item 1', fontsize = 15)
        ax.set_ylabel('Item 1', fontsize = 15)
        ax.set_frame_on(False)
        
        
        plt.subplot(1,3,2)
        ax = plt.gca()
        
        im = ax.imshow(cosPsi_22T[tt].mean(0), cmap=cmap, norm=norm, aspect='auto')
#        for i in range(len(checkpoints)):
#            for j in range(len(checkpoints)):
#                if 0.05 < pcosPsi_22T[i,j] <= 0.1:
#                    text = ax.text(j, i, '+', ha="center", va="center", fontsize=6)
#                elif 0.01 < pcosPsi_22T[i,j] <= 0.05:
#                    text = ax.text(j, i, '*', ha="center", va="center", fontsize=6)
#                elif pcosPsi_22T[i,j] <= 0.01:
#                    text = ax.text(j, i, '**', ha="center", va="center", fontsize=6)
                    
        ax.set_title('Item2-Item2', fontsize = 15, pad = 15)
        ax.set_xticks([n for n in range(len(checkpoints))])
        ax.set_xticklabels(checkpointsLabels, fontsize = 10)
        ax.set_yticks([n for n in range(len(checkpoints))])
        ax.set_yticklabels(checkpointsLabels, fontsize = 10, rotation=90)
        ax.set_xlabel('Item 2', fontsize = 15)
        ax.set_ylabel('Item 2', fontsize = 15)
        ax.set_frame_on(False)
        
        
        plt.subplot(1,3,3)
        ax = plt.gca()
        im = ax.imshow(cosPsi_12T[tt].mean(0), cmap=cmap, norm=norm, aspect='auto')
#        for i in range(len(checkpoints)):
#            for j in range(len(checkpoints)):
#                if 0.05 < pcosPsi_12T[i,j] <= 0.1:
#                    text = ax.text(j, i, '+', ha="center", va="center", fontsize=6)
#                elif 0.01 < pcosPsi_12T[i,j] <= 0.05:
#                    text = ax.text(j, i, '*', ha="center", va="center", fontsize=6)
#                elif pcosPsi_12T[i,j] <= 0.01:
#                    text = ax.text(j, i, '**', ha="center", va="center", fontsize=6)
        ax.set_title('Item1-Item2', fontsize = 15, pad = 15)
        ax.set_xticks([n for n in range(len(checkpoints))])
        ax.set_xticklabels(checkpointsLabels, fontsize = 10)
        ax.set_yticks([n for n in range(len(checkpoints))])
        ax.set_yticklabels(checkpointsLabels, fontsize = 10, rotation=90)
        ax.set_xlabel('Item 2', fontsize = 15)
        ax.set_ylabel('Item 1', fontsize = 15)
        ax.set_frame_on(False)
        
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_ticklabels(np.flip(np.degrees(angleCheckPoints).round(1).astype(int)))
        cbar.set_label('cos(Ψ)', fontsize = 15, rotation = 270, labelpad=20)
        
        plt.subplots_adjust(top = 0.8)
        plt.suptitle(f'Representational Alignment (Ψ), {strategyLabel_}, {ttype}', fontsize = 20, y=1)
        #plt.suptitle(f'abs(cosPsi), {region}, ttype={tt}', fontsize = 15, y=1)
        plt.tight_layout()
        plt.show()

#%% plot choice item v choice item
for nfi, kfi in enumerate(fitIntervals):
    label = f'fit interval:{kfi}'
    strategyLabel = 'Rehearse & Update' if kfi[-2:] == '12' else 'Retrieve at Recall'
    strategyLabel_ = 'R&U' if kfi[-2:] == '12' else 'R@R'
    
    angleCheckPoints = np.linspace(0,np.pi,7).round(5)
    
    cosThetas_choiceT = np.array([cosThetas_choice[kfi][n] for n in range(100)]).mean(1)
    cosThetas_nonchoiceT = np.array([cosThetas_nonchoice[kfi][n] for n in range(100)]).mean(1)
    cosPsis_choiceT = np.array([cosPsis_choice[kfi][n] for n in range(100)]).mean(1)
    cosPsis_nonchoiceT = np.array([cosPsis_nonchoice[kfi][n] for n in range(100)]).mean(1)
    
    ############################################
    ### cosTheta
    plt.figure(figsize=(10, 3), dpi=300)
    plt.subplot(1,2,1)
    ax = plt.gca()
    ax.errorbar(np.arange(0, len(checkpoints), 1), cosThetas_choiceT.mean(0), yerr = cosThetas_choiceT.std(0), marker = 'o', color = 'c', label = 'Choice', capsize=4)
    #ax.plot(np.arange(0, len(checkpoints), 1), pcosTheta_choice, alpha = 0.3, linestyle = '-')
    ax.errorbar(np.arange(0, len(checkpoints), 1), cosThetas_nonchoiceT.mean(0), yerr = cosThetas_nonchoiceT.std(0), marker = 'o', color = 'y', label = 'Nonchoice', capsize=4)
    #ax.plot(np.arange(0, len(checkpoints), 1), pcosTheta_nonchoice, alpha = 0.3, linestyle = '-')
    
            
    #ax.set_title('Choice', pad = 10)
    ax.set_xticks([n for n in range(len(checkpoints))])
    ax.set_xticklabels(checkpointsLabels, fontsize = 10)
    ax.set_ylim((-1,1))
    ax.set_yticks(np.flip(np.cos(angleCheckPoints).round(5)), labels = np.flip(np.degrees(angleCheckPoints).round(1).astype(int)))
    ax.set_ylabel('cos(θ)',fontsize=15,rotation = 90)
    #ax.legend(loc='lower right')
   
    ### cosPsi
    plt.subplot(1,2,2)
    ax = plt.gca()
    ax.errorbar(np.arange(0, len(checkpoints), 1), cosPsis_choiceT.mean(0), yerr = cosPsis_choiceT.std(0), marker = 'o', color = 'c', label = 'Choice', capsize=4)
    #ax.plot(np.arange(0, len(checkpoints), 1), pcosPsi_choice, alpha = 0.3, linestyle = '-')
    ax.errorbar(np.arange(0, len(checkpoints), 1), cosPsis_nonchoiceT.mean(0), yerr = cosPsis_nonchoiceT.std(0), marker = 'o', color = 'y', label = 'Nonchoice', capsize=4)
    #ax.plot(np.arange(0, len(checkpoints), 1), pcosPsi_nonchoice, alpha = 0.3, linestyle = '-')
    
    
    #ax.set_title('Choice', pad = 10)
    ax.set_xticks([n for n in range(len(checkpoints))])
    ax.set_xticklabels(checkpointsLabels, fontsize = 10)
    ax.set_ylim((-1,1))
    ax.set_yticks(np.flip(np.cos(angleCheckPoints).round(5)), labels = np.flip(np.degrees(angleCheckPoints).round(1).astype(int)))
    ax.set_ylabel('cos(Ψ)',fontsize=15,rotation = 90)
    ax.legend(bbox_to_anchor=(1, 0.5), fontsize = 12)#loc='lower right'
    
    plt.subplots_adjust(top = 0.8)
    plt.suptitle(f'Choice/Nonchoice-Item Subspaces, {strategyLabel_}', fontsize = 15, y=1)
    #plt.suptitle(f'abs(cosPsi), {region}, choice/nonchoice', fontsize = 15, y=1)
    plt.tight_layout()
    plt.show()    
    
#%% plot item v readout
for nfi, kfi in enumerate(fitIntervals):
    label = f'fit interval:{kfi}'
    strategyLabel = 'Rehearse & Update' if kfi[-2:] == '12' else 'Retrieve at Recall'
    strategyLabel_ = 'R&U' if kfi[-2:] == '12' else 'R@R'
    
    angleCheckPoints = np.linspace(0,np.pi,7).round(5)
    color1, color2 = 'b', 'm'
    
    
    cosThetas_1C_T = {tt:np.array([cosThetas_1C[kfi][n][tt] for n in range(100)]).mean(1) for tt in ttypes}
    cosThetas_2C_T = {tt:np.array([cosThetas_2C[kfi][n][tt] for n in range(100)]).mean(1) for tt in ttypes}
    cosPsis_1C_T = {tt:np.array([cosPsis_1C[kfi][n][tt] for n in range(100)]).mean(1) for tt in ttypes}
    cosPsis_2C_T = {tt:np.array([cosPsis_2C[kfi][n][tt] for n in range(100)]).mean(1) for tt in ttypes}
    
    
    ### cosTheta
    fig, axes = plt.subplots(2,2, figsize=(8,6), dpi=300, sharex=True, sharey=True)
    
    for tt in ttypes:
        
        ttype = 'Retarget' if tt == 1 else 'Distraction'
        
        
        ax = axes.flatten()[tt-1]
        
        # Item1
        ax.errorbar(np.arange(0, len(checkpoints), 1), cosThetas_1C_T[tt].mean(0), 
                    #yerr = [np.maximum(0,cosThetas_1C_T[tt].mean(0)-stats.scoreatpercentile(cosThetas_1C_T[tt], 25, axis=0)), 
                    #        np.maximum(0,stats.scoreatpercentile(cosThetas_1C_T[tt], 75, axis=0)-cosThetas_1C_T[tt].mean(0))], 
                    yerr = cosThetas_1C_T[tt].std(0),
                    marker = 'o', color = color1, label = 'Item1', capsize=4, linewidth=2)
        
        
        # Item2
        ax.errorbar(np.arange(0, len(checkpoints), 1), cosThetas_2C_T[tt].mean(0), 
                    #yerr = [np.maximum(0,cosThetas_2C_T[tt].mean(0)-stats.scoreatpercentile(cosThetas_2C_T[tt], 25, axis=0)), 
                    #        np.maximum(0,stats.scoreatpercentile(cosThetas_2C_T[tt], 75, axis=0)-cosThetas_2C_T[tt].mean(0))], 
                    yerr = cosThetas_2C_T[tt].std(0),
                    marker = 'o', color = color2, label = 'Item2', capsize=4, linewidth=2)
        
        
        #trans = ax.get_xaxis_transform()
        #for nc, cp in enumerate(checkpoints):
        #    if 0.05 < pcosTheta_1C[tt][nc] <= 0.1:
        #        ax.annotate('+', xy=(nc, 0.05), xycoords = trans, annotation_clip = False, ha = 'center', va = 'center', color = color1)
        #    elif 0.01 < pcosTheta_1C[tt][nc] <= 0.05:
        #        ax.annotate('*', xy=(nc, 0.05), xycoords = trans, annotation_clip = False, ha = 'center', va = 'center', color = color1)
        #    elif pcosTheta_1C[tt][nc] <= 0.01:
        #        ax.annotate('**', xy=(nc, 0.05), xycoords = trans, annotation_clip = False, ha = 'center', va = 'center', color = color1)
        
        #for nc, cp in enumerate(checkpoints):
        #    if 0.05 < pcosTheta_2C[tt][nc] <= 0.1:
        #        ax.annotate('+', xy=(nc, 0.05), xycoords = trans, annotation_clip = False, ha = 'center', va = 'center', color = color2)
        #    elif 0.01 < pcosTheta_2C[tt][nc] <= 0.05:
        #        ax.annotate('*', xy=(nc, 0.05), xycoords = trans, annotation_clip = False, ha = 'center', va = 'center', color = color2)
        #    elif pcosTheta_2C[tt][nc] <= 0.01:
        #        ax.annotate('**', xy=(nc, 0.05), xycoords = trans, annotation_clip = False, ha = 'center', va = 'center', color = color2)
                
        ax.set_title(f'{ttype}', pad = 10, fontsize=15)
        ax.set_xticks([n for n in range(len(checkpoints))])
        ax.set_xticklabels(checkpointsLabels, fontsize = 10)
        #ax.set_ylim((-1.1,1.1))
        ax.set_yticks(np.flip(np.cos(angleCheckPoints).round(5)), labels = np.flip(np.degrees(angleCheckPoints).round(1).astype(int)))
        ax.tick_params(axis='both', labelsize=12)
        #ax.set_xlabel('Timebin',fontsize=15)#, labelpad = 10
        
        if tt==1:
            ax.set_ylabel('cos(θ)',fontsize=15,rotation = 90)
        if tt==2:
            ax.legend(loc='lower right')#,bbox_to_anchor=(1, 0.6), fontsize=10
        
        
        
        ax = axes.flatten()[tt-1+2]
        
        ax.errorbar(np.arange(0, len(checkpoints), 1), cosPsis_1C_T[tt].mean(0), 
                    #yerr = [np.maximum(0,cosPsis_1C_T[tt].mean(0)-stats.scoreatpercentile(cosPsis_1C_T[tt], 25, axis=0)), 
                    #        np.maximum(0,stats.scoreatpercentile(cosPsis_1C_T[tt], 75, axis=0)-cosPsis_1C_T[tt].mean(0))], 
                    yerr = cosPsis_1C_T[tt].std(0),
                    marker = 'o', color = color1, label = 'Item1', capsize=4, linewidth=2)
        #ax.plot(np.arange(0, len(checkpoints), 1), pcosPsi_1C[tt], alpha = 0.3, linestyle = '-', color = color1)
        
        ax.errorbar(np.arange(0, len(checkpoints), 1), cosPsis_2C_T[tt].mean(0), 
                    #yerr = [np.maximum(0,cosPsis_2C_T[tt].mean(0)-stats.scoreatpercentile(cosPsis_2C_T[tt], 25, axis=0)), 
                    #        np.maximum(0,stats.scoreatpercentile(cosPsis_2C_T[tt], 75, axis=0)-cosPsis_2C_T[tt].mean(0))],
                    yerr = cosPsis_2C_T[tt].std(0),
                    marker = 'o', color = color2, label = 'Item2', capsize=4, linewidth=2)
        #ax.plot(np.arange(0, len(checkpoints), 1), pcosPsi_2C[tt], alpha = 0.3, linestyle = '-', color = color2)
        
        
        #trans = ax.get_xaxis_transform()
        
        #for nc, cp in enumerate(checkpoints):
        #    if 0.05 < pcosPsi_1C[tt][nc] <= 0.1:
        #        ax.annotate('+', xy=(nc, 0.05), xycoords = trans, annotation_clip = False, ha = 'center', va = 'center', color = color1)
        #    elif 0.01 < pcosPsi_1C[tt][nc] <= 0.05:
        #        ax.annotate('*', xy=(nc, 0.05), xycoords = trans, annotation_clip = False, ha = 'center', va = 'center', color = color1)
        #    elif pcosPsi_1C[tt][nc] <= 0.01:
        #        ax.annotate('**', xy=(nc, 0.05), xycoords = trans, annotation_clip = False, ha = 'center', va = 'center', color = color1)
        
        #for nc, cp in enumerate(checkpoints):
        #    if 0.05 < pcosPsi_2C[tt][nc] <= 0.1:
        #        ax.annotate('+', xy=(nc, 0.05), xycoords = trans, annotation_clip = False, ha = 'center', va = 'center', color = color2)
        #    elif 0.01 < pcosPsi_2C[tt][nc] <= 0.05:
        #        ax.annotate('*', xy=(nc, 0.05), xycoords = trans, annotation_clip = False, ha = 'center', va = 'center', color = color2)
        #    elif pcosPsi_2C[tt][nc] <= 0.01:
        #        ax.annotate('**', xy=(nc, 0.05), xycoords = trans, annotation_clip = False, ha = 'center', va = 'center', color = color2)        
        
        #ax.set_title(f'{ttype}', pad = 10, fontsize=15)
        ax.set_xticks([n for n in range(len(checkpoints))])
        ax.set_xticklabels(checkpointsLabels, fontsize = 10)
        ax.set_ylim((-1.1,1.1))
        ax.set_yticks(np.flip(np.cos(angleCheckPoints).round(5)), labels = np.flip(np.degrees(angleCheckPoints).round(1).astype(int)))
        ax.tick_params(axis='both', labelsize=12)
        ax.set_xlabel('Timebin',fontsize=15)#, labelpad = 10
        
        if tt==1:
            ax.set_ylabel('cos(ψ)',fontsize=15,rotation = 90)
        #if tt==2:
        #    ax.legend(bbox_to_anchor=(1, 0.6), fontsize=10)#loc='lower right'
    
    plt.subplots_adjust(top = 0.8)
    plt.suptitle(f'Item vs. Readout Subspaces, {strategyLabel_}', fontsize = 20, y=1)
    plt.tight_layout()
    plt.show()  
    
    fig.savefig(f'{phd_path}/outputs/rnns/item_v_readout_{strategyLabel_}.tif', bbox_inches='tight')
    




#%% plot code transferability between items
for nfi, kfi in enumerate(fitIntervals):
    label = f'fit interval:{kfi}'
    strategyLabel = 'Rehearse & Update' if kfi[-2:] == '12' else 'Retrieve at Recall'
    strategyLabel_ = 'R&U' if kfi[-2:] == '12' else 'R@R'
    
    performanceX_12T = {tt:np.array([performanceX_12[kfi][n][tt] for n in range(100)]).mean(1).mean(1) for tt in ttypes}
    performanceX_21T = {tt:np.array([performanceX_21[kfi][n][tt] for n in range(100)]).mean(1).mean(1) for tt in ttypes}
    performanceX_12T_shuff = {tt:np.array([performanceX_12_shuff[kfi][n][tt] for n in range(100)]).mean(1).mean(1) for tt in ttypes}
    performanceX_21T_shuff = {tt:np.array([performanceX_21_shuff[kfi][n][tt] for n in range(100)]).mean(1).mean(1) for tt in ttypes}

    fig, axes = plt.subplots(2,2, figsize=(12,10), dpi=100, sharex=True, sharey=True)#
    
    for tt in ttypes:
        
        #ax = axes.flatten()[tt-1]
        
        #colorT = 'b' if tt == 1 else 'm'
        condT = 'Retarget' if tt == 1 else 'Distraction'
        
        pPerms_12 = np.zeros((len(checkpoints), len(checkpoints)))
        pPerms_21 = np.zeros((len(checkpoints), len(checkpoints)))
        
        for nc,cp in enumerate(checkpoints):
            for nc_, cp_ in enumerate(checkpoints):
                pPerms_12[nc,nc_] = f_stats.permutation_pCI(performanceX_12T[tt][:,nc,nc_], performanceX_12T_shuff[tt][:,nc,nc_], tail='greater',alpha=5)
                pPerms_21[nc,nc_] = f_stats.permutation_pCI(performanceX_21T[tt][:,nc,nc_], performanceX_21T_shuff[tt][:,nc,nc_], tail='greater',alpha=5)
                
        
        plt.subplot(2,2,(tt-1)*2+1)
        ax = plt.gca()
        im = ax.imshow(performanceX_12T[tt].mean(0), cmap='magma', aspect='auto',vmax=1)
        for i in range(len(checkpoints)):
            for j in range(len(checkpoints)):
                if 0.05 < pPerms_12[i,j] <= 0.1:
                    text = ax.text(j, i, '+', ha="center", va="center", color='cornflowerblue', fontsize=15, weight='bold')
                elif 0.01 < pPerms_12[i,j] <= 0.05:
                    text = ax.text(j, i, '*', ha="center", va="center", color='cornflowerblue', fontsize=15, weight='bold')
                elif pPerms_12[i,j] <= 0.01:
                    text = ax.text(j, i, '**', ha="center", va="center", color='cornflowerblue', fontsize=15, weight='bold')
        
        ax.set_title(f'{condT}, Train I1, Test I2', fontsize = 15, pad = 15)
        ax.set_xticks([n for n in range(len(checkpoints))])
        ax.set_xticklabels(checkpointsLabels, fontsize = 10)
        ax.set_yticks([n for n in range(len(checkpoints))])
        ax.set_yticklabels(checkpointsLabels, fontsize = 10)
        ax.set_xlabel('Test', fontsize = 15)
        ax.set_ylabel('Train', fontsize = 15)
        
        cbar = plt.colorbar(im, ax=ax)
        
        plt.subplot(2,2,(tt-1)*2+2)
        ax = plt.gca()
        im = ax.imshow(performanceX_21T[tt].mean(0), cmap='magma', aspect='auto',vmax=1)
        for i in range(len(checkpoints)):
            for j in range(len(checkpoints)):
                if 0.05 < pPerms_21[i,j] <= 0.1:
                    text = ax.text(j, i, '+', ha="center", va="center", color='cornflowerblue', fontsize=15, weight='bold')
                elif 0.01 < pPerms_21[i,j] <= 0.05:
                    text = ax.text(j, i, '*', ha="center", va="center", color='cornflowerblue', fontsize=15, weight='bold')
                elif pPerms_21[i,j] <= 0.01:
                    text = ax.text(j, i, '**', ha="center", va="center", color='cornflowerblue', fontsize=15, weight='bold')
                    
        ax.set_title(f'{condT}, Train I2, Test I1', fontsize = 15, pad = 15)
        ax.set_xticks([n for n in range(len(checkpoints))])
        ax.set_xticklabels(checkpointsLabels, fontsize = 10)
        ax.set_yticks([n for n in range(len(checkpoints))])
        ax.set_yticklabels(checkpointsLabels, fontsize = 10)
        ax.set_xlabel('Test', fontsize = 15)
        ax.set_ylabel('Train', fontsize = 15)
        
        cbar = plt.colorbar(im, ax=ax)
        
    
    plt.subplots_adjust(top = 0.8)
    plt.suptitle(f'Item Subspace Code Transferability, {strategyLabel_}', fontsize = 25, y=1)
    plt.tight_layout()
    plt.show()  
    
    fig.savefig(f'{phd_path}/outputs/rnns/codeTransferability_{strategyLabel_}_rnn.tif', bbox_inches='tight')
#%% plot code transferability between choices

for nfi, kfi in enumerate(fitIntervals):
    label = f'fit interval:{kfi}'
    strategyLabel = 'Rehearse & Update' if kfi[-2:] == '12' else 'Retrieve at Recall'
    strategyLabel_ = 'R&U' if kfi[-2:] == '12' else 'R@R'
    
    performanceX_rdcT = np.array(performanceX_rdc[kfi]).mean(1).mean(1)
    performanceX_drcT = np.array(performanceX_drc[kfi]).mean(1).mean(1)
    performanceX_rdcT_shuff = np.array(performanceX_rdc_shuff[kfi]).mean(1).mean(1)
    performanceX_drcT_shuff = np.array(performanceX_drc_shuff[kfi]).mean(1).mean(1)
    
    performanceX_rdncT = np.array(performanceX_rdnc[kfi]).mean(1).mean(1)
    performanceX_drncT = np.array(performanceX_drnc[kfi]).mean(1).mean(1)
    performanceX_rdncT_shuff = np.array(performanceX_rdnc_shuff[kfi]).mean(1).mean(1)
    performanceX_drncT_shuff = np.array(performanceX_drnc_shuff[kfi]).mean(1).mean(1)



    fig, axes = plt.subplots(2,2, figsize=(12,10), dpi=100, sharex=True, sharey=True)#
       
    pPerms_rdc = np.zeros((len(checkpoints), len(checkpoints)))
    pPerms_drc = np.zeros((len(checkpoints), len(checkpoints)))
    pPerms_rdnc = np.zeros((len(checkpoints), len(checkpoints)))
    pPerms_drnc = np.zeros((len(checkpoints), len(checkpoints)))
    
    for nc,cp in enumerate(checkpoints):
        for nc_, cp_ in enumerate(checkpoints):
            pPerms_rdc[nc,nc_] = f_stats.permutation_pCI(performanceX_rdcT[:,nc,nc_], performanceX_rdcT_shuff[:,nc,nc_], tail='greater',alpha=5)
            pPerms_drc[nc,nc_] = f_stats.permutation_pCI(performanceX_drcT[:,nc,nc_], performanceX_drcT_shuff[:,nc,nc_], tail='greater',alpha=5)
            pPerms_rdnc[nc,nc_] = f_stats.permutation_pCI(performanceX_rdncT[:,nc,nc_], performanceX_rdncT_shuff[:,nc,nc_], tail='greater',alpha=5)
            pPerms_drnc[nc,nc_] = f_stats.permutation_pCI(performanceX_drncT[:,nc,nc_], performanceX_drncT_shuff[:,nc,nc_], tail='greater',alpha=5)
    
            
    
    plt.subplot(2,2,1)
    ax1 = plt.gca()
    im1 = ax1.imshow(performanceX_rdcT.mean(0), cmap='magma', aspect='auto',vmax=1)
    for i in range(len(checkpoints)):
        for j in range(len(checkpoints)):
            if 0.05 < pPerms_rdc[i,j] <= 0.1:
                text = ax1.text(j, i, '+', ha="center", va="center", color='cornflowerblue', fontsize=15, weight='bold')
            elif 0.01 < pPerms_rdc[i,j] <= 0.05:
                text = ax1.text(j, i, '*', ha="center", va="center", color='cornflowerblue', fontsize=15, weight='bold')
            elif pPerms_rdc[i,j] <= 0.01:
                text = ax1.text(j, i, '**', ha="center", va="center", color='cornflowerblue', fontsize=15, weight='bold')
    
    ax1.set_title(f'Choice Item', fontsize = 15, pad = 15)
    ax1.set_xticks([n for n in range(len(checkpoints))])
    ax1.set_xticklabels(checkpointsLabels, fontsize = 10)
    ax1.set_yticks([n for n in range(len(checkpoints))])
    ax1.set_yticklabels(checkpointsLabels, fontsize = 10)
    ax1.set_xlabel('Test Distraction', fontsize = 15)
    ax1.set_ylabel('Train Retarget', fontsize = 15)
    
    cbar1 = plt.colorbar(im1, ax=ax1)
    
    plt.subplot(2,2,2)
    ax2 = plt.gca()
    im2 = ax2.imshow(performanceX_drcT.mean(0), cmap='magma', aspect='auto',vmax=1)
    for i in range(len(checkpoints)):
        for j in range(len(checkpoints)):
            if 0.05 < pPerms_drc[i,j] <= 0.1:
                text = ax2.text(j, i, '+', ha="center", va="center", color='cornflowerblue', fontsize=15, weight='bold')
            elif 0.01 < pPerms_drc[i,j] <= 0.05:
                text = ax2.text(j, i, '*', ha="center", va="center", color='cornflowerblue', fontsize=15, weight='bold')
            elif pPerms_drc[i,j] <= 0.01:
                text = ax2.text(j, i, '**', ha="center", va="center", color='cornflowerblue', fontsize=15, weight='bold')
                
    ax2.set_title(f'Choice Item', fontsize = 15, pad = 15)
    ax2.set_xticks([n for n in range(len(checkpoints))])
    ax2.set_xticklabels(checkpointsLabels, fontsize = 10)
    ax2.set_yticks([n for n in range(len(checkpoints))])
    ax2.set_yticklabels(checkpointsLabels, fontsize = 10)
    ax2.set_xlabel('Test Retarget', fontsize = 15)
    ax2.set_ylabel('Train Distraction', fontsize = 15)
    
    cbar2 = plt.colorbar(im2, ax=ax2)
    
    ##############
    # non choice #
    ##############
    
    plt.subplot(2,2,3)
    ax3 = plt.gca()
    im3 = ax3.imshow(performanceX_rdncT.mean(0), cmap='magma', aspect='auto',vmax=1)
    for i in range(len(checkpoints)):
        for j in range(len(checkpoints)):
            if 0.05 < pPerms_rdnc[i,j] <= 0.1:
                text = ax3.text(j, i, '+', ha="center", va="center", color='cornflowerblue', fontsize=15, weight='bold')
            elif 0.01 < pPerms_rdnc[i,j] <= 0.05:
                text = ax3.text(j, i, '*', ha="center", va="center", color='cornflowerblue', fontsize=15, weight='bold')
            elif pPerms_rdnc[i,j] <= 0.01:
                text = ax3.text(j, i, '**', ha="center", va="center", color='cornflowerblue', fontsize=15, weight='bold')
    
    ax3.set_title(f'Non-choice Item', fontsize = 15, pad = 15)
    ax3.set_xticks([n for n in range(len(checkpoints))])
    ax3.set_xticklabels(checkpointsLabels, fontsize = 10)
    ax3.set_yticks([n for n in range(len(checkpoints))])
    ax3.set_yticklabels(checkpointsLabels, fontsize = 10)
    ax3.set_xlabel('Test Distraction', fontsize = 15)
    ax3.set_ylabel('Train Retarget', fontsize = 15)
    
    cbar3 = plt.colorbar(im3, ax=ax3)
    
    plt.subplot(2,2,4)
    ax4 = plt.gca()
    im4 = ax4.imshow(performanceX_drncT.mean(0), cmap='magma', aspect='auto',vmax=1)
    for i in range(len(checkpoints)):
        for j in range(len(checkpoints)):
            if 0.05 < pPerms_drnc[i,j] <= 0.1:
                text = ax4.text(j, i, '+', ha="center", va="center", color='cornflowerblue', fontsize=15, weight='bold')
            elif 0.01 < pPerms_drnc[i,j] <= 0.05:
                text = ax4.text(j, i, '*', ha="center", va="center", color='cornflowerblue', fontsize=15, weight='bold')
            elif pPerms_drnc[i,j] <= 0.01:
                text = ax4.text(j, i, '**', ha="center", va="center", color='cornflowerblue', fontsize=15, weight='bold')
                
    ax4.set_title(f'Non-choice Item', fontsize = 15, pad = 15)
    ax4.set_xticks([n for n in range(len(checkpoints))])
    ax4.set_xticklabels(checkpointsLabels, fontsize = 10)
    ax4.set_yticks([n for n in range(len(checkpoints))])
    ax4.set_yticklabels(checkpointsLabels, fontsize = 10)
    ax4.set_xlabel('Test Retarget', fontsize = 15)
    ax4.set_ylabel('Train Distraction', fontsize = 15)
    
    cbar4 = plt.colorbar(im4, ax=ax4)
    
    
    plt.subplots_adjust(top = 0.8)
    plt.suptitle(f'Choice/Nonchoice Item Code Transferability, {strategyLabel_}', fontsize = 25, y=1)
    plt.tight_layout()
    plt.show()  
    
    fig.savefig(f'{phd_path}/outputs/rnns/codeTransferability_cnc_{strategyLabel_}_rnn.tif', bbox_inches='tight')

#%%








































# In[]

################
### Plot EVR ###
################
for nfi, kfi in enumerate(fitIntervals):
    print(kfi)
    
    label = f'fit interval:{kfi}'
    strategyLabel = 'Direct' if kfi[-2:] == '12' else 'Rotation'
    
    evrs[kfi] = np.array(evrs[kfi])

#%% save evrs
np.save(f'{save_path}/evrs.npy', evrs, allow_pickle=True)


plt.figure(figsize=(2, 4), dpi=100)
ax = plt.gca()
ax.boxplot([evrs['ed2'].mean(1).sum(1), evrs['ed12'].mean(1).sum(1)], tick_labels=['Rotation','Direct'])#, showfliers = False
ax.set_xlabel('Strategy', labelpad = 3, fontsize = 12)
ax.set_ylabel('Sum of EVR from top 3 PCs', labelpad = 3, fontsize = 12)
plt.show()

# In[] cosTheta, cosPsi, sse. Compare within type, between time points, between locations
angleCheckPoints = np.linspace(0,np.pi,7).round(5)
cmap = plt.get_cmap('coolwarm')
norm = mcolors.BoundaryNorm(np.flip(np.cos(angleCheckPoints).round(5)),cmap.N, clip=True)

ed1x, ed2x = checkpointsLabels.index('ED1'), checkpointsLabels.index('ED2')
ld1x, ld2x = checkpointsLabels.index('LD1'), checkpointsLabels.index('LD2')

cseq = mpl.color_sequences['Set2']

for nfi, kfi in enumerate(fitIntervals):
    strategyLabel = 'Rehearse & Update' if kfi[-2:] == '12' else 'Retrieve at Time'

    cosThetas_12_1 = np.array([cosThetas_12[kfi][i][1].mean(0) for i in range(len(cosThetas_12[kfi]))])
    cosThetas_12_2 = np.array([cosThetas_12[kfi][i][2].mean(0) for i in range(len(cosThetas_12[kfi]))])
    cosPsis_12_1 = np.array([cosPsis_12[kfi][i][1].mean(0) for i in range(len(cosPsis_12[kfi]))])
    cosPsis_12_2 = np.array([cosPsis_12[kfi][i][2].mean(0) for i in range(len(cosPsis_12[kfi]))])

    plt.figure(figsize=(10, 4), dpi=100)
    
    # angle
    plt.subplot(1,2,1)
    bpl_L = [cosThetas_12_1[:,:,ed1x,ed2x].mean(1), cosThetas_12_1[:,:,ed1x,ld2x].mean(1), cosThetas_12_1[:,:,ld1x,ed2x].mean(1), cosThetas_12_1[:,:,ld1x,ld2x].mean(1)]
    bpr_L = [cosThetas_12_2[:,:,ed1x,ed2x].mean(1), cosThetas_12_2[:,:,ed1x,ld2x].mean(1), cosThetas_12_2[:,:,ld1x,ed2x].mean(1), cosThetas_12_2[:,:,ld1x,ld2x].mean(1)]
    
    bpl = plt.boxplot(bpl_L, positions=[0.3,1.3,2.3,3.3], flierprops=dict(markeredgecolor='black'), widths = 0.3) #
    bpr = plt.boxplot(bpr_L, positions=[0.7,1.7,2.7,3.7], flierprops=dict(markeredgecolor='grey'), widths = 0.3) #
    f_plotting.set_box_color(bpl, 'black') # colors are from http://colorbrewer2.org/
    f_plotting.set_box_color(bpr, 'grey')
    
    plt.plot([], c='black', label='Retarget')
    plt.plot([], c='grey', label='Distraction')
    
    plt.xticks([0.5,1.5, 2.5, 3.5],['ED1-ED2','ED1-LD2','LD1-ED2','LD1-LD2'],rotation=45)
    
    plt.yticks(np.flip(np.cos(angleCheckPoints).round(5)), labels = np.flip(np.degrees(angleCheckPoints).round(1).astype(int)))
    #plt.xlabel('Conditions', labelpad = 3, fontsize = 12)
    plt.ylabel('cos(θ)', labelpad = 1, fontsize = 12)
    plt.title(f'Principal Angle', fontsize = 15)
    
    # alignment
    plt.subplot(1,2,2)
    bpl_L = [cosPsis_12_1[:,:,ed1x,ed2x].mean(1), cosPsis_12_1[:,:,ed1x,ld2x].mean(1), cosPsis_12_1[:,:,ld1x,ed2x].mean(1), cosPsis_12_1[:,:,ld1x,ld2x].mean(1)]
    bpr_L = [cosPsis_12_2[:,:,ed1x,ed2x].mean(1), cosPsis_12_2[:,:,ed1x,ld2x].mean(1), cosPsis_12_2[:,:,ld1x,ed2x].mean(1), cosPsis_12_2[:,:,ld1x,ld2x].mean(1)]
    
    bpl = plt.boxplot(bpl_L, positions=[0.3,1.3,2.3,3.3], flierprops=dict(markeredgecolor='black'), widths = 0.3) #
    bpr = plt.boxplot(bpr_L, positions=[0.7,1.7,2.7,3.7], flierprops=dict(markeredgecolor='grey'), widths = 0.3) #
    f_plotting.set_box_color(bpl, 'black') # colors are from http://colorbrewer2.org/
    f_plotting.set_box_color(bpr, 'grey')
    
    plt.plot([], c='black', label='Retarget')
    plt.plot([], c='grey', label='Distraction')
    
    plt.xticks([0.5,1.5, 2.5, 3.5],['ED1-ED2','ED1-LD2','LD1-ED2','LD1-LD2'],rotation=45)
    
    plt.yticks(np.flip(np.cos(angleCheckPoints).round(5)), labels = np.flip(np.degrees(angleCheckPoints).round(1).astype(int)))
    #plt.xlabel('Conditions', labelpad = 3, fontsize = 12)
    plt.ylabel('cos(Ψ)', labelpad = 1, fontsize = 12)
    plt.title(f'Representational Alignment', fontsize = 15)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    
    
    plt.suptitle(f'I1D1 vs. I2D2, {strategyLabel}', fontsize = 20, y=1)
    #plt.suptitle(f'abs(cosPsi), {region}, ttype={tt}', fontsize = 15, y=1)
    plt.tight_layout()
    plt.show()
    
    
# In[] just retarget
color1, color2 = 'mediumvioletred', 'darkcyan'

boxprops1 = dict(facecolor=color1, edgecolor='none')
flierprops1 = dict(markeredgecolor=color1, markerfacecolor=color1)
capprops1 = dict(color=color1)
whiskerprops1 = dict(color=color1)

boxprops2 = dict(facecolor=color2, edgecolor='none')
flierprops2 = dict(markeredgecolor=color2, markerfacecolor=color2)
capprops2 = dict(color=color2)
whiskerprops2 = dict(color=color2)

medianprops = dict(linestyle='--', linewidth=1, color='w')
meanpointprops1 = dict(marker='^', markeredgecolor=color1, markerfacecolor='w')
meanpointprops2 = dict(marker='^', markeredgecolor=color2, markerfacecolor='w')

cosThetas_12_1 = np.array([cosThetas_12['ed12'][i][1].mean(0) for i in range(len(cosThetas_12['ed12']))])
cosThetas_12_2 = np.array([cosThetas_12['ed2'][i][1].mean(0) for i in range(len(cosThetas_12['ed2']))])
cosPsis_12_1 = np.array([cosPsis_12['ed12'][i][1].mean(0) for i in range(len(cosPsis_12['ed12']))])
cosPsis_12_2 = np.array([cosPsis_12['ed2'][i][1].mean(0) for i in range(len(cosPsis_12['ed2']))])

plt.figure(figsize=(10, 4), dpi=300)

# angle
plt.subplot(1,2,1)
bpl_L = [cosThetas_12_1[:,:,ed1x,ed2x].mean(1), cosThetas_12_1[:,:,ed1x,ld2x].mean(1), cosThetas_12_1[:,:,ld1x,ed2x].mean(1), cosThetas_12_1[:,:,ld1x,ld2x].mean(1)]
bpr_L = [cosThetas_12_2[:,:,ed1x,ed2x].mean(1), cosThetas_12_2[:,:,ed1x,ld2x].mean(1), cosThetas_12_2[:,:,ld1x,ed2x].mean(1), cosThetas_12_2[:,:,ld1x,ld2x].mean(1)]

bpl = plt.boxplot(bpl_L, positions=[0.3,1.3,2.3,3.3], patch_artist=True, widths = 0.3, boxprops=boxprops1, flierprops=flierprops1, meanprops=meanpointprops1, medianprops=medianprops, capprops = capprops1, whiskerprops = whiskerprops1, meanline=False, showmeans=True) #
bpr = plt.boxplot(bpr_L, positions=[0.7,1.7,2.7,3.7], patch_artist=True, widths = 0.3, boxprops=boxprops2, flierprops=flierprops2, meanprops=meanpointprops2, medianprops=medianprops, capprops = capprops2, whiskerprops = whiskerprops2, meanline=False, showmeans=True) #


plt.plot([], c=color1, label='Rehearse & Update')
plt.plot([], c=color2, label='Retrieve at Time')

plt.xticks([0.5,1.5, 2.5, 3.5],['ED1-ED2','ED1-LD2','LD1-ED2','LD1-LD2'],rotation=45)

plt.yticks(np.flip(np.cos(angleCheckPoints).round(5)), labels = np.flip(np.degrees(angleCheckPoints).round(1).astype(int)))
#plt.xlabel('Conditions', labelpad = 3, fontsize = 12)
plt.ylabel('cos(θ)', labelpad = 1, fontsize = 12)
plt.title(f'Principal Angle', fontsize = 15)

# alignment
plt.subplot(1,2,2)
bpl_L = [cosPsis_12_1[:,:,ed1x,ed2x].mean(1), cosPsis_12_1[:,:,ed1x,ld2x].mean(1), cosPsis_12_1[:,:,ld1x,ed2x].mean(1), cosPsis_12_1[:,:,ld1x,ld2x].mean(1)]
bpr_L = [cosPsis_12_2[:,:,ed1x,ed2x].mean(1), cosPsis_12_2[:,:,ed1x,ld2x].mean(1), cosPsis_12_2[:,:,ld1x,ed2x].mean(1), cosPsis_12_2[:,:,ld1x,ld2x].mean(1)]

bpl = plt.boxplot(bpl_L, positions=[0.3,1.3,2.3,3.3], patch_artist=True, widths = 0.3, boxprops=boxprops1, flierprops=flierprops1, meanprops=meanpointprops1, medianprops=medianprops, capprops = capprops1, whiskerprops = whiskerprops1, meanline=False, showmeans=True) #
bpr = plt.boxplot(bpr_L, positions=[0.7,1.7,2.7,3.7], patch_artist=True, widths = 0.3, boxprops=boxprops2, flierprops=flierprops2, meanprops=meanpointprops2, medianprops=medianprops, capprops = capprops2, whiskerprops = whiskerprops2, meanline=False, showmeans=True) #

plt.plot([], c=color1, label='Rehearse & Update')
plt.plot([], c=color2, label='Retrieve at Time')

plt.xticks([0.5,1.5, 2.5, 3.5],['ED1-ED2','ED1-LD2','LD1-ED2','LD1-LD2'],rotation=45)

plt.yticks(np.flip(np.cos(angleCheckPoints).round(5)), labels = np.flip(np.degrees(angleCheckPoints).round(1).astype(int)))
#plt.xlabel('Conditions', labelpad = 3, fontsize = 12)
plt.ylabel('cos(Ψ)', labelpad = 1, fontsize = 12)
plt.title(f'Representational Alignment', fontsize = 15)
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))


plt.suptitle(f'I1D1 vs. I2D2, Retarget (RNNs)', fontsize = 20, y=1)
#plt.suptitle(f'abs(cosPsi), {region}, ttype={tt}', fontsize = 15, y=1)
plt.tight_layout()
plt.show()

# In[] plot RNNs accuracy 
for nfi, kfi in enumerate(fitIntervals):
    models_dict = modelDicts[kfi]

    label = f'fit interval:{kfi}'
    strategyLabel = 'Rehearse & Update' if kfi[-2:] == '12' else 'Retrieve at Recall'
    
    accs = []
    for i in range(len(models_dict)):
        
        modelD = models_dict[i]['rnn']
        acc_memo = f_evaluateRNN.evaluate_acc(modelD, test_X, test_label, toPrint=False, checkpointX = np.where(tRange <= fitIntervals[kfi][-1][-1])[0].max()) #
        
        #modelD.hidden_noise = 0.5
        
        if acc_memo >=90:
            
            accs += [acc_memo]
    
    accs = np.array(accs)

    print(f'Mean Accuracy = {accs.mean()}, {strategyLabel} (N = {len(accs)})')
    

# In[] compare eu-distance D1-D2
#%%

################
# euDist drift #
################

#%%
nPerms = 10
nBoots = 1
infoMethod='lda'

bins = 50
tslice = (tRange.min(),tRange.max()+dt)
tbins = np.arange(tslice[0], tslice[1], bins)
end_D1s, end_D2s = np.arange(800,1300,bins), np.arange(2100,2600,bins)

euDists = {}
euDists_centroids = {}
euDists_centroids2 = {}

for nfi, kfi in enumerate(fitIntervals):
    print(kfi)
    
    label = f'fit interval:{kfi}'
    strategyLabel = 'Rehearse & Update' if kfi[-2:] == '12' else 'Retrieve at Recall'
    
    models_dict = modelDicts[kfi]
    
    euDists[kfi] = {tt:[] for tt in ttypes}
    euDists_centroids[kfi] = {tt:[] for tt in ttypes}
    euDists_centroids2[kfi] = {tt:[] for tt in ttypes}

    vecs_C, projs_C, projsAll_C, trialInfos_C, data_3pc_C, _, evrs_C, vecs_C_shuff, projs_C_shuff, projsAll_C_shuff, trialInfos_C_shuff, data_3pc_C_shuff, _ = f_evaluateRNN.generate_choiceVectors(models_dict, trialInfo, X_, Y0_, tRange, nBoots=nBoots, pca_tWins=((300,1300),(1600,2600),), 
                                                                                                                                                                                                    dt=dt, toPlot = False, label=f'{strategyLabel}',plot3d=False)
    
    
    for i in range(len(modelDicts[kfi])):
        
        geomsC_valid, geomsC_shuff = (vecs_C[i], projs_C[i], projsAll_C[i], trialInfos_C[i], data_3pc_C[i]), (vecs_C_shuff[i], projs_C_shuff[i], projsAll_C_shuff[i], trialInfos_C_shuff[i], data_3pc_C_shuff[i])
        
        
        euDistT = f_evaluateRNN.get_euDist(geomsC_valid, locs = locs, ttypes = ttypes, bins = bins, dt = dt, tslice = tslice, end_D1s = end_D1s, end_D2s = end_D2s, zscore = False)
        euDist_centroidsT = f_evaluateRNN.get_euDist_centroids(geomsC_valid, locs = locs, ttypes = ttypes, bins = bins, dt = dt, tslice = tslice, end_D1s = end_D1s, end_D2s = end_D2s, zscore = False)
        euDist_centroids2T = f_evaluateRNN.get_euDist_centroids2(geomsC_valid, locs = locs, ttypes = ttypes, bins = bins, dt = dt, tslice = tslice, end_D1s = end_D1s, end_D2s = end_D2s, zscore = False)
        
        for tt in ttypes:
            euDists[kfi][tt] += [euDistT[tt]]
            euDists_centroids[kfi][tt] += [euDist_centroidsT[tt]]
            euDists_centroids2[kfi][tt] += [euDist_centroids2T[tt]]

#%%
#np.save(f'{save_path}/euDists_rnns.npy', euDists, allow_pickle=True)
#np.save(f'{save_path}/euDists_rnns_centroids.npy', euDists_centroids, allow_pickle=True)
#np.save(f'{save_path}/euDists_rnns_centroids2.npy', euDists_centroids2, allow_pickle=True)            
# In[]
#cseq = mpl.color_sequences['Paired']
#color1, color1_, color2, color2_ = cseq[7], cseq[6], cseq[9], cseq[8]
color1, color1_, color2, color2_ = '#d29c2f', '#f5df7a', '#3c79b4', '#b3cde4'
#color1, color2 = 'darkcyan', 'rebeccapurple'
#alpha=0.65

boxprops1 = dict(facecolor=color1, edgecolor='none')
flierprops1 = dict(markeredgecolor=color1, markerfacecolor=color1)
capprops1 = dict(color=color1)
whiskerprops1 = dict(color=color1)
meanpointprops1 = dict(marker='^', markeredgecolor=color1, markerfacecolor='w')

boxprops2 = dict(facecolor=color2, edgecolor='none',alpha = 1)
flierprops2 = dict(markeredgecolor=color2, markerfacecolor=color2,alpha = 1)
capprops2 = dict(color=color2,alpha = 1)
whiskerprops2 = dict(color=color2,alpha = 1)
meanpointprops2 = dict(marker='^', markeredgecolor=color2, markerfacecolor='w',alpha = 1)

medianprops = dict(linestyle='--', linewidth=1, color='w')


fig = plt.figure(figsize=(3, 3), dpi=100)

plt.boxplot([euDists['ed2'][1]], positions=[0.3], patch_artist=True, widths = 0.2, boxprops=boxprops1, flierprops=flierprops1, 
                  meanprops=meanpointprops1, medianprops=medianprops, capprops = capprops1, whiskerprops = whiskerprops1, meanline=False, showmeans=True)
plt.boxplot([euDists['ed12'][1]], positions=[1.3], patch_artist=True, widths = 0.2, boxprops=boxprops2, flierprops=flierprops2, 
                  meanprops=meanpointprops2, medianprops=medianprops, capprops = capprops2, whiskerprops = whiskerprops2, meanline=False, showmeans=True)

plt.boxplot([euDists['ed2'][2]], positions=[0.7], patch_artist=True, widths = 0.2, boxprops=dict(boxprops1,facecolor=color1_), flierprops=dict(flierprops1,markeredgecolor=color1_, markerfacecolor=color1_), 
                  meanprops=dict(meanpointprops1,markeredgecolor=color1_), medianprops=medianprops, capprops = dict(capprops1,color=color1_), whiskerprops = dict(whiskerprops1,color=color1_), meanline=False, showmeans=True)
plt.boxplot([euDists['ed12'][2]], positions=[1.7], patch_artist=True, widths = 0.2, boxprops=dict(boxprops2,facecolor=color2_), flierprops=dict(flierprops2,markeredgecolor=color2_, markerfacecolor=color2_), 
                  meanprops=dict(meanpointprops2,markeredgecolor=color2_), medianprops=medianprops, capprops = dict(capprops2,color=color2_), whiskerprops = dict(whiskerprops2,color=color2_), meanline=False, showmeans=True)

lineh = np.arange(0.3,0.7,0.001)
linev = np.arange(6.15,6.25,0.001)
#p1 = scipy.stats.ttest_ind(euDists['ed2'][1], euDists['ed2'][2])[-1]
p1 = f_stats.permutation_p_diff(euDists['ed2'][1], euDists['ed2'][2])
plt.plot(lineh, np.full_like(lineh, 6.25), 'k-')
plt.plot(np.full_like(linev, 0.3), linev, 'k-')
plt.plot(np.full_like(linev, 0.7), linev, 'k-')
plt.text(0.5,6.25, f'{f_plotting.sig_marker(p1)}',horizontalalignment='center', fontsize=12)


#p2 = scipy.stats.ttest_ind(euDists['ed12'][1], euDists['ed12'][2])[-1]
p2 = f_stats.permutation_p_diff(euDists['ed12'][1], euDists['ed12'][2])
plt.plot(lineh+1, np.full_like(lineh, 6.25), 'k-')
plt.plot(np.full_like(linev, 1.3), linev, 'k-')
plt.plot(np.full_like(linev, 1.7), linev, 'k-')
plt.text(1.5,6.25, f'{f_plotting.sig_marker(p2)}',horizontalalignment='center', fontsize=12)

# draw temporary red and blue lines and use them to create a legend
plt.plot([], c='dimgrey', label='Retarget')
plt.plot([], c='lightgrey', label='Distraction')
plt.legend(bbox_to_anchor=(1.65, 0.5))#loc = 'right'

plt.xticks([0.5,1.5],['Retrieve at Recall','Rehearse & Update'], rotation=30)
plt.xlabel('Strategy', labelpad = 5, fontsize = 12)
plt.ylabel('Euclidean Distance (Std. Scaled)', labelpad = 3, fontsize = 10)
plt.ylim(top=7)
plt.title('Mean Projection Drift, LD2-LD1', fontsize = 15, pad=10)
plt.show()
        
fig.savefig(f'{save_path}/driftDist_rnns.tif')
# In[]

















































#%%

####################
# unused things... #
####################

#%%









































#%%

##################
# generate geoms #
##################

#%%
checkpoints = [150, 550, 1150, 1450, 1850, 2350]#
checkpointsLabels = ['S1','ED1','LD1','S2','ED2','LD2'] #, 2800
avgInterval = {150:150, 550:250, 1150:250, 1450:150, 1850:250, 2350:250} #, 2800:200
nPerms = 10
nBoots = 1
infoMethod='lda'

bins = 50
tslice = (tRange.min(),tRange.max()+dt)
tbins = np.arange(tslice[0], tslice[1], bins)

evrs = {}

vecs_All, projs_All, projsAll_All, trialInfos_All  = {},{},{},{}
vecs_All_shuff, projs_All_shuff, projsAll_All_shuff, trialInfos_All_shuff = {},{},{},{}

vecsC_All, projsC_All, projsCAll_All, trialInfosC_All, pcaC_All = {},{},{},{},{}
vecsC_All_shuff, projsC_All_shuff, projsCAll_All_shuff, trialInfosC_All_shuff = {},{},{},{}

vecs_All_bsl, projs_All_bsl, projsAll_All_bsl, trialInfos_All_bsl = {},{},{},{}

for nfi, kfi in enumerate(fitIntervals):
    print(kfi)
    
    label = f'fit interval:{kfi}'
    strategyLabel = 'Rehearse & Update' if kfi[-2:] == '12' else 'Retrieve at Recall'
    
    models_dict = modelDicts[kfi]
    
    evrsThreshold = 0.9#5 if kfi[-2:] == '12' else 0.9
    evrs[kfi] = []
    

    info3d_1, info3d_2 = [],[]
    info_C1, info_C2 = [],[]
    info_C1X, info_C2X = [],[]

    info3d_1_shuff, info3d_2_shuff = [],[]
    info_C1_shuff, info_C2_shuff = [],[]
    info_C1X_shuff, info_C2X_shuff = [],[]
    
    vecs, projs, projsAll, trialInfos, _, _, vecs_shuff, projs_shuff, projsAll_shuff, trialInfos_shuff, _ = f_evaluateRNN.generate_itemVectors(models_dict, trialInfo, X_, Y0_, tRange, checkpoints, avgInterval, nBoots=nBoots, pca_tWins=((300,1300),(1600,2600),), dt=dt,) #
    
    vecs_C, projs_C, projsAll_C, trialInfos_C, data_3pc_C, _, evrs_C, vecs_C_shuff, projs_C_shuff, projsAll_C_shuff, trialInfos_C_shuff, data_3pc_C_shuff, _ = f_evaluateRNN.generate_choiceVectors(models_dict, trialInfo, X_, Y0_, tRange, nBoots=nBoots, pca_tWins=((300,1300),(1600,2600),), 
                                                                                                                                                                                                    dt=dt, toPlot = False, label=f'{strategyLabel}',plot3d=False) #,{i}
    

# In[]
lineh = np.arange(0.3,0.7,0.001)
linev = np.arange(0,0.1,0.001)
#linev = np.arange(1.15,1.25,0.001)
#linev = np.arange(2.47,2.5,0.001)

fig, axes = plt.subplots(1,1, figsize=(6,4),dpi=300, sharey=True)

################
# figure props #
################

color1, color1_, color2, color2_ = '#d29c2f', '#f5df7a', '#3c79b4', '#b3cde4'
color3, color3_, color4, color4_ = '#185337', '#96d9ad', '#804098', '#c4a2d1'

boxprops1 = dict(facecolor=color1, edgecolor='none')
flierprops1 = dict(markeredgecolor=color1, markerfacecolor=color1)
capprops1 = dict(color=color1)
whiskerprops1 = dict(color=color1)
meanpointprops1 = dict(marker='^', markeredgecolor=color1, markerfacecolor='w')

boxprops2 = dict(facecolor=color2, edgecolor='none',alpha = 1)
flierprops2 = dict(markeredgecolor=color2, markerfacecolor=color2,alpha = 1)
capprops2 = dict(color=color2,alpha = 1)
whiskerprops2 = dict(color=color2,alpha = 1)
meanpointprops2 = dict(marker='^', markeredgecolor=color2, markerfacecolor='w',alpha = 1)

#medianprops = dict(linestyle='--', linewidth=1, color='w')

boxprops3 = dict(facecolor=color3, edgecolor='none')
flierprops3 = dict(markeredgecolor=color3, markerfacecolor=color3)
capprops3 = dict(color=color3)
whiskerprops3 = dict(color=color3)
meanpointprops3 = dict(marker='^', markeredgecolor=color3, markerfacecolor='w')

boxprops4 = dict(facecolor=color4, edgecolor='none',alpha = 1)
flierprops4 = dict(markeredgecolor=color4, markerfacecolor=color4,alpha = 1)
capprops4 = dict(color=color4,alpha = 1)
whiskerprops4 = dict(color=color4,alpha = 1)
meanpointprops4 = dict(marker='^', markeredgecolor=color4, markerfacecolor='w',alpha = 1)

medianprops = dict(linestyle='--', linewidth=1, color='w')

ax = axes#.flatten()[l1]

########
# rnns #
########


euDists_rnns_ed2_1 = np.array(euDists_centroids2['ed2'][1]).mean(1).mean(1)
euDists_rnns_ed2_2 = np.array(euDists_centroids2['ed2'][2]).mean(1).mean(1)
euDists_rnns_ed12_1 = np.array(euDists_centroids2['ed12'][1]).mean(1).mean(1)
euDists_rnns_ed12_2 = np.array(euDists_centroids2['ed12'][2]).mean(1).mean(1)


ax.boxplot([euDists_rnns_ed2_1], positions=[0.3], patch_artist=True, widths = 0.2, boxprops=boxprops1, flierprops=flierprops1,
                    meanprops=meanpointprops1, medianprops=medianprops, capprops = capprops1, whiskerprops = whiskerprops1, meanline=False, showmeans=True)
ax.boxplot([euDists_rnns_ed12_1], positions=[1.3], patch_artist=True, widths = 0.2, boxprops=boxprops2, flierprops=flierprops2,
                    meanprops=meanpointprops2, medianprops=medianprops, capprops = capprops2, whiskerprops = whiskerprops2, meanline=False, showmeans=True)

ax.boxplot([euDists_rnns_ed2_2], positions=[0.7], patch_artist=True, widths = 0.2, boxprops=dict(boxprops1,facecolor=color1_), flierprops=dict(flierprops1,markeredgecolor=color1_, markerfacecolor=color1_),
                    meanprops=dict(meanpointprops1,markeredgecolor=color1_), medianprops=medianprops, capprops = dict(capprops1,color=color1_), whiskerprops = dict(whiskerprops1,color=color1_), meanline=False, showmeans=True)
ax.boxplot([euDists_rnns_ed12_2], positions=[1.7], patch_artist=True, widths = 0.2, boxprops=dict(boxprops2,facecolor=color2_), flierprops=dict(flierprops2,markeredgecolor=color2_, markerfacecolor=color2_),
                    meanprops=dict(meanpointprops2,markeredgecolor=color2_), medianprops=medianprops, capprops = dict(capprops2,color=color2_), whiskerprops = dict(whiskerprops2,color=color2_), meanline=False, showmeans=True)



ylims = ax.get_ylim()
yscale = (ylims[1] - ylims[0])//(ylims[1]//2)

#p1 = scipy.stats.ttest_ind(euDists['ed2'][1], euDists['ed2'][2])[-1]
p1 = f_stats.permutation_p_diff(euDists_rnns_ed2_1, euDists_rnns_ed2_2)
ax.plot(lineh, np.full_like(lineh, ylims[1].round(2)+0.1), 'k-')
#ax.plot(lineh, np.full_like(lineh, 1.25), 'k-')
ax.plot(np.full_like(linev+ylims[1].round(2), 0.3), linev+ylims[1].round(2), 'k-')
ax.plot(np.full_like(linev+ylims[1].round(2), 0.7), linev+ylims[1].round(2), 'k-')
ax.text(0.5,ylims[1].round(2)+(yscale//5), f'{f_plotting.sig_marker(p1,ns_note=True)}',horizontalalignment='center', fontsize=12)
#ax.text(0.5,1.25, f'{f_plotting.sig_marker(p1)}',horizontalalignment='center', fontsize=12)


#p2 = scipy.stats.ttest_ind(euDists['ed12'][1], euDists['ed12'][2])[-1]
p2 = f_stats.permutation_p_diff(euDists_rnns_ed12_1, euDists_rnns_ed12_2)
ax.plot(lineh+1, np.full_like(lineh, ylims[1].round(2)+0.1), 'k-')
#ax.plot(lineh+1, np.full_like(lineh, 1.25), 'k-')
ax.plot(np.full_like(linev+ylims[1].round(2), 1.3), linev+ylims[1].round(2), 'k-')
ax.plot(np.full_like(linev+ylims[1].round(2), 1.7), linev+ylims[1].round(2), 'k-')
ax.text(1.5,ylims[1].round(2)+(yscale//5), f'{f_plotting.sig_marker(p2,ns_note=True)}',horizontalalignment='center', fontsize=12)
#ax.text(1.5,1.25, f'{f_plotting.sig_marker(p2)}',horizontalalignment='center', fontsize=12)

ax.plot([], c='dimgrey', label='Retarget')
ax.plot([], c='lightgrey', label='Distraction')
ax.legend(bbox_to_anchor=(1.3, 0.6), fontsize = 10)#loc = 'right',

ax.set_xticks([0.5,1.5],['R@R','R&U'], rotation=0)
ax.set_xlabel('Strategies/Regions', labelpad = 5, fontsize = 12)

ax.set_ylabel('Euclidean Distance (Std. Scaled)', labelpad = 3, fontsize = 10)

ax.set_ylim(top=ylims[1].round(2)+(yscale//2))


plt.suptitle('Mean Projection Drift', fontsize = 20, y=1)
#plt.tight_layout()
plt.show()
# In[] stability ratio
nIters = 5
nPerms = 20
d1 = np.arange(800,1300+bins,bins)
d1x = [tbins.tolist().index(t) for t in d1]

d2 = np.arange(2100,2600+bins,bins)
d2x = [tbins.tolist().index(t) for t in d2]

##############
# full space #
##############
stab_ratioD1s_f, stab_ratioD2s_f = {}, {}
stab_ratioD1s_shuff_f, stab_ratioD2s_shuff_f = {}, {}

for nfi, kfi in enumerate(fitIntervals):

    pfmX1 = performancesX1[kfi]
    pfmX2 = performancesX2[kfi]
    pfmX1_shuff = performancesX1_shuff[kfi]
    pfmX2_shuff = performancesX2_shuff[kfi]

    pfmX1T = {1:np.array(pfmX1['Retarget']), 2:np.array(pfmX1['Distractor'])}
    pfmX2T = {1:np.array(pfmX2['Retarget']), 2:np.array(pfmX2['Distractor'])}
    
    pfmX1T_shuff = {1:np.swapaxes(np.swapaxes(np.concatenate(pfmX1_shuff['Retarget'],-1),1,2),0,1), 2:np.swapaxes(np.swapaxes(np.concatenate(pfmX1_shuff['Distractor'],-1),1,2),0,1)}
    pfmX2T_shuff = {1:np.swapaxes(np.swapaxes(np.concatenate(pfmX2_shuff['Retarget'],-1),1,2),0,1), 2:np.swapaxes(np.swapaxes(np.concatenate(pfmX2_shuff['Distractor'],-1),1,2),0,1)}
    

    stab_ratioD1, stab_ratioD2 = [], []
    stab_ratioD1_shuff, stab_ratioD2_shuff = [], []
    
    for n in range(nIters):
        
        pfm1_ret, pfm1_dis = pfmX1T[1][n], pfmX1T[2][n]
        pfm2_ret, pfm2_dis = pfmX2T[1][n], pfmX2T[2][n]
        
        stab_ratioD1 += [np.mean((f_decoding.stability_ratio(pfm1_ret[d1x,:][:,d1x]), f_decoding.stability_ratio(pfm1_dis[d1x,:][:,d1x])))] # for d1, use only I1
        stab_ratioD2 += [np.mean((f_decoding.stability_ratio(pfm2_ret[d2x,:][:,d2x]), f_decoding.stability_ratio(pfm1_dis[d2x,:][:,d2x])))] # for d2, use only choice item
        
        for npm in range(nIters*nPerms):
            pfm1_ret_shuff, pfm1_dis_shuff = pfmX1T_shuff[1][npm], pfmX1T_shuff[2][npm]
            pfm2_ret_shuff, pfm2_dis_shuff = pfmX2T_shuff[1][npm], pfmX2T_shuff[2][npm]
            
            stab_ratioD1_shuff += [np.mean((f_decoding.stability_ratio(pfm1_ret_shuff[d1x,:][:,d1x]), f_decoding.stability_ratio(pfm1_dis_shuff[d1x,:][:,d1x])))] # for d1, use only I1
            stab_ratioD2_shuff += [np.mean((f_decoding.stability_ratio(pfm2_ret_shuff[d2x,:][:,d2x]), f_decoding.stability_ratio(pfm1_dis_shuff[d2x,:][:,d2x])))] # for d2, use only choice item
        
    stab_ratioD1s_f[kfi], stab_ratioD2s_f[kfi] = np.array(stab_ratioD1), np.array(stab_ratioD2)
    stab_ratioD1s_shuff_f[kfi], stab_ratioD2s_shuff_f[kfi] = np.array(stab_ratioD1_shuff), np.array(stab_ratioD2_shuff)


###########
# readout #
###########

stab_ratioD1s_r, stab_ratioD2s_r = {}, {}
stab_ratioD1s_shuff_r, stab_ratioD2s_shuff_r = {}, {}

for nfi, kfi in enumerate(fitIntervals):

    info_C1X,info_C2X = infos_C1X[kfi], infos_C2X[kfi] 
    info_C1X_shuff,info_C2X_shuff = infos_C1X_shuff[kfi], infos_C2X_shuff[kfi]

    decode_projC1X_3d = {1:np.array([info_C1X[i][1].mean(0) for i in range(len(info_C1X))]),
                         2:np.array([info_C1X[j][2].mean(0) for j in range(len(info_C1X))])}
    decode_projC2X_3d = {1:np.array([info_C2X[i][1].mean(0) for i in range(len(info_C2X))]),
                         2:np.array([info_C2X[j][2].mean(0) for j in range(len(info_C2X))])}
    
    decode_projC1X_3d_shuff = {1:np.concatenate(np.array([info_C1X_shuff[i][1] for i in range(len(info_C1X_shuff))])),
                         2:np.concatenate(np.array([info_C1X_shuff[j][2] for j in range(len(info_C1X_shuff))]))}
    decode_projC2X_3d_shuff = {1:np.concatenate(np.array([info_C2X_shuff[i][1] for i in range(len(info_C2X_shuff))])),
                         2:np.concatenate(np.array([info_C2X_shuff[j][2] for j in range(len(info_C2X_shuff))]))}
    

    stab_ratioD1, stab_ratioD2 = [], []
    stab_ratioD1_shuff, stab_ratioD2_shuff = [], []
    
    for n in range(nIters):
        
        pfm1_ret, pfm1_dis = decode_projC1X_3d[1][n], decode_projC1X_3d[2][n]
        pfm2_ret, pfm2_dis = decode_projC2X_3d[1][n], decode_projC2X_3d[2][n]
        
        stab_ratioD1 += [np.mean((f_decoding.stability_ratio(pfm1_ret[d1x,:][:,d1x]), f_decoding.stability_ratio(pfm1_dis[d1x,:][:,d1x])))] # for d1, use only I1
        stab_ratioD2 += [np.mean((f_decoding.stability_ratio(pfm2_ret[d2x,:][:,d2x]), f_decoding.stability_ratio(pfm1_dis[d2x,:][:,d2x])))] # for d2, use only choice item
        
        for npm in range(nIters*nPerms):
            pfm1_ret_shuff, pfm1_dis_shuff = decode_projC1X_3d_shuff[1][npm], decode_projC1X_3d_shuff[2][npm]
            pfm2_ret_shuff, pfm2_dis_shuff = decode_projC2X_3d_shuff[1][npm], decode_projC2X_3d_shuff[2][npm]
            
            stab_ratioD1_shuff += [np.mean((f_decoding.stability_ratio(pfm1_ret_shuff[d1x,:][:,d1x]), f_decoding.stability_ratio(pfm1_dis_shuff[d1x,:][:,d1x])))] # for d1, use only I1
            stab_ratioD2_shuff += [np.mean((f_decoding.stability_ratio(pfm2_ret_shuff[d2x,:][:,d2x]), f_decoding.stability_ratio(pfm1_dis_shuff[d2x,:][:,d2x])))] # for d2, use only choice item
        
    stab_ratioD1s_r[kfi], stab_ratioD2s_r[kfi] = np.array(stab_ratioD1), np.array(stab_ratioD2)
    stab_ratioD1s_shuff_r[kfi], stab_ratioD2s_shuff_r[kfi] = np.array(stab_ratioD1_shuff), np.array(stab_ratioD2_shuff)

# In[]
###############
# plot params #
###############

color1, color1_, color2, color2_ = '#d29c2f', '#f5df7a', '#3c79b4', '#b3cde4'


boxprops1 = dict(facecolor=color1, edgecolor='none')
flierprops1 = dict(markeredgecolor=color1, markerfacecolor=color1)
capprops1 = dict(color=color1)
whiskerprops1 = dict(color=color1)
meanpointprops1 = dict(marker='^', markeredgecolor=color1, markerfacecolor='w')

boxprops2 = dict(facecolor=color2, edgecolor='none',alpha = 1)
flierprops2 = dict(markeredgecolor=color2, markerfacecolor=color2,alpha = 1)
capprops2 = dict(color=color2,alpha = 1)
whiskerprops2 = dict(color=color2,alpha = 1)
meanpointprops2 = dict(marker='^', markeredgecolor=color2, markerfacecolor='w',alpha = 1)

medianprops = dict(linestyle='--', linewidth=1, color='w')

########
# plot #
########

lineh = np.arange(0.3,0.7,0.001)
linev = np.arange(1.045,1.05,0.0001)

#fig = plt.figure(figsize=(3, 3), dpi=300)
fig, axes = plt.subplots(1,2, sharey=True, figsize=(6,3), dpi=300)

# full space
ax = axes.flatten()[0]

ax.boxplot([stab_ratioD1s_f['ed2']], positions=[0.3], patch_artist=True, widths = 0.2, boxprops=boxprops1, flierprops=flierprops1, 
                  meanprops=meanpointprops1, medianprops=medianprops, capprops = capprops1, whiskerprops = whiskerprops1, meanline=False, showmeans=True)
ax.boxplot([stab_ratioD1s_f['ed12']], positions=[1.3], patch_artist=True, widths = 0.2, boxprops=boxprops2, flierprops=flierprops2, 
                  meanprops=meanpointprops2, medianprops=medianprops, capprops = capprops2, whiskerprops = whiskerprops2, meanline=False, showmeans=True)

ax.boxplot([stab_ratioD2s_f['ed2']], positions=[0.7], patch_artist=True, widths = 0.2, boxprops=dict(boxprops1,facecolor=color1_), flierprops=dict(flierprops1,markeredgecolor=color1_, markerfacecolor=color1_), 
                  meanprops=dict(meanpointprops1,markeredgecolor=color1_), medianprops=medianprops, capprops = dict(capprops1,color=color1_), whiskerprops = dict(whiskerprops1,color=color1_), meanline=False, showmeans=True)
ax.boxplot([stab_ratioD2s_f['ed12']], positions=[1.7], patch_artist=True, widths = 0.2, boxprops=dict(boxprops2,facecolor=color2_), flierprops=dict(flierprops2,markeredgecolor=color2_, markerfacecolor=color2_), 
                  meanprops=dict(meanpointprops2,markeredgecolor=color2_), medianprops=medianprops, capprops = dict(capprops2,color=color2_), whiskerprops = dict(whiskerprops2,color=color2_), meanline=False, showmeans=True)



p1 = f_stats.permutation_p(stab_ratioD1s_f['ed2'].mean(), stab_ratioD1s_shuff_f['ed2'])
ax.text(0.3,1.01, f'{f_plotting.sig_marker(p1)}',horizontalalignment='center', fontsize=12)

p2 = f_stats.permutation_p(stab_ratioD2s_f['ed2'].mean(), stab_ratioD2s_shuff_f['ed2'])
ax.text(0.7,1.01, f'{f_plotting.sig_marker(p2)}',horizontalalignment='center', fontsize=12)

p3 = f_stats.permutation_p(stab_ratioD1s_f['ed12'].mean(), stab_ratioD1s_shuff_f['ed12'])
ax.text(1.3,1.01, f'{f_plotting.sig_marker(p3)}',horizontalalignment='center', fontsize=12)

p4 = f_stats.permutation_p(stab_ratioD2s_f['ed12'].mean(), stab_ratioD2s_shuff_f['ed12'])
ax.text(1.7,1.01, f'{f_plotting.sig_marker(p4)}',horizontalalignment='center', fontsize=12)

#p5,_,_ = f_stats.bootstrap95_p(stab_ratioD1s_f['ed2'], stab_ratioD2s_f['ed2'])
p5 = f_stats.permutation_p_diff(stab_ratioD1s_f['ed2'], stab_ratioD2s_f['ed2'])
ax.plot(lineh, np.full_like(lineh, 1.05), 'k-')
ax.plot(np.full_like(linev, 0.3), linev, 'k-')
ax.plot(np.full_like(linev, 0.7), linev, 'k-')
ax.text(0.5,1.055, f'{f_plotting.sig_marker(p5,ns_note=True)}',horizontalalignment='center', fontsize=12)

#p6,_,_ = f_stats.bootstrap95_p(stab_ratioD1s_f['ed12'], stab_ratioD2s_f['ed12'])
p6 = f_stats.permutation_p_diff(stab_ratioD1s_f['ed12'], stab_ratioD2s_f['ed12'])
ax.plot(lineh+1, np.full_like(lineh, 1.05), 'k-')
ax.plot(np.full_like(linev, 0.3)+1, linev, 'k-')
ax.plot(np.full_like(linev, 0.7)+1, linev, 'k-')
ax.text(1.5,1.055, f'{f_plotting.sig_marker(p6,ns_note=True)}',horizontalalignment='center', fontsize=12)

xlims = ax.get_xlim()
lineThresholds = np.arange(xlims[0],xlims[1]+0.01,0.01)
#plt.plot(lineThresholds, np.full_like(lineThresholds,0), 'k-.', alpha = 0.5)
ax.plot(lineThresholds, np.full_like(lineThresholds,1), 'k--', alpha = 0.5, linewidth=1)

# draw temporary red and blue lines and use them to create a legend
#ax.plot([], c='dimgrey', label='Delay1')
#ax.plot([], c='lightgrey', label='Delay2')
#ax.legend(bbox_to_anchor=(1.55, 0.5))#loc = 'right',

ax.set_xticks([0.5,1.5],['R@R','R&U'], rotation=30)
ax.set_xlabel('Strategy', labelpad = 5, fontsize = 12)
ax.set_ylabel('Code Stability Ratio', labelpad = 3, fontsize = 12)
ax.set_ylim(top=1.1)
ax.set_title('Full Space', fontsize = 15, pad=10)

# readout
ax = axes.flatten()[1]
#fig = plt.figure(figsize=(3, 6), dpi=100)
ax.boxplot([stab_ratioD1s_r['ed2']], positions=[0.3], patch_artist=True, widths = 0.2, boxprops=boxprops1, flierprops=flierprops1, 
                  meanprops=meanpointprops1, medianprops=medianprops, capprops = capprops1, whiskerprops = whiskerprops1, meanline=False, showmeans=True)
ax.boxplot([stab_ratioD1s_r['ed12']], positions=[1.3], patch_artist=True, widths = 0.2, boxprops=boxprops2, flierprops=flierprops2, 
                  meanprops=meanpointprops2, medianprops=medianprops, capprops = capprops2, whiskerprops = whiskerprops2, meanline=False, showmeans=True)

ax.boxplot([stab_ratioD2s_r['ed2']], positions=[0.7], patch_artist=True, widths = 0.2, boxprops=dict(boxprops1,facecolor=color1_), flierprops=dict(flierprops1,markeredgecolor=color1_, markerfacecolor=color1_), 
                  meanprops=dict(meanpointprops1,markeredgecolor=color1_), medianprops=medianprops, capprops = dict(capprops1,color=color1_), whiskerprops = dict(whiskerprops1,color=color1_), meanline=False, showmeans=True)
ax.boxplot([stab_ratioD2s_r['ed12']], positions=[1.7], patch_artist=True, widths = 0.2, boxprops=dict(boxprops2,facecolor=color2_), flierprops=dict(flierprops2,markeredgecolor=color2_, markerfacecolor=color2_), 
                  meanprops=dict(meanpointprops2,markeredgecolor=color2_), medianprops=medianprops, capprops = dict(capprops2,color=color2_), whiskerprops = dict(whiskerprops2,color=color2_), meanline=False, showmeans=True)



p1 = f_stats.permutation_p(stab_ratioD1s_r['ed2'].mean(), stab_ratioD1s_shuff_r['ed2'])
ax.text(0.3,1.01, f'{f_plotting.sig_marker(p1)}',horizontalalignment='center', fontsize=12)

p2 = f_stats.permutation_p(stab_ratioD2s_r['ed2'].mean(), stab_ratioD2s_shuff_r['ed2'])
ax.text(0.7,1.01, f'{f_plotting.sig_marker(p2)}',horizontalalignment='center', fontsize=12)

p3 = f_stats.permutation_p(stab_ratioD1s_r['ed12'].mean(), stab_ratioD1s_shuff_r['ed12'])
ax.text(1.3,1.01, f'{f_plotting.sig_marker(p3)}',horizontalalignment='center', fontsize=12)

p4 = f_stats.permutation_p(stab_ratioD2s_r['ed12'].mean(), stab_ratioD2s_shuff_r['ed12'])
ax.text(1.7,1.01, f'{f_plotting.sig_marker(p4)}',horizontalalignment='center', fontsize=12)

#p5,_,_ = f_stats.bootstrap95_p(stab_ratioD1s_r['ed2'], stab_ratioD2s_r['ed2'])
p5 = f_stats.permutation_p_diff(stab_ratioD1s_r['ed2'], stab_ratioD2s_r['ed2'])
ax.plot(lineh, np.full_like(lineh, 1.05), 'k-')
ax.plot(np.full_like(linev, 0.3), linev, 'k-')
ax.plot(np.full_like(linev, 0.7), linev, 'k-')
ax.text(0.5,1.055, f'{f_plotting.sig_marker(p5,ns_note=True)}',horizontalalignment='center', fontsize=12)

#p6,_,_ = f_stats.bootstrap95_p(stab_ratioD1s_r['ed12'], stab_ratioD2s_r['ed12'])
p6 = f_stats.permutation_p_diff(stab_ratioD1s_r['ed12'], stab_ratioD2s_r['ed12'])
ax.plot(lineh+1, np.full_like(lineh, 1.05), 'k-')
ax.plot(np.full_like(linev, 0.3)+1, linev, 'k-')
ax.plot(np.full_like(linev, 0.7)+1, linev, 'k-')
ax.text(1.5,1.055, f'{f_plotting.sig_marker(p6,ns_note=True)}',horizontalalignment='center', fontsize=12)

xlims = ax.get_xlim()
lineThresholds = np.arange(xlims[0],xlims[1]+0.01,0.01)
#plt.plot(lineThresholds, np.full_like(lineThresholds,0), 'k-.', alpha = 0.5)
ax.plot(lineThresholds, np.full_like(lineThresholds,1), 'k--', alpha = 0.5, linewidth=1)

# draw temporary red and blue lines and use them to create a legend
ax.plot([], c='dimgrey', label='Delay1')
ax.plot([], c='lightgrey', label='Delay2')
ax.legend(bbox_to_anchor=(1.55, 0.5))#loc = 'right',

ax.set_xticks([0.5,1.5],['R@R','R&U'], rotation=30)
ax.set_xlabel('Strategy', labelpad = 5, fontsize = 12)
#ax.set_ylabel('Code Stability Ratio', labelpad = 3, fontsize = 12)
ax.set_ylim(top=1.1)
ax.set_title('Readout Subspace', fontsize = 15, pad=10)


plt.suptitle('Information Stability, RNNs', fontsize = 20, y=1.1)
plt.show()

fig.savefig(f'{save_path}/infoStabRatio_rnns.tif', bbox_inches='tight')

# In[] code morphing
nIters = 5
nPerms = 20
d1 = np.arange(800,1300+bins,bins)
d1x = [tbins.tolist().index(t) for t in d1]

d2 = np.arange(2100,2600+bins,bins)
d2x = [tbins.tolist().index(t) for t in d2]

##############
# full space #
##############
codeMorphDs_f = {}#, {}, codeMorphD21s_f
codeMorphDs_shuff_f = {}#, {}, codeMorphD21s_shuff_f

for nfi, kfi in enumerate(fitIntervals):

    pfmX1 = performancesX1[kfi]
    #pfmX2 = performancesX2[kfi]
    pfmX1_shuff = performancesX1_shuff[kfi]
    #pfmX2_shuff = performancesX2_shuff[kfi]

    pfmX1T = {1:np.array(pfmX1['Retarget']), 2:np.array(pfmX1['Distractor'])}
    #pfmX2T = {1:np.array(pfmX2['Retarget']), 2:np.array(pfmX2['Distractor'])}
    
    pfmX1T_shuff = {1:np.swapaxes(np.swapaxes(np.concatenate(pfmX1_shuff['Retarget'],-1),1,2),0,1), 2:np.swapaxes(np.swapaxes(np.concatenate(pfmX1_shuff['Distractor'],-1),1,2),0,1)}
    #pfmX2T_shuff = {1:np.swapaxes(np.swapaxes(np.concatenate(pfmX2_shuff['Retarget'],-1),1,2),0,1), 2:np.swapaxes(np.swapaxes(np.concatenate(pfmX2_shuff['Distractor'],-1),1,2),0,1)}
    

    codeMorphDs = [] #, [], stab_ratioD2
    codeMorphDs_shuff = []#, [], stab_ratioD2_shuff
    
    for n in range(nIters):
        
        pfm1_ret, pfm1_dis = pfmX1T[1][n], pfmX1T[2][n]
        #pfm2_ret, pfm2_dis = pfmX2T[1][n], pfmX2T[2][n]
        
        codeMorphDs += [np.mean((f_decoding.code_morphing(pfm1_dis[d1x,d1x], pfm1_dis[d1x,d2x]), f_decoding.code_morphing(pfm1_dis[d2x,d2x], pfm1_dis[d2x,d1x])))] # for d1, use only I1
        #stab_ratioD2 += [np.mean((f_decoding.stability_ratio(pfm2_ret[d2x,:][:,d2x]), f_decoding.stability_ratio(pfm1_dis[d2x,:][:,d2x])))] # for d2, use only choice item
        
        for npm in range(nIters*nPerms):
            pfm1_ret_shuff, pfm1_dis_shuff = pfmX1T_shuff[1][npm], pfmX1T_shuff[2][npm]
            #pfm2_ret_shuff, pfm2_dis_shuff = pfmX2T_shuff[1][npm], pfmX2T_shuff[2][npm]
            
            codeMorphDs_shuff += [np.mean((f_decoding.code_morphing(pfm1_dis_shuff[d1x,d1x], pfm1_dis_shuff[d1x,d2x]), f_decoding.code_morphing(pfm1_dis_shuff[d2x,d2x], pfm1_dis_shuff[d2x,d1x])))] # for d1, use only I1
            #stab_ratioD2_shuff += [np.mean((f_decoding.stability_ratio(pfm2_ret_shuff[d2x,:][:,d2x]), f_decoding.stability_ratio(pfm1_dis_shuff[d2x,:][:,d2x])))] # for d2, use only choice item
        
    codeMorphDs_f[kfi] = np.array(codeMorphDs)#, np.array(stab_ratioD2), stab_ratioD2s_f[kfi]
    codeMorphDs_shuff_f[kfi] = np.array(codeMorphDs_shuff)#, np.array(stab_ratioD2_shuff), stab_ratioD2s_shuff_f[kfi]


###########
# readout #
###########
codeMorphDs_r = {}#, {}, codeMorphD21s_f
codeMorphDs_shuff_r = {}#, {}, codeMorphD21s_shuff_f

for nfi, kfi in enumerate(fitIntervals):

    info_C1X = infos_C1X[kfi]#, infos_C2X[kfi] , info_C2X
    info_C1X_shuff = infos_C1X_shuff[kfi]#, infos_C2X_shuff[kfi], info_C2X_shuff

    decode_projC1X_3d = {1:np.array([info_C1X[i][1].mean(0) for i in range(len(info_C1X))]), 2:np.array([info_C1X[j][2].mean(0) for j in range(len(info_C1X))])}
    #decode_projC2X_3d = {1:np.array([info_C2X[i][1].mean(0) for i in range(len(info_C2X))]), 2:np.array([info_C2X[j][2].mean(0) for j in range(len(info_C2X))])}
    
    decode_projC1X_3d_shuff = {1:np.concatenate(np.array([info_C1X_shuff[i][1] for i in range(len(info_C1X_shuff))])), 2:np.concatenate(np.array([info_C1X_shuff[j][2] for j in range(len(info_C1X_shuff))]))}
    #decode_projC2X_3d_shuff = {1:np.concatenate(np.array([info_C2X_shuff[i][1] for i in range(len(info_C2X_shuff))])), 2:np.concatenate(np.array([info_C2X_shuff[j][2] for j in range(len(info_C2X_shuff))]))}
    

    codeMorphDs = []#, [], stab_ratioD2
    codeMorphDs_shuff = []#, [], stab_ratioD2_shuff
    
    for n in range(nIters):
        
        pfm1_ret, pfm1_dis = decode_projC1X_3d[1][n], decode_projC1X_3d[2][n]
        #pfm2_ret, pfm2_dis = decode_projC2X_3d[1][n], decode_projC2X_3d[2][n]
        
        codeMorphDs += [np.mean((f_decoding.code_morphing(pfm1_dis[d1x,d1x], pfm1_dis[d1x,d2x]), f_decoding.code_morphing(pfm1_dis[d2x,d2x], pfm1_dis[d2x,d1x])))] # for d1, use only I1
        #stab_ratioD2 += [np.mean((f_decoding.stability_ratio(pfm2_ret[d2x,:][:,d2x]), f_decoding.stability_ratio(pfm1_dis[d2x,:][:,d2x])))] # for d2, use only choice item
        
        for npm in range(nIters*nPerms):
            pfm1_ret_shuff, pfm1_dis_shuff = decode_projC1X_3d_shuff[1][npm], decode_projC1X_3d_shuff[2][npm]
            #pfm2_ret_shuff, pfm2_dis_shuff = decode_projC2X_3d_shuff[1][npm], decode_projC2X_3d_shuff[2][npm]
            
            codeMorphDs_shuff += [np.mean((f_decoding.code_morphing(pfm1_dis_shuff[d1x,d1x], pfm1_dis_shuff[d1x,d2x]), f_decoding.code_morphing(pfm1_dis_shuff[d2x,d2x], pfm1_dis_shuff[d2x,d1x])))] # for d1, use only I1
            #stab_ratioD2_shuff += [np.mean((f_decoding.stability_ratio(pfm2_ret_shuff[d2x,:][:,d2x]), f_decoding.stability_ratio(pfm1_dis_shuff[d2x,:][:,d2x])))] # for d2, use only choice item
        
    codeMorphDs_r[kfi] = np.array(codeMorphDs)#, np.array(stab_ratioD2), stab_ratioD2s_r[kfi]
    codeMorphDs_shuff_r[kfi] = np.array(codeMorphDs_shuff)#, np.array(stab_ratioD2_shuff), stab_ratioD2s_shuff_r[kfi]

# In[]
###############
# plot params #
###############

color1, color2 = '#d29c2f', '#3c79b4'#, '#f5df7a', '#b3cde4', color1_, color2_


boxprops1 = dict(facecolor=color1, edgecolor='none')
flierprops1 = dict(markeredgecolor=color1, markerfacecolor=color1)
capprops1 = dict(color=color1)
whiskerprops1 = dict(color=color1)
meanpointprops1 = dict(marker='^', markeredgecolor=color1, markerfacecolor='w')

boxprops2 = dict(facecolor=color2, edgecolor='none',alpha = 1)
flierprops2 = dict(markeredgecolor=color2, markerfacecolor=color2,alpha = 1)
capprops2 = dict(color=color2,alpha = 1)
whiskerprops2 = dict(color=color2,alpha = 1)
meanpointprops2 = dict(marker='^', markeredgecolor=color2, markerfacecolor='w',alpha = 1)

medianprops = dict(linestyle='--', linewidth=1, color='w')

########
# plot #
########

lineh = np.arange(0.3,0.7,0.001)
linev = np.arange(1.045,1.05,0.0001)

#fig = plt.figure(figsize=(3, 3), dpi=300)
fig, axes = plt.subplots(1,2, sharey=True, figsize=(6,3), dpi=300)

# full space
ax = axes.flatten()[0]

ax.boxplot([codeMorphDs_f['ed2']], positions=[0.5], patch_artist=True, widths = 0.2, boxprops=boxprops1, flierprops=flierprops1, 
                  meanprops=meanpointprops1, medianprops=medianprops, capprops = capprops1, whiskerprops = whiskerprops1, meanline=False, showmeans=True)
ax.boxplot([codeMorphDs_f['ed12']], positions=[1.5], patch_artist=True, widths = 0.2, boxprops=boxprops2, flierprops=flierprops2, 
                  meanprops=meanpointprops2, medianprops=medianprops, capprops = capprops2, whiskerprops = whiskerprops2, meanline=False, showmeans=True)



p1 = f_stats.permutation_p(codeMorphDs_f['ed2'].mean(), codeMorphDs_shuff_f['ed2'])
ax.text(0.5,2.7, f'{f_plotting.sig_marker(p1)}',horizontalalignment='center', fontsize=12)

p2 = f_stats.permutation_p(codeMorphDs_f['ed12'].mean(), codeMorphDs_shuff_f['ed12'])
ax.text(1.5,2.7, f'{f_plotting.sig_marker(p2)}',horizontalalignment='center', fontsize=12)


xlims = ax.get_xlim()
lineThresholds = np.arange(xlims[0],xlims[1]+0.01,0.01)
#plt.plot(lineThresholds, np.full_like(lineThresholds,0), 'k-.', alpha = 0.5)
ax.plot(lineThresholds, np.full_like(lineThresholds,1), 'k--', alpha = 0.5, linewidth=1)

# draw temporary red and blue lines and use them to create a legend
#ax.plot([], c='dimgrey', label='Delay1')
#ax.plot([], c='lightgrey', label='Delay2')
#ax.legend(bbox_to_anchor=(1.55, 0.5))#loc = 'right',

ax.set_xticks([0.5,1.5],['R@R','R&U'], rotation=30)
ax.set_xlabel('Strategy', labelpad = 5, fontsize = 12)
ax.set_ylabel('D1-D1/D1-D2 (& vice versa)', labelpad = 3, fontsize = 12)
ax.set_ylim(0.9,2.9)
ax.set_title('Full Space', fontsize = 15, pad=10)

# readout
ax = axes.flatten()[1]
#fig = plt.figure(figsize=(3, 6), dpi=100)
ax.boxplot([codeMorphDs_r['ed2']], positions=[0.5], patch_artist=True, widths = 0.2, boxprops=boxprops1, flierprops=flierprops1, 
                  meanprops=meanpointprops1, medianprops=medianprops, capprops = capprops1, whiskerprops = whiskerprops1, meanline=False, showmeans=True)
ax.boxplot([codeMorphDs_r['ed12']], positions=[1.5], patch_artist=True, widths = 0.2, boxprops=boxprops2, flierprops=flierprops2, 
                  meanprops=meanpointprops2, medianprops=medianprops, capprops = capprops2, whiskerprops = whiskerprops2, meanline=False, showmeans=True)


p1 = f_stats.permutation_p(codeMorphDs_r['ed2'].mean(), codeMorphDs_shuff_r['ed2'])
ax.text(0.5,2.7, f'{f_plotting.sig_marker(p1)}',horizontalalignment='center', fontsize=12)

p2 = f_stats.permutation_p(codeMorphDs_r['ed12'].mean(), codeMorphDs_shuff_r['ed12'])
ax.text(1.5,2.7, f'{f_plotting.sig_marker(p2)}',horizontalalignment='center', fontsize=12)

xlims = ax.get_xlim()
lineThresholds = np.arange(xlims[0],xlims[1]+0.01,0.01)
#plt.plot(lineThresholds, np.full_like(lineThresholds,0), 'k-.', alpha = 0.5)
ax.plot(lineThresholds, np.full_like(lineThresholds,1), 'k--', alpha = 0.5, linewidth=1)

ax.set_xticks([0.5,1.5],['R@R','R&U'], rotation=30)
ax.set_xlabel('Strategy', labelpad = 5, fontsize = 12)
#ax.set_ylabel('Code Stability Ratio', labelpad = 3, fontsize = 12)
#ax.set_ylim(top=1.1)
ax.set_title('Readout Subspace', fontsize = 15, pad=10)


plt.suptitle('Code Morphing, RNNs', fontsize = 20, y=1.1)
plt.show()

fig.savefig(f'{save_path}/codeMorphing_rnns.tif', bbox_inches='tight')






# In[] distractor information quantification
nIters = 5
nPerms = 20
d1 = np.arange(1600,2100+bins,bins)
d1x = [tbins.tolist().index(t) for t in d1]

#pfm22_full = {k:np.array(performancesX2[k]['Distractor'])[:,d1x,:][:,:,d1x].mean(-1).mean(-1) for k in performancesX2.keys()}
#pfm22_readout = {k: np.concatenate([infos_C2X[k][n][2] for n in range(len(infos_C2X[k]))])[:,d1x,:][:,:,d1x].mean(-1).mean(-1) for k in infos_C2X.keys()}

#pfm22_full_shuff = {k:np.concatenate(performancesX2_shuff[k]['Distractor'],axis=-1).swapaxes(1,2).swapaxes(0,1)[:,d1x,:][:,:,d1x].mean(-1).mean(-1) for k in performancesX2_shuff.keys()}
#pfm22_readout_shuff = {k: np.concatenate([infos_C2X_shuff[k][n][2] for n in range(len(infos_C2X_shuff[k]))])[:,d1x,:][:,:,d1x].mean(-1).mean(-1) for k in infos_C2X_shuff.keys()}

pfm22_full = {k:np.array(performancesX2[k]['Distractor'])[:,d1x,d1x].mean(-1) for k in performancesX2.keys()}
pfm22_readout = {k: np.concatenate([infos_C2X[k][n][2] for n in range(len(infos_C2X[k]))])[:,d1x,d1x].mean(-1) for k in infos_C2X.keys()}

pfm22_full_shuff = {k:np.concatenate(performancesX2_shuff[k]['Distractor'],axis=-1).swapaxes(1,2).swapaxes(0,1)[:,d1x,d1x].mean(-1) for k in performancesX2_shuff.keys()}
pfm22_readout_shuff = {k: np.concatenate([infos_C2X_shuff[k][n][2] for n in range(len(infos_C2X_shuff[k]))])[:,d1x,d1x].mean(-1) for k in infos_C2X_shuff.keys()}


###############
# plot params #
###############

color1, color1_, color2, color2_ = '#d29c2f', '#f5df7a', '#3c79b4', '#b3cde4'


boxprops1 = dict(facecolor=color1, edgecolor='none')
flierprops1 = dict(markeredgecolor=color1, markerfacecolor=color1)
capprops1 = dict(color=color1)
whiskerprops1 = dict(color=color1)
meanpointprops1 = dict(marker='^', markeredgecolor=color1, markerfacecolor='w')

boxprops2 = dict(facecolor=color2, edgecolor='none',alpha = 1)
flierprops2 = dict(markeredgecolor=color2, markerfacecolor=color2,alpha = 1)
capprops2 = dict(color=color2,alpha = 1)
whiskerprops2 = dict(color=color2,alpha = 1)
meanpointprops2 = dict(marker='^', markeredgecolor=color2, markerfacecolor='w',alpha = 1)

medianprops = dict(linestyle='--', linewidth=1, color='w')

########
# plot #
########

lineh = np.arange(0.5,1.5,0.001)
linev = np.arange(0.71,0.72,0.0001)

#fig = plt.figure(figsize=(3, 3), dpi=300)
fig, axes = plt.subplots(1,2, sharey=True, figsize=(6,3), dpi=300)

# full space
ax = axes.flatten()[0]

ax.boxplot([pfm22_full['ed2']], positions=[0.5], patch_artist=True, widths = 0.2, boxprops=boxprops1, flierprops=flierprops1, 
                  meanprops=meanpointprops1, medianprops=medianprops, capprops = capprops1, whiskerprops = whiskerprops1, meanline=False, showmeans=True)
ax.boxplot([pfm22_full['ed12']], positions=[1.5], patch_artist=True, widths = 0.2, boxprops=boxprops2, flierprops=flierprops2, 
                  meanprops=meanpointprops2, medianprops=medianprops, capprops = capprops2, whiskerprops = whiskerprops2, meanline=False, showmeans=True)



p1 = f_stats.permutation_p(pfm22_full['ed2'].mean(), pfm22_full_shuff['ed2'])
ax.text(0.5,0.7, f'{f_plotting.sig_marker(p1)}',horizontalalignment='center', fontsize=12)

p2 = f_stats.permutation_p(pfm22_full['ed12'].mean(), pfm22_full_shuff['ed12'])
ax.text(1.5,0.7, f'{f_plotting.sig_marker(p2)}',horizontalalignment='center', fontsize=12)


xlims = ax.get_xlim()
lineThresholds = np.arange(xlims[0],xlims[1]+0.01,0.01)
#plt.plot(lineThresholds, np.full_like(lineThresholds,0), 'k-.', alpha = 0.5)
ax.plot(lineThresholds, np.full_like(lineThresholds,0.33), 'k--', alpha = 0.5, linewidth=1)

ax.set_xticks([0.5,1.5],['R@R','R&U'], rotation=30)
ax.set_xlabel('Strategy', labelpad = 5, fontsize = 12)
ax.set_ylabel('Decodability', labelpad = 3, fontsize = 12)
#ax.set_ylim(top=1)
ax.set_title('Full Space', fontsize = 15, pad=10)


# readout subspace
ax = axes.flatten()[1]

ax.boxplot([pfm22_readout['ed2']], positions=[0.5], patch_artist=True, widths = 0.2, boxprops=boxprops1, flierprops=flierprops1, 
                  meanprops=meanpointprops1, medianprops=medianprops, capprops = capprops1, whiskerprops = whiskerprops1, meanline=False, showmeans=True)
ax.boxplot([pfm22_readout['ed12']], positions=[1.5], patch_artist=True, widths = 0.2, boxprops=boxprops2, flierprops=flierprops2, 
                  meanprops=meanpointprops2, medianprops=medianprops, capprops = capprops2, whiskerprops = whiskerprops2, meanline=False, showmeans=True)



p1 = f_stats.permutation_p(pfm22_readout['ed2'].mean(), pfm22_readout_shuff['ed2'])
ax.text(0.5,0.7, f'{f_plotting.sig_marker(p1)}',horizontalalignment='center', fontsize=12)

p2 = f_stats.permutation_p(pfm22_readout['ed12'].mean(), pfm22_readout_shuff['ed12'])
ax.text(1.5,0.7, f'{f_plotting.sig_marker(p2)}',horizontalalignment='center', fontsize=12)


xlims = ax.get_xlim()
lineThresholds = np.arange(xlims[0],xlims[1]+0.01,0.01)
#plt.plot(lineThresholds, np.full_like(lineThresholds,0), 'k-.', alpha = 0.5)
ax.plot(lineThresholds, np.full_like(lineThresholds,0.33), 'k--', alpha = 0.5, linewidth=1)



# draw temporary red and blue lines and use them to create a legend
#ax.plot([], c='dimgrey', label='Full Space')
#ax.plot([], c='lightgrey', label='Readout Subspace')
#ax.legend(bbox_to_anchor=(1.0, 0.6))#loc = 'right',

ax.set_xticks([0.5,1.5],['R@R','R&U'], rotation=30)
ax.set_xlabel('Strategy', labelpad = 5, fontsize = 12)
#ax.set_ylabel('LDA Decodability', labelpad = 3, fontsize = 12)
ax.set_ylim(top=0.75)
ax.set_title('Readout Subspace', fontsize = 15, pad=10)


plt.suptitle('Distractor Information, ED2, RNNs', fontsize = 20, y=1.1)
plt.show()

fig.savefig(f'{save_path}/distractorInfo_rnns.tif', bbox_inches='tight')


# In[]
plot_samples = (0,1,)
checkpoints = [150, 550, 1150, 1450, 1850, 2350]#
checkpointsLabels = ['S1','ED1','LD1','S2','ED2','LD2'] #, 2800
avgInterval = {150:150, 550:250, 1150:250, 1450:150, 1850:250, 2350:250} #, 2800:200
nIters = 5
nPerms = 50
infoMethod='lda'


cosThetas_choice, cosPsis_choice = {},{}
cosThetas_nonchoice, cosPsis_nonchoice = {},{}

cosThetas_11, cosThetas_22, cosThetas_12 = {},{},{}
cosPsis_11, cosPsis_22, cosPsis_12 = {},{},{}

cosThetas_C1, cosPsis_C1 = {},{}
cosThetas_C2, cosPsis_C2 = {},{}


for nfi, kfi in enumerate(fitIntervals):
    print(kfi)
    
    label = f'fit interval:{kfi}'
    strategyLabel = 'Rehearse & Update' if kfi[-2:] == '12' else 'Retrieve at Recall'
    
    models_dict = modelDicts[kfi]
    
    evrsThreshold = 0.9#5 if kfi[-2:] == '12' else 0.9
    evrs[kfi] = []
    

    cosTheta_choice, cosPsi_choice = [],[]
    cosTheta_nonchoice, cosPsi_nonchoice = [],[]
    
    cosTheta_11, cosTheta_22, cosTheta_12 = [],[],[]
    cosPsi_11, cosPsi_22, cosPsi_12 = [],[],[]
    
    cosTheta_C1, cosPsi_C1 = [],[]
    cosTheta_C2, cosPsi_C2 = [],[]
    
    
    ii = 0
    
    for i in range(len(modelDicts[kfi])):
        print(i)
        modelD = models_dict[i]['rnn']
        acc_memo = f_evaluateRNN.evaluate_acc(modelD, test_X, test_label, toPrint=True, checkpointX = np.where(tRange <= fitIntervals[kfi][-1][-1])[0].max()) #
        
        #modelD.hidden_noise = 0.5
        
        if acc_memo >=90:
            
            ii+=1
            
            tplt = True if (ii in plot_samples) else False
            
            vecs, projs, projsAll, _, trialInfos, _, _, vecs_shuff, projs_shuff, projsAll_shuff, _, trialInfos_shuff, _ = f_evaluateRNN.generate_itemVectors(modelD, trialInfo, X_, Y0_, tRange, trialEvents, checkpoints, avgInterval, nIters = nIters, nPerms = nPerms, pca_tWins=((300,1300),(1600,2600),), dt=dt) #
            geoms_valid, geoms_shuff = (vecs, projs, projsAll, trialInfos), (vecs_shuff, projs_shuff, projsAll_shuff, trialInfos_shuff)
            
            vecs_C, projs_C, projsAll_C, trialInfos_C, data_3pc_C, _, evrs_C, vecs_C_shuff, projs_C_shuff, projsAll_C_shuff, trialInfos_C_shuff, data_3pc_C_shuff, _ = f_evaluateRNN.generate_choiceVectors(modelD, trialInfo, X_, Y0_, tRange, trialEvents, nIters = nIters, nPerms = nPerms, 
                                                                                                                                                                                              pca_tWins=((300,1300),(1600,2600),), dt=dt, toPlot = tplt, label=f'{strategyLabel},{i}')
            geomsC_valid, geomsC_shuff = (vecs_C, projs_C, projsAll_C, trialInfos_C, data_3pc_C), (vecs_C_shuff, projs_C_shuff, projsAll_C_shuff, trialInfos_C_shuff, data_3pc_C_shuff)
            
            #print(evrs_C.sum(1).mean())
            
            #if evrs_C.sum(1).mean() > evrsThreshold:
            
            #if i not in (21, 22): 
            cosThetas, cosPsis,_,_,_ = f_evaluateRNN.plot_planeAlignment(geoms_valid, geoms_shuff, checkpoints, checkpointsLabels, nIters = nIters, nPerms = nPerms, toPlot=False)
            cosThetas_C, cosPsis_C, _, _, _ = f_evaluateRNN.plot_choiceItemSpace(geoms_valid, geoms_shuff, checkpoints, checkpointsLabels, nIters = nIters, nPerms = nPerms, toPlot=False, label=f'{strategyLabel} ')
            cosThetas_CI, cosPsis_CI = f_evaluateRNN.plot_choiceVSItem(geoms_valid, geoms_shuff, geomsC_valid, geomsC_shuff, checkpoints, checkpointsLabels, nIters = nIters, nPerms = nPerms)
            
            evrs[kfi] += [evrs_C]
            
            cosTheta_choice += [cosThetas_C[0]]
            cosPsi_choice += [cosPsis_C[0]]
            #cosSimi_choice += [cosSimis[0]]
            #ai_choice += [ais[0]]
            
            cosTheta_nonchoice += [cosThetas_C[1]]
            cosPsi_nonchoice += [cosPsis_C[1]]
            #cosSimi_nonchoice += [cosSimis[1]]
            #ai_nonchoice += [ais[1]]
    
            cosTheta_11 += [cosThetas[0]]
            cosPsi_11 += [cosPsis[0]]
            
            cosTheta_12 += [cosThetas[1]]
            cosPsi_12 += [cosPsis[1]]
            
            cosTheta_22 += [cosThetas[2]]
            cosPsi_22 += [cosPsis[2]]
            
            cosTheta_C1 += [cosThetas_CI[0]]
            cosPsi_C1 += [cosPsis_CI[0]]
            cosTheta_C2 += [cosThetas_CI[1]]
            cosPsi_C2 += [cosPsis_CI[1]]
            
            
            del modelD
            torch.cuda.empty_cache()
    
            gc.collect()

    # store
    cosThetas_choice[kfi], cosPsis_choice[kfi] = cosTheta_choice,cosPsi_choice
    cosThetas_nonchoice[kfi], cosPsis_nonchoice[kfi] = cosTheta_nonchoice,cosPsi_nonchoice

    cosThetas_11[kfi], cosThetas_22[kfi], cosThetas_12[kfi] = cosTheta_11,cosTheta_22,cosTheta_12
    cosPsis_11[kfi], cosPsis_22[kfi], cosPsis_12[kfi] = cosPsi_11,cosPsi_22,cosPsi_12

    cosThetas_C1[kfi], cosPsis_C1[kfi] = cosTheta_C1,cosPsi_C1
    cosThetas_C2[kfi], cosPsis_C2[kfi] = cosTheta_C2,cosPsi_C2

    
    
    if len(cosTheta_11)>0:
        
        angleCheckPoints = np.linspace(0,np.pi,13).round(5)
        cmap = plt.get_cmap('coolwarm')
        norm = mcolors.BoundaryNorm(np.flip(np.cos(angleCheckPoints).round(5)),cmap.N, clip=True)
        
        for tt in ttypes:
            
            cosTheta_11T = np.array([cosTheta_11[i][tt].mean(0) for i in range(len(cosTheta_11))])
            cosTheta_22T = np.array([cosTheta_22[i][tt].mean(0) for i in range(len(cosTheta_22))])
            cosTheta_12T = np.array([cosTheta_12[i][tt].mean(0) for i in range(len(cosTheta_12))])
            
            cosPsi_11T = np.array([cosPsi_11[i][tt].mean(0) for i in range(len(cosPsi_11))])
            cosPsi_22T = np.array([cosPsi_22[i][tt].mean(0) for i in range(len(cosPsi_22))])
            cosPsi_12T = np.array([cosPsi_12[i][tt].mean(0) for i in range(len(cosPsi_12))])
            
            #colorT = 'b' if tt == 1 else 'm'
            condT = 'Retarget' if tt == 1 else 'Distraction'
            
            ################
            ### cosTheta ###
            ################
            
            ### cosTheta
            plt.figure(figsize=(16, 5), dpi=100)
            plt.subplot(1,3,1)
            ax = plt.gca()
            im = ax.imshow(cosTheta_11T.mean(axis=1).mean(axis=0), cmap=cmap, norm=norm, aspect='auto')
                        
            ax.set_title('Item1-Item1', fontsize = 15, pad = 15)
            ax.set_xticks([n for n in range(len(checkpoints))])
            ax.set_xticklabels(checkpointsLabels, fontsize = 10)
            ax.set_yticks([n for n in range(len(checkpoints))])
            ax.set_yticklabels(checkpointsLabels, fontsize = 10)
            ax.set_xlabel('Item 1', fontsize = 15)
            ax.set_ylabel('Item 1', fontsize = 15)
            
            
            plt.subplot(1,3,2)
            ax = plt.gca()
            im = ax.imshow(cosTheta_22T.mean(axis=1).mean(axis=0), cmap=cmap, norm=norm, aspect='auto')
                        
            ax.set_title('Item2-Item2', fontsize = 15, pad = 15)
            ax.set_xticks([n for n in range(len(checkpoints))])
            ax.set_xticklabels(checkpointsLabels, fontsize = 10)
            ax.set_yticks([n for n in range(len(checkpoints))])
            ax.set_yticklabels(checkpointsLabels, fontsize = 10)
            ax.set_xlabel('Item 2', fontsize = 15)
            ax.set_ylabel('Item 2', fontsize = 15)
            
            
            
            plt.subplot(1,3,3)
            ax = plt.gca()
            im = ax.imshow(cosTheta_12T.mean(axis=1).mean(axis=0), cmap=cmap, norm=norm, aspect='auto')
            
            ax.set_title('Item1-Item2', fontsize = 15, pad = 15)
            ax.set_xticks([n for n in range(len(checkpoints))])
            ax.set_xticklabels(checkpointsLabels, fontsize = 10)
            ax.set_yticks([n for n in range(len(checkpoints))])
            ax.set_yticklabels(checkpointsLabels, fontsize = 10)
            ax.set_xlabel('Item 2', fontsize = 15)
            ax.set_ylabel('Item 1', fontsize = 15)
            
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_ticklabels(np.flip(np.degrees(angleCheckPoints).round(1).astype(int)))
            cbar.set_label('cos(θ)', fontsize = 15, rotation = 270, labelpad=20)
            
            plt.subplots_adjust(top = 0.8)
            plt.suptitle(f'Principal Angle (θ), {condT}, {strategyLabel}', fontsize = 20, y=1)
            plt.show()
    
            ##############
            ### cosPsi ###
            ##############
        
            plt.figure(figsize=(16, 5), dpi=100)
            plt.subplot(1,3,1)
            ax = plt.gca()
            im = ax.imshow(cosPsi_11T.mean(axis=1).mean(axis=0), cmap=cmap, norm=norm, aspect='auto')
                                    
            ax.set_title('Item1-Item1', fontsize = 15, pad = 15)
            ax.set_xticks([n for n in range(len(checkpoints))])
            ax.set_xticklabels(checkpointsLabels, fontsize = 10)
            ax.set_yticks([n for n in range(len(checkpoints))])
            ax.set_yticklabels(checkpointsLabels, fontsize = 10)
            ax.set_xlabel('Item 1', fontsize = 15)
            ax.set_ylabel('Item 1', fontsize = 15)
            
            
            
            plt.subplot(1,3,2)
            ax = plt.gca()
            im = ax.imshow(cosPsi_22T.mean(axis=1).mean(axis=0), cmap=cmap, norm=norm, aspect='auto')
                                    
            ax.set_title('Item2-Item2', fontsize = 15, pad = 15)
            ax.set_xticks([n for n in range(len(checkpoints))])
            ax.set_xticklabels(checkpointsLabels, fontsize = 10)
            ax.set_yticks([n for n in range(len(checkpoints))])
            ax.set_yticklabels(checkpointsLabels, fontsize = 10)
            ax.set_xlabel('Item 2', fontsize = 15)
            ax.set_ylabel('Item 2', fontsize = 15)
            
            
            
            plt.subplot(1,3,3)
            ax = plt.gca()
            im = ax.imshow(cosPsi_12T.mean(axis=1).mean(axis=0), cmap=cmap, norm=norm, aspect='auto')
            
            ax.set_title('Item1-Item2', fontsize = 15, pad = 15)
            ax.set_xticks([n for n in range(len(checkpoints))])
            ax.set_xticklabels(checkpointsLabels, fontsize = 10)
            ax.set_yticks([n for n in range(len(checkpoints))])
            ax.set_yticklabels(checkpointsLabels, fontsize = 10)
            ax.set_xlabel('Item 2', fontsize = 15)
            ax.set_ylabel('Item 1', fontsize = 15)
            
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_ticklabels(np.flip(np.degrees(angleCheckPoints).round(1).astype(int)))
            cbar.set_label('cos(Ψ)', fontsize = 15, rotation = 270, labelpad=20)
            
            plt.subplots_adjust(top = 0.8)
            plt.suptitle(f'Representational Alignment (Ψ), {condT}, {strategyLabel}', fontsize = 20, y=1)
            #plt.suptitle(f'abs(cosPsi),  ttype={tt}', fontsize = 15, y=1)
            plt.show()
    
    
    # choice-Item 
    if len(cosTheta_choice)>0:
        
        angleCheckPoints = np.linspace(0,np.pi,7).round(5)
        
        ################
        ### cosTheta ###
        ################
        
        cosTheta_choiceT = np.array(cosTheta_choice).squeeze()
        cosTheta_nonchoiceT = np.array(cosTheta_nonchoice).squeeze()
        
        plt.figure(figsize=(8, 3), dpi=100)
        plt.subplot(1,2,1)
        ax = plt.gca()
        ax.errorbar(np.arange(0, len(checkpoints), 1), cosTheta_choiceT.mean(axis=1).mean(axis=0), yerr = cosTheta_choiceT.mean(axis=1).std(axis=0), marker = 'o', capsize=4)
    
        ax.set_title('Choice', pad = 10)
        ax.set_xticks([n for n in range(len(checkpoints))])
        ax.set_xticklabels(checkpointsLabels, fontsize = 6)
        ax.set_ylim((-1,1))
        ax.set_yticks(np.flip(np.cos(angleCheckPoints).round(5)), labels = np.flip(np.degrees(angleCheckPoints).round(1).astype(int)))
        ax.set_ylabel('cos(θ)',fontsize=15,rotation = 90)
        
        plt.subplot(1,2,2)
        ax = plt.gca()
        ax.errorbar(np.arange(0, len(checkpoints), 1), cosTheta_nonchoiceT.mean(axis=1).mean(axis=0), yerr = cosTheta_nonchoiceT.mean(axis=1).std(axis=0), marker = 'o', capsize=4)
    
        ax.set_title('Non-choice', pad = 10)
        ax.set_xticks([n for n in range(len(checkpoints))])
        ax.set_xticklabels(checkpointsLabels, fontsize = 6)
        ax.set_ylim((-1,1))
        ax.set_yticks(np.flip(np.cos(angleCheckPoints).round(5)), labels = np.flip(np.degrees(angleCheckPoints).round(1).astype(int)))
        ax.set_ylabel('cos(θ)',fontsize=15,rotation = 90)
        
        
        plt.subplots_adjust(top = 0.8)
        plt.suptitle(f'Principal Angle (θ), {strategyLabel}', fontsize = 15, y=1)
        plt.tight_layout()
        plt.show()
    
        ##############
        ### cosPsi ###
        ##############
    
        cosPsi_choiceT = np.array(cosPsi_choice).squeeze()
        cosPsi_nonchoiceT = np.array(cosPsi_nonchoice).squeeze()
        
        plt.figure(figsize=(8, 3), dpi=100)
        plt.subplot(1,2,1)
        ax = plt.gca()
        ax.errorbar(np.arange(0, len(checkpoints), 1), cosPsi_choiceT.mean(axis=1).mean(axis=0), yerr = cosPsi_choiceT.mean(axis=1).std(axis=0), marker = 'o', capsize=4)
        #ax.errorbar(np.arange(0, len(checkpoints), 1), np.abs(cosPsi_choice[region]).mean(axis=1).mean(axis=0), yerr = np.abs(cosPsi_choice[region]).mean(axis=1).std(axis=0), marker = 'o')
    
        ax.set_title('Choice', pad = 10)
        ax.set_xticks([n for n in range(len(checkpoints))])
        ax.set_xticklabels(checkpointsLabels, fontsize = 6)
        ax.set_ylim((-1,1))
        ax.set_yticks(np.flip(np.cos(angleCheckPoints).round(5)), labels = np.flip(np.degrees(angleCheckPoints).round(1).astype(int)))
        ax.set_ylabel('cos(Ψ)',fontsize=15,rotation = 90)
        
        plt.subplot(1,2,2)
        ax = plt.gca()
        ax.errorbar(np.arange(0, len(checkpoints), 1), cosPsi_nonchoiceT.mean(axis=1).mean(axis=0), yerr = cosPsi_nonchoiceT.mean(axis=1).std(axis=0), marker = 'o', capsize=4)
        #ax.errorbar(np.arange(0, len(checkpoints), 1), np.abs(cosPsi_choice[region]).mean(axis=1).mean(axis=0), yerr = np.abs(cosPsi_choice[region]).mean(axis=1).std(axis=0), marker = 'o')
    
        ax.set_title('Non-choice', pad = 10)
        ax.set_xticks([n for n in range(len(checkpoints))])
        ax.set_xticklabels(checkpointsLabels, fontsize = 6)
        ax.set_ylim((-1,1))
        ax.set_yticks(np.flip(np.cos(angleCheckPoints).round(5)), labels = np.flip(np.degrees(angleCheckPoints).round(1).astype(int)))
        ax.set_ylabel('cos(Ψ)',fontsize=15,rotation = 90)
        
        plt.subplots_adjust(top = 0.8)
        plt.suptitle(f'Representational Alignment (Ψ), {strategyLabel}', fontsize = 15, y=1)
        #plt.suptitle(f'abs(cosPsi), {region}, choice/nonchoice', fontsize = 15, y=1)
        plt.tight_layout()
        plt.show()
    
    
    
    # choice-Item 
    if len(cosTheta_C1)>0:
        
        angleCheckPoints = np.linspace(0,np.pi,7).round(5)
        
        ################
        ### cosTheta ###
        ################
        
        plt.figure(figsize=(8, 3), dpi=100)
        
        for tt in ttypes:
            
            condT = 'Retarget' if tt == 1 else 'Distraction'
            
            
            cosTheta_choice1T = np.array([cosTheta_C1[i][tt] for i in range(len(cosTheta_C1))]).squeeze()
            cosTheta_choice2T = np.array([cosTheta_C2[j][tt] for j in range(len(cosTheta_C2))]).squeeze()
            
            
            plt.subplot(1,2,tt)
            ax = plt.gca()
            ax.errorbar(np.arange(0, len(checkpoints), 1), cosTheta_choice1T.mean(axis=1).mean(axis=0), yerr = cosTheta_choice1T.mean(axis=1).std(axis=0), marker = 'o', color = 'b', label = 'Item1', capsize=4)
            ax.errorbar(np.arange(0, len(checkpoints), 1), cosTheta_choice2T.mean(axis=1).mean(axis=0), yerr = cosTheta_choice2T.mean(axis=1).std(axis=0), marker = 'o', color = 'm', label = 'Item2', capsize=4)
            
            ax.set_title(f'{condT}', pad = 10)
            ax.set_xticks([n for n in range(len(checkpoints))])
            ax.set_xticklabels(checkpointsLabels, fontsize = 6)
            ax.set_ylim((-1,1))
            ax.set_yticks(np.flip(np.cos(angleCheckPoints).round(5)), labels = np.flip(np.degrees(angleCheckPoints).round(1).astype(int)))
            ax.set_ylabel('cos(θ)',fontsize=15,rotation = 90)
            ax.legend(loc='lower right')
            
        plt.subplots_adjust(top = 0.8)
        plt.suptitle(f'Principal Angle (θ), {strategyLabel}', fontsize = 15, y=1)
        plt.tight_layout()
        plt.show()
        
        ##############
        ### cosPsi ###
        ##############
        
        plt.figure(figsize=(8, 3), dpi=100)
        
        for tt in ttypes:
            
            condT = 'Retarget' if tt == 1 else 'Distraction'

        
            cosPsi_choice1T = np.array([cosPsi_C1[i][tt] for i in range(len(cosPsi_C1))]).squeeze()
            cosPsi_choice2T = np.array([cosPsi_C2[j][tt] for j in range(len(cosPsi_C2))]).squeeze()
            
            
            plt.subplot(1,2,tt)
            ax = plt.gca()
            ax.errorbar(np.arange(0, len(checkpoints), 1), cosPsi_choice1T.mean(axis=1).mean(axis=0), yerr = cosPsi_choice1T.mean(axis=1).std(axis=0), marker = 'o', color = 'b', label = 'Item1', capsize=4)
            ax.errorbar(np.arange(0, len(checkpoints), 1), cosPsi_choice2T.mean(axis=1).mean(axis=0), yerr = cosPsi_choice2T.mean(axis=1).std(axis=0), marker = 'o', color = 'm', label = 'Item2', capsize=4)
        
            ax.set_title(f'{condT}', pad = 10)
            ax.set_xticks([n for n in range(len(checkpoints))])
            ax.set_xticklabels(checkpointsLabels, fontsize = 6)
            ax.set_ylim((-1,1))
            ax.set_yticks(np.flip(np.cos(angleCheckPoints).round(5)), labels = np.flip(np.degrees(angleCheckPoints).round(1).astype(int)))
            ax.set_ylabel('cos(Ψ)',fontsize=15,rotation = 90)
            
            ax.legend(loc=7)
            
        #plt.legend(bbox_to_anchor=(1, 0.5))
        plt.subplots_adjust(top = 0.8)
        plt.suptitle(f'Representational Alignment (Ψ), {strategyLabel}', fontsize = 15, y=1)
        #plt.suptitle(f'abs(cosPsi), {region}, choice/nonchoice', fontsize = 15, y=1)
        plt.tight_layout()
        plt.show()
    
    ########################
    # plot info in omega^2 #
    ########################

    decode_proj1_3d = {1:np.array([info3d_1[i][1].mean(0) for i in range(len(info3d_1))]),2:np.array([info3d_1[j][2].mean(0) for j in range(len(info3d_1))])}
    decode_proj2_3d = {1:np.array([info3d_2[i][1].mean(0) for i in range(len(info3d_2))]),2:np.array([info3d_2[j][2].mean(0) for j in range(len(info3d_2))])}
    
    if len(decode_proj1_3d[1])>0:
        plt.figure(figsize=(12, 4), dpi=100)
        for tt in ttypes:
    
            #colorT = 'b' if tt == 1 else 'm'
            condT = 'Retarget' if tt == 1 else 'Distraction'
    
            infoLabel = 'Accuracy' if infoMethod=='lda' else 'PEV'
            
            plt.subplot(1,2,tt)
            ax = plt.gca()
            ax.errorbar(np.arange(0, len(checkpoints), 1), decode_proj1_3d[tt].mean(axis=-1).mean(axis=0), yerr = decode_proj1_3d[tt].mean(axis=-1).std(axis=0), marker = 'o', color = 'b', label = 'Item1', capsize=4)
            #trans = ax.get_xaxis_transform()
            #for nc, cp in enumerate(checkpoints):
                #if 0.05 < pPerms_decode1_3d[nc] <= 0.1:
                #    ax.annotate('+', xy=(nc, 0), xycoords = trans, annotation_clip = False, color = 'b', ha = 'center', va = 'center')
                #el
                #if 0.01 < pPerms_decode1_3d[nc] <= 0.05:
                #    ax.annotate('*', xy=(nc, 0), xycoords = trans, annotation_clip = False, color = 'b', ha = 'center', va = 'center')
                #elif pPerms_decode1_3d[nc] <= 0.01:
                #    ax.annotate('**', xy=(nc, 0), xycoords = trans, annotation_clip = False, color = 'b', ha = 'center', va = 'center')
    
    
            ax.errorbar(np.arange(0, len(checkpoints), 1), decode_proj2_3d[tt].mean(axis=-1).mean(axis=0), yerr = decode_proj2_3d[tt].mean(axis=-1).std(axis=0), marker = 'o', color = 'm', label = 'Item2', capsize=4)
            #trans = ax.get_xaxis_transform()
            #for nc, cp in enumerate(checkpoints):
                #if 0.05 < pPerms_decode2_3d[nc] <= 0.1:
                #    ax.annotate('+', xy=(nc, 0.05), xycoords = trans, annotation_clip = False, color = 'm', ha = 'center', va = 'center')
                #el
                #if 0.01 < pPerms_decode2_3d[nc] <= 0.05:
                #    ax.annotate('*', xy=(nc, 0.05), xycoords = trans, annotation_clip = False, color = 'm', ha = 'center', va = 'center')
                #elif pPerms_decode2_3d[nc] <= 0.01:
                #    ax.annotate('**', xy=(nc, 0.05), xycoords = trans, annotation_clip = False, color = 'm', ha = 'center', va = 'center')
    
            ax.set_title(f'{condT}', fontsize = 15, pad = 20)
            ax.set_xticks([n for n in range(len(checkpoints))])
            ax.set_xticklabels(checkpointsLabels, fontsize = 10)
            ax.set_xlabel('Time', fontsize = 15)
            ax.set_ylim((-0.1,1))
            #ax.set_yticklabels(checkpoints, fontsize = 10)
            ax.set_ylabel(f'{infoLabel}', fontsize = 15)
            ax.legend(loc='lower right')
    
        plt.subplots_adjust(top = 0.8)
        plt.suptitle(f'Mean Information, {strategyLabel}', fontsize = 20, y=0.9)
        plt.tight_layout()
        plt.show()
    
    
    #############################
    # plot info on choice plane #
    #############################

    decode_projC1_3d = {1:np.array([info_C1[i][1].mean(0) for i in range(len(info_C1))]),2:np.array([info_C1[j][2].mean(0) for j in range(len(info_C1))])}
    decode_projC2_3d = {1:np.array([info_C2[i][1].mean(0) for i in range(len(info_C2))]),2:np.array([info_C2[j][2].mean(0) for j in range(len(info_C2))])}
    
    if len(decode_projC1_3d[1])>0:
        plt.figure(figsize=(12, 4), dpi=100)
        for tt in ttypes:
    
            #colorT = 'b' if tt == 1 else 'm'
            condT = 'Retarget' if tt == 1 else 'Distraction'
    
            infoLabel = 'Accuracy' if infoMethod=='lda' else 'PEV'
            
            plt.subplot(1,2,tt)
            ax = plt.gca()
            ax.plot(np.arange(0, decode_projC1_3d[tt].shape[1], 1), decode_projC1_3d[tt].mean(axis=-1).mean(axis=0), color = 'b', label = 'Item1')
            ax.plot(np.arange(0, decode_projC2_3d[tt].shape[1], 1), decode_projC2_3d[tt].mean(axis=-1).mean(axis=0), color = 'm', label = 'Item2')
            ax.fill_between(np.arange(0, decode_projC1_3d[tt].shape[1], 1), (decode_projC1_3d[tt].mean(axis=-1).mean(axis=0)-decode_projC1_3d[tt].mean(axis=-1).std(axis=0)), (decode_projC1_3d[tt].mean(axis=-1).mean(axis=0)+decode_projC1_3d[tt].mean(axis=-1).std(axis=0)), alpha = 0.1, color = 'b')
            ax.fill_between(np.arange(0, decode_projC2_3d[tt].shape[1], 1), (decode_projC2_3d[tt].mean(axis=-1).mean(axis=0)-decode_projC2_3d[tt].mean(axis=-1).std(axis=0)), (decode_projC2_3d[tt].mean(axis=-1).mean(axis=0)+decode_projC2_3d[tt].mean(axis=-1).std(axis=0)), alpha = 0.1, color = 'm')
            
            # event lines
            for i in [0, 300, 1300, 1600, 2600]:
                
                #ax.plot(tbins, np.full_like(tbins,list(tbins).index(i)), 'k-.', linewidth=4, alpha = 0.25)
                ax.plot(np.full_like(tbins,list(tbins).index(i)), tbins, 'k-.', linewidth=2, alpha = 0.25)
            
           #ax.set_title(f'{condT}, 3d', pad = 10)
            ax.set_title(f'{condT}', fontsize = 20, pad = 20)
            ax.set_xticks([list(tbins).index(i) for i in [0, 300, 1300, 1600, 2600]])
            ax.set_xticklabels([0, 300, 1300, 1600, 2600], fontsize = 10)
            ax.set_xlabel('Time', fontsize = 15)
            #ax.set_xlim((list(tbins).index(0),list(tbins).index(2600))) #(0,)
            ax.set_ylim((0,1.1))
            #ax.set_yticklabels(checkpoints, fontsize = 10)
            ax.set_ylabel(f'{infoLabel}', fontsize = 15)
            ax.legend(loc='upper right')
    
        plt.subplots_adjust(top = 0.8)
        plt.suptitle(f'Mean Information, {strategyLabel}', fontsize = 20, y=0.9)
        plt.tight_layout()
        plt.show()
        
    ########################################
    # plot info on choice plane cross temp #
    ########################################
    
    infoLabel = 'Accuracy' if infoMethod=='lda' else 'PEV'
    #plt.figure(figsize=(12, 8), dpi=100)
    plt.figure(figsize=(28, 24), dpi=100)
    
    for tt in ttypes:
        
        condT = 'Retarget' if tt == 1 else 'Distraction'
        h0 = 0 if infoMethod == 'omega2' else 1/len(locs)
        
        pfm1, pfm2 = decode_proj1_3d[region][tt].mean(axis=0), decode_proj2_3d[region][tt].mean(axis=0)
        
        
        vmax = 0.6 if region == 'dlpfc' else 0.8
        
        # item1
        plt.subplot(2,2,tt)
        ax = plt.gca()
        sns.heatmap(pd.DataFrame(pfm1, index=tbins,columns=tbins), cmap = 'magma', vmin = 0.25, vmax = vmax, ax = ax)#, vmax = 0.6, xticklabels = 100, yticklabels = 100, vmin = cbar_min, vmax = cbar_max
        
        #from scipy import ndimage
        smooth_scale = 10
        z = ndimage.zoom(pPerms_decode1_3d, smooth_scale)
        ax.contour(np.linspace(0, len(tbins), len(tbins) * smooth_scale),
                 np.linspace(0, len(tbins), len(tbins) * smooth_scale),
                  z, levels=([0.05]), colors='white', alpha = 1)
        
        ax.invert_yaxis()
        
        
        # event lines
        for i in [0, 300, 1300, 1600, 2600]:
            ax.plot(tbins, np.full_like(tbins,list(tbins).index(i)), 'w-.', linewidth=4)
            ax.plot(np.full_like(tbins,list(tbins).index(i)), tbins, 'w-.', linewidth=4)
        
        ax.set_xticks([list(tbins).index(i) for i in [0, 300, 1300, 1600, 2600]])
        ax.set_xticklabels([0, 300, 1300, 1600, 2600], rotation=0, fontsize = 20)
        ax.set_xlabel('Test Timebin (ms)', labelpad = 10, fontsize = 25)
        ax.set_yticks([list(tbins).index(i) for i in [0, 300, 1300, 1600, 2600]])
        ax.set_yticklabels([0, 300, 1300, 1600, 2600], fontsize = 20)
        ax.set_ylabel('Train Timebin (ms)', labelpad = 10, fontsize = 25)
        
        cbar = ax.collections[0].colorbar
        # here set the labelsize by 20
        cbar.ax.tick_params(labelsize=20)
        
        ax.set_title(f'{condT}, Item1', fontsize = 30, pad = 20)
        
        # item2
        plt.subplot(2,2,tt+2)
        ax = plt.gca()
        sns.heatmap(pd.DataFrame(pfm2, index=tbins,columns=tbins), cmap = 'magma', vmin = 0.25, vmax = vmax, ax = ax)#, vmax = 0.6, xticklabels = 100, yticklabels = 100, vmin = cbar_min, vmax = cbar_max
        
        #from scipy import ndimage
        smooth_scale = 10
        z = ndimage.zoom(pPerms_decode2_3d, smooth_scale)
        ax.contour(np.linspace(0, len(tbins), len(tbins) * smooth_scale),
                 np.linspace(0, len(tbins), len(tbins) * smooth_scale),
                  z, levels=([0.05]), colors='white', alpha = 1)
        
        ax.invert_yaxis()
        
        
        # event lines
        for i in [0, 300, 1300, 1600, 2600]:
            ax.plot(tbins, np.full_like(tbins,list(tbins).index(i)), 'w-.', linewidth=4)
            ax.plot(np.full_like(tbins,list(tbins).index(i)), tbins, 'w-.', linewidth=4)
        
        ax.set_xticks([list(tbins).index(i) for i in [0, 300, 1300, 1600, 2600]])
        ax.set_xticklabels([0, 300, 1300, 1600, 2600], rotation=0, fontsize = 20)
        ax.set_xlabel('Test Timebin (ms)', labelpad = 10, fontsize = 25)
        ax.set_yticks([list(tbins).index(i) for i in [0, 300, 1300, 1600, 2600]])
        ax.set_yticklabels([0, 300, 1300, 1600, 2600], fontsize = 20)
        ax.set_ylabel('Train Timebin (ms)', labelpad = 10, fontsize = 25)
        
        cbar = ax.collections[0].colorbar
        # here set the labelsize by 20
        cbar.ax.tick_params(labelsize=20)
        
        ax.set_title(f'{condT}, Item2', fontsize = 30, pad = 20)
    
    plt.tight_layout()
    plt.subplots_adjust(top = 0.95)
    plt.suptitle(f'{region.upper()}, Readout Subspace', fontsize = 35, y=1) #, Arti_Noise = {arti_noise_level}
    plt.show()

#%%

###########################
# Full Space Decodability #
###########################

# In[]

plot_samples = (0,1)

#performances1, performances2 = {},{}
performancesX1, performancesX2 = {},{}
performancesX1_shuff, performancesX2_shuff = {},{}

for nfi, kfi in enumerate(fitIntervals):
    print(kfi)
    models_dict = modelDicts[kfi]
    
    label = f'fit interval:{kfi}'
    strategyLabel = 'Rehearse & Update' if kfi[-2:] == '12' else 'Retrieve at Recall'
    
    #pfm = {'Retarget':[], 'Distractor':[]}
    
    pfmX1 = {'Retarget':[], 'Distractor':[]}
    pfmX2 = {'Retarget':[], 'Distractor':[]}
    
    pfmX1_shuff = {'Retarget':[], 'Distractor':[]}
    pfmX2_shuff = {'Retarget':[], 'Distractor':[]}
    
    #pfm1 = {'Retarget':[], 'Distractor':[]}
    #pfm2 = {'Retarget':[], 'Distractor':[]}
    
    ii=0
    
    for i in range(len(modelDicts[kfi])):
        modelD = models_dict[i]['rnn']
        acc_memo = f_evaluateRNN.evaluate_acc(modelD, test_X, test_label, toPrint=True, checkpointX = np.where(tRange <= fitIntervals[kfi][-1][-1])[0].max()) #
        
        if acc_memo >=90:
            ii+=1
            if (ii in plot_samples):
                f_evaluateRNN.plot_states(modelD, test_Info, test_X, tRange, trialEvents, dt=dt, locs = (0,1,2,3), ttypes = (1,2), lcX = np.arange(0,1,1), cues=False, cseq = None, label = strategyLabel, 
                                          withhidden=False, withannote=False, save_path=save_path, savefig=False)
                #f_evaluateRNN.plot_weights_mixed(modelD)
            
            if ii <=30:
                #pfmT,_ = f_evaluateRNN.plot_crossTemp_lda(modelD, trialInfo, X_, Y0_, tRange, trialEvents, nIters=5, dt=dt,bins=50, pca_tWins=((300,1300),(1600,2600),), label=f'{kfi} ', toPlot = False) # (100,800),
                pfmX12, pfmX12_shuff = f_evaluateRNN.plot_crossTemp_lda12(modelD, trialInfo, X_, Y0_, tRange, trialEvents, nIters=5, nPerms=20, dt=dt, bins=50, pca_tWins=((300,1300),(1600,2600),), 
                                                                          label=f'{strategyLabel} ', toPlot = False, permDummy=False) # (100,800),
                #pfmT12,_ = f_evaluateRNN.plot_withinTime_lda12(modelD, trialInfo, X_, Y0_, tRange, trialEvents, nIters=5, dt=dt,bins=50, pca_tWins=((300,1300),(1600,2600),), label=f'{strategyLabel} ', toPlot = False) # (100,800),
                
                #pfmT,_ = f_evaluateRNN.plot_lda(modelD, trialInfo, X_, Y0_, tRange, trialEvents, nIters=5, dt = dt,bins=50, targetItem='locX', pca_tWins=((800,1300),(2100,2600),), label=f'{kfi}, locX', toPlot = False) #(100,800),
                
                pfmX1['Retarget'] += [np.array(pfmX12[0]['Retarget']).mean(0)]
                pfmX1['Distractor'] += [np.array(pfmX12[0]['Distractor']).mean(0)]
                
                pfmX2['Retarget'] += [np.array(pfmX12[1]['Retarget']).mean(0)]
                pfmX2['Distractor'] += [np.array(pfmX12[1]['Distractor']).mean(0)]
                
                pfmX1_shuff['Retarget'] += [np.array(pfmX12_shuff[0]['Retarget']).mean(0)]
                pfmX1_shuff['Distractor'] += [np.array(pfmX12_shuff[0]['Distractor']).mean(0)]
                
                pfmX2_shuff['Retarget'] += [np.array(pfmX12_shuff[1]['Retarget']).mean(0)]
                pfmX2_shuff['Distractor'] += [np.array(pfmX12_shuff[1]['Distractor']).mean(0)]
                
                #pfm1['Retarget'] += [np.array(pfmT12[0]['Retarget']).mean(0)]
                #pfm1['Distractor'] += [np.array(pfmT12[0]['Distractor']).mean(0)]
                
                #pfm2['Retarget'] += [np.array(pfmT12[1]['Retarget']).mean(0)]
                #pfm2['Distractor'] += [np.array(pfmT12[1]['Distractor']).mean(0)]
                
                del modelD
                torch.cuda.empty_cache()
        
                gc.collect()
            
    #performances1[kfi] = pfm1
    #performances2[kfi] = pfm2
    performancesX1[kfi] = pfmX1
    performancesX2[kfi] = pfmX2
    performancesX1_shuff[kfi] = pfmX1_shuff
    performancesX2_shuff[kfi] = pfmX2_shuff





# In[]
#fitIntervals = {'ed2':((1600,2600),), 'go':((2600,3000),), }  # 
# intervals to fit the model outputs 
#fitIntervals = {'ed12':((300,1300),(1600,2600),), } #'s2':((1300,2600),), 'ld2':((2100,2600),), 
fitIntervals = {'ed12':((300,1300),(1600,2600),), 'ed2':((1600,2600),), 'ld12':((700,1300),(2100,2600),), 'ld2':((2100,2600),), 'go':((2600,3000),),} #'s2':((1300,2600),), 's12':((0,1300), (1300,2600),), 
# In[]
nModels = 30 # number of models to be trained per interval
withBsl = True if tRange.min() <0 else False
# In[]
modelDicts = {} # save trained models
expected1 = 1#0.5 #
expected2 = 1#0.5 #
for nfi, kfi in enumerate(fitIntervals):
    
    print(kfi)
    modelDicts[kfi] = {i:{} for i in range(nModels)}
    
    fi1 = fitIntervals[kfi][0]
    
    Y1_Bonds = fi1
    
    if len(fitIntervals[kfi]) == 1:
        Y1 = f_simulation.generate_Y(N_out, trialInfo.choice.values, Y1_Bonds[0], Y1_Bonds[1], dt, expected1)
    else:
        Y1 = f_simulation.generate_Y(N_out, trialInfo.loc1.values, Y1_Bonds[0], Y1_Bonds[1], dt, expected1)
        
    #Y1 = f_simulation.generate_Y_circular(N_out, trialInfo.loc1.values, Y1_Bonds[0], Y1_Bonds[1], dt, expected1, kappa)
    Y1_ = torch.tensor(Y1, dtype=torch.float32).to(device)
    
    if len(fitIntervals[kfi])>1:
        fi2 = fitIntervals[kfi][1]
        
        Y2_Bonds = fi2
        Y2 = f_simulation.generate_Y(N_out, trialInfo.choice.values, Y2_Bonds[0], Y2_Bonds[1], dt, expected2)
        #Y2 = f_simulation.generate_Y_circular(N_out, trialInfo.choice.values, Y2_Bonds[0], Y2_Bonds[1], dt, expected2, kappa)
        Y2_ = torch.tensor(Y2, dtype=torch.float32).to(device)
    
    Ys_ = ((Y0_,Y0_Bonds,1,), ) if withBsl else ()
    if len(fitIntervals[kfi])>1:
        Ys_ += ((Y1_,Y1_Bonds,1,),(Y2_,Y2_Bonds,1,),) 
    else: 
        Ys_ += ((Y1_,Y1_Bonds,1,),) #(Y0_,Y0_Bonds,1,), 
    
    for i in range(nModels):
        print(f'model n={i}')
        
        # if decay RNN
        modelD = myRNNs.decayRNN(N_in, N_hidden, N_out, dt = dt, tau = tau, external = ext, hidden_noise = hidden_noise, ceiling = ceiling, 
                                 F_hidden = F_hidden, F_out = F_out, device= device, init_hidden='orthogonal_', useLinear_hidden = False).to(device)
        losses = f_trainRNN.train_model(modelD, trialInfo, X_, Ys_, tRange, frac = 0.2, criterion = nn.MSELoss(), learning_rate = 0.001, n_iter = 1000*2, loss_cutoff = 0.0001, lr_cutoff = 1e-7)
        #losses = f_trainRNN.train_model_circular(modelD, trialInfo, X_, Ys_, tRange, frac = 0.2, learning_rate = 0.0001, n_iter = 1000*2, loss_cutoff = 0.001, lr_cutoff = 1e-7)
        
        modelDicts[kfi][i]['rnn'] = modelD
        modelDicts[kfi][i]['losses'] = losses
        
        _, _, _, test_setID, test_X, _ = f_simulation.split_dataset(X_, Ys_[0][0], frac = 0.8, ranseed=114514)
        test_label = trialInfo.loc[test_setID,'choice'].astype('int').values
        acc_memo = f_evaluateRNN.evaluate_acc(modelD, test_X, test_label, toPrint=True)
        
        del modelD
        torch.cuda.empty_cache()
        
        gc.collect()

    # save/load models
    #Ybounds = [''.join([str(int(j[1][0]/100)), str(int(j[1][1]/100))]) for j in Ys_[1:]]
    #modelName = taskVersion + '_' + '_'.join([str(i) for i in [N_in, N_hidden]]+Ybounds)#, tau, dt, int(ext), ceiling, F_hidden, F_out
# In[]
np.save(f'{save_path}/modelDicts8_basic_2_{tau}_{expected1}.npy', modelDicts, allow_pickle=True) #_{expected2}
modelDicts = np.load(f'{save_path}/modelDicts8_basic_2_{tau}_{expected1}.npy', allow_pickle=True).item()

# In[]
fitIntervals = {'s2':((300,1300),(1300,2600),), 'ed2':((300,1300),(1600,2600),), }  # 'end':((2990,3000),), 'go':((300,1300), (2600,3000),), 'ld2':((300,1300), (2100,2600),), 
# intervals to fit the model outputs 
#fitIntervals = {'s12':((0,1300), (1300,2600),), 'ed12':((300,1300),(1600,2600),), 'ld12':((700,1300),(2000,2600),), } #'s2':((1300,2600),), 'ed2':((1600,2600),), 'ld2':((2100,2600),), 
# In[]
nModels = 5 # number of models to be trained per interval
withBsl = True if tRange.min() <0 else False

# In[]
modelDicts = {} # save trained models
expected1 = 0.25#0.5 #
expected2 = 1#0.5 #
for nfi, kfi in enumerate(fitIntervals):
    
    print(kfi)
    modelDicts[kfi] = {i:{} for i in range(nModels)}
    
    fi1 = fitIntervals[kfi][0]
    
    Y1_Bonds = fi1
    
    if len(fitIntervals[kfi]) == 1:
        Y1 = f_simulation.generate_Y(N_out, trialInfo.choice.values, Y1_Bonds[0], Y1_Bonds[1], dt, expected1)
    else:
        Y1 = f_simulation.generate_Y(N_out, trialInfo.loc1.values, Y1_Bonds[0], Y1_Bonds[1], dt, expected1)
        
    #Y1 = f_simulation.generate_Y_circular(N_out, trialInfo.loc1.values, Y1_Bonds[0], Y1_Bonds[1], dt, expected1, kappa)
    Y1_ = torch.tensor(Y1, dtype=torch.float32).to(device)
    
    if len(fitIntervals[kfi])>1:
        fi2 = fitIntervals[kfi][1]
        
        Y2_Bonds = fi2
        Y2 = f_simulation.generate_Y(N_out, trialInfo.choice.values, Y2_Bonds[0], Y2_Bonds[1], dt, expected2)
        #Y2 = f_simulation.generate_Y_circular(N_out, trialInfo.choice.values, Y2_Bonds[0], Y2_Bonds[1], dt, expected2, kappa)
        Y2_ = torch.tensor(Y2, dtype=torch.float32).to(device)
    
    Ys_ = ((Y0_,Y0_Bonds,1,), ) if withBsl else ()
    if len(fitIntervals[kfi])>1:
        Ys_ += ((Y1_,Y1_Bonds,1,),(Y2_,Y2_Bonds,1,),) 
    else: 
        Ys_ += ((Y1_,Y1_Bonds,1,),) #(Y0_,Y0_Bonds,1,), 
        
    for i in range(nModels):
        print(f'model n={i}')
        
        # if decay RNN
        modelD = myRNNs.decayRNN(N_in, N_hidden, N_out, dt = dt, tau = tau, external = ext, hidden_noise = hidden_noise, ceiling = ceiling, 
                                 F_hidden = F_hidden, F_out = F_out, device= device, init_hidden='orthogonal_', useLinear_hidden = False, seed = i).to(device)
        losses = f_trainRNN.train_model(modelD, trialInfo, X_, Ys_, tRange, frac = 0.2, criterion = nn.MSELoss(), 
                                        learning_rate = 0.0001, n_iter = 1000*2, loss_cutoff = 0.001, lr_cutoff = 1e-7, l2reg=False)
        #losses = f_trainRNN.train_model_circular(modelD, trialInfo, X_, Ys_, tRange, frac = 0.2, learning_rate = 0.0001, n_iter = 1000*2, loss_cutoff = 0.001, lr_cutoff = 1e-7)
        
        modelDicts[kfi][i]['rnn'] = modelD
        modelDicts[kfi][i]['losses'] = losses
        
        _, _, _, test_setID, test_X, _ = f_simulation.split_dataset(X_, Ys_[0][0], frac = 0.8, ranseed=114514)
        test_label = trialInfo.loc[test_setID,'choice'].astype('int').values
        acc_memo = f_evaluateRNN.evaluate_acc(modelD, test_X, test_label, toPrint=True)
        
        del modelD
        torch.cuda.empty_cache()
        
        gc.collect()

    # save/load models
    #Ybounds = [''.join([str(int(j[1][0]/100)), str(int(j[1][1]/100))]) for j in Ys_[1:]]
    #modelName = taskVersion + '_' + '_'.join([str(i) for i in [N_in, N_hidden]]+Ybounds)#, tau, dt, int(ext), ceiling, F_hidden, F_out
# In[]
np.save(f'{save_path}/modelDicts8_basic_1_{tau}_{expected1}_rotate.npy', modelDicts, allow_pickle=True) #_{expected2}
modelDicts = np.load(f'{save_path}/modelDicts8_basic_1_{tau}_{expected1}.npy', allow_pickle=True).item()



# In[]

############################
# 8ch version multi output #
############################

# In[]
# task version 1: sequntial input, distractor & retarget, our version of task
taskVersion = 'seqSingle'
trialEvents = {'s1':[0,300],'d1':[300,1300],'s2':[1300,1600],'d2':[1600,2600],'go':[2600,3000]} # event markers # 'bsl':[-300,0], 
X = f_simulation.generate_X_8ch(trialInfo, trialEvents, tRange, dt, vloc=vloc, vttype=vttype, noise=noise, gocue=gocue)

X_ = torch.tensor(X, dtype=torch.float32).to(device)
# In[]
kappa = 3
# In[]
#N_in = len(locs) + len(ttypes) + int(gocue) # The number of network inputs.
#N_in = len(locs) * len(ttypes) + int(gocue) # The number of network inputs.
N_in = X.shape[-1]
N_out = len(locs)
N_hidden = 128

# In[] genereate expected output values at different time windows
# always force baseline output to stay at chance level
expected0_memo = 1/N_out
expected0_resp = 1/N_out

Y0_Bonds_memo, Y0_Bonds_resp = (-300,0), (-300,0)
Y0_memo = f_simulation.generate_Y(N_out, trialInfo.choice.values, Y0_Bonds_memo[0], Y0_Bonds_memo[1], dt, expected0_memo)
Y0_resp = f_simulation.generate_Y(N_out, trialInfo.choice.values, Y0_Bonds_resp[0], Y0_Bonds_resp[1], dt, expected0_resp)

Y0_memo_ = torch.tensor(Y0_memo, dtype=torch.float32).to(device)
Y0_resp_ = torch.tensor(Y0_resp, dtype=torch.float32).to(device)

# In[]
fitIntervals = {'ld12':((800,1300),(2100,2600),(2600,3000)),'ed12':((300,1300),(1600,2600),(2600,3000)),'s12':((0,1300),(1300,2600),(2600,3000))} # 'go':((2600,3000),(2600,3000)),intervals to fit the model outputs
nModels = 10 # number of models to be trained per interval

# In[]
modelDicts = {} # save trained models

for nfi, kfi in enumerate(fitIntervals):
    
    print(kfi)
    modelDicts[kfi] = {i:{} for i in range(nModels)}
    
    fi_memo1, fi_memo2, fi_resp = fitIntervals[kfi][0], fitIntervals[kfi][1], fitIntervals[kfi][2]
    
    # pre go cue
    Y1_Bonds_memo1, Y1_Bonds_memo2, Y1_Bonds_resp = fi_memo1, fi_memo2, fi_resp
    
    expected1_memo1 = 0.5 #if Y1_Bonds[1] <= trialEvents['go'][0] else 1/N_out
    expected1_memo2 = 0.5 #if Y1_Bonds[1] <= trialEvents['go'][0] else 1/N_out
    expected1_resp = 1 #if Y1_Bonds[0] >= trialEvents['d2'][1] else 1/N_out
    
    Y1_memo1 = f_simulation.generate_Y(N_out, trialInfo.loc1.values, Y1_Bonds_memo1[0], Y1_Bonds_memo1[1], dt, expected1_memo1)
    Y1_memo2 = f_simulation.generate_Y(N_out, trialInfo.choice.values, Y1_Bonds_memo2[0], Y1_Bonds_memo2[1], dt, expected1_memo2)
    Y1_resp = f_simulation.generate_Y(N_out, trialInfo.choice.values, Y1_Bonds_resp[0], Y1_Bonds_resp[1], dt, expected1_resp)
    
    Y1_memo1_ = torch.tensor(Y1_memo1, dtype=torch.float32).to(device)
    Y1_memo2_ = torch.tensor(Y1_memo2, dtype=torch.float32).to(device)
    
    Y1_resp_ = torch.tensor(Y1_resp, dtype=torch.float32).to(device)
    
    Ys_memo_ = ((Y1_memo1_,Y1_Bonds_memo1,2,),(Y1_memo2_,Y1_Bonds_memo2,2,),) #(Y0_memo_,Y0_Bonds_memo,1,),
    Ys_resp_ = ((Y1_resp_,Y1_Bonds_resp,1,),) #(Y0_resp_,Y0_Bonds_resp,1,),
    
    for i in range(nModels):
        print(f'model n={i}')
        
        # if decay RNN
        modelD = myRNNs.decayRNN(N_in, N_hidden, N_out*2, dt = dt, tau = tau, external = ext, hidden_noise = hidden_noise, ceiling = ceiling, 
                                 F_hidden = F_hidden, F_out = F_out, device= device, init_hidden='orthogonal_', useLinear_hidden = False).to(device)
        
        losses = f_trainRNN.train_model_multiOutput(modelD, trialInfo, X_, Ys_memo_, Ys_resp_, tRange, frac = 0.2, criterion = nn.MSELoss(), learning_rate = 0.0001, 
                                                    n_iter = 1000*3, loss_cutoff = 0.01, lr_cutoff = 1e-7)
        
        modelDicts[kfi][i]['rnn'] = modelD
        modelDicts[kfi][i]['losses'] = losses
        
        _, _, _, test_setID, test_X, _ = f_simulation.split_dataset(X_, Ys_memo_[0][0], frac = 0.1, ranseed=114514)
        test_label = trialInfo.loc[test_setID,'choice'].astype('int').values
        acc_memo, acc_resp = f_evaluateRNN.evaluate_acc_multi(modelD, test_X, test_label, checkpoint1X= 260, toPrint=True)
        
        del modelD
        torch.cuda.empty_cache()
        
        gc.collect()

# In[]
np.save('E:/NUS/PhD/fitting planes/pooled/' + 'modelDicts_multi8_2.npy', modelDicts, allow_pickle=True)
modelDicts = np.load('E:/NUS/PhD/fitting planes/pooled/' + 'modelDicts_multi8_2.npy', allow_pickle=True).item()
# In[]

_, _, _, test_setID, test_X, _ = f_simulation.split_dataset(X_, Y0_memo_, frac = 0.5, ranseed=114514)
test_Info = trialInfo.loc[test_setID,:].reset_index(drop = True)
test_label = trialInfo.loc[test_setID,'choice'].astype('int').values

# In[]
for nfi, kfi in enumerate(fitIntervals):
    print(kfi)
    models_dict = modelDicts[kfi]
    
    label = f'fit interval:{kfi},{i}'
    
    pfm = {'Retarget':[], 'Distractor':[]}
    #pfm_X = {'Retarget':[], 'Distractor':[]}
    
    for i in range(nModels):
        modelD = models_dict[i]['rnn']
        acc_memo, acc_resp = f_evaluateRNN.evaluate_acc_multi(modelD, test_X, test_label, toPrint=True, 
                                                              checkpoint1X = np.where(tRange <= fitIntervals[kfi][1][-1])[0].max(), checkpoint2X = np.where(tRange <= fitIntervals[kfi][2][-1])[0].max()) #
        
        if (acc_memo >=95) and (acc_resp >=95):
            #if i == 0:
                #f_evaluateRNN.plot_states(modelD, test_Info, test_X, tRange, trialEvents, locs = (0,1,2,3), ttypes = (1,2), lcX = np.arange(0,1,1), cues=False, cseq = None)
                #f_evaluateRNN.plot_weights_mixed(modelD)
    
            pfmT,_ = f_evaluateRNN.plot_lda(modelD, trialInfo, X_, Y0_memo_, tRange, trialEvents, nIters=5, dt=dt,bins=50, pca_tWins=((300,1300),(1600,2600),), label=f'{kfi} ', toPlot = False) # (100,800),
            #pfmT,_ = f_evaluateRNN.plot_lda(modelD, trialInfo, X_, Y0_, tRange, trialEvents, nIters=5, dt = dt,bins=50, targetItem='locX', pca_tWins=((800,1300),(2100,2600),), label=f'{kfi}, locX', toPlot = False) #(100,800),
            
            pfm['Retarget'] += [np.array(pfmT['Retarget']).mean(0)]
            pfm['Distractor'] += [np.array(pfmT['Distractor']).mean(0)]
            
            del modelD
            torch.cuda.empty_cache()
    
            gc.collect()
        
    
    # decodability with/without permutation P value
    bins = 50
    tslice = (tRange.min(), tRange.max())
    #tRange = np.arange(-300,3000,dt)
    
    tbins = np.arange(tslice[0], tslice[1], bins) #+dt
    
    conditions = (('ttype', 1), ('ttype', 2))
    
    for condition in conditions:
        ttypeT = 'Retarget' if condition[-1]==1 else 'Distractor'
        
        if len(pfm[ttypeT])>0:
            
            pfmT = np.array(pfm[ttypeT])
            
            vmax = 1
            
            plt.figure(figsize=(15,12), dpi = 100)
            ax = plt.gca()
            sns.heatmap(pd.DataFrame(pfmT.mean(axis = 0), index=tbins,columns=tbins), cmap = 'jet', vmin = 0.25, ax = ax, vmax = vmax)#, vmax = 0.6, xticklabels = 100, yticklabels = 100, vmin = cbar_min, vmax = cbar_max
            
            
            ax.invert_yaxis()
            
            evts = [i[0] for i in trialEvents.values()][1:]
            # event lines
            for i in evts:
                ax.plot(tbins, np.full_like(tbins,list(tbins).index(i)), 'w-.', linewidth=4)
                ax.plot(np.full_like(tbins,list(tbins).index(i)), tbins, 'w-.', linewidth=4)
            
            ax.set_xticks([list(tbins).index(i) for i in evts])
            ax.set_xticklabels(evts, rotation=0, fontsize = 20)
            ax.set_xlabel('Test Timebin (ms)', labelpad = 10, fontsize = 35)
            ax.set_yticks([list(tbins).index(i) for i in evts])
            ax.set_yticklabels(evts, fontsize = 20)
            ax.set_ylabel('Train Timebin (ms)', labelpad = 10, fontsize = 35)
            
            cbar = ax.collections[0].colorbar
            # here set the labelsize by 20
            cbar.ax.tick_params(labelsize=20)
            
            plt.title(f'{ttypeT}, {label}', pad = 10, fontsize = 25)
            plt.show()



# In[]
checkpoints = [150, 550, 1150, 1450, 1850, 2350, 2800] #
avgInterval = {150:150, 550:250, 1150:250, 1450:150, 1850:250, 2350:250, 2800:200} #
nIters = 5
nPerms = 50

for nfi, kfi in enumerate(fitIntervals):
    print(kfi)
    models_dict = modelDicts[kfi]

    info3d_1, info3d_2 = [],[]
    cosTheta_choice, cosPsi_choice, cosSimi_choice, ai_choice = [],[],[],[]
    cosTheta_nonchoice, cosPsi_nonchoice, cosSimi_nonchoice, ai_nonchoice = [],[],[],[]
    
    cosTheta_11, cosTheta_22, cosTheta_12 = [],[],[]
    cosPsi_11, cosPsi_22, cosPsi_12 = [],[],[]
    
    for i in range(nModels):
        modelD = models_dict[i]['rnn']
        acc_memo, acc_resp = f_evaluateRNN.evaluate_acc_multi(modelD, test_X, test_label, toPrint=True, 
                                                              checkpoint1X = np.where(tRange <= fitIntervals[kfi][1][-1])[0].max(), checkpoint2X = np.where(tRange <= fitIntervals[kfi][2][-1])[0].max()) #
        
        if (acc_memo >=95) and (acc_resp >=95):
            vecs, projs, projsAll, _, trialInfos, vecs_shuff, projs_shuff, projsAll_shuff, _, trialInfos_shuff = f_evaluateRNN.generate_itemVectors(modelD, trialInfo, X_, Y0_memo_, tRange, trialEvents, checkpoints, avgInterval, nIters = nIters, nPerms = nPerms, pca_tWins=((300,1300),(1600,2600),), dt=dt) #
            geoms_valid, geoms_shuff = (vecs, projs, projsAll, trialInfos), (vecs_shuff, projs_shuff, projsAll_shuff, trialInfos_shuff)
    
            cosThetas, cosPsis,_,_,_ = f_evaluateRNN.plot_planeAlignment(geoms_valid, geoms_shuff, checkpoints, nIters = nIters, nPerms = nPerms, toPlot=False)
            info3d,_ = f_evaluateRNN.plot_itemInfo(geoms_valid, geoms_shuff, checkpoints, nIters = nIters, nPerms = nPerms, toPlot=False, label=f'{kfi} ')
            cosThetas_C, cosPsis_C, _, _, _ = f_evaluateRNN.plot_choiceItemSpace(geoms_valid, geoms_shuff, checkpoints, nIters = nIters, nPerms = nPerms, toPlot=False, label=f'{kfi} ')
    
            info3d_1 += [info3d[0]]
            info3d_2 += [info3d[1]]
    
            cosTheta_choice += [cosThetas_C[0]]
            cosPsi_choice += [cosPsis_C[0]]
            #cosSimi_choice += [cosSimis[0]]
            #ai_choice += [ais[0]]
            
            cosTheta_nonchoice += [cosThetas_C[1]]
            cosPsi_nonchoice += [cosPsis_C[1]]
            #cosSimi_nonchoice += [cosSimis[1]]
            #ai_nonchoice += [ais[1]]
    
            cosTheta_11 += [cosThetas[0]]
            cosPsi_11 += [cosPsis[0]]
            
            cosTheta_12 += [cosThetas[1]]
            cosPsi_12 += [cosPsis[1]]
            
            cosTheta_22 += [cosThetas[2]]
            cosPsi_22 += [cosPsis[2]]
            
            del modelD
            torch.cuda.empty_cache()
    
            gc.collect()

    ########################
    # plot info in omega^2 #
    ########################

    decode_proj1_3d = {1:np.array([info3d_1[i][1].mean(0) for i in range(len(info3d_1))]),2:np.array([info3d_1[j][2].mean(0) for j in range(len(info3d_1))])}
    decode_proj2_3d = {1:np.array([info3d_2[i][1].mean(0) for i in range(len(info3d_2))]),2:np.array([info3d_2[j][2].mean(0) for j in range(len(info3d_2))])}
    
    if len(decode_proj1_3d[1])>0:
        plt.figure(figsize=(12, 4), dpi=100)
        for tt in ttypes:
    
            #colorT = 'b' if tt == 1 else 'm'
            condT = 'retarget' if tt == 1 else 'distractor'
    
    
            plt.subplot(1,2,tt)
            ax = plt.gca()
            ax.errorbar(np.arange(0, len(checkpoints), 1), decode_proj1_3d[tt].mean(axis=-1).mean(axis=0), yerr = decode_proj1_3d[tt].mean(axis=-1).std(axis=0), marker = 'o', color = 'b', label = 'loc1')
            #trans = ax.get_xaxis_transform()
            #for nc, cp in enumerate(checkpoints):
                #if 0.05 < pPerms_decode1_3d[nc] <= 0.1:
                #    ax.annotate('+', xy=(nc, 0), xycoords = trans, annotation_clip = False, color = 'b', ha = 'center', va = 'center')
                #el
                #if 0.01 < pPerms_decode1_3d[nc] <= 0.05:
                #    ax.annotate('*', xy=(nc, 0), xycoords = trans, annotation_clip = False, color = 'b', ha = 'center', va = 'center')
                #elif pPerms_decode1_3d[nc] <= 0.01:
                #    ax.annotate('**', xy=(nc, 0), xycoords = trans, annotation_clip = False, color = 'b', ha = 'center', va = 'center')
    
    
            ax.errorbar(np.arange(0, len(checkpoints), 1), decode_proj2_3d[tt].mean(axis=-1).mean(axis=0), yerr = decode_proj2_3d[tt].mean(axis=-1).std(axis=0), marker = 'o', color = 'm', label = 'loc2')
            #trans = ax.get_xaxis_transform()
            #for nc, cp in enumerate(checkpoints):
                #if 0.05 < pPerms_decode2_3d[nc] <= 0.1:
                #    ax.annotate('+', xy=(nc, 0.05), xycoords = trans, annotation_clip = False, color = 'm', ha = 'center', va = 'center')
                #el
                #if 0.01 < pPerms_decode2_3d[nc] <= 0.05:
                #    ax.annotate('*', xy=(nc, 0.05), xycoords = trans, annotation_clip = False, color = 'm', ha = 'center', va = 'center')
                #elif pPerms_decode2_3d[nc] <= 0.01:
                #    ax.annotate('**', xy=(nc, 0.05), xycoords = trans, annotation_clip = False, color = 'm', ha = 'center', va = 'center')
    
            ax.set_title(f'{condT}, 3d', pad = 10)
            ax.set_xticks([n for n in range(len(checkpoints))])
            ax.set_xticklabels(checkpoints, fontsize = 6)
            ax.set_ylim((-0.1,1))
            ax.legend()
    
        plt.subplots_adjust(top = 0.8)
        plt.suptitle(f'{kfi}, Mean Info', fontsize = 15, y=0.9)
        plt.tight_layout()
        plt.show()
    
    if len(cosTheta_11)>0:
        
        angleCheckPoints = np.linspace(0,np.pi,13).round(5)
        cmap = plt.get_cmap('coolwarm')
        norm = mcolors.BoundaryNorm(np.flip(np.cos(angleCheckPoints).round(5)),cmap.N, clip=True)
        
        for tt in ttypes:
            
            cosTheta_11T = np.array([cosTheta_11[i][tt].mean(0) for i in range(len(cosTheta_11))])
            cosTheta_22T = np.array([cosTheta_22[i][tt].mean(0) for i in range(len(cosTheta_22))])
            cosTheta_12T = np.array([cosTheta_12[i][tt].mean(0) for i in range(len(cosTheta_12))])
            
            cosPsi_11T = np.array([cosPsi_11[i][tt].mean(0) for i in range(len(cosPsi_11))])
            cosPsi_22T = np.array([cosPsi_22[i][tt].mean(0) for i in range(len(cosPsi_22))])
            cosPsi_12T = np.array([cosPsi_12[i][tt].mean(0) for i in range(len(cosPsi_12))])
            
            #colorT = 'b' if tt == 1 else 'm'
            condT = 'retarget' if tt == 1 else 'distractor'
            
            ################
            ### cosTheta ###
            ################
            
            ### cosTheta
            plt.figure(figsize=(15, 5), dpi=100)
            plt.subplot(1,3,1)
            ax = plt.gca()
            im = ax.imshow(cosTheta_11T.mean(axis=1).mean(axis=0), cmap=cmap, norm=norm, aspect='auto')
                        
            ax.set_title('11', pad = 10)
            ax.set_xticks([n for n in range(len(checkpoints))])
            ax.set_xticklabels(checkpoints, fontsize = 6)
            ax.set_yticks([n for n in range(len(checkpoints))])
            ax.set_yticklabels(checkpoints, fontsize = 6)
            ax.set_xlabel('Item 1', fontsize = 10)
            ax.set_ylabel('Item 1', fontsize = 10)
            
            
            plt.subplot(1,3,2)
            ax = plt.gca()
            im = ax.imshow(cosTheta_22T.mean(axis=1).mean(axis=0), cmap=cmap, norm=norm, aspect='auto')
                        
            ax.set_title('22', pad = 10)
            ax.set_xticks([n for n in range(len(checkpoints))])
            ax.set_xticklabels(checkpoints, fontsize = 6)
            ax.set_yticks([n for n in range(len(checkpoints))])
            ax.set_yticklabels(checkpoints, fontsize = 6)
            ax.set_xlabel('Item 2', fontsize = 10)
            ax.set_ylabel('Item 2', fontsize = 10)
            
            
            
            plt.subplot(1,3,3)
            ax = plt.gca()
            im = ax.imshow(cosTheta_12T.mean(axis=1).mean(axis=0), cmap=cmap, norm=norm, aspect='auto')
            
            ax.set_title('12', pad = 10)
            ax.set_xticks([n for n in range(len(checkpoints))])
            ax.set_xticklabels(checkpoints, fontsize = 6)
            ax.set_yticks([n for n in range(len(checkpoints))])
            ax.set_yticklabels(checkpoints, fontsize = 6)
            ax.set_xlabel('Item 2', fontsize = 10)
            ax.set_ylabel('Item 1', fontsize = 10)
            
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_ticklabels(np.flip(np.degrees(angleCheckPoints).round(1).astype(int)))
            cbar.set_label('cos(x)', fontsize = 15, rotation = 270, labelpad=20)
            
            plt.subplots_adjust(top = 0.8)
            plt.suptitle(f'cosTheta, ttype={tt}, {kfi}', fontsize = 15, y=1)
            plt.show()
    
            ##############
            ### cosPsi ###
            ##############
        
            plt.figure(figsize=(15, 5), dpi=100)
            plt.subplot(1,3,1)
            ax = plt.gca()
            im = ax.imshow(cosPsi_11T.mean(axis=1).mean(axis=0), cmap=cmap, norm=norm, aspect='auto')
                                    
            ax.set_title('11', pad = 10)
            ax.set_xticks([n for n in range(len(checkpoints))])
            ax.set_xticklabels(checkpoints, fontsize = 6)
            ax.set_yticks([n for n in range(len(checkpoints))])
            ax.set_yticklabels(checkpoints, fontsize = 6)
            ax.set_xlabel('Item 1', fontsize = 10)
            ax.set_ylabel('Item 1', fontsize = 10)
            
            
            
            plt.subplot(1,3,2)
            ax = plt.gca()
            im = ax.imshow(cosPsi_22T.mean(axis=1).mean(axis=0), cmap=cmap, norm=norm, aspect='auto')
                                    
            ax.set_title('22', pad = 10)
            ax.set_xticks([n for n in range(len(checkpoints))])
            ax.set_xticklabels(checkpoints, fontsize = 6)
            ax.set_yticks([n for n in range(len(checkpoints))])
            ax.set_yticklabels(checkpoints, fontsize = 6)
            ax.set_xlabel('Item 2', fontsize = 10)
            ax.set_ylabel('Item 2', fontsize = 10)
            
            
            
            plt.subplot(1,3,3)
            ax = plt.gca()
            im = ax.imshow(cosPsi_12T.mean(axis=1).mean(axis=0), cmap=cmap, norm=norm, aspect='auto')
            
            ax.set_title('12', pad = 10)
            ax.set_xticks([n for n in range(len(checkpoints))])
            ax.set_xticklabels(checkpoints, fontsize = 6)
            ax.set_yticks([n for n in range(len(checkpoints))])
            ax.set_yticklabels(checkpoints, fontsize = 6)
            ax.set_xlabel('Item 2', fontsize = 10)
            ax.set_ylabel('Item 1', fontsize = 10)
            
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_ticklabels(np.flip(np.degrees(angleCheckPoints).round(1).astype(int)))
            cbar.set_label('cos(x)', fontsize = 15, rotation = 270, labelpad=20)
            
            plt.subplots_adjust(top = 0.8)
            plt.suptitle(f'cosPsi, ttype={tt}, {kfi}', fontsize = 15, y=1)
            #plt.suptitle(f'abs(cosPsi),  ttype={tt}', fontsize = 15, y=1)
            plt.show()
    
    
    if len(cosTheta_choice)>0:
        
        angleCheckPoints = np.linspace(0,np.pi,13).round(5)
        
        ################
        ### cosTheta ###
        ################
        
        cosTheta_choiceT = np.array(cosTheta_choice).squeeze()
        cosTheta_nonchoiceT = np.array(cosTheta_nonchoice).squeeze()
        
        plt.figure(figsize=(8, 3), dpi=100)
        plt.subplot(1,2,1)
        ax = plt.gca()
        ax.errorbar(np.arange(0, len(checkpoints), 1), cosTheta_choiceT.mean(axis=1).mean(axis=0), yerr = cosTheta_choiceT.mean(axis=1).std(axis=0), marker = 'o')
    
        ax.set_title('choice', pad = 10)
        ax.set_xticks([n for n in range(len(checkpoints))])
        ax.set_xticklabels(checkpoints, fontsize = 6)
        ax.set_ylim((-1,1))
        ax.set_yticks(np.flip(np.cos(angleCheckPoints).round(5)), labels = np.flip(np.degrees(angleCheckPoints).round(1).astype(int)))
        ax.set_ylabel('cos(x)',fontsize=15,rotation = 90)
        
        plt.subplot(1,2,2)
        ax = plt.gca()
        ax.errorbar(np.arange(0, len(checkpoints), 1), cosTheta_nonchoiceT.mean(axis=1).mean(axis=0), yerr = cosTheta_nonchoiceT.mean(axis=1).std(axis=0), marker = 'o')
    
        ax.set_title('nonchoice', pad = 10)
        ax.set_xticks([n for n in range(len(checkpoints))])
        ax.set_xticklabels(checkpoints, fontsize = 6)
        ax.set_ylim((-1,1))
        ax.set_yticks(np.flip(np.cos(angleCheckPoints).round(5)), labels = np.flip(np.degrees(angleCheckPoints).round(1).astype(int)))
        ax.set_ylabel('cos(x)',fontsize=15,rotation = 90)
        
        
        plt.subplots_adjust(top = 0.8)
        plt.suptitle(f'{kfi}, cosTheta, choice/nonchoice', fontsize = 15, y=1)
        plt.tight_layout()
        plt.show()
    
        ##############
        ### cosPsi ###
        ##############
    
        cosPsi_choiceT = np.array(cosPsi_choice).squeeze()
        cosPsi_nonchoiceT = np.array(cosPsi_nonchoice).squeeze()
        
        plt.figure(figsize=(8, 3), dpi=100)
        plt.subplot(1,2,1)
        ax = plt.gca()
        ax.errorbar(np.arange(0, len(checkpoints), 1), cosPsi_choiceT.mean(axis=1).mean(axis=0), yerr = cosPsi_choiceT.mean(axis=1).std(axis=0), marker = 'o')
        #ax.errorbar(np.arange(0, len(checkpoints), 1), np.abs(cosPsi_choice[region]).mean(axis=1).mean(axis=0), yerr = np.abs(cosPsi_choice[region]).mean(axis=1).std(axis=0), marker = 'o')
    
        ax.set_title('choice', pad = 10)
        ax.set_xticks([n for n in range(len(checkpoints))])
        ax.set_xticklabels(checkpoints, fontsize = 6)
        ax.set_ylim((-1,1))
        ax.set_yticks(np.flip(np.cos(angleCheckPoints).round(5)), labels = np.flip(np.degrees(angleCheckPoints).round(1).astype(int)))
        ax.set_ylabel('cos(x)',fontsize=15,rotation = 90)
        
        plt.subplot(1,2,2)
        ax = plt.gca()
        ax.errorbar(np.arange(0, len(checkpoints), 1), cosPsi_nonchoiceT.mean(axis=1).mean(axis=0), yerr = cosPsi_nonchoiceT.mean(axis=1).std(axis=0), marker = 'o')
        #ax.errorbar(np.arange(0, len(checkpoints), 1), np.abs(cosPsi_choice[region]).mean(axis=1).mean(axis=0), yerr = np.abs(cosPsi_choice[region]).mean(axis=1).std(axis=0), marker = 'o')
    
        ax.set_title('nonchoice', pad = 10)
        ax.set_xticks([n for n in range(len(checkpoints))])
        ax.set_xticklabels(checkpoints, fontsize = 6)
        ax.set_ylim((-1,1))
        ax.set_yticks(np.flip(np.cos(angleCheckPoints).round(5)), labels = np.flip(np.degrees(angleCheckPoints).round(1).astype(int)))
        ax.set_ylabel('cos(x)',fontsize=15,rotation = 90)
        
        plt.subplots_adjust(top = 0.8)
        plt.suptitle(f'{kfi}, cosPsi, choice/nonchoice', fontsize = 15, y=1)
        #plt.suptitle(f'abs(cosPsi), {region}, choice/nonchoice', fontsize = 15, y=1)
        plt.tight_layout()
        plt.show()
    



























# In[]
for nfi, kfi in enumerate(fitIntervals):
    print(kfi)
    models_dict = modelDicts[kfi]

    ckp = np.where((tRange<=fitIntervals[kfi][-1][-1]))[0][0].max()

    for i in range(nModels):
        modelD = models_dict[i]['rnn']
        
        acc_memo = f_evaluateRNN.evaluate_acc(modelD, test_X, test_label, toPrint=True, checkpointX = np.where(tRange <= fitIntervals[kfi][-1][-1])[0].max()) #

        del modelD
        torch.cuda.empty_cache()

        gc.collect()
        

    
        

































































# In[]

########################
# 8ch version 1 window #
########################

# In[]
# task version 1: sequntial input, distractor & retarget, our version of task
taskVersion = 'seqSingle'
trialEvents = {'s1':[0,300],'d1':[300,1300],'s2':[1300,1600],'d2':[1600,2600],'go':[2600,3000]} # event markers # 'bsl':[-300,0], 
X = f_simulation.generate_X_8ch(trialInfo, trialEvents, tRange, dt, vloc=vloc, vttype=vttype, noise=noise, gocue=gocue)

X_ = torch.tensor(X, dtype=torch.float32).to(device)

# In[]
# task version 2: sequentially displayed, 4 loc channels, cued recall
#taskVersion = 'seqMulti'
#trialEvents = {'bsl':[-300,0], 's1':[0,300],'d1':[300,1300],'s2':[1300,1600],'d2':[1600,2600],'cue':[2600,2900], 'd3':[2900,3900],'go':[3900,4500]} # event markers
#X = f_simulation.generate_X_6ch_seqMulti(trialInfo, trialEvents, tRange, dt, vloc=vloc, vttype=vttype, noise=noise, gocue=gocue)
#X_ = torch.tensor(X, dtype=torch.float32).to(device)

# In[]
# task version 3: simultaneous displayed, 8 loc channels, cued selection
#taskVersion = 'simMulti'
#trialEvents = {'bsl':[-300,0], 's1':[0,500],'d1':[500,1500],'cue':[1500,1800],'d2':[1800,2800],'go':[2800,3300]} # event markers
#X = f_simulation.generate_X_8ch_simMulti(trialInfo, trialEvents, tRange, dt, vloc=vloc, vttype=vttype, noise=noise, gocue=gocue)
#X_ = torch.tensor(X, dtype=torch.float32).to(device)

# In[]
kappa = 3
# In[]
#N_in = len(locs) + len(ttypes) + int(gocue) # The number of network inputs.
#N_in = len(locs) * len(ttypes) + int(gocue) # The number of network inputs.
N_in = X.shape[-1]
N_out = len(locs)
N_hidden = 128

# In[] genereate expected output values at different time windows
# always force baseline output to stay at chance level
expected0 = 1/N_out
Y0_Bonds = [-300,0]
Y0 = f_simulation.generate_Y(N_out, trialInfo.choice.values, Y0_Bonds[0], Y0_Bonds[1], dt, expected0)

Y0_ = torch.tensor(Y0, dtype=torch.float32).to(device)

# In[]
fitIntervals = {'end':((2990,3000),),'ld2':((2000,2600),),'ed2':((1600,2600),),} # intervals to fit the model outputs # 's12':((0,1300),(1300,2600),)
# In[]
nModels = 20 # number of models to be trained per interval

# In[]
modelDicts = {} # save trained models
expected1 = 1
for nfi, kfi in enumerate(fitIntervals):
    
    print(kfi)
    modelDicts[kfi] = {i:{} for i in range(nModels)}
    
    fi1 = fitIntervals[kfi][0]
    
    Y1_Bonds = fi1
    Y1 = f_simulation.generate_Y(N_out, trialInfo.choice.values, Y1_Bonds[0], Y1_Bonds[1], dt, expected1)
    #Y1 = f_simulation.generate_Y_circular(N_out, trialInfo.choice.values, Y1_Bonds[0], Y1_Bonds[1], dt, expected1, kappa)
    Y1_ = torch.tensor(Y1, dtype=torch.float32).to(device)
    
    
    Ys_ = ((Y1_,Y1_Bonds,1,),)#(Y0_,Y0_Bonds,1,),(Y2_,Y2_Bonds,1,), if len(fitIntervals[kfi])>1 else ((Y1_,Y1_Bonds,1,),) 
    
    for i in range(nModels):
        print(f'model n={i}')
        
        # if decay RNN
        modelD = myRNNs.decayRNN(N_in, N_hidden, N_out, dt = dt, tau = tau, external = ext, hidden_noise = hidden_noise, ceiling = ceiling, 
                                 F_hidden = F_hidden, F_out = F_out, device= device, init_hidden='orthogonal_', useLinear_hidden = False).to(device)
        losses = f_trainRNN.train_model(modelD, trialInfo, X_, Ys_, tRange, frac = 0.2, criterion = nn.MSELoss(), learning_rate = 0.0001, n_iter = 1000*2, loss_cutoff = 0.001, lr_cutoff = 1e-7)
        #losses = f_trainRNN.train_model_tbt(modelD, trialInfo, X_, Ys_, tRange, frac = 0.2, criterion = nn.MSELoss(), learning_rate = 0.0001, n_iter = 1000*2, loss_cutoff = 0.001, lr_cutoff = 1e-7)
        
        modelDicts[kfi][i]['rnn'] = modelD
        modelDicts[kfi][i]['losses'] = losses
        
        _, _, _, test_setID, test_X, _ = f_simulation.split_dataset(X_, Ys_[0][0], frac = 0.8, ranseed=114514)
        test_label = trialInfo.loc[test_setID,'choice'].astype('int').values
        acc_memo = f_evaluateRNN.evaluate_acc(modelD, test_X, test_label, toPrint=True)
        
        del modelD
        torch.cuda.empty_cache()
        
        gc.collect()

    # save/load models
    #Ybounds = [''.join([str(int(j[1][0]/100)), str(int(j[1][1]/100))]) for j in Ys_[1:]]
    #modelName = taskVersion + '_' + '_'.join([str(i) for i in [N_in, N_hidden]]+Ybounds)#, tau, dt, int(ext), ceiling, F_hidden, F_out
# In[]
np.save('E:/NUS/PhD/fitting planes/pooled/' + 'modelDicts8_basic_1.npy', modelDicts, allow_pickle=True)
modelDicts = np.load('E:/NUS/PhD/fitting planes/pooled/' + 'modelDicts8_basic_1.npy', allow_pickle=True).item()


























# In[]
checkpoints = [150, 550, 1150, 1450, 1850, 2350, 2800]
avgInterval = {150:150, 550:250, 1150:250, 1450:150, 1850:250, 2350:250, 2800:200}

pca_tWins=((800,1300),(2100,2600))

for kfi, _ in fitIntervals.items():
    for i in range(10):
        modelD = modelDicts[kfi][i]['rnn']
        label = f'fit interval:{kfi},{i}'
        
        #test_label = trialInfo.loc[test_setID,'choice'].astype('int').values
        acc_memo = f_evaluateRNN.evaluate_acc(modelD, test_X, test_label, toPrint=True, checkpointX = 290)
        
        #f_evaluateRNN.plot_states(modelD, test_Info, test_X, tRange, trialEvents, locs = (0,1,2,3), ttypes = (1,2), lcX = np.arange(0,1,1), cues=False, cseq = None, label = label)
        #f_evaluateRNN.plot_weights(modelD, label = label)
        
        #_,_ = f_evaluateRNN.plot_lda(modelD, trialInfo, X_, Y0_, tRange, trialEvents, nIters=10, label = label)
        #_,_ = f_evaluateRNN.plot_lda(modelD, trialInfo, X_, Y0_, tRange, trialEvents, nIters=10, label = label, targetItem='locX')
        
        vecs, projs, projsAll, _, trialInfos, vecs_shuff, projs_shuff, projsAll_shuff, _, trialInfos_shuff = f_evaluateRNN.generate_vectors(modelD, trialInfo, X_, Y0_, tRange, trialEvents, checkpoints, avgInterval, nIters = 20, nPerms = 50, pca_tWins=pca_tWins)
        geoms_valid, geoms_shuff = (vecs, projs, projsAll, trialInfos), (vecs_shuff, projs_shuff, projsAll_shuff, trialInfos_shuff)
        
        _,_,_,_,_ = f_evaluateRNN.plot_planeAlignment(geoms_valid, geoms_shuff, checkpoints, label = label)
        _,_ = f_evaluateRNN.plot_itemInfo(geoms_valid, geoms_shuff, checkpoints, label = label)
        _,_,_,_,_ = f_evaluateRNN.plot_choiceSpace(geoms_valid, geoms_shuff, checkpoints, label = label)

    
    













































# In[]


###############
# 4ch version #
###############


# In[]
# task version 1: sequntial input, distractor & retarget, our version of task
taskVersion = 'seqSingle'
trialEvents = {'s1':[0,300],'d1':[300,1300],'s2':[1300,1600],'d2':[1600,2600],'go':[2600,3000]} # event markers #'bsl':[-300,0], 
X = f_simulation.generate_X_4ch(trialInfo, trialEvents, tRange, dt, vtgt=vtgt, vdis=vdis, noise=noise, gocue=gocue)

X_ = torch.tensor(X, dtype=torch.float32).to(device)

# In[]
kappa = 3

# In[]
#N_in = len(locs) + len(ttypes) + int(gocue) # The number of network inputs.
#N_in = len(locs) * len(ttypes) + int(gocue) # The number of network inputs.
N_in = X.shape[-1]
N_out = len(locs)
N_hidden = 128

# In[] genereate expected output values at different time windows
# always force baseline output to stay at chance level
expected0 = 1/N_out
Y0_Bonds = [-300,0]
Y0 = f_simulation.generate_Y(N_out, trialInfo.choice.values, Y0_Bonds[0], Y0_Bonds[1], dt, expected0)

Y0_ = torch.tensor(Y0, dtype=torch.float32).to(device)

# In[]
fitIntervals = {'ld12':((800,1300),(2100,2600),),'ed12':((300,1300),(1600,2600),),} # intervals to fit the model outputs # 'go':((),(2600,3000),),'s12':((0,1300),(1300,2600),)
nModels = 5 # number of models to be trained per interval

# In[]
modelDicts = {} # save trained models
expected1 = 1
expected2 = 1
for nfi, kfi in enumerate(fitIntervals):
    
    print(kfi)
    
    fi1 = fitIntervals[kfi][0]
    fi2 = fitIntervals[kfi][1]
    modelDicts[kfi] = {i:{} for i in range(nModels)}
    
    Y1_Bonds = fi1
    #Y1 = generate_Y(N_out, trialInfo.choice.values, Y1_Bonds[0], Y1_Bonds[1], dt, expected1)
    Y1 = f_simulation.generate_Y_circular(N_out, trialInfo.loc1.values, Y1_Bonds[0], Y1_Bonds[1], dt, expected1, kappa)
    Y1_ = torch.tensor(Y1, dtype=torch.float32).to(device)
    
    Y2_Bonds = fi2
    #Y1 = generate_Y(N_out, trialInfo.choice.values, Y1_Bonds[0], Y1_Bonds[1], dt, expected1)
    Y2 = f_simulation.generate_Y_circular(N_out, trialInfo.choice.values, Y2_Bonds[0], Y2_Bonds[1], dt, expected2, kappa)
    Y2_ = torch.tensor(Y2, dtype=torch.float32).to(device)
    
    Ys_ = ((Y1_,Y1_Bonds,1,),(Y2_,Y2_Bonds,1,),) #(Y0_,Y0_Bonds,1,),
    
    for i in range(nModels):
        print(f'model n={i}')
        
        # if decay RNN
        modelD = myRNNs.decayRNN(N_in, N_hidden, N_out, dt = dt, tau = tau, external = ext, hidden_noise = hidden_noise, ceiling = ceiling, F_hidden = F_hidden, F_out = F_out, init_hidden='orthogonal_', useLinear_hidden = False).to(device)
        losses = f_trainRNN.train_model(modelD, trialInfo, X_, Ys_, tRange, frac = 0.2, criterion = nn.MSELoss(), learning_rate = 0.0001, n_iter = 1000*2, loss_cutoff = 0.001, lr_cutoff = 1e-7)
        
        modelDicts[kfi][i]['rnn'] = modelD
        modelDicts[kfi][i]['losses'] = losses
        
        _, _, _, test_setID, test_X, _ = f_simulation.split_dataset(X_, Ys_[0][0], frac = 0.1, ranseed=114514)
        test_label = trialInfo.loc[test_setID,'choice'].astype('int').values
        acc_memo = f_evaluateRNN.evaluate_acc(modelD, test_X, test_label, toPrint=True)
        
        del modelD
        torch.cuda.empty_cache()
        
        gc.collect()

    # save/load models
    #Ybounds = [''.join([str(int(j[1][0]/100)), str(int(j[1][1]/100))]) for j in Ys_[1:]]
    #modelName = taskVersion + '_' + '_'.join([str(i) for i in [N_in, N_hidden]]+Ybounds)#, tau, dt, int(ext), ceiling, F_hidden, F_out
    


# In[]
np.save('E:/NUS/PhD/fitting planes/pooled/' + 'modelDicts4_2.npy', modelDicts, allow_pickle=True)
modelDicts = np.load('E:/NUS/PhD/fitting planes/pooled/' + 'modelDicts4_2.npy', allow_pickle=True).item()
# In[]

_, _, _, test_setID, test_X, _ = f_simulation.split_dataset(X_, Y0_, frac = 0.1, ranseed=114514)
test_Info = trialInfo.loc[test_setID,:].reset_index(drop = True)


# In[]
checkpoints = [150, 550, 1150, 1450, 1850, 2350, 2800]
avgInterval = {150:150, 550:250, 1150:250, 1450:150, 1850:250, 2350:250, 2800:200}

for kfi, _ in fitIntervals.items():
    for i in range(1):
        modelD = modelDicts[kfi][i]['rnn']
        label = f'fit interval:{kfi},{i}'
        
        test_label = trialInfo.loc[test_setID,'choice'].astype('int').values
        acc_memo = f_evaluateRNN.evaluate_acc(modelD, test_X, test_label, toPrint=True)
        
        f_evaluateRNN.plot_states(modelD, test_Info, test_X, tRange, trialEvents, locs = (0,1,2,3), ttypes = (1,2), lcX = np.arange(0,1,1), cues=False, cseq = None, label = label)
        f_evaluateRNN.plot_weights(modelD, label = label)
        
        _,_ = f_evaluateRNN.plot_lda(modelD, trialInfo, X_, Y0_, tRange, trialEvents, nIters=10, label = label)
        _,_ = f_evaluateRNN.plot_lda(modelD, trialInfo, X_, Y0_, tRange, trialEvents, nIters=10, label = label, targetItem='locX')
        
        vecs, projs, projsAll, _, trialInfos, vecs_shuff, projs_shuff, projsAll_shuff, _, trialInfos_shuff = f_evaluateRNN.generate_vectors(modelD, trialInfo, X_, Y0_, tRange, trialEvents, checkpoints, avgInterval, nIters = 20, nPerms = 50)
        geoms_valid, geoms_shuff = (vecs, projs, projsAll, trialInfos), (vecs_shuff, projs_shuff, projsAll_shuff, trialInfos_shuff)
        
        _,_,_,_,_ = f_evaluateRNN.plot_planeAlignment(geoms_valid, geoms_shuff, checkpoints, label = label)
        _,_ = f_evaluateRNN.plot_itemInfo(geoms_valid, geoms_shuff, checkpoints, label = label)
        _,_,_,_,_ = f_evaluateRNN.plot_choiceSpace(geoms_valid, geoms_shuff, checkpoints, label = label)













































# In[]

##################
###  nondecay  ###
##################

# In[]

###############
# 8ch version #
###############

# In[]
# task version 1: sequntial input, distractor & retarget, our version of task
taskVersion = 'seqSingle'
trialEvents = {'bsl':[-300,0], 's1':[0,300],'d1':[300,1300],'s2':[1300,1600],'d2':[1600,2600],'go':[2600,3000]} # event markers
X = f_simulation.generate_X_8ch(trialInfo, trialEvents, tRange, dt, vloc=vloc, vttype=vttype, noise=noise, gocue=gocue)

X_ = torch.tensor(X, dtype=torch.float32).to(device)

# In[]
# task version 2: sequentially displayed, 4 loc channels, cued recall
#taskVersion = 'seqMulti'
#trialEvents = {'bsl':[-300,0], 's1':[0,300],'d1':[300,1300],'s2':[1300,1600],'d2':[1600,2600],'cue':[2600,2900], 'd3':[2900,3900],'go':[3900,4500]} # event markers
#X = f_simulation.generate_X_6ch_seqMulti(trialInfo, trialEvents, tRange, dt, vloc=vloc, vttype=vttype, noise=noise, gocue=gocue)
#X_ = torch.tensor(X, dtype=torch.float32).to(device)

# In[]
# task version 3: simultaneous displayed, 8 loc channels, cued selection
#taskVersion = 'simMulti'
#trialEvents = {'bsl':[-300,0], 's1':[0,500],'d1':[500,1500],'cue':[1500,1800],'d2':[1800,2800],'go':[2800,3300]} # event markers
#X = f_simulation.generate_X_8ch_simMulti(trialInfo, trialEvents, tRange, dt, vloc=vloc, vttype=vttype, noise=noise, gocue=gocue)
#X_ = torch.tensor(X, dtype=torch.float32).to(device)

# In[]
kappa = 3
# In[]
#N_in = len(locs) + len(ttypes) + int(gocue) # The number of network inputs.
#N_in = len(locs) * len(ttypes) + int(gocue) # The number of network inputs.
N_in = X.shape[-1]
N_out = len(locs)
N_hidden = 128

# In[] genereate expected output values at different time windows
# always force baseline output to stay at chance level
expected0 = 1/N_out
Y0_Bonds = [-300,0]
Y0 = f_simulation.generate_Y(N_out, trialInfo.choice.values, Y0_Bonds[0], Y0_Bonds[1], dt, expected0)

Y0_ = torch.tensor(Y0, dtype=torch.float32).to(device)

# In[]
fitIntervals = {'go':(2600,3000),'ld2':(2100,2600),'ed2':(1600,2600),'s2':(1300,2600)} # intervals to fit the model outputs
nModels = 1 # number of models to be trained per interval

# In[]
modelDicts = {} # save trained models
expected1 = 1
for nfi, kfi in enumerate(fitIntervals):
    
    print(kfi)
    
    fi = fitIntervals[kfi]
    modelDicts[kfi] = {i:{} for i in range(nModels)}
    
    Y1_Bonds = fi
    #Y1 = generate_Y(N_out, trialInfo.choice.values, Y1_Bonds[0], Y1_Bonds[1], dt, expected1)
    Y1 = f_simulation.generate_Y_circular(N_out, trialInfo.choice.values, Y1_Bonds[0], Y1_Bonds[1], dt, expected1, kappa)
    Y1_ = torch.tensor(Y1, dtype=torch.float32).to(device)
    
    Ys_ = ((Y0_,Y0_Bonds,1,),(Y1_,Y1_Bonds,1,),) #
    
    for i in range(nModels):
        print(f'model n={i}')
        
        # if decay RNN
        modelD = myRNNs.nondecayRNN(N_in, N_hidden, N_out, dt = dt, tau = 100, external = ext, hidden_noise = hidden_noise, ceiling = ceiling, F_hidden = F_hidden, F_out = F_out).to(device)
        losses = f_trainRNN.train_model(modelD, trialInfo, X_, Ys_, tRange, frac = 0.2, criterion = nn.MSELoss(), learning_rate = 0.0001, n_iter = 1000*2, loss_cutoff = 0.001, lr_cutoff = 1e-7)
        
        modelDicts[kfi][i]['rnn'] = modelD
        modelDicts[kfi][i]['losses'] = losses
        
        _, _, _, test_setID, test_X, _ = f_simulation.split_dataset(X_, Ys_[0][0], frac = 0.1, ranseed=114514)
        test_label = trialInfo.loc[test_setID,'choice'].astype('int').values
        acc_memo = f_evaluateRNN.evaluate_acc(modelD, test_X, test_label, toPrint=True)
        
        del modelD
        torch.cuda.empty_cache()
        
        gc.collect()

    # save/load models
    #Ybounds = [''.join([str(int(j[1][0]/100)), str(int(j[1][1]/100))]) for j in Ys_[1:]]
    #modelName = taskVersion + '_' + '_'.join([str(i) for i in [N_in, N_hidden]]+Ybounds)#, tau, dt, int(ext), ceiling, F_hidden, F_out
# In[]
np.save('E:/NUS/PhD/fitting planes/pooled/' + 'modelDicts8_nondecay.npy', modelDicts, allow_pickle=True)
modelDicts = np.load('E:/NUS/PhD/fitting planes/pooled/' + 'modelDicts8_nondecay.npy', allow_pickle=True).item()
# In[]

_, _, _, test_setID, test_X, _ = f_simulation.split_dataset(X_, Y0_, frac = 0.1, ranseed=114514)
test_Info = trialInfo.loc[test_setID,:].reset_index(drop = True)


# In[]
checkpoints = [150, 550, 1150, 1450, 1850, 2350, 2800]
avgInterval = {150:150, 550:250, 1150:250, 1450:150, 1850:250, 2350:250, 2800:200}

for kfi, _ in fitIntervals.items():
    for i in range(1):
        modelD = modelDicts[kfi][i]['rnn']
        label = f'fit interval:{kfi},{i}'
        
        test_label = trialInfo.loc[test_setID,'choice'].astype('int').values
        acc_memo = f_evaluateRNN.evaluate_acc(modelD, test_X, test_label, toPrint=True, checkpointX = 290)
        
        f_evaluateRNN.plot_states(modelD, test_Info, test_X, tRange, trialEvents, locs = (0,1,2,3), ttypes = (1,2), lcX = np.arange(0,1,1), cues=False, cseq = None, label = label)
        f_evaluateRNN.plot_weights(modelD, label = label)
        
        _,_ = f_evaluateRNN.plot_lda(modelD, trialInfo, X_, Y0_, tRange, trialEvents, nIters=10, label = label)
        _,_ = f_evaluateRNN.plot_lda(modelD, trialInfo, X_, Y0_, tRange, trialEvents, nIters=10, label = label, targetItem='locX')
        
        vecs, projs, projsAll, _, trialInfos, vecs_shuff, projs_shuff, projsAll_shuff, _, trialInfos_shuff = f_evaluateRNN.generate_vectors(modelD, trialInfo, X_, Y0_, tRange, trialEvents, checkpoints, avgInterval, nIters = 20, nPerms = 50)
        geoms_valid, geoms_shuff = (vecs, projs, projsAll, trialInfos), (vecs_shuff, projs_shuff, projsAll_shuff, trialInfos_shuff)
        
        _,_,_,_,_ = f_evaluateRNN.plot_planeAlignment(geoms_valid, geoms_shuff, checkpoints, label = label)
        _,_ = f_evaluateRNN.plot_itemInfo(geoms_valid, geoms_shuff, checkpoints, label = label)
        _,_,_,_,_ = f_evaluateRNN.plot_choiceSpace(geoms_valid, geoms_shuff, checkpoints, label = label)

    
    








    
































































# In[]
modelD = modelDicts['s2'][4]['rnn']
# In[]
# visualize loss decrease
plt.figure()
plt.plot(losses)
plt.title('Losses')
plt.show()
# In[]

fitTimeWins = []

plt.figure(figsize=(10,1),dpi=100)

plt.plot(tRange, np.zeros_like(tRange), 'k-')
for i in Ys_:
   fitTimeWins += [i[1]]
   fitValue = i[-1]
   y = np.arange(0,fitValue,0.05)
   x1, x2 = i[1]
   plt.fill_betweenx(y=y, x1=np.full_like(y,x1), x2=np.full_like(y,x2), color = 'b', alpha = 0.3)

plt.text(tRange[0],0.5, f'dt:{dt}, tau:{tau}, ext:{ext}, hNoise:{hidden_noise}, ceil:{ceiling}, {F_hidden}, {F_out}')
plt.ylim(0,1)
plt.title('RNN fitted time windows')
plt.show()

# In[]
# test samples
_, _, _, test_setID, test_X, _ = f_simulation.split_dataset(X_, Y0_, frac = 0.1, ranseed=114514)
test_label = trialInfo.loc[test_setID,'choice'].astype('int').values
acc_memo = f_trainRNN.evaluate_acc(modelD, test_X, test_label, toPrint=True)

    


# In[]
test_Info = trialInfo.loc[test_setID,:].reset_index(drop = True)
test_set = test_X.cpu().numpy()

test_Info.loc1 = test_Info.loc1.astype(int)
test_Info.loc2 = test_Info.loc2.astype(int)
test_Info.choice = test_Info.choice.astype(int)
# In[]
hidden_states, out_states = modelD(test_X)
hidden_states = hidden_states.data.cpu().detach().numpy()

out_states = out_states.data.cpu().detach().numpy()

# In[] plot activities at each layer in different conditions
#

cseq = mpl.color_sequences['tab10']
cseq_r = cseq[::-1]

for l in locCombs[0:8]:
    #color = cseq[l]
    l1, l2 = l[0], l[1]

    for tp in ttypes:
        idx = test_Info[(test_Info.loc1 == l1) & (test_Info.loc2 == l2) & (test_Info.ttype == tp)].index
        inputsT = test_set[idx,:,:].mean(axis=0)
        hiddensT = hidden_states[idx,:,:].mean(axis=0)
        outputsT = out_states[idx,:,:].mean(axis=0)

        plt.figure(figsize=(30,10), dpi = 100)

        plt.subplot(3,1,1)
        ax1 = plt.gca()

        for ll in locs:
            ax1.plot(inputsT[:,ll], linestyle = '-', color = cseq_r[ll], label = str(ll) + 'T')
            #if 8 > N_in >= 6: # if 6 & 7 channels, plot as dash lines
                #ax1.plot(inputsT[:,ll+4], linestyle = '--', color = cseq[ll], label = str(ll) + 'D')
            if N_in >= 8:
                ax1.plot(inputsT[:,ll+4], linestyle = '--', color = cseq_r[ll], label = str(ll) + 'D')
        
        
        if N_in%2 == 1:
            ax1.plot(inputsT[:,-3], linestyle = ':', color = 'r', label = 'cue red')
            ax1.plot(inputsT[:,-2], linestyle = ':', color = 'g', label = 'cue green')
            ax1.plot(inputsT[:,-1], linestyle = ':', color = 'grey', label = 'fix')
        
        ax1.legend()
        ax1.set_title('Input')
        ax1.set_xticks([list(tRange).index(i[0]) for i in trialEvents.values()])
        ax1.set_xticklabels([i[0] for i in trialEvents.values()])

        plt.subplot(3,1,2)
        ax2 = plt.gca()
        #ax.plot(hiddensT[:,:], linestyle = '-')
        im2 = ax2.imshow(hiddensT.T, cmap = 'jet', aspect = 'auto')
        ax2.set_xticks([list(tRange).index(i[0]) for i in trialEvents.values()])
        ax2.set_xticklabels([i[0] for i in trialEvents.values()])
        ax2.set_title('Hidden')
        plt.colorbar(im2, ax = ax2, extend='both')

        plt.subplot(3,1,3)
        ax3 = plt.gca()
        for ll in locs:
            ax3.plot(outputsT[:,ll], linestyle = '-', color = cseq_r[ll], label = f'{str(ll)}, memo')
            if N_out > 4:
                ax3.plot(outputsT[:,ll+4], linestyle = '--', color = cseq_r[ll], label = f'{str(ll)}, motor')
        ax3.legend()
        ax3.set_title('Output')
        ax3.set_xticks([list(tRange).index(i[0]) for i in trialEvents.values()])
        ax3.set_xticklabels([i[0] for i in trialEvents.values()])
        ax3.set_ylim(0,1)

        plt.suptitle(f'Loc1:{l1} & Loc2:{l2} & Ttype:{tp}')
        plt.show()

# In[]
# get trained parameters (weight matrices)
paras = {}
for names, param in modelD.named_parameters():
    paras[names] = param

Wrec = paras['h2h.weight'].data.cpu().detach().numpy()
Win = paras['i2h.weight'].data.cpu().detach().numpy()
#Win = WinB
Wout = paras['h2o.weight'].data.cpu().detach().numpy().T
# In[] plot weight matrices
plt.figure(figsize=(30,10), dpi = 100)
ax1 = plt.subplot(1,3,1)
im1 = plt.imshow(Win,cmap='coolwarm')
#plt.imshow(WinB,cmap='jet')
cbar = plt.colorbar(im1, ax = ax1)
cbar.ax.tick_params(labelsize=15)

ax2 = plt.subplot(1,3,2)
im2 = plt.imshow(Wrec,cmap='coolwarm')
cbar = plt.colorbar(im2, ax = ax2)
cbar.ax.tick_params(labelsize=15)

ax3 = plt.subplot(1,3,3)
im3 = plt.imshow(Wout,cmap='coolwarm')
cbar = plt.colorbar(im3, ax = ax3)
cbar.ax.tick_params(labelsize=15)

plt.show()
# In[]







# In[]
import sklearn
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression
from scipy.ndimage import gaussian_filter1d

#from sklearn.metrics.pairwise import euclidean_distances
from scipy.spatial import ConvexHull
from scipy.linalg import orthogonal_procrustes
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from mpl_toolkits.mplot3d import Axes3D

#import dPCA


    
# In[] decode from pseudo population
pd.options.mode.chained_assignment = None
epsilon = 0.00001
# In[] decodability with/without permutation P value
bins = 50
tslice = (tRange.min(), tRange.max())
#tRange = np.arange(-300,3000,dt)
# In[]
nIters = 10#0
nPerms = 50
tbins = np.arange(tslice[0], tslice[1], bins)

tBsl = (-300,0)
idxxBsl = [tRange.tolist().index(tBsl[0]), tRange.tolist().index(tBsl[1])] #

nPCs = [0,10]#pseudo_Pop.shape[2]

conditions = (('ttype', 1), ('ttype', 2))


performance = {'Retarget':[], 'Distractor':[]}
performance_shuff = {'Retarget':[], 'Distractor':[]}
#pvalues = {'Retarget':{'dlpfc':[],'fef':[]}, 'Distractor':{'dlpfc':[],'fef':[]}}

# In[]
for n in range(nIters):
    t_IterOn = time.time()
    print(f'Iter = {n}')
    
    
    _, _, _, test_setID, test_X, _ = f_simulation.split_dataset(X_, Y0_, frac = 0.8, ranseed=n)

    test_Info = trialInfo.loc[test_setID,:].reset_index(drop = True)
    test_set = test_X.cpu().numpy()

    test_Info.loc1 = test_Info.loc1.astype(int)
    test_Info.loc2 = test_Info.loc2.astype(int)
    test_Info.choice = test_Info.choice.astype(int)
    
    #test_Info['locX'] = 0
    #for i in range(len(test_Info)):
    #    test_Info.loc[i,'locX'] = test_Info.loc[i,'loc1'] if test_Info.loc[i,'ttype'] == 1 else test_Info.loc[i,'loc2']
    
    hidden_states, out_states = modelD(test_X)
    hidden_states = hidden_states.data.cpu().detach().numpy()
    hidden_states = np.swapaxes(hidden_states, 1, 2) # shape as ntrials * nchs * ntime

    #out_states = out_states.data.cpu().detach().numpy()

    #for ch in range(N_hidden):
    #    hiddens = hidden_states[:,:,ch]
    #    hidden_states[:,:,ch] = scale(hiddens, method = '01')
    
    #pseudo_TrialInfo = pseudo_Session(locCombs, ttypes, samplePerCon=samplePerCon, sampleRounds=sampleRounds)
    #pseudo_region = pseudo_PopByRegion_pooled(sessions, cellsToUse, monkey_Info, samplePerCon = samplePerCon, sampleRounds=sampleRounds, arti_noise_level = arti_noise_level)
    
    #pseudo_data = {'pseudo_TrialInfo':pseudo_TrialInfo, 'pseudo_region':pseudo_region}
    #np.save(save_path + f'/pseudo_data{n}.npy', pseudo_data, allow_pickle=True)
    
    for condition in conditions:
        ttypeT = 'Retarget' if condition[-1]==1 else 'Distractor'
        
        if bool(condition):
            test_InfoT = test_Info[test_Info[condition[0]] == condition[1]]
        else:
            test_InfoT = test_Info.copy()
        
        Y = test_InfoT.loc[:,['choice','ttype','loc1','loc2','locX']].values
        ntrial = len(test_InfoT)
        
        ### decode for each region
        #for region in ('dlpfc','fef'):
            
        hidden_statesT = hidden_states[test_InfoT.index.values,:,:] # shape2 trial * cell * time
        
        # if detrend with standardization
        #for ch in range(pseudo_PopT.shape[1]):
        #    temp = pseudo_PopT[:,ch,:]
        #    pseudo_PopT[:,ch,:] = (temp - temp.mean(axis=0))/(temp.std(axis = 0) + epsilon) # to avoid divid by 0 issue
            
        #X_region = hidden_statesT
        
        ### scaling
        for ch in range(hidden_statesT.shape[1]):
            #hidden_statesT[:,ch,:] = scale(hidden_statesT[:,ch,:]) #01 scale for each channel
            hidden_statesT[:,ch,:] = (hidden_statesT[:,ch,:] - hidden_statesT[:,ch,:].mean()) / hidden_statesT[:,ch,:].std() #standard scaler
            #hidden_statesT[:,ch,:] = (hidden_statesT[:,ch,:] - hidden_statesT[:,ch,:].mean(axis=0)) / hidden_statesT[:,ch,:].std(axis=0) #detrended standard scaler
            #hidden_statesT[:,ch,:] = (hidden_statesT[:,ch,:] - hidden_statesT[:,ch,idxxBsl[0]:idxxBsl[1]].mean())/hidden_statesT[:,ch,idxxBsl[0]:idxxBsl[1]].std() # Z to baseline
        
        
        ### if specify applied time window of pca
        pca_tWin = np.hstack((np.arange(800,1300,dt, dtype = int),np.arange(2100,2600,dt, dtype = int))).tolist() #
        pca_tWinX = [tRange.tolist().index(i) for i in pca_tWin]
        
        #pca_tWinX = None
        hidden_statesTT = hidden_statesT[:,:,pca_tWinX] if pca_tWinX != None else hidden_statesT[:,:,:]
        
        
        
        ### averaging method
        # if average across all trials, shape = chs,time
        #X_regionT = X_regionT.mean(axis=0)
        
        # if none average, concatenate all trials, shape = chs, time*trials
        #X_regionT = np.concatenate(X_region, axis=-1)
        
        # if average across all trials all times, shape = chs,trials
        #X_regionT = X_regionT.mean(axis=-1).T
        
        # if conditional average, get mean for each condition, then concatenate condition_avgs, shape = chs, time*n_conditions
        #X_regionT_ = []
        #pseudo_TrialInfoT_ = pseudo_TrialInfoT.reset_index(drop=True)
        #for lc in locCombs:
        #    lcs = str(lc[0]) + '_' + str(lc[1])
        #    idxx = pseudo_TrialInfoT_[pseudo_TrialInfoT_.locs == lcs].index.tolist()
        #    X_regionT_ += [X_regionT[idxx,:,:].mean(axis=0)]
        
        #X_regionT = np.concatenate(X_regionT_, axis=-1)
        
        # if conditional average + avged over time, get mean for each condition, then vstack condition_avgs, shape = chs, n_conditions
        hidden_statesTT_ = []
        test_InfoT_ = test_InfoT.reset_index(drop=True)
        for lc in locCombs:
            l1, l2 = lc[0], lc[1]
            idxx = test_InfoT_[(test_InfoT_.loc1 == l1) & (test_InfoT_.loc2 == l2)].index.tolist()
            hidden_statesTT_ += [hidden_statesTT[idxx,:,:].mean(axis=0).mean(axis=-1)]
        
        hidden_statesTT = np.vstack(hidden_statesTT_).T
        
        
        ### fit & transform pca
        pcFrac = 0.9
        npc = min(int(pcFrac * hidden_statesTT.shape[0]), hidden_statesTT.shape[1])
        pca = PCA(n_components=npc)
        
        pca.fit(hidden_statesTT.T)
        evr = pca.explained_variance_ratio_
        print(f'{condition[1]}, {evr.round(4)[0:5]}')
        
        hidden_statesTP = np.zeros((hidden_statesT.shape[0], npc, hidden_statesT.shape[2]))
        for trial in range(hidden_statesT.shape[0]):
            hidden_statesTP[trial,:,:] = pca.transform(hidden_statesT[trial,:,:].T).T
        
        
        ### split into train and test sets
        train_setID = np.sort(np.random.choice(ntrial, round(0.5*ntrial),replace = False))
        test_setID = np.sort(np.random.choice(np.setdiff1d(np.arange(ntrial), train_setID, assume_unique=True), (ntrial-train_setID.shape[0]),replace = False))
        #test_setID = np.sort(np.random.choice(ntrial, round(0.5*ntrial),replace = False))

        train_setP = hidden_statesTP[train_setID,:,:]
        test_setP = hidden_statesTP[test_setID,:,:]
        
        #train_setP, test_setP, _ = PCA_DataSet(train_set, test_set, mode = 0, pc_frac=0.5, pca_tWinX=pca_tWinX, avgMethod='all')
        #pc_list = [1,2]
        #if (train_setP.shape[1]>1 and test_setP.shape[1]>1):    
        #    train_setP, test_setP = train_setP[:,nPCs[0]:nPCs[1],:], test_setP[:,nPCs[0]:nPCs[1],:] # specify PCs to be used
            
            #train_setP, test_setP = np.reshape(train_setP, (train_setP.shape[0],1,train_setP.shape[-1])), np.reshape(test_setP, (test_setP.shape[0],1,test_setP.shape[-1]))
        #else:
        #    train_setP, test_setP = np.random.randn(*train_setP.shape), np.random.randn(*test_setP.shape)
        
        train_label = Y[train_setID,0].astype('int').astype('str') # locKey
        #train_label = Y[train_setID,-1].astype('int').astype('str') # to be ignored location
        
        #train_label = Y[train_setID,2].astype('str') # type
        #train_label = np.char.add(Y[train_setID,0].astype('int').astype('str'), Y[train_setID,2].astype('str'))#Y[train_setID,0].astype('int') # locKey+Type
        # (locKey = 0,'locs','type','loc1','loc2', 'locX')
        
        
        test_label = Y[test_setID,0].astype('int').astype('str') # locKey
        #test_label = Y[test_setID,-1].astype('int').astype('str') # to be ignored location
        
        #test_label = Y[test_setID,2].astype('str') # Type
        #test_label = np.char.add(Y[test_setID,0].astype('int').astype('str'), Y[test_setID,2].astype('str')) # locKey+Type
        
        
        
        ### down sample to 50ms/bin
        ntrialT, ncellT, ntimeT = train_setP.shape
        train_setP = np.mean(train_setP.reshape(ntrialT, ncellT, int(ntimeT/(bins/dt)),int(bins/dt)),axis = 3)
        
        ntrialT, ncellT, ntimeT = test_setP.shape
        test_setP = np.mean(test_setP.reshape(ntrialT, ncellT, int(ntimeT/(bins/dt)),int(bins/dt)),axis = 3)
        
        # if normalize at each time point
        #train_setP = (train_setP - train_setP.mean(axis=0))/train_setP.std(axis=0)
        #test_setP = (test_setP - test_setP.mean(axis=0))/test_setP.std(axis=0)
        
        
        ### LDA decodability
        performanceX = []
        #pX = []
        for t in range(len(tbins)):
            performanceX_ = []
            #pX_ = []
            for t_ in range(len(tbins)):
                
                #performanceX_ += [f_decoding.LDAPerformance(train_set[:,nPCs[0]:nPCs[1],t], test_set[:,nPCs[0]:nPCs[1],t_], train_label, test_label)]
                pfmTT_, pTT_ = f_decoding.LDAPerformance(train_setP[:,nPCs[0]:nPCs[1],t], test_setP[:,nPCs[0]:nPCs[1],t_], train_label, test_label), 1
                #pfmTT_, pTT_ = f_decoding.LDAPerformance_P(train_setP[:,nPCs[0]:nPCs[1],t], test_setP[:,nPCs[0]:nPCs[1],t_], train_label, test_label, n_permutations=50)
                performanceX_ += [pfmTT_]
                #pX_ += [pTT_]
            
            performanceX += [np.array(performanceX_)]
            #pX += [np.array(pX_)]
            
        performance[ttypeT] += [np.array(performanceX)]
        #pvalues[ttypeT] += [np.array(pX)]
        
        
        # permutation with shuffled label
        performanceX_shuff = []
        for t in range(len(tbins)):
            performanceX_shuff_ = []
            
            for t_ in range(len(tbins)):
                performanceX_shuff_p = []
                
                #for npm in range(nPerms):
                    #np.random.seed(0)
                #    train_label_shuff, test_label_shuff = np.random.permutation(train_label), np.random.permutation(test_label)
                #    pfmTT_shuff = f_decoding.LDAPerformance(train_setP[:,nPCs[0]:nPCs[1],t], test_setP[:,nPCs[0]:nPCs[1],t_], train_label_shuff, test_label_shuff)
                #    performanceX_shuff_p += [pfmTT_shuff]
                
                #performanceX_shuff_ += [np.array(performanceX_shuff_p)]
                
                performanceX_shuff_ += [np.ones(nPerms)] # dummy
            
            performanceX_shuff += [np.array(performanceX_shuff_)]
        
        performance_shuff[ttypeT] += [np.array(performanceX_shuff)]
    
    print(f'tIter = {(time.time() - t_IterOn):.4f}s')

# In[]
#np.save('E:/NUS/PhD/fitting planes/pooled/w&w/' + 'performance.npy', performance, allow_pickle=True)
#np.save('E:/NUS/PhD/fitting planes/pooled/w&w/' + 'performance_shuff.npy', performance_shuff, allow_pickle=True)
#np.save('E:/NUS/PhD/fitting planes/pooled/w&w/' + 'performance_X.npy', performance, allow_pickle=True)
#np.save('E:/NUS/PhD/fitting planes/pooled/w&w/' + 'performance_shuff_X.npy', performance_shuff, allow_pickle=True)


#performance = np.load('E:/NUS/PhD/fitting planes/pooled/w&w/' + 'performance.npy', allow_pickle=True).item()
#performance_shuff = np.load('E:/NUS/PhD/fitting planes/pooled/w&w/' + 'performance_shuff.npy', allow_pickle=True).item()
#performance = np.load('E:/NUS/PhD/fitting planes/pooled/w&w/' + 'performance_X.npy', allow_pickle=True).item()
#performance_shuff = np.load('E:/NUS/PhD/fitting planes/pooled/w&w/' + 'performance_shuff_X.npy', allow_pickle=True).item()
# In[]
for condition in conditions:
    
    ttypeT = 'Retarget' if condition[-1]==1 else 'Distractor'
    
       
    performanceT = performance[ttypeT]
    performanceT_shuff = performance_shuff[ttypeT]
    
    pfm = np.array(performanceT)
    pfm_shuff = np.concatenate(np.array(performanceT_shuff),axis=2)
    
    pvalues = np.zeros((len(tbins), len(tbins)))
    for t in range(len(tbins)):
        for t_ in range(len(tbins)):
            pvalues[t,t_] = f_stats.permutation_p(pfm.mean(axis = 0)[t,t_], pfm_shuff[t,t_,:], tail = 'greater')
            #pvalues[t,t_] = stats.ttest_1samp(pfm[:,t,t_], 0.25, alternative = 'greater')[1]
            
    
    vmax = 1
    
    plt.figure(figsize=(15,12), dpi = 100)
    ax = plt.gca()
    sns.heatmap(pd.DataFrame(pfm.mean(axis = 0), index=tbins,columns=tbins), cmap = 'jet', vmin = 0.25, ax = ax, vmax = vmax)#, vmax = 0.6, xticklabels = 100, yticklabels = 100, vmin = cbar_min, vmax = cbar_max
    
    #from scipy import ndimage
    smooth_scale = 10
    z = ndimage.zoom(pvalues, smooth_scale)
    ax.contour(np.linspace(0, len(tbins), len(tbins) * smooth_scale),
             np.linspace(0, len(tbins), len(tbins) * smooth_scale),
              z, levels=(0,0.05), colors='white', alpha = 1)
    
    ax.invert_yaxis()
    
    
    # event lines
    for i in [0, 300, 1300, 1600, 2600]:
        ax.plot(tbins, np.full_like(tbins,list(tbins).index(i)), 'w-.', linewidth=4)
        ax.plot(np.full_like(tbins,list(tbins).index(i)), tbins, 'w-.', linewidth=4)
    
    ax.set_xticks([list(tbins).index(i) for i in [0, 300, 1300, 1600, 2600]])
    ax.set_xticklabels([0, 300, 1300, 1600, 2600], rotation=0, fontsize = 25)
    ax.set_xlabel('Test Timebin (ms)', labelpad = 10, fontsize = 35)
    ax.set_yticks([list(tbins).index(i) for i in [0, 300, 1300, 1600, 2600]])
    ax.set_yticklabels([0, 300, 1300, 1600, 2600], fontsize = 25)
    ax.set_ylabel('Train Timebin (ms)', labelpad = 10, fontsize = 35)
    
    cbar = ax.collections[0].colorbar
    # here set the labelsize by 20
    cbar.ax.tick_params(labelsize=20)
    
    plt.title(f'{ttypeT}, nPCs = {nPCs}', pad = 10, fontsize = 25)
    plt.show()






# In[]


# In[]


# In[]


# In[]


# In[]




# In[]


# In[]


# In[]


# In[]


# In[]


# In[]


# In[]






# In[]


# In[]


# In[]


# In[]


# In[]


# In[]


# In[]







# In[]


# In[]


# In[]


# In[]


# In[]


# In[]


# In[]





# In[]


# In[]


# In[]


# In[]


# In[]


# In[]


# In[]





# In[]


# In[]


# In[]


# In[]


# In[]


# In[]


# In[]







# In[]


# In[]


# In[]


# In[]


# In[]


# In[]


# In[]









# In[]


# In[]


# In[]


# In[]


# In[]


# In[]


# In[]


# In[] if non-decay RNN

locs = [0,1,2,3] # location conditions
ttypes = [1,2] # ttype conditions

locCombs = list(permutations(locs,2))
# In[]
dt = 10 # The simulation timestep.
# time axis
tRange = np.arange(0,3300,dt)
tLength = len(tRange) # The trial length.

# In[]
N_batch = 1000 # trial pool size
accFracs = (1, 0, 0) # fraction of correct, random incorrect (non-displayed-irrelavent), non-random incorrect (displayed-irrelavent) trials

trialInfo = f_simulation.generate_trials(N_batch, locs, ttypes, accFracs)


# In[]
# simulate input values for each trial across time

gocue = True
vloc = 1.0 # input strength in location channels
vttype = 1.0 # input strength in type(color) channel
noise = 0.1 # background random noise level

#X = generate_X_6ch(trialInfo, trialEvents, tRange, dt, vloc=vloc, vttype=vttype, noise=noise, gocue=True)
#X = generate_X_8ch(trialInfo, trialEvents, tRange, dt, vloc=vloc, vttype=vttype, noise=noise, gocue=gocue)

# In[] setting parameters for RNN
torch.cuda.empty_cache() # empty GPU
tau = 100

hidden_noise = 0.1
ext = 0
ceiling = None # [0,10]# 

#hNonlinear = 'relu'
#oNonlinear = 'softmax'
F_hidden = 'relu'
F_out = 'softmax'

# In[]
# task version 1: sequntial input, distractor & retarget, our version of task
taskVersion = 'seqSingle'
trialEvents = {'s1':[0,300],'d1':[300,1300],'s2':[1300,1600],'d2':[1600,2600],'go':[2600,3300]} # event markers
X = f_simulation.generate_X_8ch(trialInfo, trialEvents, tRange, dt, vloc=vloc, vttype=vttype, noise=noise, gocue=gocue)

X_ = torch.tensor(X, dtype=torch.float32).to(device)

# In[]
kappa = 3
# In[]
#N_in = len(locs) + len(ttypes) + int(gocue) # The number of network inputs.
#N_in = len(locs) * len(ttypes) + int(gocue) # The number of network inputs.
N_in = X.shape[-1]
N_out = len(locs)
N_hidden = 128

# In[] genereate expected output values at different time windows
# always force baseline output to stay at chance level
#expected0 = 1/N_out
#Y0_Bonds = [-300,0]
#Y0 = f_simulation.generate_Y(N_out, trialInfo.choice.values, Y0_Bonds[0], Y0_Bonds[1], dt, expected0)

#Y0_ = torch.tensor(Y0, dtype=torch.float32).to(device)

# In[]
fitIntervals = {'go':(2600,3300),'ld2':(2100,3300),'ed2':(1600,3300),'s2':(1300,3300)} # intervals to fit the model outputs
nModels = 5 # number of models to be trained per interval

ceiling = [0,10] # None #

expected1 = 1
Y1_Bonds = (2100,3300)
#Y1 = generate_Y(N_out, trialInfo.choice.values, Y1_Bonds[0], Y1_Bonds[1], dt, expected1)
Y1 = f_simulation.generate_Y_circular(N_out, trialInfo.choice.values, Y1_Bonds[0], Y1_Bonds[1], dt, expected1, kappa)
Y1_ = torch.tensor(Y1, dtype=torch.float32).to(device)

Ys_ = ((Y1_,Y1_Bonds,1,),) #(Y0_,Y0_Bonds,1,),

# 
modelD = myRNNs.nondecayRNN(N_in, N_hidden, N_out, dt = dt, tau = 100, external = ext, hidden_noise = hidden_noise, ceiling = ceiling, F_hidden = F_hidden, F_out = F_out).to(device)
losses = f_trainRNN.train_model(modelD, trialInfo, X_, Ys_, tRange, frac = 0.2, criterion = nn.MSELoss(), learning_rate = 0.0001, n_iter = 1000*5, loss_cutoff = 0.01, lr_cutoff = 1e-7)

#losses = f_trainRNN.train_model2(modelD, trialInfo, X_, Y1_, tRange, frac = 0.2, criterion = nn.MSELoss(), learning_rate = 0.0001, n_iter = 1000*5, loss_cutoff = 0.01, lr_cutoff = 1e-7)


del modelD
torch.cuda.empty_cache()

gc.collect()



































# archive
# In[]
Y1_Bonds = [2600,3300]
#Y1 = generate_Y(N_out, trialInfo.choice.values, Y1_Bonds[0], Y1_Bonds[1], dt, expected1)
Y1 = f_simulation.generate_Y_circular(N_out, trialInfo.choice.values, Y1_Bonds[0], Y1_Bonds[1], dt, expected1, kappa)

#expected2 = 1
#Y2_Bonds = [2100,2600]
#Y2 = f_simulation.generate_Y_circular(N_out, trialInfo.choice.values, Y2_Bonds[0], Y2_Bonds[1], dt, expected2, kappa)

#expected3 = 1
#Y3_Bonds = [2600,3300]
#Y3 = f_simulation.generate_Y_circular(N_out, trialInfo.choice.values, Y3_Bonds[0], Y3_Bonds[1], dt, expected3, kappa)

# In[] stack expected outputs together
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


Y1_ = torch.tensor(Y1, dtype=torch.float32).to(device)
#Y2_ = torch.tensor(Y2, dtype=torch.float32).to(device)
#Y3_ = torch.tensor(Y3, dtype=torch.float32).to(device)
# In[]
Ys_ = ((Y0_,Y0_Bonds,1,),(Y1_,Y1_Bonds,1,),) #,(Y2_,Y2_Bonds,1,),(Y3_,Y3_Bonds,1,),



# In[]
# unconstrained model, with decay factor
modelD = myRNNs.decayRNN(N_in, N_hidden, N_out, dt = dt, tau = 100, external = ext, hidden_noise = hidden_noise, ceiling = ceiling, F_hidden = F_hidden, F_out = F_out).to(device)

# del modelD
#torch.cuda.empty_cache() 
# In[] 

losses = f_trainRNN.train_model(modelD, trialInfo, X_, Ys_, tRange, frac = 0.2, criterion = nn.MSELoss(), learning_rate = 0.0001, n_iter = 1000*5, loss_cutoff = 0.001, lr_cutoff = 1e-7)



# In[] Initialize loss function and optimizer
learning_rate = 0.0001
n_iter = 1000*10
cutoff = 0.001

criterion = nn.MSELoss()
optimizer = torch.optim.NAdam(modelD.parameters(), lr=learning_rate, weight_decay=0.001)

losses = []
counts = 0

# In[]
# Train the model
for epoch in range(n_iter):

    train_setID, train_X, train_Y1, _, _, _ = f_simulation.split_dataset(X_, Ys_[0][0], frac = 0.2, ranseed=epoch)

    # Forward pass
    _, outs = modelD(train_X)

    loss = 0

    lambda_reg = 0.001

    for y in range(len(Ys_)):
        train_Yt = Ys_[y][0][train_setID, :, :]
        y_boundary = Ys_[y][1]
        lweight_t = Ys_[y][2]

        lowBt, upBt = np.where((tRange>=y_boundary[0])&(tRange<=y_boundary[-1]))[0][0], np.where((tRange>=y_boundary[0])&(tRange<=y_boundary[-1]))[0][-1]

        if upBt == tLength-1:
            loss_t = criterion(outs[:, lowBt:, :], train_Yt)
        else:
            loss_t = criterion(outs[:, lowBt:upBt, :], train_Yt)

        loss += loss_t * lweight_t

    # l2 regularization if need penalize and make the weights to be sparse
    #l2_reg = 0.0
    #for name, param in modelD.named_parameters():
    #    if 'h2h' in name:
    #        l2_reg += torch.norm(param, p=2)

    #loss = loss + lambda_reg * l2_reg

    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # record loss every 50 iterations
    if epoch%50 == 0:
        counts += 1
        losses += [float(loss.detach())]

        # print loss every 100 iterations
        if epoch % 100 == 0:
            #n_correct = 0
            n_correct_memo, n_correct_motor = 0,0
            n_samples = 0

            train_label = trialInfo.loc[train_setID,'choice'].astype('int').values
            labels = torch.tensor(train_label, dtype=torch.float32).to(device)
            #_, predicted = torch.max(outs[:,-1,:].detach().data, 1) # channel with max output at last timepoint -> choice
            
            n_samples += labels.size(0)
            
            _, predicted_memo = torch.max(outs[:,-1,:].data, 1) # channel with max output at last timepoint -> choice
            n_correct_memo += (predicted_memo == labels).sum().item()
            acc_memo = 100.0 * n_correct_memo / n_samples  
            
            print (f'Epoch [{epoch}/{n_iter}], Loss: {loss.item():.4f}, Acc_memo: {acc_memo:.2f}%')

        # adaptively update learning rate
        if counts >= 10:
            if False not in ((np.array(losses[-10:]) - float(loss.detach()))<=learning_rate):
                learning_rate = learning_rate/2
                optimizer.param_groups[0]['lr'] = learning_rate
                print(f'loss = {loss:.4f}, updated learning rate = {learning_rate}')
                counts = 0 # reset counts

                if learning_rate < 1e-7:
                    print(f'learning rate too small: {learning_rate} < 1e-7')
                    break

    # if loss lower than the cutoff value, end training
    if loss < cutoff:
        print(f'Loss = {loss.item():.4f}, lower then cutoff = {cutoff}')
        break



# In[]
# save/load models
Ybounds = [''.join([str(int(j[1][0]/100)), str(int(j[1][1]/100))]) for j in Ys_[1:]]
modelName = taskVersion + '_' + '_'.join([str(i) for i in [N_in, N_hidden]]+Ybounds)#, tau, dt, int(ext), ceiling, F_hidden, F_out
#modelName = taskVersion + '_'.join([str(i) for i in [N_in, N_hidden]]+Ybounds)#, tau, dt, int(ext), ceiling, F_hidden, F_out

# In[]
def generate_Y_multiple(output_size, labels1, labels2, lowB, upB, dt, expected1, expected2, expectedRest=None):
    '''
    lowB : fitting epoch start time
    upB : fitting epoch ending time
    dt : timestep
    output_expected: expected value at the correct output channel
    '''
    lowB, upB = lowB, upB

    tx = np.arange(lowB, upB, dt)

    Y = np.full((len(labels1), len(tx), output_size),0,dtype=float)

    for i in range(len(labels1)):
        loc1 = labels1[i]
        loc2 = labels2[i]
        #loc1 = train_Info.loc1[i]
        #loc2 = train_Info.loc2[i]
        #ttype = train_Info.ttype[i]

        indices = np.arange(output_size)

        expected1 = expected1
        expected2 = expected2
        
        if expected1 == 0 and expected2 == 0:
            Y[i,:,:] = 0
            #Y[i,tx[0]:tx[-1]+1,indices!=choice] = output_expected

        else:
            loc1 = loc1
            loc2 = loc2
            locRest = ((indices!=loc1) * (indices!=loc2))
            Y[i,:,loc1] = expected1
            Y[i,:,loc2] = expected2
            
            if expectedRest == None:
                Y[i,:,locRest] += (1-expected1-expected2)/(output_size-2) # use this for softmax F_out; if sigmoid can adjust
            else:
                Y[i,:,locRest] += expectedRest

    return Y

