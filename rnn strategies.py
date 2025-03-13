# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 14:23:51 2024

@author: aka2333
"""
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


#%%

#################
# evaluate RNNs #
#################

#%% generate test set and corresponding trial info
_, _, _, test_setID, test_X, _ = f_simulation.split_dataset_balance(X_, Y0_, trialInfo, frac = 0.5, ranseed=114514)
test_Info = trialInfo.loc[test_setID,:].reset_index(drop = True)
test_label = trialInfo.loc[test_setID,'choice'].astype('int').values

#%% downsampling for plotting
bins = 50 # downsampling bins, same as simulation timestep, so doesn't really affect the results
tslice = (tRange.min(),tRange.max()+dt)
tbins = np.arange(tslice[0], tslice[1], bins)

#%% [Figure 1C, 1D, 1E] model performance, plot states

plot_samples = (0,1,2) # show example RNNs only
accuracies_rnns = {} # store RNN accuracies

for nfi, kfi in enumerate(fitIntervals):
    print(kfi)
    models_dict = modelDicts[kfi]
    
    label = f'fit interval:{kfi}'
    strategyLabel = 'Rehearse & Update' if kfi == 'R&U' else 'Retrieve at Recall'
    
    ii=0
    accuraciesT = []

    for i in range(len(modelDicts[kfi])):
        print(i)
        modelD = models_dict[i]['rnn']
        acc_memo = f_evaluateRNN.evaluate_acc(modelD, test_X, test_label, toPrint=True, 
                                              checkpointX = np.where(tRange <= fitIntervals[kfi][-1][-1])[0].max()) #
        accuraciesT += [acc_memo]

        # only plot examples if the model performs well enough
        if acc_memo >=75:
            
            if (ii in plot_samples):
                f_evaluateRNN.plot_states(modelD, test_Info, test_X, tRange, trialEvents, dt=dt, locs = (0,1,2,3), ttypes = (1,2), 
                                          lcX = np.arange(0,1,1), cues=False, cseq = None, label = strategyLabel, 
                                          withhidden=False, withannote=False, save_path=data_path, savefig=False)
                            
            ii+=1
        
    accuraciesT = np.array(accuraciesT)
    accuracies_rnns[strategyLabel] = accuraciesT
    print(f"{strategyLabel}: M(SD) = {accuraciesT.mean():.3f}({accuraciesT.std():.3f})") # print the mean and std of the accuracies

#%%

###########################
# Full Space Decodability #
###########################

#%% compute decodability
performancesX1, performancesX2 = {},{}
performancesX1_shuff, performancesX2_shuff = {},{}
performancesXtt, performancesXtt_shuff = {},{}
EVRs = {}
for nfi, kfi in enumerate(fitIntervals):
    print(kfi)
    models_dict = modelDicts[kfi]
    
    label = f'fit interval:{kfi}'
    strategyLabel = 'Rehearse & Update' if kfi == 'R&U' else 'Retrieve at Recall'
    
    pfmsX1 = {tt:[] for tt in ttypes}
    pfmsX2 = {tt:[] for tt in ttypes}
    pfmsX1_shuff = {tt:[] for tt in ttypes}
    pfmsX2_shuff = {tt:[] for tt in ttypes}
    
    pfmsXtt = []
    pfmsXtt_shuff = []
    
    evrs = []
    
    for i in range(len(modelDicts[kfi])):
        print(i)
        modelD = models_dict[i]['rnn']
        
        # calculate EVR of full space (PC1-15)        
        evrsT = f_evaluateRNN.rnns_EVRs(modelD, trialInfo, X_, Y0_, tRange, dt = 50, bins = 50, 
                                        nPCs = (0,15), nIters = 1, nBoots = 10, pca_tWins = ((300,1300), (1600,2600),), 
                                        pDummy = True, label=f'{strategyLabel} ', toPlot = False, shuff_excludeInv = False)

        # calculate decodability of item1 and item2
        pfmsX12T, pfmsX12_shuffT, evrsT = f_evaluateRNN.rnns_lda12X(modelD, trialInfo, X_, Y0_, tRange, dt = 50, bins = 50, 
                                                        nPCs = (0,15), nIters = 1, nBoots = 10, pca_tWins = ((300,1300), (1600,2600),), 
                                                        pDummy = True, label=f'{strategyLabel} ', toPlot = False, shuff_excludeInv = False) # (100,800),
        
        # calculate decodability of trial type
        pfmsXttT, pfmsXtt_shuffT, _ = f_evaluateRNN.rnns_ldattX(modelD, trialInfo, X_, Y0_, tRange, dt = 50, bins = 50, 
                                                        nPCs = (0,15), nIters = 1, nBoots = 10, pca_tWins = ((300,1300), (1600,2600),), 
                                                        pDummy = True, label=f'{strategyLabel} ', toPlot = False, shuff_excludeInv = False) # (100,800),
        
        # store the results
        pfmsX1[1] += [np.array(pfmsX12T[0][1])] 
        pfmsX1[2] += [np.array(pfmsX12T[0][2])] 
        pfmsX2[1] += [np.array(pfmsX12T[1][1])] 
        pfmsX2[2] += [np.array(pfmsX12T[1][2])] 
        
        pfmsX1_shuff[1] += [np.array(pfmsX12_shuffT[0][1])] 
        pfmsX1_shuff[2] += [np.array(pfmsX12_shuffT[0][2])] 
        pfmsX2_shuff[1] += [np.array(pfmsX12_shuffT[1][1])] 
        pfmsX2_shuff[2] += [np.array(pfmsX12_shuffT[1][2])] 
        
        evrs += [np.array(evrsT)]
        
        pfmsXtt += [np.array(pfmsXttT)]
        pfmsXtt_shuff += [np.array(pfmsXtt_shuffT)]
        
        del modelD
        torch.cuda.empty_cache()

        gc.collect()
    
    performancesX1[kfi] = pfmsX1
    performancesX2[kfi] = pfmsX2
    performancesX1_shuff[kfi] = pfmsX1_shuff
    performancesX2_shuff[kfi] = pfmsX2_shuff
    EVRs[kfi] = np.array(evrs)
    
    performancesXtt[kfi] = pfmsXtt
    performancesXtt_shuff[kfi] = pfmsXtt_shuff

#%% save computed decodability
np.save(f'{data_path}/' + 'performanceX1_full_rnn.npy', performancesX1, allow_pickle=True)
np.save(f'{data_path}/' + 'performanceX2_full_rnn.npy', performancesX2, allow_pickle=True)
np.save(f'{data_path}/' + 'performanceX1_full_shuff_rnn.npy', performancesX1_shuff, allow_pickle=True)
np.save(f'{data_path}/' + 'performanceX2_full_shuff_rnn.npy', performancesX2_shuff, allow_pickle=True)

np.save(f'{data_path}/' + 'performanceXtt_full_rnn.npy', performancesXtt, allow_pickle=True)
np.save(f'{data_path}/' + 'performanceXtt_full_shuff_rnn.npy', performancesXtt_shuff, allow_pickle=True)

#%% [Figure 2A] plot item decodability full space
for nfi, kfi in enumerate(fitIntervals):
    print(kfi)
    strategyLabel = 'Rehearse & Update' if kfi == 'R&U' else 'Retrieve at Recall'

    pfmX1 = performancesX1[kfi]
    pfmX2 = performancesX2[kfi]
    pfmX1_shuff = performancesX1_shuff[kfi]
    pfmX2_shuff = performancesX2_shuff[kfi]
    
    bins = 50
    tslice = (tRange.min(), tRange.max()+dt)
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
            pfmTX1_shuff = np.array(pfmX1_shuff[tt]).squeeze().mean(1)
            pfmTX2_shuff = np.array(pfmX2_shuff[tt]).squeeze().mean(1)
            
            # calculate permutation p values at each timebin
            pPerms_pfm1 = np.ones((len(tbins), len(tbins)))
            pPerms_pfm2 = np.ones((len(tbins), len(tbins)))
            for t in range(len(tbins)):
                for t_ in range(len(tbins)):
                    pPerms_pfm1[t, t_] = f_stats.permutation_pCI(pfmTX1[:,t,t_], pfmTX1_shuff[:,t,t_], tail='greater', alpha=5)
                    pPerms_pfm2[t, t_] = f_stats.permutation_pCI(pfmTX2[:,t,t_], pfmTX2_shuff[:,t,t_], tail='greater', alpha=5)
            
            
            # plot decoding performance of item 1
            vmax = 1
            plt.subplot(2,2,tt)
            ax = plt.gca()
            sns.heatmap(pd.DataFrame(pfmTX1.mean(0), index=tbins,columns=tbins), cmap = 'magma', vmin = 0.0, vmax = vmax, ax = ax)#, vmax = 0.6, xticklabels = 100, yticklabels = 100, vmin = cbar_min, vmax = cbar_max
            
            # outline significant timebins
            smooth_scale = 10
            z = ndimage.zoom(pPerms_pfm1, smooth_scale)
            ax.contour(np.linspace(0, len(tbins), len(tbins) * smooth_scale),
                    np.linspace(0, len(tbins), len(tbins) * smooth_scale),
                    z, levels=([0.05]), colors='white', alpha = 1, linewidths = 3)
            
            ax.invert_yaxis()
            
            # adjust labels
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
            cbar.ax.tick_params(labelsize=20)
            ax.set_title(f'{ttypeT_}, Item1', fontsize = 30, pad = 20)
            
            
            # plot decoding performance of item 2
            plt.subplot(2,2,tt+2)
            ax = plt.gca()
            sns.heatmap(pd.DataFrame(pfmTX2.mean(0), index=tbins,columns=tbins), cmap = 'magma', vmin = 0.0, vmax = vmax, ax = ax)#, vmax = 0.6, xticklabels = 100, yticklabels = 100, vmin = cbar_min, vmax = cbar_max
            
            # outline significant timebins
            smooth_scale = 10
            z = ndimage.zoom(pPerms_pfm2, smooth_scale)
            ax.contour(np.linspace(0, len(tbins), len(tbins) * smooth_scale),
                    np.linspace(0, len(tbins), len(tbins) * smooth_scale),
                    z, levels=([0.05]), colors='white', alpha = 1, linewidths = 3)
            
            ax.invert_yaxis()
            
            # adjust labels
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
            cbar.ax.tick_params(labelsize=20)
            ax.set_title(f'{ttypeT_}, Item2', fontsize = 30, pad = 20)
            
        plt.tight_layout()
        plt.subplots_adjust(top = 0.95)
        plt.suptitle(f'{strategyLabel}, Full Space', fontsize = 35, y=1) #, Arti_Noise = {arti_noise_level}
        plt.show()
        
#%% plot trial type decodability full space        
for nfi, kfi in enumerate(fitIntervals):
    print(kfi)
    strategyLabel = 'Rehearse & Update' if kfi == 'R&U' else 'Retrieve at Recall'

    pfmX1 = performancesXtt[kfi]
    pfmX1_shuff = performancesXtt_shuff[kfi]
    
    bins = 50
    tslice = (tRange.min(), tRange.max()+dt)
    tbins = np.arange(tslice[0], tslice[1], bins) #
    
    if len(pfmX1[1])>0:
        
        fig = plt.figure(figsize=(7, 6), dpi=300)
                            
        pfmTX1 = np.array(pfmX1).squeeze().mean(1)
        pfmTX1_shuff = np.array(pfmX1_shuff).squeeze().mean(1)
        
        # calculate permutation p values at each timebin
        pPerms_pfm1 = np.ones((len(tbins), len(tbins)))
        for t in range(len(tbins)):
            for t_ in range(len(tbins)):
                pPerms_pfm1[t, t_] = f_stats.permutation_pCI(pfmTX1[:,t,t_], pfmTX1_shuff[:,t,t_], tail='greater', alpha=5)
        
        # plot decoding performance of trial type
        vmax = 1
        plt.subplot(1,1,1)
        ax = plt.gca()
        sns.heatmap(pd.DataFrame(pfmTX1.mean(0), index=tbins,columns=tbins), cmap = 'magma', vmin = 0.25, vmax = vmax, ax = ax)#, vmax = 0.6, xticklabels = 100, yticklabels = 100, vmin = cbar_min, vmax = cbar_max
        
        # outline significant timebins
        smooth_scale = 10
        z = ndimage.zoom(pPerms_pfm1, smooth_scale)
        ax.contour(np.linspace(0, len(tbins), len(tbins) * smooth_scale),
                np.linspace(0, len(tbins), len(tbins) * smooth_scale),
                z, levels=([0.05]), colors='white', alpha = 1, linewidths = 3)
        
        ax.invert_yaxis()
        
        # adjust labels
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
        cbar.ax.tick_params(labelsize=15)
        ax.set_title(f'Trial Type', fontsize = 25, pad = 20)
            
        plt.tight_layout()
        plt.subplots_adjust(top = 1)
        plt.suptitle(f'{strategyLabel}, Full Space', fontsize = 25, y=1.25) #, Arti_Noise = {arti_noise_level}
        plt.show()



#%%


##########################
# subspace relationships #
##########################


#%% initialize parameters and dictionaries
checkpoints = [150, 550, 1050, 1450, 1850, 2350]#
checkpointsLabels = ['S1','ED1','LD1','S2','ED2','LD2'] #, 2800
avgInterval = {150:150, 550:250, 1050:250, 1450:150, 1850:250, 2350:250} #, 2800:200
nPerms = 100
nBoots = 100
infoMethod='lda'

# items v item
cosThetas_11, cosThetas_22, cosThetas_12 = {},{},{}
cosPsis_11, cosPsis_22, cosPsis_12 = {},{},{}

cosThetas_11_shuff, cosThetas_22_shuff, cosThetas_12_shuff = {},{},{}
cosPsis_11_shuff, cosPsis_22_shuff, cosPsis_12_shuff = {},{},{}

cosThetas_11_bsl, cosThetas_22_bsl, cosThetas_12_bsl = {},{},{}
cosPsis_11_bsl, cosPsis_22_bsl, cosPsis_12_bsl = {},{},{}

# choice items
cosThetas_choice, cosPsis_choice = {},{}
cosThetas_nonchoice, cosPsis_nonchoice = {},{}

cosThetas_choice_shuff, cosPsis_choice_shuff = {},{}
cosThetas_nonchoice_shuff, cosPsis_nonchoice_shuff = {},{}

# code transferability items
performanceX_12, performanceX_21 = {},{}
performanceX_12_shuff, performanceX_21_shuff = {},{}

# code transferability choices/nonchoices
performanceX_rdc, performanceX_drc = {},{}
performanceX_rdnc, performanceX_drnc = {},{}
performanceX_rdc_shuff, performanceX_drc_shuff = {},{}
performanceX_rdnc_shuff, performanceX_drnc_shuff = {},{}

# item info by item subspace projections
info3ds_1, info3ds_2 = {},{}
info3ds_1_shuff, info3ds_2_shuff = {},{}

# cross-temporal decodability of items by readout subspace projections
infos_C1X, infos_C2X = {},{}
infos_C1X_shuff, infos_C2X_shuff = {},{}

# EVRs of top3 PCs
EVRs_C1 = {}
EVRs_C2 = {}

#%% [Figure 4A] calculate geoms & plot with example RNNs
for nfi, kfi in enumerate(fitIntervals):
    print(kfi)
    
    label = f'fit interval:{kfi}'
    strategyLabel = 'Rehearse & Update' if kfi == 'R&U' else 'Retrieve at Recall'
    
    models_dict = modelDicts[kfi]
    
    vecs_C, projs_C, projsAll_C, trialInfos_C, data_3pc_C, pcas_C, evrs_C, evrs2nd_C, vecs_C_shuff, projs_C_shuff, projsAll_C_shuff, trialInfos_C_shuff, data_3pc_C_shuff, _ = f_evaluateRNN.generate_choiceVectors(models_dict, trialInfo, X_, Y0_, tRange, nBoots=nBoots, nPerms=nPerms, pca_tWins=((300,1300),(1600,2600),), 
                                                                                                                                                                                                    dt=dt, toPlot = True, label=f'{strategyLabel}',plot3d=False,
                                                                                                                                                                                                    hideLocs = (0,2), savefig=False, save_path = f'{data_path}/', normalizeMinMax=(-1,1), separatePlot=False) #
    EVRs_C1[kfi] = evrs_C
    EVRs_C2[kfi] = evrs2nd_C
    
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
    
    # cross-temporal item info by readout subspace projection
    infos_C1X[kfi], infos_C2X[kfi] = [],[]
    infos_C1X_shuff[kfi], infos_C2X_shuff[kfi] = [],[]
    
    for i in range(len(modelDicts[kfi])):
        print(i)

        # wrapping the vectors to be used
        geoms_valid, geoms_shuff = (vecs[i], projs[i], projsAll[i], trialInfos[i]), (vecs_shuff[i], projs_shuff[i], projsAll_shuff[i], trialInfos_shuff[i])
        geomsC_valid, geomsC_shuff = (vecs_C[i], projs_C[i], projsAll_C[i], trialInfos_C[i]), (vecs_C_shuff[i], projs_C_shuff[i], projsAll_C_shuff[i], trialInfos_C_shuff[i])
        geoms_bsl_train, geoms_bsl_test = (vecs_bsl_train[i], projs_bsl_train[i], projsAll_bsl_train[i], trialInfos_bsl_train[i]), (vecs_bsl_test[i], projs_bsl_test[i], projsAll_bsl_test[i], trialInfos_bsl_test[i])
        
        # angle alignment between item pairs
        cosThetas, cosThetas_shuff, cosPsis, cosPsis_shuff = f_evaluateRNN.get_angleAlignment_itemPairs(geoms_valid, geoms_shuff, checkpoints)
        
        # angle alignment between choice pairs
        cosThetas_C, cosThetas_C_shuff, cosPsis_C, cosPsis_C_shuff = f_evaluateRNN.get_angleAlignment_choicePairs(geoms_valid, geoms_shuff, checkpoints)
        
        # baseline angle alignment distributions between item pairs
        cosThetas_bsl, cosPsis_bsl = f_evaluateRNN.get_angleAlignment_itemPairs_bsl(geoms_bsl_train, geoms_bsl_test, checkpoints)
        
        # angle alignment between item-specific subspace and the readout subspace
        cosThetas_ivr, cosThetas_ivr_shuff, cosPsis_ivr, cosPsis_ivr_shuff = f_evaluateRNN.get_angleAlignment_itemRead(geoms_valid, geoms_shuff, geomsC_valid, geomsC_shuff, checkpoints)
        
        # code transferability between item-specific subspaces
        pfmTrans, pfmTrans_shuff = f_evaluateRNN.itemInfo_by_plane_Trans(geoms_valid, checkpoints, nPerms=nPerms, shuff_excludeInv = False)
        
        # code transferability between choice-item subspaces
        pfmTransc, pfmTransc_shuff, pfmTransnc, pfmTransnc_shuff = f_evaluateRNN.chioceInfo_by_plane_Trans(geoms_valid, checkpoints, nPerms=nPerms, shuff_excludeInv = False)
        
        # item info by item subspace projections        
        info3d, info3d_shuff = f_evaluateRNN.itemInfo_by_plane(geoms_valid, checkpoints, nPerms = nPerms, infoMethod=infoMethod, shuff_excludeInv = False)
        
        # cross-temporal item info by readout subspace projection
        info_CIX, info_CIX_shuff = f_evaluateRNN.itemInfo_by_planeCX(geomsC_valid, nPerms = nPerms, bins = bins, tslice = tslice, shuff_excludeInv = False)
        
        # store results for choice item
        cosThetas_choice[kfi] += [cosThetas_C[0]]
        cosPsis_choice[kfi] += [cosPsis_C[0]]
        
        cosThetas_nonchoice[kfi] += [cosThetas_C[1]]
        cosPsis_nonchoice[kfi] += [cosPsis_C[1]]

        cosThetas_choice_shuff[kfi] += [cosThetas_C_shuff[0]]
        cosPsis_choice_shuff[kfi] += [cosPsis_C_shuff[0]]
        
        cosThetas_nonchoice_shuff[kfi] += [cosThetas_C_shuff[1]]
        cosPsis_nonchoice_shuff[kfi] += [cosPsis_C_shuff[1]]

        # store results for item vs item
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
        
        # store results for code transferability between item1 and item2 subspaces
        performanceX_12[kfi] += [pfmTrans[0]]
        performanceX_21[kfi] += [pfmTrans[1]]
        performanceX_12_shuff[kfi] += [pfmTrans_shuff[0]]
        performanceX_21_shuff[kfi] += [pfmTrans_shuff[1]]
        
        # store results for code transferability choice nonchoice
        performanceX_rdc[kfi] += [pfmTransc[0]]
        performanceX_drc[kfi] += [pfmTransc[1]]
        performanceX_rdnc[kfi] += [pfmTransnc[0]]
        performanceX_drnc[kfi] += [pfmTransnc[1]]
        
        performanceX_rdc_shuff[kfi] += [pfmTransc_shuff[0]]
        performanceX_drc_shuff[kfi] += [pfmTransc_shuff[1]]
        performanceX_rdnc_shuff[kfi] += [pfmTransnc_shuff[0]]
        performanceX_drnc_shuff[kfi] += [pfmTransnc_shuff[1]]

        # store results for item info by item subspace
        info3ds_1[kfi] += [info3d[0]]
        info3ds_2[kfi] += [info3d[1]]
        info3ds_1_shuff[kfi] += [info3d_shuff[0]]
        info3ds_2_shuff[kfi] += [info3d_shuff[1]]
        
        # store results for cross-temporal item info by readout subspace
        infos_C1X[kfi] += [info_CIX[0]]
        infos_C2X[kfi] += [info_CIX[1]]
        infos_C1X_shuff[kfi] += [info_CIX_shuff[0]]
        infos_C2X_shuff[kfi] += [info_CIX_shuff[1]]
        

#%% save computed geometry results

np.save(f'{data_path}/' + 'cosTheta_choice_rnn.npy', cosThetas_choice, allow_pickle=True)
np.save(f'{data_path}/' + 'cosPsi_choice_rnn.npy', cosPsis_choice, allow_pickle=True)
np.save(f'{data_path}/' + 'cosTheta_nonchoice_rnn.npy', cosThetas_nonchoice, allow_pickle=True)
np.save(f'{data_path}/' + 'cosPsi_nonchoice_rnn.npy', cosPsis_nonchoice, allow_pickle=True)

np.save(f'{data_path}/' + 'cosTheta_choice_shuff_rnn.npy', cosThetas_choice_shuff, allow_pickle=True)
np.save(f'{data_path}/' + 'cosPsi_choice_shuff_rnn.npy', cosPsis_choice_shuff, allow_pickle=True)
np.save(f'{data_path}/' + 'cosTheta_nonchoice_shuff_rnn.npy', cosThetas_nonchoice_shuff, allow_pickle=True)
np.save(f'{data_path}/' + 'cosPsi_nonchoice_shuff_rnn.npy', cosPsis_nonchoice_shuff, allow_pickle=True)

np.save(f'{data_path}/' + 'cosTheta_11_rnn.npy', cosThetas_11, allow_pickle=True)
np.save(f'{data_path}/' + 'cosTheta_22_rnn.npy', cosThetas_22, allow_pickle=True)
np.save(f'{data_path}/' + 'cosTheta_12_rnn.npy', cosThetas_12, allow_pickle=True)
np.save(f'{data_path}/' + 'cosPsi_11_rnn.npy', cosPsis_11, allow_pickle=True)
np.save(f'{data_path}/' + 'cosPsi_22_rnn.npy', cosPsis_22, allow_pickle=True)
np.save(f'{data_path}/' + 'cosPsi_12_rnn.npy', cosPsis_12, allow_pickle=True)

np.save(f'{data_path}/' + 'cosTheta_11_shuff_rnn.npy', cosThetas_11_shuff, allow_pickle=True)
np.save(f'{data_path}/' + 'cosTheta_22_shuff_rnn.npy', cosThetas_22_shuff, allow_pickle=True)
np.save(f'{data_path}/' + 'cosTheta_12_shuff_rnn.npy', cosThetas_12_shuff, allow_pickle=True)
np.save(f'{data_path}/' + 'cosPsi_11_shuff_rnn.npy', cosPsis_11_shuff, allow_pickle=True)
np.save(f'{data_path}/' + 'cosPsi_22_shuff_rnn.npy', cosPsis_22_shuff, allow_pickle=True)
np.save(f'{data_path}/' + 'cosPsi_12_shuff_rnn.npy', cosPsis_12_shuff, allow_pickle=True)

np.save(f'{data_path}/' + 'cosTheta_11_bsl_rnn.npy', cosThetas_11_bsl, allow_pickle=True)
np.save(f'{data_path}/' + 'cosTheta_22_bsl_rnn.npy', cosThetas_22_bsl, allow_pickle=True)
np.save(f'{data_path}/' + 'cosTheta_12_bsl_rnn.npy', cosThetas_12_bsl, allow_pickle=True)
np.save(f'{data_path}/' + 'cosPsi_11_bsl_rnn.npy', cosPsis_11_bsl, allow_pickle=True)
np.save(f'{data_path}/' + 'cosPsi_22_bsl_rnn.npy', cosPsis_22_bsl, allow_pickle=True)
np.save(f'{data_path}/' + 'cosPsi_12_bsl_rnn.npy', cosPsis_12_bsl, allow_pickle=True)

np.save(f'{data_path}/' + 'performanceX_12_rnn.npy', performanceX_12, allow_pickle=True)
np.save(f'{data_path}/' + 'performanceX_12_shuff_rnn.npy', performanceX_12_shuff, allow_pickle=True)
np.save(f'{data_path}/' + 'performanceX_21_rnn.npy', performanceX_21, allow_pickle=True)
np.save(f'{data_path}/' + 'performanceX_21_shuff_rnn.npy', performanceX_21_shuff, allow_pickle=True)

np.save(f'{data_path}/' + 'performanceX_rdc_rnn.npy', performanceX_rdc, allow_pickle=True)
np.save(f'{data_path}/' + 'performanceX_rdc_shuff_rnn.npy', performanceX_rdc_shuff, allow_pickle=True)
np.save(f'{data_path}/' + 'performanceX_drc_rnn.npy', performanceX_drc, allow_pickle=True)
np.save(f'{data_path}/' + 'performanceX_drc_shuff_rnn.npy', performanceX_drc_shuff, allow_pickle=True)

np.save(f'{data_path}/' + 'performanceX_rdnc_rnn.npy', performanceX_rdnc, allow_pickle=True)
np.save(f'{data_path}/' + 'performanceX_rdnc_shuff_rnn.npy', performanceX_rdnc_shuff, allow_pickle=True)
np.save(f'{data_path}/' + 'performanceX_drnc_rnn.npy', performanceX_drnc, allow_pickle=True)
np.save(f'{data_path}/' + 'performanceX_drnc_shuff_rnn.npy', performanceX_drnc_shuff, allow_pickle=True)

np.save(f'{data_path}/' + 'performance1_item_rnn.npy', info3ds_1, allow_pickle=True)
np.save(f'{data_path}/' + 'performance2_item_rnn.npy', info3ds_2, allow_pickle=True)
np.save(f'{data_path}/' + 'performance1_item_shuff_rnn.npy', info3ds_1_shuff, allow_pickle=True)
np.save(f'{data_path}/' + 'performance2_item_shuff_rnn.npy', info3ds_2_shuff, allow_pickle=True)

np.save(f'{data_path}/' + 'performanceX1_readout_rnn.npy', infos_C1X, allow_pickle=True)
np.save(f'{data_path}/' + 'performanceX2_readout_rnn.npy', infos_C2X, allow_pickle=True)
np.save(f'{data_path}/' + 'performanceX1_readout_shuff_rnn.npy', infos_C1X_shuff, allow_pickle=True)
np.save(f'{data_path}/' + 'performanceX2_readout_shuff_rnn.npy', infos_C2X_shuff, allow_pickle=True)

#%% [Figure S4A] plot readout decodability
for nfi, kfi in enumerate(fitIntervals):
    print(kfi)
    info3d_1,info3d_2 = info3ds_1[kfi], info3ds_2[kfi]
    info_C1X,info_C2X = infos_C1X[kfi], infos_C2X[kfi] 

    info3d_1_shuff,info3d_2_shuff = info3ds_1_shuff[kfi], info3ds_2_shuff[kfi]
    info_C1X_shuff,info_C2X_shuff = infos_C1X_shuff[kfi], infos_C2X_shuff[kfi] 


    label = f'fit interval:{kfi}'
    strategyLabel = 'Rehearse & Update' if kfi == 'R&U' else 'Retrieve at Recall'
        
            
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
            sns.heatmap(pd.DataFrame(pfm1.mean(0), index=tbins,columns=tbins), cmap = 'magma', vmin = 0.0, vmax = 1, ax = ax)
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
            sns.heatmap(pd.DataFrame(pfm2.mean(0), index=tbins,columns=tbins), cmap = 'magma', vmin = 0.0, vmax = 1, ax = ax)
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
#%%

######################################
# drift distance on readout subspace #
######################################

#%% calculate euclidean distance between centroids on readout subspace
hideLocs=(0,2)

nPerms = 100
nBoots = 100

bins = 50
end_D1s, end_D2s = np.arange(800,1300+bins,bins), np.arange(2100,2600+bins,bins)

euDists_centroids2 = {}
euDists_centroids2_shuff = {}

for nfi, kfi in enumerate(fitIntervals):
    print(kfi)
    
    label = f'fit interval:{kfi}'
    strategyLabel = 'Rehearse & Update' if kfi == 'R&U' else 'Retrieve at Recall'
    
    models_dict = modelDicts[kfi]
    
    euDists_centroids2[kfi] = {tt:[] for tt in ttypes}
    euDists_centroids2_shuff[kfi] = {tt:[] for tt in ttypes}

    # compute readout space vectors
    vecs_C, projs_C, projsAll_C, trialInfos_C, data_3pc_C, _, evrs_C, vecs_C_shuff, projs_C_shuff, projsAll_C_shuff, trialInfos_C_shuff, data_3pc_C_shuff, _ = f_evaluateRNN.generate_choiceVectors(models_dict, trialInfo, X_, Y0_, tRange, nBoots=nBoots, nPerms=nPerms, pca_tWins=((300,1300),(1600,2600),), 
                                                                                                                                                                                                    dt=dt, toPlot = False, label=f'{strategyLabel}',plot3d=False)
    
    # for each model, compute euclidean distance between centroids
    for i in range(len(modelDicts[kfi])):
        
        for tt in ttypes:
            euDists_centroids2[kfi][tt].append([])
            euDists_centroids2_shuff[kfi][tt].append([])    
        
        for nbt in range(nBoots):
            geomsC_valid = (vecs_C[i][nbt], projs_C[i][nbt], projsAll_C[i][nbt], trialInfos_C[i][nbt], data_3pc_C[i][nbt])
            euDist_centroids2T = f_evaluateRNN.get_euDist_centroids2(geomsC_valid, locs = locs, ttypes = ttypes, bins = bins, dt = dt, tslice = tslice, end_D1s = end_D1s, end_D2s = end_D2s, zscore = False, normalizeMinMax=(-1,1), hideLocs=hideLocs)
            
            for tt in ttypes:
                euDists_centroids2[kfi][tt][i] += [euDist_centroids2T[tt]]
        
        # genereate label-shuffled baseline distribution
        for npm in range(nPerms): 
            euDist_centroids2T_shuff = f_evaluateRNN.get_euDist_centroids2(geomsC_valid, locs = locs, ttypes = ttypes, bins = bins, dt = dt, tslice = tslice, end_D1s = end_D1s, end_D2s = end_D2s, zscore = False, normalizeMinMax=(-1,1), hideLocs=hideLocs, shuffleBsl=True)
        
            for tt in ttypes:
                euDists_centroids2_shuff[kfi][tt][i] += [euDist_centroids2T_shuff[tt]]

#%% save computed drift distances
np.save(f'{data_path}/' + 'euDists_rnns_centroids2_normalized.npy', euDists_centroids2, allow_pickle=True)
np.save(f'{data_path}/' + 'euDists_rnns_centroids2_shuff_normalized.npy', euDists_centroids2_shuff, allow_pickle=True)

































