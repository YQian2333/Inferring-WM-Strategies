# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 20:03:31 2024

@author: aka2333
"""
# In[]
import numpy as np
from scipy import ndimage

import pandas as pd

# basic plot functions
import matplotlib as mpl
import matplotlib.pyplot as plt
import re, seaborn as sns

# turn off warning messages
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import gc
import time # timer
from itertools import permutations, combinations, product # itertools

import sklearn
from sklearn.decomposition import PCA

import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # define device

import f_simulation
import f_subspace
import f_stats
import f_decoding

#%%
def sort_by_columns(arr):
    """
    Sorts the rows of a 2D numpy array in ascending order based on the values 
    in each column hierarchically and returns the sorted array along with the 
    original row indices in the new order.
    
    Parameters:
    arr (np.ndarray): 2D array to be sorted.

    Returns:
    tuple: (sorted_array, original_indices)
    """
    original_indices = np.lexsort(tuple(arr[:,arr.shape[1]-n-1] for n in range(arr.shape[1])))
    sorted_array = arr[original_indices]
    return sorted_array, original_indices

#%%
def evaluate_acc(modelD, test_X, test_label, checkpointX = -1, label = 'Acc', toPrint = False):
    
    test_X = test_X
    test_label = test_label
    
    with torch.no_grad():
        n_correct_memo = 0
        n_samples = 0
        _, outputs_p = modelD(test_X)
        
        labels = torch.tensor(test_label, dtype=torch.float32).to(device)
        n_samples += labels.size(0)
        
        
        _, predicted_memo = torch.max(outputs_p[:,checkpointX,:].data, 1) # channel with max output at last timepoint -> choice
        n_correct_memo += (predicted_memo == labels).sum().item()
        acc_memo = 100.0 * n_correct_memo / n_samples
        if toPrint:
            print(f'{label} of the network on the test set: {acc_memo} %')
    
    return acc_memo


#%%       
def plot_states(modelD, test_Info, test_X, tRange, trialEvents, dt = 10, locs = (0,1,2,3), ttypes = (1,2), 
                lcX = np.arange(0,2,1), cues=False, cseq = None, label = '', vmin = 0, vmax = 10, 
                withhidden = True, withannote =True, savefig=False, save_path=''):

    """
    lcX: index of locCombs that are chosen to be plotted
    cues: show cue channel or not
    cseq: color sequence
    label: title of the plot
    withhidden: show hidden states or not
    withannote: show annotation or not
    savefig: save figure or not
    save_path: path to save figure
    """

    test_set = test_X.cpu().numpy()
    
    hidden_states, out_states = modelD(test_X)
    hidden_states = hidden_states.data.cpu().detach().numpy()
    out_states = out_states.data.cpu().detach().numpy()
    
    N_in, N_out = test_set.shape[-1], out_states.shape[-1]
    
    locs = test_Info.loc1.unique() if len(locs)==0 else locs
    ttypes = test_Info.ttype.unique() if len(ttypes)==0 else ttypes
    
    locCombs = list(permutations(locs,2))
    cseq = cseq if cseq != None else mpl.color_sequences['tab10']
    cseq_r = cseq[::-1]

    for l in np.array(locCombs)[lcX,:]:
        l1, l2 = l
        figsize=(30,12) if withhidden else (20,10)
        fig = plt.figure(figsize=figsize, dpi = 100)
        
        for tt in ttypes:
            
            ttypeT_ = 'Retarget' if tt==1 else 'Distraction'
            ttypeT = 'Retarget' if tt==1 else 'Distractor'
            
            idx = test_Info[(test_Info.loc1 == l1) & (test_Info.loc2 == l2) & (test_Info.ttype == tt)].index
            inputsT = test_set[idx,:,:].mean(axis=0)
            hiddensT = hidden_states[idx,:,:].mean(axis=0)
            outputsT = out_states[idx,:,:].mean(axis=0)

            
            # input states
            plt.subplot(3,2,tt)
            ax1 = plt.gca()

            for ll in locs:
                ax1.plot(inputsT[:,ll], linestyle = '-', linewidth = 6, color = cseq_r[ll], label = f'Location {str(int(ll+1))}, Target')
                if N_in >= 8:
                    ax1.plot(inputsT[:,ll+4], linestyle = '--', dashes=(3, 1), linewidth = 6, color = cseq_r[ll], label = f'Location {str(int(ll+1))}, Distractor')
                if cues:
                    if N_in%2 == 1:
                        ax1.plot(inputsT[:,-1], linestyle = ':', color = 'grey', dashes=(3, 1), linewidth = 6, label = 'Fixation')
                
                        ax1.plot(inputsT[:,-3], linestyle = ':', color = 'r', dashes=(3, 1), linewidth = 6, label = 'cue red')
                        ax1.plot(inputsT[:,-2], linestyle = ':', color = 'g', dashes=(3, 1), linewidth = 6, label = 'cue green')
                        
                    else:
                        ax1.plot(inputsT[:,-2], linestyle = ':', color = 'r', dashes=(3, 1), linewidth = 6, label = 'cue red')
                        ax1.plot(inputsT[:,-1], linestyle = ':', color = 'g', dashes=(3, 1), linewidth = 6, label = 'cue green')
            
            if withannote:
                arrowprops = dict(arrowstyle="->", facecolor='k', linewidth = 6, alpha = 0.8)
                ax1.annotate('Item1 (Target) Location', xy=(tRange.tolist().index(150), 0.9), xytext=(tRange.tolist().index(350), 0.8), xycoords='data', arrowprops=arrowprops, fontsize = 20, alpha = 0.8)
                ax1.annotate('Alternative Locations', xy=(tRange.tolist().index(150), 0.1), xytext=(tRange.tolist().index(250), 0.4), xycoords='data', arrowprops=arrowprops, fontsize = 20, alpha = 0.8)
                ax1.annotate(f'Item2 ({ttypeT}) Location', xy=(tRange.tolist().index(1450), 0.9), xytext=(tRange.tolist().index(1650), 0.8), xycoords='data', arrowprops=arrowprops, fontsize = 20, alpha = 0.8)
            
            
            if tt==2:
                ax1.legend(loc = 'center left', bbox_to_anchor=(1, 0.5), fontsize = 15)
            ax1.set_title(f'Input', fontsize = 20)
            ax1.set_xticks([list(tRange).index(i[0]) for i in trialEvents.values()])
            ax1.set_xticklabels([i[0] for i in trialEvents.values()], fontsize = 15)
            ax1.tick_params(axis='y', labelsize=15)
            ax1.set_xlim(left=0)
            
            
            # output states
            plt.subplot(3,2,tt+2)
            ax3 = plt.gca()
            for ll in locs:
                ax3.plot(outputsT[:,ll], linestyle = '-', linewidth = 6, color = cseq_r[ll], label = f'Location {str(int(ll+1))}')
                if N_out > 4:
                    ax3.plot(outputsT[:,ll+4], linestyle = '--', linewidth = 6, color = cseq_r[ll], label = f'{str(ll)}, resp')
            
            if withannote:
                ax3.text(tRange.tolist().index(800), 0.5, f'Target -> {ttypeT}', fontsize = 20, alpha = 0.8)
            
            
            if tt==2:
                ax3.legend(loc = 'center left', bbox_to_anchor=(1, 0.5), fontsize = 15)
            ax3.set_title(f'Output', fontsize = 20)
            ax3.set_xticks([list(tRange).index(i[0]) for i in trialEvents.values()])
            ax3.set_xticklabels([i[0] for i in trialEvents.values()], fontsize = 15)
            ax3.tick_params(axis='y', labelsize=15)
            ax3.set_xlim(left=0)
            

            if withhidden:
                # hidden states
                plt.subplot(3,2,tt+4)
                ax2 = plt.gca()
                im2 = ax2.imshow(hiddensT.T, cmap = 'magma', aspect = 'auto', vmin=vmin, vmax=vmax)
                ax2.set_xticks([list(tRange).index(i[0]) for i in trialEvents.values()])
                ax2.set_xticklabels([i[0] for i in trialEvents.values()], fontsize = 15)
                ax2.set_title(f'Hidden', fontsize = 20)
                ax2.tick_params(axis='y', labelsize=15)
                ax2.set_xlim(left=0)

        plt.suptitle(f'{label}, Item1:{l1+1} & Item2:{l2+1}', fontsize = 25)
        plt.tight_layout()
        plt.show()
        if savefig:
            fig.savefig(f'{save_path}/{label}_states.tif')


#%% plot weight matrices
def plot_weights(modelD, label = ''):
    # get trained parameters (weight matrices)
    paras = {}
    for names, param in modelD.named_parameters():
        paras[names] = param
    
    Wrec = paras['Wrec'].data.cpu().detach().numpy() if ('Wrec' in paras.keys()) else paras['h2h.weight'].data.cpu().detach().numpy()
    Win = paras['Win'].data.cpu().detach().numpy() if ('Win' in paras.keys()) else paras['i2h.weight'].data.cpu().detach().numpy()
    
    #sort Win by columns
    Win, sortingIdx = sort_by_columns(Win)
    Wrec = Wrec[sortingIdx,:] # sort Wrec by Win columns
    Wout = paras['h2o.weight'].data.cpu().detach().numpy().T
    
    plt.figure(figsize=(20,10), dpi = 100)
    ax1 = plt.subplot(1,3,1)
    vabs = np.abs(Win).max()
    im1 = plt.imshow(Win,cmap='coolwarm', vmin=-1*vabs, vmax=vabs)
    cbar = plt.colorbar(im1, ax = ax1)
    cbar.ax.tick_params(labelsize=15)
    plt.tight_layout()
    
    ax2 = plt.subplot(1,3,2)
    vabs = np.abs(Wrec).max()
    im2 = plt.imshow(Wrec,cmap='coolwarm', vmin=-1*vabs, vmax=vabs)
    cbar = plt.colorbar(im2, ax = ax2)
    cbar.ax.tick_params(labelsize=15)
    plt.tight_layout()
    
    ax3 = plt.subplot(1,3,3)
    vabs = np.abs(Wout).max()
    im3 = plt.imshow(Wout,cmap='coolwarm', vmin=-1*vabs, vmax=vabs)
    cbar = plt.colorbar(im3, ax = ax3)
    cbar.ax.tick_params(labelsize=15)
    plt.tight_layout()
    
    plt.suptitle(f'{label}')
    plt.show()

#%%
def generate_itemVectors(models_dict, trialInfo, X_, Y_, tRange, checkpoints, avgInterval, dt = 10, locs = (0,1,2,3), ttypes = (1,2), 
                         adaptPCA=None, adaptEVR = None, pca_tWins = ((300,1300), (1600,2600),), avgMethod='conditional_time', 
                         nBoots = 1, fracBoots = 1.0, nPerms = 100, toPlot=False):
    
    """
    Compute the item-specific subspace vectors for all models
    
    adaptPCA: if applicable, adapt the PCA transformation from existing matrices
    adaptEVR: same as adaptPCA
    pca_tWins: time windows to compute PCA
    nBoots: number of bootstraps for each model
    fracBoots: fraction of data to use per bootstrap for each model
    nPerms: number of permutations within each bootstrap for each model to create null distribution
    toPlot: whether to plot the results
    """

    epsilon = 1e-7
    
    nIters = len(models_dict) # number of models
    
    vecs_All = []
    projs_All = []
    projsAll_All = []
    trialInfos_All = []
    pca1s_All = []

    vecs_shuff_All = []
    projs_shuff_All = []
    projsAll_shuff_All = []
    trialInfos_shuff_All = []
    pca1s_shuff_All = []
    
    evrs_1st_All = []
    
    _, _, _, test_setID, test_X, _ = f_simulation.split_dataset_balance(X_, Y_, trialInfo, frac = 0.5)
    
    test_Info = trialInfo.loc[test_setID,:].reset_index(drop = True)
    test_set = test_X.cpu().numpy()

    test_Info.loc1 = test_Info.loc1.astype(int)
    test_Info.loc2 = test_Info.loc2.astype(int)
    test_Info.choice = test_Info.choice.astype(int)
    
    # specify applied time window of pca
    pca_tWin = np.hstack([np.arange(i[0],i[1],dt, dtype = int) for i in pca_tWins]).tolist() #
    pca_tWinX = [tRange.tolist().index(i) for i in pca_tWin]
    
    for n in range(nIters):
        
        vecs = {}
        projs = {}
        projsAll = {}
        trialInfos = {}
        
        pca1s = []

        vecs_shuff = {}
        projs_shuff = {}
        projsAll_shuff = {}
        trialInfos_shuff = {}
        pca1s_shuff = []


        for tt in ttypes:
            trialInfos[tt] = []
            trialInfos_shuff[tt] = []
            

        for cp in checkpoints:
            vecs[cp] = {}
            projs[cp] = {}
            projsAll[cp] = {}

            vecs_shuff[cp] = {}
            projs_shuff[cp] = {}
            projsAll_shuff[cp] = {}
            
            for tt in ttypes:
                vecs[cp][tt] = {1:[], 2:[]}
                projs[cp][tt] = {1:[], 2:[]}
                projsAll[cp][tt] = {1:[], 2:[]}
                

                vecs_shuff[cp][tt] = {1:[], 2:[]}
                projs_shuff[cp][tt] = {1:[], 2:[]}
                projsAll_shuff[cp][tt] = {1:[], 2:[]}
        
        evrs_1st = np.zeros((nBoots, 3))
        
        modelD = models_dict[n]['rnn']
        hidden_states, _ = modelD(test_X)
        hidden_states = hidden_states.data.cpu().detach().numpy()       
        dataN = hidden_states.swapaxes(1,2) # reshape to [trials, channels, time]
        
        for ch in range(dataN.shape[1]):
            dataN[:,ch,:] = (dataN[:,ch,:] - dataN[:,ch,:].mean() + epsilon) / (dataN[:,ch,:].std() + epsilon) #standard scaler
        

        for nbt in range(nBoots):
               
            # initialization
            pca1s.append([])
            pca1s_shuff.append([])
            
            for tt in ttypes:
                trialInfos[tt].append([])
                trialInfos_shuff[tt].append([])
            
            for cp in checkpoints:
                for tt in ttypes:
                    for ll in (1,2,):
                        vecs[cp][tt][ll].append([])
                        projs[cp][tt][ll].append([])
                        projsAll[cp][tt][ll].append([])
                        
                        vecs_shuff[cp][tt][ll].append([])
                        projs_shuff[cp][tt][ll].append([])
                        projsAll_shuff[cp][tt][ll].append([])
                        
            
            idxT,_ = f_subspace.split_set(np.arange(dataN.shape[0]), frac=fracBoots, ranseed=nbt)
            dataT = dataN[idxT,:,:]
            trialInfoT = test_Info.loc[idxT,:].reset_index(drop=True)
            trialInfoT['locs'] = trialInfoT['loc1'].astype(str) + '_' + trialInfoT['loc2'].astype(str)
                        
            trialInfoT = trialInfoT.rename(columns={'ttype':'type'})
            
            adaptPCA_T = None if (adaptPCA is None) else adaptPCA[n][nbt]
            adaptEVR_T = None if (adaptEVR is None) else adaptEVR[n][nbt]
            
            # generate item-specific subspace vectors
            vecs_D, projs_D, projsAll_D, _, trialInfos_D, _, _, evr_1st, pca1 = f_subspace.plane_fitting_analysis(dataT, trialInfoT, pca_tWinX, checkpoints, tRange, avgInterval, locs, ttypes, 
                                                                                                                  adaptPCA=adaptPCA_T, adaptEVR = adaptEVR_T, toPlot=toPlot, avgMethod = avgMethod) #Dummy, decode_method = decode_method
            
            pca1s[nbt] += [pca1]
            
            for tt in ttypes:
                trialInfos[tt][nbt] = trialInfos_D[tt]
                
            for cp in checkpoints:
                for tt in ttypes:
                    for ll in (1,2,):
                        vecs[cp][tt][ll][nbt] = vecs_D[cp][tt][ll]
                        projs[cp][tt][ll][nbt] = projs_D[cp][tt][ll]
                        projsAll[cp][tt][ll][nbt] = projsAll_D[cp][tt][ll]
                        
            
            
            adaptPCA_shuffT = pca1 if (adaptPCA is None) else adaptPCA_T
            adaptEVR_shuffT = evr_1st if (adaptEVR is None) else adaptEVR_T
            
            print(f'EVRs: {evr_1st.round(5)}')
            evrs_1st[nbt,:] = evr_1st
            
            for nperm in range(nPerms):
                
                # generate item-specific subspace vectors using shuffled-labels to get null distribution
                trialInfoT_shuff = trialInfoT.copy().sample(frac=1).reset_index(drop=True) # shuffle trials
                vecs_D_shuff, projs_D_shuff, projsAll_D_shuff, _, trialInfos_D_shuff, _, _, _, pca1_shuff = f_subspace.plane_fitting_analysis(dataT, trialInfoT_shuff, pca_tWinX, checkpoints, tRange, avgInterval, locs, ttypes, 
                                                                                                                                              adaptPCA=adaptPCA_shuffT, adaptEVR = adaptEVR_shuffT, toPlot=toPlot, avgMethod = avgMethod) #Dummy, decode_method = decode_method 
                
                pca1s_shuff[nbt] += [pca1_shuff]
                
                for tt in ttypes:
                    trialInfos_shuff[tt][nbt] += [trialInfos_D_shuff[tt]]
                    
                for cp in checkpoints:
                    for tt in ttypes: 
                        for ll in (1,2,):
                            vecs_shuff[cp][tt][ll][nbt] += [vecs_D_shuff[cp][tt][ll]]
                            projs_shuff[cp][tt][ll][nbt] += [projs_D_shuff[cp][tt][ll]]
                            projsAll_shuff[cp][tt][ll][nbt] += [projsAll_D_shuff[cp][tt][ll]]
                            
        vecs_All += [vecs]
        projs_All += [projs]
        projsAll_All += [projsAll]
        trialInfos_All += [trialInfos]
        
        pca1s_All += [pca1s]

        vecs_shuff_All += [vecs_shuff]
        projs_shuff_All += [projs_shuff]
        projsAll_shuff_All += [projsAll_shuff]
        trialInfos_shuff_All += [trialInfos_shuff]
        pca1s_shuff_All += [pca1s_shuff]
        
        evrs_1st_All += [evrs_1st]
        
        del modelD
        torch.cuda.empty_cache()
        gc.collect()

    return vecs_All, projs_All, projsAll_All, trialInfos_All, pca1s_All, evrs_1st_All, vecs_shuff_All, projs_shuff_All, projsAll_shuff_All, trialInfos_shuff_All, pca1s_shuff_All


#%%
def generate_choiceVectors(models_dict, trialInfo, X_, Y_, tRange, locs = (0,1,2,3), ttypes = (1,2), dt = 10, bins = 50, 
                           adaptPCA=None, adaptEVR = None, pca_tWins = ((300,1300), (1600,2600),), avgMethod='conditional_time', 
                           choice_tRange = (2100,2600), nBoots = 1, fracBoots = 1.0, nPerms = 100, 
                           toPlot=False, toplot_samples = (0,), sequence=(0,1,3,2), plotlayout = (0,1,2,3), 
                           indScatters=False, legend_on=False, gau_kappa = 2.0,
                           plot_traj=True, plot3d = True, 
                           traj_checkpoints=(1300,2600), traj_start=1300, traj_end=2600, 
                           label = '', hideLocs = (), hideType = (), normalizeMinMax = False, separatePlot = True,
                           savefig=False, save_path=''):
    
    """
    Compute the readout subspace vectors for all models
    
    Vector calculation parameters:
    adaptPCA: if applicable, adapt the PCA transformation from existing matrices
    adaptEVR: same as adaptPCA
    pca_tWins: time windows to compute PCA
    nBoots: number of bootstraps for each model
    fracBoots: fraction of data to use per bootstrap for each model
    nPerms: number of permutations within each bootstrap for each model to create null distribution
    
    Plotting Parameters:
    toPlot: whether to plot the results
    toplot_samples: indices of models to plot
    sequence: configuration sequence used for normal vector direction correction
    plotlayout: layout of subplots
    indScatters: whether to plot individual scatters
    legend_on: whether to plot legend
    gau_kappa: kappa parameter for Gaussian kernel used for line smoothing
    plot_traj: whether to plot trajectory
    plot3d: whether to plot 3D or 2D spaces
    traj_checkpoints: checkpoints to plot trajectory
    traj_start: start of trajectory
    traj_end: end of trajectory
    hideLocs: locations to hide
    hideType: types to hide
    normalizeMinMax: whether to normalize to min-max range
    separatePlot: whether to plot each condition (by Loc1) in a separate plot
    """

    epsilon = 1e-7
    
    nIters = len(models_dict)
    
    vecs_C_All = []
    projs_C_All = []
    projsAll_C_All = []
    trialInfos_C_All = []
    data_3pc_C_All = []
    pca1s_C_All = []

    vecs_C_shuff_All = []
    projs_C_shuff_All = []
    projsAll_C_shuff_All = []
    trialInfos_C_shuff_All = []
    data_3pc_C_shuff_All = []
    pca1s_C_shuff_All = []
    
    evrs_C_1st_All = []
    evrs_C_2nd_All = []
    
    
    _, _, _, test_setID, test_X, _ = f_simulation.split_dataset_balance(X_, Y_, trialInfo, frac = 0.5)
    
    test_Info = trialInfo.loc[test_setID,:].reset_index(drop = True)
    test_set = test_X.cpu().numpy()

    test_Info.loc1 = test_Info.loc1.astype(int)
    test_Info.loc2 = test_Info.loc2.astype(int)
    test_Info.choice = test_Info.choice.astype(int)
    
    # specify applied time window of pca
    pca_tWin = np.hstack([np.arange(i[0],i[1],dt, dtype = int) for i in pca_tWins]).tolist() #
    pca_tWinX = [tRange.tolist().index(i) for i in pca_tWin]
    
    
    for n in range(nIters):
        tplt = toPlot if n in toplot_samples else False
        
        # initialization
        vecs_C = []
        projs_C = []
        projsAll_C = []
        trialInfos_C = []
        data_3pc_C = []
        pca1s_C = []

        vecs_C_shuff = []
        projs_C_shuff = []
        projsAll_C_shuff = []
        trialInfos_C_shuff = []
        data_3pc_C_shuff = []
        pca1s_C_shuff = []

        evrs_1st = np.zeros((nBoots, 3))
        evrs_2nd = np.zeros((nBoots, 2))
        
        modelD = models_dict[n]['rnn']
        
        hidden_states, _ = modelD(test_X)
        hidden_states = hidden_states.data.cpu().detach().numpy()
        dataN = hidden_states.swapaxes(1,2)
        
        
        for ch in range(dataN.shape[1]):
            dataN[:,ch,:] = (dataN[:,ch,:] - dataN[:,ch,:].mean() + epsilon) / (dataN[:,ch,:].std() + epsilon) #standard scaler
        
        for nbt in range(nBoots):
            
            # append to store each iteration separately
            vecs_C.append([])
            projs_C.append([])
            projsAll_C.append([])
            trialInfos_C.append([])
            data_3pc_C.append([])
            pca1s_C.append([])
            
            vecs_C_shuff.append([])
            projs_C_shuff.append([])
            projsAll_C_shuff.append([])
            trialInfos_C_shuff.append([])
            data_3pc_C_shuff.append([])
            pca1s_C_shuff.append([])

            idxT,_ = f_subspace.split_set(np.arange(dataN.shape[0]), frac=fracBoots, ranseed=nbt)
            dataT = dataN[idxT,:,:]
            trialInfoT = test_Info.loc[idxT,:].reset_index(drop=True)
            trialInfoT['locs'] = trialInfoT['loc1'].astype(str) + '_' + trialInfoT['loc2'].astype(str)
            
            trialInfoT = trialInfoT.rename(columns={'ttype':'type'})
            
            adaptPCA_T = None if (adaptPCA is None) else adaptPCA[n][nbt]
            adaptEVR_T = None if (adaptEVR is None) else adaptEVR[n][nbt]
            
            # generate readout subspace vectors
            vecs_CT, projs_CT, projsAll_CT, _, trialInfo_CT, data_3pc_CT, _, evr_1stT, pca1_C, evr_2C = f_subspace.planeC_fitting_analysis(dataT, trialInfoT, pca_tWinX, tRange, choice_tRange, locs, ttypes, 
                                                                                                                                   adaptPCA=adaptPCA_T, adaptEVR=adaptEVR_T,
                                                                                                                                   toPlot=tplt, avgMethod = avgMethod, plot_traj=plot_traj, traj_checkpoints=traj_checkpoints, plot3d = plot3d, plotlayout=plotlayout, indScatters=indScatters,
                                                                                                                                   traj_start=traj_start, sequence=sequence, traj_end=traj_end, region_label=label, 
                                                                                                                                   savefig=savefig, save_path=save_path,legend_on=legend_on, gau_kappa=gau_kappa,
                                                                                                                                   hideLocs=hideLocs, hideType=hideType, normalizeMinMax = normalizeMinMax, separatePlot=separatePlot) #Dummy, decode_method = decode_method
            
            
            # smooth to 50ms bins
            ntrialT, ncellT, ntimeT = data_3pc_CT.shape
            data_3pc_CT_smooth = np.mean(data_3pc_CT.reshape(ntrialT, ncellT, int(ntimeT/(bins/dt)),int(bins/dt)),axis = 3)  
            
            # store results
            vecs_C[nbt] = vecs_CT
            projs_C[nbt] = projs_CT
            projsAll_C[nbt] = projsAll_CT
            trialInfos_C[nbt] = trialInfo_CT
            data_3pc_C[nbt] = data_3pc_CT_smooth
            
            pca1s_C[nbt] = pca1_C
            
            print(f'EVRs: {evr_1stT.round(5)}')
            evrs_1st[nbt,:] = evr_1stT
            evrs_2nd[nbt,:] = evr_2C

            
            adaptPCA_shuffT = pca1_C if (adaptPCA is None) else adaptPCA_T
            adaptEVR_shuffT = evr_1stT if (adaptEVR is None) else adaptEVR_T
            
            for nperm in range(nPerms):

                # generate readout subspace vectors using shuffled-labels to get null distribution
                trialInfoT_shuff = trialInfoT.copy().sample(frac=1).reset_index(drop=True) # shuffle trials
                vecs_CT_shuff, projs_CT_shuff, projsAll_CT_shuff, _, trialInfo_CT_shuff, data_3pc_CT_shuff, _, _, pca1_C_shuff, _ = f_subspace.planeC_fitting_analysis(dataT, trialInfoT_shuff, pca_tWinX, tRange, choice_tRange, locs, ttypes, 
                                                                                                                                                                    adaptPCA=adaptPCA_shuffT, adaptEVR = adaptEVR_shuffT, toPlot=False, avgMethod = avgMethod) #Dummy, decode_method = decode_method 
                
                # smooth to 50ms bins
                ntrialT, ncellT, ntimeT = data_3pc_CT_shuff.shape
                data_3pc_CT_smooth_shuff = np.mean(data_3pc_CT_shuff.reshape(ntrialT, ncellT, int(ntimeT/(bins/dt)),int(bins/dt)),axis = 3)
                
                vecs_C_shuff[nbt] += [vecs_CT_shuff]
                projs_C_shuff[nbt] += [projs_CT_shuff]
                projsAll_C_shuff[nbt] += [projsAll_CT_shuff]
                trialInfos_C_shuff[nbt] += [trialInfo_CT_shuff]
                data_3pc_C_shuff[nbt] += [data_3pc_CT_smooth_shuff]
                
                pca1s_C_shuff[nbt] += [pca1_C_shuff]
        
        vecs_C_All += [vecs_C]
        projs_C_All += [projs_C]
        projsAll_C_All += [projsAll_C]
        trialInfos_C_All += [trialInfos_C]
        data_3pc_C_All += [data_3pc_C]
        pca1s_C_All += [pca1s_C]

        vecs_C_shuff_All += [vecs_C_shuff]
        projs_C_shuff_All += [projs_C_shuff]
        projsAll_C_shuff_All += [projsAll_C_shuff]
        trialInfos_C_shuff_All += [trialInfos_C_shuff]
        data_3pc_C_shuff_All += [data_3pc_C_shuff]
        pca1s_C_shuff_All += [pca1s_C_shuff]
        
        evrs_C_1st_All += [evrs_1st]
        evrs_C_2nd_All += [evrs_2nd]
        
        del modelD
        torch.cuda.empty_cache()
        gc.collect()
    
    return vecs_C_All, projs_C_All, projsAll_C_All, trialInfos_C_All, data_3pc_C_All, pca1s_C_All, evrs_C_1st_All, evrs_C_2nd_All, vecs_C_shuff_All, projs_C_shuff_All, projsAll_C_shuff_All, trialInfos_C_shuff_All, data_3pc_C_shuff_All, pca1s_C_shuff_All  
    
#%% item decodability by projections on item-specific subspace
def itemInfo_by_plane(geoms_valid, checkpoints, locs = (0,1,2,3), ttypes = (1,2), Y_columnsLabels = ['choice','type','loc1','loc2','locX'], 
                      toDecode_labels1 = 'loc1', toDecode_labels2 = 'loc2', shuff_excludeInv = False, nPerms = 10, infoMethod = 'lda'):
    
    """
    Calculate decodability of item 1 and item 2 by projections on the item-specific plane (defined by the vectors) for each model.
    
    infoMethod: 'lda' (default) or 'omega2'
    """

    vecs, projs, projsAll, trialInfos = geoms_valid
    
    nBoots = len(trialInfos[1])
    
    decode_proj1_3d, decode_proj2_3d = {},{}
    decode_proj1_shuff_all_3d, decode_proj2_shuff_all_3d = {},{}
    
    
    for tt in ttypes:

        # initialization
        decode_proj1T_3d = np.zeros((nBoots, nPerms, len(checkpoints))) # pca1st 3d coordinates
        decode_proj2T_3d = np.zeros((nBoots, nPerms, len(checkpoints)))
        
        # shuff
        decode_proj1T_3d_shuff = np.zeros((nBoots, nPerms, len(checkpoints)))
        decode_proj2T_3d_shuff = np.zeros((nBoots, nPerms, len(checkpoints)))
        
        
        for nbt in range(nBoots):
            trialInfoT = trialInfos[tt][nbt]
            
            for npm in range(nPerms):
                
                # labels
                Y = trialInfoT.loc[:,Y_columnsLabels].values
                ntrial = len(trialInfoT)
                
                # shuff
                toDecode_X1 = Y_columnsLabels.index(toDecode_labels1)
                toDecode_X2 = Y_columnsLabels.index(toDecode_labels2)
                
                ### labels: ['locKey','locs','type','loc1','loc2','locX']
                label1 = Y[:,toDecode_X1].astype('int') 
                label2 = Y[:,toDecode_X2].astype('int') 
                
                
                toDecode_labels1_inv = f_stats.inversed_label(Y_columnsLabels, toDecode_labels1, tt)
                toDecode_X1_inv = Y_columnsLabels.index(toDecode_labels1_inv)
                label1_inv = Y[:,toDecode_X1_inv]
                
                
                toDecode_labels2_inv = f_stats.inversed_label(Y_columnsLabels, toDecode_labels2, tt)
                toDecode_X2_inv = Y_columnsLabels.index(toDecode_labels2_inv)
                label2_inv = Y[:,toDecode_X2_inv]
                
                if shuff_excludeInv:
                    # except for the inverse ones
                    label1_shuff = np.full_like(label1_inv,9, dtype=int)
                    label2_shuff = np.full_like(label2_inv,9, dtype=int)
                    
                    for ni1, i1 in enumerate(label1_inv.astype(int)):
                        label1_shuff[ni1] = np.random.choice(np.array(locs)[np.array(locs)!=i1]).astype(int)

                    for ni2, i2 in enumerate(label2_inv.astype(int)):
                        label2_shuff[ni2] = np.random.choice(np.array(locs)[np.array(locs)!=i2]).astype(int)
                    
                    
                    
                else:
                    label1_shuff = np.random.permutation(label1)
                    label2_shuff = np.random.permutation(label2)
                
                trialInfoT_shuff = trialInfoT.copy()
                trialInfoT_shuff[toDecode_labels1] = label1_shuff
                trialInfoT_shuff[toDecode_labels2] = label2_shuff
                    
                # calculate item1 and item2 decodability at each time bin
                for nc,cp in enumerate(checkpoints):
                    vecs1, vecs2 = vecs[cp][tt][1][nbt], vecs[cp][tt][2][nbt]
                    projs1, projs2 = projs[cp][tt][1][nbt], projs[cp][tt][2][nbt]
                    projs1_allT_3d, projs2_allT_3d = projsAll[cp][tt][1][nbt], projsAll[cp][tt][2][nbt]
                    
                    info1_3d, _ = f_subspace.plane_decodability(vecs1, projs1, projs1_allT_3d, trialInfoT, 'loc1', method = infoMethod)
                    info2_3d, _ = f_subspace.plane_decodability(vecs2, projs2, projs2_allT_3d, trialInfoT, 'loc2', method = infoMethod)
                    
                    info1_3d_shuff, _ = f_subspace.plane_decodability(vecs1, projs1, projs1_allT_3d, trialInfoT_shuff, 'loc1', method = infoMethod, sequence=(0,1,3,2))
                    info2_3d_shuff, _ = f_subspace.plane_decodability(vecs2, projs2, projs2_allT_3d, trialInfoT_shuff, 'loc2', method = infoMethod, sequence=(0,1,3,2))
                    
                    
                    decode_proj1T_3d[nbt,npm,nc] = info1_3d
                    decode_proj2T_3d[nbt,npm,nc] = info2_3d
                    
                    decode_proj1T_3d_shuff[nbt,npm,nc] = info1_3d_shuff
                    decode_proj2T_3d_shuff[nbt,npm,nc] = info2_3d_shuff
        
        decode_proj1_3d[tt] = decode_proj1T_3d
        decode_proj2_3d[tt] = decode_proj2T_3d
        decode_proj1_shuff_all_3d[tt] = decode_proj1T_3d_shuff
        decode_proj2_shuff_all_3d[tt] = decode_proj2T_3d_shuff

    
    info3d = (decode_proj1_3d, decode_proj2_3d)
    info3d_shuff = (decode_proj1_shuff_all_3d, decode_proj2_shuff_all_3d)
    
    return info3d, info3d_shuff

#%% item code tranferability between item-specific subspaces
def itemInfo_by_plane_Trans(geoms_valid, checkpoints, locs = (0,1,2,3), ttypes = (1,2), Y_columnsLabels = ['choice','type','loc1','loc2','locX'], 
                      toDecode_labels1 = 'loc1', toDecode_labels2 = 'loc2', shuff_excludeInv = False, nPerms = 10, infoMethod = 'lda'):
    
    vecs, projs, projsAll, trialInfos = geoms_valid
    
    nBoots = len(trialInfos[1])
    
    performanceX_12, performanceX_21 = {},{} # train with item1, test with item2; vice versa
    performanceX_12_shuff, performanceX_21_shuff = {},{}
        
    
    for tt in ttypes:

        # initialization
        performanceX_12T = np.zeros((nBoots, nPerms, len(checkpoints), len(checkpoints)))
        performanceX_21T = np.zeros((nBoots, nPerms, len(checkpoints), len(checkpoints)))
        
        # shuff
        performanceX_12T_shuff = np.zeros((nBoots, nPerms, len(checkpoints), len(checkpoints)))
        performanceX_21T_shuff = np.zeros((nBoots, nPerms, len(checkpoints), len(checkpoints)))
        
        
        for nbt in range(nBoots):
            trialInfoT = trialInfos[tt][nbt]
            
            
            for npm in range(nPerms):
                
                # labels
                Y = trialInfoT.loc[:,Y_columnsLabels].values
                ntrial = len(trialInfoT)
                
                # shuff
                toDecode_X1 = Y_columnsLabels.index(toDecode_labels1)
                toDecode_X2 = Y_columnsLabels.index(toDecode_labels2)
                
                ### labels: ['locKey','locs','type','loc1','loc2','locX']
                label1 = Y[:,toDecode_X1].astype('int') #.astype('str') # locKey
                label2 = Y[:,toDecode_X2].astype('int') #.astype('str') # locKey
                
                
                toDecode_labels1_inv = f_stats.inversed_label(Y_columnsLabels, toDecode_labels1, tt)
                toDecode_X1_inv = Y_columnsLabels.index(toDecode_labels1_inv)
                label1_inv = Y[:,toDecode_X1_inv]
                
                
                toDecode_labels2_inv = f_stats.inversed_label(Y_columnsLabels, toDecode_labels2, tt)
                toDecode_X2_inv = Y_columnsLabels.index(toDecode_labels2_inv)
                label2_inv = Y[:,toDecode_X2_inv]
                
                if shuff_excludeInv:
                    # except for the inverse ones
                    label1_shuff = np.full_like(label1_inv,9, dtype=int)
                    label2_shuff = np.full_like(label2_inv,9, dtype=int)
                    
                    for ni1, i1 in enumerate(label1_inv.astype(int)):
                        label1_shuff[ni1] = np.random.choice(np.array(locs)[np.array(locs)!=i1]).astype(int)

                    for ni2, i2 in enumerate(label2_inv.astype(int)):
                        label2_shuff[ni2] = np.random.choice(np.array(locs)[np.array(locs)!=i2]).astype(int)
                    
                    
                else:
                    label1_shuff = np.random.permutation(label1)
                    label2_shuff = np.random.permutation(label2)
                
                trialInfoT_shuff = trialInfoT.copy()
                trialInfoT_shuff[toDecode_labels1] = label1_shuff
                trialInfoT_shuff[toDecode_labels2] = label2_shuff
                    
                # per time bin
                for nc,cp in enumerate(checkpoints):
                    for nc_,cp_ in enumerate(checkpoints):
                                                
                        vecs1, vecs2 = vecs[cp][tt][1][nbt], vecs[cp_][tt][2][nbt]
                        projs1, projs2 = projs[cp][tt][1][nbt], projs[cp_][tt][2][nbt]
                        projs1_allT_3d, projs2_allT_3d = projsAll[cp][tt][1][nbt], projsAll[cp_][tt][2][nbt]
                        
                        geom1 = (vecs1, projs1, projs1_allT_3d, trialInfoT, 'loc1')
                        geom2 = (vecs2, projs2, projs2_allT_3d, trialInfoT, 'loc2')
                                                
                        info12, _ = f_subspace.plane_decodability_trans(geom1, geom2) # train decoders of item1 with item1 subspace projections, test to decode item2 with item2 subspace projection
                        info21, _ = f_subspace.plane_decodability_trans(geom2, geom1) # vice versa
                        
                        geom1_shuff = (vecs1, projs1, projs1_allT_3d, trialInfoT_shuff, 'loc1')
                        geom2_shuff = (vecs2, projs2, projs2_allT_3d, trialInfoT_shuff, 'loc2')
                        
                        info12_shuff, _ = f_subspace.plane_decodability_trans(geom1_shuff, geom2_shuff)
                        info21_shuff, _ = f_subspace.plane_decodability_trans(geom2_shuff, geom1_shuff)
                        
                        performanceX_12T[nbt,npm,nc,nc_] = info12
                        performanceX_21T[nbt,npm,nc_,nc] = info21
                        
                        performanceX_12T_shuff[nbt,npm,nc,nc_] = info12_shuff
                        performanceX_21T_shuff[nbt,npm,nc_,nc] = info21_shuff
                        
        performanceX_12[tt] = performanceX_12T
        performanceX_21[tt] = performanceX_21T
        
        performanceX_12_shuff[tt] = performanceX_12T_shuff
        performanceX_21_shuff[tt] = performanceX_21T_shuff
        
    
    infoTrans = (performanceX_12, performanceX_21)
    infoTrans_shuff = (performanceX_12_shuff, performanceX_21_shuff)
    
    return infoTrans, infoTrans_shuff

#%% choice-item code transferability between corresponding item-specific subspaces
def chioceInfo_by_plane_Trans(geoms_valid, checkpoints, locs = (0,1,2,3), ttypes = (1,2), Y_columnsLabels = ['choice','type','loc1','loc2','locX'], 
                      toDecode_labels1c = 'loc2', toDecode_labels2c = 'loc1', toDecode_labels1nc = 'loc1', toDecode_labels2nc = 'loc2', 
                      shuff_excludeInv = False, nPerms = 10, infoMethod = 'lda'):
    
    vecs, projs, projsAll, trialInfos = geoms_valid
    
    nBoots = len(trialInfos[1])
    
    performanceX_rdc, performanceX_drc = np.zeros((nBoots, nPerms, len(checkpoints), len(checkpoints))), np.zeros((nBoots, nPerms, len(checkpoints), len(checkpoints)))
    performanceX_rdnc, performanceX_drnc = np.zeros((nBoots, nPerms, len(checkpoints), len(checkpoints))), np.zeros((nBoots, nPerms, len(checkpoints), len(checkpoints)))
    performanceX_rdc_shuff, performanceX_drc_shuff = np.zeros((nBoots, nPerms, len(checkpoints), len(checkpoints))), np.zeros((nBoots, nPerms, len(checkpoints), len(checkpoints)))
    performanceX_rdnc_shuff, performanceX_drnc_shuff = np.zeros((nBoots, nPerms, len(checkpoints), len(checkpoints))), np.zeros((nBoots, nPerms, len(checkpoints), len(checkpoints)))      
    
    
    for nbt in range(nBoots):
        
        trialInfoT1 = trialInfos[1][nbt]
        trialInfoT2 = trialInfos[2][nbt]
        
        for npm in range(nPerms):
            
            # labels
            Y1 = trialInfoT1.loc[:,Y_columnsLabels].values
            Y2 = trialInfoT2.loc[:,Y_columnsLabels].values
            ntrial1 = len(trialInfoT1)
            ntrial2 = len(trialInfoT2)
            
            # shuff
            toDecode_X1c = Y_columnsLabels.index(toDecode_labels1c)
            toDecode_X2c = Y_columnsLabels.index(toDecode_labels2c)
            toDecode_X1nc = Y_columnsLabels.index(toDecode_labels1nc)
            toDecode_X2nc = Y_columnsLabels.index(toDecode_labels2nc)
            
            ### labels: ['locKey','locs','type','loc1','loc2','locX']
            label_c1 = Y1[:,toDecode_X1c].astype('int') #.astype('str') # locKey
            label_c2 = Y2[:,toDecode_X2c].astype('int') #.astype('str') # locKey
            
            label_nc1 = Y1[:,toDecode_X1nc].astype('int') #.astype('str') # locKey
            label_nc2 = Y2[:,toDecode_X2nc].astype('int') #.astype('str') # locKey
            
            
            toDecode_labels_c1_inv = f_stats.inversed_label(Y_columnsLabels, toDecode_labels1c, 1)
            toDecode_X_c1_inv = Y_columnsLabels.index(toDecode_labels_c1_inv)
            label_c1_inv = Y1[:,toDecode_X_c1_inv]
            
            
            toDecode_labels_c2_inv = f_stats.inversed_label(Y_columnsLabels, toDecode_labels2c, 2)
            toDecode_X_c2_inv = Y_columnsLabels.index(toDecode_labels_c2_inv)
            label_c2_inv = Y2[:,toDecode_X_c2_inv]
            
            
            toDecode_labels_nc1_inv = f_stats.inversed_label(Y_columnsLabels, toDecode_labels1nc, 1)
            toDecode_X_nc1_inv = Y_columnsLabels.index(toDecode_labels_nc1_inv)
            label_nc1_inv = Y1[:,toDecode_X_nc1_inv]
            
            
            toDecode_labels_nc2_inv = f_stats.inversed_label(Y_columnsLabels, toDecode_labels2nc, 2)
            toDecode_X_nc2_inv = Y_columnsLabels.index(toDecode_labels_nc2_inv)
            label_nc2_inv = Y2[:,toDecode_X_nc2_inv]
            
            if shuff_excludeInv:
                # except for the inverse ones
                label_c1_shuff = np.full_like(label_c1_inv,9, dtype=int)
                label_c2_shuff = np.full_like(label_c2_inv,9, dtype=int)
                label_nc1_shuff = np.full_like(label_nc1_inv,9, dtype=int)
                label_nc2_shuff = np.full_like(label_nc2_inv,9, dtype=int)
                
                for ni1, i1 in enumerate(label_c1_inv.astype(int)):
                    label_c1_shuff[ni1] = np.random.choice(np.array(locs)[np.array(locs)!=i1]).astype(int)

                for ni2, i2 in enumerate(label_c2_inv.astype(int)):
                    label_c2_shuff[ni2] = np.random.choice(np.array(locs)[np.array(locs)!=i2]).astype(int)
                
                for nj1, j1 in enumerate(label_nc1_inv.astype(int)):
                    label_nc1_shuff[nj1] = np.random.choice(np.array(locs)[np.array(locs)!=j1]).astype(int)

                for nj2, j2 in enumerate(label_nc2_inv.astype(int)):
                    label_nc2_shuff[nj2] = np.random.choice(np.array(locs)[np.array(locs)!=j2]).astype(int)
                
                
            else:
                
                label_c1_shuff = np.random.permutation(label_c1)
                label_c2_shuff = np.random.permutation(label_c2)
                label_nc1_shuff = np.random.permutation(label_nc1)
                label_nc2_shuff = np.random.permutation(label_nc2)
            
            trialInfoT1_shuff, trialInfoT2_shuff = trialInfoT1.copy(), trialInfoT2.copy()
            trialInfoT1_shuff[toDecode_labels1c] = label_c1_shuff
            trialInfoT2_shuff[toDecode_labels2c] = label_c2_shuff
            trialInfoT1_shuff[toDecode_labels1nc] = label_nc1_shuff
            trialInfoT2_shuff[toDecode_labels2nc] = label_nc2_shuff
                
            # per time bin
            for nc,cp in enumerate(checkpoints):
                for nc_,cp_ in enumerate(checkpoints):
                    
                    ###########
                    # choices #
                    ###########
                    
                    vecs1c, vecs2c = vecs[cp][1][2][nbt], vecs[cp_][2][1][nbt]
                    projs1c, projs2c = projs[cp][1][2][nbt], projs[cp_][2][1][nbt]
                    projs1c_allT_3d, projs2c_allT_3d = projsAll[cp][1][2][nbt], projsAll[cp_][2][1][nbt]
                    
                    geom1c = (vecs1c, projs1c, projs1c_allT_3d, trialInfoT1, toDecode_labels1c)
                    geom2c = (vecs2c, projs2c, projs2c_allT_3d, trialInfoT2, toDecode_labels2c)
                                            
                    infordc, _ = f_subspace.plane_decodability_trans(geom1c, geom2c)
                    infodrc, _ = f_subspace.plane_decodability_trans(geom2c, geom1c)
                    
                    geom1c_shuff = (vecs1c, projs1c, projs1c_allT_3d, trialInfoT1_shuff, toDecode_labels1c)
                    geom2c_shuff = (vecs2c, projs2c, projs2c_allT_3d, trialInfoT2_shuff, toDecode_labels2c)
                    
                    infordc_shuff, _ = f_subspace.plane_decodability_trans(geom1c_shuff, geom2c_shuff)
                    infodrc_shuff, _ = f_subspace.plane_decodability_trans(geom2c_shuff, geom1c_shuff)
                    
                    performanceX_rdc[nbt,npm,nc,nc_] = infordc
                    performanceX_drc[nbt,npm,nc_,nc] = infodrc
                    performanceX_rdc_shuff[nbt,npm,nc,nc_] = infordc_shuff
                    performanceX_drc_shuff[nbt,npm,nc_,nc] = infodrc_shuff
                    
                    #############
                    # nonchoice #
                    #############
                    
                    vecs1nc, vecs2nc = vecs[cp][1][1][nbt], vecs[cp_][2][2][nbt]
                    projs1nc, projs2nc = projs[cp][1][1][nbt], projs[cp_][2][2][nbt]
                    projs1nc_allT_3d, projs2nc_allT_3d = projsAll[cp][1][1][nbt], projsAll[cp_][2][2][nbt]
                    
                    geom1nc = (vecs1nc, projs1nc, projs1nc_allT_3d, trialInfoT1, toDecode_labels1nc)
                    geom2nc = (vecs2nc, projs2nc, projs2nc_allT_3d, trialInfoT2, toDecode_labels2nc)
                                            
                    infordnc, _ = f_subspace.plane_decodability_trans(geom1nc, geom2nc)
                    infodrnc, _ = f_subspace.plane_decodability_trans(geom2nc, geom1nc)
                    
                    geom1nc_shuff = (vecs1nc, projs1nc, projs1nc_allT_3d, trialInfoT1_shuff, toDecode_labels1nc)
                    geom2nc_shuff = (vecs2nc, projs2nc, projs2nc_allT_3d, trialInfoT2_shuff, toDecode_labels2nc)
                    
                    infordnc_shuff, _ = f_subspace.plane_decodability_trans(geom1nc_shuff, geom2nc_shuff)
                    infodrnc_shuff, _ = f_subspace.plane_decodability_trans(geom2nc_shuff, geom1nc_shuff)
                    
                    performanceX_rdnc[nbt,npm,nc,nc_] = infordnc
                    performanceX_drnc[nbt,npm,nc_,nc] = infodrnc
                    performanceX_rdnc_shuff[nbt,npm,nc,nc_] = infordnc_shuff
                    performanceX_drnc_shuff[nbt,npm,nc_,nc] = infodrnc_shuff
                                        
    
    
    infoTransc = (performanceX_rdc, performanceX_drc)
    infoTransc_shuff = (performanceX_rdc_shuff, performanceX_drc_shuff)
    infoTransnc = (performanceX_rdnc, performanceX_drnc)
    infoTransnc_shuff = (performanceX_rdnc_shuff, performanceX_drnc_shuff)
    
    return infoTransc, infoTransc_shuff, infoTransnc, infoTransnc_shuff

#%% calculate angle and alignment measures between item-specific subspace pairs
def get_angleAlignment_itemPairs(geoms_valid, geoms_shuff, checkpoints, locs = (0,1,2,3), ttypes = (1,2), sequence=(0,1,3,2)):
    
    vecs, projs, projsAll, trialInfos = geoms_valid
    vecs_shuff, projs_shuff, projsAll_shuff, trialInfos_shuff = geoms_shuff
    
    nBoots = len(trialInfos[1])
    nPerms = len(trialInfos_shuff[1][0])

    cosTheta_11, cosTheta_12, cosTheta_22 = {},{},{}
    cosPsi_11, cosPsi_12, cosPsi_22 = {},{},{}
    
    cosTheta_11_shuff, cosTheta_12_shuff, cosTheta_22_shuff = {},{},{}
    cosPsi_11_shuff, cosPsi_12_shuff, cosPsi_22_shuff = {},{},{}
    
    for tt in ttypes: #Dummy
        
        cosTheta_11T = np.zeros((nBoots, len(checkpoints), len(checkpoints)))
        cosTheta_22T = np.zeros((nBoots, len(checkpoints), len(checkpoints)))
        cosTheta_12T = np.zeros((nBoots, len(checkpoints), len(checkpoints)))
        
        cosPsi_11T = np.zeros((nBoots, len(checkpoints), len(checkpoints)))
        cosPsi_22T = np.zeros((nBoots, len(checkpoints), len(checkpoints)))
        cosPsi_12T = np.zeros((nBoots, len(checkpoints), len(checkpoints)))
        
        cosTheta_11T_shuff = np.zeros((nBoots, nPerms, len(checkpoints), len(checkpoints)))
        cosTheta_22T_shuff = np.zeros((nBoots, nPerms, len(checkpoints), len(checkpoints)))
        cosTheta_12T_shuff = np.zeros((nBoots, nPerms, len(checkpoints), len(checkpoints)))
        
        cosPsi_11T_shuff = np.zeros((nBoots, nPerms, len(checkpoints), len(checkpoints)))
        cosPsi_22T_shuff = np.zeros((nBoots, nPerms, len(checkpoints), len(checkpoints)))
        cosPsi_12T_shuff = np.zeros((nBoots, nPerms, len(checkpoints), len(checkpoints)))
        
        for nbt in range(nBoots):
            for nc, cp in enumerate(checkpoints):
                for nc_, cp_ in enumerate(checkpoints):
                    cT11, _, cP11, _, _ = f_subspace.angle_alignment_coplanar(vecs[cp][tt][1][nbt], projs[cp][tt][1][nbt], 
                                                                            vecs[cp_][tt][1][nbt], projs[cp_][tt][1][nbt], sequence=sequence)
                    cT22, _, cP22, _, _ = f_subspace.angle_alignment_coplanar(vecs[cp][tt][2][nbt], projs[cp][tt][2][nbt], 
                                                                            vecs[cp_][tt][2][nbt], projs[cp_][tt][2][nbt], sequence=sequence)
                    cT12, _, cP12, _, _ = f_subspace.angle_alignment_coplanar(vecs[cp][tt][1][nbt], projs[cp][tt][1][nbt], 
                                                                            vecs[cp_][tt][2][nbt], projs[cp_][tt][2][nbt], sequence=sequence)
                                    
                    cosTheta_11T[nbt,nc,nc_], cosTheta_22T[nbt,nc,nc_], cosTheta_12T[nbt,nc,nc_] = cT11, cT22, cT12
                    cosPsi_11T[nbt,nc,nc_], cosPsi_22T[nbt,nc,nc_], cosPsi_12T[nbt,nc,nc_] = cP11, cP22, cP12
        
            for npm in range(nPerms):
                for nc, cp in enumerate(checkpoints):
                    for nc_, cp_ in enumerate(checkpoints):
                        cT11_shuff, _, cP11_shuff, _, _ = f_subspace.angle_alignment_coplanar(vecs_shuff[cp][tt][1][nbt][npm], projs_shuff[cp][tt][1][nbt][npm], 
                                                                                vecs_shuff[cp_][tt][1][nbt][npm], projs_shuff[cp_][tt][1][nbt][npm], sequence=sequence)
                        cT22_shuff, _, cP22_shuff, _, _ = f_subspace.angle_alignment_coplanar(vecs_shuff[cp][tt][2][nbt][npm], projs_shuff[cp][tt][2][nbt][npm], 
                                                                                vecs_shuff[cp_][tt][2][nbt][npm], projs_shuff[cp_][tt][2][nbt][npm], sequence=sequence)
                        cT12_shuff, _, cP12_shuff, _, _ = f_subspace.angle_alignment_coplanar(vecs_shuff[cp][tt][1][nbt][npm], projs_shuff[cp][tt][1][nbt][npm], 
                                                                                vecs_shuff[cp_][tt][2][nbt][npm], projs_shuff[cp_][tt][2][nbt][npm], sequence=sequence)
                                        
                        cosTheta_11T_shuff[nbt,npm,nc,nc_], cosTheta_22T_shuff[nbt,npm,nc,nc_], cosTheta_12T_shuff[nbt,npm,nc,nc_] = cT11_shuff, cT22_shuff, cT12_shuff
                        cosPsi_11T_shuff[nbt,npm,nc,nc_], cosPsi_22T_shuff[nbt,npm,nc,nc_], cosPsi_12T_shuff[nbt,npm,nc,nc_] = cP11_shuff, cP22_shuff, cP12_shuff
                            
                    
        cosTheta_11[tt] = cosTheta_11T
        cosTheta_22[tt] = cosTheta_22T
        cosTheta_12[tt] = cosTheta_12T
        
        cosPsi_11[tt] = cosPsi_11T
        cosPsi_22[tt] = cosPsi_22T
        cosPsi_12[tt] = cosPsi_12T
        
        cosTheta_11_shuff[tt] = cosTheta_11T_shuff
        cosTheta_22_shuff[tt] = cosTheta_22T_shuff
        cosTheta_12_shuff[tt] = cosTheta_12T_shuff
        
        cosPsi_11_shuff[tt] = cosPsi_11T_shuff
        cosPsi_22_shuff[tt] = cosPsi_22T_shuff
        cosPsi_12_shuff[tt] = cosPsi_12T_shuff

    cosThetas_valid = (cosTheta_11, cosTheta_12, cosTheta_22)
    cosThetas_shuff = (cosTheta_11_shuff, cosTheta_12_shuff, cosTheta_22_shuff)
    cosPsis_valid = (cosPsi_11, cosPsi_12, cosPsi_22)
    cosPsis_shuff = (cosPsi_11_shuff, cosPsi_12_shuff, cosPsi_22_shuff)
    
    return cosThetas_valid, cosThetas_shuff, cosPsis_valid, cosPsis_shuff


#%% calculate angle and alignment measures between choice-item subspace pairs
def get_angleAlignment_choicePairs(geoms_valid, geoms_shuff, checkpoints, locs = (0,1,2,3), ttypes = (1,2), sequence=(0,1,3,2)):
    
    vecs, projs, projsAll, trialInfos = geoms_valid
    vecs_shuff, projs_shuff, projsAll_shuff, trialInfos_shuff = geoms_shuff
    
    nBoots = len(trialInfos[1])
    nPerms = len(trialInfos_shuff[1][0])

    cosTheta_choice, cosTheta_nonchoice = np.zeros((nBoots, len(checkpoints),)), np.zeros((nBoots, len(checkpoints),))
    cosPsi_choice, cosPsi_nonchoice = np.zeros((nBoots, len(checkpoints),)), np.zeros((nBoots, len(checkpoints),))
    
    cosTheta_choice_shuff, cosTheta_nonchoice_shuff = np.zeros((nBoots, nPerms, len(checkpoints),)), np.zeros((nBoots, nPerms, len(checkpoints),))
    cosPsi_choice_shuff, cosPsi_nonchoice_shuff = np.zeros((nBoots, nPerms, len(checkpoints),)), np.zeros((nBoots, nPerms, len(checkpoints),))
    
    for nbt in range(nBoots):
        for nc,cp in enumerate(checkpoints):
            cT_C, _, cP_C, _, _ = f_subspace.angle_alignment_coplanar(vecs[cp][1][2][nbt], projs[cp][1][2][nbt], 
                                                                      vecs[cp][2][1][nbt], projs[cp][2][1][nbt], sequence=(0,1,3,2))
            cT_NC, _, cP_NC, _, _ = f_subspace.angle_alignment_coplanar(vecs[cp][1][1][nbt], projs[cp][1][1][nbt], 
                                                                        vecs[cp][2][2][nbt], projs[cp][2][2][nbt], sequence=(0,1,3,2))
            
            cosTheta_choice[nbt,nc], cosPsi_choice[nbt,nc] = cT_C, cP_C
            cosTheta_nonchoice[nbt,nc], cosPsi_nonchoice[nbt,nc] = cT_NC, cP_NC
    
        
        for npm in range(nPerms):
            for nc,cp in enumerate(checkpoints):
                cT_C_shuff, _, cP_C_shuff, _, _ = f_subspace.angle_alignment_coplanar(vecs_shuff[cp][1][2][nbt][npm], projs_shuff[cp][1][2][nbt][npm], 
                                                                        vecs_shuff[cp][2][1][nbt][npm], projs_shuff[cp][2][1][nbt][npm], sequence=(0,1,3,2))
                cT_NC_shuff, _, cP_NC_shuff, _, _ = f_subspace.angle_alignment_coplanar(vecs_shuff[cp][1][1][nbt][npm], projs_shuff[cp][1][1][nbt][npm], 
                                                                            vecs_shuff[cp][2][2][nbt][npm], projs_shuff[cp][2][2][nbt][npm], sequence=(0,1,3,2))
                
                cosTheta_choice_shuff[nbt,npm,nc], cosPsi_choice_shuff[nbt,npm,nc] = cT_C_shuff, cP_C_shuff
                cosTheta_nonchoice_shuff[nbt,npm,nc], cosPsi_nonchoice_shuff[nbt,npm,nc] = cT_NC_shuff, cP_NC_shuff
    
    cosThetas_valid = (cosTheta_choice, cosTheta_nonchoice)
    cosThetas_shuff = (cosTheta_choice_shuff, cosTheta_nonchoice_shuff)
    cosPsis_valid = (cosPsi_choice, cosPsi_nonchoice)
    cosPsis_shuff = (cosPsi_choice_shuff, cosPsi_nonchoice_shuff)
    
    return cosThetas_valid, cosThetas_shuff, cosPsis_valid, cosPsis_shuff

#%% calculate angle and alignment between item-specific subspaces vs readout subspace
def get_angleAlignment_itemRead(geoms_valid, geoms_shuff, geomsC_valid, geomsC_shuff, checkpoints, locs = (0,1,2,3), ttypes = (1,2), sequence=(0,1,3,2)):
    
    vecs, projs, projsAll, trialInfos = geoms_valid
    vecs_shuff, projs_shuff, projsAll_shuff, trialInfos_shuff = geoms_shuff
    
    vecsC, projsC, projsAllC, trialInfosC = geomsC_valid
    vecsC_shuff, projsC_shuff, projsAllC_shuff, trialInfosC_shuff = geomsC_shuff
    
    nBoots = len(trialInfos[1])
    nPerms = len(trialInfos_shuff[1][0])

    cosTheta_1C, cosTheta_2C = {},{}
    cosPsi_1C, cosPsi_2C = {},{}
    
    cosTheta_1C_shuff, cosTheta_2C_shuff = {},{}
    cosPsi_1C_shuff, cosPsi_2C_shuff = {},{}
    
    for tt in ttypes: #Dummy
        
        cosTheta_1CT = np.zeros((nBoots, len(checkpoints),))
        cosTheta_2CT = np.zeros((nBoots, len(checkpoints),))
        
        cosPsi_1CT = np.zeros((nBoots, len(checkpoints), ))
        cosPsi_2CT = np.zeros((nBoots, len(checkpoints), ))
        
        cosTheta_1CT_shuff = np.zeros((nBoots, nPerms, len(checkpoints),))
        cosTheta_2CT_shuff = np.zeros((nBoots, nPerms, len(checkpoints), ))
        
        cosPsi_1CT_shuff = np.zeros((nBoots, nPerms, len(checkpoints),))
        cosPsi_2CT_shuff = np.zeros((nBoots, nPerms, len(checkpoints),))
        
        for nbt in range(nBoots):
            for nc, cp in enumerate(checkpoints):
                cT1C, _, cP1C, _, _ = f_subspace.angle_alignment_coplanar(vecs[cp][tt][1][nbt], projs[cp][tt][1][nbt], 
                                                                        vecsC[nbt], projsC[nbt], sequence=sequence)
                cT2C, _, cP2C, _, _ = f_subspace.angle_alignment_coplanar(vecs[cp][tt][2][nbt], projs[cp][tt][2][nbt], 
                                                                        vecsC[nbt], projsC[nbt], sequence=sequence)
                                    
                cosTheta_1CT[nbt,nc], cosTheta_2CT[nbt,nc] = cT1C, cT2C
                cosPsi_1CT[nbt,nc], cosPsi_2CT[nbt,nc] = cP1C, cP2C
        
            for npm in range(nPerms):
                for nc, cp in enumerate(checkpoints):
                    
                    cT1C_shuff, _, cP1C_shuff, _, _ = f_subspace.angle_alignment_coplanar(vecs_shuff[cp][tt][1][nbt][npm], projs_shuff[cp][tt][1][nbt][npm], 
                                                                            vecsC_shuff[nbt][npm], projsC_shuff[nbt][npm], sequence=sequence)
                    cT2C_shuff, _, cP2C_shuff, _, _ = f_subspace.angle_alignment_coplanar(vecs_shuff[cp][tt][2][nbt][npm], projs_shuff[cp][tt][2][nbt][npm], 
                                                                            vecsC_shuff[nbt][npm], projsC_shuff[nbt][npm], sequence=sequence)
                                    
                    cosTheta_1CT_shuff[nbt,npm,nc], cosTheta_2CT_shuff[nbt,npm,nc] = cT1C_shuff, cT2C_shuff
                    cosPsi_1CT_shuff[nbt,npm,nc], cosPsi_2CT_shuff[nbt,npm,nc] = cP1C_shuff, cP2C_shuff
                        
                    
        cosTheta_1C[tt] = cosTheta_1CT
        cosTheta_2C[tt] = cosTheta_2CT
        
        cosPsi_1C[tt] = cosPsi_1CT
        cosPsi_2C[tt] = cosPsi_2CT
        
        cosTheta_1C_shuff[tt] = cosTheta_1CT_shuff
        cosTheta_2C_shuff[tt] = cosTheta_2CT_shuff
        
        cosPsi_1C_shuff[tt] = cosPsi_1CT_shuff
        cosPsi_2C_shuff[tt] = cosPsi_2CT_shuff
        
    cosThetas_valid = (cosTheta_1C, cosTheta_2C)
    cosThetas_shuff = (cosTheta_1C_shuff, cosTheta_2C_shuff)
    cosPsis_valid = (cosPsi_1C, cosPsi_2C)
    cosPsis_shuff = (cosPsi_1C_shuff, cosPsi_2C_shuff)
    
    return cosThetas_valid, cosThetas_shuff, cosPsis_valid, cosPsis_shuff

#%% item decodability by the projections on the readout subspace
def itemInfo_by_planeC(geomsC_valid, locs = (0,1,2,3), ttypes = (1,2), bins = 50, dt = 10, tslice = (0,3000), Y_columnsLabels = ['choice','type','loc1','loc2','locX'], 
                   toDecode_labels1 = 'loc1', toDecode_labels2 = 'loc2', shuff_excludeInv = False, nPerms = 10, infoMethod = 'lda'):
    
    
    vecs_C, projs_C, projsAll_C, trialInfos_C, data_3pc_C = geomsC_valid
    
    nBoots = len(trialInfos_C)
    
    bins = bins
    dt = dt
    tbins = np.arange(tslice[0], tslice[1], bins)
    
    decode_proj1_3d, decode_proj2_3d = {},{}
    
    decode_proj1_shuff_all_3d, decode_proj2_shuff_all_3d = {},{}
    
    

    # estimate decodability by ttype
    for tt in ttypes:
        
        decode_proj1T_3d = np.zeros((nBoots, nPerms, len(tbins))) # pca1st 3d coordinates
        decode_proj2T_3d = np.zeros((nBoots, nPerms, len(tbins)))
        
        
        # shuff
        decode_proj1T_3d_shuff = np.zeros((nBoots, nPerms, len(tbins)))
        decode_proj2T_3d_shuff = np.zeros((nBoots, nPerms, len(tbins)))
        
        for nbt in range(nBoots):

            # trial info
            trialInfo_CT = trialInfos_C[nbt]#
            trialInfo_CT_tt = trialInfo_CT[trialInfo_CT.type == tt]
            idx_tt = trialInfo_CT_tt.index.tolist()
            
            vecs_CT = vecs_C[nbt]# choice plane vecs
            projs_CT = projs_C[nbt]

            vec_normal_CT = np.cross(vecs_CT[0],vecs_CT[1]) # plane normal vec
            center_CT = projs_CT.mean(0) # plane center
            
            data_3pc_CT_smooth = data_3pc_C[nbt][idx_tt,:,:] # 3pc states from tt trials
            projs_All_CT = np.array([[f_subspace.proj_on_plane(p[:,t], vec_normal_CT, center_CT) for t in range(len(tbins))] for p in data_3pc_CT_smooth]) # projections on the plane
            projs_All_CT = np.swapaxes(projs_All_CT,1,2)
            
            # labels
            Y = trialInfo_CT_tt.loc[:,Y_columnsLabels].values
            ntrial = len(trialInfo_CT_tt)
            
            
            toDecode_X1 = Y_columnsLabels.index(toDecode_labels1)
            toDecode_X2 = Y_columnsLabels.index(toDecode_labels2)
            
            ### labels: ['locKey','locs','type','loc1','loc2','locX']
            label1 = Y[:,toDecode_X1].astype('int')
            label2 = Y[:,toDecode_X2].astype('int')
            
            
            toDecode_labels1_inv = f_stats.inversed_label(Y_columnsLabels, toDecode_labels1, tt)
            toDecode_X1_inv = Y_columnsLabels.index(toDecode_labels1_inv)
            label1_inv = Y[:,toDecode_X1_inv]
            
            
            toDecode_labels2_inv = f_stats.inversed_label(Y_columnsLabels, toDecode_labels2, tt)
            toDecode_X2_inv = Y_columnsLabels.index(toDecode_labels2_inv)
            label2_inv = Y[:,toDecode_X2_inv]

            #shuff
            for npm in range(nPerms):
                
                if shuff_excludeInv:
                    # except for the inverse ones
                    label1_shuff = np.full_like(label1_inv,9, dtype=int)
                    label2_shuff = np.full_like(label2_inv,9, dtype=int)
                    
                    for ni1, i1 in enumerate(label1_inv.astype(int)):
                        label1_shuff[ni1] = np.random.choice(np.array(locs)[np.array(locs)!=i1]).astype(int)

                    for ni2, i2 in enumerate(label2_inv.astype(int)):
                        label2_shuff[ni2] = np.random.choice(np.array(locs)[np.array(locs)!=i2]).astype(int)
                    
                    trialInfo_CT_tt_shuff = trialInfo_CT_tt.copy()
                    trialInfo_CT_tt_shuff[toDecode_labels1] = label1_shuff
                    trialInfo_CT_tt_shuff[toDecode_labels2] = label2_shuff
                    
                    
                else:
                    label1_shuff = np.random.permutation(label1)
                    label2_shuff = np.random.permutation(label2)
                
                trialInfo_CT_tt_shuff = trialInfo_CT_tt.copy()
                trialInfo_CT_tt_shuff[toDecode_labels1] = label1_shuff
                trialInfo_CT_tt_shuff[toDecode_labels2] = label2_shuff
                
                for t in range(len(tbins)):
                    info1_3d, _ = f_subspace.plane_decodability(vecs_CT, projs_CT, projs_All_CT[:,:,t], trialInfo_CT_tt.reset_index(drop = True), 'loc1', method = infoMethod)
                    info2_3d, _ = f_subspace.plane_decodability(vecs_CT, projs_CT, projs_All_CT[:,:,t], trialInfo_CT_tt.reset_index(drop = True), 'loc2', method = infoMethod)
                    
                    decode_proj1T_3d[nbt,npm,t] = info1_3d
                    decode_proj2T_3d[nbt,npm,t] = info2_3d
                
                    info1_3d_shuff, _ = f_subspace.plane_decodability(vecs_CT, projs_CT, projs_All_CT[:,:,t], trialInfo_CT_tt_shuff.reset_index(drop = True), 'loc1', method = infoMethod)
                    info2_3d_shuff, _ = f_subspace.plane_decodability(vecs_CT, projs_CT, projs_All_CT[:,:,t], trialInfo_CT_tt_shuff.reset_index(drop = True), 'loc2', method = infoMethod)
                    
                    decode_proj1T_3d_shuff[nbt,npm,t] = info1_3d_shuff
                    decode_proj2T_3d_shuff[nbt,npm,t] = info2_3d_shuff
                    
                    
        decode_proj1_3d[tt] = decode_proj1T_3d
        decode_proj2_3d[tt] = decode_proj2T_3d
        
                  
        decode_proj1_shuff_all_3d[tt] = decode_proj1T_3d_shuff
        decode_proj2_shuff_all_3d[tt] = decode_proj2T_3d_shuff
        
    decode_projs = (decode_proj1_3d, decode_proj2_3d)
    decode_projs_shuff = (decode_proj1_shuff_all_3d, decode_proj2_shuff_all_3d)
    
    return decode_projs, decode_projs_shuff
       

#%% cross temporal item decodability by the projections on the readout subspace
def itemInfo_by_planeCX(geomsC_valid, locs = (0,1,2,3), ttypes = (1,2), bins = 50, dt = 10, tslice = (0,3000), Y_columnsLabels = ['choice','type','loc1','loc2','locX'], 
                   toDecode_labels1 = 'loc1', toDecode_labels2 = 'loc2', shuff_excludeInv = False, nPerms = 10, infoMethod = 'lda'):
    
    
    vecs_C, projs_C, projsAll_C, trialInfos_C, data_3pc_C = geomsC_valid
    
    nBoots = len(trialInfos_C)
    
    bins = bins
    dt = dt
    tbins = np.arange(tslice[0], tslice[1], bins)
    
    decode_proj1_3d, decode_proj2_3d = {},{}
    
    decode_proj1_shuff_all_3d, decode_proj2_shuff_all_3d = {},{}

    # estimate decodability by ttype
    for tt in ttypes:
        
        decode_proj1T_3d = np.zeros((nBoots, nPerms, len(tbins), len(tbins))) # pca1st 3d coordinates
        decode_proj2T_3d = np.zeros((nBoots, nPerms, len(tbins), len(tbins)))
        
        
        # shuff
        decode_proj1T_3d_shuff = np.zeros((nBoots, nPerms, len(tbins), len(tbins)))
        decode_proj2T_3d_shuff = np.zeros((nBoots, nPerms, len(tbins), len(tbins)))
        
        for nbt in range(nBoots):

            # trial info
            trialInfo_CT = trialInfos_C[nbt]
            trialInfo_CT_tt = trialInfo_CT[trialInfo_CT.type == tt]
            idx_tt = trialInfo_CT_tt.index.tolist()
            
            vecs_CT = vecs_C[nbt]# choice plane vecs
            projs_CT = projs_C[nbt]

            vec_normal_CT = np.cross(vecs_CT[0],vecs_CT[1]) # plane normal vec
            center_CT = projs_CT.mean(0) # plane center
            
            data_3pc_CT_smooth = data_3pc_C[nbt][idx_tt,:,:] # 3pc states from tt trials
            projs_All_CT = np.array([[f_subspace.proj_on_plane(p[:,t], vec_normal_CT, center_CT) for t in range(len(tbins))] for p in data_3pc_CT_smooth]) # projections on the plane
            projs_All_CT = np.swapaxes(projs_All_CT,1,2)
            
            # labels
            Y = trialInfo_CT_tt.loc[:,Y_columnsLabels].values
            ntrial = len(trialInfo_CT_tt)
            
            
            toDecode_X1 = Y_columnsLabels.index(toDecode_labels1)
            toDecode_X2 = Y_columnsLabels.index(toDecode_labels2)
            
            ### labels: ['locKey','locs','type','loc1','loc2','locX']
            label1 = Y[:,toDecode_X1].astype('int')
            label2 = Y[:,toDecode_X2].astype('int')
            
            
            toDecode_labels1_inv = f_stats.inversed_label(Y_columnsLabels, toDecode_labels1, tt)
            toDecode_X1_inv = Y_columnsLabels.index(toDecode_labels1_inv)
            label1_inv = Y[:,toDecode_X1_inv]
            
            
            toDecode_labels2_inv = f_stats.inversed_label(Y_columnsLabels, toDecode_labels2, tt)
            toDecode_X2_inv = Y_columnsLabels.index(toDecode_labels2_inv)
            label2_inv = Y[:,toDecode_X2_inv]
            
            
            if shuff_excludeInv:
                # except for the inverse ones
                label1_shuff = np.full_like(label1_inv,9, dtype=int)
                label2_shuff = np.full_like(label2_inv,9, dtype=int)
                
                for ni1, i1 in enumerate(label1_inv.astype(int)):
                    label1_shuff[ni1] = np.random.choice(np.array(locs)[np.array(locs)!=i1]).astype(int)

                for ni2, i2 in enumerate(label2_inv.astype(int)):
                    label2_shuff[ni2] = np.random.choice(np.array(locs)[np.array(locs)!=i2]).astype(int)
                                
            else:
                # fully random
                label1_shuff = np.random.permutation(label1)
                label2_shuff = np.random.permutation(label2)

            #shuff
            for npm in range(nPerms):
                
                ### split into train and test sets
                train_setID = np.sort(np.random.choice(ntrial, round(0.5*ntrial),replace = False))
                test_setID = np.sort(np.random.choice(np.setdiff1d(np.arange(ntrial), train_setID, assume_unique=True), (ntrial-len(train_setID)),replace = False))
                ### labels: ['locKey','locs','type','loc1','loc2','locX']
                train_label1 = label1[train_setID]
                train_label2 = label2[train_setID]
                
                test_label1 = label1[test_setID]
                test_label2 = label2[test_setID]
                
                train_label1_shuff = label1_shuff[train_setID]
                train_label2_shuff = label2_shuff[train_setID]
                
                test_label1_shuff = label1_shuff[test_setID]
                test_label2_shuff = label2_shuff[test_setID]     
                
                for t in range(len(tbins)):
                    for t_ in range(len(tbins)):
                        info1_3d = f_decoding.LDAPerformance(projs_All_CT[train_setID,:,t], projs_All_CT[test_setID,:,t_], train_label1,test_label1)
                        info2_3d = f_decoding.LDAPerformance(projs_All_CT[train_setID,:,t], projs_All_CT[test_setID,:,t_], train_label2,test_label2)
                        
                        decode_proj1T_3d[nbt,npm,t,t_] = info1_3d
                        decode_proj2T_3d[nbt,npm,t,t_] = info2_3d
                    
                        info1_3d_shuff = f_decoding.LDAPerformance(projs_All_CT[train_setID,:,t], projs_All_CT[test_setID,:,t_], train_label1_shuff,test_label1_shuff)
                        info2_3d_shuff = f_decoding.LDAPerformance(projs_All_CT[train_setID,:,t], projs_All_CT[test_setID,:,t_], train_label1_shuff,test_label1_shuff)
                        
                        decode_proj1T_3d_shuff[nbt,npm,t,t_] = info1_3d_shuff
                        decode_proj2T_3d_shuff[nbt,npm,t,t_] = info2_3d_shuff
                        
                    
        decode_proj1_3d[tt] = decode_proj1T_3d
        decode_proj2_3d[tt] = decode_proj2T_3d
        
                  
        decode_proj1_shuff_all_3d[tt] = decode_proj1T_3d_shuff
        decode_proj2_shuff_all_3d[tt] = decode_proj2T_3d_shuff
        
    decode_projs = (decode_proj1_3d, decode_proj2_3d)
    decode_projs_shuff = (decode_proj1_shuff_all_3d, decode_proj2_shuff_all_3d)
    
    return decode_projs, decode_projs_shuff
               
#%% cross temporal trial type decodability by the projections on the readout subspace
def ttypeInfo_by_planeCX(geomsC_valid, locs = (0,1,2,3), ttypes = (1,2), bins = 50, dt = 10, tslice = (0,3000), Y_columnsLabels = ['choice','type','loc1','loc2','locX'], 
                   toDecode_labels1 = 'type', nPerms = 10):
    
    
    vecs_C, projs_C, projsAll_C, trialInfos_C, data_3pc_C = geomsC_valid
    
    nBoots = len(trialInfos_C)
    
    bins = bins
    dt = dt
    tbins = np.arange(tslice[0], tslice[1], bins)
    
    decode_proj1_3d = {}
    decode_proj1_shuff_all_3d = {}
    
    decode_proj1T_3d = np.zeros((nBoots, nPerms, len(tbins), len(tbins))) # pca1st 3d coordinates
    decode_proj1T_3d_shuff = np.zeros((nBoots, nPerms, len(tbins), len(tbins))) # shuff
    
    for nbt in range(nBoots):
        #for nbt in range(nBoots):
            
        # trial info
        trialInfo_CT = trialInfos_C[nbt]
        idx_tt = trialInfo_CT.index.tolist()
        
        vecs_CT = vecs_C[nbt]# choice plane vecs
        projs_CT = projs_C[nbt]

        vec_normal_CT = np.cross(vecs_CT[0],vecs_CT[1]) # plane normal vec
        center_CT = projs_CT.mean(0) # plane center
        
        data_3pc_CT_smooth = data_3pc_C[nbt][idx_tt,:,:] # 3pc states from tt trials
        projs_All_CT = np.array([[f_subspace.proj_on_plane(p[:,t], vec_normal_CT, center_CT) for t in range(len(tbins))] for p in data_3pc_CT_smooth]) # projections on the plane
        projs_All_CT = np.swapaxes(projs_All_CT,1,2)
        
        # labels
        Y = trialInfo_CT.loc[:,Y_columnsLabels].values
        ntrial = len(trialInfo_CT)
        
        
        toDecode_X1 = Y_columnsLabels.index(toDecode_labels1)
        
        ### labels: ['locKey','locs','type','loc1','loc2','locX']
        label1 = Y[:,toDecode_X1].astype('int')
        
        label1_shuff = np.random.permutation(label1)

        #shuff
        for npm in range(nPerms):
            
            ### split into train and test sets
            train_setID = np.sort(np.random.choice(ntrial, round(0.5*ntrial),replace = False))
            test_setID = np.sort(np.random.choice(np.setdiff1d(np.arange(ntrial), train_setID, assume_unique=True), (ntrial-len(train_setID)),replace = False))
            ### labels: ['locKey','locs','type','loc1','loc2','locX']
            train_label1 = label1[train_setID]
            
            test_label1 = label1[test_setID]
            
            train_label1_shuff = label1_shuff[train_setID]
            
            test_label1_shuff = label1_shuff[test_setID]
            
            for t in range(len(tbins)):
                for t_ in range(len(tbins)):
                    info1_3d = f_decoding.LDAPerformance(projs_All_CT[train_setID,:,t], projs_All_CT[test_setID,:,t_], train_label1,test_label1)
                    
                    decode_proj1T_3d[nbt,npm,t,t_] = info1_3d
                    
                    info1_3d_shuff = f_decoding.LDAPerformance(projs_All_CT[train_setID,:,t], projs_All_CT[test_setID,:,t_], train_label1_shuff,test_label1_shuff)
                    
                    decode_proj1T_3d_shuff[nbt,npm,t,t_] = info1_3d_shuff
                    
                
    decode_proj1_3d = decode_proj1T_3d
                
    decode_proj1_shuff_all_3d = decode_proj1T_3d_shuff
        
    decode_projs = decode_proj1_3d
    decode_projs_shuff = decode_proj1_shuff_all_3d
    
    return decode_projs, decode_projs_shuff
     
    
#%% [non-used] calculate the mean euclidean distance of readout projections between end D1 and end D2 across all trials
def get_euDist(geomsC_valid, locs = (0,1,2,3), ttypes = (1,2), bins = 50, dt = 10, tslice = (0,2700),
                   end_D1s = (1300,), end_D2s = (2600,), zscore = False, 
                   nPerms = 50, bslMethod = 'x', end_D1b = (2100,), end_D2b = (2600,),
                   hideLocs = (), normalizeMinMax = False):
    
    epsilon = 1e-7
    vecs_C, projs_C, projsAll_C, trialInfos_C, data_3pc_C = geomsC_valid
    
    nBoots = len(trialInfos_C)
    
    remainedLocs = tuple(l for l in locs if l not in hideLocs)
    
    bins = bins
    dt = dt
    tbins = np.arange(tslice[0], tslice[1], bins)
    
    euDists = {}

    # estimate decodability by ttype
    for tt in ttypes:

        # trial info
        trialInfo_CT = trialInfos_C
        trialInfo_CT_tt = trialInfo_CT[trialInfo_CT.type == tt]#.reset_index(drop = True)
        trialInfo_CT_tt = trialInfo_CT_tt[trialInfo_CT_tt.loc1.isin(remainedLocs) & trialInfo_CT_tt.loc2.isin(remainedLocs)]
        idx_tt = trialInfo_CT_tt.index.tolist()#.trial_index.values
        
        vecs_CT = vecs_C # choice plane vecs
        projs_CT = projs_C #
        
        

        vec_normal_CT = np.cross(vecs_CT[0],vecs_CT[1]) # plane normal vec
        center_CT = projs_CT.mean(0) # plane center

        data_3pc_CT_smooth = data_3pc_C[idx_tt,:,:] #[0] 3pc states from tt trials
        projs_All_CT = np.array([[f_subspace.proj_on_plane(p[:,t], vec_normal_CT, center_CT) for t in range(len(tbins))] for p in data_3pc_CT_smooth]) # projections on the plane
        projs_All_CT = np.swapaxes(projs_All_CT,1,2)
        
        #compress to 2d
        vecX_, vecY_ = f_subspace.vec_quad_side(projs_CT, sequence = (3,1,2,0)) 
        vecs_new = np.array(f_subspace.basis_correction_3d(vecs_CT, vecX_, vecY_))
        
        projs_All_CT_2d = np.array([f_subspace.proj_2D_coordinates(projs_All_CT[:,:,t], vecs_new) for t in range(projs_All_CT.shape[2])])
        projs_All_CT_2d = np.swapaxes(np.swapaxes(projs_All_CT_2d, 0, 1), 1, 2)
    
        if bool(normalizeMinMax):
            vmin, vmax = normalizeMinMax if len(normalizeMinMax)==2 else (-1,1)
            # normalize -1 to 1
            for d in range(projs_All_CT_2d.shape[1]):
                projs_All_CT_2d[:,d,:] = ((projs_All_CT_2d[:,d,:] - projs_All_CT_2d[:,d,:].min()) / (projs_All_CT_2d[:,d,:].max() - projs_All_CT_2d[:,d,:].min())) * (vmax - vmin) + vmin
            
        
        endX_D1s, endX_D2s = [tbins.tolist().index(ed1) for ed1 in end_D1s], [tbins.tolist().index(ed2) for ed2 in end_D2s]
        
        
        euDistT = np.sqrt(np.sum((projs_All_CT_2d[:,:,endX_D1s].mean(-1) - projs_All_CT_2d[:,:,endX_D2s].mean(-1))**2, axis=1))

        
        euDistT = np.array(euDistT)
        euDists[tt] = euDistT

    # condition general std for each pseudo pop
    euDistT = np.concatenate((euDists[1], euDists[2]))

    for tt in ttypes:
        if zscore:
            for i in range(len(euDists[tt])):
                euDists[tt][i] = (euDists[tt][i]- euDistT[i].mean())

        else:
            for i in range(len(euDists[tt])):
                euDists[tt][i] = (euDists[tt][i])

    
    return euDists

#%% [non-used] euclidean distance between the centroid of readout projection between end D1 and D2. centorids calculated as the condition mean
def get_euDist_centroids(geomsC_valid, locs = (0,1,2,3), ttypes = (1,2), bins = 50, dt = 10, tslice = (0,2700),
                   end_D1s = (1300,), end_D2s = (2600,), zscore = False, 
                   nPerms = 50, bslMethod = 'x', end_D1b = (2100,), end_D2b = (2600,)):
    
    

    vecs_C, projs_C, projsAll_C, trialInfos_C, data_3pc_C = geomsC_valid
    
    nBoots = len(trialInfos_C)
    
    bins = bins
    dt = dt
    tbins = np.arange(tslice[0], tslice[1], bins)
    
    euDists = {}

    # estimate decodability by ttype
    for tt in ttypes:
        
        euDistT = [] # pca1st 3d coordinates

        # trial info
        trialInfo_CT = trialInfos_C
        trialInfo_CT_tt = trialInfo_CT[trialInfo_CT.type == tt]
        idx_tt = trialInfo_CT_tt.index.tolist()
        
        vecs_CT = vecs_C# choice plane vecs
        projs_CT = projs_C#
        
        

        vec_normal_CT = np.cross(vecs_CT[0],vecs_CT[1]) # plane normal vec
        center_CT = projs_CT.mean(0) # plane center

        data_3pc_CT_smooth = data_3pc_C[idx_tt,:,:] #[0] 3pc states from tt trials
        projs_All_CT = np.array([[f_subspace.proj_on_plane(p[:,t], vec_normal_CT, center_CT) for t in range(len(tbins))] for p in data_3pc_CT_smooth]) # projections on the plane
        projs_All_CT = np.swapaxes(projs_All_CT,1,2)
        
        #compress to 2d
        vecX_, vecY_ = f_subspace.vec_quad_side(projs_CT, sequence = (3,1,2,0)) 
        vecs_new = np.array(f_subspace.basis_correction_3d(vecs_CT, vecX_, vecY_))
        
        projs_All_CT_2d = np.array([f_subspace.proj_2D_coordinates(projs_All_CT[:,:,t], vecs_new) for t in range(projs_All_CT.shape[2])])
        projs_All_CT_2d = np.swapaxes(np.swapaxes(projs_All_CT_2d, 0, 1), 1, 2)
        
        endX_D1s, endX_D2s = [tbins.tolist().index(ed1) for ed1 in end_D1s], [tbins.tolist().index(ed2) for ed2 in end_D2s]
        
        euDistTT = []
        
        trialInfo_temp = trialInfo_CT_tt.copy().reset_index(drop=True)
        
        for l1 in locs:
            for l2 in locs:
                if l1!=l2:
                    idxT = trialInfo_temp[(trialInfo_temp.loc1==l1)&(trialInfo_temp.loc2==l2)].index
                    centroidD1 = projs_All_CT_2d[idxT][:,:,endX_D1s].mean(2).mean(0)
                    centroidD2 = projs_All_CT_2d[idxT][:,:,endX_D2s].mean(2).mean(0)
                    
                    euDistTT += [np.sqrt(np.sum((centroidD1 - centroidD2)**2))]
                
        euDistT += [np.array(euDistTT)]

        euDistT = np.array(euDistT)
        euDists[tt] = euDistT

    # condition general std for each pseudo pop
    euDistT = np.concatenate((euDists[1], euDists[2]))

    for tt in ttypes:
        if zscore:
            for i in range(len(euDists[tt])):
                euDists[tt][i] = (euDists[tt][i]- euDistT[i].mean())/euDistT[i].std()

        else:
            for i in range(len(euDists[tt])):
                euDists[tt][i] = (euDists[tt][i])/euDistT[i].std()

    return euDists

#%% [non-used] euclidean distance between the centorids of readout projections between end D1 and D2. Centroid 1 as the mean of by loc1 regardless ttype; Centroid 2 as the mean by loc1&2&ttype
def get_euDist_centroids2(geomsC_valid, locs = (0,1,2,3), ttypes = (1,2), bins = 50, dt = 10, tslice = (0,2700),
                   end_D1s = (1300,), end_D2s = (2600,), zscore = False, 
                   nPerms = 50, bslMethod = 'x', end_D1b = (2100,), end_D2b = (2600,),
                   hideLocs = (), normalizeMinMax = False, shuffleBsl = False):
    
    remainedLocs = tuple(l for l in locs if l not in hideLocs)

    epsilon = 1e-7
    
    vecs_C, projs_C, projsAll_C, trialInfos_C, data_3pc_C = geomsC_valid
    
    nBoots = len(trialInfos_C)
    
    bins = bins
    dt = dt
    tbins = np.arange(tslice[0], tslice[1], bins)

    
    euDists = {tt:[] for tt in ttypes}
        
    # trial info
    trialInfo_CT = trialInfos_C
    trialInfo_CT = trialInfos_C[(trialInfo_CT.loc1.isin(remainedLocs))&(trialInfo_CT.loc2.isin(remainedLocs))]
    idx = trialInfo_CT.index.tolist()
    
    vecs_CT = vecs_C# choice plane vecs
    projs_CT = projs_C#
    

    vec_normal_CT = np.cross(vecs_CT[0],vecs_CT[1]) # plane normal vec
    center_CT = projs_CT.mean(0) # plane center

    data_3pc_CT_smooth = data_3pc_C[idx,:,:] #[0] 3pc states from tt trials
    projs_All_CT = np.array([[f_subspace.proj_on_plane(p[:,t], vec_normal_CT, center_CT) for t in range(len(tbins))] for p in data_3pc_CT_smooth]) # projections on the plane
    projs_All_CT = np.swapaxes(projs_All_CT,1,2)
    
    #compress to 2d
    vecX_, vecY_ = f_subspace.vec_quad_side(projs_CT, sequence = (3,1,2,0)) 
    vecs_new = np.array(f_subspace.basis_correction_3d(vecs_CT, vecX_, vecY_))
    
    projs_All_CT_2d = np.array([f_subspace.proj_2D_coordinates(projs_All_CT[:,:,t], vecs_new) for t in range(projs_All_CT.shape[2])])
    projs_All_CT_2d = np.swapaxes(np.swapaxes(projs_All_CT_2d, 0, 1), 1, 2)
    
    if bool(normalizeMinMax):
        vmin, vmax = normalizeMinMax if len(normalizeMinMax)==2 else (-1,1)
        # normalize -1 to 1
        for d in range(projs_All_CT_2d.shape[1]):
            projs_All_CT_2d[:,d,:] = ((projs_All_CT_2d[:,d,:] - projs_All_CT_2d[:,d,:].min()) / (projs_All_CT_2d[:,d,:].max() - projs_All_CT_2d[:,d,:].min())) * (vmax - vmin) + vmin
        

    endX_D1s, endX_D2s = [tbins.tolist().index(ed1) for ed1 in end_D1s], [tbins.tolist().index(ed2) for ed2 in end_D2s]
            
    if shuffleBsl is False:
        trialInfo_temp = trialInfo_CT.copy().reset_index(drop=True)
    else:
        trialInfo_temp = trialInfo_CT.copy().sample(frac=1).reset_index(drop=True)
    
    euDistT = {tt:[] for tt in ttypes}

    for l1 in remainedLocs:
        idxT1 = trialInfo_temp[(trialInfo_temp.loc1==l1)].index
        centroidD1 = projs_All_CT_2d[idxT1][:,:,endX_D1s].mean(2).mean(0) # type-general centroid

        for tt in ttypes:
            for l2 in remainedLocs:
                if l1!=l2:
                    idxT2 = trialInfo_temp[(trialInfo_temp.loc1==l1)&(trialInfo_temp.loc2==l2)&(trialInfo_temp.type==tt)].index
                    centroidD2 = projs_All_CT_2d[idxT2][:,:,endX_D2s].mean(2).mean(0)
                    
                    euDistT[tt] += [np.sqrt(np.sum((centroidD1 - centroidD2)**2))]

    for tt in ttypes:
        euDistT[tt] = np.array(euDistT[tt])
        euDists[tt] += [euDistT[tt]]

    return euDists

#%% euclidean distance using the centroids2 method, but with all projections normalized.
def get_euDist_normalized_centroids2(geomsC_valid, locs = (0,1,2,3), ttypes = (1,2), bins = 50, dt = 10, tslice = (0,2700),
                   end_D1s = (1300,), end_D2s = (2600,), zscore = False, vmin = -1, vmax = 1,
                   nPerms = 50, bslMethod = 'x', end_D1b = (2100,), end_D2b = (2600,)):
    
    

    epsilon = 1e-7
    
    vecs_C, projs_C, projsAll_C, trialInfos_C, data_3pc_C = geomsC_valid
    
    nBoots = len(trialInfos_C)
    
    bins = bins
    dt = dt
    tbins = np.arange(tslice[0], tslice[1], bins)

    euDists = {tt:[] for tt in ttypes}
        
    # trial info
    trialInfo_CT = trialInfos_C

    vecs_CT = vecs_C # choice plane vecs
    projs_CT = projs_C#
    

    vec_normal_CT = np.cross(vecs_CT[0],vecs_CT[1]) # plane normal vec
    center_CT = projs_CT.mean(0) # plane center

    data_3pc_CT_smooth = data_3pc_C[:,:,:] #[0] 3pc states from tt trials
    projs_All_CT = np.array([[f_subspace.proj_on_plane(p[:,t], vec_normal_CT, center_CT) for t in range(len(tbins))] for p in data_3pc_CT_smooth]) # projections on the plane
    projs_All_CT = np.swapaxes(projs_All_CT,1,2)
    
    #compress to 2d
    vecX_, vecY_ = f_subspace.vec_quad_side(projs_CT, sequence = (3,1,2,0)) 
    vecs_new = np.array(f_subspace.basis_correction_3d(vecs_CT, vecX_, vecY_))
    
    projs_All_CT_2d = np.array([f_subspace.proj_2D_coordinates(projs_All_CT[:,:,t], vecs_new) for t in range(projs_All_CT.shape[2])])

    # normalize -1 to 1
    projs_All_CT_2d = np.array([((projs_All_CT_2d[:,:,d] - projs_All_CT_2d[:,:,d].min()) / (projs_All_CT_2d[:,:,d].max() - projs_All_CT_2d[:,:,d].min())) * (1 - -1) + -1 for d in range(projs_All_CT_2d.shape[-1])])
    projs_All_CT_2d = np.swapaxes(np.swapaxes(projs_All_CT_2d, 0, 1), 0, 2)
    
    endX_D1s, endX_D2s = [tbins.tolist().index(ed1) for ed1 in end_D1s], [tbins.tolist().index(ed2) for ed2 in end_D2s]
            
    trialInfo_temp = trialInfo_CT.copy().reset_index(drop=True)
    
    euDistT = {tt:[] for tt in ttypes}

    for l1 in locs:
        idxT1 = trialInfo_temp[(trialInfo_temp.loc1==l1)].index
        centroidD1 = projs_All_CT_2d[idxT1][:,:,endX_D1s].mean(2).mean(0) # type-general centroid

        for tt in ttypes:
            for l2 in locs:
                if l1!=l2:
                    idxT2 = trialInfo_temp[(trialInfo_temp.loc1==l1)&(trialInfo_temp.loc2==l2)&(trialInfo_temp.type==tt)].index
                    centroidD2 = projs_All_CT_2d[idxT2][:,:,endX_D2s].mean(2).mean(0)
                    
                    euDistT[tt] += [np.sqrt(np.sum((centroidD1 - centroidD2)**2))]

    for tt in ttypes:
        euDists[tt] += [euDistT[tt]]


    return euDists


#%% cross-temporal item decodability with given test set
def lda12X(test_InfoT, hidden_statesT, tRange, locs = (0,1,2,3), ttypes = (1,2), nPCs = (0,15), dt = 10, bins = 50, 
           pca_tWinX=None, avg_method = 'conditional_mean', nBoots = 50, permDummy = True, shuff_excludeInv = False,
           Y_columnsLabels = ['choice','ttype','loc1','loc2','locX'], zbsl = True, tbsl = (-300,0)):
    
    pd.options.mode.chained_assignment = None
    epsilon = 1e-7
    
    tslice = (tRange.min(), tRange.max()+dt)

    tbins = np.arange(tslice[0], tslice[1], bins) # 
    locCombs = list(permutations(locs,2))
    subConditions = list(product(locCombs, ttypes))
    
    
    if zbsl and (tbsl[0] in tRange):
        bslx1, bslx2 = tRange.tolist().index(tbsl[0]), tRange.tolist().index(tbsl[1])

    ### scaling
    for ch in range(hidden_statesT.shape[1]):
        hidden_statesT[:,ch,:] = (hidden_statesT[:,ch,:] - hidden_statesT[:,ch,:].mean() + epsilon) / (hidden_statesT[:,ch,:].std() + epsilon) #standard scaler
        
        if zbsl and (tbsl[0] in tRange):
            bsl_mean, bsl_std = hidden_statesT[:,ch,bslx1:bslx2].mean(), hidden_statesT[:,ch,bslx1:bslx2].std()
            hidden_statesT[:,ch,:] = (hidden_statesT[:,ch,:] - bsl_mean + epsilon) / (bsl_std + epsilon) #standard scaler

    hidden_statesTT = hidden_statesT[:,:,pca_tWinX] if pca_tWinX != None else hidden_statesT[:,:,:]
    
    ### averaging method
    if avg_method == 'trial':
        # if average across all trials, shape = chs,time
        hidden_statesTT = hidden_statesTT.mean(axis=0)
    
    elif avg_method == 'none':
        # if none average, concatenate all trials, shape = chs, time*trials
        hidden_statesTT = np.concatenate(hidden_statesTT, axis=-1)
    
    elif avg_method == 'all':
        # if average across all trials all times, shape = chs,trials
        hidden_statesTT = hidden_statesTT.mean(axis=-1).T
    
    elif avg_method == 'conditional':
        # if conditional average, get mean for each condition, then concatenate condition_avgs, shape = chs, time*n_conditions
        X_regionT_ = []
        test_InfoT_ = test_InfoT.reset_index(drop=True)
        for sc in subConditions:
            lc,tt = sc
            l1, l2 = lc[0], lc[1]
            idxx = test_InfoT_[(test_InfoT_.loc1 == l1) & (test_InfoT_.loc2 == l2) & (test_InfoT_.ttype == tt)].index.tolist()
            X_regionT_ += [hidden_statesTT[idxx,:,:].mean(axis=0)]
        
        hidden_statesTT = np.concatenate(X_regionT_, axis=-1)
        
    elif avg_method == 'conditional_mean':
        # if conditional average + avged over time, get mean for each condition, then vstack condition_avgs, shape = chs, n_conditions
        hidden_statesTT_ = []
        test_InfoT_ = test_InfoT.reset_index(drop=True)
        for sc in subConditions:
            lc, tt = sc
            l1, l2 = lc[0], lc[1]
            idxx = test_InfoT_[(test_InfoT_.loc1 == l1) & (test_InfoT_.loc2 == l2) & (test_InfoT_.ttype == tt)].index.tolist()
            hidden_statesTT_ += [hidden_statesTT[idxx,:,:].mean(axis=0).mean(axis=-1)]
        
        hidden_statesTT = np.vstack(hidden_statesTT_).T
    
    
    ### fit & transform pca
    pcFrac = 1.0
    npc = min(int(pcFrac * hidden_statesTT.shape[0]), hidden_statesTT.shape[1])
    pca = PCA(n_components=npc)
    
    # double check number of maximal PCs
    nPCs = (min(nPCs[0], npc-1), min(nPCs[1], npc-1))

    pca.fit(hidden_statesTT.T)
    evr = pca.explained_variance_ratio_

    # apply transform to all trials
    hidden_statesTP = np.zeros((hidden_statesT.shape[0], npc, hidden_statesT.shape[2]))
    for trial in range(hidden_statesT.shape[0]):
        hidden_statesTP[trial,:,:] = pca.transform(hidden_statesT[trial,:,:].T).T
    
    # downsample to tbins
    ntrialT, ncellT, ntimeT = hidden_statesTP.shape
    hidden_statesTP = np.mean(hidden_statesTP.reshape(ntrialT, ncellT, int(ntimeT/(bins/dt)),int(bins/dt)),axis = 3)
    
    
    performanceX1, performanceX2 = {tt:[] for tt in ttypes}, {tt:[] for tt in ttypes}
    performanceX1_shuff, performanceX2_shuff = {tt:[] for tt in ttypes}, {tt:[] for tt in ttypes}

    for tt in ttypes:
        test_InfoT_ = test_InfoT[test_InfoT.ttype == tt]
        idxT = test_InfoT_.index
        Y = test_InfoT_.loc[:,Y_columnsLabels].values
        full_setP = hidden_statesTP[idxT,:,:]
        ntrial = len(test_InfoT_)

        
        toDecode_X1 = Y_columnsLabels.index('loc1')
        toDecode_X2 = Y_columnsLabels.index('loc2')
        
        full_label1 = Y[:,toDecode_X1].astype('int')
        full_label2 = Y[:,toDecode_X2].astype('int')


        if shuff_excludeInv:
            toDecode_labels1_inv = f_stats.inversed_label(Y_columnsLabels, 'loc1', tt)
            toDecode_X1_inv = Y_columnsLabels.index(toDecode_labels1_inv)
            full_label1_inv = Y[:,toDecode_X1_inv]
            
            
            toDecode_labels2_inv = f_stats.inversed_label(Y_columnsLabels, 'loc2', tt)
            toDecode_X2_inv = Y_columnsLabels.index(toDecode_labels2_inv)
            full_label2_inv = Y[:,toDecode_X2_inv]

            full_label1_shuff = np.full_like(full_label1_inv,9, dtype=int)
            full_label2_shuff = np.full_like(full_label2_inv,9, dtype=int)
            
            for ni1, i1 in enumerate(full_label1_inv.astype(int)):
                full_label1_shuff[ni1] = np.random.choice(np.array(locs)[np.array(locs)!=i1]).astype(int)

            for ni2, i2 in enumerate(full_label2_inv.astype(int)):
                full_label2_shuff[ni2] = np.random.choice(np.array(locs)[np.array(locs)!=i2]).astype(int)
            

        else:
            # fully random
            full_label1_shuff = np.random.permutation(full_label1)
            full_label2_shuff = np.random.permutation(full_label2)
        

        # 
        performanceX1T = np.zeros((nBoots, len(tbins),len(tbins)))
        performanceX2T = np.zeros((nBoots, len(tbins),len(tbins)))
        
        # permutation with shuffled label
        performanceX1_shuffT = np.zeros((nBoots, len(tbins),len(tbins)))
        performanceX2_shuffT = np.zeros((nBoots, len(tbins),len(tbins)))

        for nbt in range(nBoots):
            ### split into train and test sets
            train_setID = np.sort(np.random.choice(ntrial, round(0.5*ntrial),replace = False))
            test_setID = np.sort(np.random.choice(np.setdiff1d(np.arange(ntrial), train_setID, assume_unique=True), (ntrial-train_setID.shape[0]),replace = False))

            train_setP = full_setP[train_setID,:,:]
            test_setP = full_setP[test_setID,:,:]
            
            train_label1 = full_label1[train_setID]
            train_label2 = full_label2[train_setID]

            test_label1 = full_label1[test_setID]
            test_label2 = full_label2[test_setID]
            
            train_label1_shuff, test_label1_shuff = full_label1_shuff[train_setID], full_label1_shuff[test_setID]
            train_label2_shuff, test_label2_shuff = full_label2_shuff[train_setID], full_label2_shuff[test_setID]
            
            
            for t in range(len(tbins)):
                for t_ in range(len(tbins)):    
                    
                    pfmT1 = f_decoding.LDAPerformance(train_setP[:,nPCs[0]:nPCs[1],t], test_setP[:,nPCs[0]:nPCs[1],t_], 
                                                      train_label1, test_label1)
                    
                    pfmT2 = f_decoding.LDAPerformance(train_setP[:,nPCs[0]:nPCs[1],t], test_setP[:,nPCs[0]:nPCs[1],t_], 
                                                      train_label2, test_label2)
                    
                    pfmT1_shuff = f_decoding.LDAPerformance(train_setP[:,nPCs[0]:nPCs[1],t], test_setP[:,nPCs[0]:nPCs[1],t_], 
                                                      train_label1_shuff, test_label1_shuff)
                    
                    pfmT2_shuff = f_decoding.LDAPerformance(train_setP[:,nPCs[0]:nPCs[1],t], test_setP[:,nPCs[0]:nPCs[1],t_], 
                                                      train_label2_shuff, test_label2_shuff)
                    
                    performanceX1T[nbt,t,t_] = pfmT1
                    performanceX2T[nbt,t,t_] = pfmT2
                    
                    performanceX1_shuffT[nbt,t,t_] = pfmT1_shuff
                    performanceX2_shuffT[nbt,t,t_] = pfmT2_shuff
        
        performanceX1[tt] = performanceX1T
        performanceX2[tt] = performanceX2T
        performanceX1_shuff[tt] = performanceX1_shuffT
        performanceX2_shuff[tt] = performanceX2_shuffT
    
    return (performanceX1, performanceX2), (performanceX1_shuff, performanceX2_shuff), evr

#%% calculate Explained variance ratio with given test set
def get_EVR(test_InfoT, hidden_statesT, tRange, locs = (0,1,2,3), ttypes = (1,2), nPCs = (0,15), dt = 10, bins = 50, 
           pca_tWinX=None, avg_method = 'conditional_mean', nBoots = 50, permDummy = True, shuff_excludeInv = False,
           Y_columnsLabels = ['choice','ttype','loc1','loc2','locX'], zbsl = True, tbsl = (-300,0)):
    
    pd.options.mode.chained_assignment = None
    epsilon = 1e-7
    
    tslice = (tRange.min(), tRange.max()+dt)

    tbins = np.arange(tslice[0], tslice[1], bins) # 
    locCombs = list(permutations(locs,2))
    subConditions = list(product(locCombs, ttypes))
    
    
    if zbsl and (tbsl[0] in tRange):
        bslx1, bslx2 = tRange.tolist().index(tbsl[0]), tRange.tolist().index(tbsl[1])

    ### scaling
    for ch in range(hidden_statesT.shape[1]):
        hidden_statesT[:,ch,:] = (hidden_statesT[:,ch,:] - hidden_statesT[:,ch,:].mean() + epsilon) / (hidden_statesT[:,ch,:].std() + epsilon) #standard scaler
        
        if zbsl and (tbsl[0] in tRange):
            bsl_mean, bsl_std = hidden_statesT[:,ch,bslx1:bslx2].mean(), hidden_statesT[:,ch,bslx1:bslx2].std()
            hidden_statesT[:,ch,:] = (hidden_statesT[:,ch,:] - bsl_mean + epsilon) / (bsl_std + epsilon) #standard scaler

    hidden_statesTT = hidden_statesT[:,:,pca_tWinX] if pca_tWinX != None else hidden_statesT[:,:,:]
    
    ### averaging method
    if avg_method == 'trial':
        # if average across all trials, shape = chs,time
        hidden_statesTT = hidden_statesTT.mean(axis=0)
    
    elif avg_method == 'none':
        # if none average, concatenate all trials, shape = chs, time*trials
        hidden_statesTT = np.concatenate(hidden_statesTT, axis=-1)
    
    elif avg_method == 'all':
        # if average across all trials all times, shape = chs,trials
        hidden_statesTT = hidden_statesTT.mean(axis=-1).T
    
    elif avg_method == 'conditional':
        # if conditional average, get mean for each condition, then concatenate condition_avgs, shape = chs, time*n_conditions
        X_regionT_ = []
        test_InfoT_ = test_InfoT.reset_index(drop=True)
        for sc in subConditions:
            lc,tt = sc
            l1, l2 = lc[0], lc[1]
            idxx = test_InfoT_[(test_InfoT_.loc1 == l1) & (test_InfoT_.loc2 == l2) & (test_InfoT_.ttype == tt)].index.tolist()
            X_regionT_ += [hidden_statesTT[idxx,:,:].mean(axis=0)]
        
        hidden_statesTT = np.concatenate(X_regionT_, axis=-1)
        
    elif avg_method == 'conditional_mean':
        # if conditional average + avged over time, get mean for each condition, then vstack condition_avgs, shape = chs, n_conditions
        hidden_statesTT_ = []
        test_InfoT_ = test_InfoT.reset_index(drop=True)
        for sc in subConditions:
            lc, tt = sc
            l1, l2 = lc[0], lc[1]
            idxx = test_InfoT_[(test_InfoT_.loc1 == l1) & (test_InfoT_.loc2 == l2) & (test_InfoT_.ttype == tt)].index.tolist()
            hidden_statesTT_ += [hidden_statesTT[idxx,:,:].mean(axis=0).mean(axis=-1)]
        
        hidden_statesTT = np.vstack(hidden_statesTT_).T
    
    
    ### fit & transform pca
    pcFrac = 1.0
    npc = min(int(pcFrac * hidden_statesTT.shape[0]), hidden_statesTT.shape[1])
    pca = PCA(n_components=npc)
    
    # double check number of maximal PCs
    nPCs = (min(nPCs[0], npc-1), min(nPCs[1], npc-1))

    pca.fit(hidden_statesTT.T)
    evr = pca.explained_variance_ratio_
    return evr

#%% cross-temporal item decodability with given model, can iterate multiple rounds to get test sets
def rnns_lda12X(modelD, trialInfo, X_, Y_, tRange, dt = 10, bins = 50, frac = 0.8, locs = (0,1,2,3), ttypes = (1,2), 
                nPCs = (0,15), nIters = 1, nBoots = 10, pca_tWins = ((300,1300), (1600,2600),), tbsl = (-300,0), 
                conditions = None, pDummy = True, toPlot = True, label = '', shuff_excludeInv = False):
    
    epsilon = 1e-7

    pDummy = pDummy
    nIters = nIters
    nBoots = nBoots

    # decodability with/without permutation P value
    bins = bins
    tslice = (tRange.min(), tRange.max()+dt)
    
    tbins = np.arange(tslice[0], tslice[1], bins) #
    
    locs = locs
    ttypes = ttypes
    nPCs = nPCs

    conditions = (('ttype', 1), ('ttype', 2)) if conditions == None else conditions
    
    performancesX1 = {tt:[] for tt in ttypes}
    performancesX2 = {tt:[] for tt in ttypes}
    performancesX1_shuff = {tt:[] for tt in ttypes}
    performancesX2_shuff = {tt:[] for tt in ttypes}
    evrs = []

    for n in range(nIters):
        t_IterOn = time.time()
        print(f'Iter = {n}')
        
        
        _, _, _, test_setID, test_X, _ = f_simulation.split_dataset_balance(X_, Y_, trialInfo, frac = frac, ranseed=n)

        test_Info = trialInfo.loc[test_setID,:].reset_index(drop = True)

        test_Info.loc1 = test_Info.loc1.astype(int)
        test_Info.loc2 = test_Info.loc2.astype(int)
        test_Info.choice = test_Info.choice.astype(int)
        
        
        hidden_states, _ = modelD(test_X)
        hidden_states = hidden_states.data.cpu().detach().numpy()
        hidden_states = np.swapaxes(hidden_states, 1, 2) # shape as ntrials * nchs * ntime
        
        pca_tWin = np.hstack([np.arange(i[0],i[1],dt, dtype = int) for i in pca_tWins]).tolist() #
        pca_tWinX = [tRange.tolist().index(i) for i in pca_tWin]

        performancesX_n, performancesX_n_shuff, evr_n = lda12X(test_Info, hidden_states, tRange, locs = locs, ttypes = ttypes, 
                                                        nPCs = nPCs, dt = dt, bins = bins, pca_tWinX=pca_tWinX, 
                                                        nBoots = nBoots, permDummy = False, shuff_excludeInv = shuff_excludeInv, tbsl=tbsl)
        
        for tt in ttypes:
            performancesX1[tt] += [performancesX_n[0][tt]] 
            performancesX2[tt] += [performancesX_n[1][tt]]
            performancesX1_shuff[tt] += [performancesX_n_shuff[0][tt]]
            performancesX2_shuff[tt] += [performancesX_n_shuff[1][tt]]

        evrs += [evr_n]
        
        print(f'tIter = {(time.time() - t_IterOn):.4f}s')
            
    performancesX, performancesX_shuff = (performancesX1, performancesX2), (performancesX1_shuff, performancesX2_shuff)
    
    # plot out
    if toPlot:
        
        plt.figure(figsize=(28, 24), dpi=100)
        
        for tt in ttypes:
            ttypeT = 'Retarget' if tt==1 else 'Distractor'
            ttypeT_ = 'Retarget' if tt==1 else 'Distraction'
            
            performanceT1 = performancesX1[tt]
            performanceT1_shuff = performancesX1_shuff[tt]
            performanceT2 = performancesX2[tt]
            performanceT2_shuff = performancesX2_shuff[tt]
            
            pfm1 = np.array(performanceT1)
            pfm1_shuff = np.concatenate(np.array(performanceT1_shuff),axis=2)
            pfm2 = np.array(performanceT2)
            pfm2_shuff = np.concatenate(np.array(performanceT2_shuff),axis=2)
            
            pvalues1 = np.ones((len(tbins), len(tbins)))
            pvalues2 = np.ones((len(tbins), len(tbins)))
            if pDummy!=True:
                for t in range(len(tbins)):
                    for t_ in range(len(tbins)):
                        pvalues1[t,t_] = f_stats.permutation_p(pfm1.mean(axis = 0)[t,t_], pfm1_shuff[t,t_,:], tail = 'greater')
                        pvalues2[t,t_] = f_stats.permutation_p(pfm2.mean(axis = 0)[t,t_], pfm2_shuff[t,t_,:], tail = 'greater')
                        
            
            vmax = 1
            
            plt.subplot(2,2,tt)
            ax = plt.gca()
            sns.heatmap(pd.DataFrame(pfm1.mean(0), index=tbins,columns=tbins), cmap = 'magma', vmin = 0.25, vmax = vmax, ax = ax)#, vmax = 0.6, xticklabels = 100, yticklabels = 100, vmin = cbar_min, vmax = cbar_max
            
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
            ax.set_xticklabels(['S1', 'S2', 'Go-Cue'], rotation=0, fontsize = 20)
            ax.set_xlabel('Test Timebin (ms)', labelpad = 10, fontsize = 25)
            ax.set_yticks([list(tbins).index(i) for i in [0, 1300, 2600]])
            ax.set_yticklabels(['S1', 'S2', 'Go-Cue'], fontsize = 20)
            ax.set_ylabel('Train Timebin (ms)', labelpad = 10, fontsize = 25)
            
            cbar = ax.collections[0].colorbar
            # here set the labelsize by 20
            cbar.ax.tick_params(labelsize=20)
            
            ax.set_title(f'{ttypeT_}, Item1', fontsize = 30, pad = 20)
            
            
            # item2
            plt.subplot(2,2,tt+2)
            ax = plt.gca()
            sns.heatmap(pd.DataFrame(pfm2.mean(0), index=tbins,columns=tbins), cmap = 'magma', vmin = 0.25, vmax = vmax, ax = ax)#, vmax = 0.6, xticklabels = 100, yticklabels = 100, vmin = cbar_min, vmax = cbar_max
            
            #from scipy import ndimage
            smooth_scale = 10
            z = ndimage.zoom(pvalues2, smooth_scale)
            ax.contour(np.linspace(0, len(tbins), len(tbins) * smooth_scale),
                     np.linspace(0, len(tbins), len(tbins) * smooth_scale),
                      z, levels=([0.05]), colors='white', alpha = 1)
            
            ax.invert_yaxis()
            
            
            # event lines
            for i in [0, 1300, 2600]:
                ax.plot(tbins, np.full_like(tbins,list(tbins).index(i)), 'w-.', linewidth=4)
                ax.plot(np.full_like(tbins,list(tbins).index(i)), tbins, 'w-.', linewidth=4)
            
            ax.set_xticks([list(tbins).index(i) for i in [0, 1300, 2600]])
            ax.set_xticklabels(['S1', 'S2', 'Go-Cue'], rotation=0, fontsize = 20)
            ax.set_xlabel('Test Timebin (ms)', labelpad = 10, fontsize = 25)
            ax.set_yticks([list(tbins).index(i) for i in [0, 1300, 2600]])
            ax.set_yticklabels(['S1', 'S2', 'Go-Cue'], fontsize = 20)
            ax.set_ylabel('Train Timebin (ms)', labelpad = 10, fontsize = 25)
            
            cbar = ax.collections[0].colorbar
            # here set the labelsize by 20
            cbar.ax.tick_params(labelsize=20)
            
            ax.set_title(f'{ttypeT_}, Item2', fontsize = 30, pad = 20)
            
        plt.tight_layout()
        plt.subplots_adjust(top = 0.95)
        plt.suptitle(f'{label}, Full Space', fontsize = 35, y=1) #, Arti_Noise = {arti_noise_level}
        plt.show()

    return performancesX, performancesX_shuff, evrs

#%% calculate explained variance ratio with multiple rounds of test set
def rnns_EVRs(modelD, trialInfo, X_, Y_, tRange, dt = 10, bins = 50, frac = 0.8, locs = (0,1,2,3), ttypes = (1,2), 
                nPCs = (0,15), nIters = 1, nBoots = 10, pca_tWins = ((300,1300), (1600,2600),), tbsl = (-300,0), 
                conditions = None, pDummy = True, toPlot = True, label = '', shuff_excludeInv = False):

    nIters = nIters
    nBoots = nBoots

    # decodability with/without permutation P value
    bins = bins
    tslice = (tRange.min(), tRange.max()+dt)

    locs = locs
    ttypes = ttypes
    nPCs = nPCs

    evrs = []

    for n in range(nIters):
        print(f'Iter = {n}')
        
        
        _, _, _, test_setID, test_X, _ = f_simulation.split_dataset_balance(X_, Y_, trialInfo, frac = frac, ranseed=n)

        test_Info = trialInfo.loc[test_setID,:].reset_index(drop = True)

        test_Info.loc1 = test_Info.loc1.astype(int)
        test_Info.loc2 = test_Info.loc2.astype(int)
        test_Info.choice = test_Info.choice.astype(int)
        
        
        hidden_states, _ = modelD(test_X)
        hidden_states = hidden_states.data.cpu().detach().numpy()
        hidden_states = np.swapaxes(hidden_states, 1, 2) # shape as ntrials * nchs * ntime
        
        pca_tWin = np.hstack([np.arange(i[0],i[1],dt, dtype = int) for i in pca_tWins]).tolist() #
        pca_tWinX = [tRange.tolist().index(i) for i in pca_tWin]

        evr_n = get_EVR(test_Info, hidden_states, tRange, locs = locs, ttypes = ttypes, 
                        nPCs = nPCs, dt = dt, bins = bins, pca_tWinX=pca_tWinX, 
                        nBoots = nBoots, permDummy = False, shuff_excludeInv = shuff_excludeInv, tbsl=tbsl)
        

        evrs += [evr_n]
        
    return evrs

#%% cross-temporal trial type decodability with given test set
def ldattX(test_InfoT, hidden_statesT, tRange, locs = (0,1,2,3), ttypes = (1,2), nPCs = (0,15), dt = 10, bins = 50, 
           pca_tWinX=None, avg_method = 'conditional_mean', nBoots = 50, permDummy = True, shuff_excludeInv = False,
           Y_columnsLabels = ['choice','ttype','loc1','loc2','locX'], zbsl = True, tbsl = (-300,0)):
    
    pd.options.mode.chained_assignment = None
    epsilon = 1e-7
    
    tslice = (tRange.min(), tRange.max()+dt)

    tbins = np.arange(tslice[0], tslice[1], bins) # 
    locCombs = list(permutations(locs,2))
    subConditions = list(product(locCombs, ttypes))
    
    
    if zbsl and (tbsl[0] in tRange):
        bslx1, bslx2 = tRange.tolist().index(tbsl[0]), tRange.tolist().index(tbsl[1])

    ### scaling
    for ch in range(hidden_statesT.shape[1]):
        hidden_statesT[:,ch,:] = (hidden_statesT[:,ch,:] - hidden_statesT[:,ch,:].mean() + epsilon) / (hidden_statesT[:,ch,:].std() + epsilon) #standard scaler
        
        if zbsl and (tbsl[0] in tRange):
            bsl_mean, bsl_std = hidden_statesT[:,ch,bslx1:bslx2].mean(), hidden_statesT[:,ch,bslx1:bslx2].std()
            hidden_statesT[:,ch,:] = (hidden_statesT[:,ch,:] - bsl_mean + epsilon) / (bsl_std + epsilon) #standard scaler

    hidden_statesTT = hidden_statesT[:,:,pca_tWinX] if pca_tWinX != None else hidden_statesT[:,:,:]
    
    ### averaging method
    if avg_method == 'trial':
        # if average across all trials, shape = chs,time
        hidden_statesTT = hidden_statesTT.mean(axis=0)
    
    elif avg_method == 'none':
        # if none average, concatenate all trials, shape = chs, time*trials
        hidden_statesTT = np.concatenate(hidden_statesTT, axis=-1)
    
    elif avg_method == 'all':
        # if average across all trials all times, shape = chs,trials
        hidden_statesTT = hidden_statesTT.mean(axis=-1).T
    
    elif avg_method == 'conditional':
        # if conditional average, get mean for each condition, then concatenate condition_avgs, shape = chs, time*n_conditions
        X_regionT_ = []
        test_InfoT_ = test_InfoT.reset_index(drop=True)
        for sc in subConditions:
            lc,tt = sc
            l1, l2 = lc[0], lc[1]
            idxx = test_InfoT_[(test_InfoT_.loc1 == l1) & (test_InfoT_.loc2 == l2) & (test_InfoT_.ttype == tt)].index.tolist()
            X_regionT_ += [hidden_statesTT[idxx,:,:].mean(axis=0)]
        
        hidden_statesTT = np.concatenate(X_regionT_, axis=-1)
        
    elif avg_method == 'conditional_mean':
        # if conditional average + avged over time, get mean for each condition, then vstack condition_avgs, shape = chs, n_conditions
        hidden_statesTT_ = []
        test_InfoT_ = test_InfoT.reset_index(drop=True)
        for sc in subConditions:
            lc, tt = sc
            l1, l2 = lc[0], lc[1]
            idxx = test_InfoT_[(test_InfoT_.loc1 == l1) & (test_InfoT_.loc2 == l2) & (test_InfoT_.ttype == tt)].index.tolist()
            hidden_statesTT_ += [hidden_statesTT[idxx,:,:].mean(axis=0).mean(axis=-1)]
        
        hidden_statesTT = np.vstack(hidden_statesTT_).T
    
    
    ### fit & transform pca
    pcFrac = 1.0
    npc = min(int(pcFrac * hidden_statesTT.shape[0]), hidden_statesTT.shape[1])
    pca = PCA(n_components=npc)
    
    # double check number of maximal PCs
    nPCs = (min(nPCs[0], npc-1), min(nPCs[1], npc-1))

    pca.fit(hidden_statesTT.T)
    evr = pca.explained_variance_ratio_

    # apply transform to all trials
    hidden_statesTP = np.zeros((hidden_statesT.shape[0], npc, hidden_statesT.shape[2]))
    for trial in range(hidden_statesT.shape[0]):
        hidden_statesTP[trial,:,:] = pca.transform(hidden_statesT[trial,:,:].T).T
    
    # downsample to tbins
    ntrialT, ncellT, ntimeT = hidden_statesTP.shape
    hidden_statesTP = np.mean(hidden_statesTP.reshape(ntrialT, ncellT, int(ntimeT/(bins/dt)),int(bins/dt)),axis = 3)
    
    
    performanceX1 = {}
    performanceX1_shuff = {}

    
    test_InfoT_ = test_InfoT.copy()###[test_InfoT.ttype == tt]
    idxT = test_InfoT_.index
    Y = test_InfoT_.loc[:,Y_columnsLabels].values
    full_setP = hidden_statesTP[idxT,:,:]
    ntrial = len(test_InfoT_)

    
    toDecode_X1 = Y_columnsLabels.index('ttype')
    
    full_label1 = Y[:,toDecode_X1].astype('int') #.astype('str') # locKey
    
    # fully random
    full_label1_shuff = np.random.permutation(full_label1)
    
    # 
    performanceX1T = np.zeros((nBoots, len(tbins),len(tbins)))
    
    # permutation with shuffled label
    performanceX1_shuffT = np.zeros((nBoots, len(tbins),len(tbins)))
    
    for nbt in range(nBoots):
        ### split into train and test sets
        train_setID = np.sort(np.random.choice(ntrial, round(0.5*ntrial),replace = False))
        test_setID = np.sort(np.random.choice(np.setdiff1d(np.arange(ntrial), train_setID, assume_unique=True), (ntrial-train_setID.shape[0]),replace = False))

        train_setP = full_setP[train_setID,:,:]
        test_setP = full_setP[test_setID,:,:]
        
        train_label1 = full_label1[train_setID]
        test_label1 = full_label1[test_setID]
        
        train_label1_shuff, test_label1_shuff = full_label1_shuff[train_setID], full_label1_shuff[test_setID]
        
        for t in range(len(tbins)):
            for t_ in range(len(tbins)):    
                
                pfmT1 = f_decoding.LDAPerformance(train_setP[:,nPCs[0]:nPCs[1],t], test_setP[:,nPCs[0]:nPCs[1],t_], 
                                                    train_label1, test_label1)
                
                
                pfmT1_shuff = f_decoding.LDAPerformance(train_setP[:,nPCs[0]:nPCs[1],t], test_setP[:,nPCs[0]:nPCs[1],t_], 
                                                    train_label1_shuff, test_label1_shuff)
                
                performanceX1T[nbt,t,t_] = pfmT1
                
                performanceX1_shuffT[nbt,t,t_] = pfmT1_shuff
                    
        performanceX1 = performanceX1T
        performanceX1_shuff = performanceX1_shuffT
        
    return performanceX1, performanceX1_shuff, evr

#%% cross-temporal trial type decodability with given model, can iterate for multiple test sets
def rnns_ldattX(modelD, trialInfo, X_, Y_, tRange, dt = 10, bins = 50, frac = 0.8, locs = (0,1,2,3), ttypes = (1,2), 
                nPCs = (0,15), nIters = 1, nBoots = 10, pca_tWins = ((300,1300), (1600,2600),), tbsl = (-300,0), 
                conditions = None, pDummy = True, toPlot = True, label = '', shuff_excludeInv = False):
    

    pDummy = pDummy
    nIters = nIters
    nBoots = nBoots
    
    # decodability with/without permutation P value
    bins = bins
    tslice = (tRange.min(), tRange.max()+dt)
    
    tbins = np.arange(tslice[0], tslice[1], bins) #
    
    locs = locs
    ttypes = ttypes
    nPCs = nPCs

    performancesX_tt = []
    performancesX_tt_shuff = []
    evrs = []

    for n in range(nIters):
        t_IterOn = time.time()
        print(f'Iter = {n}')
        
        
        _, _, _, test_setID, test_X, _ = f_simulation.split_dataset_balance(X_, Y_, trialInfo, frac = frac, ranseed=n)
        test_Info = trialInfo.loc[test_setID,:].reset_index(drop = True)

        test_Info.loc1 = test_Info.loc1.astype(int)
        test_Info.loc2 = test_Info.loc2.astype(int)
        test_Info.choice = test_Info.choice.astype(int)
        
        
        hidden_states, _ = modelD(test_X)
        hidden_states = hidden_states.data.cpu().detach().numpy()
        hidden_states = np.swapaxes(hidden_states, 1, 2) # shape as ntrials * nchs * ntime
        
        pca_tWin = np.hstack([np.arange(i[0],i[1],dt, dtype = int) for i in pca_tWins]).tolist() #
        pca_tWinX = [tRange.tolist().index(i) for i in pca_tWin]

        performancesX_tt_n, performancesX_tt_n_shuff, evr_n = ldattX(test_Info, hidden_states, tRange, locs = locs, ttypes = ttypes, 
                                                        nPCs = nPCs, dt = dt, bins = bins, pca_tWinX=pca_tWinX, 
                                                        nBoots = nBoots, permDummy = False, shuff_excludeInv = shuff_excludeInv, tbsl=tbsl)
        
        
        performancesX_tt += [performancesX_tt_n] 
        performancesX_tt_shuff += [performancesX_tt_n_shuff]
            
        evrs += [evr_n]
        
        print(f'tIter = {(time.time() - t_IterOn):.4f}s')
            
    performancesX, performancesX_shuff = performancesX_tt, performancesX_tt_shuff
    
    # plot out
    if toPlot:
        
        plt.figure(figsize=(7, 6), dpi=100)
        
        performanceT1 = performancesX_tt
        performanceT1_shuff = performancesX_tt_shuff
        
        pfm1 = np.array(performanceT1)
        pfm1_shuff = np.concatenate(np.array(performanceT1_shuff),axis=2)
        
        pvalues1 = np.ones((len(tbins), len(tbins)))
        if pDummy!=True:
            for t in range(len(tbins)):
                for t_ in range(len(tbins)):
                    pvalues1[t,t_] = f_stats.permutation_p(pfm1.mean(axis = 0)[t,t_], pfm1_shuff[t,t_,:], tail = 'greater')
                    
        
        vmax = 1
        
        plt.subplot(1,1,1)
        ax = plt.gca()
        sns.heatmap(pd.DataFrame(pfm1.mean(0), index=tbins,columns=tbins), cmap = 'magma', vmin = 0.25, vmax = vmax, ax = ax)#, vmax = 0.6, xticklabels = 100, yticklabels = 100, vmin = cbar_min, vmax = cbar_max

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
        
        ax.set_title(f'Trial Type', fontsize = 25, pad = 20)
        plt.tight_layout()
        plt.subplots_adjust(top = 0.95)
        plt.suptitle(f'{label}, Full Space', fontsize = 25, y=1) #, Arti_Noise = {arti_noise_level}
        plt.show()

    return performancesX, performancesX_shuff, evrs

#%% generate vectors for baseline distribution use, i.e. calculate the same vectors but based on different subsets of data
def generate_bslVectors(models_dict, trialInfo, X_, Y_, tRange, checkpoints, avgInterval, dt = 10, locs = (0,1,2,3), ttypes = (1,2), 
                         adaptPCA=None, adaptEVR = None, pca_tWins = ((300,1300), (1600,2600),), nBoots = 1, fracBoots = 0.5, nPerms = 100, toPlot=False, avgMethod='conditional_time'):
    
    epsilon = 1e-7
    
    nIters = len(models_dict)
    

    vecs_bsl_All_train = []
    projs_bsl_All_train = []
    projsAll_bsl_All_train = []
    trialInfos_bsl_All_train = []
    pca1s_bsl_All_train = []

    vecs_bsl_All_test = []
    projs_bsl_All_test = []
    projsAll_bsl_All_test = []
    trialInfos_bsl_All_test = []
    pca1s_bsl_All_test = []
    
    evrs_1st_All_train = []       
    evrs_1st_All_test = []
    
    _, _, _, test_setID, test_X, _ = f_simulation.split_dataset_balance(X_, Y_, trialInfo, frac = 0.5)
    
    test_Info = trialInfo.loc[test_setID,:].reset_index(drop = True)
    test_set = test_X.cpu().numpy()

    test_Info.loc1 = test_Info.loc1.astype(int)
    test_Info.loc2 = test_Info.loc2.astype(int)
    test_Info.choice = test_Info.choice.astype(int)
    
    # if specify applied time window of pca
    pca_tWin = np.hstack([np.arange(i[0],i[1],dt, dtype = int) for i in pca_tWins]).tolist() #
    pca_tWinX = [tRange.tolist().index(i) for i in pca_tWin]
    
    for n in range(nIters):
        vecs_bsl_train = {}
        projs_bsl_train = {}
        projsAll_bsl_train = {}
        trialInfos_bsl_train = {}
        pca1s_bsl_train = []

        vecs_bsl_test = {}
        projs_bsl_test = {}
        projsAll_bsl_test = {}
        trialInfos_bsl_test = {}
        pca1s_bsl_test = []


        for tt in ttypes:
            trialInfos_bsl_train[tt] = []
            trialInfos_bsl_test[tt] = []
            

        for cp in checkpoints:
            vecs_bsl_train[cp] = {}
            projs_bsl_train[cp] = {}
            projsAll_bsl_train[cp] = {}

            vecs_bsl_test[cp] = {}
            projs_bsl_test[cp] = {}
            projsAll_bsl_test[cp] = {}
            
            for tt in ttypes:
                vecs_bsl_train[cp][tt] = {1:[], 2:[]}
                projs_bsl_train[cp][tt] = {1:[], 2:[]}
                projsAll_bsl_train[cp][tt] = {1:[], 2:[]}

                vecs_bsl_test[cp][tt] = {1:[], 2:[]}
                projs_bsl_test[cp][tt] = {1:[], 2:[]}
                projsAll_bsl_test[cp][tt] = {1:[], 2:[]}
                
        
        evrs_1st_train = np.zeros((nBoots, 3))
        evrs_1st_test = np.zeros((nBoots, 3))
        
        modelD = models_dict[n]['rnn']
        hidden_states, _ = modelD(test_X)
        hidden_states = hidden_states.data.cpu().detach().numpy()

        dataN = hidden_states.swapaxes(1,2)
        
        
        for ch in range(dataN.shape[1]):
            dataN[:,ch,:] = (dataN[:,ch,:] - dataN[:,ch,:].mean() + epsilon) / (dataN[:,ch,:].std() + epsilon) #standard scaler

        for nbt in range(nPerms):
            # if run multiple iterations, enable the following
            pca1s_bsl_train.append([])
            pca1s_bsl_test.append([])
            
            for tt in ttypes: #Dummy
                trialInfos_bsl_train[tt].append([])
                trialInfos_bsl_test[tt].append([])
            
            for cp in checkpoints:
                for tt in ttypes: #Dummy
                    for ll in (1,2,):
                        vecs_bsl_train[cp][tt][ll].append([])
                        projs_bsl_train[cp][tt][ll].append([])
                        projsAll_bsl_train[cp][tt][ll].append([])
                        
                        vecs_bsl_test[cp][tt][ll].append([])
                        projs_bsl_test[cp][tt][ll].append([])
                        projsAll_bsl_test[cp][tt][ll].append([])
                        
            test_InfoT = test_Info.rename(columns={'ttype':'type'})
            idxT1,idxT2 = f_subspace.split_set_balance(np.arange(dataN.shape[0]), test_InfoT, frac=fracBoots, ranseed=nbt)
            dataT1, dataT2 = dataN[idxT1,:,:], dataN[idxT2,:,:]
            trialInfoT1, trialInfoT2 = test_InfoT.loc[idxT1,:].reset_index(drop=True), test_InfoT.loc[idxT2,:].reset_index(drop=True)
            trialInfoT1['locs'] = trialInfoT1['loc1'].astype(str) + '_' + trialInfoT1['loc2'].astype(str)
            trialInfoT2['locs'] = trialInfoT2['loc1'].astype(str) + '_' + trialInfoT2['loc2'].astype(str)
            
            
            adaptPCA_T = None if (adaptPCA is None) else adaptPCA[n][0]
            adaptEVR_T = None if (adaptEVR is None) else adaptEVR[n][0]
            
            vecs_D1, projs_D1, projsAll_D1, _, trialInfos_D1, _, _, evr_1st_1, _ = f_subspace.plane_fitting_analysis(dataT1, trialInfoT1, pca_tWinX, checkpoints, tRange, avgInterval, locs, ttypes, 
                                                                                                                     adaptPCA=adaptPCA_T, adaptEVR = adaptEVR_T, toPlot=toPlot, avgMethod = avgMethod) #Dummy, decode_method = decode_method
            vecs_D2, projs_D2, projsAll_D2, _, trialInfos_D2, _, _, evr_1st_2, _ = f_subspace.plane_fitting_analysis(dataT2, trialInfoT2, pca_tWinX, checkpoints, tRange, avgInterval, locs, ttypes,
                                                                                                                     adaptPCA=adaptPCA_T, adaptEVR = adaptEVR_T, toPlot=toPlot, avgMethod = avgMethod)
            

            for cp in checkpoints:
                for tt in ttypes: #Dummy
                    for ll in (1,2,):
                        vecs_bsl_train[cp][tt][ll][nbt] = vecs_D1[cp][tt][ll]
                        projs_bsl_train[cp][tt][ll][nbt] = projs_D1[cp][tt][ll]
                        projsAll_bsl_train[cp][tt][ll][nbt] = projsAll_D1[cp][tt][ll]
                        
                        vecs_bsl_test[cp][tt][ll][nbt] = vecs_D2[cp][tt][ll]
                        projs_bsl_test[cp][tt][ll][nbt] = projs_D2[cp][tt][ll]
                        projsAll_bsl_test[cp][tt][ll][nbt] = projsAll_D2[cp][tt][ll]

            for tt in ttypes: #Dummy
                trialInfos_bsl_train[tt][nbt] = trialInfos_D1[tt]
                trialInfos_bsl_test[tt][nbt] = trialInfos_D2[tt]
                
            print(f'EVRs: {evr_1st_1.round(5)}, {evr_1st_2.round(5)}')
            evrs_1st_train[0,:] = evr_1st_1
            evrs_1st_test[0,:] = evr_1st_2
            
            
        vecs_bsl_All_train += [vecs_bsl_train]
        projs_bsl_All_train += [projs_bsl_train]
        projsAll_bsl_All_train += [projsAll_bsl_train]
        trialInfos_bsl_All_train += [trialInfos_bsl_train]
        pca1s_bsl_All_train += [pca1s_bsl_train]
        evrs_1st_All_train += [evrs_1st_train]

        vecs_bsl_All_test += [vecs_bsl_test]
        projs_bsl_All_test += [projs_bsl_test]
        projsAll_bsl_All_test += [projsAll_bsl_test]
        trialInfos_bsl_All_test += [trialInfos_bsl_test]
        pca1s_bsl_All_test += [pca1s_bsl_test]
        evrs_1st_All_test += [evrs_1st_test]
        
        del modelD
        torch.cuda.empty_cache()
        gc.collect()

    return vecs_bsl_All_train, projs_bsl_All_train, projsAll_bsl_All_train, trialInfos_bsl_All_train, pca1s_bsl_All_train, evrs_1st_All_train, vecs_bsl_All_test, projs_bsl_All_test, projsAll_bsl_All_test, trialInfos_bsl_All_test, pca1s_bsl_All_test, evrs_1st_All_test

#%% calculate the angle and alignments based on the baseline vectors
def get_angleAlignment_itemPairs_bsl(geoms_valid, geoms_bsl, checkpoints, locs = (0,1,2,3), ttypes = (1,2), sequence=(0,1,3,2)):
    
    vecs, projs, projsAll, trialInfos = geoms_valid
    vecs_bsl, projs_bsl, projsAll_bsl, trialInfos_bsl = geoms_bsl
    
    nBoots = 1#len(trialInfos[1])
    nPerms = len(trialInfos_bsl[1])
    
    cosTheta_11_bsl, cosTheta_12_bsl, cosTheta_22_bsl = {},{},{}
    cosPsi_11_bsl, cosPsi_12_bsl, cosPsi_22_bsl = {},{},{}
    
    for tt in ttypes: #Dummy
        
        
        cosTheta_11T_shuff = np.zeros((nBoots, nPerms, len(checkpoints)))
        cosTheta_22T_shuff = np.zeros((nBoots, nPerms, len(checkpoints)))
        cosTheta_12T_shuff = np.zeros((nBoots, nPerms, len(checkpoints)))
        
        cosPsi_11T_shuff = np.zeros((nBoots, nPerms, len(checkpoints)))
        cosPsi_22T_shuff = np.zeros((nBoots, nPerms, len(checkpoints)))
        cosPsi_12T_shuff = np.zeros((nBoots, nPerms, len(checkpoints)))
        
        for nbt in range(nBoots):
            
            for npm in range(nPerms):
                for nc, cp in enumerate(checkpoints):
                    
                    cT11_shuff, _, cP11_shuff, _, _ = f_subspace.angle_alignment_coplanar(vecs[cp][tt][1][npm], projs[cp][tt][1][npm], 
                                                                            vecs_bsl[cp][tt][1][npm], projs_bsl[cp][tt][1][npm], sequence=sequence)
                    cT22_shuff, _, cP22_shuff, _, _ = f_subspace.angle_alignment_coplanar(vecs[cp][tt][2][npm], projs[cp][tt][2][npm], 
                                                                            vecs_bsl[cp][tt][2][npm], projs_bsl[cp][tt][2][npm], sequence=sequence)
                    cT12_shuff, _, cP12_shuff, _, _ = f_subspace.angle_alignment_coplanar(vecs[cp][tt][1][npm], projs[cp][tt][1][npm], 
                                                                            vecs_bsl[cp][tt][2][npm], projs_bsl[cp][tt][2][npm], sequence=sequence)
                                    
                    cosTheta_11T_shuff[nbt,npm,nc], cosTheta_22T_shuff[nbt,npm,nc], cosTheta_12T_shuff[nbt,npm,nc] = cT11_shuff, cT22_shuff, cT12_shuff
                    cosPsi_11T_shuff[nbt,npm,nc], cosPsi_22T_shuff[nbt,npm,nc], cosPsi_12T_shuff[nbt,npm,nc] = cP11_shuff, cP22_shuff, cP12_shuff
                        
        
        
        cosTheta_11_bsl[tt] = cosTheta_11T_shuff
        cosTheta_22_bsl[tt] = cosTheta_22T_shuff
        cosTheta_12_bsl[tt] = cosTheta_12T_shuff
        
        cosPsi_11_bsl[tt] = cosPsi_11T_shuff
        cosPsi_22_bsl[tt] = cosPsi_22T_shuff
        cosPsi_12_bsl[tt] = cosPsi_12T_shuff

    cosThetas_bsl = (cosTheta_11_bsl, cosTheta_12_bsl, cosTheta_22_bsl)
    cosPsis_bsl = (cosPsi_11_bsl, cosPsi_12_bsl, cosPsi_22_bsl)
    
    return cosThetas_bsl, cosPsis_bsl
