# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 20:03:31 2024

@author: aka2333
"""
# In[]
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
from PIL import Image
from io import BytesIO
# turn off warning messages
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# system paths
import os
import sys
import gc
sys.path.append(r'C:\Users\aka2333\OneDrive\phd\project')

import time # timer

from itertools import permutations, combinations, product # itertools


import matplotlib.ticker as ticker
from matplotlib.pyplot import MultipleLocator #从pyplot导入MultipleLocator类，这个类用于设置刻度间隔
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec  # 导入专门用于网格分割及子图绘制的扩展包Gridspec

# In[]
import sklearn
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA

#from sklearn.metrics.pairwise import euclidean_distances

from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import Axes3D

#import dPCA
# In[] import pyTorch
import torch
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # define device, may need to change with Mac OS
# In[]
import myRNNs
import f_simulation
import f_trainRNN
# In[] testing analysis
import f_subspace
import f_stats
import f_decoding
import f_plotting
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

# In[]
def evaluate_acc(modelD, test_X, test_label, checkpointX = -1, label = 'Acc', toPrint = False):
    
    test_X = test_X
    test_label = test_label
    
    # In test phase, we don't need to compute gradients (for memory efficiency)
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

# In[]
def evaluate_acc_seqs(modelD, test_X, test_labels, checkpointX = -1, label = 'Acc', toPrint = False):
    
    test_X = test_X
    test_label = test_labels
    
    # In test phase, we don't need to compute gradients (for memory efficiency)
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

# In[]
def evaluate_acc_multi(modelD, test_X, test_label, checkpoint1X = -1, checkpoint2X = -1,  label1 = 'Acc_memo', label2 = 'Acc_resp', toPrint = False):
    N_out = modelD.output_size
    N_out_half = N_out//2
    test_X = test_X
    test_label = test_label
    
    # In test phase, we don't need to compute gradients (for memory efficiency)
    with torch.no_grad():
        labels = torch.tensor(test_label, dtype=torch.float32).to(device)
        n_samples = test_label.shape[0]
        
        n_correct_memo = 0
        _, outputs_p = modelD(test_X)
                
        _, predicted_memo = torch.max(outputs_p[:,checkpoint1X,:N_out_half].data, 1) # channel with max output at last timepoint -> choice
        n_correct_memo += (predicted_memo == labels).sum().item()
        acc_memo = 100.0 * n_correct_memo / n_samples  
        
        if toPrint:
            print (f'{label1}: {acc_memo:.2f}%')
        
        
        ###
        n_correct_resp = 0
        _, outputs_p = modelD(test_X)
                
        _, predicted_resp = torch.max(outputs_p[:,checkpoint2X,N_out_half:].data, 1) # channel with max output at last timepoint -> choice
        n_correct_resp += (predicted_resp == labels).sum().item()
        acc_resp = 100.0 * n_correct_resp / n_samples
        
        if toPrint:
            print(f'{label2}: {acc_resp} %')
    
    return acc_memo, acc_resp
# In[]       
def plot_states(modelD, test_Info, test_X, tRange, trialEvents, dt = 10, locs = (0,1,2,3), ttypes = (1,2), 
                lcX = np.arange(0,2,1), cues=False, cseq = None, label = '', vmin = 0, vmax = 10, 
                withhidden = True, withannote =True, savefig=False, save_path=''):
    
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
        #color = cseq[l]
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

            

            plt.subplot(3,2,tt)
            ax1 = plt.gca()

            for ll in locs:
                ax1.plot(inputsT[:,ll], linestyle = '-', linewidth = 6, color = cseq_r[ll], label = f'Location {str(int(ll+1))}, Target')
                #if 8 > N_in >= 6: # if 6 & 7 channels, plot as dash lines
                    #ax1.plot(inputsT[:,ll+4], linestyle = '--', color = cseq[ll], label = str(ll) + 'D')
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
            #plt.tight_layout()
            
            
            
            
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
            #ax3.set_ylim(-0.05,1.05)
            
            if withhidden:
                # hiddens
                plt.subplot(3,2,tt+4)
                ax2 = plt.gca()
                #ax.plot(hiddensT[:,:], linestyle = '-')
                im2 = ax2.imshow(hiddensT.T, cmap = 'magma', aspect = 'auto', vmin=vmin, vmax=vmax)
                ax2.set_xticks([list(tRange).index(i[0]) for i in trialEvents.values()])
                ax2.set_xticklabels([i[0] for i in trialEvents.values()], fontsize = 15)
                ax2.set_title(f'Hidden', fontsize = 20)
                ax2.tick_params(axis='y', labelsize=15)
                #cbar = plt.colorbar(im2, ax = ax2, extend='both')
                #cbar.ax.tick_params(axis='y', labelsize=15)
                ax2.set_xlim(left=0)
                #plt.tight_layout()
            
            
            

        plt.suptitle(f'{label}, Item1:{l1+1} & Item2:{l2+1}', fontsize = 25)
        plt.tight_layout()
        plt.show()
        if savefig:
            fig.savefig(f'{save_path}/{label}_states.tif')

# In[]       
def plot_states_varGo(modelD, test_Info, test_X, tRange, trialEvents, dt = 10, locs = (), ttypes = (), varGo = (),
                lcX = np.arange(0,2,1), cues=False, cseq = None, label = '', vmin = 0, vmax = 10, 
                withhidden = True, withannote =True, savefig=False, save_path=''):
    
    test_set = test_X.cpu().numpy()
    
    hidden_states, out_states = modelD(test_X)
    hidden_states = hidden_states.data.cpu().detach().numpy()
    out_states = out_states.data.cpu().detach().numpy()
    
    N_in, N_out = test_set.shape[-1], out_states.shape[-1]
    
    locs = test_Info.loc1.unique() if len(locs)==0 else locs
    ttypes = test_Info.ttype.unique() if len(ttypes)==0 else ttypes
    varGo = test_Info.go.unique() if len(varGo)==0 else varGo
    
    locCombs = list(permutations(locs,2))
    locCombsGo = list(product(locCombs, varGo))
    
    cseq = cseq if cseq != None else mpl.color_sequences['tab10']
    cseq_r = cseq[::-1]

    for lcgx in list(lcX):
        lcg = locCombsGo[lcgx]
        l1, l2 = lcg[0]
        go = lcg[1]
        
        trialEvents_ = trialEvents.copy()
        trialEvents_['d2'] = [trialEvents_['d2'][0], trialEvents_['d2'][1] + go]
        trialEvents_['go'] = [trialEvents_['go'][0] + go, trialEvents_['go'][1]]
        
        figsize=(30,12) if withhidden else (20,10)
        fig = plt.figure(figsize=figsize, dpi = 100)
        
        for tt in ttypes:
            
            ttypeT_ = 'Retarget' if tt==1 else 'Distraction'
            ttypeT = 'Retarget' if tt==1 else 'Distractor'
            
            idx = test_Info[(test_Info.loc1 == l1) & (test_Info.loc2 == l2) & (test_Info.ttype == tt) & (test_Info.go == go)].index
            inputsT = test_set[idx,:,:].mean(axis=0)
            hiddensT = hidden_states[idx,:,:].mean(axis=0)
            outputsT = out_states[idx,:,:].mean(axis=0)

            

            plt.subplot(3,2,tt)
            ax1 = plt.gca()

            for ll in locs:
                ax1.plot(inputsT[:,ll], linestyle = '-', linewidth = 6, color = cseq_r[ll], label = f'Location {str(int(ll+1))}, Target')
                #if 8 > N_in >= 6: # if 6 & 7 channels, plot as dash lines
                    #ax1.plot(inputsT[:,ll+4], linestyle = '--', color = cseq[ll], label = str(ll) + 'D')
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
            ax1.set_xticks([list(tRange).index(i[0]) for i in trialEvents_.values()])
            ax1.set_xticklabels([i[0] for i in trialEvents_.values()], fontsize = 15)
            ax1.tick_params(axis='y', labelsize=15)
            ax1.set_xlim(left=0)
            #plt.tight_layout()
            
            
            
            
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
            ax3.set_xticks([list(tRange).index(i[0]) for i in trialEvents_.values()])
            ax3.set_xticklabels([i[0] for i in trialEvents_.values()], fontsize = 15)
            ax3.tick_params(axis='y', labelsize=15)
            ax3.set_xlim(left=0)
            #ax3.set_ylim(-0.05,1.05)
            
            if withhidden:
                # hiddens
                plt.subplot(3,2,tt+4)
                ax2 = plt.gca()
                #ax.plot(hiddensT[:,:], linestyle = '-')
                im2 = ax2.imshow(hiddensT.T, cmap = 'magma', aspect = 'auto', vmin=vmin, vmax=vmax)
                ax2.set_xticks([list(tRange).index(i[0]) for i in trialEvents_.values()])
                ax2.set_xticklabels([i[0] for i in trialEvents_.values()], fontsize = 15)
                ax2.set_title(f'Hidden', fontsize = 20)
                ax2.tick_params(axis='y', labelsize=15)
                #cbar = plt.colorbar(im2, ax = ax2, extend='both')
                #cbar.ax.tick_params(axis='y', labelsize=15)
                ax2.set_xlim(left=0)
                #plt.tight_layout()
            
            
            

        plt.suptitle(f'{label}, Item1:{l1+1} & Item2:{l2+1}', fontsize = 25)
        plt.tight_layout()
        plt.show()
        if savefig:
            fig.savefig(f'{save_path}/{label}_states.tif')
#%%
def plot_states_chunkRNNs(modelD, test_Info, test_X, tRange, trialEvents, dt = 10, items = (), chunk = (), delayLength = (), 
                lcX = np.arange(0,2,1), cues=False, cseq = None, label = '', vmin = 0, vmax = 10, 
                withhidden = True, withannote =True, savefig=False, save_path=''):
    
    test_set = test_X.cpu().numpy()
    
    hidden_states, out_states = modelD(test_X)
    hidden_states = hidden_states.data.cpu().detach().numpy()
    out_states = out_states.data.cpu().detach().numpy()
    
    N_in, N_out = test_set.shape[-1], out_states.shape[-1]
    
    items = test_Info.stim1.unique() if len(items)==0 else items
    chunk = test_Info.chunk.unique() if len(chunk)==0 else chunk
    delayLength = test_Info.delayLength.unique() if len(delayLength)==0 else delayLength
    
    locs = list(range(len(items)))
    locCombs = list(permutations(locs,4))
    itemPairs = list(permutations(items,2))
    
    itemLocs = list(product(itemPairs, locCombs))
    itemLocsDelay = list(product(itemPairs, locCombs, delayLength))
    subConditions = list(product(itemPairs, locCombs, chunk))
    
    cseq = cseq if cseq != None else mpl.color_sequences['tab10']
    cseq_r = cseq[::-1]

    for ilX in list(lcX):
        il = itemLocsDelay[ilX]
        #color = cseq[l]
        i1, i2 = il[0]
        l1, l2, l3, l4 = il[1]
        dl = il[2]
        
        trialEvents_ = trialEvents.copy()
        trialEvents_['d1'] = [trialEvents_['d1'][0], trialEvents_['d1'][1] + dl]
        trialEvents_['s2'] = [trialEvents_['s2'][0] + dl, trialEvents_['s2'][1] + dl]
        trialEvents_['d2'] = [trialEvents_['d2'][0] + dl, trialEvents_['d2'][1] + dl*2]
        trialEvents_['go'] = [trialEvents_['go'][0] + dl*2, trialEvents_['go'][1]]
        
        figsize=(30,12) if withhidden else (20,10)
        fig = plt.figure(figsize=figsize, dpi = 100)
        
        for ck in chunk:
            
            chunkName = 'Chunked' if ck==1 else 'Non-chunked'
            
            idx = test_Info[(test_Info.stim1 == i1) & (test_Info.stim2 == i2) & (test_Info.chunk == ck) & 
                            (test_Info.item1Loc == l1) & (test_Info.item2Loc == l2) & 
                            (test_Info.item3Loc == l3) & (test_Info.item4Loc == l4) & (test_Info.delayLength == dl)].index
            inputsT = test_set[idx,:,:].mean(axis=0)
            hiddensT = hidden_states[idx,:,:].mean(axis=0)
            outputsT = out_states[idx,:,:].mean(axis=0)

            

            plt.subplot(3,2,ck+1)
            ax1 = plt.gca()

            for ll in locs:
                
                ax1.imshow(inputsT.T)
                #ax1.plot(inputsT[:,ll], linestyle = '-', linewidth = 6, color = cseq_r[ll], label = f'Location {str(int(ll+1))}, Target')
                #if N_in >= 8:
                #    ax1.plot(inputsT[:,ll+4], linestyle = '--', dashes=(3, 1), linewidth = 6, color = cseq_r[ll], label = f'Location {str(int(ll+1))}, Distractor')
            
            
            
                #if cues:
                #    if N_in%2 == 1:
                #        ax1.plot(inputsT[:,-1], linestyle = ':', color = 'grey', dashes=(3, 1), linewidth = 6, label = 'Fixation')
                
                #        ax1.plot(inputsT[:,-3], linestyle = ':', color = 'r', dashes=(3, 1), linewidth = 6, label = 'cue red')
                #        ax1.plot(inputsT[:,-2], linestyle = ':', color = 'g', dashes=(3, 1), linewidth = 6, label = 'cue green')
                        
                #    else:
                #        ax1.plot(inputsT[:,-2], linestyle = ':', color = 'r', dashes=(3, 1), linewidth = 6, label = 'cue red')
                #        ax1.plot(inputsT[:,-1], linestyle = ':', color = 'g', dashes=(3, 1), linewidth = 6, label = 'cue green')
            
            if withannote:
                arrowprops = dict(arrowstyle="->", facecolor='k', linewidth = 6, alpha = 0.8)
                ax1.annotate('Item1', xy=(tRange.tolist().index(150), 0.9), xytext=(tRange.tolist().index(350), 0.8), xycoords='data', arrowprops=arrowprops, fontsize = 20, alpha = 0.8)
                ax1.annotate('Alternative Locations', xy=(tRange.tolist().index(150), 0.1), xytext=(tRange.tolist().index(250), 0.4), xycoords='data', arrowprops=arrowprops, fontsize = 20, alpha = 0.8)
                ax1.annotate(f'Item2', xy=(tRange.tolist().index(1450), 0.9), xytext=(tRange.tolist().index(1650), 0.8), xycoords='data', arrowprops=arrowprops, fontsize = 20, alpha = 0.8)
            
            
            if ck==1:
                ax1.legend(loc = 'center left', bbox_to_anchor=(1, 0.5), fontsize = 15)
            ax1.set_title(f'Input', fontsize = 20)
            ax1.set_xticks([list(tRange).index(i[0]) for i in trialEvents_.values()])
            ax1.set_xticklabels([i[0] for i in trialEvents_.values()], fontsize = 15)
            ax1.tick_params(axis='y', labelsize=15)
            ax1.set_xlim(left=0)
            #plt.tight_layout()
            
            
            
            
            plt.subplot(3,2,ck+1+2)
            ax3 = plt.gca()
            for ll in locs:
                ax3.imshow(outputsT.T)
                #ax3.plot(outputsT[:,ll], linestyle = '-', linewidth = 6, color = cseq_r[ll], label = f'Location {str(int(ll+1))}')
                #if N_out > 4:
                #    ax3.plot(outputsT[:,ll+4], linestyle = '--', linewidth = 6, color = cseq_r[ll], label = f'{str(ll)}, resp')
            
            if withannote:
                ax3.text(tRange.tolist().index(800), 0.5, f'{chunkName}', fontsize = 20, alpha = 0.8)
            
            
            if ck==2:
                ax3.legend(loc = 'center left', bbox_to_anchor=(1, 0.5), fontsize = 15)
            ax3.set_title(f'Output', fontsize = 20)
            ax3.set_xticks([list(tRange).index(i[0]) for i in trialEvents_.values()])
            ax3.set_xticklabels([i[0] for i in trialEvents_.values()], fontsize = 15)
            ax3.tick_params(axis='y', labelsize=15)
            ax3.set_xlim(left=0)
            #ax3.set_ylim(-0.05,1.05)
            
            if withhidden:
                # hiddens
                plt.subplot(3,2,ck+1+4)
                ax2 = plt.gca()
                #ax.plot(hiddensT[:,:], linestyle = '-')
                im2 = ax2.imshow(hiddensT.T, cmap = 'magma', aspect = 'auto', vmin=vmin, vmax=vmax)
                ax2.set_xticks([list(tRange).index(i[0]) for i in trialEvents_.values()])
                ax2.set_xticklabels([i[0] for i in trialEvents_.values()], fontsize = 15)
                ax2.set_title(f'Hidden', fontsize = 20)
                ax2.tick_params(axis='y', labelsize=15)
                #cbar = plt.colorbar(im2, ax = ax2, extend='both')
                #cbar.ax.tick_params(axis='y', labelsize=15)
                ax2.set_xlim(left=0)
                #plt.tight_layout()
            
            
            

        plt.suptitle(f'{label}, Item1:{l1+1} & Item2:{l2+1}', fontsize = 25)
        plt.tight_layout()
        plt.show()
        if savefig:
            fig.savefig(f'{save_path}/{label}_states.tif')
# In[] plot weight matrices
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
    #Win = WinB
    Wout = paras['h2o.weight'].data.cpu().detach().numpy().T
    
    plt.figure(figsize=(20,10), dpi = 100)
    ax1 = plt.subplot(1,3,1)
    vabs = np.abs(Win).max()
    im1 = plt.imshow(Win,cmap='coolwarm', vmin=-1*vabs, vmax=vabs)
    #plt.imshow(WinB,cmap='jet')
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

# In[] plot weight matrices
def plot_weights_mixed(modelD, label = '', gen_in = True, gen_bump=True):
    # get trained parameters (weight matrices)
    paras = {}
    for names, param in modelD.named_parameters():
        paras[names] = param
    
    Wrec = modelD.generate_Wrec().cpu().detach().numpy() if gen_bump else paras['h2h.weight'].data.cpu().detach().numpy()
    Win = modelD.generate_Win().cpu().detach().numpy() if gen_in else paras['i2h.weight'].data.cpu().detach().numpy()
    #paras['Win'].data.cpu().detach().numpy() if ('Win' in paras.keys()) else paras['i2h.weight'].data.cpu().detach().numpy()
    #Win = WinB
    Wout = paras['h2o.weight'].data.cpu().detach().numpy().T
    
    plt.figure(figsize=(20,10), dpi = 100)
    ax1 = plt.subplot(1,3,1)
    vabs = np.abs(Win).max()
    im1 = plt.imshow(Win,cmap='coolwarm', vmin=-1*vabs, vmax=vabs)
    #plt.imshow(WinB,cmap='jet')
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




# In[]

def generate_itemVectors(models_dict, trialInfo, X_, Y_, tRange, checkpoints, avgInterval, dt = 10, locs = (0,1,2,3), ttypes = (1,2), 
                         adaptPCA=None, adaptEVR = None, pca_tWins = ((300,1300), (1600,2600),), nBoots = 1, fracBoots = 1.0, nPerms = 100, toPlot=False, avgMethod='conditional_time'):
    
    epsilon = 0.0000001
    
    #avgInterval = 50
    #avgInterval = {150:150, 550:250, 1150:250, 1450:150, 1850:250, 2350:250, 2800:200}


    #checkpoints = [300, 800, 1300, 1600, 2100, 2600, 2900, 3400, 3900, 4500]#
    #checkpoints = [250, 500, 750, 1000, 1250, 1500, 1800, 2300, 2800, 3300]
    #checkpoints = [150, 550, 1150, 1450, 1850, 2350, 2800]

    # 
    #decode_method = 'omega2' #'polyArea''lda' 
    # 'conditional' 'all' 'none'

    nIters = len(models_dict)
    
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
    
    # if specify applied time window of pca
    pca_tWin = np.hstack([np.arange(i[0],i[1],dt, dtype = int) for i in pca_tWins]).tolist() #
    #pca_tWin = np.hstack((np.arange(800,1300,dt, dtype = int),np.arange(2100,2600,dt, dtype = int))).tolist() #
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

        #idx1 = test_Info.id # original index
        #idx2 = test_Info.index.to_list() # reset index
        
        ### main test
        dataN = hidden_states.swapaxes(1,2)
        
        
        for ch in range(dataN.shape[1]):
            #dataN[:,ch,:] = (dataN[:,ch,:] - dataN[:,ch,idxxBsl[0]:idxxBsl[1]].mean())/dataN[:,ch,idxxBsl[0]:idxxBsl[1]].std() # Z to baseline, 
            #dataN[:,ch,:] = scale(dataN[:,ch,:])#01 scale for each channel
            dataN[:,ch,:] = (dataN[:,ch,:] - dataN[:,ch,:].mean() + epsilon) / (dataN[:,ch,:].std() + epsilon) #standard scaler
            #dataN[:,ch,:] = (dataN[:,ch,:] - dataN[:,ch,:].mean(axis=0)) / dataN[:,ch,:].std(axis=0) # detrened standard scaler
            
        ttypesDummy = (1,)
        
        
        for nbt in range(nBoots):
            # if run multiple iterations, enable the following
            pca1s.append([])
            pca1s_shuff.append([])
            
            for tt in ttypes: #Dummy
                trialInfos[tt].append([])
                trialInfos_shuff[tt].append([])
            
            for cp in checkpoints:
                for tt in ttypes: #Dummy
                    for ll in (1,2,):
                        vecs[cp][tt][ll].append([])
                        projs[cp][tt][ll].append([])
                        projsAll[cp][tt][ll].append([])
                        
                        vecs_shuff[cp][tt][ll].append([])
                        projs_shuff[cp][tt][ll].append([])
                        projsAll_shuff[cp][tt][ll].append([])
                        
            #idxT,_ = f_subspace.split_set(dataN, frac=fracBoots, ranseed=nboot)
            idxT,_ = f_subspace.split_set(np.arange(dataN.shape[0]), frac=fracBoots, ranseed=nbt)
            dataT = dataN[idxT,:,:]
            trialInfoT = test_Info.loc[idxT,:].reset_index(drop=True)
            trialInfoT['locs'] = trialInfoT['loc1'].astype(str) + '_' + trialInfoT['loc2'].astype(str)
            
            
            trialInfoT = trialInfoT.rename(columns={'ttype':'type'})
            #trialInfoT['type'] = 1
            
            adaptPCA_T = None if (adaptPCA is None) else adaptPCA[n][nbt]
            adaptEVR_T = None if (adaptEVR is None) else adaptEVR[n][nbt]
            
            vecs_D, projs_D, projsAll_D, _, trialInfos_D, _, _, evr_1st, pca1 = f_subspace.plane_fitting_analysis(dataT, trialInfoT, pca_tWinX, checkpoints, tRange, avgInterval, locs, ttypes, 
                                                                                                                  adaptPCA=adaptPCA_T, adaptEVR = adaptEVR_T, toPlot=toPlot, avgMethod = avgMethod) #Dummy, decode_method = decode_method
            #decodability_projD
            
            pca1s[nbt] += [pca1]
            
            for tt in ttypes: #Dummy
                trialInfos[tt][nbt] = trialInfos_D[tt]
                #trialInfos[tt] += [trialInfos_D[tt]]
                
            for cp in checkpoints:
                for tt in ttypes: #Dummy
                    for ll in (1,2,):
                        vecs[cp][tt][ll][nbt] = vecs_D[cp][tt][ll]
                        projs[cp][tt][ll][nbt] = projs_D[cp][tt][ll]
                        projsAll[cp][tt][ll][nbt] = projsAll_D[cp][tt][ll]
                        
                        #vecs[cp][tt][ll] += [vecs_D[cp][tt][ll]]
                        #projs[cp][tt][ll] += [projs_D[cp][tt][ll]]
                        #projsAll[cp][tt][ll] += [projsAll_D[cp][tt][ll]]
                        
            
            
            adaptPCA_shuffT = pca1 if (adaptPCA is None) else adaptPCA_T
            adaptEVR_shuffT = evr_1st if (adaptEVR is None) else adaptEVR_T
            
            print(f'EVRs: {evr_1st.round(5)}')
            evrs_1st[nbt,:] = evr_1st
            
            for nperm in range(nPerms):
                trialInfoT_shuff = trialInfoT.copy().sample(frac=1).reset_index(drop=True) # default method
                
                #trialInfoT_shuff = trialInfoT.copy()
                #for trial in range(len(trialInfoT)):
                #    l1, l2 = trialInfoT.loc[trial,'loc1'].astype('int'), trialInfoT.loc[trial,'loc2'].astype('int')
                #    l1s = np.random.choice([l for l in locs if l not in (l1,)]).astype('int')
                #    l2s = np.random.choice([l for l in locs if l not in (l1s,l2)]).astype('int')
                #    trialInfoT_shuff.loc[trial,'loc1'], trialInfoT_shuff.loc[trial,'loc2'] = l1s, l2s
                #    trialInfoT_shuff.loc[trial,'locs'] = '_'.join([l1s.astype('str'), l2s.astype('str')])
        
        
                vecs_D_shuff, projs_D_shuff, projsAll_D_shuff, _, trialInfos_D_shuff, _, _, _, pca1_shuff = f_subspace.plane_fitting_analysis(dataT, trialInfoT_shuff, pca_tWinX, checkpoints, tRange, avgInterval, locs, ttypes, 
                                                                                                                                              adaptPCA=adaptPCA_shuffT, adaptEVR = adaptEVR_shuffT, toPlot=toPlot, avgMethod = avgMethod) #Dummy, decode_method = decode_method 
                #decodability_projD_shuff
                
                pca1s_shuff[nbt] += [pca1_shuff]
                
                for tt in ttypes: #Dummy
                    trialInfos_shuff[tt][nbt] += [trialInfos_D_shuff[tt]]
                    #trialInfos_shuff[tt] += [trialInfos_D_shuff[tt]]
                    
                for cp in checkpoints:
                    for tt in ttypes: 
                        for ll in (1,2,):
                            vecs_shuff[cp][tt][ll][nbt] += [vecs_D_shuff[cp][tt][ll]]
                            projs_shuff[cp][tt][ll][nbt] += [projs_D_shuff[cp][tt][ll]]
                            projsAll_shuff[cp][tt][ll][nbt] += [projsAll_D_shuff[cp][tt][ll]]
                            
                            #vecs_shuff[cp][tt][ll] += [vecs_D_shuff[cp][tt][ll]]
                            #projs_shuff[cp][tt][ll] += [projs_D_shuff[cp][tt][ll]]
                            #projsAll_shuff[cp][tt][ll] += [projsAll_D_shuff[cp][tt][ll]]
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


# In[]
def generate_choiceVectors(models_dict, trialInfo, X_, Y_, tRange, locs = (0,1,2,3), ttypes = (1,2), dt = 10, bins = 50, 
                           adaptPCA=None, adaptEVR = None, pca_tWins = ((300,1300), (1600,2600),), avgMethod='conditional_time', 
                           choice_tRange = (2100,2600), nBoots = 1, fracBoots = 1.0, nPerms = 100, 
                           toPlot=False, toplot_samples = (0,), sequence=(0,1,3,2), plotlayout = (0,1,2,3), 
                           indScatters=False, legend_on=False, gau_kappa = 2.0,
                           plot_traj=True, plot3d = True, 
                           traj_checkpoints=(1300,2600), traj_start=1300, traj_end=2600, 
                           label = '', hideLocs = (), hideType = (), normalizeMinMax = False, separatePlot = True,
                           savefig=False, save_path=''):
    
    epsilon = 0.0000001
    
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
    
    # if specify applied time window of pca
    pca_tWin = np.hstack([np.arange(i[0],i[1],dt, dtype = int) for i in pca_tWins]).tolist() #
    #pca_tWin = np.hstack((np.arange(800,1300,dt, dtype = int),np.arange(2100,2600,dt, dtype = int))).tolist() #
    pca_tWinX = [tRange.tolist().index(i) for i in pca_tWin]
    
    
    for n in range(nIters):
        tplt = toPlot if n in toplot_samples else False
            
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
        
        ### main test
        
        dataN = hidden_states.swapaxes(1,2)
        
        
        for ch in range(dataN.shape[1]):
            #dataN[:,ch,:] = (dataN[:,ch,:] - dataN[:,ch,idxxBsl[0]:idxxBsl[1]].mean())/dataN[:,ch,idxxBsl[0]:idxxBsl[1]].std() # Z to baseline, 
            #dataN[:,ch,:] = scale(dataN[:,ch,:])#01 scale for each channel
            dataN[:,ch,:] = (dataN[:,ch,:] - dataN[:,ch,:].mean() + epsilon) / (dataN[:,ch,:].std() + epsilon) #standard scaler
            #dataN[:,ch,:] = (dataN[:,ch,:] - dataN[:,ch,:].mean(axis=0)) / dataN[:,ch,:].std(axis=0) # detrened standard scaler
            
        
        
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

            
            #idxT,_ = f_subspace.split_set(dataN, frac=fracBoots, ranseed=nboot)
            idxT,_ = f_subspace.split_set(np.arange(dataN.shape[0]), frac=fracBoots, ranseed=nbt)
            dataT = dataN[idxT,:,:]
            trialInfoT = test_Info.loc[idxT,:].reset_index(drop=True)
            trialInfoT['locs'] = trialInfoT['loc1'].astype(str) + '_' + trialInfoT['loc2'].astype(str)
            
            
            trialInfoT = trialInfoT.rename(columns={'ttype':'type'})
            
            adaptPCA_T = None if (adaptPCA is None) else adaptPCA[n][nbt]
            adaptEVR_T = None if (adaptEVR is None) else adaptEVR[n][nbt]
            
            vecs_CT, projs_CT, projsAll_CT, _, trialInfo_CT, data_3pc_CT, _, evr_1stT, pca1_C, evr_2C = f_subspace.planeC_fitting_analysis(dataT, trialInfoT, pca_tWinX, tRange, choice_tRange, locs, ttypes, 
                                                                                                                                   adaptPCA=adaptPCA_T, adaptEVR=adaptEVR_T,
                                                                                                                                   toPlot=tplt, avgMethod = avgMethod, plot_traj=plot_traj, traj_checkpoints=traj_checkpoints, plot3d = plot3d, plotlayout=plotlayout, indScatters=indScatters,
                                                                                                                                   traj_start=traj_start, sequence=sequence, traj_end=traj_end, region_label=label, 
                                                                                                                                   savefig=savefig, save_path=save_path,legend_on=legend_on, gau_kappa=gau_kappa,
                                                                                                                                   hideLocs=hideLocs, hideType=hideType, normalizeMinMax = normalizeMinMax, separatePlot=separatePlot) #Dummy, decode_method = decode_method
            
            
            # smooth to 50ms bins
            ntrialT, ncellT, ntimeT = data_3pc_CT.shape
            data_3pc_CT_smooth = np.mean(data_3pc_CT.reshape(ntrialT, ncellT, int(ntimeT/(bins/dt)),int(bins/dt)),axis = 3)  
                        
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
                trialInfoT_shuff = trialInfoT.copy().sample(frac=1).reset_index(drop=True) # default method
                
        
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
    
#%%
def generate_memoryVectors(models_dict, trialInfo, X_, Y_, tRange, locs = (0,1,2,3), ttypes = (1,2), dt = 10, bins = 50, 
                           adaptPCA=None, adaptEVR = None, pca_tWins = ((300,1300), (1600,2600),), avgMethod='conditional_time', 
                           memory_tRange = (300,1300), nBoots = 1, fracBoots = 1.0, nPerms = 100, 
                           toPlot=False, toplot_samples = (0,), sequence=(0,1,3,2), plotlayout = (0,1,2,3), 
                           indScatters=False, legend_on=False, gau_kappa = 2.0,
                           plot_traj=True, plot3d = True, 
                           traj_checkpoints=(1300,2600), traj_start=1300, traj_end=2600, 
                           label = '', hideLocs = (), hideType = (), normalizeMinMax = False, separatePlot = True,
                           savefig=False, save_path=''):
    
    epsilon = 0.0000001
    
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
    
    # if specify applied time window of pca
    pca_tWin = np.hstack([np.arange(i[0],i[1],dt, dtype = int) for i in pca_tWins]).tolist() #
    #pca_tWin = np.hstack((np.arange(800,1300,dt, dtype = int),np.arange(2100,2600,dt, dtype = int))).tolist() #
    pca_tWinX = [tRange.tolist().index(i) for i in pca_tWin]
    
    
    for n in range(nIters):
        tplt = toPlot if n in toplot_samples else False
            
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
        
        ### main test
        
        dataN = hidden_states.swapaxes(1,2)
        
        
        for ch in range(dataN.shape[1]):
            #dataN[:,ch,:] = (dataN[:,ch,:] - dataN[:,ch,idxxBsl[0]:idxxBsl[1]].mean())/dataN[:,ch,idxxBsl[0]:idxxBsl[1]].std() # Z to baseline, 
            #dataN[:,ch,:] = scale(dataN[:,ch,:])#01 scale for each channel
            dataN[:,ch,:] = (dataN[:,ch,:] - dataN[:,ch,:].mean() + epsilon) / (dataN[:,ch,:].std() + epsilon) #standard scaler
            #dataN[:,ch,:] = (dataN[:,ch,:] - dataN[:,ch,:].mean(axis=0)) / dataN[:,ch,:].std(axis=0) # detrened standard scaler
            
        
        
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

            
            #idxT,_ = f_subspace.split_set(dataN, frac=fracBoots, ranseed=nboot)
            idxT,_ = f_subspace.split_set(np.arange(dataN.shape[0]), frac=fracBoots, ranseed=nbt)
            dataT = dataN[idxT,:,:]
            trialInfoT = test_Info.loc[idxT,:].reset_index(drop=True)
            trialInfoT['locs'] = trialInfoT['loc1'].astype(str) + '_' + trialInfoT['loc2'].astype(str)
            
            
            trialInfoT = trialInfoT.rename(columns={'ttype':'type'})
            
            adaptPCA_T = None if (adaptPCA is None) else adaptPCA[n][nbt]
            adaptEVR_T = None if (adaptEVR is None) else adaptEVR[n][nbt]
            
            vecs_CT, projs_CT, projsAll_CT, _, trialInfo_CT, data_3pc_CT, _, evr_1stT, pca1_C, evr2_C = f_subspace.planeM_fitting_analysis(dataT, trialInfoT, pca_tWinX, tRange, memory_tRange, locs, ttypes, 
                                                                                                                                   adaptPCA=adaptPCA_T, adaptEVR=adaptEVR_T,
                                                                                                                                   toPlot=tplt, avgMethod = avgMethod, plot_traj=plot_traj, traj_checkpoints=traj_checkpoints, plot3d = plot3d, plotlayout=plotlayout, indScatters=indScatters,
                                                                                                                                   traj_start=traj_start, sequence=sequence, traj_end=traj_end, region_label=label, 
                                                                                                                                   savefig=savefig, save_path=save_path,legend_on=legend_on, gau_kappa=gau_kappa,
                                                                                                                                   hideLocs=hideLocs, hideType=hideType, normalizeMinMax = normalizeMinMax, separatePlot=separatePlot) #Dummy, decode_method = decode_method
            
            
            # smooth to 50ms bins
            ntrialT, ncellT, ntimeT = data_3pc_CT.shape
            data_3pc_CT_smooth = np.mean(data_3pc_CT.reshape(ntrialT, ncellT, int(ntimeT/(bins/dt)),int(bins/dt)),axis = 3)  
                        
            vecs_C[nbt] = vecs_CT
            projs_C[nbt] = projs_CT
            projsAll_C[nbt] = projsAll_CT
            trialInfos_C[nbt] = trialInfo_CT
            data_3pc_C[nbt] = data_3pc_CT_smooth
            
            pca1s_C[nbt] = pca1_C
            
            print(f'EVRs: {evr_1stT.round(5)}')
            evrs_1st[nbt,:] = evr_1stT
            evrs_2nd[nbt,:] = evr2_C
            print(f'EVRs: {(evr_1stT.sum()*evr2_C.sum()).round(5)}')
            
            adaptPCA_shuffT = pca1_C if (adaptPCA is None) else adaptPCA_T
            adaptEVR_shuffT = evr_1stT if (adaptEVR is None) else adaptEVR_T
            
            for nperm in range(nPerms):
                trialInfoT_shuff = trialInfoT.copy().sample(frac=1).reset_index(drop=True) # default method
                
        
                vecs_CT_shuff, projs_CT_shuff, projsAll_CT_shuff, _, trialInfo_CT_shuff, data_3pc_CT_shuff, _, _, pca1_C_shuff, _ = f_subspace.planeC_fitting_analysis(dataT, trialInfoT_shuff, pca_tWinX, tRange, memory_tRange, locs, ttypes, 
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
        
# In[] decodability plane projection by omega2
def itemInfo_by_plane(geoms_valid, checkpoints, locs = (0,1,2,3), ttypes = (1,2), Y_columnsLabels = ['choice','type','loc1','loc2','locX'], 
                      toDecode_labels1 = 'loc1', toDecode_labels2 = 'loc2', shuff_excludeInv = False, nPerms = 10, infoMethod = 'lda'):
    
    vecs, projs, projsAll, trialInfos = geoms_valid
    
    nBoots = len(trialInfos[1])
    
    decode_proj1_3d, decode_proj2_3d = {},{}
    
    decode_proj1_shuff_all_3d, decode_proj2_shuff_all_3d = {},{}
    
    
    for tt in ttypes:
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
                    vecs1, vecs2 = vecs[cp][tt][1][nbt], vecs[cp][tt][2][nbt]
                    projs1, projs2 = projs[cp][tt][1][nbt], projs[cp][tt][2][nbt]
                    projs1_allT_3d, projs2_allT_3d = projsAll[cp][tt][1][nbt], projsAll[cp][tt][2][nbt]
                    
                    info1_3d, _ = f_subspace.plane_decodability(vecs1, projs1, projs1_allT_3d, trialInfoT, 'loc1', method = infoMethod)
                    info2_3d, _ = f_subspace.plane_decodability(vecs2, projs2, projs2_allT_3d, trialInfoT, 'loc2', method = infoMethod)
                    
                    info1_3d_shuff, _ = f_subspace.plane_decodability(vecs1, projs1, projs1_allT_3d, trialInfoT_shuff, 'loc1', method = infoMethod, sequence=(0,1,3,2))
                    info2_3d_shuff, _ = f_subspace.plane_decodability(vecs2, projs2, projs2_allT_3d, trialInfoT_shuff, 'loc2', method = infoMethod, sequence=(0,1,3,2))
                    
                    
                    decode_proj1T_3d[nbt,npm,nc] = info1_3d#.mean(-1) #.mean(axis=-1).mean(axis=-1)
                    decode_proj2T_3d[nbt,npm,nc] = info2_3d#.mean(-1) #.mean(axis=-1).mean(axis=-1)
                    
                    decode_proj1T_3d_shuff[nbt,npm,nc] = info1_3d_shuff#.mean(-1) #.mean(axis=-1).mean(axis=-1)
                    decode_proj2T_3d_shuff[nbt,npm,nc] = info2_3d_shuff#.mean(-1) #.mean(axis=-1).mean(axis=-1)
        
        decode_proj1_3d[tt] = decode_proj1T_3d
        decode_proj2_3d[tt] = decode_proj2T_3d
        decode_proj1_shuff_all_3d[tt] = decode_proj1T_3d_shuff
        decode_proj2_shuff_all_3d[tt] = decode_proj2T_3d_shuff

    
    info3d = (decode_proj1_3d, decode_proj2_3d)
    info3d_shuff = (decode_proj1_shuff_all_3d, decode_proj2_shuff_all_3d)
    
    return info3d, info3d_shuff

#%% plane code transferability
def itemInfo_by_plane_Trans(geoms_valid, checkpoints, locs = (0,1,2,3), ttypes = (1,2), Y_columnsLabels = ['choice','type','loc1','loc2','locX'], 
                      toDecode_labels1 = 'loc1', toDecode_labels2 = 'loc2', shuff_excludeInv = False, nPerms = 10, infoMethod = 'lda'):
    
    vecs, projs, projsAll, trialInfos = geoms_valid
    
    nBoots = len(trialInfos[1])
    
    performanceX_12, performanceX_21 = {},{}
    performanceX_12_shuff, performanceX_21_shuff = {},{}
        
    
    for tt in ttypes:
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
                                                
                        info12, _ = f_subspace.plane_decodability_trans(geom1, geom2)
                        info21, _ = f_subspace.plane_decodability_trans(geom2, geom1)
                        
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

#%%
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

#In[]
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


#%%
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

#%%
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

# In[] decodability plane projection by lda
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
            #for nbt in range(nBoots):
                
            # trial info
            trialInfo_CT = trialInfos_C[nbt]#[0]
            trialInfo_CT_tt = trialInfo_CT[trialInfo_CT.type == tt]#.reset_index(drop = True)
            idx_tt = trialInfo_CT_tt.index.tolist()#.trial_index.values
            
            vecs_CT = vecs_C[nbt]#[0] # choice plane vecs
            projs_CT = projs_C[nbt]#[0] #
            
            

            vec_normal_CT = np.cross(vecs_CT[0],vecs_CT[1]) # plane normal vec
            center_CT = projs_CT.mean(0) # plane center
            
            data_3pc_CT_smooth = data_3pc_C[nbt][idx_tt,:,:] #[0] 3pc states from tt trials
            projs_All_CT = np.array([[f_subspace.proj_on_plane(p[:,t], vec_normal_CT, center_CT) for t in range(len(tbins))] for p in data_3pc_CT_smooth]) # projections on the plane
            projs_All_CT = np.swapaxes(projs_All_CT,1,2)
            
            # labels
            Y = trialInfo_CT_tt.loc[:,Y_columnsLabels].values
            ntrial = len(trialInfo_CT_tt)
            
            
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


            #for t in range(len(tbins)):
                
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
                    
                    decode_proj1T_3d[nbt,npm,t] = info1_3d #.mean(axis=-1).mean(axis=-1)
                    decode_proj2T_3d[nbt,npm,t] = info2_3d #.mean(axis=-1).mean(axis=-1)
                
                    info1_3d_shuff, _ = f_subspace.plane_decodability(vecs_CT, projs_CT, projs_All_CT[:,:,t], trialInfo_CT_tt_shuff.reset_index(drop = True), 'loc1', method = infoMethod)
                    info2_3d_shuff, _ = f_subspace.plane_decodability(vecs_CT, projs_CT, projs_All_CT[:,:,t], trialInfo_CT_tt_shuff.reset_index(drop = True), 'loc2', method = infoMethod)
                    
                    decode_proj1T_3d_shuff[nbt,npm,t] = info1_3d_shuff #.mean(axis=-1).mean(axis=-1)
                    decode_proj2T_3d_shuff[nbt,npm,t] = info2_3d_shuff #.mean(axis=-1).mean(axis=-1)
                    
                    
        decode_proj1_3d[tt] = decode_proj1T_3d#.mean(axis=1)
        decode_proj2_3d[tt] = decode_proj2T_3d#.mean(axis=1) 
        
                  
        decode_proj1_shuff_all_3d[tt] = decode_proj1T_3d_shuff
        decode_proj2_shuff_all_3d[tt] = decode_proj2T_3d_shuff
        
    decode_projs = (decode_proj1_3d, decode_proj2_3d)
    decode_projs_shuff = (decode_proj1_shuff_all_3d, decode_proj2_shuff_all_3d)
    
    return decode_projs, decode_projs_shuff
       

# In[] cross temporal
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
            #for nbt in range(nBoots):
                
            # trial info
            trialInfo_CT = trialInfos_C[nbt]#[0]
            trialInfo_CT_tt = trialInfo_CT[trialInfo_CT.type == tt]#.reset_index(drop = True)
            idx_tt = trialInfo_CT_tt.index.tolist()#.trial_index.values
            
            vecs_CT = vecs_C[nbt]#[0] # choice plane vecs
            projs_CT = projs_C[nbt]#[0] #
            
            

            vec_normal_CT = np.cross(vecs_CT[0],vecs_CT[1]) # plane normal vec
            center_CT = projs_CT.mean(0) # plane center
            
            data_3pc_CT_smooth = data_3pc_C[nbt][idx_tt,:,:] #[0] 3pc states from tt trials
            projs_All_CT = np.array([[f_subspace.proj_on_plane(p[:,t], vec_normal_CT, center_CT) for t in range(len(tbins))] for p in data_3pc_CT_smooth]) # projections on the plane
            projs_All_CT = np.swapaxes(projs_All_CT,1,2)
            
            # labels
            Y = trialInfo_CT_tt.loc[:,Y_columnsLabels].values
            ntrial = len(trialInfo_CT_tt)
            
            
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
                # fully random
                label1_shuff = np.random.permutation(label1)
                label2_shuff = np.random.permutation(label2)
            #for t in range(len(tbins)):
                
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
                        
                        decode_proj1T_3d[nbt,npm,t,t_] = info1_3d #.mean(axis=-1).mean(axis=-1)
                        decode_proj2T_3d[nbt,npm,t,t_] = info2_3d #.mean(axis=-1).mean(axis=-1)
                    
                        info1_3d_shuff = f_decoding.LDAPerformance(projs_All_CT[train_setID,:,t], projs_All_CT[test_setID,:,t_], train_label1_shuff,test_label1_shuff)
                        info2_3d_shuff = f_decoding.LDAPerformance(projs_All_CT[train_setID,:,t], projs_All_CT[test_setID,:,t_], train_label1_shuff,test_label1_shuff)
                        
                        decode_proj1T_3d_shuff[nbt,npm,t,t_] = info1_3d_shuff #.mean(axis=-1).mean(axis=-1)
                        decode_proj2T_3d_shuff[nbt,npm,t,t_] = info2_3d_shuff #.mean(axis=-1).mean(axis=-1)
                        
                    
        decode_proj1_3d[tt] = decode_proj1T_3d#.mean(axis=1)
        decode_proj2_3d[tt] = decode_proj2T_3d#.mean(axis=1) 
        
                  
        decode_proj1_shuff_all_3d[tt] = decode_proj1T_3d_shuff
        decode_proj2_shuff_all_3d[tt] = decode_proj2T_3d_shuff
        
    decode_projs = (decode_proj1_3d, decode_proj2_3d)
    decode_projs_shuff = (decode_proj1_shuff_all_3d, decode_proj2_shuff_all_3d)
    
    return decode_projs, decode_projs_shuff
               
# In[] cross temporal
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
        trialInfo_CT = trialInfos_C[nbt]#[0]
        idx_tt = trialInfo_CT.index.tolist()#.trial_index.values
        
        vecs_CT = vecs_C[nbt]#[0] # choice plane vecs
        projs_CT = projs_C[nbt]#[0] #
        
        

        vec_normal_CT = np.cross(vecs_CT[0],vecs_CT[1]) # plane normal vec
        center_CT = projs_CT.mean(0) # plane center
        
        data_3pc_CT_smooth = data_3pc_C[nbt][idx_tt,:,:] #[0] 3pc states from tt trials
        projs_All_CT = np.array([[f_subspace.proj_on_plane(p[:,t], vec_normal_CT, center_CT) for t in range(len(tbins))] for p in data_3pc_CT_smooth]) # projections on the plane
        projs_All_CT = np.swapaxes(projs_All_CT,1,2)
        
        # labels
        Y = trialInfo_CT.loc[:,Y_columnsLabels].values
        ntrial = len(trialInfo_CT)
        
        
        toDecode_X1 = Y_columnsLabels.index(toDecode_labels1)
        
        ### labels: ['locKey','locs','type','loc1','loc2','locX']
        label1 = Y[:,toDecode_X1].astype('int') #.astype('str') # locKey
        
        label1_shuff = np.random.permutation(label1)
        #for t in range(len(tbins)):
            
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
                    
                    decode_proj1T_3d[nbt,npm,t,t_] = info1_3d #.mean(axis=-1).mean(axis=-1)
                    
                    info1_3d_shuff = f_decoding.LDAPerformance(projs_All_CT[train_setID,:,t], projs_All_CT[test_setID,:,t_], train_label1_shuff,test_label1_shuff)
                    
                    decode_proj1T_3d_shuff[nbt,npm,t,t_] = info1_3d_shuff #.mean(axis=-1).mean(axis=-1)
                    
                
    decode_proj1_3d = decode_proj1T_3d#.mean(axis=1)
                
    decode_proj1_shuff_all_3d = decode_proj1T_3d_shuff
        
    decode_projs = decode_proj1_3d
    decode_projs_shuff = decode_proj1_shuff_all_3d
    
    return decode_projs, decode_projs_shuff
     
    
# In[] retarget vs distraction state changes
def get_euDist(geomsC_valid, locs = (0,1,2,3), ttypes = (1,2), bins = 50, dt = 10, tslice = (0,2700),
                   end_D1s = (1300,), end_D2s = (2600,), zscore = False, 
                   nPerms = 50, bslMethod = 'x', end_D1b = (2100,), end_D2b = (2600,),
                   hideLocs = (), normalizeMinMax = False):
    
    epsilon = 0.0000001
    vecs_C, projs_C, projsAll_C, trialInfos_C, data_3pc_C = geomsC_valid
    
    nBoots = len(trialInfos_C)
    
    remainedLocs = tuple(l for l in locs if l not in hideLocs)
    
    bins = bins
    dt = dt
    tbins = np.arange(tslice[0], tslice[1], bins)
    
    euDists = {}
    euDists_shuff = {}

    # estimate decodability by ttype
    for tt in ttypes:
        
        #euDistT = [] # pca1st 3d coordinates
        #euDistT_shuff = [] # pca1st 3d coordinates
        
        
        # trial info
        trialInfo_CT = trialInfos_C
        trialInfo_CT_tt = trialInfo_CT[trialInfo_CT.type == tt]#.reset_index(drop = True)
        trialInfo_CT_tt = trialInfo_CT_tt[trialInfo_CT_tt.loc1.isin(remainedLocs) & trialInfo_CT_tt.loc2.isin(remainedLocs)]
        idx_tt = trialInfo_CT_tt.index.tolist()#.trial_index.values
        
        vecs_CT = vecs_C # choice plane vecs
        projs_CT = projs_C #
        
        

        vec_normal_CT = np.cross(vecs_CT[0],vecs_CT[1]) # plane normal vec
        center_CT = projs_CT.mean(0) # plane center
        
        #data_3pc_CT = data_3pc_C[region][n][nbt][idx_tt,:,:] # 3pc states from tt trials
        
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
        
        #if bslMethod == 'shuff':
        #    for npm in range(nPerms):
        #        # time-shuffled time series
        #        rng = np.random.default_rng()
        #        projs_All_CT_shuff = rng.permuted(projs_All_CT, axis=2)
                
        #        euDistT_shuff += [np.sqrt(np.sum((projs_All_CT_shuff[:,:,endX_D1s].mean(-1)-projs_All_CT_shuff[:,:,endX_D2s].mean(-1))**2, axis=1))]
        
        #else:
            #specific baseline period
        #    endX_D1b, endX_D2b = [tbins.tolist().index(ed1b) for ed1b in end_D1b], [tbins.tolist().index(ed2b) for ed2b in end_D2b]
        #    euDistT_shuff += [np.sqrt(np.sum((projs_All_CT[:,:,endX_D1b].mean(-1)-projs_All_CT[:,:,endX_D2b].mean(-1))**2, axis=1))]
        
        
        euDistT = np.array(euDistT)
        #euDistT_shuff = np.array(euDistT_shuff)
        
        euDists[tt] = euDistT
        #euDists_shuff[tt] = euDistT_shuff
    
    # condition general std for each pseudo pop
    euDistT = np.concatenate((euDists[1], euDists[2]))
    #euDistT_shuff = np.concatenate((euDists_shuff[1], euDists_shuff[2]),axis=1)
    
    
    for tt in ttypes:
        if zscore:
            for i in range(len(euDists[tt])):
                euDists[tt][i] = (euDists[tt][i]- euDistT[i].mean())#/(euDistT[i].std()+epsilon) # euDistT_shuff[j,:].std()#
            
            #for j in range(len(euDists_shuff[tt])):
            #    euDists_shuff[tt][j,:] = (euDists_shuff[tt][j,:]- euDistT[i,:].mean())/ euDistT[i,:].std()
        else:
            for i in range(len(euDists[tt])):
                euDists[tt][i] = (euDists[tt][i])#/(euDistT[i].std()+epsilon) # euDistT_shuff[j,:].std()#- euDistT[i,:].mean()
            
            #for j in range(len(euDists_shuff[tt])):
            #    euDists_shuff[tt][j,:] = (euDists_shuff[tt][j,:])/euDistT[i,:].std()# euDistT_shuff[j,:].std() # - euDistT[i,:].mean()

    #euDistances = (euDists, euDists_shuff)
    
    return euDists#, euDists_shuff

# In[] retarget vs distraction state changes
def get_euDist_centroids(geomsC_valid, locs = (0,1,2,3), ttypes = (1,2), bins = 50, dt = 10, tslice = (0,2700),
                   end_D1s = (1300,), end_D2s = (2600,), zscore = False, 
                   nPerms = 50, bslMethod = 'x', end_D1b = (2100,), end_D2b = (2600,)):
    
    

    #-300,0 #
    
    vecs_C, projs_C, projsAll_C, trialInfos_C, data_3pc_C = geomsC_valid
    
    nBoots = len(trialInfos_C)
    
    bins = bins
    dt = dt
    tbins = np.arange(tslice[0], tslice[1], bins)
    
    euDists = {}
    euDists_shuff = {}

    # estimate decodability by ttype
    for tt in ttypes:
        
        euDistT = [] # pca1st 3d coordinates
        #euDistT_shuff = [] # pca1st 3d coordinates
        
        
        # trial info
        trialInfo_CT = trialInfos_C
        trialInfo_CT_tt = trialInfo_CT[trialInfo_CT.type == tt]#.reset_index(drop = True)
        idx_tt = trialInfo_CT_tt.index.tolist()#.trial_index.values
        
        vecs_CT = vecs_C# choice plane vecs
        projs_CT = projs_C#
        
        

        vec_normal_CT = np.cross(vecs_CT[0],vecs_CT[1]) # plane normal vec
        center_CT = projs_CT.mean(0) # plane center
        
        #data_3pc_CT = data_3pc_C[region][n][nbt][idx_tt,:,:] # 3pc states from tt trials
        
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
        
        
            #if bslMethod == 'shuff':
            #    for npm in range(nPerms):
            #        # time-shuffled time series
            #        rng = np.random.default_rng()
            #        projs_All_CT_shuff = rng.permuted(projs_All_CT, axis=2)
                    
            #        euDistT_shuff += [np.sqrt(np.sum((projs_All_CT_shuff[:,:,endX_D1s].mean(2)-projs_All_CT_shuff[:,:,endX_D2s].mean(2))**2, axis=1))]
            
            #else:
            #    endX_D1b, endX_D2b = [tbins.tolist().index(ed1b) for ed1b in end_D1b], [tbins.tolist().index(ed2b) for ed2b in end_D2b]
            #    euDistT_shuff += [np.sqrt(np.sum((projs_All_CT[:,:,endX_D1b].mean(2)-projs_All_CT[:,:,endX_D2b].mean(2))**2, axis=1))]
            
        
        euDistT = np.array(euDistT)
        #euDistT_shuff = np.array(euDistT_shuff)
        
        euDists[tt] = euDistT
        #euDists_shuff[tt] = euDistT_shuff
    
    # condition general std for each pseudo pop
    euDistT = np.concatenate((euDists[1], euDists[2]))
    #euDistT_shuff = np.concatenate((euDists_shuff[1], euDists_shuff[2]),axis=1)
    
    for tt in ttypes:
        if zscore:
            for i in range(len(euDists[tt])):
                euDists[tt][i] = (euDists[tt][i]- euDistT[i].mean())/euDistT[i].std() # euDistT_shuff[j,:].std()#
            
            #for j in range(len(euDists_shuff[tt])):
            #    euDists_shuff[tt][j,:] = (euDists_shuff[tt][j,:]- euDistT[i,:].mean())/ euDistT[i,:].std()
        else:
            for i in range(len(euDists[tt])):
                euDists[tt][i] = (euDists[tt][i])/euDistT[i].std() # euDistT_shuff[j,:].std()#- euDistT[i,:].mean()
            
            #for j in range(len(euDists_shuff[tt])):
            #    euDists_shuff[tt][j,:] = (euDists_shuff[tt][j,:])/euDistT[i,:].std()# euDistT_shuff[j,:].std() # - euDistT[i,:].mean()

    #euDistances = (euDists, euDists_shuff)
    
    return euDists#, euDists_shuff

# In[] retarget vs distraction state changes
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
    
    #euDists_shuff = {}
    
    euDists = {tt:[] for tt in ttypes}
        
    # trial info
    trialInfo_CT = trialInfos_C
    #trialInfo_CT_tt = trialInfo_CT[trialInfo_CT.type == tt]#.reset_index(drop = True)
    #idx_tt = trialInfo_CT_tt.index.tolist()#.trial_index.values
    trialInfo_CT = trialInfos_C[(trialInfo_CT.loc1.isin(remainedLocs))&(trialInfo_CT.loc2.isin(remainedLocs))]
    idx = trialInfo_CT.index.tolist()
    
    vecs_CT = vecs_C# choice plane vecs
    projs_CT = projs_C#
    

    vec_normal_CT = np.cross(vecs_CT[0],vecs_CT[1]) # plane normal vec
    center_CT = projs_CT.mean(0) # plane center
    
    #data_3pc_CT = data_3pc_C[region][n][nbt][idx_tt,:,:] # 3pc states from tt trials
    
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

    #std_pooled = np.concatenate([euDistT[1],euDistT[2]]).std() + epsilon
    for tt in ttypes:
        euDistT[tt] = np.array(euDistT[tt])# / std_pooled
        #euDistT = np.array(euDistT)
        euDists[tt] += [euDistT[tt]]


    return euDists#, euDists_shuff

#%%
def get_euDist_normalized_centroids2(geomsC_valid, locs = (0,1,2,3), ttypes = (1,2), bins = 50, dt = 10, tslice = (0,2700),
                   end_D1s = (1300,), end_D2s = (2600,), zscore = False, vmin = -1, vmax = 1,
                   nPerms = 50, bslMethod = 'x', end_D1b = (2100,), end_D2b = (2600,)):
    
    

    epsilon = 1e-7
    
    vecs_C, projs_C, projsAll_C, trialInfos_C, data_3pc_C = geomsC_valid
    
    nBoots = len(trialInfos_C)
    
    bins = bins
    dt = dt
    tbins = np.arange(tslice[0], tslice[1], bins)
    
    #euDists_shuff = {}
    
    euDists = {tt:[] for tt in ttypes}
        
    # trial info
    trialInfo_CT = trialInfos_C
    #trialInfo_CT_tt = trialInfo_CT[trialInfo_CT.type == tt]#.reset_index(drop = True)
    #idx_tt = trialInfo_CT_tt.index.tolist()#.trial_index.values
    
    vecs_CT = vecs_C# choice plane vecs
    projs_CT = projs_C#
    

    vec_normal_CT = np.cross(vecs_CT[0],vecs_CT[1]) # plane normal vec
    center_CT = projs_CT.mean(0) # plane center
    
    #data_3pc_CT = data_3pc_C[region][n][nbt][idx_tt,:,:] # 3pc states from tt trials
    
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

    #std_pooled = np.concatenate([euDistT[1],euDistT[2]]).std() + epsilon
    for tt in ttypes:
        #euDistT[tt] = np.array(euDistT[tt]) / std_pooled
        #euDistT = np.array(euDistT)
        euDists[tt] += [euDistT[tt]]


    return euDists#, euDists_shuff
#%%
def get_euDist2(geomsC_valid, locs = (0,1,2,3), ttypes = (1,2), bins = 50, dt = 10, tslice = (0,2700),
                   end_D1s = (1300,), end_D2s = (2600,), zscore = False, 
                   nPerms = 50, bslMethod = 'x', end_D1b = (2100,), end_D2b = (2600,)):
    
    
    vecs_C, projs_C, projsAll_C, trialInfos_C, data_3pc_C = geomsC_valid
    
    nBoots = len(trialInfos_C)
    
    bins = bins
    dt = dt
    tbins = np.arange(tslice[0], tslice[1], bins)
    
    euDists = {}
    euDists_shuff = {}

    # estimate decodability by ttype
    for tt in ttypes:
        
        euDistT = [] # pca1st 3d coordinates
        euDistT_shuff = [] # pca1st 3d coordinates
        
        
        for nbt in range(nBoots):
            #for nbt in range(nBoots):
                
            # trial info
            trialInfo_CT = trialInfos_C[nbt]
            trialInfo_CT_tt = trialInfo_CT[trialInfo_CT.type == tt]#.reset_index(drop = True)
            idx_tt = trialInfo_CT_tt.index.tolist()#.trial_index.values
            
            vecs_CT = vecs_C[nbt] # choice plane vecs
            projs_CT = projs_C[nbt] #
            
            

            vec_normal_CT = np.cross(vecs_CT[0],vecs_CT[1]) # plane normal vec
            center_CT = projs_CT.mean(0) # plane center
            
            #data_3pc_CT = data_3pc_C[region][n][nbt][idx_tt,:,:] # 3pc states from tt trials
            
            data_3pc_CT_smooth = data_3pc_C[nbt][idx_tt,:,:] #[0] 3pc states from tt trials
            projs_All_CT = np.array([[f_subspace.proj_on_plane(p[:,t], vec_normal_CT, center_CT) for t in range(len(tbins))] for p in data_3pc_CT_smooth]) # projections on the plane
            projs_All_CT = np.swapaxes(projs_All_CT,1,2)
            
            #compress to 2d
            vecX_, vecY_ = f_subspace.vec_quad_side(projs_CT, sequence = (3,1,2,0)) 
            vecs_new = np.array(f_subspace.basis_correction_3d(vecs_CT, vecX_, vecY_))
            
            projs_All_CT_2d = np.array([f_subspace.proj_2D_coordinates(projs_All_CT[:,:,t], vecs_new) for t in range(projs_All_CT.shape[2])])
            projs_All_CT_2d = np.swapaxes(np.swapaxes(projs_All_CT_2d, 0, 1), 1, 2)
            
            endX_D1s, endX_D2s = [tbins.tolist().index(ed1) for ed1 in end_D1s], [tbins.tolist().index(ed2) for ed2 in end_D2s]
            
            
            euDistT += [np.sqrt(np.sum((projs_All_CT_2d[:,:,endX_D1s].mean(-1) - projs_All_CT_2d[:,:,endX_D2s].mean(-1))**2, axis=1))]
            
            #if bslMethod == 'shuff':
            #    for npm in range(nPerms):
            #        # time-shuffled time series
            #        rng = np.random.default_rng()
            #        projs_All_CT_shuff = rng.permuted(projs_All_CT, axis=2)
                    
            #        euDistT_shuff += [np.sqrt(np.sum((projs_All_CT_shuff[:,:,endX_D1s].mean(-1)-projs_All_CT_shuff[:,:,endX_D2s].mean(-1))**2, axis=1))]
            
            #else:
                #specific baseline period
            #    endX_D1b, endX_D2b = [tbins.tolist().index(ed1b) for ed1b in end_D1b], [tbins.tolist().index(ed2b) for ed2b in end_D2b]
            #    euDistT_shuff += [np.sqrt(np.sum((projs_All_CT[:,:,endX_D1b].mean(-1)-projs_All_CT[:,:,endX_D2b].mean(-1))**2, axis=1))]
            
        
        euDistT = np.array(euDistT)
        #euDistT_shuff = np.array(euDistT_shuff)
        
        euDists[tt] = euDistT
        #euDists_shuff[tt] = euDistT_shuff
    
    # condition general std for each pseudo pop
    euDistT = np.concatenate((euDists[1], euDists[2]),axis=1)
    #euDistT_shuff = np.concatenate((euDists_shuff[1], euDists_shuff[2]),axis=1)
    
    for tt in ttypes:
        if zscore:
            for i in range(len(euDists[tt])):
                euDists[tt][i,:] = (euDists[tt][i,:]- euDistT[i,:].mean())/euDistT[i,:].std() # euDistT_shuff[j,:].std()#
            
            #for j in range(len(euDists_shuff[tt])):
            #    euDists_shuff[tt][j,:] = (euDists_shuff[tt][j,:]- euDistT[i,:].mean())/ euDistT[i,:].std()
        else:
            for i in range(len(euDists[tt])):
                euDists[tt][i,:] = (euDists[tt][i,:])/euDistT[i,:].std() # euDistT_shuff[j,:].std()#- euDistT[i,:].mean()
            
            #for j in range(len(euDists_shuff[tt])):
            #    euDists_shuff[tt][j,:] = (euDists_shuff[tt][j,:])/euDistT[i,:].std()# euDistT_shuff[j,:].std() # - euDistT[i,:].mean()

    #euDistances = (euDists, euDists_shuff)
    
    return euDists#, euDists_shuff

# In[] retarget vs distraction state changes
def get_euDist2_centroids(geomsC_valid, locs = (0,1,2,3), ttypes = (1,2), bins = 50, dt = 10, tslice = (0,2700),
                   end_D1s = (1300,), end_D2s = (2600,), zscore = False, 
                   nPerms = 50, bslMethod = 'x', end_D1b = (2100,), end_D2b = (2600,)):
    
    

    #-300,0 #
    
    vecs_C, projs_C, projsAll_C, trialInfos_C, data_3pc_C = geomsC_valid
    
    nBoots = len(trialInfos_C)
    
    bins = bins
    dt = dt
    tbins = np.arange(tslice[0], tslice[1], bins)
    
    euDists = {}
    euDists_shuff = {}

    # estimate decodability by ttype
    for tt in ttypes:
        
        euDistT = [] # pca1st 3d coordinates
        #euDistT_shuff = [] # pca1st 3d coordinates
        
        
        for nbt in range(nBoots):
            #for nbt in range(nBoots):
                
            # trial info
            trialInfo_CT = trialInfos_C[nbt]
            trialInfo_CT_tt = trialInfo_CT[trialInfo_CT.type == tt]#.reset_index(drop = True)
            idx_tt = trialInfo_CT_tt.index.tolist()#.trial_index.values
            
            vecs_CT = vecs_C[nbt] # choice plane vecs
            projs_CT = projs_C[nbt] #
            
            

            vec_normal_CT = np.cross(vecs_CT[0],vecs_CT[1]) # plane normal vec
            center_CT = projs_CT.mean(0) # plane center
            
            #data_3pc_CT = data_3pc_C[region][n][nbt][idx_tt,:,:] # 3pc states from tt trials
            
            data_3pc_CT_smooth = data_3pc_C[nbt][idx_tt,:,:] #[0] 3pc states from tt trials
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
            
            
            #if bslMethod == 'shuff':
            #    for npm in range(nPerms):
            #        # time-shuffled time series
            #        rng = np.random.default_rng()
            #        projs_All_CT_shuff = rng.permuted(projs_All_CT, axis=2)
                    
            #        euDistT_shuff += [np.sqrt(np.sum((projs_All_CT_shuff[:,:,endX_D1s].mean(2)-projs_All_CT_shuff[:,:,endX_D2s].mean(2))**2, axis=1))]
            
            #else:
            #    endX_D1b, endX_D2b = [tbins.tolist().index(ed1b) for ed1b in end_D1b], [tbins.tolist().index(ed2b) for ed2b in end_D2b]
            #    euDistT_shuff += [np.sqrt(np.sum((projs_All_CT[:,:,endX_D1b].mean(2)-projs_All_CT[:,:,endX_D2b].mean(2))**2, axis=1))]
            
        
        euDistT = np.array(euDistT)
        #euDistT_shuff = np.array(euDistT_shuff)
        
        euDists[tt] = euDistT
        #euDists_shuff[tt] = euDistT_shuff
    
    # condition general std for each pseudo pop
    euDistT = np.concatenate((euDists[1], euDists[2]),axis=1)
    #euDistT_shuff = np.concatenate((euDists_shuff[1], euDists_shuff[2]),axis=1)
    
    for tt in ttypes:
        if zscore:
            for i in range(len(euDists[tt])):
                euDists[tt][i,:] = (euDists[tt][i,:]- euDistT[i,:].mean())/euDistT[i,:].std() # euDistT_shuff[j,:].std()#
            
            #for j in range(len(euDists_shuff[tt])):
            #    euDists_shuff[tt][j,:] = (euDists_shuff[tt][j,:]- euDistT[i,:].mean())/ euDistT[i,:].std()
        else:
            for i in range(len(euDists[tt])):
                euDists[tt][i,:] = (euDists[tt][i,:])/euDistT[i,:].std() # euDistT_shuff[j,:].std()#- euDistT[i,:].mean()
            
            #for j in range(len(euDists_shuff[tt])):
            #    euDists_shuff[tt][j,:] = (euDists_shuff[tt][j,:])/euDistT[i,:].std()# euDistT_shuff[j,:].std() # - euDistT[i,:].mean()

    #euDistances = (euDists, euDists_shuff)
    
    return euDists#, euDists_shuff

# In[] retarget vs distraction state changes
def get_euDist2_centroids2(geomsC_valid, locs = (0,1,2,3), ttypes = (1,2), bins = 50, dt = 10, tslice = (0,2700),
                   end_D1s = (1300,), end_D2s = (2600,), zscore = False, 
                   nPerms = 50, bslMethod = 'x', end_D1b = (2100,), end_D2b = (2600,)):
    
    

    epsilon = 1e-7
    
    vecs_C, projs_C, projsAll_C, trialInfos_C, data_3pc_C = geomsC_valid
    
    nBoots = len(trialInfos_C)
    
    bins = bins
    dt = dt
    tbins = np.arange(tslice[0], tslice[1], bins)
    
    #euDists_shuff = {}
    
    euDists = {tt:[] for tt in ttypes}
        
    for nbt in range(nBoots):
        #for nbt in range(nBoots):
            
        # trial info
        trialInfo_CT = trialInfos_C[nbt]
        #trialInfo_CT_tt = trialInfo_CT[trialInfo_CT.type == tt]#.reset_index(drop = True)
        #idx_tt = trialInfo_CT_tt.index.tolist()#.trial_index.values
        
        vecs_CT = vecs_C[nbt] # choice plane vecs
        projs_CT = projs_C[nbt] #
        
        

        vec_normal_CT = np.cross(vecs_CT[0],vecs_CT[1]) # plane normal vec
        center_CT = projs_CT.mean(0) # plane center
        
        #data_3pc_CT = data_3pc_C[region][n][nbt][idx_tt,:,:] # 3pc states from tt trials
        
        data_3pc_CT_smooth = data_3pc_C[nbt][:,:,:] #[0] 3pc states from tt trials
        projs_All_CT = np.array([[f_subspace.proj_on_plane(p[:,t], vec_normal_CT, center_CT) for t in range(len(tbins))] for p in data_3pc_CT_smooth]) # projections on the plane
        projs_All_CT = np.swapaxes(projs_All_CT,1,2)
        
        #compress to 2d
        vecX_, vecY_ = f_subspace.vec_quad_side(projs_CT, sequence = (3,1,2,0)) 
        vecs_new = np.array(f_subspace.basis_correction_3d(vecs_CT, vecX_, vecY_))
        
        projs_All_CT_2d = np.array([f_subspace.proj_2D_coordinates(projs_All_CT[:,:,t], vecs_new) for t in range(projs_All_CT.shape[2])])
        projs_All_CT_2d = np.swapaxes(np.swapaxes(projs_All_CT_2d, 0, 1), 1, 2)
        
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

        std_pooled = np.concatenate([euDistT[1],euDistT[2]]).std() + epsilon
        for tt in ttypes:
            euDistT[tt] = np.array(euDistT[tt]) / std_pooled
            #euDistT = np.array(euDistT)
            euDists[tt] += [euDistT[tt]]
    
    
    return euDists#, euDists_shuff



#%%
def lda12X(test_InfoT, hidden_statesT, tRange, locs = (0,1,2,3), ttypes = (1,2), nPCs = (0,15), dt = 10, bins = 50, 
           pca_tWinX=None, avg_method = 'conditional_mean', nBoots = 50, permDummy = True, shuff_excludeInv = False,
           Y_columnsLabels = ['choice','ttype','loc1','loc2','locX'], zbsl = True, tbsl = (-300,0)):
    
    pd.options.mode.chained_assignment = None
    epsilon = 0.0000001
    
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
        #hidden_statesT[:,ch,:] = scale(hidden_statesT[:,ch,:]) #01 scale for each channel
        #hidden_statesT[:,ch,:] = (hidden_statesT[:,ch,:] - hidden_statesT[:,ch,:].mean(axis=0)) / hidden_statesT[:,ch,:].std(axis=0) #detrended standard scaler
        #hidden_statesT[:,ch,:] = (hidden_statesT[:,ch,:] - hidden_statesT[:,ch,idxxBsl[0]:idxxBsl[1]].mean())/hidden_statesT[:,ch,idxxBsl[0]:idxxBsl[1]].std() # Z to baseline
    
    
    #pca_tWinX = None
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
    #print(f'{condition[1]}, {evr.round(4)[0:5]}')
    
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
        
        full_label1 = Y[:,toDecode_X1].astype('int') #.astype('str') # locKey
        full_label2 = Y[:,toDecode_X2].astype('int') #.astype('str') # locKey


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
            
            train_label1 = full_label1[train_setID]#Y[train_setID,toDecode_X1].astype('int') #.astype('str') # locKey
            train_label2 = full_label2[train_setID]#Y[train_setID,toDecode_X2].astype('int') #.astype('str') # locKey

            test_label1 = full_label1[test_setID]#Y[test_setID,toDecode_X1].astype('int') #.astype('str') # locKey
            test_label2 = full_label2[test_setID]#Y[test_setID,toDecode_X2].astype('int') #.astype('str') # locKey
            
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

#%%
def get_EVR(test_InfoT, hidden_statesT, tRange, locs = (0,1,2,3), ttypes = (1,2), nPCs = (0,15), dt = 10, bins = 50, 
           pca_tWinX=None, avg_method = 'conditional_mean', nBoots = 50, permDummy = True, shuff_excludeInv = False,
           Y_columnsLabels = ['choice','ttype','loc1','loc2','locX'], zbsl = True, tbsl = (-300,0)):
    
    pd.options.mode.chained_assignment = None
    epsilon = 0.0000001
    
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
        #hidden_statesT[:,ch,:] = scale(hidden_statesT[:,ch,:]) #01 scale for each channel
        #hidden_statesT[:,ch,:] = (hidden_statesT[:,ch,:] - hidden_statesT[:,ch,:].mean(axis=0)) / hidden_statesT[:,ch,:].std(axis=0) #detrended standard scaler
        #hidden_statesT[:,ch,:] = (hidden_statesT[:,ch,:] - hidden_statesT[:,ch,idxxBsl[0]:idxxBsl[1]].mean())/hidden_statesT[:,ch,idxxBsl[0]:idxxBsl[1]].std() # Z to baseline
    
    
    #pca_tWinX = None
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
    #print(f'{condition[1]}, {evr.round(4)[0:5]}')
    return evr
#%%
def rnns_lda12X(modelD, trialInfo, X_, Y_, tRange, dt = 10, bins = 50, frac = 0.8, locs = (0,1,2,3), ttypes = (1,2), 
                nPCs = (0,15), nIters = 1, nBoots = 10, pca_tWins = ((300,1300), (1600,2600),), tbsl = (-300,0), 
                conditions = None, pDummy = True, toPlot = True, label = '', shuff_excludeInv = False):
    
    epsilon = 0.0000001

    
    pDummy = pDummy
    nIters = nIters#0
    nBoots = nBoots
    
    
    # decodability with/without permutation P value
    bins = bins
    tslice = (tRange.min(), tRange.max()+dt)
    
    tbins = np.arange(tslice[0], tslice[1], bins) #
    
    locs = locs
    ttypes = ttypes
    nPCs = nPCs
    
    nPCs = nPCs#[0,10]#pseudo_Pop.shape[2]

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
        #test_set = test_X.cpu().numpy()

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

#%%
def rnns_EVRs(modelD, trialInfo, X_, Y_, tRange, dt = 10, bins = 50, frac = 0.8, locs = (0,1,2,3), ttypes = (1,2), 
                nPCs = (0,15), nIters = 1, nBoots = 10, pca_tWins = ((300,1300), (1600,2600),), tbsl = (-300,0), 
                conditions = None, pDummy = True, toPlot = True, label = '', shuff_excludeInv = False):
    
    epsilon = 0.0000001

    
    pDummy = pDummy
    nIters = nIters#0
    nBoots = nBoots
    
    
    # decodability with/without permutation P value
    bins = bins
    tslice = (tRange.min(), tRange.max()+dt)
    
    tbins = np.arange(tslice[0], tslice[1], bins) #
    
    locs = locs
    ttypes = ttypes
    nPCs = nPCs
    
    nPCs = nPCs#[0,10]#pseudo_Pop.shape[2]

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
        #test_set = test_X.cpu().numpy()

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
#%%
def ldattX(test_InfoT, hidden_statesT, tRange, locs = (0,1,2,3), ttypes = (1,2), nPCs = (0,15), dt = 10, bins = 50, 
           pca_tWinX=None, avg_method = 'conditional_mean', nBoots = 50, permDummy = True, shuff_excludeInv = False,
           Y_columnsLabels = ['choice','ttype','loc1','loc2','locX'], zbsl = True, tbsl = (-300,0)):
    
    pd.options.mode.chained_assignment = None
    epsilon = 0.0000001
    
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
        #hidden_statesT[:,ch,:] = scale(hidden_statesT[:,ch,:]) #01 scale for each channel
        #hidden_statesT[:,ch,:] = (hidden_statesT[:,ch,:] - hidden_statesT[:,ch,:].mean(axis=0)) / hidden_statesT[:,ch,:].std(axis=0) #detrended standard scaler
        #hidden_statesT[:,ch,:] = (hidden_statesT[:,ch,:] - hidden_statesT[:,ch,idxxBsl[0]:idxxBsl[1]].mean())/hidden_statesT[:,ch,idxxBsl[0]:idxxBsl[1]].std() # Z to baseline
    
    
    #pca_tWinX = None
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
    #print(f'{condition[1]}, {evr.round(4)[0:5]}')
    
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
        
        train_label1 = full_label1[train_setID]#Y[train_setID,toDecode_X1].astype('int') #.astype('str') # locKey
        test_label1 = full_label1[test_setID]#Y[test_setID,toDecode_X1].astype('int') #.astype('str') # locKey
        
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

#%%
def rnns_ldattX(modelD, trialInfo, X_, Y_, tRange, dt = 10, bins = 50, frac = 0.8, locs = (0,1,2,3), ttypes = (1,2), 
                nPCs = (0,15), nIters = 1, nBoots = 10, pca_tWins = ((300,1300), (1600,2600),), tbsl = (-300,0), 
                conditions = None, pDummy = True, toPlot = True, label = '', shuff_excludeInv = False):
    
    epsilon = 0.0000001

    
    pDummy = pDummy
    nIters = nIters#0
    nBoots = nBoots
    
    
    # decodability with/without permutation P value
    bins = bins
    tslice = (tRange.min(), tRange.max()+dt)
    
    tbins = np.arange(tslice[0], tslice[1], bins) #
    
    locs = locs
    ttypes = ttypes
    nPCs = nPCs
    
    nPCs = nPCs#[0,10]#pseudo_Pop.shape[2]

    performancesX_tt = []
    performancesX_tt_shuff = []
    evrs = []

    for n in range(nIters):
        t_IterOn = time.time()
        print(f'Iter = {n}')
        
        
        _, _, _, test_setID, test_X, _ = f_simulation.split_dataset_balance(X_, Y_, trialInfo, frac = frac, ranseed=n)

        test_Info = trialInfo.loc[test_setID,:].reset_index(drop = True)
        #test_set = test_X.cpu().numpy()

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
        
        ax.set_title(f'Trial Type', fontsize = 25, pad = 20)
            
            
        plt.tight_layout()
        plt.subplots_adjust(top = 0.95)
        plt.suptitle(f'{label}, Full Space', fontsize = 25, y=1) #, Arti_Noise = {arti_noise_level}
        plt.show()

    return performancesX, performancesX_shuff, evrs

#%%
def generate_bslVectors(models_dict, trialInfo, X_, Y_, tRange, checkpoints, avgInterval, dt = 10, locs = (0,1,2,3), ttypes = (1,2), 
                         adaptPCA=None, adaptEVR = None, pca_tWins = ((300,1300), (1600,2600),), nBoots = 1, fracBoots = 0.5, nPerms = 100, toPlot=False, avgMethod='conditional_time'):
    
    epsilon = 0.0000001
    
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
    #pca_tWin = np.hstack((np.arange(800,1300,dt, dtype = int),np.arange(2100,2600,dt, dtype = int))).tolist() #
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

        #idx1 = test_Info.id # original index
        #idx2 = test_Info.index.to_list() # reset index
        
        ### main test
        dataN = hidden_states.swapaxes(1,2)
        
        
        for ch in range(dataN.shape[1]):
            #dataN[:,ch,:] = (dataN[:,ch,:] - dataN[:,ch,idxxBsl[0]:idxxBsl[1]].mean())/dataN[:,ch,idxxBsl[0]:idxxBsl[1]].std() # Z to baseline, 
            #dataN[:,ch,:] = scale(dataN[:,ch,:])#01 scale for each channel
            dataN[:,ch,:] = (dataN[:,ch,:] - dataN[:,ch,:].mean() + epsilon) / (dataN[:,ch,:].std() + epsilon) #standard scaler
            #dataN[:,ch,:] = (dataN[:,ch,:] - dataN[:,ch,:].mean(axis=0)) / dataN[:,ch,:].std(axis=0) # detrened standard scaler
            
        ttypesDummy = (1,)
        
        
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
                        
            #idxT,_ = f_subspace.split_set(dataN, frac=fracBoots, ranseed=nboot)
            test_InfoT = test_Info.rename(columns={'ttype':'type'})
            idxT1,idxT2 = f_subspace.split_set_balance(np.arange(dataN.shape[0]), test_InfoT, frac=fracBoots, ranseed=nbt)
            dataT1, dataT2 = dataN[idxT1,:,:], dataN[idxT2,:,:]
            trialInfoT1, trialInfoT2 = test_InfoT.loc[idxT1,:].reset_index(drop=True), test_InfoT.loc[idxT2,:].reset_index(drop=True)
            trialInfoT1['locs'] = trialInfoT1['loc1'].astype(str) + '_' + trialInfoT1['loc2'].astype(str)
            trialInfoT2['locs'] = trialInfoT2['loc1'].astype(str) + '_' + trialInfoT2['loc2'].astype(str)
            
            
            #trialInfoT = trialInfoT.rename(columns={'ttype':'type'})
            #trialInfoT['type'] = 1
            
            adaptPCA_T = None if (adaptPCA is None) else adaptPCA[n][0]
            adaptEVR_T = None if (adaptEVR is None) else adaptEVR[n][0]
            
            vecs_D1, projs_D1, projsAll_D1, _, trialInfos_D1, _, _, evr_1st_1, _ = f_subspace.plane_fitting_analysis(dataT1, trialInfoT1, pca_tWinX, checkpoints, tRange, avgInterval, locs, ttypes, 
                                                                                                                     adaptPCA=adaptPCA_T, adaptEVR = adaptEVR_T, toPlot=toPlot, avgMethod = avgMethod) #Dummy, decode_method = decode_method
            vecs_D2, projs_D2, projsAll_D2, _, trialInfos_D2, _, _, evr_1st_2, _ = f_subspace.plane_fitting_analysis(dataT2, trialInfoT2, pca_tWinX, checkpoints, tRange, avgInterval, locs, ttypes,
                                                                                                                     adaptPCA=adaptPCA_T, adaptEVR = adaptEVR_T, toPlot=toPlot, avgMethod = avgMethod)
            
            #decodability_projD
              
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

#%%
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
#%%
def generate_bslVectors2(models_dict, trialInfo, X_, Y_, tRange, checkpoints, avgInterval, dt = 10, locs = (0,1,2,3), ttypes = (1,2), 
                         adaptPCA=None, adaptEVR = None, pca_tWins = ((300,1300), (1600,2600),), nBoots = 1, fracBoots = 0.8, nPerms = 100, toPlot=False, avgMethod='conditional_time'):
    
    epsilon = 0.0000001
    
    nIters = len(models_dict)
    

    vecs_bsl_All = []
    projs_bsl_All = []
    projsAll_bsl_All = []
    trialInfos_bsl_All = []
    pca1s_bsl_All = []
    
    
    evrs_1st_All = []
    
    _, _, _, test_setID, test_X, _ = f_simulation.split_dataset_balance(X_, Y_, trialInfo, frac = 0.5)
    
    test_Info = trialInfo.loc[test_setID,:].reset_index(drop = True)
    test_set = test_X.cpu().numpy()

    test_Info.loc1 = test_Info.loc1.astype(int)
    test_Info.loc2 = test_Info.loc2.astype(int)
    test_Info.choice = test_Info.choice.astype(int)
    
    # if specify applied time window of pca
    pca_tWin = np.hstack([np.arange(i[0],i[1],dt, dtype = int) for i in pca_tWins]).tolist() #
    #pca_tWin = np.hstack((np.arange(800,1300,dt, dtype = int),np.arange(2100,2600,dt, dtype = int))).tolist() #
    pca_tWinX = [tRange.tolist().index(i) for i in pca_tWin]
    
    for n in range(nIters):
        
        vecs_bsl = {}
        projs_bsl = {}
        projsAll_bsl = {}
        trialInfos_bsl = {}
        pca1s_bsl = []


        for tt in ttypes:
            
            trialInfos_bsl[tt] = []
            

        for cp in checkpoints:
            
            vecs_bsl[cp] = {}
            projs_bsl[cp] = {}
            projsAll_bsl[cp] = {}
            
            for tt in ttypes:
                
                vecs_bsl[cp][tt] = {1:[], 2:[]}
                projs_bsl[cp][tt] = {1:[], 2:[]}
                projsAll_bsl[cp][tt] = {1:[], 2:[]}
        
        evrs_1st = np.zeros((nBoots, 3))
        
        modelD = models_dict[n]['rnn']
        hidden_states, _ = modelD(test_X)
        hidden_states = hidden_states.data.cpu().detach().numpy()

        #idx1 = test_Info.id # original index
        #idx2 = test_Info.index.to_list() # reset index
        
        ### main test
        dataN = hidden_states.swapaxes(1,2)
        
        
        for ch in range(dataN.shape[1]):
            #dataN[:,ch,:] = (dataN[:,ch,:] - dataN[:,ch,idxxBsl[0]:idxxBsl[1]].mean())/dataN[:,ch,idxxBsl[0]:idxxBsl[1]].std() # Z to baseline, 
            #dataN[:,ch,:] = scale(dataN[:,ch,:])#01 scale for each channel
            dataN[:,ch,:] = (dataN[:,ch,:] - dataN[:,ch,:].mean() + epsilon) / (dataN[:,ch,:].std() + epsilon) #standard scaler
            #dataN[:,ch,:] = (dataN[:,ch,:] - dataN[:,ch,:].mean(axis=0)) / dataN[:,ch,:].std(axis=0) # detrened standard scaler
            
        ttypesDummy = (1,)
        
        
        for nbt in range(nPerms):
            # if run multiple iterations, enable the following
            pca1s_bsl.append([])
            
            for tt in ttypes: #Dummy
                trialInfos_bsl[tt].append([])
            
            for cp in checkpoints:
                for tt in ttypes: #Dummy
                    for ll in (1,2,):
                        vecs_bsl[cp][tt][ll].append([])
                        projs_bsl[cp][tt][ll].append([])
                        projsAll_bsl[cp][tt][ll].append([])
                        
            #idxT,_ = f_subspace.split_set(dataN, frac=fracBoots, ranseed=nboot)
            test_InfoT = test_Info.rename(columns={'ttype':'type'})
            idxT,_ = f_subspace.split_set_balance(np.arange(dataN.shape[0]), test_InfoT, frac=fracBoots, ranseed=nbt)
            dataT = dataN[idxT,:,:]
            trialInfoT = test_InfoT.loc[idxT,:].reset_index(drop=True)
            trialInfoT['locs'] = trialInfoT['loc1'].astype(str) + '_' + trialInfoT['loc2'].astype(str)
            
            
            #trialInfoT = trialInfoT.rename(columns={'ttype':'type'})
            #trialInfoT['type'] = 1
            
            adaptPCA_T = None if (adaptPCA is None) else adaptPCA[n][0]
            adaptEVR_T = None if (adaptEVR is None) else adaptEVR[n][0]
            
            vecs_D, projs_D, projsAll_D, _, trialInfos_D, _, _, evr_1st, pca1 = f_subspace.plane_fitting_analysis(dataT, trialInfoT, pca_tWinX, checkpoints, tRange, avgInterval, locs, ttypes, 
                                                                                                                  adaptPCA=adaptPCA_T, adaptEVR = adaptEVR_T, toPlot=toPlot, avgMethod = avgMethod) #Dummy, decode_method = decode_method
            #decodability_projD
              
            for cp in checkpoints:
                for tt in ttypes: #Dummy
                    for ll in (1,2,):
                        vecs_bsl[cp][tt][ll][nbt] = vecs_D[cp][tt][ll]
                        projs_bsl[cp][tt][ll][nbt] = projs_D[cp][tt][ll]
                        projsAll_bsl[cp][tt][ll][nbt] = projsAll_D[cp][tt][ll]
                        
                        #vecs[cp][tt][ll] += [vecs_D[cp][tt][ll]]
                        #projs[cp][tt][ll] += [projs_D[cp][tt][ll]]
                        #projsAll[cp][tt][ll] += [projsAll_D[cp][tt][ll]]
            
            for tt in ttypes: #Dummy
                trialInfos_bsl[tt][nbt] = trialInfos_D[tt]
                #trialInfos[tt] += [trialInfos_D[tt]]
                
            print(f'EVRs: {evr_1st.round(5)}')
            evrs_1st[0,:] = evr_1st
            
            
        vecs_bsl_All += [vecs_bsl]
        projs_bsl_All += [projs_bsl]
        projsAll_bsl_All += [projsAll_bsl]
        trialInfos_bsl_All += [trialInfos_bsl]
        pca1s_bsl_All += [pca1s_bsl]
        
        evrs_1st_All += [evrs_1st]
        
        del modelD
        torch.cuda.empty_cache()
        gc.collect()

    return vecs_bsl_All, projs_bsl_All, projsAll_bsl_All, trialInfos_bsl_All, pca1s_bsl_All, evrs_1st_All
#%%
def get_angleAlignment_itemPairs_bsl2(geoms_valid, geoms_bsl, checkpoints, locs = (0,1,2,3), ttypes = (1,2), sequence=(0,1,3,2)):
    
    vecs, projs, projsAll, trialInfos = geoms_valid
    vecs_bsl, projs_bsl, projsAll_bsl, trialInfos_bsl = geoms_bsl
    
    nBoots = len(trialInfos[1])
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
                    
                    cT11_shuff, _, cP11_shuff, _, _ = f_subspace.angle_alignment_coplanar(vecs[cp][tt][1][nbt], projs[cp][tt][1][nbt], 
                                                                            vecs_bsl[cp][tt][1][npm], projs_bsl[cp][tt][1][npm], sequence=sequence)
                    cT22_shuff, _, cP22_shuff, _, _ = f_subspace.angle_alignment_coplanar(vecs[cp][tt][2][nbt], projs[cp][tt][2][nbt], 
                                                                            vecs_bsl[cp][tt][2][npm], projs_bsl[cp][tt][2][npm], sequence=sequence)
                    cT12_shuff, _, cP12_shuff, _, _ = f_subspace.angle_alignment_coplanar(vecs[cp][tt][1][nbt], projs[cp][tt][1][nbt], 
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
#%%
def ldaX_chunking(test_InfoT, hidden_statesT, tRange, items = (), chunk = (), nPCs = (0,15), dt = 10, bins = 50, 
                  pca_tWinX=None, avg_method = 'conditional_mean', nBoots = 50, 
                  Y_columnsLabels = ['stim1','stim2','chunk', 'itemPairs'], zbsl = True, tbsl = (-300,0)):
    
    pd.options.mode.chained_assignment = None
    epsilon = 1e-7
    
    tslice = (tRange.min(), tRange.max()+dt)

    items = items if len(items) > 0 else tuple(test_InfoT['stim1'].unique())
    chunk = chunk if len(chunk) > 0 else tuple(test_InfoT['chunk'].unique())

    tbins = np.arange(tslice[0], tslice[1], bins) # 
    itemPairs = tuple(permutations(items,2))
    subConditions = tuple(product(itemPairs, chunk))
    
    if zbsl and (tbsl[0] in tRange):
        bslx1, bslx2 = tRange.tolist().index(tbsl[0]), tRange.tolist().index(tbsl[1])

    ### scaling
    for ch in range(hidden_statesT.shape[1]):
        hidden_statesT[:,ch,:] = (hidden_statesT[:,ch,:] - hidden_statesT[:,ch,:].mean() + epsilon) / (hidden_statesT[:,ch,:].std() + epsilon) #standard scaler
        
        if zbsl and (tbsl[0] in tRange):
            bsl_mean, bsl_std = hidden_statesT[:,ch,bslx1:bslx2].mean(), hidden_statesT[:,ch,bslx1:bslx2].std()
            hidden_statesT[:,ch,:] = (hidden_statesT[:,ch,:] - bsl_mean + epsilon) / (bsl_std + epsilon) #standard scaler
        #hidden_statesT[:,ch,:] = scale(hidden_statesT[:,ch,:]) #01 scale for each channel
        #hidden_statesT[:,ch,:] = (hidden_statesT[:,ch,:] - hidden_statesT[:,ch,:].mean(axis=0)) / hidden_statesT[:,ch,:].std(axis=0) #detrended standard scaler
        #hidden_statesT[:,ch,:] = (hidden_statesT[:,ch,:] - hidden_statesT[:,ch,idxxBsl[0]:idxxBsl[1]].mean())/hidden_statesT[:,ch,idxxBsl[0]:idxxBsl[1]].std() # Z to baseline
    
    
    #pca_tWinX = None
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
            i1, i2 = sc[0]
            ck = sc[1]
            idxx = test_InfoT_[(test_InfoT_.stim1 == i1) & (test_InfoT_.stim2 == i2) & (test_InfoT_.chunk == ck)].index.tolist()
            if len(idxx)>0:
                X_regionT_ += [hidden_statesTT[idxx,:,:].mean(axis=0)]
        
        hidden_statesTT = np.concatenate(X_regionT_, axis=-1)
        
    elif avg_method == 'conditional_mean':
        # if conditional average + avged over time, get mean for each condition, then vstack condition_avgs, shape = chs, n_conditions
        hidden_statesTT_ = []
        test_InfoT_ = test_InfoT.reset_index(drop=True)
        for sc in subConditions:
            i1, i2 = sc[0]
            ck = sc[1]
            
            idxx = test_InfoT_[(test_InfoT_.stim1 == i1) & (test_InfoT_.stim2 == i2) & (test_InfoT_.chunk == ck)].index.tolist()
            if len(idxx)>0:
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
    #print(f'{condition[1]}, {evr.round(4)[0:5]}')
    
    # apply transform to all trials
    hidden_statesTP = np.zeros((hidden_statesT.shape[0], npc, hidden_statesT.shape[2]))
    for trial in range(hidden_statesT.shape[0]):
        hidden_statesTP[trial,:,:] = pca.transform(hidden_statesT[trial,:,:].T).T
    
    # downsample to resolution of tbins
    ntrialT, ncellT, ntimeT = hidden_statesTP.shape
    hidden_statesTP = np.mean(hidden_statesTP.reshape(ntrialT, ncellT, int(ntimeT/(bins/dt)),int(bins/dt)),axis = 3)
    
    
    performance1X, performance2X = {ck:[] for ck in chunk}, {ck:[] for ck in chunk}
    performancePX = {ck:[] for ck in chunk}
    performanceCX = []

    performance1X_shuff, performance2X_shuff = {ck:[] for ck in chunk}, {ck:[] for ck in chunk}
    performancePX_shuff = {ck:[] for ck in chunk}
    performanceCX_shuff = []
    
    # item1 item2 and itemPairs decoding
    for ck in chunk:
        test_InfoT_ = test_InfoT[test_InfoT.chunk == ck]
        idxT = test_InfoT_.index
        Y = test_InfoT_.loc[:,Y_columnsLabels].values
        full_setP = hidden_statesTP[idxT,:,:]
        ntrial = len(test_InfoT_)

        toDecode1_X = Y_columnsLabels.index('stim1')
        toDecode2_X = Y_columnsLabels.index('stim2')
        toDecodeP_X = Y_columnsLabels.index('itemPairs')
        
        full_label1 = Y[:,toDecode1_X].astype('int') #.astype('str') 
        full_label2 = Y[:,toDecode2_X].astype('int') #.astype('str') 
        full_labelP = Y[:,toDecodeP_X].astype('str') 


        # fully random
        full_label1_shuff = np.random.permutation(full_label1)
        full_label2_shuff = np.random.permutation(full_label2)
        full_labelP_shuff = np.random.permutation(full_labelP)
        
        # initialize array to store decoding results
        performance1_XT = np.zeros((nBoots, len(tbins),len(tbins)))
        performance2_XT = np.zeros((nBoots, len(tbins),len(tbins)))
        performanceP_XT = np.zeros((nBoots, len(tbins),len(tbins)))
        
        # permutation with shuffled label
        performance1_XT_shuff = np.zeros((nBoots, len(tbins),len(tbins)))
        performance2_XT_shuff = np.zeros((nBoots, len(tbins),len(tbins)))
        performanceP_XT_shuff = np.zeros((nBoots, len(tbins),len(tbins)))

        for nbt in range(nBoots):
            ### split into train and test sets
            train_setID = np.sort(np.random.choice(ntrial, round(0.5*ntrial),replace = False))
            test_setID = np.sort(np.random.choice(np.setdiff1d(np.arange(ntrial), train_setID, assume_unique=True), (ntrial-train_setID.shape[0]),replace = False))

            # split dataset by index
            train_setP = full_setP[train_setID,:,:]
            test_setP = full_setP[test_setID,:,:]
            
            # split labels by index
            train_label1 = full_label1[train_setID]
            train_label2 = full_label2[train_setID]
            train_labelP = full_labelP[train_setID]
            
            test_label1 = full_label1[test_setID]
            test_label2 = full_label2[test_setID]
            test_labelP = full_labelP[test_setID]
            
            train_label1_shuff, test_label1_shuff = full_label1_shuff[train_setID], full_label1_shuff[test_setID]
            train_label2_shuff, test_label2_shuff = full_label2_shuff[train_setID], full_label2_shuff[test_setID]
            train_labelP_shuff, test_labelP_shuff = full_labelP_shuff[train_setID], full_labelP_shuff[test_setID]
            
            
            for t in range(len(tbins)):
                for t_ in range(len(tbins)):    
                    
                    # decode with valid labels
                    pfmT1 = f_decoding.LDAPerformance(train_setP[:,nPCs[0]:nPCs[1],t], test_setP[:,nPCs[0]:nPCs[1],t_], 
                                                      train_label1, test_label1)
                    
                    pfmT2 = f_decoding.LDAPerformance(train_setP[:,nPCs[0]:nPCs[1],t], test_setP[:,nPCs[0]:nPCs[1],t_], 
                                                      train_label2, test_label2)
                                        
                    pfmTP = f_decoding.LDAPerformance(train_setP[:,nPCs[0]:nPCs[1],t], test_setP[:,nPCs[0]:nPCs[1],t_], 
                                                      train_labelP, test_labelP)
                    
                    # decode with shuffled labels
                    pfmT1_shuff = f_decoding.LDAPerformance(train_setP[:,nPCs[0]:nPCs[1],t], test_setP[:,nPCs[0]:nPCs[1],t_], 
                                                      train_label1_shuff, test_label1_shuff)
                    
                    pfmT2_shuff = f_decoding.LDAPerformance(train_setP[:,nPCs[0]:nPCs[1],t], test_setP[:,nPCs[0]:nPCs[1],t_], 
                                                      train_label2_shuff, test_label2_shuff)
                    
                    pfmTP_shuff = f_decoding.LDAPerformance(train_setP[:,nPCs[0]:nPCs[1],t], test_setP[:,nPCs[0]:nPCs[1],t_], 
                                                      train_labelP_shuff, test_labelP_shuff)
                    
                    # store decoding results
                    performance1_XT[nbt,t,t_] = pfmT1
                    performance2_XT[nbt,t,t_] = pfmT2
                    performanceP_XT[nbt,t,t_] = pfmTP
                    
                    performance1_XT_shuff[nbt,t,t_] = pfmT1_shuff
                    performance2_XT_shuff[nbt,t,t_] = pfmT2_shuff
                    performanceP_XT_shuff[nbt,t,t_] = pfmTP_shuff
        
        performance1X[ck] = performance1_XT
        performance2X[ck] = performance2_XT
        performancePX[ck] = performanceP_XT
        
        performance1X_shuff[ck] = performance1_XT_shuff
        performance2X_shuff[ck] = performance2_XT_shuff
        performancePX_shuff[ck] = performanceP_XT_shuff
    
    
    # chunk decoding
    Y = test_InfoT.loc[:,Y_columnsLabels].values
    full_setP = hidden_statesTP[:,:,:]
    ntrial = len(test_InfoT_)

    toDecodeC_X = Y_columnsLabels.index('chunk')
    full_labelC = Y[:,toDecodeC_X].astype('int') #.astype('str') 

    # fully random
    full_labelC_shuff = np.random.permutation(full_labelC)
    
    # initialize array to store decoding results
    performanceC_XT = np.zeros((nBoots, len(tbins),len(tbins)))
    # permutation with shuffled label
    performanceC_XT_shuff = np.zeros((nBoots, len(tbins),len(tbins)))
    
    for nbt in range(nBoots):
        ### split into train and test sets
        train_setID = np.sort(np.random.choice(ntrial, round(0.5*ntrial),replace = False))
        test_setID = np.sort(np.random.choice(np.setdiff1d(np.arange(ntrial), train_setID, assume_unique=True), (ntrial-train_setID.shape[0]),replace = False))

        # split dataset by index
        train_setP = full_setP[train_setID,:,:]
        test_setP = full_setP[test_setID,:,:]
        
        # split labels by index
        train_labelC = full_labelC[train_setID]
        test_labelC = full_labelC[test_setID]
        train_labelC_shuff, test_labelC_shuff = full_labelC_shuff[train_setID], full_labelC_shuff[test_setID]
        
        for t in range(len(tbins)):
            for t_ in range(len(tbins)):    
                
                # decode with valid labels
                pfmTC = f_decoding.LDAPerformance(train_setP[:,nPCs[0]:nPCs[1],t], test_setP[:,nPCs[0]:nPCs[1],t_], 
                                                    train_labelC, test_labelC)
                
                # decode with shuffled labels
                pfmTC_shuff = f_decoding.LDAPerformance(train_setP[:,nPCs[0]:nPCs[1],t], test_setP[:,nPCs[0]:nPCs[1],t_], 
                                                    train_labelC_shuff, test_labelC_shuff)
                
                
                # store decoding results
                performanceC_XT[nbt,t,t_] = pfmTC
                performanceC_XT_shuff[nbt,t,t_] = pfmTC_shuff
                        
    performanceCX = performanceC_XT
    performanceCX_shuff = performanceC_XT_shuff
        
        
    return (performance1X, performance2X, performancePX, performanceCX), (performance1X_shuff, performance2X_shuff, performancePX_shuff, performanceCX_shuff), evr
#%%
def rnns_ldaX_chunking(modelD, trialInfo, X_, Y_, tRange, dt = 10, bins = 50, frac = 0.8, items = (), chunk = (), 
                       nPCs = (0,15), nIters = 1, nBoots = 10, pca_tWins = ((300,1300), (1600,2600),), tbsl = (-300,0), 
                       conditions = None, pDummy = True, toPlot = True, label = ''):
    
    epsilon = 0.0000001

    
    pDummy = pDummy
    nIters = nIters#0
    nBoots = nBoots
    
    
    # decodability with/without permutation P value
    bins = bins
    tslice = (tRange.min(), tRange.max()+dt)
    
    tbins = np.arange(tslice[0], tslice[1], bins) #
    
    items = items if len(items) > 0 else tuple(trialInfo['stim1'].unique())
    chunk = chunk if len(chunk) > 0 else tuple(trialInfo['chunk'].unique())
    itemPairs = tuple(permutations(items,2))
    subConditions = tuple(product(itemPairs, chunk))
    
    nPCs = nPCs
    
    conditions = (('ttype', 1), ('ttype', 2)) if conditions == None else conditions
    
    performances1X = {ck:[] for ck in chunk}
    performances2X = {ck:[] for ck in chunk}
    performancesPX = {ck:[] for ck in chunk}
    performancesCX = []
    
    performances1X_shuff = {ck:[] for ck in chunk}
    performances2X_shuff = {ck:[] for ck in chunk}
    performancesPX_shuff = {ck:[] for ck in chunk}
    performancesCX_shuff = []
    
    evrs = []

    for n in range(nIters):
        t_IterOn = time.time()
        print(f'Iter = {n}')
        
        
        _, _, _, test_setID, test_X, _ = f_simulation.split_dataset_balance_chunking(X_, Y_, trialInfo, frac = frac, ranseed=n)

        test_Info = trialInfo.loc[test_setID,:].reset_index(drop = True)
               
        
        hidden_states, _ = modelD(test_X)
        hidden_states = hidden_states.data.cpu().detach().numpy()
        hidden_states = np.swapaxes(hidden_states, 1, 2) # shape as ntrials * nchs * ntime
        
        pca_tWin = np.hstack([np.arange(i[0],i[1],dt, dtype = int) for i in pca_tWins]).tolist() #
        pca_tWinX = [tRange.tolist().index(i) for i in pca_tWin]

        performancesX_n, performancesX_n_shuff, evr_n = ldaX_chunking(test_Info, hidden_states, tRange, items = items, chunk = chunk, 
                                                                      nPCs = nPCs, dt = dt, bins = bins, pca_tWinX=pca_tWinX, 
                                                                      nBoots = nBoots, tbsl=tbsl)
        
        for ck in chunk:
            performances1X[ck] += [performancesX_n[0][ck]] 
            performances2X[ck] += [performancesX_n[1][ck]] 
            performancesPX[ck] += [performancesX_n[2][ck]]
            
            performances1X_shuff[ck] += [performancesX_n_shuff[0][ck]]
            performances2X_shuff[ck] += [performancesX_n_shuff[1][ck]]
            performancesPX_shuff[ck] += [performancesX_n_shuff[2][ck]]
        
        performancesCX += [performancesX_n[3]]
        performancesCX_shuff += [performancesX_n_shuff[3]]

        evrs += [evr_n]
        
        print(f'tIter = {(time.time() - t_IterOn):.4f}s')
            
    performancesX  = (performances1X, performances2X, performancesPX, performancesCX)
    performancesX_shuff = (performances1X_shuff, performances2X_shuff, performancesPX_shuff, performancesCX_shuff)
    
    # plot out
    if toPlot:
        
        plt.figure(figsize=(50, 24), dpi=100)
        
        for ck in chunk:
            chunkT = 'Chunked' if ck==1 else 'Non-Chunked'
            chunkT_ = 'Chunked' if ck==1 else 'Non-Chunked'
            
            performanceT1 = performances1X[ck]
            performanceT1_shuff = performances1X_shuff[ck]
            
            performanceT2 = performances2X[ck]
            performanceT2_shuff = performances2X_shuff[ck]
            
            performanceTC = performancesCX[ck]
            performanceTC_shuff = performancesCX_shuff[ck]
            
            performanceTP = performancesPX[ck]
            performanceTP_shuff = performancesPX_shuff[ck]
            
            pfm1 = np.array(performanceT1)
            pfm1_shuff = np.concatenate(np.array(performanceT1_shuff),axis=2)
            
            pfm2 = np.array(performanceT2)
            pfm2_shuff = np.concatenate(np.array(performanceT2_shuff),axis=2)
            
            pfmC = np.array(performanceTC)
            pfmC_shuff = np.concatenate(np.array(performanceTC_shuff),axis=2)
            
            pfmP = np.array(performanceTP)
            pfmP_shuff = np.concatenate(np.array(performanceTP_shuff),axis=2)
            
            pvalues1 = np.ones((len(tbins), len(tbins)))
            pvalues2 = np.ones((len(tbins), len(tbins)))
            pvaluesC = np.ones((len(tbins), len(tbins)))
            pvaluesP = np.ones((len(tbins), len(tbins)))
            
            if pDummy!=True:
                for t in range(len(tbins)):
                    for t_ in range(len(tbins)):
                        pvalues1[t,t_] = f_stats.permutation_p(pfm1.mean(axis = 0)[t,t_], pfm1_shuff[t,t_,:], tail = 'greater')
                        pvalues2[t,t_] = f_stats.permutation_p(pfm2.mean(axis = 0)[t,t_], pfm2_shuff[t,t_,:], tail = 'greater')
                        pvaluesC[t,t_] = f_stats.permutation_p(pfmC.mean(axis = 0)[t,t_], pfmC_shuff[t,t_,:], tail = 'greater')
                        pvaluesP[t,t_] = f_stats.permutation_p(pfmP.mean(axis = 0)[t,t_], pfmP_shuff[t,t_,:], tail = 'greater')
                        
            
            vmax = 1
            
            plt.subplot(4,2,ck+1)
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
            ax.set_xticklabels(['S1', 'S2', 'Go-Cue'], rotation=0, fontsize = 20)
            ax.set_xlabel('Test Timebin (ms)', labelpad = 10, fontsize = 25)
            ax.set_yticks([list(tbins).index(i) for i in [0, 1300, 2600]])
            ax.set_yticklabels(['S1', 'S2', 'Go-Cue'], fontsize = 20)
            ax.set_ylabel('Train Timebin (ms)', labelpad = 10, fontsize = 25)
            
            cbar = ax.collections[0].colorbar
            # here set the labelsize by 20
            cbar.ax.tick_params(labelsize=20)
            
            ax.set_title(f'{chunkT_}, Item1', fontsize = 30, pad = 20)
            
            
            # item2
            plt.subplot(4,2,ck+3)
            ax = plt.gca()
            sns.heatmap(pd.DataFrame(pfm2.mean(0), index=tbins,columns=tbins), cmap = 'magma', vmin = 0.25, vmax = vmax, ax = ax)#, vmax = 0.6, xticklabels = 100, yticklabels = 100, vmin = cbar_min, vmax = cbar_max
            
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
            
            ax.set_title(f'{chunkT_}, Item2', fontsize = 30, pad = 20)
            
            # chunk
            plt.subplot(4,2,ck+5)
            ax = plt.gca()
            sns.heatmap(pd.DataFrame(pfmC.mean(0), index=tbins,columns=tbins), cmap = 'magma', vmin = 0.25, vmax = vmax, ax = ax)#, vmax = 0.6, xticklabels = 100, yticklabels = 100, vmin = cbar_min, vmax = cbar_max
            
            smooth_scale = 10
            z = ndimage.zoom(pvaluesC, smooth_scale)
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
            
            ax.set_title(f'{chunkT_}, Chunk', fontsize = 30, pad = 20)
            
            
            # itemPairs
            plt.subplot(4,2,ck+7)
            ax = plt.gca()
            sns.heatmap(pd.DataFrame(pfmP.mean(0), index=tbins,columns=tbins), cmap = 'magma', vmin = 0.25, vmax = vmax, ax = ax)#, vmax = 0.6, xticklabels = 100, yticklabels = 100, vmin = cbar_min, vmax = cbar_max
            
            smooth_scale = 10
            z = ndimage.zoom(pvaluesP, smooth_scale)
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
            
            ax.set_title(f'{chunkT_}, ItemPairs', fontsize = 30, pad = 20)
            
        plt.tight_layout()
        plt.subplots_adjust(top = 0.95)
        plt.suptitle(f'{label}, Full Space', fontsize = 35, y=1) #, Arti_Noise = {arti_noise_level}
        plt.show()

    return performancesX, performancesX_shuff, evrs
#%%

#%%






























































#%%
##################
# UNUSED FOR NOW #
##################
#%%
# In[] cosTheta, cosPsi, sse. Compare within type, between time points, between locations

def plot_planeAlignment(geoms_valid, geoms_shuff, checkpoints, checkpointsLabels, locs = (0,1,2,3), ttypes = (1,2), nIters = 20, nPerms = 50, nBoots = 1, label = '', pdummy = True, toPlot = True, sequence=(0,1,3,2)):
    
    vecs, projs, projsAll, trialInfos = geoms_valid
    vecs_shuff, projs_shuff, projsAll_shuff, trialInfos_shuff = geoms_shuff
    
    cosTheta_11, cosTheta_12, cosTheta_22 = {},{},{}
    cosPsi_11, cosPsi_12, cosPsi_22 = {},{},{}
    sse_11, sse_12, sse_22 = {},{},{}
    cosSimi_11, cosSimi_22, cosSimi_12 = {},{},{}
    ai_11, ai_22, ai_12 = {},{},{}
    
    for tt in ttypes: #Dummy
        ttype = 'Retarget' if tt==1 else 'Distraction'
        
        cosTheta_11T = np.zeros((nIters, nBoots, len(checkpoints), len(checkpoints)))
        cosTheta_22T = np.zeros((nIters, nBoots, len(checkpoints), len(checkpoints)))
        cosTheta_12T = np.zeros((nIters, nBoots, len(checkpoints), len(checkpoints)))
        
        cosPsi_11T = np.zeros((nIters, nBoots, len(checkpoints), len(checkpoints)))
        cosPsi_22T = np.zeros((nIters, nBoots, len(checkpoints), len(checkpoints)))
        cosPsi_12T = np.zeros((nIters, nBoots, len(checkpoints), len(checkpoints)))
        
        sse_11T = np.zeros((nIters, nBoots, len(checkpoints), len(checkpoints)))
        sse_22T = np.zeros((nIters, nBoots, len(checkpoints), len(checkpoints)))
        sse_12T = np.zeros((nIters, nBoots, len(checkpoints), len(checkpoints)))
        
        cS_11T = np.zeros((nIters, nBoots, len(checkpoints), len(checkpoints)))
        cS_22T = np.zeros((nIters, nBoots, len(checkpoints), len(checkpoints)))
        cS_12T = np.zeros((nIters, nBoots, len(checkpoints), len(checkpoints)))
        
        ai_11T = np.zeros((nIters, nBoots, len(checkpoints), len(checkpoints)))
        ai_22T = np.zeros((nIters, nBoots, len(checkpoints), len(checkpoints)))
        ai_12T = np.zeros((nIters, nBoots, len(checkpoints), len(checkpoints)))
        
        for n in range(nIters):
            for nbt in range(nBoots):
                for nc, cp in enumerate(checkpoints):
                    for nc_, cp_ in enumerate(checkpoints):
                        cT11, _, cP11, _, s11 = f_subspace.angle_alignment_coplanar(vecs[cp][tt][1][n][nbt], projs[cp][tt][1][n][nbt], vecs[cp_][tt][1][n][nbt], projs[cp_][tt][1][n][nbt], sequence=sequence)
                        cT22, _, cP22, _, s22 = f_subspace.angle_alignment_coplanar(vecs[cp][tt][2][n][nbt], projs[cp][tt][2][n][nbt], vecs[cp_][tt][2][n][nbt], projs[cp_][tt][2][n][nbt], sequence=sequence)
                        cT12, _, cP12, _, s12 = f_subspace.angle_alignment_coplanar(vecs[cp][tt][1][n][nbt], projs[cp][tt][1][n][nbt], vecs[cp_][tt][2][n][nbt], projs[cp_][tt][2][n][nbt], sequence=sequence)
                        
                        cS11 = f_subspace.config_correlation(vecs[cp][tt][1][n][nbt], projs[cp][tt][1][n][nbt], vecs[cp_][tt][1][n][nbt], projs[cp_][tt][1][n][nbt])
                        cS22 = f_subspace.config_correlation(vecs[cp][tt][2][n][nbt], projs[cp][tt][2][n][nbt], vecs[cp_][tt][2][n][nbt], projs[cp_][tt][2][n][nbt])
                        cS12 = f_subspace.config_correlation(vecs[cp][tt][1][n][nbt], projs[cp][tt][1][n][nbt], vecs[cp_][tt][2][n][nbt], projs[cp_][tt][2][n][nbt])
                        
                        ai11 = f_subspace.get_simple_AI(projs[cp][tt][1][n][nbt], projs[cp_][tt][1][n][nbt], max_dim=2)
                        ai22 = f_subspace.get_simple_AI(projs[cp][tt][2][n][nbt], projs[cp_][tt][2][n][nbt], max_dim=2)
                        ai12 = f_subspace.get_simple_AI(projs[cp][tt][1][n][nbt], projs[cp_][tt][2][n][nbt], max_dim=2)
                        
                        cosTheta_11T[n,nbt,nc,nc_], cosTheta_22T[n,nbt,nc,nc_], cosTheta_12T[n,nbt,nc,nc_] = cT11, cT22, cT12
                        cosPsi_11T[n,nbt,nc,nc_], cosPsi_22T[n,nbt,nc,nc_], cosPsi_12T[n,nbt,nc,nc_] = cP11, cP22, cP12
                        sse_11T[n,nbt,nc,nc_], sse_22T[n,nbt,nc,nc_], sse_12T[n,nbt,nc,nc_] = s11, s22, s12
                        
                        cS_11T[n,nbt,nc,nc_], cS_22T[n,nbt,nc,nc_], cS_12T[n,nbt,nc,nc_] = cS11, cS22, cS12
                        ai_11T[n,nbt,nc,nc_], ai_22T[n,nbt,nc,nc_], ai_12T[n,nbt,nc,nc_] = ai11, ai22, ai12
                        
                    
        cosTheta_11[tt] = cosTheta_11T
        cosTheta_22[tt] = cosTheta_22T
        cosTheta_12[tt] = cosTheta_12T
        
        cosPsi_11[tt] = cosPsi_11T
        cosPsi_22[tt] = cosPsi_22T
        cosPsi_12[tt] = cosPsi_12T
        
        sse_11[tt] = sse_11T
        sse_22[tt] = sse_22T
        sse_12[tt] = sse_12T
        
        cosSimi_11[tt] = cS_11T
        cosSimi_22[tt] = cS_22T
        cosSimi_12[tt] = cS_12T
        
        ai_11[tt] = ai_11T
        ai_22[tt] = ai_22T
        ai_12[tt] = ai_12T
        
        
        
        ## shuff
        pcosTheta_11T, pcosTheta_22T, pcosTheta_12T = np.ones((len(checkpoints), len(checkpoints))), np.ones((len(checkpoints), len(checkpoints))), np.ones((len(checkpoints), len(checkpoints)))
        pcosPsi_11T, pcosPsi_22T, pcosPsi_12T = np.ones((len(checkpoints), len(checkpoints))), np.ones((len(checkpoints), len(checkpoints))), np.ones((len(checkpoints), len(checkpoints)))
        psse_11T, psse_22T, psse_12T = np.ones((len(checkpoints), len(checkpoints))), np.ones((len(checkpoints), len(checkpoints))), np.ones((len(checkpoints), len(checkpoints)))
        
        pcS_11T, pcS_22T, pcS_12T = np.ones((len(checkpoints), len(checkpoints))), np.ones((len(checkpoints), len(checkpoints))), np.ones((len(checkpoints), len(checkpoints)))
        pai_11T, pai_22T, pai_12T = np.ones((len(checkpoints), len(checkpoints))), np.ones((len(checkpoints), len(checkpoints))), np.ones((len(checkpoints), len(checkpoints)))
        
        if pdummy == False:
            
            
            cosTheta_11_shuffT = np.zeros((nIters, nBoots*nPerms, len(checkpoints), len(checkpoints)))
            cosTheta_22_shuffT = np.zeros((nIters, nBoots*nPerms, len(checkpoints), len(checkpoints)))
            cosTheta_12_shuffT = np.zeros((nIters, nBoots*nPerms, len(checkpoints), len(checkpoints)))
            
            cosPsi_11_shuffT = np.zeros((nIters, nBoots*nPerms, len(checkpoints), len(checkpoints)))
            cosPsi_22_shuffT = np.zeros((nIters, nBoots*nPerms, len(checkpoints), len(checkpoints)))
            cosPsi_12_shuffT = np.zeros((nIters, nBoots*nPerms, len(checkpoints), len(checkpoints)))
            
            sse_11_shuffT = np.zeros((nIters, nBoots*nPerms, len(checkpoints), len(checkpoints)))
            sse_22_shuffT = np.zeros((nIters, nBoots*nPerms, len(checkpoints), len(checkpoints)))
            sse_12_shuffT = np.zeros((nIters, nBoots*nPerms, len(checkpoints), len(checkpoints)))
            
            cS_11_shuffT = np.zeros((nIters, nBoots*nPerms, len(checkpoints), len(checkpoints)))
            cS_22_shuffT = np.zeros((nIters, nBoots*nPerms, len(checkpoints), len(checkpoints)))
            cS_12_shuffT = np.zeros((nIters, nBoots*nPerms, len(checkpoints), len(checkpoints)))
            
            ai_11_shuffT = np.zeros((nIters, nBoots*nPerms, len(checkpoints), len(checkpoints)))
            ai_22_shuffT = np.zeros((nIters, nBoots*nPerms, len(checkpoints), len(checkpoints)))
            ai_12_shuffT = np.zeros((nIters, nBoots*nPerms, len(checkpoints), len(checkpoints)))
            
            for n in range(nIters):
                for nbt in range(nBoots*nPerms):
                    for nc, cp in enumerate(checkpoints):
                        for nc_, cp_ in enumerate(checkpoints):
                            cT11, _, cP11, _, s11 = f_subspace.angle_alignment_coplanar(vecs_shuff[cp][tt][1][n][nbt], projs_shuff[cp][tt][1][n][nbt], vecs_shuff[cp_][tt][1][n][nbt], projs_shuff[cp_][tt][1][n][nbt], sequence=sequence)
                            cT22, _, cP22, _, s22 = f_subspace.angle_alignment_coplanar(vecs_shuff[cp][tt][2][n][nbt], projs_shuff[cp][tt][2][n][nbt], vecs_shuff[cp_][tt][2][n][nbt], projs_shuff[cp_][tt][2][n][nbt], sequence=sequence)
                            cT12, _, cP12, _, s12 = f_subspace.angle_alignment_coplanar(vecs_shuff[cp][tt][1][n][nbt], projs_shuff[cp][tt][1][n][nbt], vecs_shuff[cp_][tt][2][n][nbt], projs_shuff[cp_][tt][2][n][nbt], sequence=sequence)
                            
                            cS11 = f_subspace.config_correlation(vecs_shuff[cp][tt][1][n][nbt], projs_shuff[cp][tt][1][n][nbt], vecs_shuff[cp_][tt][1][n][nbt], projs_shuff[cp_][tt][1][n][nbt])
                            cS22 = f_subspace.config_correlation(vecs_shuff[cp][tt][2][n][nbt], projs_shuff[cp][tt][2][n][nbt], vecs_shuff[cp_][tt][2][n][nbt], projs_shuff[cp_][tt][2][n][nbt])
                            cS12 = f_subspace.config_correlation(vecs_shuff[cp][tt][1][n][nbt], projs_shuff[cp][tt][1][n][nbt], vecs_shuff[cp_][tt][2][n][nbt], projs_shuff[cp_][tt][2][n][nbt])
                            
                            ai11 = f_subspace.get_simple_AI(projs_shuff[cp][tt][1][n][nbt], projs_shuff[cp_][tt][1][n][nbt], max_dim=2)
                            ai22 = f_subspace.get_simple_AI(projs_shuff[cp][tt][2][n][nbt], projs_shuff[cp_][tt][2][n][nbt], max_dim=2)
                            ai12 = f_subspace.get_simple_AI(projs_shuff[cp][tt][1][n][nbt], projs_shuff[cp_][tt][2][n][nbt], max_dim=2)
                            
                            
                            cosTheta_11_shuffT[n,nbt,nc,nc_], cosTheta_22_shuffT[n,nbt,nc,nc_], cosTheta_12_shuffT[n,nbt,nc,nc_] = cT11, cT22, cT12
                            cosPsi_11_shuffT[n,nbt,nc,nc_], cosPsi_22_shuffT[n,nbt,nc,nc_], cosPsi_12_shuffT[n,nbt,nc,nc_] = cP11, cP22, cP12
                            sse_11_shuffT[n,nbt,nc,nc_], sse_22_shuffT[n,nbt,nc,nc_], sse_12_shuffT[n,nbt,nc,nc_] = s11, s22, s12
                                                    
                            cS_11_shuffT[n,nbt,nc,nc_], cS_22_shuffT[n,nbt,nc,nc_], cS_12_shuffT[n,nbt,nc,nc_] = cS11, cS22, cS12
                            ai_11_shuffT[n,nbt,nc,nc_], ai_22_shuffT[n,nbt,nc,nc_], ai_12_shuffT[n,nbt,nc,nc_] = ai11, ai22, ai12
            
            
            cosTheta_11_shuff_all = np.concatenate([cosTheta_11_shuffT[:,nbt,:,:] for nbt in range(nBoots*nPerms)])
            cosTheta_22_shuff_all = np.concatenate([cosTheta_22_shuffT[:,nbt,:,:] for nbt in range(nBoots*nPerms)])
            cosTheta_12_shuff_all = np.concatenate([cosTheta_12_shuffT[:,nbt,:,:] for nbt in range(nBoots*nPerms)])
            
            cosPsi_11_shuff_all = np.concatenate([cosPsi_11_shuffT[:,nbt,:,:] for nbt in range(nBoots*nPerms)])
            cosPsi_22_shuff_all = np.concatenate([cosPsi_22_shuffT[:,nbt,:,:] for nbt in range(nBoots*nPerms)])
            cosPsi_12_shuff_all = np.concatenate([cosPsi_12_shuffT[:,nbt,:,:] for nbt in range(nBoots*nPerms)])
            
            sse_11_shuff_all = np.concatenate([sse_11_shuffT[:,nbt,:,:] for nbt in range(nBoots*nPerms)])
            sse_22_shuff_all = np.concatenate([sse_22_shuffT[:,nbt,:,:] for nbt in range(nBoots*nPerms)])
            sse_12_shuff_all = np.concatenate([sse_12_shuffT[:,nbt,:,:] for nbt in range(nBoots*nPerms)])
            
            cS_11_shuff_all = np.concatenate([cS_11_shuffT[:,nbt,:,:] for nbt in range(nBoots*nPerms)])
            cS_22_shuff_all = np.concatenate([cS_22_shuffT[:,nbt,:,:] for nbt in range(nBoots*nPerms)])
            cS_12_shuff_all = np.concatenate([cS_12_shuffT[:,nbt,:,:] for nbt in range(nBoots*nPerms)])
            
            ai_11_shuff_all = np.concatenate([ai_11_shuffT[:,nbt,:,:] for nbt in range(nBoots*nPerms)])
            ai_22_shuff_all = np.concatenate([ai_22_shuffT[:,nbt,:,:] for nbt in range(nBoots*nPerms)])
            ai_12_shuff_all = np.concatenate([ai_12_shuffT[:,nbt,:,:] for nbt in range(nBoots*nPerms)])
            
            
            for i in range(len(checkpoints)):
                for j in range(len(checkpoints)):
                    #pPerm_11T[i,j] = f_stats.permutation_p(psi_11T.mean(axis=1).mean(axis=0)[i,j], psi_11_shuff_all[:,i,j], tail = 'two')#/2
                    #pPerm_22T[i,j] = f_stats.permutation_p(psi_22T.mean(axis=1).mean(axis=0)[i,j], psi_22_shuff_all[:,i,j], tail = 'two')#/2
                    #pPerm_12T[i,j] = f_stats.permutation_p(psi_12T.mean(axis=1).mean(axis=0)[i,j], psi_12_shuff_all[:,i,j], tail = 'two')#/2
                    
                    #pPerm_11T[i,j] = f_stats.permutation_pCI(psi_11T.mean(axis=1)[:,i,j].round(3), psi_11_shuff_all[:,i,j].round(3), CI_size = 1.96, value_range=(0,1)) #, CI_range = (2.5,97.5)
                    #pPerm_22T[i,j] = f_stats.permutation_pCI(psi_22T.mean(axis=1)[:,i,j].round(3), psi_22_shuff_all[:,i,j].round(3), CI_size = 1.96, value_range=(0,1)) #, CI_range = (2.5,97.5)
                    #pPerm_12T[i,j] = f_stats.permutation_pCI(psi_12T.mean(axis=1)[:,i,j].round(3), psi_12_shuff_all[:,i,j].round(3), CI_size = 1.96, value_range=(0,1)) #, CI_range = (2.5,97.5)
                    
                    #pPerm_11T[i,j] = f_stats.permutation_p(np.median(psi_11T.mean(axis=1)[:,i,j]), psi_11_shuff_all[:,i,j], tail = 'two')
                    #pPerm_22T[i,j] = f_stats.permutation_p(np.median(psi_22T.mean(axis=1)[:,i,j]), psi_22_shuff_all[:,i,j], tail = 'two')
                    #pPerm_12T[i,j] = f_stats.permutation_p(np.median(psi_12T.mean(axis=1)[:,i,j]), psi_12_shuff_all[:,i,j], tail = 'two')
                    
                    cT11, cT22, cT12 = cosTheta_11T.mean(axis=1)[:,i,j].round(5), cosTheta_22T.mean(axis=1)[:,i,j].round(5), cosTheta_12T.mean(axis=1)[:,i,j].round(5)
                    cP11, cP22, cP12 = cosPsi_11T.mean(axis=1)[:,i,j].round(5), cosPsi_22T.mean(axis=1)[:,i,j].round(5), cosPsi_12T.mean(axis=1)[:,i,j].round(5)
                    s11, s22, s12 = sse_11T.mean(axis=1)[:,i,j].round(5), sse_22T.mean(axis=1)[:,i,j].round(5), sse_12T.mean(axis=1)[:,i,j].round(5)
                    
                    cS11, cS22, cS12 = cS_11T.mean(axis=1)[:,i,j].round(5), cS_22T.mean(axis=1)[:,i,j].round(5), cS_12T.mean(axis=1)[:,i,j].round(5)
                    ai11, ai22, ai12 = ai_11T.mean(axis=1)[:,i,j].round(5), ai_22T.mean(axis=1)[:,i,j].round(5), ai_12T.mean(axis=1)[:,i,j].round(5)
                    
                    # drop nan values
                    #cP11, cP22, cP12 = cP11[~np.isnan(cP11)], cP22[~np.isnan(cP22)], cP12[~np.isnan(cP12)]
                    #s11, s22, s12 = s11[~np.isnan(s11)], s22[~np.isnan(s22)], s12[~np.isnan(s12)]
                    
                    # shuff distribution
                    cT11_shuff, cT22_shuff, cT12_shuff = cosTheta_11_shuff_all[:,i,j].round(5), cosTheta_22_shuff_all[:,i,j].round(5), cosTheta_12_shuff_all[:,i,j].round(5)
                    cP11_shuff, cP22_shuff, cP12_shuff = cosPsi_11_shuff_all[:,i,j].round(5), cosPsi_22_shuff_all[:,i,j].round(5), cosPsi_12_shuff_all[:,i,j].round(5)
                    s11_shuff, s22_shuff, s12_shuff = sse_11_shuff_all[:,i,j].round(5), sse_22_shuff_all[:,i,j].round(5), sse_12_shuff_all[:,i,j].round(5)
                    
                    #cP11_shuff, cP22_shuff, cP12_shuff = cP11_shuff[~np.isnan(cP11_shuff)], cP22_shuff[~np.isnan(cP22_shuff)], cP12_shuff[~np.isnan(cP12_shuff)]
                    #s11_shuff, s22_shuff, s12_shuff = s11_shuff[~np.isnan(s11_shuff)], s22_shuff[~np.isnan(s22_shuff)], s12_shuff[~np.isnan(s12_shuff)]
                    
                    cS11_shuff, cS22_shuff, cS12_shuff = cS_11_shuff_all[:,i,j].round(5), cS_22_shuff_all[:,i,j].round(5), cS_12_shuff_all[:,i,j].round(5)
                    ai11_shuff, ai22_shuff, ai12_shuff = ai_11_shuff_all[:,i,j].round(5), ai_22_shuff_all[:,i,j].round(5), ai_12_shuff_all[:,i,j].round(5)
                    
                    # compare distributions and calculate p values
                    pcosTheta_11T[i,j] = stats.kstest(cT11, cT11_shuff)[-1]
                    pcosTheta_22T[i,j] = stats.kstest(cT22, cT22_shuff)[-1]
                    pcosTheta_12T[i,j] = stats.kstest(cT12, cT12_shuff)[-1]
                    
                    pcosPsi_11T[i,j] = stats.kstest(cP11, cP11_shuff)[-1]
                    pcosPsi_22T[i,j] = stats.kstest(cP22, cP22_shuff)[-1]
                    pcosPsi_12T[i,j] = stats.kstest(cP12, cP12_shuff)[-1]
                    
                    #pcosPsi_11T[i,j] = stats.kstest(np.abs(cP11), np.abs(cP11_shuff))[-1]
                    #pcosPsi_22T[i,j] = stats.kstest(np.abs(cP22), np.abs(cP22_shuff))[-1]
                    #pcosPsi_12T[i,j] = stats.kstest(np.abs(cP12), np.abs(cP12_shuff))[-1]
                    
                    psse_11T[i,j] = stats.kstest(s11, s11_shuff)[-1]
                    psse_22T[i,j] = stats.kstest(s22, s22_shuff)[-1]
                    psse_12T[i,j] = stats.kstest(s12, s12_shuff)[-1]
                    
                    pcS_11T[i,j] = stats.kstest(cS11, cS11_shuff)[-1] # #stats.uniform.cdf
                    pcS_22T[i,j] = stats.kstest(cS22, cS22_shuff)[-1] # #stats.uniform.cdf
                    pcS_12T[i,j] = stats.kstest(cS12, cS12_shuff)[-1] # #stats.uniform.cdf
                    
                    pai_11T[i,j] = f_stats.permutation_p(ai11.mean(), ai11_shuff) # stats.kstest(ai11, ai11_shuff)[-1] # 
                    pai_22T[i,j] = f_stats.permutation_p(ai22.mean(), ai22_shuff) # stats.kstest(ai22, ai22_shuff)[-1] # 
                    pai_12T[i,j] = f_stats.permutation_p(ai12.mean(), ai12_shuff) # stats.kstest(ai12, ai12_shuff)[-1] # 
        
        if toPlot:
            
            angleCheckPoints = np.linspace(0,np.pi,13).round(5)
            cmap = plt.get_cmap('coolwarm')
            norm = mcolors.BoundaryNorm(np.flip(np.cos(angleCheckPoints).round(5)),cmap.N, clip=True)
            
            ### cosTheta
            plt.figure(figsize=(16, 6), dpi=100)
            plt.subplot(1,3,1)
            ax = plt.gca()
            im = ax.imshow(cosTheta_11T.mean(axis=1).mean(axis=0), cmap=cmap, norm=norm, aspect='auto')
            #im = ax.imshow(cosTheta_11T.mean(axis=1).mean(axis=0), cmap='coolwarm', aspect='auto', norm=mcolors.SymLogNorm(linthresh=0.1, linscale=0.1, vmin=-1.0, vmax=1.0, base=10))
            #im = ax.imshow(np.median(cosTheta_11T.mean(axis=1), axis=0), cmap='jet', aspect='auto', vmin=0, vmax=1)
            for i in range(len(checkpoints)):
                for j in range(len(checkpoints)):
                    if 0.05 < pcosTheta_11T[i,j] <= 0.1:
                        text = ax.text(j, i, '+', ha="center", va="center", fontsize=6)
                    elif 0.01 < pcosTheta_11T[i,j] <= 0.05:
                        text = ax.text(j, i, '*', ha="center", va="center", fontsize=6)
                    elif pcosTheta_11T[i,j] <= 0.01:
                        text = ax.text(j, i, '**', ha="center", va="center", fontsize=6)
                        
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
            #im = ax.imshow(cosTheta_22T.mean(axis=1).mean(axis=0), cmap='coolwarm', aspect='auto', norm=mcolors.SymLogNorm(linthresh=0.1, linscale=0.1, vmin=-1.0, vmax=1.0, base=10))
            #im = ax.imshow(np.median(cosTheta_22T.mean(axis=1), axis=0), cmap='jet', aspect='auto', vmin=0, vmax=1)
            for i in range(len(checkpoints)):
                for j in range(len(checkpoints)):
                    if 0.05 < pcosTheta_22T[i,j] <= 0.1:
                        text = ax.text(j, i, '+', ha="center", va="center", fontsize=6)
                    elif 0.01 < pcosTheta_22T[i,j] <= 0.05:
                        text = ax.text(j, i, '*', ha="center", va="center", fontsize=6)
                    elif pcosTheta_22T[i,j] <= 0.01:
                        text = ax.text(j, i, '**', ha="center", va="center", fontsize=6)
                        
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
            #im = ax.imshow(cosTheta_12T.mean(axis=1).mean(axis=0), cmap='coolwarm', aspect='auto', norm=mcolors.SymLogNorm(linthresh=0.1, linscale=0.1, vmin=-1.0, vmax=1.0, base=10))
            #im = ax.imshow(np.median(cosTheta_12T.mean(axis=1), axis=0), cmap='jet', aspect='auto', vmin=0, vmax=1)
            for i in range(len(checkpoints)):
                for j in range(len(checkpoints)):
                    if 0.05 < pcosTheta_12T[i,j] <= 0.1:
                        text = ax.text(j, i, '+', ha="center", va="center", fontsize=6)
                    elif 0.01 < pcosTheta_12T[i,j] <= 0.05:
                        text = ax.text(j, i, '*', ha="center", va="center", fontsize=6)
                    elif pcosTheta_12T[i,j] <= 0.01:
                        text = ax.text(j, i, '**', ha="center", va="center", fontsize=6)
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
            plt.suptitle(f'Principal Angle (θ), {ttype}, {label}', fontsize = 20, y=1)
            plt.show()
            
            
            
            
            ### cosPsi
            plt.figure(figsize=(16, 6), dpi=100)
            plt.subplot(1,3,1)
            ax = plt.gca()
            im = ax.imshow(cosPsi_11T.mean(axis=1).mean(axis=0), cmap=cmap, norm=norm, aspect='auto')
            #im = ax.imshow(cosPsi_11T.mean(axis=1).mean(axis=0), cmap='coolwarm', aspect='auto', norm=mcolors.SymLogNorm(linthresh=0.1, linscale=0.1, vmin=-1.0, vmax=1.0, base=10))
            #im = ax.imshow(np.abs(cosPsi_11T).mean(axis=1).mean(axis=0), cmap='coolwarm', aspect='auto', norm=mcolors.SymLogNorm(linthresh=0.1, linscale=0.1, vmin=0, vmax=1.0, base=10))
            for i in range(len(checkpoints)):
                for j in range(len(checkpoints)):
                    if 0.05 < pcosPsi_11T[i,j] <= 0.1:
                        text = ax.text(j, i, '+', ha="center", va="center", fontsize=6)
                    elif 0.01 < pcosPsi_11T[i,j] <= 0.05:
                        text = ax.text(j, i, '*', ha="center", va="center", fontsize=6)
                    elif pcosPsi_11T[i,j] <= 0.01:
                        text = ax.text(j, i, '**', ha="center", va="center", fontsize=6)
                        
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
            #im = ax.imshow(cosPsi_22T.mean(axis=1).mean(axis=0), cmap='coolwarm', aspect='auto', norm=mcolors.SymLogNorm(linthresh=0.1, linscale=0.1, vmin=-1.0, vmax=1.0, base=10))
            #im = ax.imshow(np.abs(cosPsi_22T).mean(axis=1).mean(axis=0), cmap='coolwarm', aspect='auto', norm=mcolors.SymLogNorm(linthresh=0.1, linscale=0.1, vmin=0, vmax=1.0, base=10))
            for i in range(len(checkpoints)):
                for j in range(len(checkpoints)):
                    if 0.05 < pcosPsi_22T[i,j] <= 0.1:
                        text = ax.text(j, i, '+', ha="center", va="center", fontsize=6)
                    elif 0.01 < pcosPsi_22T[i,j] <= 0.05:
                        text = ax.text(j, i, '*', ha="center", va="center", fontsize=6)
                    elif pcosPsi_22T[i,j] <= 0.01:
                        text = ax.text(j, i, '**', ha="center", va="center", fontsize=6)
                        
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
            #im = ax.imshow(cosPsi_12T.mean(axis=1).mean(axis=0), cmap='coolwarm', aspect='auto', norm=mcolors.SymLogNorm(linthresh=0.1, linscale=0.1, vmin=-1.0, vmax=1.0, base=10))
            #im = ax.imshow(np.abs(cosPsi_12T).mean(axis=1).mean(axis=0), cmap='coolwarm', aspect='auto', norm=mcolors.SymLogNorm(linthresh=0.1, linscale=0.1, vmin=0, vmax=1.0, base=10))
            for i in range(len(checkpoints)):
                for j in range(len(checkpoints)):
                    if 0.05 < pcosPsi_12T[i,j] <= 0.1:
                        text = ax.text(j, i, '+', ha="center", va="center", fontsize=6)
                    elif 0.01 < pcosPsi_12T[i,j] <= 0.05:
                        text = ax.text(j, i, '*', ha="center", va="center", fontsize=6)
                    elif pcosPsi_12T[i,j] <= 0.01:
                        text = ax.text(j, i, '**', ha="center", va="center", fontsize=6)
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
            plt.suptitle(f'Representational Alignment (Ψ), {ttype}, {label}', fontsize = 20, y=1)
            #plt.suptitle(f'abs(cosPsi),  ttype={tt}', fontsize = 15, y=1)
            plt.show()
        
            ### sse
            plt.figure(figsize=(15, 5), dpi=100)
            plt.subplot(1,3,1)
            ax = plt.gca()
            im = ax.imshow(sse_11T.mean(axis=1).mean(axis=0), cmap='coolwarm', aspect='auto', vmin=0) #, vmax=1
            #im = ax.imshow(np.median(cosTheta_11T.mean(axis=1), axis=0), cmap='jet', aspect='auto', vmin=0, vmax=1)
            for i in range(len(checkpoints)):
                for j in range(len(checkpoints)):
                    if 0.05 < psse_11T[i,j] <= 0.1:
                        text = ax.text(j, i, '+', ha="center", va="center", fontsize=6)
                    elif 0.01 < psse_11T[i,j] <= 0.05:
                        text = ax.text(j, i, '*', ha="center", va="center", fontsize=6)
                    elif psse_11T[i,j] <= 0.01:
                        text = ax.text(j, i, '**', ha="center", va="center", fontsize=6)
                        
            ax.set_title('11', pad = 10)
            ax.set_xticks([n for n in range(len(checkpoints))])
            ax.set_xticklabels(checkpointsLabels, fontsize = 6)
            ax.set_yticks([n for n in range(len(checkpoints))])
            ax.set_yticklabels(checkpointsLabels, fontsize = 6)
            ax.set_xlabel('Item 1', fontsize = 10)
            ax.set_ylabel('Item 1', fontsize = 10)
            
            
            
            plt.subplot(1,3,2)
            ax = plt.gca()
            im = ax.imshow(sse_22T.mean(axis=1).mean(axis=0), cmap='coolwarm', aspect='auto', vmin=0) #, vmax=1
            #im = ax.imshow(np.median(cosTheta_22T.mean(axis=1), axis=0), cmap='jet', aspect='auto', vmin=0, vmax=1)
            for i in range(len(checkpoints)):
                for j in range(len(checkpoints)):
                    if 0.05 < psse_22T[i,j] <= 0.1:
                        text = ax.text(j, i, '+', ha="center", va="center", fontsize=6)
                    elif 0.01 < psse_22T[i,j] <= 0.05:
                        text = ax.text(j, i, '*', ha="center", va="center", fontsize=6)
                    elif psse_22T[i,j] <= 0.01:
                        text = ax.text(j, i, '**', ha="center", va="center", fontsize=6)
                        
            ax.set_title('22', pad = 10)
            ax.set_xticks([n for n in range(len(checkpoints))])
            ax.set_xticklabels(checkpointsLabels, fontsize = 6)
            ax.set_yticks([n for n in range(len(checkpoints))])
            ax.set_yticklabels(checkpointsLabels, fontsize = 6)
            ax.set_xlabel('Item 2', fontsize = 10)
            ax.set_ylabel('Item 2', fontsize = 10)
            
            
            
            plt.subplot(1,3,3)
            ax = plt.gca()
            im = ax.imshow(sse_12T.mean(axis=1).mean(axis=0), cmap='coolwarm', aspect='auto', vmin=0) #, vmax=1
            #im = ax.imshow(np.median(cosTheta_12T.mean(axis=1), axis=0), cmap='jet', aspect='auto', vmin=0, vmax=1)
            for i in range(len(checkpoints)):
                for j in range(len(checkpoints)):
                    if 0.05 < psse_12T[i,j] <= 0.1:
                        text = ax.text(j, i, '+', ha="center", va="center", fontsize=6)
                    elif 0.01 < psse_12T[i,j] <= 0.05:
                        text = ax.text(j, i, '*', ha="center", va="center", fontsize=6)
                    elif psse_12T[i,j] <= 0.01:
                        text = ax.text(j, i, '**', ha="center", va="center", fontsize=6)
            ax.set_title('12', pad = 10)
            ax.set_xticks([n for n in range(len(checkpoints))])
            ax.set_xticklabels(checkpointsLabels, fontsize = 6)
            ax.set_yticks([n for n in range(len(checkpoints))])
            ax.set_yticklabels(checkpointsLabels, fontsize = 6)
            ax.set_xlabel('Item 2', fontsize = 10)
            ax.set_ylabel('Item 1', fontsize = 10)
            
            plt.colorbar(im, ax=ax)
            
            plt.subplots_adjust(top = 0.8)
            plt.suptitle(f'SSE, ttype={tt}, {label}', fontsize = 15, y=1)
            plt.show()
            
            
            ### cosSimilarity
            plt.figure(figsize=(15, 5), dpi=100)
            plt.subplot(1,3,1)
            ax = plt.gca()
            im = ax.imshow(cS_11T.mean(axis=1).mean(axis=0), cmap='OrRd', aspect='auto', vmin=0, vmax=1) #, norm=mcolors.SymLogNorm(linthresh=0.1, linscale=0.1, vmin=0, vmax=1.0, base=10)
            #im = ax.imshow(f_plotting.mask_triangle(cS_11T.mean(axis=1).mean(axis=0), ul='u', diag=0), cmap='Reds', aspect='auto', norm=mcolors.SymLogNorm(linthresh=0.1, linscale=0.1, vmin=0, vmax=1.0, base=10)) #, vmax=1
            #im = ax.imshow(np.median(cosTheta_11T.mean(axis=1), axis=0), cmap='jet', aspect='auto', vmin=0, vmax=1)
            for i in range(len(checkpoints)):
                for j in range(len(checkpoints)):
                    if 0.05 < pcS_11T[i,j] <= 0.1:
                        text = ax.text(j, i, '+', ha="center", va="center", fontsize=6)
                    elif 0.01 < pcS_11T[i,j] <= 0.05:
                        text = ax.text(j, i, '*', ha="center", va="center", fontsize=6)
                    elif pcS_11T[i,j] <= 0.01:
                        text = ax.text(j, i, '**', ha="center", va="center", fontsize=6)
                        
            ax.set_title('11', pad = 10)
            ax.set_xticks([n for n in range(len(checkpoints))])
            ax.set_xticklabels(checkpointsLabels, fontsize = 6)
            ax.set_yticks([n for n in range(len(checkpoints))])
            ax.set_yticklabels(checkpointsLabels, fontsize = 6)
            ax.set_xlabel('Item 1', fontsize = 10)
            ax.set_ylabel('Item 1', fontsize = 10)
            
            
            
            plt.subplot(1,3,2)
            ax = plt.gca()
            im = ax.imshow(cS_22T.mean(axis=1).mean(axis=0), cmap='OrRd', aspect='auto', vmin=0, vmax=1) #, norm=mcolors.SymLogNorm(linthresh=0.1, linscale=0.1, vmin=0, vmax=1.0, base=10)
            #im = ax.imshow(f_plotting.mask_triangle(cS_22T.mean(axis=1).mean(axis=0), ul='l', diag=0), cmap='Reds', aspect='auto', norm=mcolors.SymLogNorm(linthresh=0.1, linscale=0.1, vmin=0, vmax=1.0, base=10)) #, vmax=1
            #im = ax.imshow(np.median(cosTheta_22T.mean(axis=1), axis=0), cmap='jet', aspect='auto', vmin=0, vmax=1)
            for i in range(len(checkpoints)):
                for j in range(len(checkpoints)):
                    if 0.05 < pcS_22T[i,j] <= 0.1:
                        text = ax.text(j, i, '+', ha="center", va="center", fontsize=6)
                    elif 0.01 < pcS_22T[i,j] <= 0.05:
                        text = ax.text(j, i, '*', ha="center", va="center", fontsize=6)
                    elif pcS_22T[i,j] <= 0.01:
                        text = ax.text(j, i, '**', ha="center", va="center", fontsize=6)
                        
            ax.set_title('22', pad = 10)
            ax.set_xticks([n for n in range(len(checkpoints))])
            ax.set_xticklabels(checkpointsLabels, fontsize = 6)
            ax.set_yticks([n for n in range(len(checkpoints))])
            ax.set_yticklabels(checkpointsLabels, fontsize = 6)
            ax.set_xlabel('Item 2', fontsize = 10)
            ax.set_ylabel('Item 2', fontsize = 10)
            
            
            
            plt.subplot(1,3,3)
            ax = plt.gca()
            im = ax.imshow(cS_12T.mean(axis=1).mean(axis=0), cmap='OrRd', aspect='auto', vmin=0, vmax=1) #, norm=mcolors.SymLogNorm(linthresh=0.1, linscale=0.1, vmin=0, vmax=1.0, base=10)
            #im = ax.imshow(np.median(cosTheta_12T.mean(axis=1), axis=0), cmap='jet', aspect='auto', vmin=0, vmax=1)
            for i in range(len(checkpoints)):
                for j in range(len(checkpoints)):
                    if 0.05 < pcS_12T[i,j] <= 0.1:
                        text = ax.text(j, i, '+', ha="center", va="center", fontsize=6)
                    elif 0.01 < pcS_12T[i,j] <= 0.05:
                        text = ax.text(j, i, '*', ha="center", va="center", fontsize=6)
                    elif pcS_12T[i,j] <= 0.01:
                        text = ax.text(j, i, '**', ha="center", va="center", fontsize=6)
            ax.set_title('12', pad = 10)
            ax.set_xticks([n for n in range(len(checkpoints))])
            ax.set_xticklabels(checkpointsLabels, fontsize = 6)
            ax.set_yticks([n for n in range(len(checkpoints))])
            ax.set_yticklabels(checkpointsLabels, fontsize = 6)
            ax.set_xlabel('Item 2', fontsize = 10)
            ax.set_ylabel('Item 1', fontsize = 10)
            
            plt.colorbar(im, ax=ax)
            
            plt.subplots_adjust(top = 0.8)
            plt.suptitle(f'correlation, ttype={tt}', fontsize = 15, y=1)
            plt.show()
            
            
            
            ### alignment index
            plt.figure(figsize=(15, 5), dpi=100)
            plt.subplot(1,3,1)
            ax = plt.gca()
            im = ax.imshow(ai_11T.mean(axis=1).mean(axis=0), cmap='OrRd', aspect='auto', vmin=0, vmax=1) #
            #im = ax.imshow(f_plotting.mask_triangle(ai_11T.mean(axis=1).mean(axis=0), ul='u', diag=0), cmap='Reds', aspect='auto', vmin=0, vmax=0.3) #
            #im = ax.imshow(np.median(cosTheta_11T.mean(axis=1), axis=0), cmap='jet', aspect='auto', vmin=0, vmax=1)
            for i in range(len(checkpoints)):
                for j in range(len(checkpoints)):
                    if 0.05 < pai_11T[i,j] <= 0.1:
                        text = ax.text(j, i, '+', ha="center", va="center", fontsize=6)
                    elif 0.01 < pai_11T[i,j] <= 0.05:
                        text = ax.text(j, i, '*', ha="center", va="center", fontsize=6)
                    elif pai_11T[i,j] <= 0.01:
                        text = ax.text(j, i, '**', ha="center", va="center", fontsize=6)
                        
            ax.set_title('11', pad = 10)
            ax.set_xticks([n for n in range(len(checkpoints))])
            ax.set_xticklabels(checkpointsLabels, fontsize = 6)
            ax.set_yticks([n for n in range(len(checkpoints))])
            ax.set_yticklabels(checkpointsLabels, fontsize = 6)
            ax.set_xlabel('Item 1', fontsize = 10)
            ax.set_ylabel('Item 1', fontsize = 10)
            
            
            
            plt.subplot(1,3,2)
            ax = plt.gca()
            im = ax.imshow(ai_22T.mean(axis=1).mean(axis=0), cmap='OrRd', aspect='auto', vmin=0, vmax=1) #
            #im = ax.imshow(f_plotting.mask_triangle(ai_22T.mean(axis=1).mean(axis=0), ul='l', diag=0), cmap='Reds', aspect='auto', vmin=0, vmax=0.3) #
            #im = ax.imshow(np.median(cosTheta_22T.mean(axis=1), axis=0), cmap='jet', aspect='auto', vmin=0, vmax=1)
            for i in range(len(checkpoints)):
                for j in range(len(checkpoints)):
                    if 0.05 < pai_22T[i,j] <= 0.1:
                        text = ax.text(j, i, '+', ha="center", va="center", fontsize=6)
                    elif 0.01 < pai_22T[i,j] <= 0.05:
                        text = ax.text(j, i, '*', ha="center", va="center", fontsize=6)
                    elif pai_22T[i,j] <= 0.01:
                        text = ax.text(j, i, '**', ha="center", va="center", fontsize=6)
                        
            ax.set_title('22', pad = 10)
            ax.set_xticks([n for n in range(len(checkpoints))])
            ax.set_xticklabels(checkpointsLabels, fontsize = 6)
            ax.set_yticks([n for n in range(len(checkpoints))])
            ax.set_yticklabels(checkpointsLabels, fontsize = 6)
            ax.set_xlabel('Item 2', fontsize = 10)
            ax.set_ylabel('Item 2', fontsize = 10)
            
            
            
            plt.subplot(1,3,3)
            ax = plt.gca()
            im = ax.imshow(ai_12T.mean(axis=1).mean(axis=0), cmap='OrRd', aspect='auto', vmin=0, vmax=1) #, vmax=1
            #im = ax.imshow(np.median(cosTheta_12T.mean(axis=1), axis=0), cmap='jet', aspect='auto', vmin=0, vmax=1)
            for i in range(len(checkpoints)):
                for j in range(len(checkpoints)):
                    if 0.05 < pai_12T[i,j] <= 0.1:
                        text = ax.text(j, i, '+', ha="center", va="center", fontsize=6)
                    elif 0.01 < pai_12T[i,j] <= 0.05:
                        text = ax.text(j, i, '*', ha="center", va="center", fontsize=6)
                    elif pai_12T[i,j] <= 0.01:
                        text = ax.text(j, i, '**', ha="center", va="center", fontsize=6)
            ax.set_title('12', pad = 10)
            ax.set_xticks([n for n in range(len(checkpoints))])
            ax.set_xticklabels(checkpointsLabels, fontsize = 6)
            ax.set_yticks([n for n in range(len(checkpoints))])
            ax.set_yticklabels(checkpointsLabels, fontsize = 6)
            ax.set_xlabel('Item 2', fontsize = 10)
            ax.set_ylabel('Item 1', fontsize = 10)
            
            plt.colorbar(im, ax=ax)
            
            plt.subplots_adjust(top = 0.8)
            plt.suptitle(f'alignment index, ttype={tt}', fontsize = 15, y=1)
            plt.show()
        

    cosThetas = (cosTheta_11, cosTheta_12, cosTheta_22)
    cosPsis = (cosPsi_11, cosPsi_12, cosPsi_22)
    sses = (sse_11, sse_12, sse_22)
    cosSimis = (cosSimi_11, cosSimi_22, cosSimi_12)
    ais = (ai_11, ai_22, ai_12)
    
    return cosThetas, cosPsis, sses, cosSimis, ais
# In[] compare choice/non-choice
def plot_choiceItemSpace(geoms_valid, geoms_shuff, checkpoints, checkpointsLabels, locs = (0,1,2,3), ttypes = (1,2), nIters = 20, nPerms = 50, nBoots = 1, infoMethod = 'omega2', label = '', pdummy = True, toPlot = True):
    
    vecs, projs, projsAll, trialInfos = geoms_valid
    vecs_shuff, projs_shuff, projsAll_shuff, trialInfos_shuff = geoms_shuff
    
    
    cosTheta_choice, cosTheta_nonchoice = {},{}
    cosTheta_choice_shuff_all, cosTheta_nonchoice_shuff_all = {}, {}
    
    cosPsi_choice, cosPsi_nonchoice = {},{}
    cosPsi_choice_shuff_all, cosPsi_nonchoice_shuff_all = {}, {}
    
    sse_choice, sse_nonchoice = {},{}
    sse_choice_shuff_all, sse_nonchoice_shuff_all = {}, {}
    
    cosSimi_choice, cosSimi_nonchoice = {},{}
    cosSimi_choice_shuff_all, cosSimi_nonchoice_shuff_all = {}, {}

    ai_choice, ai_nonchoice = {},{}
    ai_choice_shuff_all, ai_nonchoice_shuff_all = {}, {}
    
            
    cosTheta_choiceT, cosTheta_nonchoiceT = np.zeros((nIters, nBoots, len(checkpoints),)), np.zeros((nIters, nBoots, len(checkpoints),))
    cosPsi_choiceT, cosPsi_nonchoiceT = np.zeros((nIters, nBoots, len(checkpoints),)), np.zeros((nIters, nBoots, len(checkpoints),))
    sse_choiceT, sse_nonchoiceT = np.zeros((nIters, nBoots, len(checkpoints),)), np.zeros((nIters, nBoots, len(checkpoints),))
    
    cosSimi_choiceT, cosSimi_nonchoiceT = np.zeros((nIters, nBoots, len(checkpoints),)), np.zeros((nIters, nBoots, len(checkpoints),))
    ai_choiceT, ai_nonchoiceT = np.zeros((nIters, nBoots, len(checkpoints),)), np.zeros((nIters, nBoots, len(checkpoints),))
    
    for n in range(nIters):
        for nbt in range(nBoots):
            for nc,cp in enumerate(checkpoints):
                cT_C, _, cP_C, _, s_C = f_subspace.angle_alignment_coplanar(vecs[cp][1][2][n][nbt], projs[cp][1][2][n][nbt], vecs[cp][2][1][n][nbt], projs[cp][2][1][n][nbt], sequence=(0,1,3,2))
                cT_NC, _, cP_NC, _, s_NC = f_subspace.angle_alignment_coplanar(vecs[cp][1][1][n][nbt], projs[cp][1][1][n][nbt], vecs[cp][2][2][n][nbt], projs[cp][2][2][n][nbt], sequence=(0,1,3,2))
                
                cS_C = f_subspace.config_correlation(vecs[cp][1][2][n][nbt], projs[cp][1][2][n][nbt], vecs[cp][2][1][n][nbt], projs[cp][2][1][n][nbt])
                ai_C = f_subspace.get_simple_AI(projs[cp][1][2][n][nbt], projs[cp][2][1][n][nbt], max_dim=2)
                
                cS_NC = f_subspace.config_correlation(vecs[cp][1][1][n][nbt], projs[cp][1][1][n][nbt], vecs[cp][2][2][n][nbt], projs[cp][2][2][n][nbt])
                ai_NC = f_subspace.get_simple_AI(projs[cp][1][1][n][nbt], projs[cp][2][2][n][nbt], max_dim=2)
                
                cosTheta_choiceT[n,nbt,nc], cosPsi_choiceT[n,nbt,nc], sse_choiceT[n,nbt,nc] = cT_C, cP_C, s_C
                cosTheta_nonchoiceT[n,nbt,nc], cosPsi_nonchoiceT[n,nbt,nc], sse_nonchoiceT[n,nbt,nc] = cT_NC, cP_NC, s_NC
                
                cosSimi_choiceT[n,nbt,nc], ai_choiceT[n,nbt,nc] = cS_C, ai_C
                cosSimi_nonchoiceT[n,nbt,nc], ai_nonchoiceT[n,nbt,nc] = cS_NC, ai_NC
    
    cosTheta_choice, cosTheta_nonchoice = cosTheta_choiceT, cosTheta_nonchoiceT
    cosPsi_choice, cosPsi_nonchoice = cosPsi_choiceT, cosPsi_nonchoiceT
    sse_choice, sse_nonchoice = sse_choiceT, sse_nonchoiceT
    cosSimi_choice, cosSimi_nonchoice = cosSimi_choiceT, cosSimi_nonchoiceT
    ai_choice, ai_nonchoice = ai_choiceT, ai_nonchoiceT
    
    ###
    pcosTheta_choice, pcosTheta_nonchoice = np.ones((len(checkpoints))), np.ones((len(checkpoints)))
    pcosPsi_choice, pcosPsi_nonchoice = np.ones((len(checkpoints))), np.ones((len(checkpoints)))
    psse_choice, psse_nonchoice = np.ones((len(checkpoints))), np.ones((len(checkpoints)))
    pcosSimi_choice, pcosSimi_nonchoice = np.ones((len(checkpoints))), np.ones((len(checkpoints)))
    pai_choice, pai_nonchoice = np.ones((len(checkpoints))), np.ones((len(checkpoints)))
    
    if pdummy == False:
        
        ### shuff 
        cosTheta_choice_shuffT, cosTheta_nonchoice_shuffT = np.zeros((nIters, nBoots*nPerms, len(checkpoints),)), np.zeros((nIters, nBoots*nPerms, len(checkpoints),))
        cosPsi_choice_shuffT, cosPsi_nonchoice_shuffT = np.zeros((nIters, nBoots*nPerms, len(checkpoints),)), np.zeros((nIters, nBoots*nPerms, len(checkpoints),))
        sse_choice_shuffT, sse_nonchoice_shuffT = np.zeros((nIters, nBoots*nPerms, len(checkpoints),)), np.zeros((nIters, nBoots*nPerms, len(checkpoints),))
        cosSimi_choice_shuffT, cosSimi_nonchoice_shuffT = np.zeros((nIters, nBoots*nPerms, len(checkpoints),)), np.zeros((nIters, nBoots*nPerms, len(checkpoints),))
        ai_choice_shuffT, ai_nonchoice_shuffT = np.zeros((nIters, nBoots*nPerms, len(checkpoints),)), np.zeros((nIters, nBoots*nPerms, len(checkpoints),))
        
        for n in range(nIters):
            for nbt in range(nBoots*nPerms):
                for nc, cp in enumerate(checkpoints):
                    cT_C, _, cP_C, _, s_C = f_subspace.angle_alignment_coplanar(vecs_shuff[cp][1][2][n][nbt], projs_shuff[cp][1][2][n][nbt], vecs_shuff[cp][2][1][n][nbt], projs_shuff[cp][2][1][n][nbt], sequence=(0,1,3,2))
                    cT_NC, _, cP_NC, _, s_NC = f_subspace.angle_alignment_coplanar(vecs_shuff[cp][1][1][n][nbt], projs_shuff[cp][1][1][n][nbt], vecs_shuff[cp][2][2][n][nbt], projs_shuff[cp][2][2][n][nbt], sequence=(0,1,3,2))
                    
                    cS_C = f_subspace.config_correlation(vecs_shuff[cp][1][2][n][nbt], projs_shuff[cp][1][2][n][nbt], vecs_shuff[cp][2][1][n][nbt], projs_shuff[cp][2][1][n][nbt])
                    ai_C = f_subspace.get_simple_AI(projs_shuff[cp][1][2][n][nbt], projs_shuff[cp][2][1][n][nbt], max_dim=2)
                    cS_NC = f_subspace.config_correlation(vecs_shuff[cp][1][1][n][nbt], projs_shuff[cp][1][1][n][nbt], vecs_shuff[cp][2][2][n][nbt], projs_shuff[cp][2][2][n][nbt])
                    ai_NC = f_subspace.get_simple_AI(projs_shuff[cp][1][1][n][nbt], projs_shuff[cp][2][2][n][nbt], max_dim=2)
                    
                    cosTheta_choice_shuffT[n,nbt,nc], cosPsi_choice_shuffT[n,nbt,nc], sse_choice_shuffT[n,nbt,nc] = cT_C, cP_C, s_C
                    cosTheta_nonchoice_shuffT[n,nbt,nc], cosPsi_nonchoice_shuffT[n,nbt,nc], sse_nonchoice_shuffT[n,nbt,nc] = cT_NC, cP_NC, s_NC
                    cosSimi_choice_shuffT[n,nbt,nc], ai_choice_shuffT[n,nbt,nc] = cS_C, ai_C
                    cosSimi_nonchoice_shuffT[n,nbt,nc], ai_nonchoice_shuffT[n,nbt,nc] = cS_NC, ai_NC
        
        cosTheta_choice_shuff_all, cosTheta_nonchoice_shuff_all = np.concatenate([cosTheta_choice_shuffT[:,nbt,:] for nbt in range(nBoots*nPerms)]), np.concatenate([cosTheta_nonchoice_shuffT[:,nbt,:] for nbt in range(nBoots*nPerms)])
        cosPsi_choice_shuff_all, cosPsi_nonchoice_shuff_all = np.concatenate([cosPsi_choice_shuffT[:,nbt,:] for nbt in range(nBoots*nPerms)]), np.concatenate([cosPsi_nonchoice_shuffT[:,nbt,:] for nbt in range(nBoots*nPerms)])
        sse_choice_shuff_all, sse_nonchoice_shuff_all = np.concatenate([sse_choice_shuffT[:,nbt,:] for nbt in range(nBoots*nPerms)]), np.concatenate([sse_nonchoice_shuffT[:,nbt,:] for nbt in range(nBoots*nPerms)])
        cosSimi_choice_shuff_all, cosSimi_nonchoice_shuff_all = np.concatenate([cosSimi_choice_shuffT[:,nbt,:] for nbt in range(nBoots*nPerms)]), np.concatenate([cosSimi_nonchoice_shuffT[:,nbt,:] for nbt in range(nBoots*nPerms)])
        ai_choice_shuff_all, ai_nonchoice_shuff_all = np.concatenate([ai_choice_shuffT[:,nbt,:] for nbt in range(nBoots*nPerms)]), np.concatenate([ai_nonchoice_shuffT[:,nbt,:] for nbt in range(nBoots*nPerms)])
        
        
        
        for i in range(len(checkpoints)):
            
            # test distribution
            cT_C, cT_NC = cosTheta_choice.mean(axis=1)[:,i].round(5), cosTheta_nonchoice.mean(axis=1)[:,i].round(5)
            cP_C, cP_NC = cosPsi_choice.mean(axis=1)[:,i].round(5), cosPsi_nonchoice.mean(axis=1)[:,i].round(5)
            s_C, s_NC = sse_choice.mean(axis=1)[:,i].round(5), sse_nonchoice.mean(axis=1)[:,i].round(5)
            cS_C, cS_NC = cosSimi_choice.mean(axis=1)[:,i].round(5), cosSimi_nonchoice.mean(axis=1)[:,i].round(5)
            ai_C, ai_NC = ai_choice.mean(axis=1)[:,i].round(5), ai_nonchoice.mean(axis=1)[:,i].round(5)
            
            # shuff distribution
            cT_C_shuff, cT_NC_shuff = cosTheta_choice_shuff_all[:,i].round(5), cosTheta_nonchoice_shuff_all[:,i].round(5)
            cP_C_shuff, cP_NC_shuff = cosPsi_choice_shuff_all[:,i].round(5), cosPsi_nonchoice_shuff_all[:,i].round(5)
            s_C_shuff, s_NC_shuff = sse_choice_shuff_all[:,i].round(5), sse_nonchoice_shuff_all[:,i].round(5)
            cS_C_shuff, cS_NC_shuff = cosSimi_choice_shuff_all[:,i].round(5), cosSimi_nonchoice_shuff_all[:,i].round(5)
            ai_C_shuff, ai_NC_shuff = ai_choice_shuff_all[:,i].round(5), ai_nonchoice_shuff_all[:,i].round(5)
            
            # compare distributions and calculate p values
            pcosTheta_choice[i], pcosTheta_nonchoice[i] = stats.kstest(cT_C, cT_C_shuff)[-1], stats.kstest(cT_NC, cT_NC_shuff)[-1]
            pcosPsi_choice[i], pcosPsi_nonchoice[i] = stats.kstest(cP_C, cP_C_shuff)[-1], stats.kstest(cP_NC, cP_NC_shuff)[-1]
            #pcosPsi_choice[i], pcosPsi_nonchoice[i] = stats.kstest(np.abs(cP_C), np.abs(cP_C_shuff))[-1], stats.kstest(np.abs(cP_NC), np.abs(cP_NC_shuff))[-1]
            psse_choice[i], psse_nonchoice[i] = stats.kstest(s_C, s_C_shuff)[-1], stats.kstest(s_NC, s_NC_shuff)[-1]
            pcosSimi_choice[i], pcosSimi_nonchoice[i] = stats.kstest(cS_C, cS_C_shuff)[-1], stats.kstest(cS_NC, cS_NC_shuff)[-1]
            pai_choice[i], pai_nonchoice[i] = stats.kstest(ai_C, ai_C_shuff)[-1], stats.kstest(ai_NC, ai_NC_shuff)[-1]
            
    if toPlot:
        
        angleCheckPoints = np.linspace(0,np.pi,7).round(5)
        
        ### cosTheta
        plt.figure(figsize=(8, 3), dpi=100)
        plt.subplot(1,2,1)
        ax = plt.gca()
        ax.errorbar(np.arange(0, len(checkpoints), 1), cosTheta_choice.mean(axis=1).mean(axis=0), yerr = cosTheta_choice.mean(axis=1).std(axis=0), marker = 'o')
        ax.plot(np.arange(0, len(checkpoints), 1), pcosTheta_choice, alpha = 0.3, linestyle = '-')
        
        trans = ax.get_xaxis_transform()
        for nc, cp in enumerate(checkpoints):
            if 0.05 < pcosTheta_choice[nc] <= 0.1:
                ax.annotate('+', xy=(nc, 0.05), xycoords = trans, annotation_clip = False, ha = 'center', va = 'center')
            elif 0.01 < pcosTheta_choice[nc] <= 0.05:
                ax.annotate('*', xy=(nc, 0.05), xycoords = trans, annotation_clip = False, ha = 'center', va = 'center')
            elif pcosTheta_choice[nc] <= 0.01:
                ax.annotate('**', xy=(nc, 0.05), xycoords = trans, annotation_clip = False, ha = 'center', va = 'center')
                
        ax.set_title('Choice', pad = 10)
        ax.set_xticks([n for n in range(len(checkpoints))])
        ax.set_xticklabels(checkpointsLabels, fontsize = 6)
        ax.set_ylim((-1,1))
        ax.set_yticks(np.flip(np.cos(angleCheckPoints).round(5)), labels = np.flip(np.degrees(angleCheckPoints).round(1).astype(int)))
        ax.set_ylabel('cos(θ)',fontsize=15,rotation = 90)
        
        plt.subplot(1,2,2)
        ax = plt.gca()
        ax.errorbar(np.arange(0, len(checkpoints), 1), cosTheta_nonchoice.mean(axis=1).mean(axis=0), yerr = cosTheta_nonchoice.mean(axis=1).std(axis=0), marker = 'o')
        ax.plot(np.arange(0, len(checkpoints), 1), pcosTheta_nonchoice, alpha = 0.3, linestyle = '-')
        
        trans = ax.get_xaxis_transform()
        for nc, cp in enumerate(checkpoints):
            if 0.05 < pcosTheta_nonchoice[nc] <= 0.1:
                ax.annotate('+', xy=(nc, 0.05), xycoords = trans, annotation_clip = False, ha = 'center', va = 'center')
            elif 0.01 < pcosTheta_nonchoice[nc] <= 0.05:
                ax.annotate('*', xy=(nc, 0.05), xycoords = trans, annotation_clip = False, ha = 'center', va = 'center')
            elif pcosTheta_nonchoice[nc] <= 0.01:
                ax.annotate('**', xy=(nc, 0.05), xycoords = trans, annotation_clip = False, ha = 'center', va = 'center')
                
        ax.set_title('Non-choice', pad = 10)
        ax.set_xticks([n for n in range(len(checkpoints))])
        ax.set_xticklabels(checkpointsLabels, fontsize = 6)
        ax.set_ylim((-1,1))
        ax.set_yticks(np.flip(np.cos(angleCheckPoints).round(5)), labels = np.flip(np.degrees(angleCheckPoints).round(1).astype(int)))
        ax.set_ylabel('cos(θ)',fontsize=15,rotation = 90)
        
        plt.subplots_adjust(top = 0.8)
        plt.suptitle(f'Principal Angle (θ), {label}', fontsize = 15, y=1)
        plt.show()    
        
        
        ### cosPsi
        plt.figure(figsize=(8, 3), dpi=100)
        plt.subplot(1,2,1)
        ax = plt.gca()
        ax.errorbar(np.arange(0, len(checkpoints), 1), cosPsi_choice.mean(axis=1).mean(axis=0), yerr = cosPsi_choice.mean(axis=1).std(axis=0), marker = 'o')
        #ax.errorbar(np.arange(0, len(checkpoints), 1), np.abs(cosPsi_choice[region]).mean(axis=1).mean(axis=0), yerr = np.abs(cosPsi_choice[region]).mean(axis=1).std(axis=0), marker = 'o')
        ax.plot(np.arange(0, len(checkpoints), 1), pcosPsi_choice, alpha = 0.3, linestyle = '-')
        
        trans = ax.get_xaxis_transform()
        for nc, cp in enumerate(checkpoints):
            if 0.05 < pcosPsi_choice[nc] <= 0.1:
                ax.annotate('+', xy=(nc, 0.05), xycoords = trans, annotation_clip = False, ha = 'center', va = 'center')
            elif 0.01 < pcosPsi_choice[nc] <= 0.05:
                ax.annotate('*', xy=(nc, 0.05), xycoords = trans, annotation_clip = False, ha = 'center', va = 'center')
            elif pcosPsi_choice[nc] <= 0.01:
                ax.annotate('**', xy=(nc, 0.05), xycoords = trans, annotation_clip = False, ha = 'center', va = 'center')
                
        ax.set_title('Choice', pad = 10)
        ax.set_xticks([n for n in range(len(checkpoints))])
        ax.set_xticklabels(checkpointsLabels, fontsize = 6)
        ax.set_ylim((-1,1))
        ax.set_yticks(np.flip(np.cos(angleCheckPoints).round(5)), labels = np.flip(np.degrees(angleCheckPoints).round(1).astype(int)))
        ax.set_ylabel('cos(Ψ)',fontsize=15,rotation = 90)
        
        plt.subplot(1,2,2)
        ax = plt.gca()
        #ax.errorbar(np.arange(0, len(checkpoints), 1), cosPsi_nonchoice.mean(axis=1).mean(axis=0), yerr = cosPsi_nonchoice.mean(axis=1).std(axis=0), marker = 'o')
        ax.errorbar(np.arange(0, len(checkpoints), 1), np.abs(cosPsi_nonchoice).mean(axis=1).mean(axis=0), yerr = np.abs(cosPsi_nonchoice).mean(axis=1).std(axis=0), marker = 'o')
        ax.plot(np.arange(0, len(checkpoints), 1), pcosPsi_nonchoice, alpha = 0.3, linestyle = '-')
        
        trans = ax.get_xaxis_transform()
        for nc, cp in enumerate(checkpoints):
            if 0.05 < pcosPsi_nonchoice[nc] <= 0.1:
                ax.annotate('+', xy=(nc, 0.05), xycoords = trans, annotation_clip = False, ha = 'center', va = 'center')
            elif 0.01 < pcosPsi_nonchoice[nc] <= 0.05:
                ax.annotate('*', xy=(nc, 0.05), xycoords = trans, annotation_clip = False, ha = 'center', va = 'center')
            elif pcosPsi_nonchoice[nc] <= 0.01:
                ax.annotate('**', xy=(nc, 0.05), xycoords = trans, annotation_clip = False, ha = 'center', va = 'center')
                
        ax.set_title('Non-choice', pad = 10)
        ax.set_xticks([n for n in range(len(checkpoints))])
        ax.set_xticklabels(checkpointsLabels, fontsize = 6)
        ax.set_ylim((-1,1))
        ax.set_yticks(np.flip(np.cos(angleCheckPoints).round(5)), labels = np.flip(np.degrees(angleCheckPoints).round(1).astype(int)))
        ax.set_ylabel('cos(Ψ)',fontsize=15,rotation = 90)
        
        plt.subplots_adjust(top = 0.8)
        plt.suptitle(f'Representational Alignment (Ψ), {label}', fontsize = 15, y=1)
        #plt.suptitle(f'abs(cosPsi), {region}, choice/nonchoice', fontsize = 15, y=1)
        plt.show()    
        
        
        ### sse
        plt.figure(figsize=(10, 3), dpi=100)
        plt.subplot(1,2,1)
        ax = plt.gca()
        ax.errorbar(np.arange(0, len(checkpoints), 1), sse_choice.mean(axis=1).mean(axis=0), yerr = sse_choice.mean(axis=1).std(axis=0), marker = 'o')
        ax.plot(np.arange(0, len(checkpoints), 1), psse_choice, alpha = 0.3, linestyle = '-')
        
        trans = ax.get_xaxis_transform()
        for nc, cp in enumerate(checkpoints):
            if 0.05 < psse_choice[nc] <= 0.1:
                ax.annotate('+', xy=(nc, 0.05), xycoords = trans, annotation_clip = False, ha = 'center', va = 'center')
            elif 0.01 < psse_choice[nc] <= 0.05:
                ax.annotate('*', xy=(nc, 0.05), xycoords = trans, annotation_clip = False, ha = 'center', va = 'center')
            elif psse_choice[nc] <= 0.01:
                ax.annotate('**', xy=(nc, 0.05), xycoords = trans, annotation_clip = False, ha = 'center', va = 'center')
                
        ax.set_title('Choice', pad = 10)
        ax.set_xticks([n for n in range(len(checkpoints))])
        ax.set_xticklabels(checkpointsLabels, fontsize = 6)
        #ax.set_ylim((-1,1))
        
        
        plt.subplot(1,2,2)
        ax = plt.gca()
        ax.errorbar(np.arange(0, len(checkpoints), 1), sse_nonchoice.mean(axis=1).mean(axis=0), yerr = sse_nonchoice.mean(axis=1).std(axis=0), marker = 'o')
        ax.plot(np.arange(0, len(checkpoints), 1), psse_nonchoice, alpha = 0.3, linestyle = '-')
        
        trans = ax.get_xaxis_transform()
        for nc, cp in enumerate(checkpoints):
            if 0.05 < psse_nonchoice[nc] <= 0.1:
                ax.annotate('+', xy=(nc, 0.05), xycoords = trans, annotation_clip = False, ha = 'center', va = 'center')
            elif 0.01 < psse_nonchoice[nc] <= 0.05:
                ax.annotate('*', xy=(nc, 0.05), xycoords = trans, annotation_clip = False, ha = 'center', va = 'center')
            elif psse_nonchoice[nc] <= 0.01:
                ax.annotate('**', xy=(nc, 0.05), xycoords = trans, annotation_clip = False, ha = 'center', va = 'center')
                
        ax.set_title('Non-choice', pad = 10)
        ax.set_xticks([n for n in range(len(checkpoints))])
        ax.set_xticklabels(checkpointsLabels, fontsize = 6)
        #ax.set_ylim((-1,1))
        
        plt.subplots_adjust(top = 0.8)
        plt.suptitle(f'SSE, choice/nonchoice, {label}', fontsize = 15, y=1)
        plt.show()    
        
        ### cosSimilarity
        plt.figure(figsize=(10, 3), dpi=100)
        plt.subplot(1,2,1)
        ax = plt.gca()
        ax.errorbar(np.arange(0, len(checkpoints), 1), cosSimi_choice.mean(axis=1).mean(axis=0), yerr = cosSimi_choice.mean(axis=1).std(axis=0), marker = 'o')
        #ax.errorbar(np.arange(0, len(checkpoints), 1), np.abs(cosPsi_choice[region]).mean(axis=1).mean(axis=0), yerr = np.abs(cosPsi_choice[region]).mean(axis=1).std(axis=0), marker = 'o')
        ax.plot(np.arange(0, len(checkpoints), 1), pcosSimi_choice, alpha = 0.3, linestyle = '-')
        
        trans = ax.get_xaxis_transform()
        for nc, cp in enumerate(checkpoints):
            if 0.05 < pcosSimi_choice[nc] <= 0.1:
                ax.annotate('+', xy=(nc, 0.05), xycoords = trans, annotation_clip = False, ha = 'center', va = 'center')
            elif 0.01 < pcosSimi_choice[nc] <= 0.05:
                ax.annotate('*', xy=(nc, 0.05), xycoords = trans, annotation_clip = False, ha = 'center', va = 'center')
            elif pcosSimi_choice[nc] <= 0.01:
                ax.annotate('**', xy=(nc, 0.05), xycoords = trans, annotation_clip = False, ha = 'center', va = 'center')
                
        ax.set_title('choice', pad = 10)
        ax.set_xticks([n for n in range(len(checkpoints))])
        ax.set_xticklabels(checkpointsLabels, fontsize = 6)
        ax.set_ylim((0,1))
        #ax.set_ylim((0,1))
        
        
        plt.subplot(1,2,2)
        ax = plt.gca()
        #ax.errorbar(np.arange(0, len(checkpoints), 1), cosPsi_nonchoice[region].mean(axis=1).mean(axis=0), yerr = cosPsi_nonchoice[region].mean(axis=1).std(axis=0), marker = 'o')
        ax.errorbar(np.arange(0, len(checkpoints), 1), np.abs(cosSimi_nonchoice).mean(axis=1).mean(axis=0), yerr = np.abs(cosSimi_nonchoice).mean(axis=1).std(axis=0), marker = 'o')
        ax.plot(np.arange(0, len(checkpoints), 1), pcosSimi_nonchoice, alpha = 0.3, linestyle = '-')
        
        trans = ax.get_xaxis_transform()
        for nc, cp in enumerate(checkpoints):
            if 0.05 < pcosSimi_nonchoice[nc] <= 0.1:
                ax.annotate('+', xy=(nc, 0.05), xycoords = trans, annotation_clip = False, ha = 'center', va = 'center')
            elif 0.01 < pcosSimi_nonchoice[nc] <= 0.05:
                ax.annotate('*', xy=(nc, 0.05), xycoords = trans, annotation_clip = False, ha = 'center', va = 'center')
            elif pcosSimi_nonchoice[nc] <= 0.01:
                ax.annotate('**', xy=(nc, 0.05), xycoords = trans, annotation_clip = False, ha = 'center', va = 'center')
                
        ax.set_title('non-choice', pad = 10)
        ax.set_xticks([n for n in range(len(checkpoints))])
        ax.set_xticklabels(checkpointsLabels, fontsize = 6)
        ax.set_ylim((0,1))
        #ax.set_ylim((0,1))
        
        plt.subplots_adjust(top = 0.8)
        plt.suptitle(f'correlation, choice/nonchoice', fontsize = 15, y=1)
        #plt.suptitle(f'abs(cosPsi), {region}, choice/nonchoice', fontsize = 15, y=1)
        plt.show()    
        
        
        ### alignment index
        plt.figure(figsize=(10, 3), dpi=100)
        plt.subplot(1,2,1)
        ax = plt.gca()
        ax.errorbar(np.arange(0, len(checkpoints), 1), ai_choice.mean(axis=1).mean(axis=0), yerr = ai_choice.mean(axis=1).std(axis=0), marker = 'o')
        #ax.errorbar(np.arange(0, len(checkpoints), 1), np.abs(cosPsi_choice[region]).mean(axis=1).mean(axis=0), yerr = np.abs(cosPsi_choice[region]).mean(axis=1).std(axis=0), marker = 'o')
        ax.plot(np.arange(0, len(checkpoints), 1), pai_choice, alpha = 0.3, linestyle = '-')
        
        trans = ax.get_xaxis_transform()
        for nc, cp in enumerate(checkpoints):
            if 0.05 < pai_choice[nc] <= 0.1:
                ax.annotate('+', xy=(nc, 0.05), xycoords = trans, annotation_clip = False, ha = 'center', va = 'center')
            elif 0.01 < pai_choice[nc] <= 0.05:
                ax.annotate('*', xy=(nc, 0.05), xycoords = trans, annotation_clip = False, ha = 'center', va = 'center')
            elif pai_choice[nc] <= 0.01:
                ax.annotate('**', xy=(nc, 0.05), xycoords = trans, annotation_clip = False, ha = 'center', va = 'center')
                
        ax.set_title('choice', pad = 10)
        ax.set_xticks([n for n in range(len(checkpoints))])
        ax.set_xticklabels(checkpointsLabels, fontsize = 6)
        ax.set_ylim((0,1))
        #ax.set_ylim((0,1))
        
        
        plt.subplot(1,2,2)
        ax = plt.gca()
        #ax.errorbar(np.arange(0, len(checkpoints), 1), cosPsi_nonchoice[region].mean(axis=1).mean(axis=0), yerr = cosPsi_nonchoice[region].mean(axis=1).std(axis=0), marker = 'o')
        ax.errorbar(np.arange(0, len(checkpoints), 1), np.abs(ai_nonchoice).mean(axis=1).mean(axis=0), yerr = np.abs(ai_nonchoice).mean(axis=1).std(axis=0), marker = 'o')
        ax.plot(np.arange(0, len(checkpoints), 1), pai_nonchoice, alpha = 0.3, linestyle = '-')
        
        trans = ax.get_xaxis_transform()
        for nc, cp in enumerate(checkpoints):
            if 0.05 < pai_nonchoice[nc] <= 0.1:
                ax.annotate('+', xy=(nc, 0.05), xycoords = trans, annotation_clip = False, ha = 'center', va = 'center')
            elif 0.01 < pai_nonchoice[nc] <= 0.05:
                ax.annotate('*', xy=(nc, 0.05), xycoords = trans, annotation_clip = False, ha = 'center', va = 'center')
            elif pai_nonchoice[nc] <= 0.01:
                ax.annotate('**', xy=(nc, 0.05), xycoords = trans, annotation_clip = False, ha = 'center', va = 'center')
                
        ax.set_title('non-choice', pad = 10)
        ax.set_xticks([n for n in range(len(checkpoints))])
        ax.set_xticklabels(checkpointsLabels, fontsize = 6)
        ax.set_ylim((0,1))
        #ax.set_ylim((0,1))
        
        plt.subplots_adjust(top = 0.8)
        plt.suptitle(f'alignment index, choice/nonchoice', fontsize = 15, y=1)
        #plt.suptitle(f'abs(cosPsi), {region}, choice/nonchoice', fontsize = 15, y=1)
        plt.show()    
    
    
    
    cosThetas = (cosTheta_choice, cosTheta_nonchoice)
    cosPsis = (cosPsi_choice, cosPsi_nonchoice)
    sses = (sse_choice, sse_nonchoice)
    cosSimis = (cosSimi_choice, cosSimi_nonchoice)
    ais = (ai_choice, ai_nonchoice)
    
    return cosThetas, cosPsis, sses, cosSimis, ais


# In[] condition mean trajectory 3d
def plot_trajectory(geoms_valid, geoms_shuff, checkpoints, mainCheckpoints = (1150, 2300, 2800), locs = (0,1,2,3), ttypes = (1,2), nIters = 20, nPerms = 50, nBoots = 1, infoMethod = 'omega2', label = '', toPlot = True):
    
    vecs, projs, projsAll, trialInfos = geoms_valid
    vecs_shuff, projs_shuff, projsAll_shuff, trialInfos_shuff = geoms_shuff
    
    cseq = mpl.color_sequences['tab20c']
    cmap1 = mpl.colormaps.get_cmap('Greens_r')
    cmap2 = mpl.colormaps.get_cmap('Reds_r')
    cmap3 = mpl.colormaps.get_cmap('Blues_r')
    c1s = [cmap1(i) for i in np.linspace(0,1,3)]
    c2s = [cmap2(i) for i in np.linspace(0,1,3)]
    c3s = [cmap3(i) for i in np.linspace(0,1,3)]
    
    shapes = ('o','s','*','^')
    
    
    # just apply to retarget trials
    for tt in ttypes: #(1,)
        # example Iterations
        for n in (0,10,15):#,30,40,50
            projs1T, projs2T = np.zeros((len(checkpoints), len(locs), 3)), np.zeros((len(checkpoints), len(locs), 3))
            vecs1T, vecs2T = np.zeros((len(checkpoints), 2, 3)), np.zeros((len(checkpoints), 2, 3))
            for ncp, cp in enumerate(checkpoints):
                projs1T[ncp,:,:] = np.concatenate(projs[cp][tt][1][n])
                projs2T[ncp,:,:] = np.concatenate(projs[cp][tt][2][n]) if tt == 1 else np.concatenate(projs[cp][tt][1][n])
                vecs1T[ncp,:,:] = np.concatenate(vecs[cp][tt][1][n])
                vecs2T[ncp,:,:] = np.concatenate(vecs[cp][tt][2][n]) if tt == 1 else np.concatenate(vecs[cp][tt][1][n])
            
            #X1, Y1, Z1 = projs1T[:,:,0], projs1T[:,:,1], projs1T[:,:,2]
            #X2, Y2, Z2 = projs2T[:,:,0], projs2T[:,:,1], projs2T[:,:,2]
            
            projs1T_C, projs2T_C = projs1T.mean(axis=1), projs2T.mean(axis=1)
            
            vecs3T = vecs2T if tt == 1 else vecs1T
            projs3T = projs2T if tt == 1 else projs1T
            projs3T_C = projs2T_C if tt == 1 else projs1T_C
            
            ### plot trajectory
            fig = plt.figure(figsize=(10,10), dpi = 100)        
            ax = fig.add_subplot(1, 1, 1, projection='3d')
            
            ax.plot(projs1T_C[:,0], projs1T_C[:,1], projs1T_C[:,2], color = 'r', alpha = 0.8, linestyle = ':')
            #ax.plot(projs2T_C[:,0], projs2T_C[:,1], projs2T_C[:,2], color = 'b', alpha = 0.8, linestyle = '--')
            #for l in range(len(locs)):
            #    ax.plot(X1[:,l], Y1[:,l], Z1[:,l], color = cseq[l*2], alpha = 0.8)
            #    ax.plot(X2[:,l], Y2[:,l], Z2[:,l], color = cseq[l*2+1], alpha = 0.8)
            
            for nm, mcp in enumerate(mainCheckpoints):
                nmc = checkpoints.index(mcp)
                #c1, c2 = cseq[nm], cseq[nm+4]
                if mcp < 1450:
                    x1_plane, y1_plane, z1_plane = f_subspace.plane_by_vecs(vecs1T[nmc], center = projs1T_C[nmc], xRange=(projs1T[nmc,:,0].min(), projs1T[nmc,:,0].max()), yRange=(projs1T[nmc,:,1].min(), projs1T[nmc,:,1].max())) #
                    ax.plot_surface(x1_plane, y1_plane, z1_plane, alpha=0.3, color = 'r') #c1s[int(nm-len(mainCheckpoints)/2)]
                    for l in range(len(locs)):
                        ax.scatter(projs1T[nmc,l,0], projs1T[nmc,l,1], projs1T[nmc,l,2], color = 'r', marker = shapes[l], alpha = 1)
                        #c1s[int(nm-len(mainCheckpoints)/2)]
                elif mcp <= 2600:
                    x2_plane, y2_plane, z2_plane = f_subspace.plane_by_vecs(vecs2T[nmc], center = projs2T_C[nmc], xRange=(projs2T[nmc,:,0].min(), projs2T[nmc,:,0].max()), yRange=(projs2T[nmc,:,1].min(), projs2T[nmc,:,1].max())) #
                    ax.plot_surface(x2_plane, y2_plane, z2_plane, alpha=0.3, color = 'g') #c2s[int(nm-len(mainCheckpoints)/2)]
                    for l in range(len(locs)):
                        ax.scatter(projs2T[nmc,l,0], projs2T[nmc,l,1], projs2T[nmc,l,2], color = 'g', marker = shapes[l], alpha = 1)
                        #c2s[int(nm-len(mainCheckpoints)/2)]
                else:
                    x3_plane, y3_plane, z3_plane = f_subspace.plane_by_vecs(vecs3T[nmc], center = projs3T_C[nmc], xRange=(projs3T[nmc,:,0].min(), projs3T[nmc,:,0].max()), yRange=(projs3T[nmc,:,1].min(), projs3T[nmc,:,1].max())) #
                    ax.plot_surface(x3_plane, y3_plane, z3_plane, alpha=0.3, color = 'b') #c3s[int(nm-len(mainCheckpoints)/2)]
                    for l in range(len(locs)):
                        ax.scatter(projs3T[nmc,l,0], projs3T[nmc,l,1], projs3T[nmc,l,2], color = 'b', marker = shapes[l], alpha = 1)
                        #c3s[int(nm-len(mainCheckpoints)/2)]
                    
            #    for l in range(len(locs)):
            #        ax.scatter(X1[nmc,l], Y1[nmc,l], Z1[nmc,l], color = cseq[l*2], marker = shapes[l])
            #        ax.scatter(X2[nmc,l], Y2[nmc,l], Z2[nmc,l], color = cseq[l*2+1], marker = shapes[l])
            
            ax.set_xlabel('PC 1')
            ax.set_ylabel('PC 2')
            ax.set_zlabel('PC 3')
            
            #ax.invert_yaxis()
            #ax.set_title(f'item 1')
            ax.view_init(elev=10, azim=-60)
            
            plt.suptitle(f'Traj example {n}, tt = {tt}')
            plt.tight_layout()
            plt.show() 
            
                
    return
    
# In[] cosTheta, cosPsi, sse. Compare within type, between time points, between locations

def plot_choiceVSItem(geoms_valid, geoms_shuff, geomsC_valid, geomsC_shuff, checkpoints, checkpointsLabels, locs = (0,1,2,3), ttypes = (1,2), nIters = 20, nPerms = 50, nBoots = 1, label = '', pdummy = True, toPlot = False):
    
    vecs, projs, projsAll, trialInfos = geoms_valid
    vecs_shuff, projs_shuff, projsAll_shuff, trialInfos_shuff = geoms_shuff
    
    vecs_C, projs_C, projsAll_C, trialInfos_C, _ = geomsC_valid
    vecs_C_shuff, projs_C_shuff, projsAll_C_shuff, trialInfos_C_shuff, _ = geomsC_shuff
    
    cosTheta_1C, cosTheta_2C = {},{}
    cosPsi_1C, cosPsi_2C = {},{}

    # shuff
    pcosTheta_1C, pcosTheta_2C = {},{}
    pcosPsi_1C, pcosPsi_2C = {},{}
    
    
    for tt in ttypes:
        cosTheta_1CT = np.zeros((nIters, nBoots, len(checkpoints)))
        cosTheta_2CT = np.zeros((nIters, nBoots, len(checkpoints)))
        
        cosPsi_1CT = np.zeros((nIters, nBoots, len(checkpoints)))
        cosPsi_2CT = np.zeros((nIters, nBoots, len(checkpoints)))
        
        
        for n in range(nIters):
            for nbt in range(nBoots):
                for nc, cp in enumerate(checkpoints):
                    cT1C, _, cP1C, _, _ = f_subspace.angle_alignment_coplanar(vecs[cp][tt][1][n][nbt], projs[cp][tt][1][n][nbt], vecs_C[n][nbt], projs_C[n][nbt], sequence=(0,1,3,2))
                    cT2C, _, cP2C, _, _ = f_subspace.angle_alignment_coplanar(vecs[cp][tt][2][n][nbt], projs[cp][tt][2][n][nbt], vecs_C[n][nbt], projs_C[n][nbt], sequence=(0,1,3,2))
                    
                    
                    
                    cosTheta_1CT[n,nbt,nc], cosTheta_2CT[n,nbt,nc] = cT1C, cT2C# theta11, theta22, theta12# 
                    cosPsi_1CT[n,nbt,nc], cosPsi_2CT[n,nbt,nc] = cP1C, cP2C# psi11, psi22, psi12# 
                    
                        
        cosTheta_1C[tt] = cosTheta_1CT
        cosTheta_2C[tt] = cosTheta_2CT
        
        cosPsi_1C[tt] = cosPsi_1CT
        cosPsi_2C[tt] = cosPsi_2CT
                
        
        
        pcosTheta_1C[tt], pcosTheta_2C[tt] = np.ones((len(checkpoints),)), np.ones((len(checkpoints),))
        pcosPsi_1C[tt], pcosPsi_2C[tt] = np.ones((len(checkpoints),)), np.ones((len(checkpoints),))
        
        if pdummy == False:
            
            cosTheta_1C_shuffT = np.zeros((nIters, nBoots*nPerms, len(checkpoints)))
            cosTheta_2C_shuffT = np.zeros((nIters, nBoots*nPerms, len(checkpoints)))
            
            
            cosPsi_1C_shuffT = np.zeros((nIters, nBoots*nPerms, len(checkpoints)))
            cosPsi_2C_shuffT = np.zeros((nIters, nBoots*nPerms, len(checkpoints)))
            
            
            
            
            for n in range(nIters):
                for nbt in range(nBoots*nPerms):
                    for nc, cp in enumerate(checkpoints):
                        
                        cT1C, _, cP1C, _, _ = f_subspace.angle_alignment_coplanar(vecs_shuff[cp][tt][1][n][nbt], projs_shuff[cp][tt][1][n][nbt], vecs_C[n][0], projs_C[n][0], sequence=(0,1,3,2))
                        cT2C, _, cP2C, _, _ = f_subspace.angle_alignment_coplanar(vecs_shuff[cp][tt][2][n][nbt], projs_shuff[cp][tt][2][n][nbt], vecs_C[n][0], projs_C[n][0], sequence=(0,1,3,2))
                        
                        
                        cosTheta_1C_shuffT[n,nbt,nc], cosTheta_2C_shuffT[n,nbt,nc] = cT1C, cT2C
                        cosPsi_1C_shuffT[n,nbt,nc], cosPsi_2C_shuffT[n,nbt,nc] = cP1C, cP2C
                        
                        
            cosTheta_1C_shuff_all = np.concatenate([cosTheta_1C_shuffT[:,nbt,:] for nbt in range(nBoots*nPerms)])
            cosTheta_2C_shuff_all = np.concatenate([cosTheta_2C_shuffT[:,nbt,:] for nbt in range(nBoots*nPerms)])
            
            cosPsi_1C_shuff_all = np.concatenate([cosPsi_1C_shuffT[:,nbt,:] for nbt in range(nBoots*nPerms)])
            cosPsi_2C_shuff_all = np.concatenate([cosPsi_2C_shuffT[:,nbt,:] for nbt in range(nBoots*nPerms)])
            
            
            for i in range(len(checkpoints)):
            
                cT1C, cT2C = cosTheta_1CT.mean(axis=1)[:,i].round(5), cosTheta_2CT.mean(axis=1)[:,i].round(5)
                cP1C, cP2C = cosPsi_1CT.mean(axis=1)[:,i].round(5), cosPsi_2CT.mean(axis=1)[:,i].round(5)
                
                # drop nan values
                #cP11, cP22, cP12 = cP11[~np.isnan(cP11)], cP22[~np.isnan(cP22)], cP12[~np.isnan(cP12)]
                #s11, s22, s12 = s11[~np.isnan(s11)], s22[~np.isnan(s22)], s12[~np.isnan(s12)]
                
                # shuff distribution
                cT1C_shuff, cT2C_shuff = cosTheta_1C_shuff_all[:,i].round(5), cosTheta_2C_shuff_all[:,i].round(5)
                cP1C_shuff, cP2C_shuff = cosPsi_1C_shuff_all[:,i].round(5), cosPsi_2C_shuff_all[:,i].round(5)
                
                
                # compare distributions and calculate p values
                #pcosTheta_11T[i,j] = cstats.rayleigh(t11)[0] # cstats.vtest(t11,cstats.descriptive.mean(t11))[0] #
                #pcosTheta_22T[i,j] = cstats.rayleigh(t22)[0] # cstats.vtest(t22,cstats.descriptive.mean(t22))[0] #
                #pcosTheta_12T[i,j] = cstats.rayleigh(t12)[0] # cstats.vtest(t12,cstats.descriptive.mean(t12))[0] #
                
                #pcosPsi_11T[i,j] = cstats.rayleigh(p11)[0] # cstats.vtest(p11,cstats.descriptive.mean(p11))[0] #
                #pcosPsi_22T[i,j] = cstats.rayleigh(p22)[0] # cstats.vtest(p22,cstats.descriptive.mean(p22))[0] #
                #pcosPsi_12T[i,j] = cstats.rayleigh(p12)[0] # cstats.vtest(p12,cstats.descriptive.mean(p12))[0] #
                
                ##################
                # permutation ps #
                ##################
                
                #pcosTheta_11T[i,j] = f_stats.permutation_p(cT11.mean(axis=0), cT11_shuff)#/2
                #pcosTheta_22T[i,j] = f_stats.permutation_p(cT22.mean(axis=0), cT22_shuff)#/2
                #pcosTheta_12T[i,j] = f_stats.permutation_p(cT12.mean(axis=0), cT12_shuff)#/2
                
                #pcosPsi_11T[i,j] = f_stats.permutation_p(cP11.mean(axis=0), cP11_shuff)#/2
                #pcosPsi_22T[i,j] = f_stats.permutation_p(cP22.mean(axis=0), cP22_shuff)#/2
                #pcosPsi_12T[i,j] = f_stats.permutation_p(cP12.mean(axis=0), cP12_shuff)#/2
                
                ###############
                # 2samp ttest #
                ###############
                
                #pcosTheta_11T[i,j] = stats.ttest_ind(cT11.mean(axis=0), cT11_shuff)
                #pcosTheta_22T[i,j] = f_stats.permutation_p(cT22.mean(axis=0), cT22_shuff)#/2
                #pcosTheta_12T[i,j] = f_stats.permutation_p(cT12.mean(axis=0), cT12_shuff)#/2
                
                #pcosPsi_11T[i,j] = f_stats.permutation_p(cP11.mean(axis=0), cP11_shuff)#/2
                #pcosPsi_22T[i,j] = f_stats.permutation_p(cP22.mean(axis=0), cP22_shuff)#/2
                #pcosPsi_12T[i,j] = f_stats.permutation_p(cP12.mean(axis=0), cP12_shuff)#/2
                
                #################
                # kstest pvalue #
                #################
                
                pcosTheta_1C[tt][i] = stats.kstest(cT1C, cT1C_shuff)[-1] # # stats.uniform.cdf
                pcosTheta_2C[tt][i] = stats.kstest(cT2C, cT2C_shuff)[-1] # # stats.uniform.cdf
                                    
                pcosPsi_1C[tt][i] = stats.kstest(cP1C, cP1C_shuff)[-1] # #stats.uniform.cdf
                pcosPsi_2C[tt][i] = stats.kstest(cP2C, cP2C_shuff)[-1] # #stats.uniform.cdf
                
                                        
    if toPlot:
        angleCheckPoints = np.linspace(0,np.pi,7).round(5)
        color1, color2 = 'b', 'm'
        ### cosTheta
        plt.figure(figsize=(8, 3), dpi=100)
        
        for tt in ttypes:
            ttype = 'Retarget' if tt == 1 else 'Distraction'
        
            plt.subplot(1,2,tt)
            ax = plt.gca()
            
            # Item1
            ax.errorbar(np.arange(0, len(checkpoints), 1), cosTheta_1C[tt].mean(axis=1).mean(axis=0), yerr = cosTheta_1C[tt].mean(axis=1).std(axis=0), marker = 'o', color = color1, label = 'Item1')
            ax.plot(np.arange(0, len(checkpoints), 1), pcosTheta_1C[tt], alpha = 0.3, linestyle = '-', color = color1)
            # Item2
            ax.errorbar(np.arange(0, len(checkpoints), 1), cosTheta_2C[tt].mean(axis=1).mean(axis=0), yerr = cosTheta_2C[tt].mean(axis=1).std(axis=0), marker = 'o', color = color2, label = 'Item2')
            ax.plot(np.arange(0, len(checkpoints), 1), pcosTheta_2C[tt], alpha = 0.3, linestyle = '-', color = color2)
            
            trans = ax.get_xaxis_transform()
            for nc, cp in enumerate(checkpoints):
                if 0.05 < pcosTheta_1C[tt][nc] <= 0.1:
                    ax.annotate('+', xy=(nc, 0.05), xycoords = trans, annotation_clip = False, ha = 'center', va = 'center', color = color1)
                elif 0.01 < pcosTheta_1C[tt][nc] <= 0.05:
                    ax.annotate('*', xy=(nc, 0.05), xycoords = trans, annotation_clip = False, ha = 'center', va = 'center', color = color1)
                elif pcosTheta_1C[tt][nc] <= 0.01:
                    ax.annotate('**', xy=(nc, 0.05), xycoords = trans, annotation_clip = False, ha = 'center', va = 'center', color = color1)
            
            for nc, cp in enumerate(checkpoints):
                if 0.05 < pcosTheta_2C[tt][nc] <= 0.1:
                    ax.annotate('+', xy=(nc, 0.05), xycoords = trans, annotation_clip = False, ha = 'center', va = 'center', color = color2)
                elif 0.01 < pcosTheta_2C[tt][nc] <= 0.05:
                    ax.annotate('*', xy=(nc, 0.05), xycoords = trans, annotation_clip = False, ha = 'center', va = 'center', color = color2)
                elif pcosTheta_2C[tt][nc] <= 0.01:
                    ax.annotate('**', xy=(nc, 0.05), xycoords = trans, annotation_clip = False, ha = 'center', va = 'center', color = color2)
                    
            ax.set_title(f'{ttype}', pad = 10)
            ax.set_xticks([n for n in range(len(checkpoints))])
            ax.set_xticklabels(checkpointsLabels, fontsize = 10)
            ax.set_ylim((-1,1))
            ax.set_yticks(np.flip(np.cos(angleCheckPoints).round(5)), labels = np.flip(np.degrees(angleCheckPoints).round(1).astype(int)))
            ax.set_ylabel('cos(θ)',fontsize=15,rotation = 90)
            ax.legend(loc='lower right')
            
            
        
        plt.subplots_adjust(top = 0.8)
        plt.suptitle(f'Principal Angle (θ), Item-Choice', fontsize = 15, y=1)
        plt.tight_layout()
        plt.show()    
        
        
        ### cosPsi
        plt.figure(figsize=(8, 3), dpi=100)
        
        for tt in ttypes:
            ttype = 'Retarget' if tt == 1 else 'Distraction'
        
            plt.subplot(1,2,tt)
            ax = plt.gca()
            
            ax.errorbar(np.arange(0, len(checkpoints), 1), cosPsi_1C[tt].mean(axis=1).mean(axis=0), yerr = cosPsi_1C[tt].mean(axis=1).std(axis=0), marker = 'o', color = color1, label = 'Item1')
            ax.plot(np.arange(0, len(checkpoints), 1), pcosPsi_1C[tt], alpha = 0.3, linestyle = '-', color = color1)
            
            ax.errorbar(np.arange(0, len(checkpoints), 1), cosPsi_2C[tt].mean(axis=1).mean(axis=0), yerr = cosPsi_2C[tt].mean(axis=1).std(axis=0), marker = 'o', color = color2, label = 'Item2')
            ax.plot(np.arange(0, len(checkpoints), 1), pcosPsi_2C[tt], alpha = 0.3, linestyle = '-', color = color2)
            
            
            trans = ax.get_xaxis_transform()
            
            for nc, cp in enumerate(checkpoints):
                if 0.05 < pcosPsi_1C[tt][nc] <= 0.1:
                    ax.annotate('+', xy=(nc, 0.05), xycoords = trans, annotation_clip = False, ha = 'center', va = 'center', color = color1)
                elif 0.01 < pcosPsi_1C[tt][nc] <= 0.05:
                    ax.annotate('*', xy=(nc, 0.05), xycoords = trans, annotation_clip = False, ha = 'center', va = 'center', color = color1)
                elif pcosPsi_1C[tt][nc] <= 0.01:
                    ax.annotate('**', xy=(nc, 0.05), xycoords = trans, annotation_clip = False, ha = 'center', va = 'center', color = color1)
            
            for nc, cp in enumerate(checkpoints):
                if 0.05 < pcosPsi_2C[tt][nc] <= 0.1:
                    ax.annotate('+', xy=(nc, 0.05), xycoords = trans, annotation_clip = False, ha = 'center', va = 'center', color = color2)
                elif 0.01 < pcosPsi_2C[tt][nc] <= 0.05:
                    ax.annotate('*', xy=(nc, 0.05), xycoords = trans, annotation_clip = False, ha = 'center', va = 'center', color = color2)
                elif pcosPsi_2C[tt][nc] <= 0.01:
                    ax.annotate('**', xy=(nc, 0.05), xycoords = trans, annotation_clip = False, ha = 'center', va = 'center', color = color2)        
            
            ax.set_title(f'{ttype}', pad = 10)
            ax.set_xticks([n for n in range(len(checkpoints))])
            ax.set_xticklabels(checkpointsLabels, fontsize = 10)
            ax.set_ylim((-1,1))
            ax.set_yticks(np.flip(np.cos(angleCheckPoints).round(5)), labels = np.flip(np.degrees(angleCheckPoints).round(1).astype(int)))
            ax.set_ylabel('cos(Ψ)',fontsize=15,rotation = 90)
            
            ax.legend(loc='right')
        
        plt.subplots_adjust(top = 0.8)
        plt.suptitle(f'Representational Alignment (Ψ), Item-Choice', fontsize = 15, y=1)
        plt.tight_layout()
        plt.show()
    
    cosThetas_C = (cosTheta_1C, cosTheta_2C)
    cosPsis_C = (cosPsi_1C, cosPsi_2C)
    
    return cosThetas_C, cosPsis_C
    
    
# In[]

def withinTime_lda(test_InfoT, hidden_statesT, tRange, locs = (0,1,2,3), ttypes = (1,2), nPCs = (0,15), dt = 10, bins = 50, targetItem = 'locKey', pca_tWinX=None, avg_method = 'conditional_mean', nPerms = 50, permDummy = True, 
                   Y_columnsLabels = ['choice','ttype','loc1','loc2','locX'], zbsl = True, tbsl = (-300,0)):
    
    pd.options.mode.chained_assignment = None
    epsilon = 0.0000001
    
    tslice = (tRange.min(), tRange.max())

    tbins = np.arange(tslice[0], tslice[1], bins) # +dt
    locCombs = list(permutations(locs,2))
    
    Y = test_InfoT.loc[:,Y_columnsLabels].values
    
    
    if zbsl and (tbsl[0] in tRange):
        bslx1, bslx2 = tRange.tolist().index(tbsl[0]), tRange.tolist().index(tbsl[1])
        for ch in range(hidden_statesT.shape[1]):
            bsl_mean, bsl_std = hidden_statesT[:,ch,bslx1:bslx2].mean(), hidden_statesT[:,ch,bslx1:bslx2].std()
            hidden_statesT[:,ch,:] = (hidden_statesT[:,ch,:] - bsl_mean + epsilon) / (bsl_std + epsilon) #standard scaler
    
    ### scaling
    for ch in range(hidden_statesT.shape[1]):
        #hidden_statesT[:,ch,:] = scale(hidden_statesT[:,ch,:]) #01 scale for each channel
        hidden_statesT[:,ch,:] = (hidden_statesT[:,ch,:] - hidden_statesT[:,ch,:].mean() + epsilon) / (hidden_statesT[:,ch,:].std() + epsilon) #standard scaler
        #hidden_statesT[:,ch,:] = (hidden_statesT[:,ch,:] - hidden_statesT[:,ch,:].mean(axis=0)) / hidden_statesT[:,ch,:].std(axis=0) #detrended standard scaler
        #hidden_statesT[:,ch,:] = (hidden_statesT[:,ch,:] - hidden_statesT[:,ch,idxxBsl[0]:idxxBsl[1]].mean())/hidden_statesT[:,ch,idxxBsl[0]:idxxBsl[1]].std() # Z to baseline
    
    
    #pca_tWinX = None
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
        for lc in locCombs:
            lcs = str(lc[0]) + '_' + str(lc[1])
            idxx = test_InfoT_[test_InfoT_.locs == lcs].index.tolist()
            X_regionT_ += [hidden_statesTT[idxx,:,:].mean(axis=0)]
        
        hidden_statesTT = np.concatenate(X_regionT_, axis=-1)
        
    elif avg_method == 'conditional_mean':
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
    #print(f'{condition[1]}, {evr.round(4)[0:5]}')
    
    hidden_statesTP = np.zeros((hidden_statesT.shape[0], npc, hidden_statesT.shape[2]))
    for trial in range(hidden_statesT.shape[0]):
        hidden_statesTP[trial,:,:] = pca.transform(hidden_statesT[trial,:,:].T).T
    
    ntrial = len(test_InfoT)
    ### split into train and test sets
    train_setID = np.sort(np.random.choice(ntrial, round(0.5*ntrial),replace = False))
    test_setID = np.sort(np.random.choice(np.setdiff1d(np.arange(ntrial), train_setID, assume_unique=True), (ntrial-train_setID.shape[0]),replace = False))
    #test_setID = np.sort(np.random.choice(ntrial, round(0.5*ntrial),replace = False))

    train_setP = hidden_statesTP[train_setID,:,:]
    test_setP = hidden_statesTP[test_setID,:,:]
    
    targetItemX = Y_columnsLabels.index(targetItem)
    
    
    train_label = Y[train_setID,targetItemX].astype('int').astype('str') # locKey
    test_label = Y[test_setID,targetItemX].astype('int').astype('str') # locKey

    
    
    #train_label = Y[train_setID,2].astype('str') # type
    #train_label = np.char.add(Y[train_setID,0].astype('int').astype('str'), Y[train_setID,2].astype('str'))#Y[train_setID,0].astype('int') # locKey+Type
    # (locKey = 0,'locs','type','loc1','loc2', 'locX')
    
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
        
        pfmT = f_decoding.LDAPerformance(train_setP[:,nPCs[0]:nPCs[1],t], test_setP[:,nPCs[0]:nPCs[1],t], train_label, test_label)
        
        performanceX += [pfmT]
        
    #performance[ttypeT] += [np.array(performanceX)]
    
    
    # permutation with shuffled label
    performanceX_shuff = []
    for t in range(len(tbins)):
        
        if permDummy:
            performanceX_shuff += [np.ones(nPerms)] # dummy
            
        else:
            performanceX_shuff_p = []
        
            for npm in range(nPerms):
                np.random.seed(0)
                train_label_shuff, test_label_shuff = np.random.permutation(train_label), np.random.permutation(test_label)
                pfmT_shuff = f_decoding.LDAPerformance(train_setP[:,nPCs[0]:nPCs[1],t], test_setP[:,nPCs[0]:nPCs[1],t], train_label_shuff, test_label_shuff)
                performanceX_shuff_p += [pfmT_shuff]
                
            performanceX_shuff += [np.array(performanceX_shuff_p)]
            
        
    
    #performance_shuff[ttypeT] += [np.array(performanceX_shuff)]
    
    return np.array(performanceX), np.array(performanceX_shuff)

# In[]
def plot_withinTime_lda(modelD, trialInfo, X_, Y_, tRange, trialEvents, dt = 10, bins = 50, frac = 0.8, locs = (0,1,2,3), ttypes = (1,2), nPCs = (0,10), nIters = 10, nPerms = 50, 
             pca_tWins = ((300,1300), (1600,2600),), targetItem = 'locKey', conditions = None, permDummy = True, toPlot = True, label = ''):
    
    performance = {'Retarget':[], 'Distractor':[]}
    performance_shuff = {'Retarget':[], 'Distractor':[]}
    #pvalues = {'Retarget':{'dlpfc':[],'fef':[]}, 'Distractor':{'dlpfc':[],'fef':[]}}
    
    locCombs = list(permutations(locs,2))
    
    
    permDummy = permDummy
    nIters = nIters#0
    nPerms = nPerms
    
    
    # decodability with/without permutation P value
    bins = bins
    tslice = (tRange.min(), tRange.max())
    #tRange = np.arange(-300,3000,dt)
    
    tbins = np.arange(tslice[0], tslice[1], bins) #+dt
    
    #tBsl = (-300,0)
    #idxxBsl = [tRange.tolist().index(tBsl[0]), tRange.tolist().index(tBsl[1])] #
    
    locs = locs
    ttypes = ttypes
    nPCs = nPCs
    
    nPCs = [0,10]#pseudo_Pop.shape[2]

    conditions = (('ttype', 1), ('ttype', 2)) if conditions == None else conditions
    
    for n in range(nIters):
        t_IterOn = time.time()
        print(f'Iter = {n}')
        
        
        _, _, _, test_setID, test_X, _ = f_simulation.split_dataset_balance(X_, Y_, trialInfo, frac = frac, ranseed=n)

        test_Info = trialInfo.loc[test_setID,:].reset_index(drop = True)
        #test_set = test_X.cpu().numpy()

        test_Info.loc1 = test_Info.loc1.astype(int)
        test_Info.loc2 = test_Info.loc2.astype(int)
        test_Info.choice = test_Info.choice.astype(int)
        
        #test_Info['locX'] = 0
        #for i in range(len(test_Info)):
        #    test_Info.loc[i,'locX'] = test_Info.loc[i,'loc1'] if test_Info.loc[i,'ttype'] == 1 else test_Info.loc[i,'loc2']
        
        hidden_states, _ = modelD(test_X)
        hidden_states = hidden_states.data.cpu().detach().numpy()
        hidden_states = np.swapaxes(hidden_states, 1, 2) # shape as ntrials * nchs * ntime

        #out_states = out_states.data.cpu().detach().numpy()

        #for ch in range(hidden_states.shape[1]):
        #    hiddens = hidden_states[:,ch,:]
        #    hidden_states[:,ch,:] = f_stats.scale(hiddens, method = '01')
        
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
            
            #Y = test_InfoT.loc[:,['choice','ttype','loc1','loc2','locX']].values
            #ntrial = len(test_InfoT)
            
            ### decode for each region
            #for region in ('dlpfc','fef'):
                
            hidden_statesT = hidden_states[test_InfoT.index.values,:,:] # shape2 trial * cell * time
            
            # if detrend with standardization
            #for ch in range(pseudo_PopT.shape[1]):
            #    temp = pseudo_PopT[:,ch,:]
            #    pseudo_PopT[:,ch,:] = (temp - temp.mean(axis=0))/(temp.std(axis = 0) + epsilon) # to avoid divid by 0 issue
                
            #X_region = hidden_statesT
            
            pca_tWin = np.hstack([np.arange(i[0],i[1],dt, dtype = int) for i in pca_tWins]).tolist() #
            #pca_tWin = np.hstack((np.arange(800,1300,dt, dtype = int),np.arange(2100,2600,dt, dtype = int))).tolist() #
            pca_tWinX = [tRange.tolist().index(i) for i in pca_tWin]
            
            pfm, pfm_shuff = withinTime_lda(test_InfoT, hidden_statesT, tRange, locs = locs, ttypes = ttypes, nPCs = nPCs, dt = dt, bins = bins, targetItem = targetItem, 
                                                                           pca_tWinX = pca_tWinX, avg_method = 'conditional_mean', nPerms = nPerms, permDummy = permDummy)
            
            performance[ttypeT] += [pfm]
            performance_shuff[ttypeT] += [pfm_shuff]
            
            print(f'tIter = {(time.time() - t_IterOn):.4f}s')
            
            
    # plot out
    if toPlot:
        
        plt.figure(figsize=(12,5), dpi = 100)
        
        for condition in conditions:
            tt = condition[-1]
            ttypeT = 'Retarget' if condition[-1]==1 else 'Distractor'
            
            performanceT = performance[ttypeT]
            performanceT_shuff = performance_shuff[ttypeT]
            
            pfm = np.array(performanceT)
            pfm_shuff = np.concatenate(np.array(performanceT_shuff),axis=1)
            
            pvalues = np.zeros((len(tbins)))
            for t in range(len(tbins)):
            
                pvalues[t] = f_stats.permutation_p(pfm.mean(axis = 0)[t], pfm_shuff[t,:], tail = 'greater')
                #pvalues[t,t_] = stats.ttest_1samp(pfm[:,t,t_], 0.25, alternative = 'greater')[1]
                
            
            vmax = 1
            
            plt.subplot(1,2,tt)
            ax = plt.gca()
            ax.plot(np.arange(0, len(tbins), 1), pfm.mean(axis = 0), marker = ' ', color = 'g', label = targetItem)#, vmax = 0.6, xticklabels = 100, yticklabels = 100, vmin = cbar_min, vmax = cbar_max
            ax.fill_between(np.arange(0, len(tbins), 1), (pfm.mean(0) - pfm.std(0)), (pfm.mean(0) + pfm.std(0)), color='g', alpha=.1)
            
            # significance line
            segs1 = f_plotting.significance_line_segs(pvalues,0.05)
            
            for start1, end1 in segs1:
                ax.plot(np.arange(start1,end1,1), np.full_like(np.arange(start1,end1,1), 0.1, dtype='float'), color='g', linestyle='-', linewidth=2)
            
            
            # event lines
            for i in [0, 300, 1300, 1600, 2600]:
                
                #ax.plot(tbins, np.full_like(tbins,list(tbins).index(i)), 'k-.', linewidth=4, alpha = 0.25)
                ax.plot(np.full_like(tbins,list(tbins).index(i)), tbins, 'k-.', linewidth=2, alpha = 0.25)
            
            ax.set_title(f'{ttypeT}', fontsize = 20, pad = 20)
            ax.set_xticks([list(tbins).index(i) for i in [0, 300, 1300, 1600, 2600]])
            ax.set_xticklabels([0, 300, 1300, 1600, 2600], fontsize = 10)
            ax.set_xlabel('Time', fontsize = 15)
            ax.set_ylim((0,1.1))
            #ax.set_yticklabels(checkpoints, fontsize = 10)
            ax.set_ylabel('Accuracy', fontsize = 15)
            ax.legend(loc='upper right')
            
        
        plt.tight_layout()
        plt.subplots_adjust(top = 0.8)
        plt.suptitle(f'{targetItem}, nPCs = {nPCs}', fontsize = 25, y=1)
        plt.show()

    return performance, performance_shuff

# In[]
def plot_withinTime_lda12(modelD, trialInfo, X_, Y_, tRange, trialEvents, dt = 10, bins = 50, frac = 0.8, locs = (0,1,2,3), ttypes = (1,2), nPCs = (0,10), nIters = 10, nPerms = 50, 
             pca_tWins = ((300,1300), (1600,2600),), targetItems = ('loc1', 'loc2'), conditions = None, permDummy = True, toPlot = True, label = ''):
    
    performance1 = {'Retarget':[], 'Distractor':[]}
    performance1_shuff = {'Retarget':[], 'Distractor':[]}
    performance2 = {'Retarget':[], 'Distractor':[]}
    performance2_shuff = {'Retarget':[], 'Distractor':[]}
    #pvalues = {'Retarget':{'dlpfc':[],'fef':[]}, 'Distractor':{'dlpfc':[],'fef':[]}}
    
    locCombs = list(permutations(locs,2))
    
    
    permDummy = permDummy
    nIters = nIters#0
    nPerms = nPerms
    
    
    # decodability with/without permutation P value
    bins = bins
    tslice = (tRange.min(), tRange.max())
    #tRange = np.arange(-300,3000,dt)
    
    tbins = np.arange(tslice[0], tslice[1], bins) #+dt
    
    #tBsl = (-300,0)
    #idxxBsl = [tRange.tolist().index(tBsl[0]), tRange.tolist().index(tBsl[1])] #
    
    locs = locs
    ttypes = ttypes
    nPCs = nPCs
    
    nPCs = [0,10]#pseudo_Pop.shape[2]

    conditions = (('ttype', 1), ('ttype', 2)) if conditions == None else conditions
    
    for n in range(nIters):
        t_IterOn = time.time()
        print(f'Iter = {n}')
        
        
        _, _, _, test_setID, test_X, _ = f_simulation.split_dataset_balance(X_, Y_, trialInfo, frac = frac, ranseed=n)

        test_Info = trialInfo.loc[test_setID,:].reset_index(drop = True)
        #test_set = test_X.cpu().numpy()

        test_Info.loc1 = test_Info.loc1.astype(int)
        test_Info.loc2 = test_Info.loc2.astype(int)
        test_Info.choice = test_Info.choice.astype(int)
        
        #test_Info['locX'] = 0
        #for i in range(len(test_Info)):
        #    test_Info.loc[i,'locX'] = test_Info.loc[i,'loc1'] if test_Info.loc[i,'ttype'] == 1 else test_Info.loc[i,'loc2']
        
        hidden_states, _ = modelD(test_X)
        hidden_states = hidden_states.data.cpu().detach().numpy()
        hidden_states = np.swapaxes(hidden_states, 1, 2) # shape as ntrials * nchs * ntime

        #out_states = out_states.data.cpu().detach().numpy()

        #for ch in range(hidden_states.shape[1]):
        #    hiddens = hidden_states[:,ch,:]
        #    hidden_states[:,ch,:] = f_stats.scale(hiddens, method = '01')
        
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
            
            #Y = test_InfoT.loc[:,['choice','ttype','loc1','loc2','locX']].values
            #ntrial = len(test_InfoT)
            
            ### decode for each region
            #for region in ('dlpfc','fef'):
                
            hidden_statesT = hidden_states[test_InfoT.index.values,:,:] # shape2 trial * cell * time
            
            # if detrend with standardization
            #for ch in range(pseudo_PopT.shape[1]):
            #    temp = pseudo_PopT[:,ch,:]
            #    pseudo_PopT[:,ch,:] = (temp - temp.mean(axis=0))/(temp.std(axis = 0) + epsilon) # to avoid divid by 0 issue
                
            #X_region = hidden_statesT
            
            pca_tWin = np.hstack([np.arange(i[0],i[1],dt, dtype = int) for i in pca_tWins]).tolist() #
            #pca_tWin = np.hstack((np.arange(800,1300,dt, dtype = int),np.arange(2100,2600,dt, dtype = int))).tolist() #
            pca_tWinX = [tRange.tolist().index(i) for i in pca_tWin]
            
            pfm1, pfm_shuff1 = withinTime_lda(test_InfoT, hidden_statesT, tRange, locs = locs, ttypes = ttypes, nPCs = nPCs, dt = dt, bins = bins, targetItem = targetItems[0], 
                                                                           pca_tWinX = pca_tWinX, avg_method = 'conditional_mean', nPerms = nPerms, permDummy = permDummy)
            
            pfm2, pfm_shuff2 = withinTime_lda(test_InfoT, hidden_statesT, tRange, locs = locs, ttypes = ttypes, nPCs = nPCs, dt = dt, bins = bins, targetItem = targetItems[1], 
                                                                           pca_tWinX = pca_tWinX, avg_method = 'conditional_mean', nPerms = nPerms, permDummy = permDummy)
            
            performance1[ttypeT] += [pfm1]
            performance1_shuff[ttypeT] += [pfm_shuff1]
            
            performance2[ttypeT] += [pfm2]
            performance2_shuff[ttypeT] += [pfm_shuff2]
            
            print(f'tIter = {(time.time() - t_IterOn):.4f}s')
            
        
    performance, performance_shuff = (performance1, performance2), (performance1_shuff, performance2_shuff)
    
    # plot out
    if toPlot:
        
        plt.figure(figsize=(12,5), dpi = 100)
        
        for condition in conditions:
            tt = condition[-1]
            ttypeT = 'Retarget' if condition[-1]==1 else 'Distractor'
            
            performanceT1 = performance1[ttypeT]
            performanceT1_shuff = performance1_shuff[ttypeT]
            
            pfm1 = np.array(performanceT1)
            pfm1_shuff = np.concatenate(np.array(performanceT1_shuff),axis=1)
            
            performanceT2 = performance2[ttypeT]
            performanceT2_shuff = performance2_shuff[ttypeT]
            
            pfm2 = np.array(performanceT2)
            pfm2_shuff = np.concatenate(np.array(performanceT2_shuff),axis=1)
            
            pvalues1 = np.zeros((len(tbins)))
            pvalues2 = np.zeros((len(tbins)))
            for t in range(len(tbins)):
            
                pvalues1[t] = f_stats.permutation_p(pfm1.mean(axis = 0)[t], pfm1_shuff[t,:], tail = 'greater')
                pvalues2[t] = f_stats.permutation_p(pfm2.mean(axis = 0)[t], pfm2_shuff[t,:], tail = 'greater')
                
            
            vmax = 1
            
            plt.subplot(1,2,tt)
            ax = plt.gca()
            ax.plot(np.arange(0, len(tbins), 1), pfm1.mean(axis = 0), marker = ' ', color = 'b', label = targetItems[0])#, vmax = 0.6, xticklabels = 100, yticklabels = 100, vmin = cbar_min, vmax = cbar_max
            ax.fill_between(np.arange(0, len(tbins), 1), (pfm1.mean(0) - pfm1.std(0)), (pfm1.mean(0) + pfm1.std(0)), color='b', alpha=.1)
            
            ax.plot(np.arange(0, len(tbins), 1), pfm2.mean(axis = 0), marker = ' ', color = 'm', label = targetItems[1])#, vmax = 0.6, xticklabels = 100, yticklabels = 100, vmin = cbar_min, vmax = cbar_max
            ax.fill_between(np.arange(0, len(tbins), 1), (pfm2.mean(0) - pfm2.std(0)), (pfm2.mean(0) + pfm2.std(0)), color='m', alpha=.1)
            
            # significance line
            segs1 = f_plotting.significance_line_segs(pvalues1,0.05)
            segs2 = f_plotting.significance_line_segs(pvalues2,0.05)
            
            for start1, end1 in segs1:
                ax.plot(np.arange(start1,end1,1), np.full_like(np.arange(start1,end1,1), 0.05, dtype='float'), color='b', linestyle='-', linewidth=2)
            for start2, end2 in segs2:
                ax.plot(np.arange(start2,end2,1), np.full_like(np.arange(start2,end2,1), 0.1, dtype='float'), color='m', linestyle='-', linewidth=2)
            
            # event lines
            for i in [0, 300, 1300, 1600, 2600]:
                
                #ax.plot(tbins, np.full_like(tbins,list(tbins).index(i)), 'k-.', linewidth=4, alpha = 0.25)
                ax.plot(np.full_like(tbins,list(tbins).index(i)), tbins, 'k-.', linewidth=2, alpha = 0.25)
            
            ax.set_title(f'{ttypeT}', fontsize = 20, pad = 20)
            ax.set_xticks([list(tbins).index(i) for i in [0, 300, 1300, 1600, 2600]])
            ax.set_xticklabels([0, 300, 1300, 1600, 2600], fontsize = 10)
            ax.set_xlabel('Time', fontsize = 15)
            ax.set_ylim((0,1.1))
            #ax.set_yticklabels(checkpoints, fontsize = 10)
            ax.set_ylabel('Accuracy', fontsize = 15)
            ax.legend(loc='upper right')
            
        
        plt.tight_layout()
        plt.subplots_adjust(top = 0.8)
        plt.suptitle(f'{targetItems}, nPCs = {nPCs}', fontsize = 25, y=1)
        plt.show()

    return performance, performance_shuff






# In[]

def crossTemp_lda(test_InfoT, hidden_statesT, tRange, locs = (0,1,2,3), ttype = 0, nPCs = (0,10), dt = 10, bins = 50, targetItem = 'locKey', 
                  pca_tWinX=None, avg_method = 'conditional_mean', nPerms = 50, permDummy = True, Y_columnsLabels = ['choice','ttype','loc1','loc2','locX'], 
                  shuff_excludeInv = True, zbsl = True, tbsl = (-300,0)):
    
    pd.options.mode.chained_assignment = None
    epsilon = 0.0000001
    
    tslice = (tRange.min(), tRange.max())

    tbins = np.arange(tslice[0], tslice[1], bins) # +dt
    locCombs = list(permutations(locs,2))
    
    Y = test_InfoT.loc[:,Y_columnsLabels].values
    
    if zbsl and (tbsl[0] in tRange):
        bslx1, bslx2 = tRange.tolist().index(tbsl[0]), tRange.tolist().index(tbsl[1])
        for ch in range(hidden_statesT.shape[1]):
            bsl_mean, bsl_std = hidden_statesT[:,ch,bslx1:bslx2].mean(), hidden_statesT[:,ch,bslx1:bslx2].std()
            hidden_statesT[:,ch,:] = (hidden_statesT[:,ch,:] - bsl_mean + epsilon) / (bsl_std + epsilon) #standard scaler
            
    
    ### scaling
    for ch in range(hidden_statesT.shape[1]):
        #hidden_statesT[:,ch,:] = scale(hidden_statesT[:,ch,:]) #01 scale for each channel
        hidden_statesT[:,ch,:] = (hidden_statesT[:,ch,:] - hidden_statesT[:,ch,:].mean() + epsilon) / (hidden_statesT[:,ch,:].std() + epsilon) #standard scaler
        #hidden_statesT[:,ch,:] = (hidden_statesT[:,ch,:] - hidden_statesT[:,ch,:].mean(axis=0)) / hidden_statesT[:,ch,:].std(axis=0) #detrended standard scaler
        #hidden_statesT[:,ch,:] = (hidden_statesT[:,ch,:] - hidden_statesT[:,ch,idxxBsl[0]:idxxBsl[1]].mean())/hidden_statesT[:,ch,idxxBsl[0]:idxxBsl[1]].std() # Z to baseline
    
    
    #pca_tWinX = None
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
        for lc in locCombs:
            lcs = str(lc[0]) + '_' + str(lc[1])
            idxx = test_InfoT_[test_InfoT_.locs == lcs].index.tolist()
            X_regionT_ += [hidden_statesTT[idxx,:,:].mean(axis=0)]
        
        hidden_statesTT = np.concatenate(X_regionT_, axis=-1)
        
    elif avg_method == 'conditional_mean':
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
    #print(f'{condition[1]}, {evr.round(4)[0:5]}')
    
    hidden_statesTP = np.zeros((hidden_statesT.shape[0], npc, hidden_statesT.shape[2]))
    for trial in range(hidden_statesT.shape[0]):
        hidden_statesTP[trial,:,:] = pca.transform(hidden_statesT[trial,:,:].T).T
    
    ntrial = len(test_InfoT)
    ### split into train and test sets
    train_setID = np.sort(np.random.choice(ntrial, round(0.5*ntrial),replace = False))
    test_setID = np.sort(np.random.choice(np.setdiff1d(np.arange(ntrial), train_setID, assume_unique=True), (ntrial-train_setID.shape[0]),replace = False))
    #test_setID = np.sort(np.random.choice(ntrial, round(0.5*ntrial),replace = False))

    train_setP = hidden_statesTP[train_setID,:,:]
    test_setP = hidden_statesTP[test_setID,:,:]
    
    targetItemX = Y_columnsLabels.index(targetItem)
    
    
    train_label = Y[train_setID,targetItemX].astype('int').astype('str') 
    test_label = Y[test_setID,targetItemX].astype('int').astype('str')
    
    
    toDecode_labels1_inv = f_stats.inversed_label(Y_columnsLabels, targetItem, ttype)
    toDecode_X1_inv = Y_columnsLabels.index(toDecode_labels1_inv)
    train_label1_inv = Y[train_setID,toDecode_X1_inv]
    test_label1_inv = Y[test_setID,toDecode_X1_inv]
    
    
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
        for t_ in range(len(tbins)):
            
            pfmTT_ = f_decoding.LDAPerformance(train_setP[:,nPCs[0]:nPCs[1],t], test_setP[:,nPCs[0]:nPCs[1],t_], train_label, test_label)
            performanceX_ += [pfmTT_]
        
        performanceX += [np.array(performanceX_)]
        
    #performance[ttypeT] += [np.array(performanceX)]
    
    
    # permutation with shuffled label
    performanceX_shuff = []
    for t in range(len(tbins)):
        performanceX_shuff_ = []
        
        for t_ in range(len(tbins)):
            if permDummy:
                performanceX_shuff_ += [np.ones(nPerms)] # dummy
                
            else:
                performanceX_shuff_p = []
            
                for npm in range(nPerms):
                    if shuff_excludeInv:
                        # except for the inverse ones
                        train_label_shuff, test_label_shuff = np.full_like(train_label1_inv,9, dtype=int), np.full_like(test_label1_inv,9, dtype=int)

                        for ni1, i1 in enumerate(train_label1_inv.astype(int)):
                            train_label_shuff[ni1] = np.random.choice(np.array(locs)[np.array(locs)!=i1]).astype(int)
                        for nj1, j1 in enumerate(test_label1_inv.astype(int)):
                            test_label_shuff[nj1] = np.random.choice(np.array(locs)[np.array(locs)!=j1]).astype(int)
                            
                        #test_InfoT_shuff = test_InfoT.copy()
                        #test_InfoT_shuff[targetItem] = label1_shuff
                        
                    else:
                        #test_InfoT_shuff = test_InfoT.sample(frac=1)
                        train_label_shuff, test_label_shuff = np.random.permutation(train_label), np.random.permutation(test_label)
                        
                    #train_label_shuff, test_label_shuff = test_InfoT_shuff.loc[train_setID,targetItem], test_InfoT_shuff.loc[test_setID,targetItem]
                    
                    pfmTT_shuff = f_decoding.LDAPerformance(train_setP[:,nPCs[0]:nPCs[1],t], test_setP[:,nPCs[0]:nPCs[1],t_], train_label_shuff, test_label_shuff)
                    performanceX_shuff_p += [pfmTT_shuff]
                
                performanceX_shuff_ += [np.array(performanceX_shuff_p)]
            
        performanceX_shuff += [np.array(performanceX_shuff_)]
    
    #performance_shuff[ttypeT] += [np.array(performanceX_shuff)]
    
    return np.array(performanceX), np.array(performanceX_shuff)


# In[]
def plot_crossTemp_lda(modelD, trialInfo, X_, Y_, tRange, trialEvents, dt = 10, bins = 50, frac = 0.8, locs = (0,1,2,3), ttypes = (1,2), nPCs = (0,10), nIters = 10, nPerms = 50, 
             pca_tWins = ((800,1300), (2100,2600),), targetItem = 'locKey', conditions = None, permDummy = True, toPlot = True, label = ''):
    
    performance = {'Retarget':[], 'Distractor':[]}
    performance_shuff = {'Retarget':[], 'Distractor':[]}
    #pvalues = {'Retarget':{'dlpfc':[],'fef':[]}, 'Distractor':{'dlpfc':[],'fef':[]}}
    
    locCombs = list(permutations(locs,2))
    
    
    permDummy = permDummy
    nIters = nIters#0
    nPerms = nPerms
    
    
    # decodability with/without permutation P value
    bins = bins
    tslice = (tRange.min(), tRange.max())
    #tRange = np.arange(-300,3000,dt)
    
    tbins = np.arange(tslice[0], tslice[1], bins) #+dt
    
    #tBsl = (-300,0)
    #idxxBsl = [tRange.tolist().index(tBsl[0]), tRange.tolist().index(tBsl[1])] #
    
    locs = locs
    ttypes = ttypes
    nPCs = nPCs
    
    nPCs = [0,10]#pseudo_Pop.shape[2]

    conditions = (('ttype', 1), ('ttype', 2)) if conditions == None else conditions
    
    for n in range(nIters):
        t_IterOn = time.time()
        print(f'Iter = {n}')
        
        
        _, _, _, test_setID, test_X, _ = f_simulation.split_dataset_balance(X_, Y_, trialInfo, frac = frac, ranseed=n)

        test_Info = trialInfo.loc[test_setID,:].reset_index(drop = True)
        #test_set = test_X.cpu().numpy()

        test_Info.loc1 = test_Info.loc1.astype(int)
        test_Info.loc2 = test_Info.loc2.astype(int)
        test_Info.choice = test_Info.choice.astype(int)
        
        #test_Info['locX'] = 0
        #for i in range(len(test_Info)):
        #    test_Info.loc[i,'locX'] = test_Info.loc[i,'loc1'] if test_Info.loc[i,'ttype'] == 1 else test_Info.loc[i,'loc2']
        
        hidden_states, _ = modelD(test_X)
        hidden_states = hidden_states.data.cpu().detach().numpy()
        hidden_states = np.swapaxes(hidden_states, 1, 2) # shape as ntrials * nchs * ntime

        #out_states = out_states.data.cpu().detach().numpy()

        #for ch in range(hidden_states.shape[1]):
            #hiddens = hidden_states[:,ch,:]
            #hidden_states[:,ch,:] = f_stats.scale(hiddens, method = '01')
        
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
            
            #Y = test_InfoT.loc[:,['choice','ttype','loc1','loc2','locX']].values
            #ntrial = len(test_InfoT)
            
            ### decode for each region
            #for region in ('dlpfc','fef'):
                
            hidden_statesT = hidden_states[test_InfoT.index.values,:,:] # shape2 trial * cell * time
            
            # if detrend with standardization
            #for ch in range(pseudo_PopT.shape[1]):
            #    temp = pseudo_PopT[:,ch,:]
            #    pseudo_PopT[:,ch,:] = (temp - temp.mean(axis=0))/(temp.std(axis = 0) + epsilon) # to avoid divid by 0 issue
                
            #X_region = hidden_statesT
            
            pca_tWin = np.hstack([np.arange(i[0],i[1],dt, dtype = int) for i in pca_tWins]).tolist() #
            #pca_tWin = np.hstack((np.arange(800,1300,dt, dtype = int),np.arange(2100,2600,dt, dtype = int))).tolist() #
            pca_tWinX = [tRange.tolist().index(i) for i in pca_tWin]
            
            pfm, pfm_shuff = crossTemp_lda(test_InfoT, hidden_statesT, tRange, locs = locs, ttype = condition[-1], nPCs = nPCs, dt = dt, bins = bins, targetItem = targetItem, 
                                                                           pca_tWinX = pca_tWinX, avg_method = 'conditional_mean', nPerms = nPerms, permDummy = permDummy)
            
            performance[ttypeT] += [pfm]
            performance_shuff[ttypeT] += [pfm_shuff]
            
            print(f'tIter = {(time.time() - t_IterOn):.4f}s')
            
            
    # plot out
    if toPlot:
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
            sns.heatmap(pd.DataFrame(pfm.mean(axis = 0), index=tbins,columns=tbins), cmap = 'magma', vmin = 0.25, ax = ax, vmax = vmax)#, vmax = 0.6, xticklabels = 100, yticklabels = 100, vmin = cbar_min, vmax = cbar_max
            
            #from scipy import ndimage
            smooth_scale = 10
            z = ndimage.zoom(pvalues, smooth_scale)
            ax.contour(np.linspace(0, len(tbins), len(tbins) * smooth_scale),
                     np.linspace(0, len(tbins), len(tbins) * smooth_scale),
                      z, levels=(0,0.05), colors='white', alpha = 1)
            
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
            
            plt.title(f'{ttypeT}, nPCs = {nPCs}, {label}', pad = 10, fontsize = 25)
            plt.show()

    return performance, performance_shuff

# In[]
def plot_crossTemp_lda12(modelD, trialInfo, X_, Y_, tRange, trialEvents, dt = 10, bins = 50, frac = 0.8, locs = (0,1,2,3), ttypes = (1,2), nPCs = (0,10), nIters = 10, nPerms = 50, 
             pca_tWins = ((800,1300), (2100,2600),), targetItems = ('loc1', 'loc2'), conditions = None, permDummy = True, toPlot = True, label = ''):
    
    performance1 = {'Retarget':[], 'Distractor':[]}
    performance1_shuff = {'Retarget':[], 'Distractor':[]}
    performance2 = {'Retarget':[], 'Distractor':[]}
    performance2_shuff = {'Retarget':[], 'Distractor':[]}
    #pvalues = {'Retarget':{'dlpfc':[],'fef':[]}, 'Distractor':{'dlpfc':[],'fef':[]}}
    
    locCombs = list(permutations(locs,2))
    
    
    permDummy = permDummy
    nIters = nIters#0
    nPerms = nPerms
    
    
    # decodability with/without permutation P value
    bins = bins
    tslice = (tRange.min(), tRange.max())
    #tRange = np.arange(-300,3000,dt)
    
    tbins = np.arange(tslice[0], tslice[1], bins) #+dt
    
    #tBsl = (-300,0)
    #idxxBsl = [tRange.tolist().index(tBsl[0]), tRange.tolist().index(tBsl[1])] #
    
    locs = locs
    ttypes = ttypes
    nPCs = nPCs
    
    nPCs = [0,10]#pseudo_Pop.shape[2]

    conditions = (('ttype', 1), ('ttype', 2)) if conditions == None else conditions
    
    for n in range(nIters):
        t_IterOn = time.time()
        print(f'Iter = {n}')
        
        
        _, _, _, test_setID, test_X, _ = f_simulation.split_dataset_balance(X_, Y_, trialInfo, frac = frac, ranseed=n)

        test_Info = trialInfo.loc[test_setID,:].reset_index(drop = True)
        #test_set = test_X.cpu().numpy()

        test_Info.loc1 = test_Info.loc1.astype(int)
        test_Info.loc2 = test_Info.loc2.astype(int)
        test_Info.choice = test_Info.choice.astype(int)
        
        #test_Info['locX'] = 0
        #for i in range(len(test_Info)):
        #    test_Info.loc[i,'locX'] = test_Info.loc[i,'loc1'] if test_Info.loc[i,'ttype'] == 1 else test_Info.loc[i,'loc2']
        
        hidden_states, _ = modelD(test_X)
        hidden_states = hidden_states.data.cpu().detach().numpy()
        hidden_states = np.swapaxes(hidden_states, 1, 2) # shape as ntrials * nchs * ntime

        #out_states = out_states.data.cpu().detach().numpy()

        #for ch in range(hidden_states.shape[1]):
            #hiddens = hidden_states[:,ch,:]
            #hidden_states[:,ch,:] = f_stats.scale(hiddens, method = '01')
        
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
            
            #Y = test_InfoT.loc[:,['choice','ttype','loc1','loc2','locX']].values
            #ntrial = len(test_InfoT)
            
            ### decode for each region
            #for region in ('dlpfc','fef'):
                
            hidden_statesT = hidden_states[test_InfoT.index.values,:,:] # shape2 trial * cell * time
            
            # if detrend with standardization
            #for ch in range(pseudo_PopT.shape[1]):
            #    temp = pseudo_PopT[:,ch,:]
            #    pseudo_PopT[:,ch,:] = (temp - temp.mean(axis=0))/(temp.std(axis = 0) + epsilon) # to avoid divid by 0 issue
                
            #X_region = hidden_statesT
            
            pca_tWin = np.hstack([np.arange(i[0],i[1],dt, dtype = int) for i in pca_tWins]).tolist() #
            #pca_tWin = np.hstack((np.arange(800,1300,dt, dtype = int),np.arange(2100,2600,dt, dtype = int))).tolist() #
            pca_tWinX = [tRange.tolist().index(i) for i in pca_tWin]
            
            pfm1, pfm1_shuff = crossTemp_lda(test_InfoT, hidden_statesT, tRange, locs = locs, ttype = condition[-1], nPCs = nPCs, dt = dt, bins = bins, targetItem = targetItems[0], 
                                                                           pca_tWinX = pca_tWinX, avg_method = 'conditional_mean', nPerms = nPerms, permDummy = permDummy)
            pfm2, pfm2_shuff = crossTemp_lda(test_InfoT, hidden_statesT, tRange, locs = locs, ttype = condition[-1], nPCs = nPCs, dt = dt, bins = bins, targetItem = targetItems[1], 
                                                                           pca_tWinX = pca_tWinX, avg_method = 'conditional_mean', nPerms = nPerms, permDummy = permDummy)
            performance1[ttypeT] += [pfm1]
            performance1_shuff[ttypeT] += [pfm1_shuff]
            performance2[ttypeT] += [pfm2]
            performance2_shuff[ttypeT] += [pfm2_shuff]
            
            print(f'tIter = {(time.time() - t_IterOn):.4f}s')
            
    performance, performance_shuff = (performance1, performance2), (performance1_shuff, performance2_shuff)
    
    # plot out
    if toPlot:
        
        plt.figure(figsize=(28, 24), dpi=100)
        
        for condition in conditions:
            
            tt = condition[-1]
            ttypeT = 'Retarget' if condition[-1]==1 else 'Distractor'
            ttypeT_ = 'Retarget' if condition[-1]==1 else 'Distraction'
            
            performanceT1 = performance1[ttypeT]
            performanceT1_shuff = performance1_shuff[ttypeT]
            performanceT2 = performance2[ttypeT]
            performanceT2_shuff = performance2_shuff[ttypeT]
            
            pfm1 = np.array(performanceT1)
            pfm1_shuff = np.concatenate(np.array(performanceT1_shuff),axis=2)
            pfm2 = np.array(performanceT2)
            pfm2_shuff = np.concatenate(np.array(performanceT2_shuff),axis=2)
            
            pvalues1 = np.zeros((len(tbins), len(tbins)))
            pvalues2 = np.zeros((len(tbins), len(tbins)))
            for t in range(len(tbins)):
                for t_ in range(len(tbins)):
                    pvalues1[t,t_] = f_stats.permutation_p(pfm1.mean(axis = 0)[t,t_], pfm1_shuff[t,t_,:], tail = 'greater')
                    pvalues2[t,t_] = f_stats.permutation_p(pfm2.mean(axis = 0)[t,t_], pfm2_shuff[t,t_,:], tail = 'greater')
                    
            
            vmax = 1
            
            plt.subplot(2,2,tt)
            ax = plt.gca()
            sns.heatmap(pd.DataFrame(pfm1.mean(axis = 0), index=tbins,columns=tbins), cmap = 'magma', vmin = 0.25, vmax = vmax, ax = ax)#, vmax = 0.6, xticklabels = 100, yticklabels = 100, vmin = cbar_min, vmax = cbar_max
            
            #from scipy import ndimage
            smooth_scale = 10
            z = ndimage.zoom(pvalues1, smooth_scale)
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
            
            ax.set_title(f'{ttypeT_}, Item1', fontsize = 30, pad = 20)
            
            
            # item2
            plt.subplot(2,2,tt+2)
            ax = plt.gca()
            sns.heatmap(pd.DataFrame(pfm2.mean(axis = 0), index=tbins,columns=tbins), cmap = 'magma', vmin = 0.25, vmax = vmax, ax = ax)#, vmax = 0.6, xticklabels = 100, yticklabels = 100, vmin = cbar_min, vmax = cbar_max
            
            #from scipy import ndimage
            smooth_scale = 10
            z = ndimage.zoom(pvalues2, smooth_scale)
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
            
            ax.set_title(f'{ttypeT_}, Item2', fontsize = 30, pad = 20)
            
        plt.tight_layout()
        plt.subplots_adjust(top = 0.95)
        plt.suptitle(f'{label}, Full Space', fontsize = 35, y=1) #, Arti_Noise = {arti_noise_level}
        plt.show()

    return performance, performance_shuff












