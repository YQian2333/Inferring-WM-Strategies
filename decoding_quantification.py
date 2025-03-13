# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 08:36:07 2024

@author: aka2333
"""
#%%
%reload_ext autoreload
%autoreload 2

from itertools import permutations, product

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import ndimage
import seaborn as sns

import f_stats
import f_decoding
import f_plotting

#%% initialize parameters

data_path = 'D:/data' 
tRangeRaw = np.arange(-500,4000,1) # -300 baseline, 0 onset, 300 pre1, 1300 delay1, 1600 pre2, 2600 delay2, response
locs = (0,1,2,3,)
locCombs = list(permutations(locs,2))
ttypes = (1,2,)
dropCombs = ()
subConditions = list(product(locCombs, ttypes))

pd.options.mode.chained_assignment = None
epsilon = 0.0000001
bins = 50

tslice = (-300,2700)
tbins = np.arange(tslice[0], tslice[1], bins)
#%% setting figure propertys and parameters

################
# figure props #
################

color1, color1_, color2, color2_ = '#d29c2f', '#f5df7a', '#3c79b4', '#b3cde4'
color3, color3_, color4, color4_ = '#185337', '#96d9ad', '#804098', '#c4a2d1'

boxprops0 = dict(facecolor='lightgrey', edgecolor='none',alpha = 1)
flierprops0 = dict(markeredgecolor='lightgrey', markerfacecolor='lightgrey',alpha = 1)
capprops0 = dict(color='lightgrey',alpha = 1)
whiskerprops0 = dict(color='lightgrey',alpha = 1)
meanpointprops0 = dict(marker='^', markeredgecolor='lightgrey', markerfacecolor='w',alpha = 1)

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

#%%

##############
# Full Space #
##############

#%% load precomputed full space decodability
performanceX1_full_data = np.load(f'{data_path}/' + 'performanceX1_full_data.npy', allow_pickle=True).item()
performanceX2_full_data = np.load(f'{data_path}/' + 'performanceX2_full_data.npy', allow_pickle=True).item()
performanceX1_full_shuff_data = np.load(f'{data_path}/' + 'performanceX1_full_shuff_data.npy', allow_pickle=True).item()
performanceX2_full_shuff_data = np.load(f'{data_path}/' + 'performanceX2_full_shuff_data.npy', allow_pickle=True).item()
EVRs = np.load(f'{data_path}/' + f'EVRs_full_data.npy', allow_pickle=True).item()

performanceX1_full_rnn = np.load(f'{data_path}/' + 'performanceX1_full_rnn.npy', allow_pickle=True).item()
performanceX2_full_rnn = np.load(f'{data_path}/' + 'performanceX2_full_rnn.npy', allow_pickle=True).item()
performanceX1_full_shuff_rnn = np.load(f'{data_path}/' + 'performanceX1_full_shuff_rnn.npy', allow_pickle=True).item()
performanceX2_full_shuff_rnn = np.load(f'{data_path}/' + 'performanceX2_full_shuff_rnn.npy', allow_pickle=True).item()

#%% [Figure 2A] plot cross temporal decoding for bio data
nPCs_region = {'dlpfc':[0,15], 'fef':[0,15]}
conditions = (('type', 1), ('type', 2))

events = [0, 1300, 2600] # if monkey A [0, 1400, 2800] # events to label on the plot

for region in ('dlpfc','fef'):
    nPCs = nPCs_region[region]
    evrSum = np.array(EVRs[region])[:,nPCs[0]:nPCs[1]].mean(0).sum().round(3)
    vmax = 0.6 if region == 'dlpfc' else 0.8
    
    regionLabel = 'LPFC' if region == 'dlpfc' else 'PAC'
    
    fig = plt.figure(figsize=(28, 24), dpi=100)
    
    for condition in conditions:
        
        ttypeT = 'Retarget' if condition[-1]==1 else 'Distractor'
        ttypeT_ = 'Retarget' if condition[-1]==1 else 'Distraction'
        tt = 1 if condition[-1]==1 else 2
        
        performanceT1 = performanceX1_full_data[ttypeT][region]
        performanceT1_shuff = performanceX1_full_shuff_data[ttypeT][region]
        performanceT2 = performanceX2_full_data[ttypeT][region]
        performanceT2_shuff = performanceX2_full_shuff_data[ttypeT][region]
        
        pfm1 = np.array(performanceT1).mean(1)
        pfm1_shuff = np.array(performanceT1_shuff).mean(1)
        pfm2 = np.array(performanceT2).mean(1)
        pfm2_shuff = np.array(performanceT2_shuff).mean(1)
        
        # calculate p values for each timebin
        pvalues1 = np.zeros((len(tbins), len(tbins)))
        pvalues2 = np.zeros((len(tbins), len(tbins)))
        for t in range(len(tbins)):
            for t_ in range(len(tbins)):
                pvalues1[t,t_] = f_stats.permutation_pCI(pfm1[:,t,t_], pfm1_shuff[:,t,t_],tail='greater',alpha=5)
                pvalues2[t,t_] = f_stats.permutation_pCI(pfm2[:,t,t_], pfm2_shuff[:,t,t_],tail='greater',alpha=5)
        
        
        # item1
        plt.subplot(2,2,tt)
        ax = plt.gca()
        sns.heatmap(pd.DataFrame(pfm1.mean(axis = 0), index=tbins,columns=tbins), cmap = 'magma', vmin = 0.25, vmax = vmax, ax = ax)
        
        #
        smooth_scale = 10
        z = ndimage.zoom(pvalues1, smooth_scale)
        ax.contour(np.linspace(0, len(tbins), len(tbins) * smooth_scale),
                 np.linspace(0, len(tbins), len(tbins) * smooth_scale),
                  z, levels=([0.05]), colors='white', alpha = 1, linewidths=3)
        
        ax.invert_yaxis()
        
        
        # event lines
        for i in events:
            ax.plot(tbins, np.full_like(tbins,list(tbins).index(i)), 'w-.', linewidth=4)
            ax.plot(np.full_like(tbins,list(tbins).index(i)), tbins, 'w-.', linewidth=4)
        
        ax.set_xticks([list(tbins).index(i) for i in events])
        ax.set_xticklabels(events, rotation=0, fontsize = 20)
        ax.set_xlabel('Test Timebin (ms)', labelpad = 10, fontsize = 25)
        ax.set_yticks([list(tbins).index(i) for i in events])
        ax.set_yticklabels(events, fontsize = 20)
        ax.set_ylabel('Train Timebin (ms)', labelpad = 10, fontsize = 25)
        
        cbar = ax.collections[0].colorbar
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
                  z, levels=([0.05]), colors='white', alpha = 1, linewidths=3)
        
        ax.invert_yaxis()
        
        
        # event lines
        for i in [0, 1300, 2600]:
            ax.plot(tbins, np.full_like(tbins,list(tbins).index(i)), 'w-.', linewidth=4)
            ax.plot(np.full_like(tbins,list(tbins).index(i)), tbins, 'w-.', linewidth=4)
        
        ax.set_xticks([list(tbins).index(i) for i in [0, 1300, 2600]])
        ax.set_xticklabels(['S1', 'S2', 'Go Cue'], rotation=0, fontsize = 20)
        ax.set_xlabel('Test Timebin (ms)', labelpad = 10, fontsize = 25)
        ax.set_yticks([list(tbins).index(i) for i in [0, 1300, 2600]])
        ax.set_yticklabels(['S1', 'S2', 'Go Cue'], fontsize = 20)
        ax.set_ylabel('Train Timebin (ms)', labelpad = 10, fontsize = 25)
        
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=20)
        
        ax.set_title(f'{ttypeT_}, Item2', fontsize = 30, pad = 20)
        
    plt.tight_layout()
    plt.subplots_adjust(top = 0.95)
        
    plt.suptitle(f'{regionLabel}, Full Space (ΣEVR={evrSum})', fontsize = 35, y=1)
    plt.show()
    
#%% [Figure 2A] plot item decodability full space for RNNs

dt = 50 # The simulation timestep.
tau = dt # time constant, here set to be the same as the timestep
tRange = np.arange(-300,2700,dt) # simulation time
tLength = len(tRange) # The trial length.

fitIntervals = {'R@R':((0,1300),(1600,2600),), 'R&U':((300,1300), (1600,2600),), } 

for nfi, kfi in enumerate(fitIntervals):
    print(kfi)
    strategyLabel = 'Rehearse & Update' if kfi == 'R&U' else 'Retrieve at Recall'

    pfmX1 = performanceX1_full_rnn[kfi]
    pfmX2 = performanceX2_full_rnn[kfi]
    pfmX1_shuff = performanceX1_full_shuff_rnn[kfi]
    pfmX2_shuff = performanceX2_full_shuff_rnn[kfi]
    
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
        

#%% [Figure 2B] evaluate and plot full space code stability

##################
# code stability #
##################

ld1 = np.arange(800,1300+bins,bins)
ld1x = [tbins.tolist().index(t) for t in ld1]

ld2 = np.arange(2100,2600+bins,bins)
ld2x = [tbins.tolist().index(t) for t in ld2]

stabRatioD1_full_rnn, stabRatioD2_full_rnn = {}, {}
stabRatioD1_full_shuff_rnn, stabRatioD2_full_shuff_rnn = {}, {}

for k in ('ed2','ed12'):
    pfmX1 = {tt:np.array(performanceX1_full_rnn[k][tt]).mean(1).mean(1) for tt in (1,2)}
    pfmX2 = {tt:np.array(performanceX2_full_rnn[k][tt]).mean(1).mean(1) for tt in (1,2)}
    pfmX1_shuff = {tt:np.array(performanceX1_full_shuff_rnn[k][tt]).mean(1).mean(1) for tt in (1,2)}
    pfmX2_shuff = {tt:np.array(performanceX2_full_shuff_rnn[k][tt]).mean(1).mean(1) for tt in (1,2)}

    stabRatioD1_full_rnn[k] = np.array([np.mean((f_decoding.stability_ratio(pfmX1[1][n][ld1x,:][:,ld1x]), 
                                                 f_decoding.stability_ratio(pfmX1[2][n][ld1x,:][:,ld1x]))) for n in range(100)])
    stabRatioD2_full_rnn[k] = np.array([np.mean((f_decoding.stability_ratio(pfmX2[1][n][ld2x,:][:,ld2x]), 
                                                 f_decoding.stability_ratio(pfmX1[2][n][ld2x,:][:,ld2x]))) for n in range(100)])
    
    stabRatioD1_full_shuff_rnn[k] = np.array([np.mean((f_decoding.stability_ratio(pfmX1_shuff[1][n][ld1x,:][:,ld1x]), 
                                                       f_decoding.stability_ratio(pfmX1_shuff[2][n][ld1x,:][:,ld1x]))) for n in range(100)])
    stabRatioD2_full_shuff_rnn[k] = np.array([np.mean((f_decoding.stability_ratio(pfmX2_shuff[1][n][ld2x,:][:,ld2x]), 
                                                       f_decoding.stability_ratio(pfmX1_shuff[2][n][ld2x,:][:,ld2x]))) for n in range(100)])


stabRatioD1_full_data, stabRatioD2_full_data = {}, {}
stabRatioD1_full_shuff_data, stabRatioD2_full_shuff_data = {}, {}

for k in ('dlpfc','fef'):
    ttypes = ('Retarget','Distractor')
    pfmX1 = {tt:np.array(performanceX1_full_data[ttypes[tt-1]][k]).mean(1) for tt in (1,2)}
    pfmX2 = {tt:np.array(performanceX2_full_data[ttypes[tt-1]][k]).mean(1) for tt in (1,2)}
    pfmX1_shuff = {tt:np.array(performanceX1_full_shuff_data[ttypes[tt-1]][k]).mean(1) for tt in (1,2)}
    pfmX2_shuff = {tt:np.array(performanceX2_full_shuff_data[ttypes[tt-1]][k]).mean(1) for tt in (1,2)}

    stabRatioD1_full_data[k] = np.array([np.mean((f_decoding.stability_ratio(pfmX1[1][n][ld1x,:][:,ld1x]), 
                                                 f_decoding.stability_ratio(pfmX1[2][n][ld1x,:][:,ld1x]))) for n in range(100)])
    stabRatioD2_full_data[k] = np.array([np.mean((f_decoding.stability_ratio(pfmX2[1][n][ld2x,:][:,ld2x]), 
                                                 f_decoding.stability_ratio(pfmX1[2][n][ld2x,:][:,ld2x]))) for n in range(100)])
    
    stabRatioD1_full_shuff_data[k] = np.array([np.mean((f_decoding.stability_ratio(pfmX1_shuff[1][n][ld1x,:][:,ld1x]), 
                                                       f_decoding.stability_ratio(pfmX1_shuff[2][n][ld1x,:][:,ld1x]))) for n in range(100)])
    stabRatioD2_full_shuff_data[k] = np.array([np.mean((f_decoding.stability_ratio(pfmX2_shuff[1][n][ld2x,:][:,ld2x]), 
                                                       f_decoding.stability_ratio(pfmX1_shuff[2][n][ld2x,:][:,ld2x]))) for n in range(100)])

lineh = np.arange(0.3,0.7,0.001)
linev = np.arange(1.16,1.165,0.0001)

fig, axes = plt.subplots(1,1, sharey=True, figsize=(3,3), dpi=300)

# full space
ax = axes#.flatten()[0]

ax.boxplot([stabRatioD1_full_rnn['ed2']], positions=[0.3], patch_artist=True, widths = 0.2, boxprops=boxprops1, flierprops=flierprops1, 
                  meanprops=meanpointprops1, medianprops=medianprops, capprops = capprops1, whiskerprops = whiskerprops1, meanline=False, showmeans=True)
ax.boxplot([stabRatioD2_full_rnn['ed2']], positions=[0.7], patch_artist=True, widths = 0.2, boxprops=dict(boxprops1,facecolor=color1_), flierprops=dict(flierprops1,markeredgecolor=color1_, markerfacecolor=color1_), 
                  meanprops=dict(meanpointprops1,markeredgecolor=color1_), medianprops=medianprops, capprops = dict(capprops1,color=color1_), whiskerprops = dict(whiskerprops1,color=color1_), meanline=False, showmeans=True)

ax.boxplot([stabRatioD1_full_rnn['ed12']], positions=[1.3], patch_artist=True, widths = 0.2, boxprops=boxprops2, flierprops=flierprops2, 
                  meanprops=meanpointprops2, medianprops=medianprops, capprops = capprops2, whiskerprops = whiskerprops2, meanline=False, showmeans=True)
ax.boxplot([stabRatioD2_full_rnn['ed12']], positions=[1.7], patch_artist=True, widths = 0.2, boxprops=dict(boxprops2,facecolor=color2_), flierprops=dict(flierprops2,markeredgecolor=color2_, markerfacecolor=color2_), 
                  meanprops=dict(meanpointprops2,markeredgecolor=color2_), medianprops=medianprops, capprops = dict(capprops2,color=color2_), whiskerprops = dict(whiskerprops2,color=color2_), meanline=False, showmeans=True)


ax.boxplot([stabRatioD1_full_data['dlpfc']], positions=[2.3], patch_artist=True, widths = 0.2, boxprops=boxprops3, flierprops=flierprops3, 
                  meanprops=meanpointprops3, medianprops=medianprops, capprops = capprops3, whiskerprops = whiskerprops3, meanline=False, showmeans=True)
ax.boxplot([stabRatioD2_full_data['dlpfc']], positions=[2.7], patch_artist=True, widths = 0.2, boxprops=dict(boxprops3,facecolor=color3_), flierprops=dict(flierprops3,markeredgecolor=color3_, markerfacecolor=color3_), 
                  meanprops=dict(meanpointprops3,markeredgecolor=color3_), medianprops=medianprops, capprops = dict(capprops3,color=color3_), whiskerprops = dict(whiskerprops3,color=color3_), meanline=False, showmeans=True)

ax.boxplot([stabRatioD1_full_data['fef']], positions=[3.3], patch_artist=True, widths = 0.2, boxprops=boxprops4, flierprops=flierprops4, 
                  meanprops=meanpointprops4, medianprops=medianprops, capprops = capprops4, whiskerprops = whiskerprops4, meanline=False, showmeans=True)
ax.boxplot([stabRatioD2_full_data['fef']], positions=[3.7], patch_artist=True, widths = 0.2, boxprops=dict(boxprops4,facecolor=color4_), flierprops=dict(flierprops4,markeredgecolor=color4_, markerfacecolor=color4_), 
                  meanprops=dict(meanpointprops4,markeredgecolor=color4_), medianprops=medianprops, capprops = dict(capprops4,color=color4_), whiskerprops = dict(whiskerprops4,color=color4_), meanline=False, showmeans=True)



p1 = f_stats.permutation_pCI(stabRatioD1_full_rnn['ed2'], stabRatioD1_full_shuff_rnn['ed2'], tail = 'smaller',alpha=5)
ax.text(0.3,0.425, f'{f_plotting.sig_marker(p1,ns_note=False)}',horizontalalignment='center', verticalalignment='bottom', fontsize=10)
p2 = f_stats.permutation_pCI(stabRatioD2_full_rnn['ed2'], stabRatioD2_full_shuff_rnn['ed2'], tail = 'smaller',alpha=5)
ax.text(0.7,0.425, f'{f_plotting.sig_marker(p2,ns_note=False)}',horizontalalignment='center', verticalalignment='bottom', fontsize=10)

p3 = f_stats.permutation_pCI(stabRatioD1_full_rnn['ed12'], stabRatioD1_full_shuff_rnn['ed12'], tail = 'smaller',alpha=5)
ax.text(1.3,0.425, f'{f_plotting.sig_marker(p3,ns_note=False)}',horizontalalignment='center', verticalalignment='bottom', fontsize=10)
p4 = f_stats.permutation_pCI(stabRatioD2_full_rnn['ed12'], stabRatioD2_full_shuff_rnn['ed12'], tail = 'smaller',alpha=5)
ax.text(1.7,0.425, f'{f_plotting.sig_marker(p4,ns_note=False)}',horizontalalignment='center', verticalalignment='bottom', fontsize=10)

p12 = f_stats.permutation_p_diff(stabRatioD1_full_rnn['ed2'], stabRatioD2_full_rnn['ed2'])
ax.plot(lineh, np.full_like(lineh, 1.165), 'k-')
ax.plot(np.full_like(linev, 0.3), linev, 'k-')
ax.plot(np.full_like(linev, 0.7), linev, 'k-')
ax.text(0.5,1.175, f'{f_plotting.sig_marker(p12,ns_note=True)}',horizontalalignment='center', verticalalignment='bottom', fontsize=10)

p34 = f_stats.permutation_p_diff(stabRatioD1_full_rnn['ed12'], stabRatioD2_full_rnn['ed12'])
ax.plot(lineh+1, np.full_like(lineh, 1.165), 'k-')
ax.plot(np.full_like(linev, 0.3)+1, linev, 'k-')
ax.plot(np.full_like(linev, 0.7)+1, linev, 'k-')
ax.text(1.5,1.175, f'{f_plotting.sig_marker(p34,ns_note=True)}',horizontalalignment='center', verticalalignment='bottom', fontsize=10)


p5 = f_stats.permutation_pCI(stabRatioD1_full_data['dlpfc'], stabRatioD1_full_shuff_data['dlpfc'], tail = 'smaller',alpha=5)
ax.text(2.3,0.425, f'{f_plotting.sig_marker(p5,ns_note=False)}',horizontalalignment='center', verticalalignment='bottom', fontsize=10)
p6 = f_stats.permutation_pCI(stabRatioD2_full_data['dlpfc'], stabRatioD2_full_shuff_data['dlpfc'], tail = 'smaller',alpha=5)
ax.text(2.7,0.425, f'{f_plotting.sig_marker(p6,ns_note=False)}',horizontalalignment='center', verticalalignment='bottom', fontsize=10)

p7 = f_stats.permutation_pCI(stabRatioD1_full_data['fef'], stabRatioD1_full_shuff_data['fef'], tail = 'smaller',alpha=5)
ax.text(3.3,0.425, f'{f_plotting.sig_marker(p7,ns_note=False)}',horizontalalignment='center', verticalalignment='bottom', fontsize=10)
p8 = f_stats.permutation_pCI(stabRatioD2_full_data['fef'], stabRatioD2_full_shuff_data['fef'], tail = 'smaller',alpha=5)
ax.text(3.7,0.425, f'{f_plotting.sig_marker(p8,ns_note=False)}',horizontalalignment='center', verticalalignment='bottom', fontsize=10)

p56 = f_stats.permutation_p_diff(stabRatioD1_full_data['dlpfc'], stabRatioD2_full_data['dlpfc'])
ax.plot(lineh+2, np.full_like(lineh, 1.165), 'k-')
ax.plot(np.full_like(linev, 0.3)+2, linev, 'k-')
ax.plot(np.full_like(linev, 0.7)+2, linev, 'k-')
ax.text(2.5,1.175, f'{f_plotting.sig_marker(p56,ns_note=True)}',horizontalalignment='center', verticalalignment='bottom', fontsize=10)

p78 = f_stats.permutation_p_diff(stabRatioD1_full_data['fef'], stabRatioD2_full_data['fef'])
ax.plot(lineh+3, np.full_like(lineh, 1.165), 'k-')
ax.plot(np.full_like(linev, 0.3)+3, linev, 'k-')
ax.plot(np.full_like(linev, 0.7)+3, linev, 'k-')
ax.text(3.5,1.175, f'{f_plotting.sig_marker(p78,ns_note=True)}',horizontalalignment='center', verticalalignment='bottom', fontsize=10)


ax.set_ylabel('Off-/On-Diagonal', labelpad = 3, fontsize = 10)


xlims = ax.get_xlim()
lineThresholds = np.arange(xlims[0],xlims[1]+0.01,0.01)
ax.plot(lineThresholds, np.full_like(lineThresholds,1), 'k--', alpha = 0.5)

# draw temporary red and blue lines and use them to create a legend
plt.plot([], c='dimgrey', label='Delay1')
plt.plot([], c='lightgrey', label='Delay2')
plt.legend(bbox_to_anchor=(1.5, 0.6))#loc = 'right',

ax.set_xticks([0.5,1.5,2.5,3.5],['R@R','R&U', 'LPFC','FEF'], rotation=0)
ax.set_xlabel('Strategy/Region', labelpad = 5, fontsize = 12)
ax.set_ylim(top=1.3, bottom=0.4)
ax.set_title('Code Stability, Full Space', fontsize = 12, pad=10)
plt.show()

# print stats
print(f"R@R, D1: M(SD) = {stabRatioD1_full_rnn['ed2'].mean():.3f}({stabRatioD1_full_rnn['ed2'].std():.3f}), p = {p1:.3f};")
print(f"R&U, D1: M(SD) = {stabRatioD1_full_rnn['ed12'].mean():.3f}({stabRatioD1_full_rnn['ed12'].std():.3f}), p = {p3:.3f};")
print(f"LPFC, D1: M(SD) = {stabRatioD1_full_data['dlpfc'].mean():.3f}({stabRatioD1_full_data['dlpfc'].std():.3f}), p = {p5:.3f};")
print(f"FEF, D1: M(SD) = {stabRatioD1_full_data['fef'].mean():.3f}({stabRatioD1_full_data['fef'].std():.3f}), p = {p7:.3f};")
print('\n')
print(f"R@R, D2: M(SD) = {stabRatioD2_full_rnn['ed2'].mean():.3f}({stabRatioD2_full_rnn['ed2'].std():.3f}), p = {p2:.3f};")
print(f"R&U, D2: M(SD) = {stabRatioD2_full_rnn['ed12'].mean():.3f}({stabRatioD2_full_rnn['ed12'].std():.3f}), p = {p4:.3f};")
print(f"LPFC, D2: M(SD) = {stabRatioD2_full_data['dlpfc'].mean():.3f}({stabRatioD2_full_data['dlpfc'].std():.3f}), p = {p6:.3f};")
print(f"FEF, D2: M(SD) = {stabRatioD2_full_data['fef'].mean():.3f}({stabRatioD2_full_data['fef'].std():.3f}), p = {p8:.3f};")
print('\n')
print(f"R@R, D1-D2: MD = {(stabRatioD1_full_rnn['ed2'].mean()-stabRatioD2_full_rnn['ed2'].mean()):.3f}, p = {p12:.3f}, g = {f_stats.hedges_g(stabRatioD1_full_rnn['ed2'],stabRatioD2_full_rnn['ed2']):.3f};")
print(f"R&U, D1-D2: MD = {(stabRatioD1_full_rnn['ed12'].mean()-stabRatioD2_full_rnn['ed12'].mean()):.3f}, p = {p34:.3f}, g = {f_stats.hedges_g(stabRatioD1_full_rnn['ed12'],stabRatioD2_full_rnn['ed12']):.3f};")
print(f"LPFC, D1-D2: MD = {(stabRatioD1_full_data['dlpfc'].mean()-stabRatioD2_full_data['dlpfc'].mean()):.3f}, p = {p56:.3f}, g = {f_stats.hedges_g(stabRatioD1_full_data['dlpfc'],stabRatioD2_full_data['dlpfc']):.3f};")
print(f"FEF, D1-D2: MD = {(stabRatioD1_full_data['fef'].mean()-stabRatioD2_full_data['fef'].mean()):.3f}, p = {p78:.3f}, g = {f_stats.hedges_g(stabRatioD1_full_data['fef'],stabRatioD2_full_data['fef']):.3f};")

#%% [Figure 2C] evaluate and plot full space code morphing

#################
# code morphing #
#################

ld1 = np.arange(800,1300+bins,bins)
ld1x = [tbins.tolist().index(t) for t in ld1]
ld2 = np.arange(2100,2600+bins,bins)
ld2x = [tbins.tolist().index(t) for t in ld2]

codeMorph_full_rnn = {}
codeMorph_full_shuff_rnn = {}

for k in ('ed2','ed12'):
    pfmX1 = {tt:np.array(performanceX1_full_rnn[k][tt]).mean(1).mean(1) for tt in (1,2)}
    pfmX2 = {tt:np.array(performanceX2_full_rnn[k][tt]).mean(1).mean(1) for tt in (1,2)}
    pfmX1_shuff = {tt:np.array(performanceX1_full_shuff_rnn[k][tt]).mean(1).mean(1) for tt in (1,2)}
    pfmX2_shuff = {tt:np.array(performanceX2_full_shuff_rnn[k][tt]).mean(1).mean(1) for tt in (1,2)}

    codeMorph_full_rnn[k] = np.array([np.mean((f_decoding.code_morphing(pfmX1[2][n][ld1x,ld1x], pfmX1[2][n][ld1x,ld2x]), 
                                               f_decoding.code_morphing(pfmX1[2][n][ld2x,ld2x], pfmX1[2][n][ld2x,ld1x]))) for n in range(100)])
    
    codeMorph_full_shuff_rnn[k] = np.array([np.mean((f_decoding.code_morphing(pfmX1_shuff[2][n][ld1x,ld1x], pfmX1_shuff[2][n][ld1x,ld2x]), 
                                                       f_decoding.code_morphing(pfmX1_shuff[2][n][ld2x,ld2x], pfmX1_shuff[2][n][ld2x,ld1x]))) for n in range(100)])
 
   
codeMorph_full_data = {}
codeMorph_full_shuff_data = {}

for k in ('dlpfc','fef'):
    ttypes = ('Retarget','Distractor')
    pfmX1 = {tt:np.array(performanceX1_full_data[ttypes[tt-1]][k]).mean(1) for tt in (1,2)}
    pfmX2 = {tt:np.array(performanceX2_full_data[ttypes[tt-1]][k]).mean(1) for tt in (1,2)}
    pfmX1_shuff = {tt:np.array(performanceX1_full_shuff_data[ttypes[tt-1]][k]).mean(1) for tt in (1,2)}
    pfmX2_shuff = {tt:np.array(performanceX2_full_shuff_data[ttypes[tt-1]][k]).mean(1) for tt in (1,2)}

    codeMorph_full_data[k] = np.array([np.mean((f_decoding.code_morphing(pfmX1[2][n][ld1x,ld1x], pfmX1[2][n][ld1x,ld2x]), 
                                               f_decoding.code_morphing(pfmX1[2][n][ld2x,ld2x], pfmX1[2][n][ld2x,ld1x]))) for n in range(100)])
    
    codeMorph_full_shuff_data[k] = np.array([np.mean((f_decoding.code_morphing(pfmX1_shuff[2][n][ld1x,ld1x], pfmX1_shuff[2][n][ld1x,ld2x]), 
                                                       f_decoding.code_morphing(pfmX1_shuff[2][n][ld2x,ld2x], pfmX1_shuff[2][n][ld2x,ld1x]))) for n in range(100)])


lineh = np.arange(0.3,0.7,0.001)
linev = np.arange(1.16,1.165,0.0001)

fig, axes = plt.subplots(1,1, sharey=True, figsize=(3,3), dpi=300)

# full space
ax = axes#.flatten()[0]

ax.boxplot([codeMorph_full_rnn['ed2']], positions=[0.3], patch_artist=True, widths = 0.2, boxprops=boxprops1, flierprops=flierprops1, 
                  meanprops=meanpointprops1, medianprops=medianprops, capprops = capprops1, whiskerprops = whiskerprops1, meanline=False, showmeans=True)
ax.boxplot([codeMorph_full_shuff_rnn['ed2']], positions=[0.7], patch_artist=True, widths = 0.2, boxprops=boxprops0, flierprops=flierprops0, 
                  meanprops=meanpointprops0, medianprops=medianprops, capprops = capprops0, whiskerprops = whiskerprops0, meanline=False, showmeans=True)

ax.boxplot([codeMorph_full_rnn['ed12']], positions=[1.3], patch_artist=True, widths = 0.2, boxprops=boxprops2, flierprops=flierprops2, 
                  meanprops=meanpointprops2, medianprops=medianprops, capprops = capprops2, whiskerprops = whiskerprops2, meanline=False, showmeans=True)
ax.boxplot([codeMorph_full_shuff_rnn['ed12']], positions=[1.7], patch_artist=True, widths = 0.2, boxprops=boxprops0, flierprops=flierprops0, 
                  meanprops=meanpointprops0, medianprops=medianprops, capprops = capprops0, whiskerprops = whiskerprops0, meanline=False, showmeans=True)


ax.boxplot([codeMorph_full_data['dlpfc']], positions=[2.3], patch_artist=True, widths = 0.2, boxprops=boxprops3, flierprops=flierprops3, 
                  meanprops=meanpointprops3, medianprops=medianprops, capprops = capprops3, whiskerprops = whiskerprops3, meanline=False, showmeans=True)
ax.boxplot([codeMorph_full_shuff_data['dlpfc']], positions=[2.7], patch_artist=True, widths = 0.2, boxprops=boxprops0, flierprops=flierprops0, 
                  meanprops=meanpointprops0, medianprops=medianprops, capprops = capprops0, whiskerprops = whiskerprops0, meanline=False, showmeans=True)

ax.boxplot([codeMorph_full_data['fef']], positions=[3.3], patch_artist=True, widths = 0.2, boxprops=boxprops4, flierprops=flierprops4, 
                  meanprops=meanpointprops4, medianprops=medianprops, capprops = capprops4, whiskerprops = whiskerprops4, meanline=False, showmeans=True)
ax.boxplot([codeMorph_full_shuff_data['fef']], positions=[3.7], patch_artist=True, widths = 0.2, boxprops=boxprops0, flierprops=flierprops0, 
                  meanprops=meanpointprops0, medianprops=medianprops, capprops = capprops0, whiskerprops = whiskerprops0, meanline=False, showmeans=True)



p1 = f_stats.permutation_pCI(codeMorph_full_rnn['ed2'], codeMorph_full_shuff_rnn['ed2'], tail = 'greater', alpha=5)
ax.text(0.5,5.25, f'{f_plotting.sig_marker(p1,ns_note=True)}',horizontalalignment='center', verticalalignment='bottom', fontsize=10)

p2 = f_stats.permutation_pCI(codeMorph_full_rnn['ed12'], codeMorph_full_shuff_rnn['ed12'], tail = 'greater', alpha=5)
ax.text(1.5,5.25, f'{f_plotting.sig_marker(p2,ns_note=True)}',horizontalalignment='center', verticalalignment='bottom', fontsize=10)


p3 = f_stats.permutation_pCI(codeMorph_full_data['dlpfc'], codeMorph_full_shuff_data['dlpfc'], tail = 'greater', alpha=5)
ax.text(2.5,5.25, f'{f_plotting.sig_marker(p3,ns_note=True)}',horizontalalignment='center', verticalalignment='bottom', fontsize=10)

p4 = f_stats.permutation_pCI(codeMorph_full_data['fef'], codeMorph_full_shuff_data['fef'], tail = 'greater', alpha=5)
ax.text(3.5,5.25, f'{f_plotting.sig_marker(p4,ns_note=True)}',horizontalalignment='center', verticalalignment='bottom', fontsize=10)


ax.set_ylabel('D1-D1/D1-D2 (& vice versa)', labelpad = 3, fontsize = 10)


xlims = ax.get_xlim()
lineThresholds = np.arange(xlims[0],xlims[1]+0.01,0.01)
ax.plot(lineThresholds, np.full_like(lineThresholds,1), 'k--', alpha = 0.5)

ax.set_xticks([0.5,1.5,2.5,3.5],['R@R','R&U', 'LPFC','FEF'], rotation=0)
ax.set_xlabel('Strategy/Region', labelpad = 5, fontsize = 12)
ax.set_ylim(top=6, bottom=0)
ax.set_title('Code Morphing, Full Space', fontsize = 12, pad=10)
plt.show()

# print stats
print(f"R@R: M(SD) = {codeMorph_full_rnn['ed2'].mean():.3f}({codeMorph_full_rnn['ed2'].std():.3f}), p = {p1:.3f};")
print(f"R&U: M(SD) = {codeMorph_full_rnn['ed12'].mean():.3f}({codeMorph_full_rnn['ed12'].std():.3f}), p = {p2:.3f};")
print(f"LPFC: M(SD) = {codeMorph_full_data['dlpfc'].mean():.3f}({codeMorph_full_data['dlpfc'].std():.3f}), p = {p3:.3f};")
print(f"FEF: M(SD) = {codeMorph_full_data['fef'].mean():.3f}({codeMorph_full_data['fef'].std():.3f}), p = {p4:.3f};")
print('\n')

#%% evaluate and plot full space distractor information quantification

##########################################
# Distractor Information during Early D2 #
##########################################

ed2 = np.arange(1600,2100+bins,bins)
ed2x = [tbins.tolist().index(t) for t in ed2]

lineh = np.arange(0.5,1.5,0.001)
linev = np.arange(0.71,0.72,0.0001)

# get diagonal decodability
pfm22_full_rnn = {k:np.array(performanceX2_full_rnn[k][2]).mean(1).mean(1).diagonal(offset=0,axis1=1,axis2=2)[:,ed2x].mean(-1) for k in ('ed2', 'ed12')}
pfm22_full_shuff_rnn = {k:np.array(performanceX2_full_shuff_rnn[k][2]).mean(1).mean(1).diagonal(offset=0,axis1=1,axis2=2)[:,ed2x].mean(-1) for k in ('ed2', 'ed12')}

pfm22_full_data = {k:np.array(performanceX2_full_data['Distractor'][k]).mean(1).diagonal(offset=0,axis1=1,axis2=2)[:,ed2x].mean(-1) for k in ('dlpfc','fef')}
pfm22_full_shuff_data = {k:np.array(performanceX2_full_shuff_data['Distractor'][k]).mean(1).diagonal(offset=0,axis1=1,axis2=2)[:,ed2x].mean(-1) for k in ('dlpfc','fef')}

fig, axes = plt.subplots(1,1, sharey=True, figsize=(3,3), dpi=300)

# full space
ax = axes#.flatten()[0]

ax.boxplot([pfm22_full_rnn['ed2']], positions=[0.3], patch_artist=True, widths = 0.2, boxprops=boxprops1, flierprops=flierprops1, 
                  meanprops=meanpointprops1, medianprops=medianprops, capprops = capprops1, whiskerprops = whiskerprops1, meanline=False, showmeans=True)
ax.boxplot([pfm22_full_rnn['ed12']], positions=[1.3], patch_artist=True, widths = 0.2, boxprops=boxprops2, flierprops=flierprops2, 
                  meanprops=meanpointprops2, medianprops=medianprops, capprops = capprops2, whiskerprops = whiskerprops2, meanline=False, showmeans=True)
ax.boxplot([pfm22_full_data['dlpfc']], positions=[2.3], patch_artist=True, widths = 0.2, boxprops=boxprops3, flierprops=flierprops3, 
                  meanprops=meanpointprops3, medianprops=medianprops, capprops = capprops3, whiskerprops = whiskerprops3, meanline=False, showmeans=True)
ax.boxplot([pfm22_full_data['fef']], positions=[3.3], patch_artist=True, widths = 0.2, boxprops=boxprops4, flierprops=flierprops4, 
                  meanprops=meanpointprops4, medianprops=medianprops, capprops = capprops4, whiskerprops = whiskerprops4, meanline=False, showmeans=True)


ax.boxplot([pfm22_full_shuff_rnn['ed2']], positions=[0.7], patch_artist=True, widths = 0.2, boxprops=boxprops0, flierprops=flierprops0, 
                  meanprops=meanpointprops0, medianprops=medianprops, capprops = capprops0, whiskerprops = whiskerprops0, meanline=False, showmeans=True)
ax.boxplot([pfm22_full_shuff_rnn['ed12']], positions=[1.7], patch_artist=True, widths = 0.2, boxprops=boxprops0, flierprops=flierprops0, 
                  meanprops=meanpointprops0, medianprops=medianprops, capprops = capprops0, whiskerprops = whiskerprops0, meanline=False, showmeans=True)
ax.boxplot([pfm22_full_shuff_data['dlpfc']], positions=[2.7], patch_artist=True, widths = 0.2, boxprops=boxprops0, flierprops=flierprops0, 
                  meanprops=meanpointprops0, medianprops=medianprops, capprops = capprops0, whiskerprops = whiskerprops0, meanline=False, showmeans=True)
ax.boxplot([pfm22_full_shuff_data['fef']], positions=[3.7], patch_artist=True, widths = 0.2, boxprops=boxprops0, flierprops=flierprops0, 
                  meanprops=meanpointprops0, medianprops=medianprops, capprops = capprops0, whiskerprops = whiskerprops0, meanline=False, showmeans=True)


p1 = f_stats.permutation_pCI(pfm22_full_rnn['ed2'], pfm22_full_shuff_rnn['ed2'], tail='greater', alpha=5)
ax.text(0.5,0.8, f'{f_plotting.sig_marker(p1,ns_note=True)}',horizontalalignment='center', fontsize=12)
print(f"R@R: Mean(SD) = {pfm22_full_rnn['ed2'].mean():.3f}({pfm22_full_rnn['ed2'].std():.3f}), p = {p1:.3f}")

p2 = f_stats.permutation_pCI(pfm22_full_rnn['ed12'], pfm22_full_shuff_rnn['ed12'], tail='greater', alpha=5)
ax.text(1.5,0.8, f'{f_plotting.sig_marker(p2,ns_note=True)}',horizontalalignment='center', fontsize=12)
print(f"R&U: Mean(SD) = {pfm22_full_rnn['ed12'].mean():.3f}({pfm22_full_rnn['ed12'].std():.3f}), p = {p2:.3f}")

p3 = f_stats.permutation_pCI(pfm22_full_data['dlpfc'], pfm22_full_shuff_data['dlpfc'], tail='greater', alpha=5)
ax.text(2.5,0.8, f'{f_plotting.sig_marker(p3,ns_note=True)}',horizontalalignment='center', fontsize=12)
print(f"LPFC: Mean(SD) = {pfm22_full_data['dlpfc'].mean():.3f}({pfm22_full_data['dlpfc'].std():.3f}), p = {p3:.3f}")

p4 = f_stats.permutation_pCI(pfm22_full_data['fef'], pfm22_full_shuff_data['fef'], tail='greater', alpha=5)
ax.text(3.5,0.8, f'{f_plotting.sig_marker(p4,ns_note=True)}',horizontalalignment='center', fontsize=12)
print(f"FEF: Mean(SD) = {pfm22_full_data['fef'].mean():.3f}({pfm22_full_data['fef'].std():.3f}), p = {p4:.3f}")


xlims = ax.get_xlim()
lineThresholds = np.arange(xlims[0],xlims[1]+0.01,0.01)
ax.set_xticks([0.5,1.5,2.5,3.5],['R@R','R&U','LPFC','FEF'], rotation=0)
ax.set_xlabel('Strategy/Region', labelpad = 5, fontsize = 12)
ax.set_ylabel('Decodability', labelpad = 3, fontsize = 12)
ax.set_ylim(top=0.9, bottom=0.1)
plt.suptitle('Distractor Information, Full Space, ED2', fontsize = 12, y=1.0)
plt.show()

#%%



#%%

#################
# Readout Space #
#################

#%% load precomputed readout space decodability
performanceX1_readout_data = np.load(f'{data_path}/' + 'performanceX1_readout_data.npy', allow_pickle=True).item()
performanceX2_readout_data = np.load(f'{data_path}/' + 'performanceX2_readout_data.npy', allow_pickle=True).item()
performanceX1_readout_shuff_data = np.load(f'{data_path}/' + 'performanceX1_readout_shuff_data.npy', allow_pickle=True).item()
performanceX2_readout_shuff_data = np.load(f'{data_path}/' + 'performanceX2_readout_shuff_data.npy', allow_pickle=True).item()

performanceX1_readout_rnn = np.load(f'{data_path}/' + 'performanceX1_readout_rnn.npy', allow_pickle=True).item()
performanceX2_readout_rnn = np.load(f'{data_path}/' + 'performanceX2_readout_rnn.npy', allow_pickle=True).item()
performanceX1_readout_shuff_rnn = np.load(f'{data_path}/' + 'performanceX1_readout_shuff_rnn.npy', allow_pickle=True).item()
performanceX2_readout_shuff_rnn = np.load(f'{data_path}/' + 'performanceX2_readout_shuff_rnn.npy', allow_pickle=True).item()

#%% [Figure S4A] plot readout decodability for bio data
for region in ('dlpfc','fef'):

    infoLabel = 'Accuracy'
    fig = plt.figure(figsize=(28, 24), dpi=100)
    
    for tt in ttypes:
        
        condT = 'Retarget' if tt == 1 else 'Distraction'
        h0 = 1/len(locs)
        
        pfm1, pfm2 = performanceX1_readout_data[region][tt].mean(1), performanceX2_readout_data[region][tt].mean(1)
        pfm1_shuff, pfm2_shuff = performanceX1_readout_shuff_data[region][tt].mean(1), performanceX2_readout_shuff_data[region][tt].mean(1)

        pPerms_decode1_3d = np.ones((len(tbins), len(tbins)))
        pPerms_decode2_3d = np.ones((len(tbins), len(tbins)))
        
        for t in range(len(tbins)):
            for t_ in range(len(tbins)):
                #pPerms_decode1_3d[t, t_] = f_stats.permutation_p(pfm1.mean(0)[t,t_], pfm1_shuff.mean(1)[:,t,t_], tail='greater')
                #pPerms_decode2_3d[t, t_] = f_stats.permutation_p(pfm2.mean(0)[t,t_], pfm2_shuff.mean(1)[:,t,t_], tail='greater')
                pPerms_decode1_3d[t, t_] = f_stats.permutation_pCI(pfm1[:,t,t_], pfm1_shuff[:,t,t_], tail='greater', alpha=5)
                pPerms_decode2_3d[t, t_] = f_stats.permutation_pCI(pfm2[:,t,t_], pfm2_shuff[:,t,t_], tail='greater', alpha=5)
                
        
        vmax = 0.6 if region == 'dlpfc' else 0.8
        regionLabel = 'LPFC' if region == 'dlpfc' else 'PAC'
        
        # item1
        plt.subplot(2,2,tt)
        ax = plt.gca()
        sns.heatmap(pd.DataFrame(pfm1.mean(0), index=tbins,columns=tbins), cmap = 'magma', vmin = 0.25, vmax = vmax, ax = ax)#, vmax = 0.6, xticklabels = 100, yticklabels = 100, vmin = cbar_min, vmax = cbar_max
        
        #from scipy import ndimage
        smooth_scale = 10
        z = ndimage.zoom(pPerms_decode1_3d, smooth_scale)
        ax.contour(np.linspace(0, len(tbins), len(tbins) * smooth_scale),
                 np.linspace(0, len(tbins), len(tbins) * smooth_scale),
                  z, levels=([0.05]), colors='white', alpha = 1, linewidths=3)
        
        ax.invert_yaxis()
        
        
        # event lines
        for i in [0, 1300, 2600]:
            ax.plot(tbins, np.full_like(tbins,list(tbins).index(i)), 'w-.', linewidth=4)
            ax.plot(np.full_like(tbins,list(tbins).index(i)), tbins, 'w-.', linewidth=4)
        
        ax.set_xticks([list(tbins).index(i) for i in [0, 1300, 2600]])
        ax.set_xticklabels(['S1', 'S2', 'Go Cue'], rotation=0, fontsize = 20)
        ax.set_xlabel('Test Timebin (ms)', labelpad = 10, fontsize = 25)
        ax.set_yticks([list(tbins).index(i) for i in [0, 1300, 2600]])
        ax.set_yticklabels(['S1', 'S2', 'Go Cue'], fontsize = 20, rotation=90)
        ax.set_ylabel('Train Timebin (ms)', labelpad = 10, fontsize = 25)
        
        cbar = ax.collections[0].colorbar
        # here set the labelsize by 20
        cbar.ax.tick_params(labelsize=20)
        
        ax.set_title(f'{condT}, Item1', fontsize = 30, pad = 20)
        
        # item2
        plt.subplot(2,2,tt+2)
        ax = plt.gca()
        sns.heatmap(pd.DataFrame(pfm2.mean(0), index=tbins,columns=tbins), cmap = 'magma', vmin = 0.25, vmax = vmax, ax = ax)#, vmax = 0.6, xticklabels = 100, yticklabels = 100, vmin = cbar_min, vmax = cbar_max
        
        #from scipy import ndimage
        smooth_scale = 10
        z = ndimage.zoom(pPerms_decode2_3d, smooth_scale)
        ax.contour(np.linspace(0, len(tbins), len(tbins) * smooth_scale),
                 np.linspace(0, len(tbins), len(tbins) * smooth_scale),
                  z, levels=([0.05]), colors='white', alpha = 1, linewidths=3)
        
        ax.invert_yaxis()
        
        
        # event lines
        for i in [0, 1300, 2600]:
            ax.plot(tbins, np.full_like(tbins,list(tbins).index(i)), 'w-.', linewidth=4)
            ax.plot(np.full_like(tbins,list(tbins).index(i)), tbins, 'w-.', linewidth=4)
        
        ax.set_xticks([list(tbins).index(i) for i in [0, 1300, 2600]])
        ax.set_xticklabels(['S1', 'S2', 'Go Cue'], rotation=0, fontsize = 20)
        ax.set_xlabel('Test Timebin (ms)', labelpad = 10, fontsize = 25)
        ax.set_yticks([list(tbins).index(i) for i in [0, 1300, 2600]])
        ax.set_yticklabels(['S1', 'S2', 'Go Cue'], fontsize = 20, rotation=90)
        ax.set_ylabel('Train Timebin (ms)', labelpad = 10, fontsize = 25)
        
        cbar = ax.collections[0].colorbar
        # here set the labelsize by 20
        cbar.ax.tick_params(labelsize=20)
        
        ax.set_title(f'{condT}, Item2', fontsize = 30, pad = 20)
    
    plt.tight_layout()
    plt.subplots_adjust(top = 0.95)
    plt.suptitle(f'{regionLabel}, Readout Subspace', fontsize = 35, y=1) #, Arti_Noise = {arti_noise_level}
    plt.show()
    
#%% [Figure S4A] plot readout decodability for RNNs
for nfi, kfi in enumerate(fitIntervals):
    print(kfi)
    info_C1X,info_C2X = performanceX1_readout_rnn[kfi], performanceX2_readout_rnn[kfi] 
    info_C1X_shuff,info_C2X_shuff = performanceX1_readout_shuff_rnn[kfi], performanceX2_readout_shuff_rnn[kfi] 


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
    

    infoLabel = 'Accuracy'
    
    if len(decode_projC1X_3d[1])>0:
        fig = plt.figure(figsize=(28, 24), dpi=100)
        
        for tt in ttypes:
            
            condT = 'Retarget' if tt == 1 else 'Distraction'
            h0 = 1/len(locs)
            
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
#%% [Figure S4B] evaluate and plot readout subspace code stability

##################
# code stability #
##################

stabRatioD1_readout_rnn, stabRatioD2_readout_rnn = {}, {}
stabRatioD1_readout_shuff_rnn, stabRatioD2_readout_shuff_rnn = {}, {}

for k in ('ed2','ed12'):
    pfmX1 = {tt:np.array([performanceX1_readout_rnn[k][n][tt] for n in range(100)]).mean(1).mean(1) for tt in (1,2)}
    pfmX2 = {tt:np.array([performanceX2_readout_rnn[k][n][tt] for n in range(100)]).mean(1).mean(1) for tt in (1,2)}
    pfmX1_shuff = {tt:np.array([performanceX1_readout_shuff_rnn[k][n][tt] for n in range(100)]).mean(1).mean(1) for tt in (1,2)}
    pfmX2_shuff = {tt:np.array([performanceX2_readout_shuff_rnn[k][n][tt] for n in range(100)]).mean(1).mean(1) for tt in (1,2)}

    stabRatioD1_readout_rnn[k] = np.array([np.mean((f_decoding.stability_ratio(pfmX1[1][n][ld1x,:][:,ld1x]), 
                                                 f_decoding.stability_ratio(pfmX1[2][n][ld1x,:][:,ld1x]))) for n in range(100)])
    stabRatioD2_readout_rnn[k] = np.array([np.mean((f_decoding.stability_ratio(pfmX2[1][n][ld2x,:][:,ld2x]), 
                                                 f_decoding.stability_ratio(pfmX1[2][n][ld2x,:][:,ld2x]))) for n in range(100)])
    
    stabRatioD1_readout_shuff_rnn[k] = np.array([np.mean((f_decoding.stability_ratio(pfmX1_shuff[1][n][ld1x,:][:,ld1x]), 
                                                       f_decoding.stability_ratio(pfmX1_shuff[2][n][ld1x,:][:,ld1x]))) for n in range(100)])
    stabRatioD2_readout_shuff_rnn[k] = np.array([np.mean((f_decoding.stability_ratio(pfmX2_shuff[1][n][ld2x,:][:,ld2x]), 
                                                       f_decoding.stability_ratio(pfmX1_shuff[2][n][ld2x,:][:,ld2x]))) for n in range(100)])


stabRatioD1_readout_data, stabRatioD2_readout_data = {}, {}
stabRatioD1_readout_shuff_data, stabRatioD2_readout_shuff_data = {}, {}

for k in ('dlpfc','fef'):
    ttypes = ('Retarget','Distractor')
    pfmX1 = {tt:np.array(performanceX1_readout_data[k][tt]).mean(1) for tt in (1,2)}
    pfmX2 = {tt:np.array(performanceX2_readout_data[k][tt]).mean(1) for tt in (1,2)}
    pfmX1_shuff = {tt:np.array(performanceX1_readout_shuff_data[k][tt]).mean(1) for tt in (1,2)}
    pfmX2_shuff = {tt:np.array(performanceX2_readout_shuff_data[k][tt]).mean(1) for tt in (1,2)}

    stabRatioD1_readout_data[k] = np.array([np.mean((f_decoding.stability_ratio(pfmX1[1][n][ld1x,:][:,ld1x]), 
                                                 f_decoding.stability_ratio(pfmX1[2][n][ld1x,:][:,ld1x]))) for n in range(100)])
    stabRatioD2_readout_data[k] = np.array([np.mean((f_decoding.stability_ratio(pfmX2[1][n][ld2x,:][:,ld2x]), 
                                                 f_decoding.stability_ratio(pfmX1[2][n][ld2x,:][:,ld2x]))) for n in range(100)])
    
    stabRatioD1_readout_shuff_data[k] = np.array([np.mean((f_decoding.stability_ratio(pfmX1_shuff[1][n][ld1x,:][:,ld1x]), 
                                                       f_decoding.stability_ratio(pfmX1_shuff[2][n][ld1x,:][:,ld1x]))) for n in range(100)])
    stabRatioD2_readout_shuff_data[k] = np.array([np.mean((f_decoding.stability_ratio(pfmX2_shuff[1][n][ld2x,:][:,ld2x]), 
                                                       f_decoding.stability_ratio(pfmX1_shuff[2][n][ld2x,:][:,ld2x]))) for n in range(100)])


lineh = np.arange(0.3,0.7,0.001)
linev = np.arange(1.16,1.165,0.0001)

fig, axes = plt.subplots(1,1, sharey=True, figsize=(3,3), dpi=300)

# readout subspace
ax = axes#.flatten()[0]

ax.boxplot([stabRatioD1_readout_rnn['ed2']], positions=[0.3], patch_artist=True, widths = 0.2, boxprops=boxprops1, flierprops=flierprops1, 
                  meanprops=meanpointprops1, medianprops=medianprops, capprops = capprops1, whiskerprops = whiskerprops1, meanline=False, showmeans=True)
ax.boxplot([stabRatioD2_readout_rnn['ed2']], positions=[0.7], patch_artist=True, widths = 0.2, boxprops=dict(boxprops1,facecolor=color1_), flierprops=dict(flierprops1,markeredgecolor=color1_, markerfacecolor=color1_), 
                  meanprops=dict(meanpointprops1,markeredgecolor=color1_), medianprops=medianprops, capprops = dict(capprops1,color=color1_), whiskerprops = dict(whiskerprops1,color=color1_), meanline=False, showmeans=True)

ax.boxplot([stabRatioD1_readout_rnn['ed12']], positions=[1.3], patch_artist=True, widths = 0.2, boxprops=boxprops2, flierprops=flierprops2, 
                  meanprops=meanpointprops2, medianprops=medianprops, capprops = capprops2, whiskerprops = whiskerprops2, meanline=False, showmeans=True)
ax.boxplot([stabRatioD2_readout_rnn['ed12']], positions=[1.7], patch_artist=True, widths = 0.2, boxprops=dict(boxprops2,facecolor=color2_), flierprops=dict(flierprops2,markeredgecolor=color2_, markerfacecolor=color2_), 
                  meanprops=dict(meanpointprops2,markeredgecolor=color2_), medianprops=medianprops, capprops = dict(capprops2,color=color2_), whiskerprops = dict(whiskerprops2,color=color2_), meanline=False, showmeans=True)


ax.boxplot([stabRatioD1_readout_data['dlpfc']], positions=[2.3], patch_artist=True, widths = 0.2, boxprops=boxprops3, flierprops=flierprops3, 
                  meanprops=meanpointprops3, medianprops=medianprops, capprops = capprops3, whiskerprops = whiskerprops3, meanline=False, showmeans=True)
ax.boxplot([stabRatioD2_readout_data['dlpfc']], positions=[2.7], patch_artist=True, widths = 0.2, boxprops=dict(boxprops3,facecolor=color3_), flierprops=dict(flierprops3,markeredgecolor=color3_, markerfacecolor=color3_), 
                  meanprops=dict(meanpointprops3,markeredgecolor=color3_), medianprops=medianprops, capprops = dict(capprops3,color=color3_), whiskerprops = dict(whiskerprops3,color=color3_), meanline=False, showmeans=True)

ax.boxplot([stabRatioD1_readout_data['fef']], positions=[3.3], patch_artist=True, widths = 0.2, boxprops=boxprops4, flierprops=flierprops4, 
                  meanprops=meanpointprops4, medianprops=medianprops, capprops = capprops4, whiskerprops = whiskerprops4, meanline=False, showmeans=True)
ax.boxplot([stabRatioD2_readout_data['fef']], positions=[3.7], patch_artist=True, widths = 0.2, boxprops=dict(boxprops4,facecolor=color4_), flierprops=dict(flierprops4,markeredgecolor=color4_, markerfacecolor=color4_), 
                  meanprops=dict(meanpointprops4,markeredgecolor=color4_), medianprops=medianprops, capprops = dict(capprops4,color=color4_), whiskerprops = dict(whiskerprops4,color=color4_), meanline=False, showmeans=True)



p1 = f_stats.permutation_pCI(stabRatioD1_readout_rnn['ed2'], stabRatioD1_readout_shuff_rnn['ed2'], tail = 'smaller',alpha=5)
ax.text(0.3,0.425, f'{f_plotting.sig_marker(p1,ns_note=False)}',horizontalalignment='center', verticalalignment='bottom', fontsize=10)
p2 = f_stats.permutation_pCI(stabRatioD2_readout_rnn['ed2'], stabRatioD2_readout_shuff_rnn['ed2'], tail = 'smaller',alpha=5)
ax.text(0.7,0.425, f'{f_plotting.sig_marker(p2,ns_note=False)}',horizontalalignment='center', verticalalignment='bottom', fontsize=10)

p3 = f_stats.permutation_pCI(stabRatioD1_readout_rnn['ed12'], stabRatioD1_readout_shuff_rnn['ed12'], tail = 'smaller',alpha=5)
ax.text(1.3,0.425, f'{f_plotting.sig_marker(p3,ns_note=False)}',horizontalalignment='center', verticalalignment='bottom', fontsize=10)
p4 = f_stats.permutation_pCI(stabRatioD2_readout_rnn['ed12'], stabRatioD2_readout_shuff_rnn['ed12'], tail = 'smaller',alpha=5)
ax.text(1.7,0.425, f'{f_plotting.sig_marker(p4,ns_note=False)}',horizontalalignment='center', verticalalignment='bottom', fontsize=10)

p12 = f_stats.permutation_p_diff(stabRatioD1_readout_rnn['ed2'], stabRatioD2_readout_rnn['ed2'])
ax.plot(lineh, np.full_like(lineh, 1.165), 'k-')
ax.plot(np.full_like(linev, 0.3), linev, 'k-')
ax.plot(np.full_like(linev, 0.7), linev, 'k-')
ax.text(0.5,1.175, f'{f_plotting.sig_marker(p12,ns_note=True)}',horizontalalignment='center', verticalalignment='bottom', fontsize=10)

p34 = f_stats.permutation_p_diff(stabRatioD1_readout_rnn['ed12'], stabRatioD2_readout_rnn['ed12'])
ax.plot(lineh+1, np.full_like(lineh, 1.165), 'k-')
ax.plot(np.full_like(linev, 0.3)+1, linev, 'k-')
ax.plot(np.full_like(linev, 0.7)+1, linev, 'k-')
ax.text(1.5,1.175, f'{f_plotting.sig_marker(p34,ns_note=True)}',horizontalalignment='center', verticalalignment='bottom', fontsize=10)


p5 = f_stats.permutation_pCI(stabRatioD1_readout_data['dlpfc'], stabRatioD1_readout_shuff_data['dlpfc'], tail = 'smaller',alpha=5)
ax.text(2.3,0.425, f'{f_plotting.sig_marker(p5,ns_note=False)}',horizontalalignment='center', verticalalignment='bottom', fontsize=10)
p6 = f_stats.permutation_pCI(stabRatioD2_readout_data['dlpfc'], stabRatioD2_readout_shuff_data['dlpfc'], tail = 'smaller',alpha=5)
ax.text(2.7,0.425, f'{f_plotting.sig_marker(p6,ns_note=False)}',horizontalalignment='center', verticalalignment='bottom', fontsize=10)

p7 = f_stats.permutation_pCI(stabRatioD1_readout_data['fef'], stabRatioD1_readout_shuff_data['fef'], tail = 'smaller',alpha=5)
ax.text(3.3,0.425, f'{f_plotting.sig_marker(p7,ns_note=False)}',horizontalalignment='center', verticalalignment='bottom', fontsize=10)
p8 = f_stats.permutation_pCI(stabRatioD2_readout_data['fef'], stabRatioD2_readout_shuff_data['fef'], tail = 'smaller',alpha=5)
ax.text(3.7,0.425, f'{f_plotting.sig_marker(p8,ns_note=False)}',horizontalalignment='center', verticalalignment='bottom', fontsize=10)

p56 = f_stats.permutation_p_diff(stabRatioD1_readout_data['dlpfc'], stabRatioD2_readout_data['dlpfc'])
ax.plot(lineh+2, np.full_like(lineh, 1.165), 'k-')
ax.plot(np.full_like(linev, 0.3)+2, linev, 'k-')
ax.plot(np.full_like(linev, 0.7)+2, linev, 'k-')
ax.text(2.5,1.175, f'{f_plotting.sig_marker(p56,ns_note=True)}',horizontalalignment='center', verticalalignment='bottom', fontsize=10)

p78 = f_stats.permutation_p_diff(stabRatioD1_readout_data['fef'], stabRatioD2_readout_data['fef'])
ax.plot(lineh+3, np.full_like(lineh, 1.165), 'k-')
ax.plot(np.full_like(linev, 0.3)+3, linev, 'k-')
ax.plot(np.full_like(linev, 0.7)+3, linev, 'k-')
ax.text(3.5,1.175, f'{f_plotting.sig_marker(p78,ns_note=True)}',horizontalalignment='center', verticalalignment='bottom', fontsize=10)


ax.set_ylabel('Off-/On-Diagonal', labelpad = 3, fontsize = 10)


xlims = ax.get_xlim()
lineThresholds = np.arange(xlims[0],xlims[1]+0.01,0.01)
ax.plot(lineThresholds, np.full_like(lineThresholds,1), 'k--', alpha = 0.5)

# draw temporary red and blue lines and use them to create a legend
plt.plot([], c='dimgrey', label='Delay1')
plt.plot([], c='lightgrey', label='Delay2')
plt.legend(bbox_to_anchor=(1.5, 0.6))#loc = 'right',

ax.set_xticks([0.5,1.5,2.5,3.5],['R@R','R&U', 'LPFC','FEF'], rotation=0)
ax.set_xlabel('Strategy/Region', labelpad = 5, fontsize = 12)
ax.set_ylim(top=1.3, bottom=0.4)
ax.set_title('Code Stability, Readout Subspace', fontsize = 12, pad=10)
plt.show()


print(f"R@R, D1: M(SD) = {stabRatioD1_readout_rnn['ed2'].mean():.3f}({stabRatioD1_readout_rnn['ed2'].std():.3f}), p = {p1:.3f};")
print(f"R&U, D1: M(SD) = {stabRatioD1_readout_rnn['ed12'].mean():.3f}({stabRatioD1_readout_rnn['ed12'].std():.3f}), p = {p3:.3f};")
print(f"LPFC, D1: M(SD) = {stabRatioD1_readout_data['dlpfc'].mean():.3f}({stabRatioD1_readout_data['dlpfc'].std():.3f}), p = {p5:.3f};")
print(f"FEF, D1: M(SD) = {stabRatioD1_readout_data['fef'].mean():.3f}({stabRatioD1_readout_data['fef'].std():.3f}), p = {p7:.3f};")
print('\n')
print(f"R@R, D2: M(SD) = {stabRatioD2_readout_rnn['ed2'].mean():.3f}({stabRatioD2_readout_rnn['ed2'].std():.3f}), p = {p2:.3f};")
print(f"R&U, D2: M(SD) = {stabRatioD2_readout_rnn['ed12'].mean():.3f}({stabRatioD2_readout_rnn['ed12'].std():.3f}), p = {p4:.3f};")
print(f"LPFC, D2: M(SD) = {stabRatioD2_readout_data['dlpfc'].mean():.3f}({stabRatioD2_readout_data['dlpfc'].std():.3f}), p = {p6:.3f};")
print(f"FEF, D2: M(SD) = {stabRatioD2_readout_data['fef'].mean():.3f}({stabRatioD2_readout_data['fef'].std():.3f}), p = {p8:.3f};")
print('\n')
print(f"R@R, D1-D2: MD = {(stabRatioD1_readout_rnn['ed2'].mean()-stabRatioD2_readout_rnn['ed2'].mean()):.3f}, p = {p12:.3f}, g = {f_stats.hedges_g(stabRatioD1_readout_rnn['ed2'],stabRatioD2_readout_rnn['ed2']):.3f};")
print(f"R&U, D1-D2: MD = {(stabRatioD1_readout_rnn['ed12'].mean()-stabRatioD2_readout_rnn['ed12'].mean()):.3f}, p = {p34:.3f}, g = {f_stats.hedges_g(stabRatioD1_readout_rnn['ed12'],stabRatioD2_readout_rnn['ed12']):.3f};")
print(f"LPFC, D1-D2: MD = {(stabRatioD1_readout_data['dlpfc'].mean()-stabRatioD2_readout_data['dlpfc'].mean()):.3f}, p = {p56:.3f}, g = {f_stats.hedges_g(stabRatioD1_readout_data['dlpfc'],stabRatioD2_readout_data['dlpfc']):.3f};")
print(f"FEF, D1-D2: MD = {(stabRatioD1_readout_data['fef'].mean()-stabRatioD2_readout_data['fef'].mean()):.3f}, p = {p78:.3f}, g = {f_stats.hedges_g(stabRatioD1_readout_data['fef'],stabRatioD2_readout_data['fef']):.3f};")

#%% [Figure S4C] evaluate and plot readout subspace code morphing

#################
# code morphing #
#################

codeMorph_readout_rnn = {}
codeMorph_readout_shuff_rnn = {}

for k in ('ed2','ed12'):
    pfmX1 = {tt:np.array([performanceX1_readout_rnn[k][n][tt] for n in range(100)]).mean(1).mean(1) for tt in (1,2)}
    pfmX2 = {tt:np.array([performanceX2_readout_rnn[k][n][tt] for n in range(100)]).mean(1).mean(1) for tt in (1,2)}
    pfmX1_shuff = {tt:np.array([performanceX1_readout_shuff_rnn[k][n][tt] for n in range(100)]).mean(1).mean(1) for tt in (1,2)}
    pfmX2_shuff = {tt:np.array([performanceX2_readout_shuff_rnn[k][n][tt] for n in range(100)]).mean(1).mean(1) for tt in (1,2)}

    codeMorph_readout_rnn[k] = np.array([np.mean((f_decoding.code_morphing(pfmX1[2][n][ld1x,ld1x], pfmX1[2][n][ld1x,ld2x]), 
                                               f_decoding.code_morphing(pfmX1[2][n][ld2x,ld2x], pfmX1[2][n][ld2x,ld1x]))) for n in range(100)])
    
    codeMorph_readout_shuff_rnn[k] = np.array([np.mean((f_decoding.code_morphing(pfmX1_shuff[2][n][ld1x,ld1x], pfmX1_shuff[2][n][ld1x,ld2x]), 
                                                       f_decoding.code_morphing(pfmX1_shuff[2][n][ld2x,ld2x], pfmX1_shuff[2][n][ld2x,ld1x]))) for n in range(100)])
 
   
codeMorph_readout_data = {}
codeMorph_readout_shuff_data = {}

for k in ('dlpfc','fef'):
    ttypes = ('Retarget','Distractor')
    pfmX1 = {tt:np.array(performanceX1_readout_data[k][tt]).mean(1) for tt in (1,2)}
    pfmX2 = {tt:np.array(performanceX2_readout_data[k][tt]).mean(1) for tt in (1,2)}
    pfmX1_shuff = {tt:np.array(performanceX1_readout_shuff_data[k][tt]).mean(1) for tt in (1,2)}
    pfmX2_shuff = {tt:np.array(performanceX2_readout_shuff_data[k][tt]).mean(1) for tt in (1,2)}

    codeMorph_readout_data[k] = np.array([np.mean((f_decoding.code_morphing(pfmX1[2][n][ld1x,ld1x], pfmX1[2][n][ld1x,ld2x]), 
                                               f_decoding.code_morphing(pfmX1[2][n][ld2x,ld2x], pfmX1[2][n][ld2x,ld1x]))) for n in range(100)])
    
    codeMorph_readout_shuff_data[k] = np.array([np.mean((f_decoding.code_morphing(pfmX1_shuff[2][n][ld1x,ld1x], pfmX1_shuff[2][n][ld1x,ld2x]), 
                                                       f_decoding.code_morphing(pfmX1_shuff[2][n][ld2x,ld2x], pfmX1_shuff[2][n][ld2x,ld1x]))) for n in range(100)])
 

lineh = np.arange(0.3,0.7,0.001)
linev = np.arange(1.16,1.165,0.0001)

fig, axes = plt.subplots(1,1, sharey=True, figsize=(3,3), dpi=300)

# readout subspace
ax = axes#.flatten()[0]

ax.boxplot([codeMorph_readout_rnn['ed2']], positions=[0.3], patch_artist=True, widths = 0.2, boxprops=boxprops1, flierprops=flierprops1, 
                  meanprops=meanpointprops1, medianprops=medianprops, capprops = capprops1, whiskerprops = whiskerprops1, meanline=False, showmeans=True)
ax.boxplot([codeMorph_readout_shuff_rnn['ed2']], positions=[0.7], patch_artist=True, widths = 0.2, boxprops=boxprops0, flierprops=flierprops0, 
                  meanprops=meanpointprops0, medianprops=medianprops, capprops = capprops0, whiskerprops = whiskerprops0, meanline=False, showmeans=True)

ax.boxplot([codeMorph_readout_rnn['ed12']], positions=[1.3], patch_artist=True, widths = 0.2, boxprops=boxprops2, flierprops=flierprops2, 
                  meanprops=meanpointprops2, medianprops=medianprops, capprops = capprops2, whiskerprops = whiskerprops2, meanline=False, showmeans=True)
ax.boxplot([codeMorph_readout_shuff_rnn['ed12']], positions=[1.7], patch_artist=True, widths = 0.2, boxprops=boxprops0, flierprops=flierprops0, 
                  meanprops=meanpointprops0, medianprops=medianprops, capprops = capprops0, whiskerprops = whiskerprops0, meanline=False, showmeans=True)


ax.boxplot([codeMorph_readout_data['dlpfc']], positions=[2.3], patch_artist=True, widths = 0.2, boxprops=boxprops3, flierprops=flierprops3, 
                  meanprops=meanpointprops3, medianprops=medianprops, capprops = capprops3, whiskerprops = whiskerprops3, meanline=False, showmeans=True)
ax.boxplot([codeMorph_readout_shuff_data['dlpfc']], positions=[2.7], patch_artist=True, widths = 0.2, boxprops=boxprops0, flierprops=flierprops0, 
                  meanprops=meanpointprops0, medianprops=medianprops, capprops = capprops0, whiskerprops = whiskerprops0, meanline=False, showmeans=True)

ax.boxplot([codeMorph_readout_data['fef']], positions=[3.3], patch_artist=True, widths = 0.2, boxprops=boxprops4, flierprops=flierprops4, 
                  meanprops=meanpointprops4, medianprops=medianprops, capprops = capprops4, whiskerprops = whiskerprops4, meanline=False, showmeans=True)
ax.boxplot([codeMorph_readout_shuff_data['fef']], positions=[3.7], patch_artist=True, widths = 0.2, boxprops=boxprops0, flierprops=flierprops0, 
                  meanprops=meanpointprops0, medianprops=medianprops, capprops = capprops0, whiskerprops = whiskerprops0, meanline=False, showmeans=True)



p1 = f_stats.permutation_pCI(codeMorph_readout_rnn['ed2'], codeMorph_readout_shuff_rnn['ed2'], tail = 'greater', alpha=5)
ax.text(0.5,5.25, f'{f_plotting.sig_marker(p1,ns_note=True)}',horizontalalignment='center', verticalalignment='bottom', fontsize=10)

p2 = f_stats.permutation_pCI(codeMorph_readout_rnn['ed12'], codeMorph_readout_shuff_rnn['ed12'], tail = 'greater', alpha=5)
ax.text(1.5,5.25, f'{f_plotting.sig_marker(p2,ns_note=True)}',horizontalalignment='center', verticalalignment='bottom', fontsize=10)


p3 = f_stats.permutation_pCI(codeMorph_readout_data['dlpfc'], codeMorph_readout_shuff_data['dlpfc'], tail = 'greater', alpha=5)
ax.text(2.5,5.25, f'{f_plotting.sig_marker(p3,ns_note=True)}',horizontalalignment='center', verticalalignment='bottom', fontsize=10)

p4 = f_stats.permutation_pCI(codeMorph_readout_data['fef'], codeMorph_readout_shuff_data['fef'], tail = 'greater', alpha=5)
ax.text(3.5,5.25, f'{f_plotting.sig_marker(p4,ns_note=True)}',horizontalalignment='center', verticalalignment='bottom', fontsize=10)


ax.set_ylabel('D1-D1/D1-D2 (& vice versa)', labelpad = 3, fontsize = 10)


xlims = ax.get_xlim()
lineThresholds = np.arange(xlims[0],xlims[1]+0.01,0.01)
ax.plot(lineThresholds, np.full_like(lineThresholds,1), 'k--', alpha = 0.5)

ax.set_xticks([0.5,1.5,2.5,3.5],['R@R','R&U', 'LPFC','FEF'], rotation=0)
ax.set_xlabel('Strategy/Region', labelpad = 5, fontsize = 12)
ax.set_ylim(top=6, bottom=0)
ax.set_title('Code Morphing, Readout Space', fontsize = 12, pad=10)
plt.show()

# print stats
print(f"R@R: M(SD) = {codeMorph_readout_rnn['ed2'].mean():.3f}({codeMorph_readout_rnn['ed2'].std():.3f}), p = {p1:.3f};")
print(f"R&U: M(SD) = {codeMorph_readout_rnn['ed12'].mean():.3f}({codeMorph_readout_rnn['ed12'].std():.3f}), p = {p2:.3f};")
print(f"LPFC: M(SD) = {codeMorph_readout_data['dlpfc'].mean():.3f}({codeMorph_readout_data['dlpfc'].std():.3f}), p = {p3:.3f};")
print(f"FEF: M(SD) = {codeMorph_readout_data['fef'].mean():.3f}({codeMorph_readout_data['fef'].std():.3f}), p = {p4:.3f};")
print('\n')

#%% evaluate and plot readout subspace distractor information quantification

###################
# distractor info #
###################

ed2 = np.arange(1600,2100+bins,bins)
ed2x = [tbins.tolist().index(t) for t in ed2]

# get diagonal decodability
pfm22_readout_rnn = {k:np.array([performanceX2_readout_rnn[k][n][2] for n in range(100)]).mean(1).mean(1).diagonal(offset=0,axis1=1,axis2=2)[:,ed2x].mean(-1) for k in ('ed2', 'ed12')}
pfm22_readout_shuff_rnn = {k:np.array([performanceX2_readout_shuff_rnn[k][n][2] for n in range(100)]).mean(1).mean(1).diagonal(offset=0,axis1=1,axis2=2)[:,ed2x].mean(-1) for k in ('ed2', 'ed12')}

pfm22_readout_data = {k:np.array(performanceX2_readout_data[k][2]).mean(1).diagonal(offset=0,axis1=1,axis2=2)[:,ed2x].mean(-1) for k in ('dlpfc','fef')}
pfm22_readout_shuff_data = {k:np.array(performanceX2_readout_shuff_data[k][2]).mean(1).diagonal(offset=0,axis1=1,axis2=2)[:,ed2x].mean(-1) for k in ('dlpfc','fef')}

fig, axes = plt.subplots(1,1, sharey=True, figsize=(3,3), dpi=300)

# readout subspace
ax = axes#.flatten()[0]

ax.boxplot([pfm22_readout_rnn['ed2']], positions=[0.3], patch_artist=True, widths = 0.2, boxprops=boxprops1, flierprops=flierprops1, 
                  meanprops=meanpointprops1, medianprops=medianprops, capprops = capprops1, whiskerprops = whiskerprops1, meanline=False, showmeans=True)
ax.boxplot([pfm22_readout_rnn['ed12']], positions=[1.3], patch_artist=True, widths = 0.2, boxprops=boxprops2, flierprops=flierprops2, 
                  meanprops=meanpointprops2, medianprops=medianprops, capprops = capprops2, whiskerprops = whiskerprops2, meanline=False, showmeans=True)
ax.boxplot([pfm22_readout_data['dlpfc']], positions=[2.3], patch_artist=True, widths = 0.2, boxprops=boxprops3, flierprops=flierprops3, 
                  meanprops=meanpointprops3, medianprops=medianprops, capprops = capprops3, whiskerprops = whiskerprops3, meanline=False, showmeans=True)
ax.boxplot([pfm22_readout_data['fef']], positions=[3.3], patch_artist=True, widths = 0.2, boxprops=boxprops4, flierprops=flierprops4, 
                  meanprops=meanpointprops4, medianprops=medianprops, capprops = capprops4, whiskerprops = whiskerprops4, meanline=False, showmeans=True)

ax.boxplot([pfm22_readout_shuff_rnn['ed2']], positions=[0.7], patch_artist=True, widths = 0.2, boxprops=boxprops0, flierprops=flierprops0, 
                  meanprops=meanpointprops0, medianprops=medianprops, capprops = capprops0, whiskerprops = whiskerprops0, meanline=False, showmeans=True)
ax.boxplot([pfm22_readout_shuff_rnn['ed12']], positions=[1.7], patch_artist=True, widths = 0.2, boxprops=boxprops0, flierprops=flierprops0, 
                  meanprops=meanpointprops0, medianprops=medianprops, capprops = capprops0, whiskerprops = whiskerprops0, meanline=False, showmeans=True)
ax.boxplot([pfm22_readout_shuff_data['dlpfc']], positions=[2.7], patch_artist=True, widths = 0.2, boxprops=boxprops0, flierprops=flierprops0, 
                  meanprops=meanpointprops0, medianprops=medianprops, capprops = capprops0, whiskerprops = whiskerprops0, meanline=False, showmeans=True)
ax.boxplot([pfm22_readout_shuff_data['fef']], positions=[3.7], patch_artist=True, widths = 0.2, boxprops=boxprops0, flierprops=flierprops0, 
                  meanprops=meanpointprops0, medianprops=medianprops, capprops = capprops0, whiskerprops = whiskerprops0, meanline=False, showmeans=True)


p1 = f_stats.permutation_pCI(pfm22_readout_rnn['ed2'], pfm22_readout_shuff_rnn['ed2'], tail='greater', alpha=5)
ax.text(0.5,0.8, f'{f_plotting.sig_marker(p1,ns_note=True)}',horizontalalignment='center', fontsize=12)
print(f"R@R: Mean(SD) = {pfm22_readout_rnn['ed2'].mean():.3f}({pfm22_readout_rnn['ed2'].std():.3f}), p = {p1:.3f};")

p2 = f_stats.permutation_pCI(pfm22_readout_rnn['ed12'], pfm22_readout_shuff_rnn['ed12'], tail='greater', alpha=5)
ax.text(1.5,0.8, f'{f_plotting.sig_marker(p2,ns_note=True)}',horizontalalignment='center', fontsize=12)
print(f"R&U: Mean(SD) = {pfm22_readout_rnn['ed12'].mean():.3f}({pfm22_readout_rnn['ed12'].std():.3f}), p = {p2:.3f};")

p3 = f_stats.permutation_pCI(pfm22_readout_data['dlpfc'], pfm22_readout_shuff_data['dlpfc'], tail='greater', alpha=5)
ax.text(2.5,0.8, f'{f_plotting.sig_marker(p3,ns_note=True)}',horizontalalignment='center', fontsize=12)
print(f"LPFC: Mean(SD) = {pfm22_readout_data['dlpfc'].mean():.3f}({pfm22_readout_data['dlpfc'].std():.3f}), p = {p3:.3f};")

p4 = f_stats.permutation_pCI(pfm22_readout_data['fef'], pfm22_readout_shuff_data['fef'], tail='greater', alpha=5)
ax.text(3.5,0.8, f'{f_plotting.sig_marker(p4,ns_note=True)}',horizontalalignment='center', fontsize=12)
print(f"FEF: Mean(SD) = {pfm22_readout_data['fef'].mean():.3f}({pfm22_readout_data['fef'].std():.3f}), p = {p4:.3f};")


xlims = ax.get_xlim()
lineThresholds = np.arange(xlims[0],xlims[1]+0.01,0.01)
ax.set_xticks([0.5,1.5,2.5,3.5],['R@R','R&U','LPFC','FEF'], rotation=0)
ax.set_xlabel('Strategy/Region', labelpad = 5, fontsize = 12)
ax.set_ylabel('Decodability', labelpad = 3, fontsize = 12)
ax.set_ylim(top=0.9, bottom=0.1)
plt.suptitle('Distractor Information, Readout Subspace, ED2', fontsize = 12, y=1.0)
plt.show()