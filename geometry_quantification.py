# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 08:36:07 2024

@author: aka2333
"""
#%%
%reload_ext autoreload
%autoreload 2

# Import useful py lib/packages
from itertools import permutations, product

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import scipy
from scipy import stats

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import f_stats
import f_decoding
import f_plotting

#%% initialize paths and parameters
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

checkpoints = [150, 550, 1050, 1450, 1850, 2350]#
checkpointsLabels = ['S1','ED1','LD1','S2','ED2','LD2']

#%% initialize figure properties

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
flierprops1 = dict(markeredgecolor=color1, markerfacecolor=color1, markersize=3)
capprops1 = dict(color=color1)
whiskerprops1 = dict(color=color1)
meanpointprops1 = dict(marker='^', markeredgecolor=color1, markerfacecolor='w')

boxprops2 = dict(facecolor=color2, edgecolor='none',alpha = 1)
flierprops2 = dict(markeredgecolor=color2, markerfacecolor=color2,alpha = 1, markersize=3)
capprops2 = dict(color=color2,alpha = 1)
whiskerprops2 = dict(color=color2,alpha = 1)
meanpointprops2 = dict(marker='^', markeredgecolor=color2, markerfacecolor='w',alpha = 1)

boxprops3 = dict(facecolor=color3, edgecolor='none')
flierprops3 = dict(markeredgecolor=color3, markerfacecolor=color3, markersize=3)
capprops3 = dict(color=color3)
whiskerprops3 = dict(color=color3)
meanpointprops3 = dict(marker='^', markeredgecolor=color3, markerfacecolor='w')

boxprops4 = dict(facecolor=color4, edgecolor='none',alpha = 1)
flierprops4 = dict(markeredgecolor=color4, markerfacecolor=color4,alpha = 1, markersize=3)
capprops4 = dict(color=color4,alpha = 1)
whiskerprops4 = dict(color=color4,alpha = 1)
meanpointprops4 = dict(marker='^', markeredgecolor=color4, markerfacecolor='w',alpha = 1)

medianprops = dict(linestyle='--', linewidth=1, color='w')


#%% load precomputed geometries from files

###############
# Monkey Data #
###############

cosTheta11_data = np.load(f'{data_path}/' + 'cosTheta_11_data.npy', allow_pickle=True).item()
cosTheta12_data = np.load(f'{data_path}/' + 'cosTheta_12_data.npy', allow_pickle=True).item()
cosTheta22_data = np.load(f'{data_path}/' + 'cosTheta_22_data.npy', allow_pickle=True).item()
cosPsi11_data = np.load(f'{data_path}/' + 'cosPsi_11_data.npy', allow_pickle=True).item()
cosPsi12_data = np.load(f'{data_path}/' + 'cosPsi_12_data.npy', allow_pickle=True).item()
cosPsi22_data = np.load(f'{data_path}/' + 'cosPsi_22_data.npy', allow_pickle=True).item()

cosTheta11_shuff_data = np.load(f'{data_path}/' + 'cosTheta_11_shuff_data.npy', allow_pickle=True).item()
cosTheta12_shuff_data = np.load(f'{data_path}/' + 'cosTheta_12_shuff_data.npy', allow_pickle=True).item()
cosTheta22_shuff_data = np.load(f'{data_path}/' + 'cosTheta_22_shuff_data.npy', allow_pickle=True).item()
cosPsi11_shuff_data = np.load(f'{data_path}/' + 'cosPsi_11_shuff_data.npy', allow_pickle=True).item()
cosPsi12_shuff_data = np.load(f'{data_path}/' + 'cosPsi_12_shuff_data.npy', allow_pickle=True).item()
cosPsi22_shuff_data = np.load(f'{data_path}/' + 'cosPsi_22_shuff_data.npy', allow_pickle=True).item()

cosTheta_choice_data = np.load(f'{data_path}/' + 'cosTheta_choice_data.npy', allow_pickle=True).item()
cosTheta_nonchoice_data = np.load(f'{data_path}/' + 'cosTheta_nonchoice_data.npy', allow_pickle=True).item()
cosPsi_choice_data = np.load(f'{data_path}/' + 'cosPsi_choice_data.npy', allow_pickle=True).item()
cosPsi_nonchoice_data = np.load(f'{data_path}/' + 'cosPsi_nonchoice_data.npy', allow_pickle=True).item()

cosTheta_choice_shuff_data = np.load(f'{data_path}/' + 'cosTheta_choice_shuff_data.npy', allow_pickle=True).item()
cosTheta_nonchoice_shuff_data = np.load(f'{data_path}/' + 'cosTheta_nonchoice_shuff_data.npy', allow_pickle=True).item()
cosPsi_choice_shuff_data = np.load(f'{data_path}/' + 'cosPsi_choice_shuff_data.npy', allow_pickle=True).item()
cosPsi_nonchoice_shuff_data = np.load(f'{data_path}/' + 'cosPsi_nonchoice_shuff_data.npy', allow_pickle=True).item()

# load precomputed performance from files
performance1_item_data = np.load(f'{data_path}/' + 'performance1_item_data.npy', allow_pickle=True).item()
performance2_item_data = np.load(f'{data_path}/' + 'performance2_item_data.npy', allow_pickle=True).item()
performance1_item_shuff_data = np.load(f'{data_path}/' + 'performance1_item_shuff_data.npy', allow_pickle=True).item()
performance2_item_shuff_data = np.load(f'{data_path}/' + 'performance2_item_shuff_data.npy', allow_pickle=True).item()

# load precomputed geometries from files for baselines
cosTheta11_bsl_data = np.load(f'{data_path}/' + 'cosTheta_11_bsl_data.npy', allow_pickle=True).item()
cosTheta12_bsl_data = np.load(f'{data_path}/' + 'cosTheta_12_bsl_data.npy', allow_pickle=True).item()
cosTheta22_bsl_data = np.load(f'{data_path}/' + 'cosTheta_22_bsl_data.npy', allow_pickle=True).item()
cosPsi11_bsl_data = np.load(f'{data_path}/' + 'cosPsi_11_bsl_data.npy', allow_pickle=True).item()
cosPsi12_bsl_data = np.load(f'{data_path}/' + 'cosPsi_12_bsl_data.npy', allow_pickle=True).item()
cosPsi22_bsl_data = np.load(f'{data_path}/' + 'cosPsi_22_bsl_data.npy', allow_pickle=True).item()


########
# rnns #
########

cosTheta11_rnn = np.load(f'{data_path}/' + 'cosTheta_11_rnn.npy', allow_pickle=True).item()
cosTheta12_rnn = np.load(f'{data_path}/' + 'cosTheta_12_rnn.npy', allow_pickle=True).item()
cosTheta22_rnn = np.load(f'{data_path}/' + 'cosTheta_22_rnn.npy', allow_pickle=True).item()
cosPsi11_rnn = np.load(f'{data_path}/' + 'cosPsi_11_rnn.npy', allow_pickle=True).item()
cosPsi12_rnn = np.load(f'{data_path}/' + 'cosPsi_12_rnn.npy', allow_pickle=True).item()
cosPsi22_rnn = np.load(f'{data_path}/' + 'cosPsi_22_rnn.npy', allow_pickle=True).item()

cosTheta11_shuff_rnn = np.load(f'{data_path}/' + 'cosTheta_11_shuff_rnn.npy', allow_pickle=True).item()
cosTheta12_shuff_rnn = np.load(f'{data_path}/' + 'cosTheta_12_shuff_rnn.npy', allow_pickle=True).item()
cosTheta22_shuff_rnn = np.load(f'{data_path}/' + 'cosTheta_22_shuff_rnn.npy', allow_pickle=True).item()
cosPsi11_shuff_rnn = np.load(f'{data_path}/' + 'cosPsi_11_shuff_rnn.npy', allow_pickle=True).item()
cosPsi12_shuff_rnn = np.load(f'{data_path}/' + 'cosPsi_12_shuff_rnn.npy', allow_pickle=True).item()
cosPsi22_shuff_rnn = np.load(f'{data_path}/' + 'cosPsi_22_shuff_rnn.npy', allow_pickle=True).item()

cosTheta_choice_rnn = np.load(f'{data_path}/' + 'cosTheta_choice_rnn.npy', allow_pickle=True).item()
cosTheta_nonchoice_rnn = np.load(f'{data_path}/' + 'cosTheta_nonchoice_rnn.npy', allow_pickle=True).item()
cosPsi_choice_rnn = np.load(f'{data_path}/' + 'cosPsi_choice_rnn.npy', allow_pickle=True).item()
cosPsi_nonchoice_rnn = np.load(f'{data_path}/' + 'cosPsi_nonchoice_rnn.npy', allow_pickle=True).item()

cosTheta_choice_shuff_rnn = np.load(f'{data_path}/' + 'cosTheta_choice_shuff_rnn.npy', allow_pickle=True).item()
cosTheta_nonchoice_shuff_rnn = np.load(f'{data_path}/' + 'cosTheta_nonchoice_shuff_rnn.npy', allow_pickle=True).item()
cosPsi_choice_shuff_rnn = np.load(f'{data_path}/' + 'cosPsi_choice_shuff_rnn.npy', allow_pickle=True).item()
cosPsi_nonchoice_shuff_rnn = np.load(f'{data_path}/' + 'cosPsi_nonchoice_shuff_rnn.npy', allow_pickle=True).item()

# load precomputed performance from files
performance1_item_rnn = np.load(f'{data_path}/' + 'performance1_item_rnn.npy', allow_pickle=True).item()
performance2_item_rnn = np.load(f'{data_path}/' + 'performance2_item_rnn.npy', allow_pickle=True).item()
performance1_item_shuff_rnn = np.load(f'{data_path}/' + 'performance1_item_shuff_rnn.npy', allow_pickle=True).item()
performance2_item_shuff_rnn = np.load(f'{data_path}/' + 'performance2_item_shuff_rnn.npy', allow_pickle=True).item()

# load precomputed geometries from files for baselines
cosTheta11_bsl_rnn = np.load(f'{data_path}/' + 'cosTheta_11_bsl_rnn.npy', allow_pickle=True).item()
cosTheta12_bsl_rnn = np.load(f'{data_path}/' + 'cosTheta_12_bsl_rnn.npy', allow_pickle=True).item()
cosTheta22_bsl_rnn = np.load(f'{data_path}/' + 'cosTheta_22_bsl_rnn.npy', allow_pickle=True).item()
cosPsi11_bsl_rnn = np.load(f'{data_path}/' + 'cosPsi_11_bsl_rnn.npy', allow_pickle=True).item()
cosPsi12_bsl_rnn = np.load(f'{data_path}/' + 'cosPsi_12_bsl_rnn.npy', allow_pickle=True).item()
cosPsi22_bsl_rnn = np.load(f'{data_path}/' + 'cosPsi_22_bsl_rnn.npy', allow_pickle=True).item()

#%%




######################
# T/T, I1D1 vs. I2D2 #
######################





#%% [Figure 3A, 3B] plot subspace coplanarity and alignment at specific checkpoints

# calculate mean and std for coplanarity and alignment
cosTheta12Ret_rnn = {k:np.array([cosTheta12_rnn[k][n][1] for n in range(100)]).mean(1) for k in ('ed2','ed12')}
cosTheta12Ret_data = {k:cosTheta12_data[k][1].mean(1) for k in ('dlpfc','fef')}
cosTheta12Ret_shuff_rnn = {k:np.array([cosTheta12_shuff_rnn[k][n][1] for n in range(100)]).mean(1) for k in ('ed2','ed12')}
cosTheta12Ret_shuff_data = {k:cosTheta12_shuff_data[k][1] for k in ('dlpfc','fef')}

cosPsi12Ret_rnn = {k:np.array([cosPsi12_rnn[k][n][1] for n in range(100)]).mean(1) for k in ('ed2','ed12')}
cosPsi12Ret_data = {k:cosPsi12_data[k][1].mean(1) for k in ('dlpfc','fef')}
cosPsi12Ret_shuff_rnn = {k:np.array([cosPsi12_shuff_rnn[k][n][1] for n in range(100)]).mean(1) for k in ('ed2','ed12')}
cosPsi12Ret_shuff_data = {k:cosPsi12_shuff_data[k][1] for k in ('dlpfc','fef')}


lineh = np.arange(0.5,1.5,0.001)
linev = np.arange(0.71,0.72,0.0001)

angleCheckPoints = np.linspace(0,np.pi,7).round(5)
cmap = plt.get_cmap('coolwarm')
norm = mcolors.BoundaryNorm(np.flip(np.cos(angleCheckPoints).round(5)),cmap.N, clip=True)

showCheckpoints = ('L',) # just check at Late delay windows
showShuffBsl = False
fig, axes = plt.subplots(1,2, sharey=True, figsize=(8,4), dpi=300)

# coplarnarity
ax = axes.flatten()[0]
xtks, xtklabs = [], []
for nk, k in enumerate(showCheckpoints):
    dk1, dk2 = k+'D1', k+'D2'
    d1x, d2x = checkpointsLabels.index(dk1), checkpointsLabels.index(dk2)
    #ld1x, ld2x = checkpointsLabels.index('LD1'), checkpointsLabels.index('LD2')

    print('############ cosTheta ############')
    
    # method: baseline from split sets
    cosTheta_bsl_rnn = {kk: np.mean((np.array([cosTheta11_bsl_rnn[kk][n][1] for n in range(100)]).squeeze()[:,:,d1x].mean(1),
                                   np.array([cosTheta22_bsl_rnn[kk][n][1] for n in range(100)]).squeeze()[:,:,d2x].mean(1)),axis=0) for kk in ('ed2','ed12')}
    cosTheta_bsl_data = {kk: np.mean((cosTheta11_bsl_data[kk][1][:,:,d1x].mean(1), 
                                     cosTheta22_bsl_data[kk][1][:,:,d2x].mean(1)),axis=0) for kk in ('dlpfc','fef')}
    
    
    # plot
    ax.boxplot([cosTheta12Ret_rnn['ed2'][:,d1x,d2x]], positions=[0.2+nk-0.035*showShuffBsl], patch_artist=True, widths = 0.05*len(showCheckpoints), boxprops=boxprops1, flierprops=flierprops1, 
                    meanprops=meanpointprops1, medianprops=medianprops, capprops = capprops1, whiskerprops = whiskerprops1, meanline=False, showmeans=True)
    p1 = f_stats.permutation_pCI(cosTheta12Ret_rnn['ed2'][:,d1x,d2x], cosTheta_bsl_rnn['ed2'], tail='smaller', alpha = 5) #.mean(1)
    ax.text(0.2+nk,1.0, f'{f_plotting.sig_marker(p1,ns_note=False)}',horizontalalignment='center', verticalalignment='bottom', fontsize=10)
    print(f"R@R: M(SD) = {cosTheta12Ret_rnn['ed2'][:,d1x,d2x].mean():.3f}({cosTheta12Ret_rnn['ed2'][:,d1x,d2x].std():.3f}), 95CI = [{np.quantile(cosTheta12Ret_rnn['ed2'][:,d1x,d2x],0.025):.3f}, {np.quantile(cosTheta12Ret_rnn['ed2'][:,d1x,d2x],0.975):.3f}], p = {p1:.3f}")


    ax.boxplot([cosTheta12Ret_rnn['ed12'][:,d1x,d2x]], positions=[0.4+nk-0.035*showShuffBsl], patch_artist=True, widths = 0.05*len(showCheckpoints), boxprops=boxprops2, flierprops=flierprops2, 
                    meanprops=meanpointprops2, medianprops=medianprops, capprops = capprops2, whiskerprops = whiskerprops2, meanline=False, showmeans=True)
    p2 = f_stats.permutation_pCI(cosTheta12Ret_rnn['ed12'][:,d1x,d2x], cosTheta_bsl_rnn['ed12'], tail='smaller', alpha = 5)
    ax.text(0.4+nk,1.0, f'{f_plotting.sig_marker(p2,ns_note=False)}',horizontalalignment='center', verticalalignment='bottom', fontsize=10)
    print(f"R&U: M(SD) = {cosTheta12Ret_rnn['ed12'][:,d1x,d2x].mean():.3f}({cosTheta12Ret_rnn['ed12'][:,d1x,d2x].std():.3f}), 95CI = [{np.quantile(cosTheta12Ret_rnn['ed12'][:,d1x,d2x],0.025):.3f}, {np.quantile(cosTheta12Ret_rnn['ed12'][:,d1x,d2x],0.975):.3f}], p = {p2:.3f}")

    
    ax.boxplot([cosTheta12Ret_data['dlpfc'][:,d1x,d2x]], positions=[0.6+nk-0.035*showShuffBsl], patch_artist=True, widths = 0.05*len(showCheckpoints), boxprops=boxprops3, flierprops=flierprops3, 
                    meanprops=meanpointprops3, medianprops=medianprops, capprops = capprops3, whiskerprops = whiskerprops3, meanline=False, showmeans=True)
    p3 = f_stats.permutation_pCI(cosTheta12Ret_data['dlpfc'][:,d1x,d2x], cosTheta_bsl_data['dlpfc'], tail='smaller', alpha = 5) #.mean(1)
    ax.text(0.6+nk,1.0, f'{f_plotting.sig_marker(p3,ns_note=False)}',horizontalalignment='center', verticalalignment='bottom', fontsize=10)
    print(f"LPFC: M(SD) = {cosTheta12Ret_data['dlpfc'][:,d1x,d2x].mean():.3f}({cosTheta12Ret_data['dlpfc'][:,d1x,d2x].std():.3f}), 95CI = [{np.quantile(cosTheta12Ret_data['dlpfc'][:,d1x,d2x],0.025):.3f}, {np.quantile(cosTheta12Ret_data['dlpfc'][:,d1x,d2x],0.975):.3f}], p = {p3:.3f}")

    
    ax.boxplot([cosTheta12Ret_data['fef'][:,d1x,d2x]], positions=[0.8+nk-0.035*showShuffBsl], patch_artist=True, widths = 0.05*len(showCheckpoints), boxprops=boxprops4, flierprops=flierprops4, 
                    meanprops=meanpointprops4, medianprops=medianprops, capprops = capprops4, whiskerprops = whiskerprops4, meanline=False, showmeans=True)
    p4 = f_stats.permutation_pCI(cosTheta12Ret_data['fef'][:,d1x,d2x], cosTheta_bsl_data['fef'], tail='smaller', alpha = 5) #.mean(1)
    ax.text(0.8+nk,1.0, f'{f_plotting.sig_marker(p4,ns_note=False)}',horizontalalignment='center', verticalalignment='bottom', fontsize=10)
    print(f"FEF: M(SD) = {cosTheta12Ret_data['fef'][:,d1x,d2x].mean():.3f}({cosTheta12Ret_data['fef'][:,d1x,d2x].std():.3f}), 95CI = [{np.quantile(cosTheta12Ret_data['fef'][:,d1x,d2x],0.025):.3f}, {np.quantile(cosTheta12Ret_data['fef'][:,d1x,d2x],0.975):.3f}], p = {p4:.3f}")

    
    # whether to include baseline distributions or not
    if showShuffBsl:
        ax.boxplot([cosTheta_bsl_rnn['ed2']], positions=[0.2+nk+0.035*showShuffBsl], patch_artist=True, widths = 0.05*len(showCheckpoints), boxprops=boxprops0, flierprops=flierprops0, 
                    meanprops=meanpointprops0, medianprops=medianprops, capprops = capprops0, whiskerprops = whiskerprops0, meanline=False, showmeans=True)
        
        ax.boxplot([cosTheta_bsl_rnn['ed12']], positions=[0.4+nk+0.035*showShuffBsl], patch_artist=True, widths = 0.05*len(showCheckpoints), boxprops=boxprops0, flierprops=flierprops0, 
                        meanprops=meanpointprops0, medianprops=medianprops, capprops = capprops0, whiskerprops = whiskerprops0, meanline=False, showmeans=True)
        
        ax.boxplot([cosTheta_bsl_data['dlpfc']], positions=[0.6+nk+0.035*showShuffBsl], patch_artist=True, widths = 0.05*len(showCheckpoints), boxprops=boxprops0, flierprops=flierprops0, 
                        meanprops=meanpointprops0, medianprops=medianprops, capprops = capprops0, whiskerprops = whiskerprops0, meanline=False, showmeans=True)
        
        ax.boxplot([cosTheta_bsl_data['fef']], positions=[0.8+nk+0.035*showShuffBsl], patch_artist=True, widths = 0.05*len(showCheckpoints), boxprops=boxprops0, flierprops=flierprops0, 
                        meanprops=meanpointprops0, medianprops=medianprops, capprops = capprops0, whiskerprops = whiskerprops0, meanline=False, showmeans=True)
    
    xtks += [0.5+nk]
    xtklabs += [f'{dk1}-{dk2}']
        
    # [Non-used] KS Test
    ks_results = {'R@R-LPFC':scipy.stats.ks_2samp(cosTheta12Ret_rnn['ed2'][:,d1x,d2x], cosTheta12Ret_data['dlpfc'][:,d1x,d2x]),
                  'R@R-FEF':scipy.stats.ks_2samp(cosTheta12Ret_rnn['ed2'][:,d1x,d2x], cosTheta12Ret_data['fef'][:,d1x,d2x]),
                  'R&U-LPFC':scipy.stats.ks_2samp(cosTheta12Ret_rnn['ed12'][:,d1x,d2x], cosTheta12Ret_data['dlpfc'][:,d1x,d2x]), 
                  'R&U-FEF':scipy.stats.ks_2samp(cosTheta12Ret_rnn['ed12'][:,d1x,d2x], cosTheta12Ret_data['fef'][:,d1x,d2x])}

    print('############### K-S Test ##############')
    for k in ks_results.keys():
        print(f'{k.upper()}: D = {ks_results[k][0]:.3f}, p = {ks_results[k][1]:.3f}')


ax.set_xticks(xtks,xtklabs)#,rotation=20
ax.set_xlim(0,len(showCheckpoints))
ax.set_ylim(-1.05,1.2)
ax.set_yticks(np.flip(np.cos(angleCheckPoints).round(5)), labels = np.flip(np.degrees(angleCheckPoints).round(1).astype(int)))
ax.set_xlabel('Timebins', labelpad = 3, fontsize = 15)
ax.set_ylabel('cos(θ)', labelpad = 0, fontsize = 15)
ax.set_title(f'Principal Angle', fontsize = 17)
ax.tick_params(axis='both', labelsize=12)

print('#################################')
print('\n')
    
# alignment
ax = axes.flatten()[1]
xtks, xtklabs = [], []
for nk, k in enumerate(showCheckpoints):
    dk1, dk2 = k+'D1', k+'D2'
    d1x, d2x = checkpointsLabels.index(dk1), checkpointsLabels.index(dk2)
    
    print('############## cosPsi ##############')

    # method: baseline from split sets
    cosPsi_bsl_rnn = {kk: np.mean((np.array([cosPsi11_bsl_rnn[kk][n][1] for n in range(100)]).squeeze()[:,:,d1x].mean(1),
                                   np.array([cosPsi22_bsl_rnn[kk][n][1] for n in range(100)]).squeeze()[:,:,d2x].mean(1)),axis=0) for kk in ('ed2','ed12')}
    cosPsi_bsl_data = {kk: np.mean((cosPsi11_bsl_data[kk][1][:,:,d1x].mean(1), 
                                     cosPsi22_bsl_data[kk][1][:,:,d2x].mean(1)),axis=0) for kk in ('dlpfc','fef')}
    
    
    ax.boxplot([cosPsi12Ret_rnn['ed2'][:,d1x,d2x]], positions=[0.2+nk-0.035*showShuffBsl], patch_artist=True, widths = 0.05*len(showCheckpoints), boxprops=boxprops1, flierprops=flierprops1, 
                    meanprops=meanpointprops1, medianprops=medianprops, capprops = capprops1, whiskerprops = whiskerprops1, meanline=False, showmeans=True)
    p1 = f_stats.permutation_pCI(cosPsi12Ret_rnn['ed2'][:,d1x,d2x], cosPsi_bsl_rnn['ed2'], tail='smaller', alpha = 5)
    ax.text(0.2+nk,1.0, f'{f_plotting.sig_marker(p1,ns_note=False)}',horizontalalignment='center', verticalalignment='bottom', fontsize=10)
    print(f"R@R: M(SD) = {cosPsi12Ret_rnn['ed2'][:,d1x,d2x].mean():.3f}({cosPsi12Ret_rnn['ed2'][:,d1x,d2x].std():.3f}), 95CI = [{np.quantile(cosPsi12Ret_rnn['ed2'][:,d1x,d2x],0.025):.3f}, {np.quantile(cosPsi12Ret_rnn['ed2'][:,d1x,d2x],0.975):.3f}], p = {p1:.3f}")

    
    ax.boxplot([cosPsi12Ret_rnn['ed12'][:,d1x,d2x]], positions=[0.4+nk-0.035*showShuffBsl], patch_artist=True, widths = 0.05*len(showCheckpoints), boxprops=boxprops2, flierprops=flierprops2, 
                    meanprops=meanpointprops2, medianprops=medianprops, capprops = capprops2, whiskerprops = whiskerprops2, meanline=False, showmeans=True)
    p2 = f_stats.permutation_pCI(cosPsi12Ret_rnn['ed12'][:,d1x,d2x], cosPsi_bsl_rnn['ed12'], tail='smaller', alpha = 5)
    ax.text(0.4+nk,1.0, f'{f_plotting.sig_marker(p2,ns_note=False)}',horizontalalignment='center', verticalalignment='bottom', fontsize=10)
    print(f"R&U: M(SD) = {cosPsi12Ret_rnn['ed12'][:,d1x,d2x].mean():.3f}({cosPsi12Ret_rnn['ed12'][:,d1x,d2x].std():.3f}), 95CI = [{np.quantile(cosPsi12Ret_rnn['ed12'][:,d1x,d2x],0.025):.3f}, {np.quantile(cosPsi12Ret_rnn['ed12'][:,d1x,d2x],0.975):.3f}], p = {p2:.3f}")

    
    ax.boxplot([cosPsi12Ret_data['dlpfc'][:,d1x,d2x]], positions=[0.6+nk-0.035*showShuffBsl], patch_artist=True, widths = 0.05*len(showCheckpoints), boxprops=boxprops3, flierprops=flierprops3, 
                    meanprops=meanpointprops3, medianprops=medianprops, capprops = capprops3, whiskerprops = whiskerprops3, meanline=False, showmeans=True)
    p3 = f_stats.permutation_pCI(cosPsi12Ret_data['dlpfc'][:,d1x,d2x], cosPsi_bsl_data['dlpfc'], tail='smaller', alpha = 5)
    ax.text(0.6+nk,1.0, f'{f_plotting.sig_marker(p3,ns_note=False)}',horizontalalignment='center', verticalalignment='bottom', fontsize=10)
    print(f"LPFC: M(SD) = {cosPsi12Ret_data['dlpfc'][:,d1x,d2x].mean():.3f}({cosPsi12Ret_data['dlpfc'][:,d1x,d2x].std():.3f}), 95CI = [{np.quantile(cosPsi12Ret_data['dlpfc'][:,d1x,d2x],0.025):.3f}, {np.quantile(cosPsi12Ret_data['dlpfc'][:,d1x,d2x],0.975):.3f}], p = {p3:.3f}")

    
    ax.boxplot([cosPsi12Ret_data['fef'][:,d1x,d2x]], positions=[0.8+nk-0.035*showShuffBsl], patch_artist=True, widths = 0.05*len(showCheckpoints), boxprops=boxprops4, flierprops=flierprops4, 
                    meanprops=meanpointprops4, medianprops=medianprops, capprops = capprops4, whiskerprops = whiskerprops4, meanline=False, showmeans=True)
    p4 = f_stats.permutation_pCI(cosPsi12Ret_data['fef'][:,d1x,d2x], cosPsi_bsl_data['fef'], tail='smaller', alpha = 5)
    ax.text(0.8+nk,1.0, f'{f_plotting.sig_marker(p4,ns_note=False)}',horizontalalignment='center', verticalalignment='bottom', fontsize=10)
    print(f"FEF: M(SD) = {cosPsi12Ret_data['fef'][:,d1x,d2x].mean():.3f}({cosPsi12Ret_data['fef'][:,d1x,d2x].std():.3f}), 95CI = [{np.quantile(cosPsi12Ret_data['fef'][:,d1x,d2x],0.025):.3f}, {np.quantile(cosPsi12Ret_data['fef'][:,d1x,d2x],0.975):.3f}], p = {p4:.3f}")

    # whether to include baseline distributions or not
    if showShuffBsl:
        ax.boxplot([cosPsi_bsl_rnn['ed2']], positions=[0.2+nk+0.035*showShuffBsl], patch_artist=True, widths = 0.05*len(showCheckpoints), boxprops=boxprops0, flierprops=flierprops0, 
                    meanprops=meanpointprops0, medianprops=medianprops, capprops = capprops0, whiskerprops = whiskerprops0, meanline=False, showmeans=True)
        
        ax.boxplot([cosPsi_bsl_rnn['ed12']], positions=[0.4+nk+0.035*showShuffBsl], patch_artist=True, widths = 0.05*len(showCheckpoints), boxprops=boxprops0, flierprops=flierprops0, 
                        meanprops=meanpointprops0, medianprops=medianprops, capprops = capprops0, whiskerprops = whiskerprops0, meanline=False, showmeans=True)
        
        ax.boxplot([cosPsi_bsl_data['dlpfc']], positions=[0.6+nk+0.035*showShuffBsl], patch_artist=True, widths = 0.05*len(showCheckpoints), boxprops=boxprops0, flierprops=flierprops0, 
                        meanprops=meanpointprops0, medianprops=medianprops, capprops = capprops0, whiskerprops = whiskerprops0, meanline=False, showmeans=True)
        
        ax.boxplot([cosPsi_bsl_data['fef']], positions=[0.8+nk+0.035*showShuffBsl], patch_artist=True, widths = 0.05*len(showCheckpoints), boxprops=boxprops0, flierprops=flierprops0, 
                        meanprops=meanpointprops0, medianprops=medianprops, capprops = capprops0, whiskerprops = whiskerprops0, meanline=False, showmeans=True)
        
    
    xtks += [0.5+nk]
    xtklabs += [f'{dk1}-{dk2}']

    # [Non-used] KS test
    ks_results = {'R@R-LPFC':scipy.stats.ks_2samp(cosPsi12Ret_rnn['ed2'][:,d1x,d2x], cosPsi12Ret_data['dlpfc'][:,d1x,d2x]),
                  'R@R-FEF':scipy.stats.ks_2samp(cosPsi12Ret_rnn['ed2'][:,d1x,d2x], cosPsi12Ret_data['fef'][:,d1x,d2x]),
                  'R&U-LPFC':scipy.stats.ks_2samp(cosPsi12Ret_rnn['ed12'][:,d1x,d2x], cosPsi12Ret_data['dlpfc'][:,d1x,d2x]), 
                  'R&U-FEF':scipy.stats.ks_2samp(cosPsi12Ret_rnn['ed12'][:,d1x,d2x], cosPsi12Ret_data['fef'][:,d1x,d2x])}

    print('############### K-S Test ##############')
    for k in ks_results.keys():
        print(f'{k.upper()}: D = {ks_results[k][0]:.3f}, p = {ks_results[k][1]:.3f}')
    

ax.set_xticks(xtks,xtklabs)#,rotation=20
ax.set_xlim(0,len(showCheckpoints))
ax.set_ylim(-1.05,1.2)
ax.set_yticks(np.flip(np.cos(angleCheckPoints).round(5)), labels = np.flip(np.degrees(angleCheckPoints).round(1).astype(int)))
ax.set_xlabel('Timebins', labelpad = 3, fontsize = 15)
ax.set_ylabel('cos(ψ)', labelpad = 0, fontsize = 15)
ax.set_title(f'Representational Alignment', fontsize = 17)
ax.tick_params(axis='both', labelsize=12)

ax.plot([], c=color1, label='R@R')
ax.plot([], c=color2, label='R&U')
ax.plot([], c=color3, label='LPFC')
ax.plot([], c=color4, label='FEF')
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize = 12)

print('#################################')
print('\n')


plt.suptitle(f'I1D1 vs. I2D2, Retarget', fontsize = 25, y=1) # (Monkeys)
plt.tight_layout()
plt.show()

#%% plot code transferability at specific checkpoints

######################################
# code transferability between items #
######################################

# load precomputed code tranferability from file
performanceX_Trans_12_data = np.load(f'{data_path}/' + 'performance12X_Trans_data.npy', allow_pickle=True).item()
performanceX_Trans_21_data = np.load(f'{data_path}/' + 'performance21X_Trans_data.npy', allow_pickle=True).item()
performanceX_Trans_12_shuff_data = np.load(f'{data_path}/' + 'performance12X_Trans_shuff_data.npy', allow_pickle=True).item()
performanceX_Trans_21_shuff_data = np.load(f'{data_path}/' + 'performance21X_Trans_shuff_data.npy', allow_pickle=True).item()

performanceX_Trans_12_rnn = np.load(f'{data_path}/' + 'performanceX_12_rnn.npy', allow_pickle=True).item()
performanceX_Trans_21_rnn = np.load(f'{data_path}/' + 'performanceX_21_rnn.npy', allow_pickle=True).item()
performanceX_Trans_12_shuff_rnn = np.load(f'{data_path}/' + 'performanceX_12_shuff_rnn.npy', allow_pickle=True).item()
performanceX_Trans_21_shuff_rnn = np.load(f'{data_path}/' + 'performanceX_21_shuff_rnn.npy', allow_pickle=True).item()


performance1_item_data = np.load(f'{data_path}/' + 'performance1_item_data.npy', allow_pickle=True).item()
performance2_item_data = np.load(f'{data_path}/' + 'performance2_item_data.npy', allow_pickle=True).item()
performance1_item_shuff_data = np.load(f'{data_path}/' + 'performance1_item_shuff_data.npy', allow_pickle=True).item()
performance2_item_shuff_data = np.load(f'{data_path}/' + 'performance2_item_shuff_data.npy', allow_pickle=True).item()

performance1_item_rnn = np.load(f'{data_path}/' + 'performance1_item_rnn.npy', allow_pickle=True).item()
performance2_item_rnn = np.load(f'{data_path}/' + 'performance2_item_rnn.npy', allow_pickle=True).item()
performance1_item_shuff_rnn = np.load(f'{data_path}/' + 'performance1_item_shuff_rnn.npy', allow_pickle=True).item()
performance2_item_shuff_rnn = np.load(f'{data_path}/' + 'performance2_item_shuff_rnn.npy', allow_pickle=True).item()


checkpoints = [150, 550, 1050, 1450, 1850, 2350]#
checkpointsLabels = ['S1','ED1','LD1','S2','ED2','LD2'] 
avgInterval = {150:150, 550:250, 1050:250, 1450:150, 1850:250, 2350:250} 

# cross-subspace decodability
codeTrans_item_data = {}
codeTrans_item_data_shuff = {}

for k in ('dlpfc','fef'):
    pfmX12_data = {tt:np.array([performanceX_Trans_12_data[k][tt][n] for n in range(100)]).mean(1) for tt in (1,2)}
    pfmX21_data = {tt:np.array([performanceX_Trans_21_data[k][tt][n] for n in range(100)]).mean(1) for tt in (1,2)}
    pfmX12_shuff_data = {tt:np.array([performanceX_Trans_12_shuff_data[k][tt][n] for n in range(100)]).mean(1) for tt in (1,2)}
    pfmX21_shuff_data = {tt:np.array([performanceX_Trans_21_shuff_data[k][tt][n] for n in range(100)]).mean(1) for tt in (1,2)}

    codeTrans_item_data[k] = {tt:np.array([f_decoding.code_transferability(pfmX12_data[tt][n],pfmX21_data[tt][n],checkpoints) for n in range(100)]) for tt in (1,2)}
    codeTrans_item_data_shuff[k] = {tt:np.array([f_decoding.code_transferability(pfmX12_shuff_data[tt][n],pfmX21_shuff_data[tt][n],checkpoints) for n in range(100)]) for tt in (1,2)}

        
codeTrans_item_rnn = {}
codeTrans_item_shuff_rnn = {}

for k in ('ed2','ed12'):
    pfmX12_rnn = {tt:np.array([performanceX_Trans_12_rnn[k][n][tt] for n in range(100)]).mean(1).mean(1) for tt in (1,2)}
    pfmX21_rnn = {tt:np.array([performanceX_Trans_21_rnn[k][n][tt] for n in range(100)]).mean(1).mean(1) for tt in (1,2)}
    pfmX12_shuff_rnn = {tt:np.array([performanceX_Trans_12_shuff_rnn[k][n][tt] for n in range(100)]).mean(1).mean(1) for tt in (1,2)}
    pfmX21_shuff_rnn = {tt:np.array([performanceX_Trans_21_shuff_rnn[k][n][tt] for n in range(100)]).mean(1).mean(1) for tt in (1,2)}
    
    codeTrans_item_rnn[k] = {tt:np.array([f_decoding.code_transferability(pfmX12_rnn[tt][n],pfmX21_rnn[tt][n],checkpoints) for n in range(100)]) for tt in (1,2)}
    codeTrans_item_shuff_rnn[k] = {tt:np.array([f_decoding.code_transferability(pfmX12_shuff_rnn[tt][n],pfmX21_shuff_rnn[tt][n],checkpoints) for n in range(100)]) for tt in (1,2)} 

cp1, cp2 = 'LD1', 'LD2'
cp1x, cp2x = checkpointsLabels.index(cp1), checkpointsLabels.index(cp2)

lineh = np.arange(0.3,0.7,0.001)
linev = np.arange(1.16,1.165,0.0001)

fig, axes = plt.subplots(1,1, sharey=True, figsize=(3,3), dpi=300)

# full space
ax = axes#.flatten()[0]
ax.boxplot([codeTrans_item_rnn['ed2'][1][:,cp1x, cp2x]], positions=[0.3], patch_artist=True, widths = 0.2, boxprops=boxprops1, flierprops=flierprops1, 
                  meanprops=meanpointprops1, medianprops=medianprops, capprops = capprops1, whiskerprops = whiskerprops1, meanline=False, showmeans=True)
ax.boxplot([codeTrans_item_shuff_rnn['ed2'][1][:,cp1x, cp2x]], positions=[0.7], patch_artist=True, widths = 0.2, boxprops=boxprops0, flierprops=flierprops0, 
                  meanprops=meanpointprops0, medianprops=medianprops, capprops = capprops0, whiskerprops = whiskerprops0, meanline=False, showmeans=True)

ax.boxplot([codeTrans_item_rnn['ed12'][1][:,cp1x, cp2x]], positions=[1.3], patch_artist=True, widths = 0.2, boxprops=boxprops2, flierprops=flierprops2, 
                  meanprops=meanpointprops2, medianprops=medianprops, capprops = capprops2, whiskerprops = whiskerprops2, meanline=False, showmeans=True)
ax.boxplot([codeTrans_item_shuff_rnn['ed12'][1][:,cp1x, cp2x]], positions=[1.7], patch_artist=True, widths = 0.2, boxprops=boxprops0, flierprops=flierprops0, 
                  meanprops=meanpointprops0, medianprops=medianprops, capprops = capprops0, whiskerprops = whiskerprops0, meanline=False, showmeans=True)


ax.boxplot([codeTrans_item_data['dlpfc'][1][:,cp1x, cp2x]], positions=[2.3], patch_artist=True, widths = 0.2, boxprops=boxprops3, flierprops=flierprops3, 
                  meanprops=meanpointprops3, medianprops=medianprops, capprops = capprops3, whiskerprops = whiskerprops3, meanline=False, showmeans=True)
ax.boxplot([codeTrans_item_data_shuff['dlpfc'][1][:,cp1x, cp2x]], positions=[2.7], patch_artist=True, widths = 0.2, boxprops=boxprops0, flierprops=flierprops0, 
                  meanprops=meanpointprops0, medianprops=medianprops, capprops = capprops0, whiskerprops = whiskerprops0, meanline=False, showmeans=True)

ax.boxplot([codeTrans_item_data['fef'][1][:,cp1x, cp2x]], positions=[3.3], patch_artist=True, widths = 0.2, boxprops=boxprops4, flierprops=flierprops4, 
                  meanprops=meanpointprops4, medianprops=medianprops, capprops = capprops4, whiskerprops = whiskerprops4, meanline=False, showmeans=True)
ax.boxplot([codeTrans_item_data_shuff['fef'][1][:,cp1x, cp2x]], positions=[3.7], patch_artist=True, widths = 0.2, boxprops=boxprops0, flierprops=flierprops0, 
                  meanprops=meanpointprops0, medianprops=medianprops, capprops = capprops0, whiskerprops = whiskerprops0, meanline=False, showmeans=True)


p1 = f_stats.permutation_pCI(codeTrans_item_rnn['ed2'][1][:,cp1x, cp2x], codeTrans_item_shuff_rnn['ed2'][1][:,cp1x, cp2x], tail = 'greater', alpha=5)
ax.text(0.5,1.05, f'{f_plotting.sig_marker(p1,ns_note=True)}',horizontalalignment='center', verticalalignment='bottom', fontsize=10)

p2 = f_stats.permutation_pCI(codeTrans_item_rnn['ed12'][1][:,cp1x, cp2x], codeTrans_item_shuff_rnn['ed12'][1][:,cp1x, cp2x], tail = 'greater', alpha=5)
ax.text(1.5,1.05, f'{f_plotting.sig_marker(p2,ns_note=True)}',horizontalalignment='center', verticalalignment='bottom', fontsize=10)


p3 = f_stats.permutation_pCI(codeTrans_item_data['dlpfc'][1][:,cp1x, cp2x], codeTrans_item_data_shuff['dlpfc'][1][:,cp1x, cp2x], tail = 'greater', alpha=5)
ax.text(2.5,1.05, f'{f_plotting.sig_marker(p3,ns_note=True)}',horizontalalignment='center', verticalalignment='bottom', fontsize=10)

p4 = f_stats.permutation_pCI(codeTrans_item_data['fef'][1][:,cp1x, cp2x], codeTrans_item_data_shuff['fef'][1][:,cp1x, cp2x], tail = 'greater', alpha=5)
ax.text(3.5,1.05, f'{f_plotting.sig_marker(p4,ns_note=True)}',horizontalalignment='center', verticalalignment='bottom', fontsize=10)


ax.set_ylabel('I1D1 <-> I2D2', labelpad = 3, fontsize = 10)


xlims = ax.get_xlim()
lineThresholds = np.arange(xlims[0],xlims[1]+0.01,0.01)
ax.set_xticks([0.5,1.5,2.5,3.5],['R@R','R&U', 'LPFC','FEF'], rotation=0)
ax.set_xlabel('Strategy/Region', labelpad = 5, fontsize = 12)
ax.set_ylim(top=1.15, bottom=0)
ax.set_title('Between-Item Transferability, Retarget', fontsize = 12, pad=10)
plt.show()

# print stats
print(f"R@R: M(SD) = {codeTrans_item_rnn['ed2'][1][:,cp1x, cp2x].mean():.3f}({codeTrans_item_rnn['ed2'][1][:,cp1x, cp2x].std():.3f}), p = {p1:.3f};")
print(f"R&U: M(SD) = {codeTrans_item_rnn['ed12'][1][:,cp1x, cp2x].mean():.3f}({codeTrans_item_rnn['ed12'][1][:,cp1x, cp2x].std():.3f}), p = {p2:.3f};")
print(f"LPFC: M(SD) = {codeTrans_item_data['dlpfc'][1][:,cp1x, cp2x].mean():.3f}({codeTrans_item_data['dlpfc'][1][:,cp1x, cp2x].std():.3f}), p = {p3:.3f};")
print(f"FEF: M(SD) = {codeTrans_item_data['fef'][1][:,cp1x, cp2x].mean():.3f}({codeTrans_item_data['fef'][1][:,cp1x, cp2x].std():.3f}), p = {p4:.3f};")

#%% [Figure 3C] plot code transferability ratio at specific checkpoints

############################################
# code transferability ratio between items #
############################################

checkpoints = [150, 550, 1050, 1450, 1850, 2350]#
checkpointsLabels = ['S1','ED1','LD1','S2','ED2','LD2'] 
avgInterval = {150:150, 550:250, 1050:250, 1450:150, 1850:250, 2350:250} 

# cross-subspace decodability
codeTrans_item_data = {}
codeTrans_item_data_shuff = {}

for k in ('dlpfc','fef'):
    pfmX12_data = {tt:np.array([performanceX_Trans_12_data[k][tt][n] for n in range(100)]).mean(1) for tt in (1,2)}
    pfmX21_data = {tt:np.array([performanceX_Trans_21_data[k][tt][n] for n in range(100)]).mean(1) for tt in (1,2)}
    pfmX12_shuff_data = {tt:np.array([performanceX_Trans_12_shuff_data[k][tt][n] for n in range(100)]).mean(1) for tt in (1,2)}
    pfmX21_shuff_data = {tt:np.array([performanceX_Trans_21_shuff_data[k][tt][n] for n in range(100)]).mean(1) for tt in (1,2)}

    codeTrans_item_data[k] = {tt:np.array([f_decoding.code_transferability(pfmX12_data[tt][n],pfmX21_data[tt][n],checkpoints) for n in range(100)]) for tt in (1,2)}
    codeTrans_item_data_shuff[k] = {tt:np.array([f_decoding.code_transferability(pfmX12_shuff_data[tt][n],pfmX21_shuff_data[tt][n],checkpoints) for n in range(100)]) for tt in (1,2)}

        
codeTrans_item_rnn = {}
codeTrans_item_shuff_rnn = {}

for k in ('ed2','ed12'):
    pfmX12_rnn = {tt:np.array([performanceX_Trans_12_rnn[k][n][tt] for n in range(100)]).mean(1).mean(1) for tt in (1,2)}
    pfmX21_rnn = {tt:np.array([performanceX_Trans_21_rnn[k][n][tt] for n in range(100)]).mean(1).mean(1) for tt in (1,2)}
    pfmX12_shuff_rnn = {tt:np.array([performanceX_Trans_12_shuff_rnn[k][n][tt] for n in range(100)]).mean(1).mean(1) for tt in (1,2)}
    pfmX21_shuff_rnn = {tt:np.array([performanceX_Trans_21_shuff_rnn[k][n][tt] for n in range(100)]).mean(1).mean(1) for tt in (1,2)}
    
    codeTrans_item_rnn[k] = {tt:np.array([f_decoding.code_transferability(pfmX12_rnn[tt][n],pfmX21_rnn[tt][n],checkpoints) for n in range(100)]) for tt in (1,2)}
    codeTrans_item_shuff_rnn[k] = {tt:np.array([f_decoding.code_transferability(pfmX12_shuff_rnn[tt][n],pfmX21_shuff_rnn[tt][n],checkpoints) for n in range(100)]) for tt in (1,2)} 

# within-subspace decodability
codeWithin_item_data = {}
codeWithin_item_data_shuff = {}
for k in ('dlpfc','fef'):
    
    pfm11_data = {tt:performance1_item_data[k][tt].mean(1) for tt in (1,2)}
    pfm22_data = {tt:performance2_item_data[k][tt].mean(1) for tt in (1,2)}
    pfm11_shuff_data = {tt:performance1_item_shuff_data[k][tt].mean(1) for tt in (1,2)} 
    pfm22_shuff_data = {tt:performance2_item_shuff_data[k][tt].mean(1) for tt in (1,2)}

    codeWithin_item_data[k] = {1:pfm11_data, 2:pfm22_data}
    codeWithin_item_data_shuff[k] = {1:pfm11_shuff_data, 2:pfm22_shuff_data}
    
codeWithin_item_rnn = {}
codeWithin_item_shuff_rnn = {}
for k in ('ed2','ed12'):
    pfm11_rnn = {tt:np.array([performance1_item_rnn[k][n][tt].squeeze().mean(0) for n in range(100)]) for tt in (1,2)}
    pfm22_rnn = {tt:np.array([performance2_item_rnn[k][n][tt].squeeze().mean(0) for n in range(100)]) for tt in (1,2)}
    pfm11_shuff_rnn = {tt:np.array([performance1_item_shuff_rnn[k][n][tt].squeeze().mean(0) for n in range(100)]) for tt in (1,2)}
    pfm22_shuff_rnn = {tt:np.array([performance2_item_shuff_rnn[k][n][tt].squeeze().mean(0) for n in range(100)]) for tt in (1,2)}
    
    codeWithin_item_rnn[k] = {1:pfm11_rnn, 2:pfm22_rnn}
    codeWithin_item_shuff_rnn[k] = {1:pfm11_shuff_rnn, 2:pfm22_shuff_rnn}


cp1, cp2 = 'LD1', 'LD2'
cp1x, cp2x = checkpointsLabels.index(cp1), checkpointsLabels.index(cp2)

# calculate code transferability ratio: cross-subspace/within-subspace
codeTrans_ratio_data = {}
codeTrans_ratio_data_shuff = {}
codeTrans_ratio_rnn = {}
codeTrans_ratio_shuff_rnn = {}

for k in ('dlpfc','fef'):
    codeTrans_ratio_data[k] = {tt:codeTrans_item_data[k][tt][:,cp1x, cp2x]/np.mean([codeWithin_item_data[k][1][tt][:,cp1x], codeWithin_item_data[k][2][tt][:,cp2x]], axis=0) for tt in (1,2)}
    codeTrans_ratio_data_shuff[k] = {tt:codeTrans_item_data_shuff[k][tt][:,cp1x, cp2x]/np.mean([codeWithin_item_data_shuff[k][1][tt][:,cp1x], codeWithin_item_data_shuff[k][2][tt][:,cp2x]], axis=0) for tt in (1,2)}

for k in ('ed2','ed12'):
    codeTrans_ratio_rnn[k] = {tt:codeTrans_item_rnn[k][tt][:,cp1x, cp2x]/np.mean([codeWithin_item_rnn[k][1][tt][:,cp1x], codeWithin_item_rnn[k][2][tt][:,cp2x]], axis=0) for tt in (1,2)}
    codeTrans_ratio_shuff_rnn[k] = {tt:codeTrans_item_shuff_rnn[k][tt][:,cp1x, cp2x]/np.mean([codeWithin_item_shuff_rnn[k][1][tt][:,cp1x], codeWithin_item_shuff_rnn[k][2][tt][:,cp2x]], axis=0) for tt in (1,2)}


pltBsl = True

lineh = np.arange(0.3,0.7,0.001)
linev = np.arange(1.16,1.165,0.0001)
fig, axes = plt.subplots(1,1, sharey=True, figsize=(3,3), dpi=300)

# full space
ax = axes#.flatten()[0]

ax.boxplot([codeTrans_ratio_rnn['ed2'][1]], positions=[0.5-pltBsl*0.2], patch_artist=True, widths = 0.2, boxprops=boxprops1, flierprops=flierprops1, 
                  meanprops=meanpointprops1, medianprops=medianprops, capprops = capprops1, whiskerprops = whiskerprops1, meanline=False, showmeans=True)
ax.boxplot([codeTrans_ratio_rnn['ed12'][1]], positions=[1.5-pltBsl*0.2], patch_artist=True, widths = 0.2, boxprops=boxprops2, flierprops=flierprops2, 
                  meanprops=meanpointprops2, medianprops=medianprops, capprops = capprops2, whiskerprops = whiskerprops2, meanline=False, showmeans=True)
ax.boxplot([codeTrans_ratio_data['dlpfc'][1]], positions=[2.5-pltBsl*0.2], patch_artist=True, widths = 0.2, boxprops=boxprops3, flierprops=flierprops3, 
                  meanprops=meanpointprops3, medianprops=medianprops, capprops = capprops3, whiskerprops = whiskerprops3, meanline=False, showmeans=True)
ax.boxplot([codeTrans_ratio_data['fef'][1]], positions=[3.5-pltBsl*0.2], patch_artist=True, widths = 0.2, boxprops=boxprops4, flierprops=flierprops4, 
                  meanprops=meanpointprops4, medianprops=medianprops, capprops = capprops4, whiskerprops = whiskerprops4, meanline=False, showmeans=True)

# whether to include baseline distributions or not
if pltBsl:
    ax.boxplot([codeTrans_ratio_shuff_rnn['ed2'][1]], positions=[0.5+pltBsl*0.2], patch_artist=True, widths = 0.2, boxprops=boxprops0, flierprops=flierprops0, 
                    meanprops=meanpointprops0, medianprops=medianprops, capprops = capprops0, whiskerprops = whiskerprops0, meanline=False, showmeans=True)
    ax.boxplot([codeTrans_ratio_shuff_rnn['ed12'][1]], positions=[1.5+pltBsl*0.2], patch_artist=True, widths = 0.2, boxprops=boxprops0, flierprops=flierprops0, 
                    meanprops=meanpointprops0, medianprops=medianprops, capprops = capprops0, whiskerprops = whiskerprops0, meanline=False, showmeans=True)
    ax.boxplot([codeTrans_ratio_data_shuff['dlpfc'][1]], positions=[2.5+pltBsl*0.2], patch_artist=True, widths = 0.2, boxprops=boxprops0, flierprops=flierprops0, 
                    meanprops=meanpointprops0, medianprops=medianprops, capprops = capprops0, whiskerprops = whiskerprops0, meanline=False, showmeans=True)
    ax.boxplot([codeTrans_ratio_data_shuff['fef'][1]], positions=[3.5+pltBsl*0.2], patch_artist=True, widths = 0.2, boxprops=boxprops0, flierprops=flierprops0, 
                    meanprops=meanpointprops0, medianprops=medianprops, capprops = capprops0, whiskerprops = whiskerprops0, meanline=False, showmeans=True)


p1 = f_stats.permutation_pCI(codeTrans_ratio_rnn['ed2'][1], codeTrans_ratio_shuff_rnn['ed2'][1], tail = 'smaller', alpha=5)
ax.text(0.5,1.8, f'{f_plotting.sig_marker(p1,ns_note=True)}',horizontalalignment='center', verticalalignment='bottom', fontsize=10)

p2 = f_stats.permutation_pCI(codeTrans_ratio_rnn['ed12'][1], codeTrans_ratio_shuff_rnn['ed12'][1], tail = 'smaller', alpha=5)
ax.text(1.5,1.8, f'{f_plotting.sig_marker(p2,ns_note=True)}',horizontalalignment='center', verticalalignment='bottom', fontsize=10)

p3 = f_stats.permutation_pCI(codeTrans_ratio_data['dlpfc'][1], codeTrans_ratio_data_shuff['dlpfc'][1], tail = 'smaller', alpha=5)
ax.text(2.5,1.8, f'{f_plotting.sig_marker(p3,ns_note=True)}',horizontalalignment='center', verticalalignment='bottom', fontsize=10)

p4 = f_stats.permutation_pCI(codeTrans_ratio_data['fef'][1], codeTrans_ratio_data_shuff['fef'][1], tail = 'smaller', alpha=5)
ax.text(3.5,1.8, f'{f_plotting.sig_marker(p4,ns_note=True)}',horizontalalignment='center', verticalalignment='bottom', fontsize=10)


ax.set_ylabel('Cross- / Within-Subspace', labelpad = 3, fontsize = 10)


xlims = ax.get_xlim()
lineThresholds = np.arange(xlims[0],xlims[1]+0.01,0.01)
ax.set_xticks([0.5,1.5,2.5,3.5],['R@R','R&U', 'LPFC','FEF'], rotation=0)
ax.set_xlabel('Strategy/Region', labelpad = 5, fontsize = 12)
ax.set_ylim(top=2, bottom=0)
ax.set_title('Between-Item Transferability Ratio, Retarget', fontsize = 12, pad=10)
plt.show()

# print stats
print(f"R@R: M(SD) = {codeTrans_ratio_rnn['ed2'][1].mean():.3f}({codeTrans_ratio_rnn['ed2'][1].std():.3f}), p = {p1:.3f};")
print(f"R&U: M(SD) = {codeTrans_ratio_rnn['ed12'][1].mean():.3f}({codeTrans_ratio_rnn['ed12'][1].std():.3f}), p = {p2:.3f};")
print(f"LPFC: M(SD) = {codeTrans_ratio_data['dlpfc'][1].mean():.3f}({codeTrans_ratio_data['dlpfc'][1].std():.3f}), p = {p3:.3f};")
print(f"FEF: M(SD) = {codeTrans_ratio_data['fef'][1].mean():.3f}({codeTrans_ratio_data['fef'][1].std():.3f}), p = {p4:.3f};")

# [Non-used] KS test 
codeTrans_ratio_ks = {'R@R-LPFC':scipy.stats.ks_2samp(codeTrans_ratio_rnn['ed2'][1], codeTrans_ratio_data['dlpfc'][1]), 
                        'R@R-FEF':scipy.stats.ks_2samp(codeTrans_ratio_rnn['ed2'][1], codeTrans_ratio_data['fef'][1]), 
                        'R&U-LPFC':scipy.stats.ks_2samp(codeTrans_ratio_rnn['ed12'][1], codeTrans_ratio_data['dlpfc'][1]), 
                        'R&U-FEF':scipy.stats.ks_2samp(codeTrans_ratio_rnn['ed12'][1], codeTrans_ratio_data['fef'][1])}
print('############### K-S Test ##############')
for k in codeTrans_ratio_ks.keys():
    print(f"{k}: D = {codeTrans_ratio_ks[k].statistic:.3f}, p = {codeTrans_ratio_ks[k].pvalue:.3f};")



#%%



#######################################
# Choice Item, I2D2, T/T vs. I1D2,T/D #
#######################################




#%% [Figure S3A, S3B] plot subspace coplanarity and alignment at specific checkpoints
# I1D2-Distraction vs I2D2-Retarget
cosThetaChoice_rnn = {k:np.array([cosTheta_choice_rnn[k][n] for n in range(100)]).mean(1) for k in ('ed2','ed12')}
cosThetaChoice_data = {k:cosTheta_choice_data[k].mean(1) for k in ('dlpfc','fef')}
cosThetaChoice_shuff_rnn = {k:np.array([cosTheta_choice_shuff_rnn[k][n] for n in range(100)]).mean(1) for k in ('ed2','ed12')}
cosThetaChoice_shuff_data = {k:cosTheta_choice_shuff_data[k] for k in ('dlpfc','fef')}

cosPsiChoice_rnn = {k:np.array([cosPsi_choice_rnn[k][n] for n in range(100)]).mean(1) for k in ('ed2','ed12')}
cosPsiChoice_data = {k:cosPsi_choice_data[k].mean(1) for k in ('dlpfc','fef')}
cosPsiChoice_shuff_rnn = {k:np.array([cosPsi_choice_shuff_rnn[k][n] for n in range(100)]).mean(1) for k in ('ed2','ed12')}
cosPsiChoice_shuff_data = {k:cosPsi_choice_shuff_data[k] for k in ('dlpfc','fef')}


angleCheckPoints = np.linspace(0,np.pi,7).round(5)
cmap = plt.get_cmap('coolwarm')
norm = mcolors.BoundaryNorm(np.flip(np.cos(angleCheckPoints).round(5)),cmap.N, clip=True)

showCheckpoints = ('L',) # just include the late delay periods
showShuffBsl = True

fig, axes = plt.subplots(1,2, sharey=True, figsize=(8,4), dpi=300)

# coplanarity
ax = axes.flatten()[0]
xtks, xtklabs = [], []
for nk, k in enumerate(showCheckpoints):
    dk2 = k+'D2'
    d2x = checkpointsLabels.index(dk2)
    
    print('############ cosTheta ############')
    cosTheta_bsl_rnn = {kk: np.mean((np.array([cosTheta11_bsl_rnn[kk][n][2] for n in range(100)]).squeeze()[:,:,d2x].mean(1),
                                   np.array([cosTheta22_bsl_rnn[kk][n][1] for n in range(100)]).squeeze()[:,:,d2x].mean(1)),axis=0) for kk in ('ed2','ed12')}
    cosTheta_bsl_data = {kk: np.mean((cosTheta11_bsl_data[kk][2][:,:,d2x].mean(1), 
                                     cosTheta22_bsl_data[kk][1][:,:,d2x].mean(1)),axis=0) for kk in ('dlpfc','fef')}
    
    
    ax.boxplot([cosThetaChoice_rnn['ed2'][:,d2x]], positions=[0.2+nk-0.035*showShuffBsl], patch_artist=True, widths = 0.05*len(showCheckpoints), boxprops=boxprops1, flierprops=flierprops1, 
                    meanprops=meanpointprops1, medianprops=medianprops, capprops = capprops1, whiskerprops = whiskerprops1, meanline=False, showmeans=True)
    p1 = f_stats.permutation_pCI(cosThetaChoice_rnn['ed2'][:,d2x], cosTheta_bsl_rnn['ed2'], tail='smaller')
    ax.text(0.2+nk,1., f'{f_plotting.sig_marker(p1,ns_note=False)}',horizontalalignment='center', verticalalignment='bottom', fontsize=10)
    print(f"R@R: M(SD) = {cosThetaChoice_rnn['ed2'][:,d2x].mean():.3f} ({cosThetaChoice_rnn['ed2'][:,d2x].std():.3f}), 95CI = [{np.quantile(cosThetaChoice_rnn['ed2'][:,d2x],0.025):.3f}, {np.quantile(cosThetaChoice_rnn['ed2'][:,d2x],0.975):.3f}], p = {p1:.3f}")

    
    ax.boxplot([cosThetaChoice_rnn['ed12'][:,d2x]], positions=[0.4+nk-0.035*showShuffBsl], patch_artist=True, widths = 0.05*len(showCheckpoints), boxprops=boxprops2, flierprops=flierprops2, 
                    meanprops=meanpointprops2, medianprops=medianprops, capprops = capprops2, whiskerprops = whiskerprops2, meanline=False, showmeans=True)
    p2 = f_stats.permutation_pCI(cosThetaChoice_rnn['ed12'][:,d2x], cosTheta_bsl_rnn['ed12'], tail='smaller')
    ax.text(0.4+nk,1., f'{f_plotting.sig_marker(p2,ns_note=False)}',horizontalalignment='center', verticalalignment='bottom', fontsize=10)
    print(f"R&U: M(SD) = {cosThetaChoice_rnn['ed12'][:,d2x].mean():.3f} ({cosThetaChoice_rnn['ed12'][:,d2x].std():.3f}), 95CI = [{np.quantile(cosThetaChoice_rnn['ed12'][:,d2x],0.025):.3f}, {np.quantile(cosThetaChoice_rnn['ed12'][:,d2x],0.975):.3f}], p = {p2:.3f}")

    
    ax.boxplot([cosThetaChoice_data['dlpfc'][:,d2x]], positions=[0.6+nk-0.035*showShuffBsl], patch_artist=True, widths = 0.05*len(showCheckpoints), boxprops=boxprops3, flierprops=flierprops3, 
                    meanprops=meanpointprops3, medianprops=medianprops, capprops = capprops3, whiskerprops = whiskerprops3, meanline=False, showmeans=True)
    p3 = f_stats.permutation_pCI(cosThetaChoice_data['dlpfc'][:,d2x], cosTheta_bsl_data['dlpfc'], tail='smaller')
    ax.text(0.6+nk,1., f'{f_plotting.sig_marker(p3,ns_note=False)}',horizontalalignment='center', verticalalignment='bottom', fontsize=10)
    print(f"LPFC: M(SD) = {cosThetaChoice_data['dlpfc'][:,d2x].mean():.3f} ({cosThetaChoice_data['dlpfc'][:,d2x].std():.3f}), 95CI = [{np.quantile(cosThetaChoice_data['dlpfc'][:,d2x],0.025):.3f}, {np.quantile(cosThetaChoice_data['dlpfc'][:,d2x],0.975):.3f}], p = {p3:.3f}")

    
    ax.boxplot([cosThetaChoice_data['fef'][:,d2x]], positions=[0.8+nk-0.035*showShuffBsl], patch_artist=True, widths = 0.05*len(showCheckpoints), boxprops=boxprops4, flierprops=flierprops4, 
                    meanprops=meanpointprops4, medianprops=medianprops, capprops = capprops4, whiskerprops = whiskerprops4, meanline=False, showmeans=True)
    p4 = f_stats.permutation_pCI(cosThetaChoice_data['fef'][:,d2x], cosTheta_bsl_data['fef'], tail='smaller')
    ax.text(0.8+nk,1., f'{f_plotting.sig_marker(p4,ns_note=False)}',horizontalalignment='center', verticalalignment='bottom', fontsize=10)
    print(f"FEF: M(SD) = {cosThetaChoice_data['fef'][:,d2x].mean():.3f} ({cosThetaChoice_data['fef'][:,d2x].std():.3f}), 95CI = [{np.quantile(cosThetaChoice_data['fef'][:,d2x],0.025):.3f}, {np.quantile(cosThetaChoice_data['fef'][:,d2x],0.975):.3f}], p = {p4:.3f}")

    
    # whether to include baseline distributions or not
    if showShuffBsl:
        ax.boxplot([cosTheta_bsl_rnn['ed2']], positions=[0.2+nk+0.035*showShuffBsl], patch_artist=True, widths = 0.05*len(showCheckpoints), boxprops=boxprops0, flierprops=flierprops0, 
                    meanprops=meanpointprops0, medianprops=medianprops, capprops = capprops0, whiskerprops = whiskerprops0, meanline=False, showmeans=True)
        
        ax.boxplot([cosTheta_bsl_rnn['ed12']], positions=[0.4+nk+0.035*showShuffBsl], patch_artist=True, widths = 0.05*len(showCheckpoints), boxprops=boxprops0, flierprops=flierprops0, 
                        meanprops=meanpointprops0, medianprops=medianprops, capprops = capprops0, whiskerprops = whiskerprops0, meanline=False, showmeans=True)
        
        ax.boxplot([cosTheta_bsl_data['dlpfc']], positions=[0.6+nk+0.035*showShuffBsl], patch_artist=True, widths = 0.05*len(showCheckpoints), boxprops=boxprops0, flierprops=flierprops0, 
                        meanprops=meanpointprops0, medianprops=medianprops, capprops = capprops0, whiskerprops = whiskerprops0, meanline=False, showmeans=True)
        
        ax.boxplot([cosTheta_bsl_data['fef']], positions=[0.8+nk+0.035*showShuffBsl], patch_artist=True, widths = 0.05*len(showCheckpoints), boxprops=boxprops0, flierprops=flierprops0, 
                        meanprops=meanpointprops0, medianprops=medianprops, capprops = capprops0, whiskerprops = whiskerprops0, meanline=False, showmeans=True)
        
    
    xtks += [0.5+nk]
    xtklabs += [f'{dk2}']
    
    # [Non-used] KS Test
    ks_results = {'R@R-LPFC':scipy.stats.ks_2samp(cosThetaChoice_rnn['ed2'][:,d2x], cosThetaChoice_data['dlpfc'][:,d2x]),
                  'R@R-FEF':scipy.stats.ks_2samp(cosThetaChoice_rnn['ed2'][:,d2x], cosThetaChoice_data['fef'][:,d2x]),
                  'R&U-LPFC':scipy.stats.ks_2samp(cosThetaChoice_rnn['ed12'][:,d2x], cosThetaChoice_data['dlpfc'][:,d2x]), 
                  'R&U-FEF':scipy.stats.ks_2samp(cosThetaChoice_rnn['ed12'][:,d2x], cosThetaChoice_data['fef'][:,d2x])}

    print('############### K-S Test ##############')
    for k in ks_results.keys():
        print(f'{k.upper()}: D = {ks_results[k][0]:.3f}, p = {ks_results[k][1]:.3f}')
    

ax.set_xticks(xtks,xtklabs)#,rotation=20
ax.set_xlim(0,len(showCheckpoints))
ax.set_ylim(-1.05,1.2)
ax.set_yticks(np.flip(np.cos(angleCheckPoints).round(5)), labels = np.flip(np.degrees(angleCheckPoints).round(1).astype(int)))
ax.set_xlabel('Timebins', labelpad = 3, fontsize = 15)
ax.set_ylabel('cos(θ)', labelpad = 0, fontsize = 15)
ax.set_title(f'Principal Angle', fontsize = 17)
ax.tick_params(axis='both', labelsize=12)
print('#################################')
print('\n')

# aligment
ax = axes.flatten()[1]
xtks, xtklabs = [], []
for nk, k in enumerate(showCheckpoints):
    dk2 = k+'D2'
    d2x = checkpointsLabels.index(dk2)

    print('############### cosPsi ##############')
    cosPsi_bsl_rnn = {kk: np.mean((np.array([cosPsi11_bsl_rnn[kk][n][2] for n in range(100)]).squeeze()[:,:,d2x].mean(1),
                                   np.array([cosPsi22_bsl_rnn[kk][n][1] for n in range(100)]).squeeze()[:,:,d2x].mean(1)),axis=0) for kk in ('ed2','ed12')}
    cosPsi_bsl_data = {kk: np.mean((cosPsi11_bsl_data[kk][2][:,:,d2x].mean(1), 
                                     cosPsi22_bsl_data[kk][1][:,:,d2x].mean(1)),axis=0) for kk in ('dlpfc','fef')}
    
    
    ax.boxplot([cosPsiChoice_rnn['ed2'][:,d2x]], positions=[0.2+nk-0.035*showShuffBsl], patch_artist=True, widths = 0.05*len(showCheckpoints), boxprops=boxprops1, flierprops=flierprops1, 
                    meanprops=meanpointprops1, medianprops=medianprops, capprops = capprops1, whiskerprops = whiskerprops1, meanline=False, showmeans=True)
    p1 = f_stats.permutation_pCI(cosPsiChoice_rnn['ed2'][:,d2x], cosPsi_bsl_rnn['ed2'], tail='smaller', alpha = 5)
    ax.text(0.2+nk,1., f'{f_plotting.sig_marker(p1,ns_note=False)}',horizontalalignment='center', verticalalignment='bottom', fontsize=10)
    print(f"R@R: M(SD) = {cosPsiChoice_rnn['ed2'][:,d2x].mean():.3f} ({cosPsiChoice_rnn['ed2'][:,d2x].std():.3f}), 95CI = [{np.quantile(cosPsiChoice_rnn['ed2'][:,d2x], 0.025):.3f}, {np.quantile(cosPsiChoice_rnn['ed2'][:,d2x], 0.975):.3f}], p = {p1:.3f}")

    
    ax.boxplot([cosPsiChoice_rnn['ed12'][:,d2x]], positions=[0.4+nk-0.035*showShuffBsl], patch_artist=True, widths = 0.05*len(showCheckpoints), boxprops=boxprops2, flierprops=flierprops2, 
                    meanprops=meanpointprops2, medianprops=medianprops, capprops = capprops2, whiskerprops = whiskerprops2, meanline=False, showmeans=True)
    p2 = f_stats.permutation_pCI(cosPsiChoice_rnn['ed12'][:,d2x], cosPsi_bsl_rnn['ed12'], tail='smaller', alpha = 5)
    ax.text(0.4+nk,1., f'{f_plotting.sig_marker(p2,ns_note=False)}',horizontalalignment='center', verticalalignment='bottom', fontsize=10)
    print(f"R&U: M(SD) = {cosPsiChoice_rnn['ed12'][:,d2x].mean():.3f} ({cosPsiChoice_rnn['ed12'][:,d2x].std():.3f}), 95CI = [{np.quantile(cosPsiChoice_rnn['ed12'][:,d2x], 0.025):.3f}, {np.quantile(cosPsiChoice_rnn['ed12'][:,d2x], 0.975):.3f}], p = {p2:.3f}")

    
    ax.boxplot([cosPsiChoice_data['dlpfc'][:,d2x]], positions=[0.6+nk-0.035*showShuffBsl], patch_artist=True, widths = 0.05*len(showCheckpoints), boxprops=boxprops3, flierprops=flierprops3, 
                    meanprops=meanpointprops3, medianprops=medianprops, capprops = capprops3, whiskerprops = whiskerprops3, meanline=False, showmeans=True)
    p3 = f_stats.permutation_pCI(cosPsiChoice_data['dlpfc'][:,d2x], cosPsi_bsl_data['dlpfc'], tail='smaller', alpha = 5)
    ax.text(0.6+nk,1., f'{f_plotting.sig_marker(p3,ns_note=False)}',horizontalalignment='center', verticalalignment='bottom', fontsize=10)
    print(f"LPFC: M(SD) = {cosPsiChoice_data['dlpfc'][:,d2x].mean():.3f} ({cosPsiChoice_data['dlpfc'][:,d2x].std():.3f}), 95CI = [{np.quantile(cosPsiChoice_data['dlpfc'][:,d2x], 0.025):.3f}, {np.quantile(cosPsiChoice_data['dlpfc'][:,d2x], 0.975):.3f}], p = {p3:.3f}")

    
    ax.boxplot([cosPsiChoice_data['fef'][:,d2x]], positions=[0.8+nk-0.035*showShuffBsl], patch_artist=True, widths = 0.05*len(showCheckpoints), boxprops=boxprops4, flierprops=flierprops4, 
                    meanprops=meanpointprops4, medianprops=medianprops, capprops = capprops4, whiskerprops = whiskerprops4, meanline=False, showmeans=True)
    p4 = f_stats.permutation_pCI(cosPsiChoice_data['fef'][:,d2x], cosPsi_bsl_data['fef'], tail='smaller', alpha = 5)
    ax.text(0.8+nk,1., f'{f_plotting.sig_marker(p4,ns_note=False)}',horizontalalignment='center', verticalalignment='bottom', fontsize=10)
    print(f"FEF: M(SD) = {cosPsiChoice_data['fef'][:,d2x].mean():.3f} ({cosPsiChoice_data['fef'][:,d2x].std():.3f}), 95CI = [{np.quantile(cosPsiChoice_data['fef'][:,d2x], 0.025):.3f}, {np.quantile(cosPsiChoice_data['fef'][:,d2x], 0.975):.3f}], p = {p4:.3f}")

    
    # whether to include baseline distributions or not
    if showShuffBsl:
        ax.boxplot([cosPsi_bsl_rnn['ed2']], positions=[0.2+nk+0.035*showShuffBsl], patch_artist=True, widths = 0.05*len(showCheckpoints), boxprops=boxprops0, flierprops=flierprops0, 
                    meanprops=meanpointprops0, medianprops=medianprops, capprops = capprops0, whiskerprops = whiskerprops0, meanline=False, showmeans=True)
        
        ax.boxplot([cosPsi_bsl_rnn['ed12']], positions=[0.4+nk+0.035*showShuffBsl], patch_artist=True, widths = 0.05*len(showCheckpoints), boxprops=boxprops0, flierprops=flierprops0, 
                        meanprops=meanpointprops0, medianprops=medianprops, capprops = capprops0, whiskerprops = whiskerprops0, meanline=False, showmeans=True)
        
        ax.boxplot([cosPsi_bsl_data['dlpfc']], positions=[0.6+nk+0.035*showShuffBsl], patch_artist=True, widths = 0.05*len(showCheckpoints), boxprops=boxprops0, flierprops=flierprops0, 
                        meanprops=meanpointprops0, medianprops=medianprops, capprops = capprops0, whiskerprops = whiskerprops0, meanline=False, showmeans=True)
        
        ax.boxplot([cosPsi_bsl_data['fef']], positions=[0.8+nk+0.035*showShuffBsl], patch_artist=True, widths = 0.05*len(showCheckpoints), boxprops=boxprops0, flierprops=flierprops0, 
                        meanprops=meanpointprops0, medianprops=medianprops, capprops = capprops0, whiskerprops = whiskerprops0, meanline=False, showmeans=True)
        
        
    xtks += [0.5+nk]
    xtklabs += [f'{dk2}']

    
    # [Non-used] KS Test
    ks_results = {'R@R-LPFC':scipy.stats.ks_2samp(cosPsiChoice_rnn['ed2'][:,d2x], cosPsiChoice_data['dlpfc'][:,d2x]),
                  'R@R-FEF':scipy.stats.ks_2samp(cosPsiChoice_rnn['ed2'][:,d2x], cosPsiChoice_data['fef'][:,d2x]),
                  'R&U-LPFC':scipy.stats.ks_2samp(cosPsiChoice_rnn['ed12'][:,d2x], cosPsiChoice_data['dlpfc'][:,d2x]), 
                  'R&U-FEF':scipy.stats.ks_2samp(cosPsiChoice_rnn['ed12'][:,d2x], cosPsiChoice_data['fef'][:,d2x])}

    print('############### K-S Test ##############')
    for k in ks_results.keys():
        print(f'{k.upper()}: D = {ks_results[k][0]:.3f}, p = {ks_results[k][1]:.3f}')
    
ax.set_xticks(xtks,xtklabs)#,rotation=20
ax.set_xlim(0,len(showCheckpoints))
ax.set_ylim(-1.05,1.2)
ax.set_yticks(np.flip(np.cos(angleCheckPoints).round(5)), labels = np.flip(np.degrees(angleCheckPoints).round(1).astype(int)))
ax.set_xlabel('Timebins', labelpad = 3, fontsize = 15)
ax.set_ylabel('cos(ψ)', labelpad = 0, fontsize = 15)
ax.set_title(f'Representational Alignment', fontsize = 17)
ax.tick_params(axis='both', labelsize=12)
print('#################################')
print('\n')


ax.plot([], c=color1, label='R@R')
ax.plot([], c=color2, label='R&U')
ax.plot([], c=color3, label='LPFC')
ax.plot([], c=color4, label='FEF')
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize = 12)


plt.suptitle(f'I2D2-Retarget vs. I1D2-Distraction', fontsize = 25, y=1) # (Monkeys)
plt.tight_layout()
plt.show()


#%% plot code transferability at specific checkpoints

######################################################
# code transferability between target-item subspaces #
######################################################


# load precomputed code transferability from files
performanceX_Trans_rdc_data = np.load(f'{data_path}/' + 'performanceX_Trans_rdc_data.npy', allow_pickle=True).item()
performanceX_Trans_drc_data = np.load(f'{data_path}/' + 'performanceX_Trans_drc_data.npy', allow_pickle=True).item()
performanceX_Trans_rdc_shuff_data = np.load(f'{data_path}/' + 'performanceX_Trans_rdc_shuff_data.npy', allow_pickle=True).item()
performanceX_Trans_drc_shuff_data = np.load(f'{data_path}/' + 'performanceX_Trans_drc_shuff_data.npy', allow_pickle=True).item()

performanceX_Trans_rdc_rnn = np.load(f'{data_path}/' + 'performanceX_rdc_rnn.npy', allow_pickle=True).item()
performanceX_Trans_drc_rnn = np.load(f'{data_path}/' + 'performanceX_drc_rnn.npy', allow_pickle=True).item()
performanceX_Trans_rdc_shuff_rnn = np.load(f'{data_path}/' + 'performanceX_rdc_shuff_rnn.npy', allow_pickle=True).item()
performanceX_Trans_drc_shuff_rnn = np.load(f'{data_path}/' + 'performanceX_drc_shuff_rnn.npy', allow_pickle=True).item()

performanceX_Trans_rdnc_data = np.load(f'{data_path}/' + 'performanceX_Trans_rdnc_data.npy', allow_pickle=True).item()
performanceX_Trans_drnc_data = np.load(f'{data_path}/' + 'performanceX_Trans_drnc_data.npy', allow_pickle=True).item()
performanceX_Trans_rdnc_shuff_data = np.load(f'{data_path}/' + 'performanceX_Trans_rdnc_shuff_data.npy', allow_pickle=True).item()
performanceX_Trans_drnc_shuff_data = np.load(f'{data_path}/' + 'performanceX_Trans_drnc_shuff_data.npy', allow_pickle=True).item()

performanceX_Trans_rdnc_rnn = np.load(f'{data_path}/' + 'performanceX_rdnc_rnn.npy', allow_pickle=True).item()
performanceX_Trans_drnc_rnn = np.load(f'{data_path}/' + 'performanceX_drnc_rnn.npy', allow_pickle=True).item()
performanceX_Trans_rdnc_shuff_rnn = np.load(f'{data_path}/' + 'performanceX_rdnc_shuff_rnn.npy', allow_pickle=True).item()
performanceX_Trans_drnc_shuff_rnn = np.load(f'{data_path}/' + 'performanceX_drnc_shuff_rnn.npy', allow_pickle=True).item()


performance1_item_data = np.load(f'{data_path}/' + 'performance1_item_data.npy', allow_pickle=True).item()
performance2_item_data = np.load(f'{data_path}/' + 'performance2_item_data.npy', allow_pickle=True).item()
performance1_item_shuff_data = np.load(f'{data_path}/' + 'performance1_item_shuff_data.npy', allow_pickle=True).item()
performance2_item_shuff_data = np.load(f'{data_path}/' + 'performance2_item_shuff_data.npy', allow_pickle=True).item()

performance1_item_rnn = np.load(f'{data_path}/' + 'performance1_item_rnn.npy', allow_pickle=True).item()
performance2_item_rnn = np.load(f'{data_path}/' + 'performance2_item_rnn.npy', allow_pickle=True).item()
performance1_item_shuff_rnn = np.load(f'{data_path}/' + 'performance1_item_shuff_rnn.npy', allow_pickle=True).item()
performance2_item_shuff_rnn = np.load(f'{data_path}/' + 'performance2_item_shuff_rnn.npy', allow_pickle=True).item()

checkpoints = [150, 550, 1150, 1450, 1850, 2350]#
checkpointsLabels = ['S1','ED1','LD1','S2','ED2','LD2'] 
avgInterval = {150:150, 550:250, 1150:250, 1450:150, 1850:250, 2350:250} 

codeTrans_choice_data = {}
codeTrans_choice_data_shuff = {}

for k in ('dlpfc','fef'):
    pfmX_rdc_data = np.array([performanceX_Trans_rdc_data[k][n] for n in range(100)]).mean(1)
    pfmX_drc_data = np.array([performanceX_Trans_drc_data[k][n] for n in range(100)]).mean(1)
    pfmX_rdc_shuff_data = np.array([performanceX_Trans_rdc_shuff_data[k][n] for n in range(100)]).mean(1)
    pfmX_drc_shuff_data = np.array([performanceX_Trans_drc_shuff_data[k][n] for n in range(100)]).mean(1)

    codeTrans_choice_data[k] = np.array([f_decoding.code_transferability(pfmX_rdc_data[n],pfmX_drc_data[n],checkpoints) for n in range(100)])
    codeTrans_choice_data_shuff[k] = np.array([f_decoding.code_transferability(pfmX_rdc_shuff_data[n],pfmX_drc_shuff_data[n],checkpoints) for n in range(100)])

        
codeTrans_choice_rnn = {}
codeTrans_choice_shuff_rnn = {}

for k in ('ed2','ed12'):
    pfmX_rdc_rnn = np.array([performanceX_Trans_rdc_rnn[k][n] for n in range(100)]).mean(1).mean(1)
    pfmX_drc_rnn = np.array([performanceX_Trans_drc_rnn[k][n] for n in range(100)]).mean(1).mean(1)
    pfmX_rdc_shuff_rnn = np.array([performanceX_Trans_rdc_shuff_rnn[k][n] for n in range(100)]).mean(1).mean(1)
    pfmX_drc_shuff_rnn = np.array([performanceX_Trans_drc_shuff_rnn[k][n] for n in range(100)]).mean(1).mean(1)
    
    codeTrans_choice_rnn[k] = np.array([f_decoding.code_transferability(pfmX_rdc_rnn[n],pfmX_drc_rnn[n],checkpoints) for n in range(100)])
    codeTrans_choice_shuff_rnn[k] = np.array([f_decoding.code_transferability(pfmX_rdc_shuff_rnn[n],pfmX_drc_shuff_rnn[n],checkpoints) for n in range(100)])

cp1, cp2 = 'LD2', 'LD2'
cp1x, cp2x = checkpointsLabels.index(cp1), checkpointsLabels.index(cp2)

lineh = np.arange(0.3,0.7,0.001)
linev = np.arange(1.16,1.165,0.0001)

fig, axes = plt.subplots(1,1, sharey=True, figsize=(3,3), dpi=300)

# plot
ax = axes#.flatten()[0]
ax.boxplot([codeTrans_choice_rnn['ed2'][:,cp1x, cp2x]], positions=[0.3], patch_artist=True, widths = 0.2, boxprops=boxprops1, flierprops=flierprops1, 
                  meanprops=meanpointprops1, medianprops=medianprops, capprops = capprops1, whiskerprops = whiskerprops1, meanline=False, showmeans=True)
ax.boxplot([codeTrans_choice_shuff_rnn['ed2'][:,cp1x, cp2x]], positions=[0.7], patch_artist=True, widths = 0.2, boxprops=boxprops0, flierprops=flierprops0, 
                  meanprops=meanpointprops0, medianprops=medianprops, capprops = capprops0, whiskerprops = whiskerprops0, meanline=False, showmeans=True)
ax.boxplot([codeTrans_choice_rnn['ed12'][:,cp1x, cp2x]], positions=[1.3], patch_artist=True, widths = 0.2, boxprops=boxprops2, flierprops=flierprops2, 
                  meanprops=meanpointprops2, medianprops=medianprops, capprops = capprops2, whiskerprops = whiskerprops2, meanline=False, showmeans=True)
ax.boxplot([codeTrans_choice_shuff_rnn['ed12'][:,cp1x, cp2x]], positions=[1.7], patch_artist=True, widths = 0.2, boxprops=boxprops0, flierprops=flierprops0, 
                  meanprops=meanpointprops0, medianprops=medianprops, capprops = capprops0, whiskerprops = whiskerprops0, meanline=False, showmeans=True)


ax.boxplot([codeTrans_choice_data['dlpfc'][:,cp1x, cp2x]], positions=[2.3], patch_artist=True, widths = 0.2, boxprops=boxprops3, flierprops=flierprops3, 
                  meanprops=meanpointprops3, medianprops=medianprops, capprops = capprops3, whiskerprops = whiskerprops3, meanline=False, showmeans=True)
ax.boxplot([codeTrans_choice_data_shuff['dlpfc'][:,cp1x, cp2x]], positions=[2.7], patch_artist=True, widths = 0.2, boxprops=boxprops0, flierprops=flierprops0, 
                  meanprops=meanpointprops0, medianprops=medianprops, capprops = capprops0, whiskerprops = whiskerprops0, meanline=False, showmeans=True)

ax.boxplot([codeTrans_choice_data['fef'][:,cp1x, cp2x]], positions=[3.3], patch_artist=True, widths = 0.2, boxprops=boxprops4, flierprops=flierprops4, 
                  meanprops=meanpointprops4, medianprops=medianprops, capprops = capprops4, whiskerprops = whiskerprops4, meanline=False, showmeans=True)
ax.boxplot([codeTrans_choice_data_shuff['fef'][:,cp1x, cp2x]], positions=[3.7], patch_artist=True, widths = 0.2, boxprops=boxprops0, flierprops=flierprops0, 
                  meanprops=meanpointprops0, medianprops=medianprops, capprops = capprops0, whiskerprops = whiskerprops0, meanline=False, showmeans=True)


# significance test
p1 = f_stats.permutation_pCI(codeTrans_choice_rnn['ed2'][:,cp1x, cp2x], codeTrans_choice_shuff_rnn['ed2'][:,cp1x, cp2x], tail = 'greater', alpha=5)
ax.text(0.5,1.05, f'{f_plotting.sig_marker(p1,ns_note=True)}',horizontalalignment='center', verticalalignment='bottom', fontsize=10)
p2 = f_stats.permutation_pCI(codeTrans_choice_rnn['ed12'][:,cp1x, cp2x], codeTrans_choice_shuff_rnn['ed12'][:,cp1x, cp2x], tail = 'greater', alpha=5)
ax.text(1.5,1.05, f'{f_plotting.sig_marker(p2,ns_note=True)}',horizontalalignment='center', verticalalignment='bottom', fontsize=10)
p3 = f_stats.permutation_pCI(codeTrans_choice_data['dlpfc'][:,cp1x, cp2x], codeTrans_choice_data_shuff['dlpfc'][:,cp1x, cp2x], tail = 'greater', alpha=5)
ax.text(2.5,1.05, f'{f_plotting.sig_marker(p3,ns_note=True)}',horizontalalignment='center', verticalalignment='bottom', fontsize=10)
p4 = f_stats.permutation_pCI(codeTrans_choice_data['fef'][:,cp1x, cp2x], codeTrans_choice_data_shuff['fef'][:,cp1x, cp2x], tail = 'greater', alpha=5)
ax.text(3.5,1.05, f'{f_plotting.sig_marker(p4,ns_note=True)}',horizontalalignment='center', verticalalignment='bottom', fontsize=10)


ax.set_ylabel('Retarget <-> Distraction', labelpad = 3, fontsize = 10)
xlims = ax.get_xlim()
lineThresholds = np.arange(xlims[0],xlims[1]+0.01,0.01)
ax.set_xticks([0.5,1.5,2.5,3.5],['R@R','R&U', 'LPFC','FEF'], rotation=0)
ax.set_xlabel('Strategy/Region', labelpad = 5, fontsize = 12)
ax.set_ylim(top=1.15, bottom=0)
ax.set_title('Between-Type Transferability, Choice Item', fontsize = 12, pad=10)
plt.show()

# print stats
print(f"R@R: M(SD) = {codeTrans_choice_rnn['ed2'].mean():.3f}({codeTrans_choice_rnn['ed2'].std():.3f}), p = {p1:.3f};")
print(f"R&U: M(SD) = {codeTrans_choice_rnn['ed12'].mean():.3f}({codeTrans_choice_rnn['ed12'].std():.3f}), p = {p2:.3f};")
print(f"LPFC: M(SD) = {codeTrans_choice_data['dlpfc'].mean():.3f}({codeTrans_choice_data['dlpfc'].std():.3f}), p = {p3:.3f};")
print(f"FEF: M(SD) = {codeTrans_choice_data['fef'].mean():.3f}({codeTrans_choice_data['fef'].std():.3f}), p = {p4:.3f};")
#%% [Figure S3C] plot code transferability ratio at specific checkpoints

############################################################
# code transferability ratio between target-item subspaces #
############################################################

checkpoints = [150, 550, 1050, 1450, 1850, 2350]#
checkpointsLabels = ['S1','ED1','LD1','S2','ED2','LD2'] #, 2800
avgInterval = {150:150, 550:250, 1050:250, 1450:150, 1850:250, 2350:250} #, 2800:200

# cross-subspace decodability
codeTrans_choice_data = {}
codeTrans_choice_data_shuff = {}

for k in ('dlpfc','fef'):
    pfmX_rdc_data = np.array([performanceX_Trans_rdc_data[k][n] for n in range(100)]).mean(1)
    pfmX_drc_data = np.array([performanceX_Trans_drc_data[k][n] for n in range(100)]).mean(1)
    pfmX_rdc_shuff_data = np.array([performanceX_Trans_rdc_shuff_data[k][n] for n in range(100)]).mean(1)
    pfmX_drc_shuff_data = np.array([performanceX_Trans_drc_shuff_data[k][n] for n in range(100)]).mean(1)

    codeTrans_choice_data[k] = np.array([f_decoding.code_transferability(pfmX_rdc_data[n],pfmX_drc_data[n],checkpoints) for n in range(100)])
    codeTrans_choice_data_shuff[k] = np.array([f_decoding.code_transferability(pfmX_rdc_shuff_data[n],pfmX_drc_shuff_data[n],checkpoints) for n in range(100)])

        
codeTrans_choice_rnn = {}
codeTrans_choice_shuff_rnn = {}

for k in ('ed2','ed12'):
    pfmX_rdc_rnn = np.array([performanceX_Trans_rdc_rnn[k][n] for n in range(100)]).mean(1).mean(1)
    pfmX_drc_rnn = np.array([performanceX_Trans_drc_rnn[k][n] for n in range(100)]).mean(1).mean(1)
    pfmX_rdc_shuff_rnn = np.array([performanceX_Trans_rdc_shuff_rnn[k][n] for n in range(100)]).mean(1).mean(1)
    pfmX_drc_shuff_rnn = np.array([performanceX_Trans_drc_shuff_rnn[k][n] for n in range(100)]).mean(1).mean(1)
    
    codeTrans_choice_rnn[k] = np.array([f_decoding.code_transferability(pfmX_rdc_rnn[n],pfmX_drc_rnn[n],checkpoints) for n in range(100)])
    codeTrans_choice_shuff_rnn[k] = np.array([f_decoding.code_transferability(pfmX_rdc_shuff_rnn[n],pfmX_drc_shuff_rnn[n],checkpoints) for n in range(100)])

# within-subspace decodability
codeWithin_choice_data = {}
codeWithin_choice_data_shuff = {}
for k in ('dlpfc','fef'):
    
    codeWithin_choice_data[k] = {1:performance2_item_data[k][1].mean(1), 2:performance1_item_data[k][2].mean(1)}
    codeWithin_choice_data_shuff[k] = {1:performance2_item_shuff_data[k][1].mean(1), 2:performance1_item_shuff_data[k][2].mean(1)}
    
codeWithin_choice_rnn = {}
codeWithin_choice_shuff_rnn = {}
for k in ('ed2','ed12'):
    
    codeWithin_choice_rnn[k] = {1:np.array([performance2_item_rnn[k][n][1].squeeze().mean(0) for n in range(100)]), 2:np.array([performance1_item_rnn[k][n][2].squeeze().mean(0) for n in range(100)])}
    codeWithin_choice_shuff_rnn[k] = {1:np.array([performance2_item_shuff_rnn[k][n][1].squeeze().mean(0) for n in range(100)]), 2:np.array([performance1_item_shuff_rnn[k][n][2].squeeze().mean(0) for n in range(100)])}


cp1, cp2 = 'LD2', 'LD2'
cp1x, cp2x = checkpointsLabels.index(cp1), checkpointsLabels.index(cp2)

# calculate code transferability ratio: cross-subspace / within-subspace
codeTrans_ratio_data = {}
codeTrans_ratio_data_shuff = {}
codeTrans_ratio_rnn = {}
codeTrans_ratio_shuff_rnn = {}

for k in ('dlpfc','fef'):
    codeTrans_ratio_data[k] = codeTrans_choice_data[k][:,cp1x, cp2x]/np.mean([codeWithin_choice_data[k][1][:,cp2x], codeWithin_choice_data[k][2][:,cp2x]], axis=0)
    codeTrans_ratio_data_shuff[k] = codeTrans_choice_data_shuff[k][:,cp1x, cp2x]/np.mean([codeWithin_choice_data_shuff[k][1][:,cp2x], codeWithin_choice_data_shuff[k][2][:,cp2x]], axis=0)

for k in ('ed2','ed12'):
    codeTrans_ratio_rnn[k] = codeTrans_choice_rnn[k][:,cp1x, cp2x]/np.mean([codeWithin_choice_rnn[k][1][:,cp2x], codeWithin_choice_rnn[k][2][:,cp2x]], axis=0)
    codeTrans_ratio_shuff_rnn[k] = codeTrans_choice_shuff_rnn[k][:,cp1x, cp2x]/np.mean([codeWithin_choice_shuff_rnn[k][1][:,cp2x], codeWithin_choice_shuff_rnn[k][2][:,cp2x]], axis=0)


pltBsl = True

lineh = np.arange(0.3,0.7,0.001)
linev = np.arange(1.16,1.165,0.0001)

fig, axes = plt.subplots(1,1, sharey=True, figsize=(3,3), dpi=300)

# plot
ax = axes#.flatten()[0]

ax.boxplot([codeTrans_ratio_rnn['ed2']], positions=[0.5-pltBsl*0.2], patch_artist=True, widths = 0.2, boxprops=boxprops1, flierprops=flierprops1, 
                  meanprops=meanpointprops1, medianprops=medianprops, capprops = capprops1, whiskerprops = whiskerprops1, meanline=False, showmeans=True)
ax.boxplot([codeTrans_ratio_rnn['ed12']], positions=[1.5-pltBsl*0.2], patch_artist=True, widths = 0.2, boxprops=boxprops2, flierprops=flierprops2, 
                  meanprops=meanpointprops2, medianprops=medianprops, capprops = capprops2, whiskerprops = whiskerprops2, meanline=False, showmeans=True)
ax.boxplot([codeTrans_ratio_data['dlpfc']], positions=[2.5-pltBsl*0.2], patch_artist=True, widths = 0.2, boxprops=boxprops3, flierprops=flierprops3, 
                  meanprops=meanpointprops3, medianprops=medianprops, capprops = capprops3, whiskerprops = whiskerprops3, meanline=False, showmeans=True)
ax.boxplot([codeTrans_ratio_data['fef']], positions=[3.5-pltBsl*0.2], patch_artist=True, widths = 0.2, boxprops=boxprops4, flierprops=flierprops4, 
                  meanprops=meanpointprops4, medianprops=medianprops, capprops = capprops4, whiskerprops = whiskerprops4, meanline=False, showmeans=True)

# whether to include baseline distributions or not
if pltBsl:
    ax.boxplot([codeTrans_ratio_shuff_rnn['ed2']], positions=[0.5+pltBsl*0.2], patch_artist=True, widths = 0.2, boxprops=boxprops0, flierprops=flierprops0, 
                    meanprops=meanpointprops0, medianprops=medianprops, capprops = capprops0, whiskerprops = whiskerprops0, meanline=False, showmeans=True)
    ax.boxplot([codeTrans_ratio_shuff_rnn['ed12']], positions=[1.5+pltBsl*0.2], patch_artist=True, widths = 0.2, boxprops=boxprops0, flierprops=flierprops0, 
                    meanprops=meanpointprops0, medianprops=medianprops, capprops = capprops0, whiskerprops = whiskerprops0, meanline=False, showmeans=True)
    ax.boxplot([codeTrans_ratio_data_shuff['dlpfc']], positions=[2.5+pltBsl*0.2], patch_artist=True, widths = 0.2, boxprops=boxprops0, flierprops=flierprops0, 
                    meanprops=meanpointprops0, medianprops=medianprops, capprops = capprops0, whiskerprops = whiskerprops0, meanline=False, showmeans=True)
    ax.boxplot([codeTrans_ratio_data_shuff['fef']], positions=[3.5+pltBsl*0.2], patch_artist=True, widths = 0.2, boxprops=boxprops0, flierprops=flierprops0, 
                    meanprops=meanpointprops0, medianprops=medianprops, capprops = capprops0, whiskerprops = whiskerprops0, meanline=False, showmeans=True)


p1 = f_stats.permutation_pCI(codeTrans_ratio_rnn['ed2'], codeTrans_ratio_shuff_rnn['ed2'], tail = 'smaller', alpha=5)
ax.text(0.5,1.8, f'{f_plotting.sig_marker(p1,ns_note=True)}',horizontalalignment='center', verticalalignment='bottom', fontsize=10)

p2 = f_stats.permutation_pCI(codeTrans_ratio_rnn['ed12'], codeTrans_ratio_shuff_rnn['ed12'], tail = 'smaller', alpha=5)
ax.text(1.5,1.8, f'{f_plotting.sig_marker(p2,ns_note=True)}',horizontalalignment='center', verticalalignment='bottom', fontsize=10)

p3 = f_stats.permutation_pCI(codeTrans_ratio_data['dlpfc'], codeTrans_ratio_data_shuff['dlpfc'], tail = 'smaller', alpha=5)
ax.text(2.5,1.8, f'{f_plotting.sig_marker(p3,ns_note=True)}',horizontalalignment='center', verticalalignment='bottom', fontsize=10)

p4 = f_stats.permutation_pCI(codeTrans_ratio_data['fef'], codeTrans_ratio_data_shuff['fef'], tail = 'smaller', alpha=5)
ax.text(3.5,1.8, f'{f_plotting.sig_marker(p4,ns_note=True)}',horizontalalignment='center', verticalalignment='bottom', fontsize=10)


ax.set_ylabel('Cross- / Within-Subspace', labelpad = 3, fontsize = 10)
xlims = ax.get_xlim()
lineThresholds = np.arange(xlims[0],xlims[1]+0.01,0.01)ax.set_xticks([0.5,1.5,2.5,3.5],['R@R','R&U', 'LPFC','FEF'], rotation=0)
ax.set_xlabel('Strategy/Region', labelpad = 5, fontsize = 12)
ax.set_ylim(top=2, bottom=0)
ax.set_title('Between-Task Transferability Ratio of Choice Items', fontsize = 12, pad=10)
plt.show()

# print stats
print(f"R@R: M(SD) = {codeTrans_ratio_rnn['ed2'].mean():.3f}({codeTrans_ratio_rnn['ed2'].std():.3f}), p = {p1:.3f};")
print(f"R&U: M(SD) = {codeTrans_ratio_rnn['ed12'].mean():.3f}({codeTrans_ratio_rnn['ed12'].std():.3f}), p = {p2:.3f};")
print(f"LPFC: M(SD) = {codeTrans_ratio_data['dlpfc'].mean():.3f}({codeTrans_ratio_data['dlpfc'].std():.3f}), p = {p3:.3f};")
print(f"FEF: M(SD) = {codeTrans_ratio_data['fef'].mean():.3f}({codeTrans_ratio_data['fef'].std():.3f}), p = {p4:.3f};")

# [Non-used] KS test
codeTrans_ratio_ks = {'R@R-LPFC':scipy.stats.ks_2samp(codeTrans_ratio_rnn['ed2'], codeTrans_ratio_data['dlpfc']), 
                        'R@R-FEF':scipy.stats.ks_2samp(codeTrans_ratio_rnn['ed2'], codeTrans_ratio_data['fef']), 
                        'R&U-LPFC':scipy.stats.ks_2samp(codeTrans_ratio_rnn['ed12'], codeTrans_ratio_data['dlpfc']), 
                        'R&U-FEF':scipy.stats.ks_2samp(codeTrans_ratio_rnn['ed12'], codeTrans_ratio_data['fef'])}
print('############### K-S Test ##############')
for k in codeTrans_ratio_ks.keys():
    print(f"{k}: D = {codeTrans_ratio_ks[k].statistic:.3f}, p = {codeTrans_ratio_ks[k].pvalue:.3f};")

