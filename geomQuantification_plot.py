# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 08:36:07 2024

@author: aka2333
"""
# In[ ]:
%reload_ext autoreload
%autoreload 2

# Import useful py lib/packages
import os
import sys
import shutil
import gc
import random
import math
import csv
import itertools
from itertools import permutations, combinations, product

import time

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
from numpy import genfromtxt
import scipy
from scipy import stats
from scipy.io import loadmat  # this is the SciPy module that loads mat-files

import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.sandbox.stats.multicomp import multipletests

#import pingouin as pg


import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.pyplot import MultipleLocator #从pyplot导入MultipleLocator类，这个类用于设置刻度间隔
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec  # 导入专门用于网格分割及子图绘制的扩展包Gridspec


import re, seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

import sklearn
from sklearn.decomposition import PCA
from scipy.ndimage import gaussian_filter1d
from scipy import ndimage

from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D
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

# In[]
sys.path.append(r'C:\Users\aka2333\OneDrive\phd\project')
sys.path.append(r'C:\Users\wudon\OneDrive\phd\project')

from functions import Get_paths, Get_trial_old, Get_trial, Get_spikes, spike_selection, spike2freq, bslFreqA, bslFreqV, tCutArray
from functions import bslTTest_1samp, bslTTest_paired, responsiveTest, consFTest, consTukeyHSD, selectiveTest, consANOVA, consSelective, bslTTest_paired_mean, consANOVA_mean
from functions import consAOV_mean, consPairT_mean, consAOV_bin, consPairT_bin, consAOV_mean3, consPairT_mean3, consSelective_mean, cohenD, pairedVS

from plotFunctions import plotEpochCrr_save
from selectivityFunctions import selectivityX, typeTTestInd_bin, typeTTest1Samp_bin,typeTTestArray_bin, diffTime
# In[]

import f_pseudoPop
import f_stats
import f_decoding
import f_plotting
# In[]
import f_subspace
import pycircstat as cstats
# In[] pooled
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
# monkeys = ['whiskey']
tRangeRaw = np.arange(-500,4000,1) # -300 baseline, 0 onset, 300 pre1, 1300 delay1, 1600 pre2, 2600 delay2, response

# In[] epoch parameters

locs = (0,1,2,3,)
locCombs = list(permutations(locs,2))
ttypes = (1,2,)

dropCombs = ()

subConditions = list(product(locCombs, ttypes))


# In[] decode from pseudo population
pd.options.mode.chained_assignment = None
epsilon = 0.0000001
bins = 50

tslice = (-300,2700)
tbins = np.arange(tslice[0], tslice[1], bins)

checkpoints = [150, 550, 1050, 1450, 1850, 2350]#
checkpointsLabels = ['S1','ED1','LD1','S2','ED2','LD2']

#%%

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

#medianprops = dict(linestyle='--', linewidth=1, color='w')

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

# In[]

###############
# Monkey Data #
###############

#%%
cosTheta11_data = np.load(f'{phd_path}/outputs/monkeys/' + 'cosTheta_11_data.npy', allow_pickle=True).item()
cosTheta12_data = np.load(f'{phd_path}/outputs/monkeys/' + 'cosTheta_12_data.npy', allow_pickle=True).item()
cosTheta22_data = np.load(f'{phd_path}/outputs/monkeys/' + 'cosTheta_22_data.npy', allow_pickle=True).item()
cosPsi11_data = np.load(f'{phd_path}/outputs/monkeys/' + 'cosPsi_11_data.npy', allow_pickle=True).item()
cosPsi12_data = np.load(f'{phd_path}/outputs/monkeys/' + 'cosPsi_12_data.npy', allow_pickle=True).item()
cosPsi22_data = np.load(f'{phd_path}/outputs/monkeys/' + 'cosPsi_22_data.npy', allow_pickle=True).item()

cosTheta11_shuff_data = np.load(f'{phd_path}/outputs/monkeys/' + 'cosTheta_11_shuff_data.npy', allow_pickle=True).item()
cosTheta12_shuff_data = np.load(f'{phd_path}/outputs/monkeys/' + 'cosTheta_12_shuff_data.npy', allow_pickle=True).item()
cosTheta22_shuff_data = np.load(f'{phd_path}/outputs/monkeys/' + 'cosTheta_22_shuff_data.npy', allow_pickle=True).item()
cosPsi11_shuff_data = np.load(f'{phd_path}/outputs/monkeys/' + 'cosPsi_11_shuff_data.npy', allow_pickle=True).item()
cosPsi12_shuff_data = np.load(f'{phd_path}/outputs/monkeys/' + 'cosPsi_12_shuff_data.npy', allow_pickle=True).item()
cosPsi22_shuff_data = np.load(f'{phd_path}/outputs/monkeys/' + 'cosPsi_22_shuff_data.npy', allow_pickle=True).item()

cosTheta_choice_data = np.load(f'{phd_path}/outputs/monkeys/' + 'cosTheta_choice_data.npy', allow_pickle=True).item()
cosTheta_nonchoice_data = np.load(f'{phd_path}/outputs/monkeys/' + 'cosTheta_nonchoice_data.npy', allow_pickle=True).item()
cosPsi_choice_data = np.load(f'{phd_path}/outputs/monkeys/' + 'cosPsi_choice_data.npy', allow_pickle=True).item()
cosPsi_nonchoice_data = np.load(f'{phd_path}/outputs/monkeys/' + 'cosPsi_nonchoice_data.npy', allow_pickle=True).item()

cosTheta_choice_shuff_data = np.load(f'{phd_path}/outputs/monkeys/' + 'cosTheta_choice_shuff_data.npy', allow_pickle=True).item()
cosTheta_nonchoice_shuff_data = np.load(f'{phd_path}/outputs/monkeys/' + 'cosTheta_nonchoice_shuff_data.npy', allow_pickle=True).item()
cosPsi_choice_shuff_data = np.load(f'{phd_path}/outputs/monkeys/' + 'cosPsi_choice_shuff_data.npy', allow_pickle=True).item()
cosPsi_nonchoice_shuff_data = np.load(f'{phd_path}/outputs/monkeys/' + 'cosPsi_nonchoice_shuff_data.npy', allow_pickle=True).item()

performance1_item_data = np.load(f'{phd_path}/outputs/monkeys/' + 'performance1_item_data.npy', allow_pickle=True).item()
performance2_item_data = np.load(f'{phd_path}/outputs/monkeys/' + 'performance2_item_data.npy', allow_pickle=True).item()
performance1_item_shuff_data = np.load(f'{phd_path}/outputs/monkeys/' + 'performance1_item_shuff_data.npy', allow_pickle=True).item()
performance2_item_shuff_data = np.load(f'{phd_path}/outputs/monkeys/' + 'performance2_item_shuff_data.npy', allow_pickle=True).item()
# In[]

########
# rnns #
########

#%%
cosTheta11_rnn = np.load(f'{phd_path}/outputs/rnns/' + 'cosTheta_11_rnn.npy', allow_pickle=True).item()
cosTheta12_rnn = np.load(f'{phd_path}/outputs/rnns/' + 'cosTheta_12_rnn.npy', allow_pickle=True).item()
cosTheta22_rnn = np.load(f'{phd_path}/outputs/rnns/' + 'cosTheta_22_rnn.npy', allow_pickle=True).item()
cosPsi11_rnn = np.load(f'{phd_path}/outputs/rnns/' + 'cosPsi_11_rnn.npy', allow_pickle=True).item()
cosPsi12_rnn = np.load(f'{phd_path}/outputs/rnns/' + 'cosPsi_12_rnn.npy', allow_pickle=True).item()
cosPsi22_rnn = np.load(f'{phd_path}/outputs/rnns/' + 'cosPsi_22_rnn.npy', allow_pickle=True).item()

cosTheta11_shuff_rnn = np.load(f'{phd_path}/outputs/rnns/' + 'cosTheta_11_shuff_rnn.npy', allow_pickle=True).item()
cosTheta12_shuff_rnn = np.load(f'{phd_path}/outputs/rnns/' + 'cosTheta_12_shuff_rnn.npy', allow_pickle=True).item()
cosTheta22_shuff_rnn = np.load(f'{phd_path}/outputs/rnns/' + 'cosTheta_22_shuff_rnn.npy', allow_pickle=True).item()
cosPsi11_shuff_rnn = np.load(f'{phd_path}/outputs/rnns/' + 'cosPsi_11_shuff_rnn.npy', allow_pickle=True).item()
cosPsi12_shuff_rnn = np.load(f'{phd_path}/outputs/rnns/' + 'cosPsi_12_shuff_rnn.npy', allow_pickle=True).item()
cosPsi22_shuff_rnn = np.load(f'{phd_path}/outputs/rnns/' + 'cosPsi_22_shuff_rnn.npy', allow_pickle=True).item()

cosTheta_choice_rnn = np.load(f'{phd_path}/outputs/rnns/' + 'cosTheta_choice_rnn.npy', allow_pickle=True).item()
cosTheta_nonchoice_rnn = np.load(f'{phd_path}/outputs/rnns/' + 'cosTheta_nonchoice_rnn.npy', allow_pickle=True).item()
cosPsi_choice_rnn = np.load(f'{phd_path}/outputs/rnns/' + 'cosPsi_choice_rnn.npy', allow_pickle=True).item()
cosPsi_nonchoice_rnn = np.load(f'{phd_path}/outputs/rnns/' + 'cosPsi_nonchoice_rnn.npy', allow_pickle=True).item()

cosTheta_choice_shuff_rnn = np.load(f'{phd_path}/outputs/rnns/' + 'cosTheta_choice_shuff_rnn.npy', allow_pickle=True).item()
cosTheta_nonchoice_shuff_rnn = np.load(f'{phd_path}/outputs/rnns/' + 'cosTheta_nonchoice_shuff_rnn.npy', allow_pickle=True).item()
cosPsi_choice_shuff_rnn = np.load(f'{phd_path}/outputs/rnns/' + 'cosPsi_choice_shuff_rnn.npy', allow_pickle=True).item()
cosPsi_nonchoice_shuff_rnn = np.load(f'{phd_path}/outputs/rnns/' + 'cosPsi_nonchoice_shuff_rnn.npy', allow_pickle=True).item()

performance1_item_rnn = np.load(f'{phd_path}/outputs/rnns/' + 'performance1_item_rnn.npy', allow_pickle=True).item()
performance2_item_rnn = np.load(f'{phd_path}/outputs/rnns/' + 'performance2_item_rnn.npy', allow_pickle=True).item()
performance1_item_shuff_rnn = np.load(f'{phd_path}/outputs/rnns/' + 'performance1_item_shuff_rnn.npy', allow_pickle=True).item()
performance2_item_shuff_rnn = np.load(f'{phd_path}/outputs/rnns/' + 'performance2_item_shuff_rnn.npy', allow_pickle=True).item()

#%% baseline
cosTheta11_bsl_data = np.load(f'{phd_path}/outputs/monkeys/' + 'cosTheta_11_bsl_data.npy', allow_pickle=True).item()
cosTheta12_bsl_data = np.load(f'{phd_path}/outputs/monkeys/' + 'cosTheta_12_bsl_data.npy', allow_pickle=True).item()
cosTheta22_bsl_data = np.load(f'{phd_path}/outputs/monkeys/' + 'cosTheta_22_bsl_data.npy', allow_pickle=True).item()
cosPsi11_bsl_data = np.load(f'{phd_path}/outputs/monkeys/' + 'cosPsi_11_bsl_data.npy', allow_pickle=True).item()
cosPsi12_bsl_data = np.load(f'{phd_path}/outputs/monkeys/' + 'cosPsi_12_bsl_data.npy', allow_pickle=True).item()
cosPsi22_bsl_data = np.load(f'{phd_path}/outputs/monkeys/' + 'cosPsi_22_bsl_data.npy', allow_pickle=True).item()

cosTheta11_bsl_rnn = np.load(f'{phd_path}/outputs/rnns/' + 'cosTheta_11_bsl_rnn.npy', allow_pickle=True).item()
cosTheta12_bsl_rnn = np.load(f'{phd_path}/outputs/rnns/' + 'cosTheta_12_bsl_rnn.npy', allow_pickle=True).item()
cosTheta22_bsl_rnn = np.load(f'{phd_path}/outputs/rnns/' + 'cosTheta_22_bsl_rnn.npy', allow_pickle=True).item()
cosPsi11_bsl_rnn = np.load(f'{phd_path}/outputs/rnns/' + 'cosPsi_11_bsl_rnn.npy', allow_pickle=True).item()
cosPsi12_bsl_rnn = np.load(f'{phd_path}/outputs/rnns/' + 'cosPsi_12_bsl_rnn.npy', allow_pickle=True).item()
cosPsi22_bsl_rnn = np.load(f'{phd_path}/outputs/rnns/' + 'cosPsi_22_bsl_rnn.npy', allow_pickle=True).item()

#%%




###########################
# retarget, I1D1 vs. I2D2 #
###########################





# In[] retarget, I1D1 vs. I2D2 

lineh = np.arange(0.5,1.5,0.001)
linev = np.arange(0.71,0.72,0.0001)

cosTheta12Ret_rnn = {k:np.array([cosTheta12_rnn[k][n][1] for n in range(100)]).mean(1) for k in ('ed2','ed12')}
cosTheta12Ret_data = {k:cosTheta12_data[k][1].mean(1) for k in ('dlpfc','fef')}
cosTheta12Ret_shuff_rnn = {k:np.array([cosTheta12_shuff_rnn[k][n][1] for n in range(100)]).mean(1) for k in ('ed2','ed12')}
cosTheta12Ret_shuff_data = {k:cosTheta12_shuff_data[k][1] for k in ('dlpfc','fef')}

cosPsi12Ret_rnn = {k:np.array([cosPsi12_rnn[k][n][1] for n in range(100)]).mean(1) for k in ('ed2','ed12')}
cosPsi12Ret_data = {k:cosPsi12_data[k][1].mean(1) for k in ('dlpfc','fef')}
cosPsi12Ret_shuff_rnn = {k:np.array([cosPsi12_shuff_rnn[k][n][1] for n in range(100)]).mean(1) for k in ('ed2','ed12')}
cosPsi12Ret_shuff_data = {k:cosPsi12_shuff_data[k][1] for k in ('dlpfc','fef')}

#%%
angleCheckPoints = np.linspace(0,np.pi,7).round(5)
cmap = plt.get_cmap('coolwarm')
norm = mcolors.BoundaryNorm(np.flip(np.cos(angleCheckPoints).round(5)),cmap.N, clip=True)

showCheckpoints = ('L',) #'E',
showShuffBsl = False
fig, axes = plt.subplots(1,2, sharey=True, figsize=(8,4), dpi=300)

# cosTheta
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
    
    
    ax.boxplot([cosTheta12Ret_rnn['ed2'][:,d1x,d2x]], positions=[0.2+nk-0.035*showShuffBsl], patch_artist=True, widths = 0.05*len(showCheckpoints), boxprops=boxprops1, flierprops=flierprops1, 
                    meanprops=meanpointprops1, medianprops=medianprops, capprops = capprops1, whiskerprops = whiskerprops1, meanline=False, showmeans=True)
    
    #p1 = stats.ks_2samp(cosTheta12Ret_rnn['ed2'][:,d1x,d2x], np.concatenate(cosTheta12Ret_shuff_rnn['ed2'][:,:,d1x,d2x]))[1].round(3) #.mean(1)
    #p1 = f_stats.permutation_pCI(cosTheta12Ret_rnn['ed2'][:,d1x,d2x], cosTheta12Ret_shuff_rnn['ed2'][:,:,d1x,d2x].mean(1), tail='two') #.mean(1)
    p1 = f_stats.permutation_pCI(cosTheta12Ret_rnn['ed2'][:,d1x,d2x], cosTheta_bsl_rnn['ed2'], tail='smaller', alpha = 5) #.mean(1)
    ax.text(0.2+nk,1.0, f'{f_plotting.sig_marker(p1,ns_note=False)}',horizontalalignment='center', verticalalignment='bottom', fontsize=10)
    print(f"R@R: M(SD) = {cosTheta12Ret_rnn['ed2'][:,d1x,d2x].mean():.3f}({cosTheta12Ret_rnn['ed2'][:,d1x,d2x].std():.3f}), 95CI = [{np.quantile(cosTheta12Ret_rnn['ed2'][:,d1x,d2x],0.025):.3f}, {np.quantile(cosTheta12Ret_rnn['ed2'][:,d1x,d2x],0.975):.3f}], p = {p1:.3f}")

    ax.boxplot([cosTheta12Ret_rnn['ed12'][:,d1x,d2x]], positions=[0.4+nk-0.035*showShuffBsl], patch_artist=True, widths = 0.05*len(showCheckpoints), boxprops=boxprops2, flierprops=flierprops2, 
                    meanprops=meanpointprops2, medianprops=medianprops, capprops = capprops2, whiskerprops = whiskerprops2, meanline=False, showmeans=True)
    #p2 = stats.ks_2samp(cosTheta12Ret_rnn['ed12'][:,d1x,d2x], np.concatenate(cosTheta12Ret_shuff_rnn['ed12'][:,:,d1x,d2x]))[1].round(3) #.mean(1)
    #p2 = f_stats.permutation_pCI(cosTheta12Ret_rnn['ed12'][:,d1x,d2x], cosTheta12Ret_shuff_rnn['ed12'][:,:,d1x,d2x].mean(1), tail='two')
    p2 = f_stats.permutation_pCI(cosTheta12Ret_rnn['ed12'][:,d1x,d2x], cosTheta_bsl_rnn['ed12'], tail='smaller', alpha = 5)
    ax.text(0.4+nk,1.0, f'{f_plotting.sig_marker(p2,ns_note=False)}',horizontalalignment='center', verticalalignment='bottom', fontsize=10)
    print(f"R&U: M(SD) = {cosTheta12Ret_rnn['ed12'][:,d1x,d2x].mean():.3f}({cosTheta12Ret_rnn['ed12'][:,d1x,d2x].std():.3f}), 95CI = [{np.quantile(cosTheta12Ret_rnn['ed12'][:,d1x,d2x],0.025):.3f}, {np.quantile(cosTheta12Ret_rnn['ed12'][:,d1x,d2x],0.975):.3f}], p = {p2:.3f}")

    ax.boxplot([cosTheta12Ret_data['dlpfc'][:,d1x,d2x]], positions=[0.6+nk-0.035*showShuffBsl], patch_artist=True, widths = 0.05*len(showCheckpoints), boxprops=boxprops3, flierprops=flierprops3, 
                    meanprops=meanpointprops3, medianprops=medianprops, capprops = capprops3, whiskerprops = whiskerprops3, meanline=False, showmeans=True)
    #p3 = stats.ks_2samp(cosTheta12Ret_data['dlpfc'][:,d1x,d2x], np.concatenate(cosTheta12Ret_shuff_data['dlpfc'][:,:,d1x,d2x]))[1].round(3) #.mean(1)
    #p3 = f_stats.permutation_pCI(cosTheta12Ret_data['dlpfc'][:,d1x,d2x], cosTheta12Ret_shuff_data['dlpfc'][:,:,d1x,d2x].mean(1), tail='two') #.mean(1)
    p3 = f_stats.permutation_pCI(cosTheta12Ret_data['dlpfc'][:,d1x,d2x], cosTheta_bsl_data['dlpfc'], tail='smaller', alpha = 5) #.mean(1)
    ax.text(0.6+nk,1.0, f'{f_plotting.sig_marker(p3,ns_note=False)}',horizontalalignment='center', verticalalignment='bottom', fontsize=10)
    print(f"LPFC: M(SD) = {cosTheta12Ret_data['dlpfc'][:,d1x,d2x].mean():.3f}({cosTheta12Ret_data['dlpfc'][:,d1x,d2x].std():.3f}), 95CI = [{np.quantile(cosTheta12Ret_data['dlpfc'][:,d1x,d2x],0.025):.3f}, {np.quantile(cosTheta12Ret_data['dlpfc'][:,d1x,d2x],0.975):.3f}], p = {p3:.3f}")

    ax.boxplot([cosTheta12Ret_data['fef'][:,d1x,d2x]], positions=[0.8+nk-0.035*showShuffBsl], patch_artist=True, widths = 0.05*len(showCheckpoints), boxprops=boxprops4, flierprops=flierprops4, 
                    meanprops=meanpointprops4, medianprops=medianprops, capprops = capprops4, whiskerprops = whiskerprops4, meanline=False, showmeans=True)
    #p4 = stats.ks_2samp(cosTheta12Ret_data['fef'][:,d1x,d2x], np.concatenate(cosTheta12Ret_shuff_data['fef'][:,:,d1x,d2x]))[1].round(3) #.mean(1)
    #p4 = f_stats.permutation_pCI(cosTheta12Ret_data['fef'][:,d1x,d2x], cosTheta12Ret_shuff_data['fef'][:,:,d1x,d2x].mean(1), tail='two') #.mean(1)
    p4 = f_stats.permutation_pCI(cosTheta12Ret_data['fef'][:,d1x,d2x], cosTheta_bsl_data['fef'], tail='smaller', alpha = 5) #.mean(1)
    ax.text(0.8+nk,1.0, f'{f_plotting.sig_marker(p4,ns_note=False)}',horizontalalignment='center', verticalalignment='bottom', fontsize=10)
    print(f"FEF: M(SD) = {cosTheta12Ret_data['fef'][:,d1x,d2x].mean():.3f}({cosTheta12Ret_data['fef'][:,d1x,d2x].std():.3f}), 95CI = [{np.quantile(cosTheta12Ret_data['fef'][:,d1x,d2x],0.025):.3f}, {np.quantile(cosTheta12Ret_data['fef'][:,d1x,d2x],0.975):.3f}], p = {p4:.3f}")

    if showShuffBsl:
        #ax.boxplot([cosTheta12Ret_shuff_rnn['ed2'][:,:,d1x,d2x].mean(1)], positions=[0.2+nk+0.035*showShuffBsl], patch_artist=True, widths = 0.05*len(showCheckpoints), boxprops=boxprops0, flierprops=flierprops0, 
        #            meanprops=meanpointprops0, medianprops=medianprops, capprops = capprops0, whiskerprops = whiskerprops0, meanline=False, showmeans=True)
        
        #ax.boxplot([cosTheta12Ret_shuff_rnn['ed12'][:,:,d1x,d2x].mean(1)], positions=[0.4+nk+0.035*showShuffBsl], patch_artist=True, widths = 0.05*len(showCheckpoints), boxprops=boxprops0, flierprops=flierprops0, 
        #                meanprops=meanpointprops0, medianprops=medianprops, capprops = capprops0, whiskerprops = whiskerprops0, meanline=False, showmeans=True)
        
        #ax.boxplot([cosTheta12Ret_shuff_data['dlpfc'][:,:,d1x,d2x].mean(1)], positions=[0.6+nk+0.035*showShuffBsl], patch_artist=True, widths = 0.05*len(showCheckpoints), boxprops=boxprops0, flierprops=flierprops0, 
        #                meanprops=meanpointprops0, medianprops=medianprops, capprops = capprops0, whiskerprops = whiskerprops0, meanline=False, showmeans=True)
        
        #ax.boxplot([cosTheta12Ret_shuff_data['fef'][:,:,d1x,d2x].mean(1)], positions=[0.8+nk+0.035*showShuffBsl], patch_artist=True, widths = 0.05*len(showCheckpoints), boxprops=boxprops0, flierprops=flierprops0, 
        #                meanprops=meanpointprops0, medianprops=medianprops, capprops = capprops0, whiskerprops = whiskerprops0, meanline=False, showmeans=True)
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
        
    # KS Test
    ks_results = {'R@R-LPFC':scipy.stats.ks_2samp(cosTheta12Ret_rnn['ed2'][:,d1x,d2x], cosTheta12Ret_data['dlpfc'][:,d1x,d2x]),
                  'R@R-FEF':scipy.stats.ks_2samp(cosTheta12Ret_rnn['ed2'][:,d1x,d2x], cosTheta12Ret_data['fef'][:,d1x,d2x]),
                  'R&U-LPFC':scipy.stats.ks_2samp(cosTheta12Ret_rnn['ed12'][:,d1x,d2x], cosTheta12Ret_data['dlpfc'][:,d1x,d2x]), 
                  'R&U-FEF':scipy.stats.ks_2samp(cosTheta12Ret_rnn['ed12'][:,d1x,d2x], cosTheta12Ret_data['fef'][:,d1x,d2x])}

    # FK Test
    fk_results = {'R@R-LPFC':scipy.stats.fligner(cosTheta12Ret_rnn['ed2'][:,d1x,d2x],cosTheta12Ret_data['dlpfc'][:,d1x,d2x]),
                  'R@R-FEF':scipy.stats.fligner(cosTheta12Ret_rnn['ed2'][:,d1x,d2x],cosTheta12Ret_data['fef'][:,d1x,d2x]),
                  'R&U-LPFC':scipy.stats.fligner(cosTheta12Ret_rnn['ed12'][:,d1x,d2x],cosTheta12Ret_data['dlpfc'][:,d1x,d2x]),
                  'R&U-FEF':scipy.stats.fligner(cosTheta12Ret_rnn['ed12'][:,d1x,d2x],cosTheta12Ret_data['fef'][:,d1x,d2x])}
    
    print('############### K-S Test ##############')
    for k in ks_results.keys():
        print(f'{k.upper()}: D = {ks_results[k][0]:.3f}, p = {ks_results[k][1]:.3f}')

    #print('############### F-K Test ##############')
    #for k in fk_results.keys():
    #    print(f'{k.upper()}: H = {fk_results[k][0]:.3f}, p = {fk_results[k][1]:.3f}')
    
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
    
# cosPsi
ax = axes.flatten()[1]
xtks, xtklabs = [], []
for nk, k in enumerate(showCheckpoints):
    dk1, dk2 = k+'D1', k+'D2'
    d1x, d2x = checkpointsLabels.index(dk1), checkpointsLabels.index(dk2)
    #ld1x, ld2x = checkpointsLabels.index('LD1'), checkpointsLabels.index('LD2')

    print('############## cosPsi ##############')

    # method: baseline from split sets
    cosPsi_bsl_rnn = {kk: np.mean((np.array([cosPsi11_bsl_rnn[kk][n][1] for n in range(100)]).squeeze()[:,:,d1x].mean(1),
                                   np.array([cosPsi22_bsl_rnn[kk][n][1] for n in range(100)]).squeeze()[:,:,d2x].mean(1)),axis=0) for kk in ('ed2','ed12')}
    cosPsi_bsl_data = {kk: np.mean((cosPsi11_bsl_data[kk][1][:,:,d1x].mean(1), 
                                     cosPsi22_bsl_data[kk][1][:,:,d2x].mean(1)),axis=0) for kk in ('dlpfc','fef')}
    
    
    ax.boxplot([cosPsi12Ret_rnn['ed2'][:,d1x,d2x]], positions=[0.2+nk-0.035*showShuffBsl], patch_artist=True, widths = 0.05*len(showCheckpoints), boxprops=boxprops1, flierprops=flierprops1, 
                    meanprops=meanpointprops1, medianprops=medianprops, capprops = capprops1, whiskerprops = whiskerprops1, meanline=False, showmeans=True)
    #p1 = stats.ks_2samp(cosPsi12Ret_rnn['ed2'][:,d1x,d2x], np.concatenate(cosPsi12Ret_shuff_rnn['ed2'][:,:,d1x,d2x]))[1].round(3) #.mean(1)
    #p1 = f_stats.permutation_pCI(cosPsi12Ret_rnn['ed2'][:,d1x,d2x], cosPsi12Ret_shuff_rnn['ed2'][:,:,d1x,d2x].mean(1), tail='two') #.mean(1)
    p1 = f_stats.permutation_pCI(cosPsi12Ret_rnn['ed2'][:,d1x,d2x], cosPsi_bsl_rnn['ed2'], tail='smaller', alpha = 5)
    ax.text(0.2+nk,1.0, f'{f_plotting.sig_marker(p1,ns_note=False)}',horizontalalignment='center', verticalalignment='bottom', fontsize=10)
    print(f"R@R: M(SD) = {cosPsi12Ret_rnn['ed2'][:,d1x,d2x].mean():.3f}({cosPsi12Ret_rnn['ed2'][:,d1x,d2x].std():.3f}), 95CI = [{np.quantile(cosPsi12Ret_rnn['ed2'][:,d1x,d2x],0.025):.3f}, {np.quantile(cosPsi12Ret_rnn['ed2'][:,d1x,d2x],0.975):.3f}], p = {p1:.3f}")

    ax.boxplot([cosPsi12Ret_rnn['ed12'][:,d1x,d2x]], positions=[0.4+nk-0.035*showShuffBsl], patch_artist=True, widths = 0.05*len(showCheckpoints), boxprops=boxprops2, flierprops=flierprops2, 
                    meanprops=meanpointprops2, medianprops=medianprops, capprops = capprops2, whiskerprops = whiskerprops2, meanline=False, showmeans=True)
    #p2 = stats.ks_2samp(cosPsi12Ret_rnn['ed12'][:,d1x,d2x], np.concatenate(cosPsi12Ret_shuff_rnn['ed12'][:,:,d1x,d2x]))[1].round(3) #.mean(1)
    #p2 = f_stats.permutation_pCI(cosPsi12Ret_rnn['ed12'][:,d1x,d2x], cosPsi12Ret_shuff_rnn['ed12'][:,:,d1x,d2x].mean(1), tail='two')
    p2 = f_stats.permutation_pCI(cosPsi12Ret_rnn['ed12'][:,d1x,d2x], cosPsi_bsl_rnn['ed12'], tail='smaller', alpha = 5)
    ax.text(0.4+nk,1.0, f'{f_plotting.sig_marker(p2,ns_note=False)}',horizontalalignment='center', verticalalignment='bottom', fontsize=10)
    print(f"R&U: M(SD) = {cosPsi12Ret_rnn['ed12'][:,d1x,d2x].mean():.3f}({cosPsi12Ret_rnn['ed12'][:,d1x,d2x].std():.3f}), 95CI = [{np.quantile(cosPsi12Ret_rnn['ed12'][:,d1x,d2x],0.025):.3f}, {np.quantile(cosPsi12Ret_rnn['ed12'][:,d1x,d2x],0.975):.3f}], p = {p2:.3f}")

    ax.boxplot([cosPsi12Ret_data['dlpfc'][:,d1x,d2x]], positions=[0.6+nk-0.035*showShuffBsl], patch_artist=True, widths = 0.05*len(showCheckpoints), boxprops=boxprops3, flierprops=flierprops3, 
                    meanprops=meanpointprops3, medianprops=medianprops, capprops = capprops3, whiskerprops = whiskerprops3, meanline=False, showmeans=True)
    #p3 = stats.ks_2samp(cosPsi12Ret_data['dlpfc'][:,d1x,d2x], np.concatenate(cosPsi12Ret_shuff_data['dlpfc'][:,:,d1x,d2x]))[1].round(3) #.mean(1)
    #p3 = f_stats.permutation_pCI(cosPsi12Ret_data['dlpfc'][:,d1x,d2x], cosPsi12Ret_shuff_data['dlpfc'][:,:,d1x,d2x].mean(1), tail='two') #.mean(1)
    p3 = f_stats.permutation_pCI(cosPsi12Ret_data['dlpfc'][:,d1x,d2x], cosPsi_bsl_data['dlpfc'], tail='smaller', alpha = 5)
    ax.text(0.6+nk,1.0, f'{f_plotting.sig_marker(p3,ns_note=False)}',horizontalalignment='center', verticalalignment='bottom', fontsize=10)
    print(f"LPFC: M(SD) = {cosPsi12Ret_data['dlpfc'][:,d1x,d2x].mean():.3f}({cosPsi12Ret_data['dlpfc'][:,d1x,d2x].std():.3f}), 95CI = [{np.quantile(cosPsi12Ret_data['dlpfc'][:,d1x,d2x],0.025):.3f}, {np.quantile(cosPsi12Ret_data['dlpfc'][:,d1x,d2x],0.975):.3f}], p = {p3:.3f}")

    ax.boxplot([cosPsi12Ret_data['fef'][:,d1x,d2x]], positions=[0.8+nk-0.035*showShuffBsl], patch_artist=True, widths = 0.05*len(showCheckpoints), boxprops=boxprops4, flierprops=flierprops4, 
                    meanprops=meanpointprops4, medianprops=medianprops, capprops = capprops4, whiskerprops = whiskerprops4, meanline=False, showmeans=True)
    #p4 = stats.ks_2samp(cosPsi12Ret_data['fef'][:,d1x,d2x], np.concatenate(cosPsi12Ret_shuff_data['fef'][:,:,d1x,d2x]))[1].round(3) #.mean(1)
    #p4 = f_stats.permutation_pCI(cosPsi12Ret_data['fef'][:,d1x,d2x], cosPsi12Ret_shuff_data['fef'][:,:,d1x,d2x].mean(1), tail='two') #.mean(1)
    p4 = f_stats.permutation_pCI(cosPsi12Ret_data['fef'][:,d1x,d2x], cosPsi_bsl_data['fef'], tail='smaller', alpha = 5)
    ax.text(0.8+nk,1.0, f'{f_plotting.sig_marker(p4,ns_note=False)}',horizontalalignment='center', verticalalignment='bottom', fontsize=10)
    print(f"FEF: M(SD) = {cosPsi12Ret_data['fef'][:,d1x,d2x].mean():.3f}({cosPsi12Ret_data['fef'][:,d1x,d2x].std():.3f}), 95CI = [{np.quantile(cosPsi12Ret_data['fef'][:,d1x,d2x],0.025):.3f}, {np.quantile(cosPsi12Ret_data['fef'][:,d1x,d2x],0.975):.3f}], p = {p4:.3f}")

    if showShuffBsl:
        #ax.boxplot([cosPsi12Ret_shuff_rnn['ed2'][:,:,d1x,d2x].mean(1)], positions=[0.2+nk+0.035*showShuffBsl], patch_artist=True, widths = 0.05*len(showCheckpoints), boxprops=boxprops0, flierprops=flierprops0, 
        #            meanprops=meanpointprops0, medianprops=medianprops, capprops = capprops0, whiskerprops = whiskerprops0, meanline=False, showmeans=True)
        
        #ax.boxplot([cosPsi12Ret_shuff_rnn['ed12'][:,:,d1x,d2x].mean(1)], positions=[0.4+nk+0.035*showShuffBsl], patch_artist=True, widths = 0.05*len(showCheckpoints), boxprops=boxprops0, flierprops=flierprops0, 
        #                meanprops=meanpointprops0, medianprops=medianprops, capprops = capprops0, whiskerprops = whiskerprops0, meanline=False, showmeans=True)
        
        #ax.boxplot([cosPsi12Ret_shuff_data['dlpfc'][:,:,d1x,d2x].mean(1)], positions=[0.6+nk+0.035*showShuffBsl], patch_artist=True, widths = 0.05*len(showCheckpoints), boxprops=boxprops0, flierprops=flierprops0, 
        #                meanprops=meanpointprops0, medianprops=medianprops, capprops = capprops0, whiskerprops = whiskerprops0, meanline=False, showmeans=True)
        
        #ax.boxplot([cosPsi12Ret_shuff_data['fef'][:,:,d1x,d2x].mean(1)], positions=[0.8+nk+0.035*showShuffBsl], patch_artist=True, widths = 0.05*len(showCheckpoints), boxprops=boxprops0, flierprops=flierprops0, 
        #                meanprops=meanpointprops0, medianprops=medianprops, capprops = capprops0, whiskerprops = whiskerprops0, meanline=False, showmeans=True)
        
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

    # KS test
    ks_results = {'R@R-LPFC':scipy.stats.ks_2samp(cosPsi12Ret_rnn['ed2'][:,d1x,d2x], cosPsi12Ret_data['dlpfc'][:,d1x,d2x]),
                  'R@R-FEF':scipy.stats.ks_2samp(cosPsi12Ret_rnn['ed2'][:,d1x,d2x], cosPsi12Ret_data['fef'][:,d1x,d2x]),
                  'R&U-LPFC':scipy.stats.ks_2samp(cosPsi12Ret_rnn['ed12'][:,d1x,d2x], cosPsi12Ret_data['dlpfc'][:,d1x,d2x]), 
                  'R&U-FEF':scipy.stats.ks_2samp(cosPsi12Ret_rnn['ed12'][:,d1x,d2x], cosPsi12Ret_data['fef'][:,d1x,d2x])}

    fk_results = {'R@R-LPFC':scipy.stats.fligner(cosPsi12Ret_rnn['ed2'][:,d1x,d2x], cosPsi12Ret_data['dlpfc'][:,d1x,d2x]),
                  'R@R-FEF':scipy.stats.fligner(cosPsi12Ret_rnn['ed2'][:,d1x,d2x], cosPsi12Ret_data['fef'][:,d1x,d2x]),
                  'R&U-LPFC':scipy.stats.fligner(cosPsi12Ret_rnn['ed12'][:,d1x,d2x], cosPsi12Ret_data['dlpfc'][:,d1x,d2x]), 
                  'R&U-FEF':scipy.stats.fligner(cosPsi12Ret_rnn['ed12'][:,d1x,d2x], cosPsi12Ret_data['fef'][:,d1x,d2x])}

    print('############### K-S Test ##############')
    for k in ks_results.keys():
        print(f'{k.upper()}: D = {ks_results[k][0]:.3f}, p = {ks_results[k][1]:.3f}')
        
    #print('############### F-K Test ##############')
    #for k in fk_results.keys():
    #    print(f'{k.upper()}: H = {fk_results[k][0]:.3f}, p = {fk_results[k][1]:.3f}')
    
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
#%%
fig.savefig(f'{phd_path}/outputs/I1D1vI2D2_ret_LD.tif', bbox_inches='tight')
#%%




#############################
# distractor, I1D1 vs. I1D2 #
#############################





# In[] distractor, I1D1 vs. I1D2 

lineh = np.arange(0.5,1.5,0.001)
linev = np.arange(0.71,0.72,0.0001)

cosTheta11Dis_rnn = {k:np.array([cosTheta11_rnn[k][n][2] for n in range(100)]).mean(1) for k in ('ed2','ed12')}
cosTheta11Dis_data = {k:cosTheta11_data[k][2].mean(1) for k in ('dlpfc','fef')}
cosTheta11Dis_shuff_rnn = {k:np.array([cosTheta11_shuff_rnn[k][n][2] for n in range(100)]).mean(1) for k in ('ed2','ed12')}
cosTheta11Dis_shuff_data = {k:cosTheta11_shuff_data[k][2] for k in ('dlpfc','fef')}

cosPsi11Dis_rnn = {k:np.array([cosPsi11_rnn[k][n][2] for n in range(100)]).mean(1) for k in ('ed2','ed12')}
cosPsi11Dis_data = {k:cosPsi11_data[k][2].mean(1) for k in ('dlpfc','fef')}
cosPsi11Dis_shuff_rnn = {k:np.array([cosPsi11_shuff_rnn[k][n][2] for n in range(100)]).mean(1) for k in ('ed2','ed12')}
cosPsi11Dis_shuff_data = {k:cosPsi11_shuff_data[k][2] for k in ('dlpfc','fef')}

#%%
angleCheckPoints = np.linspace(0,np.pi,7).round(5)
cmap = plt.get_cmap('coolwarm')
norm = mcolors.BoundaryNorm(np.flip(np.cos(angleCheckPoints).round(5)),cmap.N, clip=True)

showCheckpoints = ('L',) #'E',
showShuffBsl = True
fig, axes = plt.subplots(1,2, sharey=True, figsize=(8,4), dpi=300)

# cosTheta
ax = axes.flatten()[0]
xtks, xtklabs = [], []
for nk, k in enumerate(showCheckpoints):
    dk1, dk2 = k+'D1', k+'D2'
    d1x, d2x = checkpointsLabels.index(dk1), checkpointsLabels.index(dk2)
    #ld1x, ld2x = checkpointsLabels.index('LD1'), checkpointsLabels.index('LD2')

    print('############ cosTheta ############')
    
    # method: baseline from split sets
    cosTheta_bsl_rnn = {kk: np.mean((np.array([cosTheta11_bsl_rnn[kk][n][2] for n in range(100)]).squeeze()[:,:,d1x].mean(1),
                                   np.array([cosTheta11_bsl_rnn[kk][n][2] for n in range(100)]).squeeze()[:,:,d2x].mean(1)),axis=0) for kk in ('ed2','ed12')}
    cosTheta_bsl_data = {kk: np.mean((cosTheta11_bsl_data[kk][2][:,:,d1x].mean(1), 
                                     cosTheta11_bsl_data[kk][2][:,:,d2x].mean(1)),axis=0) for kk in ('dlpfc','fef')}
    
    
    ax.boxplot([cosTheta11Dis_rnn['ed2'][:,d1x,d2x]], positions=[0.2+nk-0.035*showShuffBsl], patch_artist=True, widths = 0.05*len(showCheckpoints), boxprops=boxprops1, flierprops=flierprops1, 
                    meanprops=meanpointprops1, medianprops=medianprops, capprops = capprops1, whiskerprops = whiskerprops1, meanline=False, showmeans=True)    
    p1 = f_stats.permutation_pCI(cosTheta11Dis_rnn['ed2'][:,d1x,d2x], cosTheta_bsl_rnn['ed2'], tail='smaller', alpha = 5) #.mean(1)
    ax.text(0.2+nk,1.0, f'{f_plotting.sig_marker(p1,ns_note=False)}',horizontalalignment='center', verticalalignment='bottom', fontsize=10)
    print(f"R@R: M(SD) = {cosTheta11Dis_rnn['ed2'][:,d1x,d2x].mean():.3f}({cosTheta11Dis_rnn['ed2'][:,d1x,d2x].std():.3f}), 95CI = [{np.quantile(cosTheta11Dis_rnn['ed2'][:,d1x,d2x],0.025):.3f}, {np.quantile(cosTheta11Dis_rnn['ed2'][:,d1x,d2x],0.975):.3f}], p = {p1:.3f}")


    ax.boxplot([cosTheta11Dis_rnn['ed12'][:,d1x,d2x]], positions=[0.4+nk-0.035*showShuffBsl], patch_artist=True, widths = 0.05*len(showCheckpoints), boxprops=boxprops2, flierprops=flierprops2, 
                    meanprops=meanpointprops2, medianprops=medianprops, capprops = capprops2, whiskerprops = whiskerprops2, meanline=False, showmeans=True)    
    p2 = f_stats.permutation_pCI(cosTheta11Dis_rnn['ed12'][:,d1x,d2x], cosTheta_bsl_rnn['ed12'], tail='smaller', alpha = 5)
    ax.text(0.4+nk,1.0, f'{f_plotting.sig_marker(p2,ns_note=False)}',horizontalalignment='center', verticalalignment='bottom', fontsize=10)
    print(f"R&U: M(SD) = {cosTheta11Dis_rnn['ed12'][:,d1x,d2x].mean():.3f}({cosTheta11Dis_rnn['ed12'][:,d1x,d2x].std():.3f}), 95CI = [{np.quantile(cosTheta11Dis_rnn['ed12'][:,d1x,d2x],0.025):.3f}, {np.quantile(cosTheta11Dis_rnn['ed12'][:,d1x,d2x],0.975):.3f}], p = {p2:.3f}")


    ax.boxplot([cosTheta11Dis_data['dlpfc'][:,d1x,d2x]], positions=[0.6+nk-0.035*showShuffBsl], patch_artist=True, widths = 0.05*len(showCheckpoints), boxprops=boxprops3, flierprops=flierprops3, 
                    meanprops=meanpointprops3, medianprops=medianprops, capprops = capprops3, whiskerprops = whiskerprops3, meanline=False, showmeans=True)
    p3 = f_stats.permutation_pCI(cosTheta11Dis_data['dlpfc'][:,d1x,d2x], cosTheta_bsl_data['dlpfc'], tail='smaller', alpha = 5) #.mean(1)
    ax.text(0.6+nk,1.0, f'{f_plotting.sig_marker(p3,ns_note=False)}',horizontalalignment='center', verticalalignment='bottom', fontsize=10)
    print(f"LPFC: M(SD) = {cosTheta11Dis_data['dlpfc'][:,d1x,d2x].mean():.3f}({cosTheta11Dis_data['dlpfc'][:,d1x,d2x].std():.3f}), 95CI = [{np.quantile(cosTheta11Dis_data['dlpfc'][:,d1x,d2x],0.025):.3f}, {np.quantile(cosTheta11Dis_data['dlpfc'][:,d1x,d2x],0.975):.3f}], p = {p3:.3f}")


    ax.boxplot([cosTheta11Dis_data['fef'][:,d1x,d2x]], positions=[0.8+nk-0.035*showShuffBsl], patch_artist=True, widths = 0.05*len(showCheckpoints), boxprops=boxprops4, flierprops=flierprops4, 
                    meanprops=meanpointprops4, medianprops=medianprops, capprops = capprops4, whiskerprops = whiskerprops4, meanline=False, showmeans=True)
    p4 = f_stats.permutation_pCI(cosTheta11Dis_data['fef'][:,d1x,d2x], cosTheta_bsl_data['fef'], tail='smaller', alpha = 5) #.mean(1)
    ax.text(0.8+nk,1.0, f'{f_plotting.sig_marker(p4,ns_note=False)}',horizontalalignment='center', verticalalignment='bottom', fontsize=10)
    print(f"FEF: M(SD) = {cosTheta11Dis_data['fef'][:,d1x,d2x].mean():.3f}({cosTheta11Dis_data['fef'][:,d1x,d2x].std():.3f}), 95CI = [{np.quantile(cosTheta11Dis_data['fef'][:,d1x,d2x],0.025):.3f}, {np.quantile(cosTheta11Dis_data['fef'][:,d1x,d2x],0.975):.3f}], p = {p4:.3f}")

    if showShuffBsl:
        #ax.boxplot([cosTheta12Ret_shuff_rnn['ed2'][:,:,d1x,d2x].mean(1)], positions=[0.2+nk+0.035*showShuffBsl], patch_artist=True, widths = 0.05*len(showCheckpoints), boxprops=boxprops0, flierprops=flierprops0, 
        #            meanprops=meanpointprops0, medianprops=medianprops, capprops = capprops0, whiskerprops = whiskerprops0, meanline=False, showmeans=True)
        
        #ax.boxplot([cosTheta12Ret_shuff_rnn['ed12'][:,:,d1x,d2x].mean(1)], positions=[0.4+nk+0.035*showShuffBsl], patch_artist=True, widths = 0.05*len(showCheckpoints), boxprops=boxprops0, flierprops=flierprops0, 
        #                meanprops=meanpointprops0, medianprops=medianprops, capprops = capprops0, whiskerprops = whiskerprops0, meanline=False, showmeans=True)
        
        #ax.boxplot([cosTheta12Ret_shuff_data['dlpfc'][:,:,d1x,d2x].mean(1)], positions=[0.6+nk+0.035*showShuffBsl], patch_artist=True, widths = 0.05*len(showCheckpoints), boxprops=boxprops0, flierprops=flierprops0, 
        #                meanprops=meanpointprops0, medianprops=medianprops, capprops = capprops0, whiskerprops = whiskerprops0, meanline=False, showmeans=True)
        
        #ax.boxplot([cosTheta12Ret_shuff_data['fef'][:,:,d1x,d2x].mean(1)], positions=[0.8+nk+0.035*showShuffBsl], patch_artist=True, widths = 0.05*len(showCheckpoints), boxprops=boxprops0, flierprops=flierprops0, 
        #                meanprops=meanpointprops0, medianprops=medianprops, capprops = capprops0, whiskerprops = whiskerprops0, meanline=False, showmeans=True)
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
        
    # KS Test
    ks_results = {'R@R-LPFC':scipy.stats.ks_2samp(cosTheta11Dis_rnn['ed2'][:,d1x,d2x], cosTheta11Dis_data['dlpfc'][:,d1x,d2x]),
                  'R@R-FEF':scipy.stats.ks_2samp(cosTheta11Dis_rnn['ed2'][:,d1x,d2x], cosTheta11Dis_data['fef'][:,d1x,d2x]),
                  'R&U-LPFC':scipy.stats.ks_2samp(cosTheta11Dis_rnn['ed12'][:,d1x,d2x], cosTheta11Dis_data['dlpfc'][:,d1x,d2x]), 
                  'R&U-FEF':scipy.stats.ks_2samp(cosTheta11Dis_rnn['ed12'][:,d1x,d2x], cosTheta11Dis_data['fef'][:,d1x,d2x])}

    # FK Test
    fk_results = {'R@R-LPFC':scipy.stats.fligner(cosTheta11Dis_rnn['ed2'][:,d1x,d2x],cosTheta11Dis_data['dlpfc'][:,d1x,d2x]),
                  'R@R-FEF':scipy.stats.fligner(cosTheta11Dis_rnn['ed2'][:,d1x,d2x],cosTheta11Dis_data['fef'][:,d1x,d2x]),
                  'R&U-LPFC':scipy.stats.fligner(cosTheta11Dis_rnn['ed12'][:,d1x,d2x],cosTheta11Dis_data['dlpfc'][:,d1x,d2x]),
                  'R&U-FEF':scipy.stats.fligner(cosTheta11Dis_rnn['ed12'][:,d1x,d2x],cosTheta11Dis_data['fef'][:,d1x,d2x])}
    
    print('############### K-S Test ##############')
    for k in ks_results.keys():
        print(f'{k.upper()}: D = {ks_results[k][0]:.3f}, p = {ks_results[k][1]:.3f}')

    print('############### F-K Test ##############')
    for k in fk_results.keys():
        print(f'{k.upper()}: H = {fk_results[k][0]:.3f}, p = {fk_results[k][1]:.3f}')
    
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
    
# cosPsi
ax = axes.flatten()[1]
xtks, xtklabs = [], []
for nk, k in enumerate(showCheckpoints):
    dk1, dk2 = k+'D1', k+'D2'
    d1x, d2x = checkpointsLabels.index(dk1), checkpointsLabels.index(dk2)
    #ld1x, ld2x = checkpointsLabels.index('LD1'), checkpointsLabels.index('LD2')

    print('############## cosPsi ##############')

    # method: baseline from split sets
    cosPsi_bsl_rnn = {kk: np.mean((np.array([cosPsi11_bsl_rnn[kk][n][2] for n in range(100)]).squeeze()[:,:,d1x].mean(1),
                                   np.array([cosPsi11_bsl_rnn[kk][n][2] for n in range(100)]).squeeze()[:,:,d2x].mean(1)),axis=0) for kk in ('ed2','ed12')}
    cosPsi_bsl_data = {kk: np.mean((cosPsi11_bsl_data[kk][2][:,:,d1x].mean(1), 
                                     cosPsi11_bsl_data[kk][2][:,:,d2x].mean(1)),axis=0) for kk in ('dlpfc','fef')}
    
    
    ax.boxplot([cosPsi11Dis_rnn['ed2'][:,d1x,d2x]], positions=[0.2+nk-0.035*showShuffBsl], patch_artist=True, widths = 0.05*len(showCheckpoints), boxprops=boxprops1, flierprops=flierprops1, 
                    meanprops=meanpointprops1, medianprops=medianprops, capprops = capprops1, whiskerprops = whiskerprops1, meanline=False, showmeans=True)
    p1 = f_stats.permutation_pCI(cosPsi11Dis_rnn['ed2'][:,d1x,d2x], cosPsi_bsl_rnn['ed2'], tail='smaller', alpha = 5)
    ax.text(0.2+nk,1.0, f'{f_plotting.sig_marker(p1,ns_note=False)}',horizontalalignment='center', verticalalignment='bottom', fontsize=10)
    print(f"R@R: M(SD) = {cosPsi11Dis_rnn['ed2'][:,d1x,d2x].mean():.3f}({cosPsi11Dis_rnn['ed2'][:,d1x,d2x].std():.3f}), 95CI = [{np.quantile(cosPsi11Dis_rnn['ed2'][:,d1x,d2x],0.025):.3f}, {np.quantile(cosPsi11Dis_rnn['ed2'][:,d1x,d2x],0.975):.3f}], p = {p1:.3f}")


    ax.boxplot([cosPsi11Dis_rnn['ed12'][:,d1x,d2x]], positions=[0.4+nk-0.035*showShuffBsl], patch_artist=True, widths = 0.05*len(showCheckpoints), boxprops=boxprops2, flierprops=flierprops2, 
                    meanprops=meanpointprops2, medianprops=medianprops, capprops = capprops2, whiskerprops = whiskerprops2, meanline=False, showmeans=True)
    p2 = f_stats.permutation_pCI(cosPsi11Dis_rnn['ed12'][:,d1x,d2x], cosPsi_bsl_rnn['ed12'], tail='smaller', alpha = 5)
    ax.text(0.4+nk,1.0, f'{f_plotting.sig_marker(p2,ns_note=False)}',horizontalalignment='center', verticalalignment='bottom', fontsize=10)
    print(f"R&U: M(SD) = {cosPsi11Dis_rnn['ed12'][:,d1x,d2x].mean():.3f}({cosPsi11Dis_rnn['ed12'][:,d1x,d2x].std():.3f}), 95CI = [{np.quantile(cosPsi11Dis_rnn['ed12'][:,d1x,d2x],0.025):.3f}, {np.quantile(cosPsi11Dis_rnn['ed12'][:,d1x,d2x],0.975):.3f}], p = {p2:.3f}")


    ax.boxplot([cosPsi11Dis_data['dlpfc'][:,d1x,d2x]], positions=[0.6+nk-0.035*showShuffBsl], patch_artist=True, widths = 0.05*len(showCheckpoints), boxprops=boxprops3, flierprops=flierprops3, 
                    meanprops=meanpointprops3, medianprops=medianprops, capprops = capprops3, whiskerprops = whiskerprops3, meanline=False, showmeans=True)
    p3 = f_stats.permutation_pCI(cosPsi11Dis_data['dlpfc'][:,d1x,d2x], cosPsi_bsl_data['dlpfc'], tail='smaller', alpha = 5)
    ax.text(0.6+nk,1.0, f'{f_plotting.sig_marker(p3,ns_note=False)}',horizontalalignment='center', verticalalignment='bottom', fontsize=10)
    print(f"LPFC: M(SD) = {cosPsi11Dis_data['dlpfc'][:,d1x,d2x].mean():.3f}({cosPsi11Dis_data['dlpfc'][:,d1x,d2x].std():.3f}), 95CI = [{np.quantile(cosPsi11Dis_data['dlpfc'][:,d1x,d2x],0.025):.3f}, {np.quantile(cosPsi11Dis_data['dlpfc'][:,d1x,d2x],0.975):.3f}], p = {p3:.3f}")


    ax.boxplot([cosPsi11Dis_data['fef'][:,d1x,d2x]], positions=[0.8+nk-0.035*showShuffBsl], patch_artist=True, widths = 0.05*len(showCheckpoints), boxprops=boxprops4, flierprops=flierprops4, 
                    meanprops=meanpointprops4, medianprops=medianprops, capprops = capprops4, whiskerprops = whiskerprops4, meanline=False, showmeans=True)
    p4 = f_stats.permutation_pCI(cosPsi11Dis_data['fef'][:,d1x,d2x], cosPsi_bsl_data['fef'], tail='smaller', alpha = 5)
    ax.text(0.8+nk,1.0, f'{f_plotting.sig_marker(p4,ns_note=False)}',horizontalalignment='center', verticalalignment='bottom', fontsize=10)
    print(f"FEF: M(SD) = {cosPsi11Dis_data['fef'][:,d1x,d2x].mean():.3f}({cosPsi11Dis_data['fef'][:,d1x,d2x].std():.3f}), 95CI = [{np.quantile(cosPsi11Dis_data['fef'][:,d1x,d2x],0.025):.3f}, {np.quantile(cosPsi11Dis_data['fef'][:,d1x,d2x],0.975):.3f}], p = {p4:.3f}")

    if showShuffBsl:
        #ax.boxplot([cosPsi12Ret_shuff_rnn['ed2'][:,:,d1x,d2x].mean(1)], positions=[0.2+nk+0.035*showShuffBsl], patch_artist=True, widths = 0.05*len(showCheckpoints), boxprops=boxprops0, flierprops=flierprops0, 
        #            meanprops=meanpointprops0, medianprops=medianprops, capprops = capprops0, whiskerprops = whiskerprops0, meanline=False, showmeans=True)
        
        #ax.boxplot([cosPsi12Ret_shuff_rnn['ed12'][:,:,d1x,d2x].mean(1)], positions=[0.4+nk+0.035*showShuffBsl], patch_artist=True, widths = 0.05*len(showCheckpoints), boxprops=boxprops0, flierprops=flierprops0, 
        #                meanprops=meanpointprops0, medianprops=medianprops, capprops = capprops0, whiskerprops = whiskerprops0, meanline=False, showmeans=True)
        
        #ax.boxplot([cosPsi12Ret_shuff_data['dlpfc'][:,:,d1x,d2x].mean(1)], positions=[0.6+nk+0.035*showShuffBsl], patch_artist=True, widths = 0.05*len(showCheckpoints), boxprops=boxprops0, flierprops=flierprops0, 
        #                meanprops=meanpointprops0, medianprops=medianprops, capprops = capprops0, whiskerprops = whiskerprops0, meanline=False, showmeans=True)
        
        #ax.boxplot([cosPsi12Ret_shuff_data['fef'][:,:,d1x,d2x].mean(1)], positions=[0.8+nk+0.035*showShuffBsl], patch_artist=True, widths = 0.05*len(showCheckpoints), boxprops=boxprops0, flierprops=flierprops0, 
        #                meanprops=meanpointprops0, medianprops=medianprops, capprops = capprops0, whiskerprops = whiskerprops0, meanline=False, showmeans=True)
        
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

    # KS test
    ks_results = {'R@R-LPFC':scipy.stats.ks_2samp(cosPsi11Dis_rnn['ed2'][:,d1x,d2x], cosPsi11Dis_data['dlpfc'][:,d1x,d2x]),
                  'R@R-FEF':scipy.stats.ks_2samp(cosPsi11Dis_rnn['ed2'][:,d1x,d2x], cosPsi11Dis_data['fef'][:,d1x,d2x]),
                  'R&U-LPFC':scipy.stats.ks_2samp(cosPsi11Dis_rnn['ed12'][:,d1x,d2x], cosPsi11Dis_data['dlpfc'][:,d1x,d2x]), 
                  'R&U-FEF':scipy.stats.ks_2samp(cosPsi11Dis_rnn['ed12'][:,d1x,d2x], cosPsi11Dis_data['fef'][:,d1x,d2x])}

    fk_results = {'R@R-LPFC':scipy.stats.fligner(cosPsi11Dis_rnn['ed2'][:,d1x,d2x], cosPsi11Dis_data['dlpfc'][:,d1x,d2x]),
                  'R@R-FEF':scipy.stats.fligner(cosPsi11Dis_rnn['ed2'][:,d1x,d2x], cosPsi11Dis_data['fef'][:,d1x,d2x]),
                  'R&U-LPFC':scipy.stats.fligner(cosPsi11Dis_rnn['ed12'][:,d1x,d2x], cosPsi11Dis_data['dlpfc'][:,d1x,d2x]), 
                  'R&U-FEF':scipy.stats.fligner(cosPsi11Dis_rnn['ed12'][:,d1x,d2x], cosPsi11Dis_data['fef'][:,d1x,d2x])}

    print('############### K-S Test ##############')
    for k in ks_results.keys():
        print(f'{k.upper()}: D = {ks_results[k][0]:.3f}, p = {ks_results[k][1]:.3f}')
        
    print('############### F-K Test ##############')
    for k in fk_results.keys():
        print(f'{k.upper()}: H = {fk_results[k][0]:.3f}, p = {fk_results[k][1]:.3f}')
    
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


plt.suptitle(f'I1D1 vs. I1D2, Distractor', fontsize = 25, y=1) # (Monkeys)
plt.tight_layout()
plt.show()
#%%
fig.savefig(f'{phd_path}/outputs/I1D1vI2D2_dis_LD.tif', bbox_inches='tight')
#%%



#%% check colinearity
import pycircstat
def create_vonmises(rads,mu=0):
    R_bar = pycircstat.descriptive.resultant_vector_length(rads)
    
    kappa = R_bar * (2 - R_bar**2) / (1 - R_bar**2)
    #if R_bar < 0.85:
    #    kappa = R_bar * (2 - R_bar**2) / (1 - R_bar**2)
    #else:
        # Approximation for larger values of R̄
    #    kappa = 1 / (R_bar * (R_bar - R_bar**3))
    
    temp_vonmises = np.random.vonmises(mu, kappa, size = len(rads))
    
    return temp_vonmises
#%%
#cosTheta12Ret_shuff_rnn_parallel = {k:np.cos(create_vonmises(np.arccos(cosTheta12Ret_rnn[k][:,d1x,d2x]))) for k in ('ed2','ed12')}
#cosTheta12Ret_shuff_data_parallel = {k:np.cos(create_vonmises(np.arccos(cosTheta12Ret_data[k][:,d1x,d2x]))) for k in ('dlpfc','fef')}
#cosTheta12Ret_shuff_rnn_parallel = {k:np.cos(create_vonmises(np.arccos(cosTheta12Ret_shuff_rnn[k][:,:,d1x,d2x].mean(1)))) for k in ('ed2','ed12')}
#cosTheta12Ret_shuff_data_parallel = {k:np.cos(create_vonmises(np.arccos(cosTheta12Ret_shuff_data[k][:,:,d1x,d2x].mean(1)))) for k in ('dlpfc','fef')}

#cosTheta12Ret_shuff_rnn_parallel = {k:np.abs(cosTheta12Ret_shuff_rnn[k][:,:,d1x,d2x].mean(1)) for k in ('ed2','ed12')}
#cosTheta12Ret_shuff_data_parallel = {k:np.abs(cosTheta12Ret_shuff_data[k][:,:,d1x,d2x].mean(1)) for k in ('dlpfc','fef')}

#%%
#cosTheta_bsl_data = {k:np.concatenate(np.concatenate(cosTheta_bsl_data[k])) for k in ('dlpfc','fef')}
#cosPsi_bsl_data = {k:np.concatenate(np.concatenate(cosPsi_bsl_data[k])) for k in ('dlpfc','fef')}

#cosTheta_bsl_rnn = {k:np.concatenate(np.concatenate(cosTheta_bsl_rnn[k])) for k in ('ed2','ed12')}
#cosPsi_bsl_rnn = {k:np.concatenate(np.concatenate(cosPsi_bsl_rnn[k])) for k in ('ed2','ed12')}
#%% NON-USED

cosTheta_bsl_data = {k:np.concatenate((cosTheta11_bsl_data[k][1], cosTheta22_bsl_data[k][1],
                                       cosTheta11_bsl_data[k][2], cosTheta22_bsl_data[k][2]), axis = 1) for k in ('dlpfc','fef')}
cosPsi_bsl_data = {k:np.concatenate((cosPsi11_bsl_data[k][1], cosPsi22_bsl_data[k][1],
                                     cosPsi11_bsl_data[k][2], cosPsi22_bsl_data[k][2]), axis = 1) for k in ('dlpfc','fef')}

cosTheta_bsl_rnn = {k:[] for k in ('ed2','ed12')}
cosPsi_bsl_rnn = {k:[] for k in ('ed2','ed12')}
for k in ('ed2','ed12'):    
    cosTheta_bsl_rnn[k] = np.concatenate((np.array([cosTheta11_bsl_rnn['ed2'][n][1] for n in range(100)]).squeeze(),
                                         np.array([cosTheta22_bsl_rnn['ed2'][n][2] for n in range(100)]).squeeze(),
                                         np.array([cosTheta11_bsl_rnn['ed12'][n][1] for n in range(100)]).squeeze(),
                                         np.array([cosTheta22_bsl_rnn['ed12'][n][2] for n in range(100)]).squeeze()), axis = 1)
    
    cosPsi_bsl_rnn[k] = np.concatenate((np.array([cosPsi11_bsl_rnn['ed2'][n][1] for n in range(100)]).squeeze(),
                                       np.array([cosPsi22_bsl_rnn['ed2'][n][2] for n in range(100)]).squeeze(),
                                       np.array([cosPsi11_bsl_rnn['ed12'][n][1] for n in range(100)]).squeeze(),
                                       np.array([cosPsi22_bsl_rnn['ed12'][n][2] for n in range(100)]).squeeze()), axis = 1)

# subset bsl method
angleCheckPoints = np.linspace(0,np.pi,7).round(5)
cmap = plt.get_cmap('coolwarm')
norm = mcolors.BoundaryNorm(np.flip(np.cos(angleCheckPoints).round(5)),cmap.N, clip=True)

showCheckpoints = ('L',) #'E',
showShuffBsl = True
fig, axes = plt.subplots(1,1, sharey=True, figsize=(5,4), dpi=300)

parallelThreshold = 15/90

# cosTheta
ax = axes#.flatten()[0]
xtks, xtklabs = [], []
for nk, k in enumerate(showCheckpoints):
    dk1, dk2 = k+'D1', k+'D2'
    d1x, d2x = checkpointsLabels.index(dk1), checkpointsLabels.index(dk2)
    
    cosTheta_bsl_data = {k:(cosTheta_bsl_data[k]).mean(1)[:,[d1x,d2x]].mean(1) for k in ('dlpfc','fef')}
    cosPsi_bsl_data = {k:(cosPsi_bsl_data[k]).mean(1)[:,[d1x,d2x]].mean(1) for k in ('dlpfc','fef')}

    cosTheta_bsl_rnn = {k:(cosTheta_bsl_rnn[k]).mean(1)[:,[d1x,d2x]].mean(1) for k in ('ed2','ed12')}
    cosPsi_bsl_rnn = {k:(cosPsi_bsl_rnn[k]).mean(1)[:,[d1x,d2x]].mean(1) for k in ('ed2','ed12')}
    
    #cosTheta12Ret_shuff_rnn_parallel = {k:np.cos(create_vonmises(np.arccos(cosTheta12Ret_rnn[k][:,d1x,d2x]))) for k in ('ed2','ed12')}
    #cosTheta12Ret_shuff_data_parallel = {k:np.cos(create_vonmises(np.arccos(cosTheta12Ret_data[k][:,d1x,d2x]))) for k in ('dlpfc','fef')}
    
    # method: move median to 0
    #cosTheta12Ret_shuff_rnn_parallel = {k:np.cos(np.arccos(cosTheta12Ret_rnn[k][:,d1x,d2x]) - np.median(np.arccos(cosTheta12Ret_rnn[k][:,d1x,d2x]))) for k in ('ed2','ed12')}
    #cosTheta12Ret_shuff_data_parallel = {k:np.cos(np.arccos(cosTheta12Ret_data[k][:,d1x,d2x]) - np.median(np.arccos(cosTheta12Ret_data[k][:,d1x,d2x]))) for k in ('dlpfc','fef')}
    
    #cosTheta12Ret_shuff_rnn_parallel = {k:np.cos(create_vonmises(np.arccos(cosTheta12Ret_shuff_rnn[k][:,:,d1x,d2x].mean(1)))) for k in ('ed2','ed12')}
    #cosTheta12Ret_shuff_data_parallel = {k:np.cos(create_vonmises(np.arccos(cosTheta12Ret_shuff_data[k][:,:,d1x,d2x].mean(1)))) for k in ('dlpfc','fef')}
    
    #cosTheta12Ret_shuff_rnn_parallel = {k:np.abs(cosTheta12Ret_shuff_rnn[k][:,:,d1x,d2x].mean(1)) for k in ('ed2','ed12')}
    #cosTheta12Ret_shuff_data_parallel = {k:np.abs(cosTheta12Ret_shuff_data[k][:,:,d1x,d2x].mean(1)) for k in ('dlpfc','fef')}

    ax.boxplot([cosTheta12Ret_rnn['ed2'][:,d1x,d2x]], positions=[0.2+nk-0.035*showShuffBsl], patch_artist=True, widths = 0.05*len(showCheckpoints), boxprops=boxprops1, flierprops=flierprops1, 
                    meanprops=meanpointprops1, medianprops=medianprops, capprops = capprops1, whiskerprops = whiskerprops1, meanline=False, showmeans=True)
    
    #p1 = stats.ks_2samp(cosTheta12Ret_rnn['ed2'][:,d1x,d2x], np.concatenate(cosTheta12Ret_shuff_rnn['ed2'][:,:,d1x,d2x]))[1].round(3) #.mean(1)
    p1 = f_stats.permutation_pCI(cosTheta12Ret_rnn['ed2'][:,d1x,d2x], cosTheta_bsl_rnn['ed2'], tail='smaller', alpha = 5) #.mean(1)
    #p1 = f_stats.p_1samp(np.abs(cosTheta12Ret_rnn['ed2'][:,d1x,d2x]), np.cos(parallelThreshold*(np.pi*0.5)), tail='greater') #.mean(1)
    #p1 = f_stats.p_1samp_bootstrap(np.abs(cosTheta12Ret_rnn['ed2'][:,d1x,d2x]), np.cos(parallelThreshold*(np.pi*0.5)), tail='greater', method='median') #.mean(1)
    
    #ax.text(0.2+nk,1.0, f'{f_plotting.sig_marker(p1,ns_note=False)}',horizontalalignment='center', verticalalignment='bottom', fontsize=10)
    ax.text(0.2+nk,1.05, f'{p1:.3f}',horizontalalignment='center', verticalalignment='bottom', fontsize=6)
    
    ax.boxplot([cosTheta12Ret_rnn['ed12'][:,d1x,d2x]], positions=[0.4+nk-0.035*showShuffBsl], patch_artist=True, widths = 0.05*len(showCheckpoints), boxprops=boxprops2, flierprops=flierprops2, 
                    meanprops=meanpointprops2, medianprops=medianprops, capprops = capprops2, whiskerprops = whiskerprops2, meanline=False, showmeans=True)
    
    #p2 = stats.ks_2samp(cosTheta12Ret_rnn['ed12'][:,d1x,d2x], np.concatenate(cosTheta12Ret_shuff_rnn['ed12'][:,:,d1x,d2x]))[1].round(3) #.mean(1)
    p2 = f_stats.permutation_pCI(cosTheta12Ret_rnn['ed12'][:,d1x,d2x], cosTheta_bsl_rnn['ed12'], tail='smaller', alpha = 5)
    #p2 = f_stats.p_1samp(np.abs(cosTheta12Ret_rnn['ed12'][:,d1x,d2x]), np.cos(parallelThreshold*(np.pi*0.5)), tail='greater') #.mean(1)
    #p2 = f_stats.p_1samp_bootstrap(np.abs(cosTheta12Ret_rnn['ed12'][:,d1x,d2x]), np.cos(parallelThreshold*(np.pi*0.5)), tail='greater', method='median') #.mean(1)
    
    #ax.text(0.4+nk,1.0, f'{f_plotting.sig_marker(p2,ns_note=False)}',horizontalalignment='center', verticalalignment='bottom', fontsize=10)
    ax.text(0.4+nk,1.05, f'{p2:.3f}',horizontalalignment='center', verticalalignment='bottom', fontsize=6)
    
    ax.boxplot([cosTheta12Ret_data['dlpfc'][:,d1x,d2x]], positions=[0.6+nk-0.035*showShuffBsl], patch_artist=True, widths = 0.05*len(showCheckpoints), boxprops=boxprops3, flierprops=flierprops3, 
                    meanprops=meanpointprops3, medianprops=medianprops, capprops = capprops3, whiskerprops = whiskerprops3, meanline=False, showmeans=True)
    
    #p3 = stats.ks_2samp(cosTheta12Ret_data['dlpfc'][:,d1x,d2x], np.concatenate(cosTheta12Ret_shuff_data['dlpfc'][:,:,d1x,d2x]))[1].round(3) #.mean(1)
    p3 = f_stats.permutation_pCI(cosTheta12Ret_data['dlpfc'][:,d1x,d2x], cosTheta_bsl_data['dlpfc'], tail='smaller', alpha = 5) #.mean(1)
    #p3 = f_stats.p_1samp(np.abs(cosTheta12Ret_data['dlpfc'][:,d1x,d2x]), np.cos(parallelThreshold*(np.pi*0.5)), tail='greater') #.mean(1)
    #p3 = f_stats.p_1samp_bootstrap(np.abs(cosTheta12Ret_data['dlpfc'][:,d1x,d2x]), np.cos(parallelThreshold*(np.pi*0.5)), tail='greater', method='median') #.mean(1)
    
    #ax.text(0.6+nk,1.0, f'{f_plotting.sig_marker(p3,ns_note=False)}',horizontalalignment='center', verticalalignment='bottom', fontsize=10)
    ax.text(0.6+nk,1.05, f'{p3:.3f}',horizontalalignment='center', verticalalignment='bottom', fontsize=6)
    
    ax.boxplot([cosTheta12Ret_data['fef'][:,d1x,d2x]], positions=[0.8+nk-0.035*showShuffBsl], patch_artist=True, widths = 0.05*len(showCheckpoints), boxprops=boxprops4, flierprops=flierprops4, 
                    meanprops=meanpointprops4, medianprops=medianprops, capprops = capprops4, whiskerprops = whiskerprops4, meanline=False, showmeans=True)
    
    #p4 = stats.ks_2samp(cosTheta12Ret_data['fef'][:,d1x,d2x], np.concatenate(cosTheta12Ret_shuff_data['fef'][:,:,d1x,d2x]))[1].round(3) #.mean(1)
    p4 = f_stats.permutation_pCI(cosTheta12Ret_data['fef'][:,d1x,d2x], cosTheta_bsl_data['fef'], tail='smaller', alpha = 5) #.mean(1)
    #p4 = f_stats.p_1samp(np.abs(cosTheta12Ret_data['fef'][:,d1x,d2x]), np.cos(parallelThreshold*(np.pi*0.5)), tail='greater') #.mean(1)
    #p4 = f_stats.p_1samp_bootstrap(np.abs(cosTheta12Ret_data['fef'][:,d1x,d2x]), np.cos(parallelThreshold*(np.pi*0.5)), tail='greater', method='median') #.mean(1)
    
    #ax.text(0.8+nk,1.0, f'{f_plotting.sig_marker(p4,ns_note=False)}',horizontalalignment='center', verticalalignment='bottom', fontsize=10)
    ax.text(0.8+nk,1.05, f'{p4:.3f}',horizontalalignment='center', verticalalignment='bottom', fontsize=6)
    
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

#ax.plot(np.arange(0,len(showCheckpoints)+0.1), np.full_like(np.arange(0,len(showCheckpoints)+0.1),(np.cos(parallelThreshold*(np.pi*0.5)))), c='k', ls='--',alpha=0.3)
ax.set_xticks(xtks,xtklabs)#,rotation=20
ax.set_xlim(0,len(showCheckpoints))
ax.set_ylim(-1.05,1.2)
ax.set_yticks(np.flip(np.cos(angleCheckPoints).round(5)), labels = np.flip(np.degrees(angleCheckPoints).round(1).astype(int)))
ax.set_xlabel('Timebins', labelpad = 3, fontsize = 15)
ax.set_ylabel('cos(θ)', labelpad = 0, fontsize = 15)
ax.set_title(f'Coplanarity', fontsize = 17)
ax.tick_params(axis='both', labelsize=12)


ax.plot([], c=color1, label='R@R')
ax.plot([], c=color2, label='R&U')
ax.plot([], c=color3, label='LPFC')
ax.plot([], c=color4, label='FEF')
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize = 12)


plt.suptitle(f'I1D1 vs. I2D2, Retarget', fontsize = 15, y=1) # (Monkeys)
plt.tight_layout()
plt.show()
#%%
#%% NON-USED
angleCheckPoints = np.linspace(0,np.pi,7).round(5)
cmap = plt.get_cmap('coolwarm')
norm = mcolors.BoundaryNorm(np.flip(np.cos(angleCheckPoints).round(5)),cmap.N, clip=True)

showCheckpoints = ('L',) #'E',
showShuffBsl = True
fig, axes = plt.subplots(1,1, sharey=True, figsize=(5,4), dpi=300)

parallelThreshold = 15/90

# cosTheta
ax = axes#.flatten()[0]
xtks, xtklabs = [], []
for nk, k in enumerate(showCheckpoints):
    dk1, dk2 = k+'D1', k+'D2'
    d1x, d2x = checkpointsLabels.index(dk1), checkpointsLabels.index(dk2)
    
    # method: move median to 0
    #cosTheta_bsl_rnn = {k:np.cos(np.arccos(cosTheta12Ret_rnn[k][:,d1x,d2x]) - np.median(np.arccos(cosTheta12Ret_rnn[k][:,d1x,d2x]))) for k in ('ed2','ed12')}
    #cosTheta_bsl_data = {k:np.cos(np.arccos(cosTheta12Ret_data[k][:,d1x,d2x]) - np.median(np.arccos(cosTheta12Ret_data[k][:,d1x,d2x]))) for k in ('dlpfc','fef')}
    
    # method: create von Mises distribution (mu, kappa)
    #cosTheta_bsl_rnn = {k:np.cos(np.random.vonmises(0, kappa=8, size = 100)) for k in ('ed2','ed12')}
    #cosTheta_bsl_data = {k:np.cos(np.random.vonmises(0, kappa=8, size = 100)) for k in ('dlpfc','fef')}
    #cosTheta_bsl_rnn = {k:np.cos(create_vonmises(np.arccos(cosTheta12Ret_rnn[k][:,d1x,d2x]))) for k in ('ed2','ed12')}
    #cosTheta_bsl_data = {k:np.cos(create_vonmises(np.arccos(cosTheta12Ret_data[k][:,d1x,d2x]))) for k in ('dlpfc','fef')}
    
    # method: baseline from split sets
    cosTheta_bsl_rnn = {kk: np.mean((np.array([cosTheta11_bsl_rnn[kk][n][1] for n in range(100)]).squeeze()[:,:,d1x].mean(1),
                                   np.array([cosTheta22_bsl_rnn[kk][n][1] for n in range(100)]).squeeze()[:,:,d2x].mean(1)),axis=0) for kk in ('ed2','ed12')}
    cosTheta_bsl_data = {kk: np.mean((cosTheta11_bsl_data[kk][1][:,:,d1x].mean(1), 
                                     cosTheta22_bsl_data[kk][1][:,:,d2x].mean(1)),axis=0) for kk in ('dlpfc','fef')}
    
    
    ax.boxplot([cosTheta12Ret_rnn['ed2'][:,d1x,d2x]], positions=[0.2+nk-0.035*showShuffBsl], patch_artist=True, widths = 0.05*len(showCheckpoints), boxprops=boxprops1, flierprops=flierprops1, 
                    meanprops=meanpointprops1, medianprops=medianprops, capprops = capprops1, whiskerprops = whiskerprops1, meanline=False, showmeans=True)
    
    p1 = f_stats.permutation_pCI(cosTheta12Ret_rnn['ed2'][:,d1x,d2x], cosTheta_bsl_rnn['ed2'], tail='smaller', alpha = 5) #.mean(1)
    #p1 = f_stats.p_1samp(np.abs(cosTheta12Ret_rnn['ed2'][:,d1x,d2x]), np.cos(parallelThreshold*(np.pi*0.5)), tail='greater') #.mean(1)
    #p1 = f_stats.p_1samp_bootstrap(np.abs(cosTheta12Ret_rnn['ed2'][:,d1x,d2x]), np.cos(parallelThreshold*(np.pi*0.5)), tail='greater', method='median') #.mean(1)
    
    #ax.text(0.2+nk,1.0, f'{f_plotting.sig_marker(p1,ns_note=False)}',horizontalalignment='center', verticalalignment='bottom', fontsize=10)
    ax.text(0.2+nk,1.05, f'{p1:.3f}',horizontalalignment='center', verticalalignment='bottom', fontsize=6)
    
    ax.boxplot([cosTheta12Ret_rnn['ed12'][:,d1x,d2x]], positions=[0.4+nk-0.035*showShuffBsl], patch_artist=True, widths = 0.05*len(showCheckpoints), boxprops=boxprops2, flierprops=flierprops2, 
                    meanprops=meanpointprops2, medianprops=medianprops, capprops = capprops2, whiskerprops = whiskerprops2, meanline=False, showmeans=True)
    
    p2 = f_stats.permutation_pCI(cosTheta12Ret_rnn['ed12'][:,d1x,d2x], cosTheta_bsl_rnn['ed12'], tail='smaller', alpha = 5)
    #p2 = f_stats.p_1samp(np.abs(cosTheta12Ret_rnn['ed12'][:,d1x,d2x]), np.cos(parallelThreshold*(np.pi*0.5)), tail='greater') #.mean(1)
    #p2 = f_stats.p_1samp_bootstrap(np.abs(cosTheta12Ret_rnn['ed12'][:,d1x,d2x]), np.cos(parallelThreshold*(np.pi*0.5)), tail='greater', method='median') #.mean(1)
    
    #ax.text(0.4+nk,1.0, f'{f_plotting.sig_marker(p2,ns_note=False)}',horizontalalignment='center', verticalalignment='bottom', fontsize=10)
    ax.text(0.4+nk,1.05, f'{p2:.3f}',horizontalalignment='center', verticalalignment='bottom', fontsize=6)
    
    ax.boxplot([cosTheta12Ret_data['dlpfc'][:,d1x,d2x]], positions=[0.6+nk-0.035*showShuffBsl], patch_artist=True, widths = 0.05*len(showCheckpoints), boxprops=boxprops3, flierprops=flierprops3, 
                    meanprops=meanpointprops3, medianprops=medianprops, capprops = capprops3, whiskerprops = whiskerprops3, meanline=False, showmeans=True)
    
    p3 = f_stats.permutation_pCI(cosTheta12Ret_data['dlpfc'][:,d1x,d2x], cosTheta_bsl_data['dlpfc'], tail='smaller', alpha = 5) #.mean(1)
    #p3 = f_stats.p_1samp(np.abs(cosTheta12Ret_data['dlpfc'][:,d1x,d2x]), np.cos(parallelThreshold*(np.pi*0.5)), tail='greater') #.mean(1)
    #p3 = f_stats.p_1samp_bootstrap(np.abs(cosTheta12Ret_data['dlpfc'][:,d1x,d2x]), np.cos(parallelThreshold*(np.pi*0.5)), tail='greater', method='median') #.mean(1)
    
    #ax.text(0.6+nk,1.0, f'{f_plotting.sig_marker(p3,ns_note=False)}',horizontalalignment='center', verticalalignment='bottom', fontsize=10)
    ax.text(0.6+nk,1.05, f'{p3:.3f}',horizontalalignment='center', verticalalignment='bottom', fontsize=6)
    
    ax.boxplot([cosTheta12Ret_data['fef'][:,d1x,d2x]], positions=[0.8+nk-0.035*showShuffBsl], patch_artist=True, widths = 0.05*len(showCheckpoints), boxprops=boxprops4, flierprops=flierprops4, 
                    meanprops=meanpointprops4, medianprops=medianprops, capprops = capprops4, whiskerprops = whiskerprops4, meanline=False, showmeans=True)
    
    p4 = f_stats.permutation_pCI(cosTheta12Ret_data['fef'][:,d1x,d2x], cosTheta_bsl_data['fef'], tail='smaller', alpha = 5) #.mean(1)
    #p4 = f_stats.p_1samp(np.abs(cosTheta12Ret_data['fef'][:,d1x,d2x]), np.cos(parallelThreshold*(np.pi*0.5)), tail='greater') #.mean(1)
    #p4 = f_stats.p_1samp_bootstrap(np.abs(cosTheta12Ret_data['fef'][:,d1x,d2x]), np.cos(parallelThreshold*(np.pi*0.5)), tail='greater', method='median') #.mean(1)
    
    #ax.text(0.8+nk,1.0, f'{f_plotting.sig_marker(p4,ns_note=False)}',horizontalalignment='center', verticalalignment='bottom', fontsize=10)
    ax.text(0.8+nk,1.05, f'{p4:.3f}',horizontalalignment='center', verticalalignment='bottom', fontsize=6)
    
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

ax.set_xticks(xtks,xtklabs)#,rotation=20
ax.set_xlim(0,len(showCheckpoints))
ax.set_ylim(-1.05,1.2)
ax.set_yticks(np.flip(np.cos(angleCheckPoints).round(5)), labels = np.flip(np.degrees(angleCheckPoints).round(1).astype(int)))
ax.set_xlabel('Timebins', labelpad = 3, fontsize = 15)
ax.set_ylabel('cos(θ)', labelpad = 0, fontsize = 15)
ax.set_title(f'Coplanarity', fontsize = 17)
ax.tick_params(axis='both', labelsize=12)


ax.plot([], c=color1, label='R@R')
ax.plot([], c=color2, label='R&U')
ax.plot([], c=color3, label='LPFC')
ax.plot([], c=color4, label='FEF')
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize = 12)


plt.suptitle(f'I1D1 vs. I2D2, Retarget', fontsize = 15, y=1) # (Monkeys)
plt.tight_layout()
plt.show()



#%% fligner test of variance homogeneity
print(f'cos(θ)')
for nk, k in enumerate(showCheckpoints):
    
    dk1, dk2 = k+'D1', k+'D2'
    d1x, d2x = checkpointsLabels.index(dk1), checkpointsLabels.index(dk2)
    print(f'{dk1}-{dk2}')
    
    print(f"R@R: Mean(STDV) = {cosTheta12Ret_rnn['ed2'][:,d1x,d2x].mean():.3f}({cosTheta12Ret_rnn['ed2'][:,d1x,d2x].std():.3f})")
    print(f"R&U: Mean(STDV) = {cosTheta12Ret_rnn['ed12'][:,d1x,d2x].mean():.3f}({cosTheta12Ret_rnn['ed12'][:,d1x,d2x].std():.3f})")
    print(f"LPFC: Mean(STDV) = {cosTheta12Ret_data['dlpfc'][:,d1x,d2x].mean():.3f}({cosTheta12Ret_data['dlpfc'][:,d1x,d2x].std():.3f})")
    print(f"FEF: Mean(STDV) = {cosTheta12Ret_data['fef'][:,d1x,d2x].mean():.3f}({cosTheta12Ret_data['fef'][:,d1x,d2x].std():.3f})")
    
    print(f'Fligner-Killeen test for equality of variance')
    fk1, p1 = stats.fligner(cosTheta12Ret_rnn['ed2'][:,d1x,d2x],cosTheta12Ret_data['dlpfc'][:,d1x,d2x])
    print(f'R@R-LPFC: fk = {fk1:.3f}, p = {p1:.3f}')
    
    fk2, p2 = stats.fligner(cosTheta12Ret_rnn['ed12'][:,d1x,d2x],cosTheta12Ret_data['dlpfc'][:,d1x,d2x])
    print(f'R&U-LPFC: fk = {fk2:.3f}, p = {p2:.3f}')
    
    fk3, p3 = stats.fligner(cosTheta12Ret_rnn['ed2'][:,d1x,d2x],cosTheta12Ret_data['fef'][:,d1x,d2x])
    print(f'R@R-FEF: fk = {fk3:.3f}, p = {p3:.3f}')
    
    fk4, p4 = stats.fligner(cosTheta12Ret_rnn['ed12'][:,d1x,d2x],cosTheta12Ret_data['fef'][:,d1x,d2x])
    print(f'R&U-FEF: fk = {fk4:.3f}, p = {p4:.3f}')
    
    print('\n')

#%%



#######################
# I2D2Ret vs. I1D2Dis #
#######################




#%% I1D2-Distraction vs I2D2-Retarget, whisker plots

cosThetaChoice_rnn = {k:np.array([cosTheta_choice_rnn[k][n] for n in range(100)]).mean(1) for k in ('ed2','ed12')}
cosThetaChoice_data = {k:cosTheta_choice_data[k].mean(1) for k in ('dlpfc','fef')}
cosThetaChoice_shuff_rnn = {k:np.array([cosTheta_choice_shuff_rnn[k][n] for n in range(100)]).mean(1) for k in ('ed2','ed12')}
cosThetaChoice_shuff_data = {k:cosTheta_choice_shuff_data[k] for k in ('dlpfc','fef')}

cosPsiChoice_rnn = {k:np.array([cosPsi_choice_rnn[k][n] for n in range(100)]).mean(1) for k in ('ed2','ed12')}
cosPsiChoice_data = {k:cosPsi_choice_data[k].mean(1) for k in ('dlpfc','fef')}
cosPsiChoice_shuff_rnn = {k:np.array([cosPsi_choice_shuff_rnn[k][n] for n in range(100)]).mean(1) for k in ('ed2','ed12')}
cosPsiChoice_shuff_data = {k:cosPsi_choice_shuff_data[k] for k in ('dlpfc','fef')}

#%%
angleCheckPoints = np.linspace(0,np.pi,7).round(5)
cmap = plt.get_cmap('coolwarm')
norm = mcolors.BoundaryNorm(np.flip(np.cos(angleCheckPoints).round(5)),cmap.N, clip=True)

showCheckpoints = ('L',) #'E',
showShuffBsl = True

fig, axes = plt.subplots(1,2, sharey=True, figsize=(8,4), dpi=300)

# cosTheta
ax = axes.flatten()[0]
xtks, xtklabs = [], []
for nk, k in enumerate(showCheckpoints):
    dk2 = k+'D2'
    d2x = checkpointsLabels.index(dk2)
    #ld1x, ld2x = checkpointsLabels.index('LD1'), checkpointsLabels.index('LD2')

    print('############ cosTheta ############')
    cosTheta_bsl_rnn = {kk: np.mean((np.array([cosTheta11_bsl_rnn[kk][n][2] for n in range(100)]).squeeze()[:,:,d2x].mean(1),
                                   np.array([cosTheta22_bsl_rnn[kk][n][1] for n in range(100)]).squeeze()[:,:,d2x].mean(1)),axis=0) for kk in ('ed2','ed12')}
    cosTheta_bsl_data = {kk: np.mean((cosTheta11_bsl_data[kk][2][:,:,d2x].mean(1), 
                                     cosTheta22_bsl_data[kk][1][:,:,d2x].mean(1)),axis=0) for kk in ('dlpfc','fef')}
    
    
    ax.boxplot([cosThetaChoice_rnn['ed2'][:,d2x]], positions=[0.2+nk-0.035*showShuffBsl], patch_artist=True, widths = 0.05*len(showCheckpoints), boxprops=boxprops1, flierprops=flierprops1, 
                    meanprops=meanpointprops1, medianprops=medianprops, capprops = capprops1, whiskerprops = whiskerprops1, meanline=False, showmeans=True)
    
    #p1 = stats.ks_2samp(cosThetaChoice_rnn['ed2'][:,d2x], np.concatenate(cosThetaChoice_shuff_rnn['ed2'][:,:,d2x]))[1].round(3) #.mean(1)
    #p1 = f_stats.permutation_pCI(cosThetaChoice_rnn['ed2'][:,d2x], cosThetaChoice_shuff_rnn['ed2'][:,:,d2x].mean(1), tail='two')
    p1 = f_stats.permutation_pCI(cosThetaChoice_rnn['ed2'][:,d2x], cosTheta_bsl_rnn['ed2'], tail='smaller')
    ax.text(0.2+nk,1., f'{f_plotting.sig_marker(p1,ns_note=False)}',horizontalalignment='center', verticalalignment='bottom', fontsize=10)
    print(f"R@R: M(SD) = {cosThetaChoice_rnn['ed2'][:,d2x].mean():.3f} ({cosThetaChoice_rnn['ed2'][:,d2x].std():.3f}), 95CI = [{np.quantile(cosThetaChoice_rnn['ed2'][:,d2x],0.025):.3f}, {np.quantile(cosThetaChoice_rnn['ed2'][:,d2x],0.975):.3f}], p = {p1:.3f}")

    ax.boxplot([cosThetaChoice_rnn['ed12'][:,d2x]], positions=[0.4+nk-0.035*showShuffBsl], patch_artist=True, widths = 0.05*len(showCheckpoints), boxprops=boxprops2, flierprops=flierprops2, 
                    meanprops=meanpointprops2, medianprops=medianprops, capprops = capprops2, whiskerprops = whiskerprops2, meanline=False, showmeans=True)
    #p2 = stats.ks_2samp(cosThetaChoice_rnn['ed12'][:,d2x], np.concatenate(cosThetaChoice_shuff_rnn['ed12'][:,:,d2x]))[1].round(3) #.mean(1)
    #p2 = f_stats.permutation_pCI(cosThetaChoice_rnn['ed12'][:,d2x], cosThetaChoice_shuff_rnn['ed12'][:,:,d2x].mean(1), tail='two')
    p2 = f_stats.permutation_pCI(cosThetaChoice_rnn['ed12'][:,d2x], cosTheta_bsl_rnn['ed12'], tail='smaller')
    ax.text(0.4+nk,1., f'{f_plotting.sig_marker(p2,ns_note=False)}',horizontalalignment='center', verticalalignment='bottom', fontsize=10)
    print(f"R&U: M(SD) = {cosThetaChoice_rnn['ed12'][:,d2x].mean():.3f} ({cosThetaChoice_rnn['ed12'][:,d2x].std():.3f}), 95CI = [{np.quantile(cosThetaChoice_rnn['ed12'][:,d2x],0.025):.3f}, {np.quantile(cosThetaChoice_rnn['ed12'][:,d2x],0.975):.3f}], p = {p2:.3f}")

    ax.boxplot([cosThetaChoice_data['dlpfc'][:,d2x]], positions=[0.6+nk-0.035*showShuffBsl], patch_artist=True, widths = 0.05*len(showCheckpoints), boxprops=boxprops3, flierprops=flierprops3, 
                    meanprops=meanpointprops3, medianprops=medianprops, capprops = capprops3, whiskerprops = whiskerprops3, meanline=False, showmeans=True)
    #p3 = stats.ks_2samp(cosThetaChoice_data['dlpfc'][:,d2x], np.concatenate(cosThetaChoice_shuff_data['dlpfc'][:,:,d2x]))[1].round(3) #.mean(1)
    #p3 = f_stats.permutation_pCI(cosThetaChoice_data['dlpfc'][:,d2x], cosThetaChoice_shuff_data['dlpfc'][:,:,d2x].mean(1), tail='two')
    p3 = f_stats.permutation_pCI(cosThetaChoice_data['dlpfc'][:,d2x], cosTheta_bsl_data['dlpfc'], tail='smaller')
    ax.text(0.6+nk,1., f'{f_plotting.sig_marker(p3,ns_note=False)}',horizontalalignment='center', verticalalignment='bottom', fontsize=10)
    print(f"LPFC: M(SD) = {cosThetaChoice_data['dlpfc'][:,d2x].mean():.3f} ({cosThetaChoice_data['dlpfc'][:,d2x].std():.3f}), 95CI = [{np.quantile(cosThetaChoice_data['dlpfc'][:,d2x],0.025):.3f}, {np.quantile(cosThetaChoice_data['dlpfc'][:,d2x],0.975):.3f}], p = {p3:.3f}")

    ax.boxplot([cosThetaChoice_data['fef'][:,d2x]], positions=[0.8+nk-0.035*showShuffBsl], patch_artist=True, widths = 0.05*len(showCheckpoints), boxprops=boxprops4, flierprops=flierprops4, 
                    meanprops=meanpointprops4, medianprops=medianprops, capprops = capprops4, whiskerprops = whiskerprops4, meanline=False, showmeans=True)
    #p4 = stats.ks_2samp(cosThetaChoice_data['fef'][:,d2x], np.concatenate(cosThetaChoice_shuff_data['fef'][:,:,d2x]))[1].round(3) #.mean(1)
    #p4 = f_stats.permutation_pCI(cosThetaChoice_data['fef'][:,d2x], cosThetaChoice_shuff_data['fef'][:,:,d2x].mean(1), tail='two')
    p4 = f_stats.permutation_pCI(cosThetaChoice_data['fef'][:,d2x], cosTheta_bsl_data['fef'], tail='smaller')
    ax.text(0.8+nk,1., f'{f_plotting.sig_marker(p4,ns_note=False)}',horizontalalignment='center', verticalalignment='bottom', fontsize=10)
    print(f"FEF: M(SD) = {cosThetaChoice_data['fef'][:,d2x].mean():.3f} ({cosThetaChoice_data['fef'][:,d2x].std():.3f}), 95CI = [{np.quantile(cosThetaChoice_data['fef'][:,d2x],0.025):.3f}, {np.quantile(cosThetaChoice_data['fef'][:,d2x],0.975):.3f}], p = {p4:.3f}")

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
    
    # KS Test
    ks_results = {'R@R-LPFC':scipy.stats.ks_2samp(cosThetaChoice_rnn['ed2'][:,d2x], cosThetaChoice_data['dlpfc'][:,d2x]),
                  'R@R-FEF':scipy.stats.ks_2samp(cosThetaChoice_rnn['ed2'][:,d2x], cosThetaChoice_data['fef'][:,d2x]),
                  'R&U-LPFC':scipy.stats.ks_2samp(cosThetaChoice_rnn['ed12'][:,d2x], cosThetaChoice_data['dlpfc'][:,d2x]), 
                  'R&U-FEF':scipy.stats.ks_2samp(cosThetaChoice_rnn['ed12'][:,d2x], cosThetaChoice_data['fef'][:,d2x])}

    # FK Test
    fk_results = {'R@R-LPFC':scipy.stats.fligner(cosThetaChoice_rnn['ed2'][:,d2x],cosThetaChoice_data['dlpfc'][:,d2x]),
                  'R@R-FEF':scipy.stats.fligner(cosThetaChoice_rnn['ed2'][:,d2x],cosThetaChoice_data['fef'][:,d2x]),
                  'R&U-LPFC':scipy.stats.fligner(cosThetaChoice_rnn['ed12'][:,d2x],cosThetaChoice_data['dlpfc'][:,d2x]),
                  'R&U-FEF':scipy.stats.fligner(cosThetaChoice_rnn['ed12'][:,d2x],cosThetaChoice_data['fef'][:,d2x])}
    
    print('############### K-S Test ##############')
    for k in ks_results.keys():
        print(f'{k.upper()}: D = {ks_results[k][0]:.3f}, p = {ks_results[k][1]:.3f}')

    print('############### F-K Test ##############')
    for k in fk_results.keys():
        print(f'{k.upper()}: H = {fk_results[k][0]:.3f}, p = {fk_results[k][1]:.3f}')
    

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

# cosPsi
ax = axes.flatten()[1]
xtks, xtklabs = [], []
for nk, k in enumerate(showCheckpoints):
    dk2 = k+'D2'
    d2x = checkpointsLabels.index(dk2)
    #ld1x, ld2x = checkpointsLabels.index('LD1'), checkpointsLabels.index('LD2')

    print('############### cosPsi ##############')
    cosPsi_bsl_rnn = {kk: np.mean((np.array([cosPsi11_bsl_rnn[kk][n][2] for n in range(100)]).squeeze()[:,:,d2x].mean(1),
                                   np.array([cosPsi22_bsl_rnn[kk][n][1] for n in range(100)]).squeeze()[:,:,d2x].mean(1)),axis=0) for kk in ('ed2','ed12')}
    cosPsi_bsl_data = {kk: np.mean((cosPsi11_bsl_data[kk][2][:,:,d2x].mean(1), 
                                     cosPsi22_bsl_data[kk][1][:,:,d2x].mean(1)),axis=0) for kk in ('dlpfc','fef')}
    
    
    ax.boxplot([cosPsiChoice_rnn['ed2'][:,d2x]], positions=[0.2+nk-0.035*showShuffBsl], patch_artist=True, widths = 0.05*len(showCheckpoints), boxprops=boxprops1, flierprops=flierprops1, 
                    meanprops=meanpointprops1, medianprops=medianprops, capprops = capprops1, whiskerprops = whiskerprops1, meanline=False, showmeans=True)
    #p1 = stats.ks_2samp(cosPsiChoice_rnn['ed2'][:,d2x], np.concatenate(cosPsiChoice_shuff_rnn['ed2'][:,:,d2x]))[1].round(3) #.mean(1)
    #p1 = f_stats.permutation_pCI(cosPsiChoice_rnn['ed2'][:,d2x], cosPsiChoice_shuff_rnn['ed2'][:,:,d2x].mean(1), tail='two')
    p1 = f_stats.permutation_pCI(cosPsiChoice_rnn['ed2'][:,d2x], cosPsi_bsl_rnn['ed2'], tail='smaller', alpha = 5)
    ax.text(0.2+nk,1., f'{f_plotting.sig_marker(p1,ns_note=False)}',horizontalalignment='center', verticalalignment='bottom', fontsize=10)
    print(f"R@R: M(SD) = {cosPsiChoice_rnn['ed2'][:,d2x].mean():.3f} ({cosPsiChoice_rnn['ed2'][:,d2x].std():.3f}), 95CI = [{np.quantile(cosPsiChoice_rnn['ed2'][:,d2x], 0.025):.3f}, {np.quantile(cosPsiChoice_rnn['ed2'][:,d2x], 0.975):.3f}], p = {p1:.3f}")

    ax.boxplot([cosPsiChoice_rnn['ed12'][:,d2x]], positions=[0.4+nk-0.035*showShuffBsl], patch_artist=True, widths = 0.05*len(showCheckpoints), boxprops=boxprops2, flierprops=flierprops2, 
                    meanprops=meanpointprops2, medianprops=medianprops, capprops = capprops2, whiskerprops = whiskerprops2, meanline=False, showmeans=True)
    #p2 = stats.ks_2samp(cosPsiChoice_rnn['ed12'][:,d2x], np.concatenate(cosPsiChoice_shuff_rnn['ed12'][:,:,d2x]))[1].round(3) #.mean(1)
    #p2 = f_stats.permutation_pCI(cosPsiChoice_rnn['ed12'][:,d2x], cosPsiChoice_shuff_rnn['ed12'][:,:,d2x].mean(1), tail='two')
    p2 = f_stats.permutation_pCI(cosPsiChoice_rnn['ed12'][:,d2x], cosPsi_bsl_rnn['ed12'], tail='smaller', alpha = 5)
    ax.text(0.4+nk,1., f'{f_plotting.sig_marker(p2,ns_note=False)}',horizontalalignment='center', verticalalignment='bottom', fontsize=10)
    print(f"R&U: M(SD) = {cosPsiChoice_rnn['ed12'][:,d2x].mean():.3f} ({cosPsiChoice_rnn['ed12'][:,d2x].std():.3f}), 95CI = [{np.quantile(cosPsiChoice_rnn['ed12'][:,d2x], 0.025):.3f}, {np.quantile(cosPsiChoice_rnn['ed12'][:,d2x], 0.975):.3f}], p = {p2:.3f}")

    ax.boxplot([cosPsiChoice_data['dlpfc'][:,d2x]], positions=[0.6+nk-0.035*showShuffBsl], patch_artist=True, widths = 0.05*len(showCheckpoints), boxprops=boxprops3, flierprops=flierprops3, 
                    meanprops=meanpointprops3, medianprops=medianprops, capprops = capprops3, whiskerprops = whiskerprops3, meanline=False, showmeans=True)
    #p3 = stats.ks_2samp(cosPsiChoice_data['dlpfc'][:,d2x], np.concatenate(cosPsiChoice_shuff_data['dlpfc'][:,:,d2x]))[1].round(3) #.mean(1)
    #p3 = f_stats.permutation_pCI(cosPsiChoice_data['dlpfc'][:,d2x], cosPsiChoice_shuff_data['dlpfc'][:,:,d2x].mean(1), tail='two')
    p3 = f_stats.permutation_pCI(cosPsiChoice_data['dlpfc'][:,d2x], cosPsi_bsl_data['dlpfc'], tail='smaller', alpha = 5)
    ax.text(0.6+nk,1., f'{f_plotting.sig_marker(p3,ns_note=False)}',horizontalalignment='center', verticalalignment='bottom', fontsize=10)
    print(f"LPFC: M(SD) = {cosPsiChoice_data['dlpfc'][:,d2x].mean():.3f} ({cosPsiChoice_data['dlpfc'][:,d2x].std():.3f}), 95CI = [{np.quantile(cosPsiChoice_data['dlpfc'][:,d2x], 0.025):.3f}, {np.quantile(cosPsiChoice_data['dlpfc'][:,d2x], 0.975):.3f}], p = {p3:.3f}")

    ax.boxplot([cosPsiChoice_data['fef'][:,d2x]], positions=[0.8+nk-0.035*showShuffBsl], patch_artist=True, widths = 0.05*len(showCheckpoints), boxprops=boxprops4, flierprops=flierprops4, 
                    meanprops=meanpointprops4, medianprops=medianprops, capprops = capprops4, whiskerprops = whiskerprops4, meanline=False, showmeans=True)
    #p4 = stats.ks_2samp(cosPsiChoice_data['fef'][:,d2x], np.concatenate(cosPsiChoice_shuff_data['fef'][:,:,d2x]))[1].round(3) #.mean(1)
    #p4 = f_stats.permutation_pCI(cosPsiChoice_data['fef'][:,d2x], cosPsiChoice_shuff_data['fef'][:,:,d2x].mean(1), tail='two')
    p4 = f_stats.permutation_pCI(cosPsiChoice_data['fef'][:,d2x], cosPsi_bsl_data['fef'], tail='smaller', alpha = 5)
    ax.text(0.8+nk,1., f'{f_plotting.sig_marker(p4,ns_note=False)}',horizontalalignment='center', verticalalignment='bottom', fontsize=10)
    print(f"FEF: M(SD) = {cosPsiChoice_data['fef'][:,d2x].mean():.3f} ({cosPsiChoice_data['fef'][:,d2x].std():.3f}), 95CI = [{np.quantile(cosPsiChoice_data['fef'][:,d2x], 0.025):.3f}, {np.quantile(cosPsiChoice_data['fef'][:,d2x], 0.975):.3f}], p = {p4:.3f}")

    if showShuffBsl:
        #ax.boxplot([cosPsiChoice_shuff_rnn['ed2'][:,:,d2x].mean(1)], positions=[0.2+nk+0.035*showShuffBsl], patch_artist=True, widths = 0.05*len(showCheckpoints), boxprops=boxprops0, flierprops=flierprops0, 
        #            meanprops=meanpointprops0, medianprops=medianprops, capprops = capprops0, whiskerprops = whiskerprops0, meanline=False, showmeans=True)
        
        #ax.boxplot([cosPsiChoice_shuff_rnn['ed12'][:,:,d2x].mean(1)], positions=[0.4+nk+0.035*showShuffBsl], patch_artist=True, widths = 0.05*len(showCheckpoints), boxprops=boxprops0, flierprops=flierprops0, 
        #                meanprops=meanpointprops0, medianprops=medianprops, capprops = capprops0, whiskerprops = whiskerprops0, meanline=False, showmeans=True)
        
        #ax.boxplot([cosPsiChoice_shuff_data['dlpfc'][:,:,d2x].mean(1)], positions=[0.6+nk+0.035*showShuffBsl], patch_artist=True, widths = 0.05*len(showCheckpoints), boxprops=boxprops0, flierprops=flierprops0, 
        #                meanprops=meanpointprops0, medianprops=medianprops, capprops = capprops0, whiskerprops = whiskerprops0, meanline=False, showmeans=True)
        
        #ax.boxplot([cosPsiChoice_shuff_data['fef'][:,:,d2x].mean(1)], positions=[0.8+nk+0.035*showShuffBsl], patch_artist=True, widths = 0.05*len(showCheckpoints), boxprops=boxprops0, flierprops=flierprops0, 
        #                meanprops=meanpointprops0, medianprops=medianprops, capprops = capprops0, whiskerprops = whiskerprops0, meanline=False, showmeans=True)
        
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

    
    # KS Test
    ks_results = {'R@R-LPFC':scipy.stats.ks_2samp(cosPsiChoice_rnn['ed2'][:,d2x], cosPsiChoice_data['dlpfc'][:,d2x]),
                  'R@R-FEF':scipy.stats.ks_2samp(cosPsiChoice_rnn['ed2'][:,d2x], cosPsiChoice_data['fef'][:,d2x]),
                  'R&U-LPFC':scipy.stats.ks_2samp(cosPsiChoice_rnn['ed12'][:,d2x], cosPsiChoice_data['dlpfc'][:,d2x]), 
                  'R&U-FEF':scipy.stats.ks_2samp(cosPsiChoice_rnn['ed12'][:,d2x], cosPsiChoice_data['fef'][:,d2x])}

    # FK Test
    fk_results = {'R@R-LPFC':scipy.stats.fligner(cosPsiChoice_rnn['ed2'][:,d2x],cosPsiChoice_data['dlpfc'][:,d2x]),
                  'R@R-FEF':scipy.stats.fligner(cosPsiChoice_rnn['ed2'][:,d2x],cosPsiChoice_data['fef'][:,d2x]),
                  'R&U-LPFC':scipy.stats.fligner(cosPsiChoice_rnn['ed12'][:,d2x],cosPsiChoice_data['dlpfc'][:,d2x]),
                  'R&U-FEF':scipy.stats.fligner(cosPsiChoice_rnn['ed12'][:,d2x],cosPsiChoice_data['fef'][:,d2x])}
    
    print('############### K-S Test ##############')
    for k in ks_results.keys():
        print(f'{k.upper()}: D = {ks_results[k][0]:.3f}, p = {ks_results[k][1]:.3f}')

    print('############### F-K Test ##############')
    for k in fk_results.keys():
        print(f'{k.upper()}: H = {fk_results[k][0]:.3f}, p = {fk_results[k][1]:.3f}')
    
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
#%%
fig.savefig(f'{phd_path}/outputs/i1d2Ret-i2d2Dis_LD.tif', bbox_inches='tight')
#%%
































































#plt.figure(figsize=(10, 4), dpi=300)

# angle
ax = axes.flatten()[0]#plt.subplot(1,2,1)
bpl_L = [cosTheta_choice['dlpfc'][:,:,ed2x].mean(1), cosTheta_choice['dlpfc'][:,:,ld2x].mean(1),]
bpr_L = [cosTheta_choice['fef'][:,:,ed2x].mean(1), cosTheta_choice['fef'][:,:,ld2x].mean(1),]

bpl = ax.boxplot(bpl_L, positions=[0.3,1.3], patch_artist=True, widths = 0.2, boxprops=boxprops1, flierprops=flierprops1, meanprops=meanpointprops1, medianprops=medianprops, capprops = capprops1, whiskerprops = whiskerprops1, meanline=False, showmeans=True) #, boxprops=dict(linewidth=3)
bpr = ax.boxplot(bpr_L, positions=[0.7,1.7], patch_artist=True, widths = 0.2, boxprops=boxprops2, flierprops=flierprops2, meanprops=meanpointprops2, medianprops=medianprops, capprops = capprops2, whiskerprops = whiskerprops2, meanline=False, showmeans=True) #, boxprops=dict(linewidth=3)

ax.set_xticks([0.5,1.5],['ED2','LD2',])#,rotation=20
ax.set_xlim(0,1)
ax.set_yticks(np.flip(np.cos(angleCheckPoints).round(5)), labels = np.flip(np.degrees(angleCheckPoints).round(1).astype(int)))
ax.set_xlabel('Timebins', labelpad = 3, fontsize = 20)
ax.set_ylabel('cos(θ)', labelpad = 0, fontsize = 20)
ax.set_title(f'Principal Angle', fontsize = 17)
ax.tick_params(axis='both', labelsize=15)




# alignment
ax = axes.flatten()[1]#plt.subplot(1,2,2)
bpl_L = [cosPsi_choice['dlpfc'][:,:,ed2x].mean(1), cosPsi_choice['dlpfc'][:,:,ld2x].mean(1),]
bpr_L = [cosPsi_choice['fef'][:,:,ed2x].mean(1), cosPsi_choice['fef'][:,:,ld2x].mean(1),]

bpl = ax.boxplot(bpl_L, positions=[0.3,1.3], patch_artist=True, widths = 0.2, boxprops=boxprops1, flierprops=flierprops1, meanprops=meanpointprops1, medianprops=medianprops, capprops = capprops1, whiskerprops = whiskerprops1, meanline=False, showmeans=True) #, boxprops=dict(linewidth=3)
bpr = ax.boxplot(bpr_L, positions=[0.7,1.7], patch_artist=True, widths = 0.2, boxprops=boxprops2, flierprops=flierprops2, meanprops=meanpointprops2, medianprops=medianprops, capprops = capprops2, whiskerprops = whiskerprops2, meanline=False, showmeans=True) #, boxprops=dict(linewidth=3)

ax.plot([], c=color1, label='LPFC')
ax.plot([], c=color2, label='FEF')

ax.set_xticks([0.5,1.5],['ED2','LD2',]) #,rotation=20
ax.set_xlim(0,2)
ax.set_yticks(np.flip(np.cos(angleCheckPoints).round(5)), labels = np.flip(np.degrees(angleCheckPoints).round(1).astype(int)))
ax.set_xlabel('Timebins', labelpad = 3, fontsize = 20)
ax.set_ylabel('cos(ψ)', labelpad = 0, fontsize = 20)
ax.set_title(f'Representational Alignment', fontsize = 17)
ax.tick_params(axis='both', labelsize=15)

ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize = 15)


plt.suptitle(f'I2-Retarget vs. I1-Distraction', fontsize = 25, y=1) # (Monkeys)
plt.tight_layout()
plt.show()

fig.savefig(f'{phd_path}/data/pooled/i1d2Ret-i2d2Dis_data.tif', bbox_inches='tight')



#%%




#%%





##################
# code stability #
##################





#%%
ld1 = np.arange(800,1300+bins,bins)
ld1x = [tbins.tolist().index(t) for t in ld1]

ld2 = np.arange(2100,2600+bins,bins)
ld2x = [tbins.tolist().index(t) for t in ld2]

#%%

##############
# full space #
##############

#%%
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


#%%
lineh = np.arange(0.3,0.7,0.001)
linev = np.arange(1.16,1.165,0.0001)

#fig = plt.figure(figsize=(3, 3), dpi=300)
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



p1 = f_stats.permutation_pCI(stabRatioD1_full_rnn['ed2'].mean(), stabRatioD1_full_shuff_rnn['ed2'], tail = 'smaller',alpha=5)
ax.text(0.3,0.425, f'{f_plotting.sig_marker(p1,ns_note=False)}',horizontalalignment='center', verticalalignment='bottom', fontsize=10)
p2 = f_stats.permutation_pCI(stabRatioD2_full_rnn['ed2'].mean(), stabRatioD2_full_shuff_rnn['ed2'], tail = 'smaller',alpha=5)
ax.text(0.7,0.425, f'{f_plotting.sig_marker(p2,ns_note=False)}',horizontalalignment='center', verticalalignment='bottom', fontsize=10)

p3 = f_stats.permutation_pCI(stabRatioD1_full_rnn['ed12'].mean(), stabRatioD1_full_shuff_rnn['ed12'], tail = 'smaller',alpha=5)
ax.text(1.3,0.425, f'{f_plotting.sig_marker(p3,ns_note=False)}',horizontalalignment='center', verticalalignment='bottom', fontsize=10)
p4 = f_stats.permutation_pCI(stabRatioD2_full_rnn['ed12'].mean(), stabRatioD2_full_shuff_rnn['ed12'], tail = 'smaller',alpha=5)
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


p5 = f_stats.permutation_pCI(stabRatioD1_full_data['dlpfc'].mean(), stabRatioD1_full_shuff_data['dlpfc'], tail = 'smaller',alpha=5)
ax.text(2.3,0.425, f'{f_plotting.sig_marker(p5,ns_note=False)}',horizontalalignment='center', verticalalignment='bottom', fontsize=10)
p6 = f_stats.permutation_pCI(stabRatioD2_full_data['dlpfc'].mean(), stabRatioD2_full_shuff_data['dlpfc'], tail = 'smaller',alpha=5)
ax.text(2.7,0.425, f'{f_plotting.sig_marker(p6,ns_note=False)}',horizontalalignment='center', verticalalignment='bottom', fontsize=10)

p7 = f_stats.permutation_pCI(stabRatioD1_full_data['fef'].mean(), stabRatioD1_full_shuff_data['fef'], tail = 'smaller',alpha=5)
ax.text(3.3,0.425, f'{f_plotting.sig_marker(p7,ns_note=False)}',horizontalalignment='center', verticalalignment='bottom', fontsize=10)
p8 = f_stats.permutation_pCI(stabRatioD2_full_data['fef'].mean(), stabRatioD2_full_shuff_data['fef'], tail = 'smaller',alpha=5)
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

fig.savefig(f'{phd_path}/outputs/infoStabRatio_full.tif', bbox_inches='tight')

#%%

####################
# readout Subspace #
####################

#%%
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


#%%
lineh = np.arange(0.3,0.7,0.001)
linev = np.arange(1.16,1.165,0.0001)

#fig = plt.figure(figsize=(3, 3), dpi=300)
fig, axes = plt.subplots(1,1, sharey=True, figsize=(3,3), dpi=300)

# full space
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



p1 = f_stats.permutation_pCI(stabRatioD1_readout_rnn['ed2'].mean(), stabRatioD1_readout_shuff_rnn['ed2'], tail = 'smaller',alpha=5)
ax.text(0.3,0.425, f'{f_plotting.sig_marker(p1,ns_note=False)}',horizontalalignment='center', verticalalignment='bottom', fontsize=10)
p2 = f_stats.permutation_pCI(stabRatioD2_readout_rnn['ed2'].mean(), stabRatioD2_readout_shuff_rnn['ed2'], tail = 'smaller',alpha=5)
ax.text(0.7,0.425, f'{f_plotting.sig_marker(p2,ns_note=False)}',horizontalalignment='center', verticalalignment='bottom', fontsize=10)

p3 = f_stats.permutation_pCI(stabRatioD1_readout_rnn['ed12'].mean(), stabRatioD1_readout_shuff_rnn['ed12'], tail = 'smaller',alpha=5)
ax.text(1.3,0.425, f'{f_plotting.sig_marker(p3,ns_note=False)}',horizontalalignment='center', verticalalignment='bottom', fontsize=10)
p4 = f_stats.permutation_pCI(stabRatioD2_readout_rnn['ed12'].mean(), stabRatioD2_readout_shuff_rnn['ed12'], tail = 'smaller',alpha=5)
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


p5 = f_stats.permutation_pCI(stabRatioD1_readout_data['dlpfc'].mean(), stabRatioD1_readout_shuff_data['dlpfc'], tail = 'smaller',alpha=5)
ax.text(2.3,0.425, f'{f_plotting.sig_marker(p5,ns_note=False)}',horizontalalignment='center', verticalalignment='bottom', fontsize=10)
p6 = f_stats.permutation_pCI(stabRatioD2_readout_data['dlpfc'].mean(), stabRatioD2_readout_shuff_data['dlpfc'], tail = 'smaller',alpha=5)
ax.text(2.7,0.425, f'{f_plotting.sig_marker(p6,ns_note=False)}',horizontalalignment='center', verticalalignment='bottom', fontsize=10)

p7 = f_stats.permutation_pCI(stabRatioD1_readout_data['fef'].mean(), stabRatioD1_readout_shuff_data['fef'], tail = 'smaller',alpha=5)
ax.text(3.3,0.425, f'{f_plotting.sig_marker(p7,ns_note=False)}',horizontalalignment='center', verticalalignment='bottom', fontsize=10)
p8 = f_stats.permutation_pCI(stabRatioD2_readout_data['fef'].mean(), stabRatioD2_readout_shuff_data['fef'], tail = 'smaller',alpha=5)
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

fig.savefig(f'{phd_path}/outputs/infoStabRatio_readout.tif', bbox_inches='tight')
#%%




#%%





#################
# code morphing #
#################





#%%
ld1 = np.arange(800,1300+bins,bins)
ld1x = [tbins.tolist().index(t) for t in ld1]

ld2 = np.arange(2100,2600+bins,bins)
ld2x = [tbins.tolist().index(t) for t in ld2]

#%%

##############
# full space #
##############

#%%
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
 

#%%
lineh = np.arange(0.3,0.7,0.001)
linev = np.arange(1.16,1.165,0.0001)

#fig = plt.figure(figsize=(3, 3), dpi=300)
fig, axes = plt.subplots(1,1, sharey=True, figsize=(3,3), dpi=300)

# full space
ax = axes#.flatten()[0]

ax.boxplot([codeMorph_full_rnn['ed2']], positions=[0.5], patch_artist=True, widths = 0.2, boxprops=boxprops1, flierprops=flierprops1, 
                  meanprops=meanpointprops1, medianprops=medianprops, capprops = capprops1, whiskerprops = whiskerprops1, meanline=False, showmeans=True)

ax.boxplot([codeMorph_full_rnn['ed12']], positions=[1.5], patch_artist=True, widths = 0.2, boxprops=boxprops2, flierprops=flierprops2, 
                  meanprops=meanpointprops2, medianprops=medianprops, capprops = capprops2, whiskerprops = whiskerprops2, meanline=False, showmeans=True)


ax.boxplot([codeMorph_full_data['dlpfc']], positions=[2.5], patch_artist=True, widths = 0.2, boxprops=boxprops3, flierprops=flierprops3, 
                  meanprops=meanpointprops3, medianprops=medianprops, capprops = capprops3, whiskerprops = whiskerprops3, meanline=False, showmeans=True)

ax.boxplot([codeMorph_full_data['fef']], positions=[3.5], patch_artist=True, widths = 0.2, boxprops=boxprops4, flierprops=flierprops4, 
                  meanprops=meanpointprops4, medianprops=medianprops, capprops = capprops4, whiskerprops = whiskerprops4, meanline=False, showmeans=True)



p1 = f_stats.permutation_pCI(codeMorph_full_rnn['ed2'].mean(), codeMorph_full_shuff_rnn['ed2'], tail = 'greater', alpha=5)
ax.text(0.5,5.25, f'{f_plotting.sig_marker(p1,ns_note=True)}',horizontalalignment='center', verticalalignment='bottom', fontsize=10)

p2 = f_stats.permutation_pCI(codeMorph_full_rnn['ed12'].mean(), codeMorph_full_shuff_rnn['ed12'], tail = 'greater', alpha=5)
ax.text(1.5,5.25, f'{f_plotting.sig_marker(p2,ns_note=True)}',horizontalalignment='center', verticalalignment='bottom', fontsize=10)


p3 = f_stats.permutation_pCI(codeMorph_full_data['dlpfc'].mean(), codeMorph_full_shuff_data['dlpfc'], tail = 'greater', alpha=5)
ax.text(2.5,5.25, f'{f_plotting.sig_marker(p3,ns_note=True)}',horizontalalignment='center', verticalalignment='bottom', fontsize=10)

p4 = f_stats.permutation_pCI(codeMorph_full_data['fef'].mean(), codeMorph_full_shuff_data['fef'], tail = 'greater', alpha=5)
ax.text(3.5,5.25, f'{f_plotting.sig_marker(p4,ns_note=True)}',horizontalalignment='center', verticalalignment='bottom', fontsize=10)


ax.set_ylabel('D1-D1/D1-D2 (& vice versa)', labelpad = 3, fontsize = 10)


xlims = ax.get_xlim()
lineThresholds = np.arange(xlims[0],xlims[1]+0.01,0.01)
ax.plot(lineThresholds, np.full_like(lineThresholds,1), 'k--', alpha = 0.5)

# draw temporary red and blue lines and use them to create a legend
plt.plot([], c='dimgrey', label='Delay1')
plt.plot([], c='lightgrey', label='Delay2')
plt.legend(bbox_to_anchor=(1.5, 0.6))#loc = 'right',

ax.set_xticks([0.5,1.5,2.5,3.5],['R@R','R&U', 'LPFC','FEF'], rotation=0)
ax.set_xlabel('Strategy/Region', labelpad = 5, fontsize = 12)
ax.set_ylim(top=6, bottom=0)
ax.set_title('Code Morphing, Full Space', fontsize = 12, pad=10)
plt.show()

fig.savefig(f'{phd_path}/outputs/codeMorphing_full.tif', bbox_inches='tight')

#%%

####################
# readout Subspace #
####################

#%%

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
 

#%%
lineh = np.arange(0.3,0.7,0.001)
linev = np.arange(1.16,1.165,0.0001)

#fig = plt.figure(figsize=(3, 3), dpi=300)
fig, axes = plt.subplots(1,1, sharey=True, figsize=(3,3), dpi=300)

# readout subspace
ax = axes#.flatten()[0]

ax.boxplot([codeMorph_readout_rnn['ed2']], positions=[0.5], patch_artist=True, widths = 0.2, boxprops=boxprops1, flierprops=flierprops1, 
                  meanprops=meanpointprops1, medianprops=medianprops, capprops = capprops1, whiskerprops = whiskerprops1, meanline=False, showmeans=True)

ax.boxplot([codeMorph_readout_rnn['ed12']], positions=[1.5], patch_artist=True, widths = 0.2, boxprops=boxprops2, flierprops=flierprops2, 
                  meanprops=meanpointprops2, medianprops=medianprops, capprops = capprops2, whiskerprops = whiskerprops2, meanline=False, showmeans=True)


ax.boxplot([codeMorph_readout_data['dlpfc']], positions=[2.5], patch_artist=True, widths = 0.2, boxprops=boxprops3, flierprops=flierprops3, 
                  meanprops=meanpointprops3, medianprops=medianprops, capprops = capprops3, whiskerprops = whiskerprops3, meanline=False, showmeans=True)

ax.boxplot([codeMorph_readout_data['fef']], positions=[3.5], patch_artist=True, widths = 0.2, boxprops=boxprops4, flierprops=flierprops4, 
                  meanprops=meanpointprops4, medianprops=medianprops, capprops = capprops4, whiskerprops = whiskerprops4, meanline=False, showmeans=True)



p1 = f_stats.permutation_pCI(codeMorph_readout_rnn['ed2'].mean(), codeMorph_full_shuff_rnn['ed2'], tail = 'greater', alpha=5)
ax.text(0.5,5.25, f'{f_plotting.sig_marker(p1,ns_note=True)}',horizontalalignment='center', verticalalignment='bottom', fontsize=10)

p2 = f_stats.permutation_pCI(codeMorph_readout_rnn['ed12'].mean(), codeMorph_full_shuff_rnn['ed12'], tail = 'greater', alpha=5)
ax.text(1.5,5.25, f'{f_plotting.sig_marker(p2,ns_note=True)}',horizontalalignment='center', verticalalignment='bottom', fontsize=10)


p3 = f_stats.permutation_pCI(codeMorph_readout_data['dlpfc'].mean(), codeMorph_full_shuff_data['dlpfc'], tail = 'greater', alpha=5)
ax.text(2.5,5.25, f'{f_plotting.sig_marker(p3,ns_note=True)}',horizontalalignment='center', verticalalignment='bottom', fontsize=10)

p4 = f_stats.permutation_pCI(codeMorph_readout_data['fef'].mean(), codeMorph_full_shuff_data['fef'], tail = 'greater', alpha=5)
ax.text(3.5,5.25, f'{f_plotting.sig_marker(p4,ns_note=True)}',horizontalalignment='center', verticalalignment='bottom', fontsize=10)


ax.set_ylabel('D1-D1/D1-D2 (& vice versa)', labelpad = 3, fontsize = 10)


xlims = ax.get_xlim()
lineThresholds = np.arange(xlims[0],xlims[1]+0.01,0.01)
ax.plot(lineThresholds, np.full_like(lineThresholds,1), 'k--', alpha = 0.5)

# draw temporary red and blue lines and use them to create a legend
plt.plot([], c='dimgrey', label='Delay1')
plt.plot([], c='lightgrey', label='Delay2')
plt.legend(bbox_to_anchor=(1.5, 0.6))#loc = 'right',

ax.set_xticks([0.5,1.5,2.5,3.5],['R@R','R&U', 'LPFC','FEF'], rotation=0)
ax.set_xlabel('Strategy/Region', labelpad = 5, fontsize = 12)
ax.set_ylim(top=6, bottom=0)
ax.set_title('Code Morphing, Readout Space', fontsize = 12, pad=10)
plt.show()

fig.savefig(f'{phd_path}/outputs/codeMorphing_readout.tif', bbox_inches='tight')

#%%











































































