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

# In[]

##############
# Full Space #
##############

#%%
performanceX1_full_data = np.load(f'{phd_path}/outputs/monkeys/' + 'performanceX1_full_data.npy', allow_pickle=True).item()
performanceX2_full_data = np.load(f'{phd_path}/outputs/monkeys/' + 'performanceX2_full_data.npy', allow_pickle=True).item()
performanceX1_full_shuff_data = np.load(f'{phd_path}/outputs/monkeys/' + 'performanceX1_full_shuff_data.npy', allow_pickle=True).item()
performanceX2_full_shuff_data = np.load(f'{phd_path}/outputs/monkeys/' + 'performanceX2_full_shuff_data.npy', allow_pickle=True).item()

performanceX1_full_rnn = np.load(f'{phd_path}/outputs/rnns/' + 'performanceX1_full_rnn.npy', allow_pickle=True).item()
performanceX2_full_rnn = np.load(f'{phd_path}/outputs/rnns/' + 'performanceX2_full_rnn.npy', allow_pickle=True).item()
performanceX1_full_shuff_rnn = np.load(f'{phd_path}/outputs/rnns/' + 'performanceX1_full_shuff_rnn.npy', allow_pickle=True).item()
performanceX2_full_shuff_rnn = np.load(f'{phd_path}/outputs/rnns/' + 'performanceX2_full_shuff_rnn.npy', allow_pickle=True).item()

# In[]

#################
# Readout Space #
#################

#%%
performanceX1_readout_data = np.load(f'{phd_path}/outputs/monkeys/' + 'performanceX1_readout_data.npy', allow_pickle=True).item()
performanceX2_readout_data = np.load(f'{phd_path}/outputs/monkeys/' + 'performanceX2_readout_data.npy', allow_pickle=True).item()
performanceX1_readout_shuff_data = np.load(f'{phd_path}/outputs/monkeys/' + 'performanceX1_readout_shuff_data.npy', allow_pickle=True).item()
performanceX2_readout_shuff_data = np.load(f'{phd_path}/outputs/monkeys/' + 'performanceX2_readout_shuff_data.npy', allow_pickle=True).item()

performanceX1_readout_rnn = np.load(f'{phd_path}/outputs/rnns/' + 'performanceX1_readout_rnn.npy', allow_pickle=True).item()
performanceX2_readout_rnn = np.load(f'{phd_path}/outputs/rnns/' + 'performanceX2_readout_rnn.npy', allow_pickle=True).item()
performanceX1_readout_shuff_rnn = np.load(f'{phd_path}/outputs/rnns/' + 'performanceX1_readout_shuff_rnn.npy', allow_pickle=True).item()
performanceX2_readout_shuff_rnn = np.load(f'{phd_path}/outputs/rnns/' + 'performanceX2_readout_shuff_rnn.npy', allow_pickle=True).item()
#%%




###################
# distractor info #
###################




#%%

##############
# Full Space #
##############

# In[] distractor information quantification

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
#ax.plot(lineThresholds, np.full_like(lineThresholds,0.33), 'k--', alpha = 0.5, linewidth=1)

ax.set_xticks([0.5,1.5,2.5,3.5],['R@R','R&U','LPFC','FEF'], rotation=0)
ax.set_xlabel('Strategy/Region', labelpad = 5, fontsize = 12)
ax.set_ylabel('Decodability', labelpad = 3, fontsize = 12)

ax.set_ylim(top=0.9, bottom=0.1)

plt.suptitle('Distractor Information, Full Space, ED2', fontsize = 12, y=1.0)
plt.show()

# save figure
#fig.savefig(f'{phd_path}/outputs/distractorInfo_full.tif', bbox_inches='tight')


# print stats

#%%

#################
# Readout Space #
#################

# In[] distractor information quantification

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
#ax.plot(lineThresholds, np.full_like(lineThresholds,0.33), 'k--', alpha = 0.5, linewidth=1)

ax.set_xticks([0.5,1.5,2.5,3.5],['R@R','R&U','LPFC','FEF'], rotation=0)
ax.set_xlabel('Strategy/Region', labelpad = 5, fontsize = 12)
ax.set_ylabel('Decodability', labelpad = 3, fontsize = 12)

ax.set_ylim(top=0.9, bottom=0.1)

plt.suptitle('Distractor Information, Readout Subspace, ED2', fontsize = 12, y=1.0)
plt.show()

# save figure
#fig.savefig(f'{phd_path}/outputs/distractorInfo_readout.tif', bbox_inches='tight')

# print stats

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

# save figure
#fig.savefig(f'{phd_path}/outputs/infoStabRatio_full.tif', bbox_inches='tight')

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

# save fig
fig.savefig(f'{phd_path}/outputs/infoStabRatio_readout.tif', bbox_inches='tight')

# print stats
# print stats
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

#%%
#%%





##########################
# code stability D1 Only #
##########################





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
showbsl = True
lineh = np.arange(0.3,0.7,0.001)
linev = np.arange(1.16,1.165,0.0001)

#fig = plt.figure(figsize=(3, 3), dpi=300)
fig, axes = plt.subplots(1,1, sharey=True, figsize=(3,3), dpi=300)

# full space
ax = axes#.flatten()[0]

ax.boxplot([stabRatioD1_full_rnn['ed2']], positions=[0.5-showbsl*0.2], patch_artist=True, widths = 0.2, boxprops=boxprops1, flierprops=flierprops1, 
                  meanprops=meanpointprops1, medianprops=medianprops, capprops = capprops1, whiskerprops = whiskerprops1, meanline=False, showmeans=True)

ax.boxplot([stabRatioD1_full_rnn['ed12']], positions=[1.5-showbsl*0.2], patch_artist=True, widths = 0.2, boxprops=boxprops2, flierprops=flierprops2, 
                  meanprops=meanpointprops2, medianprops=medianprops, capprops = capprops2, whiskerprops = whiskerprops2, meanline=False, showmeans=True)


ax.boxplot([stabRatioD1_full_data['dlpfc']], positions=[2.5-showbsl*0.2], patch_artist=True, widths = 0.2, boxprops=boxprops3, flierprops=flierprops3, 
                  meanprops=meanpointprops3, medianprops=medianprops, capprops = capprops3, whiskerprops = whiskerprops3, meanline=False, showmeans=True)

ax.boxplot([stabRatioD1_full_data['fef']], positions=[3.5-showbsl*0.2], patch_artist=True, widths = 0.2, boxprops=boxprops4, flierprops=flierprops4, 
                  meanprops=meanpointprops4, medianprops=medianprops, capprops = capprops4, whiskerprops = whiskerprops4, meanline=False, showmeans=True)

if showbsl:
    ax.boxplot([stabRatioD1_full_shuff_rnn['ed2']], positions=[0.5+showbsl*0.2], patch_artist=True, widths = 0.2, boxprops=boxprops0, flierprops=flierprops0, 
                  meanprops=meanpointprops0, medianprops=medianprops, capprops = capprops0, whiskerprops = whiskerprops0, meanline=False, showmeans=True)

    ax.boxplot([stabRatioD1_full_shuff_rnn['ed12']], positions=[1.5+showbsl*0.2], patch_artist=True, widths = 0.2, boxprops=boxprops0, flierprops=flierprops0, 
                    meanprops=meanpointprops0, medianprops=medianprops, capprops = capprops0, whiskerprops = whiskerprops0, meanline=False, showmeans=True)


    ax.boxplot([stabRatioD1_full_shuff_data['dlpfc']], positions=[2.5+showbsl*0.2], patch_artist=True, widths = 0.2, boxprops=boxprops0, flierprops=flierprops0, 
                    meanprops=meanpointprops0, medianprops=medianprops, capprops = capprops0, whiskerprops = whiskerprops0, meanline=False, showmeans=True)

    ax.boxplot([stabRatioD1_full_shuff_data['fef']], positions=[3.5+showbsl*0.2], patch_artist=True, widths = 0.2, boxprops=boxprops0, flierprops=flierprops0, 
                    meanprops=meanpointprops0, medianprops=medianprops, capprops = capprops0, whiskerprops = whiskerprops0, meanline=False, showmeans=True)
    

p1 = f_stats.permutation_pCI(stabRatioD1_full_rnn['ed2'], stabRatioD1_full_shuff_rnn['ed2'], tail = 'smaller',alpha=5)
ax.text(0.3,0.425, f'{f_plotting.sig_marker(p1,ns_note=False)}',horizontalalignment='center', verticalalignment='bottom', fontsize=10)

p3 = f_stats.permutation_pCI(stabRatioD1_full_rnn['ed12'], stabRatioD1_full_shuff_rnn['ed12'], tail = 'smaller',alpha=5)
ax.text(1.3,0.425, f'{f_plotting.sig_marker(p3,ns_note=False)}',horizontalalignment='center', verticalalignment='bottom', fontsize=10)


p5 = f_stats.permutation_pCI(stabRatioD1_full_data['dlpfc'], stabRatioD1_full_shuff_data['dlpfc'], tail = 'smaller',alpha=5)
ax.text(2.3,0.425, f'{f_plotting.sig_marker(p5,ns_note=False)}',horizontalalignment='center', verticalalignment='bottom', fontsize=10)

p7 = f_stats.permutation_pCI(stabRatioD1_full_data['fef'], stabRatioD1_full_shuff_data['fef'], tail = 'smaller',alpha=5)
ax.text(3.3,0.425, f'{f_plotting.sig_marker(p7,ns_note=False)}',horizontalalignment='center', verticalalignment='bottom', fontsize=10)


ax.set_ylabel('Off-/On-Diagonal', labelpad = 3, fontsize = 10)


xlims = ax.get_xlim()
lineThresholds = np.arange(xlims[0],xlims[1]+0.01,0.01)
ax.plot(lineThresholds, np.full_like(lineThresholds,1), 'k--', alpha = 0.5)


ax.set_xticks([0.5,1.5,2.5,3.5],['R@R','R&U', 'LPFC','FEF'], rotation=0)
ax.set_xlabel('Strategy/Region', labelpad = 5, fontsize = 12)
ax.set_ylim(top=1.3, bottom=0.4)
ax.set_title('Code Stability (D1), Full Space', fontsize = 12, pad=10)
plt.show()

# save figure
#fig.savefig(f'{phd_path}/outputs/infoStabRatio_d1_full.tif', bbox_inches='tight')

# print stats
print(f"R@R, D1: M(SD) = {stabRatioD1_full_rnn['ed2'].mean():.3f}({stabRatioD1_full_rnn['ed2'].std():.3f}), p = {p1:.3f};")
print(f"R&U, D1: M(SD) = {stabRatioD1_full_rnn['ed12'].mean():.3f}({stabRatioD1_full_rnn['ed12'].std():.3f}), p = {p3:.3f};")
print(f"LPFC, D1: M(SD) = {stabRatioD1_full_data['dlpfc'].mean():.3f}({stabRatioD1_full_data['dlpfc'].std():.3f}), p = {p5:.3f};")
print(f"FEF, D1: M(SD) = {stabRatioD1_full_data['fef'].mean():.3f}({stabRatioD1_full_data['fef'].std():.3f}), p = {p7:.3f};")
print('\n')
# KS Test
ks_results = {'R@R-LPFC':scipy.stats.ks_2samp(stabRatioD1_full_rnn['ed2'], stabRatioD1_full_data['dlpfc']),
                'R@R-FEF':scipy.stats.ks_2samp(stabRatioD1_full_rnn['ed2'], stabRatioD1_full_data['fef']),
                'R&U-LPFC':scipy.stats.ks_2samp(stabRatioD1_full_rnn['ed12'], stabRatioD1_full_data['dlpfc']), 
                'R&U-FEF':scipy.stats.ks_2samp(stabRatioD1_full_rnn['ed12'], stabRatioD1_full_data['fef'])}

# FK Test
fk_results = {'R@R-LPFC':scipy.stats.fligner(stabRatioD1_full_rnn['ed2'], stabRatioD1_full_data['dlpfc']),
                'R@R-FEF':scipy.stats.fligner(stabRatioD1_full_rnn['ed2'], stabRatioD1_full_data['fef']),
                'R&U-LPFC':scipy.stats.fligner(stabRatioD1_full_rnn['ed12'], stabRatioD1_full_data['dlpfc']),
                'R&U-FEF':scipy.stats.fligner(stabRatioD1_full_rnn['ed12'], stabRatioD1_full_data['fef'])}

print('############### K-S Test ##############')
for k in ks_results.keys():
    print(f'{k.upper()}: D = {ks_results[k][0]:.3f}, p = {ks_results[k][1]:.3f}')

print('############### F-K Test ##############')
for k in fk_results.keys():
    print(f'{k.upper()}: H = {fk_results[k][0]:.3f}, p = {fk_results[k][1]:.3f}')
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

ax.boxplot([stabRatioD1_readout_rnn['ed12']], positions=[1.3], patch_artist=True, widths = 0.2, boxprops=boxprops2, flierprops=flierprops2, 
                  meanprops=meanpointprops2, medianprops=medianprops, capprops = capprops2, whiskerprops = whiskerprops2, meanline=False, showmeans=True)


ax.boxplot([stabRatioD1_readout_data['dlpfc']], positions=[2.3], patch_artist=True, widths = 0.2, boxprops=boxprops3, flierprops=flierprops3, 
                  meanprops=meanpointprops3, medianprops=medianprops, capprops = capprops3, whiskerprops = whiskerprops3, meanline=False, showmeans=True)

ax.boxplot([stabRatioD1_readout_data['fef']], positions=[3.3], patch_artist=True, widths = 0.2, boxprops=boxprops4, flierprops=flierprops4, 
                  meanprops=meanpointprops4, medianprops=medianprops, capprops = capprops4, whiskerprops = whiskerprops4, meanline=False, showmeans=True)



p1 = f_stats.permutation_pCI(stabRatioD1_readout_rnn['ed2'], stabRatioD1_readout_shuff_rnn['ed2'], tail = 'smaller',alpha=5)
ax.text(0.3,0.425, f'{f_plotting.sig_marker(p1,ns_note=False)}',horizontalalignment='center', verticalalignment='bottom', fontsize=10)
p2 = f_stats.permutation_pCI(stabRatioD2_readout_rnn['ed2'], stabRatioD2_readout_shuff_rnn['ed2'], tail = 'smaller',alpha=5)
ax.text(0.7,0.425, f'{f_plotting.sig_marker(p2,ns_note=False)}',horizontalalignment='center', verticalalignment='bottom', fontsize=10)

p3 = f_stats.permutation_pCI(stabRatioD1_readout_rnn['ed12'], stabRatioD1_readout_shuff_rnn['ed12'], tail = 'smaller',alpha=5)
ax.text(1.3,0.425, f'{f_plotting.sig_marker(p3,ns_note=False)}',horizontalalignment='center', verticalalignment='bottom', fontsize=10)
p4 = f_stats.permutation_pCI(stabRatioD2_readout_rnn['ed12'], stabRatioD2_readout_shuff_rnn['ed12'], tail = 'smaller',alpha=5)
ax.text(1.7,0.425, f'{f_plotting.sig_marker(p4,ns_note=False)}',horizontalalignment='center', verticalalignment='bottom', fontsize=10)


p5 = f_stats.permutation_pCI(stabRatioD1_readout_data['dlpfc'], stabRatioD1_readout_shuff_data['dlpfc'], tail = 'smaller',alpha=5)
ax.text(2.3,0.425, f'{f_plotting.sig_marker(p5,ns_note=False)}',horizontalalignment='center', verticalalignment='bottom', fontsize=10)
p6 = f_stats.permutation_pCI(stabRatioD2_readout_data['dlpfc'], stabRatioD2_readout_shuff_data['dlpfc'], tail = 'smaller',alpha=5)
ax.text(2.7,0.425, f'{f_plotting.sig_marker(p6,ns_note=False)}',horizontalalignment='center', verticalalignment='bottom', fontsize=10)

p7 = f_stats.permutation_pCI(stabRatioD1_readout_data['fef'], stabRatioD1_readout_shuff_data['fef'], tail = 'smaller',alpha=5)
ax.text(3.3,0.425, f'{f_plotting.sig_marker(p7,ns_note=False)}',horizontalalignment='center', verticalalignment='bottom', fontsize=10)
p8 = f_stats.permutation_pCI(stabRatioD2_readout_data['fef'], stabRatioD2_readout_shuff_data['fef'], tail = 'smaller',alpha=5)
ax.text(3.7,0.425, f'{f_plotting.sig_marker(p8,ns_note=False)}',horizontalalignment='center', verticalalignment='bottom', fontsize=10)


ax.set_ylabel('Off-/On-Diagonal', labelpad = 3, fontsize = 10)


xlims = ax.get_xlim()
lineThresholds = np.arange(xlims[0],xlims[1]+0.01,0.01)
ax.plot(lineThresholds, np.full_like(lineThresholds,1), 'k--', alpha = 0.5)

ax.set_xticks([0.5,1.5,2.5,3.5],['R@R','R&U', 'LPFC','FEF'], rotation=0)
ax.set_xlabel('Strategy/Region', labelpad = 5, fontsize = 12)
ax.set_ylim(top=1.3, bottom=0.4)
ax.set_title('Code Stability (D1), Readout Subspace', fontsize = 12, pad=10)
plt.show()

# save fig
fig.savefig(f'{phd_path}/outputs/infoStabRatio_d1_readout.tif', bbox_inches='tight')

# print stats
# print stats
print(f"R@R, D1: M(SD) = {stabRatioD1_readout_rnn['ed2'].mean():.3f}({stabRatioD1_readout_rnn['ed2'].std():.3f}), p = {p1:.3f};")
print(f"R&U, D1: M(SD) = {stabRatioD1_readout_rnn['ed12'].mean():.3f}({stabRatioD1_readout_rnn['ed12'].std():.3f}), p = {p3:.3f};")
print(f"LPFC, D1: M(SD) = {stabRatioD1_readout_data['dlpfc'].mean():.3f}({stabRatioD1_readout_data['dlpfc'].std():.3f}), p = {p5:.3f};")
print(f"FEF, D1: M(SD) = {stabRatioD1_readout_data['fef'].mean():.3f}({stabRatioD1_readout_data['fef'].std():.3f}), p = {p7:.3f};")
print('\n')
# KS Test
ks_results = {'R@R-LPFC':scipy.stats.ks_2samp(stabRatioD1_readout_rnn['ed2'], stabRatioD1_readout_data['dlpfc']),
                'R@R-FEF':scipy.stats.ks_2samp(stabRatioD1_readout_rnn['ed2'], stabRatioD1_readout_data['fef']),
                'R&U-LPFC':scipy.stats.ks_2samp(stabRatioD1_readout_rnn['ed12'], stabRatioD1_readout_data['dlpfc']), 
                'R&U-FEF':scipy.stats.ks_2samp(stabRatioD1_readout_rnn['ed12'], stabRatioD1_readout_data['fef'])}

# FK Test
fk_results = {'R@R-LPFC':scipy.stats.fligner(stabRatioD1_readout_rnn['ed2'], stabRatioD1_readout_data['dlpfc']),
                'R@R-FEF':scipy.stats.fligner(stabRatioD1_readout_rnn['ed2'], stabRatioD1_readout_data['fef']),
                'R&U-LPFC':scipy.stats.fligner(stabRatioD1_readout_rnn['ed12'], stabRatioD1_readout_data['dlpfc']),
                'R&U-FEF':scipy.stats.fligner(stabRatioD1_readout_rnn['ed12'], stabRatioD1_readout_data['fef'])}

print('############### K-S Test ##############')
for k in ks_results.keys():
    print(f'{k.upper()}: D = {ks_results[k][0]:.3f}, p = {ks_results[k][1]:.3f}')

print('############### F-K Test ##############')
for k in fk_results.keys():
    print(f'{k.upper()}: H = {fk_results[k][0]:.3f}, p = {fk_results[k][1]:.3f}')
#%%
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

#ax.boxplot([codeMorph_full_rnn['ed2']], positions=[0.5], patch_artist=True, widths = 0.2, boxprops=boxprops1, flierprops=flierprops1, 
#                  meanprops=meanpointprops1, medianprops=medianprops, capprops = capprops1, whiskerprops = whiskerprops1, meanline=False, showmeans=True)

#ax.boxplot([codeMorph_full_rnn['ed12']], positions=[1.5], patch_artist=True, widths = 0.2, boxprops=boxprops2, flierprops=flierprops2, 
#                  meanprops=meanpointprops2, medianprops=medianprops, capprops = capprops2, whiskerprops = whiskerprops2, meanline=False, showmeans=True)

#ax.boxplot([codeMorph_full_data['dlpfc']], positions=[2.5], patch_artist=True, widths = 0.2, boxprops=boxprops3, flierprops=flierprops3, 
#                  meanprops=meanpointprops3, medianprops=medianprops, capprops = capprops3, whiskerprops = whiskerprops3, meanline=False, showmeans=True)

#ax.boxplot([codeMorph_full_data['fef']], positions=[3.5], patch_artist=True, widths = 0.2, boxprops=boxprops4, flierprops=flierprops4, 
#                  meanprops=meanpointprops4, medianprops=medianprops, capprops = capprops4, whiskerprops = whiskerprops4, meanline=False, showmeans=True)

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

# draw temporary red and blue lines and use them to create a legend
#plt.plot([], c='dimgrey', label='Delay1')
#plt.plot([], c='lightgrey', label='Delay2')
#plt.legend(bbox_to_anchor=(1.5, 0.6))#loc = 'right',

ax.set_xticks([0.5,1.5,2.5,3.5],['R@R','R&U', 'LPFC','FEF'], rotation=0)
ax.set_xlabel('Strategy/Region', labelpad = 5, fontsize = 12)
ax.set_ylim(top=6, bottom=0)
ax.set_title('Code Morphing, Full Space', fontsize = 12, pad=10)
plt.show()

# save fig
#fig.savefig(f'{phd_path}/outputs/codeMorphing_full.tif', bbox_inches='tight')

# print stats
print(f"R@R: M(SD) = {codeMorph_full_rnn['ed2'].mean():.3f}({codeMorph_full_rnn['ed2'].std():.3f}), p = {p1:.3f};")
print(f"R&U: M(SD) = {codeMorph_full_rnn['ed12'].mean():.3f}({codeMorph_full_rnn['ed12'].std():.3f}), p = {p2:.3f};")
print(f"LPFC: M(SD) = {codeMorph_full_data['dlpfc'].mean():.3f}({codeMorph_full_data['dlpfc'].std():.3f}), p = {p3:.3f};")
print(f"FEF: M(SD) = {codeMorph_full_data['fef'].mean():.3f}({codeMorph_full_data['fef'].std():.3f}), p = {p4:.3f};")
print('\n')
############
# k-s test #
############
codeMorph_full_ks = {'R@R-LPFC':scipy.stats.ks_2samp(codeMorph_full_rnn['ed2'], codeMorph_full_data['dlpfc']), 
                        'R@R-FEF':scipy.stats.ks_2samp(codeMorph_full_rnn['ed2'], codeMorph_full_data['fef']), 
                        'R&U-LPFC':scipy.stats.ks_2samp(codeMorph_full_rnn['ed12'], codeMorph_full_data['dlpfc']), 
                        'R&U-FEF':scipy.stats.ks_2samp(codeMorph_full_rnn['ed12'], codeMorph_full_data['fef'])}

print('############### K-S Test ##############')
for k in codeMorph_full_ks.keys():
    print(f"{k}: D = {codeMorph_full_ks[k].statistic:.3f}, p = {codeMorph_full_ks[k].pvalue:.3f};")


# FK Test
fk_results = {'R@R-LPFC':scipy.stats.fligner(codeMorph_full_rnn['ed2'], codeMorph_full_data['dlpfc']),
                'R@R-FEF':scipy.stats.fligner(codeMorph_full_rnn['ed2'], codeMorph_full_data['fef']),
                'R&U-LPFC':scipy.stats.fligner(codeMorph_full_rnn['ed12'], codeMorph_full_data['dlpfc']),
                'R&U-FEF':scipy.stats.fligner(codeMorph_full_rnn['ed12'], codeMorph_full_data['fef'])}

print('############### F-K Test ##############')
for k in fk_results.keys():
    print(f'{k.upper()}: H = {fk_results[k][0]:.3f}, p = {fk_results[k][1]:.3f}')
#%%
############
# CvM test #
############
#codeMorph_full_cvm = {'R@R-LPFC':scipy.stats.cramervonmises_2samp(codeMorph_full_rnn['ed2'], codeMorph_full_data['dlpfc']), 
#                         'R@R-FEF':scipy.stats.cramervonmises_2samp(codeMorph_full_rnn['ed2'], codeMorph_full_data['fef']), 
#                         'R&U-LPFC':scipy.stats.cramervonmises_2samp(codeMorph_full_rnn['ed12'], codeMorph_full_data['dlpfc']), 
#                         'R&U-FEF':scipy.stats.cramervonmises_2samp(codeMorph_full_rnn['ed12'], codeMorph_full_data['fef'])}

#############################
# Wasserstein Distance test #
#############################
#codeMorph_full_wd = {'R@R-LPFC':scipy.stats.wasserstein_distance(codeMorph_full_rnn['ed2'], codeMorph_full_data['dlpfc']), 
#                        'R@R-FEF':scipy.stats.wasserstein_distance(codeMorph_full_rnn['ed2'], codeMorph_full_data['fef']), 
#                        'R&U-LPFC':scipy.stats.wasserstein_distance(codeMorph_full_rnn['ed12'], codeMorph_full_data['dlpfc']), 
#                        'R&U-FEF':scipy.stats.wasserstein_distance(codeMorph_full_rnn['ed12'], codeMorph_full_data['fef'])}

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

#ax.boxplot([codeMorph_readout_rnn['ed2']], positions=[0.5], patch_artist=True, widths = 0.2, boxprops=boxprops1, flierprops=flierprops1, 
#                  meanprops=meanpointprops1, medianprops=medianprops, capprops = capprops1, whiskerprops = whiskerprops1, meanline=False, showmeans=True)

#ax.boxplot([codeMorph_readout_rnn['ed12']], positions=[1.5], patch_artist=True, widths = 0.2, boxprops=boxprops2, flierprops=flierprops2, 
#                  meanprops=meanpointprops2, medianprops=medianprops, capprops = capprops2, whiskerprops = whiskerprops2, meanline=False, showmeans=True)


#ax.boxplot([codeMorph_readout_data['dlpfc']], positions=[2.5], patch_artist=True, widths = 0.2, boxprops=boxprops3, flierprops=flierprops3, 
#                  meanprops=meanpointprops3, medianprops=medianprops, capprops = capprops3, whiskerprops = whiskerprops3, meanline=False, showmeans=True)

#ax.boxplot([codeMorph_readout_data['fef']], positions=[3.5], patch_artist=True, widths = 0.2, boxprops=boxprops4, flierprops=flierprops4, 
#                  meanprops=meanpointprops4, medianprops=medianprops, capprops = capprops4, whiskerprops = whiskerprops4, meanline=False, showmeans=True)


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

# draw temporary red and blue lines and use them to create a legend
#plt.plot([], c='dimgrey', label='Delay1')
#plt.plot([], c='lightgrey', label='Delay2')
#plt.legend(bbox_to_anchor=(1.5, 0.6))#loc = 'right',

ax.set_xticks([0.5,1.5,2.5,3.5],['R@R','R&U', 'LPFC','FEF'], rotation=0)
ax.set_xlabel('Strategy/Region', labelpad = 5, fontsize = 12)
ax.set_ylim(top=6, bottom=0)
ax.set_title('Code Morphing, Readout Space', fontsize = 12, pad=10)
plt.show()

# save fig
#fig.savefig(f'{phd_path}/outputs/codeMorphing_readout.tif', bbox_inches='tight')

# print stats
print(f"R@R: M(SD) = {codeMorph_readout_rnn['ed2'].mean():.3f}({codeMorph_readout_rnn['ed2'].std():.3f}), p = {p1:.3f};")
print(f"R&U: M(SD) = {codeMorph_readout_rnn['ed12'].mean():.3f}({codeMorph_readout_rnn['ed12'].std():.3f}), p = {p2:.3f};")
print(f"LPFC: M(SD) = {codeMorph_readout_data['dlpfc'].mean():.3f}({codeMorph_readout_data['dlpfc'].std():.3f}), p = {p3:.3f};")
print(f"FEF: M(SD) = {codeMorph_readout_data['fef'].mean():.3f}({codeMorph_readout_data['fef'].std():.3f}), p = {p4:.3f};")
print('\n')

############
# k-s test #
############
codeMorph_readout_ks = {'R@R-LPFC':scipy.stats.ks_2samp(codeMorph_readout_rnn['ed2'], codeMorph_readout_data['dlpfc']), 
                        'R@R-FEF':scipy.stats.ks_2samp(codeMorph_readout_rnn['ed2'], codeMorph_readout_data['fef']), 
                        'R&U-LPFC':scipy.stats.ks_2samp(codeMorph_readout_rnn['ed12'], codeMorph_readout_data['dlpfc']), 
                        'R&U-FEF':scipy.stats.ks_2samp(codeMorph_readout_rnn['ed12'], codeMorph_readout_data['fef'])}
print('############### K-S Test ##############')
for k in codeMorph_readout_ks.keys():
    print(f"{k}: D = {codeMorph_readout_ks[k].statistic:.3f}, p = {codeMorph_readout_ks[k].pvalue:.3f};")


# FK Test
fk_results = {'R@R-LPFC':scipy.stats.fligner(codeMorph_readout_rnn['ed2'], codeMorph_readout_data['dlpfc']),
                'R@R-FEF':scipy.stats.fligner(codeMorph_readout_rnn['ed2'], codeMorph_readout_data['fef']),
                'R&U-LPFC':scipy.stats.fligner(codeMorph_readout_rnn['ed12'], codeMorph_readout_data['dlpfc']),
                'R&U-FEF':scipy.stats.fligner(codeMorph_readout_rnn['ed12'], codeMorph_readout_data['fef'])}

print('############### F-K Test ##############')
for k in fk_results.keys():
    print(f'{k.upper()}: H = {fk_results[k][0]:.3f}, p = {fk_results[k][1]:.3f}')
#%%
############
# CvM test #
############
#codeMorph_readout_cvm = {'R@R-LPFC':scipy.stats.cramervonmises_2samp(codeMorph_readout_rnn['ed2'], codeMorph_readout_data['dlpfc']), 
#                         'R@R-FEF':scipy.stats.cramervonmises_2samp(codeMorph_readout_rnn['ed2'], codeMorph_readout_data['fef']), 
#                         'R&U-LPFC':scipy.stats.cramervonmises_2samp(codeMorph_readout_rnn['ed12'], codeMorph_readout_data['dlpfc']), 
#                         'R&U-FEF':scipy.stats.cramervonmises_2samp(codeMorph_readout_rnn['ed12'], codeMorph_readout_data['fef'])}

#############################
# Wasserstein Distance test #
#############################
#codeMorph_readout_wd = {'R@R-LPFC':scipy.stats.wasserstein_distance(codeMorph_readout_rnn['ed2'], codeMorph_readout_data['dlpfc']), 
#                        'R@R-FEF':scipy.stats.wasserstein_distance(codeMorph_readout_rnn['ed2'], codeMorph_readout_data['fef']), 
#                        'R&U-LPFC':scipy.stats.wasserstein_distance(codeMorph_readout_rnn['ed12'], codeMorph_readout_data['dlpfc']), 
#                        'R&U-FEF':scipy.stats.wasserstein_distance(codeMorph_readout_rnn['ed12'], codeMorph_readout_data['fef'])}
#%%






















#%%

######################################
# code transferability between items #
######################################

#%%
performanceX_Trans_12_data = np.load(f'{phd_path}/outputs/monkeys/' + 'performance12X_Trans_data.npy', allow_pickle=True).item()
performanceX_Trans_21_data = np.load(f'{phd_path}/outputs/monkeys/' + 'performance21X_Trans_data.npy', allow_pickle=True).item()
performanceX_Trans_12_shuff_data = np.load(f'{phd_path}/outputs/monkeys/' + 'performance12X_Trans_shuff_data.npy', allow_pickle=True).item()
performanceX_Trans_21_shuff_data = np.load(f'{phd_path}/outputs/monkeys/' + 'performance21X_Trans_shuff_data.npy', allow_pickle=True).item()

performanceX_Trans_12_rnn = np.load(f'{phd_path}/outputs/rnns/' + 'performanceX_12_rnn.npy', allow_pickle=True).item()
performanceX_Trans_21_rnn = np.load(f'{phd_path}/outputs/rnns/' + 'performanceX_21_rnn.npy', allow_pickle=True).item()
performanceX_Trans_12_shuff_rnn = np.load(f'{phd_path}/outputs/rnns/' + 'performanceX_12_shuff_rnn.npy', allow_pickle=True).item()
performanceX_Trans_21_shuff_rnn = np.load(f'{phd_path}/outputs/rnns/' + 'performanceX_21_shuff_rnn.npy', allow_pickle=True).item()
#%% late delay
checkpoints = [150, 550, 1050, 1450, 1850, 2350]#
checkpointsLabels = ['S1','ED1','LD1','S2','ED2','LD2'] #, 2800
avgInterval = {150:150, 550:250, 1050:250, 1450:150, 1850:250, 2350:250} #, 2800:200

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

#%%
cp1, cp2 = 'LD1', 'LD2'
cp1x, cp2x = checkpointsLabels.index(cp1), checkpointsLabels.index(cp2)
#%%
lineh = np.arange(0.3,0.7,0.001)
linev = np.arange(1.16,1.165,0.0001)

#fig = plt.figure(figsize=(3, 3), dpi=300)
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
#ax.plot(lineThresholds, np.full_like(lineThresholds,0.25), 'k--', alpha = 0.5)

# draw temporary red and blue lines and use them to create a legend
#plt.plot([], c='dimgrey', label='Delay1')
#plt.plot([], c='lightgrey', label='Delay2')
#plt.legend(bbox_to_anchor=(1.5, 0.6))#loc = 'right',

ax.set_xticks([0.5,1.5,2.5,3.5],['R@R','R&U', 'LPFC','FEF'], rotation=0)
ax.set_xlabel('Strategy/Region', labelpad = 5, fontsize = 12)
ax.set_ylim(top=1.15, bottom=0)
ax.set_title('Between-Item Transferability, Retarget', fontsize = 12, pad=10)
plt.show()

# save figure
#fig.savefig(f'{phd_path}/outputs/codeTrans_item.tif', bbox_inches='tight')

# print stats
print(f"R@R: M(SD) = {codeTrans_item_rnn['ed2'][1][:,cp1x, cp2x].mean():.3f}({codeTrans_item_rnn['ed2'][1][:,cp1x, cp2x].std():.3f}), p = {p1:.3f};")
print(f"R&U: M(SD) = {codeTrans_item_rnn['ed12'][1][:,cp1x, cp2x].mean():.3f}({codeTrans_item_rnn['ed12'][1][:,cp1x, cp2x].std():.3f}), p = {p2:.3f};")
print(f"LPFC: M(SD) = {codeTrans_item_data['dlpfc'][1][:,cp1x, cp2x].mean():.3f}({codeTrans_item_data['dlpfc'][1][:,cp1x, cp2x].std():.3f}), p = {p3:.3f};")
print(f"FEF: M(SD) = {codeTrans_item_data['fef'][1][:,cp1x, cp2x].mean():.3f}({codeTrans_item_data['fef'][1][:,cp1x, cp2x].std():.3f}), p = {p4:.3f};")
#%% multiple checkpoints

cps = (('ED1','ED2'),('LD1','LD2'),)

lineh = np.arange(0.3,0.7,0.001)
linev = np.arange(1.16,1.165,0.0001)

#fig = plt.figure(figsize=(3, 3), dpi=300)
fig, axes = plt.subplots(1,2, sharey=True, figsize=(8,4), dpi=300)

for ncp, cp in enumerate(cps):
    cp1, cp2 = cp
    cp1x, cp2x = checkpointsLabels.index(cp1), checkpointsLabels.index(cp2)
    cpn = 'Early Delay' if f'{cp1[0].upper()}' == 'E' else 'Late Delay'
    # full space
    ax = axes.flatten()[ncp]
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
    #ax.plot(lineThresholds, np.full_like(lineThresholds,0.25), 'k--', alpha = 0.5)

    # draw temporary red and blue lines and use them to create a legend
    #plt.plot([], c='dimgrey', label='Delay1')
    #plt.plot([], c='lightgrey', label='Delay2')
    #plt.legend(bbox_to_anchor=(1.5, 0.6))#loc = 'right',

    ax.set_xticks([0.5,1.5,2.5,3.5],['R@R','R&U', 'LPFC','FEF'], rotation=0)
    ax.set_xlabel('Strategy/Region', labelpad = 5, fontsize = 12)
    ax.set_ylim(top=1.15, bottom=0)
    ax.set_title(f'{cpn}', fontsize = 12, pad=10)
    
plt.suptitle('Between-Item Transferability, Retarget', fontsize = 12)
plt.show()



#%%

######################################
# code transferability between items ratio#
######################################

#%%
performanceX_Trans_12_data = np.load(f'{phd_path}/outputs/monkeys/' + 'performance12X_Trans_data.npy', allow_pickle=True).item()
performanceX_Trans_21_data = np.load(f'{phd_path}/outputs/monkeys/' + 'performance21X_Trans_data.npy', allow_pickle=True).item()
performanceX_Trans_12_shuff_data = np.load(f'{phd_path}/outputs/monkeys/' + 'performance12X_Trans_shuff_data.npy', allow_pickle=True).item()
performanceX_Trans_21_shuff_data = np.load(f'{phd_path}/outputs/monkeys/' + 'performance21X_Trans_shuff_data.npy', allow_pickle=True).item()

performanceX_Trans_12_rnn = np.load(f'{phd_path}/outputs/rnns/' + 'performanceX_12_rnn.npy', allow_pickle=True).item()
performanceX_Trans_21_rnn = np.load(f'{phd_path}/outputs/rnns/' + 'performanceX_21_rnn.npy', allow_pickle=True).item()
performanceX_Trans_12_shuff_rnn = np.load(f'{phd_path}/outputs/rnns/' + 'performanceX_12_shuff_rnn.npy', allow_pickle=True).item()
performanceX_Trans_21_shuff_rnn = np.load(f'{phd_path}/outputs/rnns/' + 'performanceX_21_shuff_rnn.npy', allow_pickle=True).item()


performance1_item_data = np.load(f'{phd_path}/outputs/monkeys/' + 'performance1_item_data.npy', allow_pickle=True).item()
performance2_item_data = np.load(f'{phd_path}/outputs/monkeys/' + 'performance2_item_data.npy', allow_pickle=True).item()
performance1_item_shuff_data = np.load(f'{phd_path}/outputs/monkeys/' + 'performance1_item_shuff_data.npy', allow_pickle=True).item()
performance2_item_shuff_data = np.load(f'{phd_path}/outputs/monkeys/' + 'performance2_item_shuff_data.npy', allow_pickle=True).item()

performance1_item_rnn = np.load(f'{phd_path}/outputs/rnns/' + 'performance1_item_rnn.npy', allow_pickle=True).item()
performance2_item_rnn = np.load(f'{phd_path}/outputs/rnns/' + 'performance2_item_rnn.npy', allow_pickle=True).item()
performance1_item_shuff_rnn = np.load(f'{phd_path}/outputs/rnns/' + 'performance1_item_shuff_rnn.npy', allow_pickle=True).item()
performance2_item_shuff_rnn = np.load(f'{phd_path}/outputs/rnns/' + 'performance2_item_shuff_rnn.npy', allow_pickle=True).item()
#%% late delay
checkpoints = [150, 550, 1050, 1450, 1850, 2350]#
checkpointsLabels = ['S1','ED1','LD1','S2','ED2','LD2'] #, 2800
avgInterval = {150:150, 550:250, 1050:250, 1450:150, 1850:250, 2350:250} #, 2800:200

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
#%%

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
    
#%%
cp1, cp2 = 'LD1', 'LD2'
cp1x, cp2x = checkpointsLabels.index(cp1), checkpointsLabels.index(cp2)

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
#%%

pltBsl = True

lineh = np.arange(0.3,0.7,0.001)
linev = np.arange(1.16,1.165,0.0001)

#fig = plt.figure(figsize=(3, 3), dpi=300)
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
#ax.plot(lineThresholds, np.full_like(lineThresholds,0.25), 'k--', alpha = 0.5)

# draw temporary red and blue lines and use them to create a legend
#plt.plot([], c='dimgrey', label='Delay1')
#plt.plot([], c='lightgrey', label='Delay2')
#plt.legend(bbox_to_anchor=(1.5, 0.6))#loc = 'right',

ax.set_xticks([0.5,1.5,2.5,3.5],['R@R','R&U', 'LPFC','FEF'], rotation=0)
ax.set_xlabel('Strategy/Region', labelpad = 5, fontsize = 12)
ax.set_ylim(top=2, bottom=0)
ax.set_title('Between-Item Transferability Ratio, Retarget', fontsize = 12, pad=10)
plt.show()

# save figure
fig.savefig(f'{phd_path}/outputs/codeTrans_ratio.tif', bbox_inches='tight', dpi=300)

# print stats
print(f"R@R: M(SD) = {codeTrans_ratio_rnn['ed2'][1].mean():.3f}({codeTrans_ratio_rnn['ed2'][1].std():.3f}), p = {p1:.3f};")
print(f"R&U: M(SD) = {codeTrans_ratio_rnn['ed12'][1].mean():.3f}({codeTrans_ratio_rnn['ed12'][1].std():.3f}), p = {p2:.3f};")
print(f"LPFC: M(SD) = {codeTrans_ratio_data['dlpfc'][1].mean():.3f}({codeTrans_ratio_data['dlpfc'][1].std():.3f}), p = {p3:.3f};")
print(f"FEF: M(SD) = {codeTrans_ratio_data['fef'][1].mean():.3f}({codeTrans_ratio_data['fef'][1].std():.3f}), p = {p4:.3f};")

############
# k-s test #
############
codeTrans_ratio_ks = {'R@R-LPFC':scipy.stats.ks_2samp(codeTrans_ratio_rnn['ed2'][1], codeTrans_ratio_data['dlpfc'][1]), 
                        'R@R-FEF':scipy.stats.ks_2samp(codeTrans_ratio_rnn['ed2'][1], codeTrans_ratio_data['fef'][1]), 
                        'R&U-LPFC':scipy.stats.ks_2samp(codeTrans_ratio_rnn['ed12'][1], codeTrans_ratio_data['dlpfc'][1]), 
                        'R&U-FEF':scipy.stats.ks_2samp(codeTrans_ratio_rnn['ed12'][1], codeTrans_ratio_data['fef'][1])}
print('############### K-S Test ##############')
for k in codeTrans_ratio_ks.keys():
    print(f"{k}: D = {codeTrans_ratio_ks[k].statistic:.3f}, p = {codeTrans_ratio_ks[k].pvalue:.3f};")

#%% multiple checkpoints


 
#%%

#%%

######################################
# code transferability between tasks #
######################################

#%%
performanceX_Trans_rdc_data = np.load(f'{phd_path}/outputs/monkeys/' + 'performanceX_Trans_rdc_data.npy', allow_pickle=True).item()
performanceX_Trans_drc_data = np.load(f'{phd_path}/outputs/monkeys/' + 'performanceX_Trans_drc_data.npy', allow_pickle=True).item()
performanceX_Trans_rdc_shuff_data = np.load(f'{phd_path}/outputs/monkeys/' + 'performanceX_Trans_rdc_shuff_data.npy', allow_pickle=True).item()
performanceX_Trans_drc_shuff_data = np.load(f'{phd_path}/outputs/monkeys/' + 'performanceX_Trans_drc_shuff_data.npy', allow_pickle=True).item()

performanceX_Trans_rdc_rnn = np.load(f'{phd_path}/outputs/rnns/' + 'performanceX_rdc_rnn.npy', allow_pickle=True).item()
performanceX_Trans_drc_rnn = np.load(f'{phd_path}/outputs/rnns/' + 'performanceX_drc_rnn.npy', allow_pickle=True).item()
performanceX_Trans_rdc_shuff_rnn = np.load(f'{phd_path}/outputs/rnns/' + 'performanceX_rdc_shuff_rnn.npy', allow_pickle=True).item()
performanceX_Trans_drc_shuff_rnn = np.load(f'{phd_path}/outputs/rnns/' + 'performanceX_drc_shuff_rnn.npy', allow_pickle=True).item()

#%%
performanceX_Trans_rdnc_data = np.load(f'{phd_path}/outputs/monkeys/' + 'performanceX_Trans_rdnc_data.npy', allow_pickle=True).item()
performanceX_Trans_drnc_data = np.load(f'{phd_path}/outputs/monkeys/' + 'performanceX_Trans_drnc_data.npy', allow_pickle=True).item()
performanceX_Trans_rdnc_shuff_data = np.load(f'{phd_path}/outputs/monkeys/' + 'performanceX_Trans_rdnc_shuff_data.npy', allow_pickle=True).item()
performanceX_Trans_drnc_shuff_data = np.load(f'{phd_path}/outputs/monkeys/' + 'performanceX_Trans_drnc_shuff_data.npy', allow_pickle=True).item()

performanceX_Trans_rdnc_rnn = np.load(f'{phd_path}/outputs/rnns/' + 'performanceX_rdnc_rnn.npy', allow_pickle=True).item()
performanceX_Trans_drnc_rnn = np.load(f'{phd_path}/outputs/rnns/' + 'performanceX_drnc_rnn.npy', allow_pickle=True).item()
performanceX_Trans_rdnc_shuff_rnn = np.load(f'{phd_path}/outputs/rnns/' + 'performanceX_rdnc_shuff_rnn.npy', allow_pickle=True).item()
performanceX_Trans_drnc_shuff_rnn = np.load(f'{phd_path}/outputs/rnns/' + 'performanceX_drnc_shuff_rnn.npy', allow_pickle=True).item()

#%%
checkpoints = [150, 550, 1150, 1450, 1850, 2350]#
checkpointsLabels = ['S1','ED1','LD1','S2','ED2','LD2'] #, 2800
avgInterval = {150:150, 550:250, 1150:250, 1450:150, 1850:250, 2350:250} #, 2800:200

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

#%%
cp1, cp2 = 'LD2', 'LD2'
cp1x, cp2x = checkpointsLabels.index(cp1), checkpointsLabels.index(cp2)
#%%
lineh = np.arange(0.3,0.7,0.001)
linev = np.arange(1.16,1.165,0.0001)

#fig = plt.figure(figsize=(3, 3), dpi=300)
fig, axes = plt.subplots(1,1, sharey=True, figsize=(3,3), dpi=300)

# full space
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
#ax.plot(lineThresholds, np.full_like(lineThresholds,0.25), 'k--', alpha = 0.5)

# draw temporary red and blue lines and use them to create a legend
#plt.plot([], c='dimgrey', label='Delay1')
#plt.plot([], c='lightgrey', label='Delay2')
#plt.legend(bbox_to_anchor=(1.5, 0.6))#loc = 'right',

ax.set_xticks([0.5,1.5,2.5,3.5],['R@R','R&U', 'LPFC','FEF'], rotation=0)
ax.set_xlabel('Strategy/Region', labelpad = 5, fontsize = 12)
ax.set_ylim(top=1.15, bottom=0)
ax.set_title('Between-Type Transferability, Choice Item', fontsize = 12, pad=10)
plt.show()

# save fig
fig.savefig(f'{phd_path}/outputs/codeTrans_choice.tif', bbox_inches='tight')

# print stats
print(f"R@R: M(SD) = {codeTrans_choice_rnn['ed2'].mean():.3f}({codeTrans_choice_rnn['ed2'].std():.3f}), p = {p1:.3f};")
print(f"R&U: M(SD) = {codeTrans_choice_rnn['ed12'].mean():.3f}({codeTrans_choice_rnn['ed12'].std():.3f}), p = {p2:.3f};")
print(f"LPFC: M(SD) = {codeTrans_choice_data['dlpfc'].mean():.3f}({codeTrans_choice_data['dlpfc'].std():.3f}), p = {p3:.3f};")
print(f"FEF: M(SD) = {codeTrans_choice_data['fef'].mean():.3f}({codeTrans_choice_data['fef'].std():.3f}), p = {p4:.3f};")
#%%

############################################
# code transferability between tasks ratio #
############################################

#%%
performanceX_Trans_rdc_data = np.load(f'{phd_path}/outputs/monkeys/' + 'performanceX_Trans_rdc_data.npy', allow_pickle=True).item()
performanceX_Trans_drc_data = np.load(f'{phd_path}/outputs/monkeys/' + 'performanceX_Trans_drc_data.npy', allow_pickle=True).item()
performanceX_Trans_rdc_shuff_data = np.load(f'{phd_path}/outputs/monkeys/' + 'performanceX_Trans_rdc_shuff_data.npy', allow_pickle=True).item()
performanceX_Trans_drc_shuff_data = np.load(f'{phd_path}/outputs/monkeys/' + 'performanceX_Trans_drc_shuff_data.npy', allow_pickle=True).item()

performanceX_Trans_rdc_rnn = np.load(f'{phd_path}/outputs/rnns/' + 'performanceX_rdc_rnn.npy', allow_pickle=True).item()
performanceX_Trans_drc_rnn = np.load(f'{phd_path}/outputs/rnns/' + 'performanceX_drc_rnn.npy', allow_pickle=True).item()
performanceX_Trans_rdc_shuff_rnn = np.load(f'{phd_path}/outputs/rnns/' + 'performanceX_rdc_shuff_rnn.npy', allow_pickle=True).item()
performanceX_Trans_drc_shuff_rnn = np.load(f'{phd_path}/outputs/rnns/' + 'performanceX_drc_shuff_rnn.npy', allow_pickle=True).item()

performanceX_Trans_rdnc_data = np.load(f'{phd_path}/outputs/monkeys/' + 'performanceX_Trans_rdnc_data.npy', allow_pickle=True).item()
performanceX_Trans_drnc_data = np.load(f'{phd_path}/outputs/monkeys/' + 'performanceX_Trans_drnc_data.npy', allow_pickle=True).item()
performanceX_Trans_rdnc_shuff_data = np.load(f'{phd_path}/outputs/monkeys/' + 'performanceX_Trans_rdnc_shuff_data.npy', allow_pickle=True).item()
performanceX_Trans_drnc_shuff_data = np.load(f'{phd_path}/outputs/monkeys/' + 'performanceX_Trans_drnc_shuff_data.npy', allow_pickle=True).item()

performanceX_Trans_rdnc_rnn = np.load(f'{phd_path}/outputs/rnns/' + 'performanceX_rdnc_rnn.npy', allow_pickle=True).item()
performanceX_Trans_drnc_rnn = np.load(f'{phd_path}/outputs/rnns/' + 'performanceX_drnc_rnn.npy', allow_pickle=True).item()
performanceX_Trans_rdnc_shuff_rnn = np.load(f'{phd_path}/outputs/rnns/' + 'performanceX_rdnc_shuff_rnn.npy', allow_pickle=True).item()
performanceX_Trans_drnc_shuff_rnn = np.load(f'{phd_path}/outputs/rnns/' + 'performanceX_drnc_shuff_rnn.npy', allow_pickle=True).item()


performance1_item_data = np.load(f'{phd_path}/outputs/monkeys/' + 'performance1_item_data.npy', allow_pickle=True).item()
performance2_item_data = np.load(f'{phd_path}/outputs/monkeys/' + 'performance2_item_data.npy', allow_pickle=True).item()
performance1_item_shuff_data = np.load(f'{phd_path}/outputs/monkeys/' + 'performance1_item_shuff_data.npy', allow_pickle=True).item()
performance2_item_shuff_data = np.load(f'{phd_path}/outputs/monkeys/' + 'performance2_item_shuff_data.npy', allow_pickle=True).item()

performance1_item_rnn = np.load(f'{phd_path}/outputs/rnns/' + 'performance1_item_rnn.npy', allow_pickle=True).item()
performance2_item_rnn = np.load(f'{phd_path}/outputs/rnns/' + 'performance2_item_rnn.npy', allow_pickle=True).item()
performance1_item_shuff_rnn = np.load(f'{phd_path}/outputs/rnns/' + 'performance1_item_shuff_rnn.npy', allow_pickle=True).item()
performance2_item_shuff_rnn = np.load(f'{phd_path}/outputs/rnns/' + 'performance2_item_shuff_rnn.npy', allow_pickle=True).item()
#%%
checkpoints = [150, 550, 1050, 1450, 1850, 2350]#
checkpointsLabels = ['S1','ED1','LD1','S2','ED2','LD2'] #, 2800
avgInterval = {150:150, 550:250, 1050:250, 1450:150, 1850:250, 2350:250} #, 2800:200

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

#%%

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
    
#%%
cp1, cp2 = 'LD2', 'LD2'
cp1x, cp2x = checkpointsLabels.index(cp1), checkpointsLabels.index(cp2)

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
#%%

pltBsl = True

lineh = np.arange(0.3,0.7,0.001)
linev = np.arange(1.16,1.165,0.0001)

#fig = plt.figure(figsize=(3, 3), dpi=300)
fig, axes = plt.subplots(1,1, sharey=True, figsize=(3,3), dpi=300)

# full space
ax = axes#.flatten()[0]

ax.boxplot([codeTrans_ratio_rnn['ed2']], positions=[0.5-pltBsl*0.2], patch_artist=True, widths = 0.2, boxprops=boxprops1, flierprops=flierprops1, 
                  meanprops=meanpointprops1, medianprops=medianprops, capprops = capprops1, whiskerprops = whiskerprops1, meanline=False, showmeans=True)
ax.boxplot([codeTrans_ratio_rnn['ed12']], positions=[1.5-pltBsl*0.2], patch_artist=True, widths = 0.2, boxprops=boxprops2, flierprops=flierprops2, 
                  meanprops=meanpointprops2, medianprops=medianprops, capprops = capprops2, whiskerprops = whiskerprops2, meanline=False, showmeans=True)
ax.boxplot([codeTrans_ratio_data['dlpfc']], positions=[2.5-pltBsl*0.2], patch_artist=True, widths = 0.2, boxprops=boxprops3, flierprops=flierprops3, 
                  meanprops=meanpointprops3, medianprops=medianprops, capprops = capprops3, whiskerprops = whiskerprops3, meanline=False, showmeans=True)
ax.boxplot([codeTrans_ratio_data['fef']], positions=[3.5-pltBsl*0.2], patch_artist=True, widths = 0.2, boxprops=boxprops4, flierprops=flierprops4, 
                  meanprops=meanpointprops4, medianprops=medianprops, capprops = capprops4, whiskerprops = whiskerprops4, meanline=False, showmeans=True)

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
lineThresholds = np.arange(xlims[0],xlims[1]+0.01,0.01)
#ax.plot(lineThresholds, np.full_like(lineThresholds,0.25), 'k--', alpha = 0.5)

# draw temporary red and blue lines and use them to create a legend
#plt.plot([], c='dimgrey', label='Delay1')
#plt.plot([], c='lightgrey', label='Delay2')
#plt.legend(bbox_to_anchor=(1.5, 0.6))#loc = 'right',

ax.set_xticks([0.5,1.5,2.5,3.5],['R@R','R&U', 'LPFC','FEF'], rotation=0)
ax.set_xlabel('Strategy/Region', labelpad = 5, fontsize = 12)
ax.set_ylim(top=2, bottom=0)
ax.set_title('Between-Task Transferability Ratio of Choice Items', fontsize = 12, pad=10)
plt.show()

# save figure
fig.savefig(f'{phd_path}/outputs/codeTrans_ratio_choice.tif', bbox_inches='tight', dpi=300)

# print stats
print(f"R@R: M(SD) = {codeTrans_ratio_rnn['ed2'].mean():.3f}({codeTrans_ratio_rnn['ed2'].std():.3f}), p = {p1:.3f};")
print(f"R&U: M(SD) = {codeTrans_ratio_rnn['ed12'].mean():.3f}({codeTrans_ratio_rnn['ed12'].std():.3f}), p = {p2:.3f};")
print(f"LPFC: M(SD) = {codeTrans_ratio_data['dlpfc'].mean():.3f}({codeTrans_ratio_data['dlpfc'].std():.3f}), p = {p3:.3f};")
print(f"FEF: M(SD) = {codeTrans_ratio_data['fef'].mean():.3f}({codeTrans_ratio_data['fef'].std():.3f}), p = {p4:.3f};")

############
# k-s test #
############
codeTrans_ratio_ks = {'R@R-LPFC':scipy.stats.ks_2samp(codeTrans_ratio_rnn['ed2'], codeTrans_ratio_data['dlpfc']), 
                        'R@R-FEF':scipy.stats.ks_2samp(codeTrans_ratio_rnn['ed2'], codeTrans_ratio_data['fef']), 
                        'R&U-LPFC':scipy.stats.ks_2samp(codeTrans_ratio_rnn['ed12'], codeTrans_ratio_data['dlpfc']), 
                        'R&U-FEF':scipy.stats.ks_2samp(codeTrans_ratio_rnn['ed12'], codeTrans_ratio_data['fef'])}
print('############### K-S Test ##############')
for k in codeTrans_ratio_ks.keys():
    print(f"{k}: D = {codeTrans_ratio_ks[k].statistic:.3f}, p = {codeTrans_ratio_ks[k].pvalue:.3f};")

#%%






































































# %%
