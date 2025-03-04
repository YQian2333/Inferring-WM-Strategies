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
# In[] decodability with/without permutation P value

euDists_monkeys = np.load(f'{phd_path}/outputs/monkeys/euDists_monkeys_centroids2_normalized_full.npy', allow_pickle=True).item()
euDists_rnns = np.load(f'{phd_path}/outputs/rnns/euDists_rnns_centroids2_normalized_full.npy', allow_pickle=True).item()

euDists_monkeys_shuff = np.load(f'{phd_path}/outputs/monkeys/euDists_shuff_monkeys_centroids2_normalized_full.npy', allow_pickle=True).item()
euDists_rnns_shuff = np.load(f'{phd_path}/outputs/rnns/euDists_rnns_centroids2_shuff_normalized_full.npy', allow_pickle=True).item()

#%%

#############
# distances #
#############

# In[]
lineh = np.arange(0.3,0.7,0.001)
linev = np.arange(0,0.1,0.001)
#linev = np.arange(1.15,1.25,0.001)
#linev = np.arange(2.47,2.5,0.001)

fig, axes = plt.subplots(2,2, figsize=(10,6),dpi=300, sharey=True)

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




for l1 in locs:

    ax = axes.flatten()[l1]

    ########
    # rnns #
    ########


    euDists_rnns_ed2_1 = np.array(euDists_rnns['ed2'][1]).squeeze()[:,l1*3:l1*3+3].mean(1)
    euDists_rnns_ed2_2 = np.array(euDists_rnns['ed2'][2]).squeeze()[:,l1*3:l1*3+3].mean(1)
    euDists_rnns_ed12_1 = np.array(euDists_rnns['ed12'][1]).squeeze()[:,l1*3:l1*3+3].mean(1)
    euDists_rnns_ed12_2 = np.array(euDists_rnns['ed12'][2]).squeeze()[:,l1*3:l1*3+3].mean(1)


    ax.boxplot([euDists_rnns_ed2_1], positions=[0.3], patch_artist=True, widths = 0.2, boxprops=boxprops1, flierprops=flierprops1,
                      meanprops=meanpointprops1, medianprops=medianprops, capprops = capprops1, whiskerprops = whiskerprops1, meanline=False, showmeans=True)
    ax.boxplot([euDists_rnns_ed12_1], positions=[1.3], patch_artist=True, widths = 0.2, boxprops=boxprops2, flierprops=flierprops2,
                      meanprops=meanpointprops2, medianprops=medianprops, capprops = capprops2, whiskerprops = whiskerprops2, meanline=False, showmeans=True)

    ax.boxplot([euDists_rnns_ed2_2], positions=[0.7], patch_artist=True, widths = 0.2, boxprops=dict(boxprops1,facecolor=color1_), flierprops=dict(flierprops1,markeredgecolor=color1_, markerfacecolor=color1_),
                      meanprops=dict(meanpointprops1,markeredgecolor=color1_), medianprops=medianprops, capprops = dict(capprops1,color=color1_), whiskerprops = dict(whiskerprops1,color=color1_), meanline=False, showmeans=True)
    ax.boxplot([euDists_rnns_ed12_2], positions=[1.7], patch_artist=True, widths = 0.2, boxprops=dict(boxprops2,facecolor=color2_), flierprops=dict(flierprops2,markeredgecolor=color2_, markerfacecolor=color2_),
                      meanprops=dict(meanpointprops2,markeredgecolor=color2_), medianprops=medianprops, capprops = dict(capprops2,color=color2_), whiskerprops = dict(whiskerprops2,color=color2_), meanline=False, showmeans=True)


    ###########
    # monkeys #
    ###########

    euDists_data_dlpfc_1 = np.array(euDists_monkeys['dlpfc'][1])[:,l1*3:l1*3+3].mean(1)
    euDists_data_dlpfc_2 = np.array(euDists_monkeys['dlpfc'][2])[:,l1*3:l1*3+3].mean(1)
    euDists_data_fef_1 = np.array(euDists_monkeys['fef'][1])[:,l1*3:l1*3+3].mean(1)
    euDists_data_fef_2 = np.array(euDists_monkeys['fef'][2])[:,l1*3:l1*3+3].mean(1)


    ax.boxplot([euDists_data_dlpfc_1], positions=[2.3], patch_artist=True, widths = 0.2, boxprops=boxprops3, flierprops=flierprops3,
                      meanprops=meanpointprops3, medianprops=medianprops, capprops = capprops3, whiskerprops = whiskerprops3, meanline=False, showmeans=True)
    ax.boxplot([euDists_data_fef_1], positions=[3.3], patch_artist=True, widths = 0.2, boxprops=boxprops4, flierprops=flierprops4,
                      meanprops=meanpointprops4, medianprops=medianprops, capprops = capprops4, whiskerprops = whiskerprops4, meanline=False, showmeans=True)

    ax.boxplot([euDists_data_dlpfc_2], positions=[2.7], patch_artist=True, widths = 0.2, boxprops=dict(boxprops3,facecolor=color3_), flierprops=dict(flierprops3,markeredgecolor=color3_, markerfacecolor=color3_),
                      meanprops=dict(meanpointprops3,markeredgecolor=color3_), medianprops=medianprops, capprops = dict(capprops3,color=color3_), whiskerprops = dict(whiskerprops3,color=color3_), meanline=False, showmeans=True)
    ax.boxplot([euDists_data_fef_2], positions=[3.7], patch_artist=True, widths = 0.2, boxprops=dict(boxprops4,facecolor=color4_), flierprops=dict(flierprops4,markeredgecolor=color4_, markerfacecolor=color4_),
                      meanprops=dict(meanpointprops4,markeredgecolor=color4_), medianprops=medianprops, capprops = dict(capprops4,color=color4_), whiskerprops = dict(whiskerprops4,color=color4_), meanline=False, showmeans=True)


    ylims = ax.get_ylim()
    yscale = (ylims[1] - ylims[0])//(ylims[1]/2)

    #p1 = scipy.stats.ttest_ind(euDists['ed2'][1], euDists['ed2'][2])[-1]
    p1 = f_stats.permutation_p_diff(euDists_rnns_ed2_1, euDists_rnns_ed2_2)
    ax.plot(lineh, np.full_like(lineh, 7+0.1), 'k-')
    #ax.plot(lineh, np.full_like(lineh, 1.25), 'k-')
    ax.plot(np.full_like(linev+7, 0.3), linev+7, 'k-')
    ax.plot(np.full_like(linev+7, 0.7), linev+7, 'k-')
    ax.text(0.5,7.25, f'{f_plotting.sig_marker(p1,ns_note=True)}',horizontalalignment='center', fontsize=12)
    #ax.text(0.5,1.25, f'{f_plotting.sig_marker(p1)}',horizontalalignment='center', fontsize=12)


    #p2 = scipy.stats.ttest_ind(euDists['ed12'][1], euDists['ed12'][2])[-1]
    p2 = f_stats.permutation_p_diff(euDists_rnns_ed12_1, euDists_rnns_ed12_2)
    ax.plot(lineh+1, np.full_like(lineh, 7+0.1), 'k-')
    #ax.plot(lineh+1, np.full_like(lineh, 1.25), 'k-')
    ax.plot(np.full_like(linev+7, 1.3), linev+7, 'k-')
    ax.plot(np.full_like(linev+7, 1.7), linev+7, 'k-')
    ax.text(1.5,7.25, f'{f_plotting.sig_marker(p2,ns_note=True)}',horizontalalignment='center', fontsize=12)
    #ax.text(1.5,1.25, f'{f_plotting.sig_marker(p2)}',horizontalalignment='center', fontsize=12)


    #p1 = scipy.stats.ttest_rel(euDists['dlpfc'][1].mean(1), euDists['dlpfc'][2].mean(1))[-1]
    #p1,_,_ = f_stats.bootstrap95_p(euDists['dlpfc'][1].mean(1), euDists['dlpfc'][2].mean(1)) # shape: nIters * ntrials
    p3 = f_stats.permutation_p_diff(euDists_data_dlpfc_1, euDists_data_dlpfc_2) # shape: nIters * ntrials

    ax.plot(lineh+2, np.full_like(lineh, 7+0.1), 'k-')
    #ax.plot(lineh, np.full_like(lineh, 1.25), 'k-')
    ax.plot(np.full_like(linev+7, 2.3), linev+7, 'k-')
    ax.plot(np.full_like(linev+7, 2.7), linev+7, 'k-')
    ax.text(2.5,7.25, f'{f_plotting.sig_marker(p3,ns_note=True)}',horizontalalignment='center', fontsize=12)
    #plt.text(0.5,1.25, f'{f_plotting.sig_marker(p1)}',horizontalalignment='center', fontsize=12)


    #p2 = scipy.stats.ttest_rel(euDists['fef'][1].mean(1), euDists['fef'][2].mean(1))[-1]
    #p2,_,_ = f_stats.bootstrap95_p(euDists['fef'][1].mean(1), euDists['fef'][2].mean(1))
    p4 = f_stats.permutation_p_diff(euDists_data_fef_1, euDists_data_fef_2)

    ax.plot(lineh+3, np.full_like(lineh, 7+0.1), 'k-')
    #ax.plot(lineh+1, np.full_like(lineh, 1.25), 'k-')
    ax.plot(np.full_like(linev+7, 3.3), linev+7, 'k-')
    ax.plot(np.full_like(linev+7, 3.7), linev+7, 'k-')
    ax.text(3.5,7.25, f'{f_plotting.sig_marker(p4,ns_note=True)}',horizontalalignment='center', fontsize=12)
    #ax.text(1.5,1.25, f'{f_plotting.sig_marker(p2)}',horizontalalignment='center', fontsize=12)

    if l1 in (3,):
        # draw temporary red and blue lines and use them to create a legend
        ax.plot([], c='dimgrey', label='Retarget')
        ax.plot([], c='lightgrey', label='Distraction')
        ax.legend(bbox_to_anchor=(1.1, 0.6), fontsize = 10)#loc = 'right',
    
    ax.set_xticks([0.5,1.5,2.5,3.5],['R@R','R&U','LPFC','FEF'], rotation=0)
    if l1 in (2,3):    
        ax.set_xlabel('Strategies/Regions', labelpad = 5, fontsize = 12)
    
    if l1 in (0,2):
        ax.set_ylabel('Euclidean Distance (Std. Scaled)', labelpad = 3, fontsize = 8)
    
    ax.set_ylim(top=8.5)

    #plt.ylim(0.5,2.75)
    ax.set_title(f'Item1 Location = {l1+1}', fontsize = 10, pad=3)

plt.suptitle('Mean Projection Drift', fontsize = 20, y=1)
#plt.tight_layout()
plt.show()

#%%
fig.savefig(f'{phd_path}/outputs/driftDist_centroids2_byloc.tif', bbox_inches='tight')
#%%

#############
# distances #
#############

# In[]
lineh = np.arange(0.3,0.7,0.001)
linev = np.arange(0,0.03,0.001)
#linev = np.arange(1.15,1.25,0.001)
#linev = np.arange(2.47,2.5,0.001)

fig, axes = plt.subplots(1,1, figsize=(4,4),dpi=300, sharey=True)

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


euDists_rnns_ed2_1 = np.array(euDists_rnns['ed2'][1]).squeeze().mean(1)
euDists_rnns_ed2_2 = np.array(euDists_rnns['ed2'][2]).squeeze().mean(1)
euDists_rnns_ed12_1 = np.array(euDists_rnns['ed12'][1]).squeeze().mean(1)
euDists_rnns_ed12_2 = np.array(euDists_rnns['ed12'][2]).squeeze().mean(1)


ax.boxplot([euDists_rnns_ed2_1], positions=[0.3], patch_artist=True, widths = 0.2, boxprops=boxprops1, flierprops=flierprops1,
                    meanprops=meanpointprops1, medianprops=medianprops, capprops = capprops1, whiskerprops = whiskerprops1, meanline=False, showmeans=True)
ax.boxplot([euDists_rnns_ed12_1], positions=[1.3], patch_artist=True, widths = 0.2, boxprops=boxprops2, flierprops=flierprops2,
                    meanprops=meanpointprops2, medianprops=medianprops, capprops = capprops2, whiskerprops = whiskerprops2, meanline=False, showmeans=True)

ax.boxplot([euDists_rnns_ed2_2], positions=[0.7], patch_artist=True, widths = 0.2, boxprops=dict(boxprops1,facecolor=color1_), flierprops=dict(flierprops1,markeredgecolor=color1_, markerfacecolor=color1_),
                    meanprops=dict(meanpointprops1,markeredgecolor=color1_), medianprops=medianprops, capprops = dict(capprops1,color=color1_), whiskerprops = dict(whiskerprops1,color=color1_), meanline=False, showmeans=True)
ax.boxplot([euDists_rnns_ed12_2], positions=[1.7], patch_artist=True, widths = 0.2, boxprops=dict(boxprops2,facecolor=color2_), flierprops=dict(flierprops2,markeredgecolor=color2_, markerfacecolor=color2_),
                    meanprops=dict(meanpointprops2,markeredgecolor=color2_), medianprops=medianprops, capprops = dict(capprops2,color=color2_), whiskerprops = dict(whiskerprops2,color=color2_), meanline=False, showmeans=True)


###########
# monkeys #
###########

euDists_data_dlpfc_1 = np.array(euDists_monkeys['dlpfc'][1]).mean(1)#_shuff
euDists_data_dlpfc_2 = np.array(euDists_monkeys['dlpfc'][2]).mean(1)#_shuff
euDists_data_fef_1 = np.array(euDists_monkeys['fef'][1]).mean(1)#_shuff
euDists_data_fef_2 = np.array(euDists_monkeys['fef'][2]).mean(1)#_shuff


ax.boxplot([euDists_data_dlpfc_1], positions=[2.3], patch_artist=True, widths = 0.2, boxprops=boxprops3, flierprops=flierprops3,
                    meanprops=meanpointprops3, medianprops=medianprops, capprops = capprops3, whiskerprops = whiskerprops3, meanline=False, showmeans=True)
ax.boxplot([euDists_data_fef_1], positions=[3.3], patch_artist=True, widths = 0.2, boxprops=boxprops4, flierprops=flierprops4,
                    meanprops=meanpointprops4, medianprops=medianprops, capprops = capprops4, whiskerprops = whiskerprops4, meanline=False, showmeans=True)

ax.boxplot([euDists_data_dlpfc_2], positions=[2.7], patch_artist=True, widths = 0.2, boxprops=dict(boxprops3,facecolor=color3_), flierprops=dict(flierprops3,markeredgecolor=color3_, markerfacecolor=color3_),
                    meanprops=dict(meanpointprops3,markeredgecolor=color3_), medianprops=medianprops, capprops = dict(capprops3,color=color3_), whiskerprops = dict(whiskerprops3,color=color3_), meanline=False, showmeans=True)
ax.boxplot([euDists_data_fef_2], positions=[3.7], patch_artist=True, widths = 0.2, boxprops=dict(boxprops4,facecolor=color4_), flierprops=dict(flierprops4,markeredgecolor=color4_, markerfacecolor=color4_),
                    meanprops=dict(meanpointprops4,markeredgecolor=color4_), medianprops=medianprops, capprops = dict(capprops4,color=color4_), whiskerprops = dict(whiskerprops4,color=color4_), meanline=False, showmeans=True)


ylims = ax.get_ylim()
yscale = (ylims[1] - ylims[0])//(ylims[1]//2)

print('R@R')
p1 = f_stats.permutation_p_diff(euDists_rnns_ed2_1, euDists_rnns_ed2_2)
ax.plot(lineh, np.full_like(lineh, 2+0.03), 'k-')
#ax.plot(lineh, np.full_like(lineh, 1.25), 'k-')
ax.plot(np.full_like(linev+2, 0.3), linev+2, 'k-')
ax.plot(np.full_like(linev+2, 0.7), linev+2, 'k-')
ax.text(0.5,2.1, f'{f_plotting.sig_marker(p1,ns_note=True)}',horizontalalignment='center', fontsize=12)
#ax.text(0.5,1.25, f'{f_plotting.sig_marker(p1)}',horizontalalignment='center', fontsize=12)
print(f"MD = {euDists_rnns_ed2_1.mean() - euDists_rnns_ed2_2.mean():.3f}, p = {p1:.3f}, g = {f_stats.hedges_g(euDists_rnns_ed2_1, euDists_rnns_ed2_2):.3f}")
#print(f'Mean Diff = {euDists_rnns_ed2_1.mean() - euDists_rnns_ed2_2.mean():.3f}')
#print(f'p = {p1:.3f}')
#print(f"Hedge's g = {f_stats.hedges_g(euDists_rnns_ed2_1, euDists_rnns_ed2_2):.3f}")

print('R&U')
p2 = f_stats.permutation_p_diff(euDists_rnns_ed12_1, euDists_rnns_ed12_2)
ax.plot(lineh+1, np.full_like(lineh, 2+0.03), 'k-')
#ax.plot(lineh+1, np.full_like(lineh, 1.25), 'k-')
ax.plot(np.full_like(linev+2, 1.3), linev+2, 'k-')
ax.plot(np.full_like(linev+2, 1.7), linev+2, 'k-')
ax.text(1.5,2.1, f'{f_plotting.sig_marker(p2,ns_note=True)}',horizontalalignment='center', fontsize=12)
#ax.text(1.5,1.25, f'{f_plotting.sig_marker(p2)}',horizontalalignment='center', fontsize=12)
print(f"MD = {euDists_rnns_ed12_1.mean() - euDists_rnns_ed12_2.mean():.3f}, p = {p2:.3f}, g = {f_stats.hedges_g(euDists_rnns_ed12_1, euDists_rnns_ed12_2):.3f}")
#print(f'Mean Diff = {euDists_rnns_ed12_1.mean() - euDists_rnns_ed12_2.mean():.3f}')
#print(f'p = {p2:.3f}')
#print(f"Hedge's g = {f_stats.hedges_g(euDists_rnns_ed12_1, euDists_rnns_ed12_2):.3f}")


print('LPFC')
p3 = f_stats.permutation_p_diff(euDists_data_dlpfc_1, euDists_data_dlpfc_2) # shape: nIters * ntrials

ax.plot(lineh+2, np.full_like(lineh, 2+0.03), 'k-')
#ax.plot(lineh, np.full_like(lineh, 1.25), 'k-')
ax.plot(np.full_like(linev+2, 2.3), linev+2, 'k-')
ax.plot(np.full_like(linev+2, 2.7), linev+2, 'k-')
ax.text(2.5,2.1, f'{f_plotting.sig_marker(p3,ns_note=True)}',horizontalalignment='center', fontsize=12)
#plt.text(0.5,1.25, f'{f_plotting.sig_marker(p1)}',horizontalalignment='center', fontsize=12)
print(f"MD = {euDists_data_dlpfc_1.mean() - euDists_data_dlpfc_2.mean():.3f}, p = {p3:.3f}, g = {f_stats.hedges_g(euDists_data_dlpfc_1, euDists_data_dlpfc_2):.3f}")
#print(f'Mean Diff = {euDists_data_dlpfc_1.mean() - euDists_data_dlpfc_2.mean():.3f}')
#print(f'p = {p3:.3f}')
#print(f"Hedge's g = {f_stats.hedges_g(euDists_data_dlpfc_1, euDists_data_dlpfc_2):.3f}")

#p2 = scipy.stats.ttest_rel(euDists['fef'][1].mean(1), euDists['fef'][2].mean(1))[-1]
#p2,_,_ = f_stats.bootstrap95_p(euDists['fef'][1].mean(1), euDists['fef'][2].mean(1))
print('FEF')
p4 = f_stats.permutation_p_diff(euDists_data_fef_1, euDists_data_fef_2)

ax.plot(lineh+3, np.full_like(lineh, 2+0.03), 'k-')
#ax.plot(lineh+1, np.full_like(lineh, 1.25), 'k-')
ax.plot(np.full_like(linev+2, 3.3), linev+2, 'k-')
ax.plot(np.full_like(linev+2, 3.7), linev+2, 'k-')
ax.text(3.5,2.1, f'{f_plotting.sig_marker(p4,ns_note=True)}',horizontalalignment='center', fontsize=12)
#ax.text(1.5,1.25, f'{f_plotting.sig_marker(p2)}',horizontalalignment='center', fontsize=12)
print(f"MD = {euDists_data_fef_1.mean() - euDists_data_fef_2.mean():.3f}, p = {p4:.3f}, g = {f_stats.hedges_g(euDists_data_fef_1, euDists_data_fef_2):.3f}")
#print(f'Mean Diff = {euDists_data_fef_1.mean() - euDists_data_fef_2.mean():.3f}')
#print(f'p = {p4:.3f}')
#print(f"Hedge' g = {f_stats.hedges_g(euDists_data_fef_1, euDists_data_fef_2):.3f}")

ax.plot([], c='dimgrey', label='Retarget')
ax.plot([], c='lightgrey', label='Distraction')
#ax.legend(bbox_to_anchor=(1.3, 0.6), fontsize = 10)#loc = 'right',

ax.set_xticks([0.5,1.5,2.5,3.5],['R@R','R&U','LPFC','FEF'], rotation=0)
ax.set_xlabel('Strategies/Regions', labelpad = 5, fontsize = 12)

ax.set_ylabel('Euclidean Distance (Std. Scaled)', labelpad = 3, fontsize = 10)

ax.set_ylim(top=2.5)


plt.suptitle('Mean Projection Drift', fontsize = 20, y=1)
#plt.tight_layout()
plt.show()

#%%
fig.savefig(f'{phd_path}/outputs/driftDist_centroids2_normalized_full.tif', bbox_inches='tight')















#%%

##################
# distance ratio #
##################

# In[]
lineh = np.arange(0.3,0.7,0.001)
linev = np.arange(0,0.1,0.001)
#linev = np.arange(1.15,1.25,0.001)
#linev = np.arange(2.47,2.5,0.001)

fig, axes = plt.subplots(2,2, figsize=(10,6),dpi=300, sharey=True)

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




for l1 in locs:

    ax = axes.flatten()[l1]

    ########
    # rnns #
    ########

    euDists_rnns_ed2_1 = np.array(euDists_rnns['ed2'][1]).squeeze()[:,l1*3:l1*3+3].mean(1)
    euDists_rnns_ed2_2 = np.array(euDists_rnns['ed2'][2]).squeeze()[:,l1*3:l1*3+3].mean(1)
    euDists_rnns_ed12_1 = np.array(euDists_rnns['ed12'][1]).squeeze()[:,l1*3:l1*3+3].mean(1)
    euDists_rnns_ed12_2 = np.array(euDists_rnns['ed12'][2]).squeeze()[:,l1*3:l1*3+3].mean(1)
    
    euDistsRatio_rnns_ed2 = euDists_rnns_ed2_2/euDists_rnns_ed2_1
    euDistsRatio_rnns_ed12 = euDists_rnns_ed12_2/euDists_rnns_ed12_1
    

    ax.boxplot([euDistsRatio_rnns_ed2], positions=[0.5], patch_artist=True, widths = 0.2, boxprops=boxprops1, flierprops=flierprops1,
                      meanprops=meanpointprops1, medianprops=medianprops, capprops = capprops1, whiskerprops = whiskerprops1, meanline=False, showmeans=True)
    ax.boxplot([euDistsRatio_rnns_ed12], positions=[1.5], patch_artist=True, widths = 0.2, boxprops=boxprops2, flierprops=flierprops2,
                      meanprops=meanpointprops2, medianprops=medianprops, capprops = capprops2, whiskerprops = whiskerprops2, meanline=False, showmeans=True)

    

    ###########
    # monkeys #
    ###########

    euDists_data_dlpfc_1 = np.array(euDists_monkeys['dlpfc'][1])[:,l1*3:l1*3+3].mean(1)
    euDists_data_dlpfc_2 = np.array(euDists_monkeys['dlpfc'][2])[:,l1*3:l1*3+3].mean(1)
    euDists_data_fef_1 = np.array(euDists_monkeys['fef'][1])[:,l1*3:l1*3+3].mean(1)
    euDists_data_fef_2 = np.array(euDists_monkeys['fef'][2])[:,l1*3:l1*3+3].mean(1)
    
    euDistsRatio_data_dlpfc = euDists_data_dlpfc_2/euDists_data_dlpfc_1
    euDistsRatio_data_fef = euDists_data_fef_2/euDists_data_fef_1


    ax.boxplot([euDistsRatio_data_dlpfc], positions=[2.5], patch_artist=True, widths = 0.2, boxprops=boxprops3, flierprops=flierprops3,
                      meanprops=meanpointprops3, medianprops=medianprops, capprops = capprops3, whiskerprops = whiskerprops3, meanline=False, showmeans=True)
    ax.boxplot([euDistsRatio_data_fef], positions=[3.5], patch_artist=True, widths = 0.2, boxprops=boxprops4, flierprops=flierprops4,
                      meanprops=meanpointprops4, medianprops=medianprops, capprops = capprops4, whiskerprops = whiskerprops4, meanline=False, showmeans=True)


    #ylims = ax.get_ylim()
    #yscale = (ylims[1] - ylims[0])//(ylims[1]/2)

#    p1 = f_stats.permutation_p_diff(euDists_rnns_ed2_1.mean(1), euDists_rnns_ed2_2.mean(1))
#    ax.plot(lineh, np.full_like(lineh, ylims[1].round(2)+0.1), 'k-')
#    ax.plot(np.full_like(linev+ylims[1].round(2), 0.3), linev+ylims[1].round(2), 'k-')
#    ax.plot(np.full_like(linev+ylims[1].round(2), 0.7), linev+ylims[1].round(2), 'k-')
#    ax.text(0.5,ylims[1].round(2)+(yscale//5), f'{f_plotting.sig_marker(p1)}',horizontalalignment='center', fontsize=12)


#    p2 = f_stats.permutation_p_diff(euDists_rnns_ed12_1.mean(1), euDists_rnns_ed12_2.mean(1))
#    ax.plot(lineh+1, np.full_like(lineh, ylims[1].round(2)+0.1), 'k-')
#    ax.plot(np.full_like(linev+ylims[1].round(2), 1.3), linev+ylims[1].round(2), 'k-')
#    ax.plot(np.full_like(linev+ylims[1].round(2), 1.7), linev+ylims[1].round(2), 'k-')
#    ax.text(1.5,ylims[1].round(2)+(yscale//5), f'{f_plotting.sig_marker(p2)}',horizontalalignment='center', fontsize=12)


#    p3 = f_stats.permutation_p_diff(euDists_rnns_dlpfc_1.mean(1), euDists_rnns_dlpfc_2.mean(1)) # shape: nIters * ntrials

#    ax.plot(lineh+2, np.full_like(lineh, ylims[1].round(2)+0.1), 'k-')
#    ax.plot(np.full_like(linev+ylims[1].round(2), 2.3), linev+ylims[1].round(2), 'k-')
#    ax.plot(np.full_like(linev+ylims[1].round(2), 2.7), linev+ylims[1].round(2), 'k-')
#    ax.text(2.5,ylims[1].round(2)+(yscale//5), f'{f_plotting.sig_marker(p3)}',horizontalalignment='center', fontsize=12)


    #p2 = scipy.stats.ttest_rel(euDists['fef'][1].mean(1), euDists['fef'][2].mean(1))[-1]
    #p2,_,_ = f_stats.bootstrap95_p(euDists['fef'][1].mean(1), euDists['fef'][2].mean(1))
#    p4 = f_stats.permutation_p_diff(euDists_rnns_fef_1.mean(1), euDists_rnns_fef_2.mean(1))

#    ax.plot(lineh+3, np.full_like(lineh, ylims[1].round(2)+0.1), 'k-')
#    ax.plot(np.full_like(linev+ylims[1].round(2), 3.3), linev+ylims[1].round(2), 'k-')
#    ax.plot(np.full_like(linev+ylims[1].round(2), 3.7), linev+ylims[1].round(2), 'k-')
#    ax.text(3.5,ylims[1].round(2)+(yscale//5), f'{f_plotting.sig_marker(p4)}',horizontalalignment='center', fontsize=12)
    ax.set_xticks([0.5,1.5,2.5,3.5],['R@R','R&U','LPFC','FEF'], rotation=0)
    if l1 in (2,3,):
        
        ax.set_xlabel('Strategies/Regions', labelpad = 5, fontsize = 12)
    if l1 in (0,2):
        ax.set_ylabel('Distractor/Retarget', labelpad = 3, fontsize = 8)
    #ax.set_ylim(top=ylims[1].round(2)+(yscale/2))

    #plt.ylim(0.5,2.75)
    ax.set_title(f'Item1 Location = {l1+1}', fontsize = 10, pad=3)
    
    print(f'I1 Location = {l1+1}')
    ############
    # k-s test #
    ############
    ks_results = {'R@R-LPFC':scipy.stats.ks_2samp(euDistsRatio_rnns_ed2, euDistsRatio_data_dlpfc), 
                'R@R-FEF':scipy.stats.ks_2samp(euDistsRatio_rnns_ed2, euDistsRatio_data_fef), 
                'R&U-LPFC':scipy.stats.ks_2samp(euDistsRatio_rnns_ed12, euDistsRatio_data_dlpfc), 
                'R&U-FEF':scipy.stats.ks_2samp(euDistsRatio_rnns_ed12, euDistsRatio_data_fef)}
    
    print(f'K-S Test: {ks_results}')
    
    ############
    # CvM test #
    ############
    #cvm_results = {'R@R-LPFC':scipy.stats.cramervonmises_2samp(euDistsRatio_rnns_ed2, euDistsRatio_data_dlpfc), 
    #            'R@R-FEF':scipy.stats.cramervonmises_2samp(euDistsRatio_rnns_ed2, euDistsRatio_data_fef), 
    #            'R&U-LPFC':scipy.stats.cramervonmises_2samp(euDistsRatio_rnns_ed12, euDistsRatio_data_dlpfc), 
    #            'R&U-FEF':scipy.stats.cramervonmises_2samp(euDistsRatio_rnns_ed12, euDistsRatio_data_fef)}

    #print(f'CvM Test: {cvm_results}')
    
    #############################
    # Wasserstein Distance test #
    #############################
    #wd_results = {'R@R-LPFC':scipy.stats.wasserstein_distance(euDistsRatio_rnns_ed2, euDistsRatio_data_dlpfc), 
    #            'R@R-FEF':scipy.stats.wasserstein_distance(euDistsRatio_rnns_ed2, euDistsRatio_data_fef), 
    #            'R&U-LPFC':scipy.stats.wasserstein_distance(euDistsRatio_rnns_ed12, euDistsRatio_data_dlpfc), 
    #            'R&U-FEF':scipy.stats.wasserstein_distance(euDistsRatio_rnns_ed12, euDistsRatio_data_fef)}
    
    #print(f'Wasserstein Distance: {wd_results}')


plt.suptitle('Mean Projection Drift Ratio', fontsize = 20, y=1)

plt.show()


#fig.savefig(f'{phd_path}/outputs/driftDistRatio_centroids2_byloc.tif', bbox_inches='tight')
#%%
#%%

##################
# distance ratio #
##################

# In[]
lineh = np.arange(0.3,0.7,0.001)
linev = np.arange(0,0.1,0.001)
#linev = np.arange(1.15,1.25,0.001)
#linev = np.arange(2.47,2.5,0.001)

pltBsl = True

fig, axes = plt.subplots(1,1, figsize=(4,6),dpi=300, sharey=True)

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


ax = axes#.flatten()[l1]

########
# rnns #
########

euDists_rnns_ed2_1 = np.array(euDists_rnns['ed2'][1]).squeeze().mean(1)
euDists_rnns_ed2_2 = np.array(euDists_rnns['ed2'][2]).squeeze().mean(1)
euDists_rnns_ed12_1 = np.array(euDists_rnns['ed12'][1]).squeeze().mean(1)
euDists_rnns_ed12_2 = np.array(euDists_rnns['ed12'][2]).squeeze().mean(1)

euDists_rnns_ed2_1_shuff = np.array(euDists_rnns_shuff['ed2'][1]).squeeze().mean(1).mean(1)
euDists_rnns_ed2_2_shuff = np.array(euDists_rnns_shuff['ed2'][2]).squeeze().mean(1).mean(1)
euDists_rnns_ed12_1_shuff = np.array(euDists_rnns_shuff['ed12'][1]).squeeze().mean(1).mean(1)
euDists_rnns_ed12_2_shuff = np.array(euDists_rnns_shuff['ed12'][2]).squeeze().mean(1).mean(1)


euDistsRatio_rnns_ed2 = euDists_rnns_ed2_2/euDists_rnns_ed2_1
euDistsRatio_rnns_ed12 = euDists_rnns_ed12_2/euDists_rnns_ed12_1
euDistsRatio_rnns_ed2_shuff = euDists_rnns_ed2_2_shuff/euDists_rnns_ed2_1_shuff
euDistsRatio_rnns_ed12_shuff = euDists_rnns_ed12_2_shuff/euDists_rnns_ed12_1_shuff


ax.boxplot([euDistsRatio_rnns_ed2], positions=[0.5-pltBsl*0.2], patch_artist=True, widths = 0.2, boxprops=boxprops1, flierprops=flierprops1,
                    meanprops=meanpointprops1, medianprops=medianprops, capprops = capprops1, whiskerprops = whiskerprops1, meanline=False, showmeans=True)
ax.boxplot([euDistsRatio_rnns_ed12], positions=[1.5-pltBsl*0.2], patch_artist=True, widths = 0.2, boxprops=boxprops2, flierprops=flierprops2,
                    meanprops=meanpointprops2, medianprops=medianprops, capprops = capprops2, whiskerprops = whiskerprops2, meanline=False, showmeans=True)

if pltBsl:
    ax.boxplot([euDistsRatio_rnns_ed2_shuff], positions=[0.5+pltBsl*0.2], patch_artist=True, widths = 0.2, boxprops=boxprops0, flierprops=flierprops0,
                        meanprops=meanpointprops0, medianprops=medianprops, capprops = capprops0, whiskerprops = whiskerprops0, meanline=False, showmeans=True)
    ax.boxplot([euDistsRatio_rnns_ed12_shuff], positions=[1.5+pltBsl*0.2], patch_artist=True, widths = 0.2, boxprops=boxprops0, flierprops=flierprops0,
                        meanprops=meanpointprops0, medianprops=medianprops, capprops = capprops0, whiskerprops = whiskerprops0, meanline=False, showmeans=True)

p1 = f_stats.permutation_pCI(euDistsRatio_rnns_ed2, euDistsRatio_rnns_ed2_shuff, tail='smaller', alpha=5)
ax.text(0.5,1.3, f'{f_plotting.sig_marker(p1,ns_note=True)}',horizontalalignment='center', fontsize=12)
print(f'R@R: M(SD) = {euDistsRatio_rnns_ed2.mean():.3f}({euDistsRatio_rnns_ed2.std():.3f}), p = {p1:.3f}, 95CI = [{np.quantile(euDistsRatio_rnns_ed2,0.025):.3f}, {np.quantile(euDistsRatio_rnns_ed2,0.975):.3f}]')

p2 = f_stats.permutation_pCI(euDistsRatio_rnns_ed12, euDistsRatio_rnns_ed12_shuff, tail='smaller', alpha=5)
ax.text(1.5,1.3, f'{f_plotting.sig_marker(p2,ns_note=True)}',horizontalalignment='center', fontsize=12)
print(f'R&U: M(SD) = {euDistsRatio_rnns_ed12.mean():.3f}({euDistsRatio_rnns_ed12.std():.3f}), p = {p2:.3f}, 95CI = [{np.quantile(euDistsRatio_rnns_ed12,0.025):.3f}, {np.quantile(euDistsRatio_rnns_ed12,0.975):.3f}]')

###########
# monkeys #
###########

euDists_data_dlpfc_1 = np.array(euDists_monkeys['dlpfc'][1]).mean(1)
euDists_data_dlpfc_2 = np.array(euDists_monkeys['dlpfc'][2]).mean(1)
euDists_data_fef_1 = np.array(euDists_monkeys['fef'][1]).mean(1)
euDists_data_fef_2 = np.array(euDists_monkeys['fef'][2]).mean(1)

euDists_data_dlpfc_1_shuff = np.array(euDists_monkeys_shuff['dlpfc'][1]).mean(1).mean(1)
euDists_data_dlpfc_2_shuff = np.array(euDists_monkeys_shuff['dlpfc'][2]).mean(1).mean(1)
euDists_data_fef_1_shuff = np.array(euDists_monkeys_shuff['fef'][1]).mean(1).mean(1)
euDists_data_fef_2_shuff = np.array(euDists_monkeys_shuff['fef'][2]).mean(1).mean(1)

euDistsRatio_data_dlpfc = euDists_data_dlpfc_2/euDists_data_dlpfc_1
euDistsRatio_data_fef = euDists_data_fef_2/euDists_data_fef_1
euDistsRatio_data_dlpfc_shuff = euDists_data_dlpfc_2_shuff/euDists_data_dlpfc_1_shuff
euDistsRatio_data_fef_shuff = euDists_data_fef_2_shuff/euDists_data_fef_1_shuff


ax.boxplot([euDistsRatio_data_dlpfc], positions=[2.5-pltBsl*0.2], patch_artist=True, widths = 0.2, boxprops=boxprops3, flierprops=flierprops3,
                    meanprops=meanpointprops3, medianprops=medianprops, capprops = capprops3, whiskerprops = whiskerprops3, meanline=False, showmeans=True)
ax.boxplot([euDistsRatio_data_fef], positions=[3.5-pltBsl*0.2], patch_artist=True, widths = 0.2, boxprops=boxprops4, flierprops=flierprops4,
                    meanprops=meanpointprops4, medianprops=medianprops, capprops = capprops4, whiskerprops = whiskerprops4, meanline=False, showmeans=True)
if pltBsl:
    ax.boxplot([euDistsRatio_data_dlpfc_shuff], positions=[2.5+pltBsl*0.2], patch_artist=True, widths = 0.2, boxprops=boxprops0, flierprops=flierprops0,
                        meanprops=meanpointprops0, medianprops=medianprops, capprops = capprops0, whiskerprops = whiskerprops0, meanline=False, showmeans=True)
    ax.boxplot([euDistsRatio_data_fef_shuff], positions=[3.5+pltBsl*0.2], patch_artist=True, widths = 0.2, boxprops=boxprops0, flierprops=flierprops0,
                        meanprops=meanpointprops0, medianprops=medianprops, capprops = capprops0, whiskerprops = whiskerprops0, meanline=False, showmeans=True)

p3 = f_stats.permutation_pCI(euDistsRatio_data_dlpfc, euDistsRatio_data_dlpfc_shuff, tail='smaller', alpha=5)
ax.text(2.5,1.3, f'{f_plotting.sig_marker(p3,ns_note=True)}',horizontalalignment='center', fontsize=12)

p4 = f_stats.permutation_pCI(euDistsRatio_data_fef, euDistsRatio_data_fef_shuff, tail='smaller', alpha=5)
ax.text(3.5,1.3, f'{f_plotting.sig_marker(p4,ns_note=True)}',horizontalalignment='center', fontsize=12)

print(f'LPFC: M(SD) = {euDistsRatio_data_dlpfc.mean():.3f}({euDistsRatio_data_dlpfc.std():.3f}), p = {p3:.3f}, 95CI = [{np.quantile(euDistsRatio_data_dlpfc,0.025):.3f}, {np.quantile(euDistsRatio_data_dlpfc,0.975):.3f}]')

print(f'FEF: M(SD) = {euDistsRatio_data_fef.mean():.3f}({euDistsRatio_data_fef.std():.3f}), p = {p4:.3f}, 95CI = [{np.quantile(euDistsRatio_data_fef,0.025):.3f}, {np.quantile(euDistsRatio_data_fef,0.975):.3f}]')

#ylims = ax.get_ylim()
#yscale = (ylims[1] - ylims[0])//(ylims[1]/2)

#    p1 = f_stats.permutation_p_diff(euDists_rnns_ed2_1.mean(1), euDists_rnns_ed2_2.mean(1))
#    ax.plot(lineh, np.full_like(lineh, ylims[1].round(2)+0.1), 'k-')
#    ax.plot(np.full_like(linev+ylims[1].round(2), 0.3), linev+ylims[1].round(2), 'k-')
#    ax.plot(np.full_like(linev+ylims[1].round(2), 0.7), linev+ylims[1].round(2), 'k-')
#    ax.text(0.5,ylims[1].round(2)+(yscale//5), f'{f_plotting.sig_marker(p1)}',horizontalalignment='center', fontsize=12)


#    p2 = f_stats.permutation_p_diff(euDists_rnns_ed12_1.mean(1), euDists_rnns_ed12_2.mean(1))
#    ax.plot(lineh+1, np.full_like(lineh, ylims[1].round(2)+0.1), 'k-')
#    ax.plot(np.full_like(linev+ylims[1].round(2), 1.3), linev+ylims[1].round(2), 'k-')
#    ax.plot(np.full_like(linev+ylims[1].round(2), 1.7), linev+ylims[1].round(2), 'k-')
#    ax.text(1.5,ylims[1].round(2)+(yscale//5), f'{f_plotting.sig_marker(p2)}',horizontalalignment='center', fontsize=12)


#    p3 = f_stats.permutation_p_diff(euDists_rnns_dlpfc_1.mean(1), euDists_rnns_dlpfc_2.mean(1)) # shape: nIters * ntrials

#    ax.plot(lineh+2, np.full_like(lineh, ylims[1].round(2)+0.1), 'k-')
#    ax.plot(np.full_like(linev+ylims[1].round(2), 2.3), linev+ylims[1].round(2), 'k-')
#    ax.plot(np.full_like(linev+ylims[1].round(2), 2.7), linev+ylims[1].round(2), 'k-')
#    ax.text(2.5,ylims[1].round(2)+(yscale//5), f'{f_plotting.sig_marker(p3)}',horizontalalignment='center', fontsize=12)


#p2 = scipy.stats.ttest_rel(euDists['fef'][1].mean(1), euDists['fef'][2].mean(1))[-1]
#p2,_,_ = f_stats.bootstrap95_p(euDists['fef'][1].mean(1), euDists['fef'][2].mean(1))
#    p4 = f_stats.permutation_p_diff(euDists_rnns_fef_1.mean(1), euDists_rnns_fef_2.mean(1))

#    ax.plot(lineh+3, np.full_like(lineh, ylims[1].round(2)+0.1), 'k-')
#    ax.plot(np.full_like(linev+ylims[1].round(2), 3.3), linev+ylims[1].round(2), 'k-')
#    ax.plot(np.full_like(linev+ylims[1].round(2), 3.7), linev+ylims[1].round(2), 'k-')
#    ax.text(3.5,ylims[1].round(2)+(yscale//5), f'{f_plotting.sig_marker(p4)}',horizontalalignment='center', fontsize=12)
ax.set_xticks([0.5,1.5,2.5,3.5],['R@R','R&U','LPFC','FEF'], rotation=0)

ax.set_xlabel('Strategies/Regions', labelpad = 5, fontsize = 12)

ax.set_ylabel('Distractor/Retarget', labelpad = 3, fontsize = 10)
ax.set_ylim(top=1.4)

plt.suptitle('Mean Projection Drift Ratio', fontsize = 20, y=1)

plt.show()
#%%
fig.savefig(f'{phd_path}/outputs/driftDistRatio_centroids2_full_withbsl.tif', bbox_inches='tight')

#%%
rnn_region_pairs = list(product(('R@R','R&U'),('LPFC','FEF')))
############
# k-s test #
############
ks_results = {'R@R-LPFC':scipy.stats.ks_2samp(euDistsRatio_rnns_ed2, euDistsRatio_data_dlpfc), 
              'R@R-FEF':scipy.stats.ks_2samp(euDistsRatio_rnns_ed2, euDistsRatio_data_fef), 
              'R&U-LPFC':scipy.stats.ks_2samp(euDistsRatio_rnns_ed12, euDistsRatio_data_dlpfc), 
              'R&U-FEF':scipy.stats.ks_2samp(euDistsRatio_rnns_ed12, euDistsRatio_data_fef)}

for k in ks_results.keys():
    print(f'{k.upper()}: D = {ks_results[k][0]:.3f}, p = {ks_results[k][1]:.3f}')
#%%
############
# f-k test #
############
fk_results = {'R@R-LPFC':scipy.stats.fligner(euDistsRatio_rnns_ed2,euDistsRatio_data_dlpfc),
              'R@R-FEF':scipy.stats.fligner(euDistsRatio_rnns_ed2,euDistsRatio_data_fef),
              'R&U-LPFC':scipy.stats.fligner(euDistsRatio_rnns_ed12,euDistsRatio_data_dlpfc),
              'R&U-FEF':scipy.stats.fligner(euDistsRatio_rnns_ed12,euDistsRatio_data_fef)}


for k in fk_results.keys():
    print(f'{k.upper()}: H = {fk_results[k][0]:.3f}, p = {fk_results[k][1]:.3f}')
#%%
############
# CvM test #
############
cvm_results = {'R@R-LPFC':scipy.stats.cramervonmises_2samp(euDistsRatio_rnns_ed2, euDistsRatio_data_dlpfc), 
              'R@R-FEF':scipy.stats.cramervonmises_2samp(euDistsRatio_rnns_ed2, euDistsRatio_data_fef), 
              'R&U-LPFC':scipy.stats.cramervonmises_2samp(euDistsRatio_rnns_ed12, euDistsRatio_data_dlpfc), 
              'R&U-FEF':scipy.stats.cramervonmises_2samp(euDistsRatio_rnns_ed12, euDistsRatio_data_fef)}
print(f'CvM Test: {cvm_results}')
#############################
# Wasserstein Distance test #
#############################
wd_results = {'R@R-LPFC':scipy.stats.wasserstein_distance(euDistsRatio_rnns_ed2, euDistsRatio_data_dlpfc), 
              'R@R-FEF':scipy.stats.wasserstein_distance(euDistsRatio_rnns_ed2, euDistsRatio_data_fef), 
              'R&U-LPFC':scipy.stats.wasserstein_distance(euDistsRatio_rnns_ed12, euDistsRatio_data_dlpfc), 
              'R&U-FEF':scipy.stats.wasserstein_distance(euDistsRatio_rnns_ed12, euDistsRatio_data_fef)}
print(f'Wasserstein Distance: {wd_results}')
#%%
#%%

################################
# distance ratio mean hideLoc2 # under construction...
################################

# In[]
lineh = np.arange(0.3,0.7,0.001)
linev = np.arange(0,0.1,0.001)
#linev = np.arange(1.15,1.25,0.001)
#linev = np.arange(2.47,2.5,0.001)

fig, axes = plt.subplots(1,1, figsize=(10,6),dpi=300, sharey=True)

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




########
# rnns #
########

euDists_rnns_ed2_1 = np.array(euDists_rnns['ed2'][1]).squeeze().mean(1)
euDists_rnns_ed2_2 = np.array(euDists_rnns['ed2'][2]).squeeze().mean(1)
euDists_rnns_ed12_1 = np.array(euDists_rnns['ed12'][1]).squeeze().mean(1)
euDists_rnns_ed12_2 = np.array(euDists_rnns['ed12'][2]).squeeze().mean(1)

euDists_rnns_ed2_1_shuff = np.array(euDists_rnns_shuff['ed2'][1]).squeeze().mean(1).mean(1)
euDists_rnns_ed2_2_shuff = np.array(euDists_rnns_shuff['ed2'][2]).squeeze().mean(1).mean(1)
euDists_rnns_ed12_1_shuff = np.array(euDists_rnns_shuff['ed12'][1]).squeeze().mean(1).mean(1)
euDists_rnns_ed12_2_shuff = np.array(euDists_rnns_shuff['ed12'][2]).squeeze().mean(1).mean(1)


euDistsRatio_rnns_ed2 = euDists_rnns_ed2_2/euDists_rnns_ed2_1
euDistsRatio_rnns_ed12 = euDists_rnns_ed12_2/euDists_rnns_ed12_1
euDistsRatio_rnns_ed2_shuff = euDists_rnns_ed2_2_shuff/euDists_rnns_ed2_1_shuff
euDistsRatio_rnns_ed12_shuff = euDists_rnns_ed12_2_shuff/euDists_rnns_ed12_1_shuff


ax.boxplot([euDistsRatio_rnns_ed2], positions=[0.5-pltBsl*0.2], patch_artist=True, widths = 0.2, boxprops=boxprops1, flierprops=flierprops1,
                    meanprops=meanpointprops1, medianprops=medianprops, capprops = capprops1, whiskerprops = whiskerprops1, meanline=False, showmeans=True)
ax.boxplot([euDistsRatio_rnns_ed12], positions=[1.5-pltBsl*0.2], patch_artist=True, widths = 0.2, boxprops=boxprops2, flierprops=flierprops2,
                    meanprops=meanpointprops2, medianprops=medianprops, capprops = capprops2, whiskerprops = whiskerprops2, meanline=False, showmeans=True)

if pltBsl:
    ax.boxplot([euDistsRatio_rnns_ed2_shuff], positions=[0.5+pltBsl*0.2], patch_artist=True, widths = 0.2, boxprops=boxprops0, flierprops=flierprops0,
                        meanprops=meanpointprops0, medianprops=medianprops, capprops = capprops0, whiskerprops = whiskerprops0, meanline=False, showmeans=True)
    ax.boxplot([euDistsRatio_rnns_ed12_shuff], positions=[1.5+pltBsl*0.2], patch_artist=True, widths = 0.2, boxprops=boxprops0, flierprops=flierprops0,
                        meanprops=meanpointprops0, medianprops=medianprops, capprops = capprops0, whiskerprops = whiskerprops0, meanline=False, showmeans=True)

p1 = f_stats.permutation_pCI(euDistsRatio_rnns_ed2, euDistsRatio_rnns_ed2_shuff, tail='two', alpha=5)
#ax.text(0.5,1.1, f'{f_plotting.sig_marker(p1,ns_note=True)}',horizontalalignment='center', fontsize=12)
print(f'R@R: M(SD) = {euDistsRatio_rnns_ed2.mean():.3f}({euDistsRatio_rnns_ed2.std():.3f}), 95CI = [{np.quantile(euDistsRatio_rnns_ed2,0.025):.3f}, {np.quantile(euDistsRatio_rnns_ed2,0.975):.3f}]')

p2 = f_stats.permutation_pCI(euDistsRatio_rnns_ed12, euDistsRatio_rnns_ed12_shuff, tail='two', alpha=5)
#ax.text(1.5,1.1, f'{f_plotting.sig_marker(p2,ns_note=True)}',horizontalalignment='center', fontsize=12)
print(f'R&U: M(SD) = {euDistsRatio_rnns_ed12.mean():.3f}({euDistsRatio_rnns_ed12.std():.3f}), 95CI = [{np.quantile(euDistsRatio_rnns_ed12,0.025):.3f}, {np.quantile(euDistsRatio_rnns_ed12,0.975):.3f}]')

###########
# monkeys #
###########

euDists_data_dlpfc_1 = np.array(euDists_monkeys['dlpfc'][1]).mean(1)
euDists_data_dlpfc_2 = np.array(euDists_monkeys['dlpfc'][2]).mean(1)
euDists_data_fef_1 = np.array(euDists_monkeys['fef'][1]).mean(1)
euDists_data_fef_2 = np.array(euDists_monkeys['fef'][2]).mean(1)

euDistsRatio_data_dlpfc = euDists_data_dlpfc_2/euDists_data_dlpfc_1
euDistsRatio_data_fef = euDists_data_fef_2/euDists_data_fef_1


ax.boxplot([euDistsRatio_data_dlpfc], positions=[2.5], patch_artist=True, widths = 0.2, boxprops=boxprops3, flierprops=flierprops3,
                    meanprops=meanpointprops3, medianprops=medianprops, capprops = capprops3, whiskerprops = whiskerprops3, meanline=False, showmeans=True)
ax.boxplot([euDistsRatio_data_fef], positions=[3.5], patch_artist=True, widths = 0.2, boxprops=boxprops4, flierprops=flierprops4,
                    meanprops=meanpointprops4, medianprops=medianprops, capprops = capprops4, whiskerprops = whiskerprops4, meanline=False, showmeans=True)

print(f'LPFC: Mean(STDV) = {euDistsRatio_data_dlpfc.mean():.3f}({euDistsRatio_data_dlpfc.std():.3f}), 95CI = [{np.quantile(euDistsRatio_data_dlpfc,0.025):.3f}, {np.quantile(euDistsRatio_data_dlpfc,0.975):.3f}]')

print(f'FEF: Mean(STDV) = {euDistsRatio_data_fef.mean():.3f}({euDistsRatio_data_fef.std():.3f}), 95CI = [{np.quantile(euDistsRatio_data_fef,0.025):.3f}, {np.quantile(euDistsRatio_data_fef,0.975):.3f}]')


ax.set_xticks([0.5,1.5,2.5,3.5],['R@R','R&U','LPFC','FEF'], rotation=0)

ax.set_xlabel('Strategies/Regions', labelpad = 5, fontsize = 12)

ax.set_ylabel('Distractor/Retarget', labelpad = 3, fontsize = 10)
#ax.set_ylim(top=ylims[1].round(2)+(yscale/2))

plt.suptitle('Mean Projection Drift Ratio', fontsize = 20, y=1)

plt.show()
fig.savefig(f'{phd_path}/outputs/driftDistRatio_centroids2.tif', bbox_inches='tight')



#%%








































#%%
##########
# unused #
##########
# In[] 
from sklearn.neighbors import KernelDensity

# Kernel Density Estimation (KDE)
kde_empirical = KernelDensity(kernel='gaussian', bandwidth=0.1).fit(euDistRatio_monkeys['dlpfc'][:, None])
kde_A = KernelDensity(kernel='gaussian', bandwidth=0.1).fit(euDistRatio_rnns['ed2'][:, None])
kde_B = KernelDensity(kernel='gaussian', bandwidth=0.1).fit(euDistRatio_rnns['ed12'][:, None])

# Score the log-likelihood of the empirical data under each model's KDE
log_likelihood_A_kde = kde_A.score(euDistRatio_monkeys['fef'][:, None])
log_likelihood_B_kde = kde_B.score(euDistRatio_monkeys['fef'][:, None])

print(f"Log-Likelihood under KDE for Model A: {log_likelihood_A_kde}")
print(f"Log-Likelihood under KDE for Model B: {log_likelihood_B_kde}")



wasserstein_A = stats.wasserstein_distance(euDistRatio_monkeys['dlpfc'], euDistRatio_rnns['ed2'])
wasserstein_B = stats.wasserstein_distance(euDistRatio_monkeys['dlpfc'], euDistRatio_rnns['ed12'])

ad_result_A = stats.anderson_ksamp([euDistRatio_monkeys['dlpfc'], euDistRatio_rnns['ed2']])
ad_result_B = stats.anderson_ksamp([euDistRatio_monkeys['dlpfc'], euDistRatio_rnns['ed12']])


from statsmodels.distributions.empirical_distribution import ECDF

def cramer_von_mises(empirical_data, model_data):
    ecdf_empirical = ECDF(empirical_data)
    ecdf_model = ECDF(model_data)
    x = np.sort(np.concatenate([empirical_data, model_data]))
    diff_ecdf = ecdf_empirical(x) - ecdf_model(x)
    cvm_statistic = np.sum(diff_ecdf ** 2) * len(empirical_data) * len(model_data) / (len(empirical_data) + len(model_data)) ** 2
    return cvm_statistic

cvm_A = cramer_von_mises(euDistRatio_monkeys['dlpfc'], euDistRatio_rnns['ed2'])
cvm_B = cramer_von_mises(euDistRatio_monkeys['dlpfc'], euDistRatio_rnns['ed12'])

#fig.savefig(f'{save_path}/driftDistRatio_centroids.tif', bbox_inches='tight')
#%%



















































# In[]
lineh = np.arange(0.3,0.7,0.001)
linev = np.arange(0,0.1,0.001)
#linev = np.arange(1.15,1.25,0.001)
#linev = np.arange(2.47,2.5,0.001)

fig, axes = plt.subplots(1,2, figsize=(6,3),dpi=300, sharey=True)


########
# rnns #
########
ax = axes.flatten()[0]

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


#fig = plt.figure(figsize=(3, 3), dpi=100)

ax.boxplot([euDists_rnns['ed2'][1].mean(1)], positions=[0.3], patch_artist=True, widths = 0.2, boxprops=boxprops1, flierprops=flierprops1, 
                  meanprops=meanpointprops1, medianprops=medianprops, capprops = capprops1, whiskerprops = whiskerprops1, meanline=False, showmeans=True)
ax.boxplot([euDists_rnns['ed12'][1].mean(1)], positions=[1.3], patch_artist=True, widths = 0.2, boxprops=boxprops2, flierprops=flierprops2, 
                  meanprops=meanpointprops2, medianprops=medianprops, capprops = capprops2, whiskerprops = whiskerprops2, meanline=False, showmeans=True)

ax.boxplot([euDists_rnns['ed2'][2].mean(1)], positions=[0.7], patch_artist=True, widths = 0.2, boxprops=dict(boxprops1,facecolor=color1_), flierprops=dict(flierprops1,markeredgecolor=color1_, markerfacecolor=color1_), 
                  meanprops=dict(meanpointprops1,markeredgecolor=color1_), medianprops=medianprops, capprops = dict(capprops1,color=color1_), whiskerprops = dict(whiskerprops1,color=color1_), meanline=False, showmeans=True)
ax.boxplot([euDists_rnns['ed12'][2].mean(1)], positions=[1.7], patch_artist=True, widths = 0.2, boxprops=dict(boxprops2,facecolor=color2_), flierprops=dict(flierprops2,markeredgecolor=color2_, markerfacecolor=color2_), 
                  meanprops=dict(meanpointprops2,markeredgecolor=color2_), medianprops=medianprops, capprops = dict(capprops2,color=color2_), whiskerprops = dict(whiskerprops2,color=color2_), meanline=False, showmeans=True)

ylims = ax.get_ylim()
yscale = (ylims[1] - ylims[0])//(ylims[1]//2)

#p1 = scipy.stats.ttest_ind(euDists['ed2'][1], euDists['ed2'][2])[-1]
p1 = f_stats.permutation_p_diff(euDists_rnns['ed2'][1].mean(1), euDists_rnns['ed2'][2].mean(1))
ax.plot(lineh, np.full_like(lineh, ylims[1].round(2)+0.1), 'k-')
#ax.plot(lineh, np.full_like(lineh, 1.25), 'k-')
ax.plot(np.full_like(linev+ylims[1].round(2), 0.3), linev+ylims[1].round(2), 'k-')
ax.plot(np.full_like(linev+ylims[1].round(2), 0.7), linev+ylims[1].round(2), 'k-')
ax.text(0.5,ylims[1].round(2)+(yscale//5), f'{f_plotting.sig_marker(p1)}',horizontalalignment='center', fontsize=12)
#ax.text(0.5,1.25, f'{f_plotting.sig_marker(p1)}',horizontalalignment='center', fontsize=12)


#p2 = scipy.stats.ttest_ind(euDists['ed12'][1], euDists['ed12'][2])[-1]
p2 = f_stats.permutation_p_diff(euDists_rnns['ed12'][1].mean(1), euDists_rnns['ed12'][2].mean(1))
ax.plot(lineh+1, np.full_like(lineh, ylims[1].round(2)+0.1), 'k-')
#ax.plot(lineh+1, np.full_like(lineh, 1.25), 'k-')
ax.plot(np.full_like(linev+ylims[1].round(2), 1.3), linev+ylims[1].round(2), 'k-')
ax.plot(np.full_like(linev+ylims[1].round(2), 1.7), linev+ylims[1].round(2), 'k-')
ax.text(1.5,ylims[1].round(2)+(yscale//5), f'{f_plotting.sig_marker(p2)}',horizontalalignment='center', fontsize=12)
#ax.text(1.5,1.25, f'{f_plotting.sig_marker(p2)}',horizontalalignment='center', fontsize=12)


# draw temporary red and blue lines and use them to create a legend
ax.plot([], c='dimgrey', label='Retarget')
ax.plot([], c='lightgrey', label='Distraction')
#ax.legend(bbox_to_anchor=(1.65, 0.5))#loc = 'right'

ax.set_xticks([0.5,1.5],['R@R','R&U'], rotation=30) #
ax.set_xlabel('Strategy', labelpad = 5, fontsize = 12)
ax.set_ylabel('Euclidean Distance (Std. Scaled)', labelpad = 3, fontsize = 10)
#ax.set_ylabel('z-scored Euclidean Distance', labelpad = 3, fontsize = 10)
ax.set_ylim(top=ylims[1].round(2)+(yscale//2))
#ax.set_ylim(top=1.5)
ax.set_title('Models', fontsize = 15, pad=10)

###########
# monkeys #
###########

ax = axes.flatten()[1]

color1, color1_, color2, color2_ = '#185337', '#96d9ad', '#804098', '#c4a2d1'


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
#fig = plt.figure(figsize=(3, 3), dpi=300)

ax.boxplot([euDists_monkeys['dlpfc'][1].mean(1)], positions=[0.3], patch_artist=True, widths = 0.2, boxprops=boxprops1, flierprops=flierprops1, 
                  meanprops=meanpointprops1, medianprops=medianprops, capprops = capprops1, whiskerprops = whiskerprops1, meanline=False, showmeans=True)
ax.boxplot([euDists_monkeys['fef'][1].mean(1)], positions=[1.3], patch_artist=True, widths = 0.2, boxprops=boxprops2, flierprops=flierprops2, 
                  meanprops=meanpointprops2, medianprops=medianprops, capprops = capprops2, whiskerprops = whiskerprops2, meanline=False, showmeans=True)

ax.boxplot([euDists_monkeys['dlpfc'][2].mean(1)], positions=[0.7], patch_artist=True, widths = 0.2, boxprops=dict(boxprops1,facecolor=color1_), flierprops=dict(flierprops1,markeredgecolor=color1_, markerfacecolor=color1_), 
                  meanprops=dict(meanpointprops1,markeredgecolor=color1_), medianprops=medianprops, capprops = dict(capprops1,color=color1_), whiskerprops = dict(whiskerprops1,color=color1_), meanline=False, showmeans=True)
ax.boxplot([euDists_monkeys['fef'][2].mean(1)], positions=[1.7], patch_artist=True, widths = 0.2, boxprops=dict(boxprops2,facecolor=color2_), flierprops=dict(flierprops2,markeredgecolor=color2_, markerfacecolor=color2_), 
                  meanprops=dict(meanpointprops2,markeredgecolor=color2_), medianprops=medianprops, capprops = dict(capprops2,color=color2_), whiskerprops = dict(whiskerprops2,color=color2_), meanline=False, showmeans=True)

#p1 = scipy.stats.ttest_rel(euDists['dlpfc'][1].mean(1), euDists['dlpfc'][2].mean(1))[-1]
#p1,_,_ = f_stats.bootstrap95_p(euDists['dlpfc'][1].mean(1), euDists['dlpfc'][2].mean(1)) # shape: nIters * ntrials
p1 = f_stats.permutation_p_diff(euDists_monkeys['dlpfc'][1].mean(1), euDists_monkeys['dlpfc'][2].mean(1)) # shape: nIters * ntrials

ax.plot(lineh, np.full_like(lineh, ylims[1].round(2)+0.1), 'k-')
#ax.plot(lineh, np.full_like(lineh, 1.25), 'k-')
ax.plot(np.full_like(linev+ylims[1].round(2), 0.3), linev+ylims[1].round(2), 'k-')
ax.plot(np.full_like(linev+ylims[1].round(2), 0.7), linev+ylims[1].round(2), 'k-')
ax.text(0.5,ylims[1].round(2)+(yscale//5), f'{f_plotting.sig_marker(p1)}',horizontalalignment='center', fontsize=12)
#plt.text(0.5,1.25, f'{f_plotting.sig_marker(p1)}',horizontalalignment='center', fontsize=12)


#p2 = scipy.stats.ttest_rel(euDists['fef'][1].mean(1), euDists['fef'][2].mean(1))[-1]
#p2,_,_ = f_stats.bootstrap95_p(euDists['fef'][1].mean(1), euDists['fef'][2].mean(1))
p2 = f_stats.permutation_p_diff(euDists_monkeys['fef'][1].mean(1), euDists_monkeys['fef'][2].mean(1))

ax.plot(lineh+1, np.full_like(lineh, ylims[1].round(2)+0.1), 'k-')
#ax.plot(lineh+1, np.full_like(lineh, 1.25), 'k-')
ax.plot(np.full_like(linev+ylims[1].round(2), 1.3), linev+ylims[1].round(2), 'k-')
ax.plot(np.full_like(linev+ylims[1].round(2), 1.7), linev+ylims[1].round(2), 'k-')
ax.text(1.5,ylims[1].round(2)+(yscale//5), f'{f_plotting.sig_marker(p2)}',horizontalalignment='center', fontsize=12)
#ax.text(1.5,1.25, f'{f_plotting.sig_marker(p2)}',horizontalalignment='center', fontsize=12)

# draw temporary red and blue lines and use them to create a legend
ax.plot([], c='dimgrey', label='Retarget')
ax.plot([], c='lightgrey', label='Distraction')
ax.legend(bbox_to_anchor=(1.7, 0.6), fontsize = 10)#loc = 'right',

ax.set_xticks([0.5,1.5],['LPFC','FEF'], rotation=30)
ax.set_xlabel('Regions', labelpad = 5, fontsize = 12)
#ax.set_ylabel('Euclidean Distance (Std. Scaled)', labelpad = 3, fontsize = 10)
#plt.ylim(0.5,2.75)
ax.set_title('Monkeys', fontsize = 15, pad=10)

plt.suptitle('Mean Projection Drift', fontsize = 20, y=1.1)

plt.show()

        
fig.savefig(f'{save_path}/driftDist_centroids.tif', bbox_inches='tight')

# In[] ratio

euDistRatio_rnns = {k:euDists_rnns[k][2].mean(1)/euDists_rnns[k][1].mean(1) for k in euDists_rnns.keys()}
euDistRatio_monkeys = {k:euDists_monkeys[k][2].mean(1)/euDists_monkeys[k][1].mean(1) for k in euDists_monkeys.keys()}

#euDistRatio_rnns = {k:(euDists_rnns[k][1].mean(1)-euDists_rnns[k][2].mean(1))/euDists_rnns[k][2].mean(1) for k in euDists_rnns.keys()}
#euDistRatio_monkeys = {k:(euDists_monkeys[k][1].mean(1)-euDists_monkeys[k][2].mean(1))/euDists_monkeys[k][2].mean(1) for k in euDists_monkeys.keys()}

# baseline
#euDistRatio_rnns_shuff = {}
#euDistRatio_monkeys_shuff = {}

#for k in euDistRatio_rnns.keys():
#    dis1_shuff, dis2_shuff = f_stats.shuff_label(euDists_rnns[k][2], euDists_rnns[k][1])
#    euDistRatio_rnns_shuff[k] = (dis1_shuff/dis2_shuff).mean(-1)#np.concatenate(dis1_shuff/dis2_shuff)

#for k in euDistRatio_monkeys.keys():
#    dis1_shuff, dis2_shuff = f_stats.shuff_label(euDists_monkeys[k][2], euDists_monkeys[k][1])
#    euDistRatio_monkeys_shuff[k] = (dis1_shuff/dis2_shuff).mean(-1)#np.concatenate(dis1_shuff/dis2_shuff)


lineh = np.arange(0.3,0.7,0.001)
linev = np.arange(6.15,6.25,0.001)
#linev = np.arange(1.15,1.25,0.001)
#linev = np.arange(2.47,2.5,0.001)

fig, axes = plt.subplots(1,1, figsize=(3,3),dpi=300, sharey=True)


########
# rnns #
########
ax = axes#.flatten()[0]

color1, color2 = '#d29c2f', '#3c79b4'
color3, color4 = '#185337', '#804098'


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


#fig = plt.figure(figsize=(3, 3), dpi=100)

ax.boxplot([euDistRatio_rnns['ed2']], positions=[0.5], patch_artist=True, widths = 0.2, boxprops=boxprops1, flierprops=flierprops1, 
                  meanprops=meanpointprops1, medianprops=medianprops, capprops = capprops1, whiskerprops = whiskerprops1, meanline=False, showmeans=True)
ax.boxplot([euDistRatio_rnns['ed12']], positions=[1.5], patch_artist=True, widths = 0.2, boxprops=boxprops2, flierprops=flierprops2, 
                  meanprops=meanpointprops2, medianprops=medianprops, capprops = capprops2, whiskerprops = whiskerprops2, meanline=False, showmeans=True)


ax.boxplot([euDistRatio_monkeys['dlpfc']], positions=[2.5], patch_artist=True, widths = 0.2, boxprops=boxprops3, flierprops=flierprops3, 
                  meanprops=meanpointprops3, medianprops=medianprops, capprops = capprops3, whiskerprops = whiskerprops3, meanline=False, showmeans=True)
ax.boxplot([euDistRatio_monkeys['fef']], positions=[3.5], patch_artist=True, widths = 0.2, boxprops=boxprops4, flierprops=flierprops4, 
                  meanprops=meanpointprops4, medianprops=medianprops, capprops = capprops4, whiskerprops = whiskerprops4, meanline=False, showmeans=True)



# draw temporary red and blue lines and use them to create a legend
#ax.plot([], c='dimgrey', label='Retarget')
#ax.plot([], c='lightgrey', label='Distraction')
#ax.legend(bbox_to_anchor=(1.0, 0.6), fontsize = 10)#loc = 'right',


ax.set_xticks([0.5,1.5,2.5,3.5],['R@R','R&U','LPFC','FEF'], rotation=30) #
ax.set_xlabel('Strategy & Region', labelpad = 5, fontsize = 12)
ax.set_ylabel('Distraction / Retarget', labelpad = 3, fontsize = 10)
#ax.set_ylim(top=7)
#ax.set_ylim(top=1.5)
ax.set_title('Mean Projection Drift Ratio', fontsize = 15, pad=10)
#plt.suptitle('Mean Projection Drift Ratio', fontsize = 20, y=1.1)

plt.show()
        
fig.savefig(f'{save_path}/driftDistRatio_centroids.tif', bbox_inches='tight')


































# In[] decodability with/without permutation P value

euDists_monkeys = np.load(f'{phd_path}/data/pseudo_ww/euDists_monkeys_z.npy', allow_pickle=True).item()
euDists_rnns = np.load(f'{phd_path}/fitting planes/pooled/euDists_rnns_z.npy', allow_pickle=True).item()
# In[]
lineh = np.arange(0.3,0.7,0.001)
#linev = np.arange(6.15,6.25,0.001)
linev = np.arange(1.15,1.25,0.001)
#linev = np.arange(2.47,2.5,0.001)

fig, axes = plt.subplots(1,2, figsize=(6,3),dpi=300, sharey=True)


########
# rnns #
########
ax = axes.flatten()[0]

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


#fig = plt.figure(figsize=(3, 3), dpi=100)

ax.boxplot([euDists_rnns['ed2'][1]], positions=[0.3], patch_artist=True, widths = 0.2, boxprops=boxprops1, flierprops=flierprops1, 
                  meanprops=meanpointprops1, medianprops=medianprops, capprops = capprops1, whiskerprops = whiskerprops1, meanline=False, showmeans=True)
ax.boxplot([euDists_rnns['ed12'][1]], positions=[1.3], patch_artist=True, widths = 0.2, boxprops=boxprops2, flierprops=flierprops2, 
                  meanprops=meanpointprops2, medianprops=medianprops, capprops = capprops2, whiskerprops = whiskerprops2, meanline=False, showmeans=True)

ax.boxplot([euDists_rnns['ed2'][2]], positions=[0.7], patch_artist=True, widths = 0.2, boxprops=dict(boxprops1,facecolor=color1_), flierprops=dict(flierprops1,markeredgecolor=color1_, markerfacecolor=color1_), 
                  meanprops=dict(meanpointprops1,markeredgecolor=color1_), medianprops=medianprops, capprops = dict(capprops1,color=color1_), whiskerprops = dict(whiskerprops1,color=color1_), meanline=False, showmeans=True)
ax.boxplot([euDists_rnns['ed12'][2]], positions=[1.7], patch_artist=True, widths = 0.2, boxprops=dict(boxprops2,facecolor=color2_), flierprops=dict(flierprops2,markeredgecolor=color2_, markerfacecolor=color2_), 
                  meanprops=dict(meanpointprops2,markeredgecolor=color2_), medianprops=medianprops, capprops = dict(capprops2,color=color2_), whiskerprops = dict(whiskerprops2,color=color2_), meanline=False, showmeans=True)


#p1 = scipy.stats.ttest_ind(euDists['ed2'][1], euDists['ed2'][2])[-1]
p1 = f_stats.permutation_p_diff(euDists_rnns['ed2'][1], euDists_rnns['ed2'][2])
#ax.plot(lineh, np.full_like(lineh, 6.25), 'k-')
ax.plot(lineh, np.full_like(lineh, 1.25), 'k-')
ax.plot(np.full_like(linev, 0.3), linev, 'k-')
ax.plot(np.full_like(linev, 0.7), linev, 'k-')
#ax.text(0.5,6.25, f'{f_plotting.sig_marker(p1)}',horizontalalignment='center', fontsize=12)
ax.text(0.5,1.25, f'{f_plotting.sig_marker(p1)}',horizontalalignment='center', fontsize=12)


#p2 = scipy.stats.ttest_ind(euDists['ed12'][1], euDists['ed12'][2])[-1]
p2 = f_stats.permutation_p_diff(euDists_rnns['ed12'][1], euDists_rnns['ed12'][2])
#ax.plot(lineh+1, np.full_like(lineh, 6.25), 'k-')
ax.plot(lineh+1, np.full_like(lineh, 1.25), 'k-')
ax.plot(np.full_like(linev, 1.3), linev, 'k-')
ax.plot(np.full_like(linev, 1.7), linev, 'k-')
#ax.text(1.5,6.25, f'{f_plotting.sig_marker(p2)}',horizontalalignment='center', fontsize=12)
ax.text(1.5,1.25, f'{f_plotting.sig_marker(p2)}',horizontalalignment='center', fontsize=12)


# draw temporary red and blue lines and use them to create a legend
ax.plot([], c='dimgrey', label='Retarget')
ax.plot([], c='lightgrey', label='Distraction')
#ax.legend(bbox_to_anchor=(1.65, 0.5))#loc = 'right'

ax.set_xticks([0.5,1.5],['R@R','R&U'], rotation=30) #
ax.set_xlabel('Strategy', labelpad = 5, fontsize = 12)
#ax.set_ylabel('Euclidean Distance (Std. Scaled)', labelpad = 3, fontsize = 10)
ax.set_ylabel('z-scored Euclidean Distance', labelpad = 3, fontsize = 10)
#ax.set_ylim(top=7)
ax.set_ylim(top=1.5)
ax.set_title('Models', fontsize = 15, pad=10)

###########
# monkeys #
###########

ax = axes.flatten()[1]

color1, color1_, color2, color2_ = '#185337', '#96d9ad', '#804098', '#c4a2d1'


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
#fig = plt.figure(figsize=(3, 3), dpi=300)

ax.boxplot([euDists_monkeys['dlpfc'][1].mean(1)], positions=[0.3], patch_artist=True, widths = 0.2, boxprops=boxprops1, flierprops=flierprops1, 
                  meanprops=meanpointprops1, medianprops=medianprops, capprops = capprops1, whiskerprops = whiskerprops1, meanline=False, showmeans=True)
ax.boxplot([euDists_monkeys['fef'][1].mean(1)], positions=[1.3], patch_artist=True, widths = 0.2, boxprops=boxprops2, flierprops=flierprops2, 
                  meanprops=meanpointprops2, medianprops=medianprops, capprops = capprops2, whiskerprops = whiskerprops2, meanline=False, showmeans=True)

ax.boxplot([euDists_monkeys['dlpfc'][2].mean(1)], positions=[0.7], patch_artist=True, widths = 0.2, boxprops=dict(boxprops1,facecolor=color1_), flierprops=dict(flierprops1,markeredgecolor=color1_, markerfacecolor=color1_), 
                  meanprops=dict(meanpointprops1,markeredgecolor=color1_), medianprops=medianprops, capprops = dict(capprops1,color=color1_), whiskerprops = dict(whiskerprops1,color=color1_), meanline=False, showmeans=True)
ax.boxplot([euDists_monkeys['fef'][2].mean(1)], positions=[1.7], patch_artist=True, widths = 0.2, boxprops=dict(boxprops2,facecolor=color2_), flierprops=dict(flierprops2,markeredgecolor=color2_, markerfacecolor=color2_), 
                  meanprops=dict(meanpointprops2,markeredgecolor=color2_), medianprops=medianprops, capprops = dict(capprops2,color=color2_), whiskerprops = dict(whiskerprops2,color=color2_), meanline=False, showmeans=True)

#p1 = scipy.stats.ttest_rel(euDists['dlpfc'][1].mean(1), euDists['dlpfc'][2].mean(1))[-1]
#p1,_,_ = f_stats.bootstrap95_p(euDists['dlpfc'][1].mean(1), euDists['dlpfc'][2].mean(1)) # shape: nIters * ntrials
p1 = f_stats.permutation_p_diff(euDists_monkeys['dlpfc'][1].mean(1), euDists_monkeys['dlpfc'][2].mean(1)) # shape: nIters * ntrials

#ax.plot(lineh, np.full_like(lineh, 6.25), 'k-')
ax.plot(lineh, np.full_like(lineh, 1.25), 'k-')
ax.plot(np.full_like(linev, 0.3), linev, 'k-')
ax.plot(np.full_like(linev, 0.7), linev, 'k-')
#ax.text(0.5,6.25, f'{f_plotting.sig_marker(p1)}',horizontalalignment='center', fontsize=12)
ax.text(0.5,1.25, f'{f_plotting.sig_marker(p1)}',horizontalalignment='center', fontsize=12)


#p2 = scipy.stats.ttest_rel(euDists['fef'][1].mean(1), euDists['fef'][2].mean(1))[-1]
#p2,_,_ = f_stats.bootstrap95_p(euDists['fef'][1].mean(1), euDists['fef'][2].mean(1))
p2 = f_stats.permutation_p_diff(euDists_monkeys['fef'][1].mean(1), euDists_monkeys['fef'][2].mean(1))

#ax.plot(lineh+1, np.full_like(lineh, 6.25), 'k-')
ax.plot(lineh+1, np.full_like(lineh, 1.25), 'k-')
ax.plot(np.full_like(linev, 1.3), linev, 'k-')
ax.plot(np.full_like(linev, 1.7), linev, 'k-')
#ax.text(1.5,6.25, f'{f_plotting.sig_marker(p2)}',horizontalalignment='center', fontsize=12)
ax.text(1.5,1.25, f'{f_plotting.sig_marker(p2)}',horizontalalignment='center', fontsize=12)

# draw temporary red and blue lines and use them to create a legend
ax.plot([], c='dimgrey', label='Retarget')
ax.plot([], c='lightgrey', label='Distraction')
ax.legend(bbox_to_anchor=(1.7, 0.6), fontsize = 10)#loc = 'right',

ax.set_xticks([0.5,1.5],['LPFC','FEF'], rotation=30)
ax.set_xlabel('Regions', labelpad = 5, fontsize = 12)
#ax.set_ylabel('Euclidean Distance (Std. Scaled)', labelpad = 3, fontsize = 10)
#plt.ylim(0.5,2.75)
ax.set_title('Monkeys', fontsize = 15, pad=10)

plt.suptitle('Mean Projection Drift', fontsize = 20, y=1.1)

plt.show()

        
fig.savefig(f'{save_path}/driftDist_monkeys_rnns_z.tif', bbox_inches='tight')


# %%
# eudist R-D difference
euDists_monkeysD = {k:(euDists_monkeys[k][1].mean(1)-euDists_monkeys[k][2].mean(1)) for k in euDists_monkeys.keys()}
euDists_rnnsD = {k:(euDists_rnns[k][1]-euDists_rnns[k][2]) for k in euDists_rnns.keys()}

plt.boxplot(euDists_monkeysD['dlpfc'],positions=[0.5])
plt.boxplot(euDists_monkeysD['fef'],positions=[1.5])
plt.boxplot(euDists_rnnsD['ed2'],positions=[2.5])
plt.boxplot(euDists_rnnsD['ed12'],positions=[3.5])
plt.show()

euDists_monkeysP = {k:np.concatenate([euDists_monkeys[k][1].mean(1),euDists_monkeys[k][2].mean(1)]) for k in euDists_monkeys.keys()}
euDists_rnnsP = {k:np.concatenate([euDists_rnns[k][1],euDists_rnns[k][2]]) for k in euDists_rnns.keys()}

plt.violinplot(euDists_monkeysP['dlpfc'],positions=[0.5])
plt.violinplot(euDists_monkeysP['fef'],positions=[1.5])
plt.violinplot(euDists_rnnsP['ed2'],positions=[2.5])
plt.violinplot(euDists_rnnsP['ed12'],positions=[3.5])
plt.show()











# %%
