# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 11:13:52 2024

@author: aka2333
"""
# In[]
%reload_ext autoreload
%autoreload 2

# In[ ]:

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
from scipy.ndimage import gaussian_filter

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
import f_saccade
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

step = 50
dt = 10 # ms, sampling rate down to 1000/dt Hz
# if step = dt, non-overlap sampling

# In[] whiskey Info
# levels: monkey 1, day 8, session 1, array 4, channel x, cell x
whis = Get_paths('whiskey', data_path)

whis_monkey_path = whis.monkey_path()
whis_session_paths = whis.session_path()
whis_cell_paths = whis.cell_path()

# In[] manual inspection and exclude cells with insufficient# of valid trials
whis_exclusionList = ['20200106/session02/array01/channel017/cell01', '20200106/session02/array03/channel066/cell01', '20200106/session02/array04/channel121/cell01', '20200121/session01/array02/channel062/cell03']
for ex in whis_exclusionList:    
    if (whis_monkey_path + '/' + ex) in whis_cell_paths:
        whis_cell_paths.remove(whis_monkey_path + '/' + ex)
    
# In[] drop session with small trial size
whis_session_drop = []#'f'{data_path}/whiskey/20200108/session03','f'{data_path}/whiskey/20200109/session04',
for i in whis_session_drop:
    if i in whis_session_paths:
        whis_session_paths.remove(i)
    
# In[] whiskey Info

whis_epochsDic = {'bsl':[-300,0],'s1':[0,300],'d1':[300,1300],'s2':[1300,1600],'d2':[1600,2600],'go':[2600,3000]}#, 'p1':[0,1300],'p2':[1300,2600]

whis_trialEvtsBoundary = sorted([b[1] for b in whis_epochsDic.values()])

whis_tRange = np.arange(-300, whis_trialEvtsBoundary[-1], dt)
whis_tBsl = whis_epochsDic['bsl']
whis_tslices = ((-300,0), (0,300), (300, 1300), (1300, 1600), (1600, 2600), (2600,3000))

whis_dlpfcArrays = (1,3)
whis_fefArrays = (2,4)


# In[] wangzi Info
# levels: monkey 1, day 8, session 1, array 4, channel x, cell x
wz = Get_paths('wangzi', data_path)

wz_monkey_path = wz.monkey_path()
wz_session_paths = wz.session_path()
wz_cell_paths = wz.cell_path()

# In[] manual inspection and exclude cells with insufficient# of valid trials
wz_exclusionList = ['20200106/session02/array01/channel017/cell01', '20200106/session02/array03/channel066/cell01', '20200106/session02/array04/channel121/cell01', '20200121/session01/array02/channel062/cell03']
for ex in wz_exclusionList:    
    if (wz_monkey_path + '/' + ex) in wz_cell_paths:
        wz_cell_paths.remove(wz_monkey_path + '/' + ex)
    
# In[] drop session with small trial size
wz_session_drop = [f'{data_path}/wangzi/20210729/session02',f'{data_path}/wangzi/20210818/session02',
                f'{data_path}/wangzi/20210819/session02',f'{data_path}/wangzi/20210826/session02',
                f'{data_path}/wangzi/20210830/session02',f'{data_path}/wangzi/20210906/session02',
                f'{data_path}/wangzi/20211005/session02',f'{data_path}/wangzi/20211011/session02',
                f'{data_path}/wangzi/20211007/session02',]
for i in wz_session_drop:
    if i in wz_session_paths:
        wz_session_paths.remove(i)
    
# In[] wangzi Info

wz_epochsDic = {'bsl':[-300,0],'s1':[0,400],'d1':[400,1400],'s2':[1400,1800],'d2':[1800,2800],'go':[2800,3200]}#, 'p1':[0,1300],'p2':[1300,2600]

wz_trialEvtsBoundary = sorted([b[1] for b in wz_epochsDic.values()])

wz_tRange = np.arange(-300, wz_trialEvtsBoundary[-1], dt)
wz_tBsl = wz_epochsDic['bsl']

wz_tslices = ((-300,0), (0,400), (400, 1300), (1400, 1800), (1800, 2700), (2800, 3200))

wz_dlpfcArrays = (1,2)
wz_fefArrays = (3,)



# In[]
whiskey_Info = {'subject': 'whiskey', 'monkey_path': whis.monkey_path(), 'session_paths': whis.session_path(), 'cell_paths': whis.cell_path(),
                'exclusionList': whis_exclusionList, 'session_drop': whis_session_drop, 
                'epochsDic': whis_epochsDic, 'trialEvtsBoundary': whis_trialEvtsBoundary, 'tRange': whis_tRange, 'tBsl': whis_tBsl, 'tslices': whis_tslices,
                'dlpfcArrays': whis_dlpfcArrays, 'fefArrays': whis_fefArrays}

wangzi_Info = {'subject': 'wangzi', 'monkey_path': wz.monkey_path(), 'session_paths': wz.session_path(), 'cell_paths': wz.cell_path(),
                'exclusionList': wz_exclusionList, 'session_drop': wz_session_drop, 
                'epochsDic': wz_epochsDic, 'trialEvtsBoundary': wz_trialEvtsBoundary, 'tRange': wz_tRange, 'tBsl': wz_tBsl, 'tslices': wz_tslices,
                'dlpfcArrays': wz_dlpfcArrays, 'fefArrays': wz_fefArrays}

monkey_Info = {'whiskey': whiskey_Info, 'wangzi': wangzi_Info, }#

monkey_names = list(monkey_Info.keys())[0] if len(monkey_Info)==1 else 'all'#

# In[] epoch parameters

locs = (0,1,2,3,)
locCombs = list(permutations(locs,2))
ttypes = (1,2,)

dropCombs = ()

subConditions = list(product(locCombs, ttypes))


# In[]

sessions = []
cellsToUse = []

for m, mDict in monkey_Info.items():
    monkey_path = mDict['monkey_path']
    cell_paths = mDict['cell_paths']
    session_paths = mDict['session_paths']
    exclusionList = mDict['exclusionList']
    session_drop = mDict['session_drop']
    
    for ss in session_drop:
        if ss in session_paths:
            session_paths.remove(ss)
    
    for c in exclusionList:
        if (monkey_path + '/' + c) in cell_paths:
            cell_paths.remove(monkey_path + '/' + c)
    
    sessions += session_paths
    cellsToUse += cell_paths



# In[] decode from pseudo population
pd.options.mode.chained_assignment = None
epsilon = 0.0000001
# In[] decodability with/without permutation P value
bins = 50 # dt #
tslice = (-300,2700)
tsliceRange = np.arange(-300,2700,dt)
slice_epochsDic = {'bsl':[-300,0],'s1':[0,300],'d1':[300,1300],'s2':[1300,1600],'d2':[1600,2600],'go':[2600,3000]}
#slice_epochsDic = {'bsl':[-300,0],'s1':[0,400],'d1':[400,1400],'s2':[1400,1800],'d2':[1800,2800],'go':[2600,3000]} #wangzi

events = [0, 1300, 2600] #300, 1600, if monkey_names=='whiskey' else [0, 400, 1400, 1800, 2800]
# In[]

############################################
######### create pseudo population #########
############################################

# In[] count for minimal number of trial per locComb across all sessions


trialCounts = {s:[] for s in subConditions}

for session in sessions:
    
    cells = [cc for cc in cellsToUse if '/'.join(cc.split('/')[:-3]) == session]
    
    #session=session_paths[0]
    
    trial_df = Get_trial(session).trial_selection_df()
    
    #saccadesErr_df = f_saccade.saccades_df(f_saccade.load_saccades_dict(f'{session}/incorrect_saccades_900b9e75987c962.mat'))
    ##trialErr_df = trial_df[(trial_df.accuracy == 0)].reset_index(drop=True)
    #fullErr_df = f_saccade.match_saccade2trial_Err(saccadesErr_df, trialErr_df)
    #fullErr_df_ = fullErr_df.copy()[['trial_index','accuracy','type','loc1','loc2','locKey','loc_endpos', 'quad_endpos', 
    #                                 'duration', 'onset', 'T_response_on', 'T_failure', 'T_trial_end']].dropna()
    
    #idxs = sorted(fullErr_df_[(fullErr_df_.locKey==fullErr_df_.loc_endpos)].trial_index.values.tolist() 
    #              + trial_df[(trial_df.accuracy == 1)].trial_index.values.tolist())
    
    trial_df = trial_df[(trial_df.accuracy == 1)].reset_index(drop = True)
    
    #trial_df = trial_df.loc[idxs,:].reset_index(drop = True)
    
    for k in sorted(trialCounts.keys()):
        loc1, loc2 = k[0][0], k[0][1]
        tt = k[1]
        temp = trial_df[(trial_df.type == tt)&(trial_df.loc1 == loc1)&(trial_df.loc2 == loc2)].reset_index(drop = True)
        trialCounts[k] += [len(temp)]
    
    counts1 = sorted(trial_df[trial_df.type==1].locs.value_counts())
    counts2 = sorted(trial_df[trial_df.type==2].locs.value_counts())
    #trialCounts += [counts1]
    #trialCounts += [counts2]
    
    print(f'{session}, nCells = {len(cells)}')
    print(f'Counts Ret: {counts1}')
    print(f'Counts Dis: {counts2}')

trialCounts = {v:min(k) for v, k in trialCounts.items()}

#trialMin# = 40 
trialMin = min(trialCounts.values())



# In[]

# per pseudo session
samplePerCon = int(trialMin*1) # number of trials per condition to sample from each session, dependent on the trialMin
sampleRounds = 1
arti_noise_level = 0

# In[]
#nIters = 100
#for n in range(nIters):
#    t_IterOn = time.time()
#    print(f'Iter = {n}')
    
    #create pseudo populations
    
#    pseudo_TrialInfo = f_pseudoPop.pseudo_Session(locCombs, ttypes, samplePerCon=samplePerCon, sampleRounds=sampleRounds)
#    pseudo_region = f_pseudoPop.pseudo_PopByRegion_pooled(sessions, cellsToUse, monkey_Info, locCombs = locCombs, ttypes = ttypes, samplePerCon = samplePerCon, sampleRounds=sampleRounds, arti_noise_level = arti_noise_level)
    
#    pseudo_data = {'pseudo_TrialInfo':pseudo_TrialInfo, 'pseudo_region':pseudo_region}
#    np.save(save_path + f'/pseudo_{monkey_names}{n}.npy', pseudo_data, allow_pickle=True)
    

# In[]

###########################################
######### cross-temporal decoding #########
###########################################

# In[]
nIters = 100
nBoots = 10
nPerms = nBoots#100
tbins = np.arange(tslice[0], tslice[1], bins)

tBsl = (-300,0)
idxxBsl = [tsliceRange.tolist().index(tBsl[0]), tsliceRange.tolist().index(tBsl[1])] #


maxDim_fef = 15 if monkey_names=='wangzi' else 15

nPCs_region = {'dlpfc':[0,15], 'fef':[0,15]}
nPCs_region_a = {'dlpfc':[0,2], 'fef':[0,2]}
nPCs_region_b = {'dlpfc':[2,15], 'fef':[2,15]}

conditions = (('type', 1), ('type', 2))
EVRs = {'dlpfc':[],'fef':[]}
#%% initialize variables
performance1X = {'Retarget':{'dlpfc':[],'fef':[]}, 'Distractor':{'dlpfc':[],'fef':[]}}
performance1X_shuff = {'Retarget':{'dlpfc':[],'fef':[]}, 'Distractor':{'dlpfc':[],'fef':[]}}
performance2X = {'Retarget':{'dlpfc':[],'fef':[]}, 'Distractor':{'dlpfc':[],'fef':[]}}
performance2X_shuff = {'Retarget':{'dlpfc':[],'fef':[]}, 'Distractor':{'dlpfc':[],'fef':[]}}

performance1W = {'Retarget':{'dlpfc':[],'fef':[]}, 'Distractor':{'dlpfc':[],'fef':[]}}
performance1W_shuff = {'Retarget':{'dlpfc':[],'fef':[]}, 'Distractor':{'dlpfc':[],'fef':[]}}
performance2W = {'Retarget':{'dlpfc':[],'fef':[]}, 'Distractor':{'dlpfc':[],'fef':[]}}
performance2W_shuff = {'Retarget':{'dlpfc':[],'fef':[]}, 'Distractor':{'dlpfc':[],'fef':[]}}
#%% initialize variables for subspaces
performance1X_a = {'Retarget':{'dlpfc':[],'fef':[]}, 'Distractor':{'dlpfc':[],'fef':[]}}
performance1X_a_shuff = {'Retarget':{'dlpfc':[],'fef':[]}, 'Distractor':{'dlpfc':[],'fef':[]}}
performance2X_a = {'Retarget':{'dlpfc':[],'fef':[]}, 'Distractor':{'dlpfc':[],'fef':[]}}
performance2X_a_shuff = {'Retarget':{'dlpfc':[],'fef':[]}, 'Distractor':{'dlpfc':[],'fef':[]}}

performance1W_a = {'Retarget':{'dlpfc':[],'fef':[]}, 'Distractor':{'dlpfc':[],'fef':[]}}
performance1W_a_shuff = {'Retarget':{'dlpfc':[],'fef':[]}, 'Distractor':{'dlpfc':[],'fef':[]}}
performance2W_a = {'Retarget':{'dlpfc':[],'fef':[]}, 'Distractor':{'dlpfc':[],'fef':[]}}
performance2W_a_shuff = {'Retarget':{'dlpfc':[],'fef':[]}, 'Distractor':{'dlpfc':[],'fef':[]}}

performance1X_b = {'Retarget':{'dlpfc':[],'fef':[]}, 'Distractor':{'dlpfc':[],'fef':[]}}
performance1X_b_shuff = {'Retarget':{'dlpfc':[],'fef':[]}, 'Distractor':{'dlpfc':[],'fef':[]}}
performance2X_b = {'Retarget':{'dlpfc':[],'fef':[]}, 'Distractor':{'dlpfc':[],'fef':[]}}
performance2X_b_shuff = {'Retarget':{'dlpfc':[],'fef':[]}, 'Distractor':{'dlpfc':[],'fef':[]}}

performance1W_b = {'Retarget':{'dlpfc':[],'fef':[]}, 'Distractor':{'dlpfc':[],'fef':[]}}
performance1W_b_shuff = {'Retarget':{'dlpfc':[],'fef':[]}, 'Distractor':{'dlpfc':[],'fef':[]}}
performance2W_b = {'Retarget':{'dlpfc':[],'fef':[]}, 'Distractor':{'dlpfc':[],'fef':[]}}
performance2W_b_shuff = {'Retarget':{'dlpfc':[],'fef':[]}, 'Distractor':{'dlpfc':[],'fef':[]}}
#%% initialize decoding params
toDecode_labels1 = 'loc1'
toDecode_labels2 = 'loc2'

Y_columnsLabels = ['locKey','locs','type','loc1','loc2','locX']

shuff_excludeInv = True 


#%%

########################################################
# Null method: Decoders trained on random-shuff labels #
########################################################

# In[] cross- and within-temporal decoding: permutation with (semi-)random-trained decoders
for n in range(nIters):
    t_IterOn = time.time()
    print(f'Iter = {n}')
    
    # load pseudo populations
    pseudo_data = np.load(save_path + f'/pseudo_{monkey_names}{n}.npy', allow_pickle=True).item()
    pseudo_region = pseudo_data['pseudo_region']
    pseudo_TrialInfo = pseudo_data['pseudo_TrialInfo']
    
    
    ### decode for each region
    for region in ('dlpfc','fef'):
        
        nPCs = nPCs_region[region]
        
        pseudo_PopT = pseudo_region[region][pseudo_TrialInfo.trial_index.values,:,:] # shape2 trial * cell * time
        
        # if detrend by subtract avg
        for ch in range(pseudo_PopT.shape[1]):
            temp = pseudo_PopT[:,ch,:]
            pseudo_PopT[:,ch,:] = (temp - temp.mean(axis=0))#/(temp.std(axis = 0) + epsilon) # to avoid divid by 0 issue
            
        X_region = pseudo_PopT
        
        ### scaling
        for ch in range(X_region.shape[1]):
            #X_region[:,ch,:] = scale(X_region[:,ch,:]) #01 scale for each channel
            X_region[:,ch,:] = (X_region[:,ch,:] - X_region[:,ch,:].mean()) / X_region[:,ch,:].std() #standard scaler
            #X_region[:,ch,:] = (X_region[:,ch,:] - X_region[:,ch,:].mean(axis=0)) / X_region[:,ch,:].std(axis=0) #detrended standard scaler
            #X_region[:,ch,:] = (X_region[:,ch,:] - X_region[:,ch,idxxBsl[0]:idxxBsl[1]].mean())/X_region[:,ch,idxxBsl[0]:idxxBsl[1]].std() # Z to baseline
        
                
        ### if specify applied time window of pca
        if monkey_names=='wangzi':
            pca_tWin = np.hstack((np.arange(400,1400,dt, dtype = int),np.arange(1800,2800,dt, dtype = int))).tolist() # wangzi
        else:
            pca_tWin = np.hstack((np.arange(300,1300,dt, dtype = int),np.arange(1600,2600,dt, dtype = int))).tolist() # whiskey
        pca_tWinX = [tsliceRange.tolist().index(i) for i in pca_tWin]
        
        #pca_tWinX = None
        X_regionT = X_region[:,:,pca_tWinX] if pca_tWinX != None else X_region[:,:,:]
        
        
        
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
        X_regionT_ = []
        pseudo_TrialInfoT_ = pseudo_TrialInfo.reset_index(drop=True)
        
        for sc in subConditions:
            l1,l2 = sc[0]
            tt = sc[1]
            #lcs = str(lc[0]) + '_' + str(lc[1])
            idxx = pseudo_TrialInfoT_[(pseudo_TrialInfoT_.loc1 == l1)&(pseudo_TrialInfoT_.loc2 == l2)&(pseudo_TrialInfoT_.type == tt)].index.tolist()
            X_regionT_ += [X_regionT[idxx,:,:].mean(axis=0).mean(axis=-1)]
        
        X_regionT = np.vstack(X_regionT_).T
        
        
        ### fit & transform pca
        pcFrac = 1
        npc = min(int(pcFrac * X_regionT.shape[0]), X_regionT.shape[1])
        pca = PCA(n_components=npc)
        
        pca.fit(X_regionT.T)
        evr = pca.explained_variance_ratio_
        EVRs[region] += [evr]
        print(f'{region}, {evr.round(4)[0:5]}')
        
        X_regionP = np.zeros((X_region.shape[0], npc, X_region.shape[2]))
        for trial in range(X_region.shape[0]):
            X_regionP[trial,:,:] = pca.transform(X_region[trial,:,:].T).T
        
        
        for condition in conditions:
            ttypeT = 'Retarget' if condition[-1]==1 else 'Distractor'
            tt = condition[-1]
            
            if bool(condition):
                pseudo_TrialInfoT = pseudo_TrialInfo[pseudo_TrialInfo[condition[0]] == condition[1]]
            else:
                pseudo_TrialInfoT = pseudo_TrialInfo.copy()
            
            idxT = pseudo_TrialInfoT.index
            Y = pseudo_TrialInfoT.loc[:,Y_columnsLabels].values
            ntrial = len(pseudo_TrialInfoT)
            
            
            X_regionPT = X_regionP[idxT,:,:]#[subConditions.index(i) for i in subConditions if i[1]==tt]
            
            ### down sample to 50ms/bin
            ntrialT, ncellT, ntimeT = X_regionPT.shape
            full_setP = np.mean(X_regionPT.reshape(ntrialT, ncellT, int(ntimeT/(bins/dt)),int(bins/dt)),axis = 3)
            
            
            toDecode_X1 = Y_columnsLabels.index(toDecode_labels1)
            toDecode_X2 = Y_columnsLabels.index(toDecode_labels2)
            
            full_label1 = Y[:,toDecode_X1].astype('int') #.astype('str') # locKey
            full_label2 = Y[:,toDecode_X2].astype('int') #.astype('str') # locKey

            ### labels: ['locKey','locs','type','loc1','loc2','locX']
            
            
            if shuff_excludeInv:
                toDecode_labels1_inv = f_stats.inversed_label(Y_columnsLabels, toDecode_labels1, tt)
                toDecode_X1_inv = Y_columnsLabels.index(toDecode_labels1_inv)
                full_label1_inv = Y[:,toDecode_X1_inv]
                
                toDecode_labels2_inv = f_stats.inversed_label(Y_columnsLabels, toDecode_labels2, tt)
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
            
            
            for subspace in range(1):
                if subspace == 0:
                    nPCs = nPCs_region[region]
                elif subspace == 1:
                    nPCs = nPCs_region_a[region]
                else:
                    nPCs = nPCs_region_b[region]


                ### LDA decodability
                performanceX1 = np.zeros((nBoots, len(tbins),len(tbins)))
                performanceX2 = np.zeros((nBoots, len(tbins),len(tbins)))
                
                performanceW1 = np.zeros((nBoots, len(tbins),))
                performanceW2 = np.zeros((nBoots, len(tbins),))
                
                # permutation with shuffled label
                performanceX1_shuff = np.zeros((nPerms, len(tbins),len(tbins)))
                performanceX2_shuff = np.zeros((nPerms, len(tbins),len(tbins)))
                
                performanceW1_shuff = np.zeros((nPerms, len(tbins),))
                performanceW2_shuff = np.zeros((nPerms, len(tbins),))

                for nbt in range(nBoots):

                    ### split into train and test sets
                    train_setID = np.sort(np.random.choice(ntrial, round(0.5*ntrial),replace = False))
                    test_setID = np.sort(np.random.choice(np.setdiff1d(np.arange(ntrial), train_setID, assume_unique=True), 
                                                            ntrial-len(train_setID), replace = False))

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
                            
                            # item1
                            #clf1 = LinearDiscriminantAnalysis()
                            #clf1.fit(train_setP[:,nPCs[0]:nPCs[1],t], train_label1)
                            #pfmTT1_ = clf1.score(test_setP[:,nPCs[0]:nPCs[1],t_],test_label1)
                            pfmTT1_ = f_decoding.LDAPerformance(train_setP[:,nPCs[0]:nPCs[1],t], test_setP[:,nPCs[0]:nPCs[1],t_],
                                                                train_label1, test_label1)

                            # item2
                            #clf2 = LinearDiscriminantAnalysis()
                            #clf2.fit(train_setP[:,nPCs[0]:nPCs[1],t], train_label2)
                            #pfmTT2_ = clf2.score(test_setP[:,nPCs[0]:nPCs[1],t_],test_label2)
                            pfmTT2_ = f_decoding.LDAPerformance(train_setP[:,nPCs[0]:nPCs[1],t], test_setP[:,nPCs[0]:nPCs[1],t_],
                                                                train_label2, test_label2)


                            performanceX1[nbt,t,t_] = pfmTT1_
                            performanceX2[nbt,t,t_] = pfmTT2_
                            
                            if t==t_:
                                performanceW1[nbt,t] = pfmTT1_
                                performanceW2[nbt,t] = pfmTT2_
                            
                            
                        # permutation null distribution
                        #for npm in range(nPerms):
                                                        
                            #pfmTT1_shuff = clf1.score(test_setP[:,nPCs[0]:nPCs[1],t_],test_label1_shuff)
                            #pfmTT2_shuff = clf2.score(test_setP[:,nPCs[0]:nPCs[1],t_],test_label2_shuff)
                            pfmTT1_shuff = f_decoding.LDAPerformance(train_setP[:,nPCs[0]:nPCs[1],t], test_setP[:,nPCs[0]:nPCs[1],t_],
                                                                     train_label1_shuff, test_label1_shuff)
                            pfmTT2_shuff = f_decoding.LDAPerformance(train_setP[:,nPCs[0]:nPCs[1],t], test_setP[:,nPCs[0]:nPCs[1],t_],
                                                                     train_label2_shuff, test_label2_shuff)
                            

                            performanceX1_shuff[nbt,t,t_] = pfmTT1_shuff
                            performanceX2_shuff[nbt,t,t_] = pfmTT2_shuff
                            
                            if t==t_:
                                performanceW1_shuff[nbt,t] = pfmTT1_shuff
                                performanceW2_shuff[nbt,t] = pfmTT2_shuff
                                
                
                if subspace == 0:
                    performance1X[ttypeT][region] += [np.array(performanceX1)]
                    performance2X[ttypeT][region] += [np.array(performanceX2)]
                    performance1W[ttypeT][region] += [np.array(performanceW1)]
                    performance2W[ttypeT][region] += [np.array(performanceW2)]
                    
                    performance1X_shuff[ttypeT][region] += [np.array(performanceX1_shuff)]
                    performance2X_shuff[ttypeT][region] += [np.array(performanceX2_shuff)]
                    performance1W_shuff[ttypeT][region] += [np.array(performanceW1_shuff)]
                    performance2W_shuff[ttypeT][region] += [np.array(performanceW2_shuff)]

                elif subspace == 1:
                    performance1X_a[ttypeT][region] += [np.array(performanceX1)]
                    performance2X_a[ttypeT][region] += [np.array(performanceX2)]
                    performance1W_a[ttypeT][region] += [np.array(performanceW1)]
                    performance2W_a[ttypeT][region] += [np.array(performanceW2)]
                    
                    performance1X_a_shuff[ttypeT][region] += [np.array(performanceX1_shuff)]
                    performance2X_a_shuff[ttypeT][region] += [np.array(performanceX2_shuff)]
                    performance1W_a_shuff[ttypeT][region] += [np.array(performanceW1_shuff)]
                    performance2W_a_shuff[ttypeT][region] += [np.array(performanceW2_shuff)]

                else:
                    performance1X_b[ttypeT][region] += [np.array(performanceX1)]
                    performance2X_b[ttypeT][region] += [np.array(performanceX2)]
                    performance1W_b[ttypeT][region] += [np.array(performanceW1)]
                    performance2W_b[ttypeT][region] += [np.array(performanceW2)]
                    
                    performance1X_b_shuff[ttypeT][region] += [np.array(performanceX1_shuff)]
                    performance2X_b_shuff[ttypeT][region] += [np.array(performanceX2_shuff)]
                    performance1W_b_shuff[ttypeT][region] += [np.array(performanceW1_shuff)]
                    performance2W_b_shuff[ttypeT][region] += [np.array(performanceW2_shuff)]
    
    print(f'tIter = {(time.time() - t_IterOn):.4f}s')

# In[] within time only
for n in range(nIters):
    t_IterOn = time.time()
    print(f'Iter = {n}')
    
    # load pseudo populations
    pseudo_data = np.load(save_path + f'/pseudo_{monkey_names}{n}.npy', allow_pickle=True).item()
    pseudo_region = pseudo_data['pseudo_region']
    pseudo_TrialInfo = pseudo_data['pseudo_TrialInfo']
    
    
    ### decode for each region
    for region in ('dlpfc','fef'):
        
        nPCs = nPCs_region[region]
        
        pseudo_PopT = pseudo_region[region][pseudo_TrialInfo.trial_index.values,:,:] # shape2 trial * cell * time
        
        # if detrend by subtract avg
        for ch in range(pseudo_PopT.shape[1]):
            temp = pseudo_PopT[:,ch,:]
            pseudo_PopT[:,ch,:] = (temp - temp.mean(axis=0))#/(temp.std(axis = 0) + epsilon) # to avoid divid by 0 issue
            
        X_region = pseudo_PopT
        
        ### scaling
        for ch in range(X_region.shape[1]):
            #X_region[:,ch,:] = scale(X_region[:,ch,:]) #01 scale for each channel
            X_region[:,ch,:] = (X_region[:,ch,:] - X_region[:,ch,:].mean()) / X_region[:,ch,:].std() #standard scaler
            #X_region[:,ch,:] = (X_region[:,ch,:] - X_region[:,ch,:].mean(axis=0)) / X_region[:,ch,:].std(axis=0) #detrended standard scaler
            #X_region[:,ch,:] = (X_region[:,ch,:] - X_region[:,ch,idxxBsl[0]:idxxBsl[1]].mean())/X_region[:,ch,idxxBsl[0]:idxxBsl[1]].std() # Z to baseline
        
                
        ### if specify applied time window of pca
        if monkey_names=='wangzi':
            pca_tWin = np.hstack((np.arange(400,1400,dt, dtype = int),np.arange(1800,2800,dt, dtype = int))).tolist() # wangzi
        else:
            pca_tWin = np.hstack((np.arange(300,1300,dt, dtype = int),np.arange(1600,2600,dt, dtype = int))).tolist() # whiskey
        pca_tWinX = [tsliceRange.tolist().index(i) for i in pca_tWin]
        
        #pca_tWinX = None
        X_regionT = X_region[:,:,pca_tWinX] if pca_tWinX != None else X_region[:,:,:]
        
        
        
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
        X_regionT_ = []
        pseudo_TrialInfoT_ = pseudo_TrialInfo.reset_index(drop=True)
        
        for sc in subConditions:
            l1,l2 = sc[0]
            tt = sc[1]
            #lcs = str(lc[0]) + '_' + str(lc[1])
            idxx = pseudo_TrialInfoT_[(pseudo_TrialInfoT_.loc1 == l1)&(pseudo_TrialInfoT_.loc2 == l2)&(pseudo_TrialInfoT_.type == tt)].index.tolist()
            X_regionT_ += [X_regionT[idxx,:,:].mean(axis=0).mean(axis=-1)]
        
        X_regionT = np.vstack(X_regionT_).T
        
        
        ### fit & transform pca
        pcFrac = 1
        npc = min(int(pcFrac * X_regionT.shape[0]), X_regionT.shape[1])
        pca = PCA(n_components=npc)
        
        pca.fit(X_regionT.T)
        evr = pca.explained_variance_ratio_
        EVRs[region] += [evr]
        print(f'{region}, {evr.round(4)[0:5]}')
        
        X_regionP = np.zeros((X_region.shape[0], npc, X_region.shape[2]))
        for trial in range(X_region.shape[0]):
            X_regionP[trial,:,:] = pca.transform(X_region[trial,:,:].T).T
        
        
        for condition in conditions:
            ttypeT = 'Retarget' if condition[-1]==1 else 'Distractor'
            tt = condition[-1]
            
            if bool(condition):
                pseudo_TrialInfoT = pseudo_TrialInfo[pseudo_TrialInfo[condition[0]] == condition[1]]
            else:
                pseudo_TrialInfoT = pseudo_TrialInfo.copy()
            
            idxT = pseudo_TrialInfoT.index
            Y = pseudo_TrialInfoT.loc[:,Y_columnsLabels].values
            ntrial = len(pseudo_TrialInfoT)
            
            
            X_regionPT = X_regionP[idxT,:,:]#[subConditions.index(i) for i in subConditions if i[1]==tt]
            
            ### down sample to 50ms/bin
            ntrialT, ncellT, ntimeT = X_regionPT.shape
            full_setP = np.mean(X_regionPT.reshape(ntrialT, ncellT, int(ntimeT/(bins/dt)),int(bins/dt)),axis = 3)
            
            
            toDecode_X1 = Y_columnsLabels.index(toDecode_labels1)
            toDecode_X2 = Y_columnsLabels.index(toDecode_labels2)
            
            full_label1 = Y[:,toDecode_X1].astype('int') #.astype('str') # locKey
            full_label2 = Y[:,toDecode_X2].astype('int') #.astype('str') # locKey

            ### labels: ['locKey','locs','type','loc1','loc2','locX']
            
            
            if shuff_excludeInv:
                toDecode_labels1_inv = f_stats.inversed_label(Y_columnsLabels, toDecode_labels1, tt)
                toDecode_X1_inv = Y_columnsLabels.index(toDecode_labels1_inv)
                full_label1_inv = Y[:,toDecode_X1_inv]
                
                toDecode_labels2_inv = f_stats.inversed_label(Y_columnsLabels, toDecode_labels2, tt)
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
            
            
            for subspace in range(3):
                if subspace == 0:
                    nPCs = nPCs_region[region]
                elif subspace == 1:
                    nPCs = nPCs_region_a[region]
                else:
                    nPCs = nPCs_region_b[region]


                ### LDA decodability
                performanceW1 = np.zeros((nBoots, len(tbins),))
                performanceW2 = np.zeros((nBoots, len(tbins),))
                
                # permutation with shuffled label
                performanceW1_shuff = np.zeros((nPerms, len(tbins),))
                performanceW2_shuff = np.zeros((nPerms, len(tbins),))

                for nbt in range(nBoots):

                    ### split into train and test sets
                    train_setID = np.sort(np.random.choice(ntrial, round(0.5*ntrial),replace = False))
                    test_setID = np.sort(np.random.choice(np.setdiff1d(np.arange(ntrial), train_setID, assume_unique=True), 
                                                            ntrial-len(train_setID), replace = False))

                    train_setP = full_setP[train_setID,:,:]
                    test_setP = full_setP[test_setID,:,:]
                    
                    train_label1 = full_label1[train_setID]#Y[train_setID,toDecode_X1].astype('int') #.astype('str') # locKey
                    train_label2 = full_label2[train_setID]#Y[train_setID,toDecode_X2].astype('int') #.astype('str') # locKey

                    test_label1 = full_label1[test_setID]#Y[test_setID,toDecode_X1].astype('int') #.astype('str') # locKey
                    test_label2 = full_label2[test_setID]#Y[test_setID,toDecode_X2].astype('int') #.astype('str') # locKey
                    
                    train_label1_shuff, test_label1_shuff = full_label1_shuff[train_setID], full_label1_shuff[test_setID]
                    train_label2_shuff, test_label2_shuff = full_label2_shuff[train_setID], full_label2_shuff[test_setID]
                    
                    
                    for t in range(len(tbins)):
                    
                        # item1
                        #clf1 = LinearDiscriminantAnalysis()
                        #clf1.fit(train_setP[:,nPCs[0]:nPCs[1],t], train_label1)
                        #pfmTT1_ = clf1.score(test_setP[:,nPCs[0]:nPCs[1],t_],test_label1)
                        pfmTT1_ = f_decoding.LDAPerformance(train_setP[:,nPCs[0]:nPCs[1],t], test_setP[:,nPCs[0]:nPCs[1],t],
                                                            train_label1, test_label1)

                        # item2
                        #clf2 = LinearDiscriminantAnalysis()
                        #clf2.fit(train_setP[:,nPCs[0]:nPCs[1],t], train_label2)
                        #pfmTT2_ = clf2.score(test_setP[:,nPCs[0]:nPCs[1],t_],test_label2)
                        pfmTT2_ = f_decoding.LDAPerformance(train_setP[:,nPCs[0]:nPCs[1],t], test_setP[:,nPCs[0]:nPCs[1],t],
                                                            train_label2, test_label2)

                    
                        performanceW1[nbt,t] = pfmTT1_
                        performanceW2[nbt,t] = pfmTT2_
                    
                        
                    # permutation null distribution
                    #for npm in range(nPerms):
                                                    
                        #pfmTT1_shuff = clf1.score(test_setP[:,nPCs[0]:nPCs[1],t_],test_label1_shuff)
                        #pfmTT2_shuff = clf2.score(test_setP[:,nPCs[0]:nPCs[1],t_],test_label2_shuff)
                        pfmTT1_shuff = f_decoding.LDAPerformance(train_setP[:,nPCs[0]:nPCs[1],t], test_setP[:,nPCs[0]:nPCs[1],t],
                                                                    train_label1_shuff, test_label1_shuff)
                        pfmTT2_shuff = f_decoding.LDAPerformance(train_setP[:,nPCs[0]:nPCs[1],t], test_setP[:,nPCs[0]:nPCs[1],t],
                                                                    train_label2_shuff, test_label2_shuff)
                        
                        performanceW1_shuff[nbt,t] = pfmTT1_shuff
                        performanceW2_shuff[nbt,t] = pfmTT2_shuff
                            
                
                if subspace == 0:
                    
                    performance1W[ttypeT][region] += [np.array(performanceW1)]
                    performance2W[ttypeT][region] += [np.array(performanceW2)]
                    
                    performance1W_shuff[ttypeT][region] += [np.array(performanceW1_shuff)]
                    performance2W_shuff[ttypeT][region] += [np.array(performanceW2_shuff)]

                elif subspace == 1:
                    
                    performance1W_a[ttypeT][region] += [np.array(performanceW1)]
                    performance2W_a[ttypeT][region] += [np.array(performanceW2)]
                    
                    performance1W_a_shuff[ttypeT][region] += [np.array(performanceW1_shuff)]
                    performance2W_a_shuff[ttypeT][region] += [np.array(performanceW2_shuff)]

                else:
                    performance1W_b[ttypeT][region] += [np.array(performanceW1)]
                    performance2W_b[ttypeT][region] += [np.array(performanceW2)]
                    
                    performance1W_b_shuff[ttypeT][region] += [np.array(performanceW1_shuff)]
                    performance2W_b_shuff[ttypeT][region] += [np.array(performanceW2_shuff)]
    
    print(f'tIter = {(time.time() - t_IterOn):.4f}s')




#%% save full space decodability
np.save(f'{phd_path}/outputs/monkeys/' + f'performanceX1_{monkey_names}_data.npy', performance1X, allow_pickle=True)
np.save(f'{phd_path}/outputs/monkeys/' + f'performanceX2_{monkey_names}_data.npy', performance2X, allow_pickle=True)
np.save(f'{phd_path}/outputs/monkeys/' + f'performanceX1_shuff_{monkey_names}_data.npy', performance1X_shuff, allow_pickle=True)
np.save(f'{phd_path}/outputs/monkeys/' + f'performanceX2_shuff_{monkey_names}_data.npy', performance2X_shuff, allow_pickle=True)
#%% save subspaces decodability
np.save(f'{phd_path}/outputs/monkeys/' + f'performanceX1_a_{monkey_names}_data.npy', performance1X_a, allow_pickle=True)
np.save(f'{phd_path}/outputs/monkeys/' + f'performanceX2_a_{monkey_names}_data.npy', performance2X_a, allow_pickle=True)
np.save(f'{phd_path}/outputs/monkeys/' + f'performanceX1_a_{monkey_names}_shuff_data.npy', performance1X_a_shuff, allow_pickle=True)
np.save(f'{phd_path}/outputs/monkeys/' + f'performanceX2_a_shuff_{monkey_names}_data.npy', performance2X_a_shuff, allow_pickle=True)

np.save(f'{phd_path}/outputs/monkeys/' + f'performanceX1_b_{monkey_names}_data.npy', performance1X_b, allow_pickle=True)
np.save(f'{phd_path}/outputs/monkeys/' + f'performanceX2_b_{monkey_names}_data.npy', performance2X_b, allow_pickle=True)
np.save(f'{phd_path}/outputs/monkeys/' + f'performanceX1_b_shuff_{monkey_names}_data.npy', performance1X_b_shuff, allow_pickle=True)
np.save(f'{phd_path}/outputs/monkeys/' + f'performanceX2_b_shuff_{monkey_names}_data.npy', performance2X_b_shuff, allow_pickle=True)

#%% save EVRs
np.save(f'{phd_path}/outputs/monkeys/' + f'EVRs_{monkey_names}_data.npy', EVRs, allow_pickle=True)

# In[] load full space decodability

performance1X = np.load(f'{phd_path}/outputs/monkeys/' + 'performanceX1_full_data.npy', allow_pickle=True).item()
performance2X = np.load(f'{phd_path}/outputs/monkeys/' + 'performanceX2_full_data.npy', allow_pickle=True).item()
performance1X_shuff = np.load(f'{phd_path}/outputs/monkeys/' + 'performanceX1_full_shuff_data.npy', allow_pickle=True).item()
performance2X_shuff = np.load(f'{phd_path}/outputs/monkeys/' + 'performanceX2_full_shuff_data.npy', allow_pickle=True).item()

performance1W = np.load(f'{phd_path}/outputs/monkeys/' + 'performanceW1_full_data.npy', allow_pickle=True).item()
performance2W = np.load(f'{phd_path}/outputs/monkeys/' + 'performanceW2_full_data.npy', allow_pickle=True).item()
performance1W_shuff = np.load(f'{phd_path}/outputs/monkeys/' + 'performanceW1_full_shuff_data.npy', allow_pickle=True).item()
performance2W_shuff = np.load(f'{phd_path}/outputs/monkeys/' + 'performanceW2_full_shuff_data.npy', allow_pickle=True).item()

EVRs = np.load(f'{phd_path}/outputs/monkeys/' + 'EVRs_full.npy', allow_pickle=True).item()
# In[] plot cross temporal decoding
for region in ('dlpfc','fef'):
    nPCs = nPCs_region[region]
    evrSum = np.array(EVRs[region])[:,nPCs[0]:nPCs[1]].mean(0).sum().round(3)
    #if monkey_names=='wangzi':
    #    vmax = 0.4 if region == 'dlpfc' else 0.5
    #else:    
    vmax = 0.6 if region == 'dlpfc' else 0.8
    fig = plt.figure(figsize=(28, 24), dpi=100)
    
    for condition in conditions:
        
        ttypeT = 'Retarget' if condition[-1]==1 else 'Distractor'
        ttypeT_ = 'Retarget' if condition[-1]==1 else 'Distraction'
        tt = 1 if condition[-1]==1 else 2
        
        performanceT1 = performance1X[ttypeT][region]
        performanceT1_shuff = performance1X_shuff[ttypeT][region]
        performanceT2 = performance2X[ttypeT][region]
        performanceT2_shuff = performance2X_shuff[ttypeT][region]
        
        pfm1 = np.array(performanceT1).mean(1)
        pfm1_shuff = np.array(performanceT1_shuff).mean(1)#np.concatenate(np.array(performanceT1_shuff),axis=2) #
        pfm2 = np.array(performanceT2).mean(1)
        pfm2_shuff = np.array(performanceT2_shuff).mean(1)#np.concatenate(np.array(performanceT2_shuff),axis=2) #
        
        pvalues1 = np.zeros((len(tbins), len(tbins)))
        pvalues2 = np.zeros((len(tbins), len(tbins)))
        for t in range(len(tbins)):
            for t_ in range(len(tbins)):
                #pvalues1[t,t_] = f_stats.permutation_p(pfm1.mean(0)[t,t_], pfm1_shuff[t,t_,:], tail = 'greater')
                #pvalues2[t,t_] = f_stats.permutation_p(pfm2.mean(0)[t,t_], pfm2_shuff[t,t_,:], tail = 'greater')
                
                #pvalues1[t,t_] = stats.ttest_1samp(pfm1[:,t,t_], 0.33333, alternative = 'greater')[1]
                #pvalues2[t,t_] = stats.ttest_1samp(pfm2[:,t,t_], 0.33333, alternative = 'greater')[1]
                
                pvalues1[t,t_] = f_stats.permutation_pCI(pfm1[:,t,t_], pfm1_shuff[:,t,t_],tail='greater',alpha=5)
                pvalues2[t,t_] = f_stats.permutation_pCI(pfm2[:,t,t_], pfm2_shuff[:,t,t_],tail='greater',alpha=5)
        
        
        #plt.figure(figsize=(15,12), dpi = 100)
        # item1
        plt.subplot(2,2,tt)
        ax = plt.gca()
        sns.heatmap(pd.DataFrame(pfm1.mean(axis = 0), index=tbins,columns=tbins), cmap = 'magma', vmin = 0.25, vmax = vmax, ax = ax)#, vmax = 0.6, xticklabels = 100, yticklabels = 100, vmin = cbar_min, vmax = cbar_max
        
        #from scipy import ndimage
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
        # here set the labelsize by 20
        cbar.ax.tick_params(labelsize=20)
        
        ax.set_title(f'{ttypeT_}, Item2', fontsize = 30, pad = 20)
        
    plt.tight_layout()
    plt.subplots_adjust(top = 0.95)
    
    #if region=='fef':
    #    if nPCs_region[region]==[0,maxDim_fef]:
    #        plt.suptitle(f'{region.upper()}_{monkey_names}, Full Space (ΣEVR={evrSum})', fontsize = 35, y=1)
    #    else:
    #        plt.suptitle(f'{region.upper()}_{monkey_names}, PC{nPCs_region[region][0]+1}-{nPCs_region[region][1]} (Σ EVR={evrSum})', fontsize = 35, y=1)
        
    #else:
    #    if nPCs_region[region]==[0,10]:
    #    plt.suptitle(f'{region.upper()}_{monkey_names}, Full Space (ΣEVR={evrSum})', fontsize = 35, y=1)
    #    else:
    #        plt.suptitle(f'{region.upper()}_{monkey_names}, PC{nPCs_region[region][0]+1}-{nPCs_region[region][1]} (Σ EVR={evrSum})', fontsize = 35, y=1)
        
    plt.suptitle(f'{region.upper()}, Full Space (ΣEVR={evrSum})', fontsize = 35, y=1)
    plt.show()
    
    #fig.savefig(f'{save_path}/decodabilityX_full_{region}_{monkey_names}.tif')
    fig.savefig(f'{phd_path}/outputs/monkeys/decodabilityX_full_{region}_all.tif')


# In[] plot within time decodability
#tbins = np.arange(tslice[0], tslice[1], 50)

for region in ('dlpfc','fef'):
    nPCs = nPCs_region[region]
    evrSum = np.array(EVRs[region])[:,nPCs[0]:nPCs[1]].mean(0).sum().round(3)
    
    if monkey_names=='wangzi':
        vmax = 0.8 if region == 'dlpfc' else 1.0
    else:    
        vmax = 0.8 if region == 'dlpfc' else 1.0
        
    fig, axes = plt.subplots(1,2, figsize=(15,5), dpi=300, sharex=True, sharey=True)
    #plt.figure(figsize=(12,5), dpi = 100)
    
    for condition in conditions:
        
        tt = condition[-1]
        ttypeT = 'Retarget' if condition[-1]==1 else 'Distractor'
        
        performanceT1 = performance1W[ttypeT][region]
        performanceT1_shuff = performance1W_shuff[ttypeT][region]
        performanceT2 = performance2W[ttypeT][region]
        performanceT2_shuff = performance2W_shuff[ttypeT][region]
        
        pfm1 = np.array(performanceT1).mean(1)
        pfm1_shuff = np.array(performanceT1_shuff).mean(1) #np.concatenate(np.array(performanceT1_shuff),axis=1) #
        pfm2 = np.array(performanceT2).mean(1)
        pfm2_shuff = np.array(performanceT2_shuff).mean(1) #np.concatenate(np.array(performanceT2_shuff),axis=1) #
        
        pvalues1 = np.zeros((len(tbins)))
        pvalues2 = np.zeros((len(tbins)))
        for t in range(len(tbins)):
            #pvalues1[t] = f_stats.permutation_pCI(pfm1[:,t], pfm1_shuff[:,t], alpha = 5, tail = 'greater')
            #pvalues2[t] = f_stats.permutation_pCI(pfm2[:,t], pfm2_shuff[:,t], alpha = 5, tail = 'greater')
            
            pvalues1[t] = stats.ttest_1samp(pfm1[:,t], 0.25, alternative = 'greater')[1]
            pvalues2[t] = stats.ttest_1samp(pfm2[:,t], 0.25, alternative = 'greater')[1]
            
            #pvalues1[t] = f_stats.permutation_p(np.percentile(pfm1[:,t],5), pfm1_shuff[:,t], tail = 'greater')
            #pvalues2[t] = f_stats.permutation_p(np.percentile(pfm2[:,t],5), pfm2_shuff[:,t], tail = 'greater')
            #pvalues1[t] = f_stats.permutation_pCI(pfm1[:,t], pfm1_shuff[t,:], tail = 'greater')
            #pvalues2[t] = f_stats.permutation_pCI(pfm2[:,t], pfm2_shuff[t,:], tail = 'greater')
            
        
        #vmax = 0.6 if region == 'dlpfc' else 0.8
        
        
        #plt.subplot(1,2,tt)
        ax = axes.flatten()[tt-1]#plt.gca()
        ax.plot(np.arange(0, len(tbins), 1), pfm1.mean(0), marker = ' ', color = 'b', label = 'Item1')#, vmax = 0.6, xticklabels = 100, yticklabels = 100, vmin = cbar_min, vmax = cbar_max
        ax.plot(np.arange(0, len(tbins), 1), pfm2.mean(0), marker = ' ', color = 'm', label = 'Item2')
        ax.fill_between(np.arange(0, len(tbins), 1), (pfm1.mean(0) - pfm1.std(0)), (pfm1.mean(0) + pfm1.std(0)), color='b', alpha=.1)
        ax.fill_between(np.arange(0, len(tbins), 1), (pfm2.mean(0) - pfm2.std(0)), (pfm2.mean(0) + pfm2.std(0)), color='m', alpha=.1)
        
        
        #if tt==1:
            # crossover point
            #crossover_t = np.array([f_stats.crossover_lineInterp(tbins, pfm1[i], pfm2[i], tBoundaries=(1300,2600), minDur=200) for i in range(pfm1.shape[0])])[:,0]
            #crossover_v = np.array([f_stats.crossover_lineInterp(tbins, pfm1[i], pfm2[i], tBoundaries=(1300,2600), minDur=200) for i in range(pfm1.shape[0])])[:,1]
        
            #temp_x2 = np.where(np.abs((tbins-crossover_t.mean()))==np.abs((tbins-crossover_t.mean())).min())[0][0]
            #temp_x1 = temp_x2-1
            #temp_x_mean = temp_x2 - (tbins[temp_x2] - crossover_t.mean())/(tbins[temp_x2] - tbins[temp_x1])
            #temp_x_std = crossover_t.std()/(tbins[temp_x2] - tbins[temp_x1])

            #ax.errorbar(temp_x_mean, crossover_v.mean(), xerr = temp_x_std/(crossover_t.shape[0]**0.5), yerr = crossover_v.std()/(crossover_v.shape[0]**0.5), 
            #            marker = 'X', markersize = 15, color='indianred', markeredgecolor='darkred',capsize=5, markeredgewidth=2)
        
        #ax.fill_between(np.arange(0, len(tbins), 1), (pfm1.mean(0) - pfm1.std(0)/pfm1.shape[0]**0.5), (pfm1.mean(0) + pfm1.std(0)/pfm1.shape[0]**0.5), color='b', alpha=.1)
        #ax.fill_between(np.arange(0, len(tbins), 1), (pfm2.mean(0) - pfm2.std(0)/pfm2.shape[0]**0.5), (pfm2.mean(0) + pfm2.std(0)/pfm2.shape[0]**0.5), color='m', alpha=.1)
        
        # significance line
        segs1 = f_plotting.significance_line_segs(pvalues1,0.05)
        segs2 = f_plotting.significance_line_segs(pvalues2,0.05)
        
        for start1, end1 in segs1:
            ax.plot(np.arange(start1,end1,1), np.full_like(np.arange(start1,end1,1), vmax-2*(vmax/20), dtype='float'), color='b', linestyle='-', linewidth=3)
            
        for start2, end2 in segs2:
            ax.plot(np.arange(start2,end2,1), np.full_like(np.arange(start2,end2,1), vmax-1*(vmax/20), dtype='float'), color='m', linestyle='-', linewidth=3)

        
        # event lines
        for i in events:
            
            #ax.plot(tbins, np.full_like(tbins,list(tbins).index(i)), 'k-.', linewidth=4, alpha = 0.25)
            ax.plot(np.full_like(tbins,list(tbins).index(i)), tbins, 'k-.', linewidth=2, alpha = 0.25)
        
        ax.set_title(f'{ttypeT}', fontsize = 20, pad = 20)
        #ax.set_xticks([list(tbins).index(i) for i in events])
        #ax.set_xticklabels(['S1','S2','Go Cue'], fontsize = 10)
        ax.set_xticks([list(tbins).index(i) for i in events] + [list(tbins).index(i) for i in np.arange(250,1250,250)] + [list(tbins).index(i) for i in np.arange(1550,2550,250)],
                    labels = ['S1','S2','Go Cue']+np.arange(250,1250,250).tolist()+np.arange(250,1250,250).tolist(), 
                    minor=False, fontsize = 10)
        
        ax.set_xlabel('Time', fontsize = 20)
        ax.set_ylim((0.1,vmax))
        ax.set_xlim((0, tbins.tolist().index(2600)))
        #ax.set_yticklabels(checkpoints, fontsize = 10)
        ax.tick_params(axis='both', labelsize=12)
        
        if tt==1:
            ax.set_ylabel('Accuracy', fontsize = 20)
        #if tt==2:
            #ax.legend(bbox_to_anchor=(1.05, 0.5), fontsize=15) #loc='lower right',
        
    
    plt.tight_layout()
    plt.subplots_adjust(top = 0.8)
    
    if region=='fef':
        if nPCs==[0,15]:
            plt.suptitle(f'{region.upper()}_{monkey_names}, Full Space (ΣEVR={evrSum})', fontsize = 25, y=1)
        else:
            plt.suptitle(f'{region.upper()}_{monkey_names}, PC{nPCs[0]+1}-{nPCs[1]} (Σ EVR={evrSum})', fontsize = 25, y=1)
        
    else:
        if nPCs==[0,15]:
            plt.suptitle(f'{region.upper()}_{monkey_names}, Full Space (ΣEVR={evrSum})', fontsize = 25, y=1)
        else:
            plt.suptitle(f'{region.upper()}_{monkey_names}, PC{nPCs[0]+1}-{nPCs[1]} (Σ EVR={evrSum})', fontsize = 25, y=1)
        
    #plt.suptitle(f'{region.upper()}, PC{nPCs[0]+1}-{nPCs[1]+1}', fontsize = 35, y=1) #, Arti_Noise = {arti_noise_level}
    plt.show()
    
    #fig.savefig(f'{phd_path}/outputs/monkeys/' + 'decodabilityW_full_{region}_{monkey_names}.tif', bbox_inches='tight')
#%%
#%%

##################
# TType Decoding #
##################


#%% initialization
toDecode_labels1 = 'type'

Y_columnsLabels = ['locKey','locs','type','loc1','loc2','locX']

shuff_excludeInv = False 
EVRs = {'dlpfc':[],'fef':[]}

performancettW = {'dlpfc':[],'fef':[]}
performancettW_shuff = {'dlpfc':[],'fef':[]}

performancettX = {'dlpfc':[],'fef':[]}
performancettX_shuff = {'dlpfc':[],'fef':[]}
# In[] permutation with (semi-)random-trained decoders
for n in range(nIters):
    t_IterOn = time.time()
    print(f'Iter = {n}')
    
    # load pseudo populations
    pseudo_data = np.load(save_path + f'/pseudo_{monkey_names}{n}.npy', allow_pickle=True).item()
    pseudo_region = pseudo_data['pseudo_region']
    pseudo_TrialInfo = pseudo_data['pseudo_TrialInfo']
    
    
    ### decode for each region
    for region in ('dlpfc','fef'):
        
        nPCs = nPCs_region[region]
        
        pseudo_PopT = pseudo_region[region][pseudo_TrialInfo.trial_index.values,:,:] # shape2 trial * cell * time
        
        # if detrend by subtract avg
        for ch in range(pseudo_PopT.shape[1]):
            temp = pseudo_PopT[:,ch,:]
            pseudo_PopT[:,ch,:] = (temp - temp.mean(axis=0))#/(temp.std(axis = 0) + epsilon) # to avoid divid by 0 issue
            
        X_region = pseudo_PopT
        
        ### scaling
        for ch in range(X_region.shape[1]):
            #X_region[:,ch,:] = scale(X_region[:,ch,:]) #01 scale for each channel
            X_region[:,ch,:] = (X_region[:,ch,:] - X_region[:,ch,:].mean()) / X_region[:,ch,:].std() #standard scaler
            #X_region[:,ch,:] = (X_region[:,ch,:] - X_region[:,ch,:].mean(axis=0)) / X_region[:,ch,:].std(axis=0) #detrended standard scaler
            #X_region[:,ch,:] = (X_region[:,ch,:] - X_region[:,ch,idxxBsl[0]:idxxBsl[1]].mean())/X_region[:,ch,idxxBsl[0]:idxxBsl[1]].std() # Z to baseline
        
                
        ### if specify applied time window of pca
        if monkey_names=='wangzi':
            pca_tWin = np.hstack((np.arange(400,1400,dt, dtype = int),np.arange(1800,2800,dt, dtype = int))).tolist() # wangzi
        else:
            pca_tWin = np.hstack((np.arange(300,1300,dt, dtype = int),np.arange(1600,2600,dt, dtype = int))).tolist() # whiskey
        pca_tWinX = [tsliceRange.tolist().index(i) for i in pca_tWin]
        
        #pca_tWinX = None
        X_regionT = X_region[:,:,pca_tWinX] if pca_tWinX != None else X_region[:,:,:]
        
        
        
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
        X_regionT_ = []
        pseudo_TrialInfoT_ = pseudo_TrialInfo.reset_index(drop=True)
        
        for sc in subConditions:
            l1,l2 = sc[0]
            tt = sc[1]
            #lcs = str(lc[0]) + '_' + str(lc[1])
            idxx = pseudo_TrialInfoT_[(pseudo_TrialInfoT_.loc1 == l1)&(pseudo_TrialInfoT_.loc2 == l2)&(pseudo_TrialInfoT_.type == tt)].index.tolist()
            X_regionT_ += [X_regionT[idxx,:,:].mean(axis=0).mean(axis=-1)]
        
        X_regionT = np.vstack(X_regionT_).T
        
        
        ### fit & transform pca
        pcFrac = 1
        npc = min(int(pcFrac * X_regionT.shape[0]), X_regionT.shape[1])
        pca = PCA(n_components=npc)
        
        pca.fit(X_regionT.T)
        evr = pca.explained_variance_ratio_
        EVRs[region] += [evr]
        print(f'{region}, {evr.round(4)[0:5]}')
        
        X_regionP = np.zeros((X_region.shape[0], npc, X_region.shape[2]))
        for trial in range(X_region.shape[0]):
            X_regionP[trial,:,:] = pca.transform(X_region[trial,:,:].T).T
        
        
    
        pseudo_TrialInfoT = pseudo_TrialInfo.copy()
    
        idxT = pseudo_TrialInfoT.index
        Y = pseudo_TrialInfoT.loc[:,Y_columnsLabels].values
        ntrial = len(pseudo_TrialInfoT)
        
        
        X_regionPT = X_regionP[idxT,:,:]#[subConditions.index(i) for i in subConditions if i[1]==tt]
        
        ### down sample to 50ms/bin
        ntrialT, ncellT, ntimeT = X_regionPT.shape
        full_setP = np.mean(X_regionPT.reshape(ntrialT, ncellT, int(ntimeT/(bins/dt)),int(bins/dt)),axis = 3)
        
        
        toDecode_X1 = Y_columnsLabels.index(toDecode_labels1)
        
        full_label1 = Y[:,toDecode_X1].astype('int') #.astype('str') # locKey

        ### labels: ['locKey','locs','type','loc1','loc2','locX']
        
    
        # fully random
        full_label1_shuff = np.random.permutation(full_label1)
        
        for subspace in range(1):
            if subspace == 0:
                nPCs = nPCs_region[region]
            elif subspace == 1:
                nPCs = nPCs_region_a[region]
            else:
                nPCs = nPCs_region_b[region]


            ### LDA decodability
            performanceX1 = np.zeros((nBoots, len(tbins),len(tbins)))
            performanceW1 = np.zeros((nBoots, len(tbins),))
            
            # permutation with shuffled label
            performanceX1_shuff = np.zeros((nPerms, len(tbins),len(tbins)))
            performanceW1_shuff = np.zeros((nPerms, len(tbins),))

            for nbt in range(nBoots):

                ### split into train and test sets
                train_setID = np.sort(np.random.choice(ntrial, round(0.5*ntrial),replace = False))
                test_setID = np.sort(np.random.choice(np.setdiff1d(np.arange(ntrial), train_setID, assume_unique=True), 
                                                        ntrial-len(train_setID), replace = False))

                train_setP = full_setP[train_setID,:,:]
                test_setP = full_setP[test_setID,:,:]
                
                train_label1 = full_label1[train_setID]#Y[train_setID,toDecode_X1].astype('int') #.astype('str') # locKey
                test_label1 = full_label1[test_setID]#Y[test_setID,toDecode_X1].astype('int') #.astype('str') # locKey
                
                train_label1_shuff, test_label1_shuff = full_label1_shuff[train_setID], full_label1_shuff[test_setID]
                
                
                for t in range(len(tbins)):
                    
                    for t_ in range(len(tbins)):
                        
                        # item1
                        #clf1 = LinearDiscriminantAnalysis()
                        #clf1.fit(train_setP[:,nPCs[0]:nPCs[1],t], train_label1)
                        #pfmTT1_ = clf1.score(test_setP[:,nPCs[0]:nPCs[1],t_],test_label1)
                        pfmTT1_ = f_decoding.LDAPerformance(train_setP[:,nPCs[0]:nPCs[1],t], test_setP[:,nPCs[0]:nPCs[1],t_],
                                                            train_label1, test_label1)

                        performanceX1[nbt,t,t_] = pfmTT1_
                        
                        if t==t_:
                            performanceW1[nbt,t] = pfmTT1_
                            
                        
                    # permutation null distribution
                    #for npm in range(nPerms):
                                                    
                        #pfmTT1_shuff = clf1.score(test_setP[:,nPCs[0]:nPCs[1],t_],test_label1_shuff)
                        #pfmTT2_shuff = clf2.score(test_setP[:,nPCs[0]:nPCs[1],t_],test_label2_shuff)
                        pfmTT1_shuff = f_decoding.LDAPerformance(train_setP[:,nPCs[0]:nPCs[1],t], test_setP[:,nPCs[0]:nPCs[1],t_],
                                                                    train_label1_shuff, test_label1_shuff)
                        
                        performanceX1_shuff[nbt,t,t_] = pfmTT1_shuff
                        
                        if t==t_:
                            performanceW1_shuff[nbt,t] = pfmTT1_shuff
                            
            
            if subspace == 0:
                performancettX[region] += [np.array(performanceX1)]
                performancettW[region] += [np.array(performanceW1)]
                
                performancettX_shuff[region] += [np.array(performanceX1_shuff)]
                performancettW_shuff[region] += [np.array(performanceW1_shuff)]
                
    
    print(f'tIter = {(time.time() - t_IterOn):.4f}s')
#%% save
np.save(f'{phd_path}/fitting planes/pooled/w&w/performancettX_{monkey_names}_data.npy', performancettX, allow_pickle=True)
np.save(f'{phd_path}/fitting planes/pooled/w&w/performancettW_{monkey_names}_data.npy', performancettW, allow_pickle=True)
np.save(f'{phd_path}/fitting planes/pooled/w&w/performancettX_shuff_{monkey_names}_data.npy', performancettX_shuff, allow_pickle=True)
np.save(f'{phd_path}/fitting planes/pooled/w&w/performancettW_shuff_{monkey_names}_data.npy', performancettW_shuff, allow_pickle=True)
#%% load
performancettX = np.load(f'{phd_path}/outputs/monkeys/' + f'performancettX_{monkey_names}_data.npy', allow_pickle=True).item()
performancettX_shuff = np.load(f'{phd_path}/outputs/monkeys/' + f'performancettX_shuff_{monkey_names}_data.npy', allow_pickle=True).item()
#%% plot cross temp decoding
for region in ('dlpfc','fef'):
    nPCs = nPCs_region[region]
    evrSum = np.array(EVRs[region])[:,nPCs[0]:nPCs[1]].mean(0).sum().round(3)
        
    #vmax = 0.6 if region == 'dlpfc' else 0.8
    fig = plt.figure(figsize=(7, 6), dpi=300)
    
    
    performanceT1 = performancettX[region]
    performanceT1_shuff = performancettX_shuff[region]
    
    pfm1 = np.array(performanceT1).mean(1)
    pfm1_shuff = np.array(performanceT1_shuff).mean(1)#np.concatenate(np.array(performanceT1_shuff),axis=2) #
    
    pvalues1 = np.zeros((len(tbins), len(tbins)))
    for t in range(len(tbins)):
        for t_ in range(len(tbins)):
            #pvalues1[t,t_] = f_stats.permutation_p(pfm1.mean(0)[t,t_], pfm1_shuff[t,t_,:], tail = 'greater')
            #pvalues2[t,t_] = f_stats.permutation_p(pfm2.mean(0)[t,t_], pfm2_shuff[t,t_,:], tail = 'greater')
            
            #pvalues1[t,t_] = stats.ttest_1samp(pfm1[:,t,t_], 0.33333, alternative = 'greater')[1]
            #pvalues2[t,t_] = stats.ttest_1samp(pfm2[:,t,t_], 0.33333, alternative = 'greater')[1]
            
            pvalues1[t,t_] = f_stats.permutation_pCI(pfm1[:,t,t_], pfm1_shuff[:,t,t_],tail='greater',alpha=5)
            
    
    plt.subplot(1,1,1)
    ax = plt.gca()
    sns.heatmap(pd.DataFrame(pfm1.mean(axis = 0), index=tbins,columns=tbins), cmap = 'magma', vmin = 0.25, ax = ax)#vmax = vmax, , vmax = 0.6, xticklabels = 100, yticklabels = 100, vmin = cbar_min, vmax = cbar_max
    
    #from scipy import ndimage
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
    ax.set_xticklabels(['S1', 'S2', 'Go Cue'], rotation=0, fontsize = 15)
    ax.set_xlabel('Test Timebin (ms)', labelpad = 10, fontsize = 20)
    ax.set_yticks([list(tbins).index(i) for i in events])
    ax.set_yticklabels(['S1', 'S2', 'Go Cue'], fontsize = 15,rotation=90)
    ax.set_ylabel('Train Timebin (ms)', labelpad = 10, fontsize = 20)
    
    cbar = ax.collections[0].colorbar
    # here set the labelsize by 20
    cbar.ax.tick_params(labelsize=15)
    
    ax.set_title(f'Trial Type', fontsize = 25, pad = 15)
    
    plt.tight_layout()
    plt.subplots_adjust(top = 1)
    
    #if region=='fef':
    #    if nPCs_region[region]==[0,maxDim_fef]:
    #        plt.suptitle(f'{region.upper()}_{monkey_names}, Full Space (ΣEVR={evrSum})', fontsize = 35, y=1)
    #    else:
    #        plt.suptitle(f'{region.upper()}_{monkey_names}, PC{nPCs_region[region][0]+1}-{nPCs_region[region][1]} (Σ EVR={evrSum})', fontsize = 35, y=1)
        
    #else:
    #    if nPCs_region[region]==[0,10]:
    #    plt.suptitle(f'{region.upper()}_{monkey_names}, Full Space (ΣEVR={evrSum})', fontsize = 35, y=1)
    #    else:
    #        plt.suptitle(f'{region.upper()}_{monkey_names}, PC{nPCs_region[region][0]+1}-{nPCs_region[region][1]} (Σ EVR={evrSum})', fontsize = 35, y=1)
        
    plt.suptitle(f'{region.upper()}, Full Space (ΣEVR={evrSum})', fontsize = 25, y=1.25)
    plt.show()
    
    #fig.savefig(f'{save_path}/decodabilityX_full_{region}_{monkey_names}.tif')
    fig.savefig(f'{phd_path}/outputs/monkeys/decodability_ttX_full_{region}.tif', bbox_inches='tight')

#%%





























#%%
############
# Non-Used #
############

#%%













