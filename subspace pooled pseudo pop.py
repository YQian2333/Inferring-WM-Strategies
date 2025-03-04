# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 20:17:42 2024

@author: aka2333
"""

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

monkey_Info = {'whiskey': whiskey_Info,'wangzi': wangzi_Info, }# 

monkey_names = 'all'#list(monkey_Info.keys())[0]


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



# In[] count for minimal number of trial per locComb across all sessions


trialCounts = {s:[] for s in subConditions}

for session in sessions:
    
    cells = [cc for cc in cellsToUse if '/'.join(cc.split('/')[:-3]) == session]
    
    #session=session_paths[0]
    trial_df = Get_trial(session).trial_selection_df()
    trial_df = trial_df[(trial_df.accuracy == 1)].reset_index(drop = True)
    
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


# In[] decode from pseudo population
pd.options.mode.chained_assignment = None
epsilon = 0.0000001
# In[] decodability with/without permutation P value
bins = 50 # dt #
tslice = (-300,2700)
tsliceRange = np.arange(-300,2700,dt)
slice_epochsDic = {'bsl':[-300,0],'s1':[0,300],'d1':[300,1300],'s2':[1300,1600],'d2':[1600,2600],'go':[2600,3000]}



# In[] choice plan vectors
vecs_C = np.load(f'{phd_path}/fitting planes/pooled/w&w/' + 'vecs_C_detrended.npy', allow_pickle=True).item() #
projs_C = np.load(f'{phd_path}/fitting planes/pooled/w&w/' + 'projs_C_detrended.npy', allow_pickle=True).item() #
projsAll_C = np.load(f'{phd_path}/fitting planes/pooled/w&w/' + 'projsAll_C_detrended.npy', allow_pickle=True).item() #
#Xs_mean_C = np.load(f'{phd_path}/fitting planes/pooled/w&w/' + 'Xs_mean_C_detrended.npy', allow_pickle=True).item() #
trialInfos_C = np.load(f'{phd_path}/fitting planes/pooled/w&w/' + 'trialInfos_C_detrended.npy', allow_pickle=True).item() #
data_3pc_C = np.load(f'{phd_path}/fitting planes/pooled/w&w/' + 'data_3pc_C_detrended.npy', allow_pickle=True).item() #
pca1s_C = np.load(f'{phd_path}/fitting planes/pooled/w&w/' + 'pca1s_C_detrended.npy', allow_pickle=True).item() #

vecs_C_shuff = np.load(f'{phd_path}/fitting planes/pooled/w&w/' + 'vecs_C_shuff_detrended.npy', allow_pickle=True).item() #
projs_C_shuff = np.load(f'{phd_path}/fitting planes/pooled/w&w/' + 'projs_C_shuff_detrended.npy', allow_pickle=True).item() #_detrended
projsAll_C_shuff = np.load(f'{phd_path}/fitting planes/pooled/w&w/' + 'projsAll_C_shuff_detrended.npy', allow_pickle=True).item() #_detrended
#Xs_mean_C_shuff = np.load(f'{phd_path}/fitting planes/pooled/w&w/' + 'Xs_mean_C_shuff_detrended.npy', allow_pickle=True).item() #_detrended
trialInfos_C_shuff = np.load(f'{phd_path}/fitting planes/pooled/w&w/' + 'trialInfos_C_shuff_detrended.npy', allow_pickle=True).item() #_detrended
data_3pc_C_shuff = np.load(f'{phd_path}/fitting planes/pooled/w&w/' + 'data_3pc_C_shuff_detrended.npy', allow_pickle=True).item() #_detrended
pca1s_C_shuff = np.load(f'{phd_path}/fitting planes/pooled/w&w/' + 'pca1s_C_shuff_detrended.npy', allow_pickle=True).item() #_detrended


# In[]

#####################################
######### subspace analysis #########
#####################################


# In[] consistent pca across time
nIters = 100
nPerms = 100
nBoots = 1
fracBoots = 1.0

tBsl = (-300,0)
idxxBsl = [tsliceRange.tolist().index(tBsl[0]), tsliceRange.tolist().index(tBsl[1])]

checkpoints = [150, 550, 1050, 1450, 1850, 2350, 2800]#
#avgInterval = 50
avgInterval = {150:150, 550:250, 1050:250, 1450:150, 1850:250, 2350:250, 2800:200}
checkpointsLabels = ['S1','ED1','LD2','S2','ED2','LD2', 'Go']
toPlot=False # 
#decode_method = 'omega2' #'polyArea''lda' 
avgMethod='conditional_time' # 'conditional' 'all' 'none'

#shuff_excludeInv = True
# In[]
vecs = {}
projs = {}
projsAll = {}
#Xs_mean = {}
trialInfos = {}
pca1s = {}
#pca2s = {}

vecs_shuff = {}
projs_shuff = {}
projsAll_shuff = {}
#Xs_mean_shuff = {}
trialInfos_shuff = {}
pca1s_shuff = {}
#pca2s_shuff = {}

for region in ('dlpfc','fef'):
    vecs[region] = {}
    projs[region] = {}
    projsAll[region] = {}
    #Xs_mean[region] = {}
    trialInfos[region] = {}
    pca1s[region] = []
    #pca2s[region] = {}
    
    vecs_shuff[region] = {}
    projs_shuff[region] = {}
    projsAll_shuff[region] = {}
    #Xs_mean_shuff[region] = {}
    trialInfos_shuff[region] = {}
    pca1s_shuff[region] = []
    #pca2s_shuff[region] = {}
    
    for tt in ttypes:
        trialInfos[region][tt] = []
        trialInfos_shuff[region][tt] = []
        
        
    
    for cp in checkpoints:
        vecs[region][cp] = {}
        projs[region][cp] = {}
        projsAll[region][cp] = {}
        #Xs_mean[region][cp] = {}
        #pca2s[region][cp] = {}

        vecs_shuff[region][cp] = {}
        projs_shuff[region][cp] = {}
        projsAll_shuff[region][cp] = {}
        #Xs_mean_shuff[region][cp] = {}
        #pca2s_shuff[region][cp] = {}
        
        for tt in ttypes:
            vecs[region][cp][tt] = {1:[], 2:[]}
            projs[region][cp][tt] = {1:[], 2:[]}
            projsAll[region][cp][tt] = {1:[], 2:[]}
            #Xs_mean[region][cp][tt] = {1:[], 2:[]}
            #pca2s[region][cp][tt] = {1:[], 2:[]}
            

            vecs_shuff[region][cp][tt] = {1:[], 2:[]}
            projs_shuff[region][cp][tt] = {1:[], 2:[]}
            projsAll_shuff[region][cp][tt] = {1:[], 2:[]}
            #Xs_mean_shuff[region][cp][tt] = {1:[], 2:[]}
            #pca2s_shuff[region][cp][tt] = {1:[], 2:[]}
            
    

# In[]

#n = 50
#while n < 100:
for n in range(nIters):
    t_IterOn = time.time()
    print(f'Iter = {n}')
    
    #pseudo_TrialInfo = f_pseudoPop.pseudo_Session(locCombs, ttypes, samplePerCon=samplePerCon, sampleRounds=sampleRounds)
    #pseudo_region = f_pseudoPop.pseudo_PopByRegion_pooled(sessions, cellsToUse, monkey_Info, locCombs = locCombs, ttypes = ttypes, samplePerCon = samplePerCon, sampleRounds=sampleRounds, arti_noise_level = arti_noise_level)
    
    #pseudo_data = {'pseudo_TrialInfo':pseudo_TrialInfo, 'pseudo_region':pseudo_region}
    #np.save(save_path + f'/pseudo_data{n}.npy', pseudo_data, allow_pickle=True)

    pseudo_data = np.load(save_path + f'/pseudo_all{n}.npy', allow_pickle=True).item()
    pseudo_region = pseudo_data['pseudo_region']
    pseudo_TrialInfo = pseudo_data['pseudo_TrialInfo']

    
    print(f'tIter = {(time.time() - t_IterOn):.4f}s')
    #n += 1
    
    for region in ('dlpfc','fef'):
        
        
        trialInfo = pseudo_TrialInfo.reset_index(names=['id'])
        
        idx1 = trialInfo.id # original index
        #idx2 = trialInfo.index.to_list() # reset index
        
        # if specify applied time window of pca
        pca_tWin = np.hstack((np.arange(300,1300,dt, dtype = int),np.arange(1600,2600,dt, dtype = int))).tolist() #
        pca_tWinX = [tsliceRange.tolist().index(i) for i in pca_tWin]
        
        ### main test
        
        dataN = pseudo_region[region][idx1,::]
        
        # if detrend by subtract avg
        for ch in range(dataN.shape[1]):
            temp = dataN[:,ch,:]
            dataN[:,ch,:] = (temp - temp.mean(axis=0))#/(temp.std(axis = 0) + epsilon) # to avoid divid by 0 issue
            
        #X_region = pseudo_PopT
        
        
        for ch in range(dataN.shape[1]):
            #dataN[:,ch,:] = scale(dataN[:,ch,:])#01 scale for each channel
            dataN[:,ch,:] = (dataN[:,ch,:] - dataN[:,ch,:].mean()) / dataN[:,ch,:].std() #standard scaler
            #dataN[:,ch,:] = (dataN[:,ch,:] - dataN[:,ch,:].mean(axis=0)) / dataN[:,ch,:].std(axis=0) # detrened standard scaler
            #dataN[:,ch,:] = (dataN[:,ch,:] - dataN[:,ch,idxxBsl[0]:idxxBsl[1]].mean())/dataN[:,ch,idxxBsl[0]:idxxBsl[1]].std() # Z to baseline, 
        
        # baseline z-normalize data (if not normalized)
        #for ch in range(dataN.shape[1]):
        
            #dataN[:,ch,:] = scale(dataN[:,ch,:])
        
        pca1s[region].append([])
        pca1s_shuff[region].append([])
        
        for tt in ttypes:
            trialInfos[region][tt].append([])
            trialInfos_shuff[region][tt].append([])
            
            
        for cp in checkpoints:
            for tt in ttypes:
                for ll in (1,2,):
                    vecs[region][cp][tt][ll].append([])
                    projs[region][cp][tt][ll].append([])
                    projsAll[region][cp][tt][ll].append([])
                    #Xs_mean[region][cp][tt][ll].append([])
                    #pca2s[region][cp][tt][ll].append([])
                    
                    vecs_shuff[region][cp][tt][ll].append([])
                    projs_shuff[region][cp][tt][ll].append([])
                    projsAll_shuff[region][cp][tt][ll].append([])
                    #Xs_mean_shuff[region][cp][tt][ll].append([])
                    #pca2s_shuff[region][cp][tt][ll].append([])
        
        for nboot in range(nBoots):
            #idxT,_ = f_subspace.split_set(dataN, frac=fracBoots, ranseed=nboot)
            idxT,_ = f_subspace.split_set(np.arange(dataN.shape[0]), frac=fracBoots, ranseed=nboot)
            dataT = dataN[idxT,:,:]
            trialInfoT = trialInfo.loc[idxT,:].reset_index(drop=True)

            pca1_C = pca1s_C[region][n][nboot]
            
            vecs_D, projs_D, projsAll_D, _, trialInfos_D, _, _, evr_1st, pca1 = f_subspace.plane_fitting_analysis(dataT, trialInfoT, pca_tWinX, checkpoints, tsliceRange, avgInterval, locs, ttypes, dropCombs, 
                                                                                                                  toPlot=toPlot, avgMethod = avgMethod, adaptPCA=pca1_C) #, decode_method = decode_method
            #decodability_projD
            
            pca1s[region][n] += [pca1]
            
            for tt in ttypes:
                trialInfos[region][tt][n] += [trialInfos_D[tt]]
                
            
            for cp in checkpoints:
                for tt in ttypes:
                    for ll in (1,2,):
                        vecs[region][cp][tt][ll][n] += [vecs_D[cp][tt][ll]]
                        projs[region][cp][tt][ll][n] += [projs_D[cp][tt][ll]]
                        projsAll[region][cp][tt][ll][n] += [projsAll_D[cp][tt][ll]]
                        #Xs_mean[region][cp][tt][ll][n] += [X_mean[cp][tt][ll]]
                        #pca2s[region][cp][tt][ll][n] += [pca2]
                        
            
            print(f'EVRs: {evr_1st.round(5)}')
            
            
            for nperm in range(nPerms):
                
                # NOT USING THIS METHOD BECAUSE OF NO GUARANTEE FOR EQUAL TRIAL SIZE FOR EACH CONDITION
                #if shuff_excludeInv:
                #    trialInfoT_shuff = trialInfoT.copy()
                #    Y_columnsLabels = trialInfoT.columns.tolist()
                #    loc1Labels_Inv, loc2Labels_Inv = trialInfoT_shuff.loc1.values, trialInfoT_shuff.loc2.values
                #    
                #    for i in range(len(trialInfoT)):
                #        l1,l2 = trialInfoT.loc1.values[i], trialInfoT.loc2.values[i]
                #        
                #        l1_inv = np.random.choice(np.array(locs)[np.array(locs)!=l2]).astype(int) # shuff l1 labels that other than true l2
                #        l2_inv = np.random.choice(np.array(locs)[np.array(locs)!=l1][np.array(locs)[np.array(locs)!=l1] != l1_inv]).astype(int)
                #        loc1Labels_Inv[i] = l1_inv
                #        loc2Labels_Inv[i] = l2_inv # shuff l2 labels that other than true l1
                #    
                #    trialInfoT_shuff.loc1 = loc1Labels_Inv
                #    trialInfoT_shuff.loc2 = loc2Labels_Inv
                #    trialInfoT_shuff.locs = trialInfoT_shuff.loc1.astype(str) + '_' + trialInfoT_shuff.loc2.astype(str)
                #    
                #    for i in range(len(trialInfoT_shuff)):
                #        trialInfoT_shuff.locKey[i] = trialInfoT_shuff.loc2[i] if trialInfoT_shuff.type[i] == 1 else trialInfoT_shuff.loc1[i]
                #        trialInfoT_shuff.locX[i] = trialInfoT_shuff.loc1[i] if trialInfoT_shuff.type[i] == 1 else trialInfoT_shuff.loc2[i]
                #    
                #else:
                #    trialInfoT_shuff = trialInfoT.copy().sample(frac=1).reset_index(drop=True) # default method
                
                trialInfoT_shuff = trialInfoT.copy().sample(frac=1).reset_index(drop=True) # just use default method
                
                vecs_D_shuff, projs_D_shuff, projsAll_D_shuff, _, trialInfos_D_shuff, _, _, _, pca1_shuff = f_subspace.plane_fitting_analysis(dataT, trialInfoT_shuff, pca_tWinX, checkpoints, tsliceRange, avgInterval, locs, ttypes, dropCombs, 
                                                                                                                                              toPlot=toPlot, avgMethod = avgMethod, adaptPCA=pca1_C) #, decode_method = decode_method 
                
                #decodability_projD_shuff
                
                pca1s_shuff[region][n] += [pca1_shuff]
                
                for tt in ttypes:
                    trialInfos_shuff[region][tt][n] += [trialInfos_D_shuff[tt]]
                    
                
                for cp in checkpoints:
                    for tt in ttypes:
                        
                        for ll in (1,2,):
                            vecs_shuff[region][cp][tt][ll][n] += [vecs_D_shuff[cp][tt][ll]]
                            projs_shuff[region][cp][tt][ll][n] += [projs_D_shuff[cp][tt][ll]]
                            projsAll_shuff[region][cp][tt][ll][n] += [projsAll_D_shuff[cp][tt][ll]]
                            #Xs_mean_shuff[region][cp][tt][ll][n] += [X_mean_shuff[cp][tt][ll]]
                            #pca2s_shuff[region][cp][tt][ll][n] += [pca2_shuff]
                            
            
            
        print(f'tPerm({nPerms*nBoots}) = {(time.time() - t_IterOn):.4f}s, {region}')
        
        #n += 1


# In[]
np.save(f'{phd_path}/fitting planes/pooled/w&w/' + 'vecs_detrended.npy', vecs, allow_pickle=True)
np.save(f'{phd_path}/fitting planes/pooled/w&w/' + 'projs_detrended.npy', projs, allow_pickle=True)
np.save(f'{phd_path}/fitting planes/pooled/w&w/' + 'projsAll_detrended.npy', projsAll, allow_pickle=True)
#np.save(f'{phd_path}/fitting planes/pooled/w&w/' + 'Xs_mean_detrended.npy', Xs_mean, allow_pickle=True)
np.save(f'{phd_path}/fitting planes/pooled/w&w/' + 'trialInfos_detrended.npy', trialInfos, allow_pickle=True)
np.save(f'{phd_path}/fitting planes/pooled/w&w/' + 'pca1s_detrended.npy', pca1s, allow_pickle=True)
#np.save(f'{phd_path}/fitting planes/pooled/w&w/' + 'pca2s_detrended.npy', pca2s, allow_pickle=True)

np.save(f'{phd_path}/fitting planes/pooled/w&w/' + 'vecs_shuff_detrended.npy', vecs_shuff, allow_pickle=True)
np.save(f'{phd_path}/fitting planes/pooled/w&w/' + 'projs_shuff_detrended.npy', projs_shuff, allow_pickle=True)
np.save(f'{phd_path}/fitting planes/pooled/w&w/' + 'projsAll_shuff_detrended.npy', projsAll_shuff, allow_pickle=True)
#np.save(f'{phd_path}/fitting planes/pooled/w&w/' + 'Xs_mean_shuff_detrended.npy', Xs_mean_shuff, allow_pickle=True)
np.save(f'{phd_path}/fitting planes/pooled/w&w/' + 'trialInfos_shuff_detrended.npy', trialInfos_shuff, allow_pickle=True)
np.save(f'{phd_path}/fitting planes/pooled/w&w/' + 'pca1s_shuff_detrended.npy', pca1s_shuff, allow_pickle=True)
#np.save(f'{phd_path}/fitting planes/pooled/w&w/' + 'pca2s_shuff_detrended.npy', pca2s_shuff, allow_pickle=True)

# In[]
vecs = np.load(f'{phd_path}/fitting planes/pooled/w&w/' + 'vecs_detrended.npy', allow_pickle=True).item() #
projs = np.load(f'{phd_path}/fitting planes/pooled/w&w/' + 'projs_detrended.npy', allow_pickle=True).item() #
projsAll = np.load(f'{phd_path}/fitting planes/pooled/w&w/' + 'projsAll_detrended.npy', allow_pickle=True).item() #
#Xs_mean = np.load(f'{phd_path}/fitting planes/pooled/w&w/' + 'Xs_mean_detrended.npy', allow_pickle=True).item() #
trialInfos = np.load(f'{phd_path}/fitting planes/pooled/w&w/' + 'trialInfos_detrended.npy', allow_pickle=True).item() #
pca1s = np.load(f'{phd_path}/fitting planes/pooled/w&w/' + 'pca1s_detrended.npy', allow_pickle=True).item() #
#pca2s = np.load(f'{phd_path}/fitting planes/pooled/w&w/' + 'pca2s_detrended.npy', allow_pickle=True).item() #

vecs_shuff = np.load(f'{phd_path}/fitting planes/pooled/w&w/' + 'vecs_shuff_detrended.npy', allow_pickle=True).item() #
projs_shuff = np.load(f'{phd_path}/fitting planes/pooled/w&w/' + 'projs_shuff_detrended.npy', allow_pickle=True).item() #
projsAll_shuff = np.load(f'{phd_path}/fitting planes/pooled/w&w/' + 'projsAll_shuff_detrended.npy', allow_pickle=True).item() #
#Xs_mean_shuff = np.load(f'{phd_path}/fitting planes/pooled/w&w/' + 'Xs_mean_shuff_detrended.npy', allow_pickle=True).item() #
trialInfos_shuff = np.load(f'{phd_path}/fitting planes/pooled/w&w/' + 'trialInfos_shuff_detrended.npy', allow_pickle=True).item() #
pca1s_shuff = np.load(f'{phd_path}/fitting planes/pooled/w&w/' + 'pca1s_shuff_detrended.npy', allow_pickle=True).item() #
#pca2s_shuff = np.load(f'{phd_path}/fitting planes/pooled/w&w/' + 'pca2s_shuff_detrended.npy', allow_pickle=True).item() #

# In[] exclude post-gocue window
checkpoints = [150, 550, 1050, 1450, 1850, 2350]#
checkpointsLabels = ['S1','ED1','LD1','S2','ED2','LD2']

#########################

# In[]
from matplotlib.colors import LinearSegmentedColormap

def create_my_cmap(color_seq, cmap_name):
    seg_points = np.linspace(0,1,len(color_seq))
    colors = []
    for i in range(len(seg_points)):
      colors += [(seg_points[i], color_seq[i])]
      #[(0, cmin),(0.5, cmid), (1, cmax)]
    cmap = LinearSegmentedColormap.from_list(cmap_name, colors)
    return cmap

red_white_blue = create_my_cmap(['#34768c', '#f6f9e4', '#de2723'], 'red_white_blue')
tint_jet = create_my_cmap(['#2d95e4', '#b3e5ef', '#ffdc87', '#ff7f46'], 'tint_jet')
mint_white_pink = create_my_cmap(('#22c67d', '#EEEEEE', '#e569a0'), 'mint_white_pink')


def rectify_angle(angle, mean_angle):
    # Compute the difference between the angle and the mean angle
    diff = angle - mean_angle
    
    # Rectify the angle to fall within the range [-pi, pi]
    rectified_angle = diff % (2 * np.pi)
    if rectified_angle > np.pi:
        rectified_angle -= 2 * np.pi
    return rectified_angle + mean_angle

def rectify_bimodal_angles(angles):
    # Compute the mean angle of the bimodal distribution
    mean_angle = cstats.descriptive.mean(angles) # np.mean
    
    # Rectify each angle
    rectified_angles = [rectify_angle(angle, mean_angle) for angle in angles]
    
    return np.array(rectified_angles)

#%%                    


#######################
# compare item v item #
#######################


# In[] item v item cosTheta, cosPsi, sse. Compare within type, between time points, between locations
pdummy = True #False #

cosTheta_11, cosTheta_12, cosTheta_22 = {},{},{}
cosPsi_11, cosPsi_12, cosPsi_22 = {},{},{}
#sse_11, sse_12, sse_22 = {},{},{}
cosTheta_11_shuff, cosTheta_12_shuff, cosTheta_22_shuff = {},{},{}
cosPsi_11_shuff, cosPsi_12_shuff, cosPsi_22_shuff = {},{},{}

for region in ('dlpfc','fef'):
    
    cosTheta_11[region], cosTheta_22[region], cosTheta_12[region] = {},{},{}
    cosPsi_11[region], cosPsi_22[region], cosPsi_12[region] = {},{},{}
    
    cosTheta_11_shuff[region], cosTheta_22_shuff[region], cosTheta_12_shuff[region] = {},{},{}
    cosPsi_11_shuff[region], cosPsi_22_shuff[region], cosPsi_12_shuff[region] = {},{},{}
    
    for tt in ttypes:
        ttype = 'Retarget' if tt==1 else 'Distraction'
        
        cosTheta_11T = np.zeros((nIters, nBoots, len(checkpoints), len(checkpoints)))
        cosTheta_22T = np.zeros((nIters, nBoots, len(checkpoints), len(checkpoints)))
        cosTheta_12T = np.zeros((nIters, nBoots, len(checkpoints), len(checkpoints)))
        
        cosPsi_11T = np.zeros((nIters, nBoots, len(checkpoints), len(checkpoints)))
        cosPsi_22T = np.zeros((nIters, nBoots, len(checkpoints), len(checkpoints)))
        cosPsi_12T = np.zeros((nIters, nBoots, len(checkpoints), len(checkpoints)))
        
        
        cosTheta_11T_shuff = np.zeros((nIters, nPerms, len(checkpoints), len(checkpoints)))
        cosTheta_22T_shuff = np.zeros((nIters, nPerms, len(checkpoints), len(checkpoints)))
        cosTheta_12T_shuff = np.zeros((nIters, nPerms, len(checkpoints), len(checkpoints)))
        
        cosPsi_11T_shuff = np.zeros((nIters, nPerms, len(checkpoints), len(checkpoints)))
        cosPsi_22T_shuff = np.zeros((nIters, nPerms, len(checkpoints), len(checkpoints)))
        cosPsi_12T_shuff = np.zeros((nIters, nPerms, len(checkpoints), len(checkpoints)))
        #sse_11T = np.zeros((nIters, nBoots, len(checkpoints), len(checkpoints)))
        #sse_22T = np.zeros((nIters, nBoots, len(checkpoints), len(checkpoints)))
        #sse_12T = np.zeros((nIters, nBoots, len(checkpoints), len(checkpoints)))
        
        #theta_11T = np.zeros((nIters, nBoots, len(checkpoints), len(checkpoints)))
        #theta_22T = np.zeros((nIters, nBoots, len(checkpoints), len(checkpoints)))
        #theta_12T = np.zeros((nIters, nBoots, len(checkpoints), len(checkpoints)))
        
        #psi_11T = np.zeros((nIters, nBoots, len(checkpoints), len(checkpoints)))
        #psi_22T = np.zeros((nIters, nBoots, len(checkpoints), len(checkpoints)))
        #psi_12T = np.zeros((nIters, nBoots, len(checkpoints), len(checkpoints)))
        
        for n in range(nIters):
            for nbt in range(nBoots):
                for nc, cp in enumerate(checkpoints):
                    for nc_, cp_ in enumerate(checkpoints):
                        cT11, _, cP11, _, _ = f_subspace.angle_alignment_coplanar(vecs[region][cp][tt][1][n][nbt], projs[region][cp][tt][1][n][nbt], vecs[region][cp_][tt][1][n][nbt], projs[region][cp_][tt][1][n][nbt])
                        cT22, _, cP22, _, _ = f_subspace.angle_alignment_coplanar(vecs[region][cp][tt][2][n][nbt], projs[region][cp][tt][2][n][nbt], vecs[region][cp_][tt][2][n][nbt], projs[region][cp_][tt][2][n][nbt])
                        cT12, _, cP12, _, _ = f_subspace.angle_alignment_coplanar(vecs[region][cp][tt][1][n][nbt], projs[region][cp][tt][1][n][nbt], vecs[region][cp_][tt][2][n][nbt], projs[region][cp_][tt][2][n][nbt])
                        
                        #t11, t22, t12 = f_subspace.angle_by_cossin(cT11, sT11), f_subspace.angle_by_cossin(cT22, sT22), f_subspace.angle_by_cossin(cT12, sT12) #, angle_range=(0,2*np.pi)
                        #p11, p22, p12 = f_subspace.angle_by_cossin(cP11, sP11), f_subspace.angle_by_cossin(cP22, sP22), f_subspace.angle_by_cossin(cP12, sP12) #, angle_range=(0,2*np.pi)
                        
                        cosTheta_11T[n,nbt,nc,nc_], cosTheta_22T[n,nbt,nc,nc_], cosTheta_12T[n,nbt,nc,nc_] = cT11, cT22, cT12# theta11, theta22, theta12# 
                        cosPsi_11T[n,nbt,nc,nc_], cosPsi_22T[n,nbt,nc,nc_], cosPsi_12T[n,nbt,nc,nc_] = cP11, cP22, cP12# psi11, psi22, psi12# 
                        #sse_11T[n,nbt,nc,nc_], sse_22T[n,nbt,nc,nc_], sse_12T[n,nbt,nc,nc_] = s11, s22, s12
                        
                        #theta_11T[n,nbt,nc,nc_], theta_22T[n,nbt,nc,nc_], theta_12T[n,nbt,nc,nc_] = t11, t22, t12
                        #psi_11T[n,nbt,nc,nc_], psi_22T[n,nbt,nc,nc_], psi_12T[n,nbt,nc,nc_] = p11, p22, p12
                        #cosTheta_11T[n,nbt,nc,nc_], cosTheta_22T[n,nbt,nc,nc_], cosTheta_12T[n,nbt,nc,nc_] = np.degrees(t11), np.degrees(t22), np.degrees(t12)# theta11, theta22, theta12# 
                        #cosPsi_11T[n,nbt,nc,nc_], cosPsi_22T[n,nbt,nc,nc_], cosPsi_12T[n,nbt,nc,nc_] = np.degrees(p11), np.degrees(p22), np.degrees(p12)# psi11, psi22, psi12# 
                
            for npm in range(nPerms):
                for nc, cp in enumerate(checkpoints):
                    for nc_, cp_ in enumerate(checkpoints):
                        cT11_shuff, _, cP11_shuff, _, _ = f_subspace.angle_alignment_coplanar(vecs_shuff[region][cp][tt][1][n][npm], projs_shuff[region][cp][tt][1][n][npm], vecs_shuff[region][cp_][tt][1][n][npm], projs_shuff[region][cp_][tt][1][n][npm])
                        cT22_shuff, _, cP22_shuff, _, _ = f_subspace.angle_alignment_coplanar(vecs_shuff[region][cp][tt][2][n][npm], projs_shuff[region][cp][tt][2][n][npm], vecs_shuff[region][cp_][tt][2][n][npm], projs_shuff[region][cp_][tt][2][n][npm])
                        cT12_shuff, _, cP12_shuff, _, _ = f_subspace.angle_alignment_coplanar(vecs_shuff[region][cp][tt][1][n][npm], projs_shuff[region][cp][tt][1][n][npm], vecs_shuff[region][cp_][tt][2][n][npm], projs_shuff[region][cp_][tt][2][n][npm])
                        
                        cosTheta_11T_shuff[n,npm,nc,nc_], cosTheta_22T_shuff[n,npm,nc,nc_], cosTheta_12T_shuff[n,npm,nc,nc_] = cT11_shuff, cT22_shuff, cT12_shuff# theta11, theta22, theta12# 
                        cosPsi_11T_shuff[n,npm,nc,nc_], cosPsi_22T_shuff[n,npm,nc,nc_], cosPsi_12T_shuff[n,npm,nc,nc_] = cP11_shuff, cP22_shuff, cP12_shuff# psi11, psi22, psi12# 
                        
        
        cosTheta_11[region][tt] = cosTheta_11T
        cosTheta_22[region][tt] = cosTheta_22T
        cosTheta_12[region][tt] = cosTheta_12T
        
        cosPsi_11[region][tt] = cosPsi_11T
        cosPsi_22[region][tt] = cosPsi_22T
        cosPsi_12[region][tt] = cosPsi_12T
        
        cosTheta_11_shuff[region][tt] = cosTheta_11T_shuff
        cosTheta_22_shuff[region][tt] = cosTheta_22T_shuff
        cosTheta_12_shuff[region][tt] = cosTheta_12T_shuff
        
        cosPsi_11_shuff[region][tt] = cosPsi_11T_shuff
        cosPsi_22_shuff[region][tt] = cosPsi_22T_shuff
        cosPsi_12_shuff[region][tt] = cosPsi_12T_shuff
        
#%%  plot      
for region in ('dlpfc','fef'):
    
    for tt in ttypes:
        ttype = 'Retarget' if tt==1 else 'Distraction'
        
        pcosTheta_11T, pcosTheta_22T, pcosTheta_12T = np.ones((len(checkpoints), len(checkpoints))), np.ones((len(checkpoints), len(checkpoints))), np.ones((len(checkpoints), len(checkpoints)))
        pcosPsi_11T, pcosPsi_22T, pcosPsi_12T = np.ones((len(checkpoints), len(checkpoints))), np.ones((len(checkpoints), len(checkpoints))), np.ones((len(checkpoints), len(checkpoints)))
        #psse_11T, psse_22T, psse_12T = np.ones((len(checkpoints), len(checkpoints))), np.ones((len(checkpoints), len(checkpoints))), np.ones((len(checkpoints), len(checkpoints)))
        
        if pdummy == False:
            
            cosTheta_11_shuffT = np.zeros((nIters, nBoots*nPerms, len(checkpoints), len(checkpoints)))
            cosTheta_22_shuffT = np.zeros((nIters, nBoots*nPerms, len(checkpoints), len(checkpoints)))
            cosTheta_12_shuffT = np.zeros((nIters, nBoots*nPerms, len(checkpoints), len(checkpoints)))
            
            cosPsi_11_shuffT = np.zeros((nIters, nBoots*nPerms, len(checkpoints), len(checkpoints)))
            cosPsi_22_shuffT = np.zeros((nIters, nBoots*nPerms, len(checkpoints), len(checkpoints)))
            cosPsi_12_shuffT = np.zeros((nIters, nBoots*nPerms, len(checkpoints), len(checkpoints)))
            
            #sse_11_shuffT = np.zeros((nIters, nBoots*nPerms, len(checkpoints), len(checkpoints)))
            #sse_22_shuffT = np.zeros((nIters, nBoots*nPerms, len(checkpoints), len(checkpoints)))
            #sse_12_shuffT = np.zeros((nIters, nBoots*nPerms, len(checkpoints), len(checkpoints)))
            
            
            for n in range(nIters):
                for nbt in range(nBoots*nPerms):
                    for nc, cp in enumerate(checkpoints):
                        for nc_, cp_ in enumerate(checkpoints):
                            
                            cT11, sT11, cP11, sP11, s11 = f_subspace.angle_alignment_coplanar(vecs_shuff[region][cp][tt][1][n][nbt], projs_shuff[region][cp][tt][1][n][nbt], vecs_shuff[region][cp_][tt][1][n][nbt], projs_shuff[region][cp_][tt][1][n][nbt])
                            cT22, sT22, cP22, sP22, s22 = f_subspace.angle_alignment_coplanar(vecs_shuff[region][cp][tt][2][n][nbt], projs_shuff[region][cp][tt][2][n][nbt], vecs_shuff[region][cp_][tt][2][n][nbt], projs_shuff[region][cp_][tt][2][n][nbt])
                            cT12, sT12, cP12, sP12, s12 = f_subspace.angle_alignment_coplanar(vecs_shuff[region][cp][tt][1][n][nbt], projs_shuff[region][cp][tt][1][n][nbt], vecs_shuff[region][cp_][tt][2][n][nbt], projs_shuff[region][cp_][tt][2][n][nbt])
                            
                            #t11, t22, t12 = f_subspace.angle_by_cossin(cT11, sT11), f_subspace.angle_by_cossin(cT22, sT22), f_subspace.angle_by_cossin(cT12, sT12) #, angle_range=(0,2*np.pi)
                            #p11, p22, p12 = f_subspace.angle_by_cossin(cP11, sP11), f_subspace.angle_by_cossin(cP22, sP22), f_subspace.angle_by_cossin(cP12, sP12) #, angle_range=(0,2*np.pi)
                            
                            
                            cosTheta_11_shuffT[n,nbt,nc,nc_], cosTheta_22_shuffT[n,nbt,nc,nc_], cosTheta_12_shuffT[n,nbt,nc,nc_] = cT11, cT22, cT12
                            cosPsi_11_shuffT[n,nbt,nc,nc_], cosPsi_22_shuffT[n,nbt,nc,nc_], cosPsi_12_shuffT[n,nbt,nc,nc_] = cP11, cP22, cP12
                            #sse_11_shuffT[n,nbt,nc,nc_], sse_22_shuffT[n,nbt,nc,nc_], sse_12_shuffT[n,nbt,nc,nc_] = s11, s22, s12
                            
                            #cosTheta_11_shuffT[n,nbt,nc,nc_], cosTheta_22_shuffT[n,nbt,nc,nc_], cosTheta_12_shuffT[n,nbt,nc,nc_] = np.degrees(t11), np.degrees(t22), np.degrees(t12)
                            #cosPsi_11_shuffT[n,nbt,nc,nc_], cosPsi_22_shuffT[n,nbt,nc,nc_], cosPsi_12_shuffT[n,nbt,nc,nc_] = np.degrees(p11), np.degrees(p22), np.degrees(p12)
                            
            cosTheta_11_shuff_all = np.concatenate([cosTheta_11_shuffT[:,nbt,:,:] for nbt in range(nBoots*nPerms)])
            cosTheta_22_shuff_all = np.concatenate([cosTheta_22_shuffT[:,nbt,:,:] for nbt in range(nBoots*nPerms)])
            cosTheta_12_shuff_all = np.concatenate([cosTheta_12_shuffT[:,nbt,:,:] for nbt in range(nBoots*nPerms)])
            
            cosPsi_11_shuff_all = np.concatenate([cosPsi_11_shuffT[:,nbt,:,:] for nbt in range(nBoots*nPerms)])
            cosPsi_22_shuff_all = np.concatenate([cosPsi_22_shuffT[:,nbt,:,:] for nbt in range(nBoots*nPerms)])
            cosPsi_12_shuff_all = np.concatenate([cosPsi_12_shuffT[:,nbt,:,:] for nbt in range(nBoots*nPerms)])
            
            #sse_11_shuff_all = np.concatenate([sse_11_shuffT[:,nbt,:,:] for nbt in range(nBoots*nPerms)])
            #sse_22_shuff_all = np.concatenate([sse_22_shuffT[:,nbt,:,:] for nbt in range(nBoots*nPerms)])
            #sse_12_shuff_all = np.concatenate([sse_12_shuffT[:,nbt,:,:] for nbt in range(nBoots*nPerms)])
            
            
            
            for i in range(len(checkpoints)):
                for j in range(len(checkpoints)):
                    
                    cT11, cT22, cT12 = cosTheta_11T.mean(axis=1)[:,i,j].round(5), cosTheta_22T.mean(axis=1)[:,i,j].round(5), cosTheta_12T.mean(axis=1)[:,i,j].round(5)
                    cP11, cP22, cP12 = cosPsi_11T.mean(axis=1)[:,i,j].round(5), cosPsi_22T.mean(axis=1)[:,i,j].round(5), cosPsi_12T.mean(axis=1)[:,i,j].round(5)
                    
                    #s11, s22, s12 = sse_11T.mean(axis=1)[:,i,j].round(5), sse_22T.mean(axis=1)[:,i,j].round(5), sse_12T.mean(axis=1)[:,i,j].round(5)
                    
                    #t11, t22, t12 = theta_11T.mean(axis=1)[:,i,j].round(5), theta_22T.mean(axis=1)[:,i,j].round(5), theta_12T.mean(axis=1)[:,i,j].round(5)
                    #p11, p22, p12 = psi_11T.mean(axis=1)[:,i,j].round(5), psi_22T.mean(axis=1)[:,i,j].round(5), psi_12T.mean(axis=1)[:,i,j].round(5)
                    
                    # drop nan values
                    #cP11, cP22, cP12 = cP11[~np.isnan(cP11)], cP22[~np.isnan(cP22)], cP12[~np.isnan(cP12)]
                    #s11, s22, s12 = s11[~np.isnan(s11)], s22[~np.isnan(s22)], s12[~np.isnan(s12)]
                    
                    # shuff distribution
                    cT11_shuff, cT22_shuff, cT12_shuff = cosTheta_11_shuff_all[:,i,j].round(5), cosTheta_22_shuff_all[:,i,j].round(5), cosTheta_12_shuff_all[:,i,j].round(5)
                    cP11_shuff, cP22_shuff, cP12_shuff = cosPsi_11_shuff_all[:,i,j].round(5), cosPsi_22_shuff_all[:,i,j].round(5), cosPsi_12_shuff_all[:,i,j].round(5)
                    #s11_shuff, s22_shuff, s12_shuff = sse_11_shuff_all[:,i,j].round(5), sse_22_shuff_all[:,i,j].round(5), sse_12_shuff_all[:,i,j].round(5)
                    
                    #cP11_shuff, cP22_shuff, cP12_shuff = cP11_shuff[~np.isnan(cP11_shuff)], cP22_shuff[~np.isnan(cP22_shuff)], cP12_shuff[~np.isnan(cP12_shuff)]
                    #s11_shuff, s22_shuff, s12_shuff = s11_shuff[~np.isnan(s11_shuff)], s22_shuff[~np.isnan(s22_shuff)], s12_shuff[~np.isnan(s12_shuff)]
                    
                    # compare distributions and calculate p values
                    #pcosTheta_11T[i,j] = cstats.rayleigh(t11)[0] # cstats.vtest(t11,cstats.descriptive.mean(t11))[0] #
                    #pcosTheta_22T[i,j] = cstats.rayleigh(t22)[0] # cstats.vtest(t22,cstats.descriptive.mean(t22))[0] #
                    #pcosTheta_12T[i,j] = cstats.rayleigh(t12)[0] # cstats.vtest(t12,cstats.descriptive.mean(t12))[0] #
                    
                    #pcosPsi_11T[i,j] = cstats.rayleigh(p11)[0] # cstats.vtest(p11,cstats.descriptive.mean(p11))[0] #
                    #pcosPsi_22T[i,j] = cstats.rayleigh(p22)[0] # cstats.vtest(p22,cstats.descriptive.mean(p22))[0] #
                    #pcosPsi_12T[i,j] = cstats.rayleigh(p12)[0] # cstats.vtest(p12,cstats.descriptive.mean(p12))[0] #
                    
                    #psse_11T[i,j] = f_stats.permutation_p(s11.mean(axis=0), s11_shuff)
                    #psse_22T[i,j] = f_stats.permutation_p(s22.mean(axis=0), s22_shuff)
                    #psse_12T[i,j] = f_stats.permutation_p(s12.mean(axis=0), s12_shuff)
                    
                    ##################
                    # permutation ps #
                    ##################
                    
                    #pcosTheta_11T[i,j] = f_stats.permutation_p(cT11.mean(axis=0), cT11_shuff)#/2
                    #pcosTheta_22T[i,j] = f_stats.permutation_p(cT22.mean(axis=0), cT22_shuff)#/2
                    #pcosTheta_12T[i,j] = f_stats.permutation_p(cT12.mean(axis=0), cT12_shuff)#/2
                    
                    #pcosPsi_11T[i,j] = f_stats.permutation_p(cP11.mean(axis=0), cP11_shuff)#/2
                    #pcosPsi_22T[i,j] = f_stats.permutation_p(cP22.mean(axis=0), cP22_shuff)#/2
                    #pcosPsi_12T[i,j] = f_stats.permutation_p(cP12.mean(axis=0), cP12_shuff)#/2
                    
                    #psse_11T[i,j] = f_stats.permutation_p(s11.mean(axis=0), s11_shuff)
                    #psse_22T[i,j] = f_stats.permutation_p(s22.mean(axis=0), s22_shuff)
                    #psse_12T[i,j] = f_stats.permutation_p(s12.mean(axis=0), s12_shuff)
                    
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
                    
                    pcosTheta_11T[i,j] = stats.kstest(cT11, cT11_shuff)[-1] # # stats.uniform.cdf
                    pcosTheta_22T[i,j] = stats.kstest(cT22, cT22_shuff)[-1] # # stats.uniform.cdf
                    pcosTheta_12T[i,j] = stats.kstest(cT12, cT12_shuff)[-1] # # stats.uniform.cdf
                    
                    pcosPsi_11T[i,j] = stats.kstest(cP11, cP11_shuff)[-1] # #stats.uniform.cdf
                    pcosPsi_22T[i,j] = stats.kstest(cP22, cP22_shuff)[-1] # #stats.uniform.cdf
                    pcosPsi_12T[i,j] = stats.kstest(cP12, cP12_shuff)[-1] # #stats.uniform.cdf
                    
                    #psse_11T[i,j] = stats.kstest(s11, s11_shuff)[-1] # #stats.uniform.cdf
                    #psse_22T[i,j] = stats.kstest(s22, s22_shuff)[-1] # #stats.uniform.cdf
                    #psse_12T[i,j] = stats.kstest(s12, s12_shuff)[-1] # #stats.uniform.cdf
        
        
        angleCheckPoints = np.linspace(0,np.pi,13).round(5)
        cmap = plt.get_cmap('coolwarm')
        norm = mcolors.BoundaryNorm(np.flip(np.cos(angleCheckPoints).round(5)),cmap.N, clip=True)
        
        ### cosTheta
        
        
        plt.figure(figsize=(16, 6), dpi=100)
        plt.subplot(1,3,1)
        ax = plt.gca()
        #im = ax.imshow(cosTheta_11T.mean(axis=1).mean(axis=0), cmap='coolwarm', aspect='auto', norm=mcolors.SymLogNorm(linthresh=0.4, linscale=1, vmin=-1.0, vmax=1.0, base=10)) #, vmin=-180, vmax=180
        im = ax.imshow(cosTheta_11[region][tt].mean(1).mean(0), cmap=cmap, norm=norm, aspect='auto')
        #im = ax.imshow(cosTheta_11T.mean(axis=1).mean(axis=0), cmap='Reds', aspect='auto', vmin=0, vmax=1.0) #, norm=mcolors.SymLogNorm(linthresh=0.25, linscale=0.25, vmin=0, vmax=1.0, base=10)
        #im = ax.imshow(f_plotting.mask_triangle(cosTheta_11T.mean(axis=1).mean(axis=0),ul='u', diag=1), cmap='Reds', aspect='auto', vmin=0, vmax=1.0)
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
        ax.set_frame_on(False)
        
        
        plt.subplot(1,3,2)
        ax = plt.gca()
        #im = ax.imshow(cosTheta_22T.mean(axis=1).mean(axis=0), cmap='coolwarm', aspect='auto', norm=mcolors.SymLogNorm(linthresh=0.4, linscale=1, vmin=-1.0, vmax=1.0, base=10)) #, vmin=-180, vmax=180
        im = ax.imshow(cosTheta_22[region][tt].mean(1).mean(0), cmap=cmap, norm=norm, aspect='auto')
        #im = ax.imshow(cosTheta_22T.mean(axis=1).mean(axis=0), cmap='Reds', aspect='auto', vmin=0, vmax=1.0)#, norm=mcolors.SymLogNorm(linthresh=0.25, linscale=0.25, vmin=0, vmax=1.0, base=10)
        #im = ax.imshow(f_plotting.mask_triangle(cosTheta_22T.mean(axis=1).mean(axis=0),ul='u',diag=1), cmap='Reds', aspect='auto', vmin=0, vmax=1.0)
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
        ax.set_frame_on(False)
        
        
        
        plt.subplot(1,3,3)
        ax = plt.gca()
        #im = ax.imshow(cosTheta_12T.mean(axis=1).mean(axis=0), cmap='coolwarm', aspect='auto', norm=mcolors.SymLogNorm(linthresh=0.4, linscale=1, vmin=-1.0, vmax=1.0, base=10)) #, vmin=-180, vmax=180
        im = ax.imshow(cosTheta_12[region][tt].mean(1).mean(0), cmap=cmap, norm=norm, aspect='auto')
        #im = ax.imshow(cosTheta_12T.mean(axis=1).mean(axis=0), cmap='Reds', aspect='auto', vmin=0, vmax=1.0) #, norm=mcolors.SymLogNorm(linthresh=0.25, linscale=0.25, vmin=0, vmax=1.0, base=10)
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
        ax.set_frame_on(False)
        
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_ticklabels(np.flip(np.degrees(angleCheckPoints).round(1).astype(int)))
        cbar.set_label('cos(θ)', fontsize = 15, rotation = 270, labelpad=20)
        
        plt.subplots_adjust(top = 0.8)
        plt.suptitle(f'Principal Angle (θ), {region.upper()}, {ttype}', fontsize = 20, y=1)
        plt.tight_layout()
        plt.show()
        
        
        
        
        ### cosPsi
        
        plt.figure(figsize=(16, 6), dpi=100)
        plt.subplot(1,3,1)
        ax = plt.gca()
        #im = ax.imshow(cosPsi_11T.mean(axis=1).mean(axis=0), cmap='coolwarm', aspect='auto', norm=mcolors.SymLogNorm(linthresh=0.4, linscale=1, vmin=-1.0, vmax=1.0, base=10)) #, vmin=-180, vmax=180
        im = ax.imshow(cosPsi_11[region][tt].mean(1).mean(0), cmap=cmap, norm=norm, aspect='auto')
        #im = ax.imshow(f_plotting.mask_triangle(cosPsi_11T.mean(axis=1).mean(axis=0),ul='u',diag=1), cmap='coolwarm', aspect='auto', vmin=-1.0, vmax=1.0)
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
        ax.set_frame_on(False)
        
        
        plt.subplot(1,3,2)
        ax = plt.gca()
        #im = ax.imshow(cosPsi_22T.mean(axis=1).mean(axis=0), cmap='coolwarm', aspect='auto', norm=mcolors.SymLogNorm(linthresh=0.4, linscale=1, vmin=-1.0, vmax=1.0, base=10)) #, vmin=-180, vmax=180
        im = ax.imshow(cosPsi_22[region][tt].mean(1).mean(0), cmap=cmap, norm=norm, aspect='auto')
        #im = ax.imshow(f_plotting.mask_triangle(cosPsi_22T.mean(axis=1).mean(axis=0),ul='u',diag=1), cmap='coolwarm', aspect='auto', vmin=-1.0, vmax=1.0)
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
        ax.set_frame_on(False)
        
        
        plt.subplot(1,3,3)
        ax = plt.gca()
        #im = ax.imshow(cosPsi_12T.mean(axis=1).mean(axis=0), cmap='coolwarm', aspect='auto', norm=mcolors.SymLogNorm(linthresh=0.4, linscale=1, vmin=-1.0, vmax=1.0, base=10)) #, vmin=-180, vmax=180
        im = ax.imshow(cosPsi_12[region][tt].mean(1).mean(0), cmap=cmap, norm=norm, aspect='auto')
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
        ax.set_frame_on(False)
        
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_ticklabels(np.flip(np.degrees(angleCheckPoints).round(1).astype(int)))
        cbar.set_label('cos(Ψ)', fontsize = 15, rotation = 270, labelpad=20)
        
        plt.subplots_adjust(top = 0.8)
        plt.suptitle(f'Representational Alignment (Ψ), {region.upper()}, {ttype}', fontsize = 20, y=1)
        #plt.suptitle(f'abs(cosPsi), {region}, ttype={tt}', fontsize = 15, y=1)
        plt.tight_layout()
        plt.show()
        
        
        
        ### sse
        #plt.figure(figsize=(16, 6), dpi=100)
        #plt.subplot(1,3,1)
        #ax = plt.gca()
        #im = ax.imshow(sse_11T.mean(axis=1).mean(axis=0), cmap='Reds', aspect='auto', vmin=0) #, vmax=1
        #im = ax.imshow(f_plotting.mask_triangle(sse_11T.mean(axis=1).mean(axis=0),ul='u',diag=0), cmap='Reds', aspect='auto', vmin=0) #, vmax=1
        #im = ax.imshow(np.median(cosTheta_11T.mean(axis=1), axis=0), cmap='jet', aspect='auto', vmin=0, vmax=1)
        #for i in range(len(checkpoints)):
        #    for j in range(len(checkpoints)):
        #        if 0.05 < psse_11T[i,j] <= 0.1:
        #            text = ax.text(j, i, '+', ha="center", va="center", fontsize=6)
        #        elif 0.01 < psse_11T[i,j] <= 0.05:
        #            text = ax.text(j, i, '*', ha="center", va="center", fontsize=6)
        #        elif psse_11T[i,j] <= 0.01:
        #            text = ax.text(j, i, '**', ha="center", va="center", fontsize=6)
                    
        #ax.set_title('11', pad = 10)
        #ax.set_xticks([n for n in range(len(checkpoints))])
        #ax.set_xticklabels(checkpoints, fontsize = 6)
        #ax.set_yticks([n for n in range(len(checkpoints))])
        #ax.set_yticklabels(checkpoints, fontsize = 6)
        #ax.set_xlabel('Item 1', fontsize = 10)
        #ax.set_ylabel('Item 1', fontsize = 10)
        
        
        
        #plt.subplot(1,3,2)
        #ax = plt.gca()
        #im = ax.imshow(sse_22T.mean(axis=1).mean(axis=0), cmap='Reds', aspect='auto', vmin=0) #, vmax=1
        #im = ax.imshow(f_plotting.mask_triangle(sse_22T.mean(axis=1).mean(axis=0),ul='l',diag=0), cmap='Reds', aspect='auto', vmin=0) #, vmax=1
        #im = ax.imshow(np.median(cosTheta_22T.mean(axis=1), axis=0), cmap='jet', aspect='auto', vmin=0, vmax=1)
        #for i in range(len(checkpoints)):
        #    for j in range(len(checkpoints)):
        #        if 0.05 < psse_22T[i,j] <= 0.1:
        #            text = ax.text(j, i, '+', ha="center", va="center", fontsize=6)
        #        elif 0.01 < psse_22T[i,j] <= 0.05:
        #            text = ax.text(j, i, '*', ha="center", va="center", fontsize=6)
        #        elif psse_22T[i,j] <= 0.01:
        #            text = ax.text(j, i, '**', ha="center", va="center", fontsize=6)
                    
        #ax.set_title('22', pad = 10)
        #ax.set_xticks([n for n in range(len(checkpoints))])
        #ax.set_xticklabels(checkpoints, fontsize = 6)
        #ax.set_yticks([n for n in range(len(checkpoints))])
        #ax.set_yticklabels(checkpoints, fontsize = 6)
        #ax.set_xlabel('Item 2', fontsize = 10)
        #ax.set_ylabel('Item 2', fontsize = 10)
        
        
        
        #plt.subplot(1,3,3)
        #ax = plt.gca()
        #im = ax.imshow(sse_12T.mean(axis=1).mean(axis=0), cmap='Reds', aspect='auto', vmin=0) #, vmax=1
        #im = ax.imshow(np.median(cosTheta_12T.mean(axis=1), axis=0), cmap='jet', aspect='auto', vmin=0, vmax=1)
        #for i in range(len(checkpoints)):
        #    for j in range(len(checkpoints)):
        #        if 0.05 < psse_12T[i,j] <= 0.1:
        #            text = ax.text(j, i, '+', ha="center", va="center", fontsize=6)
        #        elif 0.01 < psse_12T[i,j] <= 0.05:
        #            text = ax.text(j, i, '*', ha="center", va="center", fontsize=6)
        #        elif psse_12T[i,j] <= 0.01:
        #            text = ax.text(j, i, '**', ha="center", va="center", fontsize=6)
        #ax.set_title('12', pad = 10)
        #ax.set_xticks([n for n in range(len(checkpoints))])
        #ax.set_xticklabels(checkpoints, fontsize = 6)
        #ax.set_yticks([n for n in range(len(checkpoints))])
        #ax.set_yticklabels(checkpoints, fontsize = 6)
        #ax.set_xlabel('Item 2', fontsize = 10)
        #ax.set_ylabel('Item 1', fontsize = 10)
        
        #plt.colorbar(im, ax=ax)
        
        #plt.subplots_adjust(top = 0.8)
        #plt.suptitle(f'SSE, {region}, ttype={tt}', fontsize = 15, y=1)
        #plt.tight_layout()
        #plt.show()
# In[] save item v item 
np.save(f'{phd_path}/outputs/monkeys/' + 'cosTheta_11_data.npy', cosTheta_11, allow_pickle=True)
np.save(f'{phd_path}/outputs/monkeys/' + 'cosTheta_12_data.npy', cosTheta_12, allow_pickle=True)
np.save(f'{phd_path}/outputs/monkeys/' + 'cosTheta_22_data.npy', cosTheta_22, allow_pickle=True)
np.save(f'{phd_path}/outputs/monkeys/' + 'cosPsi_11_data.npy', cosPsi_11, allow_pickle=True)
np.save(f'{phd_path}/outputs/monkeys/' + 'cosPsi_12_data.npy', cosPsi_12, allow_pickle=True)
np.save(f'{phd_path}/outputs/monkeys/' + 'cosPsi_22_data.npy', cosPsi_22, allow_pickle=True)

np.save(f'{phd_path}/outputs/monkeys/' + 'cosTheta_11_shuff_data.npy', cosTheta_11_shuff, allow_pickle=True)
np.save(f'{phd_path}/outputs/monkeys/' + 'cosTheta_12_shuff_data.npy', cosTheta_12_shuff, allow_pickle=True)
np.save(f'{phd_path}/outputs/monkeys/' + 'cosTheta_22_shuff_data.npy', cosTheta_22_shuff, allow_pickle=True)
np.save(f'{phd_path}/outputs/monkeys/' + 'cosPsi_11_shuff_data.npy', cosPsi_11_shuff, allow_pickle=True)
np.save(f'{phd_path}/outputs/monkeys/' + 'cosPsi_12_shuff_data.npy', cosPsi_12_shuff, allow_pickle=True)
np.save(f'{phd_path}/outputs/monkeys/' + 'cosPsi_22_shuff_data.npy', cosPsi_22_shuff, allow_pickle=True)
# In[] load item v item

cosTheta_11 = np.load(f'{phd_path}/outputs/monkeys/' + 'cosTheta_11_data.npy', allow_pickle=True).item()
cosTheta_12 = np.load(f'{phd_path}/outputs/monkeys/' + 'cosTheta_12_data.npy', allow_pickle=True).item()
cosTheta_22 = np.load(f'{phd_path}/outputs/monkeys/' + 'cosTheta_22_data.npy', allow_pickle=True).item()
cosPsi_11 = np.load(f'{phd_path}/outputs/monkeys/' + 'cosPsi_11_data.npy', allow_pickle=True).item()
cosPsi_12 = np.load(f'{phd_path}/outputs/monkeys/' + 'cosPsi_12_data.npy', allow_pickle=True).item()
cosPsi_22 = np.load(f'{phd_path}/outputs/monkeys/' + 'cosPsi_22_data.npy', allow_pickle=True).item()


# In[] whisker plots, retarget, I1D1 vs. I2D2, whisker plots
angleCheckPoints = np.linspace(0,np.pi,7).round(5)
cmap = plt.get_cmap('coolwarm')
norm = mcolors.BoundaryNorm(np.flip(np.cos(angleCheckPoints).round(5)),cmap.N, clip=True)

ed1x, ed2x = checkpointsLabels.index('ED1'), checkpointsLabels.index('ED2')
ld1x, ld2x = checkpointsLabels.index('LD1'), checkpointsLabels.index('LD2')
 
color1, color1_, color2, color2_ = '#185337', '#96d9ad', '#804098', '#c4a2d1'

boxprops1 = dict(facecolor=color1, edgecolor='none', linewidth=2)
flierprops1 = dict(markeredgecolor=color1, markerfacecolor=color1)
capprops1 = dict(color=color1, linewidth=2)
whiskerprops1 = dict(color=color1, linewidth=2)

boxprops2 = dict(facecolor=color2, edgecolor='none', linewidth=2)
flierprops2 = dict(markeredgecolor=color2, markerfacecolor=color2)
capprops2 = dict(color=color2, linewidth=2)
whiskerprops2 = dict(color=color2, linewidth=2)

medianprops = dict(linestyle='--', linewidth=2, color='w')
meanpointprops1 = dict(marker='^', markeredgecolor=color1, markerfacecolor='w', markersize=8)
meanpointprops2 = dict(marker='^', markeredgecolor=color2, markerfacecolor='w', markersize=8)

fig, axes = plt.subplots(1,2, figsize=(10,6), dpi=100, sharex=True)#, sharey=True
#plt.figure(figsize=(10, 4), dpi=300)

# angle
ax = axes.flatten()[0]#plt.subplot(1,2,1)
bpl_L = [cosTheta_12['dlpfc'][1][:,:,ed1x,ed2x].mean(1), cosTheta_12['dlpfc'][1][:,:,ld1x,ld2x].mean(1),]
bpr_L = [cosTheta_12['fef'][1][:,:,ed1x,ed2x].mean(1), cosTheta_12['fef'][1][:,:,ld1x,ld2x].mean(1),]

bpl = ax.boxplot(bpl_L, positions=[0.3,1.3], patch_artist=True, widths = 0.2, boxprops=boxprops1, flierprops=flierprops1, meanprops=meanpointprops1, medianprops=medianprops, capprops = capprops1, whiskerprops = whiskerprops1, meanline=False, showmeans=True) #, boxprops=dict(linewidth=3)
bpr = ax.boxplot(bpr_L, positions=[0.7,1.7], patch_artist=True, widths = 0.2, boxprops=boxprops2, flierprops=flierprops2, meanprops=meanpointprops2, medianprops=medianprops, capprops = capprops2, whiskerprops = whiskerprops2, meanline=False, showmeans=True) #, boxprops=dict(linewidth=3)

ax.set_xticks([0.5,1.5],['ED1-ED2','LD1-LD2',])#,rotation=20
ax.set_xlim(0,1)
ax.set_yticks(np.flip(np.cos(angleCheckPoints).round(5)), labels = np.flip(np.degrees(angleCheckPoints).round(1).astype(int)))
ax.set_xlabel('Timebins', labelpad = 3, fontsize = 20)
ax.set_ylabel('cos(θ)', labelpad = 0, fontsize = 20)
ax.set_title(f'Principal Angle', fontsize = 17)
ax.tick_params(axis='both', labelsize=15)




# alignment
ax = axes.flatten()[1]#plt.subplot(1,2,2)
bpl_L = [cosPsi_12['dlpfc'][1][:,:,ed1x,ed2x].mean(1), cosPsi_12['dlpfc'][1][:,:,ld1x,ld2x].mean(1),]
bpr_L = [cosPsi_12['fef'][1][:,:,ed1x,ed2x].mean(1), cosPsi_12['fef'][1][:,:,ld1x,ld2x].mean(1),]

bpl = ax.boxplot(bpl_L, positions=[0.3,1.3], patch_artist=True, widths = 0.2, boxprops=boxprops1, flierprops=flierprops1, meanprops=meanpointprops1, medianprops=medianprops, capprops = capprops1, whiskerprops = whiskerprops1, meanline=False, showmeans=True) #, boxprops=dict(linewidth=3)
bpr = ax.boxplot(bpr_L, positions=[0.7,1.7], patch_artist=True, widths = 0.2, boxprops=boxprops2, flierprops=flierprops2, meanprops=meanpointprops2, medianprops=medianprops, capprops = capprops2, whiskerprops = whiskerprops2, meanline=False, showmeans=True) #, boxprops=dict(linewidth=3)

ax.plot([], c=color1, label='LPFC')
ax.plot([], c=color2, label='FEF')

ax.set_xticks([0.5,1.5],['ED1-ED2','LD1-LD2',]) #,rotation=20
ax.set_xlim(0,2)
ax.set_yticks(np.flip(np.cos(angleCheckPoints).round(5)), labels = np.flip(np.degrees(angleCheckPoints).round(1).astype(int)))
ax.set_xlabel('Timebins', labelpad = 3, fontsize = 20)
ax.set_ylabel('cos(ψ)', labelpad = 0, fontsize = 20)
ax.set_title(f'Representational Alignment', fontsize = 17)
ax.tick_params(axis='both', labelsize=15)

ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize = 15)


plt.suptitle(f'I1D1 vs. I2D2, Retarget', fontsize = 25, y=1) # (Monkeys)
plt.tight_layout()
plt.show()

#fig.savefig(f'{phd_path}/data/pooled/i1d1-i2d2_ret_data.tif', bbox_inches='tight')


#%%                    


#############################
# compare choice/non-choice #
############################# 


# In[] compute choice v choice
pdummy = True

cosTheta_choice, cosTheta_nonchoice = {},{}
cosTheta_choice_shuff, cosTheta_nonchoice_shuff = {}, {}

cosPsi_choice, cosPsi_nonchoice = {},{}
cosPsi_choice_shuff, cosPsi_nonchoice_shuff = {}, {}


for region in ('dlpfc','fef'):
        
    cosTheta_choiceT, cosTheta_nonchoiceT = np.zeros((nIters, nBoots, len(checkpoints),)), np.zeros((nIters, nBoots, len(checkpoints),))
    cosPsi_choiceT, cosPsi_nonchoiceT = np.zeros((nIters, nBoots, len(checkpoints),)), np.zeros((nIters, nBoots, len(checkpoints),))
    
    cosTheta_choiceT_shuff, cosTheta_nonchoiceT_shuff = np.zeros((nIters, nPerms, len(checkpoints),)), np.zeros((nIters, nPerms, len(checkpoints),))
    cosPsi_choiceT_shuff, cosPsi_nonchoiceT_shuff = np.zeros((nIters, nPerms, len(checkpoints),)), np.zeros((nIters, nPerms, len(checkpoints),))
    
    for n in range(nIters):
        for nbt in range(nBoots):
            for nc,cp in enumerate(checkpoints):
                cT_C, _, cP_C, _, _ = f_subspace.angle_alignment_coplanar(vecs[region][cp][1][2][n][nbt], projs[region][cp][1][2][n][nbt], vecs[region][cp][2][1][n][nbt], projs[region][cp][2][1][n][nbt])
                cT_NC, _, cP_NC, _, _ = f_subspace.angle_alignment_coplanar(vecs[region][cp][1][1][n][nbt], projs[region][cp][1][1][n][nbt], vecs[region][cp][2][2][n][nbt], projs[region][cp][2][2][n][nbt])
                
                cosTheta_choiceT[n,nbt,nc], cosPsi_choiceT[n,nbt,nc] = cT_C, cP_C
                cosTheta_nonchoiceT[n,nbt,nc], cosPsi_nonchoiceT[n,nbt,nc] = cT_NC, cP_NC
        
        for npm in range(nPerms):
            for nc,cp in enumerate(checkpoints):
                cT_C_shuff, _, cP_C_shuff, _, _ = f_subspace.angle_alignment_coplanar(vecs_shuff[region][cp][1][2][n][npm], projs_shuff[region][cp][1][2][n][npm], vecs_shuff[region][cp][2][1][n][npm], projs_shuff[region][cp][2][1][n][npm])
                cT_NC_shuff, _, cP_NC_shuff, _, _ = f_subspace.angle_alignment_coplanar(vecs_shuff[region][cp][1][1][n][npm], projs_shuff[region][cp][1][1][n][npm], vecs_shuff[region][cp][2][2][n][npm], projs_shuff[region][cp][2][2][n][npm])
                
                cosTheta_choiceT_shuff[n,npm,nc], cosPsi_choiceT_shuff[n,npm,nc] = cT_C_shuff, cP_C_shuff
                cosTheta_nonchoiceT_shuff[n,npm,nc], cosPsi_nonchoiceT_shuff[n,npm,nc] = cT_NC_shuff, cP_NC_shuff
    
    cosTheta_choice[region], cosTheta_nonchoice[region] = cosTheta_choiceT, cosTheta_nonchoiceT
    cosPsi_choice[region], cosPsi_nonchoice[region] = cosPsi_choiceT, cosPsi_nonchoiceT
    cosTheta_choice_shuff[region], cosTheta_nonchoice_shuff[region] = cosTheta_choiceT_shuff, cosTheta_nonchoiceT_shuff
    cosPsi_choice_shuff[region], cosPsi_nonchoice_shuff[region] = cosPsi_choiceT_shuff, cosPsi_nonchoiceT_shuff

# In[] save choice geoms 
np.save(f'{phd_path}/outputs/monkeys/' + 'cosTheta_choice_data.npy', cosTheta_choice, allow_pickle=True)
np.save(f'{phd_path}/outputs/monkeys/' + 'cosTheta_nonchoice_data.npy', cosTheta_nonchoice, allow_pickle=True)
np.save(f'{phd_path}/outputs/monkeys/' + 'cosPsi_choice_data.npy', cosPsi_choice, allow_pickle=True)
np.save(f'{phd_path}/outputs/monkeys/' + 'cosPsi_nonchoice_data.npy', cosPsi_nonchoice, allow_pickle=True)

np.save(f'{phd_path}/outputs/monkeys/' + 'cosTheta_choice_shuff_data.npy', cosTheta_choice_shuff, allow_pickle=True)
np.save(f'{phd_path}/outputs/monkeys/' + 'cosTheta_nonchoice_shuff_data.npy', cosTheta_nonchoice_shuff, allow_pickle=True)
np.save(f'{phd_path}/outputs/monkeys/' + 'cosPsi_choice_shuff_data.npy', cosPsi_choice_shuff, allow_pickle=True)
np.save(f'{phd_path}/outputs/monkeys/' + 'cosPsi_nonchoice_shuff_data.npy', cosPsi_nonchoice_shuff, allow_pickle=True)

# In[] load choice geoms 

cosTheta_choice = np.load(f'{phd_path}/fitting planes/pooled/w&w/' + 'cosTheta_choice_data.npy', allow_pickle=True).item()
cosTheta_nonchoice = np.load(f'{phd_path}/fitting planes/pooled/w&w/' + 'cosTheta_nonchoice_data.npy', allow_pickle=True).item()
cosPsi_choice = np.load(f'{phd_path}/fitting planes/pooled/w&w/' + 'cosPsi_choice_data.npy', allow_pickle=True).item()
cosPsi_nonchoice = np.load(f'{phd_path}/fitting planes/pooled/w&w/' + 'cosPsi_nonchoice_data.npy', allow_pickle=True).item()

cosTheta_choice_shuff_all = np.load(f'{phd_path}/fitting planes/pooled/w&w/' + 'cosTheta_choice_shuff_data.npy', allow_pickle=True).item()
cosTheta_nonchoice_shuff_all = np.load(f'{phd_path}/fitting planes/pooled/w&w/' + 'cosTheta_nonchoice_shuff_data.npy', allow_pickle=True).item()
cosPsi_choice_shuff_all = np.load(f'{phd_path}/fitting planes/pooled/w&w/' + 'cosPsi_choice_shuff_data.npy', allow_pickle=True).item()
cosPsi_nonchoice_shuff_all = np.load(f'{phd_path}/fitting planes/pooled/w&w/' + 'cosPsi_nonchoice_shuff_data.npy', allow_pickle=True).item()
            
#%%  nonused  
    ###
    pcosTheta_choice, pcosTheta_nonchoice = np.ones((len(checkpoints))), np.ones((len(checkpoints)))
    pcosPsi_choice, pcosPsi_nonchoice = np.ones((len(checkpoints))), np.ones((len(checkpoints)))
    
    if pdummy == False:
        ### shuff 
        cosTheta_choice_shuffT, cosTheta_nonchoice_shuffT = np.zeros((nIters, nBoots*nPerms, len(checkpoints),)), np.zeros((nIters, nBoots*nPerms, len(checkpoints),))
        cosPsi_choice_shuffT, cosPsi_nonchoice_shuffT = np.zeros((nIters, nBoots*nPerms, len(checkpoints),)), np.zeros((nIters, nBoots*nPerms, len(checkpoints),))
        
        for n in range(nIters):
            for nbt in range(nBoots*nPerms):
                for nc, cp in enumerate(checkpoints):
                    cT_C, _, cP_C, _, s_C = f_subspace.angle_alignment_coplanar(vecs_shuff[region][cp][1][2][n][nbt], projs_shuff[region][cp][1][2][n][nbt], vecs_shuff[region][cp][2][1][n][nbt], projs_shuff[region][cp][2][1][n][nbt])
                    
                    cT_NC, _, cP_NC, _, s_NC = f_subspace.angle_alignment_coplanar(vecs_shuff[region][cp][1][1][n][nbt], projs_shuff[region][cp][1][1][n][nbt], vecs_shuff[region][cp][2][2][n][nbt], projs_shuff[region][cp][2][2][n][nbt])
                    
                    cosTheta_choice_shuffT[n,nbt,nc], cosPsi_choice_shuffT[n,nbt,nc] = cT_C, cP_C
                    cosTheta_nonchoice_shuffT[n,nbt,nc], cosPsi_nonchoice_shuffT[n,nbt,nc] = cT_NC, cP_NC
                    
        cosTheta_choice_shuff_all[region], cosTheta_nonchoice_shuff_all[region] = np.concatenate([cosTheta_choice_shuffT[:,nbt,:] for nbt in range(nBoots*nPerms)]), np.concatenate([cosTheta_nonchoice_shuffT[:,nbt,:] for nbt in range(nBoots*nPerms)])
        cosPsi_choice_shuff_all[region], cosPsi_nonchoice_shuff_all[region] = np.concatenate([cosPsi_choice_shuffT[:,nbt,:] for nbt in range(nBoots*nPerms)]), np.concatenate([cosPsi_nonchoice_shuffT[:,nbt,:] for nbt in range(nBoots*nPerms)])
        
        
        for i in range(len(checkpoints)):
            
            # test distribution
            cT_C, cT_NC = cosTheta_choice[region].mean(axis=1)[:,i].round(5), cosTheta_nonchoice[region].mean(axis=1)[:,i].round(5)
            cP_C, cP_NC = cosPsi_choice[region].mean(axis=1)[:,i].round(5), cosPsi_nonchoice[region].mean(axis=1)[:,i].round(5)
            
            # shuff distribution
            cT_C_shuff, cT_NC_shuff = cosTheta_choice_shuff_all[region][:,i].round(5), cosTheta_nonchoice_shuff_all[region][:,i].round(5)
            cP_C_shuff, cP_NC_shuff = cosPsi_choice_shuff_all[region][:,i].round(5), cosPsi_nonchoice_shuff_all[region][:,i].round(5)
            
            # compare distributions and calculate p values
            pcosTheta_choice[i], pcosTheta_nonchoice[i] = stats.kstest(cT_C, cT_C_shuff)[-1], stats.kstest(cT_NC, cT_NC_shuff)[-1]
            pcosPsi_choice[i], pcosPsi_nonchoice[i] = stats.kstest(cP_C, cP_C_shuff)[-1], stats.kstest(cP_NC, cP_NC_shuff)[-1]
## In[] whisker plot, I1D2-Distraction vs I2D2-Retarget, whisker plots

angleCheckPoints = np.linspace(0,np.pi,7).round(5)
cmap = plt.get_cmap('coolwarm')
norm = mcolors.BoundaryNorm(np.flip(np.cos(angleCheckPoints).round(5)),cmap.N, clip=True)

ed1x, ed2x = checkpointsLabels.index('ED1'), checkpointsLabels.index('ED2')
ld1x, ld2x = checkpointsLabels.index('LD1'), checkpointsLabels.index('LD2')
 
color1, color1_, color2, color2_ = '#185337', '#96d9ad', '#804098', '#c4a2d1'

boxprops1 = dict(facecolor=color1, edgecolor='none', linewidth=2)
flierprops1 = dict(markeredgecolor=color1, markerfacecolor=color1)
capprops1 = dict(color=color1, linewidth=2)
whiskerprops1 = dict(color=color1, linewidth=2)

boxprops2 = dict(facecolor=color2, edgecolor='none', linewidth=2)
flierprops2 = dict(markeredgecolor=color2, markerfacecolor=color2)
capprops2 = dict(color=color2, linewidth=2)
whiskerprops2 = dict(color=color2, linewidth=2)

medianprops = dict(linestyle='--', linewidth=2, color='w')
meanpointprops1 = dict(marker='^', markeredgecolor=color1, markerfacecolor='w', markersize=8)
meanpointprops2 = dict(marker='^', markeredgecolor=color2, markerfacecolor='w', markersize=8)

fig, axes = plt.subplots(1,2, figsize=(10,6), dpi=100, sharex=True)#, sharey=True
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



# In[] plot choice v choice
for region in ('dlpfc','fef'):
    
    #angleCheckPoints = np.linspace(0,np.pi,13).round(5)
    angleCheckPoints = np.linspace(0,np.pi,7).round(5)
    
    ############################################
    ### cosTheta
    plt.figure(figsize=(8, 3), dpi=100)
    plt.subplot(1,2,1)
    ax = plt.gca()
    ax.errorbar(np.arange(0, len(checkpoints), 1), cosTheta_choice[region].mean(1).mean(0), yerr = cosTheta_choice[region].mean(axis=1).std(axis=0), marker = 'o', color = 'c', label = 'Choice', capsize=4)
    #ax.plot(np.arange(0, len(checkpoints), 1), pcosTheta_choice, alpha = 0.3, linestyle = '-')
    ax.errorbar(np.arange(0, len(checkpoints), 1), cosTheta_nonchoice[region].mean(1).mean(0), yerr = cosTheta_nonchoice[region].mean(axis=1).std(axis=0), marker = 'o', color = 'y', label = 'Nonchoice', capsize=4)
    #ax.plot(np.arange(0, len(checkpoints), 1), pcosTheta_nonchoice, alpha = 0.3, linestyle = '-')
    
            
    #ax.set_title('Choice', pad = 10)
    ax.set_xticks([n for n in range(len(checkpoints))])
    ax.set_xticklabels(checkpointsLabels, fontsize = 10)
    ax.set_ylim((-1,1))
    ax.set_yticks(np.flip(np.cos(angleCheckPoints).round(5)), labels = np.flip(np.degrees(angleCheckPoints).round(1).astype(int)))
    ax.set_ylabel('cos(θ)',fontsize=15,rotation = 90)
    ax.legend(loc='lower right')
   
    ### cosPsi
    plt.subplot(1,2,2)
    ax = plt.gca()
    ax.errorbar(np.arange(0, len(checkpoints), 1), cosPsi_choice[region].mean(1).mean(0), yerr = cosPsi_choice[region].mean(axis=1).std(axis=0), marker = 'o', color = 'c', label = 'Choice', capsize=4)
    #ax.plot(np.arange(0, len(checkpoints), 1), pcosPsi_choice, alpha = 0.3, linestyle = '-')
    ax.errorbar(np.arange(0, len(checkpoints), 1), cosPsi_nonchoice[region].mean(1).mean(0), yerr = cosPsi_nonchoice[region].mean(axis=1).std(axis=0), marker = 'o', color = 'y', label = 'Nonchoice', capsize=4)
    #ax.plot(np.arange(0, len(checkpoints), 1), pcosPsi_nonchoice, alpha = 0.3, linestyle = '-')
    
    
    #ax.set_title('Choice', pad = 10)
    ax.set_xticks([n for n in range(len(checkpoints))])
    ax.set_xticklabels(checkpointsLabels, fontsize = 10)
    ax.set_ylim((-1,1))
    ax.set_yticks(np.flip(np.cos(angleCheckPoints).round(5)), labels = np.flip(np.degrees(angleCheckPoints).round(1).astype(int)))
    ax.set_ylabel('cos(Ψ)',fontsize=15,rotation = 90)
    ax.legend(loc='lower right')
    
    plt.subplots_adjust(top = 0.8)
    plt.suptitle(f'Choice/Nonchoice-Item Subspaces, {region.upper()}', fontsize = 15, y=1)
    #plt.suptitle(f'abs(cosPsi), {region}, choice/nonchoice', fontsize = 15, y=1)
    plt.tight_layout()
    plt.show()    
    
    #############################################
    
    
    
    
    
#%%  nonused  
    ############################################
    ### cosTheta
    plt.figure(figsize=(8, 3), dpi=100)
    plt.subplot(1,2,1)
    ax = plt.gca()
    ax.errorbar(np.arange(0, len(checkpoints), 1), cosTheta_choice[region].mean(axis=1).mean(axis=0), yerr = cosTheta_choice[region].mean(axis=1).std(axis=0), marker = 'o', capsize=4)
    ax.plot(np.arange(0, len(checkpoints), 1), pcosTheta_choice, alpha = 0.3, linestyle = '-')
    
    trans = ax.get_xaxis_transform()
    for nc, cp in enumerate(checkpoints):
        if 0.05 < pcosTheta_choice[nc] <= 0.1:
            ax.annotate('+', xy=(nc, 0.05), xycoords = trans, annotation_clip = False, ha = 'center', va = 'center')
        elif 0.01 < pcosTheta_choice[nc] <= 0.05:
            ax.annotate('*', xy=(nc, 0.05), xycoords = trans, annotation_clip = False, ha = 'center', va = 'center')
        elif pcosTheta_choice[nc] <= 0.01:
            ax.annotate('**', xy=(nc, 0.05), xycoords = trans, annotation_clip = False, ha = 'center', va = 'center')
            
    #ax.set_title('Choice', pad = 10)
    ax.set_xticks([n for n in range(len(checkpoints))])
    ax.set_xticklabels(checkpointsLabels, fontsize = 15)
    ax.set_ylim((-1,1))
    ax.set_yticks(np.flip(np.cos(angleCheckPoints).round(5)), labels = np.flip(np.degrees(angleCheckPoints).round(1).astype(int)))
    ax.set_ylabel('cos(θ)',fontsize=15,rotation = 90)
    
   
    ### cosPsi
    plt.subplot(1,2,2)
    ax = plt.gca()
    ax.errorbar(np.arange(0, len(checkpoints), 1), cosPsi_choice[region].mean(axis=1).mean(axis=0), yerr = cosPsi_choice[region].mean(axis=1).std(axis=0), marker = 'o', capsize=4)
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
            
    #ax.set_title('Choice', pad = 10)
    ax.set_xticks([n for n in range(len(checkpoints))])
    ax.set_xticklabels(checkpointsLabels, fontsize = 15)
    ax.set_ylim((-1,1))
    ax.set_yticks(np.flip(np.cos(angleCheckPoints).round(5)), labels = np.flip(np.degrees(angleCheckPoints).round(1).astype(int)))
    ax.set_ylabel('cos(Ψ)',fontsize=15,rotation = 90)
    
    
    plt.subplots_adjust(top = 0.8)
    plt.suptitle(f'Choice-Item Subspaces, {region.upper()}', fontsize = 15, y=1)
    #plt.suptitle(f'abs(cosPsi), {region}, choice/nonchoice', fontsize = 15, y=1)
    plt.tight_layout()
    plt.show()    
    
    #############################################
    
    
    
    ############################################
    ### cosTheta
    plt.figure(figsize=(8, 3), dpi=100)
    plt.subplot(1,2,1)
    ax = plt.gca()
    ax.errorbar(np.arange(0, len(checkpoints), 1), cosTheta_nonchoice[region].mean(axis=1).mean(axis=0), yerr = cosTheta_nonchoice[region].mean(axis=1).std(axis=0), marker = 'o', capsize=4)
    ax.plot(np.arange(0, len(checkpoints), 1), pcosTheta_nonchoice, alpha = 0.3, linestyle = '-')
    
    trans = ax.get_xaxis_transform()
    for nc, cp in enumerate(checkpoints):
        if 0.05 < pcosTheta_nonchoice[nc] <= 0.1:
            ax.annotate('+', xy=(nc, 0.05), xycoords = trans, annotation_clip = False, ha = 'center', va = 'center')
        elif 0.01 < pcosTheta_nonchoice[nc] <= 0.05:
            ax.annotate('*', xy=(nc, 0.05), xycoords = trans, annotation_clip = False, ha = 'center', va = 'center')
        elif pcosTheta_nonchoice[nc] <= 0.01:
            ax.annotate('**', xy=(nc, 0.05), xycoords = trans, annotation_clip = False, ha = 'center', va = 'center')
            
    #ax.set_title('Choice', pad = 10)
    ax.set_xticks([n for n in range(len(checkpoints))])
    ax.set_xticklabels(checkpointsLabels, fontsize = 15)
    ax.set_ylim((-1,1))
    ax.set_yticks(np.flip(np.cos(angleCheckPoints).round(5)), labels = np.flip(np.degrees(angleCheckPoints).round(1).astype(int)))
    ax.set_ylabel('cos(θ)',fontsize=15,rotation = 90)
    
   
    ### cosPsi
    plt.subplot(1,2,2)
    ax = plt.gca()
    ax.errorbar(np.arange(0, len(checkpoints), 1), cosPsi_nonchoice[region].mean(axis=1).mean(axis=0), yerr = cosPsi_nonchoice[region].mean(axis=1).std(axis=0), marker = 'o', capsize=4)
    #ax.errorbar(np.arange(0, len(checkpoints), 1), np.abs(cosPsi_nonchoice[region]).mean(axis=1).mean(axis=0), yerr = np.abs(cosPsi_nonchoice[region]).mean(axis=1).std(axis=0), marker = 'o')
    ax.plot(np.arange(0, len(checkpoints), 1), pcosPsi_nonchoice, alpha = 0.3, linestyle = '-')
    
    trans = ax.get_xaxis_transform()
    for nc, cp in enumerate(checkpoints):
        if 0.05 < pcosPsi_nonchoice[nc] <= 0.1:
            ax.annotate('+', xy=(nc, 0.05), xycoords = trans, annotation_clip = False, ha = 'center', va = 'center')
        elif 0.01 < pcosPsi_nonchoice[nc] <= 0.05:
            ax.annotate('*', xy=(nc, 0.05), xycoords = trans, annotation_clip = False, ha = 'center', va = 'center')
        elif pcosPsi_nonchoice[nc] <= 0.01:
            ax.annotate('**', xy=(nc, 0.05), xycoords = trans, annotation_clip = False, ha = 'center', va = 'center')
            
    #ax.set_title('Choice', pad = 10)
    ax.set_xticks([n for n in range(len(checkpoints))])
    ax.set_xticklabels(checkpointsLabels, fontsize = 15)
    ax.set_ylim((-1,1))
    ax.set_yticks(np.flip(np.cos(angleCheckPoints).round(5)), labels = np.flip(np.degrees(angleCheckPoints).round(1).astype(int)))
    ax.set_ylabel('cos(Ψ)',fontsize=15,rotation = 90)
    
    
    plt.subplots_adjust(top = 0.8)
    plt.suptitle(f'Nonchoice-Item Subspaces, {region.upper()}', fontsize = 15, y=1)
    #plt.suptitle(f'abs(cosPsi), {region}, choice/nonchoice', fontsize = 15, y=1)
    plt.tight_layout()
    plt.show()    
    
    #############################################
    
    #
    ### cosTheta
    plt.figure(figsize=(8, 3), dpi=100)
    plt.subplot(1,2,1)
    ax = plt.gca()
    ax.errorbar(np.arange(0, len(checkpoints), 1), cosTheta_choice[region].mean(axis=1).mean(axis=0), yerr = cosTheta_choice[region].mean(axis=1).std(axis=0), marker = 'o', capsize=4)
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
    ax.set_xticklabels(checkpointsLabels, fontsize = 15)
    ax.set_ylim((-1,1))
    ax.set_yticks(np.flip(np.cos(angleCheckPoints).round(5)), labels = np.flip(np.degrees(angleCheckPoints).round(1).astype(int)))
    ax.set_ylabel('cos(θ)',fontsize=15,rotation = 90)
    
    plt.subplot(1,2,2)
    ax = plt.gca()
    ax.errorbar(np.arange(0, len(checkpoints), 1), cosTheta_nonchoice[region].mean(axis=1).mean(axis=0), yerr = cosTheta_nonchoice[region].mean(axis=1).std(axis=0), marker = 'o', capsize=4)
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
    ax.set_xticklabels(checkpointsLabels, fontsize = 15)
    ax.set_ylim((-1,1))
    ax.set_yticks(np.flip(np.cos(angleCheckPoints).round(5)), labels = np.flip(np.degrees(angleCheckPoints).round(1).astype(int)))
    ax.set_ylabel('cos(θ)',fontsize=15,rotation = 90)
    
    plt.subplots_adjust(top = 0.8)
    plt.suptitle(f'Principal Angle (θ), {region.upper()}', fontsize = 15, y=1)
    plt.tight_layout()
    plt.show()    
    
    
    ### cosPsi
    plt.figure(figsize=(8, 3), dpi=100)
    plt.subplot(1,2,1)
    ax = plt.gca()
    ax.errorbar(np.arange(0, len(checkpoints), 1), cosPsi_choice[region].mean(axis=1).mean(axis=0), yerr = cosPsi_choice[region].mean(axis=1).std(axis=0), marker = 'o', capsize=4)
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
    ax.set_xticklabels(checkpointsLabels, fontsize = 15)
    ax.set_ylim((-1,1))
    ax.set_yticks(np.flip(np.cos(angleCheckPoints).round(5)), labels = np.flip(np.degrees(angleCheckPoints).round(1).astype(int)))
    ax.set_ylabel('cos(Ψ)',fontsize=15,rotation = 90)
    
    
    plt.subplot(1,2,2)
    ax = plt.gca()
    ax.errorbar(np.arange(0, len(checkpoints), 1), cosPsi_nonchoice[region].mean(axis=1).mean(axis=0), yerr = cosPsi_nonchoice[region].mean(axis=1).std(axis=0), marker = 'o', capsize=4)
    #ax.errorbar(np.arange(0, len(checkpoints), 1), np.abs(cosPsi_nonchoice[region]).mean(axis=1).mean(axis=0), yerr = np.abs(cosPsi_nonchoice[region]).mean(axis=1).std(axis=0), marker = 'o')
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
    ax.set_xticklabels(checkpointsLabels, fontsize = 15)
    ax.set_ylim((-1,1))
    ax.set_yticks(np.flip(np.cos(angleCheckPoints).round(5)), labels = np.flip(np.degrees(angleCheckPoints).round(1).astype(int)))
    ax.set_ylabel('cos(Ψ)',fontsize=15,rotation = 90)
    
    plt.subplots_adjust(top = 0.8)
    plt.suptitle(f'Representational Alignment (Ψ), {region.upper()}', fontsize = 15, y=1)
    #plt.suptitle(f'abs(cosPsi), {region}, choice/nonchoice', fontsize = 15, y=1)
    plt.tight_layout()
    plt.show()    
    
    
    

# In[] export descriptive stats
theta_mean, theta_std = pd.DataFrame(), pd.DataFrame()
psi_mean, psi_std = pd.DataFrame(), pd.DataFrame()

theta_mean.to_excel(f'{save_path}/theta_mean.xlsx')
theta_std.to_excel(f'{save_path}/theta_std.xlsx')
psi_mean.to_excel(f'{save_path}/psi_mean.xlsx')
psi_std.to_excel(f'{save_path}/psi_std.xlsx')

with pd.ExcelWriter(f'{save_path}/theta_mean.xlsx') as writer:
    for region in ('dlpfc','fef'):
        pd.DataFrame(cosTheta_choice[region].mean(1).mean(0)).to_excel(writer, sheet_name=f'{region}_theta_choice')
        pd.DataFrame(cosTheta_nonchoice[region].mean(1).mean(0)).to_excel(writer, sheet_name=f'{region}_theta_nonchoice')
        
        for tt in ttypes:
            pd.DataFrame(cosTheta_11[region][tt].mean(1).mean(0)).to_excel(writer, sheet_name=f'{region}_{tt}_theta_11')
            pd.DataFrame(cosTheta_22[region][tt].mean(1).mean(0)).to_excel(writer, sheet_name=f'{region}_{tt}_theta_22')
            pd.DataFrame(cosTheta_12[region][tt].mean(1).mean(0)).to_excel(writer, sheet_name=f'{region}_{tt}_theta_12')
            
            
with pd.ExcelWriter(f'{save_path}/theta_std.xlsx') as writer:
    for region in ('dlpfc','fef'):
        pd.DataFrame(cosTheta_choice[region].mean(1).std(0)).to_excel(writer, sheet_name=f'{region}_theta_choice')
        pd.DataFrame(cosTheta_nonchoice[region].mean(1).std(0)).to_excel(writer, sheet_name=f'{region}_theta_nonchoice')
        
        for tt in ttypes:            
            pd.DataFrame(cosTheta_11[region][tt].mean(1).std(0)).to_excel(writer, sheet_name=f'{region}_{tt}_theta_11')
            pd.DataFrame(cosTheta_22[region][tt].mean(1).std(0)).to_excel(writer, sheet_name=f'{region}_{tt}_theta_22')
            pd.DataFrame(cosTheta_12[region][tt].mean(1).std(0)).to_excel(writer, sheet_name=f'{region}_{tt}_theta_12')
            

with pd.ExcelWriter(f'{save_path}/psi_mean.xlsx') as writer:
    for region in ('dlpfc','fef'):
        pd.DataFrame(cosPsi_choice[region].mean(1).mean(0)).to_excel(writer, sheet_name=f'{region}_psi_choice')
        pd.DataFrame(cosPsi_nonchoice[region].mean(1).mean(0)).to_excel(writer, sheet_name=f'{region}_psi_nonchoice')
        
        for tt in ttypes:
            pd.DataFrame(cosPsi_11[region][tt].mean(1).mean(0)).to_excel(writer, sheet_name=f'{region}_{tt}_psi_11')
            pd.DataFrame(cosPsi_22[region][tt].mean(1).mean(0)).to_excel(writer, sheet_name=f'{region}_{tt}_psi_22')
            pd.DataFrame(cosPsi_12[region][tt].mean(1).mean(0)).to_excel(writer, sheet_name=f'{region}_{tt}_psi_12')


with pd.ExcelWriter(f'{save_path}/psi_std.xlsx') as writer:
    for region in ('dlpfc','fef'):
        pd.DataFrame(cosPsi_choice[region].mean(1).std(0)).to_excel(writer, sheet_name=f'{region}_psi_choice')
        pd.DataFrame(cosPsi_nonchoice[region].mean(1).std(0)).to_excel(writer, sheet_name=f'{region}_psi_nonchoice')
        
        for tt in ttypes:
            pd.DataFrame(cosPsi_11[region][tt].mean(1).std(0)).to_excel(writer, sheet_name=f'{region}_{tt}_psi_11')
            pd.DataFrame(cosPsi_22[region][tt].mean(1).std(0)).to_excel(writer, sheet_name=f'{region}_{tt}_psi_22')
            pd.DataFrame(cosPsi_12[region][tt].mean(1).std(0)).to_excel(writer, sheet_name=f'{region}_{tt}_psi_12')
            


#%%  


#############################################
# decoadability of item subspace projection #
#############################################


#%% compute decodability of item subspace, permutation here used to create baseline of the decodability of randomly organized subspaces, not perdicting random labels

Y_columnsLabels = ['locKey','locs','type','loc1','loc2','locX']
toDecode_labels1 = 'loc1'
toDecode_labels2 = 'loc2'
shuff_excludeInv = False

nPerms = 10
infoMethod = 'lda' #  'omega2' #

decode_proj1_3d, decode_proj2_3d = {},{}
decode_proj1_shuff_all_3d, decode_proj2_shuff_all_3d = {},{}


for region in ('dlpfc','fef'):
    print(f'Region={region}')
    decode_proj1_3d[region], decode_proj2_3d[region] = {}, {}
    #decode_proj1_2d[region], decode_proj2_2d[region] = {}, {}
    
    decode_proj1_shuff_all_3d[region], decode_proj2_shuff_all_3d[region] = {},{}
    #decode_proj1_shuff_all_2d[region], decode_proj2_shuff_all_2d[region] = {},{}
    
    for tt in ttypes:
        print(f'TType={tt}')
        decode_proj1T_3d = np.zeros((nIters, nPerms, len(checkpoints))) # pca1st 3d coordinates
        decode_proj2T_3d = np.zeros((nIters, nPerms, len(checkpoints)))
        
        # shuff
        decode_proj1T_3d_shuff = np.zeros((nIters, nPerms, len(checkpoints)))
        decode_proj2T_3d_shuff = np.zeros((nIters, nPerms, len(checkpoints)))
        
        for n in range(nIters):
            
            if n%20 == 0:
                print(f'{n}')
            
            for npm in range(nPerms):
                trialInfoT = trialInfos[region][tt][n][0]
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
                    vecs1, vecs2 = vecs[region][cp][tt][1][n][0], vecs[region][cp][tt][2][n][0]
                    projs1, projs2 = projs[region][cp][tt][1][n][0], projs[region][cp][tt][2][n][0]
                    projs1_allT_3d, projs2_allT_3d = projsAll[region][cp][tt][1][n][0], projsAll[region][cp][tt][2][n][0]

                    info1_3d, _ = f_subspace.plane_decodability(vecs1, projs1, projs1_allT_3d, trialInfoT, 'loc1', method = infoMethod)
                    info2_3d, _ = f_subspace.plane_decodability(vecs2, projs2, projs2_allT_3d, trialInfoT, 'loc2', method = infoMethod)
                    
                    decode_proj1T_3d[n,npm,nc] = info1_3d #.mean(axis=-1)
                    decode_proj2T_3d[n,npm,nc] = info2_3d #.mean(axis=-1)
                    
                    # shuff
                    info1_3d_shuff, _ = f_subspace.plane_decodability(vecs1, projs1, projs1_allT_3d, trialInfoT_shuff, 'loc1', method = infoMethod)
                    info2_3d_shuff, _ = f_subspace.plane_decodability(vecs2, projs2, projs2_allT_3d, trialInfoT_shuff, 'loc2', method = infoMethod)
                    
                    decode_proj1T_3d_shuff[n,npm,nc] = info1_3d_shuff #.mean(axis=-1).mean(axis=-1)
                    decode_proj2T_3d_shuff[n,npm,nc] = info2_3d_shuff #.mean(axis=-1).mean(axis=-1)
                    
                    
        decode_proj1_3d[region][tt] = decode_proj1T_3d
        decode_proj2_3d[region][tt] = decode_proj2T_3d
                    
        decode_proj1_shuff_all_3d[region][tt] = decode_proj1T_3d_shuff
        decode_proj2_shuff_all_3d[region][tt] = decode_proj2T_3d_shuff



# In[] save
np.save(f'{phd_path}/outputs/monkeys/' + 'performance1_item_data.npy', decode_proj1_3d, allow_pickle=True)
np.save(f'{phd_path}/outputs/monkeys/' + 'performance2_item_data.npy', decode_proj2_3d, allow_pickle=True)
np.save(f'{phd_path}/outputs/monkeys/' + 'performance1_item_shuff_data.npy', decode_proj1_shuff_all_3d, allow_pickle=True)
np.save(f'{phd_path}/outputs/monkeys/' + 'performance2_item_shuff_data.npy', decode_proj2_shuff_all_3d, allow_pickle=True)

# In[] load

decode_proj1_3d = np.load(f'{phd_path}/outputs/monkeys/' + 'performance1_item_data.npy', allow_pickle=True).item()
decode_proj2_3d = np.load(f'{phd_path}/outputs/monkeys/' + 'performance2_item_data.npy', allow_pickle=True).item()
decode_proj1_shuff_all_3d = np.load(f'{phd_path}/outputs/monkeys/' + 'performance1_item_shuff_data.npy', allow_pickle=True).item()
decode_proj2_shuff_all_3d = np.load(f'{phd_path}/outputs/monkeys/' + 'performance2_item_shuff_data.npy', allow_pickle=True).item()
        
# In[] plot
for region in ('dlpfc','fef'):
        
    infoLabel = 'Accuracy' if infoMethod=='lda' else 'PEV'
    
    fig, axes = plt.subplots(1,2, figsize=(12,4), dpi=100, sharex=True, sharey=True)#
    
    for tt in ttypes:
        
        ax = axes.flatten()[tt-1]
        
        #colorT = 'b' if tt == 1 else 'm'
        condT = 'Retarget' if tt == 1 else 'Distraction'
        h0 = 0 if infoMethod == 'omega2' else 1/len(locs)
        
        pPerms_decode1_3d = np.array([f_stats.permutation_pCI(decode_proj1_3d[region][tt].mean(1)[:,nc], decode_proj1_shuff_all_3d[region][tt].mean(1)[:,nc], tail='greater',alpha=5) for nc, cp in enumerate(checkpoints)])
        pPerms_decode2_3d = np.array([f_stats.permutation_pCI(decode_proj2_3d[region][tt].mean(1)[:,nc], decode_proj2_shuff_all_3d[region][tt].mean(1)[:,nc], tail='greater',alpha=5) for nc, cp in enumerate(checkpoints)])
        
        
        plt.subplot(1,2,tt)
        ax = plt.gca()
        ax.errorbar(np.arange(0, len(checkpoints), 1), decode_proj1_3d[region][tt].mean(1).mean(0), yerr = decode_proj1_3d[region][tt].mean(1).std(0), marker = 'o', color = 'b', label = 'Item1', capsize=5)
        trans = ax.get_xaxis_transform()
        for nc, cp in enumerate(checkpoints):
            
            if 0.01 < pPerms_decode1_3d[nc] <= 0.05:
                ax.annotate('*', xy=(nc, 0.05), xycoords = trans, annotation_clip = False, color = 'b', ha = 'center', va = 'center', fontsize=15)
            elif pPerms_decode1_3d[nc] <= 0.01:
                ax.annotate('**', xy=(nc, 0.05), xycoords = trans, annotation_clip = False, color = 'b', ha = 'center', va = 'center', fontsize=15)
            elif 0.05 < pPerms_decode1_3d[nc] <= 0.1:
                ax.annotate('+', xy=(nc, 0.05), xycoords = trans, annotation_clip = False, color = 'b', ha = 'center', va = 'center', fontsize=15)
                
        
        ax.errorbar(np.arange(0, len(checkpoints), 1), decode_proj2_3d[region][tt].mean(1).mean(0), yerr = decode_proj2_3d[region][tt].mean(1).std(0), marker = 'o', color = 'm', label = 'Item2', capsize=5)
        trans = ax.get_xaxis_transform()
        for nc, cp in enumerate(checkpoints):
            
            if 0.01 < pPerms_decode2_3d[nc] <= 0.05:
                ax.annotate('*', xy=(nc, 0.1), xycoords = trans, annotation_clip = False, color = 'm', ha = 'center', va = 'center', fontsize=15)
            elif pPerms_decode2_3d[nc] <= 0.01:
                ax.annotate('**', xy=(nc, 0.1), xycoords = trans, annotation_clip = False, color = 'm', ha = 'center', va = 'center', fontsize=15)
            elif 0.05 < pPerms_decode2_3d[nc] <= 0.1:
                ax.annotate('+', xy=(nc, 0.1), xycoords = trans, annotation_clip = False, color = 'm', ha = 'center', va = 'center', fontsize=15)
                
        
        ax.set_title(f'{condT}', fontsize = 20, pad = 10)
        ax.set_xticks([n for n in range(len(checkpoints))])
        ax.set_xticklabels(checkpointsLabels, fontsize = 15)
        ax.set_xlabel('Timebin', fontsize = 20)
        ax.set_ylim((-0.1,1))
        ax.tick_params(axis='both', labelsize=15)
        
        if tt==1:
            ax.set_ylabel(f'{infoLabel}', fontsize = 20)
        if tt==2:
            ax.legend(bbox_to_anchor=(1, 0.6),fontsize=15) #loc='upper right',
        
        
        
        
        
    plt.tight_layout()
    plt.subplots_adjust(top = 0.8)
    plt.suptitle(f'Item Subspace Information, {region.upper()}', fontsize = 25, y=1)
    plt.show()  
    
    fig.savefig(f'{phd_path}/outputs/monkeys/decodability_items_data_{region}.tif', bbox_inches='tight')
#%%


########################
# code transferability #
########################



#%%  item1 v item2 trans

Y_columnsLabels = ['locKey','locs','type','loc1','loc2','locX']
toDecode_labels1 = 'loc1'
toDecode_labels2 = 'loc2'
shuff_excludeInv = False

nPerms = 10
infoMethod = 'lda' #  'omega2' #

performanceX_Trans12, performanceX_Trans21 = {},{}
performanceX_Trans12_shuff, performanceX_Trans21_shuff = {},{}


for region in ('dlpfc','fef'):
    print(f'Region={region}')
    performanceX_Trans12[region], performanceX_Trans21[region] = {}, {}
    #decode_proj1_2d[region], decode_proj2_2d[region] = {}, {}
    
    performanceX_Trans12_shuff[region], performanceX_Trans21_shuff[region] = {},{}
    #decode_proj1_shuff_all_2d[region], decode_proj2_shuff_all_2d[region] = {},{}
    
    for tt in ttypes:
        print(f'TType={tt}')
        performanceX_Trans12T = np.zeros((nIters, nPerms, len(checkpoints), len(checkpoints))) # pca1st 3d coordinates
        performanceX_Trans21T = np.zeros((nIters, nPerms, len(checkpoints), len(checkpoints)))
        
        # shuff
        performanceX_Trans12_shuffT = np.zeros((nIters, nPerms, len(checkpoints), len(checkpoints)))
        performanceX_Trans21_shuffT = np.zeros((nIters, nPerms, len(checkpoints), len(checkpoints)))
        
        for n in range(nIters):
            
            if n%20 == 0:
                print(f'{n}')
            
            for npm in range(nPerms):
                trialInfoT = trialInfos[region][tt][n][0]
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
                    for nc_, cp_ in enumerate(checkpoints):
                        vecs1, vecs2 = vecs[region][cp][tt][1][n][0], vecs[region][cp_][tt][2][n][0]
                        projs1, projs2 = projs[region][cp][tt][1][n][0], projs[region][cp_][tt][2][n][0]
                        projs1_allT_3d, projs2_allT_3d = projsAll[region][cp][tt][1][n][0], projsAll[region][cp_][tt][2][n][0]

                        geom1 = (vecs1, projs1, projs1_allT_3d, trialInfoT, toDecode_labels1)
                        geom2 = (vecs2, projs2, projs2_allT_3d, trialInfoT, toDecode_labels2)
                        
                        info12, _ = f_subspace.plane_decodability_trans(geom1, geom2)
                        info21, _ = f_subspace.plane_decodability_trans(geom2, geom1)
                        
                        performanceX_Trans12T[n,npm,nc,nc_] = info12 #.mean(axis=-1)
                        performanceX_Trans21T[n,npm,nc_,nc] = info21 #.mean(axis=-1)
                        
                        # shuff
                        geom1_shuff = (vecs1, projs1, projs1_allT_3d, trialInfoT_shuff, toDecode_labels1)
                        geom2_shuff = (vecs2, projs2, projs2_allT_3d, trialInfoT_shuff, toDecode_labels2)
                        
                        info12_shuff, _ = f_subspace.plane_decodability_trans(geom1_shuff, geom2_shuff)
                        info21_shuff, _ = f_subspace.plane_decodability_trans(geom2_shuff, geom1_shuff)
                        
                        performanceX_Trans12_shuffT[n,npm,nc,nc_] = info12_shuff #.mean(axis=-1).mean(axis=-1)
                        performanceX_Trans21_shuffT[n,npm,nc_,nc] = info21_shuff #.mean(axis=-1).mean(axis=-1)
                    
                    
        performanceX_Trans12[region][tt] = performanceX_Trans12T
        performanceX_Trans21[region][tt] = performanceX_Trans21T
                    
        performanceX_Trans12_shuff[region][tt] = performanceX_Trans12_shuffT
        performanceX_Trans21_shuff[region][tt] = performanceX_Trans21_shuffT
        
# In[] save
np.save(f'{phd_path}/outputs/monkeys/' + 'performance12X_Trans_data.npy', performanceX_Trans12, allow_pickle=True)
np.save(f'{phd_path}/outputs/monkeys/' + 'performance21X_Trans_data.npy', performanceX_Trans21, allow_pickle=True)
np.save(f'{phd_path}/outputs/monkeys/' + 'performance12X_Trans_shuff_data.npy', performanceX_Trans12_shuff, allow_pickle=True)
np.save(f'{phd_path}/outputs/monkeys/' + 'performance21X_Trans_shuff_data.npy', performanceX_Trans21_shuff, allow_pickle=True)
#%% load
performanceX_Trans12 = np.load(f'{phd_path}/outputs/monkeys/' + 'performance12X_Trans_data.npy', allow_pickle=True).item()
performanceX_Trans21 = np.load(f'{phd_path}/outputs/monkeys/' + 'performance21X_Trans_data.npy', allow_pickle=True).item()
performanceX_Trans12_shuff = np.load(f'{phd_path}/outputs/monkeys/' + 'performance12X_Trans_shuff_data.npy', allow_pickle=True).item()
performanceX_Trans21_shuff = np.load(f'{phd_path}/outputs/monkeys/' + 'performance21X_Trans_shuff_data.npy', allow_pickle=True).item()
# In[] plot
for region in ('dlpfc','fef'):
    vmax = 0.6 if region == 'dlpfc' else 0.8
    infoLabel = 'Accuracy' if infoMethod=='lda' else 'PEV'
    
    fig, axes = plt.subplots(2,2, figsize=(12,10), dpi=100, sharex=True, sharey=True)#
    
    for tt in ttypes:
        
        #ax = axes.flatten()[tt-1]
        
        #colorT = 'b' if tt == 1 else 'm'
        condT = 'Retarget' if tt == 1 else 'Distraction'
        h0 = 0 if infoMethod == 'omega2' else 1/len(locs)
        
        pPerms_12 = np.zeros((len(checkpoints), len(checkpoints)))
        pPerms_21 = np.zeros((len(checkpoints), len(checkpoints)))
        
        for nc,cp in enumerate(checkpoints):
            for nc_, cp_ in enumerate(checkpoints):
                pPerms_12[nc,nc_] = f_stats.permutation_pCI(performanceX_Trans12[region][tt].mean(1)[:,nc,nc_], performanceX_Trans12_shuff[region][tt].mean(1)[:,nc,nc_], tail='greater',alpha=5)
                pPerms_21[nc,nc_] = f_stats.permutation_pCI(performanceX_Trans21[region][tt].mean(1)[:,nc,nc_], performanceX_Trans21_shuff[region][tt].mean(1)[:,nc,nc_], tail='greater',alpha=5)
                
        
        plt.subplot(2,2,(tt-1)*2+1)
        ax = plt.gca()
        im = ax.imshow(performanceX_Trans12[region][tt].mean(1).mean(0), cmap='magma', aspect='auto',vmax=vmax)
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
        im = ax.imshow(performanceX_Trans21[region][tt].mean(1).mean(0), cmap='magma', aspect='auto',vmax=vmax)
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
    plt.suptitle(f'Item Subspace Code Transferability, {region.upper()}', fontsize = 25, y=1)
    plt.tight_layout()
    plt.show()  
    
    fig.savefig(f'{phd_path}/outputs/monkeys/codeTransferability_data_{region}.tif', bbox_inches='tight')
#%%


#####################################
# code transferability between task #
#####################################



#%%  permutation here used to create baseline of the decodability of randomly organized subspaces, not perdicting random labels

Y_columnsLabels = ['locKey','locs','type','loc1','loc2','locX']
toDecode_labels1c = 'loc2'
toDecode_labels2c = 'loc1'
toDecode_labels1nc = 'loc1'
toDecode_labels2nc = 'loc2'
shuff_excludeInv = False

nPerms = 10
infoMethod = 'lda' #  'omega2' #

performanceX_Trans_rdc, performanceX_Trans_drc = {},{}
performanceX_Trans_rdnc, performanceX_Trans_drnc = {},{}
performanceX_Trans_rdc_shuff, performanceX_Trans_drc_shuff = {},{}
performanceX_Trans_rdnc_shuff, performanceX_Trans_drnc_shuff = {},{}


for region in ('dlpfc','fef'):
    print(f'Region={region}')
    performanceX_Trans_rdc[region], performanceX_Trans_drc[region] = {},{}
    performanceX_Trans_rdnc[region], performanceX_Trans_drnc[region] = {},{}
    
    performanceX_Trans_rdc_shuff[region], performanceX_Trans_drc_shuff[region] = {},{}
    performanceX_Trans_rdnc_shuff[region], performanceX_Trans_drnc_shuff[region] = {},{}
    
    
    performanceX_Trans_rdcT = np.zeros((nIters, nPerms, len(checkpoints), len(checkpoints))) # pca1st 3d coordinates
    performanceX_Trans_drcT = np.zeros((nIters, nPerms, len(checkpoints), len(checkpoints)))
    performanceX_Trans_rdncT = np.zeros((nIters, nPerms, len(checkpoints), len(checkpoints))) # pca1st 3d coordinates
    performanceX_Trans_drncT = np.zeros((nIters, nPerms, len(checkpoints), len(checkpoints)))
    
    # shuff
    performanceX_Trans_rdc_shuffT = np.zeros((nIters, nPerms, len(checkpoints), len(checkpoints)))
    performanceX_Trans_drc_shuffT = np.zeros((nIters, nPerms, len(checkpoints), len(checkpoints)))
    performanceX_Trans_rdnc_shuffT = np.zeros((nIters, nPerms, len(checkpoints), len(checkpoints)))
    performanceX_Trans_drnc_shuffT = np.zeros((nIters, nPerms, len(checkpoints), len(checkpoints)))
    
    for n in range(nIters):
        
        if n%20 == 0:
            print(f'{n}')
        
        for npm in range(nPerms):
            trialInfoT1 = trialInfos[region][1][n][0]
            trialInfoT2 = trialInfos[region][2][n][0]
            
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
                for nc_, cp_ in enumerate(checkpoints):
                    #choice item
                    vecs1c, vecs2c = vecs[region][cp][1][2][n][0], vecs[region][cp_][2][1][n][0]
                    projs1c, projs2c = projs[region][cp][1][2][n][0], projs[region][cp_][2][1][n][0]
                    projs1c_allT_3d, projs2c_allT_3d = projsAll[region][cp][1][2][n][0], projsAll[region][cp_][2][1][n][0]

                    geom1c = (vecs1c, projs1c, projs1c_allT_3d, trialInfoT1, toDecode_labels1c)
                    geom2c = (vecs2c, projs2c, projs2c_allT_3d, trialInfoT2, toDecode_labels2c)
                    
                    info_rdc, _ = f_subspace.plane_decodability_trans(geom1c, geom2c)
                    info_drc, _ = f_subspace.plane_decodability_trans(geom2c, geom1c)
                    
                    performanceX_Trans_rdcT[n,npm,nc,nc_] = info_rdc #.mean(axis=-1)
                    performanceX_Trans_drcT[n,npm,nc_,nc] = info_drc #.mean(axis=-1)
                    
                    # shuff
                    geom1c_shuff = (vecs1c, projs1c, projs1c_allT_3d, trialInfoT1_shuff, toDecode_labels1c)
                    geom2c_shuff = (vecs2c, projs2c, projs2c_allT_3d, trialInfoT2_shuff, toDecode_labels2c)
                    
                    info_rdc_shuff, _ = f_subspace.plane_decodability_trans(geom1c_shuff, geom2c_shuff)
                    info_drc_shuff, _ = f_subspace.plane_decodability_trans(geom2c_shuff, geom1c_shuff)
                    
                    performanceX_Trans_rdc_shuffT[n,npm,nc,nc_] = info_rdc_shuff #.mean(axis=-1).mean(axis=-1)
                    performanceX_Trans_drc_shuffT[n,npm,nc_,nc] = info_drc_shuff #.mean(axis=-1).mean(axis=-1)
                    
                    # non choice item
                    vecs1nc, vecs2nc = vecs[region][cp][1][1][n][0], vecs[region][cp_][2][2][n][0]
                    projs1nc, projs2nc = projs[region][cp][1][1][n][0], projs[region][cp_][2][2][n][0]
                    projs1nc_allT_3d, projs2nc_allT_3d = projsAll[region][cp][1][1][n][0], projsAll[region][cp_][2][2][n][0]

                    geom1nc = (vecs1nc, projs1nc, projs1nc_allT_3d, trialInfoT1, toDecode_labels1nc)
                    geom2nc = (vecs2nc, projs2nc, projs2nc_allT_3d, trialInfoT2, toDecode_labels2nc)
                    
                    info_rdnc, _ = f_subspace.plane_decodability_trans(geom1nc, geom2nc)
                    info_drnc, _ = f_subspace.plane_decodability_trans(geom2nc, geom1nc)
                    
                    performanceX_Trans_rdncT[n,npm,nc,nc_] = info_rdnc #.mean(axis=-1)
                    performanceX_Trans_drncT[n,npm,nc_,nc] = info_drnc #.mean(axis=-1)
                    
                    # shuff
                    geom1nc_shuff = (vecs1nc, projs1nc, projs1nc_allT_3d, trialInfoT1_shuff, toDecode_labels1nc)
                    geom2nc_shuff = (vecs2nc, projs2nc, projs2nc_allT_3d, trialInfoT2_shuff, toDecode_labels2nc)
                    
                    info_rdnc_shuff, _ = f_subspace.plane_decodability_trans(geom1nc_shuff, geom2nc_shuff)
                    info_drnc_shuff, _ = f_subspace.plane_decodability_trans(geom2nc_shuff, geom1nc_shuff)
                    
                    performanceX_Trans_rdnc_shuffT[n,npm,nc,nc_] = info_rdnc_shuff #.mean(axis=-1).mean(axis=-1)
                    performanceX_Trans_drnc_shuffT[n,npm,nc_,nc] = info_drnc_shuff #.mean(axis=-1).mean(axis=-1)
                
                
    performanceX_Trans_rdc[region] = performanceX_Trans_rdcT
    performanceX_Trans_drc[region] = performanceX_Trans_drcT
    performanceX_Trans_rdnc[region] = performanceX_Trans_rdncT
    performanceX_Trans_drnc[region] = performanceX_Trans_drncT
                
    performanceX_Trans_rdc_shuff[region] = performanceX_Trans_rdc_shuffT
    performanceX_Trans_drc_shuff[region] = performanceX_Trans_drc_shuffT
    performanceX_Trans_rdnc_shuff[region] = performanceX_Trans_rdnc_shuffT
    performanceX_Trans_drnc_shuff[region] = performanceX_Trans_drnc_shuffT
        
# In[] save
np.save(f'{phd_path}/outputs/monkeys/' + 'performanceX_Trans_rdc_data.npy', performanceX_Trans_rdc, allow_pickle=True)
np.save(f'{phd_path}/outputs/monkeys/' + 'performanceX_Trans_drc_data.npy', performanceX_Trans_drc, allow_pickle=True)
np.save(f'{phd_path}/outputs/monkeys/' + 'performanceX_Trans_rdnc_data.npy', performanceX_Trans_rdnc, allow_pickle=True)
np.save(f'{phd_path}/outputs/monkeys/' + 'performanceX_Trans_drnc_data.npy', performanceX_Trans_drnc, allow_pickle=True)

np.save(f'{phd_path}/outputs/monkeys/' + 'performanceX_Trans_rdc_shuff_data.npy', performanceX_Trans_rdc_shuff, allow_pickle=True)
np.save(f'{phd_path}/outputs/monkeys/' + 'performanceX_Trans_drc_shuff_data.npy', performanceX_Trans_drc_shuff, allow_pickle=True)
np.save(f'{phd_path}/outputs/monkeys/' + 'performanceX_Trans_rdnc_shuff_data.npy', performanceX_Trans_rdnc_shuff, allow_pickle=True)
np.save(f'{phd_path}/outputs/monkeys/' + 'performanceX_Trans_drnc_shuff_data.npy', performanceX_Trans_drnc_shuff, allow_pickle=True)
#%% load
performanceX_Trans_rdc = np.load(f'{phd_path}/outputs/monkeys/' + 'performanceX_Trans_rdc_data.npy', allow_pickle=True).item()
performanceX_Trans_drc = np.load(f'{phd_path}/outputs/monkeys/' + 'performanceX_Trans_drc_data.npy', allow_pickle=True).item()
performanceX_Trans_rdc_shuff = np.load(f'{phd_path}/outputs/monkeys/' + 'performanceX_Trans_rdc_shuff_data.npy', allow_pickle=True).item()
performanceX_Trans_drc_shuff = np.load(f'{phd_path}/outputs/monkeys/' + 'performanceX_Trans_drc_shuff_data.npy', allow_pickle=True).item()

performanceX_Trans_rdnc = np.load(f'{phd_path}/outputs/monkeys/' + 'performanceX_Trans_rdnc_data.npy', allow_pickle=True).item()
performanceX_Trans_drnc = np.load(f'{phd_path}/outputs/monkeys/' + 'performanceX_Trans_drnc_data.npy', allow_pickle=True).item()
performanceX_Trans_rdnc_shuff = np.load(f'{phd_path}/outputs/monkeys/' + 'performanceX_Trans_rdnc_shuff_data.npy', allow_pickle=True).item()
performanceX_Trans_drnc_shuff = np.load(f'{phd_path}/outputs/monkeys/' + 'performanceX_Trans_drnc_shuff_data.npy', allow_pickle=True).item()
# In[] plot
for region in ('dlpfc','fef'):
    vmax = 0.6 if region == 'dlpfc' else 0.8
    infoLabel = 'Accuracy' if infoMethod=='lda' else 'PEV'
    
    fig, axes = plt.subplots(2,2, figsize=(12,10), dpi=100, sharex=True, sharey=True)#
    
        
    pPerms_rdc = np.zeros((len(checkpoints), len(checkpoints)))
    pPerms_drc = np.zeros((len(checkpoints), len(checkpoints)))
    pPerms_rdnc = np.zeros((len(checkpoints), len(checkpoints)))
    pPerms_drnc = np.zeros((len(checkpoints), len(checkpoints)))
    
    for nc,cp in enumerate(checkpoints):
        for nc_, cp_ in enumerate(checkpoints):
            pPerms_rdc[nc,nc_] = f_stats.permutation_pCI(performanceX_Trans_rdc[region].mean(1)[:,nc,nc_], performanceX_Trans_rdc_shuff[region].mean(1)[:,nc,nc_], tail='greater',alpha=5)
            pPerms_drc[nc,nc_] = f_stats.permutation_pCI(performanceX_Trans_drc[region].mean(1)[:,nc,nc_], performanceX_Trans_drc_shuff[region].mean(1)[:,nc,nc_], tail='greater',alpha=5)
            pPerms_rdnc[nc,nc_] = f_stats.permutation_pCI(performanceX_Trans_rdnc[region].mean(1)[:,nc,nc_], performanceX_Trans_rdnc_shuff[region].mean(1)[:,nc,nc_], tail='greater',alpha=5)
            pPerms_drnc[nc,nc_] = f_stats.permutation_pCI(performanceX_Trans_drnc[region].mean(1)[:,nc,nc_], performanceX_Trans_drnc_shuff[region].mean(1)[:,nc,nc_], tail='greater',alpha=5)
    
            
    
    plt.subplot(2,2,1)
    ax1 = plt.gca()
    im1 = ax1.imshow(performanceX_Trans_rdc[region].mean(1).mean(0), cmap='magma', aspect='auto',vmax=vmax)
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
    im2 = ax2.imshow(performanceX_Trans_drc[region].mean(1).mean(0), cmap='magma', aspect='auto',vmax=vmax)
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
    im3 = ax3.imshow(performanceX_Trans_rdnc[region].mean(1).mean(0), cmap='magma', aspect='auto',vmax=vmax)
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
    im4 = ax4.imshow(performanceX_Trans_drnc[region].mean(1).mean(0), cmap='magma', aspect='auto',vmax=vmax)
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
    plt.suptitle(f'Choice/Nonchoice Item Code Transferability, {region.upper()}', fontsize = 25, y=1)
    plt.tight_layout()
    plt.show()  
    
    fig.savefig(f'{phd_path}/outputs/monkeys/codeTransferability_cnc_data_{region}.tif', bbox_inches='tight')
#%%

#####################################
######### parallel baseline #########
#####################################

#%%
nIters = 100
nPerms = 100
nBoots = 1
fracBoots = 1.0

tBsl = (-300,0)
idxxBsl = [tsliceRange.tolist().index(tBsl[0]), tsliceRange.tolist().index(tBsl[1])]

checkpoints = [150, 550, 1150, 1450, 1850, 2350, 2800]#
#avgInterval = 50
avgInterval = {150:150, 550:250, 1150:250, 1450:150, 1850:250, 2350:250, 2800:200}
checkpointsLabels = ['S1','ED1','LD2','S2','ED2','LD2', 'Go']
toPlot=False # 
#decode_method = 'omega2' #'polyArea''lda' 
avgMethod='conditional_time' # 'conditional' 'all' 'none'

#shuff_excludeInv = True
# In[]
vecs_bsl = {}
projs_bsl = {}
projsAll_bsl = {}
trialInfos_bsl = {}
pca1s_bsl = {}


for region in ('dlpfc','fef'):
    vecs_bsl[region] = {}
    projs_bsl[region] = {}
    projsAll_bsl[region] = {}
    trialInfos_bsl[region] = {}
    pca1s_bsl[region] = []
    
    for tt in ttypes:
        trialInfos_bsl[region][tt] = []
        
    
    for cp in checkpoints:
        vecs_bsl[region][cp] = {}
        projs_bsl[region][cp] = {}
        projsAll_bsl[region][cp] = {}
        
        for tt in ttypes:
            vecs_bsl[region][cp][tt] = {1:[], 2:[]}
            projs_bsl[region][cp][tt] = {1:[], 2:[]}
            projsAll_bsl[region][cp][tt] = {1:[], 2:[]}
            
# In[]

#n = 50
#while n < 100:
for n in range(nIters):
    t_IterOn = time.time()
    print(f'Iter = {n}')
    
    pseudo_data = np.load(save_path + f'/pseudo_all{n}.npy', allow_pickle=True).item()
    pseudo_region = pseudo_data['pseudo_region']
    pseudo_TrialInfo = pseudo_data['pseudo_TrialInfo']

    
    print(f'tIter = {(time.time() - t_IterOn):.4f}s')
    #n += 1
    
    for region in ('dlpfc','fef'):
        
        
        trialInfo = pseudo_TrialInfo.reset_index(names=['id'])
        
        idx1 = trialInfo.id # original index
        #idx2 = trialInfo.index.to_list() # reset index
        
        # if specify applied time window of pca
        pca_tWin = np.hstack((np.arange(300,1300,dt, dtype = int),np.arange(1600,2600,dt, dtype = int))).tolist() #
        pca_tWinX = [tsliceRange.tolist().index(i) for i in pca_tWin]
        
        ### main test
        
        dataN = pseudo_region[region][idx1,::]
        
        # if detrend by subtract avg
        for ch in range(dataN.shape[1]):
            temp = dataN[:,ch,:]
            dataN[:,ch,:] = (temp - temp.mean(axis=0))#/(temp.std(axis = 0) + epsilon) # to avoid divid by 0 issue
            
        #X_region = pseudo_PopT
        
        
        for ch in range(dataN.shape[1]):
            #dataN[:,ch,:] = scale(dataN[:,ch,:])#01 scale for each channel
            dataN[:,ch,:] = (dataN[:,ch,:] - dataN[:,ch,:].mean()) / dataN[:,ch,:].std() #standard scaler
            #dataN[:,ch,:] = (dataN[:,ch,:] - dataN[:,ch,:].mean(axis=0)) / dataN[:,ch,:].std(axis=0) # detrened standard scaler
            #dataN[:,ch,:] = (dataN[:,ch,:] - dataN[:,ch,idxxBsl[0]:idxxBsl[1]].mean())/dataN[:,ch,idxxBsl[0]:idxxBsl[1]].std() # Z to baseline, 
        
        # baseline z-normalize data (if not normalized)
        #for ch in range(dataN.shape[1]):
        
            #dataN[:,ch,:] = scale(dataN[:,ch,:])
        
        pca1s_bsl[region].append([])
        
        for tt in ttypes:
            trialInfos_bsl[region][tt].append([])
            
            
        for cp in checkpoints:
            for tt in ttypes:
                for ll in (1,2,):
                    vecs_bsl[region][cp][tt][ll].append([])
                    projs_bsl[region][cp][tt][ll].append([])
                    projsAll_bsl[region][cp][tt][ll].append([])
                    #Xs_mean[region][cp][tt][ll].append([])
                    #pca2s[region][cp][tt][ll].append([])
        
        for nboot in range(nPerms): 
            # nBoots
            #idxT,_ = f_subspace.split_set(np.arange(dataN.shape[0]), frac=fracBoots, ranseed=nboot)
            idxT,_ = f_subspace.split_set_balance(np.arange(dataN.shape[0]), trialInfo, frac=0.5, ranseed=nboot)
            dataT = dataN[idxT,:,:]
            trialInfoT = trialInfo.loc[idxT,:].reset_index(drop=True)

            pca1_C = pca1s_C[region][n][0]
            
            vecs_D, projs_D, projsAll_D, _, trialInfos_D, _, _, evr_1st, pca1 = f_subspace.plane_fitting_analysis(dataT, trialInfoT, pca_tWinX, checkpoints, tsliceRange, avgInterval, locs, ttypes, dropCombs, 
                                                                                                                  toPlot=toPlot, avgMethod = avgMethod, adaptPCA=pca1_C) #, decode_method = decode_method
            #decodability_projD
            
            pca1s_bsl[region][n] += [pca1]
            
            for tt in ttypes:
                trialInfos_bsl[region][tt][n] += [trialInfos_D[tt]]
                
            
            for cp in checkpoints:
                for tt in ttypes:
                    for ll in (1,2,):
                        vecs_bsl[region][cp][tt][ll][n] += [vecs_D[cp][tt][ll]]
                        projs_bsl[region][cp][tt][ll][n] += [projs_D[cp][tt][ll]]
                        projsAll_bsl[region][cp][tt][ll][n] += [projsAll_D[cp][tt][ll]]
                        #Xs_mean[region][cp][tt][ll][n] += [X_mean[cp][tt][ll]]
                        #pca2s[region][cp][tt][ll][n] += [pca2]
                        
            
            print(f'EVRs: {evr_1st.round(5)}')
            
            
            
        print(f'tPerm({nPerms*nBoots}) = {(time.time() - t_IterOn):.4f}s, {region}')
        
        #n += 1


# In[]
np.save(f'{phd_path}/fitting planes/pooled/w&w/' + 'vecs_bsl_detrended.npy', vecs_bsl, allow_pickle=True)
np.save(f'{phd_path}/fitting planes/pooled/w&w/' + 'projs_bsl_detrended.npy', projs_bsl, allow_pickle=True)
np.save(f'{phd_path}/fitting planes/pooled/w&w/' + 'projsAll_bsl_detrended.npy', projsAll_bsl, allow_pickle=True)
#np.save(f'{phd_path}/fitting planes/pooled/w&w/' + 'Xs_mean_detrended.npy', Xs_mean, allow_pickle=True)
np.save(f'{phd_path}/fitting planes/pooled/w&w/' + 'trialInfos_bsl_detrended.npy', trialInfos_bsl, allow_pickle=True)
# In[] exclude post-gocue window
checkpoints = [150, 550, 1150, 1450, 1850, 2350]#
checkpointsLabels = ['S1','ED1','LD1','S2','ED2','LD2']

#%%                    


#######################
# compare item v item #
#######################


# In[] item v item cosTheta, cosPsi, sse. Compare within type, between time points, between locations
pdummy = True #False #

cosTheta_11_bsl, cosTheta_12_bsl, cosTheta_22_bsl = {},{},{}
cosPsi_11_bsl, cosPsi_12_bsl, cosPsi_22_bsl = {},{},{}


for region in ('dlpfc','fef'):
    
    cosTheta_11_bsl[region], cosTheta_22_bsl[region], cosTheta_12_bsl[region] = {},{},{}
    cosPsi_11_bsl[region], cosPsi_22_bsl[region], cosPsi_12_bsl[region] = {},{},{}
    
    
    for tt in ttypes:
        ttype = 'Retarget' if tt==1 else 'Distraction'
        
        cosTheta_11T = np.zeros((nIters, nPerms, len(checkpoints)))
        cosTheta_22T = np.zeros((nIters, nPerms, len(checkpoints)))
        cosTheta_12T = np.zeros((nIters, nPerms, len(checkpoints)))
        
        cosPsi_11T = np.zeros((nIters, nPerms, len(checkpoints)))
        cosPsi_22T = np.zeros((nIters, nPerms, len(checkpoints)))
        cosPsi_12T = np.zeros((nIters, nPerms, len(checkpoints)))
        
        
        for n in range(nIters):
                
            for npm in range(nPerms):
                for nc, cp in enumerate(checkpoints):
                    
                    cT11_bsl, _, cP11_bsl, _, _ = f_subspace.angle_alignment_coplanar(vecs[region][cp][tt][1][n][0], projs[region][cp][tt][1][n][0], vecs_bsl[region][cp][tt][1][n][npm], projs_bsl[region][cp][tt][1][n][npm])
                    cT22_bsl, _, cP22_bsl, _, _ = f_subspace.angle_alignment_coplanar(vecs[region][cp][tt][2][n][0], projs[region][cp][tt][2][n][0], vecs_bsl[region][cp][tt][2][n][npm], projs_bsl[region][cp][tt][2][n][npm])
                    cT12_bsl, _, cP12_bsl, _, _ = f_subspace.angle_alignment_coplanar(vecs[region][cp][tt][1][n][0], projs[region][cp][tt][1][n][0], vecs_bsl[region][cp][tt][2][n][npm], projs_bsl[region][cp][tt][2][n][npm])
                    
                    cosTheta_11T[n,npm,nc], cosTheta_22T[n,npm,nc], cosTheta_12T[n,npm,nc] = cT11_bsl, cT22_bsl, cT12_bsl# theta11, theta22, theta12# 
                    cosPsi_11T[n,npm,nc], cosPsi_22T[n,npm,nc], cosPsi_12T[n,npm,nc] = cP11_bsl, cP22_bsl, cP12_bsl# psi11, psi22, psi12# 
                    
        
        cosTheta_11_bsl[region][tt] = cosTheta_11T
        cosTheta_22_bsl[region][tt] = cosTheta_22T
        cosTheta_12_bsl[region][tt] = cosTheta_12T
        
        cosPsi_11_bsl[region][tt] = cosPsi_11T
        cosPsi_22_bsl[region][tt] = cosPsi_22T
        cosPsi_12_bsl[region][tt] = cosPsi_12T
        
#%%
np.save(f'{phd_path}/outputs/monkeys/' + 'cosTheta_11_bsl_data.npy', cosTheta_11_bsl, allow_pickle=True)
np.save(f'{phd_path}/outputs/monkeys/' + 'cosTheta_12_bsl_data.npy', cosTheta_12_bsl, allow_pickle=True)
np.save(f'{phd_path}/outputs/monkeys/' + 'cosTheta_22_bsl_data.npy', cosTheta_22_bsl, allow_pickle=True)
np.save(f'{phd_path}/outputs/monkeys/' + 'cosPsi_11_bsl_data.npy', cosPsi_11_bsl, allow_pickle=True)
np.save(f'{phd_path}/outputs/monkeys/' + 'cosPsi_12_bsl_data.npy', cosPsi_12_bsl, allow_pickle=True)
np.save(f'{phd_path}/outputs/monkeys/' + 'cosPsi_22_bsl_data.npy', cosPsi_22_bsl, allow_pickle=True)     

#%%

#####################################
######### parallel baseline #########
#####################################

#%%
nIters = 100
nPerms = 100
nBoots = 1
fracBoots = 1.0

tBsl = (-300,0)
idxxBsl = [tsliceRange.tolist().index(tBsl[0]), tsliceRange.tolist().index(tBsl[1])]

checkpoints = [150, 550, 1050, 1450, 1850, 2350, 2800]#
#avgInterval = 50
avgInterval = {150:150, 550:250, 1050:250, 1450:150, 1850:250, 2350:250, 2800:200}
checkpointsLabels = ['S1','ED1','LD2','S2','ED2','LD2', 'Go']
toPlot=False # 
#decode_method = 'omega2' #'polyArea''lda' 
avgMethod='conditional_time' # 'conditional' 'all' 'none'


# In[] method 2: train-test sets
vecs_bsl_train = {}
projs_bsl_train = {}
projsAll_bsl_train = {}
trialInfos_bsl_train = {}
pca1s_bsl_train = {}

vecs_bsl_test = {}
projs_bsl_test = {}
projsAll_bsl_test = {}
trialInfos_bsl_test = {}
pca1s_bsl_test = {}

for region in ('dlpfc','fef'):
    vecs_bsl_train[region] = {}
    projs_bsl_train[region] = {}
    projsAll_bsl_train[region] = {}
    trialInfos_bsl_train[region] = {}
    pca1s_bsl_train[region] = []
    
    vecs_bsl_test[region] = {}
    projs_bsl_test[region] = {}
    projsAll_bsl_test[region] = {}
    trialInfos_bsl_test[region] = {}
    pca1s_bsl_test[region] = []

    for tt in ttypes:
        trialInfos_bsl_train[region][tt] = []
        trialInfos_bsl_test[region][tt] = []
        
    
    for cp in checkpoints:
        vecs_bsl_train[region][cp] = {}
        projs_bsl_train[region][cp] = {}
        projsAll_bsl_train[region][cp] = {}
        
        vecs_bsl_test[region][cp] = {}
        projs_bsl_test[region][cp] = {}
        projsAll_bsl_test[region][cp] = {}

        for tt in ttypes:
            vecs_bsl_train[region][cp][tt] = {1:[], 2:[]}
            projs_bsl_train[region][cp][tt] = {1:[], 2:[]}
            projsAll_bsl_train[region][cp][tt] = {1:[], 2:[]}

            vecs_bsl_test[region][cp][tt] = {1:[], 2:[]}
            projs_bsl_test[region][cp][tt] = {1:[], 2:[]}
            projsAll_bsl_test[region][cp][tt] = {1:[], 2:[]}

#%%
for n in range(nIters):
    t_IterOn = time.time()
    print(f'Iter = {n}')
    
    pseudo_data = np.load(save_path + f'/pseudo_all{n}.npy', allow_pickle=True).item()
    pseudo_region = pseudo_data['pseudo_region']
    pseudo_TrialInfo = pseudo_data['pseudo_TrialInfo']

    
    print(f'tIter = {(time.time() - t_IterOn):.4f}s')

    for region in ('dlpfc','fef'):
        
        
        trialInfo = pseudo_TrialInfo.reset_index(names=['id'])
        
        idx1 = trialInfo.id # original index
        
        # if specify applied time window of pca
        pca_tWin = np.hstack((np.arange(300,1300,dt, dtype = int),np.arange(1600,2600,dt, dtype = int))).tolist() #
        pca_tWinX = [tsliceRange.tolist().index(i) for i in pca_tWin]
        
        ### main test
        
        dataN = pseudo_region[region][idx1,::]
        
        # if detrend by subtract avg
        for ch in range(dataN.shape[1]):
            temp = dataN[:,ch,:]
            dataN[:,ch,:] = (temp - temp.mean(axis=0))#/(temp.std(axis = 0) + epsilon) # to avoid divid by 0 issue
            
        
        for ch in range(dataN.shape[1]):
            dataN[:,ch,:] = (dataN[:,ch,:] - dataN[:,ch,:].mean()) / dataN[:,ch,:].std() #standard scaler
            
        pca1s_bsl_train[region].append([])
        pca1s_bsl_test[region].append([])
        
        for tt in ttypes:
            trialInfos_bsl_train[region][tt].append([])
            trialInfos_bsl_test[region][tt].append([])
            
            
        for cp in checkpoints:
            for tt in ttypes:
                for ll in (1,2,):
                    vecs_bsl_train[region][cp][tt][ll].append([])
                    projs_bsl_train[region][cp][tt][ll].append([])
                    projsAll_bsl_train[region][cp][tt][ll].append([])

                    vecs_bsl_test[region][cp][tt][ll].append([])
                    projs_bsl_test[region][cp][tt][ll].append([])
                    projsAll_bsl_test[region][cp][tt][ll].append([])

                    
        for nboot in range(nPerms): 
            # nBoots
            idxT1,idxT2 = f_subspace.split_set_balance(np.arange(dataN.shape[0]), trialInfo, frac=0.5, ranseed=nboot)
            dataT1, dataT2 = dataN[idxT1,:,:], dataN[idxT2,:,:]
            trialInfoT1, trialInfoT2 = trialInfo.loc[idxT1,:].reset_index(drop=True), trialInfo.loc[idxT2,:].reset_index(drop=True)
            #trialInfoT = trialInfo.loc[idxT,:].reset_index(drop=True)

            pca1_C = pca1s_C[region][n][0]
            
            vecs_D1, projs_D1, projsAll_D1, _, trialInfos_D1, _, _, evr_1st_1, pca1_1 = f_subspace.plane_fitting_analysis(dataT1, trialInfoT1, pca_tWinX, checkpoints, tsliceRange, avgInterval, locs, ttypes, dropCombs, 
                                                                                                                          toPlot=toPlot, avgMethod = avgMethod, adaptPCA=pca1_C) #, decode_method = decode_method
            vecs_D2, projs_D2, projsAll_D2, _, trialInfos_D2, _, _, evr_1st_2, pca1_2 = f_subspace.plane_fitting_analysis(dataT2, trialInfoT2, pca_tWinX, checkpoints, tsliceRange, avgInterval, locs, ttypes, dropCombs,
                                                                                                                          toPlot=toPlot, avgMethod = avgMethod, adaptPCA=pca1_C)

            #decodability_projD
            
            pca1s_bsl_train[region][n] += [pca1_1]
            pca1s_bsl_test[region][n] += [pca1_2]
            
            for tt in ttypes:
                trialInfos_bsl_train[region][tt][n] += [trialInfos_D1[tt]]
                trialInfos_bsl_test[region][tt][n] += [trialInfos_D2[tt]]
                
            
            for cp in checkpoints:
                for tt in ttypes:
                    for ll in (1,2,):
                        vecs_bsl_train[region][cp][tt][ll][n] += [vecs_D1[cp][tt][ll]]
                        projs_bsl_train[region][cp][tt][ll][n] += [projs_D1[cp][tt][ll]]
                        projsAll_bsl_train[region][cp][tt][ll][n] += [projsAll_D1[cp][tt][ll]]

                        vecs_bsl_test[region][cp][tt][ll][n] += [vecs_D2[cp][tt][ll]]
                        projs_bsl_test[region][cp][tt][ll][n] += [projs_D2[cp][tt][ll]]
                        projsAll_bsl_test[region][cp][tt][ll][n] += [projsAll_D2[cp][tt][ll]]
                        
            
            print(f'EVRs: {evr_1st_1.round(5)}, {evr_1st_2.round(5)}')
            
            
            
        print(f'tPerm({nPerms*nBoots}) = {(time.time() - t_IterOn):.4f}s, {region}')
        
        #n += 1


# In[]
np.save(f'{phd_path}/fitting planes/pooled/w&w/' + 'vecs_bsl_train_detrended.npy', vecs_bsl_train, allow_pickle=True)
np.save(f'{phd_path}/fitting planes/pooled/w&w/' + 'projs_bsl_train_detrended.npy', projs_bsl_train, allow_pickle=True)
np.save(f'{phd_path}/fitting planes/pooled/w&w/' + 'projsAll_bsl_train_detrended.npy', projsAll_bsl_train, allow_pickle=True)
np.save(f'{phd_path}/fitting planes/pooled/w&w/' + 'trialInfos_bsl_train_detrended.npy', trialInfos_bsl_train, allow_pickle=True)

np.save(f'{phd_path}/fitting planes/pooled/w&w/' + 'vecs_bsl_test_detrended.npy', vecs_bsl_test, allow_pickle=True)
np.save(f'{phd_path}/fitting planes/pooled/w&w/' + 'projs_bsl_test_detrended.npy', projs_bsl_test, allow_pickle=True)
np.save(f'{phd_path}/fitting planes/pooled/w&w/' + 'projsAll_bsl_test_detrended.npy', projsAll_bsl_test, allow_pickle=True)
np.save(f'{phd_path}/fitting planes/pooled/w&w/' + 'trialInfos_bsl_test_detrended.npy', trialInfos_bsl_test, allow_pickle=True)
#%%
checkpoints = [150, 550, 1050, 1450, 1850, 2350]#
checkpointsLabels = ['S1','ED1','LD1','S2','ED2','LD2']

#%%                    


#######################
# compare item v item #
#######################


# In[] item v item cosTheta, cosPsi, sse. Compare within type, between time points, between locations
pdummy = True #False #

cosTheta_11_bsl, cosTheta_12_bsl, cosTheta_22_bsl = {},{},{}
cosPsi_11_bsl, cosPsi_12_bsl, cosPsi_22_bsl = {},{},{}


for region in ('dlpfc','fef'):
    
    cosTheta_11_bsl[region], cosTheta_22_bsl[region], cosTheta_12_bsl[region] = {},{},{}
    cosPsi_11_bsl[region], cosPsi_22_bsl[region], cosPsi_12_bsl[region] = {},{},{}
    
    
    for tt in ttypes:
        ttype = 'Retarget' if tt==1 else 'Distraction'
        
        cosTheta_11T = np.zeros((nIters, nPerms, len(checkpoints)))
        cosTheta_22T = np.zeros((nIters, nPerms, len(checkpoints)))
        cosTheta_12T = np.zeros((nIters, nPerms, len(checkpoints)))
        
        cosPsi_11T = np.zeros((nIters, nPerms, len(checkpoints)))
        cosPsi_22T = np.zeros((nIters, nPerms, len(checkpoints)))
        cosPsi_12T = np.zeros((nIters, nPerms, len(checkpoints)))
        
        
        for n in range(nIters):
                
            for npm in range(nPerms):
                for nc, cp in enumerate(checkpoints):
                    
                    cT11_bsl, _, cP11_bsl, _, _ = f_subspace.angle_alignment_coplanar(vecs_bsl_train[region][cp][tt][1][n][npm], projs_bsl_train[region][cp][tt][1][n][npm], vecs_bsl_test[region][cp][tt][1][n][npm], projs_bsl_test[region][cp][tt][1][n][npm])
                    cT22_bsl, _, cP22_bsl, _, _ = f_subspace.angle_alignment_coplanar(vecs_bsl_train[region][cp][tt][2][n][npm], projs_bsl_train[region][cp][tt][2][n][npm], vecs_bsl_test[region][cp][tt][2][n][npm], projs_bsl_test[region][cp][tt][2][n][npm])
                    cT12_bsl, _, cP12_bsl, _, _ = f_subspace.angle_alignment_coplanar(vecs_bsl_train[region][cp][tt][1][n][npm], projs_bsl_train[region][cp][tt][1][n][npm], vecs_bsl_test[region][cp][tt][2][n][npm], projs_bsl_test[region][cp][tt][2][n][npm])
                    
                    cosTheta_11T[n,npm,nc], cosTheta_22T[n,npm,nc], cosTheta_12T[n,npm,nc] = cT11_bsl, cT22_bsl, cT12_bsl# theta11, theta22, theta12# 
                    cosPsi_11T[n,npm,nc], cosPsi_22T[n,npm,nc], cosPsi_12T[n,npm,nc] = cP11_bsl, cP22_bsl, cP12_bsl# psi11, psi22, psi12# 
                    
        
        cosTheta_11_bsl[region][tt] = cosTheta_11T
        cosTheta_22_bsl[region][tt] = cosTheta_22T
        cosTheta_12_bsl[region][tt] = cosTheta_12T
        
        cosPsi_11_bsl[region][tt] = cosPsi_11T
        cosPsi_22_bsl[region][tt] = cosPsi_22T
        cosPsi_12_bsl[region][tt] = cosPsi_12T
        
#%%
np.save(f'{phd_path}/outputs/monkeys/' + 'cosTheta_11_bsl_data.npy', cosTheta_11_bsl, allow_pickle=True)
np.save(f'{phd_path}/outputs/monkeys/' + 'cosTheta_12_bsl_data.npy', cosTheta_12_bsl, allow_pickle=True)
np.save(f'{phd_path}/outputs/monkeys/' + 'cosTheta_22_bsl_data.npy', cosTheta_22_bsl, allow_pickle=True)
np.save(f'{phd_path}/outputs/monkeys/' + 'cosPsi_11_bsl_data.npy', cosPsi_11_bsl, allow_pickle=True)
np.save(f'{phd_path}/outputs/monkeys/' + 'cosPsi_12_bsl_data.npy', cosPsi_12_bsl, allow_pickle=True)
np.save(f'{phd_path}/outputs/monkeys/' + 'cosPsi_22_bsl_data.npy', cosPsi_22_bsl, allow_pickle=True)     
#%%















































































































#%%
############
# non used #
############
# In[] I1I2_LD1 vs I1I2_LD2

ld1x, ld2x = checkpointsLabels.index('LD1'), checkpointsLabels.index('LD2')


for region in ('dlpfc','fef'):
    
        
    cosTheta_choice_d1 = cosTheta_choice[region][:,:,ld1x].mean(1)
    cosTheta_choice_d2 = cosTheta_choice[region][:,:,ld2x].mean(1)
    cosPsi_choice_d1 = cosPsi_choice[region][:,:,ld1x].mean(1)
    cosPsi_choice_d2 = cosPsi_choice[region][:,:,ld2x].mean(1)
    
    #stats.kstest(cosTheta_1121, cosTheta_1122)[-1]
    print(f'choice theta, {region}: ', stats.ttest_rel(cosTheta_choice_d1, cosTheta_choice_d2))
    print(f'choice psi, {region}: ', stats.ttest_rel(cosPsi_choice_d1, cosPsi_choice_d2))
    
    cosTheta_nonchoice_d1 = cosTheta_nonchoice[region][:,:,ld1x].mean(1)
    cosTheta_nonchoice_d2 = cosTheta_nonchoice[region][:,:,ld2x].mean(1)
    cosPsi_nonchoice_d1 = cosPsi_nonchoice[region][:,:,ld1x].mean(1)
    cosPsi_nonchoice_d2 = cosPsi_nonchoice[region][:,:,ld2x].mean(1)
    
    print(f'nonchoice theta, {region}: ', stats.ttest_rel(cosTheta_nonchoice_d1, cosTheta_nonchoice_d2))
    print(f'nonchoice psi, {region}: ', stats.ttest_rel(cosPsi_nonchoice_d1, cosPsi_nonchoice_d2))
    
    plt.figure(figsize=(5, 4), dpi=100)
    
    bpl = plt.boxplot([cosTheta_choice[region][:,:,ld1x].mean(1), cosTheta_nonchoice[region][:,:,ld1x].mean(1)], positions=[0.4,1.4], flierprops=dict(markeredgecolor='#00A7C8')) #
    bpr = plt.boxplot([cosTheta_choice[region][:,:,ld2x].mean(1), cosTheta_nonchoice[region][:,:,ld2x].mean(1)], positions=[0.7,1.7], flierprops=dict(markeredgecolor='#FF8787')) #
    f_plotting.set_box_color(bpl, '#00A7C8') # colors are from http://colorbrewer2.org/
    f_plotting.set_box_color(bpr, '#FF8787')

    # draw temporary red and blue lines and use them to create a legend
    plt.plot([], c='#00A7C8', label='LD1')
    plt.plot([], c='#FF8787', label='LD2')
    plt.title(f'LD1 vs. LD2, {region.upper()}', fontsize = 15)
    plt.legend()

    plt.xticks([0.5,1.5],['Choice','Non-Choice'])
    
    plt.yticks(np.flip(np.cos(angleCheckPoints).round(5)), labels = np.flip(np.degrees(angleCheckPoints).round(1).astype(int)))
    plt.xlabel('Item', labelpad = 3, fontsize = 12)
    plt.ylabel('cos(θ)', labelpad = 3, fontsize = 12)
    plt.show()
            
    plt.figure(figsize=(5, 4), dpi=100)
    
    bpl = plt.boxplot([cosPsi_choice[region][:,:,ld1x].mean(1), cosPsi_nonchoice[region][:,:,ld1x].mean(1)], positions=[0.4,1.4], flierprops=dict(markeredgecolor='#00A7C8')) #
    bpr = plt.boxplot([cosPsi_choice[region][:,:,ld2x].mean(1), cosPsi_nonchoice[region][:,:,ld2x].mean(1)], positions=[0.7,1.7], flierprops=dict(markeredgecolor='#FF8787')) #
    f_plotting.set_box_color(bpl, '#00A7C8') # colors are from http://colorbrewer2.org/
    f_plotting.set_box_color(bpr, '#FF8787')

    # draw temporary red and blue lines and use them to create a legend
    plt.plot([], c='#00A7C8', label='LD1')
    plt.plot([], c='#FF8787', label='LD2')
    plt.legend()

    plt.xticks([0.5,1.5],['Choice','Non-Choice'])
    
    plt.yticks(np.flip(np.cos(angleCheckPoints).round(5)), labels = np.flip(np.degrees(angleCheckPoints).round(1).astype(int)))
    plt.xlabel('Item', labelpad = 3, fontsize = 12)
    plt.ylabel('cos(Ψ)', labelpad = 3, fontsize = 12)
    plt.title(f'LD1 vs. LD2, {region.upper()}', fontsize = 15)
    plt.show()

# In[] decodability plane projection by omega2

Y_columnsLabels = ['locKey','locs','type','loc1','loc2','locX']
toDecode_labels1 = 'loc1'
toDecode_labels2 = 'loc2'
shuff_excludeInv = True


infoMethod = 'lda' #  'omega2' #

decode_proj1_3d, decode_proj2_3d = {},{}
decode_proj1_shuff_all_3d, decode_proj2_shuff_all_3d = {},{}


for region in ('dlpfc','fef'):
    print(f'Region={region}')
    decode_proj1_3d[region], decode_proj2_3d[region] = {}, {}
    #decode_proj1_2d[region], decode_proj2_2d[region] = {}, {}
    
    decode_proj1_shuff_all_3d[region], decode_proj2_shuff_all_3d[region] = {},{}
    #decode_proj1_shuff_all_2d[region], decode_proj2_shuff_all_2d[region] = {},{}
    
    for tt in ttypes:
        print(f'TType={tt}')
        decode_proj1T_3d = np.zeros((nIters, nBoots, len(checkpoints),3)) # pca1st 3d coordinates
        decode_proj2T_3d = np.zeros((nIters, nBoots, len(checkpoints),3))
        
        # shuff
        decode_proj1T_3d_shuff = np.zeros((nIters, nBoots*nPerms, len(checkpoints),3))
        decode_proj2T_3d_shuff = np.zeros((nIters, nBoots*nPerms, len(checkpoints),3))
        
        for n in range(nIters):
            
            if n%20 == 0:
                print(f'{n}')
            
            for nbt in range(nBoots):
                trialInfoT = trialInfos[region][tt][n][nbt]
                
                for nc,cp in enumerate(checkpoints):
                    vecs1, vecs2 = vecs[region][cp][tt][1][n][nbt], vecs[region][cp][tt][2][n][nbt]
                    projs1, projs2 = projs[region][cp][tt][1][n][nbt], projs[region][cp][tt][2][n][nbt]
                    projs1_allT_3d, projs2_allT_3d = projsAll[region][cp][tt][1][n][nbt], projsAll[region][cp][tt][2][n][nbt]
                    
                    info1_3d, _ = f_subspace.plane_decodability(vecs1, projs1, projs1_allT_3d, trialInfoT, 'loc1', method = infoMethod)
                    info2_3d, _ = f_subspace.plane_decodability(vecs2, projs2, projs2_allT_3d, trialInfoT, 'loc2', method = infoMethod)
                    
                    decode_proj1T_3d[n,nbt,nc,:] = info1_3d #.mean(axis=-1).mean(axis=-1)
                    decode_proj2T_3d[n,nbt,nc,:] = info2_3d #.mean(axis=-1).mean(axis=-1)
                    
            
            for nbt in range(nBoots*nPerms):
                
                # labels
                Y = trialInfoT.loc[:,Y_columnsLabels].values
                ntrial = len(trialInfoT)
                
                
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
                    
                    trialInfoT_shuff = trialInfoT.copy()
                    trialInfoT_shuff[toDecode_labels1] = label1_shuff
                    trialInfoT_shuff[toDecode_labels2] = label2_shuff
                    
                    
                else:
                    trialInfoT_shuff = trialInfoT.sample(frac=1)


                for nc, cp in enumerate(checkpoints):

                    info1_3d_shuff, _ = f_subspace.plane_decodability(vecs1, projs1, projs1_allT_3d, trialInfoT_shuff, 'loc1', method = infoMethod, sequence=(0,1,3,2))
                    info2_3d_shuff, _ = f_subspace.plane_decodability(vecs2, projs2, projs2_allT_3d, trialInfoT_shuff, 'loc2', method = infoMethod, sequence=(0,1,3,2))
                    
                    decode_proj1T_3d_shuff[n,nbt,nc,:] = info1_3d_shuff #.mean(axis=-1).mean(axis=-1)
                    decode_proj2T_3d_shuff[n,nbt,nc,:] = info2_3d_shuff #.mean(axis=-1).mean(axis=-1)
                    
                    
        decode_proj1_3d[region][tt] = decode_proj1T_3d.mean(axis=1)
        decode_proj2_3d[region][tt] = decode_proj2T_3d.mean(axis=1) 
                    
        decode_proj1_shuff_all_3d[region][tt] = np.concatenate(decode_proj1T_3d_shuff, axis=0)
        decode_proj2_shuff_all_3d[region][tt] = np.concatenate(decode_proj2T_3d_shuff, axis=0)
# In[] cross-temp decodability plane projection

infoMethod = 'lda' #  'omega2' #

decode_proj1_3d, decode_proj2_3d = {},{}
#decode_proj1_2d, decode_proj2_2d = {},{}

decode_proj1_shuff_all_3d, decode_proj2_shuff_all_3d = {},{}
#decode_proj1_shuff_all_2d, decode_proj2_shuff_all_2d = {},{}

for region in ('dlpfc','fef'):
    
    decode_proj1_3d[region], decode_proj2_3d[region] = {}, {}
    #decode_proj1_2d[region], decode_proj2_2d[region] = {}, {}
    
    decode_proj1_shuff_all_3d[region], decode_proj2_shuff_all_3d[region] = {},{}
    #decode_proj1_shuff_all_2d[region], decode_proj2_shuff_all_2d[region] = {},{}
    
    for tt in ttypes:
        decode_proj1T_3d = np.zeros((nIters, nBoots, len(checkpoints), len(checkpoints))) # pca1st 3d coordinates
        decode_proj2T_3d = np.zeros((nIters, nBoots, len(checkpoints), len(checkpoints)))
        
        #decode_proj1T_2d = np.zeros((nIters, nBoots, len(checkpoints),2)) # pca1st 3d coordinates
        #decode_proj2T_2d = np.zeros((nIters, nBoots, len(checkpoints),2))
        
        for n in range(nIters):
            for nbt in range(nBoots):
                trialInfoT = trialInfos[region][tt][n][nbt]
                
                for nc1,cp1 in enumerate(checkpoints):
                    vecs1_1, vecs2_1 = vecs[region][cp1][tt][1][n][nbt], vecs[region][cp1][tt][2][n][nbt]
                    projs1_1, projs2_1 = projs[region][cp1][tt][1][n][nbt], projs[region][cp1][tt][2][n][nbt]
                    projs1_allT_3d_1, projs2_allT_3d_1 = projsAll[region][cp1][tt][1][n][nbt], projsAll[region][cp1][tt][2][n][nbt]
                    
                    for nc2,cp2 in enumerate(checkpoints):
                        vecs1_2, vecs2_2 = vecs[region][cp2][tt][1][n][nbt], vecs[region][cp2][tt][2][n][nbt]
                        projs1_2, projs2_2 = projs[region][cp2][tt][1][n][nbt], projs[region][cp2][tt][2][n][nbt]
                        projs1_allT_3d_2, projs2_allT_3d_2 = projsAll[region][cp2][tt][1][n][nbt], projsAll[region][cp2][tt][2][n][nbt]
                        
                        geoms1_1, geoms1_2 = (vecs1_1, projs1_1, projs1_allT_3d_1), (vecs1_2, projs1_2, projs1_allT_3d_2)
                        geoms2_1, geoms2_2 = (vecs2_1, projs2_1, projs2_allT_3d_1), (vecs2_2, projs2_2, projs2_allT_3d_2)
                        
                        #info1_3d, info1_2d = f_subspace.plane_decodability(vecs1, projs1, projs1_allT_3d, trialInfoT, 'loc1', method = infoMethod)
                        #info2_3d, info2_2d = f_subspace.plane_decodability(vecs2, projs2, projs2_allT_3d, trialInfoT, 'loc2', method = infoMethod)
                        info1_3d, _ = f_subspace.plane_decodability_LDA(geoms1_1, geoms1_2, trialInfoT, 'loc1')
                        info2_3d, _ = f_subspace.plane_decodability_LDA(geoms2_1, geoms2_2, trialInfoT, 'loc2')
                    
                        decode_proj1T_3d[n,nbt,nc1,nc2] = info1_3d.mean() #.mean(axis=-1).mean(axis=-1)
                        decode_proj2T_3d[n,nbt,nc1,nc2] = info2_3d.mean() #.mean(axis=-1).mean(axis=-1)
                        #decode_proj1T_2d[n,nbt,nc,:] = info1_2d
                        #decode_proj2T_2d[n,nbt,nc,:] = info2_2d
        
        decode_proj1_3d[region][tt] = decode_proj1T_3d.mean(axis=1)
        decode_proj2_3d[region][tt] = decode_proj2T_3d.mean(axis=1) 
        #decode_proj1_2d[region][tt] = decode_proj1T_2d.mean(axis=1)
        #decode_proj2_2d[region][tt] = decode_proj2T_2d.mean(axis=1)
        
        # shuff
        decode_proj1T_3d_shuff = np.ones((nIters, nBoots*nPerms, len(checkpoints),len(checkpoints)))
        decode_proj2T_3d_shuff = np.ones((nIters, nBoots*nPerms, len(checkpoints),len(checkpoints)))
        
        #decode_proj1T_2d_shuff = np.zeros((nIters, nBoots*nPerms, len(checkpoints),2))
        #decode_proj2T_2d_shuff = np.zeros((nIters, nBoots*nPerms, len(checkpoints),2))
        
        for n in range(nIters):
            for nbt in range(nBoots*nPerms):
                trialInfoT_shuff = trialInfos_shuff[region][tt][n][nbt]
                
                for nc1, cp1 in enumerate(checkpoints):
                    vecs1_shuff_1, vecs2_shuff_1 = vecs_shuff[region][cp1][tt][1][n][nbt], vecs_shuff[region][cp1][tt][2][n][nbt]
                    projs1_shuff_1, projs2_shuff_1 = projs_shuff[region][cp1][tt][1][n][nbt], projs_shuff[region][cp1][tt][2][n][nbt]
                    projs1_allT_3d_shuff_1, projs2_allT_3d_shuff_1 = projsAll_shuff[region][cp1][tt][1][n][nbt], projsAll_shuff[region][cp1][tt][2][n][nbt]
                    
                    for nc2, cp2 in enumerate(checkpoints):
                        
                        vecs1_shuff_2, vecs2_shuff_2 = vecs_shuff[region][cp2][tt][1][n][nbt], vecs_shuff[region][cp2][tt][2][n][nbt]
                        projs1_shuff_2, projs2_shuff_2 = projs_shuff[region][cp2][tt][1][n][nbt], projs_shuff[region][cp2][tt][2][n][nbt]
                        projs1_allT_3d_shuff_2, projs2_allT_3d_shuff_2 = projsAll_shuff[region][cp2][tt][1][n][nbt], projsAll_shuff[region][cp2][tt][2][n][nbt]
                        
                        geoms1_1_shuff, geoms1_2_shuff = (vecs1_shuff_1, projs1_shuff_1, projs1_allT_3d_shuff_1), (vecs1_shuff_2, projs1_shuff_2, projs1_allT_3d_shuff_2)
                        geoms2_1_shuff, geoms2_2_shuff = (vecs2_shuff_1, projs2_shuff_1, projs2_allT_3d_shuff_1), (vecs2_shuff_2, projs2_shuff_2, projs2_allT_3d_shuff_2)
                        
                        info1_3d_shuff, _ = f_subspace.plane_decodability_LDA(geoms1_1_shuff, geoms1_2_shuff, trialInfoT_shuff, 'loc1')
                        info2_3d_shuff, _ = f_subspace.plane_decodability_LDA(geoms2_1_shuff, geoms2_2_shuff, trialInfoT_shuff, 'loc2')
                        
                        decode_proj1T_3d_shuff[n,nbt,nc1,nc2] = info1_3d_shuff.mean() #.mean(axis=-1).mean(axis=-1)
                        decode_proj2T_3d_shuff[n,nbt,nc1,nc2] = info2_3d_shuff.mean() #.mean(axis=-1).mean(axis=-1)
                        #decode_proj1T_2d_shuff[n,nbt,nc,:] = info1_2d_shuff #.mean(axis=-1).mean(axis=-1)
                        #decode_proj2T_2d_shuff[n,nbt,nc,:] = info2_2d_shuff #.mean(axis=-1).mean(axis=-1)
                    
                    
        decode_proj1_shuff_all_3d[region][tt] = np.concatenate(decode_proj1T_3d_shuff, axis=0)
        decode_proj2_shuff_all_3d[region][tt] = np.concatenate(decode_proj2T_3d_shuff, axis=0)
        #decode_proj1_shuff_all_2d[region][tt] = np.concatenate(decode_proj1T_2d_shuff, axis=0)
        #decode_proj2_shuff_all_2d[region][tt] = np.concatenate(decode_proj2T_2d_shuff, axis=0)
        
        
    
    #plt.figure(figsize=(12, 8), dpi=100)
    plt.figure(figsize=(20, 18), dpi=100)    
    for tt in ttypes:
        
        #colorT = 'b' if tt == 1 else 'm'
        condT = 'retarget' if tt == 1 else 'distractor'
        
        
        performanceT1 = decode_proj1_3d[region][tt]
        performanceT2 = decode_proj2_3d[region][tt]
        performanceT1_shuff = decode_proj1_shuff_all_3d[region][tt]
        performanceT2_shuff = decode_proj2_shuff_all_3d[region][tt]
        
        # significance test
        pvalues1 = np.zeros((len(checkpoints),len(checkpoints)))
        pvalues2 = np.zeros((len(checkpoints),len(checkpoints)))
        for t in range(len(checkpoints)):
            for t_ in range(len(checkpoints)):
                pvalues1[t,t_] = f_stats.permutation_p(performanceT1.mean(axis = 0)[t,t_], performanceT1_shuff[:,t,t_], tail = 'greater')
                pvalues2[t,t_] = f_stats.permutation_p(performanceT2.mean(axis = 0)[t,t_], performanceT2_shuff[:,t,t_], tail = 'greater')
                #pvalues[t,t_] = stats.ttest_1samp(pfm[:,t,t_], 0.25, alternative = 'greater')[1]
                
        
        vmax = 0.6 if region == 'dlpfc' else 0.8
        
        plt.subplot(2,2,(tt-1)*2+1)
        ax = plt.gca()
        sns.heatmap(pd.DataFrame(performanceT1.mean(axis = 0), index=checkpoints,columns=checkpoints), cmap = 'magma', vmin = 0.25, vmax = vmax, ax = ax)#, vmax = 0.6, xticklabels = 100, yticklabels = 100, vmin = cbar_min, vmax = cbar_max
        
        #from scipy import ndimage
        smooth_scale = 10
        z = ndimage.zoom(pvalues1, smooth_scale)
        ax.contour(np.linspace(0, len(checkpoints), len(checkpoints) * smooth_scale),
                 np.linspace(0, len(checkpoints), len(checkpoints) * smooth_scale),
                  z, levels=([0.05]), colors='white', alpha = 1)
        
        ax.invert_yaxis()
        
        
        #ax.set_xticks([list(tbins).index(i) for i in [0, 300, 1300, 1600, 2600]])
        ax.set_xticklabels(checkpointsLabels, rotation=0, fontsize = 25)
        ax.set_xlabel('Test Timebin (ms)', labelpad = 10, fontsize = 35)
        #ax.set_yticks([list(tbins).index(i) for i in [0, 300, 1300, 1600, 2600]])
        ax.set_yticklabels(checkpointsLabels, fontsize = 25)
        ax.set_ylabel('Train Timebin (ms)', labelpad = 10, fontsize = 35)
        
        cbar = ax.collections[0].colorbar
        # here set the labelsize by 20
        cbar.ax.tick_params(labelsize=20)
        
        ax.set_title(f'{condT}, item1', pad = 10, fontsize = 25)
        
        
        plt.subplot(2,2,(tt-1)*2+2)
        ax = plt.gca()
        sns.heatmap(pd.DataFrame(performanceT2.mean(axis = 0), index=checkpoints,columns=checkpoints), cmap = 'magma', vmin = 0.25, vmax = vmax, ax = ax)#, vmax = 0.6, xticklabels = 100, yticklabels = 100, vmin = cbar_min, vmax = cbar_max
        
        #from scipy import ndimage
        smooth_scale = 10
        z = ndimage.zoom(pvalues2, smooth_scale)
        ax.contour(np.linspace(0, len(checkpoints), len(checkpoints) * smooth_scale),
                 np.linspace(0, len(checkpoints), len(checkpoints) * smooth_scale),
                  z, levels=([0.05]), colors='white', alpha = 1)
        
        ax.invert_yaxis()
        
        
        #ax.set_xticks([list(tbins).index(i) for i in [0, 300, 1300, 1600, 2600]])
        ax.set_xticklabels(checkpointsLabels, rotation=0, fontsize = 25)
        ax.set_xlabel('Test Timebin (ms)', labelpad = 10, fontsize = 35)
        #ax.set_yticks([list(tbins).index(i) for i in [0, 300, 1300, 1600, 2600]])
        ax.set_yticklabels(checkpointsLabels, fontsize = 25)
        ax.set_ylabel('Train Timebin (ms)', labelpad = 10, fontsize = 35)
        
        cbar = ax.collections[0].colorbar
        # here set the labelsize by 20
        cbar.ax.tick_params(labelsize=20)
        
        ax.set_title(f'{condT}, item2', pad = 10, fontsize = 25)
        
        
    plt.tight_layout()
    plt.subplots_adjust(top = 0.95)
    plt.suptitle(f'{region.upper()}', fontsize = 25, y=1)
    plt.show()  

# In[] cross-temp decodability plane projection for choice/nonchoice

infoMethod = 'lda' #  'omega2' #

decode_proj12_21, decode_proj21_12 = {},{} #choices, 12 = train retarget, test distractor; 21 = train distractor, test retarget
decode_proj11_22, decode_proj22_11 = {},{} #nonchioces

decode_proj12_21_shuff_all, decode_proj21_12_shuff_all = {},{}
decode_proj11_22_shuff_all, decode_proj22_11_shuff_all = {},{}

for region in ('dlpfc','fef'):
    
    decode_proj12_21[region], decode_proj21_12[region] = {}, {}
    decode_proj11_22[region], decode_proj22_11[region] = {}, {}
    
    decode_proj12_21_shuff_all[region], decode_proj21_12_shuff_all[region] = {},{}
    decode_proj11_22_shuff_all[region], decode_proj22_11_shuff_all[region] = {},{}
    
    
    decode_proj12_21T = np.zeros((nIters, nBoots, len(checkpoints), len(checkpoints))) # pca1st 3d coordinates
    decode_proj21_12T = np.zeros((nIters, nBoots, len(checkpoints), len(checkpoints)))
    
    decode_proj11_22T = np.zeros((nIters, nBoots, len(checkpoints), len(checkpoints))) # pca1st 3d coordinates
    decode_proj22_11T = np.zeros((nIters, nBoots, len(checkpoints), len(checkpoints)))
    
    for n in range(nIters):
        for nbt in range(nBoots):
            trialInfoT1 = trialInfos[region][1][n][nbt] # retarget trials
            trialInfoT2 = trialInfos[region][2][n][nbt] # distractor trials
            
            for nc1,cp1 in enumerate(checkpoints):
                vecs11_1, vecs12_1 = vecs[region][cp1][1][1][n][nbt], vecs[region][cp1][1][2][n][nbt]
                projs11_1, projs12_1 = projs[region][cp1][1][1][n][nbt], projs[region][cp1][1][2][n][nbt]
                projs11_allT_1, projs12_allT_1 = projsAll[region][cp1][1][1][n][nbt], projsAll[region][cp1][1][2][n][nbt]
                
                vecs21_1, vecs22_1 = vecs[region][cp1][2][1][n][nbt], vecs[region][cp1][2][2][n][nbt]
                projs21_1, projs22_1 = projs[region][cp1][2][1][n][nbt], projs[region][cp1][2][2][n][nbt]
                projs21_allT_1, projs22_allT_1 = projsAll[region][cp1][2][1][n][nbt], projsAll[region][cp1][2][2][n][nbt]
                
                for nc2,cp2 in enumerate(checkpoints):
                    vecs11_2, vecs12_2 = vecs[region][cp2][1][1][n][nbt], vecs[region][cp2][1][2][n][nbt]
                    projs11_2, projs12_2 = projs[region][cp2][1][1][n][nbt], projs[region][cp2][1][2][n][nbt]
                    projs11_allT_2, projs12_allT_2 = projsAll[region][cp2][1][1][n][nbt], projsAll[region][cp2][1][2][n][nbt]
                    
                    vecs21_2, vecs22_2 = vecs[region][cp2][2][1][n][nbt], vecs[region][cp2][2][2][n][nbt]
                    projs21_2, projs22_2 = projs[region][cp2][2][1][n][nbt], projs[region][cp2][2][2][n][nbt]
                    projs21_allT_2, projs22_allT_2 = projsAll[region][cp2][2][1][n][nbt], projsAll[region][cp2][2][2][n][nbt]
                    
                    geoms11_1, geoms11_2 = (vecs11_1, projs11_1, projs11_allT_1), (vecs11_2, projs11_2, projs11_allT_2)
                    geoms12_1, geoms12_2 = (vecs12_1, projs12_1, projs12_allT_1), (vecs12_2, projs12_2, projs12_allT_2)
                    
                    geoms21_1, geoms21_2 = (vecs21_1, projs21_1, projs21_allT_1), (vecs21_2, projs21_2, projs21_allT_2)
                    geoms22_1, geoms22_2 = (vecs22_1, projs22_1, projs22_allT_1), (vecs22_2, projs22_2, projs22_allT_2)
                    
                    info12_21, _ = f_subspace.plane_decodability_LDA_choice(geoms12_1, geoms21_2, trialInfoT1, trialInfoT2,'loc2','loc1')
                    info21_12, _ = f_subspace.plane_decodability_LDA_choice(geoms21_1, geoms12_2, trialInfoT2, trialInfoT1,'loc1','loc2')
                    
                    info11_22, _ = f_subspace.plane_decodability_LDA_choice(geoms11_1, geoms22_2, trialInfoT1, trialInfoT2,'loc1','loc2')
                    info22_11, _ = f_subspace.plane_decodability_LDA_choice(geoms22_1, geoms11_2, trialInfoT2, trialInfoT1,'loc2','loc1')
                    
                    decode_proj12_21T[n,nbt,nc1,nc2] = info12_21.mean() #.mean(axis=-1).mean(axis=-1)
                    decode_proj21_12T[n,nbt,nc1,nc2] = info21_12.mean() #.mean(axis=-1).mean(axis=-1)
                    decode_proj11_22T[n,nbt,nc1,nc2] = info11_22.mean() #.mean(axis=-1).mean(axis=-1)
                    decode_proj22_11T[n,nbt,nc1,nc2] = info22_11.mean()
        
    decode_proj12_21[region] = decode_proj12_21T.mean(axis=1)
    decode_proj21_12[region] = decode_proj21_12T.mean(axis=1) 
    decode_proj11_22[region] = decode_proj11_22T.mean(axis=1)
    decode_proj22_11[region] = decode_proj22_11T.mean(axis=1) 
        
    # shuff
    decode_proj12_21T_shuff = np.ones((nIters, nBoots*nPerms, len(checkpoints),len(checkpoints)))
    decode_proj21_12T_shuff = np.ones((nIters, nBoots*nPerms, len(checkpoints),len(checkpoints)))
    decode_proj11_22T_shuff = np.ones((nIters, nBoots*nPerms, len(checkpoints),len(checkpoints)))
    decode_proj22_11T_shuff = np.ones((nIters, nBoots*nPerms, len(checkpoints),len(checkpoints)))
        
    for n in range(nIters):
        for nbt in range(nBoots*nPerms):
            trialInfoT1_shuff = trialInfos_shuff[region][1][n][nbt]
            trialInfoT2_shuff = trialInfos_shuff[region][2][n][nbt]
            
            for nc1, cp1 in enumerate(checkpoints):
                vecs11_shuff_1, vecs12_shuff_1 = vecs_shuff[region][cp1][1][1][n][nbt], vecs_shuff[region][cp1][1][2][n][nbt]
                projs11_shuff_1, projs12_shuff_1 = projs_shuff[region][cp1][1][1][n][nbt], projs_shuff[region][cp1][1][2][n][nbt]
                projs11_allT_shuff_1, projs12_allT_shuff_1 = projsAll_shuff[region][cp1][1][1][n][nbt], projsAll_shuff[region][cp1][1][2][n][nbt]
                
                vecs21_shuff_1, vecs22_shuff_1 = vecs_shuff[region][cp1][2][1][n][nbt], vecs_shuff[region][cp1][2][2][n][nbt]
                projs21_shuff_1, projs22_shuff_1 = projs_shuff[region][cp1][2][1][n][nbt], projs_shuff[region][cp1][2][2][n][nbt]
                projs21_allT_shuff_1, projs22_allT_shuff_1 = projsAll_shuff[region][cp1][2][1][n][nbt], projsAll_shuff[region][cp1][2][2][n][nbt]
                
                for nc2, cp2 in enumerate(checkpoints):
                    
                    vecs11_shuff_2, vecs12_shuff_2 = vecs_shuff[region][cp2][1][1][n][nbt], vecs_shuff[region][cp2][1][2][n][nbt]
                    projs11_shuff_2, projs12_shuff_2 = projs_shuff[region][cp2][1][1][n][nbt], projs_shuff[region][cp2][1][2][n][nbt]
                    projs11_allT_shuff_2, projs12_allT_shuff_2 = projsAll_shuff[region][cp2][1][1][n][nbt], projsAll_shuff[region][cp2][1][2][n][nbt]
                    
                    vecs21_shuff_2, vecs22_shuff_2 = vecs_shuff[region][cp2][2][1][n][nbt], vecs_shuff[region][cp2][2][2][n][nbt]
                    projs21_shuff_2, projs22_shuff_2 = projs_shuff[region][cp2][2][1][n][nbt], projs_shuff[region][cp2][2][2][n][nbt]
                    projs21_allT_shuff_2, projs22_allT_shuff_2 = projsAll_shuff[region][cp2][2][1][n][nbt], projsAll_shuff[region][cp2][2][2][n][nbt]
                    
                    geoms11_1_shuff, geoms11_2_shuff = (vecs11_shuff_1, projs11_shuff_1, projs11_allT_shuff_1), (vecs11_shuff_2, projs11_shuff_2, projs11_allT_shuff_2)
                    geoms12_1_shuff, geoms12_2_shuff = (vecs12_shuff_1, projs12_shuff_1, projs12_allT_shuff_1), (vecs12_shuff_2, projs12_shuff_2, projs12_allT_shuff_2)
                    
                    geoms21_1_shuff, geoms21_2_shuff = (vecs21_shuff_1, projs21_shuff_1, projs21_allT_shuff_1), (vecs21_shuff_2, projs21_shuff_2, projs21_allT_shuff_2)
                    geoms22_1_shuff, geoms22_2_shuff = (vecs22_shuff_1, projs22_shuff_1, projs22_allT_shuff_1), (vecs22_shuff_2, projs22_shuff_2, projs22_allT_shuff_2)
                    
                    info12_21_shuff, _ = f_subspace.plane_decodability_LDA_choice(geoms12_1_shuff, geoms21_2_shuff, trialInfoT1_shuff, trialInfoT2_shuff,'loc2','loc1')
                    info21_12_shuff, _ = f_subspace.plane_decodability_LDA_choice(geoms21_1_shuff, geoms12_2_shuff, trialInfoT2_shuff, trialInfoT1_shuff,'loc1','loc2')
                    
                    info11_22_shuff, _ = f_subspace.plane_decodability_LDA_choice(geoms11_1_shuff, geoms22_2_shuff, trialInfoT1_shuff, trialInfoT2_shuff,'loc1','loc2')
                    info22_11_shuff, _ = f_subspace.plane_decodability_LDA_choice(geoms22_1_shuff, geoms11_2_shuff, trialInfoT2_shuff, trialInfoT1_shuff,'loc2','loc1')
                    
                    decode_proj12_21T_shuff[n,nbt,nc1,nc2] = info12_21_shuff.mean() #.mean(axis=-1).mean(axis=-1)
                    decode_proj21_12T_shuff[n,nbt,nc1,nc2] = info21_12_shuff.mean() #.mean(axis=-1).mean(axis=-1)
                    decode_proj11_22T_shuff[n,nbt,nc1,nc2] = info11_22_shuff.mean() #.mean(axis=-1).mean(axis=-1)
                    decode_proj22_11T_shuff[n,nbt,nc1,nc2] = info22_11_shuff.mean()
                
                
    decode_proj12_21_shuff_all[region] = np.concatenate(decode_proj12_21T_shuff, axis=0)
    decode_proj21_12_shuff_all[region] = np.concatenate(decode_proj21_12T_shuff, axis=0)
    decode_proj11_22_shuff_all[region] = np.concatenate(decode_proj11_22T_shuff, axis=0)
    decode_proj22_11_shuff_all[region] = np.concatenate(decode_proj22_11T_shuff, axis=0)
        
        
    
    
    performance12_21 = decode_proj12_21[region]
    performance21_12 = decode_proj21_12[region]
    performance11_22 = decode_proj11_22[region]
    performance22_11 = decode_proj22_11[region]
    
    performance12_21_shuff = decode_proj12_21_shuff_all[region]
    performance21_12_shuff = decode_proj21_12_shuff_all[region]
    performance11_22_shuff = decode_proj11_22_shuff_all[region]
    performance22_11_shuff = decode_proj22_11_shuff_all[region]
    
    # significance test
    pvalues12_21 = np.zeros((len(checkpoints),len(checkpoints)))
    pvalues21_12 = np.zeros((len(checkpoints),len(checkpoints)))
    pvalues11_22 = np.zeros((len(checkpoints),len(checkpoints)))
    pvalues22_11 = np.zeros((len(checkpoints),len(checkpoints)))
    
    for t in range(len(checkpoints)):
        for t_ in range(len(checkpoints)):
            pvalues12_21[t,t_] = f_stats.permutation_p(performance12_21.mean(axis = 0)[t,t_], performance12_21_shuff[:,t,t_], tail = 'greater')
            pvalues21_12[t,t_] = f_stats.permutation_p(performance21_12.mean(axis = 0)[t,t_], performance21_12_shuff[:,t,t_], tail = 'greater')
            pvalues11_22[t,t_] = f_stats.permutation_p(performance11_22.mean(axis = 0)[t,t_], performance11_22_shuff[:,t,t_], tail = 'greater')
            pvalues22_11[t,t_] = f_stats.permutation_p(performance22_11.mean(axis = 0)[t,t_], performance22_11_shuff[:,t,t_], tail = 'greater')
            
    
    vmax = 0.6 if region == 'dlpfc' else 0.8
    
    #plt.figure(figsize=(12, 8), dpi=100)
    plt.figure(figsize=(20, 18), dpi=100)    
    
    
    plt.subplot(2,2,1)
    ax = plt.gca()
    sns.heatmap(pd.DataFrame(performance12_21.mean(axis = 0), index=checkpoints,columns=checkpoints), cmap = 'magma', vmin = 0.25, vmax = vmax, ax = ax)#, vmax = 0.6, xticklabels = 100, yticklabels = 100, vmin = cbar_min, vmax = cbar_max
    
    #from scipy import ndimage
    smooth_scale = 5
    z = ndimage.zoom(pvalues12_21, smooth_scale)
    ax.contour(np.linspace(0, len(checkpoints), len(checkpoints) * smooth_scale),
             np.linspace(0, len(checkpoints), len(checkpoints) * smooth_scale),
              z, levels=([0.05]), colors='white', alpha = 1)
    
    ax.invert_yaxis()
    
    
    #ax.set_xticks([list(tbins).index(i) for i in [0, 300, 1300, 1600, 2600]])
    ax.set_xticklabels(checkpointsLabels, rotation=0, fontsize = 25)
    ax.set_xlabel('Test Timebin (ms)', labelpad = 10, fontsize = 35)
    #ax.set_yticks([list(tbins).index(i) for i in [0, 300, 1300, 1600, 2600]])
    ax.set_yticklabels(checkpointsLabels, fontsize = 25)
    ax.set_ylabel('Train Timebin (ms)', labelpad = 10, fontsize = 35)
    
    cbar = ax.collections[0].colorbar
    # here set the labelsize by 20
    cbar.ax.tick_params(labelsize=20)
    
    ax.set_title(f'Train Retarget; Test Distraction, Choice Item', pad = 10, fontsize = 25)
    
    
    plt.subplot(2,2,2)
    ax = plt.gca()
    sns.heatmap(pd.DataFrame(performance21_12.mean(axis = 0), index=checkpoints,columns=checkpoints), cmap = 'magma', vmin = 0.25, vmax = vmax, ax = ax)#, vmax = 0.6, xticklabels = 100, yticklabels = 100, vmin = cbar_min, vmax = cbar_max
    
    #from scipy import ndimage
    smooth_scale = 10
    z = ndimage.zoom(pvalues21_12, smooth_scale)
    ax.contour(np.linspace(0, len(checkpoints), len(checkpoints) * smooth_scale),
             np.linspace(0, len(checkpoints), len(checkpoints) * smooth_scale),
              z, levels=([0.05]), colors='white', alpha = 1)
    
    ax.invert_yaxis()
    
    
    #ax.set_xticks([list(tbins).index(i) for i in [0, 300, 1300, 1600, 2600]])
    ax.set_xticklabels(checkpointsLabels, rotation=0, fontsize = 25)
    ax.set_xlabel('Test Timebin (ms)', labelpad = 10, fontsize = 35)
    #ax.set_yticks([list(tbins).index(i) for i in [0, 300, 1300, 1600, 2600]])
    ax.set_yticklabels(checkpointsLabels, fontsize = 25)
    ax.set_ylabel('Train Timebin (ms)', labelpad = 10, fontsize = 35)
    
    cbar = ax.collections[0].colorbar
    # here set the labelsize by 20
    cbar.ax.tick_params(labelsize=20)
    
    ax.set_title(f'Train Distraction; Test Retarget, Choice Item', pad = 10, fontsize = 25)
    
    
    plt.subplot(2,2,3)
    ax = plt.gca()
    sns.heatmap(pd.DataFrame(performance11_22.mean(axis = 0), index=checkpoints,columns=checkpoints), cmap = 'magma', vmin = 0.25, vmax = vmax, ax = ax)#, vmax = 0.6, xticklabels = 100, yticklabels = 100, vmin = cbar_min, vmax = cbar_max
    
    #from scipy import ndimage
    smooth_scale = 5
    z = ndimage.zoom(pvalues11_22, smooth_scale)
    ax.contour(np.linspace(0, len(checkpoints), len(checkpoints) * smooth_scale),
             np.linspace(0, len(checkpoints), len(checkpoints) * smooth_scale),
              z, levels=([0.05]), colors='white', alpha = 1)
    
    ax.invert_yaxis()
    
    
    #ax.set_xticks([list(tbins).index(i) for i in [0, 300, 1300, 1600, 2600]])
    ax.set_xticklabels(checkpointsLabels, rotation=0, fontsize = 25)
    ax.set_xlabel('Test Timebin (ms)', labelpad = 10, fontsize = 35)
    #ax.set_yticks([list(tbins).index(i) for i in [0, 300, 1300, 1600, 2600]])
    ax.set_yticklabels(checkpointsLabels, fontsize = 25)
    ax.set_ylabel('Train Timebin (ms)', labelpad = 10, fontsize = 35)
    
    cbar = ax.collections[0].colorbar
    # here set the labelsize by 20
    cbar.ax.tick_params(labelsize=20)
    
    ax.set_title(f'Train Retarget; Test Distraction, Non-Choice Item', pad = 10, fontsize = 25)
    
    
    plt.subplot(2,2,4)
    ax = plt.gca()
    sns.heatmap(pd.DataFrame(performance22_11.mean(axis = 0), index=checkpoints,columns=checkpoints), cmap = 'magma', vmin = 0.25, vmax = vmax, ax = ax)#, vmax = 0.6, xticklabels = 100, yticklabels = 100, vmin = cbar_min, vmax = cbar_max
    
    #from scipy import ndimage
    smooth_scale = 10
    z = ndimage.zoom(pvalues22_11, smooth_scale)
    ax.contour(np.linspace(0, len(checkpoints), len(checkpoints) * smooth_scale),
             np.linspace(0, len(checkpoints), len(checkpoints) * smooth_scale),
              z, levels=([0.05]), colors='white', alpha = 1)
    
    ax.invert_yaxis()
    
    
    #ax.set_xticks([list(tbins).index(i) for i in [0, 300, 1300, 1600, 2600]])
    ax.set_xticklabels(checkpointsLabels, rotation=0, fontsize = 25)
    ax.set_xlabel('Test Timebin (ms)', labelpad = 10, fontsize = 35)
    #ax.set_yticks([list(tbins).index(i) for i in [0, 300, 1300, 1600, 2600]])
    ax.set_yticklabels(checkpointsLabels, fontsize = 25)
    ax.set_ylabel('Train Timebin (ms)', labelpad = 10, fontsize = 35)
    
    cbar = ax.collections[0].colorbar
    # here set the labelsize by 20
    cbar.ax.tick_params(labelsize=20)
    
    ax.set_title(f'Train Distraction; Test Retarget, Non-Choice Item', pad = 10, fontsize = 25)
        
    plt.tight_layout()
    plt.subplots_adjust(top = 0.95)
    plt.suptitle(f'{region.upper()}', fontsize = 25, y=1)
    plt.show()  
    
# In[] item 1 vs 2 within time (diagnal)
pdummy = True
cosTheta_12, cosTheta_12_shuff_all = {},{}
cosPsi_12, cosPsi_12_shuff_all = {},{}
sse_12, sse_12_shuff_all = {},{}
cosSimi_12, cosSimi_12_shuff_all = {},{}
ai_12, ai_12_shuff_all = {},{}

for region in ('dlpfc','fef'):
    
    cosTheta_12[region], cosPsi_12[region], sse_12[region], cosSimi_12[region], ai_12[region] = {},{},{},{},{}
    cosTheta_12_shuff_all[region], cosPsi_12_shuff_all[region], sse_12_shuff_all[region], cosSimi_12_shuff_all[region], ai_12_shuff_all[region] = {},{},{},{},{}
    
    for tt in ttypes:
        
        cosTheta_12T = np.zeros((nIters, nBoots, len(checkpoints),))
        cosPsi_12T = np.zeros((nIters, nBoots, len(checkpoints),))
        sse_12T = np.zeros((nIters, nBoots, len(checkpoints),))
        cosSimi_12T = np.zeros((nIters, nBoots, len(checkpoints),))
        ai_12T = np.zeros((nIters, nBoots, len(checkpoints),))
        
        for n in range(nIters):
            for nbt in range(nBoots):
                for nc,cp in enumerate(checkpoints):
                    cT_12, _, cP_12, _, s_12 = f_subspace.angle_alignment_coplanar(vecs[region][cp][tt][1][n][nbt], projs[region][cp][tt][1][n][nbt], vecs[region][cp][tt][2][n][nbt], projs[region][cp][tt][2][n][nbt])
                    cS_12 = f_subspace.config_correlation(vecs[region][cp][tt][1][n][nbt], projs[region][cp][tt][1][n][nbt], vecs[region][cp][tt][2][n][nbt], projs[region][cp][tt][2][n][nbt])
                    ai12 = f_subspace.get_simple_AI(projs[region][cp][tt][1][n][nbt], projs[region][cp][tt][2][n][nbt], max_dim=2)
                    
                    cosTheta_12T[n,nbt,nc], cosPsi_12T[n,nbt,nc], sse_12T[n,nbt,nc] = cT_12, cP_12, s_12
                    cosSimi_12T[n,nbt,nc], ai_12T[n,nbt,nc] = cS_12, ai12
                    
        cosTheta_12[region][tt], cosPsi_12[region][tt], sse_12[region][tt] = cosTheta_12T, cosPsi_12T, sse_12T
        cosSimi_12[region][tt], ai_12[region][tt] = cosSimi_12T, ai_12T
        
        ###
        pcosTheta_12 = np.ones((len(checkpoints)))
        pcosPsi_12 = np.ones((len(checkpoints)))
        psse_12 = np.ones((len(checkpoints)))
        pcosSimi_12 = np.ones((len(checkpoints)))
        pai_12 = np.ones((len(checkpoints)))
        
        if pdummy==False:
            ### shuff 
            cosTheta_12_shuffT = np.zeros((nIters, nBoots*nPerms, len(checkpoints),))
            cosPsi_12_shuffT = np.zeros((nIters, nBoots*nPerms, len(checkpoints),))
            sse_12_shuffT = np.zeros((nIters, nBoots*nPerms, len(checkpoints),))
            cosSimi_12_shuffT = np.zeros((nIters, nBoots*nPerms, len(checkpoints),))
            ai_12_shuffT = np.zeros((nIters, nBoots*nPerms, len(checkpoints),))
            
            for n in range(nIters):
                for nbt in range(nBoots*nPerms):
                    for nc, cp in enumerate(checkpoints):
                        cT_12, _, cP_12, _, s_12 = f_subspace.angle_alignment_coplanar(vecs_shuff[region][cp][tt][1][n][nbt], projs_shuff[region][cp][tt][1][n][nbt], vecs_shuff[region][cp][tt][2][n][nbt], projs_shuff[region][cp][tt][2][n][nbt])
                        cS_12 = f_subspace.config_correlation(vecs_shuff[region][cp][tt][1][n][nbt], projs_shuff[region][cp][tt][1][n][nbt], vecs_shuff[region][cp][tt][2][n][nbt], projs_shuff[region][cp][tt][2][n][nbt])
                        ai12 = f_subspace.get_simple_AI(projs_shuff[region][cp][tt][1][n][nbt], projs_shuff[region][cp][tt][2][n][nbt], max_dim=2)
                        
                        cosTheta_12_shuffT[n,nbt,nc], cosPsi_12_shuffT[n,nbt,nc], sse_12_shuffT[n,nbt,nc] = cT_12, cP_12, s_12
                        cosSimi_12_shuffT[n,nbt,nc], ai_12_shuffT[n,nbt,nc] = cS_12, ai12
                        
            
            
            cosTheta_12_shuff_all[region][tt] = np.concatenate([cosTheta_12_shuffT[:,nbt,:] for nbt in range(nBoots*nPerms)])
            cosPsi_12_shuff_all[region][tt] = np.concatenate([cosPsi_12_shuffT[:,nbt,:] for nbt in range(nBoots*nPerms)])
            sse_12_shuff_all[region][tt] = np.concatenate([sse_12_shuffT[:,nbt,:] for nbt in range(nBoots*nPerms)])
            cosSimi_12_shuff_all[region][tt] = np.concatenate([cosSimi_12_shuffT[:,nbt,:] for nbt in range(nBoots*nPerms)])
            ai_12_shuff_all[region][tt] = np.concatenate([ai_12_shuffT[:,nbt,:] for nbt in range(nBoots*nPerms)])
            
            
            for i in range(len(checkpoints)):
                
                # test distribution
                cT_12 = cosTheta_12[region][tt].mean(axis=1)[:,i].round(5)
                cP_12 = cosPsi_12[region][tt].mean(axis=1)[:,i].round(5)
                s_12 = sse_12[region][tt].mean(axis=1)[:,i].round(5)
                cS_12 = cosSimi_12[region][tt].mean(axis=1)[:,i].round(5)
                ai12 = ai_12[region][tt].mean(axis=1)[:,i].round(5)
                
                # shuff distribution
                cT_12_shuff = cosTheta_12_shuff_all[region][tt][:,i].round(5)
                cP_12_shuff = cosPsi_12_shuff_all[region][tt][:,i].round(5)
                s_12_shuff = sse_12_shuff_all[region][tt][:,i].round(5)
                cS_12_shuff = cosSimi_12_shuff_all[region][tt][:,i].round(5)
                ai12_shuff = ai_12_shuff_all[region][tt][:,i].round(5)
                
                # compare distributions and calculate p values
                pcosTheta_12[i] = stats.kstest(cT_12, cT_12_shuff)[-1]
                pcosPsi_12[i] = stats.kstest(cP_12, cP_12_shuff)[-1]
                psse_12[i] = stats.kstest(s_12, s_12_shuff)[-1]
                pcosSimi_12[i] = stats.kstest(cS_12, cS_12_shuff)[-1]
                pai_12[i] = stats.kstest(ai12, ai12_shuff)[-1]
            
        
        angleCheckPoints = np.linspace(0,np.pi,13).round(5)
        
        ### cosTheta
        plt.figure(figsize=(8, 3), dpi=100)
        #plt.subplot(1,5,1)
        plt.subplot(1,2,1)
        ax = plt.gca()
        ax.errorbar(np.arange(0, len(checkpoints), 1), cosTheta_12[region][tt].mean(axis=1).mean(axis=0), yerr = cosTheta_12[region][tt].mean(axis=1).std(axis=0), marker = 'o', capsize=4)
        ax.plot(np.arange(0, len(checkpoints), 1), pcosTheta_12, alpha = 0.3, linestyle = '-')
        
        trans = ax.get_xaxis_transform()
        for nc, cp in enumerate(checkpoints):
            if 0.05 < pcosTheta_12[nc] <= 0.1:
                ax.annotate('+', xy=(nc, 0.05), xycoords = trans, annotation_clip = False, ha = 'center', va = 'center')
            elif 0.01 < pcosTheta_12[nc] <= 0.05:
                ax.annotate('*', xy=(nc, 0.05), xycoords = trans, annotation_clip = False, ha = 'center', va = 'center')
            elif pcosTheta_12[nc] <= 0.01:
                ax.annotate('**', xy=(nc, 0.05), xycoords = trans, annotation_clip = False, ha = 'center', va = 'center')
                
        ax.set_title('cosTheta', pad = 10)
        ax.set_xticks([n for n in range(len(checkpoints))])
        ax.set_xticklabels(checkpointsLabels, fontsize = 6)
        ax.set_ylim((-1,1))
        ax.set_yticks(np.flip(np.cos(angleCheckPoints).round(5)), labels = np.flip(np.degrees(angleCheckPoints).round(1).astype(int)))
        ax.set_ylabel('cos(x)',fontsize=15,rotation = 90)
        
        
        ### cosPsi
        #plt.subplot(1,5,2)
        plt.subplot(1,2,2)
        ax = plt.gca()
        ax.errorbar(np.arange(0, len(checkpoints), 1), cosPsi_12[region][tt].mean(axis=1).mean(axis=0), yerr = cosPsi_12[region][tt].mean(axis=1).std(axis=0), marker = 'o', capsize=4)
        #ax.errorbar(np.arange(0, len(checkpoints), 1), np.abs(cosPsi_12[region][tt]).mean(axis=1).mean(axis=0), yerr = np.abs(cosPsi_12[region][tt]).mean(axis=1).std(axis=0), marker = 'o')
        
        ax.plot(np.arange(0, len(checkpoints), 1), pcosPsi_12, alpha = 0.3, linestyle = '-')
        
        trans = ax.get_xaxis_transform()
        for nc, cp in enumerate(checkpoints):
            if 0.05 < pcosPsi_12[nc] <= 0.1:
                ax.annotate('+', xy=(nc, 0.05), xycoords = trans, annotation_clip = False, ha = 'center', va = 'center')
            elif 0.01 < pcosPsi_12[nc] <= 0.05:
                ax.annotate('*', xy=(nc, 0.05), xycoords = trans, annotation_clip = False, ha = 'center', va = 'center')
            elif pcosPsi_12[nc] <= 0.01:
                ax.annotate('**', xy=(nc, 0.05), xycoords = trans, annotation_clip = False, ha = 'center', va = 'center')
                
        ax.set_title('cosPsi', pad = 10)
        #ax.set_title('abs(cosPsi)', pad = 10)
        ax.set_xticks([n for n in range(len(checkpoints))])
        ax.set_xticklabels(checkpointsLabels, fontsize = 6)
        ax.set_ylim((-1,1))
        ax.set_yticks(np.flip(np.cos(angleCheckPoints).round(5)), labels = np.flip(np.degrees(angleCheckPoints).round(1).astype(int)))
        ax.set_ylabel('cos(x)',fontsize=15,rotation = 90)
        
        
        plt.subplots_adjust(top = 0.8)
        plt.suptitle(f'{region}, {tt}, item 1 vs 2, within time', fontsize = 15, y=1)
        plt.tight_layout()
        plt.show()    
        
        
        ### sse
        #plt.subplot(1,5,3)
        #ax = plt.gca()
        #ax.errorbar(np.arange(0, len(checkpoints), 1), sse_12[region][tt].mean(axis=1).mean(axis=0), yerr = sse_12[region][tt].mean(axis=1).std(axis=0), marker = 'o')
        #ax.plot(np.arange(0, len(checkpoints), 1), psse_12, alpha = 0.3, linestyle = '-')
        
        #trans = ax.get_xaxis_transform()
        #for nc, cp in enumerate(checkpoints):
        #    if 0.05 < psse_12[nc] <= 0.1:
        #        ax.annotate('+', xy=(nc, 0.05), xycoords = trans, annotation_clip = False, ha = 'center', va = 'center')
        #    elif 0.01 < psse_12[nc] <= 0.05:
        #        ax.annotate('*', xy=(nc, 0.05), xycoords = trans, annotation_clip = False, ha = 'center', va = 'center')
        #    elif psse_12[nc] <= 0.01:
        #        ax.annotate('**', xy=(nc, 0.05), xycoords = trans, annotation_clip = False, ha = 'center', va = 'center')
                
        #ax.set_title('sse', pad = 10)
        #ax.set_xticks([n for n in range(len(checkpoints))])
        #ax.set_xticklabels(checkpoints, fontsize = 6)
        #ax.set_ylim((-1,1))
        
        
        ### cosSimilarity
        #plt.subplot(1,5,4)
        #ax = plt.gca()
        #ax.errorbar(np.arange(0, len(checkpoints), 1), cosSimi_12[region][tt].mean(axis=1).mean(axis=0), yerr = cosSimi_12[region][tt].mean(axis=1).std(axis=0), marker = 'o')
        #ax.errorbar(np.arange(0, len(checkpoints), 1), np.abs(cosPsi_12[region][tt]).mean(axis=1).mean(axis=0), yerr = np.abs(cosPsi_12[region][tt]).mean(axis=1).std(axis=0), marker = 'o')
        
        #ax.plot(np.arange(0, len(checkpoints), 1), pcosSimi_12, alpha = 0.3, linestyle = '-')
        
        #trans = ax.get_xaxis_transform()
        #for nc, cp in enumerate(checkpoints):
        #    if 0.05 < pcosSimi_12[nc] <= 0.1:
        #        ax.annotate('+', xy=(nc, 0.05), xycoords = trans, annotation_clip = False, ha = 'center', va = 'center')
        #    elif 0.01 < pcosSimi_12[nc] <= 0.05:
        #        ax.annotate('*', xy=(nc, 0.05), xycoords = trans, annotation_clip = False, ha = 'center', va = 'center')
        #    elif pcosSimi_12[nc] <= 0.01:
        #        ax.annotate('**', xy=(nc, 0.05), xycoords = trans, annotation_clip = False, ha = 'center', va = 'center')
                
        #ax.set_title('correlation', pad = 10)
        #ax.set_title('abs(cosPsi)', pad = 10)
        #ax.set_xticks([n for n in range(len(checkpoints))])
        #ax.set_xticklabels(checkpoints, fontsize = 6)
        #ax.set_ylim((0,1))
        #ax.set_ylim((0,1))
        
        
        ### Alignment index
        #plt.subplot(1,5,5)
        #ax = plt.gca()
        #ax.errorbar(np.arange(0, len(checkpoints), 1), ai_12[region][tt].mean(axis=1).mean(axis=0), yerr = ai_12[region][tt].mean(axis=1).std(axis=0), marker = 'o')
        #ax.errorbar(np.arange(0, len(checkpoints), 1), np.abs(cosPsi_12[region][tt]).mean(axis=1).mean(axis=0), yerr = np.abs(cosPsi_12[region][tt]).mean(axis=1).std(axis=0), marker = 'o')
        
        #ax.plot(np.arange(0, len(checkpoints), 1), pai_12, alpha = 0.3, linestyle = '-')
        
        #trans = ax.get_xaxis_transform()
        #for nc, cp in enumerate(checkpoints):
        #    if 0.05 < pai_12[nc] <= 0.1:
        #        ax.annotate('+', xy=(nc, 0.05), xycoords = trans, annotation_clip = False, ha = 'center', va = 'center')
        #    elif 0.01 < pai_12[nc] <= 0.05:
        #        ax.annotate('*', xy=(nc, 0.05), xycoords = trans, annotation_clip = False, ha = 'center', va = 'center')
        #    elif pai_12[nc] <= 0.01:
        #        ax.annotate('**', xy=(nc, 0.05), xycoords = trans, annotation_clip = False, ha = 'center', va = 'center')
                
        #ax.set_title('Alignment index', pad = 10)
        #ax.set_title('abs(cosPsi)', pad = 10)
        #ax.set_xticks([n for n in range(len(checkpoints))])
        #ax.set_xticklabels(checkpoints, fontsize = 6)
        #ax.set_ylim((0,1))
        #ax.set_ylim((0,1))
        
        
        #plt.subplots_adjust(top = 0.8)
        #plt.suptitle(f'{region}, {tt}, item 1 vs 2, within time', fontsize = 15, y=1)
        #plt.tight_layout()
        #plt.show()    
        




        
# In[] correlation between cosTheta_12 and decodability (omega2PEV)

validCheckpoints = {'dlpfc': (1850,), 'fef': (1450, )} # 

for region in ('dlpfc','fef'):
    
    cT12 = []
    cP12 = []
    info1 = []
    info2 = []
    
    ttypesT = (1,) #(1,2) # 
    
    for tt in ttypesT:  #
        
        for n in range(nIters):
            for nbt in range(nBoots):
                for nc, cp in enumerate(checkpoints):
                    if cp in validCheckpoints[region]:
                        cT12 += [cosTheta_12[region][tt][n,nbt,nc,nc]]
                        cP12 += [cosPsi_12[region][tt][n,nbt,nc,nc]]
                        info1 += [decode_proj1_3d[region][tt][n,nc,:].mean()] # if use omega2
                        info2 += [decode_proj2_3d[region][tt][n,nc,:].mean()]
                        #info1 += [decodability_proj[region][cp][tt][1][n][nbt]] # if use lda performance
                        #info2 += [decodability_proj[region][cp][tt][2][n][nbt]]
    
    cT12 = np.array(cT12)
    cP12 = np.array(cP12)
    info1 = np.array(info1)
    info2 = np.array(info2)
    
    infoDiff = np.abs(info1 - info2)/(info1 + info2)
    
    zT = np.polyfit(infoDiff, cT12, 1)
    pT = np.poly1d(zT)
    
    zP = np.polyfit(infoDiff, cP12, 1)
    pP = np.poly1d(zP)
    
    
    method = 'spearman' #'pearson' #
    
    if method == 'spearman':
        corr, p_corr = stats.spearmanr(np.abs(cT12), infoDiff)
    elif method == 'pearson':
        corr, p_corr = stats.pearsonr(np.abs(cT12), infoDiff)
    
    plt.figure(figsize=(5, 3), dpi=100)
    
    plt.subplot(1,1,1)
    ax = plt.gca()
    
    ax.scatter(infoDiff, cT12, marker = '.')
    ax.plot(infoDiff,pT(infoDiff), color = 'r', linestyle='--', alpha = 0.6)
    
    ax.set_title(f'{method} r = {corr:.3f}, p = {p_corr:.3f}', pad = 10)
    ax.set_xlabel('infoDiff')
    ax.set_ylabel('abs(cosTheta)')
    #ax.set_ylim((-1,1))
    #ax.legend()
    
    plt.subplots_adjust(top = 0.8)
    plt.suptitle(f'{region}, types: {ttypesT}, timepoints: {validCheckpoints[region]}', fontsize = 10, y=1)
    plt.show()
    
    
    method = 'pearson' #'spearman' #
    
    if method == 'spearman':
        corr, p_corr = stats.spearmanr(np.abs(cP12), infoDiff)
    elif method == 'pearson':
        corr, p_corr = stats.pearsonr(np.abs(cP12), infoDiff)
        
    plt.figure(figsize=(5, 3), dpi=100)
    
    plt.subplot(1,1,1)
    ax = plt.gca()
    
    ax.scatter(infoDiff, cP12, marker = '.')
    ax.plot(infoDiff,pP(infoDiff), color = 'r', linestyle='--', alpha = 0.6)
    
    ax.set_title(f'{method} r = {corr:.3f}, p = {p_corr:.3f}', pad = 10)
    ax.set_xlabel('infoDiff')
    ax.set_ylabel('abs(cosPsi)')
    #ax.set_ylim((-1,1))
    #ax.legend()
    
    plt.subplots_adjust(top = 0.8)
    plt.suptitle(f'{region}, types: {ttypesT}, timepoints: {validCheckpoints[region]}', fontsize = 10, y=1)
    plt.show()






    
# In[] condition mean trajectory 3d

mainCheckpoints = (1300, 2600, 3000)

cseq = mpl.color_sequences['tab20c']
cmap1 = mpl.colormaps.get_cmap('Greens_r')
cmap2 = mpl.colormaps.get_cmap('Reds_r')
cmap3 = mpl.colormaps.get_cmap('Blues_r')
c1s = [cmap1(i) for i in np.linspace(0,1,3)]
c2s = [cmap2(i) for i in np.linspace(0,1,3)]
c3s = [cmap3(i) for i in np.linspace(0,1,3)]

shapes = ('o','s','*','^')

for region in ('dlpfc','fef'):
    
    # just apply to retarget trials
    for tt in ttypes: #(1,)
        # example Iterations
        for n in (0,10,20):#,30,40,50
            projs1T, projs2T = np.zeros((len(checkpoints), len(locs), 3)), np.zeros((len(checkpoints), len(locs), 3))
            vecs1T, vecs2T = np.zeros((len(checkpoints), 2, 3)), np.zeros((len(checkpoints), 2, 3))
            for ncp, cp in enumerate(checkpoints):
                projs1T[ncp,:,:] = np.concatenate(projs[region][cp][tt][1][n])
                projs2T[ncp,:,:] = np.concatenate(projs[region][cp][tt][2][n]) if tt == 1 else np.concatenate(projs[region][cp][tt][1][n])
                vecs1T[ncp,:,:] = np.concatenate(vecs[region][cp][tt][1][n])
                vecs2T[ncp,:,:] = np.concatenate(vecs[region][cp][tt][2][n]) if tt == 1 else np.concatenate(vecs[region][cp][tt][1][n])
            
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
            
            plt.suptitle(f'Traj {region}, example {n}, tt = {tt}')
            plt.tight_layout()
            plt.show() 
            
            
# In[] correlation(cosSimilarity), alignment index. Compare within type, between time points, between locations
pdummy = True
cosSimi_11, cosSimi_22, cosSimi_12 = {},{},{}

for region in ('dlpfc','fef'):
    
    cosSimi_11[region], cosSimi_22[region], cosSimi_12[region] = {},{},{}
    
    for tt in ttypes:
                
        cS_11T = np.zeros((nIters, nBoots, len(checkpoints), len(checkpoints)))
        cS_22T = np.zeros((nIters, nBoots, len(checkpoints), len(checkpoints)))
        cS_12T = np.zeros((nIters, nBoots, len(checkpoints), len(checkpoints)))
        
        for n in range(nIters):
            for nbt in range(nBoots):
                for nc, cp in enumerate(checkpoints):
                    for nc_, cp_ in enumerate(checkpoints):
                        
                        cS11 = f_subspace.config_correlation(vecs[region][cp][tt][1][n][nbt], projs[region][cp][tt][1][n][nbt], vecs[region][cp_][tt][1][n][nbt], projs[region][cp_][tt][1][n][nbt])
                        cS22 = f_subspace.config_correlation(vecs[region][cp][tt][2][n][nbt], projs[region][cp][tt][2][n][nbt], vecs[region][cp_][tt][2][n][nbt], projs[region][cp_][tt][2][n][nbt])
                        cS12 = f_subspace.config_correlation(vecs[region][cp][tt][1][n][nbt], projs[region][cp][tt][1][n][nbt], vecs[region][cp_][tt][2][n][nbt], projs[region][cp_][tt][2][n][nbt])
                        
                        cS_11T[n,nbt,nc,nc_], cS_22T[n,nbt,nc,nc_], cS_12T[n,nbt,nc,nc_] = cS11, cS22, cS12
                        
        
        cosSimi_11[region][tt] = cS_11T
        cosSimi_22[region][tt] = cS_22T
        cosSimi_12[region][tt] = cS_12T
        
        # shuff
    
        
        pcS_11T, pcS_22T, pcS_12T = np.ones((len(checkpoints), len(checkpoints))), np.ones((len(checkpoints), len(checkpoints))), np.ones((len(checkpoints), len(checkpoints)))
        
        if pdummy == False:
            
            cS_11_shuffT = np.zeros((nIters, nBoots*nPerms, len(checkpoints), len(checkpoints)))
            cS_22_shuffT = np.zeros((nIters, nBoots*nPerms, len(checkpoints), len(checkpoints)))
            cS_12_shuffT = np.zeros((nIters, nBoots*nPerms, len(checkpoints), len(checkpoints)))
            
            for n in range(nIters):
                for nbt in range(nBoots*nPerms):
                    for nc, cp in enumerate(checkpoints):
                        for nc_, cp_ in enumerate(checkpoints):
                            
                            
                            cS11 = f_subspace.config_correlation(vecs_shuff[region][cp][tt][1][n][nbt], projs_shuff[region][cp][tt][1][n][nbt], vecs_shuff[region][cp_][tt][1][n][nbt], projs_shuff[region][cp_][tt][1][n][nbt])
                            cS22 = f_subspace.config_correlation(vecs_shuff[region][cp][tt][2][n][nbt], projs_shuff[region][cp][tt][2][n][nbt], vecs_shuff[region][cp_][tt][2][n][nbt], projs_shuff[region][cp_][tt][2][n][nbt])
                            cS12 = f_subspace.config_correlation(vecs_shuff[region][cp][tt][1][n][nbt], projs_shuff[region][cp][tt][1][n][nbt], vecs_shuff[region][cp_][tt][2][n][nbt], projs_shuff[region][cp_][tt][2][n][nbt])
                                                    
                            cS_11_shuffT[n,nbt,nc,nc_], cS_22_shuffT[n,nbt,nc,nc_], cS_12_shuffT[n,nbt,nc,nc_] = cS11, cS22, cS12
                            
            
            cS_11_shuff_all = np.concatenate([cS_11_shuffT[:,nbt,:,:] for nbt in range(nBoots*nPerms)])
            cS_22_shuff_all = np.concatenate([cS_22_shuffT[:,nbt,:,:] for nbt in range(nBoots*nPerms)])
            cS_12_shuff_all = np.concatenate([cS_12_shuffT[:,nbt,:,:] for nbt in range(nBoots*nPerms)])
            
            for i in range(len(checkpoints)):
                for j in range(len(checkpoints)):
                    
                    cS11, cS22, cS12 = cS_11T.mean(axis=1)[:,i,j].round(5), cS_22T.mean(axis=1)[:,i,j].round(5), cS_12T.mean(axis=1)[:,i,j].round(5)
                    
                    # shuff distribution
                    
                    cS11_shuff, cS22_shuff, cS12_shuff = cS_11_shuff_all[:,i,j].round(5), cS_22_shuff_all[:,i,j].round(5), cS_12_shuff_all[:,i,j].round(5)
                    
                                   
                    # kstest pvalue
                    
                    pcS_11T[i,j] = stats.kstest(cS11, cS11_shuff)[-1] # #stats.uniform.cdf
                    pcS_22T[i,j] = stats.kstest(cS22, cS22_shuff)[-1] # #stats.uniform.cdf
                    pcS_12T[i,j] = stats.kstest(cS12, cS12_shuff)[-1] # #stats.uniform.cdf
                    
            
        
        ### cosSimilarity
        plt.figure(figsize=(16, 6), dpi=100)
        plt.subplot(1,3,1)
        ax = plt.gca()
        #im = ax.imshow(cS_11T.mean(axis=1).mean(axis=0), cmap='OrRd', aspect='auto', vmin=0, vmax=1) #, norm=mcolors.SymLogNorm(linthresh=0.1, linscale=0.1, vmin=0, vmax=1.0, base=10)
        im = ax.imshow(f_plotting.mask_triangle(cS_11T.mean(axis=1).mean(axis=0), ul='u', diag=1), cmap='OrRd', aspect='auto', vmin=0, vmax=1) #, vmax=1, norm=mcolors.SymLogNorm(linthresh=0.1, linscale=0.1, vmin=0, vmax=1.0, base=10)
        #im = ax.imshow(np.median(cosTheta_11T.mean(axis=1), axis=0), cmap='jet', aspect='auto', vmin=0, vmax=1)
        for i in range(len(checkpoints)):
            for j in range(len(checkpoints)):
                if 0.05 < pcS_11T[i,j] <= 0.1:
                    text = ax.text(j, i, '+', ha="center", va="center", fontsize=6)
                elif 0.01 < pcS_11T[i,j] <= 0.05:
                    text = ax.text(j, i, '*', ha="center", va="center", fontsize=6)
                elif pcS_11T[i,j] <= 0.01:
                    text = ax.text(j, i, '**', ha="center", va="center", fontsize=6)
                    
        ax.set_title('11', pad = 15)
        ax.set_xticks([n for n in range(len(checkpoints))])
        ax.set_xticklabels(checkpoints, fontsize = 10)
        ax.set_yticks([n for n in range(len(checkpoints))])
        ax.set_yticklabels(checkpoints, fontsize = 10)
        ax.set_xlabel('Item 1', fontsize = 15)
        ax.set_ylabel('Item 1', fontsize = 15)
        ax.set_frame_on(False)
        
        
        plt.subplot(1,3,2)
        ax = plt.gca()
        #im = ax.imshow(cS_22T.mean(axis=1).mean(axis=0), cmap='OrRd', aspect='auto', vmin=0, vmax=1) #, norm=mcolors.SymLogNorm(linthresh=0.1, linscale=0.1, vmin=0, vmax=1.0, base=10)
        im = ax.imshow(f_plotting.mask_triangle(cS_22T.mean(axis=1).mean(axis=0), ul='u', diag=1), cmap='OrRd', aspect='auto', vmin=0, vmax=1) #, vmax=1, norm=mcolors.SymLogNorm(linthresh=0.1, linscale=0.1, vmin=0, vmax=1.0, base=10)
        #im = ax.imshow(np.median(cosTheta_22T.mean(axis=1), axis=0), cmap='jet', aspect='auto', vmin=0, vmax=1)
        for i in range(len(checkpoints)):
            for j in range(len(checkpoints)):
                if 0.05 < pcS_22T[i,j] <= 0.1:
                    text = ax.text(j, i, '+', ha="center", va="center", fontsize=6)
                elif 0.01 < pcS_22T[i,j] <= 0.05:
                    text = ax.text(j, i, '*', ha="center", va="center", fontsize=6)
                elif pcS_22T[i,j] <= 0.01:
                    text = ax.text(j, i, '**', ha="center", va="center", fontsize=6)
                    
        ax.set_title('22', pad = 15)
        ax.set_xticks([n for n in range(len(checkpoints))])
        ax.set_xticklabels(checkpoints, fontsize = 10)
        ax.set_yticks([n for n in range(len(checkpoints))])
        ax.set_yticklabels(checkpoints, fontsize = 10)
        ax.set_xlabel('Item 2', fontsize = 15)
        ax.set_ylabel('Item 2', fontsize = 15)
        ax.set_frame_on(False)
        
        
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
        ax.set_title('12', pad = 15)
        ax.set_xticks([n for n in range(len(checkpoints))])
        ax.set_xticklabels(checkpoints, fontsize = 10)
        ax.set_yticks([n for n in range(len(checkpoints))])
        ax.set_yticklabels(checkpoints, fontsize = 10)
        ax.set_xlabel('Item 2', fontsize = 15)
        ax.set_ylabel('Item 1', fontsize = 15)
        ax.set_frame_on(False)
        
        plt.colorbar(im, ax=ax)
        
        plt.subplots_adjust(top = 0.8)
        plt.suptitle(f'correlation, {region}, ttype={tt}', fontsize = 20, y=1)
        plt.tight_layout()
        plt.show()
        
        
        
        
# In[] alignment index. Compare within type, between time points, between locations
pdummy = True
ai_11, ai_22, ai_12 = {},{},{}

for region in ('dlpfc','fef'):
    
    ai_11[region], ai_22[region], ai_12[region] = {},{},{}
    
    for tt in ttypes:
        
        ai_11T = np.zeros((nIters, nBoots, len(checkpoints), len(checkpoints)))
        ai_22T = np.zeros((nIters, nBoots, len(checkpoints), len(checkpoints)))
        ai_12T = np.zeros((nIters, nBoots, len(checkpoints), len(checkpoints)))
        
        for n in range(nIters):
            for nbt in range(nBoots):
                for nc, cp in enumerate(checkpoints):
                    for nc_, cp_ in enumerate(checkpoints):
                        
                        #ai11 = f_subspace.get_simple_AI(Xs_mean[region][cp][tt][1][n][nbt], Xs_mean[region][cp_][tt][1][n][nbt], max_dim=2)
                        #ai22 = f_subspace.get_simple_AI(Xs_mean[region][cp][tt][2][n][nbt], Xs_mean[region][cp_][tt][2][n][nbt], max_dim=2)
                        #ai12 = f_subspace.get_simple_AI(Xs_mean[region][cp][tt][1][n][nbt], Xs_mean[region][cp_][tt][2][n][nbt], max_dim=2)
                        
                        ai11 = f_subspace.get_simple_AI(projs[region][cp][tt][1][n][nbt], projs[region][cp_][tt][1][n][nbt], max_dim=2)
                        ai22 = f_subspace.get_simple_AI(projs[region][cp][tt][2][n][nbt], projs[region][cp_][tt][2][n][nbt], max_dim=2)
                        ai12 = f_subspace.get_simple_AI(projs[region][cp][tt][1][n][nbt], projs[region][cp_][tt][2][n][nbt], max_dim=2)
                        
                        #ai11 = f_subspace.get_simple_AI_2d(vecs[region][cp][tt][1][n][nbt], projs[region][cp][tt][1][n][nbt], vecs[region][cp_][tt][1][n][nbt], projs[region][cp_][tt][1][n][nbt])
                        #ai22 = f_subspace.get_simple_AI_2d(vecs[region][cp][tt][2][n][nbt], projs[region][cp][tt][2][n][nbt], vecs[region][cp_][tt][2][n][nbt], projs[region][cp_][tt][2][n][nbt])
                        #ai12 = f_subspace.get_simple_AI_2d(vecs[region][cp][tt][1][n][nbt], projs[region][cp][tt][1][n][nbt], vecs[region][cp_][tt][2][n][nbt], projs[region][cp_][tt][2][n][nbt])
                        
                        ai_11T[n,nbt,nc,nc_], ai_22T[n,nbt,nc,nc_], ai_12T[n,nbt,nc,nc_] = ai11, ai22, ai12
                        
        
        ai_11[region][tt] = ai_11T
        ai_22[region][tt] = ai_22T
        ai_12[region][tt] = ai_12T
        
        
        # shuff
        
        pai_11T, pai_22T, pai_12T = np.ones((len(checkpoints), len(checkpoints))), np.ones((len(checkpoints), len(checkpoints))), np.ones((len(checkpoints), len(checkpoints)))
        
        if pdummy== False:
            ai_11_shuffT = np.zeros((nIters, nBoots*nPerms, len(checkpoints), len(checkpoints)))
            ai_22_shuffT = np.zeros((nIters, nBoots*nPerms, len(checkpoints), len(checkpoints)))
            ai_12_shuffT = np.zeros((nIters, nBoots*nPerms, len(checkpoints), len(checkpoints)))
            
            for n in range(nIters):
                for nbt in range(nBoots*nPerms):
                    for nc, cp in enumerate(checkpoints):
                        for nc_, cp_ in enumerate(checkpoints):
                                                        
                            #ai11 = f_subspace.get_simple_AI(Xs_mean_shuff[region][cp][tt][1][n][nbt], Xs_mean_shuff[region][cp_][tt][1][n][nbt])
                            #ai22 = f_subspace.get_simple_AI(Xs_mean_shuff[region][cp][tt][2][n][nbt], Xs_mean_shuff[region][cp_][tt][2][n][nbt])
                            #ai12 = f_subspace.get_simple_AI(Xs_mean_shuff[region][cp][tt][1][n][nbt], Xs_mean_shuff[region][cp_][tt][2][n][nbt])
                            
                            ai11 = f_subspace.get_simple_AI(projs_shuff[region][cp][tt][1][n][nbt], projs_shuff[region][cp_][tt][1][n][nbt], max_dim=2)
                            ai22 = f_subspace.get_simple_AI(projs_shuff[region][cp][tt][2][n][nbt], projs_shuff[region][cp_][tt][2][n][nbt], max_dim=2)
                            ai12 = f_subspace.get_simple_AI(projs_shuff[region][cp][tt][1][n][nbt], projs_shuff[region][cp_][tt][2][n][nbt], max_dim=2)
                            
                            ai_11_shuffT[n,nbt,nc,nc_], ai_22_shuffT[n,nbt,nc,nc_], ai_12_shuffT[n,nbt,nc,nc_] = ai11, ai22, ai12
                            
            
            ai_11_shuff_all = np.concatenate([ai_11_shuffT[:,nbt,:,:] for nbt in range(nBoots*nPerms)])
            ai_22_shuff_all = np.concatenate([ai_22_shuffT[:,nbt,:,:] for nbt in range(nBoots*nPerms)])
            ai_12_shuff_all = np.concatenate([ai_12_shuffT[:,nbt,:,:] for nbt in range(nBoots*nPerms)])
            
            for i in range(len(checkpoints)):
                for j in range(len(checkpoints)):
                    
                    ai11, ai22, ai12 = ai_11T.mean(axis=1)[:,i,j].round(5), ai_22T.mean(axis=1)[:,i,j].round(5), ai_12T.mean(axis=1)[:,i,j].round(5)
                    
                    
                    # shuff distribution
                    
                    ai11_shuff, ai22_shuff, ai12_shuff = ai_11_shuff_all[:,i,j].round(5), ai_22_shuff_all[:,i,j].round(5), ai_12_shuff_all[:,i,j].round(5)
                    
                    # kstest pvalue
                    
                    pai_11T[i,j] = f_stats.permutation_p(ai11.mean(), ai11_shuff) # stats.kstest(ai11, ai11_shuff)[-1] # 
                    pai_22T[i,j] = f_stats.permutation_p(ai22.mean(), ai22_shuff) # stats.kstest(ai22, ai22_shuff)[-1] # 
                    pai_12T[i,j] = f_stats.permutation_p(ai12.mean(), ai12_shuff) # stats.kstest(ai12, ai12_shuff)[-1] # 
            
        
        
        ### alignment index
        plt.figure(figsize=(16, 6), dpi=100)
        plt.subplot(1,3,1)
        ax = plt.gca()
        #im = ax.imshow(ai_11T.mean(axis=1).mean(axis=0), cmap='OrRd', aspect='auto', vmin=0, vmax=1) #
        im = ax.imshow(f_plotting.mask_triangle(ai_11T.mean(axis=1).mean(axis=0), ul='u', diag=1), cmap='OrRd', aspect='auto', vmin=0, vmax=1) #
        #im = ax.imshow(np.median(cosTheta_11T.mean(axis=1), axis=0), cmap='jet', aspect='auto', vmin=0, vmax=1)
        for i in range(len(checkpoints)):
            for j in range(len(checkpoints)):
                if 0.05 < pai_11T[i,j] <= 0.1:
                    text = ax.text(j, i, '+', ha="center", va="center", fontsize=6)
                elif 0.01 < pai_11T[i,j] <= 0.05:
                    text = ax.text(j, i, '*', ha="center", va="center", fontsize=6)
                elif pai_11T[i,j] <= 0.01:
                    text = ax.text(j, i, '**', ha="center", va="center", fontsize=6)
                    
        ax.set_title('Item1-Item1', fontsize = 15, pad = 15)
        ax.set_xticks([n for n in range(len(checkpoints))])
        ax.set_xticklabels(checkpoints, fontsize = 10)
        ax.set_yticks([n for n in range(len(checkpoints))])
        ax.set_yticklabels(checkpoints, fontsize = 10)
        ax.set_xlabel('Item 1', fontsize = 15)
        ax.set_ylabel('Item 1', fontsize = 15)
        ax.set_frame_on(False)
        
        
        plt.subplot(1,3,2)
        ax = plt.gca()
        #im = ax.imshow(ai_22T.mean(axis=1).mean(axis=0), cmap='OrRd', aspect='auto', vmin=0, vmax=1) #
        im = ax.imshow(f_plotting.mask_triangle(ai_22T.mean(axis=1).mean(axis=0), ul='u', diag=1), cmap='OrRd', aspect='auto', vmin=0, vmax=1) #
        #im = ax.imshow(np.median(cosTheta_22T.mean(axis=1), axis=0), cmap='jet', aspect='auto', vmin=0, vmax=1)
        for i in range(len(checkpoints)):
            for j in range(len(checkpoints)):
                if 0.05 < pai_22T[i,j] <= 0.1:
                    text = ax.text(j, i, '+', ha="center", va="center", fontsize=6)
                elif 0.01 < pai_22T[i,j] <= 0.05:
                    text = ax.text(j, i, '*', ha="center", va="center", fontsize=6)
                elif pai_22T[i,j] <= 0.01:
                    text = ax.text(j, i, '**', ha="center", va="center", fontsize=6)
                    
        ax.set_title('Item2-Item2', fontsize = 15, pad = 15)
        ax.set_xticks([n for n in range(len(checkpoints))])
        ax.set_xticklabels(checkpoints, fontsize = 10)
        ax.set_yticks([n for n in range(len(checkpoints))])
        ax.set_yticklabels(checkpoints, fontsize = 10)
        ax.set_xlabel('Item 2', fontsize = 15)
        ax.set_ylabel('Item 2', fontsize = 15)
        ax.set_frame_on(False)
        
        
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
        ax.set_title('Item1-Item2', fontsize = 15, pad = 15)
        ax.set_xticks([n for n in range(len(checkpoints))])
        ax.set_xticklabels(checkpoints, fontsize = 10)
        ax.set_yticks([n for n in range(len(checkpoints))])
        ax.set_yticklabels(checkpoints, fontsize = 10)
        ax.set_xlabel('Item 2', fontsize = 15)
        ax.set_ylabel('Item 1', fontsize = 15)
        ax.set_frame_on(False)
        
        plt.colorbar(im, ax=ax)
        
        plt.subplots_adjust(top = 0.8)
        plt.suptitle(f'alignment index, {region}, ttype={tt}', fontsize = 20, y=1)
        plt.tight_layout()
        plt.show()
        

                    

# In[]
########### test plane replacement
# In[] test plane replacement: lda trained on d1i1 test on d2i2

validCheckpoints = {'dlpfc': {'d1':(150, 550, 1150, 1450, 1850, 2350), 'd2':(150, 550, 1150, 1450, 1850, 2350)}, 
                    'fef': {'d1':(150, 550, 1150, 1450, 1850, 2350), 'd2':(150, 550, 1150, 1450, 1850, 2350)}} # , 2800, 2800, 2800, 2800


for region in ('dlpfc','fef'):
    
    validCps1 = validCheckpoints[region]['d1'] #checkpoints
    validCps2 = validCheckpoints[region]['d2'] #checkpoints
        
    # just apply to retarget trials
    for tt in ttypes: #(1,)
        
        ttype = 'Retarget' if tt==1 else 'Distraction'
        
        ### 3d projs
        xtemp_pfms12 = np.zeros((nIters,len(validCps1), len(validCps2)))
        xtemp_pfms12_shuff = np.zeros((nIters, nPerms,len(validCps1), len(validCps2)))
        xtemp_pfms21 = np.zeros((nIters,len(validCps2), len(validCps1)))
        xtemp_pfms21_shuff = np.zeros((nIters, nPerms,len(validCps2), len(validCps1)))
        
        ### 2d projs
        #xtemp_pfms12_2d = np.zeros((nIters,len(validCps1), len(validCps2)))
        #xtemp_pfms12_2d_shuff = np.zeros((nIters, nPerms,len(validCps1), len(validCps2)))
        #xtemp_pfms21_2d = np.zeros((nIters,len(validCps2), len(validCps1)))
        #xtemp_pfms21_2d_shuff = np.zeros((nIters, nPerms,len(validCps2), len(validCps1)))
        
        for n in range(nIters):
            pseudo_TrialInfoT = np.load(save_path + f'/pseudo_all{n}.npy', allow_pickle=True).item()['pseudo_TrialInfo']
            pseudo_TrialInfoT = pseudo_TrialInfoT[pseudo_TrialInfoT.type == tt].reset_index(drop=True)
            
            loc1_labelsT, loc2_labelsT = pseudo_TrialInfoT.loc1, pseudo_TrialInfoT.loc2
            
            for nv1, vcp1 in enumerate(validCps1):
                data1_T1, data2_T1 = np.concatenate(projsAll[region][vcp1][tt][1][n]), np.concatenate(projsAll[region][vcp1][tt][2][n])
                
                for nv2, vcp2 in enumerate(validCps2):
                    
                    ### 3d
                    data1_T2, data2_T2 = np.concatenate(projsAll[region][vcp2][tt][1][n]), np.concatenate(projsAll[region][vcp2][tt][2][n])
                    #train_dataT_mean, test_dataT_mean = np.concatenate(projs[region][vcp1][tt][1][n]), np.concatenate(projs[region][vcp2][tt][2][n])
                    
                    # standard scaling
                    #train_dataT_Z, test_dataT_Z = train_dataT.copy(), test_dataT.copy()
                    #for ch in range(3):
                    #    train_dataT_Z[:,ch] = (train_dataT[:,ch] - train_dataT[:,ch].mean())/train_dataT[:,ch].std()
                    #    test_dataT_Z[:,ch] = (test_dataT[:,ch] - test_dataT[:,ch].mean())/test_dataT[:,ch].std()
                        
                    xtemp_pfms12[n, nv1, nv2] = f_decoding.LDAPerformance(data1_T1, data2_T2, loc1_labelsT, loc2_labelsT) # train 1 test 2
                    xtemp_pfms21[n, nv1, nv2] = f_decoding.LDAPerformance(data2_T1, data1_T2, loc2_labelsT, loc1_labelsT) # train 2 test 1
                   
                    
                   
                    ### 2d
                    #train_dataT_2d = f_subspace.projsAll_to_2d(np.concatenate(vecs[region][vcp1][tt][1][n]), train_dataT_mean, train_dataT)
                    #test_dataT_2d = f_subspace.projsAll_to_2d(np.concatenate(vecs[region][vcp2][tt][2][n]), test_dataT_mean, test_dataT)
                    #train_dataT_2d, test_dataT_2d = (train_dataT_2d - train_dataT_2d.mean(axis=0)), (test_dataT_2d - test_dataT_2d.mean(axis=0))
                    
                    #train_dataT_2d_mean, test_dataT_2d_mean = f_subspace.proj_2D_coordinates(train_dataT_mean, np.concatenate(vecs[region][vcp1][tt][1][n])).T, f_subspace.proj_2D_coordinates(test_dataT_mean, np.concatenate(vecs[region][vcp2][tt][2][n])).T
                    #train_dataT_2d_mean, test_dataT_2d_mean = (train_dataT_2d_mean - train_dataT_2d_mean.mean(axis=0)), (test_dataT_2d_mean - test_dataT_2d_mean.mean(axis=0))
                    
                    # rotate to the minimal Forb. Norm in conditional avg configs
                    #R_, _ = scipy.linalg.orthogonal_procrustes(train_dataT_2d_mean, test_dataT_2d_mean)
                    #test_dataT_2dR = np.dot(test_dataT_2d, R_.T)
                    
                    # standard scaling
                    #train_dataT_2d_Z, test_dataT_2d_Z = train_dataT_2d.copy(), test_dataT_2d.copy()
                    #for ch in range(2):
                    #    train_dataT_2d_Z[:,ch] = (train_dataT_2d[:,ch] - train_dataT_2d[:,ch].mean())/train_dataT_2d[:,ch].std()
                    #    test_dataT_2d_Z[:,ch] = (test_dataT_2d[:,ch] - test_dataT_2d[:,ch].mean())/test_dataT_2d[:,ch].std()
                        
                    #xtemp_pfms12_2d[n, nv1, nv2] = f_decoding.LDAPerformance(train_dataT_2d_Z, test_dataT_2d_Z, train_labelsT, test_labelsT)
                    #xtemp_pfms21_2d[n, nv2, nv1] = f_decoding.LDAPerformance(test_dataT_2d_Z, train_dataT_2d_Z, test_labelsT, train_labelsT)
                    
                    #permutation p value checks
                    for npm in range(nPerms): #nPerms
                        loc1_labelsT_shuff, loc2_labelsT_shuff = np.random.permutation(pseudo_TrialInfoT.loc1), np.random.permutation(pseudo_TrialInfoT.loc2)
                        
                        xtemp_pfms12_shuff[n, npm, nv1, nv2] = f_decoding.LDAPerformance(data1_T1, data2_T2, loc1_labelsT_shuff, loc2_labelsT_shuff)
                        xtemp_pfms21_shuff[n, npm, nv1, nv2] = f_decoding.LDAPerformance(data2_T1, data1_T2, loc2_labelsT_shuff, loc1_labelsT_shuff)
                        
                        #xtemp_pfms12_2d_shuff[n, npm, nv1, nv2] = f_decoding.LDAPerformance(train_dataT_2d_Z, test_dataT_2d_Z, train_labelsT_shuff, test_labelsT_shuff)
                        #xtemp_pfms21_2d_shuff[n, npm, nv2, nv1] = f_decoding.LDAPerformance(test_dataT_2d_Z, train_dataT_2d_Z, test_labelsT_shuff, train_labelsT_shuff)
        
        ### pvalues
        pPfms12 = np.zeros((len(validCps1), len(validCps2)))
        pPfms21 = np.zeros((len(validCps2), len(validCps1)))
        
        #pPfms12_2d = np.zeros((len(validCps1), len(validCps2)))
        #pPfms21_2d = np.zeros((len(validCps2), len(validCps1)))
        
        
        for nv1, _ in enumerate(validCps1):
            for nv2, _ in enumerate(validCps2):
                pPfms12[nv1, nv2] = f_stats.permutation_p(xtemp_pfms12.mean(axis=0)[nv1, nv2], np.concatenate(xtemp_pfms12_shuff,axis=0)[:, nv1, nv2], tail = 'greater')
                pPfms21[nv1, nv2] = f_stats.permutation_p(xtemp_pfms21.mean(axis=0)[nv1, nv2], np.concatenate(xtemp_pfms21_shuff,axis=0)[:, nv1, nv2], tail = 'greater')
                
                #pPfms12_2d[nv1, nv2] = f_stats.permutation_p(xtemp_pfms12_2d.mean(axis=0)[nv1, nv2], np.concatenate(xtemp_pfms12_2d_shuff,axis=0)[:, nv1, nv2], tail = 'greater')
                #pPfms21_2d[nv2, nv1] = f_stats.permutation_p(xtemp_pfms21_2d.mean(axis=0)[nv2, nv1], np.concatenate(xtemp_pfms21_2d_shuff,axis=0)[:, nv2, nv1], tail = 'greater')
        
        
        ### plot
        vmax = 0.6 if region == 'dlpfc' else 0.8
        #plt.figure(figsize=(10,9), dpi = 100)
        plt.figure(figsize=(10,5), dpi = 100)
        ### 3d - train1 test2
        #plt.subplot(2,2,1)
        plt.subplot(1,2,1)
        ax = plt.gca()
        im = ax.imshow(xtemp_pfms12.mean(axis=0), cmap='magma', aspect='auto', vmin = 0.1, vmax = vmax) #
        
        for i in range(len(validCps1)):
            for j in range(len(validCps2)):
                if 0.05 < pPfms12[i,j] <= 0.1:
                    text = ax.text(j, i, '+', ha="center", va="center", fontsize=15, color = 'white') #
                elif 0.01 < pPfms12[i,j] <= 0.05:
                    text = ax.text(j, i, '*', ha="center", va="center", fontsize=15, color = 'white') #
                elif pPfms12[i,j] <= 0.01:
                    text = ax.text(j, i, '**', ha="center", va="center", fontsize=15, color = 'white') #
                    
        ax.set_xticks([n for n in range(len(validCps2))])
        ax.set_xticklabels(checkpointsLabels, fontsize = 10)
        ax.set_yticks([n for n in range(len(validCps1))])
        ax.set_yticklabels(checkpointsLabels, fontsize = 10)
        ax.set_xlabel('Test', fontsize = 15)
        ax.set_ylabel('Train', fontsize = 15)
        
        #ax.invert_yaxis()
        ax.set_title('Item1 -> Item2', pad = 15)
        plt.colorbar(im,ax=ax)
        
        
        ### 2d - 12
        #plt.subplot(2,2,2)
        #ax = plt.gca()
        #im = ax.imshow(xtemp_pfms12_2d.mean(axis=0), cmap='YlOrRd', aspect='auto', vmin = 0.1, vmax = 0.8) #
        #for i in range(len(validCps1)):
        #    for j in range(len(validCps2)):
        #        if 0.05 < pPfms12_2d[i,j] <= 0.1:
        #            text = ax.text(j, i, '+', ha="center", va="center", fontsize=6) #, color = 'yellow'
        #        elif 0.01 < pPfms12_2d[i,j] <= 0.05:
        #            text = ax.text(j, i, '*', ha="center", va="center", fontsize=6) #, color = 'yellow'
        #        elif pPfms12_2d[i,j] <= 0.01:
        #            text = ax.text(j, i, '**', ha="center", va="center", fontsize=6) #, color = 'yellow'
                   
        #ax.set_xticks([n for n in range(len(validCps2))])
        #ax.set_xticklabels(validCps2, fontsize = 6)
        #ax.set_yticks([n for n in range(len(validCps1))])
        #ax.set_yticklabels(validCps1, fontsize = 6)
        #ax.set_xlabel('Test Loc2')
        #ax.set_ylabel('Train Loc1')
        
        #ax.invert_yaxis()
        #ax.set_title('2d Projs')
        #plt.colorbar(im,ax=ax)
        
        
        ### 3d - train2 test1
        #plt.subplot(2,2,3)
        plt.subplot(1,2,2)
        ax = plt.gca()
        im = ax.imshow(xtemp_pfms21.mean(axis=0), cmap='magma', aspect='auto', vmin = 0.1, vmax = vmax) #
        for i in range(len(validCps2)):
            for j in range(len(validCps1)):
                if 0.05 < pPfms21[i,j] <= 0.1:
                    text = ax.text(j, i, '+', ha="center", va="center", fontsize=15, color = 'white') #
                elif 0.01 < pPfms21[i,j] <= 0.05:
                    text = ax.text(j, i, '*', ha="center", va="center", fontsize=15, color = 'white') #
                elif pPfms21[i,j] <= 0.01:
                    text = ax.text(j, i, '**', ha="center", va="center", fontsize=15, color = 'white') #
                    
        ax.set_xticks([n for n in range(len(validCps1))])
        ax.set_xticklabels(checkpointsLabels, fontsize = 10)
        ax.set_yticks([n for n in range(len(validCps2))])
        ax.set_yticklabels(checkpointsLabels, fontsize = 10)
        ax.set_xlabel('Test', fontsize = 15)
        ax.set_ylabel('Train', fontsize = 15)
        
        #ax.invert_yaxis()
        ax.set_title('Item2 -> Item1', pad = 15)
        plt.colorbar(im,ax=ax)
        
        
        ### 2d - 21
        #plt.subplot(2,2,4)
        #ax = plt.gca()
        #im = ax.imshow(xtemp_pfms21_2d.mean(axis=0), cmap='YlOrRd', aspect='auto', vmin = 0.1, vmax = 0.8) #
        #for i in range(len(validCps2)):
        #    for j in range(len(validCps1)):
        #        if 0.05 < pPfms21_2d[i,j] <= 0.1:
        #            text = ax.text(j, i, '+', ha="center", va="center", fontsize=6) #, color = 'yellow'
        #        elif 0.01 < pPfms21_2d[i,j] <= 0.05:
        #            text = ax.text(j, i, '*', ha="center", va="center", fontsize=6) #, color = 'yellow'
        #        elif pPfms21_2d[i,j] <= 0.01:
        #            text = ax.text(j, i, '**', ha="center", va="center", fontsize=6) #, color = 'yellow'
                    
        #ax.set_xticks([n for n in range(len(validCps1))])
        #ax.set_xticklabels(validCps1, fontsize = 6)
        #ax.set_yticks([n for n in range(len(validCps2))])
        #ax.set_yticklabels(validCps2, fontsize = 6)
        #ax.set_xlabel('Test Loc1')
        #ax.set_ylabel('Train Loc2')
        
        #ax.invert_yaxis()
        #ax.set_title('2d Projs')
        #plt.colorbar(im,ax=ax)
        
        
        plt.subplots_adjust(top = 0.8)
        plt.suptitle(f'Item Code Transferability, {region.upper()}, {ttype}', fontsize = 20, y=1)
        plt.tight_layout()
        plt.show()

# In[] test plane replacement: lda decode sequence (i1 or i2)

validCheckpoints = {'dlpfc': {'d1':(150, 550, 1150, 1450, 1850, 2350), 'd2':(150, 550, 1150, 1450, 1850, 2350)}, 
                    'fef': {'d1':(150, 550, 1150, 1450, 1850, 2350), 'd2':(150, 550, 1150, 1450, 1850, 2350)}} # , 2800, 2800, 2800, 2800


for region in ('dlpfc','fef'):
    
    validCps1 = validCheckpoints[region]['d1'] #checkpoints
    validCps2 = validCheckpoints[region]['d2'] #checkpoints
    
    plt.figure(figsize=(10,5), dpi = 100)
    
    for tt in ttypes: # (1,)
        ttype = 'Retarget' if tt==1 else 'Distraction'
        
        
        ### 3d projs
        xtemp_pfms = np.zeros((nIters, len(validCps1), len(validCps2)))
        xtemp_pfms_mean = np.zeros((nIters, len(validCps1), len(validCps2)))
        xtemp_pfms_shuff = np.zeros((nIters, nPerms, len(validCps1), len(validCps2)))
        xtemp_pfms_mean_shuff = np.zeros((nIters, nPerms, len(validCps1), len(validCps2)))
        
        for n in range(nIters):
            
            for nv1, vcp1 in enumerate(validCps1):
                for nv2, vcp2 in enumerate(validCps2):
                    
                    item1T, item2T = np.concatenate(projsAll[region][vcp1][tt][1][n]), np.concatenate(projsAll[region][vcp2][tt][2][n])
                    item1T_mean, item2T_mean = np.concatenate(projs[region][vcp1][tt][1][n]), np.concatenate(projs[region][vcp2][tt][2][n])
                    
                    label1T, label2T = np.full((item1T.shape[0]), '1'), np.full((item2T.shape[0]), '2')
                    label1T_mean, label2T_mean = np.full((item1T_mean.shape[0]), '1'), np.full((item2T_mean.shape[0]), '2')
                    
                    # standard scaling
                    item1T_Z, item2T_Z = item1T.copy(), item2T.copy()
                    item1T_Z_mean, item2T_Z_mean = item1T_mean.copy(), item2T_mean.copy()
                    for ch in range(3):
                        item1T_Z[:,ch] = (item1T[:,ch] - item1T[:,ch].mean())/item1T[:,ch].std()
                        item2T_Z[:,ch] = (item2T[:,ch] - item2T[:,ch].mean())/item2T[:,ch].std()
                        item1T_Z_mean[:,ch] = (item1T_mean[:,ch] - item1T_mean[:,ch].mean())/item1T_mean[:,ch].std()
                        item2T_Z_mean[:,ch] = (item2T_mean[:,ch] - item2T_mean[:,ch].mean())/item2T_mean[:,ch].std()
                    
                    
                    itemsFull_T = np.concatenate((item1T_Z, item2T_Z), axis=0)
                    labelsFull_T = np.concatenate((label1T, label2T), axis=0)
                    itemsFull_T_mean = np.concatenate((item1T_Z_mean, item2T_Z_mean), axis=0)
                    labelsFull_T_mean = np.concatenate((label1T_mean, label2T_mean), axis=0)
                    
                    
                    ## all projs
                    train_IDT, test_IDT = f_subspace.split_set(itemsFull_T,frac = 0.5)
                    train_setT, test_setT = itemsFull_T[train_IDT,:], itemsFull_T[test_IDT,:]
                    train_labelsT, test_labelsT = labelsFull_T[train_IDT], labelsFull_T[test_IDT]
                    
                    xtemp_pfms[n, nv1, nv2] = f_decoding.LDAPerformance(train_setT, test_setT, train_labelsT, test_labelsT)
                    
                    
                    ## mean projs
                    train_ID_meanT, test_ID_meanT = f_subspace.split_set(itemsFull_T_mean,frac = 0.5)
                    train_set_meanT, test_set_meanT = itemsFull_T_mean[train_ID_meanT,:], itemsFull_T_mean[test_ID_meanT,:]
                    train_labels_meanT, test_labels_meanT = labelsFull_T_mean[train_ID_meanT], labelsFull_T_mean[test_ID_meanT]
                    
                    xtemp_pfms_mean[n, nv1, nv2] = f_decoding.LDAPerformance(train_set_meanT, test_set_meanT, train_labels_meanT, test_labels_meanT)
                    
                    
                    ## permutation p values
                    for npm in range(nPerms): #nPerms
                        train_labelsT_shuff, test_labelsT_shuff = np.random.permutation(train_labelsT), np.random.permutation(test_labelsT)
                        train_labels_meanT_shuff, test_labels_meanT_shuff = np.random.permutation(train_labels_meanT), np.random.permutation(test_labels_meanT)
                        
                        xtemp_pfms_shuff[n, npm, nv1, nv2] = f_decoding.LDAPerformance(train_setT, test_setT, train_labelsT_shuff, test_labelsT_shuff)
                        xtemp_pfms_mean_shuff[n, npm, nv1, nv2] = f_decoding.LDAPerformance(train_set_meanT, test_set_meanT, train_labels_meanT_shuff, test_labels_meanT_shuff)
                    
                        
        ### pvalues
        pPfms = np.zeros((len(validCps1), len(validCps2)))
        pPfms_mean = np.zeros((len(validCps1), len(validCps2)))
        
        
        for nv1, _ in enumerate(validCps1):
            for nv2, _ in enumerate(validCps2):
                pPfms[nv1, nv2] = f_stats.permutation_p(xtemp_pfms.mean(axis=0)[nv1, nv2], np.concatenate(xtemp_pfms_shuff,axis=0)[:, nv1, nv2], tail = 'greater')
                pPfms_mean[nv1, nv2] = f_stats.permutation_p(xtemp_pfms_mean.mean(axis=0)[nv1, nv2], np.concatenate(xtemp_pfms_mean_shuff,axis=0)[:, nv1, nv2], tail = 'greater')
                
        
        ### plot        
        #plt.figure(figsize=(10,4), dpi = 100)
        #plt.figure(figsize=(4,4), dpi = 100)
        ### 3d - 12
        #.subplot(1,2,1)
        plt.subplot(1,2,tt)
        ax = plt.gca()
        im = ax.imshow(xtemp_pfms.mean(axis=0), cmap='YlOrRd', aspect='auto', vmin = 0.1, vmax = 0.8) #
        for i in range(len(validCps1)):
            for j in range(len(validCps2)):
                if 0.05 < pPfms[i,j] <= 0.1:
                    text = ax.text(j, i, '+', ha="center", va="center", fontsize=6) #, color = 'yellow'
                elif 0.01 < pPfms[i,j] <= 0.05:
                    text = ax.text(j, i, '*', ha="center", va="center", fontsize=6) #, color = 'yellow'
                elif pPfms[i,j] <= 0.01:
                    text = ax.text(j, i, '**', ha="center", va="center", fontsize=6) #, color = 'yellow'
                    
        ax.set_xticks([n for n in range(len(validCps2))])
        ax.set_xticklabels(checkpointsLabels, fontsize = 10)
        ax.set_yticks([n for n in range(len(validCps1))])
        ax.set_yticklabels(checkpointsLabels, fontsize = 10)
        ax.set_xlabel('Test', fontsize = 15)
        ax.set_ylabel('Train', fontsize = 15)
        
        #ax.invert_yaxis()
        ax.set_title(f'{ttype}', fontsize = 15)
        plt.colorbar(im,ax=ax)
        
        
        ### 2d - 12
        #plt.subplot(1,2,2)
        #ax = plt.gca()
        #im = ax.imshow(xtemp_pfms_mean.mean(axis=0), cmap='YlOrRd', aspect='auto', vmin = 0.1, vmax = 0.8) #
        #for i in range(len(validCps1)):
        #    for j in range(len(validCps2)):
        #        if 0.05 < pPfms_mean[i,j] <= 0.1:
        #            text = ax.text(j, i, '+', ha="center", va="center", fontsize=6) #, color = 'yellow'
        #        elif 0.01 < pPfms_mean[i,j] <= 0.05:
        #            text = ax.text(j, i, '*', ha="center", va="center", fontsize=6) #, color = 'yellow'
        #        elif pPfms_mean[i,j] <= 0.01:
        #            text = ax.text(j, i, '**', ha="center", va="center", fontsize=6) #, color = 'yellow'
                    
        #ax.set_xticks([n for n in range(len(validCps2))])
        #ax.set_xticklabels(checkpointsLabels, fontsize = 6)
        #ax.set_yticks([n for n in range(len(validCps1))])
        #ax.set_yticklabels(checkpointsLabels, fontsize = 6)
        #ax.set_xlabel('D1')
        #ax.set_ylabel('D2')
        
        ##ax.invert_yaxis()
        #ax.set_title('Projs Mean')
        #plt.colorbar(im,ax=ax)
        
        
    plt.subplots_adjust(top = 0.8)
    plt.suptitle(f'Sequence Decodability, {region.upper()}', fontsize = 20, y=1)#, {ttype}
    plt.tight_layout()
    plt.show()
    
# In[] condition mean eu-distance between plane centers

validCheckpoints = {'dlpfc': {'d1':(150, 550, 1150, 1450, 1850, 2350, 2800), 'd2':(150, 550, 1150, 1450, 1850, 2350, 2800)}, 
                    'fef': {'d1':(150, 550, 1150, 1450, 1850, 2350, 2800), 'd2':(150, 550, 1150, 1450, 1850, 2350, 2800)}} # 


for region in ('dlpfc','fef'):
    
    validCps1 =  validCheckpoints[region]['d1'] #checkpoints #
    validCps2 =  validCheckpoints[region]['d2'] #checkpoints #
    
    # just apply to retarget trials
    for tt in ttypes: #(1,)
        
        xtemp_dist = np.zeros((nIters,len(validCps1), len(validCps2)))
        
        for n in range(nIters):
            pointsT_mean = [] 
            #pointsT_all = []
            
            for ncp, cp in enumerate(checkpoints): #validCps1
                for tt_ in (1,2,):
                    pointsT_mean += [np.concatenate((np.concatenate(projs[region][cp][tt_][1][n]), np.concatenate(projs[region][cp][tt_][2][n])))]
                    #pointsT_all += [np.concatenate((np.concatenate(projsAll[region][cp][tt_][1][n]), np.concatenate(projsAll[region][cp][tt_][2][n])))]
            
            
            pointsT_mean = np.concatenate(pointsT_mean)
            meanT_mean, stdT_mean = pointsT_mean.mean(axis=0), pointsT_mean.std(axis=0)
            #pointsT_all = np.concatenate(pointsT_all)
            #meanT_all, stdT_all = pointsT_all.mean(axis=0), pointsT_all.std(axis=0)
            
            for nv1, vcp1 in enumerate(validCps1):
                for nv2, vcp2 in enumerate(validCps2):
                    center1T = np.concatenate(projs[region][vcp1][tt][1][n]).mean(axis=0) 
                    center2T = np.concatenate(projs[region][vcp2][tt][2][n]).mean(axis=0) if tt == 1 else np.concatenate(projs[region][vcp2][tt][1][n]).mean(axis=0)
                    center1Z, center2Z = (center1T - meanT_mean)/stdT_mean, (center2T - meanT_mean)/stdT_mean
                    
                    xtemp_dist[n, nv1, nv2] = f_subspace.euclidean_distance(center1Z, center2Z)
        
        # shuffles
        xtemp_dist_shuff = np.zeros((nIters, nPerms,len(validCps1), len(validCps2)))
        
        for n in range(nIters):
            for npm in range(nPerms):
                pointsT_mean_shuff = [] 
                
                for ncp, cp in enumerate(checkpoints): #validCps1
                    for tt_ in (1,2,):
                        pointsT_mean_shuff += [np.concatenate((projs_shuff[region][cp][tt_][1][n][npm], projs_shuff[region][cp][tt_][2][n][npm]))]
                        #pointsT_all_shuff += [np.concatenate((np.concatenate(projsAll_shuff[region][cp][tt_][1][n]), np.concatenate(projsAll_shuff[region][cp][tt_][2][n])))]
                
                pointsT_mean_shuff = np.concatenate(pointsT_mean_shuff)
                meanT_mean_shuff, stdT_mean_shuff = pointsT_mean_shuff.mean(axis=0), pointsT_mean_shuff.std(axis=0)
                
                for nv1, vcp1 in enumerate(validCps1):
                    for nv2, vcp2 in enumerate(validCps2):
                        center1T_shuff = projs_shuff[region][vcp1][tt][1][n][npm].mean(axis=0)
                        center2T_shuff = projs_shuff[region][vcp2][tt][2][n][npm].mean(axis=0) if tt == 1 else projs_shuff[region][vcp2][tt][1][n][npm].mean(axis=0)
                        center1Z_shuff, center2Z_shuff = (center1T_shuff - meanT_mean_shuff)/stdT_mean_shuff, (center2T_shuff - meanT_mean_shuff)/stdT_mean_shuff
                        
                        xtemp_dist_shuff[n, npm, nv1, nv2] = f_subspace.euclidean_distance(center1Z_shuff, center2Z_shuff)
        
        #xtemp_distZ = (xtemp_dist - xtemp_dist_shuff.mean(axis=1)) / xtemp_dist_shuff.std(axis=1)
        
        #xtemp_dist_shuffZ = []
        #for n in range(nIters):
        #    xtemp_dist_shuffZ += [(xtemp_dist_shuff[n] - xtemp_dist_shuff[n].mean(axis=0)) / xtemp_dist_shuff[n].std(axis=0)]
            
        #xtemp_dist_shuffZ = np.array(xtemp_dist_shuffZ)
        
        # xtemp_dist_shuffZ = (xtemp_dist_shuff - xtemp_dist_shuff.mean(axis=1)) / xtemp_dist_shuff.mean(axis=1)
        
        pDists = np.zeros((len(validCps1), len(validCps2)))
        for nv1, _ in enumerate(validCps1):
            for nv2, _ in enumerate(validCps2):
                pDists[nv1, nv2] = f_stats.permutation_p(xtemp_dist.round(5).mean(axis=0)[nv1, nv2], np.concatenate(xtemp_dist_shuff,axis=0).round(5)[:, nv1, nv2], tail = 'greater')
                #pDists[nv1, nv2] = f_stats.permutation_p(xtemp_distZ.round(5).mean(axis=0)[nv1, nv2], np.concatenate(xtemp_dist_shuffZ,axis=0).round(5)[:, nv1, nv2], tail = 'two')
        
        ### plot        
        plt.figure(figsize=(5,4), dpi = 100)
        ### 3d
        plt.subplot(1,1,1)
        ax = plt.gca()
        im = ax.imshow(xtemp_dist.mean(axis=0), cmap='YlOrRd', aspect='auto', vmin = 0, vmax = 5)#
        #im = ax.imshow(xtemp_distZ.mean(axis=0).T, cmap='Blues_r', aspect='auto', vmin = 0, vmax = 5)#
        for i in range(len(validCps1)):
            for j in range(len(validCps2)):
                if 0.05 < pDists[i,j] <= 0.1:
                    text = ax.text(j, i, '+', ha="center", va="center", fontsize=6)
                elif 0.01 < pDists[i,j] <= 0.05:
                    text = ax.text(j, i, '*', ha="center", va="center", fontsize=6)
                elif pDists[i,j] <= 0.01:
                    text = ax.text(j, i, '**', ha="center", va="center", fontsize=6)
                    
        ax.set_xticks([n for n in range(len(validCps2))])
        ax.set_xticklabels(checkpointsLabels, fontsize = 6)
        ax.set_yticks([n for n in range(len(validCps1))])
        ax.set_yticklabels(checkpointsLabels, fontsize = 6)
        ax.set_xlabel('Time (Item2)')
        ax.set_ylabel('Time (Item1)')
        
        #ax.invert_yaxis()
        #ax.set_title('3d Projs')
        plt.colorbar(im,ax=ax)
        
        plt.suptitle(f'{region}, retarget, Z_EuDist Item1-Item2, ttype = {tt}')
        plt.show()
        









































































#########################################
#########################################
#########################################
#########################################
#########################################
#########################################
#########################################
#########################################
#########################################
# In[] decodability plane projection
decode_proj1, decode_proj2 = {},{}
decode_proj1_shuff_mean,decode_proj1_shuff_all = {},{}
decode_proj2_shuff_mean,decode_proj2_shuff_all = {},{}

for region in ('dlpfc','fef'):
    
    decode_proj1[region] = {}
    decode_proj2[region] = {}
    decode_proj1_shuff_mean[region], decode_proj1_shuff_all[region] = {},{}
    decode_proj2_shuff_mean[region], decode_proj2_shuff_all[region] = {},{}

    
    for tt in ttypes:
        decode_proj1T = np.zeros((nIters, nBoots, len(checkpoints),))
        decode_proj2T = np.zeros((nIters, nBoots, len(checkpoints),))
        
        for n in range(nIters):
            for nbt in range(nBoots):
                for nc,cp in enumerate(checkpoints):
                    decode_proj1T[n,nbt,nc] = decodability_proj[region][cp][tt][1][n][nbt]
                    decode_proj2T[n,nbt,nc] = decodability_proj[region][cp][tt][2][n][nbt]
        
        decode_proj1[region][tt] = decode_proj1T
        decode_proj2[region][tt] = decode_proj2T 
        
        decode_proj1_shuffT = np.zeros((nIters, nBoots*nPerms, len(checkpoints),))
        decode_proj2_shuffT = np.zeros((nIters, nBoots*nPerms, len(checkpoints),))
        
        for n in range(nIters):
            for nbt in range(nBoots*nPerms):
                for nc, cp in enumerate(checkpoints):
                    decode_proj1_shuffT[n,nbt,nc,] = decodability_proj_shuff[region][cp][tt][1][n][nbt]
                    decode_proj2_shuffT[n,nbt,nc,] = decodability_proj_shuff[region][cp][tt][2][n][nbt]
        
        decode_proj1_shuffT_mean, decode_proj1_shuffT_all = decode_proj1_shuffT.mean(axis=1), np.concatenate([decode_proj1_shuffT[:,nbt,:] for nbt in range(nBoots*nPerms)])
        decode_proj1_shuff_mean[region][tt], decode_proj1_shuff_all[region][tt] = decode_proj1_shuffT_mean, decode_proj1_shuffT_all
        
        decode_proj2_shuffT_mean, decode_proj2_shuffT_all = decode_proj2_shuffT.mean(axis=1), np.concatenate([decode_proj2_shuffT[:,nbt,:] for nbt in range(nBoots*nPerms)])
        decode_proj2_shuff_mean[region][tt], decode_proj2_shuff_all[region][tt] = decode_proj2_shuffT_mean, decode_proj2_shuffT_all
        
    
    plt.figure(figsize=(10, 3), dpi=100)    
    for tt in ttypes:
        
        #colorT = 'b' if tt == 1 else 'm'
        condT = 'retarget' if tt == 1 else 'distractor'
        
        pPerms_decode1 = np.array([f_stats.permutation_p(decode_proj1[region][tt].mean(axis=1).mean(axis=0)[nc], decode_proj1_shuff_all[region][tt][:,nc], tail='greater') for nc, cp in enumerate(checkpoints)])
        pPerms_decode2 = np.array([f_stats.permutation_p(decode_proj2[region][tt].mean(axis=1).mean(axis=0)[nc], decode_proj2_shuff_all[region][tt][:,nc], tail='greater') for nc, cp in enumerate(checkpoints)])
        
        
        plt.subplot(1,2,tt)
        ax = plt.gca()
        ax.errorbar(np.arange(0, len(checkpoints), 1), decode_proj1[region][tt].mean(axis=1).mean(axis=0), yerr = decode_proj1[region][tt].mean(axis=1).std(axis=0), marker = 'o', color = 'b', label = 'loc1')
        #ax.plot(np.arange(0, len(checkpoints), 1), pPerms_decode1, color = 'b', marker = '*', linestyle='', alpha = 0.5)
        trans = ax.get_xaxis_transform()
        for nc, cp in enumerate(checkpoints):
            if 0.05 < pPerms_decode1[nc] <= 0.1:
                ax.annotate('+', xy=(nc, 0.01), xycoords = trans, annotation_clip = False, color = 'b', ha = 'center', va = 'center')
            elif 0.01 < pPerms_decode1[nc] <= 0.05:
                ax.annotate('*', xy=(nc, 0.01), xycoords = trans, annotation_clip = False, color = 'b', ha = 'center', va = 'center')
            elif pPerms_decode1[nc] <= 0.01:
                ax.annotate('**', xy=(nc, 0.01), xycoords = trans, annotation_clip = False, color = 'b', ha = 'center', va = 'center')
                
        ax.errorbar(np.arange(0, len(checkpoints), 1), decode_proj2[region][tt].mean(axis=1).mean(axis=0), yerr = decode_proj2[region][tt].mean(axis=1).std(axis=0), marker = 'o', color = 'm', label = 'loc2')
        #ax.plot(np.arange(0, len(checkpoints), 1), pPerms_decode2, color = 'm', marker = '*', linestyle='', alpha = 0.5)
        trans = ax.get_xaxis_transform()
        for nc, cp in enumerate(checkpoints):
            if 0.05 < pPerms_decode2[nc] <= 0.1:
                ax.annotate('+', xy=(nc, 0.05), xycoords = trans, annotation_clip = False, color = 'm', ha = 'center', va = 'center')
            elif 0.01 < pPerms_decode2[nc] <= 0.05:
                ax.annotate('*', xy=(nc, 0.05), xycoords = trans, annotation_clip = False, color = 'm', ha = 'center', va = 'center')
            elif pPerms_decode2[nc] <= 0.01:
                ax.annotate('**', xy=(nc, 0.05), xycoords = trans, annotation_clip = False, color = 'm', ha = 'center', va = 'center')
                
        ax.set_title(f'{condT}', pad = 10)
        ax.set_xticks([n for n in range(len(checkpoints))])
        ax.set_xticklabels(checkpoints, fontsize = 6)
        ax.set_ylim((0,1))
        ax.legend()
        
    
    plt.subplots_adjust(top = 0.8)
    plt.suptitle(f'decodability_proj, {region}', fontsize = 15, y=1)
    plt.show()  



########### test plane replacement

# In[] test plane replacement: lda trained on d1i1 test on d2i2

validCheckpoints = {'dlpfc': {'d1':(150, 300, 500, 700, 900, 1100, 1300), 'd2':(1450, 1600, 1800, 2000, 2200, 2400, 2600)}, 
                    'fef': {'d1':(150, 300, 500, 700, 900, 1100, 1300), 'd2':(1450, 1600, 1800, 2000, 2200, 2400, 2600)}} # 


for region in ('dlpfc','fef'):
    
    validCps1 = validCheckpoints[region]['d1'] #checkpoints
    validCps2 = validCheckpoints[region]['d2'] #checkpoints
        
    # just apply to retarget trials
    for tt in ttypes: #(1,)
        
        ### 3d projs
        xtemp_pfms12 = np.zeros((nIters,len(validCps1), len(validCps2)))
        xtemp_pfms12_shuff = np.zeros((nIters, nPerms,len(validCps1), len(validCps2)))
        xtemp_pfms21 = np.zeros((nIters,len(validCps2), len(validCps1)))
        xtemp_pfms21_shuff = np.zeros((nIters, nPerms,len(validCps2), len(validCps1)))
        
        ### 2d projs
        xtemp_pfms12_2d = np.zeros((nIters,len(validCps1), len(validCps2)))
        xtemp_pfms12_2d_shuff = np.zeros((nIters, nPerms,len(validCps1), len(validCps2)))
        xtemp_pfms21_2d = np.zeros((nIters,len(validCps2), len(validCps1)))
        xtemp_pfms21_2d_shuff = np.zeros((nIters, nPerms,len(validCps2), len(validCps1)))
        
        for n in range(nIters):
            pseudo_TrialInfoT = np.load(save_path + f'/pseudo_all{n}.npy', allow_pickle=True).item()['pseudo_TrialInfo']
            pseudo_TrialInfoT = pseudo_TrialInfoT[pseudo_TrialInfoT.type == tt].reset_index(drop=True)
            
            for nv1, vcp1 in enumerate(validCps1):
                for nv2, vcp2 in enumerate(validCps2):
                    train_labelsT, test_labelsT = pseudo_TrialInfoT.loc1, pseudo_TrialInfoT.loc2
                    
                    ### 3d
                    train_dataT, test_dataT = np.concatenate(projsAll[region][vcp1][tt][1][n]), np.concatenate(projsAll[region][vcp2][tt][2][n])
                    train_dataT_mean, test_dataT_mean = np.concatenate(projs[region][vcp1][tt][1][n]), np.concatenate(projs[region][vcp2][tt][2][n])
                    
                    # standard scaling
                    train_dataT_Z, test_dataT_Z = train_dataT.copy(), test_dataT.copy()
                    for ch in range(3):
                        train_dataT_Z[:,ch] = (train_dataT[:,ch] - train_dataT[:,ch].mean())/train_dataT[:,ch].std()
                        test_dataT_Z[:,ch] = (test_dataT[:,ch] - test_dataT[:,ch].mean())/test_dataT[:,ch].std()
                        
                    xtemp_pfms12[n, nv1, nv2] = f_decoding.LDAPerformance(train_dataT_Z, test_dataT_Z, train_labelsT, test_labelsT)
                    xtemp_pfms21[n, nv2, nv1] = f_decoding.LDAPerformance(test_dataT_Z, train_dataT_Z, test_labelsT, train_labelsT)
                   
                    
                   
                    ### 2d
                    train_dataT_2d, test_dataT_2d = proj_2D_coordinates(train_dataT, np.concatenate(vecs[region][vcp1][tt][1][n])).T, proj_2D_coordinates(test_dataT, np.concatenate(vecs[region][vcp2][tt][2][n])).T
                    train_dataT_2d, test_dataT_2d = (train_dataT_2d - train_dataT_2d.mean(axis=0)), (test_dataT_2d - test_dataT_2d.mean(axis=0))
                    
                    train_dataT_2d_mean, test_dataT_2d_mean = proj_2D_coordinates(train_dataT_mean, np.concatenate(vecs[region][vcp1][tt][1][n])).T, proj_2D_coordinates(test_dataT_mean, np.concatenate(vecs[region][vcp2][tt][2][n])).T
                    train_dataT_2d_mean, test_dataT_2d_mean = (train_dataT_2d_mean - train_dataT_2d_mean.mean(axis=0)), (test_dataT_2d_mean - test_dataT_2d_mean.mean(axis=0))
                    
                    # rotate to the minimal Forb. Norm in conditional avg configs
                    R_, _ = scipy.linalg.orthogonal_procrustes(train_dataT_2d_mean, test_dataT_2d_mean)
                    test_dataT_2dR = np.dot(test_dataT_2d, R_.T)
                    
                    # standard scaling
                    train_dataT_2d_Z, test_dataT_2dR_Z = train_dataT_2d.copy(), test_dataT_2dR.copy()
                    for ch in range(2):
                        train_dataT_2d_Z[:,ch] = (train_dataT_2d[:,ch] - train_dataT_2d[:,ch].mean())/train_dataT_2d[:,ch].std()
                        test_dataT_2dR_Z[:,ch] = (test_dataT_2dR[:,ch] - test_dataT_2dR[:,ch].mean())/test_dataT_2dR[:,ch].std()
                        
                    xtemp_pfms12_2d[n, nv1, nv2] = f_decoding.LDAPerformance(train_dataT_2d_Z, test_dataT_2dR_Z, train_labelsT, test_labelsT)
                    xtemp_pfms21_2d[n, nv2, nv1] = f_decoding.LDAPerformance(test_dataT_2dR_Z, train_dataT_2d_Z, test_labelsT, train_labelsT)
                    
                    #permutation p value checks
                    for npm in range(50): #nPerms
                        train_labelsT_shuff, test_labelsT_shuff = np.random.permutation(pseudo_TrialInfoT.loc1), np.random.permutation(pseudo_TrialInfoT.loc2)
                        
                        xtemp_pfms12_shuff[n, npm, nv1, nv2] = f_decoding.LDAPerformance(train_dataT_Z, test_dataT_Z, train_labelsT_shuff, test_labelsT_shuff)
                        xtemp_pfms21_shuff[n, npm, nv2, nv1] = f_decoding.LDAPerformance(test_dataT_Z, train_dataT_Z, test_labelsT_shuff, train_labelsT_shuff)
                        
                        xtemp_pfms12_2d_shuff[n, npm, nv1, nv2] = f_decoding.LDAPerformance(train_dataT_2d_Z, test_dataT_2dR_Z, train_labelsT_shuff, test_labelsT_shuff)
                        xtemp_pfms21_2d_shuff[n, npm, nv2, nv1] = f_decoding.LDAPerformance(test_dataT_2dR_Z, train_dataT_2d_Z, test_labelsT_shuff, train_labelsT_shuff)
        
        ### pvalues
        pPfms12 = np.zeros((len(validCps1), len(validCps2)))
        pPfms21 = np.zeros((len(validCps2), len(validCps1)))
        
        pPfms12_2d = np.zeros((len(validCps1), len(validCps2)))
        pPfms21_2d = np.zeros((len(validCps2), len(validCps1)))
        
        
        for nv1, _ in enumerate(validCps1):
            for nv2, _ in enumerate(validCps2):
                pPfms12[nv1, nv2] = f_stats.permutation_p(xtemp_pfms12.mean(axis=0)[nv1, nv2], np.concatenate(xtemp_pfms12_shuff,axis=0)[:, nv1, nv2], tail = 'greater')
                pPfms21[nv2, nv1] = f_stats.permutation_p(xtemp_pfms21.mean(axis=0)[nv2, nv1], np.concatenate(xtemp_pfms21_shuff,axis=0)[:, nv2, nv1], tail = 'greater')
                
                pPfms12_2d[nv1, nv2] = f_stats.permutation_p(xtemp_pfms12_2d.mean(axis=0)[nv1, nv2], np.concatenate(xtemp_pfms12_2d_shuff,axis=0)[:, nv1, nv2], tail = 'greater')
                pPfms21_2d[nv2, nv1] = f_stats.permutation_p(xtemp_pfms21_2d.mean(axis=0)[nv2, nv1], np.concatenate(xtemp_pfms21_2d_shuff,axis=0)[:, nv2, nv1], tail = 'greater')
        
        
        ### plot        
        #plt.figure(figsize=(10,10), dpi = 100)
        plt.figure(figsize=(10,4), dpi = 100)
        ### 3d - 12
        #plt.subplot(2,2,1)
        plt.subplot(1,2,1)
        ax = plt.gca()
        im = ax.imshow(xtemp_pfms12.mean(axis=0).T, cmap='Blues_r', aspect='auto', vmin = 0.1, vmax = 0.6) #
        for i in range(len(validCps1)):
            for j in range(len(validCps2)):
                if 0.05 < pPfms12[i,j] <= 0.1:
                    text = ax.text(i, j, '+', ha="center", va="center", fontsize=6) #, color = 'yellow'
                elif 0.01 < pPfms12[i,j] <= 0.05:
                    text = ax.text(i, j, '*', ha="center", va="center", fontsize=6) #, color = 'yellow'
                elif pPfms12[i,j] <= 0.01:
                    text = ax.text(i, j, '**', ha="center", va="center", fontsize=6) #, color = 'yellow'
                    
        ax.set_xticks([n for n in range(len(validCps1))])
        ax.set_xticklabels(validCps1, fontsize = 6)
        ax.set_yticks([n for n in range(len(validCps2))])
        ax.set_yticklabels(validCps2, fontsize = 6)
        ax.set_xlabel('Train Loc1')
        ax.set_ylabel('Test Loc2')
        
        #ax.invert_yaxis()
        ax.set_title('3d Projs')
        plt.colorbar(im,ax=ax)
        
        
        ### 2d - 12
        #plt.subplot(2,2,2)
        #ax = plt.gca()
        #im = ax.imshow(xtemp_pfms12_2d.mean(axis=0).T, cmap='Blues_r', aspect='auto', vmin = 0.1, vmax = 0.6) #
        #for i in range(len(validCps1)):
        #    for j in range(len(validCps2)):
        #        if 0.05 < pPfms12_2d[i,j] <= 0.1:
        #            text = ax.text(i, j, '+', ha="center", va="center", fontsize=6) #, color = 'yellow'
        #        elif 0.01 < pPfms12_2d[i,j] <= 0.05:
        #            text = ax.text(i, j, '*', ha="center", va="center", fontsize=6) #, color = 'yellow'
        #        elif pPfms12_2d[i,j] <= 0.01:
        #            text = ax.text(i, j, '**', ha="center", va="center", fontsize=6) #, color = 'yellow'
                    
        #ax.set_xticks([n for n in range(len(validCps1))])
        #ax.set_xticklabels(validCps1, fontsize = 6)
        #ax.set_yticks([n for n in range(len(validCps2))])
        #ax.set_yticklabels(validCps2, fontsize = 6)
        #ax.set_xlabel('Train Loc1')
        #ax.set_ylabel('Test Loc2')
        
        #ax.invert_yaxis()
        #ax.set_title('2d Projs')
        #plt.colorbar(im,ax=ax)
        
        
        ### 3d - 21
        #plt.subplot(2,2,3)
        plt.subplot(1,2,2)
        ax = plt.gca()
        im = ax.imshow(xtemp_pfms21.mean(axis=0).T, cmap='Blues_r', aspect='auto', vmin = 0.1, vmax = 0.6) #
        for i in range(len(validCps2)):
            for j in range(len(validCps1)):
                if 0.05 < pPfms21[i,j] <= 0.1:
                    text = ax.text(i, j, '+', ha="center", va="center", fontsize=6) #, color = 'yellow'
                elif 0.01 < pPfms21[i,j] <= 0.05:
                    text = ax.text(i, j, '*', ha="center", va="center", fontsize=6) #, color = 'yellow'
                elif pPfms21[i,j] <= 0.01:
                    text = ax.text(i, j, '**', ha="center", va="center", fontsize=6) #, color = 'yellow'
                    
        ax.set_xticks([n for n in range(len(validCps2))])
        ax.set_xticklabels(validCps2, fontsize = 6)
        ax.set_yticks([n for n in range(len(validCps1))])
        ax.set_yticklabels(validCps1, fontsize = 6)
        ax.set_xlabel('Train Loc2')
        ax.set_ylabel('Test Loc1')
        
        #ax.invert_yaxis()
        ax.set_title('3d Projs')
        plt.colorbar(im,ax=ax)
        
        
        ### 2d - 21
        #plt.subplot(2,2,4)
        #ax = plt.gca()
        #im = ax.imshow(xtemp_pfms21_2d.mean(axis=0).T, cmap='Blues_r', aspect='auto', vmin = 0.1, vmax = 0.6) #
        #for i in range(len(validCps2)):
        #    for j in range(len(validCps1)):
        #        if 0.05 < pPfms21_2d[i,j] <= 0.1:
        #            text = ax.text(i, j, '+', ha="center", va="center", fontsize=6) #, color = 'yellow'
        #        elif 0.01 < pPfms21_2d[i,j] <= 0.05:
        #            text = ax.text(i, j, '*', ha="center", va="center", fontsize=6) #, color = 'yellow'
        #        elif pPfms21_2d[i,j] <= 0.01:
        #            text = ax.text(i, j, '**', ha="center", va="center", fontsize=6) #, color = 'yellow'
                    
        #ax.set_xticks([n for n in range(len(validCps2))])
        #ax.set_xticklabels(validCps1, fontsize = 6)
        #ax.set_yticks([n for n in range(len(validCps1))])
        #ax.set_yticklabels(validCps1, fontsize = 6)
        #ax.set_xlabel('Train Loc2')
        #ax.set_ylabel('Test Loc1')
        
        #ax.invert_yaxis()
        #ax.set_title('2d Projs')
        #plt.colorbar(im,ax=ax)
        
        
        
        plt.suptitle(f'{region}, ttype={tt}. retarget, decodability Train1-Test2')
        plt.show()




# In[] test plane replacement: project d2i2 on d1i1

validCheckpoints = {'dlpfc': {'d1':(150, 300, 500, 700, 900, 1100, 1300), 'd2':(1450, 1600, 1800, 2000, 2200, 2400, 2600)}, 
                    'fef': {'d1':(150, 300, 500, 700, 900, 1100, 1300), 'd2':(1450, 1600, 1800, 2000, 2200, 2400, 2600)}} # 


for region in ('dlpfc','fef'):
    
    validCps1 = validCheckpoints[region]['d1'] #checkpoints
    validCps2 = validCheckpoints[region]['d2'] #checkpoints
    
    
    # just apply to retarget trials
    for tt in ttypes: #(1,)
        
        ### 3d projs
        xtemp_pfms = np.zeros((nIters,len(validCps1), len(validCps2)))
        xtemp_pfms_shuff = np.zeros((nIters, nPerms,len(validCps1), len(validCps2)))
        
        ### 2d projs
        xtemp_pfms2d = np.zeros((nIters,len(validCps1), len(validCps2)))
        xtemp_pfms2d_shuff = np.zeros((nIters, nPerms,len(validCps1), len(validCps2)))
        
        
        for n in range(nIters):
            pseudo_TrialInfoT = np.load(save_path + f'/pseudo_all{n}.npy', allow_pickle=True).item()['pseudo_TrialInfo']
            pseudo_TrialInfoT = pseudo_TrialInfoT[pseudo_TrialInfoT.type == tt].reset_index(drop=True)
            
            for nv1, vcp1 in enumerate(validCps1):
                for nv2, vcp2 in enumerate(validCps2):
                    train_labelsT, test_labelsT = pseudo_TrialInfoT.loc1, pseudo_TrialInfoT.loc2
                    
                    ### 3d
                    train_dataT, test_dataT = np.concatenate(projsAll[region][vcp1][tt][1][n]), np.concatenate(projsAll[region][vcp2][tt][2][n])
                    train_dataT_mean, test_dataT_mean = np.concatenate(projs[region][vcp1][tt][1][n]), np.concatenate(projs[region][vcp2][tt][2][n])
                    
                    # project 2 on 1
                    train_normalT, test_normalT = np.concatenate(vecs_normal[region][vcp1][tt][1][n]), np.concatenate(vecs_normal[region][vcp2][tt][2][n])
                    train_centerT, test_centerT = train_dataT_mean.mean(axis=0), test_dataT_mean.mean(axis=0)
                    
                    test_dataT = np.array([proj_on_plane(p, train_normalT, train_centerT) for p in test_dataT]) # project d2i2 projs on d1i1 plane
                    #train_dataT = np.array([proj_on_plane(p, test_normalT, test_centerT) for p in train_dataT]) # vice versa
                    
                    # standard scaling
                    train_dataT_Z, test_dataT_Z = train_dataT.copy(), test_dataT.copy()
                    for ch in range(3):
                        train_dataT[:,ch] = (train_dataT[:,ch] - train_dataT[:,ch].mean())/train_dataT[:,ch].std()
                        test_dataT[:,ch] = (test_dataT[:,ch] - test_dataT[:,ch].mean())/test_dataT[:,ch].std()
                        
                    xtemp_pfms[n, nv1, nv2] = f_decoding.LDAPerformance(train_dataT_Z, test_dataT_Z, train_labelsT, test_labelsT)
                    
                    ### 2d
                    train_dataT_2d, test_dataT_2d = proj_2D_coordinates(train_dataT, np.concatenate(vecs[region][vcp1][tt][1][n])).T, proj_2D_coordinates(test_dataT, np.concatenate(vecs[region][vcp1][tt][1][n])).T # project d2i2 projs on d1i1 plane
                    #train_dataT_2d, test_dataT_2d = proj_2D_coordinates(train_dataT, np.concatenate(vecs[region][vcp2][tt][2][n])).T, proj_2D_coordinates(test_dataT, np.concatenate(vecs[region][vcp2][tt][2][n])).T # vice versa
                    train_dataT_2d, test_dataT_2d = (train_dataT_2d - train_dataT_2d.mean(axis=0)), (test_dataT_2d - test_dataT_2d.mean(axis=0))
                    
                    #train_dataT_2d_mean, test_dataT_2d_mean = proj_2D_coordinates(train_dataT_mean, np.concatenate(vecs[region][vcp1][tt][1][n])).T, proj_2D_coordinates(test_dataT_mean, np.concatenate(vecs[region][vcp2][tt][2][n])).T
                    #train_dataT_2d_mean, test_dataT_2d_mean = proj_2D_coordinates(train_dataT_mean, np.concatenate(vecs[region][vcp1][tt][1][n])).T, proj_2D_coordinates(test_dataT_mean, np.concatenate(vecs[region][vcp2][tt][2][n])).T
                    #train_dataT_2d_mean, test_dataT_2d_mean = (train_dataT_2d_mean - train_dataT_2d_mean.mean(axis=0)), (test_dataT_2d_mean - test_dataT_2d_mean.mean(axis=0))
                    
                    # rotate to the minimal Forb. Norm in conditional avg configs
                    #R_, _ = scipy.linalg.orthogonal_procrustes(train_dataT_2d_mean, test_dataT_2d_mean)
                    #test_dataT_2dR = np.dot(test_dataT_2d, R_.T)
                    test_dataT_2dR = test_dataT_2d
                    
                    # standard scaling
                    train_dataT_2d_Z, test_dataT_2dR_Z = train_dataT_2d.copy(), test_dataT_2dR.copy()
                    for ch in range(2):
                        train_dataT_2d_Z[:,ch] = (train_dataT_2d[:,ch] - train_dataT_2d[:,ch].mean())/train_dataT_2d[:,ch].std()
                        test_dataT_2dR_Z[:,ch] = (test_dataT_2dR[:,ch] - test_dataT_2dR[:,ch].mean())/test_dataT_2dR[:,ch].std()
                        
                    xtemp_pfms2d[n, nv1, nv2] = f_decoding.LDAPerformance(train_dataT_2d_Z, test_dataT_2dR_Z, train_labelsT, test_labelsT)
                    
                    for npm in range(50): #nPerms
                        train_labelsT_shuff, test_labelsT_shuff = np.random.permutation(pseudo_TrialInfoT.loc1), np.random.permutation(pseudo_TrialInfoT.loc2)
                        xtemp_pfms_shuff[n, npm, nv1, nv2] = f_decoding.LDAPerformance(train_dataT_Z, test_dataT_Z, train_labelsT_shuff, test_labelsT_shuff)
                        xtemp_pfms2d_shuff[n, npm, nv1, nv2] = f_decoding.LDAPerformance(train_dataT_2d_Z, test_dataT_2dR_Z, train_labelsT_shuff, test_labelsT_shuff)
        
        
        ### pvalues
        pPfms = np.zeros((len(validCps1), len(validCps2)))
        pPfms2d = np.zeros((len(validCps1), len(validCps2)))
        for nv1, _ in enumerate(validCps1):
            for nv2, _ in enumerate(validCps2):
                pPfms[nv1, nv2] = f_stats.permutation_p(xtemp_pfms.mean(axis=0)[nv1, nv2], np.concatenate(xtemp_pfms_shuff,axis=0)[:, nv1, nv2], tail = 'greater')
                pPfms2d[nv1, nv2] = f_stats.permutation_p(xtemp_pfms2d.mean(axis=0)[nv1, nv2], np.concatenate(xtemp_pfms2d_shuff,axis=0)[:, nv1, nv2], tail = 'greater')
        
        
        
        ### plot        
        plt.figure(figsize=(10,4), dpi = 100)
        ### 3d
        plt.subplot(1,2,1)
        ax = plt.gca()
        im = ax.imshow(xtemp_pfms.mean(axis=0).T, cmap='Blues_r', aspect='auto', vmin = 0.1, vmax = 0.6)
        for i in range(len(validCps1)):
            for j in range(len(validCps2)):
                if 0.05 < pPfms[i,j] <= 0.1:
                    text = ax.text(i, j, '+', ha="center", va="center", fontsize=6) #, color = 'yellow'
                elif 0.01 < pPfms[i,j] <= 0.05:
                    text = ax.text(i, j, '*', ha="center", va="center", fontsize=6) #, color = 'yellow'
                elif pPfms[i,j] <= 0.01:
                    text = ax.text(i, j, '**', ha="center", va="center", fontsize=6) #, color = 'yellow'
                    
        ax.set_xticks([n for n in range(len(validCps1))])
        ax.set_xticklabels(validCps1, fontsize = 6)
        ax.set_yticks([n for n in range(len(validCps2))])
        ax.set_yticklabels(validCps2, fontsize = 6)
        ax.set_xlabel('Train Loc1')
        ax.set_ylabel('Test Loc2')
        
        #ax.invert_yaxis()
        ax.set_title('3d Projs')
        plt.colorbar(im,ax=ax)
        
        
        ### 2d
        plt.subplot(1,2,2)
        ax = plt.gca()
        im = ax.imshow(xtemp_pfms2d.mean(axis=0).T, cmap='Blues_r', aspect='auto', vmin = 0.1, vmax = 0.6)
        for i in range(len(validCps1)):
            for j in range(len(validCps2)):
                if 0.05 < pPfms2d[i,j] <= 0.1:
                    text = ax.text(i, j, '+', ha="center", va="center", fontsize=6) #, color = 'yellow'
                elif 0.01 < pPfms2d[i,j] <= 0.05:
                    text = ax.text(i, j, '*', ha="center", va="center", fontsize=6) #, color = 'yellow'
                elif pPfms2d[i,j] <= 0.01:
                    text = ax.text(i, j, '**', ha="center", va="center", fontsize=6) #, color = 'yellow'
                    
        ax.set_xticks([n for n in range(len(validCps1))])
        ax.set_xticklabels(validCps1, fontsize = 6)
        ax.set_yticks([n for n in range(len(validCps2))])
        ax.set_yticklabels(validCps2, fontsize = 6)
        ax.set_xlabel('Train Loc1')
        ax.set_ylabel('Test Loc2')
        
        #ax.invert_yaxis()
        ax.set_title('2d Projs')
        plt.colorbar(im,ax=ax)
        
        
        plt.suptitle(f'{region}, retarget, decodability Train1-Test2')
        plt.show()

# In[] test plane replacement: lda decode sequence (i1 or i2)

validCheckpoints = {'dlpfc': {'d1':(150, 300, 500, 700, 900, 1100, 1300), 'd2':(1450, 1600, 1800, 2000, 2200, 2400, 2600)}, 
                    'fef': {'d1':(150, 300, 500, 700, 900, 1100, 1300), 'd2':(1450, 1600, 1800, 2000, 2200, 2400, 2600)}} # 


for region in ('dlpfc','fef'):
    
    validCps1 = validCheckpoints[region]['d1'] #checkpoints
    validCps2 = validCheckpoints[region]['d2'] #checkpoints
    
    for tt in ttypes: # (1,)
        
        ### 3d projs
        xtemp_pfms = np.zeros((nIters, len(validCps1), len(validCps2)))
        xtemp_pfms_mean = np.zeros((nIters, len(validCps1), len(validCps2)))
        xtemp_pfms_shuff = np.zeros((nIters, nPerms, len(validCps1), len(validCps2)))
        xtemp_pfms_mean_shuff = np.zeros((nIters, nPerms, len(validCps1), len(validCps2)))
        
        for n in range(nIters):
            
            for nv1, vcp1 in enumerate(validCps1):
                for nv2, vcp2 in enumerate(validCps2):
                    
                    item1T, item2T = np.concatenate(projsAll[region][vcp1][tt][1][n]), np.concatenate(projsAll[region][vcp2][tt][2][n])
                    item1T_mean, item2T_mean = np.concatenate(projs[region][vcp1][tt][1][n]), np.concatenate(projs[region][vcp2][tt][2][n])
                    
                    label1T, label2T = np.full((item1T.shape[0]), '1'), np.full((item2T.shape[0]), '2')
                    label1T_mean, label2T_mean = np.full((item1T_mean.shape[0]), '1'), np.full((item2T_mean.shape[0]), '2')
                    
                    # standard scaling
                    item1T_Z, item2T_Z = item1T.copy(), item2T.copy()
                    item1T_Z_mean, item2T_Z_mean = item1T_mean.copy(), item2T_mean.copy()
                    for ch in range(3):
                        item1T_Z[:,ch] = (item1T[:,ch] - item1T[:,ch].mean())/item1T[:,ch].std()
                        item2T_Z[:,ch] = (item2T[:,ch] - item2T[:,ch].mean())/item2T[:,ch].std()
                        item1T_Z_mean[:,ch] = (item1T_mean[:,ch] - item1T_mean[:,ch].mean())/item1T_mean[:,ch].std()
                        item2T_Z_mean[:,ch] = (item2T_mean[:,ch] - item2T_mean[:,ch].mean())/item2T_mean[:,ch].std()
                    
                    
                    itemsFull_T = np.concatenate((item1T_Z, item2T_Z), axis=0)
                    labelsFull_T = np.concatenate((label1T, label2T), axis=0)
                    itemsFull_T_mean = np.concatenate((item1T_Z_mean, item2T_Z_mean), axis=0)
                    labelsFull_T_mean = np.concatenate((label1T_mean, label2T_mean), axis=0)
                    
                    
                    ## all projs
                    train_IDT, test_IDT = split_set(itemsFull_T,frac = 0.5)
                    train_setT, test_setT = itemsFull_T[train_IDT,:], itemsFull_T[test_IDT,:]
                    train_labelsT, test_labelsT = labelsFull_T[train_IDT], labelsFull_T[test_IDT]
                    
                    xtemp_pfms[n, nv1, nv2] = f_decoding.LDAPerformance(train_setT, test_setT, train_labelsT, test_labelsT)
                    
                    
                    ## mean projs
                    train_ID_meanT, test_ID_meanT = split_set(itemsFull_T_mean,frac = 0.5)
                    train_set_meanT, test_set_meanT = itemsFull_T_mean[train_ID_meanT,:], itemsFull_T_mean[test_ID_meanT,:]
                    train_labels_meanT, test_labels_meanT = labelsFull_T_mean[train_ID_meanT], labelsFull_T_mean[test_ID_meanT]
                    
                    xtemp_pfms_mean[n, nv1, nv2] = f_decoding.LDAPerformance(train_set_meanT, test_set_meanT, train_labels_meanT, test_labels_meanT)
                    
                    
                    ## permutation p values
                    for npm in range(50): #nPerms
                        train_labelsT_shuff, test_labelsT_shuff = np.random.permutation(train_labelsT), np.random.permutation(test_labelsT)
                        train_labels_meanT_shuff, test_labels_meanT_shuff = np.random.permutation(train_labels_meanT), np.random.permutation(test_labels_meanT)
                        
                        xtemp_pfms_shuff[n, npm, nv1, nv2] = f_decoding.LDAPerformance(train_setT, test_setT, train_labelsT_shuff, test_labelsT_shuff)
                        xtemp_pfms_mean_shuff[n, npm, nv1, nv2] = f_decoding.LDAPerformance(train_set_meanT, test_set_meanT, train_labels_meanT_shuff, test_labels_meanT_shuff)
                    
                        
        ### pvalues
        pPfms = np.zeros((len(validCps1), len(validCps2)))
        pPfms_mean = np.zeros((len(validCps1), len(validCps2)))
        
        
        for nv1, _ in enumerate(validCps1):
            for nv2, _ in enumerate(validCps2):
                pPfms[nv1, nv2] = f_stats.permutation_p(xtemp_pfms.mean(axis=0)[nv1, nv2], np.concatenate(xtemp_pfms_shuff,axis=0)[:, nv1, nv2], tail = 'greater')
                pPfms_mean[nv1, nv2] = f_stats.permutation_p(xtemp_pfms_mean.mean(axis=0)[nv1, nv2], np.concatenate(xtemp_pfms_mean_shuff,axis=0)[:, nv1, nv2], tail = 'greater')
                
        
        ### plot        
        plt.figure(figsize=(10,4), dpi = 100)
        ### 3d - 12
        plt.subplot(1,2,1)
        ax = plt.gca()
        im = ax.imshow(xtemp_pfms.mean(axis=0).T, cmap='Blues_r', aspect='auto', vmin = 0.1, vmax = 0.8) #
        for i in range(len(validCps1)):
            for j in range(len(validCps2)):
                if 0.05 < pPfms[i,j] <= 0.1:
                    text = ax.text(i, j, '+', ha="center", va="center", fontsize=6) #, color = 'yellow'
                elif 0.01 < pPfms[i,j] <= 0.05:
                    text = ax.text(i, j, '*', ha="center", va="center", fontsize=6) #, color = 'yellow'
                elif pPfms[i,j] <= 0.01:
                    text = ax.text(i, j, '**', ha="center", va="center", fontsize=6) #, color = 'yellow'
                    
        ax.set_xticks([n for n in range(len(validCps1))])
        ax.set_xticklabels(validCps1, fontsize = 6)
        ax.set_yticks([n for n in range(len(validCps2))])
        ax.set_yticklabels(validCps2, fontsize = 6)
        ax.set_xlabel('D1')
        ax.set_ylabel('D2')
        
        #ax.invert_yaxis()
        ax.set_title('Projs All')
        plt.colorbar(im,ax=ax)
        
        
        ### 2d - 12
        plt.subplot(1,2,2)
        ax = plt.gca()
        im = ax.imshow(xtemp_pfms_mean.mean(axis=0).T, cmap='Blues_r', aspect='auto', vmin = 0.1, vmax = 0.8) #
        for i in range(len(validCps1)):
            for j in range(len(validCps2)):
                if 0.05 < pPfms_mean[i,j] <= 0.1:
                    text = ax.text(i, j, '+', ha="center", va="center", fontsize=6) #, color = 'yellow'
                elif 0.01 < pPfms_mean[i,j] <= 0.05:
                    text = ax.text(i, j, '*', ha="center", va="center", fontsize=6) #, color = 'yellow'
                elif pPfms_mean[i,j] <= 0.01:
                    text = ax.text(i, j, '**', ha="center", va="center", fontsize=6) #, color = 'yellow'
                    
        ax.set_xticks([n for n in range(len(validCps1))])
        ax.set_xticklabels(validCps1, fontsize = 6)
        ax.set_yticks([n for n in range(len(validCps2))])
        ax.set_yticklabels(validCps2, fontsize = 6)
        ax.set_xlabel('D1')
        ax.set_ylabel('D2')
        
        #ax.invert_yaxis()
        ax.set_title('Projs Mean')
        plt.colorbar(im,ax=ax)
        
        
        
        plt.suptitle(f'{region}, ttype={tt}, retarget, decodability of Sequence')
        plt.show()
    



    












######### Test plane by distance measure
# In[] condition mean eu-distance between plane centers

validCheckpoints = {'dlpfc': {'d1':(150, 300, 500, 700, 900, 1100, 1300), 'd2':(1450, 1600, 1800, 2000, 2200, 2400, 2600)}, 
                    'fef': {'d1':(150, 300, 500, 700, 900, 1100, 1300), 'd2':(1450, 1600, 1800, 2000, 2200, 2400, 2600)}} # 


for region in ('dlpfc','fef'):
    
    validCps1 =  validCheckpoints[region]['d1'] #checkpoints #
    validCps2 =  validCheckpoints[region]['d2'] #checkpoints #
    
    # just apply to retarget trials
    for tt in ttypes: #(1,)
        
        xtemp_dist = np.zeros((nIters,len(validCps1), len(validCps2)))
        
        for n in range(nIters):
            pointsT_mean = [] 
            #pointsT_all = []
            
            for ncp, cp in enumerate(checkpoints): #validCps1
                for tt_ in (1,2,):
                    pointsT_mean += [np.concatenate((np.concatenate(projs[region][cp][tt_][1][n]), np.concatenate(projs[region][cp][tt_][2][n])))]
                    #pointsT_all += [np.concatenate((np.concatenate(projsAll[region][cp][tt_][1][n]), np.concatenate(projsAll[region][cp][tt_][2][n])))]
            
            
            pointsT_mean = np.concatenate(pointsT_mean)
            meanT_mean, stdT_mean = pointsT_mean.mean(axis=0), pointsT_mean.std(axis=0)
            #pointsT_all = np.concatenate(pointsT_all)
            #meanT_all, stdT_all = pointsT_all.mean(axis=0), pointsT_all.std(axis=0)
            
            for nv1, vcp1 in enumerate(validCps1):
                for nv2, vcp2 in enumerate(validCps2):
                    center1T = np.concatenate(projs[region][vcp1][tt][1][n]).mean(axis=0) 
                    center2T = np.concatenate(projs[region][vcp2][tt][2][n]).mean(axis=0) if tt == 1 else np.concatenate(projs[region][vcp2][tt][1][n]).mean(axis=0)
                    center1Z, center2Z = (center1T - meanT_mean)/stdT_mean, (center2T - meanT_mean)/stdT_mean
                    
                    xtemp_dist[n, nv1, nv2] = euclidean_distance(center1Z, center2Z)
        
        # shuffles
        xtemp_dist_shuff = np.zeros((nIters, nPerms,len(validCps1), len(validCps2)))
        
        for n in range(nIters):
            for npm in range(nPerms):
                pointsT_mean_shuff = [] 
                
                for ncp, cp in enumerate(checkpoints): #validCps1
                    for tt_ in (1,2,):
                        pointsT_mean_shuff += [np.concatenate((projs_shuff[region][cp][tt_][1][n][npm], projs_shuff[region][cp][tt_][2][n][npm]))]
                        #pointsT_all_shuff += [np.concatenate((np.concatenate(projsAll_shuff[region][cp][tt_][1][n]), np.concatenate(projsAll_shuff[region][cp][tt_][2][n])))]
                
                pointsT_mean_shuff = np.concatenate(pointsT_mean_shuff)
                meanT_mean_shuff, stdT_mean_shuff = pointsT_mean_shuff.mean(axis=0), pointsT_mean_shuff.std(axis=0)
                
                for nv1, vcp1 in enumerate(validCps1):
                    for nv2, vcp2 in enumerate(validCps2):
                        center1T_shuff = projs_shuff[region][vcp1][tt][1][n][npm].mean(axis=0)
                        center2T_shuff = projs_shuff[region][vcp2][tt][2][n][npm].mean(axis=0) if tt == 1 else projs_shuff[region][vcp2][tt][1][n][npm].mean(axis=0)
                        center1Z_shuff, center2Z_shuff = (center1T_shuff - meanT_mean_shuff)/stdT_mean_shuff, (center2T_shuff - meanT_mean_shuff)/stdT_mean_shuff
                        
                        xtemp_dist_shuff[n, npm, nv1, nv2] = euclidean_distance(center1Z_shuff, center2Z_shuff)
        
        #xtemp_distZ = (xtemp_dist - xtemp_dist_shuff.mean(axis=1)) / xtemp_dist_shuff.std(axis=1)
        
        #xtemp_dist_shuffZ = []
        #for n in range(nIters):
        #    xtemp_dist_shuffZ += [(xtemp_dist_shuff[n] - xtemp_dist_shuff[n].mean(axis=0)) / xtemp_dist_shuff[n].std(axis=0)]
            
        #xtemp_dist_shuffZ = np.array(xtemp_dist_shuffZ)
        
        # xtemp_dist_shuffZ = (xtemp_dist_shuff - xtemp_dist_shuff.mean(axis=1)) / xtemp_dist_shuff.mean(axis=1)
        
        pDists = np.zeros((len(validCps1), len(validCps2)))
        for nv1, _ in enumerate(validCps1):
            for nv2, _ in enumerate(validCps2):
                pDists[nv1, nv2] = f_stats.permutation_p(xtemp_dist.round(5).mean(axis=0)[nv1, nv2], np.concatenate(xtemp_dist_shuff,axis=0).round(5)[:, nv1, nv2], tail = 'greater')
                #pDists[nv1, nv2] = f_stats.permutation_p(xtemp_distZ.round(5).mean(axis=0)[nv1, nv2], np.concatenate(xtemp_dist_shuffZ,axis=0).round(5)[:, nv1, nv2], tail = 'two')
        
        ### plot        
        plt.figure(figsize=(5,4), dpi = 100)
        ### 3d
        plt.subplot(1,1,1)
        ax = plt.gca()
        im = ax.imshow(xtemp_dist.mean(axis=0).T, cmap='Blues_r', aspect='auto', vmin = 0, vmax = 5)#
        #im = ax.imshow(xtemp_distZ.mean(axis=0).T, cmap='Blues_r', aspect='auto', vmin = 0, vmax = 5)#
        for i in range(len(validCps1)):
            for j in range(len(validCps2)):
                if 0.05 < pDists[i,j] <= 0.1:
                    text = ax.text(i, j, '+', ha="center", va="center", fontsize=6)
                elif 0.01 < pDists[i,j] <= 0.05:
                    text = ax.text(i, j, '*', ha="center", va="center", fontsize=6)
                elif pDists[i,j] <= 0.01:
                    text = ax.text(i, j, '**', ha="center", va="center", fontsize=6)
                    
        ax.set_xticks([n for n in range(len(validCps1))])
        ax.set_xticklabels(validCps1, fontsize = 6)
        ax.set_yticks([n for n in range(len(validCps2))])
        ax.set_yticklabels(validCps2, fontsize = 6)
        ax.set_xlabel('Time (Item1)')
        ax.set_ylabel('Time (Item2)')
        
        #ax.invert_yaxis()
        #ax.set_title('3d Projs')
        plt.colorbar(im,ax=ax)
        
        plt.suptitle(f'{region}, retarget, Z_EuDist Item1-Item2, ttype = {tt}')
        plt.show()
        

# In[] test plane replacement: project d2i2 on d1i1 pairwise distance
### add significance measure
validCheckpoints = {'dlpfc': {'d1':(150, 300, 500, 700, 900, 1100, 1300), 'd2':(1450, 1600, 1800, 2000, 2200, 2400, 2600)}, 
                    'fef': {'d1':(150, 300, 500, 700, 900, 1100, 1300), 'd2':(1450, 1600, 1800, 2000, 2200, 2400, 2600)}} # 


for region in ('dlpfc','fef'):
    
    validCps1 = validCheckpoints[region]['d1'] #checkpoints
    validCps2 = validCheckpoints[region]['d2'] #checkpoints
    
    
    # just apply to retarget trials
    for tt in (1,):
        
        ### 3d projs
        xtemp_dist = np.zeros((nIters,len(validCps1), len(validCps2)))
        xtemp_dist_shuff = np.zeros((nIters, nPerms,len(validCps1), len(validCps2)))
        
        ### 2d projs
        xtemp_dist2d = np.zeros((nIters,len(validCps1), len(validCps2)))
        xtemp_dist2d_shuff = np.zeros((nIters, nPerms,len(validCps1), len(validCps2)))
        
        
        for n in range(nIters):
            pseudo_TrialInfoT = np.load(save_path + f'/pseudo_all{n}.npy', allow_pickle=True).item()['pseudo_TrialInfo']
            pseudo_TrialInfoT = pseudo_TrialInfoT[pseudo_TrialInfoT.type == tt].reset_index(drop=True)
            
            for nv1, vcp1 in enumerate(validCps1):
                for nv2, vcp2 in enumerate(validCps2):
                    train_labelsT, test_labelsT = pseudo_TrialInfoT.loc1, pseudo_TrialInfoT.loc2
                    
                    ### 3d
                    train_dataT, test_dataT = np.concatenate(projsAll[region][vcp1][tt][1][n]), np.concatenate(projsAll[region][vcp2][tt][2][n])
                    train_dataT_mean, test_dataT_mean = np.concatenate(projs[region][vcp1][tt][1][n]), np.concatenate(projs[region][vcp2][tt][2][n])
                    
                    # project 2 on 1
                    train_normalT, test_normalT = np.concatenate(vecs_normal[region][vcp1][tt][1][n]), np.concatenate(vecs_normal[region][vcp2][tt][2][n])
                    train_centerT, test_centerT = train_dataT_mean.mean(axis=0), test_dataT_mean.mean(axis=0)
                    
                    test_dataT = np.array([proj_on_plane(p, train_normalT, train_centerT) for p in test_dataT]) # project d2i2 projs on d1i1 plane
                    #train_dataT = np.array([proj_on_plane(p, test_normalT, test_centerT) for p in train_dataT]) # vice versa
                    
                        
                    xtemp_dist[n, nv1, nv2] = np.array([euclidean_distance(train_dataT[i,:], test_dataT[i,:]) for i in range(train_dataT.shape[0])]).mean()
                    
                    
                    ### 2d
                    train_dataT_2d, test_dataT_2d = proj_2D_coordinates(train_dataT, np.concatenate(vecs[region][vcp1][tt][1][n])).T, proj_2D_coordinates(test_dataT, np.concatenate(vecs[region][vcp1][tt][1][n])).T # project d2i2 projs on d1i1 plane
                    #train_dataT_2d, test_dataT_2d = proj_2D_coordinates(train_dataT, np.concatenate(vecs[region][vcp2][tt][2][n])).T, proj_2D_coordinates(test_dataT, np.concatenate(vecs[region][vcp2][tt][2][n])).T # vice versa
                    train_dataT_2d, test_dataT_2d = (train_dataT_2d - train_dataT_2d.mean(axis=0)), (test_dataT_2d - test_dataT_2d.mean(axis=0))
                    #test_dataT_2dR = test_dataT_2d
                    
                    xtemp_dist2d[n, nv1, nv2] = np.array([euclidean_distance(train_dataT_2d[i,:], test_dataT_2d[i,:]) for i in range(len(train_dataT_Z))]).mean()
                    
                    
                    for npm in range(50): #nPerms
                        ### 3d
                        train_data_shuffT, test_data_shuffT = projsAll_shuff[region][vcp1][tt][1][n][npm], projsAll_shuff[region][vcp2][tt][2][n][npm]
                        train_data_shuffT_mean, test_data_shuffT_mean = projs_shuff[region][vcp1][tt][1][n][npm], projs_shuff[region][vcp2][tt][2][n][npm]
                        
                        # project 2 on 1
                        train_normal_shuffT, test_normal_shuffT = vecs_normal_shuff[region][vcp1][tt][1][n][npm], vecs_normal_shuff[region][vcp2][tt][2][n][npm]
                        train_center_shuffT, test_center_shuffT = train_data_shuffT_mean.mean(axis=0), test_data_shuffT_mean.mean(axis=0)
                        
                        test_data_shuffT = np.array([proj_on_plane(p, train_normal_shuffT, train_center_shuffT) for p in test_data_shuffT]) # project d2i2 projs on d1i1 plane
                        #train_data_shuffT = np.array([proj_on_plane(p, test_normal_shuffT, test_center_shuffT) for p in train_data_shuffT]) # vice versa
                    
                        xtemp_dist_shuff[n, npm, nv1, nv2] = np.array([euclidean_distance(train_data_shuffT[i,:], test_data_shuffT[i,:]) for i in range(train_data_shuffT.shape[0])]).mean() 
                        
                        
                        ### 2d
                        train_data_shuffT_2d, test_data_shuffT_2d = proj_2D_coordinates(train_data_shuffT, vecs_shuff[region][vcp1][tt][1][n][npm]).T, proj_2D_coordinates(test_data_shuffT, vecs_shuff[region][vcp1][tt][1][n][npm]).T # project d2i2 projs on d1i1 plane
                        #train_dataT_2d, test_dataT_2d = proj_2D_coordinates(train_dataT, np.concatenate(vecs[region][vcp2][tt][2][n])).T, proj_2D_coordinates(test_dataT, np.concatenate(vecs[region][vcp2][tt][2][n])).T # vice versa
                        train_data_shuffT_2d, test_data_shuffT_2d = (train_data_shuffT_2d - train_data_shuffT_2d.mean(axis=0)), (test_data_shuffT_2d - test_data_shuffT_2d.mean(axis=0))
                        
                        #train_dataT_2d_mean, test_dataT_2d_mean = proj_2D_coordinates(train_dataT_mean, np.concatenate(vecs[region][vcp1][tt][1][n])).T, proj_2D_coordinates(test_dataT_mean, np.concatenate(vecs[region][vcp2][tt][2][n])).T
                        #train_dataT_2d_mean, test_dataT_2d_mean = proj_2D_coordinates(train_dataT_mean, np.concatenate(vecs[region][vcp1][tt][1][n])).T, proj_2D_coordinates(test_dataT_mean, np.concatenate(vecs[region][vcp2][tt][2][n])).T
                        #train_dataT_2d_mean, test_dataT_2d_mean = (train_dataT_2d_mean - train_dataT_2d_mean.mean(axis=0)), (test_dataT_2d_mean - test_dataT_2d_mean.mean(axis=0))
                        
                        # rotate to the minimal Forb. Norm in conditional avg configs
                        #R_, _ = scipy.linalg.orthogonal_procrustes(train_dataT_2d_mean, test_dataT_2d_mean)
                        #test_dataT_2dR = np.dot(test_dataT_2d, R_.T)
                        #test_data_shuffT_2dR = test_data_shuffT_2d
                        
                        xtemp_dist2d_shuff[n, npm, nv1, nv2] = np.array([euclidean_distance(train_data_shuffT_2d[i,:], test_data_shuffT_2d[i,:]) for i in range(train_data_shuffT_2d.shape[0])]).mean() 
        
                    
        # standardize distance sacle
        xtemp_distZ = (xtemp_dist - xtemp_dist_shuff.mean(axis=1)) / xtemp_dist_shuff.std(axis=1)
        xtemp_dist2dZ = (xtemp_dist2d - xtemp_dist2d_shuff.mean(axis=1)) / xtemp_dist2d_shuff.std(axis=1)
        
        xtemp_dist_shuffZ = []
        xtemp_dist2d_shuffZ = []
        for n in range(nIters):
            xtemp_dist_shuffZ += [(xtemp_dist_shuff[n] - xtemp_dist_shuff[n].mean(axis=0)) / xtemp_dist_shuff[n].std(axis=0)]
            xtemp_dist2d_shuffZ += [(xtemp_dist2d_shuff[n] - xtemp_dist2d_shuff[n].mean(axis=0)) / xtemp_dist2d_shuff[n].std(axis=0)]
        
        xtemp_dist_shuffZ = np.array(xtemp_dist_shuffZ)
        xtemp_dist2d_shuffZ = np.array(xtemp_dist2d_shuffZ)
                    
        
        ### pvalues
        pDist = np.zeros((len(validCps1), len(validCps2)))
        pDist2d = np.zeros((len(validCps1), len(validCps2)))
        for nv1, _ in enumerate(validCps1):
            for nv2, _ in enumerate(validCps2):
                pDist[nv1, nv2] = f_stats.permutation_p(xtemp_distZ.mean(axis=0)[nv1, nv2], np.concatenate(xtemp_dist_shuffZ,axis=0)[:, nv1, nv2], tail = 'greater')
                pDist2d[nv1, nv2] = f_stats.permutation_p(xtemp_dist2dZ.mean(axis=0)[nv1, nv2], np.concatenate(xtemp_dist2d_shuffZ,axis=0)[:, nv1, nv2], tail = 'greater')
        
        
        
        ### plot        
        plt.figure(figsize=(10,4), dpi = 100)
        ### 3d
        plt.subplot(1,2,1)
        ax = plt.gca()
        im = ax.imshow(xtemp_distZ.mean(axis=0).T, cmap='Blues_r', aspect='auto', vmin = -3, vmax = 3)#, vmin = 0.1, vmax = 0.6
        for i in range(len(validCps1)):
            for j in range(len(validCps2)):
                if 0.05 < pDist[i,j] <= 0.1:
                    text = ax.text(i, j, '+', ha="center", va="center", fontsize=6) #, color = 'yellow'
                elif 0.01 < pDist[i,j] <= 0.05:
                    text = ax.text(i, j, '*', ha="center", va="center", fontsize=6) #, color = 'yellow'
                elif pDist[i,j] <= 0.01:
                    text = ax.text(i, j, '**', ha="center", va="center", fontsize=6) #, color = 'yellow'
                    
        ax.set_xticks([n for n in range(len(validCps1))])
        ax.set_xticklabels(validCps1, fontsize = 6)
        ax.set_yticks([n for n in range(len(validCps2))])
        ax.set_yticklabels(validCps2, fontsize = 6)
        ax.set_xlabel('Loc1')
        ax.set_ylabel('Loc2')
        
        #ax.invert_yaxis()
        ax.set_title('3d Projs')
        plt.colorbar(im,ax=ax)
        
        
        ### 2d
        plt.subplot(1,2,2)
        ax = plt.gca()
        im = ax.imshow(xtemp_dist2dZ.mean(axis=0).T, cmap='Blues_r', aspect='auto', vmin = -3, vmax = 3)#, vmin = 0.1, vmax = 0.6
        for i in range(len(validCps1)):
            for j in range(len(validCps2)):
                if 0.05 < pDist2d[i,j] <= 0.1:
                    text = ax.text(i, j, '+', ha="center", va="center", fontsize=6) #, color = 'yellow'
                elif 0.01 < pDist2d[i,j] <= 0.05:
                    text = ax.text(i, j, '*', ha="center", va="center", fontsize=6) #, color = 'yellow'
                elif pDist2d[i,j] <= 0.01:
                    text = ax.text(i, j, '**', ha="center", va="center", fontsize=6) #, color = 'yellow'
                    
        ax.set_xticks([n for n in range(len(validCps1))])
        ax.set_xticklabels(validCps1, fontsize = 6)
        ax.set_yticks([n for n in range(len(validCps2))])
        ax.set_yticklabels(validCps2, fontsize = 6)
        ax.set_xlabel('Loc1')
        ax.set_ylabel('Loc2')
        
        #ax.invert_yaxis()
        ax.set_title('2d Projs')
        plt.colorbar(im,ax=ax)
        
        
        plt.suptitle(f'{region}, retarget, mean pairwise dist d2i2->d1i1')
        plt.show()










































###########################################################################################################################################################
# In[] correlation between cosTheta_12 and decodability (omega2PEV)

validCheckpoints = {'dlpfc': (1450, 1600, 1800, 2000, 2200, 2400, 2600), 'fef': (1450, )} # 1600, 1800, 2000, 2200, 2400, 2600

for region in ('dlpfc','fef'):
    
    psi12 = []
    info1 = []
    info2 = []
    
    for tt in ttypes:  # (1,)
        
        for n in range(nIters):
            for nbt in range(nBoots):
                for nc, cp in enumerate(checkpoints):
                    if cp in validCheckpoints[region]:
                        psi12 += [np.cos(phase_alignment(projs[region][cp][tt][1][n][nbt], vecs[region][cp][tt][1][n][nbt], projs[region][cp][tt][2][n][nbt], vecs[region][cp][tt][2][n][nbt]))]# / np.pi
                        info1 += [decodability_proj[region][cp][tt][1][n][nbt].mean(axis=-1)] # if use omega2
                        info2 += [decodability_proj[region][cp][tt][2][n][nbt].mean(axis=-1)]
                        #info1 += [decodability_proj[region][cp][tt][1][n][nbt]] # if use lda performance
                        #info2 += [decodability_proj[region][cp][tt][2][n][nbt]]
    
    psi12 = np.array(psi12)
    info1 = np.array(info1)
    info2 = np.array(info2)
    
    infoDiff = np.abs(info1 - info2)/(info1 + info2)
    
    z = np.polyfit(infoDiff, psi12, 1)
    p = np.poly1d(z)
    
    corr, p_corr = stats.pearsonr(psi12, infoDiff)
    
    plt.figure(figsize=(5, 3), dpi=100)
    
    plt.subplot(1,1,1)
    ax = plt.gca()
    
    ax.scatter(infoDiff, psi12, marker = '.')
    ax.plot(infoDiff,p(infoDiff), color = 'r', linestyle='--', alpha = 0.6)
    
    ax.set_title(f'r = {corr:.3f}, p = {p_corr:.3f}', pad = 10)
    ax.set_xlabel('infoDiff')
    ax.set_ylabel('cosPsi')
    #ax.set_ylim((-1,1))
    #ax.legend()
    
    plt.subplots_adjust(top = 0.8)
    plt.suptitle(f'{region}, timepoints: {validCheckpoints[region]}', fontsize = 10, y=1)
    plt.show()
        

# In[] correlation between cosTheta_12 and decodability, at each timepoints (omega2PEV)
corr_InfoTheta_12 = {}

for region in ('dlpfc','fef'):
    
    corr_InfoTheta_12[region] = {}
    
    theta12 = []
    info1 = []
    info2 = []
    
    for tt in ttypes:

        cosTheta_12DiagT = np.zeros((nIters, nBoots, len(checkpoints)))
        decode_proj1T = np.zeros((nIters, nBoots, len(checkpoints))) # pca1st 3d coordinates
        decode_proj2T = np.zeros((nIters, nBoots, len(checkpoints)))
        
        for n in range(nIters):
            for nbt in range(nBoots):
                for nc, cp in enumerate(checkpoints):
                    cosTheta_12DiagT[n,nbt,nc,] = np.abs(angle_btw_planes(vecs_normal[region][cp][tt][1][n][nbt], vecs_normal[region][cp][tt][2][n][nbt]))
                    decode_proj1T[n,nbt,nc] = decodability_proj[region][cp][tt][1][n][nbt].mean(axis=-1)
                    decode_proj2T[n,nbt,nc] = decodability_proj[region][cp][tt][2][n][nbt].mean(axis=-1)
                    
                    
        theta12 += [cosTheta_12DiagT.mean(axis=1)]
        info1 += [decode_proj1T.mean(axis=1)]
        info2 += [decode_proj2T.mean(axis=1)]
    
    theta12 = np.concatenate(theta12, axis = 0)
    info1 = np.concatenate(info1, axis = 0)
    info2 = np.concatenate(info2, axis = 0)
    
    infoDiff = np.abs(info1 - info2)/(info1 + info2)
    
    
    plt.figure(figsize=(5, 3), dpi=100)
    
    corr = np.array([stats.pearsonr(theta12[:,nc], infoDiff[:,nc])[0] for nc, cp in enumerate(checkpoints)])
    p_corr = np.array([stats.pearsonr(theta12[:,nc], infoDiff[:,nc])[1] for nc, cp in enumerate(checkpoints)])
    
    plt.subplot(1,1,1)
    ax = plt.gca()
    ax.plot(np.arange(0, len(checkpoints), 1), corr, marker = 'o', linestyle='-')
    
    trans = ax.get_xaxis_transform()
    for nc, cp in enumerate(checkpoints):
        if 0.05 < p_corr[nc] <= 0.1:
            ax.annotate('+', xy=(nc, 0.01), xycoords = trans, annotation_clip = False, color = 'b', ha = 'center', va = 'center')
        elif 0.01 < p_corr[nc] <= 0.05:
            ax.annotate('*', xy=(nc, 0.01), xycoords = trans, annotation_clip = False, color = 'b', ha = 'center', va = 'center')
        elif p_corr[nc] <= 0.01:
            ax.annotate('**', xy=(nc, 0.01), xycoords = trans, annotation_clip = False, color = 'b', ha = 'center', va = 'center')
    
    #ax.set_title(f'{region}', pad = 10)
    ax.set_xticks([n for n in range(len(checkpoints))])
    ax.set_xticklabels(checkpoints, fontsize = 6)
    ax.set_xlabel('time (ms)')
    ax.set_ylim((-1,1))
    ax.set_ylabel('pearson r')
    #ax.legend()
    
    plt.subplots_adjust(top = 0.8)
    plt.suptitle(f'cosTheta_infoDiff, {region}', fontsize = 15, y=1)
    plt.show()
    
    
    plt.figure(figsize=(15, 5), dpi=100)
    for nc, cp in enumerate(checkpoints):
        
        trendline = np.polyfit(theta12[:,nc], infoDiff[:,nc], 1)
        trendline_fit = np.poly1d(trendline)
        
        plt.subplot(2,7,nc+1)
        ax = plt.gca()
        ax.scatter(theta12[:,nc], infoDiff[:,nc], marker = 'o')
        ax.plot(theta12[:,nc], trendline_fit(theta12[:,nc]), color = 'r', linestyle='--', alpha = 0.3)
        
        ax.set_xlabel('cosTheta')
        ax.set_ylabel('infoDiff')
        ax.set_title(f'{cp}', pad = 10)
    
    plt.subplots_adjust(top = 0.8)
    plt.suptitle(f'scatter cosTheta_infoDiff, {region}', fontsize = 15, y=1)
    plt.tight_layout()
    plt.show()

# In[] correlation between cosTheta_12 and decodability, at each timepoints, by ttype (omega2PEV)
corr_InfoTheta_12 = {}

for region in ('dlpfc','fef'):
    
    corr_InfoTheta_12[region] = {}
    
    plt.figure(figsize=(10, 3), dpi=100)
    for tt in ttypes:
        
        cosTheta_12DiagT = np.zeros((nIters, nBoots, len(checkpoints)))
        decode_proj1T = np.zeros((nIters, nBoots, len(checkpoints))) # pca1st 3d coordinates
        decode_proj2T = np.zeros((nIters, nBoots, len(checkpoints)))
        
        for n in range(nIters):
            for nbt in range(nBoots):
                for nc, cp in enumerate(checkpoints):
                    cosTheta_12DiagT[n,nbt,nc,] = np.abs(angle_btw_planes(vecs_normal[region][cp][tt][1][n][nbt], vecs_normal[region][cp][tt][2][n][nbt]))
                    decode_proj1T[n,nbt,nc] = decodability_proj[region][cp][tt][1][n][nbt].mean(axis=-1)
                    decode_proj2T[n,nbt,nc] = decodability_proj[region][cp][tt][2][n][nbt].mean(axis=-1)
                    
                    
        theta12 = cosTheta_12DiagT.mean(axis=1)
        info1 = decode_proj1T.mean(axis=1)
        info2 = decode_proj2T.mean(axis=1)
    
        infoDiff = np.abs(info1 - info2)/(info1 + info2)
        
        corr = np.array([stats.pearsonr(theta12[:,nc], infoDiff[:,nc])[0] for nc, cp in enumerate(checkpoints)])
        p_corr = np.array([stats.pearsonr(theta12[:,nc], infoDiff[:,nc])[1] for nc, cp in enumerate(checkpoints)])
    
        plt.subplot(1,2,tt)
        ax = plt.gca()
        ax.plot(np.arange(0, len(checkpoints), 1), corr, marker = 'o', linestyle='-')
        
        trans = ax.get_xaxis_transform()
        for nc, cp in enumerate(checkpoints):
            if 0.05 < p_corr[nc] <= 0.1:
                ax.annotate('+', xy=(nc, 0.01), xycoords = trans, annotation_clip = False, color = 'b', ha = 'center', va = 'center')
            elif 0.01 < p_corr[nc] <= 0.05:
                ax.annotate('*', xy=(nc, 0.01), xycoords = trans, annotation_clip = False, color = 'b', ha = 'center', va = 'center')
            elif p_corr[nc] <= 0.01:
                ax.annotate('**', xy=(nc, 0.01), xycoords = trans, annotation_clip = False, color = 'b', ha = 'center', va = 'center')
        
        ax.set_title(f'type = {tt}', pad = 10)
        ax.set_xticks([n for n in range(len(checkpoints))])
        ax.set_xticklabels(checkpoints, fontsize = 6)
        ax.set_xlabel('time (ms)')
        ax.set_ylim((-1,1))
        ax.set_ylabel('pearson r')
        #ax.legend()
    
    plt.subplots_adjust(top = 0.8)
    plt.suptitle(f'cosTheta_infoDiff, {region}', fontsize = 15, y=1)
    plt.show()
    
    for tt in ttypes:
        
        cosTheta_12DiagT = np.zeros((nIters, nBoots, len(checkpoints)))
        decode_proj1T = np.zeros((nIters, nBoots, len(checkpoints))) # pca1st 3d coordinates
        decode_proj2T = np.zeros((nIters, nBoots, len(checkpoints)))
        
        for n in range(nIters):
            for nbt in range(nBoots):
                for nc, cp in enumerate(checkpoints):
                    cosTheta_12DiagT[n,nbt,nc,] = np.abs(angle_btw_planes(vecs_normal[region][cp][tt][1][n][nbt], vecs_normal[region][cp][tt][2][n][nbt]))
                    decode_proj1T[n,nbt,nc] = decodability_proj[region][cp][tt][1][n][nbt].mean(axis=-1)
                    decode_proj2T[n,nbt,nc] = decodability_proj[region][cp][tt][2][n][nbt].mean(axis=-1)
                    
                    
        theta12 = cosTheta_12DiagT.mean(axis=1)
        info1 = decode_proj1T.mean(axis=1)
        info2 = decode_proj2T.mean(axis=1)
    
        infoDiff = np.abs(info1 - info2)
        
        plt.figure(figsize=(15, 5), dpi=100)
        for nc, cp in enumerate(checkpoints):
            
            trendline = np.polyfit(theta12[:,nc], infoDiff[:,nc], 1)
            trendline_fit = np.poly1d(trendline)
            
            plt.subplot(2,7,nc+1)
            ax = plt.gca()
            ax.scatter(theta12[:,nc], infoDiff[:,nc], marker = 'o')
            ax.plot(theta12[:,nc], trendline_fit(theta12[:,nc]), color = 'r', linestyle='--', alpha = 0.3)
            
            ax.set_xlabel('cosTheta')
            ax.set_ylabel('infoDiff')
            ax.set_title(f'{cp}', pad = 10)
        
        plt.subplots_adjust(top = 0.8)
        plt.suptitle(f'scatter cosTheta_infoDiff, {region}', fontsize = 15, y=1)
        plt.tight_layout()
        plt.show()       
        
        
        
# In[] correlation between cosPsi_12 and decodability (omega2PEV)

validCheckpoints = {'dlpfc': (1450, 1600, 1800, 2000, ), 'fef': (1450, )} # 1600, 1800, 2000, 2200, 2400, 2600 2200, 2400, 2600

for region in ('dlpfc','fef'):
    
    psi12 = []
    info1 = []
    info2 = []
    
    ttypesT = (1,2) # (1,)
    
    for tt in ttypesT:  #
        
        for n in range(nIters):
            for nbt in range(nBoots):
                for nc, cp in enumerate(checkpoints):
                    if cp in validCheckpoints[region]:
                        psi12 += [np.abs(np.cos(phase_alignment(projs[region][cp][tt][1][n][nbt], vecs[region][cp][tt][1][n][nbt], projs[region][cp][tt][2][n][nbt], vecs[region][cp][tt][2][n][nbt])))]
                        info1 += [decodability_proj[region][cp][tt][1][n][nbt].mean(axis=-1)] # if use omega2
                        info2 += [decodability_proj[region][cp][tt][2][n][nbt].mean(axis=-1)]
                        #info1 += [decodability_proj[region][cp][tt][1][n][nbt]] # if use lda performance
                        #info2 += [decodability_proj[region][cp][tt][2][n][nbt]]
    
    psi12 = np.array(psi12)
    info1 = np.array(info1)
    info2 = np.array(info2)
    
    infoDiff = np.abs(info1 - info2)/(info1 + info2)
    
    z = np.polyfit(infoDiff, psi12, 1)
    p = np.poly1d(z)
    
    method = 'spearman' #'pearson'
    
    if method == 'spearman':
        corr, p_corr = stats.spearmanr(psi12, infoDiff)
    elif method == 'pearson':
        corr, p_corr = stats.pearsonr(psi12, infoDiff)
    
    
    plt.figure(figsize=(5, 3), dpi=100)
    
    plt.subplot(1,1,1)
    ax = plt.gca()
    
    ax.scatter(infoDiff, psi12, marker = '.')
    ax.plot(infoDiff,p(infoDiff), color = 'r', linestyle='--', alpha = 0.6)
    
    ax.set_title(f'{method} r = {corr:.3f}, p = {p_corr:.3f}', pad = 10)
    ax.set_xlabel('infoDiff')
    ax.set_ylabel('abs(cosPsi)')
    #ax.set_ylim((-1,1))
    #ax.legend()
    
    plt.subplots_adjust(top = 0.8)
    plt.suptitle(f'{region}, types: {ttypesT}, timepoints: {validCheckpoints[region]}', fontsize = 10, y=1)
    plt.show()

# In[] compare choice/non-choice 

discriminability_choice = {} 
#discriminability_nonchoice = {}
#discriminability_choice_shuff_mean = {}
discriminability_choice_shuff_all = {}
#discriminability_nonchoice_shuff_mean = {}
#discriminability_nonchoice_shuff_all = {}

for region in ('dlpfc','fef'):
        
    projs_choice1 = np.zeros((nIters, nBoots, len(checkpoints), len(locs), 3))
    projs_choice2 = np.zeros((nIters, nBoots, len(checkpoints), len(locs), 3))
    for n in range(nIters):
        for nbt in range(nBoots):
            for nc,cp in enumerate(checkpoints):
                projs_choice1[n,nbt,nc, :, :] = projs[region][cp][1][2][n][nbt]
                projs_choice2[n,nbt,nc, :, :] = projs[region][cp][2][1][n][nbt]
    
    projs_choice1 = np.concatenate(projs_choice1, axis = 0)
    projs_choice2 = np.concatenate(projs_choice2, axis = 0)
    
    discriminability_choiceT = np.zeros((nIters, len(checkpoints),))
    #discriminability_nonchoiceT = np.zeros((nIters, len(checkpoints),))
    
    for n in range(nIters):
        for nc,cp in enumerate(checkpoints):
            
            projs_choice1T = projs_choice1[n,nc,:,:]
            projs_choice2T = projs_choice2[n,nc,:,:]
            labels1T = np.full((projs_choice1T.shape[0]), '1')
            labels2T = np.full((projs_choice1T.shape[0]), '2')
            
            projs_choiceT = np.concatenate((projs_choice1T, projs_choice2T))
            labelsT = np.concatenate((labels1T, labels2T))
            
            clfT = sklearn.linear_model.LogisticRegression()
            
            discriminability_choiceT[n,nc] = clfT.fit(projs_choiceT, labelsT).score(projs_choiceT, labelsT)
            #discriminability_nonchoiceT[n,nbt,nc] = np.abs(angle_btw_planes(vecs_normal[region][cp][1][1][n][nbt], vecs_normal[region][cp][2][2][n][nbt]))
    
    discriminability_choice[region] = discriminability_choiceT
    #discriminability_nonchoice[region] = discriminability_nonchoiceT 
    
    
    
    
    projs_choice1_shuff = np.zeros((nIters, nBoots*nPerms, len(checkpoints), len(locs), 3))
    projs_choice2_shuff = np.zeros((nIters, nBoots*nPerms, len(checkpoints), len(locs), 3))
    #discriminability_nonchoice_shuffT = np.zeros((nIters, nBoots*nPerms, len(checkpoints),))
    
    for n in range(nIters):
        for nbt in range(nBoots*nPerms):
            for nc, cp in enumerate(checkpoints):
                projs_choice1_shuff[n,nbt,nc, :, :] = projs_shuff[region][cp][1][2][n][nbt]
                projs_choice2_shuff[n,nbt,nc, :, :] = projs_shuff[region][cp][2][1][n][nbt]
                #discriminability_nonchoice_shuffT[n,nbt,nc,] = np.abs(angle_btw_planes(vecs_normal_shuff[region][cp][1][1][n][nbt], vecs_normal_shuff[region][cp][2][2][n][nbt]))
    
    discriminability_choice_shuffT = np.zeros((nIters, nBoots*nPerms, len(checkpoints),))
    
    for n in range(nIters):
        for nbt in range(nBoots*nPerms):
            for nc,cp in enumerate(checkpoints):
                projs_choice1_shuffT = projs_choice1_shuff[n,nbt,nc,:,:]
                projs_choice2_shuffT = projs_choice2_shuff[n,nbt,nc,:,:]
                labels1_shuffT = np.full((projs_choice1_shuffT.shape[0]), '1')
                labels2_shuffT = np.full((projs_choice2_shuffT.shape[0]), '2')
                
                projs_choice_shuffT = np.concatenate((projs_choice1_shuffT, projs_choice2_shuffT))
                labels_shuffT = np.concatenate((labels1_shuffT, labels2_shuffT))
                
                clfT = sklearn.linear_model.LogisticRegression()
                
                discriminability_choice_shuffT[n,nbt,nc] = clfT.fit(projs_choice_shuffT, labels_shuffT).score(projs_choice_shuffT, labels_shuffT)
    
    discriminability_choice_shuffT = np.concatenate(discriminability_choice_shuffT, axis=0)
    discriminability_choice_shuff_all[region] = discriminability_choice_shuffT
    
    
    #discriminability_choiceT_shuff_mean, discriminability_choiceT_shuff_all = discriminability_choice_shuffT.mean(axis=1), np.concatenate([discriminability_choice_shuffT[:,nbt,:] for nbt in range(nBoots*nPerms)])
    #discriminability_choice_shuff_mean[region], discriminability_choice_shuff_all[region] = discriminability_choiceT_shuff_mean, discriminability_choiceT_shuff_all
    
    #discriminability_nonchoiceT_shuff_mean, discriminability_nonchoiceT_shuff_all = discriminability_nonchoice_shuffT.mean(axis=1), np.concatenate([discriminability_nonchoice_shuffT[:,nbt,:] for nbt in range(nBoots*nPerms)])
    #discriminability_nonchoice_shuff_mean[region], discriminability_nonchoice_shuff_all[region] = discriminability_nonchoiceT_shuff_mean, discriminability_nonchoiceT_shuff_all
    
    
    #pPerms_choice = np.array([f_stats.permutation_p(discriminability_choice[region].mean(axis=0)[nc], discriminability_choice_shuff_all[region][:,nc], tail='greater') for nc, cp in enumerate(checkpoints)]) #/2
    pPerms_choice = np.array([stats.ttest_1samp(discriminability_choice[region][:, nc], 0.5, alternative='greater')[1] for nc, cp in enumerate(checkpoints)])
    #pPerms_nonchoice = np.array([f_stats.permutation_p(discriminability_nonchoice[region].mean(axis=1).mean(axis=0)[nc], discriminability_nonchoice_shuff_all[region][:,nc], tail='two') for nc, cp in enumerate(checkpoints)]) /2
    
    #pPerms_choice = np.array([f_stats.permutation_pCI(discriminability_choice[region].mean(axis=1)[:,nc].round(5), discriminability_choice_shuff_all[region][:,nc].round(5), CI_size=1.96) for nc, cp in enumerate(checkpoints)])
    #pPerms_nonchoice = np.array([f_stats.permutation_pCI(discriminability_nonchoice[region].mean(axis=1)[:,nc].round(5), discriminability_nonchoice_shuff_all[region][:,nc].round(5), CI_size=1.96) for nc, cp in enumerate(checkpoints)])
    
    
    plt.figure(figsize=(10, 3), dpi=100)
    plt.subplot(1,2,1)
    ax = plt.gca()
    ax.errorbar(np.arange(0, len(checkpoints), 1), discriminability_choiceT.mean(axis=1).mean(axis=0), yerr = discriminability_choiceT.mean(axis=1).std(axis=0), marker = 'o')
    ax.plot(np.arange(0, len(checkpoints), 1), pPerms_choice, alpha = 0.3, linestyle = '-')
    
    trans = ax.get_xaxis_transform()
    for nc, cp in enumerate(checkpoints):
        if 0.05 < pPerms_choice[nc] <= 0.1:
            ax.annotate('+', xy=(nc, 0.05), xycoords = trans, annotation_clip = False, ha = 'center', va = 'center')
        elif 0.01 < pPerms_choice[nc] <= 0.05:
            ax.annotate('*', xy=(nc, 0.05), xycoords = trans, annotation_clip = False, ha = 'center', va = 'center')
        elif pPerms_choice[nc] <= 0.01:
            ax.annotate('**', xy=(nc, 0.05), xycoords = trans, annotation_clip = False, ha = 'center', va = 'center')
            
    ax.set_title('choice', pad = 10)
    ax.set_xticks([n for n in range(len(checkpoints))])
    ax.set_xticklabels(checkpoints, fontsize = 6)
    ax.set_ylim((0,1))
    
    
    plt.subplot(1,2,2)
    ax = plt.gca()
    ax.errorbar(np.arange(0, len(checkpoints), 1), discriminability_nonchoiceT.mean(axis=1).mean(axis=0), yerr = discriminability_nonchoiceT.mean(axis=1).std(axis=0), marker = 'o')
    ax.plot(np.arange(0, len(checkpoints), 1), pPerms_nonchoice, alpha = 0.3, linestyle = '-')
    
    trans = ax.get_xaxis_transform()
    for nc, cp in enumerate(checkpoints):
        if 0.05 < pPerms_nonchoice[nc] <= 0.1:
            ax.annotate('+', xy=(nc, 0.05), xycoords = trans, annotation_clip = False, ha = 'center', va = 'center')
        elif 0.01 < pPerms_nonchoice[nc] <= 0.05:
            ax.annotate('*', xy=(nc, 0.05), xycoords = trans, annotation_clip = False, ha = 'center', va = 'center')
        elif pPerms_nonchoice[nc] <= 0.01:
            ax.annotate('**', xy=(nc, 0.05), xycoords = trans, annotation_clip = False, ha = 'center', va = 'center')
            
    ax.set_title('non-choice', pad = 10)
    ax.set_xticks([n for n in range(len(checkpoints))])
    ax.set_xticklabels(checkpoints, fontsize = 6)
    ax.set_ylim((0,1))
    
    plt.subplots_adjust(top = 0.8)
    plt.suptitle(f'discriminability, {region}, choice/nonchoice', fontsize = 15, y=1)
    plt.show()    
    



# In[]
discriminability = {} 
#discriminability_nonchoice = {}
#discriminability_choice_shuff_mean = {}
discriminability_shuff_all = {}
#discriminability_nonchoice_shuff_mean = {}
#discriminability_nonchoice_shuff_all = {}

for region in ('dlpfc','fef'):
    
    checkT = (1100,1300,2400,2600)
    
    discriminability_T = np.zeros((nIters))
    
    for n in range(nIters):
        projs_1T, projs_2T = [], []
        labels_1T, labels_2T = [], []
        
        for nbt in range(nBoots):
            for nc,cp in enumerate(checkT):
                if cp < 1450:
                    projs_1T += [projs[region][cp][1][1][n][nbt]]
                    labels_1T += [np.full((projs[region][cp][1][1][n][nbt].shape[0]), '1')] 
                else:
                    projs_2T += [projs[region][cp][1][2][n][nbt]]
                    labels_2T += [np.full((projs[region][cp][1][2][n][nbt].shape[0]), '2')] 
            
        projs_T = np.concatenate((np.concatenate(projs_1T), np.concatenate(projs_2T)))
        labels_T = np.concatenate((np.concatenate(labels_1T), np.concatenate(labels_2T)))
            
        clfT = sklearn.linear_model.LogisticRegression()
        
        discriminability_T[n] = clfT.fit(projs_T, labels_T).score(projs_T, labels_T)
        #discriminability_nonchoiceT[n,nbt,nc] = np.abs(angle_btw_planes(vecs_normal[region][cp][1][1][n][nbt], vecs_normal[region][cp][2][2][n][nbt]))
    
    discriminability[region] = discriminability_T

# In[] check distribution

for region in ('dlpfc', 'fef'):
    for tt in ttypes:
        
        plt.figure(figsize=(40,10), dpi=100)
        
        for nc, cp in enumerate(checkpoints):
            
            kde1 = scipy.stats.gaussian_kde(decode_proj1[region][tt].mean(axis=1).mean(axis=-1)[:,nc])
            kde_shuff1 = scipy.stats.gaussian_kde(decode_proj1_shuff_all[region][tt].mean(axis=-1)[:,nc])
            kde2 = scipy.stats.gaussian_kde(decode_proj2[region][tt].mean(axis=1).mean(axis=-1)[:,nc])
            kde_shuff2 = scipy.stats.gaussian_kde(decode_proj2_shuff_all[region][tt].mean(axis=-1)[:,nc])
            dist_space = np.linspace(0,1,100)
            
            plt.subplot(2,7,nc+1)
            
            plt.plot(dist_space, kde1(dist_space), label = f'data1, {decode_proj1[region][tt].mean(axis=1).mean(axis=-1)[:,nc].mean():.4f}')
            plt.plot(dist_space, kde_shuff1(dist_space), label = 'shuffle1')
            plt.plot(dist_space, kde2(dist_space), label = f'data2, {decode_proj2[region][tt].mean(axis=1).mean(axis=-1)[:,nc].mean():.4f}')
            plt.plot(dist_space, kde_shuff2(dist_space), label = 'shuffle2')
            
            #plt.hist(decode_proj1[region][tt].mean(axis=1).mean(axis=-1)[nc], label = f'data1, {decode_proj1[region][tt].mean(axis=1).mean(axis=-1)[nc].mean():.4f}')
            #plt.hist(decode_proj2[region][tt].mean(axis=1).mean(axis=-1)[nc], label = f'data2, {decode_proj2[region][tt].mean(axis=1).mean(axis=-1)[nc].mean():.4f}')
            
            plt.legend()
            plt.title(f'{cp}')
        
        plt.suptitle(f'{region},{tt}')
        plt.plot()












































































# In[] compare within location, between types

cosTheta_loc1_ttype, cosTheta_loc2_ttype = {},{}

for region in ('dlpfc','fef'):
    
    cosTheta_loc1_ttype[region] = {}
    cosTheta_loc2_ttype[region] = {}
    
    
    cosTheta_loc1_ttypeT = np.zeros((nIters, len(checkpoints),))
    cosTheta_loc2_ttypeT = np.zeros((nIters, len(checkpoints),))
    
    for n in range(nIters):
        for nc, cp in enumerate(checkpoints):        
            cosTheta_loc1_ttypeT[n,nc] = np.abs(angle_btw_planes(vecs_normal[region][cp][1][1][n], vecs_normal[region][cp][2][1][n]))
            cosTheta_loc2_ttypeT[n,nc] = np.abs(angle_btw_planes(vecs_normal[region][cp][1][2][n], vecs_normal[region][cp][2][2][n]))
    
    cosTheta_loc1_ttype[region] = cosTheta_loc1_ttypeT
    cosTheta_loc2_ttype[region] = cosTheta_loc2_ttypeT   
    
    plt.figure(figsize=(10, 3), dpi=100)
    plt.subplot(1,2,1)
    ax = plt.gca()
    ax.errorbar(np.arange(0, len(checkpoints), 1), cosTheta_loc1_ttypeT.mean(axis=0), yerr = cosTheta_loc1_ttypeT.std(axis=0), marker = 'o')
    ax.set_title('loc1', pad = 10)
    ax.set_xticks([n for n in range(len(checkpoints))])
    ax.set_xticklabels(checkpoints, fontsize = 6)
    ax.set_ylim((0,1))
    
    plt.subplot(1,2,2)
    ax = plt.gca()
    ax.errorbar(np.arange(0, len(checkpoints), 1), cosTheta_loc2_ttypeT.mean(axis=0), yerr = cosTheta_loc2_ttypeT.std(axis=0), marker = 'o')
    ax.set_title('loc2', pad = 10)
    ax.set_xticks([n for n in range(len(checkpoints))])
    ax.set_xticklabels(checkpoints, fontsize = 6)
    ax.set_ylim((0,1))
    
    plt.subplots_adjust(top = 0.8)
    plt.suptitle(f'cosTheta, {region}, between ttype', fontsize = 15, y=1)
    plt.show()    
        
# In[]            
################    
        # cosTheta between location planes (11,22,12)
        cosTheta_11D, cosTheta_22D, cosTheta_12D = cosTheta_between_locs(vecs_normalD, checkpoints, ttypes)
        for tt in ttypes:
            cosTheta_11[region][tt] += [cosTheta_11D[tt]]
            cosTheta_22[region][tt] += [cosTheta_22D[tt]]
            cosTheta_12[region][tt] += [cosTheta_12D[tt]]
        
        # cosTheta by choices
        cosTheta_choiceD, cosTheta_nonchoiceD = cosTheta_choices(vecs_normalD, checkpoints)
        cosTheta_choice[region] += [cosTheta_choiceD]
        cosTheta_nonchoice[region] += [cosTheta_nonchoiceD]
        
        
        ### permutation
        cosTheta_11D_shuff, cosTheta_22D_shuff, cosTheta_12D_shuff = {1:[],2:[]}, {1:[],2:[]}, {1:[],2:[]}
        cosTheta_choiceD_shuff, cosTheta_nonchoiceD_shuff = [],[] 
            

            
            
            # cosTheta between location planes (11,22,12)
            cosTheta_11D_shuffT, cosTheta_22D_shuffT, cosTheta_12D_shuffT = cosTheta_between_locs(vecs_normalD_shuff, checkpoints, ttypes)
            for tt in ttypes:
                cosTheta_11D_shuff[tt] += [cosTheta_11D_shuffT[tt]]
                cosTheta_22D_shuff[tt] += [cosTheta_22D_shuffT[tt]]
                cosTheta_12D_shuff[tt] += [cosTheta_12D_shuffT[tt]]
                
            # cosTheta by choices
            cosTheta_choiceD_shuffT, cosTheta_nonchoiceD_shuffT = cosTheta_choices(vecs_normalD_shuff, checkpoints)
            
            cosTheta_choiceD_shuff += [cosTheta_choiceD_shuffT]
            cosTheta_nonchoiceD_shuff += [cosTheta_nonchoiceD_shuffT]
            
        for tt in ttypes:
            cosTheta_11D_shuff[tt] = np.array(cosTheta_11D_shuff[tt])
            cosTheta_22D_shuff[tt] = np.array(cosTheta_22D_shuff[tt])
            cosTheta_12D_shuff[tt] = np.array(cosTheta_12D_shuff[tt])
        
        cosTheta_choiceD_shuff = np.array(cosTheta_choiceD_shuff)
        cosTheta_nonchoiceD_shuff = np.array(cosTheta_nonchoiceD_shuff)
        
        
        ### pvalues
        p = []
        for ncos, cos in enumerate(cosTheta_choiceD):
            p += [f_stats.permutation_p(cos, cosTheta_choiceD_shuff[:,ncos])]









# In[]

toPlot = -1

for n in range(nIter):
    t_IterOn = time.time()
    print(f'Iter = {n}')
    
    pseudo_TrialInfo = f_pseudoPop.pseudo_Session(samplePerCon=samplePerCon, sampleRounds=sampleRounds)
    pseudo_region = f_pseudoPop.pseudo_PopByRegion_pooled(sessions, cellsToUse, monkey_Info, samplePerCon = samplePerCon, sampleRounds=sampleRounds, arti_noise_level = arti_noise_level)
    
    print(f'tIter = {(time.time() - t_IterOn):.4f}s')
    
    for region in ('dlpfc','fef'):
        
        
        trialInfo = pseudo_TrialInfo.reset_index(names=['id'])
        
        idx1 = trialInfo.id # original index
        idx2 = trialInfo.index.to_list() # reset index
        
        #idxxBsl = [tsliceRange.tolist().index(tBsl[0]), tsliceRange.tolist().index(tBsl[1])]
        
        
        dataT = pseudo_region[region][idx1,::]
        
        # baseline z-normalize data (if not normalized)
        #for ch in range(dataT.shape[1]):
        #    dataT[:,ch,:] = (dataT[:,ch,:] - dataT[:,ch,idxxBsl[0]:idxxBsl[1]].mean())/dataT[:,ch,idxxBsl[0]:idxxBsl[1]].std()
            #dataT[:,ch,:] = scale(dataT[:,ch,:])
        
        # if specify applied time window of pca
        pca_tWin = np.hstack((np.arange(800,1300,dt, dtype = int),np.arange(2100,2600,dt, dtype = int))).tolist()
        pca_tWinX = [tsliceRange.tolist().index(i) for i in pca_tWin]
        
        # condition average data. within each ttype, shape = (4*3) * ncells * nt
        dataT_avg = []
        dataT_avg_full = []
        
        for sc in subConditions:
            lc, tt = sc[0], sc[1]
            slc = '_'.join([str(l) for l in lc])
            idxx = trialInfo[(trialInfo.locs == slc)&(trialInfo.type == tt)].index.tolist()
            
            x = dataT[idxx,:,:].mean(axis=0)
            
            dataT_avg += [x[:,pca_tWinX]]
            dataT_avg_full += [x]
    
        # hstack as ncells * (nt * (4*3))
        dataT_avg = np.hstack(dataT_avg)
        dataT_avg_full = np.hstack(dataT_avg_full)
        
        ###
        # 1st order pca, reduce from ncells * (nt*12) -> 3pcs * (nt*12) to create a 3pc space shared by all locComb conditions, across all timepoints
        pca_1st = PCA(n_components=3)
        
        # fit the PCA model to the data
        pca_1st.fit(dataT_avg.T)
    
        dataT_3pc_meanT  = pca_1st.transform(dataT_avg_full.T).T
    
        evr_1st = pca_1st.explained_variance_ratio_
        
        dataT_3pc_mean = []
        for i in range(len(subConditions)):
            b1, b2 = i*len(tsliceRange), (i+1)*len(tsliceRange)
            xp = dataT_3pc_meanT[:,b1:b2]
            dataT_3pc_mean += [xp]
            
        dataT_3pc_mean = np.array(dataT_3pc_mean) # reshape as (4*3) * 3pcs * nt
        
        dataT_3pc = []
        for trial in range(dataT.shape[0]):
            dataT_3pc += [pca_1st.transform(dataT[trial,:,:].T).T]
        
        dataT_3pc = np.array(dataT_3pc)
        
        # test plots
        #for pc in range(3):
        #    plt.figure()
        #    for i in range(len(subConditions)):
        #        plt.plot(tsliceRange, dataT_3pc_mean[i,pc,:])
        #    plt.title(f'PC{pc}')
        #    plt.show()
        
        
        for cp in checkpoints:
            t1 = tsliceRange.tolist().index(cp-avgInterval) if cp-avgInterval >= tsliceRange.min() else tsliceRange.tolist().index(tsliceRange.min())
            t2 = tsliceRange.tolist().index(cp+avgInterval) if cp+avgInterval <= tsliceRange.max() else tsliceRange.tolist().index(tsliceRange.max())
            
            tempX = dataT_3pc[:,:,t1:t2].mean(axis=2) if t1!=t2 else dataT_3pc[:,:,t1]
            
            tempX_mean = dataT_3pc_mean[:,:,t1:t2].mean(axis=2) if t1!=t2 else dataT_3pc_mean[:,:,t1]
            
            for tt in ttypes:
                
                idx = trialInfo[(trialInfo.type == tt)].index.tolist()
                
                trialInfoT = trialInfo[(trialInfo.type == tt)].reset_index(drop=True)
                tempX_tt = tempX[idx,:]
                
                tempX1_mean = []
                tempX2_mean = []
                
                for l in locs:
                    
                    conx1 = [subConditions.index(sc) for sc in subConditions if (sc[0][0] == l and sc[1] == tt)]
                    conx2 = [subConditions.index(sc) for sc in subConditions if (sc[0][1] == l and sc[1] == tt)]
                
                    tempX1_mean += [tempX_mean[conx1,:].mean(axis = 0)]
                    tempX2_mean += [tempX_mean[conx2,:].mean(axis = 0)]
                    
                    #tempX3 += [temp3[conx2,:].mean(axis = 0)] if tt == 1 else [temp3[conx1,:].mean(axis = 0)]
                
                tempX1_mean = np.array(tempX1_mean)
                #tempX1 = tempX1 - tempX1.mean()
                tempX2_mean = np.array(tempX2_mean)
                #tempX2 = tempX2 - tempX2.mean()
                
                ### loc1 2nd pca
                pca_2nd_1 = PCA(n_components=2)
                pca_2nd_1.fit(tempX1_mean)# - tempX1_mean.mean()
                vecs1, evr2_1 = pca_2nd_1.components_, pca_2nd_1.explained_variance_ratio_
                vec_normal1 = np.cross(vecs1[0], vecs1[1])
                
                center1 = tempX1_mean.mean(axis=0)
                
                proj1 = np.array([proj_on_plane(p1, vec_normal1, center1) for p1 in tempX1_mean])
                #project_to_plane(tempX1_mean, vecs1)
                
                #x1_plane, y1_plane, z1_plane = plane_by_vecs(vecs1, center = center1, xRange=(tempX1_mean[:,0].min(), tempX1_mean[:,0].max()), yRange=(tempX1_mean[:,1].min(), tempX1_mean[:,1].max()))
                x1_plane, y1_plane, z1_plane = plane_by_vecs(vecs1, center = center1, xRange=(proj1[:,0].min(), proj1[:,0].max()), yRange=(proj1[:,1].min(), proj1[:,1].max()))
                #, scaleX=scaleX1, scaleY=scaleY1
                # single trials
                tempX1_pca = pca_2nd_1.transform(tempX_tt) # 2nd pca transformed data
                tempX1_proj = np.array([proj_on_plane(p1, vec_normal1, center1) for p1 in tempX_tt]) # projections on the 2nd pca plane
                
                
                
                ### loc2 2nd pca
                pca_2nd_2 = PCA(n_components=2)
                pca_2nd_2.fit(tempX2_mean)# - tempX2_mean.mean()
                vecs2, evr2_2 = pca_2nd_2.components_, pca_2nd_2.explained_variance_ratio_
                vec_normal2 = np.cross(vecs2[0], vecs2[1])
                
                center2 = tempX2_mean.mean(axis=0)
                
                proj2 = np.array([proj_on_plane(p2, vec_normal2, center2) for p2 in tempX2_mean])
                #project_to_plane(tempX2_mean, vecs2)
                
                #x2_plane, y2_plane, z2_plane = plane_by_vecs(vecs2, center = center2, xRange=(tempX2_mean[:,0].min(), tempX2_mean[:,0].max()), yRange=(tempX2_mean[:,1].min(), tempX2_mean[:,1].max()))
                x2_plane, y2_plane, z2_plane = plane_by_vecs(vecs2, center = center2, xRange=(proj2[:,0].min(), proj2[:,0].max()), yRange=(proj2[:,1].min(), proj2[:,1].max()))
                #, scaleX=scaleX2, scaleY=scaleY2
                # single trials
                tempX2_pca = pca_2nd_2.transform(tempX_tt) # 2nd pca transformed data
                tempX2_proj = np.array([proj_on_plane(p2, vec_normal2, center2) for p2 in tempX_tt]) # projections on the 2nd pca plane
                
                
                
                ### angle between two planes
                cos_theta = angle_btw_planes(vec_normal1, vec_normal2)
                theta = np.degrees(np.arccos(cos_theta))
                
                # store the normal vectors for plane Loc1 and Loc2 at each time
                vecs_normal[region][cp][tt][1] += [vec_normal1]
                vecs_normal[region][cp][tt][2] += [vec_normal2]
                
                ### decodability 
                train_pca, test_pca = split_set(tempX_tt) #, ranseed = n
                #train_pca, test_pca = np.array(idx)[train_pca], np.array(idx)[test_pca]
                
                train_label1, train_label2 = trialInfoT.loc[train_pca,'loc1'], trialInfoT.loc[train_pca,'loc2']
                test_label1, test_label2 = trialInfoT.loc[test_pca,'loc1'], trialInfoT.loc[test_pca,'loc2']
                
                # loc1/loc2 by 1st pca
                performance1_pca1 = f_decoding.LDAPerformance(tempX_tt[train_pca,:], tempX_tt[test_pca,:], train_label1, test_label1)
                performance2_pca1 = f_decoding.LDAPerformance(tempX_tt[train_pca,:], tempX_tt[test_pca,:], train_label2, test_label2)
                
                # loc1/loc2 by 2nd pca
                performance1_pca2 = f_decoding.LDAPerformance(tempX1_pca[train_pca,:], tempX1_pca[test_pca,:], train_label1, test_label1)
                performance2_pca2 = f_decoding.LDAPerformance(tempX2_pca[train_pca,:], tempX2_pca[test_pca,:], train_label2, test_label2)
                
                # loc1/loc2 by 1nd pca
                performance1_pcaProj = f_decoding.LDAPerformance(tempX1_proj[train_pca,:], tempX1_proj[test_pca,:], train_label1, test_label1)
                performance2_pcaProj = f_decoding.LDAPerformance(tempX2_proj[train_pca,:], tempX2_proj[test_pca,:], train_label2, test_label2)
                
                decodability_pca2[region][cp][tt][1] += [performance1_pca2]
                decodability_pca2[region][cp][tt][2] += [performance2_pca2]
                decodability_proj[region][cp][tt][1] += [performance1_pcaProj]
                decodability_proj[region][cp][tt][2] += [performance2_pcaProj]
                
                
                
                ### run only if to plot
                ### plot the best-fitting plane
                # find vertices, then sort vertices according to the shortest path - so that plotted plane will be a quadrilateral
                # only plot in the selected iteration
                if n == toPlot :
                    
                    color1 = 'r'
                    color2 = 'g'
                    #color3 = 'b'
                    
                    #colors = np.array([color1, color2, color3])
                    
                    fig = plt.figure(figsize=(8, 6))
                    ax = fig.add_subplot(111, projection='3d')
                    
                    shapes = ('o','s','*','^')
                    for i in range(len(tempX1_mean)):
                        ax.scatter(tempX1_mean[i,0], tempX1_mean[i,1], tempX1_mean[i,2], marker = shapes[i], color = color1, alpha = 0.7, label = f'loc1, {i}')
                        ax.scatter(proj1[i,0], proj1[i,1], proj1[i,2], marker = shapes[i], color = color1, alpha = 0.2)#, label = f'loc1, {i}'
                        
                        if cp >= slice_epochsDic['s2'][0]:
                            ax.scatter(tempX2_mean[i,0], tempX2_mean[i,1], tempX2_mean[i,2], marker = shapes[i], color = color2, alpha = 0.7, label = f'loc2, {i}')
                            ax.scatter(proj2[i,0], proj2[i,1], proj2[i,2], marker = shapes[i], color = color2, alpha = 0.2)#, label = f'loc2, {i}'
                        #ax.scatter(tempX3[i,0], tempX3[i,1], tempX3[i,2], marker = shapes[i], color = color3, label = f'locChoice, {i}')
                    
                    #ax.scatter(tempX1[:,0], tempX1[:,1], tempX1[:,2], marker = '.', color = color1, alpha = 0.3)
                    #ax.scatter(tempX1_proj[:,0], tempX1_proj[:,1], tempX1_proj[:,2], marker = '.', color = color1, alpha = 0.1)#, label = f'loc1, {i}'
                    #ax.scatter(tempX2_proj[:,0], tempX2_proj[:,1], tempX2_proj[:,2], marker = '.', color = color2, alpha = 0.1)#, label = f'loc1, {i}'
                    
                    ax.plot_surface(x1_plane, y1_plane, z1_plane, alpha=0.5, color = color1)
                    #ax.add_collection3d(Poly3DCollection([sorted_verts1], facecolor=color1, edgecolor=[], alpha=0.2))#
                    #ax.quiver(x1_plane.mean(), y1_plane.mean(), z1_plane.mean(), vec_normal1[0], vec_normal1[1], vec_normal1[2], color = color1, alpha = 0.2)#, arrow_length_ratio = 0.001
                    #ax.quiver(x1_plane.mean(), y1_plane.mean(), z1_plane.mean(), vecs1[0,0], vecs1[0,1], vecs1[0,2], color = color1, alpha = 1)
                    #ax.quiver(x1_plane.mean(), y1_plane.mean(), z1_plane.mean(), vecs1[1,0], vecs1[1,1], vecs1[1,2], color = color1, alpha = 1)
                    ax.quiver(tempX1_mean[:,0].mean(),tempX1_mean[:,1].mean(),tempX1_mean[:,2].mean(), vec_normal1[0], vec_normal1[1], vec_normal1[2], color = color1, alpha = 0.2)#, arrow_length_ratio = 0.001
                    #ax.quiver(tempX1_mean[:,0].mean(),tempX1_mean[:,1].mean(),tempX1_mean[:,2].mean(), vecs1[0,0], vecs1[0,1], vecs1[0,2], color = color1, alpha = 1)
                    #ax.quiver(tempX1_mean[:,0].mean(),tempX1_mean[:,1].mean(),tempX1_mean[:,2].mean(), vecs1[1,0], vecs1[1,1], vecs1[1,2], color = color1, alpha = 1)
                    #ax.text(x1_plane.min(),y1_plane.min(),z1_plane.min(),f'Loc1 EVR:{evr2_1[0]:.4f}; {evr2_1[1]:.4f}')
                    
                    if cp >= slice_epochsDic['s2'][0]:
                        
                        ax.plot_surface(x2_plane, y2_plane, z2_plane, alpha=0.5, color = color2)
                        #ax.add_collection3d(Poly3DCollection([sorted_verts2], facecolor=color2, edgecolor=[], alpha=0.2))#
                        #ax.quiver(x2_plane.mean(), y2_plane.mean(), z2_plane.mean(), vec_normal2[0], vec_normal2[1], vec_normal2[2], color = color2, alpha = 0.2)#, arrow_length_ratio = 0.001
                        #ax.quiver(x2_plane.mean(), y2_plane.mean(), z2_plane.mean(), vecs2[0,0], vecs2[0,1], vecs2[0,2], color = color2, alpha = 1)
                        #ax.quiver(x2_plane.mean(), y2_plane.mean(), z2_plane.mean(), vecs2[1,0], vecs2[1,1], vecs2[1,2], color = color2, alpha = 1)
                        ax.quiver(tempX2_mean[:,0].mean(),tempX2_mean[:,1].mean(),tempX2_mean[:,2].mean(), vec_normal2[0], vec_normal2[1], vec_normal2[2], color = color2, alpha = 0.2)#, arrow_length_ratio = 0.001
                        #ax.quiver(tempX2_mean[:,0].mean(),tempX2_mean[:,1].mean(),tempX2_mean[:,2].mean(), vecs2[0,0], vecs2[0,1], vecs2[0,2], color = color2, alpha = 1)
                        #ax.quiver(tempX2_mean[:,0].mean(),tempX2_mean[:,1].mean(),tempX2_mean[:,2].mean(), vecs2[1,0], vecs2[1,1], vecs2[1,2], color = color2, alpha = 1)
                        #ax.text(x2_plane.min(),y2_plane.min(),z2_plane.min(),f'Loc2 EVR:{evr2_2[0]:.4f}; {evr2_2[1]:.4f}')
                    
                    
                    ax.set_xlabel(f'PC1 ({evr_1st[0]:.4f})')
                    ax.set_ylabel(f'PC2 ({evr_1st[1]:.4f})')
                    ax.set_zlabel(f'PC3 ({evr_1st[2]:.4f})')
                    
                    #ax.view_init(elev=45, azim=45, roll=0)
                    
                    plt.legend()
                    plt.title(f'{region}, t = {cp}, type = {tt}, cosTheta = {cos_theta:.3f}, loc1% = {performance1_pcaProj:.2f}; loc2% = {performance2_pcaProj:.2f}')
                    plt.show()





































# In[]

toPlot = -1

for n in range(nIter):
    t_IterOn = time.time()
    print(f'Iter = {n}')
    
    pseudo_TrialInfo = pseudo_Session(samplePerCon=samplePerCon, sampleRounds=sampleRounds)
    pseudo_region = pseudo_PopByRegion_pooled(sessions, cellsToUse, monkey_Info, samplePerCon = samplePerCon, sampleRounds=sampleRounds, arti_noise_level = arti_noise_level)
    
    print(f'tIter = {(time.time() - t_IterOn):.4f}s')
    
    for region in ('dlpfc','fef'):
        
        
        trialInfo = pseudo_TrialInfo.reset_index(names=['id'])
        
        idx1 = trialInfo.id # original index
        idx2 = trialInfo.index.to_list() # reset index
        
        #idxxBsl = [tsliceRange.tolist().index(tBsl[0]), tsliceRange.tolist().index(tBsl[1])]
        
        
        dataT = pseudo_region[region][idx1,::]
        
        # for permutation p value check use
        dataT_shuffled = dataT[np.random.permutation(dataT.shape[0]),:,:]# should shuffle along axis=0
        
        # baseline z-normalize data (if not normalized)
        #for ch in range(dataT.shape[1]):
        #    dataT[:,ch,:] = (dataT[:,ch,:] - dataT[:,ch,idxxBsl[0]:idxxBsl[1]].mean())/dataT[:,ch,idxxBsl[0]:idxxBsl[1]].std()
            #dataT[:,ch,:] = scale(dataT[:,ch,:])
        
        # if specify applied time window of pca
        pca_tWin = np.hstack((np.arange(800,1300,dt, dtype = int),np.arange(2100,2600,dt, dtype = int))).tolist()
        pca_tWinX = [tsliceRange.tolist().index(i) for i in pca_tWin]
        
        # condition average data. within each ttype, shape = (4*3) * ncells * nt
        dataT_avg = []
        dataT_avg_full = []
        dataT_avg_shuffled = []
        
        for sc in subConditions:
            lc, tt = sc[0], sc[1]
            slc = '_'.join([str(l) for l in lc])
            idxx = trialInfo[(trialInfo.locs == slc)&(trialInfo.type == tt)].index.tolist()
            
            x = dataT[idxx,:,:].mean(axis=0)
            
            dataT_avg += [x[:,pca_tWinX]]
            dataT_avg_full += [x]
            dataT_avg_shuffled += [dataT_shuffled[idxx,:,:].mean(axis=0)]
    
        # hstack as ncells * (nt * (4*3))
        dataT_avg = np.hstack(dataT_avg)
        dataT_avg_full = np.hstack(dataT_avg_full)
        dataT_avg_shuffled = np.hstack(dataT_avg_shuffled)
        
        ###
        # 1st order pca, reduce from ncells * (nt*12) -> 3pcs * (nt*12) to create a 3pc space shared by all locComb conditions, across all timepoints
        pca_1st = PCA(n_components=3)
        
        # fit the PCA model to the data
        pca_1st.fit(dataT_avg.T)
    
        dataT_3pc_meanT  = pca_1st.transform(dataT_avg_full.T).T
        dataT_3pc_meanT_shuffled = pca_1st.transform(dataT_avg_shuffled.T).T
    
        evr_1st = pca_1st.explained_variance_ratio_
        
        dataT_3pc_mean = []
        dataT_3pc_mean_shuffled = []
        
        for i in range(len(subConditions)):
            b1, b2 = i*len(tsliceRange), (i+1)*len(tsliceRange)
            dataT_3pc_mean += [dataT_3pc_meanT[:,b1:b2]]
            dataT_3pc_mean_shuffled += [dataT_3pc_meanT_shuffled[:,b1:b2]]
            
        dataT_3pc_mean = np.array(dataT_3pc_mean) # reshape as (4*3) * 3pcs * nt
        dataT_3pc_mean_shuffled = np.array(dataT_3pc_mean_shuffled)
        
        dataT_3pc = []
        dataT_3pc_shuffled = []
        for trial in range(dataT.shape[0]):
            dataT_3pc += [pca_1st.transform(dataT[trial,:,:].T).T]
            dataT_3pc_shuffled += [pca_1st.transform(dataT_shuffled[trial,:,:].T).T]
        
        dataT_3pc = np.array(dataT_3pc)
        dataT_3pc_shuffled = np.array(dataT_3pc_shuffled)
        
        # test plots
        #for pc in range(3):
        #    plt.figure()
        #    for i in range(len(subConditions)):
        #        plt.plot(tsliceRange, dataT_3pc_mean[i,pc,:])
        #    plt.title(f'PC{pc}')
        #    plt.show()
        
        
        for cp in checkpoints:
            t1 = tsliceRange.tolist().index(cp-avgInterval) if cp-avgInterval >= tsliceRange.min() else tsliceRange.tolist().index(tsliceRange.min())
            t2 = tsliceRange.tolist().index(cp+avgInterval) if cp+avgInterval <= tsliceRange.max() else tsliceRange.tolist().index(tsliceRange.max())
            
            tempX = dataT_3pc[:,:,t1:t2].mean(axis=2) if t1!=t2 else dataT_3pc[:,:,t1]
            tempX_shuffled = dataT_3pc_shuffled[:,:,t1:t2].mean(axis=2) if t1!=t2 else dataT_3pc_shuffled[:,:,t1]
            
            tempX_mean = dataT_3pc_mean[:,:,t1:t2].mean(axis=2) if t1!=t2 else dataT_3pc_mean[:,:,t1]
            tempX_mean_shuffled = dataT_3pc_mean_shuffled[:,:,t1:t2].mean(axis=2) if t1!=t2 else dataT_3pc_mean_shuffled[:,:,t1]
            
            for tt in ttypes:
                
                idx = trialInfo[(trialInfo.type == tt)].index.tolist()
                
                trialInfoT = trialInfo[(trialInfo.type == tt)].reset_index(drop=True)
                tempX_tt = tempX[idx,:]
                tempX_tt_shuffled = tempX_shuffled[idx,:]
                
                tempX1_mean = []
                tempX2_mean = []
                
                for l in locs:
                    
                    conx1 = [subConditions.index(sc) for sc in subConditions if (sc[0][0] == l and sc[1] == tt)]
                    conx2 = [subConditions.index(sc) for sc in subConditions if (sc[0][1] == l and sc[1] == tt)]
                
                    tempX1_mean += [tempX_mean[conx1,:].mean(axis = 0)]
                    tempX2_mean += [tempX_mean[conx2,:].mean(axis = 0)]
                    
                    #tempX3 += [temp3[conx2,:].mean(axis = 0)] if tt == 1 else [temp3[conx1,:].mean(axis = 0)]
                
                tempX1_mean = np.array(tempX1_mean)
                #tempX1 = tempX1 - tempX1.mean()
                tempX2_mean = np.array(tempX2_mean)
                #tempX2 = tempX2 - tempX2.mean()
                
                ### loc1 2nd pca
                pca_2nd_1 = PCA(n_components=2)
                pca_2nd_1.fit(tempX1_mean)# - tempX1_mean.mean()
                vecs1, evr2_1 = pca_2nd_1.components_, pca_2nd_1.explained_variance_ratio_
                vec_normal1 = np.cross(vecs1[0], vecs1[1])
                
                center1 = tempX1_mean.mean(axis=0)
                
                proj1 = np.array([proj_on_plane(p1, vec_normal1, center1) for p1 in tempX1_mean])
                #project_to_plane(tempX1_mean, vecs1)
                
                #x1_plane, y1_plane, z1_plane = plane_by_vecs(vecs1, center = center1, xRange=(tempX1_mean[:,0].min(), tempX1_mean[:,0].max()), yRange=(tempX1_mean[:,1].min(), tempX1_mean[:,1].max()))
                x1_plane, y1_plane, z1_plane = plane_by_vecs(vecs1, center = center1, xRange=(proj1[:,0].min(), proj1[:,0].max()), yRange=(proj1[:,1].min(), proj1[:,1].max()))
                #, scaleX=scaleX1, scaleY=scaleY1
                # single trials
                tempX1_pca = pca_2nd_1.transform(tempX_tt) # 2nd pca transformed data
                tempX1_proj = np.array([proj_on_plane(p1, vec_normal1, center1) for p1 in tempX_tt]) # projections on the 2nd pca plane
                
                
                
                ### loc2 2nd pca
                pca_2nd_2 = PCA(n_components=2)
                pca_2nd_2.fit(tempX2_mean)# - tempX2_mean.mean()
                vecs2, evr2_2 = pca_2nd_2.components_, pca_2nd_2.explained_variance_ratio_
                vec_normal2 = np.cross(vecs2[0], vecs2[1])
                
                center2 = tempX2_mean.mean(axis=0)
                
                proj2 = np.array([proj_on_plane(p2, vec_normal2, center2) for p2 in tempX2_mean])
                #project_to_plane(tempX2_mean, vecs2)
                
                #x2_plane, y2_plane, z2_plane = plane_by_vecs(vecs2, center = center2, xRange=(tempX2_mean[:,0].min(), tempX2_mean[:,0].max()), yRange=(tempX2_mean[:,1].min(), tempX2_mean[:,1].max()))
                x2_plane, y2_plane, z2_plane = plane_by_vecs(vecs2, center = center2, xRange=(proj2[:,0].min(), proj2[:,0].max()), yRange=(proj2[:,1].min(), proj2[:,1].max()))
                #, scaleX=scaleX2, scaleY=scaleY2
                # single trials
                tempX2_pca = pca_2nd_2.transform(tempX_tt) # 2nd pca transformed data
                tempX2_proj = np.array([proj_on_plane(p2, vec_normal2, center2) for p2 in tempX_tt]) # projections on the 2nd pca plane
                
                
                
                ### angle between two planes
                cos_theta = angle_btw_planes(vec_normal1, vec_normal2)
                theta = np.degrees(np.arccos(cos_theta))
                
                # store the normal vectors for plane Loc1 and Loc2 at each time
                vecs_normal[region][cp][tt][1] += [vec_normal1]
                vecs_normal[region][cp][tt][2] += [vec_normal2]
                
                ### decodability 
                train_pca, test_pca = split_set(tempX_tt) #, ranseed = n
                #train_pca, test_pca = np.array(idx)[train_pca], np.array(idx)[test_pca]
                
                train_label1, train_label2 = trialInfoT.loc[train_pca,'loc1'], trialInfoT.loc[train_pca,'loc2']
                test_label1, test_label2 = trialInfoT.loc[test_pca,'loc1'], trialInfoT.loc[test_pca,'loc2']
                
                # loc1/loc2 by 1st pca
                performance1_pca1 = f_decoding.LDAPerformance(tempX_tt[train_pca,:], tempX_tt[test_pca,:], train_label1, test_label1)
                performance2_pca1 = f_decoding.LDAPerformance(tempX_tt[train_pca,:], tempX_tt[test_pca,:], train_label2, test_label2)
                
                # loc1/loc2 by 2nd pca
                performance1_pca2 = f_decoding.LDAPerformance(tempX1_pca[train_pca,:], tempX1_pca[test_pca,:], train_label1, test_label1)
                performance2_pca2 = f_decoding.LDAPerformance(tempX2_pca[train_pca,:], tempX2_pca[test_pca,:], train_label2, test_label2)
                
                # loc1/loc2 by 1nd pca
                performance1_pcaProj = f_decoding.LDAPerformance(tempX1_proj[train_pca,:], tempX1_proj[test_pca,:], train_label1, test_label1)
                performance2_pcaProj = f_decoding.LDAPerformance(tempX2_proj[train_pca,:], tempX2_proj[test_pca,:], train_label2, test_label2)
                
                decodability_pca2[region][cp][tt][1] += [performance1_pca2]
                decodability_pca2[region][cp][tt][2] += [performance2_pca2]
                decodability_proj[region][cp][tt][1] += [performance1_pcaProj]
                decodability_proj[region][cp][tt][2] += [performance2_pcaProj]
                
                
                
                ### run only if to plot
                ### plot the best-fitting plane
                # find vertices, then sort vertices according to the shortest path - so that plotted plane will be a quadrilateral
                # only plot in the selected iteration
                if n == toPlot :
                    
                    color1 = 'r'
                    color2 = 'g'
                    #color3 = 'b'
                    
                    #colors = np.array([color1, color2, color3])
                    
                    fig = plt.figure(figsize=(8, 6))
                    ax = fig.add_subplot(111, projection='3d')
                    
                    shapes = ('o','s','*','^')
                    for i in range(len(tempX1_mean)):
                        ax.scatter(tempX1_mean[i,0], tempX1_mean[i,1], tempX1_mean[i,2], marker = shapes[i], color = color1, alpha = 0.7, label = f'loc1, {i}')
                        ax.scatter(proj1[i,0], proj1[i,1], proj1[i,2], marker = shapes[i], color = color1, alpha = 0.2)#, label = f'loc1, {i}'
                        
                        if cp >= slice_epochsDic['s2'][0]:
                            ax.scatter(tempX2_mean[i,0], tempX2_mean[i,1], tempX2_mean[i,2], marker = shapes[i], color = color2, alpha = 0.7, label = f'loc2, {i}')
                            ax.scatter(proj2[i,0], proj2[i,1], proj2[i,2], marker = shapes[i], color = color2, alpha = 0.2)#, label = f'loc2, {i}'
                        #ax.scatter(tempX3[i,0], tempX3[i,1], tempX3[i,2], marker = shapes[i], color = color3, label = f'locChoice, {i}')
                    
                    #ax.scatter(tempX1[:,0], tempX1[:,1], tempX1[:,2], marker = '.', color = color1, alpha = 0.3)
                    #ax.scatter(tempX1_proj[:,0], tempX1_proj[:,1], tempX1_proj[:,2], marker = '.', color = color1, alpha = 0.1)#, label = f'loc1, {i}'
                    #ax.scatter(tempX2_proj[:,0], tempX2_proj[:,1], tempX2_proj[:,2], marker = '.', color = color2, alpha = 0.1)#, label = f'loc1, {i}'
                    
                    ax.plot_surface(x1_plane, y1_plane, z1_plane, alpha=0.5, color = color1)
                    #ax.add_collection3d(Poly3DCollection([sorted_verts1], facecolor=color1, edgecolor=[], alpha=0.2))#
                    #ax.quiver(x1_plane.mean(), y1_plane.mean(), z1_plane.mean(), vec_normal1[0], vec_normal1[1], vec_normal1[2], color = color1, alpha = 0.2)#, arrow_length_ratio = 0.001
                    #ax.quiver(x1_plane.mean(), y1_plane.mean(), z1_plane.mean(), vecs1[0,0], vecs1[0,1], vecs1[0,2], color = color1, alpha = 1)
                    #ax.quiver(x1_plane.mean(), y1_plane.mean(), z1_plane.mean(), vecs1[1,0], vecs1[1,1], vecs1[1,2], color = color1, alpha = 1)
                    ax.quiver(tempX1_mean[:,0].mean(),tempX1_mean[:,1].mean(),tempX1_mean[:,2].mean(), vec_normal1[0], vec_normal1[1], vec_normal1[2], color = color1, alpha = 0.2)#, arrow_length_ratio = 0.001
                    #ax.quiver(tempX1_mean[:,0].mean(),tempX1_mean[:,1].mean(),tempX1_mean[:,2].mean(), vecs1[0,0], vecs1[0,1], vecs1[0,2], color = color1, alpha = 1)
                    #ax.quiver(tempX1_mean[:,0].mean(),tempX1_mean[:,1].mean(),tempX1_mean[:,2].mean(), vecs1[1,0], vecs1[1,1], vecs1[1,2], color = color1, alpha = 1)
                    #ax.text(x1_plane.min(),y1_plane.min(),z1_plane.min(),f'Loc1 EVR:{evr2_1[0]:.4f}; {evr2_1[1]:.4f}')
                    
                    if cp >= slice_epochsDic['s2'][0]:
                        
                        ax.plot_surface(x2_plane, y2_plane, z2_plane, alpha=0.5, color = color2)
                        #ax.add_collection3d(Poly3DCollection([sorted_verts2], facecolor=color2, edgecolor=[], alpha=0.2))#
                        #ax.quiver(x2_plane.mean(), y2_plane.mean(), z2_plane.mean(), vec_normal2[0], vec_normal2[1], vec_normal2[2], color = color2, alpha = 0.2)#, arrow_length_ratio = 0.001
                        #ax.quiver(x2_plane.mean(), y2_plane.mean(), z2_plane.mean(), vecs2[0,0], vecs2[0,1], vecs2[0,2], color = color2, alpha = 1)
                        #ax.quiver(x2_plane.mean(), y2_plane.mean(), z2_plane.mean(), vecs2[1,0], vecs2[1,1], vecs2[1,2], color = color2, alpha = 1)
                        ax.quiver(tempX2_mean[:,0].mean(),tempX2_mean[:,1].mean(),tempX2_mean[:,2].mean(), vec_normal2[0], vec_normal2[1], vec_normal2[2], color = color2, alpha = 0.2)#, arrow_length_ratio = 0.001
                        #ax.quiver(tempX2_mean[:,0].mean(),tempX2_mean[:,1].mean(),tempX2_mean[:,2].mean(), vecs2[0,0], vecs2[0,1], vecs2[0,2], color = color2, alpha = 1)
                        #ax.quiver(tempX2_mean[:,0].mean(),tempX2_mean[:,1].mean(),tempX2_mean[:,2].mean(), vecs2[1,0], vecs2[1,1], vecs2[1,2], color = color2, alpha = 1)
                        #ax.text(x2_plane.min(),y2_plane.min(),z2_plane.min(),f'Loc2 EVR:{evr2_2[0]:.4f}; {evr2_2[1]:.4f}')
                    
                    
                    ax.set_xlabel(f'PC1 ({evr_1st[0]:.4f})')
                    ax.set_ylabel(f'PC2 ({evr_1st[1]:.4f})')
                    ax.set_zlabel(f'PC3 ({evr_1st[2]:.4f})')
                    
                    #ax.view_init(elev=45, azim=45, roll=0)
                    
                    plt.legend()
                    plt.title(f'{region}, t = {cp}, type = {tt}, cosTheta = {cos_theta:.3f}, loc1% = {performance1_pcaProj:.2f}; loc2% = {performance2_pcaProj:.2f}')
                    plt.show()
                    