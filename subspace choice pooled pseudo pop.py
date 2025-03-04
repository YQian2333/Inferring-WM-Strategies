# -*- coding: utf-8 -*-
"""
Created on Sat Jun 29 00:23:18 2024

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

# In[] decode from pseudo population
pd.options.mode.chained_assignment = None
epsilon = 0.0000001
# In[] decodability with/without permutation P value
bins = 50 # dt #
tslice = (-300,3000)
tsliceRange = np.arange(-300,3000,dt)
slice_epochsDic = {'bsl':[-300,0],'s1':[0,300],'d1':[300,1300],'s2':[1300,1600],'d2':[1600,2600],'go':[2600,3000]}





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

choice_tRange = (2100,2600)
toplot_samples = np.arange(0,1,1)

# smooth to 50ms bins
bins = 50
dt = dt
tbins = np.arange(tslice[0], tslice[1], bins)
epsilon = 0.0000001
# In[]

################################################
########## choice subspace trajectory ##########
################################################    
    
#%% initialization choice subspace

vecs_C = {}
projs_C = {}
projsAll_C = {}
#Xs_mean_C = {}
trialInfos_C = {}
data_3pc_C = {}
pca1s_C = {}


vecs_C_shuff = {}
projs_C_shuff = {}
projsAll_C_shuff = {}
#Xs_mean_C_shuff = {}
trialInfos_C_shuff = {}
data_3pc_C_shuff = {}
pca1s_C_shuff = {}

for region in ('dlpfc','fef'):
    vecs_C[region] = []
    projs_C[region] = []
    projsAll_C[region] = []
    #Xs_mean_C[region] = []
    trialInfos_C[region] = []
    data_3pc_C[region] = []
    pca1s_C[region] = []
    
    vecs_C_shuff[region] = []
    projs_C_shuff[region] = []
    projsAll_C_shuff[region] = []
    #Xs_mean_C_shuff[region] = []
    trialInfos_C_shuff[region] = []
    data_3pc_C_shuff[region] = []
    pca1s_C_shuff[region] = []


evrs = {'dlpfc':np.zeros((nIters, 3)), 'fef':np.zeros((nIters, 3))}
#n = 50

#%% estimate choice subspace geoms
#while n < 100:
for n in range(nIters):
    t_IterOn = time.time()
    print(f'Iter = {n}')
    
    #pseudo_TrialInfo = f_pseudoPop.pseudo_Session(locCombs, ttypes, samplePerCon=samplePerCon, sampleRounds=sampleRounds)
    #pseudo_region = f_pseudoPop.pseudo_PopByRegion_pooled(sessions, cellsToUse, monkey_Info, locCombs = locCombs, ttypes = ttypes, samplePerCon = samplePerCon, sampleRounds=sampleRounds, arti_noise_level = arti_noise_level)
    
    #pseudo_data = {'pseudo_TrialInfo':pseudo_TrialInfo, 'pseudo_region':pseudo_region}
    #np.save(save_path + f'/pseudo_all{n}.npy', pseudo_data, allow_pickle=True)

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
            dataN[:,ch,:] = (dataN[:,ch,:] - dataN[:,ch,:].mean()) / (dataN[:,ch,:].std()+epsilon) #standard scaler
            #dataN[:,ch,:] = (dataN[:,ch,:] - dataN[:,ch,:].mean(axis=0)) / dataN[:,ch,:].std(axis=0) # detrened standard scaler
            #dataN[:,ch,:] = (dataN[:,ch,:] - dataN[:,ch,idxxBsl[0]:idxxBsl[1]].mean())/dataN[:,ch,idxxBsl[0]:idxxBsl[1]].std() # Z to baseline, 
        
        # append to store each iteration separately
        vecs_C[region].append([])
        projs_C[region].append([])
        projsAll_C[region].append([])
        #Xs_mean_C[region].append([])
        trialInfos_C[region].append([])
        data_3pc_C[region].append([])
        pca1s_C[region].append([])
        
        vecs_C_shuff[region].append([])
        projs_C_shuff[region].append([])
        projsAll_C_shuff[region].append([])
        #Xs_mean_C_shuff[region].append([])
        trialInfos_C_shuff[region].append([])
        data_3pc_C_shuff[region].append([])
        pca1s_C_shuff[region].append([])
        
        
        
        for nboot in range(nBoots):
            tplt = True if (nboot == 0 and n in toplot_samples) else False
            #idxT,_ = f_subspace.split_set(dataN, frac=fracBoots, ranseed=nboot)
            idxT,_ = f_subspace.split_set(np.arange(dataN.shape[0]), frac=fracBoots, ranseed=nboot)
            dataT = dataN[idxT,:,:]
            trialInfoT = trialInfo.loc[idxT,:].reset_index(drop=True)
            
            vecs_CT, projs_CT, projsAll_CT, _, trialInfo_CT, data_3pc_CT, _, evr_1stT, pca1_CT, _ = f_subspace.planeC_fitting_analysis(dataT, trialInfoT, pca_tWinX, tsliceRange, choice_tRange, locs, ttypes, dropCombs, 
                                                                                                                                              toPlot=tplt, avgMethod = avgMethod, region_label=f'{region.upper()}', 
                                                                                                                                              plot_traj=True, traj_checkpoints=(1300,2600), traj_start=1300, traj_end=2600,
                                                                                                                                              plot3d=False,savefig=False, save_path=save_path, plotlayout = (2,3,0,1),
                                                                                                                                              hideLocs=(0,2,),legend_on=False) #, decode_method = decode_method
            
            #decodability_projD
            
            # smooth to 50ms bins
            ntrialT, ncellT, ntimeT = data_3pc_CT.shape
            data_3pc_CT_smooth = np.mean(data_3pc_CT.reshape(ntrialT, ncellT, int(ntimeT/(bins/dt)),int(bins/dt)),axis = 3)
            
            vecs_C[region][n] += [vecs_CT]
            projs_C[region][n] += [projs_CT]
            projsAll_C[region][n] += [projsAll_CT]
            #Xs_mean_C[region][n] += [X_meanT]
            trialInfos_C[region][n] += [trialInfo_CT]
            data_3pc_C[region][n] += [data_3pc_CT_smooth]
            pca1s_C[region][n] += [pca1_CT]
            
            evrs[region][n,:] = evr_1stT
            
            print(f'EVRs: {evr_1stT.round(5)}')
            
            
            for nperm in range(nPerms):
                
                
                trialInfoT_shuff = trialInfoT.copy().sample(frac=1).reset_index(drop=True) # just use default method
                
                vecs_CT_shuff, projs_CT_shuff, projsAll_CT_shuff, _, trialInfo_CT_shuff, data_3pc_CT_shuff, _, _, pca1_CT_shuff, _ = f_subspace.planeC_fitting_analysis(dataT, trialInfoT_shuff, pca_tWinX, tsliceRange, choice_tRange, locs, ttypes, dropCombs, 
                                                                                                                                                                     toPlot=False, avgMethod = avgMethod, adaptPCA=pca1_CT)
                
                # smooth to 50ms bins
                ntrialT, ncellT, ntimeT = data_3pc_CT_shuff.shape
                data_3pc_CT_smooth_shuff = np.mean(data_3pc_CT_shuff.reshape(ntrialT, ncellT, int(ntimeT/(bins/dt)),int(bins/dt)),axis = 3)
                
                vecs_C_shuff[region][n] += [vecs_CT_shuff]
                projs_C_shuff[region][n] += [projs_CT_shuff]
                projsAll_C_shuff[region][n] += [projsAll_CT_shuff]
                #Xs_mean_C_shuff[region][n] += [X_meanT_shuff]
                trialInfos_C_shuff[region][n] += [trialInfo_CT_shuff]
                data_3pc_C_shuff[region][n] += [data_3pc_CT_smooth_shuff]
                pca1s_C_shuff[region][n] += [pca1_CT_shuff]
                
                            
            
            
        print(f'tPerm({nPerms*nBoots}) = {(time.time() - t_IterOn):.4f}s, {region}')
            

# In[] save 
np.save(f'{phd_path}/fitting planes/pooled/w&w/' + 'vecs_C_detrended.npy', vecs_C, allow_pickle=True)
np.save(f'{phd_path}/fitting planes/pooled/w&w/' + 'projs_C_detrended.npy', projs_C, allow_pickle=True)
np.save(f'{phd_path}/fitting planes/pooled/w&w/' + 'projsAll_C_detrended.npy', projsAll_C, allow_pickle=True)
#np.save(f'{phd_path}/fitting planes/pooled/w&w/' + 'Xs_mean_C_detrended.npy', Xs_mean_C, allow_pickle=True)
np.save(f'{phd_path}/fitting planes/pooled/w&w/' + 'trialInfos_C_detrended.npy', trialInfos_C, allow_pickle=True)
np.save(f'{phd_path}/fitting planes/pooled/w&w/' + 'data_3pc_C_detrended.npy', data_3pc_C, allow_pickle=True)
np.save(f'{phd_path}/fitting planes/pooled/w&w/' + 'pca1s_C_detrended.npy', pca1s_C, allow_pickle=True)


np.save(f'{phd_path}/fitting planes/pooled/w&w/' + 'vecs_C_shuff_detrended.npy', vecs_C_shuff, allow_pickle=True)
np.save(f'{phd_path}/fitting planes/pooled/w&w/' + 'projs_C_shuff_detrended.npy', projs_C_shuff, allow_pickle=True)
np.save(f'{phd_path}/fitting planes/pooled/w&w/' + 'projsAll_C_shuff_detrended.npy', projsAll_C_shuff, allow_pickle=True)
#np.save(f'{phd_path}/fitting planes/pooled/w&w/' + 'Xs_mean_C_shuff_detrended.npy', Xs_mean_C_shuff, allow_pickle=True)
np.save(f'{phd_path}/fitting planes/pooled/w&w/' + 'trialInfos_C_shuff_detrended.npy', trialInfos_C_shuff, allow_pickle=True)
np.save(f'{phd_path}/fitting planes/pooled/w&w/' + 'data_3pc_C_shuff_detrended.npy', data_3pc_C_shuff, allow_pickle=True)
np.save(f'{phd_path}/fitting planes/pooled/w&w/' + 'pca1s_C_shuff_detrended.npy', pca1s_C_shuff, allow_pickle=True)

# In[] load choice plan vectors
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

# In[] load item-specific plane vectors
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
# In[] choice subspace vs item subspaces
pdummy = True #False #
nIters = 100
nPerms = 100
nBoots = 1
cosTheta_1C, cosTheta_2C = {},{}
cosPsi_1C, cosPsi_2C = {},{}

# shuff
cosTheta_1C_shuff, cosTheta_2C_shuff = {},{}
cosPsi_1C_shuff, cosPsi_2C_shuff = {},{}

for region in ('dlpfc','fef'):
    
    cosTheta_1C[region], cosTheta_2C[region] = {},{}
    cosPsi_1C[region], cosPsi_2C[region] = {},{}
    
    cosTheta_1C_shuff[region], cosTheta_2C_shuff[region] = {},{}
    cosPsi_1C_shuff[region], cosPsi_2C_shuff[region] = {},{}
    
    for tt in ttypes:
        cosTheta_1CT = np.zeros((nIters, nBoots, len(checkpoints)))
        cosTheta_2CT = np.zeros((nIters, nBoots, len(checkpoints)))
        cosPsi_1CT = np.zeros((nIters, nBoots, len(checkpoints)))
        cosPsi_2CT = np.zeros((nIters, nBoots, len(checkpoints)))
        
    
        cosTheta_1C_shuffT = np.zeros((nIters, nPerms, len(checkpoints)))
        cosTheta_2C_shuffT = np.zeros((nIters, nPerms, len(checkpoints)))
        cosPsi_1C_shuffT = np.zeros((nIters, nPerms, len(checkpoints)))
        cosPsi_2C_shuffT = np.zeros((nIters, nPerms, len(checkpoints)))
        
        for n in range(nIters):
            if n%20==0:
                print(n)
                
            for nbt in range(nBoots):
                for nc, cp in enumerate(checkpoints):
                    cT1C, _, cP1C, _, _ = f_subspace.angle_alignment_coplanar(vecs[region][cp][tt][1][n][nbt], projs[region][cp][tt][1][n][nbt], vecs_C[region][n][nbt], projs_C[region][n][nbt])
                    cT2C, _, cP2C, _, _ = f_subspace.angle_alignment_coplanar(vecs[region][cp][tt][2][n][nbt], projs[region][cp][tt][2][n][nbt], vecs_C[region][n][nbt], projs_C[region][n][nbt])
                    
                    cosTheta_1CT[n,nbt,nc], cosTheta_2CT[n,nbt,nc] = cT1C, cT2C# theta11, theta22, theta12# 
                    cosPsi_1CT[n,nbt,nc], cosPsi_2CT[n,nbt,nc] = cP1C, cP2C# psi11, psi22, psi12# 
                    
            
            for npm in range(nPerms):
                    for nc, cp in enumerate(checkpoints):
                        
                        cT1C_shuff, _, cP1C_shuff, _, _ = f_subspace.angle_alignment_coplanar(vecs_shuff[region][cp][tt][1][n][npm], projs_shuff[region][cp][tt][1][n][npm], 
                                                                                  vecs_C[region][n][0], projs_C[region][n][0])
                        cT2C_shuff, _, cP2C_shuff, _, _ = f_subspace.angle_alignment_coplanar(vecs_shuff[region][cp][tt][2][n][npm], projs_shuff[region][cp][tt][2][n][npm], 
                                                                                  vecs_C[region][n][0], projs_C[region][n][0])
                        
                        
                        cosTheta_1C_shuffT[n,npm,nc], cosTheta_2C_shuffT[n,npm,nc] = cT1C_shuff, cT2C_shuff
                        cosPsi_1C_shuffT[n,npm,nc], cosPsi_2C_shuffT[n,npm,nc] = cP1C_shuff, cP2C_shuff
                        
        cosTheta_1C[region][tt] = cosTheta_1CT
        cosTheta_2C[region][tt] = cosTheta_2CT
        cosPsi_1C[region][tt] = cosPsi_1CT
        cosPsi_2C[region][tt] = cosPsi_2CT
        
        cosTheta_1C_shuff[region][tt] = cosTheta_1C_shuffT
        cosTheta_2C_shuff[region][tt] = cosTheta_2C_shuffT
        cosPsi_1C_shuff[region][tt] = cosPsi_1C_shuffT
        cosPsi_2C_shuff[region][tt] = cosPsi_2C_shuffT
                
        

# In[] save
np.save(f'{phd_path}/outputs/monkeys/' + 'cosTheta_1Read_data.npy', cosTheta_1C, allow_pickle=True)
np.save(f'{phd_path}/outputs/monkeys/' + 'cosTheta_2Read_data.npy', cosTheta_2C, allow_pickle=True)
np.save(f'{phd_path}/outputs/monkeys/' + 'cosPsi_1Read_data.npy', cosPsi_1C, allow_pickle=True)
np.save(f'{phd_path}/outputs/monkeys/' + 'cosPsi_2Read_data.npy', cosPsi_2C, allow_pickle=True)

# In[] load

cosTheta_1C = np.load(f'{phd_path}/outputs/monkeys/' + 'cosTheta_1Read_data.npy', allow_pickle=True).item()
cosTheta_2C = np.load(f'{phd_path}/outputs/monkeys/' + 'cosTheta_2Read_data.npy', allow_pickle=True).item()
cosPsi_1C = np.load(f'{phd_path}/outputs/monkeys/' + 'cosPsi_1Read_data.npy', allow_pickle=True).item()
cosPsi_2C = np.load(f'{phd_path}/outputs/monkeys/' + 'cosPsi_2Read_data.npy', allow_pickle=True).item()


# In[] plot item v readout                        
for region in ('dlpfc','fef'):
    
    angleCheckPoints = np.linspace(0,np.pi,7).round(5)
    color1, color2 = 'b', 'm'
    
    ### cosTheta
    fig, axes = plt.subplots(2,2, figsize=(8,6), dpi=300, sharex=True, sharey=True)
    
    for tt in ttypes:
        
        ttype = 'Retarget' if tt == 1 else 'Distraction'
        
        
        ax = axes.flatten()[tt-1]
        
        # Item1
        ax.errorbar(np.arange(0, len(checkpoints), 1), cosTheta_1C[region][tt].mean(1).mean(0), yerr = cosTheta_1C[region][tt].mean(1).std(0), marker = 'o', color = color1, label = 'Item1', capsize=4, linewidth=2)
        #ax.plot(np.arange(0, len(checkpoints), 1), pcosTheta_1C[tt], alpha = 0.3, linestyle = '-', color = color1)
        
        # Item2
        ax.errorbar(np.arange(0, len(checkpoints), 1), cosTheta_2C[region][tt].mean(1).mean(0), yerr = cosTheta_2C[region][tt].mean(1).std(0), marker = 'o', color = color2, label = 'Item2', capsize=4, linewidth=2)
        #ax.plot(np.arange(0, len(checkpoints), 1), pcosTheta_2C[tt], alpha = 0.3, linestyle = '-', color = color2)
        
        #trans = ax.get_xaxis_transform()
        #for nc, cp in enumerate(checkpoints):
        #    if 0.05 < pcosTheta_1C[tt][nc] <= 0.1:
        #        ax.annotate('+', xy=(nc, 0.05), xycoords = trans, annotation_clip = False, ha = 'center', va = 'center', color = color1)
        #    elif 0.01 < pcosTheta_1C[tt][nc] <= 0.05:
        #        ax.annotate('*', xy=(nc, 0.05), xycoords = trans, annotation_clip = False, ha = 'center', va = 'center', color = color1)
        #    elif pcosTheta_1C[tt][nc] <= 0.01:
        #        ax.annotate('**', xy=(nc, 0.05), xycoords = trans, annotation_clip = False, ha = 'center', va = 'center', color = color1)
        
        #for nc, cp in enumerate(checkpoints):
        #    if 0.05 < pcosTheta_2C[tt][nc] <= 0.1:
        #        ax.annotate('+', xy=(nc, 0.05), xycoords = trans, annotation_clip = False, ha = 'center', va = 'center', color = color2)
        #    elif 0.01 < pcosTheta_2C[tt][nc] <= 0.05:
        #        ax.annotate('*', xy=(nc, 0.05), xycoords = trans, annotation_clip = False, ha = 'center', va = 'center', color = color2)
        #    elif pcosTheta_2C[tt][nc] <= 0.01:
        #        ax.annotate('**', xy=(nc, 0.05), xycoords = trans, annotation_clip = False, ha = 'center', va = 'center', color = color2)
                
        ax.set_title(f'{ttype}', pad = 10, fontsize=15)
        ax.set_xticks([n for n in range(len(checkpoints))])
        ax.set_xticklabels(checkpointsLabels, fontsize = 10)
        #ax.set_ylim((-1.1,1.1))
        ax.set_yticks(np.flip(np.cos(angleCheckPoints).round(5)), labels = np.flip(np.degrees(angleCheckPoints).round(1).astype(int)))
        ax.tick_params(axis='both', labelsize=12)
        #ax.set_xlabel('Timebin',fontsize=15)#, labelpad = 10
        
        if tt==1:
            ax.set_ylabel('cos(θ)',fontsize=15,rotation = 90)
        if tt==2:
            ax.legend(loc='lower right')#,bbox_to_anchor=(1, 0.6), fontsize=10
        
        
        
        ax = axes.flatten()[tt-1+2]
        
        ax.errorbar(np.arange(0, len(checkpoints), 1), cosPsi_1C[region][tt].mean(1).mean(0), yerr = cosPsi_1C[region][tt].mean(1).std(0), marker = 'o', color = color1, label = 'Item1', capsize=4, linewidth=2)
        #ax.plot(np.arange(0, len(checkpoints), 1), pcosPsi_1C[tt], alpha = 0.3, linestyle = '-', color = color1)
        
        ax.errorbar(np.arange(0, len(checkpoints), 1), cosPsi_2C[region][tt].mean(1).mean(0), yerr = cosPsi_2C[region][tt].mean(1).std(0), marker = 'o', color = color2, label = 'Item2', capsize=4, linewidth=2)
        #ax.plot(np.arange(0, len(checkpoints), 1), pcosPsi_2C[tt], alpha = 0.3, linestyle = '-', color = color2)
        
        
        #trans = ax.get_xaxis_transform()
        
        #for nc, cp in enumerate(checkpoints):
        #    if 0.05 < pcosPsi_1C[tt][nc] <= 0.1:
        #        ax.annotate('+', xy=(nc, 0.05), xycoords = trans, annotation_clip = False, ha = 'center', va = 'center', color = color1)
        #    elif 0.01 < pcosPsi_1C[tt][nc] <= 0.05:
        #        ax.annotate('*', xy=(nc, 0.05), xycoords = trans, annotation_clip = False, ha = 'center', va = 'center', color = color1)
        #    elif pcosPsi_1C[tt][nc] <= 0.01:
        #        ax.annotate('**', xy=(nc, 0.05), xycoords = trans, annotation_clip = False, ha = 'center', va = 'center', color = color1)
        
        #for nc, cp in enumerate(checkpoints):
        #    if 0.05 < pcosPsi_2C[tt][nc] <= 0.1:
        #        ax.annotate('+', xy=(nc, 0.05), xycoords = trans, annotation_clip = False, ha = 'center', va = 'center', color = color2)
        #    elif 0.01 < pcosPsi_2C[tt][nc] <= 0.05:
        #        ax.annotate('*', xy=(nc, 0.05), xycoords = trans, annotation_clip = False, ha = 'center', va = 'center', color = color2)
        #    elif pcosPsi_2C[tt][nc] <= 0.01:
        #        ax.annotate('**', xy=(nc, 0.05), xycoords = trans, annotation_clip = False, ha = 'center', va = 'center', color = color2)        
        
        #ax.set_title(f'{ttype}', pad = 10, fontsize=15)
        ax.set_xticks([n for n in range(len(checkpoints))])
        ax.set_xticklabels(checkpointsLabels, fontsize = 10)
        ax.set_ylim((-1.1,1.1))
        ax.set_yticks(np.flip(np.cos(angleCheckPoints).round(5)), labels = np.flip(np.degrees(angleCheckPoints).round(1).astype(int)))
        ax.tick_params(axis='both', labelsize=12)
        ax.set_xlabel('Timebin',fontsize=15)#, labelpad = 10
        
        if tt==1:
            ax.set_ylabel('cos(ψ)',fontsize=15,rotation = 90)
        #if tt==2:
        #    ax.legend(bbox_to_anchor=(1, 0.6), fontsize=10)#loc='lower right'
    
    plt.subplots_adjust(top = 0.8)
    plt.suptitle(f'Item vs. Readout Subspaces, {region.upper()}', fontsize = 20, y=1)
    plt.tight_layout()
    plt.show()  
    
    fig.savefig(f'{phd_path}/outputs/monkeys/item_v_readout_{region}.tif', bbox_inches='tight')
    














        
# In[] NON-USED plot 

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




# In[] NON-USED plot i1-i2-choice planes with mean projections, by ttypes
shapes = ('o','*','s','^')
colors = plt.get_cmap('Paired').colors#('r','b','g','m')

colorC = 'g'
labelC = 'Choice'

color1, color2 = 'b','m'
label1, label2 = 'Item1','Item2'

for n in (0,1,2,3,4):
    
    for region in ('dlpfc','fef'):
        
        # choice plane and loc_mean projs
        vecs_CT = vecs_C[region][n][0]
        projs_CT = projs_C[region][n][0]
        
        # create plane grids
        xC_planeT, yC_planeT, zC_planeT = f_subspace.plane_by_vecs(vecs_CT, center = projs_CT.mean(0), xRange=(projs_CT.min(0)[0], projs_CT.max(0)[0]), yRange=(projs_CT.min(0)[1], projs_CT.max(0)[1]))
        
        for tt in ttypes:
            
            ttype = 'Retarget' if tt==1 else 'Distraction'
            
            fig = plt.figure(figsize=(30, 20), dpi=100)
            
            for nc, cp in enumerate(checkpoints):
                
                # Item1 & 2 Plane
                vecs_1T, vecs_2T = vecs[region][cp][tt][1][n][0], vecs[region][cp][tt][2][n][0]
                projs_1T, projs_2T = projs[region][cp][tt][1][n][0], projs[region][cp][tt][2][n][0]
                
                x1_planeT, y1_planeT, z1_planeT = f_subspace.plane_by_vecs(vecs_1T, center = projs_1T.mean(0), xRange=(projs_1T.min(0)[0], projs_1T.max(0)[0]), yRange=(projs_1T.min(0)[1], projs_1T.max(0)[1]))
                x2_planeT, y2_planeT, z2_planeT = f_subspace.plane_by_vecs(vecs_2T, center = projs_2T.mean(0), xRange=(projs_2T.min(0)[0], projs_2T.max(0)[0]), yRange=(projs_2T.min(0)[1], projs_2T.max(0)[1]))
                
                ax = fig.add_subplot(2,3,nc+1, projection='3d')
                
                # choice plane
                ax.plot_surface(xC_planeT, yC_planeT, zC_planeT, alpha=0.3, color = colorC, label = labelC)
                ax.plot_surface(x1_planeT, y1_planeT, z1_planeT, alpha=0.3, color = color1, label = label1)
                ax.plot_surface(x2_planeT, y2_planeT, z2_planeT, alpha=0.3, color = color2, label = label2)
                
                for l in locs:
                    ax.scatter(projs_CT[l,0], projs_CT[l,1], projs_CT[l,2], marker = f'${l}$', color = colorC, alpha = 1, s = 100)#, label = f'loc1, {i}'
                    ax.scatter(projs_1T[l,0], projs_1T[l,1], projs_1T[l,2], marker = f'${l}$', color = color1, alpha = 1, s = 100)#, label = f'loc1, {i}'
                    ax.scatter(projs_2T[l,0], projs_2T[l,1], projs_2T[l,2], marker = f'${l}$', color = color2, alpha = 1, s = 100)#, label = f'loc1, {i}'
                
                
                
                ax.set_title(f'{checkpointsLabels[nc]}', fontsize=20)
                ax.set_xlabel(f'PC1', fontsize=20, labelpad=20)
                ax.tick_params(axis='x', labelsize=15)
                ax.set_ylabel(f'PC2', fontsize=20, labelpad=20)
                ax.tick_params(axis='y', labelsize=15)
                ax.set_zlabel(f'PC3', fontsize=20, labelpad=20)
                ax.tick_params(axis='z', labelsize=15)
                
                ax.legend(loc='upper right', fontsize=20)
                #ax.view_init(elev=0, azim=0, roll=0)
                
            plt.tight_layout()
            plt.subplots_adjust(top = 0.8)
            plt.suptitle(f'Subspace Relationship, {region.upper()}, {ttype}', fontsize = 30, y=0.85) #, {region}, cosTheta = {cos_theta:.3f}, loc1% = {performance1_pcaProj:.2f}; loc2% = {performance2_pcaProj:.2f}
            plt.show()
#%%

#################################
# readout subspace decodability #
#################################

# In[] cross temp decodability plane projection by lda

Y_columnsLabels = ['locKey','locs','type','loc1','loc2','locX']
toDecode_labels1 = 'loc1'
toDecode_labels2 = 'loc2'
shuff_excludeInv = False

nIters = 100
nBoots = 10
nPerms = nBoots#100

infoMethod = 'lda' #  'omega2' #
bins = 50
dt = dt

tslice = (-300,2700)
tbins = np.arange(tslice[0], tslice[1], bins)
#%% initialization
decode_proj1_3dX, decode_proj2_3dX = {},{}
decode_proj1_shuff_all_3dX, decode_proj2_shuff_all_3dX = {},{}

decode_proj1_3dW, decode_proj2_3dW = {},{}
decode_proj1_shuff_all_3dW, decode_proj2_shuff_all_3dW = {},{}

#%% calculate readout subspace decodability
for region in ('dlpfc','fef'):
    
    print(f'{region}')
    
    decode_proj1_3dX[region], decode_proj2_3dX[region] = {}, {}
    decode_proj1_shuff_all_3dX[region], decode_proj2_shuff_all_3dX[region] = {},{}
    
    decode_proj1_3dW[region], decode_proj2_3dW[region] = {}, {}
    decode_proj1_shuff_all_3dW[region], decode_proj2_shuff_all_3dW[region] = {},{}
    

    # estimate decodability by ttype
    for tt in ttypes:
        
        decode_proj1T_3dX = np.zeros((nIters, nBoots, len(tbins), len(tbins))) # pca1st 3d coordinates
        decode_proj2T_3dX = np.zeros((nIters, nBoots, len(tbins), len(tbins)))
        
        decode_proj1T_3dW = np.zeros((nIters, nBoots, len(tbins), )) # pca1st 3d coordinates
        decode_proj2T_3dW = np.zeros((nIters, nBoots, len(tbins), ))
        
        # shuff
        decode_proj1T_3d_shuffX = np.zeros((nIters, nBoots, len(tbins), len(tbins)))
        decode_proj2T_3d_shuffX = np.zeros((nIters, nBoots, len(tbins), len(tbins)))
        
        decode_proj1T_3d_shuffW = np.zeros((nIters, nBoots, len(tbins),))
        decode_proj2T_3d_shuffW = np.zeros((nIters, nBoots, len(tbins),))
        
        for n in range(nIters):
            #for nbt in range(nBoots):
            
            print(f'{n}')
            
            # trial info
            trialInfo_CT = trialInfos_C[region][n][0]
            trialInfo_CT_tt = trialInfo_CT[trialInfo_CT.type == tt]#.reset_index(drop = True)
            idx_tt = trialInfo_CT_tt.index.tolist()#.trial_index.values
            
            vecs_CT = vecs_C[region][n][0] # choice plane vecs
            projs_CT = projs_C[region][n][0] #
            
            

            vec_normal_CT = np.cross(vecs_CT[0],vecs_CT[1]) # plane normal vec
            center_CT = projs_CT.mean(0) # plane center
            
            #data_3pc_CT = data_3pc_C[region][n][nbt][idx_tt,:,:] # 3pc states from tt trials
            
            # smooth to 50ms bins
            #ntrialT, ncellT, ntimeT = data_3pc_CT.shape
            #data_3pc_CT_smooth = np.mean(data_3pc_CT.reshape(ntrialT, ncellT, int(ntimeT/(bins/dt)),int(bins/dt)),axis = 3)
            
            data_3pc_CT_smooth = data_3pc_C[region][n][0][idx_tt,:,:] # 3pc states from tt trials
            projs_All_CT = np.array([[f_subspace.proj_on_plane(p[:,t], vec_normal_CT, center_CT) for t in range(len(tbins))] for p in data_3pc_CT_smooth]) # projections on the plane
            projs_All_CT = np.swapaxes(projs_All_CT,1,2)
            
            # labels
            Y = trialInfo_CT_tt.loc[:,Y_columnsLabels].values
            ntrial = len(trialInfo_CT_tt)


            toDecode_X1 = Y_columnsLabels.index(toDecode_labels1)
            toDecode_X2 = Y_columnsLabels.index(toDecode_labels2)
            
            ### labels: ['locKey','locs','type','loc1','loc2','locX']
            full_label1 = Y[:,toDecode_X1].astype('int') #.astype('str') # locKey
            full_label2 = Y[:,toDecode_X2].astype('int') #.astype('str') # locKey
            
            #test_label1 = Y[test_setID,toDecode_X1].astype('int') #.astype('str') # locKey
            #test_label2 = Y[test_setID,toDecode_X2].astype('int') #.astype('str') # locKey
            
            if shuff_excludeInv:
                toDecode_labels1_inv = f_stats.inversed_label(Y_columnsLabels, toDecode_labels1, tt)
                toDecode_X1_inv = Y_columnsLabels.index(toDecode_labels1_inv)
                full_label1_inv = Y[:,toDecode_X1_inv]
                #test_label1_inv = Y[test_setID,toDecode_X1_inv]
                
                
                toDecode_labels2_inv = f_stats.inversed_label(Y_columnsLabels, toDecode_labels2, tt)
                toDecode_X2_inv = Y_columnsLabels.index(toDecode_labels2_inv)
                full_label2_inv = Y[:,toDecode_X2_inv]
                #test_label2_inv = Y[test_setID,toDecode_X2_inv]

                # except for the inverse ones
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
                
            for nbt in range(nBoots):
                ### split into train and test sets
                train_setID = np.sort(np.random.choice(ntrial, round(0.5*ntrial),replace = False))
                test_setID = np.sort(np.random.choice(np.setdiff1d(np.arange(ntrial), train_setID, assume_unique=True), (ntrial-len(train_setID)),replace = False))
                
                train_label1 = full_label1[train_setID] #.astype('str') # locKey
                train_label2 = full_label2[train_setID] #.astype('str') # locKey
                test_label1 = full_label1[test_setID] #.astype('str') # locKey
                test_label2 = full_label2[test_setID] #.astype('str') # locKey

                train_label1_shuff = full_label1_shuff[train_setID] #.astype('str') # locKey
                train_label2_shuff = full_label2_shuff[train_setID] #.astype('str') # locKey
                test_label1_shuff = full_label1_shuff[test_setID] #.astype('str') # locKey
                test_label2_shuff = full_label2_shuff[test_setID] #.astype('str') # locKey


                # cross temp decoding
                
                for t in range(len(tbins)):
                    for t_ in range(len(tbins)):
                        
                        info1_3d = f_decoding.LDAPerformance(projs_All_CT[train_setID,:,t],projs_All_CT[test_setID,:,t_], 
                                                             train_label1, test_label1)
                        info2_3d = f_decoding.LDAPerformance(projs_All_CT[train_setID,:,t],projs_All_CT[test_setID,:,t_], 
                                                             train_label2, test_label2)
                    
                        decode_proj1T_3dX[n,nbt,t,t_] = info1_3d #.mean(axis=-1).mean(axis=-1)
                        decode_proj2T_3dX[n,nbt,t,t_] = info2_3d #.mean(axis=-1).mean(axis=-1)

                        if t==t_:                            
                            decode_proj1T_3dW[n,nbt,t] = info1_3d #.mean(axis=-1).mean(axis=-1)
                            decode_proj2T_3dW[n,nbt,t] = info2_3d #.mean(axis=-1).mean(axis=-1)
                        
                    
                        # permutation null distribution
                        info1_3d_shuff = f_decoding.LDAPerformance(projs_All_CT[train_setID,:,t],projs_All_CT[test_setID,:,t_], 
                                                             train_label1_shuff, test_label1_shuff)
                        info2_3d_shuff = f_decoding.LDAPerformance(projs_All_CT[train_setID,:,t],projs_All_CT[test_setID,:,t_], 
                                                             train_label2_shuff, test_label2_shuff)
                        
                        decode_proj1T_3d_shuffX[n,nbt,t,t_] = info1_3d_shuff #.mean(axis=-1).mean(axis=-1)
                        decode_proj2T_3d_shuffX[n,nbt,t,t_] = info2_3d_shuff #.mean(axis=-1).mean(axis=-1)
                        
                        if t==t_:
                            decode_proj1T_3d_shuffW[n,nbt,t] = info1_3d_shuff #.mean(axis=-1).mean(axis=-1)
                            decode_proj2T_3d_shuffW[n,nbt,t] = info2_3d_shuff #.mean(axis=-1).mean(axis=-1)

                            
        decode_proj1_3dX[region][tt] = decode_proj1T_3dX
        decode_proj2_3dX[region][tt] = decode_proj2T_3dX
        decode_proj1_shuff_all_3dX[region][tt] = decode_proj1T_3d_shuffX
        decode_proj2_shuff_all_3dX[region][tt] = decode_proj2T_3d_shuffX

        decode_proj1_3dW[region][tt] = decode_proj1T_3dW
        decode_proj2_3dW[region][tt] = decode_proj2T_3dW
        decode_proj1_shuff_all_3dW[region][tt] = decode_proj1T_3d_shuffW
        decode_proj2_shuff_all_3dW[region][tt] = decode_proj2T_3d_shuffW
        
# In[] save within temp readout decodability
np.save(f'{phd_path}/outputs/monkeys/' + 'performanceW1_readout_data.npy', decode_proj1_3dW, allow_pickle=True)
np.save(f'{phd_path}/outputs/monkeys/' + 'performanceW2_readout_data.npy', decode_proj2_3dW, allow_pickle=True)
np.save(f'{phd_path}/outputs/monkeys/' + 'performanceW1_readout_shuff_data.npy', decode_proj1_shuff_all_3dW, allow_pickle=True)
np.save(f'{phd_path}/outputs/monkeys/' + 'performanceW2_readout_shuff_data.npy', decode_proj2_shuff_all_3dW, allow_pickle=True)
# In[] save cross temp readout decodability
np.save(f'{phd_path}/outputs/monkeys/' + 'performanceX1_readout_data.npy', decode_proj1_3dX, allow_pickle=True)
np.save(f'{phd_path}/outputs/monkeys/' + 'performanceX2_readout_data.npy', decode_proj2_3dX, allow_pickle=True)
np.save(f'{phd_path}/outputs/monkeys/' + 'performanceX1_readout_shuff_data.npy', decode_proj1_shuff_all_3dX, allow_pickle=True)
np.save(f'{phd_path}/outputs/monkeys/' + 'performanceX2_readout_shuff_data.npy', decode_proj2_shuff_all_3dX, allow_pickle=True)

# In[] load within temp readout decodability
decode_proj1_3dW = np.load(f'{phd_path}/outputs/monkeys/' + 'performanceW1_readout_data.npy', allow_pickle=True).item()
decode_proj2_3dW = np.load(f'{phd_path}/outputs/monkeys/' + 'performanceW2_readout_data.npy', allow_pickle=True).item()
decode_proj1_shuff_all_3dW = np.load(f'{phd_path}/outputs/monkeys/' + 'performanceW1_readout_shuff_data.npy', allow_pickle=True).item()
decode_proj2_shuff_all_3dW = np.load(f'{phd_path}/outputs/monkeys/' + 'performanceW2_readout_shuff_data.npy', allow_pickle=True).item()

#%% load cross temp readout decodability
decode_proj1_3dX = np.load(f'{phd_path}/outputs/monkeys/' + 'performanceX1_readout_data.npy', allow_pickle=True).item()
decode_proj2_3dX = np.load(f'{phd_path}/outputs/monkeys/' + 'performanceX2_readout_data.npy', allow_pickle=True).item()
decode_proj1_shuff_all_3dX = np.load(f'{phd_path}/outputs/monkeys/' + 'performanceX1_readout_shuff_data.npy', allow_pickle=True).item()
decode_proj2_shuff_all_3dX = np.load(f'{phd_path}/outputs/monkeys/' + 'performanceX2_readout_shuff_data.npy', allow_pickle=True).item()

#%%
#for region in ('dlpfc','fef'):
#    for tt in ttypes:
#        for n in range(nIters):
#            for t in range(len(tbins)):
#                for npm in range(nPerms):
#                    decode_proj1_shuff_all_3dW[region][tt][n,npm,t] = decode_proj1_shuff_all_3dX[region][tt][n,npm,t,t]
#                    decode_proj2_shuff_all_3dW[region][tt][n,npm,t] = decode_proj2_shuff_all_3dX[region][tt][n,npm,t,t]
#%% plot within temp readout decodability
for region in ('dlpfc','fef'):
           
    infoLabel = 'Accuracy' if infoMethod=='lda' else 'PEV'
    #plt.figure(figsize=(12, 8), dpi=100)
    #plt.figure(figsize=(12, 4.75), dpi=100)
    
    fig, axes = plt.subplots(1,2, figsize=(15,5), dpi=100, sharex=True, sharey=True)
    
    for tt in ttypes:
        
        condT = 'Retarget' if tt == 1 else 'Distraction'
        h0 = 0 if infoMethod == 'omega2' else 1/len(locs)
        
        #pPerms_decode1_3d = np.array([f_stats.permutation_p(decode_proj1_3dW[region][tt].mean(1).mean(0)[t], 
        ##                                                    decode_proj1_shuff_all_3dW[region][tt].mean(1)[:,t], tail='greater') for t in range(len(tbins))])
        #pPerms_decode2_3d = np.array([f_stats.permutation_p(decode_proj2_3dW[region][tt].mean(1).mean(0)[t], 
        #                                                    decode_proj2_shuff_all_3dW[region][tt].mean(1)[:,t], tail='greater') for t in range(len(tbins))])
        
        pPerms_decode1_3d = np.array([f_stats.permutation_pCI(decode_proj1_3dW[region][tt].mean(1)[:,t], 
                                                            decode_proj1_shuff_all_3dW[region][tt].mean(1)[:,t], 
                                                            alpha=5,tail='greater') for t in range(len(tbins))])
        pPerms_decode2_3d = np.array([f_stats.permutation_pCI(decode_proj2_3dW[region][tt].mean(1)[:,t], 
                                                            decode_proj2_shuff_all_3dW[region][tt].mean(1)[:,t], 
                                                            alpha=5,tail='greater') for t in range(len(tbins))])
        
        #pPerms_decode1_3d = np.array([stats.mannwhitneyu(decode_proj1_3dW[region][tt].mean(1)[:,t], 
        #                                                    np.concatenate(decode_proj1_shuff_all_3dW[region][tt])[:,t], alternative='greater')[1] for t in range(len(tbins))])
        #pPerms_decode2_3d = np.array([stats.mannwhitneyu(decode_proj2_3dW[region][tt].mean(1)[:,t], 
        #                                                    np.concatenate(decode_proj2_shuff_all_3dW[region][tt])[:,t], alternative='greater')[1] for t in range(len(tbins))])
        
        #pPerms_decode1_3d = np.array([stats.ttest_1samp(decode_proj1_3dW[region][tt].mean(1)[:,t], 0.25, alternative='greater')[1] for t in range(len(tbins))])
        #pPerms_decode2_3d = np.array([stats.ttest_1samp(decode_proj2_3dW[region][tt].mean(1)[:,t], 0.25, alternative='greater')[1] for t in range(len(tbins))])
        
        
        #plt.subplot(1,2,tt)
        #ax = plt.gca()
        ax = axes.flatten()[tt-1]
        
        ax.plot(np.arange(0, len(tbins), 1), decode_proj1_3dW[region][tt].mean(1).mean(0), color = 'b', label = 'Item1')
        ax.plot(np.arange(0, len(tbins), 1), decode_proj2_3dW[region][tt].mean(1).mean(0), color = 'm', label = 'Item2')
        ax.fill_between(np.arange(0, len(tbins), 1), (decode_proj1_3dW[region][tt].mean(1).mean(0) - decode_proj1_3dW[region][tt].mean(1).std(0)), (decode_proj1_3dW[region][tt].mean(1).mean(0) + decode_proj1_3dW[region][tt].mean(1).std(0)), color = 'b', alpha = 0.1)
        ax.fill_between(np.arange(0, len(tbins), 1), (decode_proj2_3dW[region][tt].mean(1).mean(0) - decode_proj2_3dW[region][tt].mean(1).std(0)), (decode_proj2_3dW[region][tt].mean(1).mean(0) + decode_proj2_3dW[region][tt].mean(1).std(0)), color = 'm', alpha = 0.1)
        
        # significance line
        segs1 = f_plotting.significance_line_segs(pPerms_decode1_3d,0.05)
        segs2 = f_plotting.significance_line_segs(pPerms_decode2_3d,0.05)
        
        for start1, end1 in segs1:
            ax.plot(np.arange(start1,end1,1), np.full_like(np.arange(start1,end1,1), 1.0, dtype='float'), color='b', linestyle='-', linewidth=2)
            
        for start2, end2 in segs2:
            ax.plot(np.arange(start2,end2,1), np.full_like(np.arange(start2,end2,1), 1.05, dtype='float'), color='m', linestyle='-', linewidth=2)

        
        # event lines
        for i in [0, 1300, 2600]:
            
            #ax.plot(tbins, np.full_like(tbins,list(tbins).index(i)), 'k-.', linewidth=4, alpha = 0.25)
            ax.plot(np.full_like(tbins,list(tbins).index(i)), tbins, 'k-.', linewidth=2, alpha = 0.25)
        
       #ax.set_title(f'{condT}, 3d', pad = 10)
        ax.set_title(f'{condT}', fontsize = 20, pad = 20)
        ax.set_xticks([list(tbins).index(i) for i in [0, 1300, 2600]])
        ax.set_xticklabels(['S1', 'S2', 'Go Cue'], fontsize = 10)
        ax.set_xlabel('Time', fontsize = 15)
        #ax.set_xlim((list(tbins).index(0),list(tbins).index(2600))) #(0,)
        ax.set_ylim((0,1.1))
        ax.set_xlim((0, len(tbins)))
        ax.tick_params(axis='both', labelsize=12)
        
        #ax.set_yticklabels(checkpoints, fontsize = 10)
        if tt==1:
            ax.set_ylabel(f'{infoLabel}', fontsize = 15)
        if tt==2:
            ax.legend(bbox_to_anchor=(1, 0.7), fontsize=15)
        
        
    plt.tight_layout()
    plt.subplots_adjust(top = 0.8)
    plt.suptitle(f'{region.upper()}, Readout Subspace', fontsize = 25, y=1)
    plt.show()
    
    #fig.savefig(f'{phd_path}/data/pseudo_ww/decodabilityW_readout_{region}.tif', bbox_inches='tight')

#%%  plot readout crosstemp decodability
infoMethod='lda'
for region in ('dlpfc','fef'):

    infoLabel = 'Accuracy' if infoMethod=='lda' else 'PEV'
    #plt.figure(figsize=(12, 8), dpi=100)
    fig = plt.figure(figsize=(28, 24), dpi=100)
    
    for tt in ttypes:
        
        condT = 'Retarget' if tt == 1 else 'Distraction'
        h0 = 0 if infoMethod == 'omega2' else 1/len(locs)
        
        pfm1, pfm2 = decode_proj1_3dX[region][tt].mean(1), decode_proj2_3dX[region][tt].mean(1)
        pfm1_shuff, pfm2_shuff = decode_proj1_shuff_all_3dX[region][tt].mean(1), decode_proj2_shuff_all_3dX[region][tt].mean(1)

        pPerms_decode1_3d = np.ones((len(tbins), len(tbins)))
        pPerms_decode2_3d = np.ones((len(tbins), len(tbins)))
        
        for t in range(len(tbins)):
            for t_ in range(len(tbins)):
                #pPerms_decode1_3d[t, t_] = f_stats.permutation_p(pfm1.mean(0)[t,t_], pfm1_shuff.mean(1)[:,t,t_], tail='greater')
                #pPerms_decode2_3d[t, t_] = f_stats.permutation_p(pfm2.mean(0)[t,t_], pfm2_shuff.mean(1)[:,t,t_], tail='greater')
                pPerms_decode1_3d[t, t_] = f_stats.permutation_pCI(pfm1[:,t,t_], pfm1_shuff[:,t,t_], tail='greater', alpha=5)
                pPerms_decode2_3d[t, t_] = f_stats.permutation_pCI(pfm2[:,t,t_], pfm2_shuff[:,t,t_], tail='greater', alpha=5)
                
        
        vmax = 0.6 if region == 'dlpfc' else 0.8
        
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
    plt.suptitle(f'{region.upper()}, Readout Subspace', fontsize = 35, y=1) #, Arti_Noise = {arti_noise_level}
    plt.show()
    
    fig.savefig(f'{phd_path}/outputs/monkeys/' + f'decodabilityX_readout_{region}.tif')

#%%

###########################################
# readout subspace decodability of ttypes #
###########################################

# In[] cross temp decodability plane projection by lda
Y_columnsLabels = ['locKey','locs','type','loc1','loc2','locX']
toDecode_labels1 = 'type'
#shuff_excludeInv = False

nIters = 100
nBoots = 10
nPerms = nBoots#100

infoMethod = 'lda' #  'omega2' #
bins = 50
dt = dt

tslice = (-300,2700)
tbins = np.arange(tslice[0], tslice[1], bins)
#%% ttype readout decodability
decode_ttX = {}
decode_ttX_shuff = {}
decode_ttW = {}
decode_ttW_shuff = {}

for region in ('dlpfc','fef'):
    
    print(f'{region}')
    
    decode_ttX[region] = {}
    decode_ttX_shuff[region] = {}
    decode_ttW[region] = {}
    decode_ttW_shuff[region] = {}


    # estimate decodability by ttype
    decode_ttXT = np.zeros((nIters, nBoots, len(tbins), len(tbins))) # pca1st 3d coordinates
    decode_ttWT = np.zeros((nIters, nBoots, len(tbins), ))
    
    # shuff
    decode_ttX_shuffT = np.zeros((nIters, nBoots, len(tbins), len(tbins)))
    decode_ttW_shuffT = np.zeros((nIters, nBoots, len(tbins),))
    
    for n in range(nIters):
        #for nbt in range(nBoots):
        
        print(f'{n}')
        
        # trial info
        trialInfo_CT = trialInfos_C[region][n][0]
        idx_tt = trialInfo_CT.index.tolist()#.trial_index.values
        
        vecs_CT = vecs_C[region][n][0] # choice plane vecs
        projs_CT = projs_C[region][n][0] #
        
        

        vec_normal_CT = np.cross(vecs_CT[0],vecs_CT[1]) # plane normal vec
        center_CT = projs_CT.mean(0) # plane center
        
        #data_3pc_CT = data_3pc_C[region][n][nbt][idx_tt,:,:] # 3pc states from tt trials
        
        # smooth to 50ms bins
        #ntrialT, ncellT, ntimeT = data_3pc_CT.shape
        #data_3pc_CT_smooth = np.mean(data_3pc_CT.reshape(ntrialT, ncellT, int(ntimeT/(bins/dt)),int(bins/dt)),axis = 3)
        
        data_3pc_CT_smooth = data_3pc_C[region][n][0][idx_tt,:,:] # 3pc states from tt trials
        projs_All_CT = np.array([[f_subspace.proj_on_plane(p[:,t], vec_normal_CT, center_CT) for t in range(len(tbins))] for p in data_3pc_CT_smooth]) # projections on the plane
        projs_All_CT = np.swapaxes(projs_All_CT,1,2)
        
        # labels
        Y = trialInfo_CT.loc[:,Y_columnsLabels].values
        ntrial = len(trialInfo_CT)


        toDecode_X1 = Y_columnsLabels.index(toDecode_labels1)
        
        ### labels: ['locKey','locs','type','loc1','loc2','locX']
        full_label1 = Y[:,toDecode_X1].astype('int') #.astype('str') # locKey
        
        #test_label1 = Y[test_setID,toDecode_X1].astype('int') #.astype('str') # locKey
        #test_label2 = Y[test_setID,toDecode_X2].astype('int') #.astype('str') # locKey
        # fully random
        full_label1_shuff = np.random.permutation(full_label1) 
        
        for nbt in range(nBoots):
            ### split into train and test sets
            train_setID = np.sort(np.random.choice(ntrial, round(0.5*ntrial),replace = False))
            test_setID = np.sort(np.random.choice(np.setdiff1d(np.arange(ntrial), train_setID, assume_unique=True), (ntrial-len(train_setID)),replace = False))
            
            train_label1 = full_label1[train_setID] #.astype('str') # locKey
            test_label1 = full_label1[test_setID] #.astype('str') # locKey
            
            train_label1_shuff = full_label1_shuff[train_setID] #.astype('str') # locKey
            test_label1_shuff = full_label1_shuff[test_setID] #.astype('str') # locKey
            

            # cross temp decoding
            
            for t in range(len(tbins)):
                for t_ in range(len(tbins)):
                    
                    info1_3d = f_decoding.LDAPerformance(projs_All_CT[train_setID,:,t],projs_All_CT[test_setID,:,t_], 
                                                            train_label1, test_label1)
                    
                    decode_ttXT[n,nbt,t,t_] = info1_3d #.mean(axis=-1).mean(axis=-1)
                    
                    if t==t_:                            
                        decode_ttWT[n,nbt,t] = info1_3d #.mean(axis=-1).mean(axis=-1)
                        
                
                    # permutation null distribution
                    info1_3d_shuff = f_decoding.LDAPerformance(projs_All_CT[train_setID,:,t],projs_All_CT[test_setID,:,t_], 
                                                            train_label1_shuff, test_label1_shuff)
                    
                    decode_ttX_shuffT[n,nbt,t,t_] = info1_3d_shuff #.mean(axis=-1).mean(axis=-1)
                    
                    if t==t_:
                        decode_ttW_shuffT[n,nbt,t] = info1_3d_shuff #.mean(axis=-1).mean(axis=-1)
                        
                            
        decode_ttX[region] = decode_ttXT
        decode_ttX_shuff[region] = decode_ttX_shuffT
        
        decode_ttW[region] = decode_ttWT
        decode_ttW_shuff[region] = decode_ttW_shuffT
        
# In[] save
np.save(f'{phd_path}/outputs/monkeys/' + 'performance_ttX_readout_data.npy', decode_ttX, allow_pickle=True)
np.save(f'{phd_path}/outputs/monkeys/' + 'performance_ttW_readout_data.npy', decode_ttW, allow_pickle=True)
np.save(f'{phd_path}/outputs/monkeys/' + 'performance_ttX_readout_shuff_data.npy', decode_ttX_shuff, allow_pickle=True)
np.save(f'{phd_path}/outputs/monkeys/' + 'performance_ttW_readout_shuff_data.npy', decode_ttW_shuff, allow_pickle=True)

#%% load
decode_ttX = np.load(f'{phd_path}/outputs/monkeys/' + 'performance_ttX_readout_data.npy', allow_pickle=True).item()
decode_ttW = np.load(f'{phd_path}/outputs/monkeys/' + 'performance_ttW_readout_data.npy', allow_pickle=True).item()
decode_ttX_shuff = np.load(f'{phd_path}/outputs/monkeys/' + 'performance_ttX_readout_shuff_data.npy', allow_pickle=True).item()
decode_ttW_shuff = np.load(f'{phd_path}/outputs/monkeys/' + 'performance_ttW_readout_shuff_data.npy', allow_pickle=True).item()
#%% plot cross temp decoding of ttype
for region in ('dlpfc','fef'):
       
    #vmax = 0.6 if region == 'dlpfc' else 0.8
    fig = plt.figure(figsize=(7, 6), dpi=300)
    
    
    performanceT1 = decode_ttX[region]
    performanceT1_shuff = decode_ttX_shuff[region]
    
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
    for i in [0, 1300, 2600]:
        ax.plot(tbins, np.full_like(tbins,list(tbins).index(i)), 'w-.', linewidth=4)
        ax.plot(np.full_like(tbins,list(tbins).index(i)), tbins, 'w-.', linewidth=4)
    
    ax.set_xticks([list(tbins).index(i) for i in [0, 1300, 2600]])
    ax.set_xticklabels(['S1', 'S2', 'Go Cue'], rotation=0, fontsize = 15)
    ax.set_xlabel('Test Timebin (ms)', labelpad = 10, fontsize = 20)
    ax.set_yticks([list(tbins).index(i) for i in [0, 1300, 2600]])
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
        
    plt.suptitle(f'{region.upper()}, Readout Subspace', fontsize = 25, y=1.25)
    plt.show()
    
    fig.savefig(f'{phd_path}/outputs/monkeys/decodability_ttX_readout_{region}.tif', bbox_inches='tight')

#%%



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
# In[] retarget vs distraction state changes
#nPerms = 10
nIters=100

bins = 50
dt = dt
tbins = np.arange(tslice[0], tslice[1], bins)

euDists = {}
euDists_shuff = {}


end_D1s, end_D2s = np.arange(800,1300,bins), np.arange(2100,2600,bins)

#, end_D2b = 2100, 2600 #-300,0 #

for region in ('dlpfc','fef'):
    
    euDists[region] = {}
    euDists_shuff[region] = {}
    
    # estimate decodability by ttype
    for tt in ttypes:
        
        euDistT = [] # pca1st 3d coordinates
        euDistT_shuff = [] # pca1st 3d coordinates
        
        for n in range(nIters):
            #for nbt in range(nBoots):
                
            # trial info
            trialInfo_CT = trialInfos_C[region][n][0]
            trialInfo_CT_tt = trialInfo_CT[trialInfo_CT.type == tt]#.reset_index(drop = True)
            idx_tt = trialInfo_CT_tt.index.tolist()#.trial_index.values
            
            vecs_CT = vecs_C[region][n][0] # choice plane vecs
            projs_CT = projs_C[region][n][0] #
            
            # shuff trial info
            trialInfo_CT_shuff = trialInfos_C_shuff[region][n][0]
            trialInfo_CT_tt_shuff = trialInfo_CT_shuff[trialInfo_CT_shuff.type == tt]#.reset_index(drop = True)
            idx_tt_shuff = trialInfo_CT_tt_shuff.index.tolist()#.trial_index.values
            
            vecs_CT_shuff = vecs_C_shuff[region][n][0] # choice plane vecs
            projs_CT_shuff = projs_C_shuff[region][n][0] #
            
            

            vec_normal_CT = np.cross(vecs_CT[0],vecs_CT[1]) # plane normal vec
            center_CT = projs_CT.mean(0) # plane center
            
            vec_normal_CT_shuff = np.cross(vecs_CT_shuff[0],vecs_CT_shuff[1]) # plane normal vec
            center_CT_shuff = projs_CT_shuff.mean(0) # plane center
            
            #data_3pc_CT = data_3pc_C[region][n][nbt][idx_tt,:,:] # 3pc states from tt trials
            
            # smooth to 50ms bins
            #ntrialT, ncellT, ntimeT = data_3pc_CT.shape
            #data_3pc_CT_smooth = np.mean(data_3pc_CT.reshape(ntrialT, ncellT, int(ntimeT/(bins/dt)),int(bins/dt)),axis = 3)
            
            data_3pc_CT_smooth = data_3pc_C[region][n][0][idx_tt,:,:] # 3pc states from tt trials
            projs_All_CT = np.array([[f_subspace.proj_on_plane(p[:,t], vec_normal_CT, center_CT) for t in range(len(tbins))] for p in data_3pc_CT_smooth]) # projections on the plane
            projs_All_CT = np.swapaxes(projs_All_CT,1,2)
            
            endX_D1s, endX_D2s = [tbins.tolist().index(ed1) for ed1 in end_D1s], [tbins.tolist().index(ed2) for ed2 in end_D2s]
            
            #shuff
            data_3pc_CT_smooth_shuff = data_3pc_C_shuff[region][n][0][idx_tt_shuff,:,:] # 3pc states from tt trials
            projs_All_CT_shuff = np.array([[f_subspace.proj_on_plane(p[:,t], vec_normal_CT_shuff, center_CT_shuff) for t in range(len(tbins))] for p in data_3pc_CT_smooth_shuff]) # projections on the plane
            projs_All_CT_shuff = np.swapaxes(projs_All_CT_shuff,1,2)
            
            #compress to 2d
            vecX_, vecY_ = f_subspace.vec_quad_side(projs_CT, sequence = (3,1,2,0)) 
            vecs_new = np.array(f_subspace.basis_correction_3d(vecs_CT, vecX_, vecY_))
            
            projs_All_CT_2d = np.array([f_subspace.proj_2D_coordinates(projs_All_CT[:,:,t], vecs_new) for t in range(projs_All_CT.shape[2])])
            projs_All_CT_2d = np.swapaxes(np.swapaxes(projs_All_CT_2d, 0, 1), 1, 2)
            
            
            euDistT += [np.sqrt(np.sum((projs_All_CT_2d[:,:,endX_D1s].mean(2) - projs_All_CT_2d[:,:,endX_D2s].mean(2))**2, axis=1))]
            
            
            # shuff
            vecX_shuff, vecY_shuff = f_subspace.vec_quad_side(projs_CT_shuff, sequence = (3,1,2,0)) 
            vecs_new_shuff = np.array(f_subspace.basis_correction_3d(vecs_CT_shuff, vecX_shuff, vecY_shuff))
            
            projs_All_CT_2d_shuff = np.array([f_subspace.proj_2D_coordinates(projs_All_CT_shuff[:,:,t], vecs_new_shuff) for t in range(projs_All_CT_shuff.shape[2])])
            projs_All_CT_2d_shuff = np.swapaxes(np.swapaxes(projs_All_CT_2d_shuff, 0, 1), 1, 2)
            
            euDistT_shuff += [np.sqrt(np.sum((projs_All_CT_2d_shuff[:,:,endX_D1s].mean(2) - projs_All_CT_2d_shuff[:,:,endX_D2s].mean(2))**2, axis=1))]
            
            #euDistT += [np.sqrt(np.sum((centroidD1 - centroidD2)**2))]
            #endX_D1b, endX_D2b = tbins.tolist().index(end_D1b), tbins.tolist().index(end_D2b)
            #euDistT_shuff += [np.sqrt(np.sum((projs_All_CT[:,:,endX_D1b]-projs_All_CT[:,:,endX_D2b])**2, axis=1))]
            
            #for npm in range(nPerms):
                # time-shuffled time series
            #    rng = np.random.default_rng()
            #    projs_All_CT_shuff = rng.permuted(projs_All_CT, axis=2)
                
            #    euDistT_shuff += [np.sqrt(np.sum((projs_All_CT_shuff[:,:,endX_D1]-projs_All_CT_shuff[:,:,endX_D2])**2, axis=1))]
                
        
        euDistT = np.array(euDistT)
        euDists[region][tt] = euDistT
        
        euDistT_shuff = np.array(euDistT_shuff)
        euDists_shuff[region][tt] = euDistT_shuff
    
    # condition general std for each pseudo pop
    #euDistT = np.concatenate((euDists[region][1], euDists[region][2]),axis=1)
    euDistT = np.concatenate((euDists[region][1], euDists[region][2]))
    euDistT_shuff = np.concatenate((euDists_shuff[region][1], euDists_shuff[region][2]))
    
    for tt in ttypes:
        for i in range(len(euDists[region][tt])):
            #euDists[region][tt][i,:] = (euDists[region][tt][i,:])/euDistT[i,:].std()
            euDists[region][tt][i] = (euDists[region][tt][i])/(euDistT[i].std()+epsilon)
            #euDists[region][tt][i,:] = (euDists[region][tt][i,:] - euDistT[i,:].mean())/euDistT[i,:].std()
            # euDistT_shuff[j,:].std()#
        
        for j in range(len(euDists_shuff[region][tt])):
            euDists_shuff[region][tt][j] = (euDists_shuff[region][tt][j])/(euDistT_shuff[j].std()+epsilon)# euDistT_shuff[j,:].std() # - euDistT[i,:].mean()
#%%
np.save(f'{phd_path}/outputs/monkeys/euDists_monkeys.npy', euDists, allow_pickle=True)
np.save(f'{phd_path}/outputs/monkeys/euDists_shuff_monkeys.npy', euDists_shuff, allow_pickle=True)
# In[] retarget vs distraction state changes, centroid method
#nPerms = 10
nIters=100

bins = 50
dt = dt
tbins = np.arange(tslice[0], tslice[1], bins)

euDists = {}
euDists_shuff = {}


end_D1s, end_D2s = np.arange(800,1300,bins), np.arange(2100,2600,bins)

#, end_D2b = 2100, 2600 #-300,0 #

for region in ('dlpfc','fef'):
    
    euDists[region] = {}
    euDists_shuff[region] = {}
    
    # estimate decodability by ttype
    for tt in ttypes:
        
        euDistT = [] # pca1st 3d coordinates
        euDistT_shuff = [] # pca1st 3d coordinates
        
        for n in range(nIters):
            #for nbt in range(nBoots):
                
            # trial info
            trialInfo_CT = trialInfos_C[region][n][0]
            trialInfo_CT_tt = trialInfo_CT[trialInfo_CT.type == tt]#.reset_index(drop = True)
            idx_tt = trialInfo_CT_tt.index.tolist()#.trial_index.values
            
            # shuff
            trialInfo_CT_shuff = trialInfos_C_shuff[region][n][0]
            trialInfo_CT_tt_shuff = trialInfo_CT_shuff[trialInfo_CT_shuff.type == tt]#.reset_index(drop = True)
            idx_tt_shuff = trialInfo_CT_tt_shuff.index.tolist()#.trial_index.values
            
            vecs_CT = vecs_C[region][n][0] # choice plane vecs
            projs_CT = projs_C[region][n][0] #
            
            vecs_CT_shuff = vecs_C_shuff[region][n][0] # choice plane vecs
            projs_CT_shuff = projs_C_shuff[region][n][0] #
            

            vec_normal_CT = np.cross(vecs_CT[0],vecs_CT[1]) # plane normal vec
            center_CT = projs_CT.mean(0) # plane center
            
            vec_normal_CT_shuff = np.cross(vecs_CT_shuff[0],vecs_CT_shuff[1]) # plane normal vec
            center_CT_shuff = projs_CT_shuff.mean(0) # plane center
            
            #data_3pc_CT = data_3pc_C[region][n][nbt][idx_tt,:,:] # 3pc states from tt trials
            
            # smooth to 50ms bins
            #ntrialT, ncellT, ntimeT = data_3pc_CT.shape
            #data_3pc_CT_smooth = np.mean(data_3pc_CT.reshape(ntrialT, ncellT, int(ntimeT/(bins/dt)),int(bins/dt)),axis = 3)
            
            data_3pc_CT_smooth = data_3pc_C[region][n][0][idx_tt,:,:] # 3pc states from tt trials
            projs_All_CT = np.array([[f_subspace.proj_on_plane(p[:,t], vec_normal_CT, center_CT) for t in range(len(tbins))] for p in data_3pc_CT_smooth]) # projections on the plane
            projs_All_CT = np.swapaxes(projs_All_CT,1,2)
            
            # shuff
            data_3pc_CT_smooth_shuff = data_3pc_C_shuff[region][n][0][idx_tt_shuff,:,:] # 3pc states from tt trials
            projs_All_CT_shuff = np.array([[f_subspace.proj_on_plane(p[:,t], vec_normal_CT_shuff, center_CT_shuff) for t in range(len(tbins))] for p in data_3pc_CT_smooth_shuff]) # projections on the plane
            projs_All_CT_shuff = np.swapaxes(projs_All_CT_shuff,1,2)
            
            #compress to 2d
            vecX_, vecY_ = f_subspace.vec_quad_side(projs_CT, sequence = (3,1,2,0)) 
            vecs_new = np.array(f_subspace.basis_correction_3d(vecs_CT, vecX_, vecY_))
            
            projs_All_CT_2d = np.array([f_subspace.proj_2D_coordinates(projs_All_CT[:,:,t], vecs_new) for t in range(projs_All_CT.shape[2])])
            projs_All_CT_2d = np.swapaxes(np.swapaxes(projs_All_CT_2d, 0, 1), 1, 2)
            
            endX_D1s, endX_D2s = [tbins.tolist().index(ed1) for ed1 in end_D1s], [tbins.tolist().index(ed2) for ed2 in end_D2s]
            
            #shuff
            vecX_shuff, vecY_shuff = f_subspace.vec_quad_side(projs_CT_shuff, sequence = (3,1,2,0)) 
            vecs_new_shuff = np.array(f_subspace.basis_correction_3d(vecs_CT_shuff, vecX_shuff, vecY_shuff))
            
            projs_All_CT_2d_shuff = np.array([f_subspace.proj_2D_coordinates(projs_All_CT_shuff[:,:,t], vecs_new_shuff) for t in range(projs_All_CT_shuff.shape[2])])
            projs_All_CT_2d_shuff = np.swapaxes(np.swapaxes(projs_All_CT_2d_shuff, 0, 1), 1, 2)
            
            endX_D1s_shuff, endX_D2s_shuff = [tbins.tolist().index(ed1) for ed1 in end_D1s], [tbins.tolist().index(ed2) for ed2 in end_D2s]
            
            euDistTT = []
            euDistTT_shuff = []
            
            trialInfo_temp = trialInfo_CT_tt.copy().reset_index(drop=True)
            trialInfo_temp_shuff = trialInfo_CT_tt_shuff.copy().reset_index(drop=True)
            
            for l1 in locs:
                for l2 in locs:
                    if l1!=l2:
                        idxT = trialInfo_temp[(trialInfo_temp.loc1==l1)&(trialInfo_temp.loc2==l2)].index
                        centroidD1 = projs_All_CT_2d[idxT][:,:,endX_D1s].mean(2).mean(0)
                        centroidD2 = projs_All_CT_2d[idxT][:,:,endX_D2s].mean(2).mean(0)
                        
                        euDistTT += [np.sqrt(np.sum((centroidD1 - centroidD2)**2))]
                        
                        # shuff
                        idxT_shuff = trialInfo_temp_shuff[(trialInfo_temp_shuff.loc1==l1)&(trialInfo_temp_shuff.loc2==l2)].index
                        centroidD1_shuff = projs_All_CT_2d_shuff[idxT_shuff][:,:,endX_D1s_shuff].mean(2).mean(0)
                        centroidD2_shuff = projs_All_CT_2d_shuff[idxT_shuff][:,:,endX_D2s_shuff].mean(2).mean(0)
                        
                        euDistTT_shuff += [np.sqrt(np.sum((centroidD1_shuff - centroidD2_shuff)**2))]
                    
            euDistT += [np.array(euDistTT)]
            euDistT_shuff += [np.array(euDistTT_shuff)]
                
        
        euDistT = np.array(euDistT)
        euDists[region][tt] = euDistT
        
        euDistT_shuff = np.array(euDistT_shuff)
        euDists_shuff[region][tt] = euDistT_shuff
    
    # condition general std for each pseudo pop
    euDistT = np.concatenate((euDists[region][1], euDists[region][2]))
    euDistT_shuff = np.concatenate((euDists_shuff[region][1], euDists_shuff[region][2]))
    
    for tt in ttypes:
        for i in range(len(euDists[region][tt])):
            #euDists[region][tt][i,:] = (euDists[region][tt][i,:])/euDistT[i,:].std()
            euDists[region][tt][i] = (euDists[region][tt][i])/(euDistT[i].std()+epsilon)
            #euDists[region][tt][i,:] = (euDists[region][tt][i,:] - euDistT[i,:].mean())/euDistT[i,:].std()
            # euDistT_shuff[j,:].std()#
        
        for j in range(len(euDists_shuff[region][tt])):
            euDists_shuff[region][tt][j] = (euDists_shuff[region][tt][j])/(euDistT_shuff[j].std()+epsilon)
# In[]
np.save(f'{phd_path}/outputs/monkeys/euDists_monkeys_centroids.npy', euDists, allow_pickle=True)
np.save(f'{phd_path}/outputs/monkeys/euDists_shuff_monkeys_centroids.npy', euDists_shuff, allow_pickle=True)
#%%
euDists = np.load(f'{phd_path}/data/pseudo_ww/euDists_monkeys_centroids.npy', allow_pickle=True).item()



# In[] retarget vs distraction state changes, centroid method2
#nPerms = 10
nIters=100

bins = 50
dt = dt
tbins = np.arange(tslice[0], tslice[1], bins)

euDists = {}
euDists_shuff = {}


end_D1s, end_D2s = np.arange(800,1300,bins), np.arange(2100,2600,bins)

#, end_D2b = 2100, 2600 #-300,0 #

for region in ('dlpfc','fef'):
    
    euDists[region] = {tt:[] for tt in ttypes}
    euDists_shuff[region] = {tt:[] for tt in ttypes}

    # estimate decodability by ttype
    for n in range(nIters):
        
        #for nbt in range(nBoots):
            
        # trial info
        trialInfo_CT = trialInfos_C[region][n][0]
        vecs_CT = vecs_C[region][n][0] # choice plane vecs
        projs_CT = projs_C[region][n][0] #
        
        vec_normal_CT = np.cross(vecs_CT[0],vecs_CT[1]) # plane normal vec
        center_CT = projs_CT.mean(0) # plane center
        
        # shuff
        trialInfo_CT_shuff = trialInfos_C_shuff[region][n][0]
        vecs_CT_shuff = vecs_C_shuff[region][n][0] # choice plane vecs
        projs_CT_shuff = projs_C_shuff[region][n][0] #
        
        vec_normal_CT_shuff = np.cross(vecs_CT_shuff[0],vecs_CT_shuff[1]) # plane normal vec
        center_CT_shuff = projs_CT_shuff.mean(0) # plane center
        
        #data_3pc_CT = data_3pc_C[region][n][nbt][idx_tt,:,:] # 3pc states from tt trials
        
        # smooth to 50ms bins
        #ntrialT, ncellT, ntimeT = data_3pc_CT.shape
        #data_3pc_CT_smooth = np.mean(data_3pc_CT.reshape(ntrialT, ncellT, int(ntimeT/(bins/dt)),int(bins/dt)),axis = 3)
        
        data_3pc_CT_smooth = data_3pc_C[region][n][0][:,:,:] # 3pc states from tt trials
        projs_All_CT = np.array([[f_subspace.proj_on_plane(p[:,t], vec_normal_CT, center_CT) for t in range(len(tbins))] for p in data_3pc_CT_smooth]) # projections on the plane
        projs_All_CT = np.swapaxes(projs_All_CT,1,2)
        
        # shuff
        data_3pc_CT_smooth_shuff = data_3pc_C_shuff[region][n][0][:,:,:] # 3pc states from tt trials
        projs_All_CT_shuff = np.array([[f_subspace.proj_on_plane(p[:,t], vec_normal_CT_shuff, center_CT_shuff) for t in range(len(tbins))] for p in data_3pc_CT_smooth_shuff]) # projections on the plane
        projs_All_CT_shuff = np.swapaxes(projs_All_CT_shuff,1,2)
        
        #compress to 2d
        vecX_, vecY_ = f_subspace.vec_quad_side(projs_CT, sequence = (3,1,2,0)) 
        vecs_new = np.array(f_subspace.basis_correction_3d(vecs_CT, vecX_, vecY_))
        
        projs_All_CT_2d = np.array([f_subspace.proj_2D_coordinates(projs_All_CT[:,:,t], vecs_new) for t in range(projs_All_CT.shape[2])])
        projs_All_CT_2d = np.swapaxes(np.swapaxes(projs_All_CT_2d, 0, 1), 1, 2)
        
        endX_D1s, endX_D2s = [tbins.tolist().index(ed1) for ed1 in end_D1s], [tbins.tolist().index(ed2) for ed2 in end_D2s]
        
        #shuff
        vecX_shuff, vecY_shuff = f_subspace.vec_quad_side(projs_CT_shuff, sequence = (3,1,2,0)) 
        vecs_new_shuff = np.array(f_subspace.basis_correction_3d(vecs_CT_shuff, vecX_shuff, vecY_shuff))
        
        projs_All_CT_2d_shuff = np.array([f_subspace.proj_2D_coordinates(projs_All_CT_shuff[:,:,t], vecs_new_shuff) for t in range(projs_All_CT_shuff.shape[2])])
        projs_All_CT_2d_shuff = np.swapaxes(np.swapaxes(projs_All_CT_2d_shuff, 0, 1), 1, 2)
        
        endX_D1s_shuff, endX_D2s_shuff = [tbins.tolist().index(ed1) for ed1 in end_D1s], [tbins.tolist().index(ed2) for ed2 in end_D2s]
        
        trialInfo_temp = trialInfo_CT.copy().reset_index(drop=True)
        trialInfo_temp_shuff = trialInfo_CT_shuff.copy().reset_index(drop=True)
        
        euDistT = {tt:[] for tt in ttypes}
        euDistT_shuff = {tt:[] for tt in ttypes}

        for l1 in locs:
            idxT1 = trialInfo_temp[(trialInfo_temp.loc1==l1)].index
            centroidD1 = projs_All_CT_2d[idxT1][:,:,endX_D1s].mean(2).mean(0) # type-general centroid

            idxT1_shuff = trialInfo_temp_shuff[(trialInfo_temp_shuff.loc1==l1)].index
            centroidD1_shuff = projs_All_CT_2d_shuff[idxT1_shuff][:,:,endX_D1s_shuff].mean(2).mean(0) # type-general centroid
            
            for tt in ttypes:
                for l2 in locs:
                    if l1!=l2:
                        idxT2 = trialInfo_temp[(trialInfo_temp.loc1==l1)&(trialInfo_temp.loc2==l2)&(trialInfo_temp.type==tt)].index
                        centroidD2 = projs_All_CT_2d[idxT2][:,:,endX_D2s].mean(2).mean(0)
                        
                        euDistT[tt] += [np.sqrt(np.sum((centroidD1 - centroidD2)**2))]
                        
                        # shuff
                        idxT2_shuff = trialInfo_temp_shuff[(trialInfo_temp_shuff.loc1==l1)&(trialInfo_temp_shuff.loc2==l2)&(trialInfo_temp_shuff.type==tt)].index
                        centroidD2_shuff = projs_All_CT_2d_shuff[idxT2_shuff][:,:,endX_D2s_shuff].mean(2).mean(0)
                        euDistT_shuff[tt] += [np.sqrt(np.sum((centroidD1_shuff - centroidD2_shuff)**2))]

        std_pooled = np.concatenate([euDistT[1],euDistT[2]]).std() + epsilon
        std_pooled_shuff = np.concatenate([euDistT_shuff[1],euDistT_shuff[2]]).std() + epsilon
        
        for tt in ttypes:
            euDistT[tt] = np.array(euDistT[tt]) / std_pooled
            euDists[region][tt] += [euDistT[tt]]
            
            euDistT_shuff[tt] = np.array(euDistT_shuff[tt]) / std_pooled_shuff
            euDists_shuff[region][tt] += [euDistT_shuff[tt]]
        
    #euDistT_shuff = np.array(euDistT_shuff)
    #euDists_shuff[region][tt] = euDistT_shuff


    # condition general std for each pseudo pop
    #euDistT = np.concatenate((euDists[region][1], euDists[region][2]))
    #euDistT_shuff = np.concatenate((euDists_shuff[region][1], euDists_shuff[region][2]),axis=1)
    
    #for tt in ttypes:
    #    for i in range(len(euDists[region][tt])):
            #euDists[region][tt][i,:] = (euDists[region][tt][i,:])/euDistT[i,:].std()
    #        euDists[region][tt][i] = (euDists[region][tt][i])/(euDistT[i].std()+epsilon)
            #euDists[region][tt][i,:] = (euDists[region][tt][i,:] - euDistT[i,:].mean())/euDistT[i,:].std()
            # euDistT_shuff[j,:].std()#
        
        #for j in range(len(euDists_shuff[region][tt])):
        #    euDists_shuff[region][tt][j,:] = (euDists_shuff[region][tt][j,:])/ euDistT[i,:].std()# euDistT_shuff[j,:].std() # - euDistT[i,:].mean()
# In[]
np.save(f'{phd_path}/outputs/monkeys/euDists_monkeys_centroids2.npy', euDists, allow_pickle=True)
np.save(f'{phd_path}/outputs/monkeys/euDists_shuff_monkeys_centroids2.npy', euDists_shuff, allow_pickle=True)
#%%
euDists = np.load(f'{phd_path}/data/pseudo_ww/euDists_monkeys_centroids.npy', allow_pickle=True).item()



# In[] retarget vs distraction state changes normalized
#nPerms = 10
nIters=100

bins = 50
dt = dt
tbins = np.arange(tslice[0], tslice[1], bins)

euDists = {}
euDists_shuff = {}


end_D1s, end_D2s = np.arange(800,1300,bins), np.arange(2100,2600,bins)

#, end_D2b = 2100, 2600 #-300,0 #
hideLocs = () #0,2
normalizeMinMax = (-1,1)
remainedLocs = tuple(l for l in locs if l not in hideLocs)

for region in ('dlpfc','fef'):
    
    euDists[region] = {}
    euDists_shuff[region] = {}
    
    # estimate decodability by ttype
    for tt in ttypes:
        
        euDistT = [] # pca1st 3d coordinates
        euDistT_shuff = [] # pca1st 3d coordinates
        
        for n in range(nIters):
            #for nbt in range(nBoots):
                
            # trial info
            trialInfo_CT = trialInfos_C[region][n][0]
            trialInfo_CT_tt = trialInfo_CT[trialInfo_CT.type == tt]#.reset_index(drop = True)
            
            #if bool(hideLocs):
            #    trialInfo_CT_tt = trialInfo_CT_tt[~trialInfo_CT_tt.loc2.isin(hideLocs)]   
            trialInfo_CT_tt = trialInfo_CT_tt[(trialInfo_CT_tt.loc1.isin(remainedLocs))&(trialInfo_CT_tt.loc2.isin(remainedLocs))]
            idx_tt = trialInfo_CT_tt.index.tolist()#.trial_index.values
            
            vecs_CT = vecs_C[region][n][0] # choice plane vecs
            projs_CT = projs_C[region][n][0] #
            
            

            vec_normal_CT = np.cross(vecs_CT[0],vecs_CT[1]) # plane normal vec
            center_CT = projs_CT.mean(0) # plane center
            
            #data_3pc_CT = data_3pc_C[region][n][nbt][idx_tt,:,:] # 3pc states from tt trials
            
            # smooth to 50ms bins
            #ntrialT, ncellT, ntimeT = data_3pc_CT.shape
            #data_3pc_CT_smooth = np.mean(data_3pc_CT.reshape(ntrialT, ncellT, int(ntimeT/(bins/dt)),int(bins/dt)),axis = 3)
            
            data_3pc_CT_smooth = data_3pc_C[region][n][0][idx_tt,:,:] # 3pc states from tt trials
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
            
            euDistT += [np.sqrt(np.sum((projs_All_CT_2d[:,:,endX_D1s].mean(2) - projs_All_CT_2d[:,:,endX_D2s].mean(2))**2, axis=1))]
            
            
            # shuff trial info
            #trialInfo_CT_shuff = trialInfos_C_shuff[region][n][0]
            #trialInfo_CT_tt_shuff = trialInfo_CT_shuff[trialInfo_CT_shuff.type == tt]#.reset_index(drop = True)
            
            #if bool(hideLocs):
            #    trialInfo_CT_tt_shuff = trialInfo_CT_tt_shuff[~trialInfo_CT_tt_shuff.loc2.isin(hideLocs)]
            #trialInfo_CT_tt_shuff = trialInfo_CT_tt_shuff[(trialInfo_CT_tt_shuff.loc1.isin(remainedLocs))&(trialInfo_CT_tt_shuff.loc2.isin(remainedLocs))]
            #idx_tt_shuff = trialInfo_CT_tt_shuff.index.tolist()#.trial_index.values
            idx_tt_shuff = trialInfo_CT_tt.sample(frac=1).reset_index(drop=True).index.tolist()
            
            #vecs_CT_shuff = vecs_C_shuff[region][n][0] # choice plane vecs
            #projs_CT_shuff = projs_C_shuff[region][n][0] #
            
            #vec_normal_CT_shuff = np.cross(vecs_CT_shuff[0],vecs_CT_shuff[1]) # plane normal vec
            #center_CT_shuff = projs_CT_shuff.mean(0) # plane center
            
            #data_3pc_CT_smooth_shuff = data_3pc_C_shuff[region][n][0][idx_tt_shuff,:,:] # 3pc states from tt trials
            #projs_All_CT_shuff = np.array([[f_subspace.proj_on_plane(p[:,t], vec_normal_CT_shuff, center_CT_shuff) for t in range(len(tbins))] for p in data_3pc_CT_smooth_shuff]) # projections on the plane
            #projs_All_CT_shuff = np.swapaxes(projs_All_CT_shuff,1,2)
            
            #shuff
            data_3pc_CT_smooth_shuff = data_3pc_C[region][n][0][idx_tt_shuff,:,:] # 3pc states from tt trials
            projs_All_CT_shuff = np.array([[f_subspace.proj_on_plane(p[:,t], vec_normal_CT, center_CT) for t in range(len(tbins))] for p in data_3pc_CT_smooth_shuff]) # projections on the plane
            projs_All_CT_shuff = np.swapaxes(projs_All_CT_shuff,1,2)
            
            
            
            # shuff
            #vecX_shuff, vecY_shuff = f_subspace.vec_quad_side(projs_CT_shuff, sequence = (3,1,2,0)) 
            #vecs_new_shuff = np.array(f_subspace.basis_correction_3d(vecs_CT_shuff, vecX_shuff, vecY_shuff))
            
            #projs_All_CT_2d_shuff = np.array([f_subspace.proj_2D_coordinates(projs_All_CT_shuff[:,:,t], vecs_new_shuff) for t in range(projs_All_CT_shuff.shape[2])])
            projs_All_CT_2d_shuff = np.array([f_subspace.proj_2D_coordinates(projs_All_CT_shuff[:,:,t], vecs_new) for t in range(projs_All_CT_shuff.shape[2])])
            projs_All_CT_2d_shuff = np.swapaxes(np.swapaxes(projs_All_CT_2d_shuff, 0, 1), 1, 2)
            
            if bool(normalizeMinMax):
                vmin, vmax = normalizeMinMax if len(normalizeMinMax)==2 else (-1,1)
                # normalize -1 to 1
                for d in range(projs_All_CT_2d_shuff.shape[1]):
                    projs_All_CT_2d_shuff[:,d,:] = ((projs_All_CT_2d_shuff[:,d,:] - projs_All_CT_2d_shuff[:,d,:].min()) / (projs_All_CT_2d_shuff[:,d,:].max() - projs_All_CT_2d_shuff[:,d,:].min())) * (vmax - vmin) + vmin
            euDistT_shuff += [np.sqrt(np.sum((projs_All_CT_2d_shuff[:,:,endX_D1s].mean(2) - projs_All_CT_2d_shuff[:,:,endX_D2s].mean(2))**2, axis=1))]
            
            #euDistT += [np.sqrt(np.sum((centroidD1 - centroidD2)**2))]
            #endX_D1b, endX_D2b = tbins.tolist().index(end_D1b), tbins.tolist().index(end_D2b)
            #euDistT_shuff += [np.sqrt(np.sum((projs_All_CT[:,:,endX_D1b]-projs_All_CT[:,:,endX_D2b])**2, axis=1))]
            
            #for npm in range(nPerms):
                # time-shuffled time series
            #    rng = np.random.default_rng()
            #    projs_All_CT_shuff = rng.permuted(projs_All_CT, axis=2)
                
            #    euDistT_shuff += [np.sqrt(np.sum((projs_All_CT_shuff[:,:,endX_D1]-projs_All_CT_shuff[:,:,endX_D2])**2, axis=1))]
                
        
        euDistT = np.array(euDistT)
        euDists[region][tt] = euDistT
        
        euDistT_shuff = np.array(euDistT_shuff)
        euDists_shuff[region][tt] = euDistT_shuff
    
    # condition general std for each pseudo pop
    #euDistT = np.concatenate((euDists[region][1], euDists[region][2]),axis=1)
    #euDistT = np.concatenate((euDists[region][1], euDists[region][2]))
    #euDistT_shuff = np.concatenate((euDists_shuff[region][1], euDists_shuff[region][2]))
    
    #for tt in ttypes:
    #    for i in range(len(euDists[region][tt])):
            #euDists[region][tt][i,:] = (euDists[region][tt][i,:])/euDistT[i,:].std()
    #        euDists[region][tt][i] = (euDists[region][tt][i])#/(euDistT[i].std()+epsilon)
            #euDists[region][tt][i,:] = (euDists[region][tt][i,:] - euDistT[i,:].mean())/euDistT[i,:].std()
            # euDistT_shuff[j,:].std()#
        
    #    for j in range(len(euDists_shuff[region][tt])):
    #        euDists_shuff[region][tt][j] = (euDists_shuff[region][tt][j])#/(euDistT_shuff[j].std()+epsilon)# euDistT_shuff[j,:].std() # - euDistT[i,:].mean()
#%%
np.save(f'{phd_path}/outputs/monkeys/euDists_monkeys_normalized_full.npy', euDists, allow_pickle=True)
np.save(f'{phd_path}/outputs/monkeys/euDists_shuff_monkeys_normalized_full.npy', euDists_shuff, allow_pickle=True)

# In[] retarget vs distraction state changes, normalization -1 to 1
#nPerms = 10
nIters = 100
nPerms = 100

bins = 50
dt = dt
tbins = np.arange(tslice[0], tslice[1], bins)

euDists = {}
euDists_shuff = {}

hideLocs = (0,2) #
normalizeMinMax = (-1,1)
remainedLocs = tuple(l for l in locs if l not in hideLocs)
end_D1s, end_D2s = np.arange(800,1300,bins), np.arange(2100,2600,bins)

#, end_D2b = 2100, 2600 #-300,0 #

for region in ('dlpfc','fef'):
    
    euDists[region] = {tt:[] for tt in ttypes}
    euDists_shuff[region] = {tt:[] for tt in ttypes}

    # estimate decodability by ttype
    for n in range(nIters):
        
        # trial info
        trialInfo_CT = trialInfos_C[region][n][0]
        #if bool(hideLocs):
        #    trialInfo_CT = trialInfo_CT[~trialInfo_CT.loc2.isin(hideLocs)]
        trialInfo_CT = trialInfo_CT[(trialInfo_CT.loc1.isin(remainedLocs))&(trialInfo_CT.loc2.isin(remainedLocs))]
            
        idx = trialInfo_CT.index.tolist()#.trial_index.values
        
        vecs_CT = vecs_C[region][n][0] # choice plane vecs
        projs_CT = projs_C[region][n][0] #
        
        vec_normal_CT = np.cross(vecs_CT[0],vecs_CT[1]) # plane normal vec
        center_CT = projs_CT.mean(0) # plane center
        
        #data_3pc_CT = data_3pc_C[region][n][nbt][idx_tt,:,:] # 3pc states from tt trials
        
        # smooth to 50ms bins
        #ntrialT, ncellT, ntimeT = data_3pc_CT.shape
        #data_3pc_CT_smooth = np.mean(data_3pc_CT.reshape(ntrialT, ncellT, int(ntimeT/(bins/dt)),int(bins/dt)),axis = 3)
        
        data_3pc_CT_smooth = data_3pc_C[region][n][0][idx,:,:] # 3pc states from tt trials
        projs_All_CT = np.array([[f_subspace.proj_on_plane(p[:,t], vec_normal_CT, center_CT) for t in range(len(tbins))] for p in data_3pc_CT_smooth]) # projections on the plane
        projs_All_CT = np.swapaxes(projs_All_CT,1,2)
        
        #compress to 2d
        vecX_, vecY_ = f_subspace.vec_quad_side(projs_CT, sequence = (3,1,2,0)) 
        vecs_new = np.array(f_subspace.basis_correction_3d(vecs_CT, vecX_, vecY_))
        
        projs_All_CT_2d = np.array([f_subspace.proj_2D_coordinates(projs_All_CT[:,:,t], vecs_new) for t in range(projs_All_CT.shape[2])])
        # normalize -1 to 1
        projs_All_CT_2d = np.array([((projs_All_CT_2d[:,:,d] - projs_All_CT_2d[:,:,d].min()) / (projs_All_CT_2d[:,:,d].max() - projs_All_CT_2d[:,:,d].min())) * (1 - -1) + -1 for d in range(projs_All_CT_2d.shape[-1])])
        projs_All_CT_2d = np.swapaxes(np.swapaxes(projs_All_CT_2d, 0, 1), 0, 2)
    
        if bool(normalizeMinMax):
            vmin, vmax = normalizeMinMax if len(normalizeMinMax)==2 else (-1,1)
            # normalize -1 to 1
            for d in range(projs_All_CT_2d.shape[1]):
                projs_All_CT_2d[:,d,:] = ((projs_All_CT_2d[:,d,:] - projs_All_CT_2d[:,d,:].min()) / (projs_All_CT_2d[:,d,:].max() - projs_All_CT_2d[:,d,:].min())) * (vmax - vmin) + vmin
            
    
        endX_D1s, endX_D2s = [tbins.tolist().index(ed1) for ed1 in end_D1s], [tbins.tolist().index(ed2) for ed2 in end_D2s]
    
        
        # shuff
        #trialInfo_CT_shuff = trialInfos_C_shuff[region][n][0]
        #if bool(hideLocs):
        #    trialInfo_CT_shuff = trialInfo_CT_shuff[~trialInfo_CT_shuff.loc2.isin(hideLocs)]
        ##trialInfo_CT_shuff = trialInfo_CT.sample(frac=1)
        ##idx_shuff = trialInfo_CT_shuff.index.tolist()#.trial_index.values
        #idx_shuff = np.random.shuffle(idx)
        
        #vecs_CT_shuff = vecs_C_shuff[region][n][0] # choice plane vecs
        #projs_CT_shuff = projs_C_shuff[region][n][0] #
        
        #vec_normal_CT_shuff = np.cross(vecs_CT_shuff[0],vecs_CT_shuff[1]) # plane normal vec
        #center_CT_shuff = projs_CT_shuff.mean(0) # plane center
        
        # shuff
        #data_3pc_CT_smooth_shuff = data_3pc_C_shuff[region][n][0][idx_shuff,:,:] # 3pc states from tt trials
        ##data_3pc_CT_smooth_shuff = data_3pc_C[region][n][0][idx_shuff,:,:] # 3pc states from tt trials
        
        #projs_All_CT_shuff = np.array([[f_subspace.proj_on_plane(p[:,t], vec_normal_CT_shuff, center_CT_shuff) for t in range(len(tbins))] for p in data_3pc_CT_smooth_shuff]) # projections on the plane
        ##projs_All_CT_shuff = np.array([[f_subspace.proj_on_plane(p[:,t], vec_normal_CT, center_CT) for t in range(len(tbins))] for p in data_3pc_CT_smooth_shuff]) # projections on the plane
        ##projs_All_CT_shuff = np.swapaxes(projs_All_CT_shuff,1,2)
        
        #shuff
        #vecX_shuff, vecY_shuff = f_subspace.vec_quad_side(projs_CT_shuff, sequence = (3,1,2,0)) 
        #vecs_new_shuff = np.array(f_subspace.basis_correction_3d(vecs_CT_shuff, vecX_shuff, vecY_shuff))
        
        #projs_All_CT_2d_shuff = np.array([f_subspace.proj_2D_coordinates(projs_All_CT_shuff[:,:,t], vecs_new_shuff) for t in range(projs_All_CT_shuff.shape[2])])
        ##projs_All_CT_2d_shuff = np.array([f_subspace.proj_2D_coordinates(projs_All_CT_shuff[:,:,t], vecs_new) for t in range(projs_All_CT_shuff.shape[2])])
        
        # normalize -1 to 1
        ##projs_All_CT_2d_shuff = np.array([((projs_All_CT_2d_shuff[:,:,d] - projs_All_CT_2d_shuff[:,:,d].min()) / (projs_All_CT_2d_shuff[:,:,d].max() - projs_All_CT_2d_shuff[:,:,d].min())) * (1 - -1) + -1 for d in range(projs_All_CT_2d_shuff.shape[-1])])
        ##projs_All_CT_2d_shuff = np.swapaxes(np.swapaxes(projs_All_CT_2d_shuff, 0, 1), 0, 2)
        
        ##if bool(normalizeMinMax):
        ##    vmin, vmax = normalizeMinMax if len(normalizeMinMax)==2 else (-1,1)
            # normalize -1 to 1
        ##    for d in range(projs_All_CT_2d_shuff.shape[1]):
        ##        projs_All_CT_2d_shuff[:,d,:] = ((projs_All_CT_2d_shuff[:,d,:] - projs_All_CT_2d_shuff[:,d,:].min()) / (projs_All_CT_2d_shuff[:,d,:].max() - projs_All_CT_2d_shuff[:,d,:].min())) * (vmax - vmin) + vmin
        
        endX_D1s_shuff, endX_D2s_shuff = [tbins.tolist().index(ed1) for ed1 in end_D1s], [tbins.tolist().index(ed2) for ed2 in end_D2s]
        
        
        euDistT = {tt:[] for tt in ttypes}
    
        
        trialInfo_temp = trialInfo_CT.copy().reset_index(drop=True)
        
        for l1 in remainedLocs:
            idxT1 = trialInfo_temp[(trialInfo_temp.loc1==l1)].index 
            centroidD1 = projs_All_CT_2d[idxT1][:,:,endX_D1s].mean(2).mean(0) # type-general centroid

            #idxT1_shuff = trialInfo_temp_shuff[(trialInfo_temp_shuff.loc1==l1)].index
            #centroidD1_shuff = projs_All_CT_2d[idxT1_shuff][:,:,endX_D1s_shuff].mean(2).mean(0) # type-general centroid
            
            for tt in ttypes:
                for l2 in remainedLocs:
                    if l1!=l2:
                        idxT2 = trialInfo_temp[(trialInfo_temp.loc1==l1)&(trialInfo_temp.loc2==l2)&(trialInfo_temp.type==tt)].index
                        centroidD2 = projs_All_CT_2d[idxT2][:,:,endX_D2s].mean(2).mean(0)
                        
                        euDistT[tt] += [np.sqrt(np.sum((centroidD1 - centroidD2)**2))]
                        
                        # shuff
                        #idxT2_shuff = trialInfo_temp_shuff[(trialInfo_temp_shuff.loc1==l1)&(trialInfo_temp_shuff.loc2==l2)&(trialInfo_temp_shuff.type==tt)].index
                        #centroidD2_shuff = projs_All_CT_2d[idxT2_shuff][:,:,endX_D2s_shuff].mean(2).mean(0)
                        #euDistT_shuff[tt] += [np.sqrt(np.sum((centroidD1_shuff - centroidD2_shuff)**2))]
        
        for tt in ttypes:
            #temp = np.array(euDistT[tt])
            #temp_shuff = np.array(euDistT_shuff[tt])
            
            #euDistT[tt] = ((temp - euDistT_pooled.mean()) / (euDistT_pooled.max() - euDistT_pooled.min())) * (1 - -1) + -1
            euDists[region][tt] += [euDistT[tt]]
            
        
        #shuffle
        euDistT_shuff = {tt:[] for tt in ttypes}
        for npm in range(nPerms):
            for tt in ttypes:
                euDistT_shuff[tt].append([])
            
            trialInfo_temp_shuff = trialInfo_CT.sample(frac=1).reset_index(drop=True)

            for l1 in remainedLocs:
                
                idxT1_shuff = trialInfo_temp_shuff[(trialInfo_temp_shuff.loc1==l1)].index
                centroidD1_shuff = projs_All_CT_2d[idxT1_shuff][:,:,endX_D1s_shuff].mean(2).mean(0) # type-general centroid
                
                for tt in ttypes:
                    for l2 in remainedLocs:
                        if l1!=l2:
                            
                            idxT2_shuff = trialInfo_temp_shuff[(trialInfo_temp_shuff.loc1==l1)&(trialInfo_temp_shuff.loc2==l2)&(trialInfo_temp_shuff.type==tt)].index
                            centroidD2_shuff = projs_All_CT_2d[idxT2_shuff][:,:,endX_D2s_shuff].mean(2).mean(0)
                            euDistT_shuff[tt][npm] += [np.sqrt(np.sum((centroidD1_shuff - centroidD2_shuff)**2))]
        
        for tt in ttypes:
            euDists_shuff[region][tt] += [euDistT_shuff[tt]]
            
    
# In[]
np.save(f'{phd_path}/outputs/monkeys/euDists_monkeys_centroids2_normalized_hide02.npy', euDists, allow_pickle=True)
np.save(f'{phd_path}/outputs/monkeys/euDists_shuff_monkeys_centroids2_normalized_hide02.npy', euDists_shuff, allow_pickle=True)
#%%
euDists = np.load(f'{phd_path}/data/pseudo_ww/euDists_monkeys_normalized.npy', allow_pickle=True).item()



# In[]      

# In[] evrs

plt.figure(figsize=(5, 4), dpi=100)
ax = plt.gca()
ax.boxplot([evrs['dlpfc'].sum(1), evrs['fef'].sum(1)], tick_labels=['DLPFC','FEF'])
ax.set_xlabel('Region', labelpad = 3, fontsize = 12)
ax.set_ylabel('Sum of EVR from top 3 PCs', labelpad = 3, fontsize = 12)
plt.show()
#%%

























































#%%

############
# non used #
############

#%%



#cseq = mpl.color_sequences['tab20b']
#color1, color1_, color2, color2_ = cseq[-3], cseq[-1], cseq[-7], cseq[-5]

#cseq = mpl.color_sequences['Paired']
#color1, color1_, color2, color2_ = cseq[7], cseq[6], cseq[9], cseq[8]
color1, color1_, color2, color2_ = '#185337', '#96d9ad', '#804098', '#c4a2d1'

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


fig = plt.figure(figsize=(3, 3), dpi=300)

plt.boxplot([euDists['dlpfc'][1].mean(1)], positions=[0.3], patch_artist=True, widths = 0.2, boxprops=boxprops1, flierprops=flierprops1, 
                  meanprops=meanpointprops1, medianprops=medianprops, capprops = capprops1, whiskerprops = whiskerprops1, meanline=False, showmeans=True)
plt.boxplot([euDists['fef'][1].mean(1)], positions=[1.3], patch_artist=True, widths = 0.2, boxprops=boxprops2, flierprops=flierprops2, 
                  meanprops=meanpointprops2, medianprops=medianprops, capprops = capprops2, whiskerprops = whiskerprops2, meanline=False, showmeans=True)

plt.boxplot([euDists['dlpfc'][2].mean(1)], positions=[0.7], patch_artist=True, widths = 0.2, boxprops=dict(boxprops1,facecolor=color1_), flierprops=dict(flierprops1,markeredgecolor=color1_, markerfacecolor=color1_), 
                  meanprops=dict(meanpointprops1,markeredgecolor=color1_), medianprops=medianprops, capprops = dict(capprops1,color=color1_), whiskerprops = dict(whiskerprops1,color=color1_), meanline=False, showmeans=True)
plt.boxplot([euDists['fef'][2].mean(1)], positions=[1.7], patch_artist=True, widths = 0.2, boxprops=dict(boxprops2,facecolor=color2_), flierprops=dict(flierprops2,markeredgecolor=color2_, markerfacecolor=color2_), 
                  meanprops=dict(meanpointprops2,markeredgecolor=color2_), medianprops=medianprops, capprops = dict(capprops2,color=color2_), whiskerprops = dict(whiskerprops2,color=color2_), meanline=False, showmeans=True)

lineh = np.arange(0.3,0.7,0.001)
linev = np.arange(2.97,3,0.001)
#p1 = scipy.stats.ttest_rel(euDists['dlpfc'][1].mean(1), euDists['dlpfc'][2].mean(1))[-1]
#p1,_,_ = f_stats.bootstrap95_p(euDists['dlpfc'][1].mean(1), euDists['dlpfc'][2].mean(1)) # shape: nIters * ntrials
p1 = f_stats.permutation_p_diff(euDists['dlpfc'][1].mean(1), euDists['dlpfc'][2].mean(1)) # shape: nIters * ntrials

plt.plot(lineh, np.full_like(lineh, 3), 'k-')
plt.plot(np.full_like(linev, 0.3), linev, 'k-')
plt.plot(np.full_like(linev, 0.7), linev, 'k-')
plt.text(0.5,3.01, f'{f_plotting.sig_marker(p1)}',horizontalalignment='center', fontsize=12)


#p2 = scipy.stats.ttest_rel(euDists['fef'][1].mean(1), euDists['fef'][2].mean(1))[-1]
#p2,_,_ = f_stats.bootstrap95_p(euDists['fef'][1].mean(1), euDists['fef'][2].mean(1))
p2 = f_stats.permutation_p_diff(euDists['fef'][1].mean(1), euDists['fef'][2].mean(1))

plt.plot(lineh+1, np.full_like(lineh, 3), 'k-')
plt.plot(np.full_like(linev, 1.3), linev, 'k-')
plt.plot(np.full_like(linev, 1.7), linev, 'k-')
plt.text(1.5,3.01, f'{f_plotting.sig_marker(p2)}',horizontalalignment='center', fontsize=12)

# draw temporary red and blue lines and use them to create a legend
plt.plot([], c='dimgrey', label='Retarget')
plt.plot([], c='lightgrey', label='Distraction')
plt.legend(bbox_to_anchor=(1.6, 0.6), fontsize = 10)#loc = 'right',

plt.xticks([0.5,1.5],['LPFC','FEF'], rotation=30)
plt.xlabel('Regions', labelpad = 5, fontsize = 12)
plt.ylabel('Euclidean Distance (Std. Scaled)', labelpad = 3, fontsize = 10)
plt.ylim(0,3.25)
plt.title('Mean Drift, LD2-LD1', fontsize = 15, pad=10)
plt.show()

#fig.savefig(f'{save_path}/driftDist_monkeys_centroids.tif', bbox_inches='tight')

    





# In[] stability ratio

performance1 = np.load(f'{phd_path}/fitting planes/pooled/w&w/' + 'performanceX1.npy', allow_pickle=True).item()
performance2 = np.load(f'{phd_path}/fitting planes/pooled/w&w/' + 'performanceX2.npy', allow_pickle=True).item()
performance1_shuff = np.load(f'{phd_path}/fitting planes/pooled/w&w/' + 'performanceX1_shuff.npy', allow_pickle=True).item()
performance2_shuff = np.load(f'{phd_path}/fitting planes/pooled/w&w/' + 'performanceX2_shuff.npy', allow_pickle=True).item()

decode_proj1_3d = np.load(f'{phd_path}/fitting planes/pooled/w&w/' + 'decode_proj1X_3d.npy', allow_pickle=True).item()
decode_proj2_3d = np.load(f'{phd_path}/fitting planes/pooled/w&w/' + 'decode_proj2X_3d.npy', allow_pickle=True).item()
decode_proj1_shuff_all_3d = np.load(f'{phd_path}/fitting planes/pooled/w&w/' + 'decode_proj1X_shuff_all_3d.npy', allow_pickle=True).item()
decode_proj2_shuff_all_3d = np.load(f'{phd_path}/fitting planes/pooled/w&w/' + 'decode_proj2X_shuff_all_3d.npy', allow_pickle=True).item()

# In[]
nPerms = 100
# define esti windows
d1 = np.arange(800,1300+bins,bins)
d1x = [tbins.tolist().index(t) for t in d1]

d2 = np.arange(2100,2600+bins,bins)
d2x = [tbins.tolist().index(t) for t in d2]

##############
# full space #
##############

stab_ratioD1s_f, stab_ratioD2s_f = {}, {}
stab_ratioD1s_shuff_f, stab_ratioD2s_shuff_f = {}, {}

for region in ('dlpfc', 'fef'):
    
    stab_ratioD1, stab_ratioD2 = [], []
    stab_ratioD1_shuff, stab_ratioD2_shuff = [], []
    
    for n in range(nIters):
        
        pfm1_ret, pfm1_dis = performance1['Retarget'][region][n], performance1['Distractor'][region][n]
        pfm2_ret, pfm2_dis = performance2['Retarget'][region][n], performance2['Distractor'][region][n]
        
        stab_ratioD1 += [np.mean((f_decoding.stability_ratio(pfm1_ret[d1x,:][:,d1x]), f_decoding.stability_ratio(pfm1_dis[d1x,:][:,d1x])))] # for d1, use only I1
        stab_ratioD2 += [np.mean((f_decoding.stability_ratio(pfm2_ret[d2x,:][:,d2x]), f_decoding.stability_ratio(pfm1_dis[d2x,:][:,d2x])))] # for d2, use only choice item
        
        for npm in range(nPerms):
            pfm1_ret_shuff, pfm1_dis_shuff = performance1_shuff['Retarget'][region][n][:,:,npm], performance1_shuff['Distractor'][region][n][:,:,npm]
            pfm2_ret_shuff, pfm2_dis_shuff = performance2_shuff['Retarget'][region][n][:,:,npm], performance2_shuff['Distractor'][region][n][:,:,npm]
            
            stab_ratioD1_shuff += [np.mean((f_decoding.stability_ratio(pfm1_ret_shuff[d1x,:][:,d1x]), f_decoding.stability_ratio(pfm1_dis_shuff[d1x,:][:,d1x])))] # for d1, use only I1
            stab_ratioD2_shuff += [np.mean((f_decoding.stability_ratio(pfm2_ret_shuff[d2x,:][:,d2x]), f_decoding.stability_ratio(pfm1_dis_shuff[d2x,:][:,d2x])))] # for d2, use only choice item
        
    stab_ratioD1s_f[region], stab_ratioD2s_f[region] = np.array(stab_ratioD1), np.array(stab_ratioD2)
    stab_ratioD1s_shuff_f[region], stab_ratioD2s_shuff_f[region] = np.array(stab_ratioD1_shuff), np.array(stab_ratioD2_shuff)


###########
# readout #
###########

stab_ratioD1s_r, stab_ratioD2s_r = {}, {}
stab_ratioD1s_shuff_r, stab_ratioD2s_shuff_r = {}, {}

for region in ('dlpfc','fef'):

    stab_ratioD1, stab_ratioD2 = [], []
    stab_ratioD1_shuff, stab_ratioD2_shuff = [], []
    
    for n in range(nIters):
        
        pfm1_ret, pfm1_dis = decode_proj1_3d[region][1][n], decode_proj1_3d[region][2][n]
        pfm2_ret, pfm2_dis = decode_proj2_3d[region][1][n], decode_proj2_3d[region][2][n]
        
        stab_ratioD1 += [np.mean((f_decoding.stability_ratio(pfm1_ret[d1x,:][:,d1x]), f_decoding.stability_ratio(pfm1_dis[d1x,:][:,d1x])))] # for d1, use only I1
        stab_ratioD2 += [np.mean((f_decoding.stability_ratio(pfm2_ret[d2x,:][:,d2x]), f_decoding.stability_ratio(pfm1_dis[d2x,:][:,d2x])))] # for d2, use only choice item
        
        for npm in range(nIters*nPerms):
            pfm1_ret_shuff, pfm1_dis_shuff = decode_proj1_shuff_all_3d[region][1][npm], decode_proj1_shuff_all_3d[region][2][npm]
            pfm2_ret_shuff, pfm2_dis_shuff = decode_proj2_shuff_all_3d[region][1][npm], decode_proj2_shuff_all_3d[region][2][npm]
            
            stab_ratioD1_shuff += [np.mean((f_decoding.stability_ratio(pfm1_ret_shuff[d1x,:][:,d1x]), f_decoding.stability_ratio(pfm1_dis_shuff[d1x,:][:,d1x])))] # for d1, use only I1
            stab_ratioD2_shuff += [np.mean((f_decoding.stability_ratio(pfm2_ret_shuff[d2x,:][:,d2x]), f_decoding.stability_ratio(pfm1_dis_shuff[d2x,:][:,d2x])))] # for d2, use only choice item
        
    stab_ratioD1s_r[region], stab_ratioD2s_r[region] = np.array(stab_ratioD1), np.array(stab_ratioD2)
    stab_ratioD1s_shuff_r[region], stab_ratioD2s_shuff_r[region] = np.array(stab_ratioD1_shuff), np.array(stab_ratioD2_shuff)


###############
# plot params #
###############

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

########
# plot #
########

lineh = np.arange(0.3,0.7,0.001)
linev = np.arange(1.11,1.115,0.001)

fig, axes = plt.subplots(1,2, sharey=True, figsize=(6,3), dpi=300)

#fig = plt.figure(figsize=(3, 6), dpi=100)

# full space
ax = axes.flatten()[0]

ax.boxplot([stab_ratioD1s_f['dlpfc']], positions=[0.3], patch_artist=True, widths = 0.2, boxprops=boxprops1, flierprops=flierprops1, 
                  meanprops=meanpointprops1, medianprops=medianprops, capprops = capprops1, whiskerprops = whiskerprops1, meanline=False, showmeans=True)
ax.boxplot([stab_ratioD1s_f['fef']], positions=[1.3], patch_artist=True, widths = 0.2, boxprops=boxprops2, flierprops=flierprops2, 
                  meanprops=meanpointprops2, medianprops=medianprops, capprops = capprops2, whiskerprops = whiskerprops2, meanline=False, showmeans=True)

ax.boxplot([stab_ratioD2s_f['dlpfc']], positions=[0.7], patch_artist=True, widths = 0.2, boxprops=dict(boxprops1,facecolor=color1_), flierprops=dict(flierprops1,markeredgecolor=color1_, markerfacecolor=color1_), 
                  meanprops=dict(meanpointprops1,markeredgecolor=color1_), medianprops=medianprops, capprops = dict(capprops1,color=color1_), whiskerprops = dict(whiskerprops1,color=color1_), meanline=False, showmeans=True)
ax.boxplot([stab_ratioD2s_f['fef']], positions=[1.7], patch_artist=True, widths = 0.2, boxprops=dict(boxprops2,facecolor=color2_), flierprops=dict(flierprops2,markeredgecolor=color2_, markerfacecolor=color2_), 
                  meanprops=dict(meanpointprops2,markeredgecolor=color2_), medianprops=medianprops, capprops = dict(capprops2,color=color2_), whiskerprops = dict(whiskerprops2,color=color2_), meanline=False, showmeans=True)



p1 = f_stats.permutation_p(stab_ratioD1s_f['dlpfc'].mean(), stab_ratioD1s_shuff_f['dlpfc'])
ax.text(0.3,1.1, f'{f_plotting.sig_marker(p1)}',horizontalalignment='center', fontsize=12)


p2 = f_stats.permutation_p(stab_ratioD2s_f['dlpfc'].mean(), stab_ratioD2s_shuff_f['dlpfc'])
ax.text(0.7,1.1, f'{f_plotting.sig_marker(p2)}',horizontalalignment='center', fontsize=12)


p3 = f_stats.permutation_p(stab_ratioD1s_f['fef'].mean(), stab_ratioD1s_shuff_f['fef'])
ax.text(1.3,1.1, f'{f_plotting.sig_marker(p3)}',horizontalalignment='center', fontsize=12)


p4 = f_stats.permutation_p(stab_ratioD2s_f['fef'].mean(), stab_ratioD2s_shuff_f['fef'])
ax.text(1.7,1.1, f'{f_plotting.sig_marker(p4)}',horizontalalignment='center', fontsize=12)

#p5,_,_ = f_stats.bootstrap95_p(stab_ratioD1s_f['dlpfc'], stab_ratioD2s_f['dlpfc'])
p5 = f_stats.permutation_p_diff(stab_ratioD1s_f['dlpfc'], stab_ratioD2s_f['dlpfc'])
ax.plot(lineh, np.full_like(lineh, 1.115), 'k-')
ax.plot(np.full_like(linev, 0.3), linev, 'k-')
ax.plot(np.full_like(linev, 0.7), linev, 'k-')
ax.text(0.5,1.125, f'{f_plotting.sig_marker(p5, ns_note=True)}',horizontalalignment='center', fontsize=12)

#p6,_,_ = f_stats.bootstrap95_p(stab_ratioD1s_f['fef'], stab_ratioD2s_f['fef'])
p6 = f_stats.permutation_p_diff(stab_ratioD1s_f['fef'], stab_ratioD2s_f['fef'])
ax.plot(lineh+1, np.full_like(lineh, 1.115), 'k-')
ax.plot(np.full_like(linev, 0.3)+1, linev, 'k-')
ax.plot(np.full_like(linev, 0.7)+1, linev, 'k-')
ax.text(1.5,1.125, f'{f_plotting.sig_marker(p6, ns_note=True)}',horizontalalignment='center', fontsize=12)

xlims = ax.get_xlim()
lineThresholds = np.arange(xlims[0],xlims[1]+0.01,0.01)
ax.plot(lineThresholds, np.full_like(lineThresholds,1), 'k--', alpha = 0.5)

# draw temporary red and blue lines and use them to create a legend
#plt.plot([], c='dimgrey', label='Delay1')
#plt.plot([], c='lightgrey', label='Delay2')
#plt.legend(bbox_to_anchor=(1.5, 0.5))#loc = 'right',

ax.set_xticks([0.5,1.5],['LPFC','FEF'], rotation=30)
ax.set_xlabel('Region', labelpad = 5, fontsize = 12)
ax.set_ylabel('Code Stability Ratio', labelpad = 3, fontsize = 12)
ax.set_ylim(0.9,1.145)
ax.set_title('Full Space', fontsize = 15, pad=10)
#plt.show()
#fig.savefig(f'{save_path}/infoStabRatio_readout_monkeys.tif')


# readout
ax = axes.flatten()[1]
#fig = plt.figure(figsize=(3, 6), dpi=100)

ax.boxplot([stab_ratioD1s_r['dlpfc']], positions=[0.3], patch_artist=True, widths = 0.2, boxprops=boxprops1, flierprops=flierprops1, 
                  meanprops=meanpointprops1, medianprops=medianprops, capprops = capprops1, whiskerprops = whiskerprops1, meanline=False, showmeans=True)
ax.boxplot([stab_ratioD1s_r['fef']], positions=[1.3], patch_artist=True, widths = 0.2, boxprops=boxprops2, flierprops=flierprops2, 
                  meanprops=meanpointprops2, medianprops=medianprops, capprops = capprops2, whiskerprops = whiskerprops2, meanline=False, showmeans=True)

ax.boxplot([stab_ratioD2s_r['dlpfc']], positions=[0.7], patch_artist=True, widths = 0.2, boxprops=dict(boxprops1,facecolor=color1_), flierprops=dict(flierprops1,markeredgecolor=color1_, markerfacecolor=color1_), 
                  meanprops=dict(meanpointprops1,markeredgecolor=color1_), medianprops=medianprops, capprops = dict(capprops1,color=color1_), whiskerprops = dict(whiskerprops1,color=color1_), meanline=False, showmeans=True)
ax.boxplot([stab_ratioD2s_r['fef']], positions=[1.7], patch_artist=True, widths = 0.2, boxprops=dict(boxprops2,facecolor=color2_), flierprops=dict(flierprops2,markeredgecolor=color2_, markerfacecolor=color2_), 
                  meanprops=dict(meanpointprops2,markeredgecolor=color2_), medianprops=medianprops, capprops = dict(capprops2,color=color2_), whiskerprops = dict(whiskerprops2,color=color2_), meanline=False, showmeans=True)



p1 = f_stats.permutation_p(stab_ratioD1s_r['dlpfc'].mean(), stab_ratioD1s_shuff_r['dlpfc'])
ax.text(0.3,1.1, f'{f_plotting.sig_marker(p1)}',horizontalalignment='center', fontsize=12)


p2 = f_stats.permutation_p(stab_ratioD2s_r['dlpfc'].mean(), stab_ratioD2s_shuff_r['dlpfc'])
ax.text(0.7,1.1, f'{f_plotting.sig_marker(p2)}',horizontalalignment='center', fontsize=12)


p3 = f_stats.permutation_p(stab_ratioD1s_r['fef'].mean(), stab_ratioD1s_shuff_r['fef'])
ax.text(1.3,1.1, f'{f_plotting.sig_marker(p3)}',horizontalalignment='center', fontsize=12)


p4 = f_stats.permutation_p(stab_ratioD2s_r['fef'].mean(), stab_ratioD2s_shuff_r['fef'])
ax.text(1.7,1.1, f'{f_plotting.sig_marker(p4)}',horizontalalignment='center', fontsize=12)

#p5,_,_ = f_stats.bootstrap95_p(stab_ratioD1s_r['dlpfc'], stab_ratioD2s_r['dlpfc'])
p5 = f_stats.permutation_p_diff(stab_ratioD1s_r['dlpfc'], stab_ratioD2s_r['dlpfc'])
ax.plot(lineh, np.full_like(lineh, 1.115), 'k-')
ax.plot(np.full_like(linev, 0.3), linev, 'k-')
ax.plot(np.full_like(linev, 0.7), linev, 'k-')
ax.text(0.5,1.125, f'{f_plotting.sig_marker(p5, ns_note=True)}',horizontalalignment='center', fontsize=12)

#p6,_,_ = f_stats.bootstrap95_p(stab_ratioD1s_r['fef'], stab_ratioD2s_r['fef'])
p6 = f_stats.permutation_p_diff(stab_ratioD1s_r['fef'], stab_ratioD2s_r['fef'])
ax.plot(lineh+1, np.full_like(lineh, 1.115), 'k-')
ax.plot(np.full_like(linev, 0.3)+1, linev, 'k-')
ax.plot(np.full_like(linev, 0.7)+1, linev, 'k-')
ax.text(1.5,1.125, f'{f_plotting.sig_marker(p6, ns_note=True)}',horizontalalignment='center', fontsize=12)

xlims = ax.get_xlim()
lineThresholds = np.arange(xlims[0],xlims[1]+0.01,0.01)
ax.plot(lineThresholds, np.full_like(lineThresholds,1), 'k--', alpha = 0.5)

# draw temporary red and blue lines and use them to create a legend
ax.plot([], c='dimgrey', label='Delay1')
ax.plot([], c='lightgrey', label='Delay2')
ax.legend(bbox_to_anchor=(1.55, 0.5))#loc = 'right',

ax.set_xticks([0.5,1.5],['LPFC','FEF'], rotation=30)
ax.set_xlabel('Regions', labelpad = 5, fontsize = 12)
#ax.set_ylabel('Code Stability Ratio', labelpad = 3, fontsize = 12)
ax.set_ylim(0.9,1.145)
ax.set_title('Readout Subspace', fontsize = 15, pad=10)

plt.suptitle('Information Stability, Monkeys', fontsize = 20, y=1.1)
plt.show()

fig.savefig(f'{save_path}/infoStabRatio_monkey_.tif', bbox_inches='tight')


# In[] code morphing

performance1 = np.load(f'{phd_path}/fitting planes/pooled/w&w/' + 'performanceX1.npy', allow_pickle=True).item()
performance2 = np.load(f'{phd_path}/fitting planes/pooled/w&w/' + 'performanceX2.npy', allow_pickle=True).item()
performance1_shuff = np.load(f'{phd_path}/fitting planes/pooled/w&w/' + 'performanceX1_shuff.npy', allow_pickle=True).item()
performance2_shuff = np.load(f'{phd_path}/fitting planes/pooled/w&w/' + 'performanceX2_shuff.npy', allow_pickle=True).item()

decode_proj1_3d = np.load(f'{phd_path}/fitting planes/pooled/w&w/' + 'decode_proj1X_3d.npy', allow_pickle=True).item()
decode_proj2_3d = np.load(f'{phd_path}/fitting planes/pooled/w&w/' + 'decode_proj2X_3d.npy', allow_pickle=True).item()
decode_proj1_shuff_all_3d = np.load(f'{phd_path}/fitting planes/pooled/w&w/' + 'decode_proj1X_shuff_all_3d.npy', allow_pickle=True).item()
decode_proj2_shuff_all_3d = np.load(f'{phd_path}/fitting planes/pooled/w&w/' + 'decode_proj2X_shuff_all_3d.npy', allow_pickle=True).item()

# In[]
nPerms = 100
# define esti windows
d1 = np.arange(800,1300+bins,bins)
d1x = [tbins.tolist().index(t) for t in d1]

d2 = np.arange(2100,2600+bins,bins)
d2x = [tbins.tolist().index(t) for t in d2]

##############
# full space #
##############

codeMorphDs_f = {}#, {}, codeMorphD21s_f
codeMorphDs_shuff_f = {}#, {}, codeMorphD21s_shuff_f

for region in ('dlpfc', 'fef'):
    
    codeMorphD = [] #, [], codeMorphD21
    codeMorphD_shuff = []#, [], codeMorphD21_shuff
    
    for n in range(nIters):
        
        pfm1_dis = performance1['Distractor'][region][n] # pfm1_ret, performance1['Retarget'][region][n], 
        #pfm2_dis = performance2['Distractor'][region][n] # performance2['Retarget'][region][n], pfm2_ret, 
        
        codeMorphD += [np.mean((f_decoding.code_morphing(pfm1_dis[d1x,d1x], pfm1_dis[d1x,d2x]), f_decoding.code_morphing(pfm1_dis[d2x,d2x], pfm1_dis[d2x,d1x])))] # distractor only, d1
        #stab_ratioD2 += [np.mean((f_decoding.stability_ratio(pfm2_ret[d2x,:][:,d2x]), f_decoding.stability_ratio(pfm1_dis[d2x,:][:,d2x])))] # for d2, use only choice item
        
        for npm in range(nPerms):
            pfm1_dis_shuff = performance1_shuff['Distractor'][region][n][:,:,npm] #performance1_shuff['Retarget'][region][n][:,:,npm], pfm1_ret_shuff, 
            #pfm2_ret_shuff, pfm2_dis_shuff = performance2_shuff['Retarget'][region][n][:,:,npm], performance2_shuff['Distractor'][region][n][:,:,npm]
            
            codeMorphD_shuff += [np.mean((f_decoding.code_morphing(pfm1_dis_shuff[d1x,d1x], pfm1_dis_shuff[d1x,d2x]), f_decoding.code_morphing(pfm1_dis_shuff[d2x,d2x], pfm1_dis_shuff[d2x,d1x])))] # for d1, use only I1
            #stab_ratioD2_shuff += [np.mean((f_decoding.stability_ratio(pfm2_ret_shuff[d2x,:][:,d2x]), f_decoding.stability_ratio(pfm1_dis_shuff[d2x,:][:,d2x])))] # for d2, use only choice item
        
    codeMorphDs_f[region] = np.array(codeMorphD) #, np.array(stab_ratioD2), stab_ratioD2s_f[region]
    codeMorphDs_shuff_f[region] = np.array(codeMorphD_shuff) #, np.array(stab_ratioD2_shuff), stab_ratioD2s_shuff_f[region]


###########
# readout #
###########

codeMorphDs_r = {}#, {}, codeMorphD21s_f
codeMorphDs_shuff_r = {}#, {}, codeMorphD21s_shuff_f

for region in ('dlpfc', 'fef'):
    
    codeMorphD = [] #, [], codeMorphD21
    codeMorphD_shuff = []#, [], codeMorphD21_shuff
    
    for n in range(nIters):
        
        pfm1_dis = decode_proj1_3d[region][2][n] # pfm1_ret, performance1['Retarget'][region][n], 
        #pfm2_dis = performance2['Distractor'][region][n] # performance2['Retarget'][region][n], pfm2_ret, 
        
        codeMorphD += [np.mean((f_decoding.code_morphing(pfm1_dis[d1x,d1x], pfm1_dis[d1x,d2x]), f_decoding.code_morphing(pfm1_dis[d2x,d2x], pfm1_dis[d2x,d1x])))] # distractor only, d1
        #stab_ratioD2 += [np.mean((f_decoding.stability_ratio(pfm2_ret[d2x,:][:,d2x]), f_decoding.stability_ratio(pfm1_dis[d2x,:][:,d2x])))] # for d2, use only choice item
        
        for npm in range(nPerms):
            pfm1_dis_shuff = decode_proj1_shuff_all_3d[region][2][npm] #performance1_shuff['Retarget'][region][n][:,:,npm], pfm1_ret_shuff, 
            #pfm2_ret_shuff, pfm2_dis_shuff = performance2_shuff['Retarget'][region][n][:,:,npm], performance2_shuff['Distractor'][region][n][:,:,npm]
            
            codeMorphD_shuff += [np.mean((f_decoding.code_morphing(pfm1_dis_shuff[d1x,d1x], pfm1_dis_shuff[d1x,d2x]), f_decoding.code_morphing(pfm1_dis_shuff[d2x,d2x], pfm1_dis_shuff[d2x,d1x])))] # for d1, use only I1
            #stab_ratioD2_shuff += [np.mean((f_decoding.stability_ratio(pfm2_ret_shuff[d2x,:][:,d2x]), f_decoding.stability_ratio(pfm1_dis_shuff[d2x,:][:,d2x])))] # for d2, use only choice item
        
    codeMorphDs_r[region] = np.array(codeMorphD) #, np.array(stab_ratioD2), stab_ratioD2s_f[region]
    codeMorphDs_shuff_r[region] = np.array(codeMorphD_shuff) #, np.array(stab_ratioD2_shuff), stab_ratioD2s_shuff_f[region]




###############
# plot params #
###############

color1, color2 = '#185337', '#804098'#, color1_, color2_, '#96d9ad', '#c4a2d1'

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

########
# plot #
########

lineh = np.arange(0.5,1.5,0.001)
linev = np.arange(1.11,1.115,0.001)

fig, axes = plt.subplots(1,2, sharey=True, figsize=(6,3), dpi=300)

#fig = plt.figure(figsize=(3, 6), dpi=100)

# full space
ax = axes.flatten()[0]

ax.boxplot([codeMorphDs_f['dlpfc']], positions=[0.5], patch_artist=True, widths = 0.2, boxprops=boxprops1, flierprops=flierprops1, 
                  meanprops=meanpointprops1, medianprops=medianprops, capprops = capprops1, whiskerprops = whiskerprops1, meanline=False, showmeans=True)
ax.boxplot([codeMorphDs_f['fef']], positions=[1.5], patch_artist=True, widths = 0.2, boxprops=boxprops2, flierprops=flierprops2, 
                  meanprops=meanpointprops2, medianprops=medianprops, capprops = capprops2, whiskerprops = whiskerprops2, meanline=False, showmeans=True)


p1 = f_stats.permutation_p(codeMorphDs_f['dlpfc'].mean(), codeMorphDs_shuff_f['dlpfc'])
ax.text(0.5,1.1, f'{f_plotting.sig_marker(p1)}',horizontalalignment='center', fontsize=12)


p2 = f_stats.permutation_p(codeMorphDs_f['fef'].mean(), codeMorphDs_shuff_f['fef'])
ax.text(1.5,1.1, f'{f_plotting.sig_marker(p2)}',horizontalalignment='center', fontsize=12)



xlims = ax.get_xlim()
lineThresholds = np.arange(xlims[0],xlims[1]+0.01,0.01)
ax.plot(lineThresholds, np.full_like(lineThresholds,1), 'k--', alpha = 0.5)

# draw temporary red and blue lines and use them to create a legend
#plt.plot([], c='dimgrey', label='Delay1')
#plt.plot([], c='lightgrey', label='Delay2')
#plt.legend(bbox_to_anchor=(1.5, 0.5))#loc = 'right',

ax.set_xticks([0.5,1.5],['LPFC','FEF'], rotation=30)
ax.set_xlabel('Region', labelpad = 5, fontsize = 12)
ax.set_ylabel('D1-D1/D1-D2 (& vice versa)', labelpad = 3, fontsize = 12)
ax.set_ylim(top=1.2)
ax.set_title('Full Space', fontsize = 15, pad=10)
#plt.show()
#fig.savefig(f'{save_path}/infoStabRatio_readout_monkeys.tif')


# readout
ax = axes.flatten()[1]
#fig = plt.figure(figsize=(3, 6), dpi=100)

ax.boxplot([codeMorphDs_r['dlpfc']], positions=[0.5], patch_artist=True, widths = 0.2, boxprops=boxprops1, flierprops=flierprops1, 
                  meanprops=meanpointprops1, medianprops=medianprops, capprops = capprops1, whiskerprops = whiskerprops1, meanline=False, showmeans=True)
ax.boxplot([codeMorphDs_r['fef']], positions=[1.5], patch_artist=True, widths = 0.2, boxprops=boxprops2, flierprops=flierprops2, 
                  meanprops=meanpointprops2, medianprops=medianprops, capprops = capprops2, whiskerprops = whiskerprops2, meanline=False, showmeans=True)


p1 = f_stats.permutation_p(codeMorphDs_r['dlpfc'].mean(), codeMorphDs_shuff_r['dlpfc'])
ax.text(0.5,1.1, f'{f_plotting.sig_marker(p1)}',horizontalalignment='center', fontsize=12)


p2 = f_stats.permutation_p(codeMorphDs_r['fef'].mean(), codeMorphDs_shuff_r['fef'])
ax.text(1.5,1.1, f'{f_plotting.sig_marker(p2)}',horizontalalignment='center', fontsize=12)



xlims = ax.get_xlim()
lineThresholds = np.arange(xlims[0],xlims[1]+0.01,0.01)
ax.plot(lineThresholds, np.full_like(lineThresholds,1), 'k--', alpha = 0.5)


ax.set_xticks([0.5,1.5],['LPFC','FEF'], rotation=30)
ax.set_xlabel('Regions', labelpad = 5, fontsize = 12)
#ax.set_ylabel('Code Stability Ratio', labelpad = 3, fontsize = 12)
#ax.set_ylim(0.9,1.145)
ax.set_title('Readout Subspace', fontsize = 15, pad=10)

plt.suptitle('Code Morphing, Monkeys', fontsize = 20, y=1.1)
plt.show()

fig.savefig(f'{save_path}/codeMorphing_monkey_.tif', bbox_inches='tight')
# In[] distractor information quantification

d1 = np.arange(1600,2100+bins,bins)
d1x = [tbins.tolist().index(t) for t in d1]

#pfm22_full = {k:np.array(performance2['Distractor'][k])[:,d1x,:][:,:,d1x].mean(-1).mean(-1) for k in performance2['Distractor'].keys()}
#pfm22_readout = {k: decode_proj2_3d[k][2][:,d1x,:][:,:,d1x].mean(-1).mean(-1) for k in decode_proj2_3d.keys()}

#pfm22_full_shuff = {k:np.concatenate(performance2_shuff['Distractor'][k],axis=-1).swapaxes(1,2).swapaxes(0,1)[:,d1x,:][:,:,d1x].mean(-1).mean(-1) for k in performance2_shuff['Distractor'].keys()}
#pfm22_readout_shuff = {k: decode_proj2_shuff_all_3d[k][2][:,d1x,:][:,:,d1x].mean(-1).mean(-1) for k in decode_proj2_shuff_all_3d.keys()}


pfm22_full = {k:np.array(performance2['Distractor'][k])[:,d1x,d1x].mean(-1) for k in performance2['Distractor'].keys()}
pfm22_readout = {k: decode_proj2_3d[k][2][:,d1x,d1x].mean(-1) for k in decode_proj2_3d.keys()}

pfm22_full_shuff = {k:np.concatenate(performance2_shuff['Distractor'][k],axis=-1).swapaxes(1,2).swapaxes(0,1)[:,d1x,d1x].mean(-1) for k in performance2_shuff['Distractor'].keys()}
pfm22_readout_shuff = {k: decode_proj2_shuff_all_3d[k][2][:,d1x,d1x].mean(-1) for k in decode_proj2_shuff_all_3d.keys()}


###############
# plot params #
###############

color1, color2 = '#185337', '#804098'#, color1_, color2_, '#96d9ad', '#c4a2d1'


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

########
# plot #
########

lineh = np.arange(0.5,1.5,0.001)
linev = np.arange(0.61,0.62,0.0001)

#fig = plt.figure(figsize=(3, 3), dpi=300)
fig, axes = plt.subplots(1,2, sharey=True, figsize=(6,3), dpi=300)

# full space
ax = axes.flatten()[0]

ax.boxplot([pfm22_full['dlpfc']], positions=[0.5], patch_artist=True, widths = 0.2, boxprops=boxprops1, flierprops=flierprops1, 
                  meanprops=meanpointprops1, medianprops=medianprops, capprops = capprops1, whiskerprops = whiskerprops1, meanline=False, showmeans=True)
ax.boxplot([pfm22_full['fef']], positions=[1.5], patch_artist=True, widths = 0.2, boxprops=boxprops2, flierprops=flierprops2, 
                  meanprops=meanpointprops2, medianprops=medianprops, capprops = capprops2, whiskerprops = whiskerprops2, meanline=False, showmeans=True)



p1 = f_stats.permutation_p(pfm22_full['dlpfc'].mean(), pfm22_full_shuff['dlpfc'], tail = 'greater')
#_,p1 = f_stats.bslTTest_1samp(pfm22_full['dlpfc'], 0.333)
ax.text(0.5,0.6, f'{f_plotting.sig_marker(p1)}',horizontalalignment='center', fontsize=12)

p2 = f_stats.permutation_p(pfm22_full['fef'].mean(), pfm22_full_shuff['fef'], tail = 'greater')
#_,p2 = f_stats.bslTTest_1samp(pfm22_full['fef'], 0.333)
ax.text(1.5,0.6, f'{f_plotting.sig_marker(p2)}',horizontalalignment='center', fontsize=12)


xlims = ax.get_xlim()
lineThresholds = np.arange(xlims[0],xlims[1]+0.01,0.01)
#plt.plot(lineThresholds, np.full_like(lineThresholds,0), 'k-.', alpha = 0.5)
ax.plot(lineThresholds, np.full_like(lineThresholds,0.33), 'k--', alpha = 0.5, linewidth=1)

ax.set_xticks([0.5,1.5],['LPFC','FEF'], rotation=30)
ax.set_xlabel('Region', labelpad = 5, fontsize = 12)
ax.set_ylabel('Decodability', labelpad = 3, fontsize = 12)
#ax.set_ylim(top=1)
ax.set_title('Full Space', fontsize = 15, pad=10)


# readout subspace
ax = axes.flatten()[1]

ax.boxplot([pfm22_readout['dlpfc']], positions=[0.5], patch_artist=True, widths = 0.2, boxprops=boxprops1, flierprops=flierprops1, 
                  meanprops=meanpointprops1, medianprops=medianprops, capprops = capprops1, whiskerprops = whiskerprops1, meanline=False, showmeans=True)
ax.boxplot([pfm22_readout['fef']], positions=[1.5], patch_artist=True, widths = 0.2, boxprops=boxprops2, flierprops=flierprops2, 
                  meanprops=meanpointprops2, medianprops=medianprops, capprops = capprops2, whiskerprops = whiskerprops2, meanline=False, showmeans=True)



p1 = f_stats.permutation_p(pfm22_readout['dlpfc'].mean(), pfm22_readout_shuff['dlpfc'], tail = 'greater')
ax.text(0.5,0.6, f'{f_plotting.sig_marker(p1)}',horizontalalignment='center', fontsize=12)

p2 = f_stats.permutation_p(pfm22_readout['fef'].mean(), pfm22_readout_shuff['fef'], tail = 'greater')
ax.text(1.5,0.6, f'{f_plotting.sig_marker(p2)}',horizontalalignment='center', fontsize=12)


xlims = ax.get_xlim()
lineThresholds = np.arange(xlims[0],xlims[1]+0.01,0.01)
#plt.plot(lineThresholds, np.full_like(lineThresholds,0), 'k-.', alpha = 0.5)
ax.plot(lineThresholds, np.full_like(lineThresholds,0.33), 'k--', alpha = 0.5, linewidth=1)



# draw temporary red and blue lines and use them to create a legend
#ax.plot([], c='dimgrey', label='Full Space')
#ax.plot([], c='lightgrey', label='Readout Subspace')
#ax.legend(bbox_to_anchor=(1.0, 0.6))#loc = 'right',

ax.set_xticks([0.5,1.5],['LPFC','FEF'], rotation=30)
ax.set_xlabel('Region', labelpad = 5, fontsize = 12)
#ax.set_ylabel('LDA Decodability', labelpad = 3, fontsize = 12)
ax.set_ylim(top=0.65)
ax.set_title('Readout Subspace', fontsize = 15, pad=10)


plt.suptitle('Distractor Information, ED2, Monkeys', fontsize = 20, y=1.1)
plt.show()

fig.savefig(f'{save_path}/distractorInfo_monkeys.tif', bbox_inches='tight')



#%% decode, permutation on test labels
decode_proj1_3dX, decode_proj2_3dX = {},{}
decode_proj1_shuff_all_3dX, decode_proj2_shuff_all_3dX = {},{}

decode_proj1_3dW, decode_proj2_3dW = {},{}
decode_proj1_shuff_all_3dW, decode_proj2_shuff_all_3dW = {},{}


for region in ('dlpfc','fef'):
    
    print(f'{region}')
    
    decode_proj1_3dX[region], decode_proj2_3dX[region] = {}, {}
    decode_proj1_shuff_all_3dX[region], decode_proj2_shuff_all_3dX[region] = {},{}
    
    decode_proj1_3dW[region], decode_proj2_3dW[region] = {}, {}
    decode_proj1_shuff_all_3dW[region], decode_proj2_shuff_all_3dW[region] = {},{}
    

    # estimate decodability by ttype
    for tt in ttypes:
        
        decode_proj1T_3dX = np.zeros((nIters, nBoots, len(tbins), len(tbins))) # pca1st 3d coordinates
        decode_proj2T_3dX = np.zeros((nIters, nBoots, len(tbins), len(tbins)))
        
        decode_proj1T_3dW = np.zeros((nIters, nBoots, len(tbins), )) # pca1st 3d coordinates
        decode_proj2T_3dW = np.zeros((nIters, nBoots, len(tbins), ))
        
        # shuff
        decode_proj1T_3d_shuffX = np.zeros((nIters, nBoots*nPerms, len(tbins), len(tbins)))
        decode_proj2T_3d_shuffX = np.zeros((nIters, nBoots*nPerms, len(tbins), len(tbins)))
        
        decode_proj1T_3d_shuffW = np.zeros((nIters, nBoots*nPerms, len(tbins),))
        decode_proj2T_3d_shuffW = np.zeros((nIters, nBoots*nPerms, len(tbins),))
        
        for n in range(nIters):
            #for nbt in range(nBoots):
            
            print(f'{n}')
            
            # trial info
            trialInfo_CT = trialInfos_C[region][n][0]
            trialInfo_CT_tt = trialInfo_CT[trialInfo_CT.type == tt]#.reset_index(drop = True)
            idx_tt = trialInfo_CT_tt.index.tolist()#.trial_index.values
            
            vecs_CT = vecs_C[region][n][0] # choice plane vecs
            projs_CT = projs_C[region][n][0] #
            
            

            vec_normal_CT = np.cross(vecs_CT[0],vecs_CT[1]) # plane normal vec
            center_CT = projs_CT.mean(0) # plane center
            
            #data_3pc_CT = data_3pc_C[region][n][nbt][idx_tt,:,:] # 3pc states from tt trials
            
            # smooth to 50ms bins
            #ntrialT, ncellT, ntimeT = data_3pc_CT.shape
            #data_3pc_CT_smooth = np.mean(data_3pc_CT.reshape(ntrialT, ncellT, int(ntimeT/(bins/dt)),int(bins/dt)),axis = 3)
            
            data_3pc_CT_smooth = data_3pc_C[region][n][0][idx_tt,:,:] # 3pc states from tt trials
            projs_All_CT = np.array([[f_subspace.proj_on_plane(p[:,t], vec_normal_CT, center_CT) for t in range(len(tbins))] for p in data_3pc_CT_smooth]) # projections on the plane
            projs_All_CT = np.swapaxes(projs_All_CT,1,2)
            
            # labels
            Y = trialInfo_CT_tt.loc[:,Y_columnsLabels].values
            ntrial = len(trialInfo_CT_tt)
            
            ### split into train and test sets
            train_setID = np.sort(np.random.choice(ntrial, round(0.5*ntrial),replace = False))
            test_setID = np.sort(np.random.choice(np.setdiff1d(np.arange(ntrial), train_setID, assume_unique=True), (ntrial-len(train_setID)),replace = False))
            
            toDecode_X1 = Y_columnsLabels.index(toDecode_labels1)
            toDecode_X2 = Y_columnsLabels.index(toDecode_labels2)
            
            ### labels: ['locKey','locs','type','loc1','loc2','locX']
            train_label1 = Y[train_setID,toDecode_X1].astype('int') #.astype('str') # locKey
            train_label2 = Y[train_setID,toDecode_X2].astype('int') #.astype('str') # locKey
            
            test_label1 = Y[test_setID,toDecode_X1].astype('int') #.astype('str') # locKey
            test_label2 = Y[test_setID,toDecode_X2].astype('int') #.astype('str') # locKey
            
            if shuff_excludeInv:
                toDecode_labels1_inv = f_stats.inversed_label(Y_columnsLabels, toDecode_labels1, tt)
                toDecode_X1_inv = Y_columnsLabels.index(toDecode_labels1_inv)
                train_label1_inv = Y[train_setID,toDecode_X1_inv]
                test_label1_inv = Y[test_setID,toDecode_X1_inv]
                
                
                toDecode_labels2_inv = f_stats.inversed_label(Y_columnsLabels, toDecode_labels2, tt)
                toDecode_X2_inv = Y_columnsLabels.index(toDecode_labels2_inv)
                train_label2_inv = Y[train_setID,toDecode_X2_inv]
                test_label2_inv = Y[test_setID,toDecode_X2_inv]
            
            
            # cross temp decoding
            
            for t in range(len(tbins)):
                for t_ in range(len(tbins)):
                    clf1, clf2 = LinearDiscriminantAnalysis(), LinearDiscriminantAnalysis()
                    # fit the training data and their corresponding labels
                    clf1.fit(projs_All_CT[train_setID,:,t], train_label1)
                    clf2.fit(projs_All_CT[train_setID,:,t], train_label2)

                    info1_3d = clf1.score(projs_All_CT[test_setID,:,t_], test_label1)
                    info2_3d = clf2.score(projs_All_CT[test_setID,:,t_], test_label2)
                
                    decode_proj1T_3dX[n,0,t,t_] = info1_3d #.mean(axis=-1).mean(axis=-1)
                    decode_proj2T_3dX[n,0,t,t_] = info2_3d #.mean(axis=-1).mean(axis=-1)

                    if t==t_:                            
                        decode_proj1T_3dW[n,0,t] = info1_3d #.mean(axis=-1).mean(axis=-1)
                        decode_proj2T_3dW[n,0,t] = info2_3d #.mean(axis=-1).mean(axis=-1)
                    
                    #shuff to create null distribution by testing the decoders predict random labels
                    for npm in range(nPerms):
                        if shuff_excludeInv:
                            # except for the inverse ones
                            train_label1_shuff, test_label1_shuff = np.full_like(train_label1_inv,9, dtype=int), np.full_like(test_label1_inv,9, dtype=int)
                            train_label2_shuff, test_label2_shuff = np.full_like(train_label2_inv,9, dtype=int), np.full_like(test_label2_inv,9, dtype=int)
                            
                            #for ni1, i1 in enumerate(train_label1_inv.astype(int)):
                            #    train_label1_shuff[ni1] = np.random.choice(np.array(locs)[np.array(locs)!=i1]).astype(int)

                            for nj1, j1 in enumerate(test_label1_inv.astype(int)):
                                test_label1_shuff[nj1] = np.random.choice(np.array(locs)[np.array(locs)!=j1]).astype(int)

                            #for ni2, i2 in enumerate(train_label2_inv.astype(int)):
                            #    train_label2_shuff[ni2] = np.random.choice(np.array(locs)[np.array(locs)!=i2]).astype(int)
                                    
                            for nj2, j2 in enumerate(test_label2_inv.astype(int)):
                                test_label2_shuff[nj2] = np.random.choice(np.array(locs)[np.array(locs)!=j2]).astype(int)

                        else:
                            # fully random
                            #train_label1_shuff = np.random.permutation(train_label1) 
                            test_label1_shuff = np.random.permutation(test_label1)
                            #train_label2_shuff = np.random.permutation(train_label2)
                            test_label2_shuff = np.random.permutation(test_label2)
                        
                        info1_3d_shuff = clf1.score(projs_All_CT[test_setID,:,t_], test_label1_shuff)
                        info2_3d_shuff = clf2.score(projs_All_CT[test_setID,:,t_], test_label2_shuff)
                        
                        decode_proj1T_3d_shuffX[n,npm,t,t_] = info1_3d_shuff #.mean(axis=-1).mean(axis=-1)
                        decode_proj2T_3d_shuffX[n,npm,t,t_] = info2_3d_shuff #.mean(axis=-1).mean(axis=-1)
                        
                        if t==t_:
                            decode_proj1T_3d_shuffW[n,npm,t] = info1_3d_shuff #.mean(axis=-1).mean(axis=-1)
                            decode_proj2T_3d_shuffW[n,npm,t] = info2_3d_shuff #.mean(axis=-1).mean(axis=-1)

        decode_proj1_3dX[region][tt] = decode_proj1T_3dX
        decode_proj2_3dX[region][tt] = decode_proj2T_3dX
        decode_proj1_shuff_all_3dX[region][tt] = decode_proj1T_3d_shuffX
        decode_proj2_shuff_all_3dX[region][tt] = decode_proj2T_3d_shuffX

        decode_proj1_3dW[region][tt] = decode_proj1T_3dW
        decode_proj2_3dW[region][tt] = decode_proj2T_3dW
        decode_proj1_shuff_all_3dW[region][tt] = decode_proj1T_3d_shuffW
        decode_proj2_shuff_all_3dW[region][tt] = decode_proj2T_3d_shuffW
# In[] within time decodability plane projection by lda
#
nPerms = 20
infoMethod = 'lda' #  'omega2' #
bins = 50
dt = dt
tbins = np.arange(tslice[0], tslice[1], bins)

Y_columnsLabels = ['locKey','locs','type','loc1','loc2','locX']
toDecode_labels1 = 'loc1'
toDecode_labels2 = 'loc2'
shuff_excludeInv = True


decode_proj1_3d, decode_proj2_3d = {},{}
decode_proj1_shuff_all_3d, decode_proj2_shuff_all_3d = {},{}

for region in ('dlpfc','fef'):
    
    print(f'Region={region}')
    
    decode_proj1_3d[region], decode_proj2_3d[region] = {}, {}
    decode_proj1_shuff_all_3d[region], decode_proj2_shuff_all_3d[region] = {},{}
    
    

    # estimate decodability by ttype
    for tt in ttypes:
        
        print(f'TType={tt}')
        
        decode_proj1T_3d = np.zeros((nIters, nBoots, len(tbins),3)) # pca1st 3d coordinates
        decode_proj2T_3d = np.zeros((nIters, nBoots, len(tbins),3))
        
        
        # shuff
        decode_proj1T_3d_shuff = np.zeros((nIters, nBoots*nPerms, len(tbins),3))
        decode_proj2T_3d_shuff = np.zeros((nIters, nBoots*nPerms, len(tbins),3))
        
        for n in range(nIters):
            
            if n%20==0:
                print(f'n={n}')
            #for nbt in range(nBoots):
                
            # trial info
            trialInfo_CT = trialInfos_C[region][n][0]
            trialInfo_CT_tt = trialInfo_CT[trialInfo_CT.type == tt]#.reset_index(drop = True)
            idx_tt = trialInfo_CT_tt.index.tolist()#.trial_index.values
            
            vecs_CT = vecs_C[region][n][0] # choice plane vecs
            projs_CT = projs_C[region][n][0] #
            
            

            vec_normal_CT = np.cross(vecs_CT[0],vecs_CT[1]) # plane normal vec
            center_CT = projs_CT.mean(0) # plane center
            
            #data_3pc_CT = data_3pc_C[region][n][nbt][idx_tt,:,:] # 3pc states from tt trials
            
            # smooth to 50ms bins
            #ntrialT, ncellT, ntimeT = data_3pc_CT.shape
            #data_3pc_CT_smooth = np.mean(data_3pc_CT.reshape(ntrialT, ncellT, int(ntimeT/(bins/dt)),int(bins/dt)),axis = 3)
            
            data_3pc_CT_smooth = data_3pc_C[region][n][0][idx_tt,:,:] # 3pc states from tt trials
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
            

            for t in range(len(tbins)):
                info1_3d, _ = f_subspace.plane_decodability(vecs_CT, projs_CT, projs_All_CT[:,:,t], trialInfo_CT_tt.reset_index(drop = True), 'loc1', method = infoMethod)
                info2_3d, _ = f_subspace.plane_decodability(vecs_CT, projs_CT, projs_All_CT[:,:,t], trialInfo_CT_tt.reset_index(drop = True), 'loc2', method = infoMethod)
                
                decode_proj1T_3d[n,0,t,:] = info1_3d #.mean(axis=-1).mean(axis=-1)
                decode_proj2T_3d[n,0,t,:] = info2_3d #.mean(axis=-1).mean(axis=-1)
            
            #shuff
            for nbt in range(nBoots*nPerms):
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
                    trialInfo_CT_tt_shuff = trialInfo_CT_tt.sample(frac=1)
                
                for t in range(len(tbins)):
                    
                    info1_3d_shuff, _ = f_subspace.plane_decodability(vecs_CT, projs_CT, projs_All_CT[:,:,t], trialInfo_CT_tt_shuff.reset_index(drop = True), 'loc1', method = infoMethod)
                    info2_3d_shuff, _ = f_subspace.plane_decodability(vecs_CT, projs_CT, projs_All_CT[:,:,t], trialInfo_CT_tt_shuff.reset_index(drop = True), 'loc2', method = infoMethod)
                    
                    decode_proj1T_3d_shuff[n,nbt,t,:] = info1_3d_shuff #.mean(axis=-1).mean(axis=-1)
                    decode_proj2T_3d_shuff[n,nbt,t,:] = info2_3d_shuff #.mean(axis=-1).mean(axis=-1)
                    
                    
        decode_proj1_3d[region][tt] = decode_proj1T_3d.mean(axis=1)
        decode_proj2_3d[region][tt] = decode_proj2T_3d.mean(axis=1) 
        
        
        #for n in range(nIters):
        decode_proj1_shuff_all_3d[region][tt] = np.concatenate(decode_proj1T_3d_shuff, axis=0)
        decode_proj2_shuff_all_3d[region][tt] = np.concatenate(decode_proj2T_3d_shuff, axis=0)
# In[] i1-i2 code transferability plane projection by lda

infoMethod = 'lda' #  'omega2' #
pdummy = False
nPerms = 20

decode_proj12_3d, decode_proj21_3d = {},{}

decode_proj12_shuff_all_3d, decode_proj21_shuff_all_3d = {},{}

for region in ('dlpfc','fef'):
    
    decode_proj12_3d[region], decode_proj21_3d[region] = {}, {}
    
    decode_proj12_shuff_all_3d[region], decode_proj21_shuff_all_3d[region] = {},{}
    
    

    # estimate decodability by ttype
    for tt in ttypes:
        
        decode_proj12T_3d = np.zeros((nIters, nBoots, len(tbins),len(tbins))) # pca1st 3d coordinates
        decode_proj21T_3d = np.zeros((nIters, nBoots, len(tbins),len(tbins)))
        
        
        # shuff
        decode_proj12T_3d_shuff = np.ones((nIters, nBoots*nPerms, len(tbins),len(tbins)))
        decode_proj21T_3d_shuff = np.ones((nIters, nBoots*nPerms, len(tbins),len(tbins)))
        
        for n in range(nIters):
            #for nbt in range(nBoots):
            t_IterOn = time.time()
            if n%5==0:
                print(f'nIter = {n}')
                
            # trial info
            trialInfo_CT = trialInfos_C[region][n][0]
            trialInfo_CT_tt = trialInfo_CT[trialInfo_CT.type == tt]#.reset_index(drop = True)
            idx_tt = trialInfo_CT_tt.index.tolist()#.trial_index.values
            
            vecs_CT = vecs_C[region][n][0] # choice plane vecs
            projs_CT = projs_C[region][n][0] #
            
            

            vec_normal_CT = np.cross(vecs_CT[0],vecs_CT[1]) # plane normal vec
            center_CT = projs_CT.mean(0) # plane center
            
            #data_3pc_CT = data_3pc_C[region][n][nbt][idx_tt,:,:] # 3pc states from tt trials
            
            # smooth to 50ms bins
            #ntrialT, ncellT, ntimeT = data_3pc_CT.shape
            #data_3pc_CT_smooth = np.mean(data_3pc_CT.reshape(ntrialT, ncellT, int(ntimeT/(bins/dt)),int(bins/dt)),axis = 3)
            
            data_3pc_CT_smooth = data_3pc_C[region][n][0][idx_tt,:,:] # 3pc states from tt trials
            projs_All_CT = np.array([[f_subspace.proj_on_plane(p[:,t], vec_normal_CT, center_CT) for t in range(len(tbins))] for p in data_3pc_CT_smooth]) # projections on the plane
            projs_All_CT = np.swapaxes(projs_All_CT,1,2)
            
            loc1_labels, loc2_labels = trialInfo_CT_tt.loc1.values, trialInfo_CT_tt.loc2.values
            
            for t in range(len(tbins)):
                for t_ in range(len(tbins)):
                    
                    info12_3d = f_decoding.LDAPerformance(projs_All_CT[:,:,t], projs_All_CT[:,:,t_], loc1_labels, loc2_labels) # train to decode item1, test to decode item2
                    info21_3d = f_decoding.LDAPerformance(projs_All_CT[:,:,t], projs_All_CT[:,:,t_], loc2_labels, loc1_labels) # train to decode item2, test to decode item1
                    
                    decode_proj12T_3d[n,0,t,t_] = info12_3d #.mean(axis=-1).mean(axis=-1)
                    decode_proj21T_3d[n,0,t,t_] = info21_3d #.mean(axis=-1).mean(axis=-1)
            
            
            
            #shuff
            if pdummy == False:
                for nbt in range(nBoots*nPerms):
                    
                    if nbt%10==0:
                        print(f'nBoot = {nbt}')
                        
                    trialInfo_CT_shuff = trialInfos_C_shuff[region][n][nbt]
                    trialInfo_CT_tt_shuff = trialInfo_CT_shuff[trialInfo_CT_shuff.type == tt]#.reset_index(drop = True)
                    idx_tt_shuff = trialInfo_CT_tt_shuff.index.tolist()#.trial_index.values
                    
                    vecs_CT_shuff = vecs_C_shuff[region][n][nbt] # choice plane vecs
                    projs_CT_shuff = projs_C_shuff[region][n][nbt] #
                    
                    vec_normal_CT_shuff = np.cross(vecs_CT_shuff[0],vecs_CT_shuff[1]) # plane normal vec
                    center_CT_shuff = projs_CT_shuff.mean(0) # plane center 
                    
                    #data_3pc_CT_shuff = data_3pc_C_shuff[region][n][nbt][idx_tt_shuff,:,:] # 3pc states from tt trials
                    
                    # smooth to 50ms bins
                    #ntrialT, ncellT, ntimeT = data_3pc_CT_shuff.shape
                    #data_3pc_CT_smooth_shuff = np.mean(data_3pc_CT_shuff.reshape(ntrialT, ncellT, int(ntimeT/(bins/dt)),int(bins/dt)),axis = 3)
                    
                    data_3pc_CT_smooth_shuff = data_3pc_C_shuff[region][n][nbt][idx_tt_shuff,:,:] 
                    
                    projs_All_CT_shuff = np.array([[f_subspace.proj_on_plane(p[:,t], vec_normal_CT_shuff, center_CT_shuff) for t in range(len(tbins))] for p in data_3pc_CT_smooth]) # projections on the plane
                    projs_All_CT_shuff = np.swapaxes(projs_All_CT_shuff,1,2)
                    
                    loc1_labels_shuff, loc2_labels_shuff = trialInfo_CT_tt_shuff.loc1.values, trialInfo_CT_tt_shuff.loc2.values
                    
                    for t in range(len(tbins)):
                        for t_ in range(len(tbins)):    
                            
                            info12_3d_shuff = f_decoding.LDAPerformance(projs_All_CT[:,:,t], projs_All_CT[:,:,t_], loc1_labels_shuff, loc2_labels_shuff) # train to decode item1, test to decode item2
                            info21_3d_shuff = f_decoding.LDAPerformance(projs_All_CT[:,:,t], projs_All_CT[:,:,t_], loc2_labels_shuff, loc1_labels_shuff) # train to decode item2, test to decode item1
                            
                            decode_proj12T_3d_shuff[n,nbt,t,t_] = info12_3d_shuff #.mean(axis=-1).mean(axis=-1)
                            decode_proj21T_3d_shuff[n,nbt,t,t_] = info21_3d_shuff #.mean(axis=-1).mean(axis=-1)
            
            print(f'tIter = {(time.time() - t_IterOn):.1f}s')
        
        decode_proj12_3d[region][tt] = decode_proj12T_3d.mean(axis=1)
        decode_proj21_3d[region][tt] = decode_proj21T_3d.mean(axis=1)
        
        decode_proj12_shuff_all_3d[region][tt] = np.concatenate(decode_proj12T_3d_shuff, axis=0)
        decode_proj21_shuff_all_3d[region][tt] = np.concatenate(decode_proj21T_3d_shuff, axis=0)
        
    
    
    
    # plotting
    vmax = 0.6 if region == 'dlpfc' else 0.8
    
    #plt.figure(figsize=(12, 8), dpi=100)
    plt.figure(figsize=(20, 18), dpi=100)    
    
    for tt in ttypes:
        
        condT = 'Retarget' if tt == 1 else 'Distraction'
        
        pPerms_decode12_3d = np.ones((len(tbins),len(tbins)))
        pPerms_decode21_3d = np.ones((len(tbins),len(tbins)))
        
        for t in range(len(tbins)):
            for t_ in range(len(tbins)):
                pPerms_decode12_3d[t,t_] = f_stats.permutation_p(decode_proj12_3d[region][tt].mean(axis=0)[t,t_], decode_proj12_shuff_all_3d[region][tt][:,t,t_], tail='greater')
                pPerms_decode21_3d[t,t_] = f_stats.permutation_p(decode_proj21_3d[region][tt].mean(axis=0)[t,t_], decode_proj21_shuff_all_3d[region][tt][:,t,t_], tail='greater')
                
        # train 1 test 2
        plt.subplot(2,2,(tt-1)*2+1)
        
        ax = plt.gca()
        im = ax.imshow(decode_proj12_3d[region][tt].mean(axis=0), cmap='magma', aspect='auto', vmin = 0.1, vmax = vmax) #
        
        smooth_scale = 10
        z = ndimage.zoom(pPerms_decode12_3d, smooth_scale)
        ax.contour(np.linspace(0, len(tbins), len(tbins) * smooth_scale),
                 np.linspace(0, len(tbins), len(tbins) * smooth_scale),
                  z, levels=([0.05]), colors='white', alpha = 1)
        
        # event lines
        for i in [0, 300, 1300, 1600, 2600]:
            ax.plot(np.arange(0,len(tbins),1), np.full_like(np.arange(0,len(tbins),1),list(tbins).index(i)), 'w-.', linewidth=4)
            ax.plot(np.full_like(np.arange(0,len(tbins),1),list(tbins).index(i)), np.arange(0,len(tbins),1), 'w-.', linewidth=4)
        
        ax.set_xlim(0,len(tbins)-1)
        ax.set_ylim(len(tbins)-1,0)
        ax.set_xticks([list(tbins).index(i) for i in [0, 300, 1300, 1600, 2600]])
        ax.set_xticklabels([0, 300, 1300, 1600, 2600], rotation=0, fontsize = 15)
        ax.set_xlabel('Test Timebin (ms)', labelpad = 10, fontsize = 20)
        ax.set_yticks([list(tbins).index(i) for i in [0, 300, 1300, 1600, 2600]])
        ax.set_yticklabels([0, 300, 1300, 1600, 2600], fontsize = 15)
        ax.set_ylabel('Train Timebin (ms)', labelpad = 10, fontsize = 20)
        ax.set_title(f'{condT}, Item1 -> Item2', pad = 10, fontsize = 20)
        ax.invert_yaxis()
        
        cbar = plt.colorbar(im,ax=ax)
        cbar.ax.tick_params(labelsize=15)
        
        
        
        # train 2 test 1
        plt.subplot(2,2,(tt-1)*2+2)
        
        ax = plt.gca()
        im = ax.imshow(decode_proj21_3d[region][tt].mean(axis=0), cmap='magma', aspect='auto', vmin = 0.1, vmax = vmax) #
        
        smooth_scale = 10
        z = ndimage.zoom(pPerms_decode21_3d, smooth_scale)
        ax.contour(np.linspace(0, len(tbins), len(tbins) * smooth_scale),
                 np.linspace(0, len(tbins), len(tbins) * smooth_scale),
                  z, levels=([0.05]), colors='white', alpha = 1)
        
        # event lines
        for i in [0, 300, 1300, 1600, 2600]:
            ax.plot(np.arange(0,len(tbins),1), np.full_like(np.arange(0,len(tbins),1),list(tbins).index(i)), 'w-.', linewidth=4)
            ax.plot(np.full_like(np.arange(0,len(tbins),1),list(tbins).index(i)), np.arange(0,len(tbins),1), 'w-.', linewidth=4)
        
        ax.set_xlim(0,len(tbins)-1)
        ax.set_ylim(len(tbins)-1,0)
        ax.set_xticks([list(tbins).index(i) for i in [0, 300, 1300, 1600, 2600]])
        ax.set_xticklabels([0, 300, 1300, 1600, 2600], rotation=0, fontsize = 15)
        ax.set_xlabel('Test Timebin (ms)', labelpad = 10, fontsize = 20)
        ax.set_yticks([list(tbins).index(i) for i in [0, 300, 1300, 1600, 2600]])
        ax.set_yticklabels([0, 300, 1300, 1600, 2600], fontsize = 15)
        ax.set_ylabel('Train Timebin (ms)', labelpad = 10, fontsize = 20)
        ax.set_title(f'{condT}, Item2 -> Item1', pad = 10, fontsize = 20)
        ax.invert_yaxis()
        
        cbar = plt.colorbar(im,ax=ax)
        cbar.ax.tick_params(labelsize=15)   
        
        
    plt.tight_layout()
    plt.subplots_adjust(top = 0.95)
    plt.suptitle(f'Item Code Transferability, {region.upper()}', fontsize = 25, y=1)
    plt.show()  


# In[] i1-i2 code transferability plane projection by lda, binned

infoMethod = 'lda' #  'omega2' #
pdummy = False
nPerms = 100

decode_proj12_3d, decode_proj21_3d = {},{}

decode_proj12_shuff_all_3d, decode_proj21_shuff_all_3d = {},{}

for region in ('dlpfc','fef'):
    
    decode_proj12_3d[region], decode_proj21_3d[region] = {}, {}
    
    decode_proj12_shuff_all_3d[region], decode_proj21_shuff_all_3d[region] = {},{}
    
    

    # estimate decodability by ttype
    for tt in ttypes:
        
        decode_proj12T_3d = np.zeros((nIters, nBoots, len(checkpoints),len(checkpoints))) # pca1st 3d coordinates
        decode_proj21T_3d = np.zeros((nIters, nBoots, len(checkpoints),len(checkpoints)))
        
        
        # shuff
        decode_proj12T_3d_shuff = np.ones((nIters, nBoots*nPerms, len(checkpoints),len(checkpoints)))
        decode_proj21T_3d_shuff = np.ones((nIters, nBoots*nPerms, len(checkpoints),len(checkpoints)))
        
        for n in range(nIters):
            #for nbt in range(nBoots):
            t_IterOn = time.time()
            if n%5==0:
                print(f'nIter = {n}')
                
            # trial info
            trialInfo_CT = trialInfos_C[region][n][0]
            trialInfo_CT_tt = trialInfo_CT[trialInfo_CT.type == tt]#.reset_index(drop = True)
            idx_tt = trialInfo_CT_tt.index.tolist()#.trial_index.values
            
            vecs_CT = vecs_C[region][n][0] # choice plane vecs
            projs_CT = projs_C[region][n][0] #
            
            

            vec_normal_CT = np.cross(vecs_CT[0],vecs_CT[1]) # plane normal vec
            center_CT = projs_CT.mean(0) # plane center
            
            #data_3pc_CT = data_3pc_C[region][n][nbt][idx_tt,:,:] # 3pc states from tt trials
            
            # smooth to 50ms bins
            #ntrialT, ncellT, ntimeT = data_3pc_CT.shape
            #data_3pc_CT_smooth = np.mean(data_3pc_CT.reshape(ntrialT, ncellT, int(ntimeT/(bins/dt)),int(bins/dt)),axis = 3)
            
            data_3pc_CT_smooth = data_3pc_C[region][n][0][idx_tt,:,:] # 3pc states from tt trials
            
            # binned by windows
            temp = []
            for cp,avgInt in avgInterval.items():
                t1 = tbins.tolist().index(cp-avgInt) if (cp-avgInt)>= tbins.min() else tbins.tolist().index(tbins.min())
                t2 = tbins.tolist().index(cp+avgInt) if (cp+avgInt)<= tbins.max() else tbins.tolist().index(tbins.max())
                temp += [data_3pc_CT_smooth[:,:,t1:t2].mean(2)]
            
            data_3pc_CT_smooth = np.array(temp).swapaxes(0,1).swapaxes(1,2)
            
            projs_All_CT = np.array([[f_subspace.proj_on_plane(p[:,t], vec_normal_CT, center_CT) for t in range(len(checkpoints))] for p in data_3pc_CT_smooth]) # projections on the plane
            projs_All_CT = np.swapaxes(projs_All_CT,1,2)
            
            loc1_labels, loc2_labels = trialInfo_CT_tt.loc1.values, trialInfo_CT_tt.loc2.values
            
            for t in range(len(checkpoints)):
                for t_ in range(len(checkpoints)):
                    
                    info12_3d = f_decoding.LDAPerformance(projs_All_CT[:,:,t], projs_All_CT[:,:,t_], loc1_labels, loc2_labels) # train to decode item1, test to decode item2
                    info21_3d = f_decoding.LDAPerformance(projs_All_CT[:,:,t], projs_All_CT[:,:,t_], loc2_labels, loc1_labels) # train to decode item2, test to decode item1
                    
                    decode_proj12T_3d[n,0,t,t_] = info12_3d #.mean(axis=-1).mean(axis=-1)
                    decode_proj21T_3d[n,0,t,t_] = info21_3d #.mean(axis=-1).mean(axis=-1)
            
            
            
            #shuff
            if pdummy == False:
                for nbt in range(nBoots*nPerms):
                    if n%5==0:
                        if nbt%10==0:
                            print(f'nBoot = {nbt}')
                        
                    trialInfo_CT_shuff = trialInfos_C_shuff[region][n][nbt]
                    trialInfo_CT_tt_shuff = trialInfo_CT_shuff[trialInfo_CT_shuff.type == tt]#.reset_index(drop = True)
                    idx_tt_shuff = trialInfo_CT_tt_shuff.index.tolist()#.trial_index.values
                    
                    vecs_CT_shuff = vecs_C_shuff[region][n][nbt] # choice plane vecs
                    projs_CT_shuff = projs_C_shuff[region][n][nbt] #
                    
                    vec_normal_CT_shuff = np.cross(vecs_CT_shuff[0],vecs_CT_shuff[1]) # plane normal vec
                    center_CT_shuff = projs_CT_shuff.mean(0) # plane center 
                    
                    #data_3pc_CT_shuff = data_3pc_C_shuff[region][n][nbt][idx_tt_shuff,:,:] # 3pc states from tt trials
                    
                    # smooth to 50ms bins
                    #ntrialT, ncellT, ntimeT = data_3pc_CT_shuff.shape
                    #data_3pc_CT_smooth_shuff = np.mean(data_3pc_CT_shuff.reshape(ntrialT, ncellT, int(ntimeT/(bins/dt)),int(bins/dt)),axis = 3)
                    
                    data_3pc_CT_smooth_shuff = data_3pc_C_shuff[region][n][nbt][idx_tt_shuff,:,:] 
                    
                    # binned by windows
                    temp_shuff = []
                    for cp,avgInt in avgInterval.items():
                        t1 = tbins.tolist().index(cp-avgInt) if (cp-avgInt)>= tbins.min() else tbins.tolist().index(tbins.min())
                        t2 = tbins.tolist().index(cp+avgInt) if (cp+avgInt)<= tbins.max() else tbins.tolist().index(tbins.max())
                        temp_shuff += [data_3pc_CT_smooth_shuff[:,:,t1:t2].mean(2)]
                    
                    data_3pc_CT_smooth_shuff = np.array(temp_shuff).swapaxes(0,1).swapaxes(1,2)
                    
                    projs_All_CT_shuff = np.array([[f_subspace.proj_on_plane(p[:,t], vec_normal_CT_shuff, center_CT_shuff) for t in range(len(checkpoints))] for p in data_3pc_CT_smooth]) # projections on the plane
                    projs_All_CT_shuff = np.swapaxes(projs_All_CT_shuff,1,2)
                    
                    loc1_labels_shuff, loc2_labels_shuff = trialInfo_CT_tt_shuff.loc1.values, trialInfo_CT_tt_shuff.loc2.values
                    
                    for t in range(len(checkpoints)):
                        for t_ in range(len(checkpoints)):    
                            
                            info12_3d_shuff = f_decoding.LDAPerformance(projs_All_CT[:,:,t], projs_All_CT[:,:,t_], loc1_labels_shuff, loc2_labels_shuff) # train to decode item1, test to decode item2
                            info21_3d_shuff = f_decoding.LDAPerformance(projs_All_CT[:,:,t], projs_All_CT[:,:,t_], loc2_labels_shuff, loc1_labels_shuff) # train to decode item2, test to decode item1
                            
                            decode_proj12T_3d_shuff[n,nbt,t,t_] = info12_3d_shuff #.mean(axis=-1).mean(axis=-1)
                            decode_proj21T_3d_shuff[n,nbt,t,t_] = info21_3d_shuff #.mean(axis=-1).mean(axis=-1)
            
            if n%5==0:
                print(f'tIter = {(time.time() - t_IterOn):.1f}s')
        
        decode_proj12_3d[region][tt] = decode_proj12T_3d.mean(axis=1)
        decode_proj21_3d[region][tt] = decode_proj21T_3d.mean(axis=1)
        
        decode_proj12_shuff_all_3d[region][tt] = np.concatenate(decode_proj12T_3d_shuff, axis=0)
        decode_proj21_shuff_all_3d[region][tt] = np.concatenate(decode_proj21T_3d_shuff, axis=0)
        
    
    
for region in ('dlpfc','fef'):    
    # plotting
    vmax = 0.6 if region == 'dlpfc' else 0.8
    
    #plt.figure(figsize=(12, 8), dpi=100)
    plt.figure(figsize=(20, 18), dpi=100)    
    
    for tt in ttypes:
        
        condT = 'Retarget' if tt == 1 else 'Distraction'
        
        pPerms_decode12_3d = np.ones((len(checkpoints),len(checkpoints)))
        pPerms_decode21_3d = np.ones((len(checkpoints),len(checkpoints)))
        
        for t in range(len(checkpoints)):
            for t_ in range(len(checkpoints)):
                pPerms_decode12_3d[t,t_] = f_stats.permutation_p(decode_proj12_3d[region][tt].mean(axis=0)[t,t_], decode_proj12_shuff_all_3d[region][tt][:,t,t_], tail='greater')
                pPerms_decode21_3d[t,t_] = f_stats.permutation_p(decode_proj21_3d[region][tt].mean(axis=0)[t,t_], decode_proj21_shuff_all_3d[region][tt][:,t,t_], tail='greater')
                
        # train 1 test 2
        plt.subplot(2,2,(tt-1)*2+1)
        
        ax = plt.gca()
        im = ax.imshow(decode_proj12_3d[region][tt].mean(axis=0), cmap='magma', aspect='auto', vmin = 0.1, vmax = vmax) #
        
        for i in range(len(checkpoints)):
            for j in range(len(checkpoints)):
                if 0.05 < pPerms_decode12_3d[i,j] <= 0.1:
                    text = ax.text(j, i, '+', ha="center", va="center", fontsize=15, color = 'white') #
                elif 0.01 < pPerms_decode12_3d[i,j] <= 0.05:
                    text = ax.text(j, i, '*', ha="center", va="center", fontsize=15, color = 'white') #
                elif pPerms_decode12_3d[i,j] <= 0.01:
                    text = ax.text(j, i, '**', ha="center", va="center", fontsize=15, color = 'white') #
        
        
        
        ax.set_xticks([n for n in range(len(checkpoints))])
        ax.set_xticklabels(checkpointsLabels, rotation=0, fontsize = 15)
        ax.set_xlabel('Test Bins', labelpad = 10, fontsize = 20)
        ax.set_yticks([n for n in range(len(checkpoints))])
        ax.set_yticklabels(checkpointsLabels, fontsize = 15)
        ax.set_ylabel('Train Bins', labelpad = 10, fontsize = 20)
        ax.set_title(f'{condT}, Item1 -> Item2', pad = 10, fontsize = 20)
        #ax.invert_yaxis()
        
        cbar = plt.colorbar(im,ax=ax)
        cbar.ax.tick_params(labelsize=15)
        
        
        
        # train 2 test 1
        plt.subplot(2,2,(tt-1)*2+2)
        
        ax = plt.gca()
        im = ax.imshow(decode_proj21_3d[region][tt].mean(axis=0), cmap='magma', aspect='auto', vmin = 0.1, vmax = vmax) #
        
        for i in range(len(checkpoints)):
            for j in range(len(checkpoints)):
                if 0.05 < pPerms_decode21_3d[i,j] <= 0.1:
                    text = ax.text(j, i, '+', ha="center", va="center", fontsize=15, color = 'white') #
                elif 0.01 < pPerms_decode21_3d[i,j] <= 0.05:
                    text = ax.text(j, i, '*', ha="center", va="center", fontsize=15, color = 'white') #
                elif pPerms_decode21_3d[i,j] <= 0.01:
                    text = ax.text(j, i, '**', ha="center", va="center", fontsize=15, color = 'white') #
        
        
        
        ax.set_xticks([n for n in range(len(checkpoints))])
        ax.set_xticklabels(checkpointsLabels, rotation=0, fontsize = 15)
        ax.set_xlabel('Test Bins', labelpad = 10, fontsize = 20)
        ax.set_yticks([n for n in range(len(checkpoints))])
        ax.set_yticklabels(checkpointsLabels, fontsize = 15)
        ax.set_ylabel('Train Bins', labelpad = 10, fontsize = 20)
        ax.set_title(f'{condT}, Item2 -> Item1', pad = 10, fontsize = 20)
        #ax.invert_yaxis()
        
        cbar = plt.colorbar(im,ax=ax)
        cbar.ax.tick_params(labelsize=15)   
        
        
    plt.tight_layout()
    plt.subplots_adjust(top = 0.95)
    plt.suptitle(f'Item Code Transferability, {region.upper()}', fontsize = 25, y=1)
    plt.show()  





















# %%
# In[] decodability plane projection by lda

infoMethod = 'lda' #  'omega2' #
bins = 50
dt = dt
tbins = np.arange(tslice[0], tslice[1], bins)

decode_proj1_3d, decode_proj2_3d = {},{}

decode_proj1_shuff_all_3d, decode_proj2_shuff_all_3d = {},{}

for region in ('dlpfc','fef'):
    
    decode_proj1_3d[region], decode_proj2_3d[region] = {}, {}
    
    decode_proj1_shuff_all_3d[region], decode_proj2_shuff_all_3d[region] = {},{}
    
    

    # estimate decodability by ttype
    for tt in ttypes:
        
        decode_proj1T_3d = np.zeros((nIters, nBoots, len(tbins),3)) # pca1st 3d coordinates
        decode_proj2T_3d = np.zeros((nIters, nBoots, len(tbins),3))
        
        
        # shuff
        decode_proj1T_3d_shuff = np.zeros((nIters, nBoots*nPerms, len(tbins),3))
        decode_proj2T_3d_shuff = np.zeros((nIters, nBoots*nPerms, len(tbins),3))
        
        for n in range(nIters):
            #for nbt in range(nBoots):
                
            # trial info
            trialInfo_CT = trialInfos_C[region][n][0]
            trialInfo_CT_tt = trialInfo_CT[trialInfo_CT.type == tt]#.reset_index(drop = True)
            idx_tt = trialInfo_CT_tt.index.tolist()#.trial_index.values
            
            vecs_CT = vecs_C[region][n][0] # choice plane vecs
            projs_CT = projs_C[region][n][0] #
            
            

            vec_normal_CT = np.cross(vecs_CT[0],vecs_CT[1]) # plane normal vec
            center_CT = projs_CT.mean(0) # plane center
            
            #data_3pc_CT = data_3pc_C[region][n][nbt][idx_tt,:,:] # 3pc states from tt trials
            
            # smooth to 50ms bins
            #ntrialT, ncellT, ntimeT = data_3pc_CT.shape
            #data_3pc_CT_smooth = np.mean(data_3pc_CT.reshape(ntrialT, ncellT, int(ntimeT/(bins/dt)),int(bins/dt)),axis = 3)
            
            data_3pc_CT_smooth = data_3pc_C[region][n][0][idx_tt,:,:] # 3pc states from tt trials
            projs_All_CT = np.array([[f_subspace.proj_on_plane(p[:,t], vec_normal_CT, center_CT) for t in range(len(tbins))] for p in data_3pc_CT_smooth]) # projections on the plane
            projs_All_CT = np.swapaxes(projs_All_CT,1,2)
            
            for t in range(len(tbins)):
                info1_3d, _ = f_subspace.plane_decodability(vecs_CT, projs_CT, projs_All_CT[:,:,t], trialInfo_CT_tt.reset_index(drop = True), 'loc1', method = infoMethod)
                info2_3d, _ = f_subspace.plane_decodability(vecs_CT, projs_CT, projs_All_CT[:,:,t], trialInfo_CT_tt.reset_index(drop = True), 'loc2', method = infoMethod)
                
                decode_proj1T_3d[n,0,t,:] = info1_3d #.mean(axis=-1).mean(axis=-1)
                decode_proj2T_3d[n,0,t,:] = info2_3d #.mean(axis=-1).mean(axis=-1)
            
            #shuff
            for nbt in range(nBoots*nPerms):
                
                trialInfo_CT_shuff = trialInfos_C_shuff[region][n][nbt]
                trialInfo_CT_tt_shuff = trialInfo_CT_shuff[trialInfo_CT_shuff.type == tt]#.reset_index(drop = True)
                idx_tt_shuff = trialInfo_CT_tt_shuff.index.tolist()#.trial_index.values
                
                vecs_CT_shuff = vecs_C_shuff[region][n][nbt] # choice plane vecs
                projs_CT_shuff = projs_C_shuff[region][n][nbt] #
                
                vec_normal_CT_shuff = np.cross(vecs_CT_shuff[0],vecs_CT_shuff[1]) # plane normal vec
                center_CT_shuff = projs_CT_shuff.mean(0) # plane center 
                
                #data_3pc_CT_shuff = data_3pc_C_shuff[region][n][nbt][idx_tt_shuff,:,:] # 3pc states from tt trials
                
                # smooth to 50ms bins
                #ntrialT, ncellT, ntimeT = data_3pc_CT_shuff.shape
                #data_3pc_CT_smooth_shuff = np.mean(data_3pc_CT_shuff.reshape(ntrialT, ncellT, int(ntimeT/(bins/dt)),int(bins/dt)),axis = 3)
                
                data_3pc_CT_smooth_shuff = data_3pc_C_shuff[region][n][nbt][idx_tt_shuff,:,:] 
                
                projs_All_CT_shuff = np.array([[f_subspace.proj_on_plane(p[:,t], vec_normal_CT_shuff, center_CT_shuff) for t in range(len(tbins))] for p in data_3pc_CT_smooth]) # projections on the plane
                projs_All_CT_shuff = np.swapaxes(projs_All_CT_shuff,1,2)
                
                for t in range(len(tbins)):
                    
                    info1_3d_shuff, _ = f_subspace.plane_decodability(vecs_CT_shuff, projs_CT_shuff, projs_All_CT_shuff[:,:,t], trialInfo_CT_tt_shuff.reset_index(drop = True), 'loc1', method = infoMethod)
                    info2_3d_shuff, _ = f_subspace.plane_decodability(vecs_CT_shuff, projs_CT_shuff, projs_All_CT_shuff[:,:,t], trialInfo_CT_tt_shuff.reset_index(drop = True), 'loc2', method = infoMethod)
                    
                    decode_proj1T_3d_shuff[n,nbt,t,:] = info1_3d_shuff #.mean(axis=-1).mean(axis=-1)
                    decode_proj2T_3d_shuff[n,nbt,t,:] = info2_3d_shuff #.mean(axis=-1).mean(axis=-1)
                    
                    
        decode_proj1_3d[region][tt] = decode_proj1T_3d.mean(axis=1)
        decode_proj2_3d[region][tt] = decode_proj2T_3d.mean(axis=1) 
        
        
        #for n in range(nIters):
                   
        decode_proj1_shuff_all_3d[region][tt] = np.concatenate(decode_proj1T_3d_shuff, axis=0)
        decode_proj2_shuff_all_3d[region][tt] = np.concatenate(decode_proj2T_3d_shuff, axis=0)
        
        
    infoLabel = 'Accuracy' if infoMethod=='lda' else 'PEV'
    #plt.figure(figsize=(12, 8), dpi=100)
    plt.figure(figsize=(12, 4.75), dpi=100)    
    for tt in ttypes:
        
        condT = 'Retarget' if tt == 1 else 'Distraction'
        h0 = 0 if infoMethod == 'omega2' else 1/len(locs)
        
        pPerms_decode1_3d = np.array([f_stats.permutation_p(decode_proj1_3d[region][tt].mean(axis=-1).mean(axis=0)[t], decode_proj1_shuff_all_3d[region][tt].mean(axis=-1)[:,t], tail='greater') for t in range(len(tbins))])
        pPerms_decode2_3d = np.array([f_stats.permutation_p(decode_proj2_3d[region][tt].mean(axis=-1).mean(axis=0)[t], decode_proj2_shuff_all_3d[region][tt].mean(axis=-1)[:,t], tail='greater') for t in range(len(tbins))])
        
        
        plt.subplot(1,2,tt)
        ax = plt.gca()
        ax.plot(np.arange(0, len(tbins), 1), decode_proj1_3d[region][tt].mean(-1).mean(0), color = 'b', label = 'Item1')
        ax.plot(np.arange(0, len(tbins), 1), decode_proj2_3d[region][tt].mean(-1).mean(0), color = 'm', label = 'Item2')
        ax.fill_between(np.arange(0, len(tbins), 1), (decode_proj1_3d[region][tt].mean(-1).mean(0) - decode_proj1_3d[region][tt].mean(-1).std(0)), (decode_proj1_3d[region][tt].mean(-1).mean(0) + decode_proj1_3d[region][tt].mean(-1).std(0)), color = 'b', alpha = 0.1)
        ax.fill_between(np.arange(0, len(tbins), 1), (decode_proj2_3d[region][tt].mean(-1).mean(0) - decode_proj2_3d[region][tt].mean(-1).std(0)), (decode_proj2_3d[region][tt].mean(-1).mean(0) + decode_proj2_3d[region][tt].mean(-1).std(0)), color = 'm', alpha = 0.1)
        
        # significance line
        segs1 = f_plotting.significance_line_segs(pPerms_decode1_3d,0.05)
        segs2 = f_plotting.significance_line_segs(pPerms_decode2_3d,0.05)
        
        for start1, end1 in segs1:
            ax.plot(np.arange(start1,end1,1), np.full_like(np.arange(start1,end1,1), 1.0, dtype='float'), color='b', linestyle='-', linewidth=2)
            
        for start2, end2 in segs2:
            ax.plot(np.arange(start2,end2,1), np.full_like(np.arange(start2,end2,1), 1.05, dtype='float'), color='m', linestyle='-', linewidth=2)

        
        # event lines
        for i in [0, 300, 1300, 1600, 2600]:
            
            #ax.plot(tbins, np.full_like(tbins,list(tbins).index(i)), 'k-.', linewidth=4, alpha = 0.25)
            ax.plot(np.full_like(tbins,list(tbins).index(i)), tbins, 'k-.', linewidth=2, alpha = 0.25)
        
       #ax.set_title(f'{condT}, 3d', pad = 10)
        ax.set_title(f'{condT}', fontsize = 20, pad = 20)
        ax.set_xticks([list(tbins).index(i) for i in [0, 300, 1300, 1600, 2600]])
        ax.set_xticklabels([0, 300, 1300, 1600, 2600], fontsize = 10)
        ax.set_xlabel('Time', fontsize = 15)
        #ax.set_xlim((list(tbins).index(0),list(tbins).index(2600))) #(0,)
        ax.set_ylim((0,1.1))
        #ax.set_yticklabels(checkpoints, fontsize = 10)
        ax.set_ylabel(f'{infoLabel}', fontsize = 15)
        ax.legend(loc='upper right')
        
        
    plt.tight_layout()
    plt.subplots_adjust(top = 0.8)
    plt.suptitle(f'Readout Subspace Decodability of Items, {region.upper()}', fontsize = 25, y=1)
    plt.show()  
