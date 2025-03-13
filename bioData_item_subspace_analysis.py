# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 20:17:42 2024

@author: aka2333
"""

#%%

from itertools import permutations, product

import time

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import pandas as pd
import f_stats
import f_subspace

#%% initialize parameters

data_path = 'D:/data' 
pseudoPop_path = 'D:/data/pseudoPop' # change to your own save path

tRangeRaw = np.arange(-500,4000,1) # -300 baseline, 0 onset, 300 pre1, 1300 delay1, 1600 pre2, 2600 delay2, response
step = 50
dt = 10 # ms, sampling rate down to 1000/dt Hz

locs = (0,1,2,3,)
locCombs = list(permutations(locs,2))
ttypes = (1,2,)
dropCombs = ()
subConditions = list(product(locCombs, ttypes))

pd.options.mode.chained_assignment = None
epsilon = 1e-7

bins = 50 # dt #
tslice = (-300,2700)
tsliceRange = np.arange(-300,2700,dt)
slice_epochsDic = {'bsl':[-300,0],'s1':[0,300],'d1':[300,1300],'s2':[1300,1600],'d2':[1600,2600],'go':[2600,3000]}


#%% load precomputed readout subspace vectors
vecs_C = np.load(f'{data_path}/' + 'vecs_C_detrended.npy', allow_pickle=True).item() #
projs_C = np.load(f'{data_path}/' + 'projs_C_detrended.npy', allow_pickle=True).item() #
projsAll_C = np.load(f'{data_path}/' + 'projsAll_C_detrended.npy', allow_pickle=True).item() #
trialInfos_C = np.load(f'{data_path}/' + 'trialInfos_C_detrended.npy', allow_pickle=True).item() #
data_3pc_C = np.load(f'{data_path}/' + 'data_3pc_C_detrended.npy', allow_pickle=True).item() #
pca1s_C = np.load(f'{data_path}/' + 'pca1s_C_detrended.npy', allow_pickle=True).item() #

vecs_C_shuff = np.load(f'{data_path}/' + 'vecs_C_shuff_detrended.npy', allow_pickle=True).item() #
projs_C_shuff = np.load(f'{data_path}/' + 'projs_C_shuff_detrended.npy', allow_pickle=True).item() #
projsAll_C_shuff = np.load(f'{data_path}/' + 'projsAll_C_shuff_detrended.npy', allow_pickle=True).item() #
trialInfos_C_shuff = np.load(f'{data_path}/' + 'trialInfos_C_shuff_detrended.npy', allow_pickle=True).item() #
data_3pc_C_shuff = np.load(f'{data_path}/' + 'data_3pc_C_shuff_detrended.npy', allow_pickle=True).item() #
pca1s_C_shuff = np.load(f'{data_path}/' + 'pca1s_C_shuff_detrended.npy', allow_pickle=True).item() #


# In[]

#####################################
######### subspace analysis #########
#####################################


#%% calculate item-specific subspace vectors
nIters = 100
nPerms = 100
nBoots = 10 # for demo purposes, we reduced the number of bootstraps to 10 to save time 
fracBoots = 1.0

tBsl = (-300,0)
idxxBsl = [tsliceRange.tolist().index(tBsl[0]), tsliceRange.tolist().index(tBsl[1])]

checkpoints = [150, 550, 1050, 1450, 1850, 2350, 2800]#
#avgInterval = 50
avgInterval = {150:150, 550:250, 1050:250, 1450:150, 1850:250, 2350:250, 2800:200}
checkpointsLabels = ['S1','ED1','LD2','S2','ED2','LD2', 'Go']
toPlot=False # 
avgMethod='conditional_time' # 'conditional' 'all' 'none'

# initialize dictionaries
vecs = {}
projs = {}
projsAll = {}
trialInfos = {}
pca1s = {}

vecs_shuff = {}
projs_shuff = {}
projsAll_shuff = {}
trialInfos_shuff = {}
pca1s_shuff = {}

for region in ('dlpfc','fef'):
    vecs[region] = {}
    projs[region] = {}
    projsAll[region] = {}
    trialInfos[region] = {}
    pca1s[region] = []
        
    vecs_shuff[region] = {}
    projs_shuff[region] = {}
    projsAll_shuff[region] = {}
    trialInfos_shuff[region] = {}
    pca1s_shuff[region] = []
    
    
    for tt in ttypes:
        trialInfos[region][tt] = []
        trialInfos_shuff[region][tt] = []    
    
    
    for cp in checkpoints:
        vecs[region][cp] = {}
        projs[region][cp] = {}
        projsAll[region][cp] = {}

        vecs_shuff[region][cp] = {}
        projs_shuff[region][cp] = {}
        projsAll_shuff[region][cp] = {}
    
        
        for tt in ttypes:
            vecs[region][cp][tt] = {1:[], 2:[]}
            projs[region][cp][tt] = {1:[], 2:[]}
            projsAll[region][cp][tt] = {1:[], 2:[]}
            
            vecs_shuff[region][cp][tt] = {1:[], 2:[]}
            projs_shuff[region][cp][tt] = {1:[], 2:[]}
            projsAll_shuff[region][cp][tt] = {1:[], 2:[]}
            
    

# calculate item-specific subspace vectors

for n in range(nIters):
    t_IterOn = time.time()
    print(f'Iter = {n}')
    
    pseudo_data = np.load(pseudoPop_path + f'/pseudo_all{n}.npy', allow_pickle=True).item()
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
                    
                    vecs_shuff[region][cp][tt][ll].append([])
                    projs_shuff[region][cp][tt][ll].append([])
                    projsAll_shuff[region][cp][tt][ll].append([])
        
        for nboot in range(nBoots):
            idxT,_ = f_subspace.split_set(np.arange(dataN.shape[0]), frac=fracBoots, ranseed=nboot)
            dataT = dataN[idxT,:,:]
            trialInfoT = trialInfo.loc[idxT,:].reset_index(drop=True)

            pca1_C = pca1s_C[region][n][nboot]
            
            vecs_D, projs_D, projsAll_D, _, trialInfos_D, _, _, evr_1st, pca1 = f_subspace.plane_fitting_analysis(dataT, trialInfoT, pca_tWinX, checkpoints, tsliceRange, avgInterval, locs, ttypes, dropCombs, 
                                                                                                                  toPlot=toPlot, avgMethod = avgMethod, adaptPCA=pca1_C) #, decode_method = decode_method
            
            pca1s[region][n] += [pca1]
            
            for tt in ttypes:
                trialInfos[region][tt][n] += [trialInfos_D[tt]]
                
            
            for cp in checkpoints:
                for tt in ttypes:
                    for ll in (1,2,):
                        vecs[region][cp][tt][ll][n] += [vecs_D[cp][tt][ll]]
                        projs[region][cp][tt][ll][n] += [projs_D[cp][tt][ll]]
                        projsAll[region][cp][tt][ll][n] += [projsAll_D[cp][tt][ll]]                        
            
            print(f'EVRs: {evr_1st.round(5)}')
            
            
            for nperm in range(nPerms):
                
                trialInfoT_shuff = trialInfoT.copy().sample(frac=1).reset_index(drop=True) # just use default method
                
                vecs_D_shuff, projs_D_shuff, projsAll_D_shuff, _, trialInfos_D_shuff, _, _, _, pca1_shuff = f_subspace.plane_fitting_analysis(dataT, trialInfoT_shuff, pca_tWinX, checkpoints, tsliceRange, avgInterval, locs, ttypes, dropCombs, 
                                                                                                                                              toPlot=toPlot, avgMethod = avgMethod, adaptPCA=pca1_C) #, decode_method = decode_method 
                
                pca1s_shuff[region][n] += [pca1_shuff]
                
                for tt in ttypes:
                    trialInfos_shuff[region][tt][n] += [trialInfos_D_shuff[tt]]
                    
                
                for cp in checkpoints:
                    for tt in ttypes:
                        
                        for ll in (1,2,):
                            vecs_shuff[region][cp][tt][ll][n] += [vecs_D_shuff[cp][tt][ll]]
                            projs_shuff[region][cp][tt][ll][n] += [projs_D_shuff[cp][tt][ll]]
                            projsAll_shuff[region][cp][tt][ll][n] += [projsAll_D_shuff[cp][tt][ll]]           
            
            
        print(f'tPerm({nPerms*nBoots}) = {(time.time() - t_IterOn):.4f}s, {region}')
        

#%% save computed vectors
np.save(f'{data_path}/' + 'vecs_detrended.npy', vecs, allow_pickle=True)
np.save(f'{data_path}/' + 'projs_detrended.npy', projs, allow_pickle=True)
np.save(f'{data_path}/' + 'projsAll_detrended.npy', projsAll, allow_pickle=True)
np.save(f'{data_path}/' + 'trialInfos_detrended.npy', trialInfos, allow_pickle=True)
np.save(f'{data_path}/' + 'pca1s_detrended.npy', pca1s, allow_pickle=True)

np.save(f'{data_path}/' + 'vecs_shuff_detrended.npy', vecs_shuff, allow_pickle=True)
np.save(f'{data_path}/' + 'projs_shuff_detrended.npy', projs_shuff, allow_pickle=True)
np.save(f'{data_path}/' + 'projsAll_shuff_detrended.npy', projsAll_shuff, allow_pickle=True)
np.save(f'{data_path}/' + 'trialInfos_shuff_detrended.npy', trialInfos_shuff, allow_pickle=True)
np.save(f'{data_path}/' + 'pca1s_shuff_detrended.npy', pca1s_shuff, allow_pickle=True)


#%% exclude post-gocue window
checkpoints = [150, 550, 1050, 1450, 1850, 2350]#
checkpointsLabels = ['S1','ED1','LD1','S2','ED2','LD2']

#%%                    


#######################
# compare item v item #
#######################


# In[] item v item cosTheta, cosPsi. Compare within type, between time points, between locations
pdummy = True 

cosTheta_11, cosTheta_12, cosTheta_22 = {},{},{}
cosPsi_11, cosPsi_12, cosPsi_22 = {},{},{}
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

        for n in range(nIters):
            for nbt in range(nBoots):
                for nc, cp in enumerate(checkpoints):
                    for nc_, cp_ in enumerate(checkpoints):
                        cT11, _, cP11, _, _ = f_subspace.angle_alignment_coplanar(vecs[region][cp][tt][1][n][nbt], projs[region][cp][tt][1][n][nbt], vecs[region][cp_][tt][1][n][nbt], projs[region][cp_][tt][1][n][nbt])
                        cT22, _, cP22, _, _ = f_subspace.angle_alignment_coplanar(vecs[region][cp][tt][2][n][nbt], projs[region][cp][tt][2][n][nbt], vecs[region][cp_][tt][2][n][nbt], projs[region][cp_][tt][2][n][nbt])
                        cT12, _, cP12, _, _ = f_subspace.angle_alignment_coplanar(vecs[region][cp][tt][1][n][nbt], projs[region][cp][tt][1][n][nbt], vecs[region][cp_][tt][2][n][nbt], projs[region][cp_][tt][2][n][nbt])
                                                
                        cosTheta_11T[n,nbt,nc,nc_], cosTheta_22T[n,nbt,nc,nc_], cosTheta_12T[n,nbt,nc,nc_] = cT11, cT22, cT12# 
                        cosPsi_11T[n,nbt,nc,nc_], cosPsi_22T[n,nbt,nc,nc_], cosPsi_12T[n,nbt,nc,nc_] = cP11, cP22, cP12# 
                        
            for npm in range(nPerms):
                for nc, cp in enumerate(checkpoints):
                    for nc_, cp_ in enumerate(checkpoints):
                        cT11_shuff, _, cP11_shuff, _, _ = f_subspace.angle_alignment_coplanar(vecs_shuff[region][cp][tt][1][n][npm], projs_shuff[region][cp][tt][1][n][npm], vecs_shuff[region][cp_][tt][1][n][npm], projs_shuff[region][cp_][tt][1][n][npm])
                        cT22_shuff, _, cP22_shuff, _, _ = f_subspace.angle_alignment_coplanar(vecs_shuff[region][cp][tt][2][n][npm], projs_shuff[region][cp][tt][2][n][npm], vecs_shuff[region][cp_][tt][2][n][npm], projs_shuff[region][cp_][tt][2][n][npm])
                        cT12_shuff, _, cP12_shuff, _, _ = f_subspace.angle_alignment_coplanar(vecs_shuff[region][cp][tt][1][n][npm], projs_shuff[region][cp][tt][1][n][npm], vecs_shuff[region][cp_][tt][2][n][npm], projs_shuff[region][cp_][tt][2][n][npm])
                        
                        cosTheta_11T_shuff[n,npm,nc,nc_], cosTheta_22T_shuff[n,npm,nc,nc_], cosTheta_12T_shuff[n,npm,nc,nc_] = cT11_shuff, cT22_shuff, cT12_shuff# 
                        cosPsi_11T_shuff[n,npm,nc,nc_], cosPsi_22T_shuff[n,npm,nc,nc_], cosPsi_12T_shuff[n,npm,nc,nc_] = cP11_shuff, cP22_shuff, cP12_shuff# 
                        
        
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
        
#%% save item v item 
np.save(f'{data_path}/' + 'cosTheta_11_data.npy', cosTheta_11, allow_pickle=True)
np.save(f'{data_path}/' + 'cosTheta_12_data.npy', cosTheta_12, allow_pickle=True)
np.save(f'{data_path}/' + 'cosTheta_22_data.npy', cosTheta_22, allow_pickle=True)
np.save(f'{data_path}/' + 'cosPsi_11_data.npy', cosPsi_11, allow_pickle=True)
np.save(f'{data_path}/' + 'cosPsi_12_data.npy', cosPsi_12, allow_pickle=True)
np.save(f'{data_path}/' + 'cosPsi_22_data.npy', cosPsi_22, allow_pickle=True)

np.save(f'{data_path}/' + 'cosTheta_11_shuff_data.npy', cosTheta_11_shuff, allow_pickle=True)
np.save(f'{data_path}/' + 'cosTheta_12_shuff_data.npy', cosTheta_12_shuff, allow_pickle=True)
np.save(f'{data_path}/' + 'cosTheta_22_shuff_data.npy', cosTheta_22_shuff, allow_pickle=True)
np.save(f'{data_path}/' + 'cosPsi_11_shuff_data.npy', cosPsi_11_shuff, allow_pickle=True)
np.save(f'{data_path}/' + 'cosPsi_12_shuff_data.npy', cosPsi_12_shuff, allow_pickle=True)
np.save(f'{data_path}/' + 'cosPsi_22_shuff_data.npy', cosPsi_22_shuff, allow_pickle=True)

#%%                    


#############################
# compare choice/non-choice #
############################# 


#%% compute choice v choice
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
np.save(f'{data_path}/' + 'cosTheta_choice_data.npy', cosTheta_choice, allow_pickle=True)
np.save(f'{data_path}/' + 'cosTheta_nonchoice_data.npy', cosTheta_nonchoice, allow_pickle=True)
np.save(f'{data_path}/' + 'cosPsi_choice_data.npy', cosPsi_choice, allow_pickle=True)
np.save(f'{data_path}/' + 'cosPsi_nonchoice_data.npy', cosPsi_nonchoice, allow_pickle=True)

np.save(f'{data_path}/' + 'cosTheta_choice_shuff_data.npy', cosTheta_choice_shuff, allow_pickle=True)
np.save(f'{data_path}/' + 'cosTheta_nonchoice_shuff_data.npy', cosTheta_nonchoice_shuff, allow_pickle=True)
np.save(f'{data_path}/' + 'cosPsi_choice_shuff_data.npy', cosPsi_choice_shuff, allow_pickle=True)
np.save(f'{data_path}/' + 'cosPsi_nonchoice_shuff_data.npy', cosPsi_nonchoice_shuff, allow_pickle=True)

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
    decode_proj1_shuff_all_3d[region], decode_proj2_shuff_all_3d[region] = {},{}
    
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
                label1 = Y[:,toDecode_X1].astype('int') #
                label2 = Y[:,toDecode_X2].astype('int') #
                
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
                    
                    decode_proj1T_3d[n,npm,nc] = info1_3d 
                    decode_proj2T_3d[n,npm,nc] = info2_3d 
                    
                    # shuff
                    info1_3d_shuff, _ = f_subspace.plane_decodability(vecs1, projs1, projs1_allT_3d, trialInfoT_shuff, 'loc1', method = infoMethod)
                    info2_3d_shuff, _ = f_subspace.plane_decodability(vecs2, projs2, projs2_allT_3d, trialInfoT_shuff, 'loc2', method = infoMethod)
                    
                    decode_proj1T_3d_shuff[n,npm,nc] = info1_3d_shuff 
                    decode_proj2T_3d_shuff[n,npm,nc] = info2_3d_shuff 
                    
                    
        decode_proj1_3d[region][tt] = decode_proj1T_3d
        decode_proj2_3d[region][tt] = decode_proj2T_3d
                    
        decode_proj1_shuff_all_3d[region][tt] = decode_proj1T_3d_shuff
        decode_proj2_shuff_all_3d[region][tt] = decode_proj2T_3d_shuff



#%% save
np.save(f'{data_path}/' + 'performance1_item_data.npy', decode_proj1_3d, allow_pickle=True)
np.save(f'{data_path}/' + 'performance2_item_data.npy', decode_proj2_3d, allow_pickle=True)
np.save(f'{data_path}/' + 'performance1_item_shuff_data.npy', decode_proj1_shuff_all_3d, allow_pickle=True)
np.save(f'{data_path}/' + 'performance2_item_shuff_data.npy', decode_proj2_shuff_all_3d, allow_pickle=True)

#%%


########################
# code transferability #
########################



#%%  item1 v item2 trans

Y_columnsLabels = ['locKey','locs','type','loc1','loc2','locX']
toDecode_labels1 = 'loc1'
toDecode_labels2 = 'loc2'
shuff_excludeInv = False

nPerms = 100
infoMethod = 'lda' #  'omega2' #

performanceX_Trans12, performanceX_Trans21 = {},{}
performanceX_Trans12_shuff, performanceX_Trans21_shuff = {},{}


for region in ('dlpfc','fef'):
    print(f'Region={region}')
    performanceX_Trans12[region], performanceX_Trans21[region] = {}, {}    
    performanceX_Trans12_shuff[region], performanceX_Trans21_shuff[region] = {},{}
    
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
        
#%% save
np.save(f'{data_path}/' + 'performance12X_Trans_data.npy', performanceX_Trans12, allow_pickle=True)
np.save(f'{data_path}/' + 'performance21X_Trans_data.npy', performanceX_Trans21, allow_pickle=True)
np.save(f'{data_path}/' + 'performance12X_Trans_shuff_data.npy', performanceX_Trans12_shuff, allow_pickle=True)
np.save(f'{data_path}/' + 'performance21X_Trans_shuff_data.npy', performanceX_Trans21_shuff, allow_pickle=True)

#%%


#####################################
# code transferability between task #
#####################################



#%%  calculate code transferability between choice-items

Y_columnsLabels = ['locKey','locs','type','loc1','loc2','locX']
toDecode_labels1c = 'loc2'
toDecode_labels2c = 'loc1'
toDecode_labels1nc = 'loc1'
toDecode_labels2nc = 'loc2'
shuff_excludeInv = False

nPerms = 100
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
            label_c1 = Y1[:,toDecode_X1c].astype('int') 
            label_c2 = Y2[:,toDecode_X2c].astype('int') 
            
            label_nc1 = Y1[:,toDecode_X1nc].astype('int') 
            label_nc2 = Y2[:,toDecode_X2nc].astype('int') 
            
            
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
                    
                    performanceX_Trans_rdcT[n,npm,nc,nc_] = info_rdc 
                    performanceX_Trans_drcT[n,npm,nc_,nc] = info_drc 
                    
                    # shuff
                    geom1c_shuff = (vecs1c, projs1c, projs1c_allT_3d, trialInfoT1_shuff, toDecode_labels1c)
                    geom2c_shuff = (vecs2c, projs2c, projs2c_allT_3d, trialInfoT2_shuff, toDecode_labels2c)
                    
                    info_rdc_shuff, _ = f_subspace.plane_decodability_trans(geom1c_shuff, geom2c_shuff)
                    info_drc_shuff, _ = f_subspace.plane_decodability_trans(geom2c_shuff, geom1c_shuff)
                    
                    performanceX_Trans_rdc_shuffT[n,npm,nc,nc_] = info_rdc_shuff 
                    performanceX_Trans_drc_shuffT[n,npm,nc_,nc] = info_drc_shuff 
                    
                    # non choice item
                    vecs1nc, vecs2nc = vecs[region][cp][1][1][n][0], vecs[region][cp_][2][2][n][0]
                    projs1nc, projs2nc = projs[region][cp][1][1][n][0], projs[region][cp_][2][2][n][0]
                    projs1nc_allT_3d, projs2nc_allT_3d = projsAll[region][cp][1][1][n][0], projsAll[region][cp_][2][2][n][0]

                    geom1nc = (vecs1nc, projs1nc, projs1nc_allT_3d, trialInfoT1, toDecode_labels1nc)
                    geom2nc = (vecs2nc, projs2nc, projs2nc_allT_3d, trialInfoT2, toDecode_labels2nc)
                    
                    info_rdnc, _ = f_subspace.plane_decodability_trans(geom1nc, geom2nc)
                    info_drnc, _ = f_subspace.plane_decodability_trans(geom2nc, geom1nc)
                    
                    performanceX_Trans_rdncT[n,npm,nc,nc_] = info_rdnc 
                    performanceX_Trans_drncT[n,npm,nc_,nc] = info_drnc 
                    
                    # shuff
                    geom1nc_shuff = (vecs1nc, projs1nc, projs1nc_allT_3d, trialInfoT1_shuff, toDecode_labels1nc)
                    geom2nc_shuff = (vecs2nc, projs2nc, projs2nc_allT_3d, trialInfoT2_shuff, toDecode_labels2nc)
                    
                    info_rdnc_shuff, _ = f_subspace.plane_decodability_trans(geom1nc_shuff, geom2nc_shuff)
                    info_drnc_shuff, _ = f_subspace.plane_decodability_trans(geom2nc_shuff, geom1nc_shuff)
                    
                    performanceX_Trans_rdnc_shuffT[n,npm,nc,nc_] = info_rdnc_shuff 
                    performanceX_Trans_drnc_shuffT[n,npm,nc_,nc] = info_drnc_shuff 
                
                
    performanceX_Trans_rdc[region] = performanceX_Trans_rdcT
    performanceX_Trans_drc[region] = performanceX_Trans_drcT
    performanceX_Trans_rdnc[region] = performanceX_Trans_rdncT
    performanceX_Trans_drnc[region] = performanceX_Trans_drncT
                
    performanceX_Trans_rdc_shuff[region] = performanceX_Trans_rdc_shuffT
    performanceX_Trans_drc_shuff[region] = performanceX_Trans_drc_shuffT
    performanceX_Trans_rdnc_shuff[region] = performanceX_Trans_rdnc_shuffT
    performanceX_Trans_drnc_shuff[region] = performanceX_Trans_drnc_shuffT
        
#%% save
np.save(f'{data_path}/' + 'performanceX_Trans_rdc_data.npy', performanceX_Trans_rdc, allow_pickle=True)
np.save(f'{data_path}/' + 'performanceX_Trans_drc_data.npy', performanceX_Trans_drc, allow_pickle=True)
np.save(f'{data_path}/' + 'performanceX_Trans_rdnc_data.npy', performanceX_Trans_rdnc, allow_pickle=True)
np.save(f'{data_path}/' + 'performanceX_Trans_drnc_data.npy', performanceX_Trans_drnc, allow_pickle=True)

np.save(f'{data_path}/' + 'performanceX_Trans_rdc_shuff_data.npy', performanceX_Trans_rdc_shuff, allow_pickle=True)
np.save(f'{data_path}/' + 'performanceX_Trans_drc_shuff_data.npy', performanceX_Trans_drc_shuff, allow_pickle=True)
np.save(f'{data_path}/' + 'performanceX_Trans_rdnc_shuff_data.npy', performanceX_Trans_rdnc_shuff, allow_pickle=True)
np.save(f'{data_path}/' + 'performanceX_Trans_drnc_shuff_data.npy', performanceX_Trans_drnc_shuff, allow_pickle=True)


#%%

#####################################
######### parallel baseline #########
#####################################

#%% baseline method: train-test sets

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
avgMethod='conditional_time' # 'conditional' 'all' 'none'


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

# calculate subspace vectors
for n in range(nIters):
    t_IterOn = time.time()
    print(f'Iter = {n}')
    
    pseudo_data = np.load(data_path + f'/pseudo_all{n}.npy', allow_pickle=True).item()
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
            dataN[:,ch,:] = (temp - temp.mean(axis=0))
        
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
            
            pca1_C = pca1s_C[region][n][0]
            
            vecs_D1, projs_D1, projsAll_D1, _, trialInfos_D1, _, _, evr_1st_1, pca1_1 = f_subspace.plane_fitting_analysis(dataT1, trialInfoT1, pca_tWinX, checkpoints, tsliceRange, avgInterval, locs, ttypes, dropCombs, 
                                                                                                                          toPlot=toPlot, avgMethod = avgMethod, adaptPCA=pca1_C) #, decode_method = decode_method
            vecs_D2, projs_D2, projsAll_D2, _, trialInfos_D2, _, _, evr_1st_2, pca1_2 = f_subspace.plane_fitting_analysis(dataT2, trialInfoT2, pca_tWinX, checkpoints, tsliceRange, avgInterval, locs, ttypes, dropCombs,
                                                                                                                          toPlot=toPlot, avgMethod = avgMethod, adaptPCA=pca1_C)

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
        


#%%
np.save(f'{data_path}/' + 'vecs_bsl_train_detrended.npy', vecs_bsl_train, allow_pickle=True)
np.save(f'{data_path}/' + 'projs_bsl_train_detrended.npy', projs_bsl_train, allow_pickle=True)
np.save(f'{data_path}/' + 'projsAll_bsl_train_detrended.npy', projsAll_bsl_train, allow_pickle=True)
np.save(f'{data_path}/' + 'trialInfos_bsl_train_detrended.npy', trialInfos_bsl_train, allow_pickle=True)

np.save(f'{data_path}/' + 'vecs_bsl_test_detrended.npy', vecs_bsl_test, allow_pickle=True)
np.save(f'{data_path}/' + 'projs_bsl_test_detrended.npy', projs_bsl_test, allow_pickle=True)
np.save(f'{data_path}/' + 'projsAll_bsl_test_detrended.npy', projsAll_bsl_test, allow_pickle=True)
np.save(f'{data_path}/' + 'trialInfos_bsl_test_detrended.npy', trialInfos_bsl_test, allow_pickle=True)

#%%                    


#################################
# compare item v item, baseline #
#################################


#%% item v item cosTheta, cosPsi. Compare within type, between time points, between locations
checkpoints = [150, 550, 1050, 1450, 1850, 2350]#
checkpointsLabels = ['S1','ED1','LD1','S2','ED2','LD2']

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
                    
                    cosTheta_11T[n,npm,nc], cosTheta_22T[n,npm,nc], cosTheta_12T[n,npm,nc] = cT11_bsl, cT22_bsl, cT12_bsl#
                    cosPsi_11T[n,npm,nc], cosPsi_22T[n,npm,nc], cosPsi_12T[n,npm,nc] = cP11_bsl, cP22_bsl, cP12_bsl#
                    
        
        cosTheta_11_bsl[region][tt] = cosTheta_11T
        cosTheta_22_bsl[region][tt] = cosTheta_22T
        cosTheta_12_bsl[region][tt] = cosTheta_12T
        
        cosPsi_11_bsl[region][tt] = cosPsi_11T
        cosPsi_22_bsl[region][tt] = cosPsi_22T
        cosPsi_12_bsl[region][tt] = cosPsi_12T
        
#%% save
np.save(f'{data_path}/' + 'cosTheta_11_bsl_data.npy', cosTheta_11_bsl, allow_pickle=True)
np.save(f'{data_path}/' + 'cosTheta_12_bsl_data.npy', cosTheta_12_bsl, allow_pickle=True)
np.save(f'{data_path}/' + 'cosTheta_22_bsl_data.npy', cosTheta_22_bsl, allow_pickle=True)
np.save(f'{data_path}/' + 'cosPsi_11_bsl_data.npy', cosPsi_11_bsl, allow_pickle=True)
np.save(f'{data_path}/' + 'cosPsi_12_bsl_data.npy', cosPsi_12_bsl, allow_pickle=True)
np.save(f'{data_path}/' + 'cosPsi_22_bsl_data.npy', cosPsi_22_bsl, allow_pickle=True)     
#%%

##############################################
# compare readout vs item-specific subspaces #
##############################################

#%% load precomputed readout subspace vectors
vecs_C = np.load(f'{data_path}/' + 'vecs_C_detrended.npy', allow_pickle=True).item() #
projs_C = np.load(f'{data_path}/' + 'projs_C_detrended.npy', allow_pickle=True).item() #
projsAll_C = np.load(f'{data_path}/' + 'projsAll_C_detrended.npy', allow_pickle=True).item() #
trialInfos_C = np.load(f'{data_path}/' + 'trialInfos_C_detrended.npy', allow_pickle=True).item() #
data_3pc_C = np.load(f'{data_path}/' + 'data_3pc_C_detrended.npy', allow_pickle=True).item() #
pca1s_C = np.load(f'{data_path}/' + 'pca1s_C_detrended.npy', allow_pickle=True).item() #

vecs_C_shuff = np.load(f'{data_path}/' + 'vecs_C_shuff_detrended.npy', allow_pickle=True).item() #
projs_C_shuff = np.load(f'{data_path}/' + 'projs_C_shuff_detrended.npy', allow_pickle=True).item() #
projsAll_C_shuff = np.load(f'{data_path}/' + 'projsAll_C_shuff_detrended.npy', allow_pickle=True).item() #
trialInfos_C_shuff = np.load(f'{data_path}/' + 'trialInfos_C_shuff_detrended.npy', allow_pickle=True).item() #
data_3pc_C_shuff = np.load(f'{data_path}/' + 'data_3pc_C_shuff_detrended.npy', allow_pickle=True).item() #
pca1s_C_shuff = np.load(f'{data_path}/' + 'pca1s_C_shuff_detrended.npy', allow_pickle=True).item() #
#%% readout vs. item-specific subspace

# exclude post-gocue window
checkpoints = [150, 550, 1050, 1450, 1850, 2350]#
checkpointsLabels = ['S1','ED1','LD1','S2','ED2','LD2']

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
                    
                    cosTheta_1CT[n,nbt,nc], cosTheta_2CT[n,nbt,nc] = cT1C, cT2C# 
                    cosPsi_1CT[n,nbt,nc], cosPsi_2CT[n,nbt,nc] = cP1C, cP2C#
                    
            
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

#%% save
np.save(f'{data_path}/' + 'cosTheta_1Read_data.npy', cosTheta_1C, allow_pickle=True)
np.save(f'{data_path}/' + 'cosTheta_2Read_data.npy', cosTheta_2C, allow_pickle=True)
np.save(f'{data_path}/' + 'cosPsi_1Read_data.npy', cosPsi_1C, allow_pickle=True)
np.save(f'{data_path}/' + 'cosPsi_2Read_data.npy', cosPsi_2C, allow_pickle=True)
