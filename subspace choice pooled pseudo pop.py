# -*- coding: utf-8 -*-
"""
Created on Sat Jun 29 00:23:18 2024

@author: aka2333
"""
#%%
%reload_ext autoreload
%autoreload 2

from itertools import permutations, product
import time
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import pandas as pd

import f_stats
import f_decoding
import f_subspace

#%% initialize parameters
data_path = 'D:/data' 
tRangeRaw = np.arange(-500,4000,1) # -300 baseline, 0 onset, 300 pre1, 1300 delay1, 1600 pre2, 2600 delay2, response

step = 50
dt = 10 # ms, sampling rate down to 1000/dt Hz

locs = (0,1,2,3,)
locCombs = list(permutations(locs,2))
ttypes = (1,2,)
dropCombs = ()
subConditions = list(product(locCombs, ttypes))

pd.options.mode.chained_assignment = None

bins = 50 # dt #
tslice = (-300,3000)
tsliceRange = np.arange(-300,3000,dt)
slice_epochsDic = {'bsl':[-300,0],'s1':[0,300],'d1':[300,1300],'s2':[1300,1600],'d2':[1600,2600],'go':[2600,3000]}

nIters = 100
nPerms = 100
nBoots = 1
fracBoots = 1.0

tBsl = (-300,0)
idxxBsl = [tsliceRange.tolist().index(tBsl[0]), tsliceRange.tolist().index(tBsl[1])]

checkpoints = [150, 550, 1050, 1450, 1850, 2350, 2800]#
avgInterval = {150:150, 550:250, 1050:250, 1450:150, 1850:250, 2350:250, 2800:200}
checkpointsLabels = ['S1','ED1','LD2','S2','ED2','LD2', 'Go']
toPlot=False # 
avgMethod='conditional_time' # 'conditional' 'all' 'none'

choice_tRange = (2100,2600)
toplot_samples = np.arange(0,1,1)

# smooth to 50ms bins
bins = 50
dt = dt
tbins = np.arange(tslice[0], tslice[1], bins)
epsilon = 1e-7
#%%

###############################################
########## readout subspace analysis ##########
###############################################    
    
#%% calculate readout subspace vectors

vecs_C = {}
projs_C = {}
projsAll_C = {}
trialInfos_C = {}
data_3pc_C = {}
pca1s_C = {}

vecs_C_shuff = {}
projs_C_shuff = {}
projsAll_C_shuff = {}
trialInfos_C_shuff = {}
data_3pc_C_shuff = {}
pca1s_C_shuff = {}

for region in ('dlpfc','fef'):
    vecs_C[region] = []
    projs_C[region] = []
    projsAll_C[region] = []
    trialInfos_C[region] = []
    data_3pc_C[region] = []
    pca1s_C[region] = []
    
    vecs_C_shuff[region] = []
    projs_C_shuff[region] = []
    projsAll_C_shuff[region] = []
    trialInfos_C_shuff[region] = []
    data_3pc_C_shuff[region] = []
    pca1s_C_shuff[region] = []

evrs = {'dlpfc':np.zeros((nIters, 3)), 'fef':np.zeros((nIters, 3))}

# estimate choice subspace geoms
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
            dataN[:,ch,:] = (temp - temp.mean(axis=0))#/(temp.std(axis = 0) + epsilon) # to avoid divid by 0 issue
            
        
        for ch in range(dataN.shape[1]):
            dataN[:,ch,:] = (dataN[:,ch,:] - dataN[:,ch,:].mean()) / (dataN[:,ch,:].std()+epsilon) #standard scaler
            
        # append to store each iteration separately
        vecs_C[region].append([])
        projs_C[region].append([])
        projsAll_C[region].append([])
        trialInfos_C[region].append([])
        data_3pc_C[region].append([])
        pca1s_C[region].append([])
        
        vecs_C_shuff[region].append([])
        projs_C_shuff[region].append([])
        projsAll_C_shuff[region].append([])
        trialInfos_C_shuff[region].append([])
        data_3pc_C_shuff[region].append([])
        pca1s_C_shuff[region].append([])
        
        
        
        for nboot in range(nBoots):
            tplt = True if (nboot == 0 and n in toplot_samples) else False
            idxT,_ = f_subspace.split_set(np.arange(dataN.shape[0]), frac=fracBoots, ranseed=nboot)
            dataT = dataN[idxT,:,:]
            trialInfoT = trialInfo.loc[idxT,:].reset_index(drop=True)
            
            vecs_CT, projs_CT, projsAll_CT, _, trialInfo_CT, data_3pc_CT, _, evr_1stT, pca1_CT, _ = f_subspace.planeC_fitting_analysis(dataT, trialInfoT, pca_tWinX, tsliceRange, choice_tRange, locs, ttypes, dropCombs, 
                                                                                                                                              toPlot=tplt, avgMethod = avgMethod, region_label=f'{region.upper()}', 
                                                                                                                                              plot_traj=True, traj_checkpoints=(1300,2600), traj_start=1300, traj_end=2600,
                                                                                                                                              plot3d=False,savefig=False, save_path=data_path, plotlayout = (2,3,0,1),
                                                                                                                                              hideLocs=(0,2,),legend_on=False) #
            
            # smooth to 50ms bins
            ntrialT, ncellT, ntimeT = data_3pc_CT.shape
            data_3pc_CT_smooth = np.mean(data_3pc_CT.reshape(ntrialT, ncellT, int(ntimeT/(bins/dt)),int(bins/dt)),axis = 3)
            
            vecs_C[region][n] += [vecs_CT]
            projs_C[region][n] += [projs_CT]
            projsAll_C[region][n] += [projsAll_CT]
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
                trialInfos_C_shuff[region][n] += [trialInfo_CT_shuff]
                data_3pc_C_shuff[region][n] += [data_3pc_CT_smooth_shuff]
                pca1s_C_shuff[region][n] += [pca1_CT_shuff]
                
                            
            
            
        print(f'tPerm({nPerms*nBoots}) = {(time.time() - t_IterOn):.4f}s, {region}')
            

#%% save 
np.save(f'{data_path}/' + 'vecs_C_detrended.npy', vecs_C, allow_pickle=True)
np.save(f'{data_path}/' + 'projs_C_detrended.npy', projs_C, allow_pickle=True)
np.save(f'{data_path}/' + 'projsAll_C_detrended.npy', projsAll_C, allow_pickle=True)
np.save(f'{data_path}/' + 'trialInfos_C_detrended.npy', trialInfos_C, allow_pickle=True)
np.save(f'{data_path}/' + 'data_3pc_C_detrended.npy', data_3pc_C, allow_pickle=True)
np.save(f'{data_path}/' + 'pca1s_C_detrended.npy', pca1s_C, allow_pickle=True)


np.save(f'{data_path}/' + 'vecs_C_shuff_detrended.npy', vecs_C_shuff, allow_pickle=True)
np.save(f'{data_path}/' + 'projs_C_shuff_detrended.npy', projs_C_shuff, allow_pickle=True)
np.save(f'{data_path}/' + 'projsAll_C_shuff_detrended.npy', projsAll_C_shuff, allow_pickle=True)
np.save(f'{data_path}/' + 'trialInfos_C_shuff_detrended.npy', trialInfos_C_shuff, allow_pickle=True)
np.save(f'{data_path}/' + 'data_3pc_C_shuff_detrended.npy', data_3pc_C_shuff, allow_pickle=True)
np.save(f'{data_path}/' + 'pca1s_C_shuff_detrended.npy', pca1s_C_shuff, allow_pickle=True)


#%%

#################################
# readout subspace decodability #
#################################

#%% cross temp decodability plane projection by lda

# initialization
Y_columnsLabels = ['locKey','locs','type','loc1','loc2','locX']
toDecode_labels1 = 'loc1'
toDecode_labels2 = 'loc2'
shuff_excludeInv = False

nIters = 100
nBoots = 100
nPerms = nBoots#

infoMethod = 'lda' #
bins = 50
dt = dt

tslice = (-300,2700)
tbins = np.arange(tslice[0], tslice[1], bins)

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
            
            data_3pc_CT_smooth = data_3pc_C[region][n][0][idx_tt,:,:] # 3pc states from tt trials
            projs_All_CT = np.array([[f_subspace.proj_on_plane(p[:,t], vec_normal_CT, center_CT) for t in range(len(tbins))] for p in data_3pc_CT_smooth]) # projections on the plane
            projs_All_CT = np.swapaxes(projs_All_CT,1,2)
            
            # labels
            Y = trialInfo_CT_tt.loc[:,Y_columnsLabels].values
            ntrial = len(trialInfo_CT_tt)


            toDecode_X1 = Y_columnsLabels.index(toDecode_labels1)
            toDecode_X2 = Y_columnsLabels.index(toDecode_labels2)
            
            ### labels: ['locKey','locs','type','loc1','loc2','locX']
            full_label1 = Y[:,toDecode_X1].astype('int') 
            full_label2 = Y[:,toDecode_X2].astype('int') 
            
            if shuff_excludeInv:
                toDecode_labels1_inv = f_stats.inversed_label(Y_columnsLabels, toDecode_labels1, tt)
                toDecode_X1_inv = Y_columnsLabels.index(toDecode_labels1_inv)
                full_label1_inv = Y[:,toDecode_X1_inv]
                
                toDecode_labels2_inv = f_stats.inversed_label(Y_columnsLabels, toDecode_labels2, tt)
                toDecode_X2_inv = Y_columnsLabels.index(toDecode_labels2_inv)
                full_label2_inv = Y[:,toDecode_X2_inv]
            
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
                
                train_label1 = full_label1[train_setID] 
                train_label2 = full_label2[train_setID] 
                test_label1 = full_label1[test_setID] 
                test_label2 = full_label2[test_setID] 

                train_label1_shuff = full_label1_shuff[train_setID] 
                train_label2_shuff = full_label2_shuff[train_setID] 
                test_label1_shuff = full_label1_shuff[test_setID] 
                test_label2_shuff = full_label2_shuff[test_setID] 


                # cross temp decoding
                for t in range(len(tbins)):
                    for t_ in range(len(tbins)):
                        
                        info1_3d = f_decoding.LDAPerformance(projs_All_CT[train_setID,:,t],projs_All_CT[test_setID,:,t_], 
                                                             train_label1, test_label1)
                        info2_3d = f_decoding.LDAPerformance(projs_All_CT[train_setID,:,t],projs_All_CT[test_setID,:,t_], 
                                                             train_label2, test_label2)
                    
                        decode_proj1T_3dX[n,nbt,t,t_] = info1_3d 
                        decode_proj2T_3dX[n,nbt,t,t_] = info2_3d 

                        if t==t_:                            
                            decode_proj1T_3dW[n,nbt,t] = info1_3d 
                            decode_proj2T_3dW[n,nbt,t] = info2_3d 
                        
                    
                        # permutation null distribution
                        info1_3d_shuff = f_decoding.LDAPerformance(projs_All_CT[train_setID,:,t],projs_All_CT[test_setID,:,t_], 
                                                             train_label1_shuff, test_label1_shuff)
                        info2_3d_shuff = f_decoding.LDAPerformance(projs_All_CT[train_setID,:,t],projs_All_CT[test_setID,:,t_], 
                                                             train_label2_shuff, test_label2_shuff)
                        
                        decode_proj1T_3d_shuffX[n,nbt,t,t_] = info1_3d_shuff 
                        decode_proj2T_3d_shuffX[n,nbt,t,t_] = info2_3d_shuff 
                        
                        if t==t_:
                            decode_proj1T_3d_shuffW[n,nbt,t] = info1_3d_shuff 
                            decode_proj2T_3d_shuffW[n,nbt,t] = info2_3d_shuff 

                            
        decode_proj1_3dX[region][tt] = decode_proj1T_3dX
        decode_proj2_3dX[region][tt] = decode_proj2T_3dX
        decode_proj1_shuff_all_3dX[region][tt] = decode_proj1T_3d_shuffX
        decode_proj2_shuff_all_3dX[region][tt] = decode_proj2T_3d_shuffX

        decode_proj1_3dW[region][tt] = decode_proj1T_3dW
        decode_proj2_3dW[region][tt] = decode_proj2T_3dW
        decode_proj1_shuff_all_3dW[region][tt] = decode_proj1T_3d_shuffW
        decode_proj2_shuff_all_3dW[region][tt] = decode_proj2T_3d_shuffW
        
#%% save within- and cross- temp readout decodability
np.save(f'{data_path}/' + 'performanceW1_readout_data.npy', decode_proj1_3dW, allow_pickle=True)
np.save(f'{data_path}/' + 'performanceW2_readout_data.npy', decode_proj2_3dW, allow_pickle=True)
np.save(f'{data_path}/' + 'performanceW1_readout_shuff_data.npy', decode_proj1_shuff_all_3dW, allow_pickle=True)
np.save(f'{data_path}/' + 'performanceW2_readout_shuff_data.npy', decode_proj2_shuff_all_3dW, allow_pickle=True)

np.save(f'{data_path}/' + 'performanceX1_readout_data.npy', decode_proj1_3dX, allow_pickle=True)
np.save(f'{data_path}/' + 'performanceX2_readout_data.npy', decode_proj2_3dX, allow_pickle=True)
np.save(f'{data_path}/' + 'performanceX1_readout_shuff_data.npy', decode_proj1_shuff_all_3dX, allow_pickle=True)
np.save(f'{data_path}/' + 'performanceX2_readout_shuff_data.npy', decode_proj2_shuff_all_3dX, allow_pickle=True)


#%%

###########################################
# readout subspace decodability of ttypes #
###########################################

#%% cross temp decodability of trial type plane projection by lda

# initialization
Y_columnsLabels = ['locKey','locs','type','loc1','loc2','locX']
toDecode_labels1 = 'type'
nIters = 100
nBoots = 100
nPerms = nBoots#100

infoMethod = 'lda' #  'omega2' #
bins = 50
dt = dt

tslice = (-300,2700)
tbins = np.arange(tslice[0], tslice[1], bins)

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
        
        print(f'{n}')
        
        # trial info
        trialInfo_CT = trialInfos_C[region][n][0]
        idx_tt = trialInfo_CT.index.tolist()#
        vecs_CT = vecs_C[region][n][0] # choice plane vecs
        projs_CT = projs_C[region][n][0] #
        vec_normal_CT = np.cross(vecs_CT[0],vecs_CT[1]) # plane normal vec
        center_CT = projs_CT.mean(0) # plane center
        
        data_3pc_CT_smooth = data_3pc_C[region][n][0][idx_tt,:,:] # 3pc states from tt trials
        projs_All_CT = np.array([[f_subspace.proj_on_plane(p[:,t], vec_normal_CT, center_CT) for t in range(len(tbins))] for p in data_3pc_CT_smooth]) # projections on the plane
        projs_All_CT = np.swapaxes(projs_All_CT,1,2)
        
        # labels
        Y = trialInfo_CT.loc[:,Y_columnsLabels].values
        ntrial = len(trialInfo_CT)
        toDecode_X1 = Y_columnsLabels.index(toDecode_labels1)
        
        ### labels: ['locKey','locs','type','loc1','loc2','locX']
        full_label1 = Y[:,toDecode_X1].astype('int') #.astype('str') # locKey

        # fully random
        full_label1_shuff = np.random.permutation(full_label1) 
        
        for nbt in range(nBoots):
            ### split into train and test sets
            train_setID = np.sort(np.random.choice(ntrial, round(0.5*ntrial),replace = False))
            test_setID = np.sort(np.random.choice(np.setdiff1d(np.arange(ntrial), train_setID, assume_unique=True), (ntrial-len(train_setID)),replace = False))
            
            train_label1 = full_label1[train_setID] #
            test_label1 = full_label1[test_setID] #
            
            train_label1_shuff = full_label1_shuff[train_setID] #
            test_label1_shuff = full_label1_shuff[test_setID] #
            
            # cross temp decoding
            
            for t in range(len(tbins)):
                for t_ in range(len(tbins)):
                    
                    info1_3d = f_decoding.LDAPerformance(projs_All_CT[train_setID,:,t],projs_All_CT[test_setID,:,t_], 
                                                            train_label1, test_label1)
                    
                    decode_ttXT[n,nbt,t,t_] = info1_3d #
                    
                    if t==t_:                            
                        decode_ttWT[n,nbt,t] = info1_3d #
                        
                
                    # permutation null distribution
                    info1_3d_shuff = f_decoding.LDAPerformance(projs_All_CT[train_setID,:,t],projs_All_CT[test_setID,:,t_], 
                                                            train_label1_shuff, test_label1_shuff)
                    
                    decode_ttX_shuffT[n,nbt,t,t_] = info1_3d_shuff #
                    
                    if t==t_:
                        decode_ttW_shuffT[n,nbt,t] = info1_3d_shuff #
                        
                            
        decode_ttX[region] = decode_ttXT
        decode_ttX_shuff[region] = decode_ttX_shuffT
        decode_ttW[region] = decode_ttWT
        decode_ttW_shuff[region] = decode_ttW_shuffT
        
#%% save
np.save(f'{data_path}/' + 'performance_ttX_readout_data.npy', decode_ttX, allow_pickle=True)
np.save(f'{data_path}/' + 'performance_ttW_readout_data.npy', decode_ttW, allow_pickle=True)
np.save(f'{data_path}/' + 'performance_ttX_readout_shuff_data.npy', decode_ttX_shuff, allow_pickle=True)
np.save(f'{data_path}/' + 'performance_ttW_readout_shuff_data.npy', decode_ttW_shuff, allow_pickle=True)

#%% readout vs. item-specific subspace
# load item-specific plane vectors
vecs = np.load(f'{data_path}/' + 'vecs_detrended.npy', allow_pickle=True).item() #
projs = np.load(f'{data_path}/' + 'projs_detrended.npy', allow_pickle=True).item() #
projsAll = np.load(f'{data_path}/' + 'projsAll_detrended.npy', allow_pickle=True).item() #
trialInfos = np.load(f'{data_path}/' + 'trialInfos_detrended.npy', allow_pickle=True).item() #
pca1s = np.load(f'{data_path}/' + 'pca1s_detrended.npy', allow_pickle=True).item() #

vecs_shuff = np.load(f'{data_path}/' + 'vecs_shuff_detrended.npy', allow_pickle=True).item() #
projs_shuff = np.load(f'{data_path}/' + 'projs_shuff_detrended.npy', allow_pickle=True).item() #
projsAll_shuff = np.load(f'{data_path}/' + 'projsAll_shuff_detrended.npy', allow_pickle=True).item() #
trialInfos_shuff = np.load(f'{data_path}/' + 'trialInfos_shuff_detrended.npy', allow_pickle=True).item() #
pca1s_shuff = np.load(f'{data_path}/' + 'pca1s_shuff_detrended.npy', allow_pickle=True).item() #

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

#%%



#%% retarget vs distraction drift distances, normalization -1 to 1

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


for region in ('dlpfc','fef'):
    
    euDists[region] = {tt:[] for tt in ttypes}
    euDists_shuff[region] = {tt:[] for tt in ttypes}

    # estimate decodability by ttype
    for n in range(nIters):
        
        # trial info
        trialInfo_CT = trialInfos_C[region][n][0]
        trialInfo_CT = trialInfo_CT[(trialInfo_CT.loc1.isin(remainedLocs))&(trialInfo_CT.loc2.isin(remainedLocs))]
            
        idx = trialInfo_CT.index.tolist()
        
        vecs_CT = vecs_C[region][n][0] # choice plane vecs
        projs_CT = projs_C[region][n][0] #
        
        vec_normal_CT = np.cross(vecs_CT[0],vecs_CT[1]) # plane normal vec
        center_CT = projs_CT.mean(0) # plane center
        
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
    
        
        endX_D1s_shuff, endX_D2s_shuff = [tbins.tolist().index(ed1) for ed1 in end_D1s], [tbins.tolist().index(ed2) for ed2 in end_D2s]
        
        
        euDistT = {tt:[] for tt in ttypes}
    
        
        trialInfo_temp = trialInfo_CT.copy().reset_index(drop=True)
        
        for l1 in remainedLocs:
            idxT1 = trialInfo_temp[(trialInfo_temp.loc1==l1)].index 
            centroidD1 = projs_All_CT_2d[idxT1][:,:,endX_D1s].mean(2).mean(0) # type-general centroid

            for tt in ttypes:
                for l2 in remainedLocs:
                    if l1!=l2:
                        idxT2 = trialInfo_temp[(trialInfo_temp.loc1==l1)&(trialInfo_temp.loc2==l2)&(trialInfo_temp.type==tt)].index
                        centroidD2 = projs_All_CT_2d[idxT2][:,:,endX_D2s].mean(2).mean(0)
                        
                        euDistT[tt] += [np.sqrt(np.sum((centroidD1 - centroidD2)**2))]
        
        for tt in ttypes:
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
            
    
#%%
np.save(f'{data_path}/euDists_monkeys_centroids2_normalized_hide02.npy', euDists, allow_pickle=True)
np.save(f'{data_path}/euDists_shuff_monkeys_centroids2_normalized_hide02.npy', euDists_shuff, allow_pickle=True)
