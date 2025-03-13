#%%
%reload_ext autoreload
%autoreload 2

from itertools import permutations, product
import time
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import pandas as pd

from sklearn.decomposition import PCA

import f_stats
import f_decoding

#%% initialize load and save paths and parameters
data_path = 'D:/data' # change to your own data path
pseudoPop_path = 'D:/data/pseudoPop' # change to your own save path
dt = 10 # sampling rate
monkey_names = 'all' # pooled pseudo population from both monkeys

bins = 50 # smoothed bins
tslice = (-300,2700)
tsliceRange = np.arange(-300,2700,dt)
slice_epochsDic = {'bsl':[-300,0],'s1':[0,300],'d1':[300,1300],'s2':[1300,1600],'d2':[1600,2600],'go':[2600,3000]} # used for pooled pseudo pop or monkey B individually
#slice_epochsDic = {'bsl':[-300,0],'s1':[0,400],'d1':[400,1400],'s2':[1400,1800],'d2':[1800,2800],'go':[2600,3000]} # used for monkey A individually

pd.options.mode.chained_assignment = None
epsilon = 1e-7

# create conditions
locs = (0,1,2,3,)
locCombs = list(permutations(locs,2))
ttypes = (1,2,)
subConditions = list(product(locCombs, ttypes))

#%%

###########################################
######### cross-temporal decoding #########
###########################################

    
#%% initialize variables and decoding params
nIters = 100
nBoots = 100
nPerms = nBoots
tbins = np.arange(tslice[0], tslice[1], bins)

tBsl = (-300,0)
idxxBsl = [tsliceRange.tolist().index(tBsl[0]), tsliceRange.tolist().index(tBsl[1])] #

nPCs_region = {'dlpfc':[0,15], 'fef':[0,15]}

conditions = (('type', 1), ('type', 2))
EVRs = {'dlpfc':[],'fef':[]}

performance1X = {'Retarget':{'dlpfc':[],'fef':[]}, 'Distractor':{'dlpfc':[],'fef':[]}}
performance1X_shuff = {'Retarget':{'dlpfc':[],'fef':[]}, 'Distractor':{'dlpfc':[],'fef':[]}}
performance2X = {'Retarget':{'dlpfc':[],'fef':[]}, 'Distractor':{'dlpfc':[],'fef':[]}}
performance2X_shuff = {'Retarget':{'dlpfc':[],'fef':[]}, 'Distractor':{'dlpfc':[],'fef':[]}}

performance1W = {'Retarget':{'dlpfc':[],'fef':[]}, 'Distractor':{'dlpfc':[],'fef':[]}}
performance1W_shuff = {'Retarget':{'dlpfc':[],'fef':[]}, 'Distractor':{'dlpfc':[],'fef':[]}}
performance2W = {'Retarget':{'dlpfc':[],'fef':[]}, 'Distractor':{'dlpfc':[],'fef':[]}}
performance2W_shuff = {'Retarget':{'dlpfc':[],'fef':[]}, 'Distractor':{'dlpfc':[],'fef':[]}}

toDecode_labels1 = 'loc1'
toDecode_labels2 = 'loc2'
Y_columnsLabels = ['locKey','locs','type','loc1','loc2','locX']
shuff_excludeInv = False 

#%% cross- and within-temporal decoding: permutation with random-trained decoders

for n in range(nIters):
    t_IterOn = time.time()
    print(f'Iter = {n}')
    
    # load pseudo populations
    pseudo_data = np.load(pseudoPop_path + f'/pseudo_{monkey_names}{n}.npy', allow_pickle=True).item()
    pseudo_region = pseudo_data['pseudo_region']
    pseudo_TrialInfo = pseudo_data['pseudo_TrialInfo']
    
    
    # decode for each region
    for region in ('dlpfc','fef'):
        
        nPCs = nPCs_region[region]
        
        pseudo_PopT = pseudo_region[region][pseudo_TrialInfo.trial_index.values,:,:] # shape2 trial * cell * time
        
        # if detrend by subtract avg
        for ch in range(pseudo_PopT.shape[1]):
            temp = pseudo_PopT[:,ch,:]
            pseudo_PopT[:,ch,:] = (temp - temp.mean(axis=0))#/(temp.std(axis = 0) + epsilon) # to avoid divid by 0 issue
            
        X_region = pseudo_PopT
        
        # scaling
        for ch in range(X_region.shape[1]):
            X_region[:,ch,:] = (X_region[:,ch,:] - X_region[:,ch,:].mean()) / X_region[:,ch,:].std() #standard scaler
            
        # PCA on pseudo populations        
        
        # specify applied time window of pca
        pca_tWin = np.hstack((np.arange(300,1300,dt, dtype = int),np.arange(1600,2600,dt, dtype = int))).tolist() # whiskey
        pca_tWinX = [tsliceRange.tolist().index(i) for i in pca_tWin]
        
        # conditional average + avged over time, get mean for each condition, then vstack condition_avgs, shape = chs, n_conditions
        X_regionT = X_region[:,:,pca_tWinX] if pca_tWinX != None else X_region[:,:,:]        
        X_regionT_ = []
        pseudo_TrialInfoT_ = pseudo_TrialInfo.reset_index(drop=True)
        
        for sc in subConditions:
            l1,l2 = sc[0]
            tt = sc[1]
            idxx = pseudo_TrialInfoT_[(pseudo_TrialInfoT_.loc1 == l1)&(pseudo_TrialInfoT_.loc2 == l2)&(pseudo_TrialInfoT_.type == tt)].index.tolist()
            X_regionT_ += [X_regionT[idxx,:,:].mean(axis=0).mean(axis=-1)]
        
        X_regionT = np.vstack(X_regionT_).T
        
        
        # fit & transform PCA to convert to full space (PC1-15)
        pcFrac = 1
        npc = min(int(pcFrac * X_regionT.shape[0]), X_regionT.shape[1])
        pca = PCA(n_components=npc)
        
        pca.fit(X_regionT.T)
        evr = pca.explained_variance_ratio_
        EVRs[region] += [evr]
        print(f'{region}, {evr.round(4)[0:5]}') # show explained variance ratio
        
        X_regionP = np.zeros((X_region.shape[0], npc, X_region.shape[2]))
        for trial in range(X_region.shape[0]):
            X_regionP[trial,:,:] = pca.transform(X_region[trial,:,:].T).T
        
        # decode for each Trial Type based on Full Space data
        for condition in conditions:
            ttypeT = 'Retarget' if condition[-1]==1 else 'Distractor' # Retarget = T/T; Distractor = T/D
            tt = condition[-1]
            
            if bool(condition):
                pseudo_TrialInfoT = pseudo_TrialInfo[pseudo_TrialInfo[condition[0]] == condition[1]]
            else:
                pseudo_TrialInfoT = pseudo_TrialInfo.copy()
            
            idxT = pseudo_TrialInfoT.index
            Y = pseudo_TrialInfoT.loc[:,Y_columnsLabels].values
            ntrial = len(pseudo_TrialInfoT)
            
            
            X_regionPT = X_regionP[idxT,:,:]
            
            ### down sample to 50ms-bins
            ntrialT, ncellT, ntimeT = X_regionPT.shape
            full_setP = np.mean(X_regionPT.reshape(ntrialT, ncellT, int(ntimeT/(bins/dt)),int(bins/dt)),axis = 3)
                        
            toDecode_X1 = Y_columnsLabels.index(toDecode_labels1)
            toDecode_X2 = Y_columnsLabels.index(toDecode_labels2)
            
            full_label1 = Y[:,toDecode_X1].astype('int') # item1 info
            full_label2 = Y[:,toDecode_X2].astype('int') # item2 info
            ### labels: ['locKey','locs','type','loc1','loc2','locX']
            
            
            # create shuffled labels to get null distribution
            if shuff_excludeInv:
                # exclude inverse pairs, non-used
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
            
            
            nPCs = nPCs_region[region]
            
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
                
                train_label1, test_label1 = full_label1[train_setID], full_label1[test_setID] # item1 info
                train_label2, test_label2 = full_label2[train_setID], full_label2[test_setID] # item2 info
                
                train_label1_shuff, test_label1_shuff = full_label1_shuff[train_setID], full_label1_shuff[test_setID]
                train_label2_shuff, test_label2_shuff = full_label2_shuff[train_setID], full_label2_shuff[test_setID]
                
                
                for t in range(len(tbins)):    
                    for t_ in range(len(tbins)):
                        
                        # item1
                        pfmTT1_ = f_decoding.LDAPerformance(train_setP[:,nPCs[0]:nPCs[1],t], test_setP[:,nPCs[0]:nPCs[1],t_],
                                                            train_label1, test_label1)

                        # item2
                        pfmTT2_ = f_decoding.LDAPerformance(train_setP[:,nPCs[0]:nPCs[1],t], test_setP[:,nPCs[0]:nPCs[1],t_],
                                                            train_label2, test_label2)

                        performanceX1[nbt,t,t_] = pfmTT1_
                        performanceX2[nbt,t,t_] = pfmTT2_
                        
                        if t==t_:
                            performanceW1[nbt,t] = pfmTT1_
                            performanceW2[nbt,t] = pfmTT2_
                        
                        # train decoders with shuffled labels to get null distribution
                        pfmTT1_shuff = f_decoding.LDAPerformance(train_setP[:,nPCs[0]:nPCs[1],t], test_setP[:,nPCs[0]:nPCs[1],t_],
                                                                    train_label1_shuff, test_label1_shuff)
                        pfmTT2_shuff = f_decoding.LDAPerformance(train_setP[:,nPCs[0]:nPCs[1],t], test_setP[:,nPCs[0]:nPCs[1],t_],
                                                                    train_label2_shuff, test_label2_shuff)
                        

                        performanceX1_shuff[nbt,t,t_] = pfmTT1_shuff
                        performanceX2_shuff[nbt,t,t_] = pfmTT2_shuff
                        
                        if t==t_:
                            performanceW1_shuff[nbt,t] = pfmTT1_shuff
                            performanceW2_shuff[nbt,t] = pfmTT2_shuff
            
            # store decodability
            performance1X[ttypeT][region] += [np.array(performanceX1)]
            performance2X[ttypeT][region] += [np.array(performanceX2)]
            performance1W[ttypeT][region] += [np.array(performanceW1)]
            performance2W[ttypeT][region] += [np.array(performanceW2)]
            
            performance1X_shuff[ttypeT][region] += [np.array(performanceX1_shuff)]
            performance2X_shuff[ttypeT][region] += [np.array(performanceX2_shuff)]
            performance1W_shuff[ttypeT][region] += [np.array(performanceW1_shuff)]
            performance2W_shuff[ttypeT][region] += [np.array(performanceW2_shuff)]

    print(f'tIter = {(time.time() - t_IterOn):.4f}s')


#%% save full space decodability and EVRs
np.save(f'{data_path}/' + f'performanceX1_full_data.npy', performance1X, allow_pickle=True)
np.save(f'{data_path}/' + f'performanceX2_full_data.npy', performance2X, allow_pickle=True)
np.save(f'{data_path}/' + f'performanceX1_shuff_full_data.npy', performance1X_shuff, allow_pickle=True)
np.save(f'{data_path}/' + f'performanceX2_shuff_full_data.npy', performance2X_shuff, allow_pickle=True)
np.save(f'{data_path}/' + f'EVRs_full_data.npy', EVRs, allow_pickle=True)

