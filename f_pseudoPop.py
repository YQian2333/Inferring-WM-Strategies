# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 15:13:03 2024

@author: aka2333
"""
#%%
import time

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import pandas as pd
from functions import Get_trial, Get_spikes, spike_selection, spike2freq, bslFreqV, tCutArray

# In[]
def pseudo_Session(locCombs, ttypes, samplePerCon = 6, sampleRounds = 1, subConditions = None):
    pseudo_Session = []
    idx = 0
    for r in range(sampleRounds):
        
        if subConditions == None:
            for comb in locCombs:
                loc1,loc2 = comb[0], comb[1]
                locs = str(loc1)+'_'+str(loc2)
                
                for tt in ttypes:
                    locKey = loc2 if tt==1 else loc1
                    locX = loc1 if tt==1 else loc2
                    
                    for s in range(samplePerCon):            
                        pseudo_Session += [[idx, loc1, loc2, locs, locKey, locX, tt]]
                    
                        idx += 1
        else:
            for comb in subConditions:
                loc1,loc2, tt = comb[0][0], comb[0][1], comb[1]
                
                locs = str(loc1)+'_'+str(loc2)
                
                locKey = loc2 if tt==1 else loc1
                locX = loc1 if tt==1 else loc2
                    
                for s in range(samplePerCon):            
                    pseudo_Session += [[idx, loc1, loc2, locs, locKey, locX, tt]]
                
                    idx += 1
                    
    pseudo_Session = pd.DataFrame(pseudo_Session, columns = ['trial_index','loc1','loc2','locs','locKey','locX','type'])
    
    return pseudo_Session
# In[] pseudo population by region, trial_matched_within_session
def pseudo_PopByRegion(session_paths, cellsToUse, dlpfcArrays, fefArrays, tRange, tslice, tBsl, locCombs, ttypes, samplePerCon = 6, sampleRounds=1, arti_noise_level = 0, 
                       tRangeRaw=np.arange(-500,4000,1), step=50, dt=10, subConditions = None):
    
    start_time = time.time()
    pseudo_PopByRegion = {'dlpfc':[], 'fef':[]}
    
    # for each region
    for k in pseudo_PopByRegion.keys():
        # pseudo pop overall
        pseudo_Pop = []
        # from real session, randomly sample one trial matching the condition, store all cell firing rates from this session and stack
        for session in session_paths:
            
            trial_df = Get_trial(session).trial_selection_df()
            trial_df = trial_df[(trial_df.accuracy == 1)].reset_index(drop = True) # correct only
            
            cells = [cc for cc in cellsToUse if '/'.join(cc.split('/')[:-3]) == session]
            
            firingsT = [] # store all firing rates from all cells from this session
            
            for c in sorted(cells):
                
                aryTemp = c.split('/')[-3]
                
                # define arrays correspond to regions
                if int(aryTemp[-1]) in dlpfcArrays: 
                    region = 'dlpfc'
                elif int(aryTemp[-1]) in fefArrays:
                    region = 'fef'
                else:
                    region = 'others'
                    
                if region == k:
                    mTsp = Get_spikes(c).spike_timeStamp()
                    spkT_all = spike_selection(mTsp, trial_df, trial_df.trial_index)
                    arrayFreqT_all = spike2freq(spkT_all, tRangeRaw, step, dt)
                    bslV = bslFreqV(tRange, tBsl[0], tBsl[1], arrayFreqT_all)
            
                    arrayFreqB_all = tCutArray(tRange, tslice[0], tslice[1], arrayFreqT_all - bslV)
            
                    bslMean = tCutArray(tRange, tBsl[0], tBsl[1], arrayFreqT_all - bslV).mean() # should be 0 for bsl corrected data
                    bslStd = tCutArray(tRange, tBsl[0], tBsl[1], arrayFreqT_all - bslV).std()
                    
                    arrayFreqB_all = (arrayFreqB_all - bslMean)/bslStd if bslStd > 0 else (arrayFreqT_all - bslMean)/1# bsl z score
                    
                    firingsT += [arrayFreqB_all] 
                
            firingsT = np.array(firingsT)
            
            pseudo_trial = []
            
            if len(firingsT) >0:
                for r in range(sampleRounds):
                    
                    if subConditions == None:
                        for comb in locCombs:
                            loc1,loc2 = comb[0], comb[1]
                            locs = str(loc1)+'_'+str(loc2)
                            
                            for tt in ttypes:
                                #for s in range(samplePerCon):
                                trial_X = trial_df[(trial_df.locs==locs) & (trial_df.type==tt)].index
                                
                                sampleFirings = firingsT[:,trial_X,:]
                                arti_noise = np.random.normal(0,arti_noise_level,size=sampleFirings.shape)
                                sampleFirings += arti_noise
                                
                                rng = np.random.default_rng()
                                rng.shuffle(sampleFirings, axis=1)
                                
                                trial_XT = np.random.choice(np.arange(len(trial_X)), samplePerCon, replace=False)
                                
                                pseudo_trial += [sampleFirings[:,trial_XT,:]]
                    else:
                        for comb in subConditions:
                            loc1,loc2, tt = comb[0][0], comb[0][1], comb[1]
                            locs = str(loc1)+'_'+str(loc2)
                            
                            trial_X = trial_df[(trial_df.locs==locs) & (trial_df.type==tt)].index
                            
                            sampleFirings = firingsT[:,trial_X,:]
                            arti_noise = np.random.normal(0,arti_noise_level,size=sampleFirings.shape)
                            sampleFirings += arti_noise
                            
                            rng = np.random.default_rng()
                            rng.shuffle(sampleFirings, axis=1)
                            
                            trial_XT = np.random.choice(np.arange(len(trial_X)), samplePerCon, replace=False)
                            
                            pseudo_trial += [sampleFirings[:,trial_XT,:]]
                        
                pseudo_trial = np.concatenate(pseudo_trial, axis=1)
            
                pseudo_Pop += [pseudo_trial]
                
        
        pseudo_Pop = np.concatenate(pseudo_Pop, axis=0) # cell * trial * time
        pseudo_Pop = np.swapaxes(pseudo_Pop, 0,1) # shape2 trial * cell * time
        
        pseudo_PopByRegion[k] = pseudo_Pop
    
    print(f'{(time.time() - start_time):.4f}')
    
    return pseudo_PopByRegion


# In[] pseudo population by region, trial_matched_within_session
def pseudo_PopByRegion_pooled(session_paths, cellsToUse, monkey_Info, locCombs, ttypes, samplePerCon = 6, sampleRounds=1, arti_noise_level = 0, tRangeRaw=np.arange(-500,4000,1), step=50, dt=10, 
                              toNormalize = True, toRmvBsl = True, accs = 1, subConditions = None):
    
    start_time = time.time()
    pseudo_PopByRegion = {'dlpfc':[], 'fef':[]}
    
    # for each region
    for k in pseudo_PopByRegion.keys():
        # pseudo pop overall
        pseudo_Pop = []
        # from real session, randomly sample one trial matching the condition, store all cell firing rates from this session and stack
        for session in session_paths:
            
            monkey = session.split('/')[4]
            dlpfcArrays = monkey_Info[monkey]['dlpfcArrays']
            fefArrays = monkey_Info[monkey]['fefArrays']
            
            tBsl = monkey_Info[monkey]['tBsl']
            tslices = monkey_Info[monkey]['tslices']
            
            trial_df = Get_trial(session).trial_selection_df()
            
            if accs == None:
                trial_df = trial_df # all trials
            else:
                trial_df = trial_df[(trial_df.accuracy == accs)].reset_index(drop = True) # correct only
                
            
            cells = [cc for cc in cellsToUse if '/'.join(cc.split('/')[:-3]) == session]
            
            firingsT = [] # store all firing rates from all cells from this session
            
            for c in sorted(cells):
                
                aryTemp = c.split('/')[-3]
                
                # define arrays correspond to regions
                if int(aryTemp[-1]) in dlpfcArrays: 
                    region = 'dlpfc'
                elif int(aryTemp[-1]) in fefArrays:
                    region = 'fef'
                else:
                    region = 'others'
                
                # if in the region
                if region == k:
                    mTsp = Get_spikes(c).spike_timeStamp()
                    spkT_all = spike_selection(mTsp, trial_df, trial_df.trial_index)
                    arrayFreqT_all = spike2freq(spkT_all, tRangeRaw, step, dt)
                    
                    tRange = np.arange(tRangeRaw[0], tRangeRaw[-1],dt)
            
                    bslMean = tCutArray(tRange, tBsl[0], tBsl[1], arrayFreqT_all).mean() # should be 0 for bsl corrected data - bslV
                    bslStd = tCutArray(tRange, tBsl[0], tBsl[1], arrayFreqT_all).std() #- bslV
                    
                    # bsl z score
                    if toNormalize:
                        arrayFreqB_all = (arrayFreqT_all - bslMean)/bslStd if bslStd > 0 else (arrayFreqT_all - bslMean)/1
                    else:
                        if toRmvBsl:
                            arrayFreqB_all = (arrayFreqT_all - bslMean)
                        else:
                            arrayFreqB_all = arrayFreqT_all
                    
                    # firing rates from selected slices only
                    firingsT_c = []
                    for tslice in tslices:
                        arrayFreqB_T = tCutArray(tRange, tslice[0], tslice[1], arrayFreqB_all)
                        firingsT_c += [arrayFreqB_T]
                        
                    firingsT_c = np.hstack(firingsT_c) # hstack
                    
                    firingsT += [firingsT_c]
                
            firingsT = np.array(firingsT)
            
            pseudo_trial = []
            
            if len(firingsT) >0:
                for r in range(sampleRounds):
                    if subConditions == None:
                        for comb in locCombs:
                            loc1,loc2 = comb[0], comb[1]
                            locs = str(loc1)+'_'+str(loc2)
                            
                            for tt in ttypes:
                                
                                trial_X = trial_df[(trial_df.locs==locs) & (trial_df.type==tt)].index
                                
                                sampleFirings = firingsT[:,trial_X,:]
                                arti_noise = np.random.normal(0,arti_noise_level,size=sampleFirings.shape)
                                sampleFirings += arti_noise
                                
                                rng = np.random.default_rng()
                                rng.shuffle(sampleFirings, axis=1)
                                
                                trial_XT = np.random.choice(np.arange(len(trial_X)), samplePerCon, replace=False)
                                
                                pseudo_trial += [sampleFirings[:,trial_XT,:]]
                    
                    else:
                        for comb in subConditions:
                            loc1,loc2, tt = comb[0][0], comb[0][1], comb[1]
                            locs = str(loc1)+'_'+str(loc2)
                            
                            trial_X = trial_df[(trial_df.locs==locs) & (trial_df.type==tt)].index
                            
                            sampleFirings = firingsT[:,trial_X,:]
                            arti_noise = np.random.normal(0,arti_noise_level,size=sampleFirings.shape)
                            sampleFirings += arti_noise
                            
                            rng = np.random.default_rng()
                            rng.shuffle(sampleFirings, axis=1)
                            
                            trial_XT = np.random.choice(np.arange(len(trial_X)), samplePerCon, replace=False)
                            
                            pseudo_trial += [sampleFirings[:,trial_XT,:]]
                    
                    
                        
                pseudo_trial = np.concatenate(pseudo_trial, axis=1)
            
                pseudo_Pop += [pseudo_trial]
                
        
        pseudo_Pop = np.concatenate(pseudo_Pop, axis=0) # cell * trial * time
        pseudo_Pop = np.swapaxes(pseudo_Pop, 0,1) # shape2 trial * cell * time
        
        pseudo_PopByRegion[k] = pseudo_Pop
    
    print(f'{(time.time() - start_time):.4f}')
    
    return pseudo_PopByRegion
