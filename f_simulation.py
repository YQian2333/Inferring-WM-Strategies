# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 15:28:12 2024

@author: aka2333
"""

#%% 
import numpy as np
import scipy
import pandas as pd

# turn off warning messages
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# system paths
from itertools import permutations, combinations, product # itertools

#%%
def generate_trials(N_batch, locs, ttypes, accFracs=(1,0,0)):
    
    """
    N_batch = number of trials
    locs = list of locations
    ttypes = list of task types
    accFracs = tuple of accuracy fractions
    """
    # generate trial information

    trialInfo = []
        
    # correct (acc=1)
    for n in range(int(accFracs[0]*N_batch)):
        np.random.seed(n)
        locpair = np.random.choice(locs, size = 2, replace = False)
        ttype = np.random.choice(ttypes)

        loc1, loc2 = tuple(locpair)

        choice = loc2 if ttype == 1 else loc1
        acc = 1

        trialInfo += [[loc1, loc2, ttype, choice, acc]]

    # random error (acc=0)
    for n in range(int(accFracs[1] * N_batch)):
        np.random.seed(n)
        locpair = np.random.choice(locs, size = 2, replace = False)
        ttype = np.random.choice(ttypes)

        loc1, loc2 = tuple(locpair)

        acc = 0
        errs = [i for i in locs if i!=loc2] if ttype == 1 else [i for i in locs if i!=loc1]
        choice = np.random.choice(errs, replace = False)

        trialInfo += [[loc1, loc2, ttype, choice, acc]]

    # non-random error (acc=-1)
    for n in range(int(accFracs[2] * N_batch)):
        np.random.seed(n)
        locpair = np.random.choice(locs, size = 2, replace = False)
        ttype = np.random.choice(ttypes)

        loc1, loc2 = tuple(locpair)

        acc = -1 # non-random errors acc coded as -1

        choice = loc1 if ttype == 1 else loc2

        trialInfo += [[loc1, loc2, ttype, choice, acc]]

    trialInfo = pd.DataFrame(trialInfo, columns = ['loc1','loc2','ttype','choice','acc'])
    
    trialInfo['locX'] = 0
    for i in range(len(trialInfo)):
        trialInfo.loc[i,'locX'] = trialInfo.loc[i,'loc1'] if trialInfo.loc[i,'ttype'] == 1 else trialInfo.loc[i,'loc2']
        
    trialInfo = trialInfo.sample(frac = 1, random_state=233).reset_index(drop = True) # shuffle order

    return trialInfo

#%%
def generate_trials_balance(N_batch, locs, ttypes, accFracs=(1,0,0), varGos = (0,)):
    
    """
    N_batch = number of trials
    locs = list of locations
    ttypes = list of task types
    accFracs = tuple of accuracy fractions
    varGos = tuple of variable go cue onset time
    """

    # generate equal number of trials per condition
    
    trialInfo = []
    
    locpairs = list(permutations(locs,2))
    subConditions = list(product(locpairs, ttypes, varGos))
    
    # correct (acc=1)
    for sc in subConditions:
        
        nPerCond = int(accFracs[0]*N_batch//len(subConditions))
        
        locpair, ttype, go = sc[0], sc[1], sc[2]
        loc1, loc2 = tuple(locpair)
        choice = loc2 if ttype == 1 else loc1
        acc = 1
        
        for n in range(nPerCond):
            trialInfo += [[loc1, loc2, ttype, go, choice, acc]]

    # random error (acc=0)
    for sc in subConditions:
        
        nPerCond = int(accFracs[1]*N_batch//len(subConditions))
        
        locpair, ttype, go = sc[0], sc[1], sc[2]
        loc1, loc2 = tuple(locpair)
        #choice = loc2 if ttype == 1 else loc1
        acc = 0
        
        errs = [i for i in locs if (i!=loc1 and i!=loc2)] # if ttype == 1 else [i for i in locs if i!=loc1]
        
        for e in errs:
            nPerErr = nPerCond//len(errs)
            choice = e
                
            for n in range(nPerErr):
                trialInfo += [[loc1, loc2, ttype, go, choice, acc]]
            

    # non-random error (acc=-1)
    for sc in subConditions:
        
        nPerCond = int(accFracs[2]*N_batch//len(subConditions))
        
        locpair, ttype, go = sc[0], sc[1], sc[2]
        loc1, loc2 = tuple(locpair)
        #choice = loc2 if ttype == 1 else loc1
        acc = -1
        
        choice = loc1 if ttype == 1 else loc2
        
        for n in range(nPerCond):
            trialInfo += [[loc1, loc2, ttype, go, choice, acc]]
    

    trialInfo = pd.DataFrame(trialInfo, columns = ['loc1','loc2','ttype', 'go','choice','acc'])
    trialInfo = trialInfo.sample(frac=1).reset_index(drop=True)
    
    trialInfo['locX'] = 0
    for i in range(len(trialInfo)):
        trialInfo.loc[i,'locX'] = trialInfo.loc[i,'loc1'] if trialInfo.loc[i,'ttype'] == 1 else trialInfo.loc[i,'loc2']
        
    trialInfo = trialInfo.sample(frac = 1, random_state=233).reset_index(drop = True) # shuffle order

    return trialInfo

#%%
def generate_trials_ttypeProportion(N_batch, locs, ttypes, accFracs=(1,0,0), retProportion=0.5):
    
    """
    N_batch = number of trials
    locs = list of locations
    ttypes = list of task types
    accFracs = tuple of accuracy fractions
    retProportion = proportion of T/T trials
    """

    # generate equal trials per condition, but with different ttype proportions

    trialInfo = []
    
    locpairs = list(permutations(locs,2))
    subConditions = list(product(locpairs, ttypes))
    
    # correct (acc=1)
    for lc in locpairs:
        loc1, loc2 = lc
        nPerLC = int(accFracs[0]*N_batch//len(locpairs))
        
        for tt in ttypes:
            nPerCond = int(nPerLC*retProportion) if tt==1 else int(nPerLC*(1-retProportion))
            
            ttype = tt            
            choice = loc2 if ttype == 1 else loc1
            acc = 1
            
            for n in range(nPerCond):
                trialInfo += [[loc1, loc2, ttype, choice, acc]]

    # random error (acc=0)
    for lc in locpairs:
        loc1, loc2 = lc
        nPerLC = int(accFracs[1]*N_batch//len(locpairs))
        
        for tt in ttypes:
            nPerCond = int(nPerLC*retProportion) if tt==1 else int(nPerLC*(1-retProportion))
            
            ttype = tt            
            #choice = loc2 if ttype == 1 else loc1
            acc = 0
            
            errs = [i for i in locs if (i!=loc1 and i!=loc2)] # if ttype == 1 else [i for i in locs if i!=loc1]
            for e in errs:
                nPerErr = nPerCond//len(errs)
                choice = e

            for n in range(nPerCond):
                trialInfo += [[loc1, loc2, ttype, choice, acc]]
            
            

    # non-random error (acc=-1)
    for lc in locpairs:
        loc1, loc2 = lc
        nPerLC = int(accFracs[2]*N_batch//len(locpairs))
        
        for tt in ttypes:
            nPerCond = int(nPerLC*retProportion) if tt==1 else int(nPerLC*(1-retProportion))
            
            ttype = tt            
            acc = -1
        
            choice = loc1 if ttype == 1 else loc2
            
            for n in range(nPerCond):
                trialInfo += [[loc1, loc2, ttype, choice, acc]]
    
    

    trialInfo = pd.DataFrame(trialInfo, columns = ['loc1','loc2','ttype','choice','acc'])
    trialInfo = trialInfo.sample(frac=1).reset_index(drop=True)
    
    trialInfo['locX'] = 0
    for i in range(len(trialInfo)):
        trialInfo.loc[i,'locX'] = trialInfo.loc[i,'loc1'] if trialInfo.loc[i,'ttype'] == 1 else trialInfo.loc[i,'loc2']
        
    trialInfo = trialInfo.sample(frac = 1, random_state=233).reset_index(drop = True) # shuffle order

    return trialInfo


#%%
def generate_X_8ch(trialInfo, trialEvents, tRange, dt, vloc=1, vttype=1, noise=0.1, gocue = False, vcue = 1):

    locs = sorted(trialInfo.loc1.unique())
    ttypes = sorted(trialInfo.ttype.unique())
    
    XlocT1 = np.sqrt(2*.01*np.sqrt(10)*np.sqrt(dt)*noise*noise)*np.random.randn(len(trialInfo), len(tRange), len(locs)) # initialize input channels filled with noises
    XlocT2 = np.sqrt(2*.01*np.sqrt(10)*np.sqrt(dt)*noise*noise)*np.random.randn(len(trialInfo), len(tRange), len(locs))
    Xfix = np.sqrt(2*.01*np.sqrt(10)*np.sqrt(dt)*noise*noise)*np.random.randn(len(trialInfo), len(tRange), 1) + vcue # fixation initially set as 1

    for n in range(len(trialInfo)):
        loc1, loc2, ttype = trialInfo.loc[n,'loc1'], trialInfo.loc[n,'loc2'], trialInfo.loc[n,'ttype']

        for t in range(len(tRange)):
            if trialEvents['s1'][0] <= tRange[t] < trialEvents['s1'][1]:
                XlocT1[n,t,loc1] += vloc

            elif trialEvents['s2'][0] <= tRange[t] < trialEvents['s2'][1]:
                if ttype == 1:
                    XlocT1[n,t,loc2] += vloc
                elif ttype == 2:
                    XlocT2[n,t,loc2] += vloc
            elif trialEvents['go'][0] <= tRange[t] < trialEvents['go'][1]:
                    Xfix[n,t,:] += -1 # fixation disappear as go cue

    X = np.concatenate((XlocT1,XlocT2,Xfix,), axis=2) if gocue else np.concatenate((XlocT1,XlocT2,), axis=2)

    return X


#%%
def generate_X_4ch(trialInfo, trialEvents, tRange, dt, vtgt=1, vdis=1, vttype=1, noise=0.1, gocue = False, vcue = 1):
    
    """
    trialInfo = dataframe of trial information
    trialEvents = dictionary of trial events
    tRange = time range of simulation
    dt = time step
    vtgt = target value
    vdis = distractor value (for 4ch version this is required)
    vttype = target type value (not relevant)
    noise = noise level
    gocue = whether to include go cue
    vcue = go cue value (if go cue included)
    """
    
    # generate input stimulus for 4-channel version of input layer

    locs = sorted(trialInfo.loc1.unique())
    ttypes = sorted(trialInfo.ttype.unique())
    
    Xloc = np.sqrt(2*.01*np.sqrt(10)*np.sqrt(dt)*noise*noise)*np.random.randn(len(trialInfo), len(tRange), len(locs)) # initialize input channels filled with noises
    Xfix = np.sqrt(2*.01*np.sqrt(10)*np.sqrt(dt)*noise*noise)*np.random.randn(len(trialInfo), len(tRange), 1) + vcue # fixation initially set as 1

    for n in range(len(trialInfo)):
        loc1, loc2, ttype = trialInfo.loc[n,'loc1'], trialInfo.loc[n,'loc2'], trialInfo.loc[n,'ttype']

        for t in range(len(tRange)):
            if trialEvents['s1'][0] <= tRange[t] < trialEvents['s1'][1]:
                Xloc[n,t,loc1] += vtgt
                
            elif trialEvents['s2'][0] <= tRange[t] < trialEvents['s2'][1]:
                if ttype == 1:
                    Xloc[n,t,loc2] += vtgt
                elif ttype == 2:
                    Xloc[n,t,loc2] += vdis
                
            elif trialEvents['go'][0] <= tRange[t] < trialEvents['go'][1]:
                Xfix[n,t,:] += -1 # fixation disappear as go cue

    X = np.concatenate((Xloc,Xfix,), axis=2) if gocue else np.concatenate((Xloc,), axis=2)
    
    return X

#%%
def generate_X_6ch(trialInfo, trialEvents, tRange, dt, vloc=1, vttype=1, noise=0.1, gocue = False, vcue = 1):

    locs = sorted(trialInfo.loc1.unique())
    ttypes = sorted(trialInfo.ttype.unique())

    Xloc = np.sqrt(2*.01*np.sqrt(10)*np.sqrt(dt)*noise*noise)*np.random.randn(len(trialInfo), len(tRange), len(locs)) # initialize input channels filled with noises
    Xttype = np.sqrt(2*.01*np.sqrt(10)*np.sqrt(dt)*noise*noise)*np.random.randn(len(trialInfo), len(tRange), len(ttypes))
    Xfix = np.sqrt(2*.01*np.sqrt(10)*np.sqrt(dt)*noise*noise)*np.random.randn(len(trialInfo), len(tRange), 1) + vcue # fixation initially set as 1

    for n in range(len(trialInfo)):
        loc1, loc2, ttype = trialInfo.loc[n,'loc1'], trialInfo.loc[n,'loc2'], trialInfo.loc[n,'ttype']

        for t in range(len(tRange)):
            if trialEvents['s1'][0] <= tRange[t] < trialEvents['s1'][1]:
                Xloc[n,t,loc1] += vloc
                Xttype[n,t,0] += vttype # s1 always target (red)

            elif trialEvents['s2'][0] <= tRange[t] < trialEvents['s2'][1]:
                Xloc[n,t,loc2] += vloc
                Xttype[n,t,(ttype-1)] += vttype # s2 can be retarget (red,ch0) or distractor (green,ch1)
            
            elif trialEvents['go'][0] <= tRange[t] < trialEvents['go'][1]:
                Xfix[n,t,:] += -1 # fixation disappear as go cue

    X = np.concatenate((Xloc,Xttype,Xfix,), axis=2) if gocue else np.concatenate((Xloc,Xttype,), axis=2)

    return X




#%%
def generate_Y(output_size, labels, lowB, upB, dt, expected, expectedAlt=None):
    '''
    lowB : fitting epoch start time
    upB : fitting epoch ending time
    dt : timestep
    output_expected: expected value at the correct output channel
    expectedAlt: expected value at the other output channel, float
    '''
    lowB, upB = lowB, upB

    tx = np.arange(lowB, upB, dt)

    Y = np.full((len(labels), len(tx), output_size),0,dtype=float)

    for i in range(len(labels)):
        loc = labels[i]
        indices = np.arange(output_size)
        expected = expected

        if expected == 0:
            Y[i,:,:] = expected
    
        else:
            loc = loc
            Y[i,:,loc] = expected
            if expectedAlt is None:
                Y[i,:,indices!=loc] += (1-expected)/(output_size -1)
            else:
                Y[i,:,indices!=loc] += expectedAlt

    return Y


#%%
# sample corresponding in-/output tensors from the X/Y pool
def split_dataset(X,Y, frac = 0.8, ranseed = None):

    ranseed = np.random.randint(0,10000) if ranseed == None else ranseed

    np.random.seed(ranseed)

    train_setID = np.sort(np.random.choice(len(X), round(frac*len(X)),replace = False))
    test_setID = np.setdiff1d(np.arange(len(X)), train_setID, assume_unique=True)

    train_x = X[train_setID,:,:]
    train_y = Y[train_setID,:,:]# (locKey = 0,'locs','type','loc1','loc2')
    
    test_x = X[test_setID,:,:]
    test_y = Y[test_setID,:,:]

    return train_setID, train_x, train_y, test_setID, test_x, test_y


#%%
def split_dataset_balance(X,Y, trialInfo, locs=(0,1,2,3), ttypes=(1,2), frac = 0.8, ranseed = None):
    
    ranseed = np.random.randint(0,10000) if ranseed == None else ranseed
    np.random.seed(ranseed)

    locpairs = list(permutations(locs,2))
    subConditions = list(product(locpairs, ttypes))
    
    nPerCond = int(len(X)*frac//len(subConditions))
    
    train_setID = []
    test_setID = []
    for sc in subConditions:
        loc1,loc2,ttype = sc[0][0], sc[0][1], sc[1]
        idxs = trialInfo[(trialInfo.loc1==loc1) & (trialInfo.loc2==loc2) & (trialInfo.ttype==ttype)].index.values
        
        train_setID += [np.sort(np.random.choice(idxs, round(frac*len(idxs)),replace = False))]
        test_setID += [np.setdiff1d(idxs, train_setID, assume_unique=True)]
    
    train_setID = np.sort(np.concatenate(train_setID))
    test_setID = np.sort(np.concatenate(test_setID))

    train_x = X[train_setID,:,:]
    train_y = Y[train_setID,:,:]# (locKey = 0,'locs','type','loc1','loc2')
    
    test_x = X[test_setID,:,:]
    test_y = Y[test_setID,:,:]

    return train_setID, train_x, train_y, test_setID, test_x, test_y

