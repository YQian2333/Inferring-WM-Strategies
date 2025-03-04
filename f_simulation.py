# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 15:28:12 2024

@author: aka2333
"""

# In[]
import numpy as np
import scipy
from scipy.stats import vonmises # circular distribution

import pandas as pd

# turn off warning messages
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# system paths
from itertools import permutations, combinations, product # itertools

import torch
import torch.nn as nn
# In[] ricker wavelet (mexican hat) bump
def gaussian_dis(x, u=0, sigma=1):
    return np.exp(-(x - u) ** 2 / (2 * sigma ** 2)) / (np.sqrt(2 * np.pi) * sigma)

def scale_distribution(x, upper=1, lower=0):
    if x.max() == x.min():
        x = x # uniform
    else:
        x = (x-x.min())/(x.max()-x.min())
        x = x * (upper-lower) + lower
    return x

def bump_Win(distribution, input_size, bump_size, specificity = 0):
    distribution = distribution.round(decimals=4)
    distribution = np.roll(distribution, len(distribution) - np.where(distribution==distribution.max())[0][0])
    W = np.zeros((bump_size, input_size)) # W: hidden-hidden recurrent weight matrix (shape = h*h)
    
    offset = 0
    for i in range(input_size):
        if i%specificity == 0 and i>0:    
            offset += int(bump_size/input_size)
        W[:, i] = np.roll(distribution, offset)
    
    return W


def bump_Wrec(distribution, bump_size, specificity = 0):
    distribution = distribution.round(decimals=4)
    distribution = np.roll(distribution, len(distribution) - np.where(distribution==distribution.max())[0][0])
    W = np.zeros((bump_size, bump_size)) # W: hidden-hidden recurrent weight matrix (shape = h*h)
    
    offset = 0
    for i in range(bump_size):
        if i%specificity == 0 and i>0:    
            offset += specificity
        W[:, i] = np.roll(distribution, offset)
    
    return W

# In[]
def gaussian_disT(x, u=0, sigma=1):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    pi = torch.acos(torch.zeros(1).to(device)) * 2 # which is 3.1415927410125732
    gau = torch.exp(-(x - u) ** 2 / (2 * sigma ** 2)) / (torch.sqrt(2 * pi) * sigma)
    return gau.to(device)


def bump_WrecT(distribution, specificity = 0):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    #distribution = distribution#.round(decimals=4)
    distribution = torch.roll(distribution, (len(distribution) - torch.where(distribution==distribution.max())[0][0]).item())
    W = torch.zeros((distribution.shape[0], distribution.shape[0])).to(device) # W: hidden-hidden recurrent weight matrix (shape = h*h)
    
    offset = 0
    for i in range(len(W)):
        if i%specificity == 0 and i>0:    
            offset += specificity
        W[:, i] = torch.roll(distribution, offset)
    
    return W.to(device)
# In[]
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

# In[]
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

# In[]
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


# In[]
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


def generate_X_6ch(trialInfo, trialEvents, tRange, dt, vloc=1, vttype=1, noise=0.1, gocue = False, vcue = 1):

    locs = sorted(trialInfo.loc1.unique())
    ttypes = sorted(trialInfo.ttype.unique())

    #N_in = len(locs) + len(ttypes) # The number of network inputs.

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


def generate_X_8ch(trialInfo, trialEvents, tRange, dt, vloc=1, vttype=1, noise=0.1, gocue = False, vcue = 1):

    locs = sorted(trialInfo.loc1.unique())
    ttypes = sorted(trialInfo.ttype.unique())

    #N_in = len(locs) + len(ttypes) # The number of network inputs.

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


def generate_X_8ch_varGos(trialInfo, trialEvents, tRange, dt, vloc=1, vttype=1, noise=0.1, gocue = True, vcue = 1):

    locs = sorted(trialInfo.loc1.unique())
    ttypes = sorted(trialInfo.ttype.unique())

    #N_in = len(locs) + len(ttypes) # The number of network inputs.

    XlocT1 = np.sqrt(2*.01*np.sqrt(10)*np.sqrt(dt)*noise*noise)*np.random.randn(len(trialInfo), len(tRange), len(locs)) # initialize input channels filled with noises
    XlocT2 = np.sqrt(2*.01*np.sqrt(10)*np.sqrt(dt)*noise*noise)*np.random.randn(len(trialInfo), len(tRange), len(locs))
    Xfix = np.sqrt(2*.01*np.sqrt(10)*np.sqrt(dt)*noise*noise)*np.random.randn(len(trialInfo), len(tRange), 1) + vcue # fixation initially set as 1

    for n in range(len(trialInfo)):
        loc1, loc2, ttype, go = trialInfo.loc[n,'loc1'], trialInfo.loc[n,'loc2'], trialInfo.loc[n,'ttype'], trialInfo.loc[n,'go']

        for t in range(len(tRange)):
            if trialEvents['s1'][0] <= tRange[t] < trialEvents['s1'][1]:
                XlocT1[n,t,loc1] += vloc

            elif trialEvents['s2'][0] <= tRange[t] < trialEvents['s2'][1]:
                if ttype == 1:
                    XlocT1[n,t,loc2] += vloc
                elif ttype == 2:
                    XlocT2[n,t,loc2] += vloc
            elif (trialEvents['go'][0]+go) <= tRange[t] < trialEvents['go'][1]:
                    Xfix[n,t,:] += -1 # fixation disappear as go cue

    X = np.concatenate((XlocT1,XlocT2,Xfix,), axis=2) if gocue else np.concatenate((XlocT1,XlocT2,), axis=2)

    return X


def generate_X_8ch_invert(trialInfo, trialEvents, tRange, dt, vloc=1, vttype=1, noise=0.1, gocue = False, vcue = 1):

    locs = sorted(trialInfo.loc1.unique())
    ttypes = sorted(trialInfo.ttype.unique())

    #N_in = len(locs) + len(ttypes) # The number of network inputs.

    XlocT1 = np.sqrt(2*.01*np.sqrt(10)*np.sqrt(dt)*noise*noise)*np.random.randn(len(trialInfo), len(tRange), len(locs)) # initialize input channels filled with noises
    XlocT2 = np.sqrt(2*.01*np.sqrt(10)*np.sqrt(dt)*noise*noise)*np.random.randn(len(trialInfo), len(tRange), len(locs))
    Xfix = np.sqrt(2*.01*np.sqrt(10)*np.sqrt(dt)*noise*noise)*np.random.randn(len(trialInfo), len(tRange), 1) + vcue # fixation initially set as 1

    for n in range(len(trialInfo)):
        loc1, loc2, ttype, inversion = trialInfo.loc[n,'loc1'], trialInfo.loc[n,'loc2'], trialInfo.loc[n,'ttype'], trialInfo.loc[n,'invertSeq']

        for t in range(len(tRange)):
            if trialEvents['s1'][0] <= tRange[t] < trialEvents['s1'][1]:
                if inversion==0:
                    XlocT1[n,t,loc1] += vloc
                elif inversion==1:
                    if ttype == 1:
                        XlocT1[n,t,loc2] += vloc
                    elif ttype == 2:
                        XlocT2[n,t,loc2] += vloc

            elif trialEvents['s2'][0] <= tRange[t] < trialEvents['s2'][1]:
                if inversion==0:
                    if ttype == 1:
                        XlocT1[n,t,loc2] += vloc
                    elif ttype == 2:
                        XlocT2[n,t,loc2] += vloc
                elif inversion==1:
                    XlocT1[n,t,loc1] += vloc

            elif trialEvents['go'][0] <= tRange[t] < trialEvents['go'][1]:
                Xfix[n,t,:] += -1 # fixation disappear as go cue

    X = np.concatenate((XlocT1,XlocT2,Xfix,), axis=2) if gocue else np.concatenate((XlocT1,XlocT2,), axis=2)

    return X

# In[]


def generate_X_6ch_seqMulti(trialInfo, trialEvents, tRange, dt, vloc=1, vttype=1, noise=0.1, gocue = False, vcue = 1):
    
    locs = sorted(trialInfo.loc1.unique())
    ttypes = sorted(trialInfo.ttype.unique())

    #N_in = len(locs) + len(ttypes) # The number of network inputs.

    Xloc = np.sqrt(2*.01*np.sqrt(10)*np.sqrt(dt)*noise*noise)*np.random.randn(len(trialInfo), len(tRange), len(locs)) # initialize input channels filled with noises
    Xttype = np.sqrt(2*.01*np.sqrt(10)*np.sqrt(dt)*noise*noise)*np.random.randn(len(trialInfo), len(tRange), len(ttypes))
    Xfix = np.sqrt(2*.01*np.sqrt(10)*np.sqrt(dt)*noise*noise)*np.random.randn(len(trialInfo), len(tRange), 1) + vcue # fixation initially set as 1

    for n in range(len(trialInfo)):
        loc1, loc2, ttype = trialInfo.loc[n,'loc1'], trialInfo.loc[n,'loc2'], trialInfo.loc[n,'ttype']

        for t in range(len(tRange)):
            if trialEvents['s1'][0] <= tRange[t] < trialEvents['s1'][1]:
                Xloc[n,t,loc1] += vloc
                #Xttype[n,t,0] += vttype # s1 always target (red)

            elif trialEvents['s2'][0] <= tRange[t] < trialEvents['s2'][1]:
                Xloc[n,t,loc2] += vloc
                #Xttype[n,t,(ttype-1)] += vttype # s2 can be retarget (red,ch0) or distractor (green,ch1)

            elif trialEvents['cue'][0] <= tRange[t] < trialEvents['cue'][1]:
                Xttype[n,t,(ttype-1)] += vttype # s2 can be retarget (red,ch0) or distractor (green,ch1)
            
            elif trialEvents['go'][0] <= tRange[t] < trialEvents['go'][1]: 
                Xfix[n,t,:] += -1 # fixation disappear as go cue

    X = np.concatenate((Xloc,Xttype,Xfix,), axis=2) if gocue else np.concatenate((Xloc,Xttype,), axis=2)

    return X


def generate_X_8ch_simMulti(trialInfo, trialEvents, tRange, dt, vloc=1, vttype=1, noise=0.1, gocue = False, vcue = 1):

    locs = sorted(trialInfo.loc1.unique())
    ttypes = sorted(trialInfo.ttype.unique())

    #N_in = len(locs) + len(ttypes) # The number of network inputs.

    XlocT1 = np.sqrt(2*.01*np.sqrt(10)*np.sqrt(dt)*noise*noise)*np.random.randn(len(trialInfo), len(tRange), len(locs)) # initialize input channels filled with noises
    XlocT2 = np.sqrt(2*.01*np.sqrt(10)*np.sqrt(dt)*noise*noise)*np.random.randn(len(trialInfo), len(tRange), len(locs))
    Xttype = np.sqrt(2*.01*np.sqrt(10)*np.sqrt(dt)*noise*noise)*np.random.randn(len(trialInfo), len(tRange), len(ttypes))
    Xfix = np.sqrt(2*.01*np.sqrt(10)*np.sqrt(dt)*noise*noise)*np.random.randn(len(trialInfo), len(tRange), 1) + vcue # fixation initially set as 1

    for n in range(len(trialInfo)):
        loc1, loc2, ttype = trialInfo.loc[n,'loc1'], trialInfo.loc[n,'loc2'], trialInfo.loc[n,'ttype']

        for t in range(len(tRange)):
            if trialEvents['s1'][0] <= tRange[t] < trialEvents['s1'][1]:
                XlocT1[n,t,loc1] += vloc
                XlocT2[n,t,loc2] += vloc
                
            elif trialEvents['cue'][0] <= tRange[t] < trialEvents['cue'][1]:
                Xttype[n,t,(ttype-1)] += vttype

            elif trialEvents['go'][0] <= tRange[t] < trialEvents['go'][1]:
                Xfix[n,t,:] += -1 # fixation disappear as go cue

    X = np.concatenate((XlocT1,XlocT2,Xttype,Xfix,), axis=2) if gocue else np.concatenate((XlocT1,XlocT2,Xttype,), axis=2)
    
    return X





# In[]
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
        #loc1 = train_Info.loc1[i]
        #loc2 = train_Info.loc2[i]
        #ttype = train_Info.ttype[i]

        indices = np.arange(output_size)

        expected = expected

        if expected == 0:
            Y[i,:,:] = expected
            #Y[i,tx[0]:tx[-1]+1,indices!=choice] = output_expected

        else:
            loc = loc
            Y[i,:,loc] = expected
            if expectedAlt is None:
                Y[i,:,indices!=loc] += (1-expected)/(output_size -1)
            else:
                Y[i,:,indices!=loc] += expectedAlt

    return Y

#%%
def generate_Y_varDelay(output_size, trialInfo, test_label, varDelay_label, lowB, upB, dt, expected=1, expectedAlt=None, multiplier_low = 1, multiplier_up = 0):
    
    '''
    lowB : fitting epoch start time
    upB : fitting epoch ending time
    dt : timestep
    output_expected: expected value at the correct output channel
    expectedAlt: expected value at the other output channel, float
    '''
    varDelays = trialInfo[varDelay_label].unique()
    expectedAlt = (1-expected)/(output_size -1) if expectedAlt is None else expectedAlt
    
    lowB, upB = lowB, upB

    tx = np.arange(lowB + varDelays.min()*multiplier_low, upB + varDelays.max()*multiplier_up, dt)

    Y = np.full((len(trialInfo), len(tx), output_size), expectedAlt, dtype=float)

    for i in range(len(trialInfo)):
        loc = trialInfo[test_label][i]
        delay = trialInfo[varDelay_label][i]
        lowB_x = tx.tolist().index(lowB + delay*multiplier_low) if lowB + delay*multiplier_low > tx[0] else 0
        upB_x = tx.tolist().index(upB + delay*multiplier_up) if upB + delay*multiplier_up < tx[-1] else len(tx)
        
        #indices = np.arange(output_size)

        expected = expected

        if expected == 0:
            Y[i,lowB_x:upB_x,:] = expected # in the case need to mute all outputs
            
        else:
            loc = loc
            Y[i,lowB_x:upB_x,loc] = expected
            

    return Y


def generate_Y_circular(output_size, labels, lowB, upB, dt, expected, kappa):
    '''
    lowB : fitting epoch start time
    upB : fitting epoch ending time
    dt : timestep
    '''
    lowB, upB = lowB, upB

    tx = np.arange(lowB, upB, dt)

    Y = np.full((len(labels), len(tx), output_size),0,dtype=float)

    locTuning = np.linspace(-np.pi, np.pi, output_size+1) # +1 to avoid loc0 fully overlaps loc3
    #scaling = 1/max(vonmises.pdf(np.linspace(-np.pi, np.pi, 100), kappa, loc=0))

    for i in range(len(labels)):
        loc = labels[i]
        #loc1 = train_Info.loc1[i]
        #loc2 = train_Info.loc2[i]
        #ttype = train_Info.ttype[i]

        #indices = np.arange(output_size)

        if expected == 0:
            Y[i,:,:] = expected
            #Y[i,tx[0]:tx[-1]+1,indices!=choice] = output_expected

        else:
            loc = loc
            y = vonmises.pdf(locTuning[loc], kappa=kappa, loc = locTuning)[:-1] # * scaling
            #Y[i,:,:] = y
            #Y[i,:,:] = np.exp(y)/sum(np.exp(y)) # apply softmax
            Y[i,:,:] = y/y.sum()

    return Y


# In[]
# sample corresponding in-/output tensors from the X/Y pool
def split_dataset(X,Y, frac = 0.8, ranseed = None):

    ranseed = np.random.randint(0,10000) if ranseed == None else ranseed

    np.random.seed(ranseed)

    train_setID = np.sort(np.random.choice(len(X), round(frac*len(X)),replace = False))
    test_setID = np.setdiff1d(np.arange(len(X)), train_setID, assume_unique=True)

    train_x = X[train_setID,:,:]
    train_y = Y[train_setID,:,:]# (locKey = 0,'locs','type','loc1','loc2')
    #train_Info = trialInfo.loc[train_setID,:].reset_index(drop = True)

    #test_set = X[test_setID,:,:]
    test_x = X[test_setID,:,:]
    test_y = Y[test_setID,:,:]

    return train_setID, train_x, train_y, test_setID, test_x, test_y



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
    #train_Info = trialInfo.loc[train_setID,:].reset_index(drop = True)

    #test_set = X[test_setID,:,:]
    test_x = X[test_setID,:,:]
    test_y = Y[test_setID,:,:]

    return train_setID, train_x, train_y, test_setID, test_x, test_y



#%%
def chunk_trials(N_batch, items, chunk, chunkedPairs, accFracs=(1,0,0,0), chunkProportion=0.5, varGos = (0,)):
    
    
    # generate equal trials per condition, but with different ttype proportions

    trialInfo = []
    
    itempairs = tuple(permutations(items,2)) # showed at trial beginning
    locCombs = tuple(permutations(items,4)) 
    itemLocs = tuple(product(itempairs, locCombs))
    itemLocsGo = tuple(product(itempairs, locCombs, varGos))
    subConditions = list(product(itempairs, locCombs, chunk, varGos))
    
    N_batch = N_batch if N_batch > len(subConditions) else len(subConditions) # make sure have at least 1 trial per condition
    accFracs = np.array(accFracs)/np.sum(accFracs) # normalize
    
    # correct (acc=1)
    for il in itemLocsGo:
        i1, i2 = il[0]
        lc = il[1]
        go = il[2]
        nPerLC = int(accFracs[0]*N_batch//len(itemLocsGo))
        
        l1 = lc.index(items[0])
        l2 = lc.index(items[1])
        l3 = lc.index(items[2])
        l4 = lc.index(items[3])
        
        for ck in chunk:
            if ck == 1:
                nPerCond = int(nPerLC*chunkProportion)
                
                if il[0] in chunkedPairs:    
                    choice1 = lc.index(i1) # index of location showing the corresponding item
                    choice2 = lc.index(i2) # index of location showing the corresponding item
                    acc = 1
                    
                    for n in range(nPerCond):
                        trialInfo += [[i1, i2, ck, l1, l2, l3, l4, go, choice1, choice2, acc]]
                
            else:
                nPerCond = int(nPerLC*(1-chunkProportion))
            
                choice1 = lc.index(i1) # index of location showing the corresponding item
                choice2 = lc.index(i2) # index of location showing the corresponding item
                acc = 1
                
                for n in range(nPerCond):
                    trialInfo += [[i1, i2, ck, l1, l2, l3, l4, go, choice1, choice2, acc]]
                    
    # error type 1: incorrect 1, chose non-i2 (random1, acc=-1)
    for il in itemLocsGo:
        i1, i2 = il[0]
        lc = il[1]
        go = il[2]
        nPerLC = int(accFracs[1]*N_batch//len(itemLocsGo))
        
        l1 = lc.index(items[0])
        l2 = lc.index(items[1])
        l3 = lc.index(items[2])
        l4 = lc.index(items[3])
        
        for ck in chunk:
            if ck == 1:
                nPerCond = int(nPerLC*chunkProportion)   
                
                if il[0] in chunkedPairs:
                    acc = -1
                    errs = [i for i in items if (i not in (i1, i2))] # if ttype == 1 else [i for i in locs if i!=loc1]
                    for e in errs:
                        nPerErr = nPerCond//len(errs)
                        choice1 = lc.index(e)
                        choice2 = lc.index(np.random.choice([i for i in items if i not in (e,)]))

                    for n in range(nPerCond):
                        trialInfo += [[i1, i2, ck, l1, l2, l3, l4, go, choice1, choice2, acc]]
            
            else:
                nPerCond = int(nPerLC*(1-chunkProportion))    
                acc = -1
                
                errs = [i for i in items if (i not in (i1, i2))] # if ttype == 1 else [i for i in locs if i!=loc1]
                for e in errs:
                    nPerErr = nPerCond//len(errs)
                    choice1 = lc.index(e)
                    choice2 = lc.index(np.random.choice([i for i in items if i not in (e,)]))

                for n in range(nPerCond):
                    trialInfo += [[i1, i2, ck, l1, l2, l3, l4, go, choice1, choice2, acc]]
    
    
    
    # error type 2: correct 1, incorrect 2, chose non-i2 (non-random2, acc=-2)
    for il in itemLocsGo:
        i1, i2 = il[0]
        lc = il[1]
        go = il[2]
        nPerLC = int(accFracs[2]*N_batch//len(itemLocsGo))
        
        l1 = lc.index(items[0])
        l2 = lc.index(items[1])
        l3 = lc.index(items[2])
        l4 = lc.index(items[3])    
    
        for ck in chunk:
            if ck == 1:
                nPerCond = int(nPerLC*chunkProportion) if ck==1 else int(nPerLC*(1-chunkProportion))        
                if il[0] in chunkedPairs:
                    acc = -2
                
                    choice1 = lc.index(i1)
                    choice2 = lc.index(np.random.choice([i for i in items if i not in (i1,i2)]))
                    
                    for n in range(nPerCond):
                        trialInfo += [[i1, i2, ck, l1, l2, l3, l4, go, choice1, choice2, acc]]
                
            else:
                nPerCond = int(nPerLC*chunkProportion) if ck==1 else int(nPerLC*(1-chunkProportion))        
                acc = -2
            
                choice1 = lc.index(i1)
                choice2 = lc.index(np.random.choice([i for i in items if i not in (i1,i2)]))
                
                for n in range(nPerCond):
                    trialInfo += [[i1, i2, ck, l1, l2, l3, l4, go, choice1, choice2, acc]]
    
    
    # error type 3: incorrect 1, chose i2 (non-random1, acc=-3)
    for il in itemLocsGo:
        i1, i2 = il[0]
        lc = il[1]
        go = il[2]
        nPerLC = int(accFracs[3]*N_batch//len(itemLocsGo))
    
    
        l1 = lc.index(items[0])
        l2 = lc.index(items[1])
        l3 = lc.index(items[2])
        l4 = lc.index(items[3])
        
        
        for ck in chunk:
            if ck == 1:
                nPerCond = int(nPerLC*chunkProportion) if ck==1 else int(nPerLC*(1-chunkProportion))
                if il[0] in chunkedPairs:
                    acc = -3
                
                    choice1 = lc.index(i2)
                    choice2 = lc.index(np.random.choice([i for i in items if i not in (i2,)]))
                    
                    for n in range(nPerCond):
                        trialInfo += [[i1, i2, ck, l1, l2, l3, l4, go, choice1, choice2, acc]]
                
            else:
                nPerCond = int(nPerLC*chunkProportion) if ck==1 else int(nPerLC*(1-chunkProportion))
                acc = -3
            
                choice1 = lc.index(i2)
                choice2 = lc.index(np.random.choice([i for i in items if i not in (i2,)]))
                
                for n in range(nPerCond):
                    trialInfo += [[i1, i2, ck, l1, l2, l3, l4, go, choice1, choice2, acc]]
    
    

    trialInfo = pd.DataFrame(trialInfo, columns = ['stim1','stim2','chunk','item1Loc','item2Loc','item3Loc','item4Loc', 'delayLength','choice1Loc','choice2Loc','acc'])
    trialInfo['itemPairs'] = trialInfo['stim1'].astype('str') + '_' + trialInfo['stim2'].astype('str')
    trialInfo = trialInfo.sample(frac=1, random_state=233).reset_index(drop = True) # shuffle order

    return trialInfo


#%%
def chunk_X(trialInfo, trialEvents, tRange, dt, vitem=1, noise=0.1, gocue = False, vcue = 1):

    items = sorted(trialInfo.stim1.unique())
    locs = list(range(len(items)))
    chunk = sorted(trialInfo.chunk.unique())

    itemLocs = tuple(product(items, locs))
    
    XitemCenter = np.sqrt(2*.01*np.sqrt(10)*np.sqrt(dt)*noise*noise)*np.random.randn(len(trialInfo), len(tRange), len(items)) # center item display channels, initialize input channels filled with noises
    XitemLocs = np.sqrt(2*.01*np.sqrt(10)*np.sqrt(dt)*noise*noise)*np.random.randn(len(trialInfo), len(tRange), len(items)*len(locs)) # surrounding item display channels
    Xchunk = np.sqrt(2*.01*np.sqrt(10)*np.sqrt(dt)*noise*noise)*np.random.randn(len(trialInfo), len(tRange), len(chunk)-1) # chunk cue display channels
    Xfix = np.sqrt(2*.01*np.sqrt(10)*np.sqrt(dt)*noise*noise)*np.random.randn(len(trialInfo), len(tRange), 1) + vcue # fixation initially set as 1

    for n in range(len(trialInfo)):
        i1, i2, chunk = trialInfo.loc[n,'stim1'], trialInfo.loc[n,'stim2'], trialInfo.loc[n,'chunk']
        l1, l2, l3, l4 = trialInfo.loc[n,'item1Loc'], trialInfo.loc[n,'item2Loc'], trialInfo.loc[n,'item3Loc'], trialInfo.loc[n,'item4Loc']
        ck = trialInfo.loc[n,'chunk']
        delayLength = trialInfo.loc[n,'delayLength']

        for t in range(len(tRange)):
            if trialEvents['chunk'][0] <= tRange[t] < trialEvents['s2'][1]+delayLength:
                Xchunk[n,t,ck-1] += 1 if ck == 1 else 0
                
            if trialEvents['s1'][0] <= tRange[t] < trialEvents['s1'][1]:
                idx = items.index(i1)
                XitemCenter[n,t,idx] += vitem

            elif trialEvents['s2'][0]+delayLength <= tRange[t] < trialEvents['s2'][1]+delayLength:
                idx = items.index(i2)
                XitemCenter[n,t,idx] += vitem
                
            elif trialEvents['go'][0]+delayLength*2 <= tRange[t] < trialEvents['go'][1]:
                Xfix[n,t,:] += -1 # fixation disappear as go cue
                
                XitemLocs[n,t,l1+(4*0)] += vitem
                XitemLocs[n,t,l2+(4*1)] += vitem
                XitemLocs[n,t,l3+(4*2)] += vitem
                XitemLocs[n,t,l4+(4*3)] += vitem

    X = np.concatenate((XitemCenter,XitemLocs,Xchunk,Xfix,), axis=2) if gocue else np.concatenate((XitemCenter,XitemLocs,Xchunk,), axis=2)

    return X

#%%
def split_dataset_balance_chunking(X,Y, trialInfo, items=(1,2,3,4), chunk=(0,1), frac = 0.8, ranseed = None):
    
    ranseed = np.random.randint(0,10000) if ranseed == None else ranseed
    np.random.seed(ranseed)

    itemPairs = list(permutations(items,2))
    subConditions = list(product(itemPairs, chunk))
    
    nPerCond = int(len(X)*frac//len(subConditions))
    
    train_setID = []
    test_setID = []
    for sc in subConditions:
        i1, i2, ck = sc[0][0], sc[0][1], sc[1]
        idxs = trialInfo[(trialInfo.stim1==i1) & (trialInfo.stim2==i2) & (trialInfo.chunk==ck)].index.values
        
        if len(idxs) > 0:
            train_setID += [np.sort(np.random.choice(idxs, round(frac*len(idxs)),replace = False))]
            test_setID += [np.setdiff1d(idxs, train_setID, assume_unique=True)]
    
    train_setID = np.sort(np.concatenate(train_setID))
    test_setID = np.sort(np.concatenate(test_setID))

    train_x = X[train_setID,:,:]
    train_y = Y[train_setID,:,:]
    
    test_x = X[test_setID,:,:]
    test_y = Y[test_setID,:,:]

    return train_setID, train_x, train_y, test_setID, test_x, test_y



