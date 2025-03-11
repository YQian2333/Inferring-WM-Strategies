# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 02:59:52 2024

@author: aka2333
"""
# In[]
import numpy as np
from scipy import stats
import pandas as pd


# In[]
def scale(x, upper=1, lower=0, method = '01'):
    if method == 'std':
      x = (x - x.mean()) / x.std() if x.std()>0 else (x - x.mean()) / 1

    elif method == '01':
      if x.max() == x.min():
          x = x
      else:
          x = (x-x.min())/(x.max()-x.min())
          x = x * (upper-lower) + lower
    return x


# In[]
def permutation_p_counts(x, permuted_xs, tail = 'two'):
    x, permuted_xs = x.round(5), permuted_xs.round(5)
    
    if tail == 'two':
        pvalue = min(len(permuted_xs[permuted_xs<x])/len(permuted_xs), 
                     len(permuted_xs[permuted_xs>x])/len(permuted_xs)) # two-tail
        
    elif tail == 'greater':
        pvalue = len(permuted_xs[permuted_xs>x])/len(permuted_xs) # one-tail greater than
        
    elif tail == 'smaller':
        pvalue = len(permuted_xs[permuted_xs<x])/len(permuted_xs) # one-tail smaller than
    return pvalue

def permutation_p(x, permuted_xs, tail = 'two'):
    x, permuted_xs = x.round(5), permuted_xs.round(5)
    #_percentile, which is equivalent to the counting the empirical probability
    if tail == 'two':
        pvalue = (50-abs(stats.percentileofscore(permuted_xs, x)-50))*2/100 # two-tail
        
    elif tail == 'greater':
        pvalue = (100-stats.percentileofscore(permuted_xs, x))/100 # one-tail greater than
        
    elif tail == 'smaller':
        pvalue = (stats.percentileofscore(permuted_xs, x))/100 # one-tail smaller than
    return pvalue



def permutation_pCI(xs, permuted_xs, alpha=5, tail = 'two', base='h0'):
    xs, permuted_xs = xs.round(5), permuted_xs.round(5)
    
    if base == 'h0':
        b1,b2 = xs.min(), xs.max()
        if tail=='two':
            b1,b2 = stats.scoreatpercentile(xs,alpha/2), stats.scoreatpercentile(xs,100-alpha/2)
            pvalue = (len(permuted_xs[np.logical_and(permuted_xs<b2, permuted_xs>b1)])+1)/(len(permuted_xs)+1)
            
        elif tail=='greater':
            b1 = stats.scoreatpercentile(xs,alpha/1)
            pvalue = (len(permuted_xs[np.logical_and(True, permuted_xs>b1)])+1)/(len(permuted_xs)+1)
            
        elif tail=='smaller':
            b2 = stats.scoreatpercentile(xs,100-alpha/1)
            pvalue = (len(permuted_xs[np.logical_and(permuted_xs<b2, True)])+1)/(len(permuted_xs)+1)
    
    elif base == 'h1':
        # if h1, should always use tail='two'
        b1,b2 = permuted_xs.min(), permuted_xs.max()
        if tail=='two':
            b1,b2 = stats.scoreatpercentile(permuted_xs,alpha/2), stats.scoreatpercentile(permuted_xs,100-alpha/2)
            pvalue = (len(xs[np.logical_and(xs<b2, xs>b1)])+1)/(len(xs)+1)
            
        elif tail=='greater':
            b1 = stats.scoreatpercentile(permuted_xs,alpha/1)
            pvalue = (len(xs[np.logical_and(xs<b2, xs>b1)])+1)/(len(xs)+1)
            
        elif tail=='smaller':
            b2 = stats.scoreatpercentile(permuted_xs,100-alpha/1)
            pvalue = (len(xs[np.logical_and(xs<b2, xs>b1)])+1)/(len(xs)+1)
            

    return pvalue

def permutation_p_overlap(xs, permuted_xs, alpha=5, tail = 'two'):
    # unfinished
    xs, permuted_xs = xs.round(5), permuted_xs.round(5)
    
    b1,b2 = xs.min(), xs.max()
    b1p, b2p = permuted_xs.min(), permuted_xs.max()
    
    if tail=='two':
        b1,b2 = stats.scoreatpercentile(xs,alpha/2), stats.scoreatpercentile(xs,100-alpha/2)
        b1p,b2p = stats.scoreatpercentile(permuted_xs,alpha/2), stats.scoreatpercentile(permuted_xs,100-alpha/2)
        
        pvalue = len(xs[np.logical_and(xs<b2, xs>b1)])/len(xs)
        
    elif tail=='greater':
        b1 = stats.scoreatpercentile(xs,alpha/1)
        b1p = stats.scoreatpercentile(permuted_xs,alpha/1)
        
        pvalue = len(xs[np.logical_and(True, xs>b1)])/len(xs)
        
    elif tail=='smaller':
        b2 = stats.scoreatpercentile(xs,100-alpha/1)
        b2p = stats.scoreatpercentile(permuted_xs,100-alpha/1)
        
        pvalue = len(xs[np.logical_and(xs<b2, True)])/len(xs)

    return pvalue

def hedges_g(x1s, x2s):
    # effect size
    s1, s2 = x1s.std(), x2s.std()
    n1, n2 = len(x1s), len(x2s)
    m1, m2 = x1s.mean(), x2s.mean()
    s_ = np.sqrt(((n1-1)*s1**2 + (n2-1)*s2**2)/(n1+n2-2))
    g = (1-3/(4*(n1+n2)-9))*((m1 - m2)/(s_))
    
    return g
#%%
def p_1samp_bootstrap(xs, expect, prop=0.8, nBoot = 1000, method='mean', tail = 'greater'):
    #xs = xs.round(5)
    
    statsBoot = []
    
    # sample from distribution
    for nbt in range(nBoot):
        np.random.seed(nbt)
        sampleID = np.sort(np.random.choice(len(xs), int(len(xs)*prop), replace = False))
        sample = xs[sampleID]

        if method == 'mean':
            statsBoot += [sample.mean()]
        elif method == 'median':
            statsBoot += [np.median(sample)]
    
    statsBoot = np.array(statsBoot)
    
    if tail=='greater':
        pvalue = len(statsBoot[statsBoot<expect])/len(statsBoot)
        
    elif tail=='smaller':
        pvalue = len(statsBoot[statsBoot>expect])/len(statsBoot)
    
    return pvalue

def p_1samp(xs, expect, tail = 'greater'):
    #xs = xs.round(5)
    
    if tail=='greater':
        pvalue = len(xs[xs<expect])/len(xs)
        
    elif tail=='smaller':
        pvalue = len(xs[xs>expect])/len(xs)
    
    return pvalue

def permutation_pCI_1samp(xs, expected, alpha=5, tail = 'greater'):
    xs = xs.round(5)
    
    if tail=='greater':
        b1 = stats.scoreatpercentile(xs,alpha/1)
        pvalue = len(xs[xs<expected])/len(xs)
        
    elif tail=='smaller':
        b2 = stats.scoreatpercentile(xs,100-alpha/1)
        pvalue = len(xs[xs>expected])/len(xs)
        

    return pvalue


# In[]
def inversed_label(Y_columnsLabels, toDecode_labels, tt):
    if toDecode_labels == 'locKey':
        toDecode_labels_inv = 'loc1' if tt == 1 else 'loc2'
    elif toDecode_labels == 'locX':
        toDecode_labels_inv = 'loc2' if tt == 1 else 'loc1'
    elif toDecode_labels == 'loc1':
        toDecode_labels_inv = 'loc2'
    elif toDecode_labels == 'loc2':
        toDecode_labels_inv = 'loc1'

    return toDecode_labels_inv


# In[]
def bootstrap95_p(distribution1, distribution2, nboots=1000, alpha=0.05):
    # Bootstrap resampling
    bootstraps1 = stats.bootstrap((distribution1,), np.mean, n_resamples=nboots).confidence_interval
    bootstraps2 = stats.bootstrap((distribution2,), np.mean, n_resamples=nboots).confidence_interval
    
    # Calculate 95% confidence intervals for each bootstrapped distribution
    ci1_lower, ci1_upper = bootstraps1.low, bootstraps1.high
    ci2_lower, ci2_upper = bootstraps2.low, bootstraps2.high
    
    # Check for overlap in 95th percentile ranges
    overlap = max(0, min(ci1_upper, ci2_upper) - max(ci1_lower, ci2_lower))
    overlap_points = int(overlap > 0)
    
    # Estimate p-value based on overlap
    p_value = (1 + overlap_points) / (nboots + 1)
    
    return p_value, (ci1_lower, ci1_upper), (ci2_lower, ci2_upper)

#In[]

def shuff_label(distribution1, distribution2, nPerms=1000):
    ndis1, ndis2 = len(distribution1), len(distribution2)
    pooled_dis = np.concatenate((distribution1, distribution2))
    
    # group shuffled distribution
    dis1_shuff = []
    dis2_shuff = []

    for npm in range(nPerms):
        np.random.seed(npm)
        dis1_shuffID = np.sort(np.random.choice(len(pooled_dis), ndis1, replace = False))
        dis2_shuffID = np.setdiff1d(np.arange(len(pooled_dis)), dis1_shuffID, assume_unique=True)
        
        # group shuffled distribution
        dis1_shuff += [pooled_dis[dis1_shuffID]]
        dis2_shuff += [pooled_dis[dis2_shuffID]]

    return np.array(dis1_shuff), np.array(dis2_shuff)

# In[]
def permutation_p_diff(distribution1, distribution2, nPerms=1000, tail = 'two'):
    
    dis1_shuff, dis2_shuff = shuff_label(distribution1, distribution2, nPerms=nPerms)
    disShuff_diff = dis1_shuff.mean(-1) - dis2_shuff.mean(-1)
    dis_diff = (distribution1.mean() - distribution2.mean())#.mean()
    
    if tail == 'two':
        pvalue = (50-abs(stats.percentileofscore(disShuff_diff, dis_diff)-50))*2/100 # two-tail
        
    elif tail == 'greater':
        pvalue = (100-stats.percentileofscore(disShuff_diff, dis_diff))/100 # one-tail greater than
        
    elif tail == 'smaller':
        pvalue = (stats.percentileofscore(disShuff_diff, dis_diff))/100 # one-tail smaller than
    
    return pvalue
