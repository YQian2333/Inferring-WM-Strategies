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


def permutation_pCI_(xs, permuted_xs, alpha=5, tail = 'two'):
    xs, permuted_xs = xs.round(5), permuted_xs.round(5)
    
    b1,b2 = permuted_xs.min(), permuted_xs.max()
    if tail=='two':
        b1,b2 = stats.scoreatpercentile(permuted_xs,alpha/2), stats.scoreatpercentile(permuted_xs,100-alpha/2)
        #b1,b2 = np.percentile(xs,alpha/2), np.percentile(xs,100-alpha/2)
        pvalue = len(xs[np.logical_and(xs<b2, xs>b1)])/len(xs)
    elif tail=='greater':
        b1 = stats.scoreatpercentile(permuted_xs,alpha/1)
        #b1 = np.percentile(xs,alpha/1)
        pvalue = len(xs[np.logical_and(True, xs>b1)])/len(xs)
    elif tail=='smaller':
        b2 = stats.scoreatpercentile(permuted_xs,100-alpha/1)
        #b2 = np.percentile(xs,100-alpha/1)
        pvalue = len(xs[np.logical_and(xs<b2, True)])/len(xs)
    
    #pvalue = len(permuted_xs[np.logical_and(permuted_xs<b2, permuted_xs>b1)])/len(permuted_xs)
    return pvalue


def permutation_pCI(xs, permuted_xs, alpha=5, tail = 'two', base='h0'):
    xs, permuted_xs = xs.round(5), permuted_xs.round(5)
    
    if base == 'h0':
        b1,b2 = xs.min(), xs.max()
        if tail=='two':
            b1,b2 = stats.scoreatpercentile(xs,alpha/2), stats.scoreatpercentile(xs,100-alpha/2)
            pvalue = len(permuted_xs[np.logical_and(permuted_xs<b2, permuted_xs>b1)])/len(permuted_xs)
            
        elif tail=='greater':
            b1 = stats.scoreatpercentile(xs,alpha/1)
            pvalue = len(permuted_xs[np.logical_and(True, permuted_xs>b1)])/len(permuted_xs)
            
        elif tail=='smaller':
            b2 = stats.scoreatpercentile(xs,100-alpha/1)
            pvalue = len(permuted_xs[np.logical_and(permuted_xs<b2, True)])/len(permuted_xs)
    
    elif base == 'h1':
        # if h1, should always use tail='two'
        b1,b2 = permuted_xs.min(), permuted_xs.max()
        if tail=='two':
            b1,b2 = stats.scoreatpercentile(permuted_xs,alpha/2), stats.scoreatpercentile(permuted_xs,100-alpha/2)
            pvalue = len(xs[np.logical_and(xs<b2, xs>b1)])/len(xs)
            
        elif tail=='greater':
            b1 = stats.scoreatpercentile(permuted_xs,alpha/1)
            pvalue = len(xs[np.logical_and(xs<b2, xs>b1)])/len(xs)
            
        elif tail=='smaller':
            b2 = stats.scoreatpercentile(permuted_xs,100-alpha/1)
            pvalue = len(xs[np.logical_and(xs<b2, xs>b1)])/len(xs)
            
    
    #pvalue = len(permuted_xs[np.logical_and(permuted_xs<b2, permuted_xs>b1)])/len(permuted_xs)
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
    
    #pvalue = len(permuted_xs[np.logical_and(permuted_xs<b2, permuted_xs>b1)])/len(permuted_xs)
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
        

    #pvalue = len(permuted_xs[np.logical_and(permuted_xs<b2, permuted_xs>b1)])/len(permuted_xs)
    return pvalue

# In[] selectivity test

def sigtest(parray, threshold = 0.05, consec=2):
    # Initialize a counter variable
    consecutive_count = 0
    # Loop through the array
    for i in range(len(parray)-1):
        if parray[i] < threshold and parray[i+1] < threshold:
            consecutive_count += 1
            if consecutive_count >= consec:
                #print("There are at least 2 consecutive elements that meet the criterion.")
                sigtest = 1
                break
        else:
            consecutive_count = 0
    
    if consecutive_count < consec:
        #print("There are no 2 consecutive elements that meet the criterion.")
        sigtest = 0
    
    return sigtest


def bslTTest_1samp(arrayFreq_E, bslV):
    t,p = stats.ttest_1samp(arrayFreq_E, bslV, alternative = 'two-sided')
    
    return t,p

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

# In[]
def factorial_anova(data, factor1, factor2, dependent):
    # Get the unique levels of each factor
    levels_factor1 = data[factor1].unique()
    levels_factor2 = data[factor2].unique()
    
    # Grand mean
    grand_mean = data[dependent].mean()
    
    # Initialize sum of squares
    ss_total = 0
    ss_factor1 = 0
    ss_factor2 = 0
    ss_interaction = 0
    ss_within = 0
    
    # Number of observations
    n = len(data)
    
    # Total sum of squares
    ss_total = np.sum((data[dependent] - grand_mean) ** 2)
    
    # Iterate over levels of factor1 and factor2 to calculate sums of squares
    for level1 in levels_factor1:
        for level2 in levels_factor2:
            # Subset data for each combination of factor levels
            subset = data[(data[factor1] == level1) & (data[factor2] == level2)]
            subset_mean = subset[dependent].mean()
            n_subset = len(subset)
            
            # Factor1 effect
            level1_mean = data[data[factor1] == level1][dependent].mean()
            ss_factor1 += n_subset * (level1_mean - grand_mean) ** 2
            
            # Factor2 effect
            level2_mean = data[data[factor2] == level2][dependent].mean()
            ss_factor2 += n_subset * (level2_mean - grand_mean) ** 2
            
            # Interaction effect
            ss_interaction += n_subset * (subset_mean - level1_mean - level2_mean + grand_mean) ** 2
            
            # Within-group sum of squares
            ss_within += np.sum((subset[dependent] - subset_mean) ** 2)
    
    # Degrees of freedom
    df_factor1 = len(levels_factor1) - 1
    df_factor2 = len(levels_factor2) - 1
    df_interaction = df_factor1 * df_factor2
    df_within = n - len(levels_factor1) * len(levels_factor2)
    
    # Mean squares
    ms_factor1 = ss_factor1 / df_factor1
    ms_factor2 = ss_factor2 / df_factor2
    ms_interaction = ss_interaction / df_interaction
    ms_within = ss_within / df_within
    
    # F-statistics
    f_factor1 = ms_factor1 / ms_within
    f_factor2 = ms_factor2 / ms_within
    f_interaction = ms_interaction / ms_within
    
    # p-values
    p_factor1 = stats.f.sf(f_factor1, df_factor1, df_within)
    p_factor2 = stats.f.sf(f_factor2, df_factor2, df_within)
    p_interaction = stats.f.sf(f_interaction, df_interaction, df_within)
    
    # Results summary
    results = pd.DataFrame({
        'Source': [factor1, factor2, f"{factor1} * {factor2}", "Within"],
        'SS': [ss_factor1, ss_factor2, ss_interaction, ss_within],
        'DF': [df_factor1, df_factor2, df_interaction, df_within],
        'MS': [ms_factor1, ms_factor2, ms_interaction, ms_within],
        'F': [f_factor1, f_factor2, f_interaction, np.nan],
        'p': [p_factor1, p_factor2, p_interaction, np.nan]
    })
    
    return results


# In[]
def crossover_lineInterp(tRange, series1, series2, tBoundaries = (None, None), minDur=100):
    """
    Estimates the crossover time points between two time series based on linear interpolation method.
    
    Args:
        time (numpy array): The time points of the series.
        series1 (numpy array): The first time series data.
        series2 (numpy array): The second time series data.
        tBoundaries (tuple): Find crossover points only within this time range.
    Returns:
        crossover_times (np.array): List of estimated time points where the two series cross over.
        crossover_values (np.array): List of estimated time points where the two series cross over.
    """
    tLow, tHigh = tBoundaries
    tLow = tLow if tLow!=None else tRange[0]
    tHigh = tHigh if tHigh!=None else tRange[-1] # if not specify boundaries, just use the limits
    
    # Find where the two series have opposite signs (i.e., crossovers)
    diff = series1 - series2
    sign_diff = np.sign(diff)
    crossover_indices = np.where(np.diff(sign_diff))[0]
    
    # Estimate exact crossover points by linear interpolation
    crossover_times = []
    crossover_values = []

    for idx in crossover_indices:

        t1, t2 = tRange[idx], tRange[idx + 1]
        s1_diff, s2_diff = diff[idx], diff[idx + 1]
        
        # Linear interpolation formula to estimate crossover time
        t_crossover = t1 - s1_diff * (t2 - t1) / (s2_diff - s1_diff)

        # Interpolate to find the value at the crossover point
        v1 = series1[idx]
        v2 = series1[idx + 1]
        v_crossover = v1 + (v2 - v1) * (t_crossover - t1) / (t2 - t1)
        
        if tLow <= t_crossover <= tHigh:
            crossover_times.append(t_crossover)
            crossover_values.append(v_crossover)
        else:
            pass
    
    # when multiple crossovers, take the first valid point
    # valid defined as if no further crossover in the next minDur ms

    crossover_times_valid = crossover_times.copy()
    #crossover_values_valid = crossover_values.copy()

    if len(crossover_times)>1:
        for ntx, tx in enumerate(crossover_times):
            if ntx < (len(crossover_times)-1):
                if tx <= crossover_times[ntx+1] <= tx+minDur:
                    crossover_times_valid.remove(tx)

                    #vx = crossover_values[crossover_times.index(tx)]
                    #crossover_values_valid.remove(vx)
                else:
                    pass#crossover_times_valid.append(crossover_times[ntx])
    else:
        crossover_times_valid = crossover_times.copy()
        #crossover_values_valid = crossover_values.copy()

    crossover_t = np.array(crossover_times_valid).min()
    crossover_v = np.array(crossover_values[crossover_times.index(crossover_t)])

    return crossover_t.round(0), crossover_v.round(5)

def valid_timepoint(times, dt = 50, minDur=100, tBoundaries = (None, None), earliest=True, returnall=False, proportion = 1.0):
    
    tLow, tHigh = tBoundaries
    tLow = tLow if tLow!=None else 0
    tHigh = tHigh if tHigh!=None else 2800 # if not specify boundaries, just use the limits
    
    times = times[(tLow<=times)&(times<tHigh)]
    times = times.astype(int)
    times_valid = times.tolist().copy()
    #times_valid = times.copy()

    if len(times)>(minDur//dt):
        for ntx, tx in enumerate(times):
            if ntx < (len(times)-(minDur//dt)):
                
                tBools = [] #
                for i in range(minDur//dt):
                    tBools += [((tx+(i*dt)).astype(int) in times)*1] #
                    
                    ##if (tx+(i*dt)).astype(int) in times: # in times
                    ##    pass
                        # if after minDur still significant, keep, else drop
                    ##else:
                    ##    times_valid.remove(tx)
                    ##    break #times_valid.append(times[ntx])
                
                tBools = np.array(tBools)
                if np.sum(tBools)>=(proportion*(minDur//dt)):
                    pass
                else:
                    times_valid.remove(tx)
    else:
        times_valid = [np.nan,]
    
    times_valid = np.array(times_valid)
    times_valid_inBoundary = times_valid#[np.where((tLow<times_valid) & (times_valid<tHigh))[0]]
    
    if returnall:
        tReturn = times_valid_inBoundary.round(0) if len(times_valid_inBoundary)>0 else np.array([])
    else:
        if earliest:
            tReturn = times_valid_inBoundary.min().round(0) if len(times_valid_inBoundary)>0 else np.nan
        else:
            tReturn = times_valid_inBoundary.max().round(0) if len(times_valid_inBoundary)>0 else np.nan
    
    return tReturn

# %%
def shuffle_array(array, axis=0):
    rng = np.random.default_rng()
    array_ = rng.permuted(array, axis=axis)
    return array_

# %%
def compute_t_stat(pop1,pop2):

    num1 = pop1.shape[0]
    num2 = pop2.shape[0];
    var1 = np.var(pop1, ddof=1)
    var2 = np.var(pop2, ddof=1)

    # The formula for t-stat when population variances differ.
    t_stat = (np.mean(pop1) - np.mean(pop2)) / np.sqrt(var1/num1 + var2/num2)

    # ADDED: The Welch-Satterthwaite degrees of freedom.
    df = ((var1/num1 + var2/num2)**(2.0))/((var1/num1)**(2.0)/(num1-1) + (var2/num2)**(2.0)/(num2-1)) 

    one_tailed_p_value = 1.0 - stats.t.cdf(t_stat,df)
    two_tailed_p_value = 1.0 - (stats.t.cdf(np.abs(t_stat),df) - stats.t.cdf(-np.abs(t_stat),df) )    


    # Computing with SciPy's built-ins
    # My results don't match theirs.
    t_ind, p_ind = stats.ttest_ind(pop1, pop2)

    return t_stat, one_tailed_p_value, two_tailed_p_value, df, t_ind, p_ind
