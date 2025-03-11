# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 22:01:51 2024

@author: aka2333
"""


import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
# In[]

def mask_triangle(data, ul='u', diag=0):
    if ul == 'u':
        offDiag = 0 if diag == 0 else 1
        mask = np.triu(np.ones_like(data), k=offDiag)
    elif ul == 'l':
        offDiag = 0 if diag == 0 else -1
        mask = np.tril(np.ones_like(data), k=offDiag)
        
    return np.ma.array(data, mask=mask)

# In[]
def plot_pdf(distributions):
    
    plt.figure()
    for d in distributions:
        kde = gaussian_kde(d)
        dist_space = np.linspace(d.min(), d.max(), 100)
        
        plt.plot(dist_space, kde(dist_space))
    
    plt.show()
    return

# In[]
# Custom function to create a mask for solid lines based on significance values
def significance_line_segs(pvalues, threshold=0.05):
    segments = []
    start_idx = None
    for i, sig in enumerate(pvalues):
        if sig <= threshold:
            if start_idx is None:
                start_idx = i
        else:
            if start_idx is not None:
                segments.append((start_idx, i))
                start_idx = None
    if start_idx is not None:
        segments.append((start_idx, len(pvalues)))
    return segments

# In[]
def set_box_color(bp, color):
    plt.setp(bp['boxes'], color=color)
    plt.setp(bp['whiskers'], color=color)
    plt.setp(bp['caps'], color=color)
    plt.setp(bp['medians'], color=color)
    plt.setp(bp['fliers'], color=color)
    
# In[]
def sig_marker(p1,ns_note=False,pmax=0.05):
    ns_ = 'n.s.' if ns_note else ''
    
    if p1 < 0.001:
        sig1 = '***' if p1<pmax else ns_
    elif p1 < 0.01:
        sig1 = '**' if p1<pmax else ns_ 
    elif p1 < 0.05:
        sig1 = '*' if p1<pmax else ns_
    elif p1 < 0.1:
        sig1 = '+' if p1<pmax else ns_
    else:
        sig1 = ns_
    return sig1