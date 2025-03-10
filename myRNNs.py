# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 02:30:50 2024

@author: aka2333
"""
# In[]
import numpy as np
import scipy
from scipy import stats
from scipy.stats import vonmises # circular distribution
from scipy import ndimage

import pandas as pd

# basic plot functions
import matplotlib as mpl
import matplotlib.pyplot as plt
import re, seaborn as sns

# turn off warning messages
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# system paths
import os
import sys

import time # timer

from itertools import permutations, combinations, product # itertools


import matplotlib.ticker as ticker
from matplotlib.pyplot import MultipleLocator #从pyplot导入MultipleLocator类，这个类用于设置刻度间隔
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec  # 导入专门用于网格分割及子图绘制的扩展包Gridspec

# In[] import pyTorch
import torch
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # define device, may need to change with Mac OS

# In[] 
import f_simulation
import f_trainRNN
import f_stats
# In[]
class decayRNN(nn.Module):
    def __init__(self,input_size, hidden_size, output_size, dt = 10, tau = 100, external = 0,
                 hidden_noise = 0.1, ceiling = None, F_hidden = 'relu', F_out = 'softmax', useLinear_hidden = True,
                 init_in = None, init_hidden = 'orthogonal_', init_out = None, device = device, seed = None, h2hInitScale = 1):
        super(decayRNN, self).__init__() #, tau = 100
        # if change class name, remember to change the super as well
        
        self.seed = seed
        if self.seed != None:
            torch.manual_seed(self.seed)
        
        ####################
        # Basic parameters #
        ####################
        
        self.device = device
        
        self.input_size = input_size
        self.output_size = output_size

        self.hidden_size = hidden_size

        self.dt = dt # time step
        self.tau = tau # intrinsic time scale, control delay rate.
        self.alpha = (1.0 * self.dt) / self.tau #Inferred Parameters:**alpha** (*float*) -- The number of unit time constants per simulation timestep.

        self.hidden_noise = hidden_noise
        self.external = external # external source of modulation, e.g., top-down modulation from higher level regions... default set to 0
        
        
        ##############################################
        # weights + bias between the 3 linear layers #
        ##############################################
        
        self.i2h = nn.Linear(self.input_size, self.hidden_size, bias=False).to(self.device)
        
        if init_in == 'xavier_normal_':    
            self.i2h.weight = nn.Parameter(nn.init.xavier_normal_(self.i2h.weight.data))
        elif init_in == 'orthogonal_':    
            self.i2h.weight = nn.Parameter(nn.init.orthogonal_(self.i2h.weight.data))
        elif init_in == 'kaiming_normal_':    
            self.i2h.weight = nn.Parameter(nn.init.kaiming_normal_(self.i2h.weight.data))
        elif init_in == None:    
            self.i2h.weight = nn.Parameter(self.i2h.weight.data)
        #self.i2h.weight = nn.Parameter(nn.init.kaiming_normal_(self.i2h.weight.data))
        #self.i2h.bias = nn.Parameter(nn.init.xavier_normal_(self.i2h.bias.size))

        #self.Win = nn.Parameter(nn.init.xavier_normal_(torch.empty((self.input_size, self.hidden_size)))) # input-hidden weight matrix + bias, trainable
        #self.Bin = nn.Parameter(nn.init.xavier_normal_(torch.empty((self.hidden_size,1))).squeeze())
        
        
        #########################################################
        # hidden layer recurrent weight matrix, orthogonal init #
        #########################################################
        
        self.useLinear_hidden = useLinear_hidden
        self.h2h = nn.Linear(self.hidden_size, self.hidden_size, bias=False).to(self.device)
        
        # initialization
        if init_hidden == 'xavier_normal_':
            self.Wrec = nn.init.xavier_normal_(self.h2h.weight.data.clone())
        elif init_hidden == 'orthogonal_':    
            self.Wrec = nn.init.orthogonal_(self.h2h.weight.data.clone())
        elif init_hidden == 'kaiming_normal_':    
            self.Wrec = nn.init.kaiming_normal_(self.h2h.weight.data.clone())
            
        if useLinear_hidden:
            self.h2h.weight = nn.Parameter(self.Wrec * h2hInitScale)
        else:
            self.h2h.weight.requires_grad = False
            self.Wrec = nn.Parameter(self.Wrec * h2hInitScale)
        
        #self.h2h.weight = nn.Parameter(nn.init.xavier_normal_(self.h2h.weight.data))
        #self.h2h.bias = nn.Parameter(nn.init.orthogonal_(self.h2h.bias))
        #self.Wrec = nn.Parameter(nn.init.orthogonal_(torch.empty((self.hidden_size, self.hidden_size))))
        
        
        #################################################
        # hidden-output weight matrix + bias, trainable #
        #################################################
        
        self.h2o = nn.Linear(self.hidden_size, self.output_size, bias=False).to(self.device)
        
        if init_out == 'xavier_normal_':
            self.h2o.weight = nn.Parameter(nn.init.xavier_normal_(self.h2o.weight.data))
        elif init_out == 'orthogonal_':    
            self.h2o.weight = nn.Parameter(nn.init.orthogonal_(self.h2o.weight.data))
        elif init_out == 'kaiming_normal_':    
            self.h2o.weight = nn.Parameter(nn.init.kaiming_normal_(self.h2o.weight.data))
        elif init_out == None:    
            self.h2o.weight = nn.Parameter(self.h2o.weight.data)
        
        #self.h2o.weight = nn.Parameter(nn.init.kaiming_normal_(self.h2o.weight.data))
        #self.h2o.bias = nn.Parameter(nn.init.xavier_normal_(self.h2o.bias))
        #self.Wout = nn.Parameter(nn.init.xavier_normal_(torch.empty((self.hidden_size, self.output_size))))
        #self.Bout = nn.Parameter(nn.init.xavier_normal_(torch.empty((self.output_size,1))).squeeze())
        
        
        ###########################
        # non-linearity & ceiling #
        ###########################
        
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.softplus = nn.Softplus()
        self.mish = nn.Mish()

        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim = 1)

        # hidden layer nonlinearity
        if F_hidden == 'relu':
            self.F_hidden = self.relu
        elif F_hidden == 'softplus':
            self.F_hidden = self.softplus
        elif F_hidden == 'mish':
            self.F_hidden = self.mish
        elif F_hidden == 'tanh':
            self.F_hidden = self.tanh

        # output layer nonlinearity
        if F_out == 'softmax':
            self.F_out = self.softmax
        elif F_out == 'sigmoid':
            self.F_out = self.sigmoid
        elif F_out == 'tanh':
            self.F_out = self.tanh

        #self.rnn = nn.RNN(input_size=input_size, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True)
        self.ceiling = ceiling

    def forward(self, x):

        # Set initial hidden states
        h0 = torch.zeros(x.size(0), self.hidden_size).to(self.device)

        # Forward propagate RNN
        hiddens = [h0] # initial states
        outs = []
        for t in range(x.size()[1]): # go through all timepoints
            ht_1 = hiddens[-1] # hidden states at the previous timepoint

            # your function here
            noise = torch.normal(mean=0, std=self.hidden_noise, size = ht_1.size()).to(self.device)

            # whether to use linear layer or directly matrix multiply
            if self.useLinear_hidden:
                ht = (self.i2h(x[:,t,:]) + self.h2h(ht_1) + noise)
            else:
                ht = (self.i2h(x[:,t,:]) + ht_1 @ self.Wrec.T + noise)
                
            #ht = ((torch.matmul(x[:,t,:], self.Win) + self.Bin) + torch.matmul(ht_1, self.Wrec) + noise)
            ht = self.F_hidden(ht)*self.alpha + (1-self.alpha)*ht_1 + self.external #

            # ceiling
            ht = ht if self.ceiling == None else torch.clamp(ht, min=self.ceiling[0], max = self.ceiling[1])

            hiddens += [ht] # store hidden state at each timepoint

            # output layer
            #out = (torch.matmul(ht, self.Wout) + self.Bout)
            out = self.h2o(ht)
            out = self.F_out(out)
            outs += [out] # store output at each timepoint

        # stack hidden states together
        hiddens = torch.stack(hiddens[1:]).swapaxes(0,1)
        outs = torch.stack(outs).swapaxes(0,1)

        return hiddens, outs

