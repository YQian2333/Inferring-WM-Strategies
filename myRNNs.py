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


# In[]
class mixed_decayRNN(nn.Module):
    def __init__(self,input_size, bump_size, random_size, output_size, dt = 10, tau = 100, external = 0,
                 hidden_noise = 0.1, ceiling = None, F_hidden = 'relu', F_out = 'softmax', 
                 rec_sigma = 2, rec_scale_up = None, rec_scale_low = None, useLinear_hidden = True,
                 init_in = 'xavier_normal_', init_hidden = 'kaiming_normal_', init_out = 'xavier_normal_', device = device):
        super(mixed_decayRNN, self).__init__() #, tau = 100
        # if change class name, remember to change the super as well
        
        
        ####################
        # Basic parameters #
        ####################
        
        self.device = device
        
        self.input_size = input_size
        self.output_size = output_size
        
        self.bump_size, self.random_size = bump_size, random_size
        
        self.hidden_size = (self.bump_size + self.random_size)

        self.dt = dt # time step
        self.tau = tau # intrinsic time scale, control delay rate.
        self.alpha = (1.0 * self.dt) / self.tau #Inferred Parameters:**alpha** (*float*) -- The number of unit time constants per simulation timestep.

        self.hidden_noise = hidden_noise
        self.external = external # external source of modulation, e.g., top-down modulation from higher level regions... default set to 0
        
        
        ########################################
        # Bump cells initialization parameters #
        ########################################
        
        # recurrent layer weight combined by bump and random
        self.rec_sigma = rec_sigma
        self.gau_bump = f_simulation.gaussian_dis(np.arange(self.bump_size), u = int(np.median(np.arange(self.bump_size))), sigma = self.rec_sigma)
        self.rec_scale_up = self.gau_bump.max()if (rec_scale_up == None) else rec_scale_up 
        self.rec_scale_low = self.gau_bump.min()if (rec_scale_low == None) else rec_scale_low 
        
        self.gau_bump = f_simulation.scale_distribution(self.gau_bump, upper=self.rec_scale_up, lower=self.rec_scale_low) # if self.rec_scale else self.gau_bump
        
        self.WrecBB = torch.tensor(f_simulation.bump_Wrec(self.gau_bump, self.bump_size, specificity=1), dtype=torch.float32).to(self.device)
        #self.WrecRR = torch.nn.init.kaiming_normal_(torch.zeros(self.random_size, self.random_size)).to(self.device)
        
        
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
        
        #self.i2h.weight = nn.Parameter(nn.init.kaiming_normal_(self.i2h.weight.data))
        #self.i2h.bias = nn.Parameter(nn.init.xavier_normal_(self.i2h.bias.size))

        #self.Win = nn.Parameter(nn.init.xavier_normal_(torch.empty((self.input_size, self.hidden_size)))) # input-hidden weight matrix + bias, trainable
        #self.Bin = nn.Parameter(nn.init.xavier_normal_(torch.empty((self.hidden_size,1))).squeeze())
        
        
        #########################################################
        # hidden layer recurrent weight matrix, orthogonal init #
        #########################################################
        
        self.useLinear_hidden = useLinear_hidden
        self.h2h = nn.Linear(self.hidden_size, self.hidden_size, bias=False).to(self.device)
        
        if init_hidden == 'xavier_normal_':
            self.Wrec = nn.init.xavier_normal_(self.h2h.weight.data.clone())
        elif init_hidden == 'orthogonal_':    
            self.Wrec = nn.init.orthogonal_(self.h2h.weight.data.clone())
        elif init_hidden == 'kaiming_normal_':    
            self.Wrec = nn.init.kaiming_normal_(self.h2h.weight.data.clone())
        
        self.Wrec[0:self.bump_size, 0:self.bump_size] = self.WrecBB
        self.h2h.weight = nn.Parameter(self.Wrec)
        
        self.Wrec = nn.Parameter(self.Wrec)
        
        #self.h2h.bias = nn.Parameter(nn.init.orthogonal_(self.h2h.bias))
        #self.Wrec = nn.Parameter(nn.init.orthogonal_(torch.empty((self.hidden_size, self.hidden_size))))
        #self.Wrec = nn.init.kaiming_normal_(self.h2h.weight.data.clone())
        
        
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
        
        # ceiling of hidden neuron activity level
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

# In[] fixed input + trainable recurrent weights with bump attractor initializations
class mixed_decayRNN_fixin(nn.Module):
    def __init__(self, input_size, bump_size, random_size, output_size, dt = 10, tau = 100, external = 0, gocue = True,
                 hidden_noise = 0.1, ceiling = None, locs = (0,1,2,3), ttypes = (1,2),
                 in_sigma = 1, in_scale_up = None, in_scale_low = None, train_in = False, useLinear_in = True,
                 rec_sigma = 2, rec_scale_up = None, rec_scale_low = None, train_rec = True, useLinear_hidden = True, F_hidden = 'relu',
                 init_in = 'xavier_normal_', init_hidden = 'kaiming_normal_', init_out = 'xavier_normal_', F_out = 'softmax', device = device):
        super(mixed_decayRNN_fixin, self).__init__() #, tau = 100
        # if change class name, remember to change the super as well
        
        
        ####################
        # Basic parameters #
        ####################
        
        self.device = device
        self.gocue = gocue
        
        self.input_size = input_size
        self.cue_size = 1 if self.gocue else 0
        
        self.output_size = output_size
        
        self.bump_size, self.random_size = bump_size, random_size
        
        self.hidden_size = (self.bump_size + self.random_size)

        self.dt = dt # time step
        self.tau = tau # intrinsic time scale, control delay rate.
        self.alpha = (1.0 * self.dt) / self.tau #Inferred Parameters:**alpha** (*float*) -- The number of unit time constants per simulation timestep.
        
        self.locs, self.ttypes = locs, ttypes
        
        self.n_bumpPops = int((self.input_size - self.cue_size) / len(self.locs))
        self.bump_size_ = int(self.bump_size/self.n_bumpPops)
        
        self.hidden_noise = hidden_noise
        self.external = external # external source of modulation, e.g., top-down modulation from higher level regions... default set to 0
        
        
        ########################################
        # Bump cells initialization parameters #
        ########################################
        
        # input layer weight combined by bump and random
        self.in_sigma = in_sigma
        self.gau_in = f_simulation.gaussian_dis(np.arange(self.bump_size_), u = int(np.median(np.arange(self.bump_size_))), sigma = self.in_sigma)
        
        self.in_scale_up = self.gau_in.max() if (in_scale_up == None) else in_scale_up 
        self.in_scale_low = self.gau_in.min() if (in_scale_low == None) else in_scale_low 
        
        self.gau_in = f_simulation.scale_distribution(self.gau_in, upper=self.in_scale_up, lower=self.in_scale_low)
        
        #self.WinB = torch.tensor(f_simulation.bump_Win(self.gau_in, (self.input_size - self.cue_size), self.bump_size, specificity=1), dtype=torch.float32).to(self.device)
        self.WinB = torch.tensor(f_simulation.bump_Win(self.gau_in, len(self.locs), self.bump_size_, specificity=1), dtype=torch.float32).to(self.device)
        
        # recurrent layer weight combined by bump and random
        self.rec_sigma = rec_sigma
        self.gau_rec = f_simulation.gaussian_dis(np.arange(self.bump_size_), u = int(np.median(np.arange(self.bump_size_))), sigma = self.rec_sigma)
        
        self.rec_scale_up = self.gau_rec.max() if (rec_scale_up == None) else rec_scale_up 
        self.rec_scale_low = self.gau_rec.min() if (rec_scale_low == None) else rec_scale_low 
        
        self.gau_rec = f_simulation.scale_distribution(self.gau_rec, upper=self.rec_scale_up, lower=self.rec_scale_low)
        
        self.WrecBB = torch.tensor(f_simulation.bump_Wrec(self.gau_rec, self.bump_size_, specificity=1), dtype=torch.float32).to(self.device)
        
        
        ##############################################
        # weights + bias between the 3 linear layers #
        ##############################################
        
        self.useLinear_in = useLinear_in
        self.train_in = train_in
        
        self.i2h = nn.Linear(self.input_size, self.hidden_size, bias=False).to(self.device)
        
        if init_in == 'xavier_normal_':    
            self.Win = nn.init.xavier_normal_(self.i2h.weight.data)
        elif init_in == 'orthogonal_':    
            self.Win = nn.init.orthogonal_(self.i2h.weight.data)
        elif init_in == 'kaiming_normal_':    
            self.Win = nn.init.kaiming_normal_(self.i2h.weight.data)
        
        # 
        for npop in range(self.n_bumpPops):
            self.Win[self.bump_size_*(npop):self.bump_size_*(npop+1), len(self.locs)*(npop):len(self.locs)*(npop+1)] = self.WinB
        
            
        self.i2h.weight = nn.Parameter(self.Win)
        
        # only optimize input weights if need
        if self.train_in:
            self.Win = nn.Parameter(self.Win)
        
        
        
        #########################################################
        # hidden layer recurrent weight matrix, orthogonal init #
        #########################################################
        
        self.useLinear_hidden = useLinear_hidden
        self.train_rec = train_rec
        
        self.h2h = nn.Linear(self.hidden_size, self.hidden_size, bias=False).to(self.device)
        
        if init_hidden == 'xavier_normal_':
            self.Wrec = nn.init.xavier_normal_(self.h2h.weight.data.clone())
        elif init_hidden == 'orthogonal_':    
            self.Wrec = nn.init.orthogonal_(self.h2h.weight.data.clone())
        elif init_hidden == 'kaiming_normal_':    
            self.Wrec = nn.init.kaiming_normal_(self.h2h.weight.data.clone())
        
        # 
        for npop in range(self.n_bumpPops):
            self.Wrec[self.bump_size_*(npop):self.bump_size_*(npop+1), self.bump_size_*(npop):self.bump_size_*(npop+1)] = self.WrecBB
            
        self.h2h.weight = nn.Parameter(self.Wrec)
        
        if self.train_rec:
            self.Wrec = nn.Parameter(self.Wrec)
        
        #self.h2h.bias = nn.Parameter(nn.init.orthogonal_(self.h2h.bias))
        #self.Wrec = nn.Parameter(nn.init.orthogonal_(torch.empty((self.hidden_size, self.hidden_size))))
        #self.Wrec = nn.init.kaiming_normal_(self.h2h.weight.data.clone())
        
        
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
        
        # ceiling of hidden neuron activity level
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
            if self.useLinear_in:
                it = self.i2h(x[:,t,:])
            else:
                it = x[:,t,:] @ self.Win.T
            
            if self.useLinear_hidden:
                ht = (it + self.h2h(ht_1) + noise)
            else:
                ht = (it + ht_1 @ self.Wrec.T + noise)
                
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
# In[] fixed input + trainable recurrent weights with bump attractor initializations
class mixed_decayRNN_fix_bump(nn.Module):
    def __init__(self, input_size, bump_size, random_size, output_size, dt = 10, tau = 100, external = 0, gocue = True,
                 hidden_noise = 0.1, ceiling = None, locs = (0,1,2,3), ttypes = (1,2), n_bumpPops = 1,
                 rec_sigma = 2, rec_scale_up = None, rec_scale_low = None, train_rec = True, train_bump = False, F_hidden = 'relu',
                 init_in = 'xavier_normal_', init_hidden = 'orthogonal', init_out = 'xavier_normal_', F_out = 'softmax', device = device):
        super(mixed_decayRNN_fix_bump, self).__init__() #, tau = 100
        # if change class name, remember to change the super as well
        
        
        ####################
        # Basic parameters #
        ####################
        
        self.device = device
        self.gocue = gocue
        
        self.input_size = input_size
        self.cue_size = 1 if self.gocue else 0
        
        self.output_size = output_size
        
        self.bump_size, self.random_size = bump_size, random_size
        
        self.hidden_size = (self.bump_size + self.random_size)

        self.dt = dt # time step
        self.tau = tau # intrinsic time scale, control delay rate.
        self.alpha = (1.0 * self.dt) / self.tau #Inferred Parameters:**alpha** (*float*) -- The number of unit time constants per simulation timestep.
        
        self.locs, self.ttypes = locs, ttypes
        
        self.n_bumpPops = n_bumpPops#int((self.input_size - self.cue_size) / len(self.locs))
        self.bump_size_ = int(self.bump_size/self.n_bumpPops)
        
        self.hidden_noise = hidden_noise
        self.external = external # external source of modulation, e.g., top-down modulation from higher level regions... default set to 0
        
        
        ########################################
        # Bump cells initialization parameters #
        ########################################
        
        
        # recurrent layer weight combined by bump and random
        self.rec_sigma = rec_sigma
        self.gau_rec = f_simulation.gaussian_dis(np.arange(self.bump_size_), u = int(np.median(np.arange(self.bump_size_))), sigma = self.rec_sigma)
        
        self.rec_scale_up = self.gau_rec.max() if (rec_scale_up == None) else rec_scale_up 
        self.rec_scale_low = self.gau_rec.min() if (rec_scale_low == None) else rec_scale_low 
        
        self.gau_rec = f_simulation.scale_distribution(self.gau_rec, upper=self.rec_scale_up, lower=self.rec_scale_low)
        
        self.WrecBB = torch.tensor(f_simulation.bump_Wrec(self.gau_rec, self.bump_size_, specificity=1), dtype=torch.float32).to(self.device)
        
        
        ##############################################
        # weights + bias between the 3 linear layers #
        ##############################################
        
        
        self.i2h = nn.Linear(self.input_size, self.hidden_size, bias=False).to(self.device)
        
        if init_out == 'xavier_normal_':
            self.i2h.weight = nn.Parameter(nn.init.xavier_normal_(self.i2h.weight.data))
        elif init_out == 'orthogonal_':    
            self.i2h.weight = nn.Parameter(nn.init.orthogonal_(self.i2h.weight.data))
        elif init_out == 'kaiming_normal_':    
            self.i2h.weight = nn.Parameter(nn.init.kaiming_normal_(self.i2h.weight.data))
        
        
        
        #########################################################
        # hidden layer recurrent weight matrix, orthogonal init #
        #########################################################
        #self.h2h = nn.Linear(self.hidden_size, self.hidden_size, bias=False).to(self.device)
        
        #if init_hidden == 'xavier_normal_':
        #    self.Wrec = nn.init.xavier_normal_(self.h2h.weight.data.clone())
        #elif init_hidden == 'orthogonal_':    
        #    self.Wrec = nn.init.orthogonal_(self.h2h.weight.data.clone())
        #elif init_hidden == 'kaiming_normal_':    
        #    self.Wrec = nn.init.kaiming_normal_(self.h2h.weight.data.clone())
        
        # 
        #for npop in range(self.n_bumpPops):
        #    self.Wrec[self.bump_size_*(npop):self.bump_size_*(npop+1), self.bump_size_*(npop):self.bump_size_*(npop+1)] = self.WrecBB
        #self.h2h.weight = nn.Parameter(self.Wrec)    
        
        self.train_bump = train_bump
        self.train_rec = train_rec
        
        if self.random_size>0:
            self.WrecRR = torch.empty((self.random_size, self.random_size)).to(self.device) # random
            self.Wrec1R = torch.empty((self.bump_size_, self.random_size)).to(self.device)
            self.WrecR1 = torch.empty((self.random_size, self.bump_size_)).to(self.device)
            
            if init_hidden == 'xavier_normal_':    
                self.WrecRR = nn.init.xavier_normal_(self.WrecRR)
                self.Wrec1R = nn.init.xavier_normal_(self.Wrec1R)
                self.WrecR1 = nn.init.xavier_normal_(self.WrecR1)
                
            elif init_hidden == 'orthogonal_':    
                self.WrecRR = nn.init.orthogonal_(self.WrecRR)
                self.Wrec1R = nn.init.orthogonal_(self.Wrec1R)
                self.WrecR1 = nn.init.orthogonal_(self.WrecR1)
                
            elif init_hidden == 'kaiming_normal_':    
                self.WrecRR = nn.init.kaiming_normal_(self.WrecRR)
                self.Wrec1R = nn.init.kaiming_normal_(self.Wrec1R)
                self.WrecR1 = nn.init.kaiming_normal_(self.WrecR1)
            
        
        if self.train_bump:
            self.WrecBB = nn.Parameter(self.WrecBB)
        
        if self.train_rec:
            if self.random_size>0:
                self.WrecRR = nn.Parameter(self.WrecRR)
                self.Wrec1R = nn.Parameter(self.Wrec1R)
                self.WrecR1 = nn.Parameter(self.WrecR1)
        
        #self.h2h.bias = nn.Parameter(nn.init.orthogonal_(self.h2h.bias))
        #self.Wrec = nn.Parameter(nn.init.orthogonal_(torch.empty((self.hidden_size, self.hidden_size))))
        #self.Wrec = nn.init.kaiming_normal_(self.h2h.weight.data.clone())
        
        
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
        
        # ceiling of hidden neuron activity level
        self.ceiling = ceiling
    
    
    def generate_Wrec(self):
        
        #################################
        # build Recurrent Weight Matrix #
        #################################
        
        Wrec = self.WrecBB #torch.vstack((torch.hstack((self.WrecBB, self.Wrec12)), torch.hstack((self.Wrec21, self.WrecBB))))
        if self.random_size>0:
            Wrec = torch.vstack((torch.hstack((Wrec, self.Wrec1R)), torch.hstack((self.WrecR1, self.WrecRR))))
            #torch.vstack((torch.hstack((Wrec, torch.vstack((self.Wrec1R,self.Wrec2R)))), torch.hstack((self.WrecR1, self.WrecR2, self.WrecRR))))
        
        return Wrec
    
    def forward(self, x):
        
        Wrec = self.generate_Wrec()
        
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
            
            it = self.i2h(x[:,t,:])
            
            ht = (it + ht_1 @ Wrec.T + noise)
                
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

# In[] fixed input + recurrent weights for bump cells
class mixed_decayRNN_fix_ib8(nn.Module):
    def __init__(self, input_size, bump_size, random_size, output_size, dt = 10, tau = 100, external = 0, gocue = True,
                 hidden_noise = 0.1, ceiling = None, locs = (0,1,2,3), ttypes = (1,2),
                 in_sigma = 1, in_scale_up = None, in_scale_low = None, train_in = False,
                 rec_sigma = 2, rec_scale_up = None, rec_scale_low = None, train_rec = True, train_bump = False,
                 init_in = 'xavier_normal_', init_hidden = 'orthogonal_', init_out = 'xavier_normal_', 
                 F_hidden = 'relu', F_out = 'softmax', device = device):
        super(mixed_decayRNN_fix_ib8, self).__init__() #, useLinear_hidden = True, useLinear_in = True
        # if change class name, remember to change the super as well
        
        
        ####################
        # Basic parameters #
        ####################
        
        self.device = device
        self.gocue = gocue
        
        self.input_size = input_size
        self.cue_size = 1 if self.gocue else 0
        
        self.output_size = output_size
        
        self.bump_size, self.random_size = bump_size, random_size
        
        self.hidden_size = (self.bump_size + self.random_size)

        self.dt = dt # time step
        self.tau = tau # intrinsic time scale, control delay rate.
        self.alpha = (1.0 * self.dt) / self.tau #Inferred Parameters:**alpha** (*float*) -- The number of unit time constants per simulation timestep.
        
        self.locs, self.ttypes = locs, ttypes
        self.loc_size, self.ttype_size = len(self.locs), len(self.ttypes)
        
        self.n_bumpPops = 2#int((self.input_size - self.cue_size) / len(self.locs))
        self.bump_size_ = int(self.bump_size/self.n_bumpPops)
        
        self.hidden_noise = hidden_noise
        self.external = external # external source of modulation, e.g., top-down modulation from higher level regions... default set to 0
        
        
        ########################################
        # Bump cells initialization parameters #
        ########################################
        
        # input layer weight combined by bump and random
        self.in_sigma = in_sigma
        self.gau_in = f_simulation.gaussian_dis(np.arange(self.bump_size_), u = int(np.median(np.arange(self.bump_size_))), sigma = self.in_sigma)
        
        self.in_scale_up = self.gau_in.max() if (in_scale_up == None) else in_scale_up 
        self.in_scale_low = self.gau_in.min() if (in_scale_low == None) else in_scale_low 
        
        self.gau_in = f_simulation.scale_distribution(self.gau_in, upper=self.in_scale_up, lower=self.in_scale_low)
        
        #self.WinB = torch.tensor(f_simulation.bump_Win(self.gau_in, (self.input_size - self.cue_size), self.bump_size, specificity=1), dtype=torch.float32).to(self.device)
        self.WinB = torch.tensor(f_simulation.bump_Win(self.gau_in, len(self.locs), self.bump_size_, specificity=1), dtype=torch.float32).to(self.device)
        
        # recurrent layer weight combined by bump and random
        self.rec_sigma = rec_sigma
        self.gau_rec = f_simulation.gaussian_dis(np.arange(self.bump_size_), u = int(np.median(np.arange(self.bump_size_))), sigma = self.rec_sigma)
        
        self.rec_scale_up = self.gau_rec.max() if (rec_scale_up == None) else rec_scale_up 
        self.rec_scale_low = self.gau_rec.min() if (rec_scale_low == None) else rec_scale_low 
        
        self.gau_rec = f_simulation.scale_distribution(self.gau_rec, upper=self.rec_scale_up, lower=self.rec_scale_low)
        
        self.WrecBB = torch.tensor(f_simulation.bump_Wrec(self.gau_rec, self.bump_size_, specificity=1), dtype=torch.float32).to(self.device)
        
        
        ##############################################
        # weights + bias between the 3 linear layers #
        ##############################################
        
        #self.useLinear_in = useLinear_in
        self.train_in = train_in
        
        self.Win1R = torch.empty((self.bump_size_, self.loc_size)).to(self.device) # pop1
        self.Win2R = torch.empty((self.bump_size_, self.loc_size)).to(self.device) # pop2
        if self.random_size>0:
            self.WinRR = torch.empty((self.random_size, (self.input_size - self.cue_size))).to(self.device) # random
        if self.gocue:
            self.WinCR = torch.empty((self.hidden_size, self.cue_size)).to(self.device) # cue
        
        if init_in == 'xavier_normal_':    
            self.Win1R = nn.init.xavier_normal_(self.Win1R)
            self.Win2R = nn.init.xavier_normal_(self.Win2R)
            if self.random_size>0:
                self.WinRR = nn.init.xavier_normal_(self.WinRR)
            if self.gocue:
                self.WinCR = nn.init.xavier_normal_(self.WinCR)
            
        elif init_in == 'orthogonal_':    
            self.Win1R = nn.init.orthogonal_(self.Win1R)
            self.Win2R = nn.init.orthogonal_(self.Win2R)
            if self.random_size>0:
                self.WinRR = nn.init.orthogonal_(self.WinRR)
            if self.gocue:
                self.WinCR = nn.init.orthogonal_(self.WinCR)
            
        elif init_in == 'kaiming_normal_':    
            self.Win1R = nn.init.kaiming_normal_(self.Win1R)
            self.Win2R = nn.init.kaiming_normal_(self.Win2R)
            if self.random_size>0:
                self.WinRR = nn.init.kaiming_normal_(self.WinRR)
            if self.gocue:
                self.WinCR = nn.init.kaiming_normal_(self.WinCR)
        
        elif init_in == 'normal_':    
            self.Win1R = nn.init.normal_(self.Win1R, mean=0.0, std=1.0,)
            self.Win2R = nn.init.normal_(self.Win2R, mean=0.0, std=1.0,)
            if self.random_size>0:
                self.WinRR = nn.init.normal_(self.WinRR, mean=0.0, std=1.0,)
            if self.gocue:
                self.WinCR = nn.init.normal_(self.WinCR, mean=0.0, std=1.0,)
        
        #for npop in range(self.n_bumpPops):
        #    self.Win[self.bump_size_*(npop):self.bump_size_*(npop+1), len(self.locs)*(npop):len(self.locs)*(npop+1)] = self.WinB
        #self.i2h.weight = nn.Parameter(self.Win)
        
        # only optimize input weights if need
        if self.train_in:
            self.Win1R = nn.Parameter(self.Win1R)
            self.Win2R = nn.Parameter(self.Win2R)
            if self.random_size>0:
                self.WinRR = nn.Parameter(self.WinRR)
            if self.gocue:
                self.WinCR = nn.Parameter(self.WinCR)
        
        
        
        #########################################################
        # hidden layer recurrent weight matrix, orthogonal init #
        #########################################################
        
        #self.useLinear_hidden = useLinear_hidden
        self.train_rec = train_rec
        self.train_bump = train_bump
        
        self.Wrec12 = torch.empty((self.bump_size_, self.bump_size_)).to(self.device) # random
        self.Wrec21 = torch.empty((self.bump_size_, self.bump_size_)).to(self.device) # random
        
        if self.random_size>0:
            self.WrecRR = torch.empty((self.random_size, self.random_size)).to(self.device) # random
            self.Wrec1R = torch.empty((self.bump_size_, self.random_size)).to(self.device)
            self.WrecR1 = torch.empty((self.random_size, self.bump_size_)).to(self.device)
            self.Wrec2R = torch.empty((self.bump_size_, self.random_size)).to(self.device)
            self.WrecR2 = torch.empty((self.random_size, self.bump_size_)).to(self.device)
        
        if init_hidden == 'xavier_normal_':    
            self.Wrec12 = nn.init.xavier_normal_(self.Wrec12)
            self.Wrec21 = nn.init.xavier_normal_(self.Wrec21)
            
            if self.random_size>0:
                
                self.WrecRR = nn.init.xavier_normal_(self.WrecRR)
                self.Wrec1R = nn.init.xavier_normal_(self.Wrec1R)
                self.WrecR1 = nn.init.xavier_normal_(self.WrecR1)
                self.Wrec2R = nn.init.xavier_normal_(self.Wrec2R)
                self.WrecR2 = nn.init.xavier_normal_(self.WrecR2)
            
        elif init_hidden == 'orthogonal_':    
            self.Wrec12 = nn.init.orthogonal_(self.Wrec12)
            self.Wrec21 = nn.init.orthogonal_(self.Wrec21)
            
            if self.random_size>0:
                
                self.WrecRR = nn.init.orthogonal_(self.WrecRR)
                self.Wrec1R = nn.init.orthogonal_(self.Wrec1R)
                self.WrecR1 = nn.init.orthogonal_(self.WrecR1)
                self.Wrec2R = nn.init.orthogonal_(self.Wrec2R)
                self.WrecR2 = nn.init.orthogonal_(self.WrecR2)
            
        elif init_hidden == 'kaiming_normal_':    
            self.Wrec12 = nn.init.kaiming_normal_(self.Wrec12)
            self.Wrec21 = nn.init.kaiming_normal_(self.Wrec21)
            
            if self.random_size>0:
                
                self.WrecRR = nn.init.kaiming_normal_(self.WrecRR)
                self.Wrec1R = nn.init.kaiming_normal_(self.Wrec1R)
                self.WrecR1 = nn.init.kaiming_normal_(self.WrecR1)
                self.Wrec2R = nn.init.kaiming_normal_(self.Wrec2R)
                self.WrecR2 = nn.init.kaiming_normal_(self.WrecR2)
          
        elif init_hidden == 'normal_':    
            self.Wrec12 = nn.init.normal_(self.Wrec12, mean=0.0, std=1.0,)
            self.Wrec21 = nn.init.normal_(self.Wrec21, mean=0.0, std=1.0,)
            
            if self.random_size>0:
                
                self.WrecRR = nn.init.normal_(self.WrecRR, mean=0.0, std=1.0,)
                self.Wrec1R = nn.init.normal_(self.Wrec1R, mean=0.0, std=1.0,)
                self.WrecR1 = nn.init.normal_(self.WrecR1, mean=0.0, std=1.0,)
                self.Wrec2R = nn.init.normal_(self.Wrec2R, mean=0.0, std=1.0,)
                self.WrecR2 = nn.init.normal_(self.WrecR2, mean=0.0, std=1.0,)
            
        #for npop in range(self.n_bumpPops):
        #    self.Win[self.bump_size_*(npop):self.bump_size_*(npop+1), len(self.locs)*(npop):len(self.locs)*(npop+1)] = self.WinB
        #self.i2h.weight = nn.Parameter(self.Win)
        
        # only optimize input weights if need
        if self.train_bump:
            self.WrecBB = nn.Parameter(self.WrecBB)
        
        if self.train_rec:
            self.Wrec12 = nn.Parameter(self.Wrec12)
            self.Wrec21 = nn.Parameter(self.Wrec21)
            if self.random_size>0:
                self.WrecRR = nn.Parameter(self.WrecRR)
                self.Wrec1R = nn.Parameter(self.Wrec1R)
                self.Wrec2R = nn.Parameter(self.Wrec2R)
                self.WrecR1 = nn.Parameter(self.WrecR1)
                self.WrecR2 = nn.Parameter(self.WrecR2)
        
        
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
        elif init_out == 'normal_':    
            self.h2o.weight = nn.Parameter(nn.init.normal_(self.h2o.weight.data, mean=0.0, std=1.0,))
        
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
        
        # ceiling of hidden neuron activity level
        self.ceiling = ceiling
    
    def generate_Win(self):
        
        #############################
        # build Input Weight Matrix #
        #############################
        
        Win = torch.vstack((torch.hstack((self.WinB, self.Win1R)), torch.hstack((self.Win2R, self.WinB))))
        if self.random_size>0:
            Win = torch.vstack((Win, self.WinRR))
        if self.gocue:
            Win = torch.hstack((Win,self.WinCR))
        
        return Win
    
    def generate_Wrec(self):
        
        #################################
        # build Recurrent Weight Matrix #
        #################################
        
        Wrec = torch.vstack((torch.hstack((self.WrecBB, self.Wrec12)), torch.hstack((self.Wrec21, self.WrecBB))))
        if self.random_size>0:
            Wrec = torch.vstack((torch.hstack((Wrec, torch.vstack((self.Wrec1R,self.Wrec2R)))), torch.hstack((self.WrecR1, self.WrecR2, self.WrecRR))))
        
        return Wrec
    
    def forward(self, x):
        
        Win = self.generate_Win()
        Wrec = self.generate_Wrec()
        
        
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
            #if self.useLinear_in:
                #it = self.i2h(x[:,t,:])
            #else:
            it = x[:,t,:] @ Win.T
            
            #if self.useLinear_hidden:
                #ht = (it + self.h2h(ht_1) + noise)
            #else:
            ht = (it + ht_1 @ Wrec.T + noise)
                
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

# In[] fixed input + recurrent weights for bump cells
class mixed_decayRNN_fix_ib4(nn.Module):
    def __init__(self, input_size, bump_size, random_size, output_size, dt = 10, tau = 100, external = 0, gocue = True,
                 hidden_noise = 0.1, ceiling = None, locs = (0,1,2,3), ttypes = (1,2),
                 in_sigma = 1, in_scale_up = None, in_scale_low = None, train_in = False,
                 rec_sigma = 2, rec_scale_up = None, rec_scale_low = None, train_rec = True, train_bump = False,
                 init_in = 'xavier_normal_', init_hidden = 'orthogonal_', init_out = 'xavier_normal_', 
                 mirror_out = False, F_hidden = 'relu', F_out = 'softmax', device = device):
        super(mixed_decayRNN_fix_ib4, self).__init__() #, useLinear_hidden = True, useLinear_in = True
        # if change class name, remember to change the super as well
        
        
        ####################
        # Basic parameters #
        ####################
        
        self.device = device
        self.gocue = gocue
        
        self.input_size = input_size
        self.cue_size = 1 if self.gocue else 0
        
        self.output_size = output_size
        
        self.bump_size, self.random_size = bump_size, random_size
        
        self.hidden_size = (self.bump_size + self.random_size)

        self.dt = dt # time step
        self.tau = tau # intrinsic time scale, control delay rate.
        self.alpha = (1.0 * self.dt) / self.tau #Inferred Parameters:**alpha** (*float*) -- The number of unit time constants per simulation timestep.
        
        self.locs, self.ttypes = locs, ttypes
        self.loc_size, self.ttype_size = len(self.locs), len(self.ttypes)
        
        self.n_bumpPops = 1 #int((self.input_size - self.cue_size) / len(self.locs))
        self.bump_size_ = int(self.bump_size/self.n_bumpPops)
        
        self.hidden_noise = hidden_noise
        self.external = external # external source of modulation, e.g., top-down modulation from higher level regions... default set to 0
        
        
        ########################################
        # Bump cells initialization parameters #
        ########################################
        
        # input layer weight combined by bump and random
        self.in_sigma = in_sigma
        self.gau_in = f_simulation.gaussian_dis(np.arange(self.bump_size_), u = int(np.median(np.arange(self.bump_size_))), sigma = self.in_sigma)
        
        self.in_scale_up = self.gau_in.max() if (in_scale_up == None) else in_scale_up 
        self.in_scale_low = self.gau_in.min() if (in_scale_low == None) else in_scale_low 
        
        self.gau_in = f_simulation.scale_distribution(self.gau_in, upper=self.in_scale_up, lower=self.in_scale_low)
        
        #self.WinB = torch.tensor(f_simulation.bump_Win(self.gau_in, (self.input_size - self.cue_size), self.bump_size, specificity=1), dtype=torch.float32).to(self.device)
        self.WinB = torch.tensor(f_simulation.bump_Win(self.gau_in, len(self.locs), self.bump_size_, specificity=1), dtype=torch.float32).to(self.device)
        
        # recurrent layer weight combined by bump and random
        self.rec_sigma = rec_sigma
        self.gau_rec = f_simulation.gaussian_dis(np.arange(self.bump_size_), u = int(np.median(np.arange(self.bump_size_))), sigma = self.rec_sigma)
        
        self.rec_scale_up = self.gau_rec.max() if (rec_scale_up == None) else rec_scale_up 
        self.rec_scale_low = self.gau_rec.min() if (rec_scale_low == None) else rec_scale_low 
        
        self.gau_rec = f_simulation.scale_distribution(self.gau_rec, upper=self.rec_scale_up, lower=self.rec_scale_low)
        
        self.WrecBB = torch.tensor(f_simulation.bump_Wrec(self.gau_rec, self.bump_size_, specificity=1), dtype=torch.float32).to(self.device)
        
        
        ##############################################
        # weights + bias between the 3 linear layers #
        ##############################################
        
        #self.useLinear_in = useLinear_in
        self.train_in = train_in
        
        
        if self.random_size>0:
            self.WinRR = torch.empty((self.random_size, (self.input_size - self.cue_size))).to(self.device) # random
        if self.gocue:
            self.WinCR = torch.empty((self.hidden_size, self.cue_size)).to(self.device) # cue
        
        if init_in == 'xavier_normal_':    
            
            if self.random_size>0:
                self.WinRR = nn.init.xavier_normal_(self.WinRR)
            if self.gocue:
                self.WinCR = nn.init.xavier_normal_(self.WinCR)
            
        elif init_in == 'orthogonal_':    
            
            if self.random_size>0:
                self.WinRR = nn.init.orthogonal_(self.WinRR)
            if self.gocue:
                self.WinCR = nn.init.orthogonal_(self.WinCR)
            
        elif init_in == 'kaiming_normal_':    
            
            if self.random_size>0:
                self.WinRR = nn.init.kaiming_normal_(self.WinRR)
            if self.gocue:
                self.WinCR = nn.init.kaiming_normal_(self.WinCR)
        
        #for npop in range(self.n_bumpPops):
        #    self.Win[self.bump_size_*(npop):self.bump_size_*(npop+1), len(self.locs)*(npop):len(self.locs)*(npop+1)] = self.WinB
        #self.i2h.weight = nn.Parameter(self.Win)
        
        # only optimize input weights if need
        if self.train_in:
            
            if self.random_size>0:
                self.WinRR = nn.Parameter(self.WinRR)
            if self.gocue:
                self.WinCR = nn.Parameter(self.WinCR)
        
        
        
        #########################################################
        # hidden layer recurrent weight matrix, orthogonal init #
        #########################################################
        
        #self.useLinear_hidden = useLinear_hidden
        self.train_rec = train_rec
        self.train_bump = train_bump
        
        
        if self.random_size>0:
            self.WrecRR = torch.empty((self.random_size, self.random_size)).to(self.device) # random
            self.Wrec1R = torch.empty((self.bump_size_, self.random_size)).to(self.device)
            self.WrecR1 = torch.empty((self.random_size, self.bump_size_)).to(self.device)
        
        if init_hidden == 'xavier_normal_':    
            
            if self.random_size>0:
                self.WrecRR = nn.init.xavier_normal_(self.WrecRR)
                self.Wrec1R = nn.init.xavier_normal_(self.Wrec1R)
                self.WrecR1 = nn.init.xavier_normal_(self.WrecR1)
            
        elif init_hidden == 'orthogonal_':    
            
            if self.random_size>0:
                self.WrecRR = nn.init.orthogonal_(self.WrecRR)
                self.Wrec1R = nn.init.orthogonal_(self.Wrec1R)
                self.WrecR1 = nn.init.orthogonal_(self.WrecR1)
            
        elif init_hidden == 'kaiming_normal_':    
            
            if self.random_size>0:
                self.WrecRR = nn.init.kaiming_normal_(self.WrecRR)
                self.Wrec1R = nn.init.kaiming_normal_(self.Wrec1R)
                self.WrecR1 = nn.init.kaiming_normal_(self.WrecR1)
            
            
        #for npop in range(self.n_bumpPops):
        #    self.Win[self.bump_size_*(npop):self.bump_size_*(npop+1), len(self.locs)*(npop):len(self.locs)*(npop+1)] = self.WinB
        #self.i2h.weight = nn.Parameter(self.Win)
        
        # only optimize input weights if need
        if self.train_bump:
            self.WrecBB = nn.Parameter(self.WrecBB)
        
        if self.train_rec:
            
            if self.random_size>0:
                self.WrecRR = nn.Parameter(self.WrecRR)
                self.Wrec1R = nn.Parameter(self.Wrec1R)
                self.WrecR1 = nn.Parameter(self.WrecR1)
        
        
        #################################################
        # hidden-output weight matrix + bias, trainable #
        #################################################
        
        self.mirror_out = mirror_out
        
        if self.mirror_out == False:
            self.h2o = nn.Linear(self.hidden_size, self.output_size, bias=False).to(self.device)
            
            if init_out == 'xavier_normal_':
                self.h2o.weight = nn.Parameter(nn.init.xavier_normal_(self.h2o.weight.data))
            elif init_out == 'orthogonal_':    
                self.h2o.weight = nn.Parameter(nn.init.orthogonal_(self.h2o.weight.data))
            elif init_out == 'kaiming_normal_':    
                self.h2o.weight = nn.Parameter(nn.init.kaiming_normal_(self.h2o.weight.data))
        
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
        
        # ceiling of hidden neuron activity level
        self.ceiling = ceiling
    
    def generate_Win(self):
        
        #############################
        # build Input Weight Matrix #
        #############################
        
        Win = self.WinB #torch.vstack((torch.hstack((self.WinB, self.Win1R)), torch.hstack((self.Win2R, self.WinB))))
        if self.random_size>0:
            Win = torch.vstack((Win, self.WinRR))
        if self.gocue:
            Win = torch.hstack((Win,self.WinCR))
        
        return Win
    
    def generate_Wrec(self):
        
        #################################
        # build Recurrent Weight Matrix #
        #################################
        
        Wrec = self.WrecBB #torch.vstack((torch.hstack((self.WrecBB, self.Wrec12)), torch.hstack((self.Wrec21, self.WrecBB))))
        if self.random_size>0:
            Wrec = torch.vstack((torch.hstack((Wrec, self.Wrec1R)), torch.hstack((self.WrecR1, self.WrecRR))))
            #torch.vstack((torch.hstack((Wrec, torch.vstack((self.Wrec1R,self.Wrec2R)))), torch.hstack((self.WrecR1, self.WrecR2, self.WrecRR))))
        
        return Wrec
    
    def forward(self, x):
        
        Win = self.generate_Win()
        Wrec = self.generate_Wrec()
        
        
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
            #if self.useLinear_in:
                #it = self.i2h(x[:,t,:])
            #else:
            it = x[:,t,:] @ Win.T
            
            #if self.useLinear_hidden:
                #ht = (it + self.h2h(ht_1) + noise)
            #else:
            ht = (it + ht_1 @ Wrec.T + noise)
                
            #ht = ((torch.matmul(x[:,t,:], self.Win) + self.Bin) + torch.matmul(ht_1, self.Wrec) + noise)
            ht = self.F_hidden(ht)*self.alpha + (1-self.alpha)*ht_1 + self.external #

            # ceiling
            ht = ht if self.ceiling == None else torch.clamp(ht, min=self.ceiling[0], max = self.ceiling[1])

            hiddens += [ht] # store hidden state at each timepoint

            # output layer
            #out = (torch.matmul(ht, self.Wout) + self.Bout)
            if self.mirror_out:
                Wout = Win.T
                out = ht @ Wout.T
            else:
                out = self.h2o(ht)
                
            out = self.F_out(out)
            outs += [out] # store output at each timepoint

        # stack hidden states together
        hiddens = torch.stack(hiddens[1:]).swapaxes(0,1)
        outs = torch.stack(outs).swapaxes(0,1)

        return hiddens, outs

# In[] piwek 2023's
class nondecayRNN(nn.Module):
    def __init__(self,input_size, hidden_size, output_size, dt = 10, tau = 100, external = 0,
                 hidden_noise = 0.1, ceiling = None, F_hidden = nn.ReLU(), F_out = nn.Softmax(dim = -1),
                 init_in = 'xavier_normal_', init_hidden = 'kaiming_normal_', init_out = 'xavier_normal_', seed = 0, device = device):
        super(nondecayRNN, self).__init__() #, tau = 100
        # if change class name, remember to change the super as well
        
        self.device = device
        
        # set seed
        torch.manual_seed(seed)
        
        self.input_size = input_size
        self.output_size = output_size

        self.hidden_size = hidden_size

        #self.dt = dt # time step
        #self.tau = tau # intrinsic time scale, control delay rate.
        #self.alpha = (1.0 * self.dt) / self.tau #Inferred Parameters:**alpha** (*float*) -- The number of unit time constants per simulation timestep.

        self.hidden_noise = hidden_noise
        self.external = external # external source of modulation, e.g., top-down modulation from higher level regions... default set to 0
        
        #self.init_in, self.init_hidden, self.init_out = init_in, init_hidden, init_out
        
        # weights + bias between the 3 linear layers
        self.i2h = nn.Linear(self.input_size, self.hidden_size).to(self.device)
        self.i2h.weight = nn.Parameter(self.i2h.weight.data)
        self.i2h.bias = nn.Parameter(self.i2h.bias.data)
        #if init_in == 'xavier_normal_':    
        #    self.i2h.weight = nn.Parameter(nn.init.xavier_normal_(self.i2h.weight.data))
        #elif init_in == 'orthogonal_':    
        #    self.i2h.weight = nn.Parameter(nn.init.orthogonal_(self.i2h.weight.data))
        #elif init_in == 'kaiming_normal_':    
        #    self.i2h.weight = nn.Parameter(nn.init.kaiming_normal_(self.i2h.weight.data))
        
        #self.Win = nn.Parameter(nn.init.xavier_normal_(torch.empty((self.input_size, self.hidden_size)))) # input-hidden weight matrix + bias, trainable
        #self.Bin = nn.Parameter(nn.init.xavier_normal_(torch.empty((self.hidden_size,1))).squeeze())
        
        
        self.Wrec = nn.Parameter(nn.init.orthogonal_(torch.empty((self.hidden_size, self.hidden_size)))).to(self.device)  # orthogonal init
        
        # hidden layer recurrent weight matrix, orthogonal init
        self.h2h = nn.Linear(self.hidden_size, self.hidden_size, bias=False).to(self.device)
        #self.h2h.weight = nn.Parameter(nn.init.orthogonal_(self.h2h.weight.data))
        self.h2h.weight = nn.Parameter(self.Wrec)
        
        #if init_hidden == 'xavier_normal_':
        #    self.h2h.weight = nn.Parameter(nn.init.xavier_normal_(self.h2h.weight.data))
        #elif init_hidden == 'orthogonal_':
        #    self.h2h.weight = nn.Parameter(nn.init.orthogonal_(self.h2h.weight.data))
        #elif init_hidden == 'kaiming_normal_':
        #    self.h2h.weight = nn.Parameter(nn.init.kaiming_normal_(self.h2h.weight.data))
        
        #self.h2h.weight = nn.Parameter(nn.init.xavier_normal_(self.h2h.weight.data))
        #self.h2h.bias = nn.Parameter(nn.init.orthogonal_(self.h2h.bias))
        #self.Wrec = nn.Parameter(nn.init.orthogonal_(torch.empty((self.hidden_size, self.hidden_size))))

        # hidden-output weight matrix + bias, trainable
        self.h2o = nn.Linear(self.hidden_size, self.output_size).to(self.device)
        self.h2o.weight = nn.Parameter(self.h2o.weight.data)
        self.h2o.bias = nn.Parameter(self.h2o.bias.data)
        #if init_out == 'xavier_normal_':
        #    self.h2o.weight = nn.Parameter(nn.init.xavier_normal_(self.h2o.weight.data))
        #elif init_out == 'orthogonal_':
        #    self.h2o.weight = nn.Parameter(nn.init.orthogonal_(self.h2o.weight.data))
        #elif init_out == 'kaiming_normal_':
        #    self.h2o.weight = nn.Parameter(nn.init.kaiming_normal_(self.h2o.weight.data))
            
        #self.h2o.bias = nn.Parameter(nn.init.xavier_normal_(self.h2o.bias))
        #self.Wout = nn.Parameter(nn.init.xavier_normal_(torch.empty((self.hidden_size, self.output_size))))
        #self.Bout = nn.Parameter(nn.init.xavier_normal_(torch.empty((self.output_size,1))).squeeze())

        # non-linearity
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.softplus = nn.Softplus()
        self.mish = nn.Mish()

        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim = -1)

        # hidden layer nonlinearity
        
        if F_hidden == 'relu':
            self.F_hidden = self.relu
        elif F_hidden == 'softplus':
            self.F_hidden = self.softplus
        elif F_hidden == 'mish':
            self.F_hidden = self.mish
        elif F_hidden == 'tanh':
            self.F_hidden = self.tanh
        #else:
        #    self.F_hidden  = F_hidden

        # output layer nonlinearity
        
        if F_out == 'softmax':
            self.F_out = self.softmax
        elif F_out == 'sigmoid':
            self.F_out = self.sigmoid
        elif F_out == 'tanh':
            self.F_out = self.tanh
        #else:
        #    self.F_out  = F_out
        
        #self.rnn = nn.RNN(input_size=input_size, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True)
        #self.ceiling = ceiling
    
    #def step(self, xt, ht_1, noise):
    #    """
    #    Run the RNN for one timestep.
    #    """
        #hidden = self.relu(self.i2h(xt.unsqueeze(1)) + (ht_1 @ self.Wrec.T) + noise) #.unsqueeze(1)
        
        #h = hidden.clone().detach()
        # We need to detach the hidden state to be able to save it into a matrix
        # in the forward method, otherwise it messes with the computational graph.

        #return hidden #h, # shape as trial * chs
    
    def forward(self, x):

        # Set initial hidden states
        #h0 = torch.zeros(x.size(0), 1, self.hidden_size).to(self.device)
        h0 = torch.zeros(x.size(0), self.hidden_size).to(self.device)
        
        # Forward propagate RNN
        hiddens = [h0] # initial states
        outs = []
        for t in range(x.size(1)): # go through all timepoints
            ht_1 = hiddens[-1] # hidden states at the previous timepoint

            # your function here
            #noise = torch.normal(mean=0, std=self.hidden_noise, size = (ht_1.size(1),)).to(self.device)
            noise = torch.zeros(h0.size()).to(self.device)#(torch.randn(h0.size()) * self.hidden_noise).to(self.device)
            
            #ht = (self.i2h(x[:,t,:].unsqueeze(1)) + self.h2h(ht_1) + noise)
            #ht = (self.i2h(x[:,t,:].unsqueeze(1)) + ht_1 @ self.Wrec.T + noise)
            ht = (self.i2h(x[:,t,:]) + ht_1 @ self.Wrec.T + noise)
            
            #ht = ((torch.matmul(x[:,t,:], self.Win) + self.Bin) + torch.matmul(ht_1, self.Wrec) + noise)
            ht = self.F_hidden(ht) + self.external # *self.alpha + (1-self.alpha)*ht_1
            
            #ht = self.step(x[:,t,:], ht_1, noise)

            # ceiling
            #ht = ht if self.ceiling == None else torch.clamp(ht, min=self.ceiling[0], max = self.ceiling[1])

            hiddens += [ht] # store hidden state at each timepoint

            # output layer
            #out = (torch.matmul(ht, self.Wout) + self.Bout)
            
            out = self.h2o(ht)
            out = self.F_out(out)
            outs += [out] # store output at each timepoint
        
        #out = (torch.matmul(ht, self.Wout) + self.Bout)
        #out = self.h2o(ht)
        #out = self.F_out(out)
        #outs = out # shape as trial * time * chs.unsqueeze(1)
        
        # stack hidden states together
        #hiddens = torch.stack(hiddens[1:], dim=1).squeeze(-2)#.swapaxes(0,1) # shape as trial*time*chs
        #outs = torch.stack(outs, dim=1).squeeze(-2)#.swapaxes(0,1) # shape as trial*time*chs
        
        hiddens = torch.stack(hiddens[1:]).swapaxes(0,1) # shape as trial*time*chs
        outs = torch.stack(outs).swapaxes(0,1) # shape as trial*time*chs
        
        return hiddens, outs

# In[]
class piwek_RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, noise_sigma, noise_timesteps, device, seed):
        super(piwek_RNN, self).__init__()
        # PARAMETERS
        self.n_rec = hidden_size  # number of recurrent neurons
        self.n_inp = input_size
        self.n_out = output_size
        self.device = device
        self.noise_sigma = noise_sigma
        #self.noise_distr = noise_distr noise_distr, 
        self.noise_timesteps = noise_timesteps

        # set seed
        torch.manual_seed(seed)

        # LAYERS
        # note: We need to add noise after the Wrec @ ht-1 step, but before the nonlinearity.
        # So, we cannot use the torch RNN class, because it does it all in one step.

        # input layer
        self.inp = nn.Linear(self.n_inp, self.n_rec).to(self.device)
        self.inp.weight = nn.Parameter(self.inp.weight)  # Xavier init * params['init_scale']
        self.inp.bias = nn.Parameter(self.inp.bias)  # Xavier init * params['init_scale']

        # recurrent layer
        self.Wrec = nn.Parameter(torch.nn.init.orthogonal_(torch.empty((self.n_rec, self.n_rec)))).to(self.device)  # orthogonal init
        self.relu = nn.ReLU()

        # output layer
        self.out = nn.Linear(self.n_rec, self.n_out).to(self.device)  # output layer
        self.out.weight = nn.Parameter(self.out.weight)  # Xavier init * params['init_scale']
        self.out.bias = nn.Parameter(self.out.bias)  # Xavier init * params['init_scale']
        self.softmax = nn.Softmax(dim=-1)

    def step(self, input_ext, hidden, noise):
        """
        Run the RNN for one timestep.
        """
        hidden = self.relu(self.inp(input_ext.unsqueeze(1)) + hidden @ self.Wrec.T + noise)
        h = hidden.clone().detach()
        # We need to detach the hidden state to be able to save it into a matrix
        # in the forward method, otherwise it messes with the computational graph.

        return h.squeeze(), hidden

    def forward(self, inputs):
        """
        Run the RNN with the input time course.
        """
        inputs = inputs.to(self.device)
        # Add noise to hidden units
        #seq_len = inputs.shape[1]
        #batch_size = inputs.shape[0]
        # To add hidden noise, need to use the expanded implementation below:

        # Initialize network state
        # hidden states from current time point
        hidden = torch.zeros((inputs.size(0), 1, self.n_rec), device=self.device)  # 0s
        # hidden states from all timepoints
        h = torch.empty((inputs.size(0), inputs.size(1), self.n_rec), device=self.device)

        # Run the input through the network - across time
        for timepoint in range(inputs.size(1)):
            if len(np.where(self.noise_timesteps == timepoint)[0]) > 0:
                # Add Gaussian noise to appropriate timesteps of the trial
                noise = (torch.randn(hidden.size(), device=self.device)) * self.noise_sigma
            else:
                # timestep without noise
                noise = torch.zeros(hidden.size(), device=self.device)

            h[:, timepoint, :], hidden = self.step(inputs[:, timepoint, :], hidden, noise)

        # pass the recurrent activation from the last timestep through the decoder layer and apply softmax
        output = self.out(hidden)
        output = self.softmax(output)
        
        return h.to(self.device), output.to(self.device) #.squeeze(), hidden













