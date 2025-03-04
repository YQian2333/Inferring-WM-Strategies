# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 03:27:25 2024

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
sys.path.append(r'C:\Users\aka2333\OneDrive\phd\project')

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

# In[]
import myRNNs
import f_simulation
import f_evaluateRNN
# In[] testing analysis
import f_subspace
import f_stats
import f_decoding
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # define device, may need to change with Mac OS

# In[] Initialize loss function and optimizer
def train_model(modelD, trialInfo, X_, Ys_, tRange, frac = 0.2, criterion = nn.MSELoss(), labelNames = ('choice',), weight_decay = 1e-7,
                learning_rate = 0.0001, n_iter = 10000, loss_cutoff = 0.001, lr_cutoff = 1e-7, l2reg = False, lambda_reg = 0.001, 
                adaptive_lr = True, adaptive_iterCounts = 10, adaptive_reduction = 0.5, givingup_prop = False, givingup_ratio = 0.8):
    
    # initialize optimizer
    learning_rate = learning_rate
    n_iter = n_iter
    loss_cutoff = loss_cutoff
    
    criterion = criterion
    optimizer = torch.optim.NAdam(modelD.parameters(), lr=learning_rate, weight_decay=weight_decay)
    #optimizer = torch.optim.Adam(modelD.parameters(), lr=learning_rate, weight_decay=weight_decay)
    #optimizer = torch.optim.RMSprop(modelD.parameters(), lr=learning_rate, weight_decay=weight_decay)
    #optimizer = torch.optim.SGD(modelD.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    #
    losses = []
    counts = 0
    
    tRange = tRange
    tLength = len(tRange)
    
    # Train the model
    for epoch in range(n_iter):

        train_setID, train_X, train_Y1, _, _, _ = f_simulation.split_dataset(X_, Ys_[0][0], frac = frac, ranseed=epoch)

        # Forward pass
        _, outs = modelD(train_X) # out shape: ntrial * nt * nout

        loss = 0

        
        checkpoint1 = []
        for y in range(len(Ys_)):
            train_Yt = Ys_[y][0][train_setID, :, :]
            y_boundary = Ys_[y][1]
            lweight_t = Ys_[y][2]

            lowBt, upBt = np.where((tRange>=y_boundary[0])&(tRange<=y_boundary[-1]))[0][0], np.where((tRange>=y_boundary[0])&(tRange<=y_boundary[-1]))[0][-1]

            if upBt == tLength-1:
                loss_t = criterion(outs[:, lowBt:, :], train_Yt)
            else:
                loss_t = criterion(outs[:, lowBt:upBt, :], train_Yt)
            
            checkpoint1 += [upBt]
            loss += loss_t * lweight_t
        
        checkpoint1X = -1  if max(checkpoint1) == tLength-1 else max(checkpoint1)
        
        
        # l2 regularization if need penalize and make the weights to be sparse
        lambda_reg = lambda_reg
        
        if l2reg:
            l2_reg = 0.0
            for name, param in modelD.named_parameters():
                if ('h2h' in name) or ('Wrec' in name):
                    l2_reg += torch.norm(param, p=2)
    
            loss = loss + lambda_reg * l2_reg

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # record loss every 50 iterations
        if epoch%50 == 0:
            counts += 1
            losses += [float(loss.detach())]

            # print loss every 100 iterations
            if epoch % 100 == 0:
                #n_correct = 0
                accs = []
                for labelName in labelNames:
                    train_label = trialInfo.loc[train_setID,labelName].astype('int').values
                    #labels = torch.tensor(train_label, dtype=torch.float32).to(device)
                    #_, predicted = torch.max(outs[:,-1,:].detach().data, 1) # channel with max output at last timepoint -> choice
                    
                    acc_memoT = f_evaluateRNN.evaluate_acc(modelD, train_X, train_label, label = 'Acc', checkpointX = checkpoint1X)
                    accs += [np.array(acc_memoT).round(4)]
                    
                
                print (f'Epoch [{epoch}/{n_iter}], Loss: {loss.item():.4f}, Acc_memo: {accs}%')
            
            # adaptively update learning rate if loss does not decrease for certain number of iterations
            if adaptive_lr:
                if counts >= adaptive_iterCounts:
                    if False not in ((np.array(losses[-adaptive_iterCounts:]) - float(loss.detach()))<=learning_rate):
                        
                        learning_rate = learning_rate * adaptive_reduction
                        optimizer.param_groups[0]['lr'] = learning_rate
                        print(f'loss = {loss:.4f}, updated learning rate = {learning_rate}')
                        counts = 0 # reset counts
                        
                        # if learning rate too small, terminate
                        if learning_rate < lr_cutoff:
                            print(f'learning rate too small: {learning_rate} < 1e-7')
                            break
            
            # give up training if no effective improvement after certain number of epochs
            if bool(givingup_prop):
                if (epoch >= givingup_prop * n_iter) and (np.array(losses[-10:]).mean() > (losses[0]*givingup_ratio)):
                    print(f'No effective improvement in {int(givingup_prop * n_iter)} epochs, giving up training')
                    break
                
            
        # if loss lower than the cutoff value, end training
        if loss < loss_cutoff:
            print(f'Loss = {loss.item():.4f}, lower then cutoff = {loss_cutoff}')
            break
        
    return losses#modelD, 

#%%
def train_model_varDelay(modelD, trialInfo, X_, Ys_, tRange, frac = 0.2, criterion = nn.MSELoss(), labelNames = ('choice',), varDelay_label = 'delayLength',
                         weight_decay = 1e-7, learning_rate = 0.0001, n_iter = 10000, loss_cutoff = 0.001, lr_cutoff = 1e-7, l2reg = False, lambda_reg = 0.001, 
                         adaptive_lr = True, adaptive_iterCounts = 10, adaptive_reduction = 0.5, givingup_prop = False, givingup_ratio = 0.8):
    
    varDelay = trialInfo[varDelay_label].unique()
    
    # initialize optimizer
    learning_rate = learning_rate
    n_iter = n_iter
    loss_cutoff = loss_cutoff
    
    criterion = criterion
    optimizer = torch.optim.NAdam(modelD.parameters(), lr=learning_rate, weight_decay=weight_decay)
    #optimizer = torch.optim.Adam(modelD.parameters(), lr=learning_rate, weight_decay=weight_decay)
    #optimizer = torch.optim.RMSprop(modelD.parameters(), lr=learning_rate, weight_decay=weight_decay)
    #optimizer = torch.optim.SGD(modelD.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    #
    losses = []
    counts = 0
    
    tRange = tRange
    tLength = len(tRange)
    
    # Train the model
    for epoch in range(n_iter):

        train_setID, train_X, train_Y1, _, _, _ = f_simulation.split_dataset(X_, Ys_[0][0], frac = frac, ranseed=epoch)

        # Forward pass
        _, outs = modelD(train_X) # out shape: ntrial * nt * nout

        loss = 0

        
        checkpoint1 = []
        for y in range(len(Ys_)):
                       
            train_Yt = Ys_[y][0][train_setID, :, :]
            y_boundary = Ys_[y][1]
            lweight_t = Ys_[y][2]
            
            multiplier_low = Ys_[y][3][0]
            multiplier_high = Ys_[y][3][1]
            
            lowBt = tRange.tolist().index(y_boundary[0])
            upBt = tRange.tolist().index(y_boundary[-1])
            outsT = outs[:, lowBt:, :].clone() if upBt == tLength-1 else outs[:, lowBt:upBt, :].clone()

            for i in train_setID:
                delayLengthT = trialInfo.loc[i, varDelay_label]
                lowB_x = tRange.tolist().index(y_boundary[0] + delayLengthT*multiplier_low) if y_boundary[0] + delayLengthT*multiplier_low > tRange[0] else 0
                upB_x = tRange.tolist().index(y_boundary[-1] + delayLengthT*multiplier_high) if y_boundary[-1] + delayLengthT*multiplier_high < tRange[-1] else tLength-1
                
                i_ = train_setID.tolist().index(i)
                outsT[i_,:,:] = outs[i_,lowB_x:upB_x,:]
            #lowBt = np.where((tRange>=y_boundary[0] + varDelay.min()*multiplier_low)&(tRange<=y_boundary[-1] + varDelay.min()*multiplier_low))[0][0]# if y_boundary[-1] + varDelay.min()*multiplier_low < tRange[-1] else np.where((tRange>=y_boundary[0] + varDelay.min()*multiplier_low))[0][0]
            #upBt = np.where((tRange>=y_boundary[0] + varDelay.max()*multiplier_high)&(tRange<=y_boundary[-1] + varDelay.max()*multiplier_high))[0][-1]

            #if upBt == tLength-1:
            #    loss_t = criterion(outs[:, lowBt:, :], train_Yt)
            #else:
            #    loss_t = criterion(outs[:, lowBt:upBt, :], train_Yt)
            
            loss_t = criterion(outsT, train_Yt)

            checkpoint1 += [upBt]
            loss += loss_t * lweight_t
        
        checkpoint1X = -1  if max(checkpoint1) == tLength-1 else max(checkpoint1)
        
        
        # l2 regularization if need penalize and make the weights to be sparse
        lambda_reg = lambda_reg
        
        if l2reg:
            l2_reg = 0.0
            for name, param in modelD.named_parameters():
                if ('h2h' in name) or ('Wrec' in name):
                    l2_reg += torch.norm(param, p=2)
    
            loss = loss + lambda_reg * l2_reg

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # record loss every 50 iterations
        if epoch%50 == 0:
            counts += 1
            losses += [float(loss.detach())]

            # print loss every 100 iterations
            if epoch % 100 == 0:
                #n_correct = 0
                accs = []
                for labelName in labelNames:
                    train_label = trialInfo.loc[train_setID,labelName].astype('int').values
                    #labels = torch.tensor(train_label, dtype=torch.float32).to(device)
                    #_, predicted = torch.max(outs[:,-1,:].detach().data, 1) # channel with max output at last timepoint -> choice
                    
                    acc_memoT = f_evaluateRNN.evaluate_acc(modelD, train_X, train_label, label = 'Acc', checkpointX = checkpoint1X)
                    accs += [np.array(acc_memoT).round(4)]
                    
                
                print (f'Epoch [{epoch}/{n_iter}], Loss: {loss.item():.4f}, Acc_memo: {accs}%')
            
            # adaptively update learning rate if loss does not decrease for certain number of iterations
            if adaptive_lr:
                if counts >= adaptive_iterCounts:
                    if False not in ((np.array(losses[-adaptive_iterCounts:]) - float(loss.detach()))<=learning_rate):
                        
                        learning_rate = learning_rate * adaptive_reduction
                        optimizer.param_groups[0]['lr'] = learning_rate
                        print(f'loss = {loss:.4f}, updated learning rate = {learning_rate}')
                        counts = 0 # reset counts
                        
                        # if learning rate too small, terminate
                        if learning_rate < lr_cutoff:
                            print(f'learning rate too small: {learning_rate} < 1e-7')
                            break
            
            # give up training if no effective improvement after certain number of epochs
            if bool(givingup_prop):
                if (epoch >= givingup_prop * n_iter) and (np.array(losses[-10:]).mean() > (losses[0]*givingup_ratio)):
                    print(f'No effective improvement in {int(givingup_prop * n_iter)} epochs, giving up training')
                    break
                
            
        # if loss lower than the cutoff value, end training
        if loss < loss_cutoff:
            print(f'Loss = {loss.item():.4f}, lower then cutoff = {loss_cutoff}')
            break
        
    return losses#modelD, 


# In[] Initialize loss function and optimizer
def train_model_multiOutput(modelD, trialInfo, X_, Ys_memo_, Ys_resp_, tRange, frac = 0.2, criterion = nn.MSELoss(), learning_rate = 0.0001, n_iter = 10000, loss_cutoff = 0.001, lr_cutoff = 1e-7):
    
    #modelD = modelD
    N_in, N_hidden, N_out = modelD.input_size, modelD.hidden_size, modelD.output_size
    N_out_half = N_out//2
    # initialize optimizer
    learning_rate = learning_rate
    n_iter = n_iter
    loss_cutoff = loss_cutoff
    
    criterion = criterion
    optimizer = torch.optim.NAdam(modelD.parameters(), lr=learning_rate, weight_decay=0.001)
    
    #
    losses = []
    counts = 0
    
    tRange = tRange
    tLength = len(tRange)
    
    # Train the model
    for epoch in range(n_iter):

        train_setID, train_X, train_Y1, _, _, _ = f_simulation.split_dataset(X_, Ys_memo_[0][0], frac = frac, ranseed=epoch)

        # Forward pass
        _, outs = modelD(train_X)
        
        loss = 0
        loss_memo, loss_resp = 0, 0
        #lambda_reg = 0.001
        
        checkpoint1 = []
        
        for y in range(len(Ys_memo_)):
            train_Yt = Ys_memo_[y][0][train_setID, :, :]
            y_memo_boundary = Ys_memo_[y][1]
            lweight_memo_t = Ys_memo_[y][2]
            
            lowBt, upBt = np.where((tRange>=y_memo_boundary[0])&(tRange<=y_memo_boundary[-1]))[0][0], np.where((tRange>=y_memo_boundary[0])&(tRange<=y_memo_boundary[-1]))[0][-1]
            checkpoint1 += [upBt]
            
            loss_memo_t = criterion(outs[:, lowBt:, :N_out_half], train_Yt) if upBt == tLength-1 else criterion(outs[:, lowBt:upBt, :N_out_half], train_Yt)
            loss_memo += loss_memo_t * lweight_memo_t
        
        checkpoint1X = -1  if max(checkpoint1) == tLength-1 else max(checkpoint1)
        
        
        checkpoint2 = []
        
        for y in range(len(Ys_resp_)):
            train_Yt = Ys_resp_[y][0][train_setID, :, :]
            y_resp_boundary = Ys_resp_[y][1]
            lweight_resp_t = Ys_resp_[y][2]

            lowBt, upBt = np.where((tRange>=y_resp_boundary[0])&(tRange<=y_resp_boundary[-1]))[0][0], np.where((tRange>=y_resp_boundary[0])&(tRange<=y_resp_boundary[-1]))[0][-1]
            checkpoint2 += [upBt]
            
            loss_resp_t = criterion(outs[:, lowBt:, N_out_half:], train_Yt) if upBt == tLength-1 else criterion(outs[:, lowBt:upBt, N_out_half:], train_Yt)
            loss_resp += loss_resp_t * lweight_resp_t
        
        checkpoint2X = -1  if max(checkpoint2) == tLength-1 else max(checkpoint2)
        
        loss = loss_memo + loss_resp
        
        # l2 regularization if need penalize and make the weights to be sparse
        #l2_reg = 0.0
        #for name, param in modelD.named_parameters():
        #    if 'h2h' in name:
        #        l2_reg += torch.norm(param, p=2)

        #loss = loss + lambda_reg * l2_reg

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # record loss every 50 iterations
        if epoch%50 == 0:
            counts += 1
            losses += [float(loss.detach())]

            # print loss every 100 iterations
            if epoch % 100 == 0:
                
                train_label = trialInfo.loc[train_setID,'choice'].astype('int').values
                
                acc_memo, acc_resp = f_evaluateRNN.evaluate_acc_multi(modelD, train_X, train_label, checkpoint1X=checkpoint1X)
                
                print (f'Epoch [{epoch}/{n_iter}], Loss: {loss.item():.4f}, Acc_memo: {acc_memo:.2f}%, Acc_resp: {acc_resp:.2f}%')

            # adaptively update learning rate
            if counts >= 10:
                if False not in ((np.array(losses[-10:]) - float(loss.detach()))<=learning_rate):
                    learning_rate = learning_rate/2
                    optimizer.param_groups[0]['lr'] = learning_rate
                    print(f'loss = {loss:.4f}, updated learning rate = {learning_rate}')
                    counts = 0 # reset counts

                    if learning_rate < lr_cutoff:
                        print(f'learning rate too small: {learning_rate} < 1e-7')
                        break

        # if loss lower than the cutoff value, end training
        if loss < loss_cutoff:
            print(f'Loss = {loss.item():.4f}, lower then cutoff = {loss_cutoff}')
            break
        
    return losses#modelD, 

# In[] fit only to the output at the end timepoint
def train_model_end(modelD, trialInfo, X_, Y_, tRange, frac = 0.2, criterion = nn.MSELoss(), learning_rate = 0.0001, n_iter = 10000, loss_cutoff = 0.001, lr_cutoff = 1e-7):
    
    #modelD = modelD
    
    # initialize optimizer
    learning_rate = learning_rate
    n_iter = n_iter
    loss_cutoff = loss_cutoff
    
    criterion = criterion
    #optimizer = torch.optim.NAdam(modelD.parameters(), lr=learning_rate, weight_decay=0.001)
    optimizer = torch.optim.RMSprop(modelD.parameters(), lr=learning_rate, weight_decay=0.001)
    
    #
    losses = []
    counts = 0
    
    tRange = tRange
    tLength = len(tRange)
    
    # Train the model
    for epoch in range(n_iter):

        train_setID, train_X, train_Y1, _, _, _ = f_simulation.split_dataset(X_, Y_, frac = frac, ranseed=epoch)

        # Forward pass
        _, outs = modelD(train_X)

        loss = 0

        #lambda_reg = 0.001
        
        loss = criterion(outs[:, -1, :], train_Y1[:, -1, :]) #.squeeze().squeeze()

        # l2 regularization if need penalize and make the weights to be sparse
        #l2_reg = 0.0
        #for name, param in modelD.named_parameters():
        #    if 'h2h' in name:
        #        l2_reg += torch.norm(param, p=2)

        #loss = loss + lambda_reg * l2_reg

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # record loss every 50 iterations
        if epoch%50 == 0:
            counts += 1
            losses += [float(loss.detach())]

            # print loss every 100 iterations
            if epoch % 100 == 0:
                #n_correct = 0
                train_label = trialInfo.loc[train_setID,'choice'].astype('int').values
                #labels = torch.tensor(train_label, dtype=torch.float32).to(device)
                #_, predicted = torch.max(outs[:,-1,:].detach().data, 1) # channel with max output at last timepoint -> choice
                
                acc_memo = f_evaluateRNN.evaluate_acc(modelD, train_X, train_label, label = 'Acc')
                
                #n_correct_memo = 0
                #n_samples = 0
                
                #n_samples += labels.size(0)
                
                #_, predicted_memo = torch.max(outs[:,-1,:].data, 1) # channel with max output at last timepoint -> choice
                #n_correct_memo += (predicted_memo == labels).sum().item()
                #acc_memo = 100.0 * n_correct_memo / n_samples  
                
                print (f'Epoch [{epoch}/{n_iter}], Loss: {loss.item():.4f}, Acc_memo: {acc_memo:.2f}%')

            # adaptively update learning rate
            if counts >= 10:
                if False not in ((np.array(losses[-10:]) - float(loss.detach()))<=learning_rate):
                    learning_rate = learning_rate/2
                    optimizer.param_groups[0]['lr'] = learning_rate
                    print(f'loss = {loss:.4f}, updated learning rate = {learning_rate}')
                    counts = 0 # reset counts

                    if learning_rate < lr_cutoff:
                        print(f'learning rate too small: {learning_rate} < 1e-7')
                        break

        # if loss lower than the cutoff value, end training
        if loss < loss_cutoff:
            print(f'Loss = {loss.item():.4f}, lower then cutoff = {loss_cutoff}')
            break
        
    return losses#modelD, 

# In[] Initialize loss function and optimizer
def train_model_tbt(modelD, trialInfo, X_, Ys_, tRange, frac = 0.2, criterion = nn.MSELoss(), learning_rate = 0.0001, n_iter = 10000, loss_cutoff = 0.001, lr_cutoff = 1e-7):
    
    #modelD = modelD
    
    # initialize optimizer
    learning_rate = learning_rate
    n_iter = n_iter
    loss_cutoff = loss_cutoff
    
    criterion = criterion
    optimizer = torch.optim.NAdam(modelD.parameters(), lr=learning_rate, weight_decay=0.001)
    #optimizer = torch.optim.RMSprop(modelD.parameters(), lr=learning_rate, weight_decay=0.001)
    
    #
    losses = []
    counts = 0
    
    tRange = tRange
    tLength = len(tRange)
    
    # Train the model
    for epoch in range(n_iter):

        train_setID, train_X, train_Y1, _, _, _ = f_simulation.split_dataset(X_, Ys_[0][0], frac = frac, ranseed=epoch)
        
        for trial in range(len(train_setID)):
            
            # Forward pass
            _, outs = modelD(train_X[trial].unsqueeze(0))
    
            loss = 0
    
            #lambda_reg = 0.001
            #checkpoint1 = []
            for y in range(len(Ys_)):
                train_Yt = Ys_[y][0][train_setID, :, :]
                y_boundary = Ys_[y][1]
                lweight_t = Ys_[y][2]
    
                lowBt, upBt = np.where((tRange>=y_boundary[0])&(tRange<=y_boundary[-1]))[0][0], np.where((tRange>=y_boundary[0])&(tRange<=y_boundary[-1]))[0][-1]
    
                if upBt == tLength-1:
                    loss_t = criterion(outs[0, lowBt:, :], train_Yt[trial])
                else:
                    loss_t = criterion(outs[0, lowBt:upBt, :], train_Yt[trial])
                
                #checkpoint1 += [upBt]
                loss += loss_t * lweight_t
            
            #checkpoint1X = -1  if max(checkpoint1) == tLength-1 else max(checkpoint1)
            # l2 regularization if need penalize and make the weights to be sparse
            #l2_reg = 0.0
            #for name, param in modelD.named_parameters():
            #    if 'h2h' in name:
            #        l2_reg += torch.norm(param, p=2)
    
            #loss = loss + lambda_reg * l2_reg
    
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # record loss every 50 iterations
        if epoch%50 == 0:
            counts += 1
            
            _, outs_batch = modelD(train_X)
            
            # batch loss
            loss_batch = 0
    
            #lambda_reg = 0.001
            checkpoint1 = []
            for y in range(len(Ys_)):
                train_Yt = Ys_[y][0][train_setID, :, :]
                y_boundary = Ys_[y][1]
                lweight_t = Ys_[y][2]
    
                lowBt, upBt = np.where((tRange>=y_boundary[0])&(tRange<=y_boundary[-1]))[0][0], np.where((tRange>=y_boundary[0])&(tRange<=y_boundary[-1]))[0][-1]
    
                if upBt == tLength-1:
                    loss_t = criterion(outs_batch[:, lowBt:, :], train_Yt)
                else:
                    loss_t = criterion(outs_batch[:, lowBt:upBt, :], train_Yt)
                
                checkpoint1 += [upBt]
                loss_batch += loss_t * lweight_t
                
            checkpoint1X = -1  if max(checkpoint1) == tLength-1 else max(checkpoint1)
            
            losses += [float(loss_batch.detach())]

            # print loss every 100 iterations
            if epoch % 100 == 0:
                #n_correct = 0
                train_label = trialInfo.loc[train_setID,'choice'].astype('int').values
                #labels = torch.tensor(train_label, dtype=torch.float32).to(device)
                #_, predicted = torch.max(outs[:,-1,:].detach().data, 1) # channel with max output at last timepoint -> choice
                
                acc_memo = f_evaluateRNN.evaluate_acc(modelD, train_X, train_label, label = 'Acc', checkpointX = checkpoint1X)
                
                #n_correct_memo = 0
                #n_samples = 0
                
                #n_samples += labels.size(0)
                
                #_, predicted_memo = torch.max(outs[:,-1,:].data, 1) # channel with max output at last timepoint -> choice
                #n_correct_memo += (predicted_memo == labels).sum().item()
                #acc_memo = 100.0 * n_correct_memo / n_samples  
                
                print (f'Epoch [{epoch}/{n_iter}], Loss: {loss_batch.item():.4f}, Acc_memo: {acc_memo:.2f}%')

            # adaptively update learning rate
            if counts >= 10:
                if False not in ((np.array(losses[-10:]) - float(loss.detach()))<=learning_rate):
                    learning_rate = learning_rate/2
                    optimizer.param_groups[0]['lr'] = learning_rate
                    print(f'loss = {loss_batch:.4f}, updated learning rate = {learning_rate}')
                    counts = 0 # reset counts

                    if learning_rate < lr_cutoff:
                        print(f'learning rate too small: {learning_rate} < 1e-7')
                        break

        # if loss lower than the cutoff value, end training
        if loss < loss_cutoff:
            print(f'Loss = {loss_batch.item():.4f}, lower then cutoff = {loss_cutoff}')
            break
        
    return losses#modelD, 



# In[] Initialize loss function and optimizer
def train_model_multiOutput_tbt(modelD, trialInfo, X_, Ys_memo_, Ys_resp_, tRange, frac = 0.2, criterion = nn.MSELoss(), learning_rate = 0.0001, n_iter = 10000, loss_cutoff = 0.001, lr_cutoff = 1e-7):
    
    #modelD = modelD
    N_in, N_hidden, N_out = modelD.input_size, modelD.hidden_size, modelD.output_size
    N_out_half = N_out//2
    # initialize optimizer
    learning_rate = learning_rate
    n_iter = n_iter
    loss_cutoff = loss_cutoff
    
    criterion = criterion
    optimizer = torch.optim.NAdam(modelD.parameters(), lr=learning_rate, weight_decay=0.001)
    
    #
    losses = []
    counts = 0
    
    tRange = tRange
    tLength = len(tRange)
    
    # Train the model
    for epoch in range(n_iter):

        train_setID, train_X, train_Y1, _, _, _ = f_simulation.split_dataset(X_, Ys_memo_[0][0], frac = frac, ranseed=epoch)
        
        for trial in range(len(train_setID)):
            
            # Forward pass
            _, outs = modelD(train_X[trial].unsqueeze(0))
            
            loss = 0
            loss_memo, loss_resp = 0, 0
            #lambda_reg = 0.001
            
            #checkpoint1 = []
            
            for y in range(len(Ys_memo_)):
                train_Yt = Ys_memo_[y][0][train_setID, :, :]
                y_memo_boundary = Ys_memo_[y][1]
                lweight_memo_t = Ys_memo_[y][2]
                
                lowBt, upBt = np.where((tRange>=y_memo_boundary[0])&(tRange<=y_memo_boundary[-1]))[0][0], np.where((tRange>=y_memo_boundary[0])&(tRange<=y_memo_boundary[-1]))[0][-1]
                #checkpoint1 += [upBt]
                
                loss_memo_t = criterion(outs[0, lowBt:, :N_out_half], train_Yt[trial]) if upBt == tLength-1 else criterion(outs[0, lowBt:upBt, :N_out_half], train_Yt[trial])
                loss_memo += loss_memo_t * lweight_memo_t
            
            #checkpoint1X = -1  if max(checkpoint1) == tLength-1 else max(checkpoint1)
            
            
            #checkpoint2 = []
            
            for y in range(len(Ys_resp_)):
                train_Yt = Ys_resp_[y][0][train_setID, :, :]
                y_resp_boundary = Ys_resp_[y][1]
                lweight_resp_t = Ys_resp_[y][2]
    
                lowBt, upBt = np.where((tRange>=y_resp_boundary[0])&(tRange<=y_resp_boundary[-1]))[0][0], np.where((tRange>=y_resp_boundary[0])&(tRange<=y_resp_boundary[-1]))[0][-1]
                #checkpoint2 += [upBt]
                
                loss_resp_t = criterion(outs[0, lowBt:, N_out_half:], train_Yt[trial]) if upBt == tLength-1 else criterion(outs[0, lowBt:upBt, N_out_half:], train_Yt[trial])
                loss_resp += loss_resp_t * lweight_resp_t
            
            #checkpoint2X = -1  if max(checkpoint2) == tLength-1 else max(checkpoint2)
            
            loss = loss_memo + loss_resp
            
            # l2 regularization if need penalize and make the weights to be sparse
            #l2_reg = 0.0
            #for name, param in modelD.named_parameters():
            #    if 'h2h' in name:
            #        l2_reg += torch.norm(param, p=2)
    
            #loss = loss + lambda_reg * l2_reg
    
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # record loss every 50 iterations
        if epoch%50 == 0:
            counts += 1
            
            _, outs_batch = modelD(train_X)
            
            loss_batch = 0
            loss_memo_batch, loss_resp_batch = 0, 0
            #lambda_reg = 0.001
            
            checkpoint1 = []
            
            for y in range(len(Ys_memo_)):
                train_Yt = Ys_memo_[y][0][train_setID, :, :]
                y_memo_boundary = Ys_memo_[y][1]
                lweight_memo_t = Ys_memo_[y][2]
                
                lowBt, upBt = np.where((tRange>=y_memo_boundary[0])&(tRange<=y_memo_boundary[-1]))[0][0], np.where((tRange>=y_memo_boundary[0])&(tRange<=y_memo_boundary[-1]))[0][-1]
                checkpoint1 += [upBt]
                
                loss_memo_t = criterion(outs_batch[:, lowBt:, :N_out_half], train_Yt) if upBt == tLength-1 else criterion(outs_batch[:, lowBt:upBt, :N_out_half], train_Yt)
                loss_memo_batch += loss_memo_t * lweight_memo_t
            
            checkpoint1X = -1  if max(checkpoint1) == tLength-1 else max(checkpoint1)
            
            
            checkpoint2 = []
            
            for y in range(len(Ys_resp_)):
                train_Yt = Ys_resp_[y][0][train_setID, :, :]
                y_resp_boundary = Ys_resp_[y][1]
                lweight_resp_t = Ys_resp_[y][2]
    
                lowBt, upBt = np.where((tRange>=y_resp_boundary[0])&(tRange<=y_resp_boundary[-1]))[0][0], np.where((tRange>=y_resp_boundary[0])&(tRange<=y_resp_boundary[-1]))[0][-1]
                checkpoint2 += [upBt]
                
                loss_resp_t = criterion(outs_batch[:, lowBt:, N_out_half:], train_Yt) if upBt == tLength-1 else criterion(outs_batch[:, lowBt:upBt, N_out_half:], train_Yt)
                loss_resp_batch += loss_resp_t * lweight_resp_t
            
            checkpoint2X = -1  if max(checkpoint2) == tLength-1 else max(checkpoint2)
            
            loss_batch = loss_memo_batch + loss_resp_batch
            
            losses += [float(loss_batch.detach())]

            # print loss every 100 iterations
            if epoch % 100 == 0:
                
                train_label = trialInfo.loc[train_setID,'choice'].astype('int').values
                
                acc_memo, acc_resp = f_evaluateRNN.evaluate_acc_multi(modelD, train_X, train_label, checkpoint1X=checkpoint1X)
                
                print (f'Epoch [{epoch}/{n_iter}], Loss: {loss_batch.item():.4f}, Acc_memo: {acc_memo:.2f}%, Acc_resp: {acc_resp:.2f}%')

            # adaptively update learning rate
            if counts >= 10:
                if False not in ((np.array(losses[-10:]) - float(loss_batch.detach()))<=learning_rate):
                    learning_rate = learning_rate/2
                    optimizer.param_groups[0]['lr'] = learning_rate
                    print(f'loss = {loss_batch:.4f}, updated learning rate = {learning_rate}')
                    counts = 0 # reset counts

                    if learning_rate < lr_cutoff:
                        print(f'learning rate too small: {learning_rate} < 1e-7')
                        break

        # if loss lower than the cutoff value, end training
        if loss_batch < loss_cutoff:
            print(f'Loss = {loss_batch.item():.4f}, lower then cutoff = {loss_cutoff}')
            break
        
    return losses#modelD, 




# In[] fit only to the output at the end timepoint
def train_model_end_tbt(modelD, trialInfo, X_, Y_, tRange, frac = 0.2, criterion = nn.MSELoss(), learning_rate = 0.0001, n_iter = 10000, loss_cutoff = 0.001, lr_cutoff = 1e-7):
    
    #modelD = modelD
    
    # initialize optimizer
    learning_rate = learning_rate
    n_iter = n_iter
    loss_cutoff = loss_cutoff
    
    criterion = criterion
    #optimizer = torch.optim.NAdam(modelD.parameters(), lr=learning_rate, weight_decay=0.001)
    optimizer = torch.optim.RMSprop(modelD.parameters(), lr=learning_rate, weight_decay=0.001)
    
    #
    losses = []
    counts = 0
    
    tRange = tRange
    tLength = len(tRange)
    
    # Train the model
    for epoch in range(n_iter):

        train_setID, train_X, train_Y1, _, _, _ = f_simulation.split_dataset(X_, Y_, frac = frac, ranseed=epoch)
        
        for trial in range(len(train_setID)):
            
            # Forward pass
            _, outs = modelD(train_X[trial].unsqueeze(0))
    
            loss = 0
    
            #lambda_reg = 0.001
            
            loss = criterion(outs[0, -1, :], train_Y1[trial, -1, :]) #.squeeze().squeeze()
    
            # l2 regularization if need penalize and make the weights to be sparse
            #l2_reg = 0.0
            #for name, param in modelD.named_parameters():
            #    if 'h2h' in name:
            #        l2_reg += torch.norm(param, p=2)
    
            #loss = loss + lambda_reg * l2_reg
    
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # record loss every 50 iterations
        if epoch%50 == 0:
            
            _, outs_batch = modelD(train_X)
            loss_batch = criterion(outs_batch[:,-1,:], train_Y1[:,-1,:]) #.squeeze() .squeeze()
            
            counts += 1
            losses += [float(loss_batch.detach())]

            # print loss every 100 iterations
            if epoch % 100 == 0:
                #n_correct = 0
                train_label = trialInfo.loc[train_setID,'choice'].astype('int').values
                #labels = torch.tensor(train_label, dtype=torch.float32).to(device)
                #_, predicted = torch.max(outs[:,-1,:].detach().data, 1) # channel with max output at last timepoint -> choice
                
                acc_memo = f_evaluateRNN.evaluate_acc(modelD, train_X, train_label, label = 'Acc')
                
                #n_correct_memo = 0
                #n_samples = 0
                
                #n_samples += labels.size(0)
                
                #_, predicted_memo = torch.max(outs[:,-1,:].data, 1) # channel with max output at last timepoint -> choice
                #n_correct_memo += (predicted_memo == labels).sum().item()
                #acc_memo = 100.0 * n_correct_memo / n_samples  
                
                print (f'Epoch [{epoch}/{n_iter}], Loss: {loss_batch.item():.4f}, Acc_memo: {acc_memo:.2f}%')

            # adaptively update learning rate
            if counts >= 10:
                if False not in ((np.array(losses[-10:]) - float(loss_batch.detach()))<=learning_rate):
                    learning_rate = learning_rate/2
                    optimizer.param_groups[0]['lr'] = learning_rate
                    print(f'loss = {loss_batch:.4f}, updated learning rate = {learning_rate}')
                    counts = 0 # reset counts

                    if learning_rate < lr_cutoff:
                        print(f'learning rate too small: {learning_rate} < 1e-7')
                        break

        # if loss lower than the cutoff value, end training
        if loss_batch < loss_cutoff:
            print(f'Loss = {loss_batch.item():.4f}, lower then cutoff = {loss_cutoff}')
            break
        
    return losses#modelD, 



# In[]

def circular_MSE(outsT, expectsT, device = device):
    # outsT, expectsT as outputs/expectations at time t, shape as ntrial*nouts
    output_size = outsT.shape[1]
    circular_scaler = torch.linspace(-np.pi, np.pi, output_size+1)[:-1].to(device)
    a_outsT, a_expectsT = circular_scaler[outsT.argmax(1)], circular_scaler[expectsT.argmax(1)]
    
    #loss = ((outsT - expectsT)**2).mean() * ((a_outsT - a_expectsT)**2).mean()
    loss = (((outsT - expectsT)**2).mean(1) * ((a_outsT - a_expectsT)**2)).mean()
    #loss = (((outsT - expectsT).mean(1) * (a_outsT - a_expectsT))**2).mean()
    
    return loss

# In[] Initialize loss function and optimizer
def train_model_circular(modelD, trialInfo, X_, Ys_, tRange, frac = 0.2, learning_rate = 0.0001, n_iter = 10000, loss_cutoff = 0.001, lr_cutoff = 1e-7):
    
    #modelD = modelD
    
    # initialize optimizer
    learning_rate = learning_rate
    n_iter = n_iter
    loss_cutoff = loss_cutoff
    
    optimizer = torch.optim.NAdam(modelD.parameters(), lr=learning_rate, weight_decay=0.001)
    #optimizer = torch.optim.RMSprop(modelD.parameters(), lr=learning_rate, weight_decay=0.001)
    
    #
    losses = []
    counts = 0
    
    tRange = tRange
    tLength = len(tRange)
    
    # Train the model
    for epoch in range(n_iter):

        train_setID, train_X, train_Y1, _, _, _ = f_simulation.split_dataset(X_, Ys_[0][0], frac = frac, ranseed=epoch)

        # Forward pass
        _, outs = modelD(train_X) # out shape: ntrial * nt * nout

        loss = 0

        #lambda_reg = 0.001
        checkpoint1 = []
        for y in range(len(Ys_)):
            train_Yt = Ys_[y][0][train_setID, :, :]
            y_boundary = Ys_[y][1]
            lweight_t = Ys_[y][2]

            lowBt, upBt = np.where((tRange>=y_boundary[0])&(tRange<=y_boundary[-1]))[0][0], np.where((tRange>=y_boundary[0])&(tRange<=y_boundary[-1]))[0][-1]

            if upBt == tLength-1:
                loss_t = circular_MSE(outs[:, lowBt:, :], train_Yt)
            else:
                loss_t = circular_MSE(outs[:, lowBt:upBt, :], train_Yt)
            
            checkpoint1 += [upBt]
            loss += loss_t * lweight_t
        
        checkpoint1X = -1  if max(checkpoint1) == tLength-1 else max(checkpoint1)
        # l2 regularization if need penalize and make the weights to be sparse
        #l2_reg = 0.0
        #for name, param in modelD.named_parameters():
        #    if 'h2h' in name:
        #        l2_reg += torch.norm(param, p=2)

        #loss = loss + lambda_reg * l2_reg

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # record loss every 50 iterations
        if epoch%50 == 0:
            counts += 1
            losses += [float(loss.detach())]

            # print loss every 100 iterations
            if epoch % 100 == 0:
                #n_correct = 0
                train_label = trialInfo.loc[train_setID,'choice'].astype('int').values
                #labels = torch.tensor(train_label, dtype=torch.float32).to(device)
                #_, predicted = torch.max(outs[:,-1,:].detach().data, 1) # channel with max output at last timepoint -> choice
                
                acc_memo = f_evaluateRNN.evaluate_acc(modelD, train_X, train_label, label = 'Acc', checkpointX = checkpoint1X)
                
                #n_correct_memo = 0
                #n_samples = 0
                
                #n_samples += labels.size(0)
                
                #_, predicted_memo = torch.max(outs[:,-1,:].data, 1) # channel with max output at last timepoint -> choice
                #n_correct_memo += (predicted_memo == labels).sum().item()
                #acc_memo = 100.0 * n_correct_memo / n_samples  
                
                print (f'Epoch [{epoch}/{n_iter}], Loss: {loss.item():.4f}, Acc_memo: {acc_memo:.2f}%')

            # adaptively update learning rate
            if counts >= 10:
                if False not in ((np.array(losses[-10:]) - float(loss.detach()))<=learning_rate):
                    learning_rate = learning_rate/2
                    optimizer.param_groups[0]['lr'] = learning_rate
                    print(f'loss = {loss:.4f}, updated learning rate = {learning_rate}')
                    counts = 0 # reset counts

                    if learning_rate < lr_cutoff:
                        print(f'learning rate too small: {learning_rate} < {lr_cutoff}')
                        break

        # if loss lower than the cutoff value, end training
        if loss < loss_cutoff:
            print(f'Loss = {loss.item():.4f}, lower then cutoff = {loss_cutoff}')
            break
        
    return losses#modelD, 

# In[] fit only to the output at the end timepoint
def train_model_end_circular(modelD, trialInfo, X_, Y_, tRange, frac = 0.2, learning_rate = 0.0001, n_iter = 10000, loss_cutoff = 0.001, lr_cutoff = 1e-7):
    
    #modelD = modelD
    
    # initialize optimizer
    learning_rate = learning_rate
    n_iter = n_iter
    loss_cutoff = loss_cutoff
    
    #optimizer = torch.optim.NAdam(modelD.parameters(), lr=learning_rate, weight_decay=0.001)
    optimizer = torch.optim.RMSprop(modelD.parameters(), lr=learning_rate, weight_decay=0.001)
    
    #
    losses = []
    counts = 0
    
    tRange = tRange
    tLength = len(tRange)
    
    # Train the model
    for epoch in range(n_iter):

        train_setID, train_X, train_Y1, _, _, _ = f_simulation.split_dataset(X_, Y_, frac = frac, ranseed=epoch)

        # Forward pass
        _, outs = modelD(train_X)

        loss = 0

        #lambda_reg = 0.001
        
        loss = circular_MSE(outs[:, -1, :], train_Y1[:, -1, :]) #.squeeze().squeeze()

        # l2 regularization if need penalize and make the weights to be sparse
        #l2_reg = 0.0
        #for name, param in modelD.named_parameters():
        #    if 'h2h' in name:
        #        l2_reg += torch.norm(param, p=2)

        #loss = loss + lambda_reg * l2_reg

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # record loss every 50 iterations
        if epoch%50 == 0:
            counts += 1
            losses += [float(loss.detach())]

            # print loss every 100 iterations
            if epoch % 100 == 0:
                #n_correct = 0
                train_label = trialInfo.loc[train_setID,'choice'].astype('int').values
                #labels = torch.tensor(train_label, dtype=torch.float32).to(device)
                #_, predicted = torch.max(outs[:,-1,:].detach().data, 1) # channel with max output at last timepoint -> choice
                
                acc_memo = f_evaluateRNN.evaluate_acc(modelD, train_X, train_label, label = 'Acc')
                
                #n_correct_memo = 0
                #n_samples = 0
                
                #n_samples += labels.size(0)
                
                #_, predicted_memo = torch.max(outs[:,-1,:].data, 1) # channel with max output at last timepoint -> choice
                #n_correct_memo += (predicted_memo == labels).sum().item()
                #acc_memo = 100.0 * n_correct_memo / n_samples  
                
                print (f'Epoch [{epoch}/{n_iter}], Loss: {loss.item():.4f}, Acc_memo: {acc_memo:.2f}%')

            # adaptively update learning rate
            if counts >= 10:
                if False not in ((np.array(losses[-10:]) - float(loss.detach()))<=learning_rate):
                    learning_rate = learning_rate/2
                    optimizer.param_groups[0]['lr'] = learning_rate
                    print(f'loss = {loss:.4f}, updated learning rate = {learning_rate}')
                    counts = 0 # reset counts

                    if learning_rate < lr_cutoff:
                        print(f'learning rate too small: {learning_rate} < 1e-7')
                        break

        # if loss lower than the cutoff value, end training
        if loss < loss_cutoff:
            print(f'Loss = {loss.item():.4f}, lower then cutoff = {loss_cutoff}')
            break
        
    return losses#modelD, 