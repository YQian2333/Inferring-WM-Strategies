# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 03:27:25 2024

@author: aka2333
"""

#%%
import numpy as np
# turn off warning messages
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# system paths
import torch
import torch.nn as nn

import f_simulation
import f_evaluateRNN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # define device

# In[] Initialize loss function and optimizer
def train_model(modelD, trialInfo, X_, Ys_, tRange, frac = 0.2, criterion = nn.MSELoss(), labelNames = ('choice',), weight_decay = 1e-7,
                learning_rate = 0.0001, n_iter = 10000, loss_cutoff = 0.001, lr_cutoff = 1e-7, l2reg = False, lambda_reg = 0.001, 
                adaptive_lr = True, adaptive_iterCounts = 10, adaptive_reduction = 0.5, givingup_prop = False, givingup_ratio = 0.8):
    
    """
    modelD: model to train
    trialInfo: trial information
    X_: input data
    Ys_: list of expected output data to fit
    frac: fraction for splitting dataset
    loss_cutoff: loss cutoff for early stopping
    lr_cutoff: learning rate cutoff for early stopping
    l2reg: whether to use l2 regularization
    lambda_reg: weight of l2 regularization if used
    adaptive_lr: whether to adaptively reduce learning rate
    adaptive_iterCounts: number of iterations before reducing learning rate
    adaptive_reduction: reduction factor for learning rate
    givingup_prop: whether to give up training if loss does not decrease for a certain number of iterations
    givingup_ratio: ratio of loss reduction to give up
    """

    # initialize optimizer
    learning_rate = learning_rate
    n_iter = n_iter
    loss_cutoff = loss_cutoff
    
    criterion = criterion

    # optimizer algorithm to use
    optimizer = torch.optim.NAdam(modelD.parameters(), lr=learning_rate, weight_decay=weight_decay)
    #optimizer = torch.optim.Adam(modelD.parameters(), lr=learning_rate, weight_decay=weight_decay)
    #optimizer = torch.optim.RMSprop(modelD.parameters(), lr=learning_rate, weight_decay=weight_decay)
    #optimizer = torch.optim.SGD(modelD.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # initialize loss and count
    losses = []
    counts = 0
    
    tRange = tRange
    tLength = len(tRange)
    
    # Train the model
    for epoch in range(n_iter):

        train_setID, train_X, _, _, _, _ = f_simulation.split_dataset(X_, Ys_[0][0], frac = frac, ranseed=epoch)

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
        
    return losses #modelD, 

