# -*- coding: utf-8 -*-
"""
Created on Sun Sep 11 01:38:56 2022

@author: aka2333
"""

# In[ ]:

# Import useful py libs
import os
import itertools

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
from numpy import genfromtxt
import scipy
from scipy import stats
from scipy.io import loadmat  # this is the SciPy module that loads mat-files

import h5py

import pandas as pd

import DataProcessingTools as DPT # https://github.com/grero/DataProcessingTools
import NeuralProcessingTools as NPT # https://github.com/grero/NeuralProcessingTools



# In[ ]:
class Get_paths(object):
    
    def __init__(self, name, parent):
        self.name = name
        self.parent = parent
    
    def monkey_path(self):
        monkey_path = self.parent + '/' + self.name.lower()
        return monkey_path
    
    def day_path(self):
        dayDirs = os.listdir(self.monkey_path())
        dayDirs1 = []
        for i in dayDirs:
            if os.path.isfile(self.monkey_path() + '/' + i) == False:
                dayDirs1 += [i]
        day_path = [self.monkey_path() + '/' + i for i in dayDirs1]
        return day_path
    
    def session_path(self):
        session_path = []
        for d in self.day_path():
            sessionDirs = os.listdir(d)
            sessionDirs1 = []
            for i in sessionDirs:
                if os.path.isfile(d + '/' + i) == False:
                    sessionDirs1 += [i]
            session_path += [d + '/' + i for i in sessionDirs1]    
        return session_path
    
    def array_path(self):
        array_path = []
        for s in self.session_path():
            arrayDirs = os.listdir(s)
            arrayDirs1 = []
            for i in arrayDirs:
                if os.path.isfile(s + '/' + i) == False:
                    arrayDirs1 += [i]
            array_path += [s + '/' + i for i in arrayDirs1]    
        return array_path

    def channel_path(self):
        channel_path = []
        for a in self.array_path():
            channelDirs = os.listdir(a)
            channelDirs1 = []
            for i in channelDirs:
                if os.path.isfile(a + '/' + i) == False:
                    channelDirs1 += [i]
            channel_path += [a + '/' + i for i in channelDirs1]    
        return channel_path

    def cell_path(self):
        cell_path = []
        for ch in self.channel_path():
            cellDirs = os.listdir(ch)
            cellDirs1 = []
            for i in cellDirs:
                if os.path.isfile(ch + '/' + i) == False:
                    cellDirs1 += [i]
            cell_path += [ch + '/' + i for i in cellDirs1]    
        return cell_path
    

class Get_trial(object):
    # specific to session
        
    def __init__(self, data_path, old = False):
        self.data_path = data_path
        
    
    
    def trial_event(self):
        session_path = self.data_path
        with DPT.misc.CWD(session_path):
            trials = NPT.trialstructures.WorkingMemoryTrials()
            
        trial_event = pd.DataFrame()
        trial_event['event'] = trials.events
        trial_event['timeStamp'] = trials.timestamps
        return trial_event
    
    
    def trial_segment(self): # time boundaries of each trial
        # first pick the index of all trial_start events
        trialStartX = []
        
        trial_event = self.trial_event()
        
        for i in range(len(trial_event)):
            temp = trial_event.iloc[i,]
            if temp['event'] == 'trial_start':
                trialStartX += [i]
            #elif temp['event'] == 'trial_end':
                #trialEndX += [i]

        # segmenting as [trial_start1: trial_start2], since each trial must start with a trial_start event
        trialSeg = []
        for s in range(len(trialStartX)):
            if s != len(trialStartX) - 1:
                sX1 = trialStartX[s]
                sX2 = trialStartX[s+1]
                
                trialSeg += [trial_event.iloc[sX1:sX2,].reset_index(drop = True)]
                
            else:
                sX1 = trialStartX[s]
                trialSeg += [trial_event.iloc[sX1:,].reset_index(drop = True)]
        
        
        faultTrial = []
        for t in range(len(trialSeg)):
            temp = trialSeg[t]
            if len(temp[temp.event == 'trial_end']) == 1:
                pass
            
            else:#if len(temp[temp.event == 'trial_end']) == 0 or len(temp[temp.event == 'trial_end']) > 1:
                faultTrial += [t]
                
                toDrop = list(temp[temp.event == 'trial_end'].index)[:-1]
                trialSeg[t] = temp.drop(toDrop).reset_index(drop = True)
                
        return trialSeg
    
    
        
    def trial_selection(self): #selection of valid trials only

        selectedTrials = []
        
        trial_segment = self.trial_segment()
        rawIdx = []
        for t in range(len(trial_segment)):
            temp = trial_segment[t]
            
            boolSOn = temp.event.str.contains('stimulus_on')
            boolSOff = temp.event.str.contains('stimulus_off')
            boolRsp = temp.event.str.contains('response')
            
            if True in list(boolSOn) and True in list(boolSOff) and True in list(boolRsp): #
                
                nSOn = boolSOn.value_counts()[True]
                nSOff = boolSOff.value_counts()[True]
                nRsp = boolRsp.value_counts()[True]
            
                if nSOn == 2 and nSOff == 2 and nRsp == 1:
                    selectedTrials += [temp]
                    rawIdx += [t]
                else:
                    pass
        return selectedTrials, rawIdx
    
    
    
    def trial_selection_df(self):
        
        corX = lambda l: 2 if l == 0 or l == 2 else -2
        corY = lambda l: 1.5 if l == 0 or l == 1 else -1.5
        corDis = lambda x1,y1,x2,y2: ((x1-x2)**2 + (y1-y2)**2)**0.5
        
        trial_selection, rawIdx = self.trial_selection()
        
        colsX = ['trial_index','raw_index']
        colsN = ['trial_start','fix_start','stimulus_1_on_type','stimulus_1_off_type','stimulus_2_on_type','stimulus_2_off_type',
                 'stimulus_1_on_loc','stimulus_1_off_loc','stimulus_2_on_loc','stimulus_2_off_loc','response_on','failure','reward_on','reward_off','trial_end']

        colsT = ['T_trial_start','T_fix_start','T_stimulus_1_on_type','T_stimulus_1_off_type','T_stimulus_2_on_type','T_stimulus_2_off_type',
                 'T_stimulus_1_on_loc','T_stimulus_1_off_loc','T_stimulus_2_on_loc','T_stimulus_2_off_loc','T_response_on','T_failure','T_reward_on','T_reward_off','T_trial_end']
        

        # stim_on_1(type)_2(location) ----> stim_1(order)_on_1_1_2
        # stimulus_displayOrder_switch_type_location1_location2

        cols = colsX + colsN + colsT

        selectedTrials_df = pd.DataFrame(columns = cols)

        for st in range(len(trial_selection)):
            temp = trial_selection[st]
            
            tempDict = {}
            tempDict['trial_index'] = st
            tempDict['raw_index'] = rawIdx[st]
            
            for c in colsN + colsT:
                
                cN = c.split('_')
                
                if cN[0] != 'T':
                    if cN[0] != 'stimulus':
                        tempDict[c] = 1 if c in list(temp.event) else 0
                        
                    else:
                        tempStim = temp[temp.event.str.contains('stimulus')].event.str.split('_', expand = True).reset_index(drop = True)
                        
                        if cN[-1] == 'type':
                            tempDict[c] = int(tempStim[2][0]) if cN[1] == '1' else int(tempStim[2][2])
                        elif cN[-1] == 'loc':
                            tempDict[c] = int(tempStim[3][0]) if cN[1] == '1' else int(tempStim[3][2])
                        
                        
                elif  cN[0] == 'T':
                    if cN[1] != 'stimulus':
                        tempDict[c] = temp[temp.event == c.split('_',1)[-1]].timeStamp.values[0] if c.split('_',1)[-1] in list(temp.event) else None
                    else:
                        tempStim = temp[temp.event.str.contains('stimulus')].reset_index(drop = True)
                        if cN[2] == '1':
                            tempDict[c] = tempStim.timeStamp[0] if cN [3] == 'on' else tempStim.timeStamp[1]
                        elif cN[2] == '2':
                            tempDict[c] = tempStim.timeStamp[2] if cN [3] == 'on' else tempStim.timeStamp[3]
            
            selectedTrials_df = selectedTrials_df._append(tempDict, ignore_index=True)

        #colsO = ['accuracy', 'conditionType', 'conditionLoc', 'conditionLoc1', 'conditionLoc2', 'conditionDistance', 'condition']

        selectedTrials_df['accuracy'] = selectedTrials_df['reward_on']
        selectedTrials_df['type'] = selectedTrials_df['stimulus_2_on_type'].astype('int')
        selectedTrials_df['locs'] = selectedTrials_df['stimulus_1_on_loc'].astype('int').astype('str') + '_' + selectedTrials_df['stimulus_2_on_loc'].astype('int').astype('str')
        selectedTrials_df['loc1'] = selectedTrials_df['stimulus_1_on_loc']
        selectedTrials_df['loc2'] = selectedTrials_df['stimulus_2_on_loc']
        
        selectedTrials_df.loc[(selectedTrials_df.type == 1), 'locKey'] = selectedTrials_df.loc[(selectedTrials_df.type == 1), 'loc2']
        selectedTrials_df.loc[(selectedTrials_df.type == 2), 'locKey'] = selectedTrials_df.loc[(selectedTrials_df.type == 2), 'loc1']
        
        x1 = [corX(selectedTrials_df.loc1[i]) for i in range(len(selectedTrials_df))]
        x2 = [corX(selectedTrials_df.loc2[i]) for i in range(len(selectedTrials_df))]
        y1 = [corY(selectedTrials_df.loc1[i]) for i in range(len(selectedTrials_df))]
        y2 = [corY(selectedTrials_df.loc2[i]) for i in range(len(selectedTrials_df))]
        
        dis = [corDis(x1[i], y1[i], x2[i], y2[i]) for i in range(len(selectedTrials_df))]
        
        selectedTrials_df['distance'] = dis
        
        #selectedTrials_df['condition'] = selectedTrials_df['conditionType'].astype('str') + '_' + selectedTrials_df['conditionLoc'].astype('str')

        # type condition: 1=tgt 2=dis
        # loc condition: loc1_loc2
        # condition: type_loc1_loc2

        selectedTrials_df['accuracy'] = selectedTrials_df['reward_on']
        
        selectedTrials_df['RT'] = 0
        for i in range(len(selectedTrials_df)):
            if selectedTrials_df.loc[i,'accuracy'] == 1:
                selectedTrials_df.loc[i,'RT'] = selectedTrials_df.loc[i,'T_reward_on'] - selectedTrials_df.loc[i,'T_response_on'] 
            else:
                selectedTrials_df.loc[i,'RT'] = selectedTrials_df.loc[i,'T_failure'] - selectedTrials_df.loc[i,'T_response_on']
        
        selectedTrials_df['id'] = 1
        selectedTrials_df['delay'] = (selectedTrials_df['T_response_on'] - selectedTrials_df['T_stimulus_2_off_type']).round(1)
        
        return selectedTrials_df


# specific to old version of task
class Get_trial_old(object):
    # specific to session
    
    
    def __init__(self, data_path, old = False):
        self.data_path = data_path
        self.old = old # if use oldWMTrial, for James/Pancake. Default 0, not to use old
    
    
    
    def trial_event(self):
        session_path = self.data_path
        with DPT.misc.CWD(session_path):
            trials = NPT.trialstructures.OldWorkingMemoryTrials(ncols=5, nrows=5)
            
        trial_event = pd.DataFrame()
        trial_event['event'] = trials.events
        trial_event['timeStamp'] = trials.timestamps
        return trial_event
    
    
    def trial_segment(self): # time boundaries of each trial
        # first pick the index of all trial_start events
        trialStartX = []
        
        trial_event = self.trial_event()
        
        for i in range(len(trial_event)):
            temp = trial_event.iloc[i,]
            if temp['event'] == 'trial_start':
                trialStartX += [i]
            #elif temp['event'] == 'trial_end':
                #trialEndX += [i]

        # segmenting as [trial_start1: trial_start2], since each trial must start with a trial_start event
        trialSeg = []
        for s in range(len(trialStartX)):
            if s != len(trialStartX) - 1:
                sX1 = trialStartX[s]
                sX2 = trialStartX[s+1]
                
                trialSeg += [trial_event.iloc[sX1:sX2,].reset_index(drop = True)]
                
            else:
                sX1 = trialStartX[s]
                trialSeg += [trial_event.iloc[sX1:,].reset_index(drop = True)]
        
        
        faultTrial = []
        for t in range(len(trialSeg)):
            temp = trialSeg[t]
            if len(temp[temp.event == 'trial_end']) == 1:
                pass
            
            elif len(temp[temp.event == 'trial_end']) == 0 or len(temp[temp.event == 'trial_end']) > 1:
                faultTrial += [t]
                
                toDrop = list(temp[temp.event == 'trial_end'].index)[:-1]
                trialSeg[t] = temp.drop(toDrop).reset_index(drop = True)
                
        return trialSeg
    
    
        
    def trial_selection(self): #selection of valid trials only

        selectedTrials = []
        
        trial_segment = self.trial_segment()
        
        for t in range(len(trial_segment)):
            temp = trial_segment[t]
            
            boolSOn = temp.event.str.contains('stimulus_on')
            #boolSOff = temp.event.str.contains('stimulus_off')
            boolRsp = temp.event.str.contains('response')
            
            if True in list(boolSOn) and True in list(boolRsp): #and True in list(boolSOff) 
                
                nSOn = boolSOn.value_counts()[True]
                #nSOff = boolSOff.value_counts()[True]
                nRsp = boolRsp.value_counts()[True]
            
                if nSOn == 2 and nRsp == 1: #and nSOff == 2 
                    selectedTrials += [temp]
                else:
                    pass
        return selectedTrials
    
    
    
    def trial_selection_df(self):
        
        corX = lambda l: 2 if l == 0 or l == 2 else -2
        corY = lambda l: 1.5 if l == 0 or l == 1 else -1.5
        corDis = lambda x1,y1,x2,y2: ((x1-x2)**2 + (y1-y2)**2)**0.5
        
        trial_selection = self.trial_selection()
        
        colsX = ['trial_index']
        colsN = ['trial_start','fix_start','stimulus_1_on_type','stimulus_2_on_type',
                 'stimulus_1_on_loc','stimulus_2_on_loc','response_on','failure','reward_on','reward_off','trial_end']

        colsT = ['T_trial_start','T_fix_start','T_stimulus_1_on_type','T_stimulus_2_on_type',
                 'T_stimulus_1_on_loc','T_stimulus_2_on_loc','T_response_on','T_failure','T_reward_on','T_reward_off','T_trial_end']
        

        # stim_on_1(type)_2(location) ----> stim_1(order)_on_1_1_2
        # stimulus_displayOrder_switch_type_location1_location2

        cols = colsX + colsN + colsT

        selectedTrials_df = pd.DataFrame(columns = cols)

        for st in range(len(trial_selection)):
            temp = trial_selection[st]
            
            tempDict = {}
            tempDict['trial_index'] = st
            
            for c in colsN + colsT:
                
                cN = c.split('_')
                
                if cN[0] != 'T':
                    if cN[0] != 'stimulus':
                        tempDict[c] = 1 if c in list(temp.event) else 0
                        
                    else:
                        tempStim = temp[temp.event.str.contains('stimulus')].event.str.split('_', expand = True).reset_index(drop = True)
                        
                        if cN[-1] == 'type':
                            tempDict[c] = int(tempStim[2][0]) if cN[1] == '1' else int(tempStim[2][1])
                        elif cN[-1] == 'loc':
                            tempDict[c] = int(tempStim[3][0]) if cN[1] == '1' else int(tempStim[3][1])
                        
                        
                elif  cN[0] == 'T':
                    if cN[1] != 'stimulus':
                        tempDict[c] = temp[temp.event == c.split('_',1)[-1]].timeStamp.values[0] if c.split('_',1)[-1] in list(temp.event) else None
                    else:
                        tempStim = temp[temp.event.str.contains('stimulus')].reset_index(drop = True)
                        if cN[2] == '1':
                            tempDict[c] = tempStim.timeStamp[0] #if cN [3] == 'on' else tempStim.timeStamp[1]
                        elif cN[2] == '2':
                            tempDict[c] = tempStim.timeStamp[1] #if cN [3] == 'on' else tempStim.timeStamp[3]
            
            selectedTrials_df = selectedTrials_df._append(tempDict, ignore_index=True)

        #colsO = ['accuracy', 'conditionType', 'conditionLoc', 'conditionLoc1', 'conditionLoc2', 'conditionDistance', 'condition']

        selectedTrials_df['accuracy'] = selectedTrials_df['reward_on']
        selectedTrials_df['type'] = selectedTrials_df['stimulus_2_on_type'].astype('int')
        selectedTrials_df['locs'] = selectedTrials_df['stimulus_1_on_loc'].astype('int').astype('str') + '_' + selectedTrials_df['stimulus_2_on_loc'].astype('int').astype('str')
        selectedTrials_df['loc1'] = selectedTrials_df['stimulus_1_on_loc']
        selectedTrials_df['loc2'] = selectedTrials_df['stimulus_2_on_loc']
        
        selectedTrials_df.loc[(selectedTrials_df.type == 1), 'locKey'] = selectedTrials_df.loc[(selectedTrials_df.type == 1), 'loc2']
        selectedTrials_df.loc[(selectedTrials_df.type == 2), 'locKey'] = selectedTrials_df.loc[(selectedTrials_df.type == 2), 'loc1']
        
        x1 = [corX(selectedTrials_df.loc1[i]) for i in range(len(selectedTrials_df))]
        x2 = [corX(selectedTrials_df.loc2[i]) for i in range(len(selectedTrials_df))]
        y1 = [corY(selectedTrials_df.loc1[i]) for i in range(len(selectedTrials_df))]
        y2 = [corY(selectedTrials_df.loc2[i]) for i in range(len(selectedTrials_df))]
        
        dis = [corDis(x1[i], y1[i], x2[i], y2[i]) for i in range(len(selectedTrials_df))]
        
        selectedTrials_df['distance'] = dis
        
        #selectedTrials_df['condition'] = selectedTrials_df['conditionType'].astype('str') + '_' + selectedTrials_df['conditionLoc'].astype('str')

        # type condition: 1=tgt 2=dis
        # loc condition: loc1_loc2
        # condition: type_loc1_loc2

        selectedTrials_df['accuracy'] = selectedTrials_df['reward_on']

        selectedTrials_df['RT'] = 0
        for i in range(len(selectedTrials_df)):
            if selectedTrials_df.loc[i,'accuracy'] == 1:
                selectedTrials_df.loc[i,'RT'] = selectedTrials_df.loc[i,'T_reward_on'] - selectedTrials_df.loc[i,'T_response_on'] 
            else:
                selectedTrials_df.loc[i,'RT'] = selectedTrials_df.loc[i,'T_failure'] - selectedTrials_df.loc[i,'T_response_on']
        
        selectedTrials_df['id'] = 1
        
        return selectedTrials_df


# In[] Fit spike timestamps into each trial

class Get_spikes(object):
    
    def __init__(self, cell_path):
        self.cell_path = cell_path
    
    def spike_load(self):
        cell_path = self.cell_path
        mat_path = cell_path + '/unit.mat'
        
        if h5py.is_hdf5(mat_path):
            mat = {}
            temp = h5py.File(mat_path)  # load mat-file old version
            for k,v in temp.items():
                mat[k] = np.array(v)
        else:
            mat = loadmat(mat_path)  # load mat-file
        return mat
    
    #def spike_load_old(self):
    #    cell_path = self.cell_path
        
    #    mat = {}
    #    temp = h5py.File(cell_path+'/unit.mat')  # load mat-file old version
    #    for k,v in temp.items():
    #        mat[k] = np.array(v)
            
    #    return mat
    
    def spike_timeStamp(self):
        mat = self.spike_load()
        mTsp = mat['timestamps']
        mTsp = mTsp.reshape([mTsp.size,]) / 1000
        return mTsp
    
    #def spike_timeStamp_old(self):
    #    mat = self.spike_load_old()
    #    mTsp = mat['timestamps'][0]/1000
    #    return mTsp
    
    def spike_form(self):
        mat = self.spike_load()
        mSpk = mat['spikeForm'][0]
        return mSpk

# In[]

def spike_selection(mTsp, trial_df, trial_idx):
    mTsp = mTsp
    
    #selectedTrials = Get_trial(session_path).trial_selection()
    trial_df = trial_df
    
    #selectedTrials_spikes = {1:[],2:[]}
    selectedSpikes = []
    for st in trial_idx:
        temp = trial_df[trial_df.trial_index == st]
        tStart, tEnd = float(temp.T_trial_start), float(temp.T_trial_end)
        tP1 = float(temp.T_stimulus_1_on_type)
        #conditionT = selectedTrials_df.conditionType[st]
        
        spikes = mTsp[np.logical_and(mTsp > tStart, mTsp < tEnd)] - tP1 # time zero set as stimulus presentation 1
        
        selectedSpikes += [spikes]
        
        #if conditionT == 1:
        #    selectedTrials_spikes[1] += [spikes]
        #elif conditionT == 2:
        #    selectedTrials_spikes[2] += [spikes]
    
    return selectedSpikes
    
    
# In[] get frequencies


def spike2freq(spikesS, tRangeRaw, step, slide):
    # spikesS array of selected spike train
    
    tRange = np.arange(tRangeRaw[0], tRangeRaw[-1], slide)
    
    arrayFreqT = np.zeros([len(spikesS), len(tRange)]) #rows * columns
    
    for tx in range(len(spikesS)):
        t0 = tRangeRaw[0] # starting point
        x = 0
        
        while (t0 >= tRangeRaw[0]) and (t0 + slide <= tRangeRaw[-1]):

            sTrainT = spikesS[tx] * 1000
            
            if step>=0:
                sTrain_slice = sTrainT[np.logical_and(sTrainT >= t0, sTrainT < t0 + step)]
            else:
                sTrain_slice = sTrainT[np.logical_and(sTrainT <= t0, sTrainT > t0 + step)]
            
            # start from a new row, then add
            arrayFreqT[tx, x] = (len(sTrain_slice)/abs(step)) * (1000)
            
            x += 1
            t0 += slide
    
    return arrayFreqT

# In[] 
def tCutArray(tRange, tsp1, tsp2, arrayFreq):
    
    idxBsl = [np.where(tRange==tsp1)[0][0], np.where(tRange==tsp2)[0][0]]
    cut = arrayFreq[:,idxBsl[0]:idxBsl[1]]
    
    return cut


# In[] baseline correction for firing rate array

def bslFreqA(tRange, tBsl1, tBsl2, arrayFreq):
    
    #idxBsl = [np.abs(tRange - tBsl1).argmin(), np.abs(tRange - tBsl2).argmin()]
    
    cut = tCutArray(tRange, tBsl1, tBsl2, arrayFreq)#arrayFreq[:,idxBsl[0]:idxBsl[1]]
    
    arrayFreqB = arrayFreq - np.mean(cut, axis = 1).reshape(cut.shape[0],1)
    
    return arrayFreqB

# In[] baseline correction for firing rate array

def bslFreqV(tRange, tBsl1, tBsl2, arrayFreq):
    
    #idxBsl = [np.abs(tRange - tBsl1).argmin(), np.abs(tRange - tBsl2).argmin()]
    cut = tCutArray(tRange, tBsl1, tBsl2, arrayFreq)
    
    #bslFreqV = np.mean(cut, axis = 1)
    bslFreqV = np.mean(cut)
    
    return bslFreqV

# In[]

def bslTTest_1samp(arrayFreq_E, bslV):
    
    tempDic = {}
    
    bn = 0
    while bn < np.shape(arrayFreq_E)[-1]:
    #for c in range(np.shape(arrayFreq_E)[-1]):
        tout = stats.ttest_1samp(arrayFreq_E[:,bn], bslV, alternative = 'two-sided')
        tempDic[bn] = {'t':float(tout[0]), 'p':float(tout[-1])}
        bn += 1
    
    tempDf = pd.DataFrame(tempDic).T
    
    return tempDf
# In[]

def bslTTest_paired(arrayFreq_E, bslA):
    
    tempDic = {}
    
    bn = 0
    while bn < np.shape(arrayFreq_E)[-1]:
    #for c in range(np.shape(arrayFreq_E)[-1]):
        tout = stats.ttest_ind(arrayFreq_E[:,bn], bslA, alternative = 'two-sided')
        tempDic[bn] = {'t':float(tout[0]), 'p':float(tout[-1])}
        bn += 1
        
    tempDf = pd.DataFrame(tempDic).T
    
    return tempDf
# In[]

def bslTTest_paired_mean(arrayFreq_E, bslA):
    
    tempDic = {}
    
    #bn = 0
    #while bn < np.shape(arrayFreq_E)[-1]:
    #for c in range(np.shape(arrayFreq_E)[-1]):
    tout = stats.ttest_ind(arrayFreq_E.mean(axis = 1), bslA, alternative = 'two-sided')
    tempDic = {'t':float(tout[0]), 'p':float(tout[-1])}
    #bn += 1
        
    #tempDf = pd.DataFrame(tempDic).T
    
    return tempDic


