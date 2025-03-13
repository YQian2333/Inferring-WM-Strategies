#%% import libraries
%reload_ext autoreload
%autoreload 2

import os
from itertools import permutations, combinations, product
import time
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np

from functions import Get_paths, Get_trial
import f_pseudoPop

#%%

############################################
######### create pseudo population #########
############################################

#%% set load and save paths
data_path = 'D:/data' # change to your own data path
pseudoPop_path = 'D:/data/pseudoPop' # change to your own save path
dt = 10 # sampling rate

#%% Monkey A Info
# Data file structure: monkey > day > session > array > channel > cell
wz = Get_paths('wangzi', data_path) # here change the monkey name to match the data folder name
wz_monkey_path = wz.monkey_path()
wz_session_paths = wz.session_path()
wz_cell_paths = wz.cell_path()

# manual inspections to exclude invalid cells & sessions with small trial size
wz_exclusionList = ['20200106/session02/array01/channel017/cell01', '20200106/session02/array03/channel066/cell01', '20200106/session02/array04/channel121/cell01', '20200121/session01/array02/channel062/cell03']
for ex in wz_exclusionList:    
    if (wz_monkey_path + '/' + ex) in wz_cell_paths:
        wz_cell_paths.remove(wz_monkey_path + '/' + ex)
    
wz_session_drop = [f'{data_path}/wangzi/20210729/session02',f'{data_path}/wangzi/20210818/session02',
                f'{data_path}/wangzi/20210819/session02',f'{data_path}/wangzi/20210826/session02',
                f'{data_path}/wangzi/20210830/session02',f'{data_path}/wangzi/20210906/session02',
                f'{data_path}/wangzi/20211005/session02',f'{data_path}/wangzi/20211011/session02',
                f'{data_path}/wangzi/20211007/session02',]
for i in wz_session_drop:
    if i in wz_session_paths:
        wz_session_paths.remove(i)
    
# trial time series parameters and event markers
wz_epochsDic = {'bsl':[-300,0],'s1':[0,400],'d1':[400,1400],'s2':[1400,1800],'d2':[1800,2800],'go':[2800,3200]}
wz_trialEvtsBoundary = sorted([b[1] for b in wz_epochsDic.values()])
wz_tRange = np.arange(-300, wz_trialEvtsBoundary[-1], dt)
wz_tBsl = wz_epochsDic['bsl']
wz_tslices = ((-300,0), (0,400), (400, 1300), (1400, 1800), (1800, 2700), (2800, 3200))

# define arrays corresponding to LPFC and FEF(PAC)
wz_dlpfcArrays = (1,2)
wz_fefArrays = (3,)

# wrap up Monkey A Info
wangzi_Info = {'subject': 'wangzi', 'monkey_path': wz.monkey_path(), 'session_paths': wz.session_path(), 'cell_paths': wz.cell_path(),
                'exclusionList': wz_exclusionList, 'session_drop': wz_session_drop, 
                'epochsDic': wz_epochsDic, 'trialEvtsBoundary': wz_trialEvtsBoundary, 'tRange': wz_tRange, 'tBsl': wz_tBsl, 'tslices': wz_tslices,
                'dlpfcArrays': wz_dlpfcArrays, 'fefArrays': wz_fefArrays}


#%% Monkey B Info
# Data file structure: monkey > day > session > array > channel > cell
whis = Get_paths('whiskey', data_path) # here change the monkey name to match the data folder name
whis_monkey_path = whis.monkey_path()
whis_session_paths = whis.session_path()
whis_cell_paths = whis.cell_path()

# manual inspections to exclude invalid cells & sessions with small trial size
whis_exclusionList = ['20200106/session02/array01/channel017/cell01', '20200106/session02/array03/channel066/cell01', '20200106/session02/array04/channel121/cell01', '20200121/session01/array02/channel062/cell03']
for ex in whis_exclusionList:    
    if (whis_monkey_path + '/' + ex) in whis_cell_paths:
        whis_cell_paths.remove(whis_monkey_path + '/' + ex)

whis_session_drop = []
for i in whis_session_drop:
    if i in whis_session_paths:
        whis_session_paths.remove(i)

# trial time series parameters and event markers
whis_epochsDic = {'bsl':[-300,0],'s1':[0,300],'d1':[300,1300],'s2':[1300,1600],'d2':[1600,2600],'go':[2600,3000]}
whis_trialEvtsBoundary = sorted([b[1] for b in whis_epochsDic.values()])
whis_tRange = np.arange(-300, whis_trialEvtsBoundary[-1], dt)
whis_tBsl = whis_epochsDic['bsl']
whis_tslices = ((-300,0), (0,300), (300, 1300), (1300, 1600), (1600, 2600), (2600,3000))

# define arrays corresponding to LPFC and FEF(PAC)
whis_dlpfcArrays = (1,3)
whis_fefArrays = (2,4)

# wrap up Monkey B Info
whiskey_Info = {'subject': 'whiskey', 'monkey_path': whis.monkey_path(), 'session_paths': whis.session_path(), 'cell_paths': whis.cell_path(),
                'exclusionList': whis_exclusionList, 'session_drop': whis_session_drop, 
                'epochsDic': whis_epochsDic, 'trialEvtsBoundary': whis_trialEvtsBoundary, 'tRange': whis_tRange, 'tBsl': whis_tBsl, 'tslices': whis_tslices,
                'dlpfcArrays': whis_dlpfcArrays, 'fefArrays': whis_fefArrays}


#%% pool both monkeys
monkey_Info = {'whiskey': whiskey_Info, 'wangzi': wangzi_Info, }
monkey_names = list(monkey_Info.keys())[0] if len(monkey_Info)==1 else 'all'

# pool sessions and cells
sessions = []
cellsToUse = []

for m, mDict in monkey_Info.items():
    monkey_path = mDict['monkey_path']
    cell_paths = mDict['cell_paths']
    session_paths = mDict['session_paths']
    exclusionList = mDict['exclusionList']
    session_drop = mDict['session_drop']
    
    for ss in session_drop:
        if ss in session_paths:
            session_paths.remove(ss)
    
    for c in exclusionList:
        if (monkey_path + '/' + c) in cell_paths:
            cell_paths.remove(monkey_path + '/' + c)
    
    sessions += session_paths
    cellsToUse += cell_paths


#%% pseudo population sample parameters

# create conditions
locs = (0,1,2,3,)
locCombs = list(permutations(locs,2))
ttypes = (1,2,)
subConditions = list(product(locCombs, ttypes))

# count for minimal number of trial per locComb across all sessions
trialCounts = {s:[] for s in subConditions}
for session in sessions:
    
    cells = [cc for cc in cellsToUse if '/'.join(cc.split('/')[:-3]) == session]
    
    trial_df = Get_trial(session).trial_selection_df()
    trial_df = trial_df[(trial_df.accuracy == 1)].reset_index(drop = True)
    
    for k in sorted(trialCounts.keys()):
        loc1, loc2 = k[0][0], k[0][1]
        tt = k[1]
        temp = trial_df[(trial_df.type == tt)&(trial_df.loc1 == loc1)&(trial_df.loc2 == loc2)].reset_index(drop = True)
        trialCounts[k] += [len(temp)]
    
    counts1 = sorted(trial_df[trial_df.type==1].locs.value_counts())
    counts2 = sorted(trial_df[trial_df.type==2].locs.value_counts())
    
    print(f'{session}, nCells = {len(cells)}')
    print(f'Counts Ret: {counts1}')
    print(f'Counts Dis: {counts2}')

trialCounts = {v:min(k) for v, k in trialCounts.items()}
trialMin = min(trialCounts.values()) # minimal number of trials per condition


#%% create pseudo populations

samplePerCon = int(trialMin*1) # number of trials per condition to sample from each session, dependent on the trialMin
sampleRounds = 1 # number of rounds to sample from each session
arti_noise_level = 0 # if need to add artificial gaussian noises to the samples

nIters = 100 # number of pseudo populations to generate

for n in range(nIters):
    t_IterOn = time.time()
    print(f'Iter = {n}')
    
    pseudo_TrialInfo = f_pseudoPop.pseudo_Session(locCombs, ttypes, samplePerCon=samplePerCon, sampleRounds=sampleRounds) # generate pseudo session trial info
    pseudo_region = f_pseudoPop.pseudo_PopByRegion_pooled(sessions, cellsToUse, monkey_Info, locCombs = locCombs, ttypes = ttypes, 
                                                          samplePerCon = samplePerCon, sampleRounds=sampleRounds, arti_noise_level = arti_noise_level) # generate pseudo population dataset
    
    # wrap up pseudo population
    pseudo_data = {'pseudo_TrialInfo':pseudo_TrialInfo, 'pseudo_region':pseudo_region}
    np.save(pseudoPop_path + f'/pseudo_{monkey_names}{n}.npy', pseudo_data, allow_pickle=True) # save pseudo population data
    
    print(f'{time.time()-t_IterOn:.3f}s') # show time cost for each iteration
