
#%%
import numpy as np
from scipy import stats

# turn off warning messages
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

#%%
def omega2(data, trialInfo, iv):
    ssTotal = ((data - data.mean(axis=0))**2).sum(axis=0)
    ssBtw = []
    mse = []
    conditions = sorted(trialInfo[iv].unique())
    df = len(conditions)-1
    epsilon = 1e-07
    
    for l in conditions:
        idxx = trialInfo[trialInfo[iv]==l].index.tolist()
        n_group = len(idxx)
        ssBtw += [len(idxx) * ((data[idxx,:].mean(axis=0) - data.mean(axis=0))**2)]
        mse += [(data[idxx,:] - data[idxx,:].mean(axis=0))**2]
    
    ssBtw = np.array(ssBtw).sum(axis=0)
    mse = np.concatenate(mse,axis=0).mean(axis=0)
    omega2 = (ssBtw - (df*mse))/(ssTotal + mse + epsilon)

    return omega2
#%%
def LDAPerformance(train_data, test_data,  train_label, test_label):
    # create an instance of the LDA classifier
    clf = LinearDiscriminantAnalysis()        
    # fit the training data and their corresponding labels
    clf.fit(train_data, train_label)
    # predict the labels for the test data    
    performance = clf.score(test_data,test_label)
    
    return performance


#%%
def stability_ratio(pfm):
    ondiag = np.diag(pfm)
    off_diagonal_mask = ~np.eye(len(pfm), dtype=bool)
    off_diag = pfm[off_diagonal_mask]
    
    stab_ratio = off_diag.mean() / ondiag.mean()
    
    return stab_ratio

def code_morphing(pfm1, pfm2):
    code_morph = pfm1.mean() / pfm2.mean()
    return code_morph


def code_transferability(pfm12, pfm21, tbins):
    transferability = np.zeros((len(tbins),len(tbins)))
    for t in range(len(tbins)):
        for t_ in range(len(tbins)):
            transferability[t,t_] = np.mean((pfm12[t,t_],pfm21[t_,t]))
    return transferability