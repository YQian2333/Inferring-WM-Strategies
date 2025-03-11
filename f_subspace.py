# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 15:21:05 2024

@author: aka2333
"""
# In[ ]:
from itertools import permutations, combinations, product
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import scipy
from scipy import stats


import matplotlib.pyplot as plt
import re, seaborn as sns

import sklearn
from sklearn.decomposition import PCA
from scipy.ndimage import gaussian_filter,gaussian_filter1d

import sklearn
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from scipy.ndimage import gaussian_filter1d

# In[]

def LDAPerformance(train_data, test_data,  train_label, test_label):
    # create an instance of the LDA classifier
    clf = LinearDiscriminantAnalysis()        
    clf.fit(train_data, train_label)    
    performance = clf.score(test_data,test_label)
    
    return performance

    
def permutation_p(x, permuted_xs, tail = 'two'):
    x, permuted_xs = x.round(5), permuted_xs.round(5)
    
    if tail == 'two':
        pvalue = (50-abs(stats.percentileofscore(permuted_xs, x)-50))*2/100 # two-tail
        
    elif tail == 'greater':
        pvalue = (100-stats.percentileofscore(permuted_xs, x))/100 # one-tail greater than
        
    elif tail == 'smaller':
        pvalue = (stats.percentileofscore(permuted_xs, x))/100 # one-tail smaller than
    return pvalue

def permutation_pCI(xs, permuted_xs, CI_size = 1, value_range = (0,1), CI_range = False):
    xs, permuted_xs = xs.round(5), permuted_xs.round(5)
    if bool(CI_range):
        b1, b2 = stats.scoreatpercentile(xs, CI_range)
    else:
        b1, b2 = (xs.mean() - CI_size * xs.std()), (xs.mean() + CI_size * xs.std())
        b1, b2 = np.max([b1, value_range[0]]), np.min([b2, value_range[1]]) # not exceed value range
    
    
    if b1 != b2:
        pvalue = (stats.percentileofscore(permuted_xs, b2) - stats.percentileofscore(permuted_xs, b1))/100 # one-tail smaller than
    else:
        pvalue = 1
    return pvalue


#%%
def plane_by_vecs(vecs, center = (0,0,0), scaleX = 1, scaleY = 1, xRange = [-1,1], yRange = [-1,1]):
    #
    epsilon = 1e-7
    
    vecs = vecs #(2 components * 3 features)
    vec1, vec2 = vecs[0,:], vecs[1,:] #(3 features,)
    vec_normal = np.cross(vec1, vec2) #(1 * 3)
    vec_normal = vec_normal / np.linalg.norm(vec_normal) if np.linalg.norm(vec_normal)!= 0 else vec_normal / (np.linalg.norm(vec_normal) + epsilon) #(1 * 3)
    
    d = -vec_normal[0]*center[0] - vec_normal[1]*center[1] - vec_normal[2]*center[2]
    
    
    x_plane, y_plane = np.meshgrid(np.linspace(xRange[0], xRange[1], 10), np.linspace(yRange[0], yRange[1], 10))
    z_plane = (-vec_normal[0]*x_plane - vec_normal[1]*y_plane - d) / vec_normal[2]
    
    return x_plane, y_plane, z_plane

def angle_btw_vecs(vec_normal1, vec_normal2):
    cos_theta = (vec_normal1 @ vec_normal2) / (np.linalg.norm(vec_normal1) * np.linalg.norm(vec_normal2))
    sin_theta = np.linalg.norm(np.cross(vec_normal1, vec_normal2)) / (np.linalg.norm(vec_normal1) * np.linalg.norm(vec_normal2))
    
    return cos_theta, sin_theta

def proj_on_plane(p, vec_normal, center):
    return p - (vec_normal @ (p - center))*vec_normal

def proj_2D_coordinates(P, vecs):
    #, center
    epsilon = 1e-7
    vec1, vec2 = vecs
    vec1_ = vec1 / np.linalg.norm(vec1) if np.linalg.norm(vec1)!=0 else vec1 / (np.linalg.norm(vec1) + epsilon)
    vec2_ = vec2 / np.linalg.norm(vec2) if np.linalg.norm(vec2)!=0 else vec2 / (np.linalg.norm(vec2) + epsilon)
    
    if len(P.shape)>1:
        P_new = np.zeros((P.shape[0],2))
        for i in range(P.shape[0]):
            P_new[i] = [np.dot(P[i], vec1_), np.dot(P[i], vec2_)]
    else:
        P_new = np.array([np.dot(P, vec1_), np.dot(P, vec2_)])
    
    return P_new


def split_setID(trialIDs, frac=0.5, ranseed = 233):
    np.random.seed(ranseed)
    nTrials = len(trialIDs)
    train_setID = np.sort(np.random.choice(nTrials, round(frac*nTrials),replace = False))
    test_setID = np.setdiff1d(np.arange(nTrials), train_setID, assume_unique=True)
    
    return train_setID, test_setID


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
    #mse = np.concatenate(mse,axis=0).sum(axis=0)
    mse = np.concatenate(mse,axis=0).mean(axis=0)
    omega2 = (ssBtw - (df*mse))/(ssTotal + mse + epsilon)

    return omega2


def angle_by_cossin(cos_angle, sin_angle, angle_range = (-np.pi, np.pi)):
    if angle_range == (-np.pi, np.pi):
        angle = np.arccos(cos_angle.round(7)) if sin_angle.round(7) > 0 else -1*np.arccos(cos_angle.round(7))
    elif angle_range == (0, 2*np.pi):
        angle = np.arccos(cos_angle.round(7)) if sin_angle.round(7) > 0 else 2*np.pi - np.arccos(cos_angle.round(7))
    
    return angle


def get_rotation_matrix(angle, axis):
    
    # Construct the rotation matrix using axis-angle representation
    c = np.cos(angle)
    s = np.sin(angle)
    t = 1 - c
    x, y, z = axis
    R = np.array([[t*x*x + c, t*x*y - z*s, t*x*z + y*s], 
                  [t*x*y + z*s, t*y*y + c, t*y*z - x*s],
                  [t*x*z - y*s, t*y*z + x*s, t*z*z + c]])
    
    return R
#%%
def phase_alignment(projs1, vecs1, projs2, vecs2):
    
    epsilon = 1e-07
    ###
    projs1C, projs2C = (projs1 - projs1.mean(axis=0)), (projs2 - projs2.mean(axis=0))
    projs1_2d, projs2_2d = proj_2D_coordinates(projs1C, vecs1).T, proj_2D_coordinates(projs2C, vecs2).T
    
    R, s = scipy.linalg.orthogonal_procrustes(projs1_2d, projs2_2d) # input A (rotated), B (reference); return R so that minimize (A@R)-B
    
    cosPsi, sinPsi = R[0,0], R[1,0]
    
    psi = -1*np.arctan(R[1,0]/R[0,0]) # R[1,0] = sin, R[0,0] = cos
    
    projs2_2dR = np.dot(R.T, projs2_2d)# * s
    
    disparity = np.sum((projs1_2d - projs2_2d)**2)
    disparityR = np.sum((projs1_2d - projs2_2dR)**2)
    
    return psi #cosPsi,  #, disparity



#%%
def vec_quad_side(projs, sequence = (3,1,2,0)):
    epsilon = 1e-7
    # default order of config vertices as the spatial arrangement of out task stimuli (0: UR; 1: UL; 2: LR; 3: LL)
    vecX = projs[sequence[1]] - projs[sequence[0]]
    vecY = projs[sequence[2]] - projs[sequence[0]]
    vecX = vecX / np.linalg.norm(vecX) if np.linalg.norm(vecX) !=0 else vecX / (np.linalg.norm(vecX)+ epsilon)
    vecY = vecY / np.linalg.norm(vecY) if np.linalg.norm(vecY) !=0 else vecY / (np.linalg.norm(vecY)+ epsilon)
    return vecX, vecY


def basis_correction_3d(vecs, vecX, vecY): 
    
    # correct plane basis depending on the two vectors, in this analysis the two edges defining the quadralateral.
    # after correction the direction of the new X axis should align with the order of quadralater vertices config, 
    # and the new Y axis align with the counter-route. In this way, the direction of normal vector of the plane will 
    # be dependent to the config sequence, which can subsequently help detect plane flipping (i.e. rotated > +-90) using cosTheta
    
    epsilon = 1e-7
    vec1, vec2 = vecs
    vecN = np.cross(vec1, vec2)
    vecN = vecN / np.linalg.norm(vecN) if np.linalg.norm(vecN) != 0 else vecN / (np.linalg.norm(vecN) + epsilon) # Axis of rotation
    vecX_ = vecX / np.linalg.norm(vecX) if np.linalg.norm(vecX) !=0 else vecX / (np.linalg.norm(vecX)+ epsilon) # Normalize the vectors
    
    R1 = get_rotation_matrix((np.pi*0.5), vecN)
    R2 = get_rotation_matrix((np.pi*1.5), vecN)
    
    vecY_1 = np.dot(R1, vecX) # new vec rotated from vec1 that is collinear to vecX
    vecY_2 = np.dot(R2, vecX)
    
    vecY_ = vecY_1 if ((vecY_1 - vecY)**2).sum() < ((vecY_2 - vecY)**2).sum() else vecY_2 # by squared error
    vecY_ = vecY_ / np.linalg.norm(vecY_) if np.linalg.norm(vecY_) !=0 else vecY_ / (np.linalg.norm(vecY_)+ epsilon)
    
    return vecX_, vecY_

# In[]
def angle_btw_planes(vecs1, projs1, vecs2, projs2, sequence = (3,1,2,0)):
    epsilon = 1e-7
    vecX1, vecY1 = vec_quad_side(projs1, sequence = sequence) 
    vecX2, vecY2 = vec_quad_side(projs2, sequence = sequence)
    
    vecs1_new = np.array(basis_correction_3d(vecs1, vecX1, vecY1))
    vecs2_new = np.array(basis_correction_3d(vecs2, vecX2, vecY2))
    
    vecN1_new, vecN2_new = np.cross(vecs1_new[0], vecs1_new[1]), np.cross(vecs2_new[0], vecs2_new[1])
    vecN1_new = vecN1_new / np.linalg.norm(vecN1_new) if np.linalg.norm(vecN1_new) !=0 else vecN1_new / (np.linalg.norm(vecN1_new) + epsilon)
    vecN2_new = vecN2_new / np.linalg.norm(vecN2_new) if np.linalg.norm(vecN2_new) !=0 else vecN2_new / (np.linalg.norm(vecN2_new) + epsilon)
    cos_theta, sin_theta = angle_btw_vecs(vecN1_new, vecN2_new) # angle to rotate plane1 to be coplanar to plane2
    
    return cos_theta, sin_theta #theta, 

# In[]
def projsAll_to_2d(vecs1, projs1, projs1_all, sequence = (3,1,2,0)):
    vecX1, vecY1 = vec_quad_side(projs1, sequence = sequence) 
    vecs1_new = np.array(basis_correction_3d(vecs1, vecX1, vecY1))
    
    projs1_all_2d = proj_2D_coordinates(projs1_all, vecs1_new)
    
    return projs1_all_2d

# In[]
def plane_decodability(vecs1, projs1, projs1_all, trialInfoT, iv, method = 'lda', sequence = (3,1,2,0), include2d=False):
    #shuff_excludeInv = True, Y_columnsLabels = ['locKey','locs','type','loc1','loc2','locX'], toDecode_labels = 'locX', tt = 1
    vecX1, vecY1 = vec_quad_side(projs1, sequence = sequence) 
    vecs1_new = np.array(basis_correction_3d(vecs1, vecX1, vecY1))
    
    projs1_all = projs1_all
    projs1_all_2d = proj_2D_coordinates(projs1_all, vecs1_new)
    
    if method == 'omega2':
        # if use 3d coords
        decodability_3d = omega2(projs1_all, trialInfoT, iv)
        
        decodability_2d = []
        if include2d:# if use 2d coords
            decodability_2d = omega2(projs1_all_2d, trialInfoT, iv)
        
        
    elif method == 'lda':
        
        
        train_id, test_id = split_setID(np.arange(projs1_all.shape[0])) #, ranseed = n
        train_label1, test_label1 = trialInfoT.loc[train_id, iv], trialInfoT.loc[test_id, iv]
        
        decodability_3d = LDAPerformance(projs1_all[train_id,:], projs1_all[test_id,:], train_label1, test_label1)
        
        decodability_2d = []
        if include2d:
        # if use 2d coords
            decodability_2d = LDAPerformance(projs1_all_2d[train_id,:], projs1_all_2d[test_id,:], train_label1, test_label1)
        
    return decodability_3d, decodability_2d

# In[]
def plane_decodability_trans(geoms1, geoms2, sequence = (3,1,2,0), include2d=False):
    #shuff_excludeInv = True, Y_columnsLabels = ['locKey','locs','type','loc1','loc2','locX'], toDecode_labels = 'locX', tt = 1
    
    vecs1, projs1, projs1_all, trialInfo1T, iv1 = geoms1
    vecs2, projs2, projs2_all, trialInfo2T, iv2 = geoms2
    
    vecX1, vecY1 = vec_quad_side(projs1, sequence = sequence) 
    vecs1_new = np.array(basis_correction_3d(vecs1, vecX1, vecY1))
    projs1_all = projs1_all
    projs1_all_2d = proj_2D_coordinates(projs1_all, vecs1_new)
    
    vecX2, vecY2 = vec_quad_side(projs2, sequence = sequence) 
    vecs2_new = np.array(basis_correction_3d(vecs2, vecX2, vecY2))
    projs2_all = projs2_all
    projs2_all_2d = proj_2D_coordinates(projs2_all, vecs2_new)
    
    #train_id, test_id = split_setID(np.arange(projs1_all.shape[0])) #, ranseed = n
    label1, label2 = trialInfo1T.loc[:, iv1], trialInfo2T.loc[:, iv2]
    
    decodability_3d = LDAPerformance(projs1_all, projs2_all, label1, label2)
    
    decodability_2d = []
    if include2d:
    # if use 2d coords
        decodability_2d = LDAPerformance(projs1_all_2d, projs2_all_2d, label1, label2)
        
    return decodability_3d, decodability_2d


#%%
def plane_decodability_LDA(geoms1, geoms2, trialInfoT, iv, sequence = (3,1,2,0), use2d = False):
    
    vecs1, projs1, projs1_all = geoms1
    

    projs1_all = projs1_all
    
    vecs2, projs2, projs2_all = geoms2

    
    projs2_all = projs2_all
    
    # LDA
    train_id, test_id = split_setID(np.arange(projs1_all.shape[0])) #, ranseed = n
    train_label1, test_label1 = trialInfoT.loc[train_id, iv], trialInfoT.loc[test_id, iv]
    
    # if use 3d coords
    decodability_3d = LDAPerformance(projs1_all[train_id,:], projs2_all[test_id,:], train_label1, test_label1)
    
    # if use 2d coords
    decodability_2d = np.zeros_like(decodability_3d)
    if use2d:
            
        vecX1, vecY1 = vec_quad_side(projs1, sequence = sequence) 
        vecs1_new = np.array(basis_correction_3d(vecs1, vecX1, vecY1))
        projs1_all_2d = proj_2D_coordinates(projs1_all, vecs1_new)

        vecX2, vecY2 = vec_quad_side(projs2, sequence = sequence) 
        vecs2_new = np.array(basis_correction_3d(vecs2, vecX2, vecY2))
        projs2_all_2d = proj_2D_coordinates(projs2_all, vecs2_new)
        
        
        decodability_2d = LDAPerformance(projs1_all_2d[train_id,:], projs2_all_2d[test_id,:], train_label1, test_label1)
    
    return decodability_3d, decodability_2d


def plane_decodability_LDA_choice(geoms1, geoms2, trialInfoT1, trialInfoT2, iv1, iv2, sequence = (3,1,2,0), use2d = False):
    
    vecs1, projs1, projs1_all = geoms1
    projs1_all = projs1_all
    

    vecs2, projs2, projs2_all = geoms2
    projs2_all = projs2_all
    
    # LDA
    #train_id, test_id = split_setID(np.arange(projs1_all.shape[0])) #, ranseed = n
    train_label, test_label = trialInfoT1.loc[:, iv1], trialInfoT2.loc[:, iv2]
    
    # if use 3d coords
    decodability_3d = LDAPerformance(projs1_all[:,:], projs2_all[:,:], train_label, test_label)
    
    # if use 2d coords
    decodability_2d = np.zeros_like(decodability_3d)
    if use2d:
            
        vecX1, vecY1 = vec_quad_side(projs1, sequence = sequence) 
        vecs1_new = np.array(basis_correction_3d(vecs1, vecX1, vecY1))
        projs1_all_2d = proj_2D_coordinates(projs1_all, vecs1_new)

        vecX2, vecY2 = vec_quad_side(projs2, sequence = sequence) 
        vecs2_new = np.array(basis_correction_3d(vecs2, vecX2, vecY2))
        projs2_all_2d = proj_2D_coordinates(projs2_all, vecs2_new)
        
        decodability_2d = LDAPerformance(projs1_all_2d[:,:], projs2_all_2d[:,:], train_label, test_label)
    
    return decodability_3d, decodability_2d


# In[]
def config_alignment(vecs1, projs1, vecs2, projs2, sequence = (3,1,2,0), scaling = False):
    epsilon = 1e-7
    
    vecX1, vecY1 = vec_quad_side(projs1, sequence = sequence) 
    vecX2, vecY2 = vec_quad_side(projs2, sequence = sequence)
    
    vecs1_new = np.array(basis_correction_3d(vecs1, vecX1, vecY1))
    vecs2_new = np.array(basis_correction_3d(vecs2, vecX2, vecY2))
    
    vecN1_new, vecN2_new = np.cross(vecs1_new[0], vecs1_new[1]), np.cross(vecs2_new[0], vecs2_new[1])
    vecN1_new = vecN1_new / np.linalg.norm(vecN1_new) if np.linalg.norm(vecN1_new) !=0 else vecN1_new / (np.linalg.norm(vecN1_new) + epsilon)
    vecN2_new = vecN2_new / np.linalg.norm(vecN2_new) if np.linalg.norm(vecN2_new) !=0 else vecN2_new / (np.linalg.norm(vecN2_new) + epsilon)
    
    # config alignment test
    # project to 2d coords within corresponding plane
    projs1_2d = proj_2D_coordinates(projs1, vecs1_new)
    projs2_2d = proj_2D_coordinates(projs2, vecs2_new)
    
    projs1_2d = projs1_2d - projs1_2d.mean(axis=0)
    projs2_2d = projs2_2d - projs2_2d.mean(axis=0)
    
    R, s = scipy.linalg.orthogonal_procrustes(projs1_2d, projs2_2d) # input A (rotated), B (reference); return R so that minimize (A@R)-B
    
    cos_psi, sin_psi = R[0,0], R[1,0]
    #psi = np.sign(np.arcsin(sin_psi)) * np.arccos(cos_psi)
    
    if scaling:
        projs1_2d_R = np.array([np.dot(R, projs1_2d[i]) for i in range(projs1_2d.shape[0])]) * s
        #projs1_2d_R = np.dot(projs1_2d, R.T) * s
        projs2_2d_R = np.array([np.dot(R.T, projs2_2d[i]) for i in range(projs2_2d.shape[0])]) * s
    else:
        projs1_2d_R = np.array([np.dot(R, projs1_2d[i]) for i in range(projs1_2d.shape[0])]) # * s
        #projs1_2d_R = np.dot(projs1_2d, R.T) * s
        projs2_2d_R = np.array([np.dot(R.T, projs2_2d[i]) for i in range(projs2_2d.shape[0])]) # * s
        
    sse = (np.sum((projs1_2d_R - projs2_2d)**2) + np.sum((projs1_2d - projs2_2d_R)**2))/2
    #np.sum((projs1_2d - projs2_2d_R)**2) / projs1_2d.shape[0]
    
    return cos_psi, sin_psi, sse #psi, 




def config_alignment_coplanar(vecs1, projs1, vecs2, projs2, sequence = (3,1,2,0), scaling = False):
    epsilon = 1e-7
    
    vecX1, vecY1 = vec_quad_side(projs1, sequence = sequence) 
    vecX2, vecY2 = vec_quad_side(projs2, sequence = sequence)
    
    vecs1_new = np.array(basis_correction_3d(vecs1, vecX1, vecY1))
    vecs2_new = np.array(basis_correction_3d(vecs2, vecX2, vecY2))
    
    vecN1_new, vecN2_new = np.cross(vecs1_new[0], vecs1_new[1]), np.cross(vecs2_new[0], vecs2_new[1])
    vecN1_new = vecN1_new / np.linalg.norm(vecN1_new) if np.linalg.norm(vecN1_new) !=0 else vecN1_new / (np.linalg.norm(vecN1_new) + epsilon)
    vecN2_new = vecN2_new / np.linalg.norm(vecN2_new) if np.linalg.norm(vecN2_new) !=0 else vecN2_new / (np.linalg.norm(vecN2_new) + epsilon)
    
    cos_theta, sin_theta = angle_btw_vecs(vecN1_new, vecN2_new) # angle to rotate plane1 to be coplanar to plane2
    theta = angle_by_cossin(cos_theta.round(5),sin_theta.round(5)) # theta radius from -pi to pi
    
    vecNN = np.cross(vecN1_new, vecN2_new) # normal vec for the vecNs, use as axis for rotating planes
    vecNN = vecNN / np.linalg.norm(vecNN) if np.linalg.norm(vecNN) !=0 else vecNN / (np.linalg.norm(vecNN)+ epsilon)
    #if cos_theta < 0:
    #    # exclude plane pairs with >90 angles, as will be flipped & intersecting shapes
    #    cos_psi, sin_psi = np.nan, np.nan
    #    sse = np.nan
    #    #R_plane = get_rotation_matrix(np.pi - theta, vecNN)
        
    #else:
    # rotate plane1 to be coplanar with plane2
    R_plane = get_rotation_matrix(theta, vecNN)
    
    #vecs1_coplanar = np.array([np.dot(R_plane, vecs1_new[i])/np.linalg.norm(np.dot(R_plane, vecs1_new[i])) for i in range(vecs1_new.shape[0])])
    #vecN1_coplanar = np.cross(vecs1_coplanar[0], vecs1_coplanar[1]) / np.linalg.norm(np.cross(vecs1_coplanar[0], vecs1_coplanar[1]))
    projs1_coplanar = np.array([np.dot(R_plane, projs1[i]) for i in range(projs1.shape[0])])

    ### config alignment test
    # project to 2d coords within corresponding plane
    # use basis of plane2 for both planes' projection, so that comparable within the same ref sys
    projs1_2d = proj_2D_coordinates(projs1_coplanar, vecs2_new)
    projs2_2d = proj_2D_coordinates(projs2, vecs2_new)
    
    #center to mean
    projs1_2d = projs1_2d - projs1_2d.mean(axis=0)
    projs2_2d = projs2_2d - projs2_2d.mean(axis=0)
    
    R, s = scipy.linalg.orthogonal_procrustes(projs1_2d, projs2_2d) # input A (rotated), B (reference); return R so that minimize (A@R)-B
    
    cos_psi, sin_psi = R[0,0], R[1,0]
    #psi = np.sign(np.arcsin(sin_psi)) * np.arccos(cos_psi)
    
    if scaling:
        projs1_2d_R = np.array([np.dot(R, projs1_2d[i]) for i in range(projs1_2d.shape[0])]) * s
        #projs1_2d_R = np.dot(projs1_2d, R.T) * s
        projs2_2d_R = np.array([np.dot(R.T, projs2_2d[i]) for i in range(projs2_2d.shape[0])]) * s
    else:
        projs1_2d_R = np.array([np.dot(R, projs1_2d[i]) for i in range(projs1_2d.shape[0])]) # * s
        #projs1_2d_R = np.dot(projs1_2d, R.T) * s
        projs2_2d_R = np.array([np.dot(R.T, projs2_2d[i]) for i in range(projs2_2d.shape[0])]) # * s
        
    sse = (np.sum((projs1_2d_R - projs2_2d)**2) + np.sum((projs1_2d - projs2_2d_R)**2))/2
    #np.sum((projs1_2d - projs2_2d_R)**2) / projs1_2d.shape[0]
        
    return cos_psi, sin_psi, sse #psi, 


# In[]
### 
def angle_alignment_coplanar(vecs1, projs1, vecs2, projs2, sequence = (3,1,2,0), scaling = False):
    epsilon = 1e-7
    
    vecX1, vecY1 = vec_quad_side(projs1, sequence = sequence) 
    vecX2, vecY2 = vec_quad_side(projs2, sequence = sequence)
    
    vecs1_new = np.array(basis_correction_3d(vecs1, vecX1, vecY1))
    vecs2_new = np.array(basis_correction_3d(vecs2, vecX2, vecY2))
    
    vecN1_new, vecN2_new = np.cross(vecs1_new[0], vecs1_new[1]), np.cross(vecs2_new[0], vecs2_new[1])
    vecN1_new = vecN1_new / np.linalg.norm(vecN1_new) if np.linalg.norm(vecN1_new) !=0 else vecN1_new / (np.linalg.norm(vecN1_new) + epsilon)
    vecN2_new = vecN2_new / np.linalg.norm(vecN2_new) if np.linalg.norm(vecN2_new) !=0 else vecN2_new / (np.linalg.norm(vecN2_new) + epsilon)
    
    cos_theta, sin_theta = angle_btw_vecs(vecN1_new, vecN2_new) # angle to rotate plane1 to be coplanar to plane2
    theta = angle_by_cossin(cos_theta.round(5) ,sin_theta.round(5)) # theta radius from -pi to pi
    
    vecNN = np.cross(vecN1_new, vecN2_new) # normal vec for the vecNs, use as axis for rotating planes
    vecNN = vecNN / np.linalg.norm(vecNN) if np.linalg.norm(vecNN) !=0 else vecNN / (np.linalg.norm(vecNN)+ epsilon)
    #if cos_theta < 0:
    #    # exclude plane pairs with >90 angles, as will be flipped & intersecting shapes
    #    cos_psi, sin_psi = np.nan, np.nan
    #    sse = np.nan
        #R_plane = get_rotation_matrix(np.pi - theta, vecNN)
        
    #else:
    # rotate plane1 to be coplanar with plane2
    R_plane = get_rotation_matrix(theta, vecNN)
    
    #vecs1_coplanar = np.array([np.dot(R_plane, vecs1_new[i])/np.linalg.norm(np.dot(R_plane, vecs1_new[i])) for i in range(vecs1_new.shape[0])])
    #vecN1_coplanar = np.cross(vecs1_coplanar[0], vecs1_coplanar[1]) / np.linalg.norm(np.cross(vecs1_coplanar[0], vecs1_coplanar[1]))
    projs1_coplanar = np.array([np.dot(R_plane, projs1[i]) for i in range(projs1.shape[0])])

    ### config alignment test
    # project to 2d coords within corresponding plane
    # use basis of plane2 for both planes' projection, so that comparable within the same ref sys
    projs1_2d = proj_2D_coordinates(projs1_coplanar, vecs2_new)
    projs2_2d = proj_2D_coordinates(projs2, vecs2_new)
    
    #center to mean
    projs1_2d = projs1_2d - projs1_2d.mean(axis=0)
    projs2_2d = projs2_2d - projs2_2d.mean(axis=0)
    
    R, s = scipy.linalg.orthogonal_procrustes(projs1_2d, projs2_2d) # input A (rotated), B (reference); return R so that minimize (A@R)-B
    
    cos_psi, sin_psi = R[0,0], R[1,0]
    #psi = np.sign(np.arcsin(sin_psi)) * np.arccos(cos_psi)
    
    if scaling:
        projs1_2d_R = np.array([np.dot(R, projs1_2d[i]) for i in range(projs1_2d.shape[0])]) * s
        #projs1_2d_R = np.dot(projs1_2d, R.T) * s
        projs2_2d_R = np.array([np.dot(R.T, projs2_2d[i]) for i in range(projs2_2d.shape[0])]) * s
    else:
        projs1_2d_R = np.array([np.dot(R, projs1_2d[i]) for i in range(projs1_2d.shape[0])]) # * s
        #projs1_2d_R = np.dot(projs1_2d, R.T) * s
        projs2_2d_R = np.array([np.dot(R.T, projs2_2d[i]) for i in range(projs2_2d.shape[0])]) # * s
        
    sse = (np.sum((projs1_2d_R - projs2_2d)**2) + np.sum((projs1_2d - projs2_2d_R)**2))/2
    #np.sum((projs1_2d - projs2_2d_R)**2) / projs1_2d.shape[0]
        
    return cos_theta, sin_theta, cos_psi, sin_psi, sse


def angle_alignment(vecs1, projs1, vecs2, projs2, sequence = (3,1,2,0), scaling = False):
    epsilon = 1e-7
    vecX1, vecY1 = vec_quad_side(projs1, sequence = sequence) 
    vecX2, vecY2 = vec_quad_side(projs2, sequence = sequence)
    
    vecs1_new = np.array(basis_correction_3d(vecs1, vecX1, vecY1))
    vecs2_new = np.array(basis_correction_3d(vecs2, vecX2, vecY2))
    
    vecN1_new, vecN2_new = np.cross(vecs1_new[0], vecs1_new[1]), np.cross(vecs2_new[0], vecs2_new[1])
    vecN1_new = vecN1_new / np.linalg.norm(vecN1_new) if np.linalg.norm(vecN1_new) !=0 else vecN1_new / (np.linalg.norm(vecN1_new) + epsilon)
    vecN2_new = vecN2_new / np.linalg.norm(vecN2_new) if np.linalg.norm(vecN2_new) !=0 else vecN2_new / (np.linalg.norm(vecN2_new) + epsilon)
    
    cos_theta, sin_theta = angle_btw_vecs(vecN1_new, vecN2_new) # angle to rotate plane1 to be coplanar to plane2
    theta = angle_by_cossin(cos_theta.round(5),sin_theta.round(5)) # theta radius from -pi to pi
    
    # config alignment test
    # project to 2d coords within corresponding plane
    projs1_2d = proj_2D_coordinates(projs1, vecs1_new)
    projs2_2d = proj_2D_coordinates(projs2, vecs2_new)
    
    projs1_2d = projs1_2d - projs1_2d.mean(axis=0)
    projs2_2d = projs2_2d - projs2_2d.mean(axis=0)
    
    R, s = scipy.linalg.orthogonal_procrustes(projs1_2d, projs2_2d) # input A (rotated), B (reference); return R so that minimize (A@R)-B
    
    cos_psi, sin_psi = R[0,0], R[1,0]
    #psi = np.sign(np.arcsin(sin_psi)) * np.arccos(cos_psi)
    
    if scaling:
        projs1_2d_R = np.array([np.dot(R, projs1_2d[i]) for i in range(projs1_2d.shape[0])]) * s
        #projs1_2d_R = np.dot(projs1_2d, R.T) * s
        projs2_2d_R = np.array([np.dot(R.T, projs2_2d[i]) for i in range(projs2_2d.shape[0])]) * s
    else:
        projs1_2d_R = np.array([np.dot(R, projs1_2d[i]) for i in range(projs1_2d.shape[0])]) # * s
        #projs1_2d_R = np.dot(projs1_2d, R.T) * s
        projs2_2d_R = np.array([np.dot(R.T, projs2_2d[i]) for i in range(projs2_2d.shape[0])]) # * s
        
    sse = (np.sum((projs1_2d_R - projs2_2d)**2) + np.sum((projs1_2d - projs2_2d_R)**2))/2
    #np.sum((projs1_2d - projs2_2d_R)**2) / projs1_2d.shape[0]
        
    return cos_theta, sin_theta, cos_psi, sin_psi, sse

# In[]
### 
def angle_alignment_coplanar_nan(vecs1, projs1, vecs2, projs2, sequence = (3,1,2,0), scaling = False):
    epsilon = 1e-7
    
    vecX1, vecY1 = vec_quad_side(projs1, sequence = sequence) 
    vecX2, vecY2 = vec_quad_side(projs2, sequence = sequence)
    
    vecs1_new = np.array(basis_correction_3d(vecs1, vecX1, vecY1))
    vecs2_new = np.array(basis_correction_3d(vecs2, vecX2, vecY2))
    
    vecN1_new, vecN2_new = np.cross(vecs1_new[0], vecs1_new[1]), np.cross(vecs2_new[0], vecs2_new[1])
    vecN1_new = vecN1_new / np.linalg.norm(vecN1_new) if np.linalg.norm(vecN1_new) !=0 else vecN1_new / (np.linalg.norm(vecN1_new) + epsilon)
    vecN2_new = vecN2_new / np.linalg.norm(vecN2_new) if np.linalg.norm(vecN2_new) !=0 else vecN2_new / (np.linalg.norm(vecN2_new) + epsilon)
    
    cos_theta, sin_theta = angle_btw_vecs(vecN1_new, vecN2_new) # angle to rotate plane1 to be coplanar to plane2
    theta = angle_by_cossin(cos_theta.round(5) ,sin_theta.round(5)) # theta radius from -pi to pi
    
    vecNN = np.cross(vecN1_new, vecN2_new) # normal vec for the vecNs, use as axis for rotating planes
    vecNN = vecNN / np.linalg.norm(vecNN) if np.linalg.norm(vecNN) !=0 else vecNN / (np.linalg.norm(vecNN)+ epsilon)
    
    if cos_theta < 0:
        # exclude plane pairs with >90 angles, as will be flipped & intersecting shapes
        cos_psi, sin_psi = np.nan, np.nan
        sse = np.nan
        #R_plane = get_rotation_matrix(np.pi - theta, vecNN)
        
    else:
    # rotate plane1 to be coplanar with plane2
        R_plane = get_rotation_matrix(theta, vecNN)
        
        #vecs1_coplanar = np.array([np.dot(R_plane, vecs1_new[i])/np.linalg.norm(np.dot(R_plane, vecs1_new[i])) for i in range(vecs1_new.shape[0])])
        #vecN1_coplanar = np.cross(vecs1_coplanar[0], vecs1_coplanar[1]) / np.linalg.norm(np.cross(vecs1_coplanar[0], vecs1_coplanar[1]))
        projs1_coplanar = np.array([np.dot(R_plane, projs1[i]) for i in range(projs1.shape[0])])
    
        ### config alignment test
        # project to 2d coords within corresponding plane
        # use basis of plane2 for both planes' projection, so that comparable within the same ref sys
        projs1_2d = proj_2D_coordinates(projs1_coplanar, vecs2_new)
        projs2_2d = proj_2D_coordinates(projs2, vecs2_new)
        
        #center to mean
        projs1_2d = projs1_2d - projs1_2d.mean(axis=0)
        projs2_2d = projs2_2d - projs2_2d.mean(axis=0)
        
        R, s = scipy.linalg.orthogonal_procrustes(projs1_2d, projs2_2d) # input A (rotated), B (reference); return R so that minimize (A@R)-B
        
        cos_psi, sin_psi = R[0,0], R[1,0]
        #psi = np.sign(np.arcsin(sin_psi)) * np.arccos(cos_psi)
        
        if scaling:
            projs1_2d_R = np.array([np.dot(R, projs1_2d[i]) for i in range(projs1_2d.shape[0])]) * s
            #projs1_2d_R = np.dot(projs1_2d, R.T) * s
            projs2_2d_R = np.array([np.dot(R.T, projs2_2d[i]) for i in range(projs2_2d.shape[0])]) * s
        else:
            projs1_2d_R = np.array([np.dot(R, projs1_2d[i]) for i in range(projs1_2d.shape[0])]) # * s
            #projs1_2d_R = np.dot(projs1_2d, R.T) * s
            projs2_2d_R = np.array([np.dot(R.T, projs2_2d[i]) for i in range(projs2_2d.shape[0])]) # * s
            
        sse = (np.sum((projs1_2d_R - projs2_2d)**2) + np.sum((projs1_2d - projs2_2d_R)**2))/2
        #np.sum((projs1_2d - projs2_2d_R)**2) / projs1_2d.shape[0]
        
    return cos_theta, sin_theta, cos_psi, sin_psi, sse


def angle_alignment_nan(vecs1, projs1, vecs2, projs2, sequence = (3,1,2,0), scaling = False):
    epsilon = 1e-7
    vecX1, vecY1 = vec_quad_side(projs1, sequence = sequence) 
    vecX2, vecY2 = vec_quad_side(projs2, sequence = sequence)
    
    vecs1_new = np.array(basis_correction_3d(vecs1, vecX1, vecY1))
    vecs2_new = np.array(basis_correction_3d(vecs2, vecX2, vecY2))
    
    vecN1_new, vecN2_new = np.cross(vecs1_new[0], vecs1_new[1]), np.cross(vecs2_new[0], vecs2_new[1])
    vecN1_new = vecN1_new / np.linalg.norm(vecN1_new) if np.linalg.norm(vecN1_new) !=0 else vecN1_new / (np.linalg.norm(vecN1_new) + epsilon)
    vecN2_new = vecN2_new / np.linalg.norm(vecN2_new) if np.linalg.norm(vecN2_new) !=0 else vecN2_new / (np.linalg.norm(vecN2_new) + epsilon)
    
    cos_theta, sin_theta = angle_btw_vecs(vecN1_new, vecN2_new) # angle to rotate plane1 to be coplanar to plane2
    theta = angle_by_cossin(cos_theta.round(5),sin_theta.round(5)) # theta radius from -pi to pi
    
    if cos_theta < 0:
        # exclude plane pairs with >90 angles, as will be flipped & intersecting shapes
        cos_psi, sin_psi = np.nan, np.nan
        sse = np.nan
        #R_plane = get_rotation_matrix(np.pi - theta, vecNN)
    else:
        # config alignment test
        # project to 2d coords within corresponding plane
        projs1_2d = proj_2D_coordinates(projs1, vecs1_new)
        projs2_2d = proj_2D_coordinates(projs2, vecs2_new)
        
        projs1_2d = projs1_2d - projs1_2d.mean(axis=0)
        projs2_2d = projs2_2d - projs2_2d.mean(axis=0)
        
        R, s = scipy.linalg.orthogonal_procrustes(projs1_2d, projs2_2d) # input A (rotated), B (reference); return R so that minimize (A@R)-B
        
        cos_psi, sin_psi = R[0,0], R[1,0]
        #psi = np.sign(np.arcsin(sin_psi)) * np.arccos(cos_psi)
        
        if scaling:
            projs1_2d_R = np.array([np.dot(R, projs1_2d[i]) for i in range(projs1_2d.shape[0])]) * s
            #projs1_2d_R = np.dot(projs1_2d, R.T) * s
            projs2_2d_R = np.array([np.dot(R.T, projs2_2d[i]) for i in range(projs2_2d.shape[0])]) * s
        else:
            projs1_2d_R = np.array([np.dot(R, projs1_2d[i]) for i in range(projs1_2d.shape[0])]) # * s
            #projs1_2d_R = np.dot(projs1_2d, R.T) * s
            projs2_2d_R = np.array([np.dot(R.T, projs2_2d[i]) for i in range(projs2_2d.shape[0])]) # * s
            
        sse = (np.sum((projs1_2d_R - projs2_2d)**2) + np.sum((projs1_2d - projs2_2d_R)**2))/2
        #np.sum((projs1_2d - projs2_2d_R)**2) / projs1_2d.shape[0]
        
    return cos_theta, sin_theta, cos_psi, sin_psi, sse

# In[]
def cos_similarity(p1, p2):
    cosSimi = np.dot(p1.T, p2) / (np.linalg.norm(p1) * np.linalg.norm(p2))
    return cosSimi

def config_correlation(vecs1, projs1, vecs2, projs2, sequence = (3,1,2,0)):
    # as per panichello & buschman 2021
    
    # basis corrected
    epsilon = 1e-7
    vecX1, vecY1 = vec_quad_side(projs1, sequence = sequence) 
    vecX2, vecY2 = vec_quad_side(projs2, sequence = sequence)
    
    vecs1_new = np.array(basis_correction_3d(vecs1, vecX1, vecY1))
    vecs2_new = np.array(basis_correction_3d(vecs2, vecX2, vecY2))
    
    vecN1_new, vecN2_new = np.cross(vecs1_new[0], vecs1_new[1]), np.cross(vecs2_new[0], vecs2_new[1])
    vecN1_new = vecN1_new / np.linalg.norm(vecN1_new) if np.linalg.norm(vecN1_new) !=0 else vecN1_new / (np.linalg.norm(vecN1_new) + epsilon)
    vecN2_new = vecN2_new / np.linalg.norm(vecN2_new) if np.linalg.norm(vecN2_new) !=0 else vecN2_new / (np.linalg.norm(vecN2_new) + epsilon)
    
    projs1_2d = proj_2D_coordinates(projs1, vecs1_new)
    projs2_2d = proj_2D_coordinates(projs2, vecs2_new)
    
    # recenterd
    #projs1, projs2 = projs1 - projs1.mean(0), projs2 - projs2.mean(0)
    
    projs1_2d = projs1_2d - projs1_2d.mean(axis=0)
    projs2_2d = projs2_2d - projs2_2d.mean(axis=0)
    
    #cosSimi_3d = np.array([cos_similarity(projs1[i], projs2[i]) for i in range(projs1.shape[0])]).mean()
    cosSimi_2d = np.array([cos_similarity(projs1_2d[i], projs2_2d[i]) for i in range(projs1_2d.shape[0])]).mean()
    return cosSimi_2d



# In[]
def plane_fitting_analysis(dataT, trialInfo, pca_tWinX, checkpoints, tsliceRange, avgInterval, locs, ttypes, dropCombs=[], 
                           adaptPCA = None, adaptEVR = None, toPlot=False, avgMethod = 'conditional_time'):
    #, decode_method = 'lda'
    epsilon = 1e-7
    vecs_D = {}
    projs_D = {}
    projsAll_D = {}
    
    trialInfo_D = {}
    X_mean = {}
    #pca2 = {}
    
    ncells = dataT.shape[1]
    
    for cp in checkpoints:
        vecs_D[cp] = {}
        projs_D[cp] = {}
        projsAll_D[cp] = {}
        
        X_mean[cp] = {}
        #pca2[cp] = {}
        
        for tt in ttypes:
            vecs_D[cp][tt] = {1:[], 2:[]}
            projs_D[cp][tt] = {1:[], 2:[]}
            projsAll_D[cp][tt] = {1:[], 2:[]}
            
            X_mean[cp][tt] = {1:[], 2:[]}
            #pca2[cp][tt] = {1:[], 2:[]}
    
    for tt in ttypes:
        trialInfo_D[tt] = []
    
    
    #locs = (0,1,2,3,)
    locCombs = list(permutations(locs,2))
    #ttypes = (1,2,)
                
    subConditions = list(product(locCombs, ttypes))
    
    # remove invalid combos
    subConditions_copy = subConditions.copy()
    for sc in subConditions_copy:
        if sc in dropCombs:
            subConditions.remove(sc)
    
    ### main analysis
    # condition average data. within each ttype, shape = (4*3) * ncells * nt
    
    dataT_slice = dataT[:,:,pca_tWinX]
    
    # all trial, time averaged
    if avgMethod == 'all':
        dataT_avg = dataT_slice.mean(axis=0)
    
    # all trial, time separate
    elif avgMethod == 'none':
        dataT_avg = []
        for trial in range(dataT_slice.shape[0]):
            dataT_avg += [dataT_slice[trial,:,:]]
        
        dataT_avg = np.hstack(dataT_avg)
    
    # conditional, time separate
    elif avgMethod == 'conditional':
        dataT_avg = []
        #dataT_avg_full = []
        
        for sc in subConditions:
            lc, tt = sc[0], sc[1]
            slc = '_'.join([str(l) for l in lc])
            idxx = trialInfo[(trialInfo.locs == slc)&(trialInfo.type == tt)].index.tolist()
            
            x = dataT_slice[idxx,:,:].mean(axis=0) #.mean(axis=-1)
            
            dataT_avg += [x]
            #dataT_avg_full += [x]
    
        # stack as ncells * (nt * (4*3))
        dataT_avg = np.hstack(dataT_avg)
    
    # conditional, time averaged
    elif avgMethod == 'conditional_time':
        dataT_avg = []
        
        for sc in subConditions:
            lc, tt = sc[0], sc[1]
            slc = '_'.join([str(l) for l in lc])
            idxx = trialInfo[(trialInfo.locs == slc)&(trialInfo.type == tt)].index.tolist()
            
            x = dataT_slice[idxx,:,:].mean(axis=0).mean(axis=-1) #
            
            dataT_avg += [x]
            #dataT_avg_full += [x]
    
        # stack as ncells * (4*3)
        dataT_avg = np.vstack(dataT_avg).T
    
    ###
    # 1st order pca, reduce from ncells * (nt*12) -> 3pcs * (nt*12) to create a 3pc space shared by all locComb conditions, across all timepoints
    dataT_3pc = []
    
    if (adaptPCA is not None) and (adaptPCA.shape==(3,ncells)):
        # all trials (ntrials * npc * ntime)
        for trial in range(dataT.shape[0]):
            dataT_3pc += [np.dot(dataT[trial,:,:].T, adaptPCA.T).T]
        
        dataT_3pc = np.array(dataT_3pc)
        
        evr_1st = np.ones(3,) if adaptEVR is None else adaptEVR # dummy, refer to the adapted PCA results
        pca1 = adaptPCA.copy()
        
    else:
        if adaptPCA is not None:
            print('Error: None input of adaptPCA or Shape not match. Will use default method.')
        
        pca_1st = PCA(n_components=3)
    
        # fit the PCA model to the data
        pca_1st.fit(dataT_avg.T)
        
        evr_1st = pca_1st.explained_variance_ratio_
        
        pca1 = pca_1st.components_
        
        # all trials (ntrials * npc * ntime)
        
        for trial in range(dataT.shape[0]):
            dataT_3pc += [pca_1st.transform(dataT[trial,:,:].T).T]
        
        dataT_3pc = np.array(dataT_3pc)

    
    # conditional mean (ncondition * npc * ntime)
    #dataT_3pc_meanT = pca_1st.transform(dataT_avg_full.T).T
    
    dataT_3pc_mean = []
    for sc in subConditions:
        lc, tt = sc[0], sc[1]
        slc = '_'.join([str(l) for l in lc])
        idxx = trialInfo[(trialInfo.locs == slc)&(trialInfo.type == tt)].index.tolist()
        
        dataT_3pc_mean += [dataT_3pc[idxx,:,:].mean(axis=0)]
        
    dataT_3pc_mean = np.array(dataT_3pc_mean) # reshape as (4*3) * 3pcs * nt
    
    for tt in ttypes:
        
        idx = trialInfo[(trialInfo.type == tt)].index.tolist()
        
        trialInfoT = trialInfo[(trialInfo.type == tt)].reset_index(drop=True)
        
        trialInfo_D[tt] = trialInfoT
        
        for cp in checkpoints:
            if len(avgInterval) == 1:
                t1 = tsliceRange.tolist().index(cp-avgInterval) if cp-avgInterval >= tsliceRange.min() else tsliceRange.tolist().index(tsliceRange.min())
                t2 = tsliceRange.tolist().index(cp+avgInterval) if cp+avgInterval <= tsliceRange.max() else tsliceRange.tolist().index(tsliceRange.max())
            
            elif (len(avgInterval) == len(checkpoints)) and (type(avgInterval) == dict):
                aInt = avgInterval[cp] # when avgInterval as a dict with keys of corresponding checkpoints
                
                t1 = tsliceRange.tolist().index(cp-aInt) if cp-aInt >= tsliceRange.min() else tsliceRange.tolist().index(tsliceRange.min())
                t2 = tsliceRange.tolist().index(cp+aInt) if cp+aInt <= tsliceRange.max() else tsliceRange.tolist().index(tsliceRange.max())
            
            
            tempX = dataT_3pc[:,:,t1:t2].mean(axis=2) if t1!=t2 else dataT_3pc[:,:,t1]
            
            tempX_mean = dataT_3pc_mean[:,:,t1:t2].mean(axis=2) if t1!=t2 else dataT_3pc_mean[:,:,t1]
            
            dataT_mean = dataT[idx,:,t1:t2].mean(axis=2) if t1!=t2 else dataT[idx,:,t1]
            
            tempX_tt = tempX[idx,:]
            
            tempX1_mean = []
            tempX2_mean = []
            
            dataT_mean1 = []
            dataT_mean2 = []
            
            for l in locs:
                
                conx1 = [subConditions.index(sc) for sc in subConditions if (sc[0][0] == l and sc[1] == tt)]
                conx2 = [subConditions.index(sc) for sc in subConditions if (sc[0][1] == l and sc[1] == tt)]
            
                tempX1_mean += [tempX_mean[conx1,:].mean(axis = 0)]
                tempX2_mean += [tempX_mean[conx2,:].mean(axis = 0)]
                
                dataT_mean1 += [dataT_3pc_mean[conx1,:].mean(axis = 0)]
                dataT_mean2 += [dataT_3pc_mean[conx2,:].mean(axis = 0)]
                
            
            tempX1_mean = np.array(tempX1_mean)
            tempX2_mean = np.array(tempX2_mean)
            
            dataT_mean1 = np.array(dataT_mean1)
            dataT_mean2 = np.array(dataT_mean2)
            
            ### loc1 2nd pca
            pca_2nd_1 = PCA(n_components=2)
            pca_2nd_1.fit(tempX1_mean)# - tempX1_mean.mean()
            vecs1, evr2_1 = pca_2nd_1.components_, pca_2nd_1.explained_variance_ratio_
            vec_normal1 = np.cross(vecs1[0], vecs1[1])
            vec_normal1 = vec_normal1 / np.linalg.norm(vec_normal1) if np.linalg.norm(vec_normal1) !=0 else vec_normal1 / (np.linalg.norm(vec_normal1)+ epsilon)
            
            # plane centered at mean of conditional mean
            center1 = tempX1_mean.mean(axis=0)
            
            # projects of conditional means on the plane, 3d
            proj1 = np.array([proj_on_plane(p1, vec_normal1, center1) for p1 in tempX1_mean])
            
            # create plane grids
            x1_plane, y1_plane, z1_plane = plane_by_vecs(vecs1, center = center1, xRange=(proj1[:,0].min(), proj1[:,0].max()), yRange=(proj1[:,1].min(), proj1[:,1].max()))
            
            # projects of single trials on the plane, 3d
            #tempX1_pca = pca_2nd_1.transform(tempX_tt) # 2nd pca transformed data
            tempX1_proj = np.array([proj_on_plane(p1, vec_normal1, center1) for p1 in tempX_tt]) # projections on the 2nd pca plane
            
            
            
            ### loc2 2nd pca
            pca_2nd_2 = PCA(n_components=2)
            pca_2nd_2.fit(tempX2_mean)# - tempX2_mean.mean()
            vecs2, evr2_2 = pca_2nd_2.components_, pca_2nd_2.explained_variance_ratio_
            vec_normal2 = np.cross(vecs2[0], vecs2[1])
            vec_normal2 = vec_normal2 / np.linalg.norm(vec_normal2) if np.linalg.norm(vec_normal2) !=0 else vec_normal2 / (np.linalg.norm(vec_normal2)+ epsilon)
            
            # plane centered at mean of conditional mean
            center2 = tempX2_mean.mean(axis=0)
            
            # projects of conditional means on the plane, 3d
            proj2 = np.array([proj_on_plane(p2, vec_normal2, center2) for p2 in tempX2_mean])
            
            # create plane grids
            x2_plane, y2_plane, z2_plane = plane_by_vecs(vecs2, center = center2, xRange=(proj2[:,0].min(), proj2[:,0].max()), yRange=(proj2[:,1].min(), proj2[:,1].max()))
            
            # projects of single trials on the plane, 3d
            #tempX2_pca = pca_2nd_2.transform(tempX_tt) # 2nd pca transformed data
            tempX2_proj = np.array([proj_on_plane(p2, vec_normal2, center2) for p2 in tempX_tt]) # projections on the 2nd pca plane
            
            
            ### angle between two planes
            #cos_theta, sin_theta = angle_btw_vecs(vec_normal1, vec_normal2)
            #theta = angle_by_cossin(cos_theta, sin_theta)
            
            ### store the normal vectors for plane Loc1 and Loc2 at each time
            vecs_D[cp][tt][1] = vecs1
            vecs_D[cp][tt][2] = vecs2
            
            projs_D[cp][tt][1] = proj1
            projs_D[cp][tt][2] = proj2
            
            projsAll_D[cp][tt][1] = tempX1_proj
            projsAll_D[cp][tt][2] = tempX2_proj
            
            X_mean[cp][tt][1], X_mean[cp][tt][2] = dataT_mean1, dataT_mean2
            
            
            
            
            
            ### run only if to plot
            ### plot the best-fitting plane
            # find vertices, then sort vertices according to the shortest path - so that plotted plane will be a quadrilateral
            # only plot in the selected iteration
            if toPlot == True :
                
                color1 = 'r'
                color2 = 'g'
                #color3 = 'b'
                
                #colors = np.array([color1, color2, color3])
                
                fig = plt.figure(figsize=(8, 6))
                ax = fig.add_subplot(111, projection='3d')
                
                shapes = ('o','s','*','^')
                for i in range(len(tempX1_mean)):
                    ax.scatter(tempX1_mean[i,0], tempX1_mean[i,1], tempX1_mean[i,2], marker = shapes[i], color = color1, alpha = 0.7, label = f'loc1, {i}')
                    ax.scatter(proj1[i,0], proj1[i,1], proj1[i,2], marker = shapes[i], color = color1, alpha = 0.2)#, label = f'loc1, {i}'
                    
                    #if cp >= slice_epochsDic['s2'][0]:
                    ax.scatter(tempX2_mean[i,0], tempX2_mean[i,1], tempX2_mean[i,2], marker = shapes[i], color = color2, alpha = 0.7, label = f'loc2, {i}')
                    ax.scatter(proj2[i,0], proj2[i,1], proj2[i,2], marker = shapes[i], color = color2, alpha = 0.2)#, label = f'loc2, {i}'
                    #ax.scatter(tempX3[i,0], tempX3[i,1], tempX3[i,2], marker = shapes[i], color = color3, label = f'locChoice, {i}')
                
                #ax.scatter(tempX1[:,0], tempX1[:,1], tempX1[:,2], marker = '.', color = color1, alpha = 0.3)
                #ax.scatter(tempX1_proj[:,0], tempX1_proj[:,1], tempX1_proj[:,2], marker = '.', color = color1, alpha = 0.1)#, label = f'loc1, {i}'
                #ax.scatter(tempX2_proj[:,0], tempX2_proj[:,1], tempX2_proj[:,2], marker = '.', color = color2, alpha = 0.1)#, label = f'loc1, {i}'
                
                ax.plot_surface(x1_plane, y1_plane, z1_plane, alpha=0.5, color = color1)
                #ax.add_collection3d(Poly3DCollection([sorted_verts1], facecolor=color1, edgecolor=[], alpha=0.2))#
                #ax.quiver(x1_plane.mean(), y1_plane.mean(), z1_plane.mean(), vec_normal1[0], vec_normal1[1], vec_normal1[2], color = color1, alpha = 0.2)#, arrow_length_ratio = 0.001
                #ax.quiver(x1_plane.mean(), y1_plane.mean(), z1_plane.mean(), vecs1[0,0], vecs1[0,1], vecs1[0,2], color = color1, alpha = 1)
                #ax.quiver(x1_plane.mean(), y1_plane.mean(), z1_plane.mean(), vecs1[1,0], vecs1[1,1], vecs1[1,2], color = color1, alpha = 1)
                ax.quiver(tempX1_mean[:,0].mean(),tempX1_mean[:,1].mean(),tempX1_mean[:,2].mean(), vec_normal1[0], vec_normal1[1], vec_normal1[2], color = color1, alpha = 0.2)#, arrow_length_ratio = 0.001
                #ax.quiver(tempX1_mean[:,0].mean(),tempX1_mean[:,1].mean(),tempX1_mean[:,2].mean(), vecs1[0,0], vecs1[0,1], vecs1[0,2], color = color1, alpha = 1)
                #ax.quiver(tempX1_mean[:,0].mean(),tempX1_mean[:,1].mean(),tempX1_mean[:,2].mean(), vecs1[1,0], vecs1[1,1], vecs1[1,2], color = color1, alpha = 1)
                #ax.text(x1_plane.min(),y1_plane.min(),z1_plane.min(),f'Loc1 EVR:{evr2_1[0]:.4f}; {evr2_1[1]:.4f}')
                
                #if cp >= slice_epochsDic['s2'][0]:
                    
                ax.plot_surface(x2_plane, y2_plane, z2_plane, alpha=0.5, color = color2)
                #ax.add_collection3d(Poly3DCollection([sorted_verts2], facecolor=color2, edgecolor=[], alpha=0.2))#
                #ax.quiver(x2_plane.mean(), y2_plane.mean(), z2_plane.mean(), vec_normal2[0], vec_normal2[1], vec_normal2[2], color = color2, alpha = 0.2)#, arrow_length_ratio = 0.001
                #ax.quiver(x2_plane.mean(), y2_plane.mean(), z2_plane.mean(), vecs2[0,0], vecs2[0,1], vecs2[0,2], color = color2, alpha = 1)
                #ax.quiver(x2_plane.mean(), y2_plane.mean(), z2_plane.mean(), vecs2[1,0], vecs2[1,1], vecs2[1,2], color = color2, alpha = 1)
                ax.quiver(tempX2_mean[:,0].mean(),tempX2_mean[:,1].mean(),tempX2_mean[:,2].mean(), vec_normal2[0], vec_normal2[1], vec_normal2[2], color = color2, alpha = 0.2)#, arrow_length_ratio = 0.001
                #ax.quiver(tempX2_mean[:,0].mean(),tempX2_mean[:,1].mean(),tempX2_mean[:,2].mean(), vecs2[0,0], vecs2[0,1], vecs2[0,2], color = color2, alpha = 1)
                #ax.quiver(tempX2_mean[:,0].mean(),tempX2_mean[:,1].mean(),tempX2_mean[:,2].mean(), vecs2[1,0], vecs2[1,1], vecs2[1,2], color = color2, alpha = 1)
                #ax.text(x2_plane.min(),y2_plane.min(),z2_plane.min(),f'Loc2 EVR:{evr2_2[0]:.4f}; {evr2_2[1]:.4f}')
            
                
                ax.set_xlabel(f'PC1 ({evr_1st[0]:.4f})')
                ax.set_ylabel(f'PC2 ({evr_1st[1]:.4f})')
                ax.set_zlabel(f'PC3 ({evr_1st[2]:.4f})')
                
                #ax.view_init(elev=45, azim=45, roll=0)
                
                plt.legend(loc='upper right')
                plt.tight_layout()
                plt.title(f't = {cp}, type = {tt}') #, {region}, cosTheta = {cos_theta:.3f}, loc1% = {performance1_pcaProj:.2f}; loc2% = {performance2_pcaProj:.2f}
                plt.show()
    
    return vecs_D, projs_D, projsAll_D, X_mean, trialInfo_D, dataT_3pc, dataT_3pc_mean, evr_1st, pca1 # , decodability_projD, pca2




#%%
def planeC_fitting_analysis(dataT, trialInfo, pca_tWinX, tsliceRange, choice_tRange, locs, ttypes, dropCombs=[], avgMethod = 'conditional_time',
                            dt = 10, bins = 100, adaptPCA = None, adaptEVR = None, toPlot=False, sequence = (3,1,2,0), 
                            plot_traj = True, traj_checkpoints = (1300, 2600), traj_start = 0, traj_end = 2600, indScatters=False, gau_kappa = 2.0,
                            plot3d = False, region_label = '', plotlayout = (0,1,2,3), legend_on=True, hideLocs = (), hideType = (), normalizeMinMax = False, separatePlot = True,
                            savefig=False, save_path=''):
    
    epsilon = 1e-7
    vecs_C = {}
    projs_C = {}
    projsAll_C = {}

    trialInfo_C = trialInfo
    
    X_mean = {}
    ncells = dataT.shape[1]
    locCombs = list(permutations(locs,2))
    subConditions = list(product(locCombs, ttypes))
    
    # remove invalid combos
    subConditions_copy = subConditions.copy()
    for sc in subConditions_copy:
        if sc in dropCombs:
            subConditions.remove(sc)
    
    ### main analysis
    # condition average data. within each ttype, shape = (4*3) * ncells * nt
    
    dataT_slice = dataT[:,:,pca_tWinX]
    
    # all trial, time averaged
    if avgMethod == 'all':
        dataT_avg = dataT_slice.mean(axis=0)
    
    # all trial, time separate
    elif avgMethod == 'none':
        dataT_avg = []
        for trial in range(dataT_slice.shape[0]):
            dataT_avg += [dataT_slice[trial,:,:]]
        
        dataT_avg = np.hstack(dataT_avg)
    
    # conditional, time separate
    elif avgMethod == 'conditional':
        dataT_avg = []
        
        for sc in subConditions:
            lc, tt = sc[0], sc[1]
            slc = '_'.join([str(l) for l in lc])
            idxx = trialInfo[(trialInfo.locs == slc)&(trialInfo.type == tt)].index.tolist()
            
            x = dataT_slice[idxx,:,:].mean(axis=0) 
            
            dataT_avg += [x]
    
        # stack as ncells * (nt * (4*3))
        dataT_avg = np.hstack(dataT_avg)
    
    # conditional, time averaged
    elif avgMethod == 'conditional_time':
        dataT_avg = []
        
        for sc in subConditions:
            lc, tt = sc[0], sc[1]
            slc = '_'.join([str(l) for l in lc])
            idxx = trialInfo[(trialInfo.locs == slc)&(trialInfo.type == tt)].index.tolist()
            
            x = dataT_slice[idxx,:,:].mean(axis=0).mean(axis=-1) #
            
            dataT_avg += [x]
            #dataT_avg_full += [x]
    
        # stack as ncells * (4*3)
        dataT_avg = np.vstack(dataT_avg).T
    
    ###
    # 1st order pca, reduce from ncells * (nt*12) -> 3pcs * (nt*12) to create a 3pc space shared by all locComb conditions, across all timepoints
    dataT_3pc = []
    
    if (adaptPCA is not None) and (adaptPCA.shape==(3,ncells)):
        # all trials (ntrials * npc * ntime)
        for trial in range(dataT.shape[0]):
            dataT_3pc += [np.dot(dataT[trial,:,:].T, adaptPCA.T).T]
        
        dataT_3pc = np.array(dataT_3pc)
        evr_1st = np.ones(3,) if adaptEVR is None else adaptEVR
        pca1 = adaptPCA.copy()
        
    else:
        if adaptPCA is not None:
            print('Error: None input of adaptPCA or Shape not match. Will use default method.')
            
        pca_1st = PCA(n_components=3)
    
        # fit the PCA model to the data
        pca_1st.fit(dataT_avg.T)
        
        evr_1st = pca_1st.explained_variance_ratio_
        
        pca1 = pca_1st.components_
        
        # all trials (ntrials * npc * ntime)
        
        for trial in range(dataT.shape[0]):
            dataT_3pc += [pca_1st.transform(dataT[trial,:,:].T).T]
        
        dataT_3pc = np.array(dataT_3pc)


    dataT_3pc_mean = []
    for sc in subConditions:
        lc, tt = sc[0], sc[1]
        slc = '_'.join([str(l) for l in lc])
        idxx = trialInfo[(trialInfo.locs == slc)&(trialInfo.type == tt)].index.tolist()
        
        dataT_3pc_mean += [dataT_3pc[idxx,:,:].mean(axis=0)]
    
    # reshape as nconditions * 3pcs * nt
    dataT_3pc_mean = np.array(dataT_3pc_mean) # this is condition mean states in 3pc space at each time point 
    
    
    # 2nd PCA time window
    t1 = tsliceRange.tolist().index(choice_tRange[0]) if choice_tRange[0] >= tsliceRange.min() else tsliceRange.tolist().index(tsliceRange.min())
    t2 = tsliceRange.tolist().index(choice_tRange[1]) if choice_tRange[1] <= tsliceRange.max() else tsliceRange.tolist().index(tsliceRange.max())

    
    tempX = dataT_3pc[:,:,t1:t2].mean(axis=2) if t1!=t2 else dataT_3pc[:,:,t1] # individual trials
    tempX_mean = dataT_3pc_mean[:,:,t1:t2].mean(axis=2) if t1!=t2 else dataT_3pc_mean[:,:,t1]  # condition average
    
    dataT_mean = dataT[:,:,t1:t2].mean(axis=2) if t1!=t2 else dataT[:,:,t1]
    tempXC_mean = []
    dataTC_mean = []
    
    for l in locs:
        
        conx = [subConditions.index(sc) for sc in subConditions if (sc[0][0] == l and sc[1] == 2) or (sc[0][1] == l and sc[1] == 1)]
        tempXC_mean += [tempX_mean[conx,:].mean(axis = 0)]
        dataTC_mean += [dataT_mean[conx,:].mean(axis = 0)]
        
    tempXC_mean = np.array(tempXC_mean)
    dataTC_mean = np.array(dataTC_mean)
    
    ### choice subspace
    pca_2nd_C = PCA(n_components=2)
    pca_2nd_C.fit(tempXC_mean)# - tempX1_mean.mean()
    vecsC, evr2_C = pca_2nd_C.components_, pca_2nd_C.explained_variance_ratio_
    vec_normalC = np.cross(vecsC[0], vecsC[1])
    vec_normalC = vec_normalC / np.linalg.norm(vec_normalC) if np.linalg.norm(vec_normalC) !=0 else vec_normalC / (np.linalg.norm(vec_normalC)+ epsilon)
    
    # plane centered at mean of conditional mean
    centerC = tempXC_mean.mean(axis=0)
    
    # projects of conditional means on the plane, 3d
    projC = np.array([proj_on_plane(p, vec_normalC, centerC) for p in tempXC_mean])
    
    # create plane grids
    x_plane, y_plane, z_plane = plane_by_vecs(vecsC, center = centerC, xRange=(dataT_3pc_mean[:,0,:].min(), dataT_3pc_mean[:,0,:].max()), yRange=(dataT_3pc_mean[:,1,:].min(), dataT_3pc_mean[:,1,:].max()))
    
    # projects of single trials on the plane, 3d
    #tempX1_pca = pca_2nd_1.transform(tempX_tt) # 2nd pca transformed data
    tempX_proj = np.array([proj_on_plane(p, vec_normalC, centerC) for p in tempX]) # projections on the 2nd pca plane
    
    
    
    
    
    ### angle between two planes
    #cos_theta, sin_theta = angle_btw_vecs(vec_normal1, vec_normal2)
    #theta = angle_by_cossin(cos_theta, sin_theta)
    
    ### store the normal vectors for plane Loc1 and Loc2 at each time
    vecs_C = vecsC
    
    projs_C = projC
    
    projsAll_C = tempX_proj
    
    X_mean = dataTC_mean
    
    
    ### run only if to plot
    ### plot the best-fitting plane
    # find vertices, then sort vertices according to the shortest path - so that plotted plane will be a quadrilateral
    # only plot in the selected iteration
    if toPlot == True :
        traj_startX, traj_endX = tsliceRange.tolist().index(traj_start), tsliceRange.tolist().index(traj_end)
        
        temp_3pc = {}
        temp_3pc_mean = {}
        xPmin, xPmax = [],[]
        yPmin, yPmax = [],[]
        
        remainedLocs = tuple(l for l in locs if l not in hideLocs)

        for l in locs:
            temp_3pc[l] = {1:[], 2:[]}
            temp_3pc_mean[l] = {1:[], 2:[]}
            
            
            for tt in ttypes:
                conx = [subConditions.index(sc) for sc in subConditions if (sc[0][0] == l and sc[1] == tt)]
                temp_3pc_mean[l][tt] = dataT_3pc_mean[conx,:,:]
                
                for l2 in locs:
                    if l!=l2:
                        trialx = trialInfo[(trialInfo.loc1==l)&(trialInfo.type==tt)&(trialInfo.loc2==l2)].index
                        temp_3pc[l][tt] += [dataT_3pc[trialx,:,:]]
                
                projs_3pc_mean = np.array([[proj_on_plane(p, vec_normalC, centerC) for p in temp_3pc_mean[l][tt][:,:,t]] for t in range(temp_3pc_mean[l][tt].shape[2])])
                projs_3pc_mean = np.swapaxes(np.swapaxes(projs_3pc_mean, 0, 1), 1, 2)
                
                # smooth for traj
                tbinsRange = np.arange(traj_start, traj_end+bins, bins)
                tsmoothX = np.array([tsliceRange.tolist().index(tb) for tb in tbinsRange])
                projs_3pc_mean_smooth = projs_3pc_mean[:,:,tsmoothX]

                xPmin += [projs_3pc_mean_smooth[:,0,:].min()]
                xPmax += [projs_3pc_mean_smooth[:,0,:].max()]
                yPmin += [projs_3pc_mean_smooth[:,1,:].min()]
                yPmax += [projs_3pc_mean_smooth[:,1,:].max()]
                
                
        if plot3d:
            
            # get global min max
            temp_pooled = []
            temp_mean_pooled = []
            for l in locs:
                for tt in ttypes:
                    for d in range(len(temp_3pc[l][tt])):
                        temp = np.array(temp_3pc[l][tt][d])
                        temp_pooled += [temp]
                    for d in range(len(temp_3pc_mean[l][tt])):
                        temp_mean = np.array([temp_3pc_mean[l][tt][d]]) # []to add dimension so that the shape keeps: condition * pcs * time
                        temp_mean_pooled += [temp_mean]

            temp_pooled = np.concatenate(temp_pooled)
            temp_mean_pooled = np.concatenate(temp_mean_pooled)
            global_min, global_max = temp_pooled.min(axis=(0,2)), temp_pooled.max(axis=(0,2))
            global_mean_min, global_mean_max = temp_mean_pooled.min(axis=(0,2)), temp_mean_pooled.max(axis=(0,2))

            
            # create plane grids
            
            x_planeT, y_planeT, z_planeT = plane_by_vecs(vecsC, center = centerC, xRange=(np.array(xPmin).min(), np.array(xPmax).max()), yRange=(np.array(yPmin).min(), np.array(yPmax).max()))
            
            
            # plots
            if separatePlot:
                fig, axes = plt.subplots(2,2, figsize=(35, 35), dpi = 100, sharex=True, sharey=True, subplot_kw={'projection': '3d'})
            else:
                fig, axes = plt.subplots(1,1, figsize=(15, 15), dpi = 100, sharex=True, sharey=True, subplot_kw={'projection': '3d'})

            shapes = ('o','*','s','^')
            colors = plt.get_cmap('Paired').colors#('r','b','g','m')
            colorC = 'grey'

            for l in locs:
                
                ll = plotlayout.index(l)
                
                ax = axes.flatten()[ll] if separatePlot else axes
                #ax = fig.add_subplot(2,2,l+1)
                ax.plot_surface(x_planeT, y_planeT, z_planeT, alpha=0.3, color = colorC)

                if l in remainedLocs:
                    
                    remainLoc2 = tuple(rl2 for rl2 in locs if rl2 != l)
                    showLoc2X = tuple(remainLoc2.index(showL) for showL in remainLoc2 if showL not in hideLocs)

                    for tt in ttypes:
                        
                        ttypeT = 'Retarget' if tt==1 else 'Distraction'
                        
                        raws_3pc = np.array(temp_3pc[l][tt])
                        projs_3pc = raws_3pc
                        
                        raws_3pc_mean = temp_3pc_mean[l][tt] # shape: condition * pcs * time
                        projs_3pc_mean = np.array(raws_3pc_mean)
                        
                        # normalize -1 to 1
                        if bool(normalizeMinMax):
                            vmin, vmax = normalizeMinMax if len(normalizeMinMax)==2 else (-1,1)

                            projs_3pc_mean = np.array([((projs_3pc_mean[:,d,:] - global_mean_min[d]) / (global_mean_max[d] - global_mean_min[d])) * (vmax - vmin) + vmin for d in range(projs_3pc_mean.shape[1])])
                            projs_3pc_mean = np.swapaxes(projs_3pc_mean,0,1) # reshape to conditions * pc * t
                            
                            for i in range(len(projs_3pc)):    
                                projs_3pcT = np.array([((projs_3pc[i][:,d,:] - global_min[d]) / (global_max[d] - global_min[d])) * (vmax - vmin) + vmin for d in range(projs_3pc[i].shape[1])])
                                projs_3pc[i] = np.swapaxes(projs_3pcT,0,1) # reshape to conditions * pc * t
                                
                        # smooth for traj
                        tbinsRange = np.arange(traj_start, traj_end+bins, bins)
                        tsmoothX = np.array([tsliceRange.tolist().index(tb) for tb in tbinsRange])
                        raws_3pc_mean_smooth = raws_3pc_mean[:,:,tsmoothX]
                        projs_3pc_mean_smooth = projs_3pc_mean[:,:,tsmoothX]
                        projs_3pc_smooth = projs_3pc[:,:,:,tsmoothX]
                        
                        
                        
                        loc2s = np.array(locs)[np.array(locs)!=l]
                        
                        for i in range(len(raws_3pc_mean)):
                            colorT_r, colorT_p = colors[loc2s[i]*2], colors[loc2s[i]*2+1] # projection and raw
                            lsT = '-' if tt ==1 else ':'
                            fcc_p = colorT_p if tt==1 else 'none'
                            egc_p = colorT_p #'none' if tt==1 else
                            fcc_r = colorT_r if tt==1 else 'none'
                            egc_r = colorT_r #'none' if tt==1 else 
                            
                            projs_3pc_mean_gausmoothT = np.array([gaussian_filter1d(projs_3pc_mean_smooth[i,d,:],gau_kappa) for d in range(projs_3pc_mean_smooth.shape[1])])
                            projs_3pc_mean_gausmoothT[:,0] = projs_3pc_mean_smooth[i,:,0]
                            projs_3pc_mean_gausmoothT[:,-1] = projs_3pc_mean_smooth[i,:,-1]
                            
                            if (tt not in hideType) and (i in showLoc2X):

                                if plot_traj:
                                    ax.plot(projs_3pc_mean_gausmoothT[0,:], projs_3pc_mean_gausmoothT[1,:], projs_3pc_mean_gausmoothT[2,:], color = colorT_p, alpha = 1, linestyle = lsT, linewidth=5)
                                    
                                for nc, cp in enumerate(traj_checkpoints):
                                    #ncp = tsliceRange.tolist().index(cp)
                                    ncp = tbinsRange.tolist().index(cp)
                                    labelT = f'Item2 = {loc2s[i]}, {ttypeT}' if nc==0 else ''
                                    ax.scatter(projs_3pc_mean_gausmoothT[0,:][ncp], projs_3pc_mean_gausmoothT[1,:][ncp], projs_3pc_mean_gausmoothT[2,:][ncp], 
                                            marker = f'{shapes[nc]}', color = colorT_p, alpha = 1, s = 500, edgecolors=egc_p, facecolors = fcc_p, linewidth=5, label = labelT)#, label = f'loc1, {i}'
                                    
                                    if indScatters:
                                        
                                        ax.scatter(gaussian_filter(projs_3pc_smooth[i][:,0,:],gau_kappa)[:,ncp], gaussian_filter(projs_3pc_smooth[i][:,1,:],gau_kappa)[:,ncp], gaussian_filter(projs_3pc_smooth[i][:,2,:],gau_kappa)[:,ncp], 
                                            marker = f'{shapes[nc]}', color = colorT_p, alpha = 0.1, s = 500, edgecolors=egc_p, facecolors = fcc_p, linewidth=5)#, label = f'loc1, {i}'
                        
                
                ax.set_title(f'Item1 = {l}', fontsize=20)
    
                ax.set_xlabel(f'PC1 ({evr_1st[0]:.4f})', fontsize=20, labelpad=20)
                ax.tick_params(axis='x', labelsize=15)
    
                ax.set_ylabel(f'PC2 ({evr_1st[1]:.4f})', fontsize=20, labelpad=20)
                ax.tick_params(axis='y', labelsize=15)
    
                ax.set_zlabel(f'PC3 ({evr_1st[2]:.4f})', fontsize=20, labelpad=20)
                ax.tick_params(axis='z', labelsize=15)
                
                
                if bool(normalizeMinMax):
                    lim_pad = (normalizeMinMax[1] - normalizeMinMax[0])/20
                    ax.set_xlim(normalizeMinMax[0]-lim_pad, normalizeMinMax[1]+lim_pad)
                    ax.set_ylim(normalizeMinMax[0]-lim_pad, normalizeMinMax[1]+lim_pad)
                    ax.set_zlim(normalizeMinMax[0]-lim_pad, normalizeMinMax[1]+lim_pad)
    
                # Calculate the azimuth and elevation
                vecRef = vec_normalC 
                vecRef = vecRef/np.sqrt(vecRef[0]**2 + vecRef[1]**2 + vecRef[2]**2)
                elev = np.degrees(np.arcsin(vecRef[2]))
                elev = elev if elev>=0 else elev+180
                azim = np.degrees(np.arctan2(vecRef[1], vecRef[0]))
                
                ax.view_init(elev=-elev, azim=0, roll=-60)#
                
                if legend_on:
                    plt.legend(loc='upper right', fontsize=20)
                
            plt.subplots_adjust(top = 0.8)
            plt.suptitle(f'{region_label}, Readout Subspace', fontsize = 30, y=0.85) #, {region}, cosTheta = {cos_theta:.3f}, loc1% = {performance1_pcaProj:.2f}; loc2% = {performance2_pcaProj:.2f}
            plt.show()
        
        
        else:
            
            vecX_, vecY_ = vec_quad_side(projC, sequence = sequence) 
            vecs_new = np.array(basis_correction_3d(vecsC, vecX_, vecY_))
            
            # get global min max
            temp_pooled_2d = []
            temp_mean_pooled_2d = []
            for l in locs:
                for tt in ttypes:
                    
                    # all trials
                    for d in range(len(temp_3pc[l][tt])):
                        temp = np.array(temp_3pc[l][tt][d])
                        #
                        temp_2d = np.array([proj_2D_coordinates(temp[:,:,t], vecs_new) for t in range(temp.shape[2])])
                        temp_2d = np.swapaxes(np.swapaxes(temp_2d, 0, 1), 1, 2)
                        
                        temp_pooled_2d += [temp_2d]
                    
                    # condition mean
                    for d in range(len(temp_3pc_mean[l][tt])):
                        temp_mean = np.array([temp_3pc_mean[l][tt][d]]) # []to add dimension, shape: condition * pcs * time
                        temp_mean_2d = np.array([proj_2D_coordinates(temp_mean[:,:,t], vecs_new) for t in range(temp_mean.shape[2])])
                        temp_mean_2d = np.swapaxes(np.swapaxes(temp_mean_2d, 0, 1), 1, 2)
                        temp_mean_pooled_2d += [temp_mean_2d]

            temp_pooled_2d = np.concatenate(temp_pooled_2d)
            temp_mean_pooled_2d = np.concatenate(temp_mean_pooled_2d)
            global_min, global_max = temp_pooled_2d.min(axis=(0,2)), temp_pooled_2d.max(axis=(0,2))
            global_mean_min, global_mean_max = temp_mean_pooled_2d.min(axis=(0,2)), temp_mean_pooled_2d.max(axis=(0,2))

            
            # plots
            if separatePlot:
                fig, axes = plt.subplots(2,2, figsize=(35, 35), dpi = 100, sharex=True, sharey=True)
            else:
                fig, axes = plt.subplots(1,1, figsize=(15, 15), dpi = 100, sharex=True, sharey=True)

            shapes = ('o','*','s','^')
            colors = plt.get_cmap('Paired').colors#('r','b','g','m')
            
            for l in locs:
                
                ll = plotlayout.index(l)
                
                ax = axes.flatten()[ll] if separatePlot else axes
                #ax = fig.add_subplot(2,2,l+1)
                
                if l in remainedLocs:
                    remainLoc2 = tuple(rl2 for rl2 in locs if rl2 != l)
                    showLoc2X = tuple(remainLoc2.index(showL) for showL in remainLoc2 if showL not in hideLocs)

                    for tt in ttypes:
                        
                        ttypeT = 'Retarget' if tt==1 else 'Distraction'
                        
                        raws_3pc = np.array(temp_3pc[l][tt])
                        #projs_3pc = []
                        #for l2 in range(raws_3pc.shape[0]):
                        #    raws_3pcT = raws_3pc[l2]
                        #    projs_3pcT = np.array([[proj_on_plane(p, vec_normalC, centerC) for p in raws_3pcT[:,:,t]] for t in range(raws_3pcT.shape[2])])
                        #    projs_3pcT = np.swapaxes(np.swapaxes(projs_3pcT, 0, 1), 1, 2)
                        #    projs_3pc += [projs_3pcT]
                        projs_3pc = raws_3pc#np.array(projs_3pc)
                        
                        raws_3pc_mean = temp_3pc_mean[l][tt]
                        #projs_3pc_mean = np.array([[proj_on_plane(p, vec_normalC, centerC) for p in raws_3pc_mean[:,:,t]] for t in range(raws_3pc_mean.shape[2])])
                        projs_3pc_mean = np.array(raws_3pc_mean)#np.swapaxes(np.swapaxes(projs_3pc_mean, 0, 1), 1, 2)
                        
                        projs_3pc_mean_2d = np.array([proj_2D_coordinates(projs_3pc_mean[:,:,t], vecs_new) for t in range(projs_3pc_mean.shape[2])])
                        projs_3pc_mean_2d = np.swapaxes(np.swapaxes(projs_3pc_mean_2d, 0, 1), 1, 2)
                        
                        projs_3pc_2d = []
                        for l2 in range(projs_3pc.shape[0]):
                            projs_3pcT = projs_3pc[l2]
                            projs_3pc_2dT = np.array([proj_2D_coordinates(projs_3pcT[:,:,t], vecs_new) for t in range(projs_3pcT.shape[2])])
                            projs_3pc_2dT = np.swapaxes(np.swapaxes(projs_3pc_2dT, 0, 1), 1, 2)
                            projs_3pc_2d += [projs_3pc_2dT]
                        projs_3pc_2d = np.array(projs_3pc_2d)
                        
                        
                        # normalize -1 to 1
                        if bool(normalizeMinMax):
                            vmin, vmax = normalizeMinMax if len(normalizeMinMax)==2 else (-1,1)

                            projs_3pc_mean_2d = np.array([((projs_3pc_mean_2d[:,d,:] - global_mean_min[d]) / (global_mean_max[d] - global_mean_min[d])) * (vmax - vmin) + vmin for d in range(projs_3pc_mean_2d.shape[1])])
                            projs_3pc_mean_2d = np.swapaxes(projs_3pc_mean_2d,0,1) # reshape to conditions * pc * t
                            
                            for i in range(len(projs_3pc_2d)):    
                                projs_3pc_2dT = np.array([((projs_3pc_2d[i][:,d,:] - global_min[d]) / (global_max[d] - global_min[d])) * (vmax - vmin) + vmin for d in range(projs_3pc_2d[i].shape[1])])
                                projs_3pc_2d[i] = np.swapaxes(projs_3pc_2dT,0,1) # reshape to conditions * pc * t
                                
                        # smooth for traj
                        tbinsRange = np.arange(traj_start, traj_end+bins, bins)
                        tsmoothX = np.array([tsliceRange.tolist().index(tb) for tb in tbinsRange])
                        
                        #ncondT, npcsT, ntimeT = projs_3pc_meanT[:,:,traj_startX:traj_endX].shape
                        #dt, bins = 10, 100
                        #raws_3pc_mean_smoothT = np.mean(temp_3pc_mean[l][tt][:,:,traj_startX:traj_endX].reshape(ncondT, npcsT, int(ntimeT/(bins/dt)), int(bins/dt)),axis = 3)
                        #projs_3pc_mean_smoothT = np.mean(projs_3pc_meanT[:,:,traj_startX:traj_endX].reshape(ncondT, npcsT, int(ntimeT/(bins/dt)), int(bins/dt)),axis = 3)
                        #raws_3pc_mean_smooth = raws_3pc_mean[:,:,tsmoothX]
                        projs_3pc_mean_smooth_2d = projs_3pc_mean_2d[:,:,tsmoothX]
                        projs_3pc_smooth_2d = projs_3pc_2d[:,:,:,tsmoothX]
                        


                        loc2s = np.array(locs)[np.array(locs)!=l]
                        
                        for i in range(len(raws_3pc_mean)):
                            colorT_r, colorT_p = colors[loc2s[i]*2], colors[loc2s[i]*2+1] # projection and raw
                            lsT = '-' if tt ==1 else ':'
                            fcc_p = colorT_p if tt==1 else 'none'
                            egc_p = colorT_p #'none' if tt==1 else
                            fcc_r = colorT_r if tt==1 else 'none'
                            egc_r = colorT_r #'none' if tt==1 else 

                            
                            projs_3pc_mean_gausmoothT = np.array([gaussian_filter1d(projs_3pc_mean_smooth_2d[i,d,:],gau_kappa) for d in range(projs_3pc_mean_smooth_2d.shape[1])])
                            projs_3pc_mean_gausmoothT[:,0] = projs_3pc_mean_smooth_2d[i,:,0]
                            projs_3pc_mean_gausmoothT[:,-1] = projs_3pc_mean_smooth_2d[i,:,-1]

                            if (tt not in hideType) and (i in showLoc2X):

                                if plot_traj:
                                    #ax.plot(raws_3pc_mean_smooth[i,0,:], raws_3pc_mean_smooth[i,1,:], raws_3pc_mean_smooth[i,2,:], color = colorT_r, alpha = 0.5, linestyle = lsT)
                                    #ax.plot(projs_3pc_mean_smooth[i,0,:], projs_3pc_mean_smooth[i,1,:], projs_3pc_mean_smooth[i,2,:], color = colorT_p, alpha = 0.5, linestyle = lsT)
                                    
                                    #ax.plot(gaussian_filter1d(raws_3pc_mean_smooth[i,0,:],2), gaussian_filter1d(raws_3pc_mean_smooth[i,1,:],2), gaussian_filter1d(raws_3pc_mean_smooth[i,2,:],2), color = colorT_r, alpha = 0.5, linestyle = lsT)
                                    ax.plot(gaussian_filter1d(projs_3pc_mean_smooth_2d[i,0,:],gau_kappa), gaussian_filter1d(projs_3pc_mean_smooth_2d[i,1,:],gau_kappa), color = colorT_p, alpha = 1, linestyle = lsT, linewidth=5)
                                    ax.plot(projs_3pc_mean_gausmoothT[0,:], projs_3pc_mean_gausmoothT[1,:], color = colorT_p, alpha = 1, linestyle = lsT, linewidth=5)

                                for nc, cp in enumerate(traj_checkpoints):
                                    #ncp = tsliceRange.tolist().index(cp)
                                    ncp = tbinsRange.tolist().index(cp)
                                    labelT = f'Item2={loc2s[i]}, {ttypeT}' if nc==0 else ''
                                    #egc = 'y' if tt==1 else 'k'
                                    #ax.scatter(raws_3pc_mean_smooth[i,0,ncp], raws_3pc_mean_smooth[i,1,ncp], raws_3pc_mean_smooth[i,2,ncp], marker = f'{shapes[nc]}', color = colorT_r, alpha = 1, s = 100, edgecolors=egc_r, facecolors = fcc_r, linewidth=2)
                                    #ax.scatter(projs_3pc_mean[i,0,ncp], projs_3pc_mean[i,1,ncp], projs_3pc_mean[i,2,ncp], marker = f'{shapes[nc]}', color = colorT_p, alpha = 1, s = 100, edgecolors=egc_r, facecolors = fcc_r, linewidth=3)#, label = f'loc1, {i}'
                                    #ax.scatter(gaussian_filter1d(projs_3pc_mean_smooth_2d[i,0,:],gau_kappa)[ncp], gaussian_filter1d(projs_3pc_mean_smooth_2d[i,1,:],gau_kappa)[ncp], 
                                    #           marker = f'{shapes[nc]}', color = colorT_p, alpha = 1, s = 500, edgecolors=egc_p, facecolors = fcc_p, linewidth=5, label = labelT)#, label = f'loc1, {i}'
                                    ax.scatter(projs_3pc_mean_gausmoothT[0,:][ncp], projs_3pc_mean_gausmoothT[1,:][ncp], 
                                            marker = f'{shapes[nc]}', color = colorT_p, alpha = 1, s = 500, edgecolors=egc_p, facecolors = fcc_p, linewidth=5, label = labelT)#, label = f'loc1, {i}'
                                    
                                    if indScatters:
                                                                                
                                        ax.scatter(gaussian_filter(projs_3pc_smooth_2d[i][:,0,:],gau_kappa)[:,ncp], gaussian_filter(projs_3pc_smooth_2d[i][:,1,:],gau_kappa)[:,ncp], 
                                            marker = f'{shapes[nc]}', color = colorT_p, alpha = 0.1, s = 500, edgecolors=egc_p, facecolors = fcc_p, linewidth=5)#, label = f'loc1, {i}'
                    

                ax.set_title(f'Item1 = {l}', fontsize=25, pad=10)
    
                #[l.set_visible(False) for (i,l) in enumerate(ax.xaxis.get_ticklabels()) if i % 1 != 0]
                ax.set_xlabel(f'Secondary PC1 ({evr2_C[0]:.4f})', fontsize=25, labelpad=10)
                ax.tick_params(axis='x', labelsize=25)
                
                
                #[l.set_visible(False) for (i,l) in enumerate(ax.yaxis.get_ticklabels()) if i % 1 != 0]
                ax.set_ylabel(f'Secondary PC2 ({evr2_C[1]:.4f})', fontsize=25, labelpad=10)
                ax.tick_params(axis='y', labelsize=25)
                
                if bool(normalizeMinMax):
                    lim_pad = (normalizeMinMax[1] - normalizeMinMax[0])/20
                    ax.set_xlim(normalizeMinMax[0]-lim_pad, normalizeMinMax[1]+lim_pad)
                    ax.set_ylim(normalizeMinMax[0]-lim_pad, normalizeMinMax[1]+lim_pad)
                
                if legend_on:
                    ax.legend(fontsize=25)#loc='upper right', 
                
            #plt.tight_layout()
            plt.subplots_adjust(top = 0.8)
            plt.suptitle(f'{region_label}, Readout Subspace', fontsize = 35, y=0.85) #, {region}, cosTheta = {cos_theta:.3f}, loc1% = {performance1_pcaProj:.2f}; loc2% = {performance2_pcaProj:.2f}
            plt.show()
        
        
        
        
        if savefig:
            dimLabel = '3d' if plot3d else '2d'
            fig.savefig(f'{save_path}/{region_label}_trajs_{dimLabel}.tif')
    
    return vecs_C, projs_C, projsAll_C, X_mean, trialInfo_C, dataT_3pc, dataT_3pc_mean, evr_1st, pca1, evr2_C# , decodability_projD, pca_2nd_C

