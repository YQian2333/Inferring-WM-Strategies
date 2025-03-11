# In[ ]:
%reload_ext autoreload
%autoreload 2

# Import useful py lib/packages
import os
from itertools import permutations, product

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import scipy
from scipy import stats
import pandas as pd

import matplotlib.pyplot as plt
import re, seaborn as sns

import f_stats
import f_plotting

#%% initialize parameters

data_path = 'D:/data' # for laptop
tRangeRaw = np.arange(-500,4000,1) # -300 baseline, 0 onset, 300 pre1, 1300 delay1, 1600 pre2, 2600 delay2, response

locs = (0,1,2,3,)
locCombs = list(permutations(locs,2))
ttypes = (1,2,)
dropCombs = ()
subConditions = list(product(locCombs, ttypes))

pd.options.mode.chained_assignment = None
epsilon = 1e-7

#%% load precomputed euclidean distances

euDists_monkeys = np.load(f'{data_path}/euDists_monkeys_centroids2_normalized_full.npy', allow_pickle=True).item()
euDists_rnns = np.load(f'{data_path}/euDists_rnns_centroids2_normalized_full.npy', allow_pickle=True).item()
euDists_monkeys_shuff = np.load(f'{data_path}/euDists_shuff_monkeys_centroids2_normalized_full.npy', allow_pickle=True).item()
euDists_rnns_shuff = np.load(f'{data_path}/euDists_rnns_centroids2_shuff_normalized_full.npy', allow_pickle=True).item()

#%%

#############
# distances #
#############

#%%

lineh = np.arange(0.3,0.7,0.001)
linev = np.arange(0,0.03,0.001)

fig, axes = plt.subplots(1,1, figsize=(4,4),dpi=300, sharey=True)

################
# figure props #
################

color1, color1_, color2, color2_ = '#d29c2f', '#f5df7a', '#3c79b4', '#b3cde4'
color3, color3_, color4, color4_ = '#185337', '#96d9ad', '#804098', '#c4a2d1'

boxprops1 = dict(facecolor=color1, edgecolor='none')
flierprops1 = dict(markeredgecolor=color1, markerfacecolor=color1)
capprops1 = dict(color=color1)
whiskerprops1 = dict(color=color1)
meanpointprops1 = dict(marker='^', markeredgecolor=color1, markerfacecolor='w')

boxprops2 = dict(facecolor=color2, edgecolor='none',alpha = 1)
flierprops2 = dict(markeredgecolor=color2, markerfacecolor=color2,alpha = 1)
capprops2 = dict(color=color2,alpha = 1)
whiskerprops2 = dict(color=color2,alpha = 1)
meanpointprops2 = dict(marker='^', markeredgecolor=color2, markerfacecolor='w',alpha = 1)

boxprops3 = dict(facecolor=color3, edgecolor='none')
flierprops3 = dict(markeredgecolor=color3, markerfacecolor=color3)
capprops3 = dict(color=color3)
whiskerprops3 = dict(color=color3)
meanpointprops3 = dict(marker='^', markeredgecolor=color3, markerfacecolor='w')

boxprops4 = dict(facecolor=color4, edgecolor='none',alpha = 1)
flierprops4 = dict(markeredgecolor=color4, markerfacecolor=color4,alpha = 1)
capprops4 = dict(color=color4,alpha = 1)
whiskerprops4 = dict(color=color4,alpha = 1)
meanpointprops4 = dict(marker='^', markeredgecolor=color4, markerfacecolor='w',alpha = 1)

medianprops = dict(linestyle='--', linewidth=1, color='w')

ax = axes#.flatten()[l1]

########
# rnns #
########

# load euclidean distances from rnns
euDists_rnns_ed2_1 = np.array(euDists_rnns['ed2'][1]).squeeze().mean(1)
euDists_rnns_ed2_2 = np.array(euDists_rnns['ed2'][2]).squeeze().mean(1)
euDists_rnns_ed12_1 = np.array(euDists_rnns['ed12'][1]).squeeze().mean(1)
euDists_rnns_ed12_2 = np.array(euDists_rnns['ed12'][2]).squeeze().mean(1)

# plot
ax.boxplot([euDists_rnns_ed2_1], positions=[0.3], patch_artist=True, widths = 0.2, boxprops=boxprops1, flierprops=flierprops1,
                    meanprops=meanpointprops1, medianprops=medianprops, capprops = capprops1, whiskerprops = whiskerprops1, meanline=False, showmeans=True)
ax.boxplot([euDists_rnns_ed12_1], positions=[1.3], patch_artist=True, widths = 0.2, boxprops=boxprops2, flierprops=flierprops2,
                    meanprops=meanpointprops2, medianprops=medianprops, capprops = capprops2, whiskerprops = whiskerprops2, meanline=False, showmeans=True)

ax.boxplot([euDists_rnns_ed2_2], positions=[0.7], patch_artist=True, widths = 0.2, boxprops=dict(boxprops1,facecolor=color1_), flierprops=dict(flierprops1,markeredgecolor=color1_, markerfacecolor=color1_),
                    meanprops=dict(meanpointprops1,markeredgecolor=color1_), medianprops=medianprops, capprops = dict(capprops1,color=color1_), whiskerprops = dict(whiskerprops1,color=color1_), meanline=False, showmeans=True)
ax.boxplot([euDists_rnns_ed12_2], positions=[1.7], patch_artist=True, widths = 0.2, boxprops=dict(boxprops2,facecolor=color2_), flierprops=dict(flierprops2,markeredgecolor=color2_, markerfacecolor=color2_),
                    meanprops=dict(meanpointprops2,markeredgecolor=color2_), medianprops=medianprops, capprops = dict(capprops2,color=color2_), whiskerprops = dict(whiskerprops2,color=color2_), meanline=False, showmeans=True)


###########
# monkeys #
###########

# load euclidean distances from monkeys
euDists_data_dlpfc_1 = np.array(euDists_monkeys['dlpfc'][1]).mean(1)#_shuff
euDists_data_dlpfc_2 = np.array(euDists_monkeys['dlpfc'][2]).mean(1)#_shuff
euDists_data_fef_1 = np.array(euDists_monkeys['fef'][1]).mean(1)#_shuff
euDists_data_fef_2 = np.array(euDists_monkeys['fef'][2]).mean(1)#_shuff

# plot
ax.boxplot([euDists_data_dlpfc_1], positions=[2.3], patch_artist=True, widths = 0.2, boxprops=boxprops3, flierprops=flierprops3,
                    meanprops=meanpointprops3, medianprops=medianprops, capprops = capprops3, whiskerprops = whiskerprops3, meanline=False, showmeans=True)
ax.boxplot([euDists_data_fef_1], positions=[3.3], patch_artist=True, widths = 0.2, boxprops=boxprops4, flierprops=flierprops4,
                    meanprops=meanpointprops4, medianprops=medianprops, capprops = capprops4, whiskerprops = whiskerprops4, meanline=False, showmeans=True)

ax.boxplot([euDists_data_dlpfc_2], positions=[2.7], patch_artist=True, widths = 0.2, boxprops=dict(boxprops3,facecolor=color3_), flierprops=dict(flierprops3,markeredgecolor=color3_, markerfacecolor=color3_),
                    meanprops=dict(meanpointprops3,markeredgecolor=color3_), medianprops=medianprops, capprops = dict(capprops3,color=color3_), whiskerprops = dict(whiskerprops3,color=color3_), meanline=False, showmeans=True)
ax.boxplot([euDists_data_fef_2], positions=[3.7], patch_artist=True, widths = 0.2, boxprops=dict(boxprops4,facecolor=color4_), flierprops=dict(flierprops4,markeredgecolor=color4_, markerfacecolor=color4_),
                    meanprops=dict(meanpointprops4,markeredgecolor=color4_), medianprops=medianprops, capprops = dict(capprops4,color=color4_), whiskerprops = dict(whiskerprops4,color=color4_), meanline=False, showmeans=True)


ylims = ax.get_ylim()
yscale = (ylims[1] - ylims[0])//(ylims[1]//2)

print('R@R')
p1 = f_stats.permutation_p_diff(euDists_rnns_ed2_1, euDists_rnns_ed2_2)
ax.plot(lineh, np.full_like(lineh, 2+0.03), 'k-')
ax.plot(np.full_like(linev+2, 0.3), linev+2, 'k-')
ax.plot(np.full_like(linev+2, 0.7), linev+2, 'k-')
ax.text(0.5,2.1, f'{f_plotting.sig_marker(p1,ns_note=True)}',horizontalalignment='center', fontsize=12)
print(f"MD = {euDists_rnns_ed2_1.mean() - euDists_rnns_ed2_2.mean():.3f}, p = {p1:.3f}, g = {f_stats.hedges_g(euDists_rnns_ed2_1, euDists_rnns_ed2_2):.3f}")

print('R&U')
p2 = f_stats.permutation_p_diff(euDists_rnns_ed12_1, euDists_rnns_ed12_2)
ax.plot(lineh+1, np.full_like(lineh, 2+0.03), 'k-')
ax.plot(np.full_like(linev+2, 1.3), linev+2, 'k-')
ax.plot(np.full_like(linev+2, 1.7), linev+2, 'k-')
ax.text(1.5,2.1, f'{f_plotting.sig_marker(p2,ns_note=True)}',horizontalalignment='center', fontsize=12)
print(f"MD = {euDists_rnns_ed12_1.mean() - euDists_rnns_ed12_2.mean():.3f}, p = {p2:.3f}, g = {f_stats.hedges_g(euDists_rnns_ed12_1, euDists_rnns_ed12_2):.3f}")


print('LPFC')
p3 = f_stats.permutation_p_diff(euDists_data_dlpfc_1, euDists_data_dlpfc_2) # shape: nIters * ntrials
ax.plot(lineh+2, np.full_like(lineh, 2+0.03), 'k-')
ax.plot(np.full_like(linev+2, 2.3), linev+2, 'k-')
ax.plot(np.full_like(linev+2, 2.7), linev+2, 'k-')
ax.text(2.5,2.1, f'{f_plotting.sig_marker(p3,ns_note=True)}',horizontalalignment='center', fontsize=12)
print(f"MD = {euDists_data_dlpfc_1.mean() - euDists_data_dlpfc_2.mean():.3f}, p = {p3:.3f}, g = {f_stats.hedges_g(euDists_data_dlpfc_1, euDists_data_dlpfc_2):.3f}")

print('FEF')
p4 = f_stats.permutation_p_diff(euDists_data_fef_1, euDists_data_fef_2)
ax.plot(lineh+3, np.full_like(lineh, 2+0.03), 'k-')
ax.plot(np.full_like(linev+2, 3.3), linev+2, 'k-')
ax.plot(np.full_like(linev+2, 3.7), linev+2, 'k-')
ax.text(3.5,2.1, f'{f_plotting.sig_marker(p4,ns_note=True)}',horizontalalignment='center', fontsize=12)
print(f"MD = {euDists_data_fef_1.mean() - euDists_data_fef_2.mean():.3f}, p = {p4:.3f}, g = {f_stats.hedges_g(euDists_data_fef_1, euDists_data_fef_2):.3f}")

ax.plot([], c='dimgrey', label='Retarget')
ax.plot([], c='lightgrey', label='Distraction')


ax.set_xticks([0.5,1.5,2.5,3.5],['R@R','R&U','LPFC','FEF'], rotation=0)
ax.set_xlabel('Strategies/Regions', labelpad = 5, fontsize = 12)
ax.set_ylabel('Euclidean Distance (Std. Scaled)', labelpad = 3, fontsize = 10)
ax.set_ylim(top=2.5)
plt.suptitle('Mean Projection Drift', fontsize = 20, y=1)
plt.show()

#%%



#%%

###############
# drift ratio #
###############

#%%

lineh = np.arange(0.3,0.7,0.001)
linev = np.arange(0,0.1,0.001)

pltBsl = True

fig, axes = plt.subplots(1,1, figsize=(4,6),dpi=300, sharey=True)

################
# figure props #
################

color1, color1_, color2, color2_ = '#d29c2f', '#f5df7a', '#3c79b4', '#b3cde4'
color3, color3_, color4, color4_ = '#185337', '#96d9ad', '#804098', '#c4a2d1'

boxprops0 = dict(facecolor='lightgrey', edgecolor='none',alpha = 1)
flierprops0 = dict(markeredgecolor='lightgrey', markerfacecolor='lightgrey',alpha = 1)
capprops0 = dict(color='lightgrey',alpha = 1)
whiskerprops0 = dict(color='lightgrey',alpha = 1)
meanpointprops0 = dict(marker='^', markeredgecolor='lightgrey', markerfacecolor='w',alpha = 1)

boxprops1 = dict(facecolor=color1, edgecolor='none')
flierprops1 = dict(markeredgecolor=color1, markerfacecolor=color1)
capprops1 = dict(color=color1)
whiskerprops1 = dict(color=color1)
meanpointprops1 = dict(marker='^', markeredgecolor=color1, markerfacecolor='w')

boxprops2 = dict(facecolor=color2, edgecolor='none',alpha = 1)
flierprops2 = dict(markeredgecolor=color2, markerfacecolor=color2,alpha = 1)
capprops2 = dict(color=color2,alpha = 1)
whiskerprops2 = dict(color=color2,alpha = 1)
meanpointprops2 = dict(marker='^', markeredgecolor=color2, markerfacecolor='w',alpha = 1)

boxprops3 = dict(facecolor=color3, edgecolor='none')
flierprops3 = dict(markeredgecolor=color3, markerfacecolor=color3)
capprops3 = dict(color=color3)
whiskerprops3 = dict(color=color3)
meanpointprops3 = dict(marker='^', markeredgecolor=color3, markerfacecolor='w')

boxprops4 = dict(facecolor=color4, edgecolor='none',alpha = 1)
flierprops4 = dict(markeredgecolor=color4, markerfacecolor=color4,alpha = 1)
capprops4 = dict(color=color4,alpha = 1)
whiskerprops4 = dict(color=color4,alpha = 1)
meanpointprops4 = dict(marker='^', markeredgecolor=color4, markerfacecolor='w',alpha = 1)

medianprops = dict(linestyle='--', linewidth=1, color='w')


ax = axes

########
# rnns #
########

# load drift distances
euDists_rnns_ed2_1 = np.array(euDists_rnns['ed2'][1]).squeeze().mean(1)
euDists_rnns_ed2_2 = np.array(euDists_rnns['ed2'][2]).squeeze().mean(1)
euDists_rnns_ed12_1 = np.array(euDists_rnns['ed12'][1]).squeeze().mean(1)
euDists_rnns_ed12_2 = np.array(euDists_rnns['ed12'][2]).squeeze().mean(1)

euDists_rnns_ed2_1_shuff = np.array(euDists_rnns_shuff['ed2'][1]).squeeze().mean(1).mean(1)
euDists_rnns_ed2_2_shuff = np.array(euDists_rnns_shuff['ed2'][2]).squeeze().mean(1).mean(1)
euDists_rnns_ed12_1_shuff = np.array(euDists_rnns_shuff['ed12'][1]).squeeze().mean(1).mean(1)
euDists_rnns_ed12_2_shuff = np.array(euDists_rnns_shuff['ed12'][2]).squeeze().mean(1).mean(1)

# calculate drift ratio
euDistsRatio_rnns_ed2 = euDists_rnns_ed2_2/euDists_rnns_ed2_1
euDistsRatio_rnns_ed12 = euDists_rnns_ed12_2/euDists_rnns_ed12_1
euDistsRatio_rnns_ed2_shuff = euDists_rnns_ed2_2_shuff/euDists_rnns_ed2_1_shuff
euDistsRatio_rnns_ed12_shuff = euDists_rnns_ed12_2_shuff/euDists_rnns_ed12_1_shuff

# plot
ax.boxplot([euDistsRatio_rnns_ed2], positions=[0.5-pltBsl*0.2], patch_artist=True, widths = 0.2, boxprops=boxprops1, flierprops=flierprops1,
                    meanprops=meanpointprops1, medianprops=medianprops, capprops = capprops1, whiskerprops = whiskerprops1, meanline=False, showmeans=True)
ax.boxplot([euDistsRatio_rnns_ed12], positions=[1.5-pltBsl*0.2], patch_artist=True, widths = 0.2, boxprops=boxprops2, flierprops=flierprops2,
                    meanprops=meanpointprops2, medianprops=medianprops, capprops = capprops2, whiskerprops = whiskerprops2, meanline=False, showmeans=True)

# whether include baseline distributions or not
if pltBsl:
    ax.boxplot([euDistsRatio_rnns_ed2_shuff], positions=[0.5+pltBsl*0.2], patch_artist=True, widths = 0.2, boxprops=boxprops0, flierprops=flierprops0,
                        meanprops=meanpointprops0, medianprops=medianprops, capprops = capprops0, whiskerprops = whiskerprops0, meanline=False, showmeans=True)
    ax.boxplot([euDistsRatio_rnns_ed12_shuff], positions=[1.5+pltBsl*0.2], patch_artist=True, widths = 0.2, boxprops=boxprops0, flierprops=flierprops0,
                        meanprops=meanpointprops0, medianprops=medianprops, capprops = capprops0, whiskerprops = whiskerprops0, meanline=False, showmeans=True)

# significance test
p1 = f_stats.permutation_pCI(euDistsRatio_rnns_ed2, euDistsRatio_rnns_ed2_shuff, tail='smaller', alpha=5)
ax.text(0.5,1.3, f'{f_plotting.sig_marker(p1,ns_note=True)}',horizontalalignment='center', fontsize=12)
print(f'R@R: M(SD) = {euDistsRatio_rnns_ed2.mean():.3f}({euDistsRatio_rnns_ed2.std():.3f}), p = {p1:.3f}, 95CI = [{np.quantile(euDistsRatio_rnns_ed2,0.025):.3f}, {np.quantile(euDistsRatio_rnns_ed2,0.975):.3f}]')

p2 = f_stats.permutation_pCI(euDistsRatio_rnns_ed12, euDistsRatio_rnns_ed12_shuff, tail='smaller', alpha=5)
ax.text(1.5,1.3, f'{f_plotting.sig_marker(p2,ns_note=True)}',horizontalalignment='center', fontsize=12)
print(f'R&U: M(SD) = {euDistsRatio_rnns_ed12.mean():.3f}({euDistsRatio_rnns_ed12.std():.3f}), p = {p2:.3f}, 95CI = [{np.quantile(euDistsRatio_rnns_ed12,0.025):.3f}, {np.quantile(euDistsRatio_rnns_ed12,0.975):.3f}]')

###########
# monkeys #
###########

# load drift distances
euDists_data_dlpfc_1 = np.array(euDists_monkeys['dlpfc'][1]).mean(1)
euDists_data_dlpfc_2 = np.array(euDists_monkeys['dlpfc'][2]).mean(1)
euDists_data_fef_1 = np.array(euDists_monkeys['fef'][1]).mean(1)
euDists_data_fef_2 = np.array(euDists_monkeys['fef'][2]).mean(1)

euDists_data_dlpfc_1_shuff = np.array(euDists_monkeys_shuff['dlpfc'][1]).mean(1).mean(1)
euDists_data_dlpfc_2_shuff = np.array(euDists_monkeys_shuff['dlpfc'][2]).mean(1).mean(1)
euDists_data_fef_1_shuff = np.array(euDists_monkeys_shuff['fef'][1]).mean(1).mean(1)
euDists_data_fef_2_shuff = np.array(euDists_monkeys_shuff['fef'][2]).mean(1).mean(1)

# calculate drift ratio
euDistsRatio_data_dlpfc = euDists_data_dlpfc_2/euDists_data_dlpfc_1
euDistsRatio_data_fef = euDists_data_fef_2/euDists_data_fef_1
euDistsRatio_data_dlpfc_shuff = euDists_data_dlpfc_2_shuff/euDists_data_dlpfc_1_shuff
euDistsRatio_data_fef_shuff = euDists_data_fef_2_shuff/euDists_data_fef_1_shuff

# plot
ax.boxplot([euDistsRatio_data_dlpfc], positions=[2.5-pltBsl*0.2], patch_artist=True, widths = 0.2, boxprops=boxprops3, flierprops=flierprops3,
                    meanprops=meanpointprops3, medianprops=medianprops, capprops = capprops3, whiskerprops = whiskerprops3, meanline=False, showmeans=True)
ax.boxplot([euDistsRatio_data_fef], positions=[3.5-pltBsl*0.2], patch_artist=True, widths = 0.2, boxprops=boxprops4, flierprops=flierprops4,
                    meanprops=meanpointprops4, medianprops=medianprops, capprops = capprops4, whiskerprops = whiskerprops4, meanline=False, showmeans=True)

# whether include baseline distributions or not
if pltBsl:
    ax.boxplot([euDistsRatio_data_dlpfc_shuff], positions=[2.5+pltBsl*0.2], patch_artist=True, widths = 0.2, boxprops=boxprops0, flierprops=flierprops0,
                        meanprops=meanpointprops0, medianprops=medianprops, capprops = capprops0, whiskerprops = whiskerprops0, meanline=False, showmeans=True)
    ax.boxplot([euDistsRatio_data_fef_shuff], positions=[3.5+pltBsl*0.2], patch_artist=True, widths = 0.2, boxprops=boxprops0, flierprops=flierprops0,
                        meanprops=meanpointprops0, medianprops=medianprops, capprops = capprops0, whiskerprops = whiskerprops0, meanline=False, showmeans=True)

# significance test
p3 = f_stats.permutation_pCI(euDistsRatio_data_dlpfc, euDistsRatio_data_dlpfc_shuff, tail='smaller', alpha=5)
ax.text(2.5,1.3, f'{f_plotting.sig_marker(p3,ns_note=True)}',horizontalalignment='center', fontsize=12)

p4 = f_stats.permutation_pCI(euDistsRatio_data_fef, euDistsRatio_data_fef_shuff, tail='smaller', alpha=5)
ax.text(3.5,1.3, f'{f_plotting.sig_marker(p4,ns_note=True)}',horizontalalignment='center', fontsize=12)

print(f'LPFC: M(SD) = {euDistsRatio_data_dlpfc.mean():.3f}({euDistsRatio_data_dlpfc.std():.3f}), p = {p3:.3f}, 95CI = [{np.quantile(euDistsRatio_data_dlpfc,0.025):.3f}, {np.quantile(euDistsRatio_data_dlpfc,0.975):.3f}]')

print(f'FEF: M(SD) = {euDistsRatio_data_fef.mean():.3f}({euDistsRatio_data_fef.std():.3f}), p = {p4:.3f}, 95CI = [{np.quantile(euDistsRatio_data_fef,0.025):.3f}, {np.quantile(euDistsRatio_data_fef,0.975):.3f}]')

ax.set_xticks([0.5,1.5,2.5,3.5],['R@R','R&U','LPFC','FEF'], rotation=0)
ax.set_xlabel('Strategies/Regions', labelpad = 5, fontsize = 12)
ax.set_ylabel('Distractor/Retarget', labelpad = 3, fontsize = 10)
ax.set_ylim(top=1.4)
plt.suptitle('Mean Projection Drift Ratio', fontsize = 20, y=1)
plt.show()


