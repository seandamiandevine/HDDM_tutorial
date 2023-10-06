# ****************************************************************************
# *                                   Setup                                  *
# ****************************************************************************

# Packages to use 
# you will not many of these—depends on your project
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt 
import seaborn as sns
from scipy.stats import gaussian_kde
import os
from tqdm import tqdm

# setting up plots to look pretty (at least in my opinion...)
plt.style.use('grayscale')
matplotlib.rcParams.update({'font.size': 18})
matplotlib.rcParams['figure.figsize'] = (10.4, 6.8)
matplotlib.rcParams['lines.linewidth'] = 2.5
matplotlib.rcParams['axes.prop_cycle'] = matplotlib.cycler(color=['0', '0.5']) 

pd.options.mode.chained_assignment = None  # ignore chaining warning; default='warn'   

os.chdir('/Users/sean/documents/misc/hddm_tutorial')

# ****************************************************************************
# *                               Simulate Data                              *
# ****************************************************************************

# Sample two patient groups
true_a_HC, true_a_PD = .75, 2
true_v_HC, true_v_PD = 1.5, 1
true_t_HC, true_t_PD = .5, 1.1

N, J = 10, 100

# Generate data
dat, true_pars = hddm.generate.gen_rand_data({'HC':{'a':true_a_HC,'v':true_v_HC,'t':true_t_HC}, 
							 'PD':{'a':true_a_PD,'v':true_v_PD,'t':true_t_PD}},
							  size=J, subjs=N)

# Create continuous signal from RT as regressors
tau0 = .11
tau1 = .075
b0   = 1
b1   = .4
r    = .9

U0js = np.random.normal(0, tau0, N)
U1js = np.random.normal(0, tau1, N)

eeg = b0 + U0js[dat.subj_idx] + (b1+U1js[dat.subj_idx])*dat.rt + np.random.normal(0,r,dat.shape[0])
dat['eeg_z'] = (eeg-eeg.mean())/eeg.std()


# eeg_bin = pd.qcut(eeg, 6, labels=False)
# sns.lineplot(x=eeg_bin,y=dat.rt)
# plt.show()

dat.to_csv('example_data.csv')

dat.head()

# ****************************************************************************
# *                             1. Fit Basic HDDM                            *
# ****************************************************************************

import hddm

# Load data 
dat = pd.read_csv('example_data.csv')

# Fit HDDM to data
simple_ddm = hddm.HDDM(dat)
simple_ddm.find_starting_values()
simple_ddm.sample(2000, burn=500, dbname=f'simple_ddm_traces', db='pickle')
simple_ddm.save('simple_ddm')

# Load model if already run
# simple_ddm = hddm.load('simple_ddm')

simple_ddm.gen_stats()

a,v,t = simple_ddm.nodes_db.node[['a','v','t']]

# traceplots
fig, ax = plt.subplots(1,3,figsize=[16,6],constrained_layout=True)

ax[0].plot(a.trace())
ax[0].set(title='Decision Threshold')

ax[1].plot(v.trace())
ax[1].set(title='Drift Rate')

ax[2].plot(t.trace())
ax[2].set(title='Non-Decision Time')

plt.show()
fig.savefig('figs/basic_hddm_traceplot.png')
plt.close()

# density plots
fig,ax = plt.subplots(1,3,figsize=[16,6],constrained_layout=True)

sns.kdeplot(x=a.trace(),ax=ax[0])
ax[0].set(title='Decision Threshold')

sns.kdeplot(x=v.trace(),ax=ax[1])
ax[1].set(title='Drift Rate')

sns.kdeplot(x=t.trace(),ax=ax[2])
ax[2].set(title='Non-Decision Time')

plt.show()
fig.savefig('figs/basic_hddm_density.png')
plt.close()

## compare two subjects
v1,v5 = simple_ddm.nodes_db.node[['v_subj.1','v_subj.5']]

xaxis = np.linspace(np.concatenate([v1.trace(),v5.trace()]).min(),np.concatenate([v1.trace(),v5.trace()]).max(), 500)

dens1 = gaussian_kde(v1.trace())(xaxis)
dens5 = gaussian_kde(v5.trace())(xaxis)

dens1_0 = dens1/dens1.max()
dens5_0 = dens5/dens5.max()

fig,ax = plt.subplots(1)

ax.plot(xaxis,dens1_0, label='Subject 1', c='darkblue')
ax.fill_between(xaxis, dens1_0, color='darkblue',alpha=0.25)

ax.plot(xaxis,dens5_0, label='Subject 5', c='darkred')
ax.fill_between(xaxis, dens5_0, color='darkred', alpha=0.25)

ax.set(title='Drift Rates', xlabel='Parameter Value', ylabel='Normalized Density')
ax.legend(title='Subject')

plt.show()
fig.savefig('figs/subject_comparison.png')
plt.close()

# ****************************************************************************
# *                          2. Fit Conditional HDDM                         *
# ****************************************************************************

# Fit conditional HDDM to data
cond_ddm = hddm.HDDM(dat, depends_on={'a':'condition', 'v':'condition', 't':'condition'})
cond_ddm.find_starting_values()
cond_ddm.sample(2000, burn=500, dbname=f'cond_ddm_traces', db='pickle')
cond_ddm.save('cond_ddm')

# Load model if already run
# cond_ddm = hddm.load('cond_ddm')

cond_ddm_stats = cond_ddm.gen_stats()
print(cond_ddm_stats.to_string())

# Density plot compare
titles = {'a':'Decision Boundary', 'v':'Drift Rate', 't':'Non-Decision Time'}

for i,p in enumerate(['a','v','t']): 
	
	t1,t2 = cond_ddm.nodes_db.node[[f'{p}(HC)', f'{p}(PD)']]

	xaxis = np.linspace(np.concatenate([t1.trace(),t2.trace()]).min(),np.concatenate([t1.trace(),t2.trace()]).max(), 500)

	densHC = gaussian_kde(t1.trace())(xaxis)
	densPD = gaussian_kde(t2.trace())(xaxis)

	densHC_0 = densHC/densHC.max()
	densPD_0 = densPD/densPD.max()

	fig,ax = plt.subplots(1)

	ax.plot(xaxis,densHC_0, label='Controls', c='darkblue')
	ax.fill_between(xaxis, densHC_0, color='darkblue',alpha=0.25)

	ax.plot(xaxis,densPD_0, label='Parkinson', c='darkred')
	ax.fill_between(xaxis, densPD_0, color='darkred', alpha=0.25)

	ax.set(title=titles[p], xlabel='Parameter Value', ylabel='Normalized Density')
	ax.legend(title='Condition')

	plt.show()
	fig.savefig(f'figs/cond_post_{p}')
	plt.close()

# Bayesian p-value

## Decision boundary
aHC,aPD = cond_ddm.nodes_db.node[['a(HC)', 'a(PD)']]
np.mean(aHC.trace() > aPD.trace())

## Drift rate
vHC,vPD = cond_ddm.nodes_db.node[['v(HC)', 'v(PD)']]
np.mean(vHC.trace() < vPD.trace())

## N.d. time
tHC,tPD = cond_ddm.nodes_db.node[['t(HC)', 't(PD)']]
np.mean(tHC.trace() > tPD.trace())

# ****************************************************************************
# *                             3. HDDM Regressor                            *
# ****************************************************************************

reg_ddm = hddm.models.HDDMRegressor(dat, ['v ~ eeg_z'])
reg_ddm.find_starting_values()
reg_ddm.sample(2000, burn=500, dbname=f'reg_ddm_traces', db='pickle')
reg_ddm.save('reg_ddm')

# Load model if already run
# reg_ddm = hddm.load('reg_ddm')

reg_ddm.gen_stats()


# Visualize v_eeg_z
v_eeg_z = reg_ddm.nodes_db.node['v_eeg_z']

xaxis = np.linspace(v_eeg_z.trace().min(), v_eeg_z.trace().max(), 500)
dens  = gaussian_kde(v_eeg_z.trace())(xaxis)
q     = np.quantile(xaxis, (.025,.975))

fig,ax = plt.subplots(1)

ax.plot(xaxis,dens)
ax.fill_between(xaxis[(xaxis > q[0]) & (xaxis < q[1])], dens[(xaxis > q[0]) & (xaxis < q[1])], 
	color='red', alpha=0.25, label='95% CI')
ax.axvline(0,linestyle='--',c='grey')
ax.set(xlim=[xaxis.min()-.05,0.05], title=r'$v_{eeg_z}$', xlabel='Regression Weight', ylabel='Density')
ax.legend()

plt.show()
fig.savefig(f'figs/v_eeg_z.png')
plt.close()

p_eeg_z_lt_0 = np.mean(v_eeg_z.trace() > 0)


q  = np.quantile(xaxis, (.025,.975))




