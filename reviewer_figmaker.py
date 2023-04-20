import numpy as np
from scipy.signal import correlate
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, pairwise_distances, adjusted_rand_score
from sklearn.mixture import GaussianMixture
import os
from helpers import *

prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']
os.chdir('files')
savepath = '../figs/'
run=True

D = np.load('../ivscc_data_n12.npz',allow_pickle=True)
all_spks = D['binned_spikes']
dt = D['bin_len']*1000 #ms
N=len(all_spks)
d = [10,20]
downsample = 5

#Autocorrelation figure
lags = 1100
if run:
    autocorrs = np.zeros((N,2*lags+1))
    for n in range(N):
        spks = np.hstack(all_spks[n])
        c = correlate(spks,spks)
        autocorrs[n,:] = c[(c.size-1)//2-lags:(c.size-1)//2+lags+1]/c[(c.size-1)//2]
    np.savez('../summary_files/ivscc_data_autocorrs',autocorrs=autocorrs)
autocorrs = np.load('../summary_files/ivscc_data_autocorrs.npz')['autocorrs']

fig,ax = plt.subplots()
plt.plot(dt*np.arange(-lags,lags+1)/1000, autocorrs[:5,:].T)
plt.xlabel('Time (s)',fontsize=14)
plt.ylabel('Cross-correlation of spiking responses',fontsize=14)
plt.savefig(savepath+'ivscc_autocorrs')
plt.close()


# IVSCC-based simulations
Kfits = np.arange(1,21)
l2s = np.logspace(-7,-1,13)

share = 'W'
trials = 18

simulBICs = np.load('../summary_files/BIC_allN_share='+share+'.npz')['simulBICs']
K_max=Kfits[np.argmax(np.max(simulBICs,axis=1))]
trial_max = np.argmax(simulBICs[Kfits==K_max,:])
D_true = np.load('ivscc_n1t2v_simulreg_share_'+share+str(trial_max)+'_K='+str(K_max)+'_sub=allN_train_reps=all.npz',allow_pickle=True)
N = D_true['Q'].shape[0]
true_Fs = D_true['Fs']
true_Ws = D_true['Ws']
true_mus = D_true['mu_k']
true_Sigmas = D_true['C_k']
thresh=-4
Sig_to_sigs = lambda X: [np.sqrt(np.diag(X[k])) if len(X[k].shape)==2 else np.sqrt(X[k]) for k in range(len(X))]

# Simultaneous
if run:
    simul_bics = -np.inf*np.ones((Kfits.size,trials))
    for Ki,Kfit in enumerate(Kfits):
        for l2_i in range(l2s.size):
            max_BIC = -np.inf
            for trial in range(trials):
                fname = 'sim_frivsccsimul_simul'+str(trial)+'_Kfit'+str(Kfit)+'_l2i'+str(l2_i)+'_share='+share
                try:
                    D = np.load(fname+'.npz',allow_pickle=True)
                    simul_bics[Ki,trial] = D['BIC']
                    if D['BIC']>max_BIC:
                        max_BIC=D['BIC']
                        if Kfit==K_max:
                            simul_D = {k:D[k] for k in ['ars','Ws','Fs','bs','mu_k','C_k','wts','l2']}
                except Exception as e:
                    if True:
                        print(fname,e)
    print(simul_D['l2'])
    simul_D['BICs'] = simul_bics
    np.savez('../summary_files/ivscc_sims_share'+share,simul_D=simul_D)
D = np.load('../summary_files/ivscc_sims_share'+share+'.npz',allow_pickle=True)
simul_D = D['simul_D'][()]

if run:
    # Sequential
    seq_bics = -np.inf*np.ones((Kfits.size,trials))
    errors = -np.inf*np.ones((l2s.size,l2s.size,2))
    for l2_stim_i in range(l2s.size):
        for l2_self_i in range(l2s.size):
            fname = 'sim_frivsccsimul_seq_l2stimi='+str(l2_stim_i)+'_l2selfi='+str(l2_self_i)+'_share='+share
            try:
                D = np.load(fname+'.npz',allow_pickle=True)
                errors[l2_stim_i,l2_self_i,0] = rms(np.ravel(np.squeeze(D['Fs'])-true_Fs)[np.ravel(true_Fs)>thresh])
                errors[l2_stim_i,l2_self_i,1] = rms(np.ravel(D['Ws']-true_Ws)[np.ravel(true_Ws)>thresh])
            except:
                print(fname)
    l2_stim_i = np.argmin(np.min(errors[:,:,0],axis=1))
    l2_self_i = np.argmin(np.min(errors[:,:,1],axis=0))
    fname = 'sim_frivsccsimul_seq_l2stimi='+str(l2_stim_i)+'_l2selfi='+str(l2_self_i)+'_share='+share
    print(l2_stim_i,l2_self_i)
    D = np.load(fname+'.npz',allow_pickle=True)
    if share=='W':
        betas = D['Ws']
    else:
        betas = np.hstack([np.squeeze(D['Fs']),D['Ws'],np.expand_dims(D['bs'],1)])

    max_bic = -np.inf
    for Ki,Kfit in enumerate(Kfits):
        for trial in range(trials):
            gmm = GaussianMixture(n_components=Kfit,covariance_type='diag', max_iter=100)
            gmm.fit(betas)
            bic = -gmm.bic(betas)
            seq_bics[Ki,trial] = bic
            if bic>max_bic:
                max_bic=bic
                seq_D = {'ars':adjusted_rand_score(D['true_ks'],gmm.predict(betas)),
                    'mu_k':gmm.means_,'C_k':gmm.covariances_,'wts':gmm.weights_}
    seq_D.update({k:D[k] for k in ['Ws','Fs','bs']})
    seq_D['BICs'] = seq_bics
    np.savez('../summary_files/ivscc_sims_share'+share,seq_D=seq_D,simul_D=simul_D)
D = np.load('../summary_files/ivscc_sims_share'+share+'.npz',allow_pickle=True)

# Clusters Figure
fig,ax = plt.subplots()
hs = []
labels = []
plot_data = [
    ('True', true_mus, Sig_to_sigs(true_Sigmas), 'k'),
    ('Simultaneous', D['simul_D'][()]['mu_k'], Sig_to_sigs(D['simul_D'][()]['C_k']), 'r'), 
    ('Sequential', D['seq_D'][()]['mu_k'], Sig_to_sigs(D['seq_D'][()]['C_k']),'c')]
for label, means, sds, color in plot_data:
    h = plt.plot(np.arange(1,d[1]+1)*dt,means.T,color+'-')
    hs.append(h[0])
    labels.append(label)
plt.xlabel('Time (ms)',fontsize=14)
plt.ylabel('$\\mu^\\mathrm{self}_k$',fontsize=14)
ax.set_ylim([-4,2])
plt.legend(hs,labels,fontsize=14)
plt.savefig(savepath+'ivscc_sims_clusters_share='+share+'.png',bbox_inches='tight')
plt.close()

# BIC figures
plot_data = [('Simultaneous',D['simul_D'][()]['BICs']), ('Sequential',D['seq_D'][()]['BICs'])]
for meth, bics in plot_data:
    print(bics)
    bics-=np.max(bics)
    fig,ax = plt.subplots()
    ax.plot(Kfits,bics,'o',c=colors[0])
    ax.plot(Kfits,np.max(bics,axis=1),c=colors[0])
    ax.set_ylim([-1600,100])
    ax.set_xticks(Kfits[::2])
    ax.set_xlabel('K',fontsize=14)
    ax.set_ylabel('BIC',fontsize=14)
    plt.title(meth+' Method',fontsize=14)
    plt.tight_layout()
    plt.savefig(savepath+'ivscc_sims_'+meth+'_BIC_share='+share)
    plt.close()




