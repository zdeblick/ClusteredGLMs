import numpy as np
from scipy.signal import correlate
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.metrics import confusion_matrix, pairwise_distances, adjusted_rand_score
from sklearn.mixture import GaussianMixture
from itertools import combinations
import os
from helpers import *
from scipy.optimize import linear_sum_assignment


prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']
os.chdir('files')
savepath = '../figs/'
run=True
Kfits = np.arange(1,21)
import colorcet as cc
cmap = cc.cm.rainbow
import matplotlib as mpl 
mpl.rcParams['xtick.labelsize'] = 14
mpl.rcParams['ytick.labelsize'] = 14
mpl.rcParams['axes.labelsize'] = 14
mpl.rcParams['figure.max_open_warning'] = 100

if False:
    # Comparing all 4 best solutions
    fnames = []
    for share in ['W','all']:
        #Simultaneous
        simulBICs = np.load('../summary_files/BIC_allN_share='+share+'.npz')['simulBICs']
        K_max=Kfits[np.argmax(np.max(simulBICs,axis=1))]
        trial_max = np.argmax(simulBICs[Kfits==K_max,:])
        fnames.append('ivscc_n1t2v_simulreg_share_'+share+str(trial_max)+'_K='+str(K_max)+'_sub=allN_train_reps=all')

        #Sequential
        seqBICs = np.load('../summary_files/BIC_allN_share='+share+'.npz')['seqBICs']
        K_max=Kfits[np.argmax(np.max(seqBICs,axis=1))]
        fnames.append('ivscc_n1t2v_seqreg_share_'+share+'_K='+str(K_max)+'_sub=allN_train_reps=all')
    Ds = [np.load(fname+'.npz',allow_pickle=True) for fname in fnames]
    cases = ['A','A','B','B']
    meths = ['Simultaneous','Sequential']*2
    for i in range(len(Ds)):
        for j in range(i):
            fig,ax = plt.subplots()
            cm = confusion_matrix(np.argmax(Ds[i]['Q'],axis=1),np.argmax(Ds[j]['Q'],axis=1))
            row_inds,col_inds = linear_sum_assignment(-cm)
            cm = cm[row_inds,:][:,col_inds]
            cm = cm[:,np.sum(cm,axis=0)>0][np.sum(cm,axis=1)>0,:]
            cm = cm/(np.sqrt(np.sum(cm,axis=1,keepdims=True)*np.sum(cm,axis=0,keepdims=True)))
            im = ax.imshow(cm,origin='lower',cmap=cc.cm.blues)#,vmin=0,vmax=50)
            ars = adjusted_rand_score(np.argmax(Ds[i]['Q'],axis=1),np.argmax(Ds[j]['Q'],axis=1))
            ax.set_title('ARS='+str(np.round(ars,2)))

            ax.set_xlim([-0.5,cm.shape[1]-0.5])
            ax.set_ylim([cm.shape[0]-0.5,-0.5])
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_ylabel(meths[i]+' method, case '+cases[i])
            ax.set_xlabel(meths[j]+' method, case '+cases[j])
            pixel_size_in = 0.35
            set_size(pixel_size_in*cm.shape[1],pixel_size_in*cm.shape[0],ax=ax)

            plt.savefig(savepath+'khat_comps_'+meths[i]+cases[i]+'_vs_'+meths[j]+cases[j]+'.png',bbox_inches='tight')
            plt.close() 





D = np.load('../ivscc_data_n12.npz',allow_pickle=True)
all_spks = D['binned_spikes']
dt = D['bin_len']*1000 #ms
N=len(all_spks)
d = [10,20]
downsample = 5

#Autocorrelation figure
lags = 5100
sm_l = 21
autocorrs = np.zeros((N,2*lags+1))
rvals = np.zeros((N,))
for n in range(N):
    trial_avg = np.mean(np.vstack(all_spks[n]),axis=0)
    trial_avg = correlate(trial_avg,np.ones((sm_l,))/sm_l,mode='same')
    # trial_avg = np.mean(np.vstack(all_spks[n]))
    resids = np.hstack([trial_spks-trial_avg for trial_spks in all_spks[n]])
    c = correlate(resids,np.tile(resids,3))
    rvals[n] = np.mean(np.array([pearsonr(tr-trial_avg,tr2-trial_avg)[0] for tr,tr2 in combinations(all_spks[n],2)]))
    autocorrs[n,:] = c[(c.size-1)//2-lags:(c.size-1)//2+lags+1]/c[(c.size-1)//2]

which_lags = np.arange(-lags,lags+1)%500!=0-1
autocorrs = autocorrs[:,which_lags]

# autocorrs = correlate(autocorrs,np.ones((1,sm_l))/sm_l,mode='same')
autocorrs/=np.max(autocorrs,keepdims=True,axis=1)
which_lags[:sm_l] = False
which_lags[-sm_l:] = False
fig,ax = plt.subplots()
plt.hist(rvals,bins = np.arange(-0.3,1.0001,0.1))
plt.xlabel('Pearson\'s r between trial residuals',fontsize=14)
plt.ylabel('Number of neurons',fontsize=14)
# plt.xlim(-1.01,-0.99)
# plt.xticks(np.arange(0,1.1,0.2))
plt.grid()
fig.tight_layout()
plt.savefig(savepath+'ivscc_trial_rvals')
plt.show()
plt.close()


fig,ax = plt.subplots()
plt.plot((dt*np.arange(-lags,lags+1)/1000)[which_lags], autocorrs[:10,sm_l:-sm_l].T,linewidth=0.8)
plt.xlabel('Time (s)',fontsize=14)
plt.ylabel('Autocorrelation of spike residuals',fontsize=14)
plt.xticks(np.arange(-9,10,3))
plt.grid()
# plt.xlim(-9.01,-8.99)
fig.tight_layout()
plt.savefig(savepath+'ivscc_autocorrs')
plt.show()
plt.close()

stop



# IVSCC-based simulations



share = 'all'
l2s = np.logspace(-7,-1,13) if share!='all' else np.array([0])
trials = 20

Kfit=5
fname = 'sim_frivsccsimul_simul'+str(0)+'_Kfit'+str(Kfit)+'_l2i'+str(0)+'_share='+share
D_true = np.load(fname+'.npz',allow_pickle=True)
N = D_true['Q'].shape[0]
true_Fs = D_true['true_betas'][:d[0]]
true_Ws = D_true['true_betas'][d[0]:-1]
true_mus = D_true['true_mus']
fname = 'sim_frivsccsimul_seq_l2stimi='+str(0)+'_l2selfi='+str(0)+'_share='+share
D_true = np.load(fname+'.npz',allow_pickle=True)
true_Sigmas = np.zeros((1,))
_,true_wts = np.unique(D_true['true_ks'],return_counts=True)
true_wts=true_wts/N
print(true_wts, true_mus.shape)
thresh=-4
Sig_to_sigs = lambda X: [np.sqrt(np.diag(X[k])) if len(X[k].shape)==2 else np.sqrt(X[k]) for k in range(len(X))]


# Simultaneous
if run:
    for l2_i in range(l2s.size):
        max_BIC = -np.inf
        for trial in range(trials):
            fname = 'sim_frivsccsimul_simul'+str(trial)+'_Kfit'+str(Kfit)+'_l2i'+str(l2_i)+'_share='+share
            try:
                D = np.load(fname+'.npz',allow_pickle=True)
                print(D['bad'])
                if D['BIC']>max_BIC:# and not np.any(D['bad']):
                    max_BIC=D['BIC']
                    simul_D = {k:D[k] for k in ['ars','Ws','Fs','bs','mu_k','C_k','wts','l2']}
            except Exception as e:
                if True:
                    print(fname,e)
    print(simul_D['l2'])
    np.savez('../summary_files/ivscc_sims_share'+share,simul_D=simul_D)
D = np.load('../summary_files/ivscc_sims_share'+share+'.npz',allow_pickle=True)
simul_D = D['simul_D'][()]

if run:
    # Sequential
    errors = -np.inf*np.ones((l2s.size,l2s.size,2))
    for l2_stim_i in range(l2s.size):
        for l2_self_i in range(l2s.size):
            fname = 'sim_frivsccsimul_seq_l2stimi='+str(l2_stim_i)+'_l2selfi='+str(l2_self_i)+'_share='+share
            try:
                D = np.load(fname+'.npz',allow_pickle=True)
                errors[l2_stim_i,l2_self_i,0] = rmst(D['Fs'],true_Fs)
                errors[l2_stim_i,l2_self_i,1] = rmst(D['Ws'],true_Ws)
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
    print(betas.shape)
    max_bic = -np.inf
    for trial in range(trials):
        gmm = GaussianMixture(n_components=Kfit,covariance_type='diag', max_iter=100)
        gmm.fit(betas)
        bic = -gmm.bic(betas)
        if bic>max_bic:
            max_bic=bic
            seq_D = {'ars':adjusted_rand_score(D['true_ks'],gmm.predict(betas)),
                'mu_k':gmm.means_,'C_k':gmm.covariances_,'wts':gmm.weights_}
    seq_D.update({k:D[k] for k in ['Ws','Fs','bs']})
    np.savez('../summary_files/ivscc_sims_share'+share,seq_D=seq_D,simul_D=simul_D)
D = np.load('../summary_files/ivscc_sims_share'+share+'.npz',allow_pickle=True)

# Clusters Figure
print(D['simul_D'][()]['ars'],D['seq_D'][()]['ars'])
fig,ax = plt.subplots()
hs = []
labels = []
plot_data = [
    ('True', true_mus[:,d[0]:-1], Sig_to_sigs(true_Sigmas), true_wts, 'k'),
    ('Simultaneous (ARS='+str(np.round(D['simul_D'][()]['ars'],2))+')', D['simul_D'][()]['mu_k'][:,d[0]:-1], Sig_to_sigs(D['simul_D'][()]['C_k']), D['simul_D'][()]['wts'], 'r'), 
    ('Sequential (ARS='+str(np.round(D['seq_D'][()]['ars'],2))+')', D['seq_D'][()]['mu_k'][:,d[0]:-1], Sig_to_sigs(D['seq_D'][()]['C_k']), D['seq_D'][()]['wts'],'c')]
for label, means, sds, wts, color in plot_data:
    h = plt.plot(np.arange(1,d[1]+1)*dt,means.T,color+'-')
    hs.append(h[0])
    labels.append(label)
plt.xlabel('Time (ms)',fontsize=14)
plt.ylabel('$\\mu^\\mathrm{self}_k$',fontsize=14)
ax.set_ylim([-5,2])
plt.legend(hs,labels,fontsize=14)
plt.savefig(savepath+'ivscc_sims_clusters_share='+share+'.png',bbox_inches='tight')
plt.close()



fig,ax = plt.subplots()
hs = []
labels = []
plot_data = [
    ('True', true_mus[:,:d[0]], Sig_to_sigs(true_Sigmas), true_wts, 'k'),
    ('Simultaneous (ARS='+str(np.round(D['simul_D'][()]['ars'],2))+')', D['simul_D'][()]['mu_k'][:,:d[0]], Sig_to_sigs(D['simul_D'][()]['C_k']), D['simul_D'][()]['wts'], 'r'), 
    ('Sequential (ARS='+str(np.round(D['seq_D'][()]['ars'],2))+')', D['seq_D'][()]['mu_k'][:,:d[0]], Sig_to_sigs(D['seq_D'][()]['C_k']), D['seq_D'][()]['wts'],'c')]
for label, means, sds, wts, color in plot_data:
    h = plt.plot(np.arange(d[0])*dt*downsample,means.T,color+'-')
    hs.append(h[0])
    labels.append(label)
plt.xlabel('Time (ms)',fontsize=14)
plt.ylabel('$\\mu^\\mathrm{stim}_k$',fontsize=14)
plt.legend(hs,labels,fontsize=14)
plt.savefig(savepath+'ivscc_sims_Fclusters_share='+share+'.png',bbox_inches='tight')
plt.close()




print(D['simul_D'][()]['mu_k'][:,:d[0]])
