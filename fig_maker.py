import numpy as np
from scipy import stats
import pandas as pd
import os

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, pairwise_distances, adjusted_rand_score
### We looked into AMI as an alternative to ARS (uncommenting inport below), results were similar
#from sklearn.metrics import adjusted_mutual_info_score as adjusted_rand_score 
from sklearn.mixture import GaussianMixture
from scipy.stats import multivariate_normal
from scipy.optimize import linear_sum_assignment
from helpers import *

import colorcet as cc
cmap = cc.cm.rainbow
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

print(os.getcwd())
os.chdir('files')
print(os.getcwd())
savepath = '../figs/'


methods = ['Simultaneous','Sequential']
nmeth = len(methods)
dt=2 #ms
downsample = 5
alg = 'trust-ncg'
d = [10,20]


### Simuation Figures
Ks = [3,5]
oracle_seeds = 10
seeds = 50
trials = 20
delta = 0.5
Wstds = list(np.logspace(-2,-0.5,10))[:-1]
Kfits = np.arange(1,9)
wi_ms=[0,7]


run = False #set to False if re-running


for share, Wtypes in [('W',True), ('all',False)]:
    if run:
        W_from_muk_inds = np.arange(d[1]) if share=='W' else np.arange(d[0],d[0]+d[1])

        arss = np.nan*np.ones((seeds,len(Ks),nmeth,len(Wstds)))
        MAD_Ws = np.nan*np.ones((seeds,len(Ks),nmeth,len(Wstds)))
        MAD_Fs = np.nan*np.ones((seeds,len(Ks),nmeth,len(Wstds)))
        Wstd_ests = np.nan*np.ones((seeds,len(Ks),nmeth,len(Wstds)))
        BICs = {wims:np.nan*np.ones((seeds,len(Ks),nmeth,Kfits.size)) for wims in wi_ms}
        VLs = {wims:np.nan*np.ones((seeds,len(Ks),nmeth,Kfits.size)) for wims in wi_ms}
        for si in range(seeds):
            seed = si+oracle_seeds
            for ki,K in enumerate(Ks):
                for wi, Wstd in enumerate(Wstds):
                    D_kek = None

                    ### Simultaneous method
                    for kfi,Kfit in enumerate(Kfits if wi in wi_ms else [K]):
                        D_best = None
                        min_loss = np.inf
                        for trial in range(trials):
                            fname = 'ivscc_gmmglm_simulations_reg'+'_Wtypes'*Wtypes+'_share_'+share+'_K='+str(K)+'_Kfit='+str(Kfit)+'_trial='+str(trial)+'_seed='+str(seed)+'_1000Wstd='+str(int(1000*Wstd))+'_100delta='+str(int(100*delta))+'_alg='+alg
                            D = np.load(fname+'.npz',allow_pickle=True)
                            if D['loss']<min_loss:
                                min_loss = D['loss']
                                D_best = dict(D)
                        if Kfit==K:
                            D_kek = dict(D_best)
                        if wi in wi_ms:
                            BICs[wi][si,ki,0,kfi] = D_best['BIC']
                            VLs[wi][si,ki,0,kfi] = D_best['val_loss']
                    D = D_kek
                    if D is None:
                        continue

                    ###Sequential Method
                    fname = 'ivscc_gmmglm_simulations_indreg'+'_Wtypes'*Wtypes+'_seed='+str(seed)+'_K='+str(K)+'_1000Wstd='+str(int(1000*Wstd))+'_100delta='+str(int(100*delta))
                    D_ind = np.load(fname+'.npz')
                    if share=='W':
                        betas = D_ind['Ws']
                        if wi in wi_ms:
                            val_betas = D_ind['Ws_val_neurons']
                    else:
                        betas = np.hstack([np.squeeze(D_ind['Fs']),D_ind['Ws'],np.expand_dims(D_ind['bs'],1)])
                        if wi in wi_ms:
                            val_betas = np.hstack([np.squeeze(D_ind['Fs_val_neurons']),D_ind['Ws_val_neurons'],np.expand_dims(D_ind['bs_val_neurons'],1)])
                    N = betas.shape[0]

                    if wi in wi_ms:
                        for kfi,Kfit in enumerate(Kfits):
                            gmm = GaussianMixture(n_components=Kfit,covariance_type='diag', max_iter=100,n_init=trials)
                            gmm.fit(betas)
                            BICs[wi][si,ki,1,kfi] = -gmm.bic(betas)
                            VLs[wi][si,ki,1,kfi] = -gmm.score(val_betas)

                    gmm = GaussianMixture(n_components=K,covariance_type='diag', max_iter=100,n_init=trials)
                    arss[si,ki,:,wi] = [D['ars'], adjusted_rand_score(gmm.fit_predict(betas),np.tile(np.arange(K),N//K))]

                    true_Ws = D['true_betas'][:,d[0]:d[0]+d[1]]
                    true_Fs = D['true_betas'][:,:d[0]]

                    Wstd_ests[si,ki,:,wi] = [mrm(D['C_k'][:,W_from_muk_inds,W_from_muk_inds]), mrm(gmm.covariances_[:,W_from_muk_inds])]
                    MAD_Ws[si,ki,:,wi] = [rmst(true_Ws,D['Ws']),rmst(true_Ws,D_ind['Ws'])]
                    MAD_Fs[si,ki,:,wi] = [rmst(true_Fs,np.squeeze(D['Fs'])),rmst(true_Fs,np.squeeze(D_ind['Fs']))]

                    #example sim
                    if (K==5) and (seed==10) and (wi==0):
                        simul_sds = np.array([np.sqrt(np.diag(D['C_k'][k]))[W_from_muk_inds] for k in range(K)])
                        fig,ax = plt.subplots()
                        hs = []
                        labels = []
                        plot_data = [
                            ('True', D['true_mus'][:,W_from_muk_inds], Wstd*np.ones_like(simul_sds), 'k'),
                            ('Simultaneous', D['mu_k'][:,W_from_muk_inds], simul_sds, 'r'), 
                            ('Sequential', gmm.means_[:,W_from_muk_inds], np.sqrt(gmm.covariances_)[:,W_from_muk_inds],'c')]
                        for label, means, sds, color in plot_data:
                            h = plt.plot(np.arange(d[1]+1)*dt,means.T,color+'-')
                            hs.append(h[0])
                            labels.append(label)
                        plt.xlabel('Time (s)',fontsize=14)
                        plt.ylabel('$\\mu^\\mathrm{self}_k$',fontsize=14)
                        ax.set_ylim([-4,2])
                        plt.legend(hs,labels,fontsize=14)
                        plt.savefig(savepath+'example_mu_self_share='+share+'.png',bbox_inches='tight')
                        plt.close()

                        if Wtypes:
                            fig,ax = plt.subplots()
                            plt.plot(np.arange(d[0])*downsample*dt,D['true_betas'][0,:d[0]],'k')
                            plt.ylabel('$\\beta^\\mathrm{stim}_i$',fontsize=14)
                            plt.xlabel('Time (s)',fontsize=14)
                            plt.savefig(savepath+'example_beta_stim.png',bbox_inches='tight')
                            plt.close()
                        else:
                            F_from_muk_inds = np.arange(d[0])
                            simul_sds = np.array([np.sqrt(np.diag(D['C_k'][k]))[F_from_muk_inds] for k in range(K)])
                            fig,ax = plt.subplots()
                            hs = []
                            labels = []
                            plot_data = [
                                ('True', D['true_mus'][:,F_from_muk_inds], Wstd*np.ones_like(simul_sds), 'k'),
                                ('Simultaneous', D['mu_k'][:,F_from_muk_inds], simul_sds, 'r'), 
                                ('Sequential', gmm.means_[:,F_from_muk_inds], np.sqrt(gmm.covariances_)[:,F_from_muk_inds],'c')]
                            for label, means, sds, color in plot_data:
                                h = plt.plot(np.arange(d[0])*downsample*dt,means.T,color+'-')
                                hs.append(h[0])
                                labels.append(label)
                            plt.xlabel('Time (s)',fontsize=14)
                            plt.ylabel('$\\mu^\\mathrm{self}_k$',fontsize=14)
                            ax.set_ylim([-4,2])
                            plt.legend(hs,labels,fontsize=14)
                            plt.savefig(savepath+'example_mu_stim.png',bbox_inches='tight')
                            plt.close()

        np.savez('../summary_files/sim_plot_data_share_'+share,arss=arss,MAD_Ws=MAD_Ws,MAD_Fs=MAD_Fs,Wstd_ests=Wstd_ests,BICs=BICs,Wstds=Wstds,Ks=Ks,Kfits=Kfits,VLs=VLs)

    D = np.load('../summary_files/sim_plot_data_share_'+share+'.npz',allow_pickle=True)

    plots = [('ARS','ARS between $k_i$ and $\\hat{k}_i$',D['arss']),
            ('W_err','$||\\beta^\\mathrm{self}_i-\\hat{\\beta}^\\mathrm{self}_i||_2$',D['MAD_Ws']),
            ('F_err','$||\\beta^\\mathrm{stim}_i-\\hat{\\beta}^\\mathrm{stim}_i||_2$',D['MAD_Fs']),
            ('Wstd_est','Median $\\sqrt{\\hat{\\Sigma}^\\mathrm{self}_k}$',D['Wstd_ests'])]
    ms_plots = [('BIC0','BIC',D['BICs'][()][0]),('BIC7','BIC',D['BICs'][()][7]),('VL0','Val. ANLL',D['VLs'][()][0]),('VL7','Val. ANLL',D['VLs'][()][7])]
    fmts = [['-r', '-c'], ['--r', '--c'], [':r', ':c']]
    lss = ['-','--',':']

    ### Figure!!!
    for (name, ylabel, data) in plots:
        fig,ax = plt.subplots()
        plt_data = np.nanmean(data,axis=0)
        plt_sem = np.nanstd(data,axis=0)/np.sqrt(np.sum(np.isfinite(data),axis=0))
        for ki in range(len(Ks)):
            for meth in range(nmeth):
                plt.errorbar(np.array(Wstds)*((2*ki+meth)*0.005+1),plt_data[ki,meth,:],yerr=plt_sem[ki,meth,:],fmt = fmts[ki][meth])

        for ki in range(len(Ks)*(name!='Wstd_est')):
            max_d = plt.ylim()[1]+(plt.ylim()[1]-plt.ylim()[0])*0.05*(1+ki)
            for wi, Wstd in enumerate(Wstds):
                diff = np.diff(data,axis=2)[:,ki,0,wi]
                p_val = stats.wilcoxon(diff[np.isfinite(diff)],alternative='less' if name=='ARS' else 'greater')[1]
                #p_val *= len(Ks)*len(Wstds) #panel-wise bonferroni correction
                plt.text(Wstd,max_d,np.format_float_scientific(p_val,precision=0),fontsize=10)
            plt.text(Wstds[0]/2,max_d,'K='+str(Ks[ki]))
        ax.set_xscale('log')

        plt.xlabel('$\\sigma$',fontsize=14)
        plt.ylabel(ylabel,fontsize=14)
        if name=='ARS':
             plt.legend(['Simultaneous','Sequential']+['']*2*(len(Ks)-1),ncol=len(Ks),markerfirst=False,title=' '*24+(' '*9).join(['K='+str(K) for K in Ks]), loc = 'upper right')
        if name== 'Wstd_est':
            plt.plot(Wstds,Wstds,'k')
            ax.set_yscale('log')
        plt.savefig(savepath+name+'_vs_Wstd_reg_share_'+share+'.png',bbox_inches='tight')
        plt.close()

    ### Figures!!!
    for (name, ylabel, data) in ms_plots:
        is_bic = name.startswith('BIC')
        fig,ax = plt.subplots()

        K_means = np.zeros((len(Ks),2))
        K_sems = np.zeros((len(Ks),2))
        for ki in range(len(Ks)):
            for meth in range(nmeth):
                if is_bic:
                    K_sel = Kfits[np.nanargmax(data[:,ki,meth,:],axis=-1)]
                else:
                    N = 40*Ks[ki]
                    se = 1.0*np.stack([plt_sem[ki,meth,:]]*seeds,axis=0)*np.sqrt(50)
                    K_sel = np.min(Kfits+1e300*(data[:,ki,meth,:]>np.min(data[:,ki,meth,:],axis=-1,keepdims=True)+se[np.argmin(data[:,ki,meth,:],axis=-1)]),axis=-1)
                a,_ = np.histogram(K_sel,bins=np.arange(Kfits[-1]+1)+0.5,density=True)
                width = 0.8/(2*len(Ks))
                xp = Kfits-0.4+width/2+(2*ki+meth)*width
                plt.bar(xp,a,width=width,color=fmts[ki][meth][-1] if ki==0 else 'white',hatch=None if ki==0 else '///' if ki==1 else '//////',edgecolor=fmts[ki][meth][-1])
                K_means[ki,meth] = np.mean(K_sel)
                K_sems[ki,meth] = np.std(K_sel)/np.sqrt(K_sel.size)
            plt.bar(K,1,width=0.9,color=(0,0,0,0),edgecolor='k',ls=lss[ki])
            plt.text(0,1.1+0.05*ki,'True K='+str(K)+', Simultaneous='+str(np.round(K_means[ki,0],2))+'$\\pm$'+str(np.round(K_sems[ki,0],2))+', Sequential='+str(np.round(K_means[ki,1],2))+'$\\pm$'+str(np.round(K_sems[ki,1],2)))
        plt.legend(['Simultaneous','Sequential','True']+['']*3*(len(Ks)-1),ncol=len(Ks),markerfirst=False,title=' '*24+(' '*9).join(['K='+str(K) for K in Ks]), loc = 'upper right')

            
        plt.xlabel(ylabel+'-Selected $K$',fontsize=14)
        plt.ylabel('Frequency',fontsize=14)

        plt.savefig(savepath+name+'_sel_hist_reg_share_'+share+'.png',bbox_inches='tight')
        plt.close()         














###Data figures
Kfits = np.arange(1,21)
trials = 20
seeds = 1

N = 634                
n_subsets = 4
seed = 0
np.random.seed(seed)
order = np.random.permutation(N)
all_vns = [order[(sub*N)//n_subsets:((sub+1)*N)//n_subsets] for sub in range(n_subsets)]
run = False

for share in ['W','all']:
    W_from_muk_inds = np.arange(d[1]) if share=='W' else np.arange(d[0],d[0]+d[1])

    ### Figures!!!
    for subsub in [True,False] if run else []:
        l2_is = np.array([0]) if share=='all' or subsub else np.arange(-2,2)
        nl2s = l2_is.size


        mets = {'seq':[],'simul':[]}
        subs = [(s1,s2) for s1 in range(n_subsets) for s2 in range(s1)] if subsub else list(range(n_subsets))
        Kopt_mets = {'seq':[None for i in subs],'simul':[None for i in subs]}
        alltr_mets = {'seq':{'BIC':np.nan*np.ones((Kfits.size,len(subs),trials*nl2s)),'TL':np.nan*np.ones((Kfits.size,len(subs),trials*nl2s))},'simul':{'BIC':np.nan*np.ones((Kfits.size,len(subs),trials*nl2s)),'TL':np.nan*np.ones((Kfits.size,len(subs),trials*nl2s))}}
        l2s = np.zeros((Kfits.size,len(subs)))
        for meth in mets.keys():
            max_BICs = [-np.inf for i in subs]
            for Ki,K in enumerate(Kfits):
                ct = []
                ctx = []
                mu_k = []
                C_k = []
                pi_k = []
                Qs = []
                TLs = []
                BICs = []
                gmm_TLs = []
                tb_losses_mean = np.nan*np.ones((N,seeds,len(subs)))
                tb_corrs_mean = np.nan*np.ones((N,seeds,len(subs)))
                train_tb_losses_mean = np.nan*np.ones((N,seeds,len(subs)))
                train_tb_corrs_mean = np.nan*np.ones((N,seeds,len(subs)))

                for subi, subset in enumerate(subs):

                    if meth=='simul': #Simultaneous
                        min_loss = np.inf
                        min_D = None
                        for trial, l2_i in [(trial,l2_i) for trial in range(trials) for l2_i in l2_is]:
                            try:
                                fname = 'ivscc_n1t2v_simulreg_share_'+share+str(trial)+'_K='+str(K)+'l2i='+str(l2_i)+'_sub='+str(subset)+'_seed='+str(seed)+'_train_reps=all'
                                D = np.load(fname+'.npz',allow_pickle=True)
                                if D['loss']<min_loss:
                                    a = D['Q_val'].size
                                    min_loss=D['loss']
                                    min_D = D
                                    l2s[Ki,subi] = D['l2']
                                alltr_mets[meth]['BIC'][Ki,subi,l2_i*trials+trial] = D['BIC']
                                alltr_mets[meth]['TL'][Ki,subi,l2_i*trials+trial] = D['val_loss']
                            except Exception as e:
                                print(e,subset,trial)
                        D = min_D
                    else: #Sequential
                        fname = 'ivscc_n1t2v_seqreg_share_'+share+'_K='+str(K)+'_sub='+str(subset)+'_seed='+str(seed)+'_train_reps=all'
                        D = np.load(fname+'.npz',allow_pickle=True)
                    neurons = D['neuron_inds']
                    sub = (subset,) if type(subset) is not tuple else subset
                    val_neurons = np.hstack([order[(s*N)//n_subsets:((s+1)*N)//n_subsets] for s in sub])
                    Q = np.zeros((N,K))
                    Q[neurons,:] = D['Q']
                    Q[val_neurons,:] = D['Q_val']
                    Qs.append(Q)
                    ct.append(np.argmax(Q,axis=1))
                    ctx.append([ct[-1][vn] for vn in all_vns])
                    mu_k.append(D['mu_k'])
                    C_k.append(D['C_k'])
                    pi_k.append(D['wts'])
                    TLs.append(D['val_losses'])
                    if meth=='simul': #Simultaneous
                        tb_losses_mean[val_neurons,seed,subi] = D['val_neurons_val_nnlls']
                        tb_corrs_mean[val_neurons,seed,subi] = np.where(np.isnan(D['val_neurons_val_corrs']),0,D['val_neurons_val_corrs'])
                        train_tb_losses_mean[neurons,seed,subi] = D['val_nnlls']
                        train_tb_corrs_mean[neurons,seed,subi] = np.where(np.isnan(D['val_corrs']),0,D['val_corrs'])
                        bic_name = 'BIC'
                        gmm_TLs.append(np.nan)
                    else: #Sequential
                        tb_losses_mean[val_neurons,seed,subi] = np.squeeze(D['val_nnlls'])[val_neurons]
                        tc = np.squeeze(D['val_corrs'][val_neurons])
                        tb_corrs_mean[val_neurons,seed,subi] = np.where(np.isnan(tc),0,tc)
                        train_tb_losses_mean[neurons,seed,subi] = np.squeeze(D['val_nnlls'])[neurons]
                        tc = np.squeeze(D['val_corrs'][neurons])
                        train_tb_corrs_mean[neurons,seed,subi] = np.where(np.isnan(tc),0,tc)
                        bic_name = 'gmm_BIC'
                        gmm_TLs.append(D['gmm_TLs'])
                    if D[bic_name]>max_BICs[subi]:
                        max_BICs[subi] = D[bic_name]
                        Kopt_mets[meth][subi] = [K, tb_losses_mean[:,seed,subi], tb_corrs_mean[:,seed,subi]]
                    BICs.append(D[bic_name])

                
                variances = [np.median([np.sqrt(np.diag(C_k[t][k])[W_from_muk_inds])*np.exp(mu_k[t][k][W_from_muk_inds]) for k in range(K)]) for t in range(len(C_k))]
                arss = []

                for t1 in range(len(mu_k)):
                    for t2 in range(t1):
                        arss.append(adjusted_rand_score(ct[t1],ct[t2]))

                mets[meth].append([gmm_TLs, arss, variances, train_tb_losses_mean, train_tb_corrs_mean, tb_losses_mean, tb_corrs_mean, TLs, BICs])
        np.savez('../summary_files/ivscc_interp_data'+'_subsub'*subsub+'_share'+share+'.npz',mets=mets,Kopt_mets=Kopt_mets, alltr_mets = alltr_mets, l2s=l2s)

    names = ['gmm_TLs', 'Between_ARS', 'Within_Sigma',
            'GLM_train_tb_losses','GLM_train_tb_corrs', 'GLM_tb_losses','GLM_tb_corrs', 'full_tn_marg_loss', 'BIC']
    labels = ['GMM Losses on held out neurons', 'ARS between runs', 'Median $e^{\mu_k}\sqrt{\Sigma_k}$',
            'ANLL on held out stimulus','$EV_{ratio}$ on held out stimulus', 'ANLL on held out neurons and stimulus','$EV_{ratio}$ of held out neurons and stimulus', 'Loss on held out neurons', 'BIC']
    # alt_less = [None,False,False,True,False,True,False,True,None]


#    savepath = '../figs/subsub_'
#    D = np.load('ivscc_interp_data_subsub_share'+share+'.npz',allow_pickle=True)
    D = np.load('../summary_files/ivscc_interp_data_share'+share+'.npz',allow_pickle=True)
    seq_mets = D['mets'][()]['seq']
    simul_mets = D['mets'][()]['simul']
    seq_mets_Kopt = D['Kopt_mets'][()]['seq']
    simul_mets_Kopt = D['Kopt_mets'][()]['simul']
    seq_mets_alltr = D['alltr_mets'][()]['seq']
    simul_mets_alltr = D['alltr_mets'][()]['simul']
    l2s = D['l2s']

    for i in range(len(names)):
        fig,ax = plt.subplots()

        if not names[i].startswith('GLM_'):
            continue
        simul_data = np.stack([np.array(simul_mets[Ki][i])-np.array(simul_mets[0][i]) for Ki in range(Kfits.size)],axis=0)
        print(names[i],simul_data.shape)
        for f in range(simul_data.shape[-1]):
            h1 = plt.errorbar(Kfits-0.15+0.1*f,np.nanmean(simul_data,axis=(1,2))[:,f],yerr=np.nanstd(simul_data,axis=(1,2))[:,f]/np.sqrt(np.sum(np.isfinite(simul_data[:,:,:,f]),axis=(1,2))))

        ax.set_xticks(Kfits)
        ax.set_xlabel('K',fontsize=14)
        ax.set_ylabel(labels[i],fontsize=14)
        fig.tight_layout()
        plt.savefig(savepath+'IVSCC_interp_'+names[i]+'_share'+share)
        plt.close()


    D = dict(np.load('../ivscc_data_n12.npz',allow_pickle=True))

    ### Color for neurons in generalization plots
    c = np.log10(D['test_spk_counts']) #order*n_subsets//N

    i = 1
    fig,ax=plt.subplots()
    plt.scatter(np.nanmean(np.vstack([seq_mets_Kopt[s][i] for s in range(n_subsets)]),axis=0),
        np.nanmean(np.vstack([simul_mets_Kopt[s][i] for s in range(n_subsets)]),axis=0),c=c,cmap=cmap)
    clb = plt.colorbar(label='$\\log_{10}$(\x23spikes)')
    lims=[0,0.5]
    plt.plot(lims,lims,'k')
    plt.axis('square')
    plt.xlim(lims)
    plt.ylim(lims)
    plt.text(lims[1]*3.5/5,lims[1]*0.5/5,'Simultaneous \n is better')
    plt.text(lims[1]/5*0.5,lims[1]/5*4,'Sequential \n is better')
    plt.xlabel('Independent GLM\'s test ANLL',fontsize=14)
    plt.ylabel('Simul GLM\'s test ANLL',fontsize=14)
    plt.tight_layout()
    plt.savefig(savepath+'ivscc_test_n_testbin_share='+share+'_nnll')
    plt.close()

    i = 2
    fig,ax=plt.subplots()
    plt.scatter(np.nanmean(np.vstack([seq_mets_Kopt[s][i] for s in range(n_subsets)]),axis=0),np.nanmean(np.vstack([simul_mets_Kopt[s][i] for s in range(n_subsets)]),axis=0),c=c,cmap=cmap)
    clb = plt.colorbar(label='$\\log_{10}$(\x23spikes)')
    lims=[-0.05,1.0]
    plt.plot(lims,lims,'k')
    plt.axis('square')
    plt.xlim(lims)
    plt.ylim(lims)
    plt.text(lims[1]*3.5/5,lims[1]*0.5/5,'Sequential \n is better')
    plt.text(lims[1]/5*0.5,lims[1]/5*4,'Simultaneous \n is better')
    plt.xlabel('Independent GLM\'s test $EV_{ratio}$',fontsize=14)
    plt.ylabel('Simul GLM\'s test $EV_{ratio}$',fontsize=14)
    plt.tight_layout()
    plt.savefig(savepath+'ivscc_test_n_testbin_share='+share+'_corr')
    plt.close()


    fig,ax=plt.subplots()
    i=1
    x1 = np.nanmean(np.vstack([seq_mets_Kopt[s][i] for s in range(n_subsets)]),axis=0)
    x2 = np.nanmean(np.vstack([simul_mets_Kopt[s][i] for s in range(n_subsets)]),axis=0)
    i=2
    y1 = np.nanmean(np.vstack([seq_mets_Kopt[s][i] for s in range(n_subsets)]),axis=0)
    y2 = np.nanmean(np.vstack([simul_mets_Kopt[s][i] for s in range(n_subsets)]),axis=0)

    x = (x2-x1)/(x1+x2)
    y = (y2-y1)/(y1+y2)

    plt.scatter(x,y,c=c,cmap=cmap)
    clb = plt.colorbar(label='$\\log_{10}$(\x23spikes)')
    lims = [-1.2,1.2]
    ld = lims[1]-lims[0]
    plt.xlim(lims)
    plt.ylim(lims)
    plt.axvline(0,c='k')
    plt.axhline(0,c='k')
    plt.text(lims[0]-ld/5,lims[0],'Sequential \n is better',rotation=90)
    plt.text(lims[0]-ld/5,lims[0]+ld*3.5/5,'Simultaneous \n is better',rotation=90)
    plt.text(lims[0]+ld*3.5/5,lims[0]-ld/5,'Sequential \n is better')
    plt.text(lims[0],lims[0]-ld/5,'Simultaneous \n is better')
    plt.title('Relative difference between \n Simultaneous and Sequential methods',fontsize=14)
    plt.xlabel('GLM\'s test \n ANLL',fontsize=14)
    plt.ylabel('GLM\'s test \n $EV_{ratio}$',fontsize=14)
    plt.tight_layout()
    plt.savefig(savepath+'ivscc_test_n_testbin_relcomp_share='+share)
    plt.close()


    methnames = {'seq':'Sequential','simul':'Simultaneous'}

    for meth in methnames.keys():
        fig,ax=plt.subplots()
        if meth=='simul':
            y2 = -np.array([[np.nanmean(m) for m in simul_mets[Ki][-2]] for Ki in range(Kfits.size)])
        else:
            y2 = np.array([[np.nanmean(m) for m in seq_mets[Ki][0]] for Ki in range(Kfits.size)])
        y2 -= y2[0:1,:] #subtract off losses for K=1
        h1 = ax.plot(Kfits,y2)[0]
        ### select K using 1se rule
        CV = np.mean(y2,axis=1)
        Ksel = Kfits[np.argmin(Kfits-1000 * ( CV>np.max(CV)-np.std(y2,axis=1)/np.sqrt(y2.shape[1]) ) )]
        h2 = ax.plot(Ksel,np.max(CV),'r*',markersize=10)[0]
        ax.set_ylabel('Loglikelihood on held out neurons',fontsize=14)
        ax.set_xticks(Kfits)
        ax.set_xlabel('K',fontsize=14)
        plt.title(methnames[meth]+' Method',fontsize=14)
        plt.legend([h1,h2],['LL of each fold', 'K selected by 1SE rule'],loc='lower right')
        plt.tight_layout()
        plt.savefig(savepath+'ivscc_'+meth+'_VLs_share='+share)
        plt.close()


    ### Figures !!!
    #Simultaneous
    BICs = np.nan*np.ones((Kfits.size,trials))
    for Ki, K in enumerate(Kfits):
        for trial in range(trials):
            D = np.load('ivscc_n1t2v_simulreg_share_'+share+str(trial)+'_K='+str(K)+'_sub=allN_train_reps=all.npz',allow_pickle=True)
            if np.any(D['bad']):
                print('bad',K,trial)
            BICs[Ki,trial] = D['BIC']
    BICs-=np.max(BICs)
    K_max=np.argmax(BICs) #replace with specific K to generate plots for other K
    trial_max = np.argmax(BICs[Kfits==K_max,:])
    

    fig,ax = plt.subplots()
    ax.plot(Kfits,BICs,'o',c=colors[0])
    ax.plot(Kfits,np.max(BICs,axis=1),c=colors[0])
    ax.set_ylim([-1600,100])
    ax.set_xticks(Kfits)
    ax.set_xlabel('K',fontsize=14)
    ax.set_ylabel('BIC',fontsize=14,c=colors[0])
    plt.title('Simultaneous Method',fontsize=14)
    plt.tight_layout()
    plt.savefig(savepath+'ivscc_simul_BIC_all_share='+share)
    plt.close()

    #Sequential
    BICs = np.zeros_like(Kfits)
    for Ki, K in enumerate(Kfits):
        fname = 'ivscc_n1t2v_seqreg_share_'+share+'_K='+str(K)+'_sub=allN_train_reps=all'
        D = np.load(fname+'.npz',allow_pickle=True)    
        BICs[Ki] = D['gmm_BIC']
    BICs-=np.max(BICs)
    VLs = np.array([[np.nanmean(m) for m in seq_mets[Ki][0]] for Ki in range(Kfits.size)])
    CVs = np.mean(VLs,axis=(1))
    sems = np.std(VLs,axis=(1))/np.sqrt(seeds*n_subsets)


    fig,ax = plt.subplots()
    ax.plot(Kfits,BICs,c=colors[0])
    ax.set_xticks(Kfits)
    plt.xlabel('$K$',fontsize=14)
    ax.set_ylabel('BIC on train neurons',fontsize=14,c=colors[0])
    plt.title('Sequential Method',fontsize=14)
    plt.tight_layout()
    plt.savefig(savepath+'ivscc_seq_BIC_all_share'+share+'.png',bbox_inches='tight')
    plt.close()


    ###Metadata figures
    pixel_size_in = 0.35
    cat_cols = ['dendrite_type','structure_hemisphere','structure_layer_name','transgenic_line']
    used_cells = pd.read_csv('../n12_cells.csv')
    which = np.arange(N)
    for meth in ['simul','seq']:
        if meth=='simul':
            D = np.load('ivscc_n1t2v_simulreg_share_'+share+str(trial_max)+'_K='+str(K_max)+'_sub=allN_train_reps=all.npz',allow_pickle=True)
        else:
            D = np.load('ivscc_n1t2v_seqreg_share_'+share+'_K='+str(K_max)+'_sub=allN_train_reps=all'+'.npz',allow_pickle=True)

        P = np.flip(np.argsort(np.sum(D['mu_k'][:,W_from_muk_inds],axis=1))) #ordering permutations
        min_N = 20-0.1 #only clusters with pi_k*N > min_N are shown

        hs = []
        labels = []
        used_ks = []
        fig,ax = plt.subplots()
        for k in P:
            if D['wts'][k]*N>min_N:
                used_ks.append(k)
                sds = np.sqrt(np.diag(D['C_k'][k])[W_from_muk_inds])
                hs.append(plt.fill_between((W_from_muk_inds-np.min(W_from_muk_inds))*dt,D['mu_k'][k,W_from_muk_inds]-sds,D['mu_k'][k,W_from_muk_inds]+sds,alpha=0.4))
                labels.append(str(len(used_ks)-1)+', '+str(np.round(D['wts'][k],2)))
                plt.plot((W_from_muk_inds-np.min(W_from_muk_inds))*dt,D['mu_k'][k,W_from_muk_inds].T)
        plt.xlabel('Time (ms)',fontsize=14)
        plt.ylabel('Self-interaction filters',fontsize=14)
        ax.set_ylim([-4,2])
        leg = plt.legend(hs,labels,loc=(1,0))
        leg.set_title('Cluster IDs and size')
        plt.title('Simultaneous Method',fontsize=14)
        plt.tight_layout()
        plt.savefig(savepath+'IVSCC_Ws_'+meth+'reg_share'+share+'_K='+str(K_max)+'.png',bbox_inches='tight')
        plt.close()

        c_types=np.argmax(D['Q'],axis=1) #memberships

        plt.set_cmap('PiYG')
        for i,cat in enumerate(cat_cols):
            fig,ax = plt.subplots()
            opts = np.unique(used_cells[cat].values)
            opts_dict = dict(zip(opts,np.arange(opts.size)))
            cm = confusion_matrix(c_types,np.vectorize(opts_dict.get)(used_cells[cat].values[which]),labels=np.arange(max(K_max,opts.size)))
            cm = cm[used_ks,:opts.size]
            which_cols = np.sum(cm,axis=0)>8
            p_clust = np.sum(cm[:,which_cols],axis=1,keepdims=True)/N
            p_attr = cm[:,which_cols]*1.0/np.sum(cm[:,which_cols],axis=0,keepdims=True)
            img = (p_attr-p_clust)/np.sqrt(p_attr*(1-p_attr)/np.sum(cm[:,which_cols],axis=0,keepdims=True)+p_clust*(1-p_clust)/N)
            im = ax.imshow(img,vmin=-5,vmax=5)
            ax.set_xticks(np.arange(np.sum(which_cols)))
            ax.set_yticks(np.arange(len(used_ks)))
            ax.set_ylim([len(used_ks)-0.5,-0.5])
            title = cat.replace('structure_','').replace('_',' ')
            ars = adjusted_rand_score(c_types,np.vectorize(opts_dict.get)(used_cells[cat].values[which]))
            ax.set_title(title.capitalize()+'\n ARS = '+str(np.round(ars,2)),fontsize=14)

            ax.set_xticklabels(np.core.defchararray.partition(opts[which_cols].astype('str'),'-')[:,0],rotation=-90*(i!=2))
            ax.set_ylabel('Cluster IDs',fontsize=14)
            set_size(pixel_size_in*img.shape[1],pixel_size_in*img.shape[0],ax=ax)
            if i==len(cat_cols)-1:
                cbar = add_colorbar(im,orientation='vertical')
                cbar.set_label('Z-scored fraction of cells\n with attribute in cluster',fontsize=12,labelpad=10)
            plt.savefig(savepath+title+meth+'_share'+share+'_K='+str(K_max)+'.png',bbox_inches='tight')
            plt.close()            

for share in ['W']: #we haven't done these analyses for case B yet
    ### Figures !!!
    K = 12
    l2_i = 0
    seed = 0
    train_reps = [1, 2, 3] #, 'all']
    sub_levels = [list(combinations(np.flip(np.arange(n_subsets)),3)), list(combinations(np.flip(np.arange(n_subsets)),2)), list(range(n_subsets)), ['allN']]
    sl_names = ['$\\frac{1}{4}$', '$\\frac{1}{2}$', '$\\frac{3}{4}$', 'all']
    tn_nnlls = np.zeros((N,len(train_reps),len(sub_levels),2))
    tn_corrs = np.zeros((N,len(train_reps),len(sub_levels),2))
    vn_nnlls = np.zeros((N,len(train_reps),len(sub_levels)-1,2))
    vn_corrs = np.zeros((N,len(train_reps),len(sub_levels)-1,2))
    cts = np.nan*np.ones((N,len(train_reps),len(sub_levels)-1,6,2))
    arss = np.nan*np.ones((15,len(train_reps),len(sub_levels)-1,2))
    hits = np.zeros((N,len(train_reps),len(sub_levels),2))
    vn_hits = np.zeros((N,len(train_reps),len(sub_levels)-1,2))

    for sli, subs in enumerate(sub_levels):
        for tri, train_rep in enumerate(train_reps):
            for mi, meth in enumerate(['simul','seq']):
                for subi,sub in enumerate(subs):
                    min_loss = np.inf
                    D = None
                    tr_D = None
                    for trial in range(trials) if meth=='simul' else ['']:
                        fname = 'ivscc_n1t2v_'+meth+'reg_share_'+share+str(trial)+'_K='+str(12)+'l2i=0'*((meth=='simul') and (sub!='allN'))+'_sub='+str(sub)+('_seed='+str(seed))*(sub!='allN')+'_train_reps='+str(train_rep)
                        tr_D = np.load(fname+'.npz',allow_pickle = True)
                        if meth=='seq' or tr_D['loss']<min_loss:# and (trial!=trial_max or tri!=2 or sli!=2):
                            min_loss = tr_D['loss'] if meth=='simul' else None
                            D = tr_D
                    neurons = D['neuron_inds']
                    tn_nnlls[neurons,tri,sli,mi] += np.squeeze(D['val_nnlls'][neurons] if meth=='seq' else D['val_nnlls'])
                    tn_corrs[neurons,tri,sli,mi] += np.squeeze(D['val_corrs'][neurons] if meth=='seq' else D['val_corrs'])
                    hits[neurons,tri,sli,mi] += 1
                    if sub!='allN':
                        sub = (sub,) if type(sub) is not tuple else sub
                        val_neurons = np.squeeze(np.hstack([order[(s*N)//n_subsets:((s+1)*N)//n_subsets] for s in sub]))
                        Q = np.zeros((N,K))
                        Q[neurons] = D['Q']
                        Q[val_neurons] = D['Q_val']
                        cts[:,tri,sli,subi,mi] = np.argmax(Q,axis=1)
                        vn_nnlls[val_neurons,tri,sli,mi] += np.squeeze(D['val_nnlls'][val_neurons] if meth=='seq' else D['val_neurons_val_nnlls'])
                        vn_corrs[val_neurons,tri,sli,mi] += np.squeeze(D['val_corrs'][val_neurons] if meth=='seq' else D['val_neurons_val_corrs'])
                        vn_hits[val_neurons,tri,sli,mi] += 1
                if len(subs)>1:
                    ARS = []
                    for subi in range(len(subs)):
                        for subi2 in range(subi):
                            ARS.append(adjusted_rand_score(cts[:,tri,sli,subi,mi],cts[:,tri,sli,subi2,mi]))
                    arss[:len(ARS),tri,sli,mi] = ARS
        np.savez('summary_files/ivscc_scaling_data',tn_nnlls=tn_nnlls/hits,tn_corrs=tn_corrs/hits, hits=hits, arss=arss,cts=cts, vn_nnlls=vn_nnlls/vn_hits, vn_corrs=vn_corrs/vn_hits, vn_hits=vn_hits)

    D = np.load('summary_files/ivscc_scaling_data.npz')
    tn_nnlls = D['tn_nnlls']
    tn_corrs = D['tn_corrs']
    vn_nnlls = D['vn_nnlls']
    vn_corrs = D['vn_corrs']
    arss = D['arss']
    hits = D['hits']
    labels = ['Train ANLL', 'Train $EV_{ratio}$', 'ARS', 'Test ANLL', 'Test $EV_{ratio}$']
    slabels = ['train_ANLL', 'train_EV', 'ARS', 'test_ANLL', 'test_EV']
    thresh = 1e-3 #significance threshold for uncorrected p-values
    for di,data in enumerate([tn_nnlls,tn_corrs,arss,vn_nnlls,vn_corrs]):
        fig,ax = plt.subplots()
        rel_data = (data[:,:,:,0]-data[:,:,:,1])/(data[:,:,:,0]+data[:,:,:,1])
        plt.imshow(np.nanmedian(rel_data,axis=0),cmap=cmap,origin='upper')
        r_diffs = []
        c_diffs = []
        for r in range(data.shape[1]):
            for c in range(data.shape[2]):
               p_val = stats.wilcoxon(rel_data[np.isfinite(rel_data[:,r,c]),r,c],alternative='less' if slabels[di].endswith('ANLL') else 'greater')[1] 
               if p_val<thresh:
                   plt.plot([c],[r],'w*')
               if r < data.shape[1]-1:
                   diff = rel_data[:,r,c]-rel_data[:,r+1,c]
                   p_val = stats.wilcoxon(diff)[1]
                   r_diffs.append([rel_data[:,r,c],rel_data[:,r+1,c]])
                   if p_val<thresh:
                       plt.plot([c,c],[r+0.4,r+0.6],'w')
               if c < data.shape[2]-1:
                   if di==2:
                       p_val = stats.ttest_ind(rel_data[:,r,c+1],rel_data[:,r,c],nan_policy='omit')[1]
                   else:
                       diff = rel_data[:,r,c+1]-rel_data[:,r,c]
                       p_val = stats.wilcoxon(diff)[1]
                   c_diffs.append([rel_data[:,r,c+1],rel_data[:,r,c]])
                   if p_val<thresh:
                       plt.plot([c+0.4,c+0.6],[r,r],'w')
        c_diffs = np.array(c_diffs)
        r_diffs = np.array(r_diffs)
        def test(x,y,paired,**kwargs):
            return stats.ttest_ind(x[np.isfinite(x)],y[np.isfinite(y)],**kwargs) if not paired else stats.wilcoxon((x-y)[np.isfinite(x-y)],**kwargs)
        p_rl = test(np.ravel(r_diffs[:,0,:]),np.ravel(r_diffs[:,1,:]),True,alternative='less')[1]
        p_cl = test(np.ravel(c_diffs[:,0,:]),np.ravel(c_diffs[:,1,:]),di!=2,alternative='less')[1]
        print(slabels[di],'less',p_rl,p_cl)
        p_rg = test(np.ravel(r_diffs[:,0,:]),np.ravel(r_diffs[:,1,:]),True,alternative='greater')[1]
        p_cg = test(np.ravel(c_diffs[:,0,:]),np.ravel(c_diffs[:,1,:]),di!=2,alternative='greater')[1]
        print(slabels[di],'greater',p_rg,p_cg)
        if di<2:
            plt.xticks(range(len(sl_names)),sl_names)
        else:
            plt.xticks(range(len(sl_names)-1),sl_names[:-1])

        plt.yticks(range(len(train_reps)),train_reps)
        plt.colorbar()
        pr_str = str(np.format_float_scientific(min(p_cl,p_cg),precision=0))
        sig_str = 'n.s.' if min(p_cl,p_cg)>0.1 else 'more are better, p='+pr_str if slabels[di].endswith('ANLL') == (p_cl<0.1) else 'fewer are better, p='+pr_str
        plt.xlabel('Fraction of neurons used in training \n' +sig_str,fontsize=14)
        pr_str = str(np.format_float_scientific(min(p_rl,p_rg),precision=0))
        sig_str = 'n.s.' if min(p_rl,p_rg)>0.1 else 'fewer are better, p='+pr_str if slabels[di].endswith('ANLL') == (p_rl<0.1) else 'more are better, p='+pr_str
        plt.ylabel('Number of stimulus presentations\n used in training\n '+sig_str,fontsize=14)
        plt.title('Relative difference in '+labels[di],fontsize=14)
        plt.tight_layout()
        plt.savefig(savepath+'ivscc_'+slabels[di]+'_scalingNT_share'+share)


