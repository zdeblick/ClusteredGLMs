#!/home/daniel.zdeblick/anaconda3/bin/python3 #used if this script is run by bash command


import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import pearsonr
import statsmodels.api as sm
from sklearn.mixture import GaussianMixture
import time
from sklearn.metrics import confusion_matrix, adjusted_rand_score
import os
import cProfile  
from helpers import *
from models import *

array_id_str = 'SLURM_ARRAY_TASK_ID' #slurm
#array_id_str = 'PBS_ARRAYID' #pbs/moab/torque
    
def main_data(seq_method=False,val_bins=False,all_n=True,subsub=False,share='W'):
    os.chdir('files')
    D = np.load('../ivscc_data_n12.npz',allow_pickle=True)

    stim = D['binned_stim']
    binned_spikes = D['binned_spikes']
    test_stim = D['test_binned_stim']
    test_spikes = D['test_binned_spikes']
    bin_len = D['bin_len']
    d = [10,20]
    N = binned_spikes.shape[0]
    id = os.getenv(array_id_str)
    id = 0 if id is None else int(id)
    seeds = 1

    n_subsets = 4
    shared_stim = False
    downsample = 5
    
    train_nm1 = True
    trials = 20
    Ks = np.arange(1,21)
    simul_tfolds = 4

    train_repss = [1, 2, 3]
    subsets = np.hstack([list(combinations(np.flip(np.arange(n_subsets)),3)), list(combinations(np.flip(np.arange(n_subsets)),2)), list(range(n_subsets))])
    train_reps = 'all'      

    if seq_method:
        ###All Neurons
        l2s_self = np.logspace(-7,-1,19)
        l2s_stim = np.logspace(-7,-1,19)
        which_l2s = np.arange(0,19)
        if all_n and val_bins:
            (l2_stim_i,l2_self_i) = np.unravel_index(id,(which_l2s.size,which_l2s.size))
            l2_stim_i = which_l2s[l2_stim_i]
            l2_self_i = which_l2s[l2_self_i]
            fname = 'ivscc_n1t2v_regind_valtime_l2stimi='+str(l2_stim_i)+'_l2selfi='+str(l2_self_i)
            subset = np.inf
        elif all_n:
            Ki = id
            K = Ks[Ki]
            subset = np.inf
            fname = 'ivscc_n1t2v_seqreg_share_'+share+'_K='+str(K)+'_sub=allN_train_reps='+str(train_reps)
        ###Subset
        else:
            if subsub:
                (subset,repi,seed) = np.unravel_index(id,(len(subsets),len(train_repss),seeds))
                Ki = 11 if share=='W' else 0
                subset = subsets[subset]
                train_reps = train_repss[repi]
            else:
                (subset,seed,Ki) = np.unravel_index(id,(n_subsets,seeds,Ks.size))
            np.random.seed(seed)
            K = Ks[Ki]
            fname = 'ivscc_n1t2v_seqreg_share_'+share+'_K='+str(K)+'_sub='+str(subset)+'_seed='+str(seed)+'_train_reps='+str(train_reps)
    else:
        l2s = np.array([0.0]) if share=='all' else np.hstack((np.logspace(-7,-1,19),np.logspace(-9,-7,7)[:-1]))

        if all_n:
            (trial,Ki) = np.unravel_index(id,(200,Ks.size))
            K = Ks[Ki]
            min_val_nnll = np.inf
            l2_i = 0 #We pre-selected this for case A based on the results of model selection with subsets of neurons from step 5a
            l2 = l2s[l2_i]
            subset = np.inf
            fname = 'ivscc_n1t2v_simulreg_share_'+share+str(trial)+'_K='+str(K)+'_sub=allN_train_reps='+str(train_reps)

        ###Subset of neurons
        else:
            if subsub:
                which_l2s = np.arange(0,1) #We pre-selected this for case A based on the results of model selection with subsets of neurons from step 5a
                (trial,subset,repi,seed,l2_i) = np.unravel_index(id,(200,len(subsets),len(train_repss),seeds,which_l2s.size))
                Ki = 11 if share=='W' else 0
                subset = subsets[subset]
                train_reps = train_repss[repi]
            else:
                which_l2s = np.arange(0,1) if share=='all' else np.arange(-3,3)
                (trial,subset,seed,Ki,l2_i) = np.unravel_index(id,(200,n_subsets,seeds,Ks.size,which_l2s.size))
            l2_i = 0 if subsub else which_l2s[l2_i]
            l2 = l2s[l2_i]
            K = Ks[Ki]
            np.random.seed(seed)
            fname = 'ivscc_n1t2v_simulreg_share_'+share+str(trial)+'_K='+str(K)+'l2i='+str(l2_i)+'_sub='+str(subset)+'_seed='+str(seed)+'_train_reps='+str(train_reps)

    order = np.random.permutation(N)
    if all_n:
        neurons = np.arange(N)
        val_neurons = np.array([])
    else:
        subset = (subset,) if type(subset) is not tuple else subset
        neurons = np.hstack([order[(s*N)//n_subsets:((s+1)*N)//n_subsets] for s in subset])
        if train_nm1:
            val_neurons = np.array(neurons)
            neurons = np.setdiff1d(np.arange(N),neurons)
        else:
            val_neurons = np.setdiff1d(np.arange(N),neurons)

    if val_bins and not seq_method:
        #randomly choose val_trials for each neuron
        val_trials = [np.random.permutation(len(s))[np.mod(fold,len(s))] for s in stim]
    else:
        val_trials = [-1]*N
    if train_reps=='all':
        tr_trials = [np.arange(len(s)) for s in stim]
    else:
        if val_bins:
            raise(NotImplementedError) # if you ever want to do this, make sure that no trial is both val and train
        tr_trials = [np.random.permutation(len(s))[:train_reps] for s in stim]

    TL = None
    Tc = None
    if seq_method and not val_bins:
        if train_reps=='all':
            VLs = np.inf*np.ones((l2s_stim.size,l2s_self.size))
            TLs = np.inf*np.ones((val_neurons.size,l2s_stim.size,l2s_self.size))
            Tcs = np.inf*np.ones((val_neurons.size,l2s_stim.size,l2s_self.size))
            for l2_stim_i in range(l2s_stim.size):
                for l2_self_i in range(l2s_self.size):
                    tfname = 'ivscc_n1t2v_regind_valtime_l2stimi='+str(l2_stim_i)+'_l2selfi='+str(l2_self_i)
                    try:
                        D = np.load(tfname+'.npz',allow_pickle=True)
                        VLs[l2_stim_i,l2_self_i] = np.median([np.mean(D['val_nnlls'][n]) for n in neurons])
                        TLs[:,l2_stim_i,l2_self_i] = [np.mean(D['val_nnlls'][n]) for n in val_neurons]
                        Tcs[:,l2_stim_i,l2_self_i] =  [np.mean(D['val_corrs'][n]) for n in val_neurons]
                    except:
                        pass
            l2_stim_i,l2_self_i = np.unravel_index(np.argmin(VLs),(l2s_stim.size,l2s_self.size))
            print(l2_stim_i,l2_self_i)
            Tc = Tcs[:,l2_stim_i,l2_self_i]
            TL = TLs[:,l2_stim_i,l2_self_i]
        else:
            l2_stim_i,l2_self_i = 5,9

    seed = int(time.time()*1000000)%1000000
    np.random.seed(seed)

    n_sims = 1 if val_bins else 3000
    if seq_method:
        train_nnlls = [[] for n in range(N)]
        train_corrs = [[] for n in range(N)]
        val_nnlls = [[] for n in range(N)]
        val_corrs = [[] for n in range(N)]
        Fs = [[] for n in range(N)]
        Ws = [[] for n in range(N)]
        bs = [[] for n in range(N)]
        is_file = False #os.path.isfile(fname+'.npz')
        if not is_file:
            for n in range(N):
                n_splits = len(binned_spikes[n]) 
                guess=None

                for split in range(n_splits) if val_bins else [-1]:
                    if shared_stim:
                        tr_stim = [stim[spl][t] for spl in range(n_splits) if spl!=split and spl in tr_trials[n]]
                        val_stim = [stim[split]] if val_bins else test_stim
                    else:
                        tr_stim = [stim[n][spl] for spl in range(n_splits) if spl!=split and spl in tr_trials[n]]
                        val_stim = [stim[n][split]] if val_bins else test_stim[n]
                    tr_spks = [binned_spikes[n][spl] for spl in range(n_splits) if spl!=split and spl in tr_trials[n]]
                    val_spks = binned_spikes[n][split] if val_bins else test_spikes[n]
                    F, W, b, train_nnll, train_corr, val_nnll, val_corr = fit_GLM(tr_stim,tr_spks,d,val_stim=val_spks,val_spks=val_spks,l2_stim=l2s_stim[l2_stim_i],l2_self=l2s_self[l2_self_i],downsample=downsample,guess=guess,n_sims=n_sims)
                    guess = np.hstack((np.squeeze(F),W,np.expand_dims(b,0)))
                    train_nnlls[n].append(train_nnll)
                    train_corrs[n].append(train_corr)
                    Fs[n].append(F)
                    Ws[n].append(W)
                    bs[n].append(b)
                    val_nnlls[n].append(val_nnll)
                    val_corrs[n].append(val_corr)
            np.savez(fname,train_nnlls=train_nnlls,train_corrs=train_corrs,val_nnlls=val_nnlls,val_corrs=val_corrs,Fs=Fs,Ws=Ws,bs=bs,TL=TL,Tc=Tc,neuron_inds=neurons)
        squelch = lambda x: np.vstack([x_i[0].reshape(1,-1) for x_i in x])

        D_ind = dict(np.load(fname+'.npz',allow_pickle=True))
        ind_p = np.hstack([squelch(D_ind['Fs']),squelch(D_ind['Ws']),squelch(D_ind['bs'])])

        if not val_bins:
            D_ind['ind_p'] = ind_p[neurons,:]
            np.savez(fname,**D_ind)
        ctm = CellTypesModel(d,K,shared_stim=False,fname=fname,share=share,reload=True,l2=l2s_stim[l2_stim_i],downsample=downsample,W_min = -np.inf,n_sims=n_sims)

        if not val_bins:
            ctm.fit_GMM(m_it_GMM=2000,n_init=trials)
            ctm.set_params()
            ctm.save()
        if not all_n:
            all_stim_valn = [[stim[n][t] for t in tr_trials[n]] for n in val_neurons]
            all_spks_valn = [[binned_spikes[n][t] for t in tr_trials[n]] for n in val_neurons]
            # ctm.val_neurons(all_stim_valn,all_spks_valn) #can be used to evaluate the loss of the hierarchical model on sequential GMM
            ctm.Q_beta(ind_p[val_neurons,:],'Q_val')
            
    else:
        reload = os.path.isfile(fname+'.npz')
        train_stim = [[stim[n][t] for t in tr_trials[n]] for n in neurons]
        train_spks = [[binned_spikes[n][t] for t in tr_trials[n]] for n in neurons]
        ctm = CellTypesModel(d,K,shared_stim=False,fname=fname,share=share,reload=reload,neuron_inds=neurons,l2=l2,downsample=downsample,n_sims=n_sims)  
        if val_bins:
            val_stim=[[stim[n][val_trials[n]]] for n in neurons]
            val_spks=[[binned_spikes[n][val_trials[n]]] for n in neurons]
        else:
            val_stim=[test_stim[n] for n in neurons]
            val_spks=[test_spikes[n] for n in neurons]
        ctm.fit(train_stim,train_spks,max_it=200,n_init=1,val_stim=val_stim,val_spks=val_spks)
        if not val_bins and val_neurons.size>0:
            ctm.val_neurons([stim[n] for n in val_neurons],[binned_spikes[n] for n in val_neurons],val_stim=[test_stim[n] for n in val_neurons],val_spks=[test_spikes[n] for n in val_neurons])
            print('Done; ', fname)
    return True


def main_sim_from_ivscc(seq_method = False, share='W'):
    os.chdir('files')
    id = os.getenv(array_id_str)
    id = 0 if id is None else int(id)
    downsample = 5
    d=[10,20]
    trials = 20
    Kfits = np.arange(1,21)


    if True: #simul betas
        simulBICs = np.load('../summary_files/BIC_allN_share='+share+'.npz')['simulBICs']
        K_max=Kfits[np.argmax(np.max(simulBICs,axis=1))]
        trial_max = np.argmax(simulBICs[Kfits==K_max,:])
        D = np.load('ivscc_n1t2v_simulreg_share_'+share+str(trial_max)+'_K='+str(K_max)+'_sub=allN_train_reps=all.npz',allow_pickle=True)
    else: #seq betas
        simulBICs = np.load('../summary_files/BIC_allN_share='+share+'.npz')['seqBICs']
        K_max=Kfits[np.argmax(simulBICs)]
        D = np.load('ivscc_n1t2v_seqreg_share_'+share+'_K='+str(K_max)+'_sub=allN_train_reps=all'+'.npz',allow_pickle=True)

    l2s_stim = np.logspace(-7,-1,13)
    l2s_self = np.logspace(-7,-1,13)
    if seq_method:
        (l2_stim_i,l2_self_i) = np.unravel_index(id,(l2s_stim.size,l2s_self.size))
        fname = 'sim_frivsccsimul_seq_l2stimi='+str(l2_stim_i)+'_l2selfi='+str(l2_self_i)+'_share='+share
    else:
        l2s = [0.0] if share=='all' else l2s_stim
        (trial,Ki,l2_i) = np.unravel_index(id,(500,Kfits.size,l2s.size))
        Kfit = Kfits[Ki]
        fname = 'sim_frivsccsimul_simul'+str(trial)+'_Kfit'+str(Kfit)+'_l2i'+str(l2_i)+'_share='+share

    seed=0
    np.random.seed(seed)
    sim_stim, sim_spikes, true_betas, true_mus, true_ks = sim_GMMGLM_from_fit(D, drange=20000, downsample=downsample)
    N = sim_stim.shape[0]

    np.random.seed(int(time.time()*10000000000000)%(2**32))
    
    if seq_method:
        Ws = []
        Fs = []
        bs = []
        train_nnlls = []
        for n in range(N):
            F, W, b, train_nnll, _ = fit_GLM(sim_stim[n],sim_spikes[n],d,l2_stim=l2s_stim[l2_stim_i],l2_self=l2s_self[l2_self_i],downsample=downsample)
            train_nnlls.append(train_nnll)
            Fs.append(F)
            Ws.append(W)
            bs.append(b)
        D = {}
        D['train_nnlls']=train_nnlls
        D['Fs']=Fs
        D['Ws']=Ws
        D['bs']=bs
        D['true_betas'] = true_betas
        D['true_mus'] = true_mus
        D['true_ks'] = true_ks
        np.savez(fname,**D)
        print('Done!', fname)
    else:
        ctm = CellTypesModel(d,Kfit,fname=fname,share=share,l2=l2s[l2_i],downsample=downsample)
        D = ctm.fit(sim_stim,sim_spikes,max_it=200,n_init=1,tol=1e-6)
        D['ars'] = adjusted_rand_score(np.argmax(D['Q'],axis=1),true_ks)
        D['true_mus'] = true_mus
        D['true_betas'] = true_betas
        np.savez(fname,**D)
        print('Done!', fname)




def main_sim(seq_method = False,oracle = False,val_neurons = False,mod_select = False,share = 'W'):
    os.chdir('files')

    id = os.getenv(array_id_str)
    id = 0 if id is None else int(id)
    downsample = 5
    trials = 20


    l2s = np.logspace(-5,-2,10)
    l2s_stim = np.logspace(-5,-2,10)
    l2s_self = np.logspace(-6.5,-3.5,10)[:-2]
    Ks = [3,5]
    deltas = [0.5]
    #deltas = list(np.linspace(0.1,1,10))
    
    oracle_seeds = 10
    Wtypes=(share=='W')
    seed_offset = 0
    
    NpK = 10 if val_neurons else 40

    if mod_select:
        Wstds = list(np.logspace(-2,-0.5,10)[np.array([0,7])])
    else:
        Wstds = list(np.logspace(-2,-0.5,10)[:-1])


    if seq_method:
        (seed,K,Wstd,delta) = np.unravel_index(id,(500,len(Ks),len(Wstds),len(deltas))) #many runs with different Omega_K
        if not oracle:
            seed+=oracle_seeds+seed_offset
    else:
        l2s = np.array([0.0]) if share=='all' else l2s
        if not oracle and not mod_select:
            (seed,trial,K,Wstd,delta) = np.unravel_index(id,(500,trials,len(Ks),len(Wstds),len(deltas))) #many runs with different Omega_K
            seed += oracle_seeds+seed_offset
            Kfit = Ks[K]
        elif not oracle and mod_select:
            Kfits = np.arange(1,9)
            nKfits = Kfits.size-(1 if not val_neurons else 0)
            nKfits = 1
            (seed,trial,K,Kfit,Wstd,delta) = np.unravel_index(id,(500,trials,len(Ks),nKfits,len(Wstds),len(deltas)))
            seed += oracle_seeds+seed_offset
            if val_neurons:
                Kfit = Kfits[Kfit]
            else:
                Kfit = Kfits[Kfits!=Ks[K]][Kfit]
        else:
            (seed,trial,K,Wstd,delta,l2_i) = np.unravel_index(id,(500,1,len(Ks),len(Wstds),len(deltas),l2s.size)) #many runs with different Omega_K
            Kfit = Ks[K]

    Wstd=Wstds[Wstd]
    K = Ks[K]
    delta = deltas[delta]
    np.random.seed(seed+1000*val_neurons)

    sim_stim, sim_spikes, true_betas, true_mus = sim_GMMGLM(K, Wstd, drange=20000, delta=delta, d=d, Wtypes=Wtypes, NpK=NpK, downsample=downsample)


    N = sim_spikes.shape[0]
    d = [10,20]
    alg = 'trust-ncg'

    ###Use oracle datasets to select lambda hyperparameters
    if not oracle:
        if seq_method:        
            errors = np.zeros((l2s_stim.size,l2s_self.size,oracle_seeds,2))
            for o_seed in range(oracle_seeds):
                for l2_stim_i in range(l2s_stim.size):
                    for l2_self_i in range(l2s_self.size):
                        fname = 'ivscc_gmmglm_simulations_indreg'+'_Wtypes'*Wtypes+'_seed='+str(o_seed)+'_K='+str(K)+'_1000Wstd='+str(int(1000*Wstd))+'_100delta='+str(int(100*delta))+'_l2stim_i='+str(l2_stim_i)+'_l2self_i='+str(l2_self_i)
                        D = np.load(fname+'.npz')
                        true_Fs = D['true_betas'][:,:d[0]]
                        errors[l2_stim_i,l2_self_i,o_seed,0] += rms(np.ravel(np.squeeze(D['Fs'])-true_Fs)[np.ravel(true_Fs)>thresh])
                        errors[l2_stim_i,l2_self_i,o_seed,1] += rms(np.ravel(D['Ws']-D['true_betas'][:,d[0]:-1])[np.ravel(D['true_betas'][:,d[0]:-1])>thresh])
            l2_stim_i = np.argmin(np.min(np.mean(errors,axis=2)[:,:,0],axis=1))
            l2_self_i = np.argmin(np.min(np.mean(errors,axis=2)[:,:,1],axis=0))
            print(l2_stim_i,l2_self_i)
            if l2_stim_i==0 or l2_stim_i==l2s_stim.size-1 or l2_self_i==0 or l2_self_i==l2s_self.size-1:
                try:
                    if os.path.isfile('bad_l2s.npz'):
                        D = np.load('bad_l2s.npz',allow_pickle=True)
                        fails = list(D['fails'])
                    else:
                        fails = []
                    fails.append({'delta':delta,'K':K,'Wstd':Wstd,'l2_stim_i':l2_stim_i,'l2_self_i':l2_self_i,'errors':errors})
                    np.savez('bad_l2s',fails=fails)
                except:
                    pass

        elif share=='W':
            errors = np.zeros((l2s.size,oracle_seeds))
            for o_seed in range(oracle_seeds):
                for l2_i in range(l2s.size):
                    fname = 'ivscc_gmmglm_simulations_reg'+'_Wtypes'*Wtypes+'_share_'+share+'_K='+str(K)+'_Kfit='+str(K)+'_trial=0_seed='+str(o_seed)+'_1000Wstd='+str(int(1000*Wstd))+'_100delta='+str(int(100*delta))+'_l2i='+str(l2_i)+'_alg='+alg
                    D = np.load(fname+'.npz')
                    true_Fs = D['true_betas'][:,:d[0]]
                    errors[l2_i,o_seed] = rmst(true_Fs,D['Fs'])
            l2_i = np.argmin(np.mean(errors,axis=1))
            if l2_i==0 or l2_i==l2s.size-1:
                try:
                    if os.path.isfile('bad_l2s.npz'):
                        D = np.load('bad_l2s.npz',allow_pickle=True)
                        fails = list(D['fails'])
                    else:
                        fails = []
                    fails.append({'delta':delta,'K':K,'Wstd':Wstd,'l2_i':l2_i,'errors':errors})
                    np.savez('bad_l2s',fails=fails)
                except:
                    pass
        elif share=='all':
            l2_i=0

    

    print(np.sum(sim_spikes,axis=1))
    print(np.max(np.ravel(sim_spikes)))

    ### fit parameters
    np.random.seed(int(time.time()*10000000000000)%(2**32))

    ### Simultaneous method
    if not seq_method:
        fname = 'ivscc_gmmglm_simulations_reg'+'_Wtypes'*Wtypes+'_share_'+share+'_K='+str(K)+'_Kfit='+str(Kfit)+'_trial='+str(trial)+'_seed='+str(seed)+'_1000Wstd='+str(int(1000*Wstd))+'_100delta='+str(int(100*delta))+('_l2i='+str(l2_i))*oracle+'_alg='+alg
        reload = val_neurons #os.path.isfile(fname+'.npz')
        ctm = CellTypesModel(d,Kfit,hess_alg=alg,shared_stim=False,cov_mode='diag',fname=fname,share=share,reload=reload,l2=l2s[l2_i],downsample=downsample)
           
        if not val_neurons:
            D = ctm.fit(sim_stim,sim_spikes,max_it=200,n_init=1,tol=1e-6)
            D['ars'] = adjusted_rand_score(np.argmax(D['Q'],axis=1),np.tile(np.arange(K),N//K))
            D['true_mus'] = true_mus
            D['true_betas'] = true_betas
            np.savez(fname,**D)
        else:
           ctm.val_neurons(sim_stim,sim_spikes)
        print('Done!', fname)

    ### Sequential method
    else:
        guess = False
        for l2_self_i in range(l2s_self.size) if oracle else [l2_self_i]:
            for l2_stim_i in range(l2s_stim.size) if oracle else [l2_stim_i]:
                fname = 'ivscc_gmmglm_simulations_indreg'+'_Wtypes'*Wtypes+'_seed='+str(seed)+'_K='+str(K)+'_1000Wstd='+str(int(1000*Wstd))+'_100delta='+str(int(100*delta))+('_l2stim_i='+str(l2_stim_i)+'_l2self_i='+str(l2_self_i))*oracle
                Ws = []
                Fs = []
                bs = []
                train_nnlls = []
                for n in range(N):
                    to_guess = np.hstack((np.squeeze(D['Fs'][n]),D['Ws'][n],np.expand_dims(D['bs'][n],0))) if guess else None
                    F, W, b, train_nnll, _ = fit_GLM(sim_stim[n],sim_spikes[n],d,l2_stim=l2s_stim[l2_stim_i],l2_self=l2s_self[l2_self_i],guess=to_guess,downsample=downsample)
                    train_nnlls.append(train_nnll)
                    Fs.append(F)
                    Ws.append(W)
                    bs.append(b)
                if not val_neurons:
                    D = {}
                    D['train_nnlls']=train_nnlls
                    D['Fs']=Fs
                    D['Ws']=Ws
                    D['bs']=bs
                    D['true_betas'] = true_betas
                else:
                    D = dict(np.load(fname+'.npz',allow_pickle=True))
                    D['train_nnlls_val_neurons']=train_nnlls
                    D['Fs_val_neurons']=Fs
                    D['Ws_val_neurons']=Ws
                    D['bs_val_neurons']=bs
                    D['true_betas_val_neurons'] = true_betas
                D['true_mus'] = true_mus
                np.savez(fname,**D)
                print('Done!', fname)
                guess = True

        
### uncomment the script you want to run

# if __name__ == "__main__":
#     main_sim()
#     # main_data()
# #    cProfile.run('main_sim()')
