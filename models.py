from scipy.stats import multivariate_normal as mult_norm
import statsmodels.api as sm
from sklearn.mixture import GaussianMixture
from sklearn.metrics import pairwise_distances_argmin as pdargmin
import time
import signal
import gc
import os
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import scipy
from scipy.optimize import minimize
from itertools import combinations
from helpers import *
   

### Simulating


def sim_GLM(stim, stim_filt, spk_filt, offset,noise_amp=None,noise_len=0,drop_rate = None, drop_len = None, downsample=None, smax=1,n_sims=1):
    downsample = 1 if downsample is None else downsample
    if len(stim.shape)==1:
        stim=np.expand_dims(stim,0)
    if len(stim_filt.shape)==1:
        stim_filt=np.expand_dims(stim_filt,0)
    d = (stim_filt.shape[-1], spk_filt.size)
    T = stim.shape[1]
    spks = np.zeros((n_sims,T)) # starts at time step d
    rate = np.zeros((n_sims,T))
    stim_filt_todot = np.fliplr(stim_filt).reshape(1,-1)
    spk_filt_todot = np.fliplr(np.expand_dims(spk_filt,0)).T
    if noise_amp is not None:
        noise_sig = np.random.normal(loc=0.0,size=(n_sims,T))
        if noise_len>0:
            t_filt = np.arange(-2*noise_len,2*noise_len+1)
            filt = np.exp(-np.square(t_filt)/(noise_len**2*2))
            noise_sig = np.vstack([np.convolve(noise_sig[s,:],filt,mode='same') for s in n_sims])
        noise_sig = noise_amp*noise_sig/np.sqrt(np.mean(np.square(noise_sig)))
    for i in range(T):
        recent_stim = stim[:,max(i-d[0]*downsample+1,0):i+1]
        recent_stim = np.hstack((np.zeros((stim.shape[0],max(0,d[0]*downsample-i-1))),recent_stim))
        recent_stim = np.mean(recent_stim.reshape(stim.shape[0],-1,downsample),axis=-1)
        recent_spikes = spks[:,max(0,i-d[1]):i]
        recent_spikes = np.hstack((np.zeros((n_sims,max(0,d[1]-i))),recent_spikes))
        rate[:,i] = np.squeeze(np.exp(stim_filt_todot.dot(recent_stim.reshape(-1,1))+recent_spikes.dot(spk_filt_todot)+offset))
        rate[:,i] = np.minimum(rate[:,i],20)
        if noise_amp is not None:
            rate[:,i] = np.maximum(0,rate[:,i]+noise_sig[:,i])
        c = np.random.poisson(rate[:,i])
        if drop_rate is None:
            spks[:,i] = c
        else:
            if drop_len is None:
                spks[:,i] = c*(np.random.uniform(size=n_sims)>drop_rate)
            else:
                spks[:,i] = c
                for s in range(n_sims):
                    if (np.random.uniform()<drop_rate/drop_len):
                        spks[s,max(0,i+1-drop_len):i+1] = 0
        
        spks[:,i] = np.minimum(spks[:,i],smax)
    return spks, rate


def sim_GMMGLM(K, Wstd, delta = 1.0, NpK=40, d = [10,20], drange = 20000, beta_seed = None, Wtypes=True, Kmax=2, downsample=None):
    N = NpK*K
    true_mus = np.zeros((K,d[1]+(d[0]+1)*(not Wtypes)))
    print(true_mus.shape)

    #Ws
    param_spacing = np.arange(K)/(Kmax-1)*delta
    t_abs = param_spacing*14+2
    t_rel = param_spacing*4+2
    a = (1-param_spacing)*0.2
    c_isi = param_spacing*16 + 3
    sig_isi = 3+2*param_spacing
    true_mus = np.zeros((K,d[1]+(d[0]+1)*(not Wtypes)))
    
    for k in range(K):
        if Wtypes:
            true_mus[k,:] = -np.exp(-(np.arange(d[1])-t_abs[k])/t_rel[k]) + a[k]*np.exp(-0.5*np.square((np.arange(d[1])-c_isi[k])/sig_isi[k]))
            true_mus[k,:] = np.maximum(true_mus[k,:],-15)
        else:
            #Fs
            if downsample==None:
                true_mus[k,:d[0]] = np.exp(-np.arange(d[0])/4)*(0.5+param_spacing[k])
            else:
                x = np.exp(-np.arange(d[0]*downsample)/4)*(0.5+param_spacing[k])        
                true_mus[k,:d[0]] = np.sum(x.reshape((-1,downsample)),axis=-1)
            #Ws
            true_mus[k,d[0]:d[0]+d[1]] = -np.exp(-(np.arange(d[1])-t_abs[k])/t_rel[k]) + a[k]*np.exp(-0.5*np.square((np.arange(d[1])-c_isi[k])/sig_isi[k]))
            true_mus[k,d[0]:d[0]+d[1]] = np.maximum(true_mus[k,d[0]:d[0]+d[1]],-15)

            #bs
            true_mus[k,-1] = -4.5-2*param_spacing[k]

    if beta_seed is not None:
        np.random.seed(beta_seed)

    true_betas = np.zeros((N,d[0]+d[1]+1))
    for k in range(K):
        for n_k in range(N//K):
            if Wtypes:
                true_betas[K*n_k+k,d[0]:d[0]+d[1]] = true_mus[k,:]+np.random.normal(scale=Wstd,size=(d[1],))   # self-interaction 
                if downsample is None:
                    true_betas[K*n_k+k,:d[0]] =  np.exp(-np.arange(d[0])/4)                 # stim
                else:
                    x = 0.9*np.exp(-np.arange(d[0]*downsample)/4)
                    true_betas[K*n_k+k,:d[0]] = np.sum(x.reshape((-1,downsample)),axis=-1)
                true_betas[K*n_k+k,-1] =  -5                                            # offset
            else:
                true_betas[K*n_k+k,:] = true_mus[k,:]+np.random.normal(scale=Wstd,size=(d[0]+d[1]+1,))

    ### Generate responses to data
    D = np.load('ivscc_data_n12.npz',allow_pickle=True)
    sim_stim = [np.concatenate( D['binned_stim'][n].extend(D['test_binned_stim'][n]) ) for n in range(N)]
    sim_stim = np.hstack([ s[(s.size-drange)//2:(s.size+drange)//2] for s in sim_stim ])
    sim_stim /= np.max(sim_stim,axis=1,keepdims=True)
    sim_spikes = np.zeros_like(sim_stim)
    for n in range(N):
        sim_spikes[n,:],_ = sim_GLM(sim_stim[n:n+1,:],true_betas[n,:d[0]],true_betas[n,d[0]:d[0]+d[1]],true_betas[n,-1],downsample=downsample)

    return sim_stim, sim_spikes, true_betas, true_mus


### Fitting

def construct_Xdsn(flat_stimuluses, binned_spikeses, d, downsample=None):
    if type(flat_stimuluses) is np.ndarray:
        flat_stimuluses = (flat_stimuluses,)
        binned_spikeses = (binned_spikeses,)
    X_dsns = []
    for fs, bs in zip(flat_stimuluses,binned_spikeses):
        T = bs.size # T is number of time bins
        sh = fs.shape # M is the size of a stimulus
        if len(sh)==1:
            M = 1
            T1 = sh[0]
            flat_stimulus = np.expand_dims(fs,axis=0)
        else:
            (M,T1) = sh
            flat_stimulus = np.array(fs)

        assert T==T1, "arrays have different number of time samples"
        d1,d2 = d
        dmax = max(d)
        if downsample is not None:
            dmax = max((d1*downsample,d2))
        binned_spikes = np.concatenate((np.zeros((dmax,)),bs))
        flat_stimulus = np.concatenate((np.zeros((M,dmax)),flat_stimulus),axis=1)
        binned_spikes = np.expand_dims(binned_spikes,0)
        T = binned_spikes.size
        X_dsn = np.ones((T-dmax,M*d1+d2+1))
        for t in range(T-dmax):
            if downsample is None:
                X_dsn[t,:M*d1] = np.fliplr(flat_stimulus[:,t+dmax+1-d1:t+dmax+1]).reshape((1,-1))  #stimulus inputs
            else:
                X_dsn[t,:M*d1] = np.fliplr(np.mean(flat_stimulus[:,t+dmax+1-d1*downsample:t+dmax+1].reshape((M,-1,downsample)),axis=-1)).reshape((1,-1))  #stimulus inputs
            X_dsn[t,M*d1:-1] = np.fliplr(binned_spikes[:,t+dmax-d2:t+dmax]).reshape((1,-1)) #spike inputs
        X_dsns.append(X_dsn)
    return np.vstack(X_dsns), np.hstack(binned_spikeses)




def fit_GLM(stim,spks,d,val_stim=None,val_spks=None,n_sims=1,sig=5,downsample=None,l2_stim=0,l2_self=0,guess=None):
    if type(stim) is np.ndarray:
        stim = (stim,)
        spks = (spks,)
    if type(val_stim) is np.ndarray:
        val_stim = (val_stim,)
        val_spks = (val_spks,)
    d1,d2 = d
    Xdsn, y = construct_Xdsn(stim,spks,d,downsample=downsample)
    model =  sm.GLM(y,Xdsn,family = sm.families.Poisson())
    m_it = 300
    
    if guess is None:
        guess = np.random.rand(Xdsn.shape[1])

    fun = lambda p: -model.loglike(p)/y.size + 1/2*l2_stim*np.sum(np.square(p[:d1])) + 1/2*l2_self*np.sum(np.square(p[d1:d1+d2]))
    jac = lambda p: -model.score(p)/y.size + np.hstack((l2_stim*p[:d1],l2_self*p[d1:d1+d2],[0]))
    hess = lambda p: -model.hessian(p)/y.size + np.diag(np.hstack(([l2_stim]*d1,[l2_self]*d2,[0])))
    res = minimize(fun,guess,jac=jac,hess=hess,method='trust-ncg',options={'maxiter':m_it, 'disp':False},tol=1e-10)
    conved = res.success
    p = res.x
    se = np.zeros_like(p)

    b = p[-1]
    bse = se[-1]
    stim_filt = p[:-1-d2].reshape([-1,d1])
    stim_se = se[:-1-d2]
    spk_filt = p[-1-d2:-1]
    spk_se = se[-1-d2:-1]
    sim_spks = sim_GLM(stim[0],np.squeeze(stim_filt),spk_filt,b,downsample = downsample,n_sims=n_sims)[0]
    train_corr = EVratio(sim_spks,spks,sig)
    train_nnll = res.fun
    dev = None
    
    if val_stim is not None:
        X_val,y_val = construct_Xdsn(val_stim,val_spks,d,downsample=downsample)
        val_model = sm.GLM(y_val,X_val,family = sm.families.Poisson())
        val_nnll = -val_model.loglike(p)/y_val.size
        sim_val_spks = sim_GLM(val_stim[0],stim_filt,spk_filt,b,downsample = downsample,n_sims=n_sims)[0]
        val_corr = EVratio(sim_val_spks,val_spks,sig)
        return stim_filt, spk_filt, b, train_nnll, train_corr, val_nnll, val_corr
    return stim_filt, spk_filt, b, train_nnll, train_corr



class CellTypesModel:
    def __init__(self,stim,spks,d,K,shared_stim=False,family=sm.families.Poisson(),cov_mode='diag',share='W',W_min = -15,hess_alg = 'trust-ncg',m_it_GLM=200,fname='run',reload = False,l2=0.0,downsample=None,sig=5,n_sims=1,**kwargs):
        ###store any additional metadata passed in in kwargs
        for k,v in kwargs.items():
            setattr(self,k,v)
        N = len(spks) if type(spks) is list else spks.shape[0]
        #create design matrices
        X_dsns = []
        ys = []
        for n in range(N):
            if shared_stim:
                X_dsn, y = construct_Xdsn(stim,spks[n],d,downsample=downsample)
            else:
                X_dsn, y = construct_Xdsn(stim[n],spks[n],d,downsample=downsample)
            X_dsns.append(X_dsn)
            ys.append(y)

        T_dsns = [X_dsns[n].shape[0] for n in range(N)]
        L_dsn = X_dsns[-1].shape[1]

        #indices of shared params
        if share=='W':
            shared_i = np.arange(d[0],d[0]+d[1])  #W
        if share=='all':
            shared_i = np.arange(L_dsn)         #all
        if share=='bW':
            shared_i = np.arange(d[0],L_dsn)    #b and W

        d_s = shared_i.size
        shared_i2 = (np.repeat(shared_i,d_s),np.tile(shared_i,d_s))
        reg_i = np.setdiff1d(np.arange(L_dsn-1),shared_i)
        d_r = reg_i.size
        reg_i2 = (np.repeat(reg_i,d_r),np.tile(reg_i,d_r))

        self.ind_mus = np.zeros((N,K,d_s))
        self.ind_sig2s = np.ones((N,K,d_s,d_s))
        self.Q = np.zeros((N,K))
        self.l2 = l2

        if not reload:
            ind_p = np.random.rand(N,L_dsn)
            self.train_spikes = [np.sum(y) for y in ys]
            for n in range(N):
                model =  sm.GLM(ys[n],X_dsns[n],family = family)
                fun = lambda p: -model.loglike(p)/ys[n].size + 1/2*self.l2*np.sum(np.square(p[reg_i]))
                jac = lambda p: -model.score(p)/ys[n].size + self.l2*self.put(np.zeros((L_dsn)),reg_i,p[reg_i])
                hess = lambda p: -model.hessian(p)/ys[n].size + self.l2*self.put(np.zeros((L_dsn,L_dsn)),reg_i2,np.eye(d_r))
                res = minimize(fun,np.random.rand(L_dsn),jac=jac,hess=hess,method=hess_alg,options={'maxiter':m_it_GLM, 'disp':False},tol=1e-6)
                ind_p[n,:] = res.x #np.maximum(res.x,W_min)
            self.ind_p = ind_p
            self.iter = 0

        self.conv = np.inf
        self.losses = []
        self.convs = []
        if reload:
            self.load(dict(np.load(fname+'.npz',allow_pickle=True)))
        try:
            self.gmm_seq = self.gmm_seq[()]
        except:
            pass
        self.downsample=downsample
        self.sig=sig
        self.n_sims = n_sims
        self.stim = stim
        self.spks = spks
        self.shared_stim = shared_stim
        self.fname=fname
        self.W_min = W_min
        self.m_it_GLM = m_it_GLM
        self.hess_alg=hess_alg
        self.cov_mode=cov_mode
        self.T_dsns=T_dsns
        self.L_dsn=L_dsn
        self.K = K
        self.N = N
        self.X_dsns=X_dsns
        self.ys = ys
        self.d_s = d_s
        self.d = d
        self.shared_i = shared_i
        self.shared_i2 = shared_i2
        self.d_r = d_r
        self.reg_i = reg_i
        self.reg_i2 = reg_i2
        self.convs = list(self.convs)
        self.losses = list(self.losses)
        self.time = os.times()[-1]
        self.start_time = self.time
        self.family = family
        self.set_params()


    def load(self,D):
        for k,v in D.items():
            setattr(self,k,v)

    def set_params(self):
        self.time = os.times()[-1]
        self.params = {k:v for k, v in self.__dict__.items() if not ((k.startswith('__') and k.endswith('__')) or k=='ys' or k=='X_dsns' or k=='params' or k=='stim' or k=='spks')}

    def save(self):
        np.savez(self.fname,**self.params)


    def put(self,arr,inds,vals):
        if type(inds)==tuple:
            arr[inds[0],inds[1]] = np.ravel(vals)
        else:
            arr[inds] = vals
        return arr


    def Q_beta(self,beta,name):
        beta_shared = beta[:,self.shared_i]
        Q = self.gmm_seq.predict_proba(beta_shared)
        self.val_losses_GLM = self.gmm_seq.score_samples(beta_shared)
        self.val_loss_GLM = np.mean(self.val_losses_GLM)
        setattr(self,name,Q)
        self.set_params()
        self.save()


    def initialize(self,m_it_GMM=100,n_init=10):
        K=self.K
        gmm = GaussianMixture(n_components=K,covariance_type=self.cov_mode, max_iter=m_it_GMM,n_init=n_init)
        gmm.fit(np.maximum(self.ind_p[:,self.shared_i],self.W_min))
        self.wts = gmm.weights_
        self.Q = gmm.predict_proba(np.maximum(self.ind_p[:,self.shared_i],self.W_min))
        self.gmm_BIC = -gmm.bic(np.maximum(self.ind_p[:,self.shared_i],self.W_min))
        if self.cov_mode == 'full':
            self.P_k = gmm.precisions_
            self.C_k = gmm.covariances_
            self.detC_k = [np.linalg.det(self.C_k[k]) for k in range(K)]
        else:
            self.P_k = [np.diag(gmm.precisions_[k]) for k in range(K)]
            self.C_k = [np.diag(gmm.covariances_[k]) for k in range(K)]
            self.detC_k = [np.prod(gmm.covariances_[k]) for k in range(K)]
        self.mu_k = gmm.means_
        self.bad = [False]*K
        self.p = np.tile(np.expand_dims(self.ind_p,1),(1,K,1)) #N x K x Ldsn
        self.gmm_seq = gmm


    def update(self,verbose=True):
        K=self.K
        N=self.N
        T_dsns=self.T_dsns
        L_dsn=self.L_dsn
        d_s = self.d_s
        shared_i = self.shared_i
        shared_i2 = self.shared_i2
        d_r = self.d_r
        reg_i = self.reg_i
        reg_i2 = self.reg_i2
        logposts = -np.inf*np.ones((N,K))

        #E-step: update q(z_i)
        time1 = time.time()
        good_Ks = np.arange(K)[~np.array(self.bad)]
        for n in range(N):
            model =  sm.GLM(self.ys[n],self.X_dsns[n],family = self.family)
            for k in good_Ks:
                fun = lambda p: -(model.loglike(p)-(p[shared_i]-self.mu_k[k,:]).dot(self.P_k[k]).dot((p[shared_i]-self.mu_k[k,:]).T)/2)/self.ys[n].size + 1/2*self.l2*np.sum(np.square(p[reg_i]))
                jac = lambda p: -(model.score(p)-self.put(np.zeros_like(p),shared_i,(p[shared_i]-self.mu_k[k,:]).dot(self.P_k[k])))/self.ys[n].size + self.l2*self.put(np.zeros((L_dsn)),reg_i,p[reg_i])
                hess = lambda p: -(model.hessian(p)-(self.put(np.zeros((L_dsn,L_dsn)),shared_i2,self.P_k[k])))/self.ys[n].size + self.l2*self.put(np.zeros((L_dsn,L_dsn)),reg_i2,np.eye(d_r))
                res = minimize(fun,self.p[n,k,:],jac=jac,hess=hess,method=self.hess_alg,options={'maxiter':self.m_it_GLM,'disp':False},tol=1e-6)
                
                self.ind_mus[n,k,:] = res.x[shared_i] #np.maximum(res.x[shared_i],self.W_min)
                self.p[n,k,:] = res.x #np.maximum(res.x,self.W_min)
                if self.cov_mode=='diag':
                    C = np.diag(1.0/(np.diag(res.hess)*self.ys[n].size+1e-300))
                else:
                    C = np.linalg.inv(res.hess*self.ys[n].size)+1e-300*np.eye(L_dsn)
                self.ind_sig2s[n,k,:,:] = C[shared_i,:][:,shared_i]
                logposts[n,k] = -(res.fun)*self.ys[n].size - 1/2*np.log(self.detC_k[k]+1e-300) + 1/2*np.log(np.linalg.det(C)+1e-300) + np.log(self.wts[k]+1e-300) + 1/2*d_r*np.log(self.l2*self.ys[n].size+1e-300) #steepest descent approx
            Qz = np.exp(logposts[n,:]-np.max(logposts[n,:]))
            Qz[self.bad] = 0
            self.Q[n,:] = Qz/(np.sum(Qz)+1e-300)
        logp = np.sum(np.log(np.sum(np.exp(logposts-np.max(logposts,axis=1,keepdims=True)),axis=1))+np.max(logposts,axis=1))
        self.loss = -logp/(np.sum(T_dsns))
        self.losses.append(self.loss)
        time2 = time.time()
        if verbose:
            print("E-Step: ", time2-time1, " seconds")

        #M-step: update MOG
        self.wts = np.sum(self.Q,axis=0)/N
        self.mu_k = np.array([self.Q[:,k].dot(self.ind_mus[:,k,:])/(N*self.wts[k]+1e-300) for k in range(K)])
        if self.cov_mode=='full':
            self.C_k = [np.sum(np.expand_dims(np.expand_dims(self.Q[:,k]/(N*self.wts[k]+1e-300),-1),-1)*(self.ind_sig2s[:,k,:,:]
                    +np.matmul(np.transpose(self.ind_mus[:,k:k+1,:],(0,2,1)),self.ind_mus[:,k:k+1,:])),axis=0)
                    -self.mu_k[k:k+1,:].T.dot(self.mu_k[k:k+1,:]) for k in range(K)]
        else:
            self.C_k = [self.Q[:,k].dot(self.ind_sig2s[:,k,np.arange(d_s),np.arange(d_s)]
                    +self.ind_mus[:,k,:]**2
                    -self.mu_k[k,:]**2)/(N*self.wts[k]) for k in range(K)]
        self.bad = [self.bad[k] or np.any(np.isnan(self.mu_k[k,:])) or np.any(np.isnan(self.C_k[k])) for k in range(K)]
        self.mu_k[self.bad,:] = -5000
        self.wts[self.bad] = 0
        self.wts/=(np.sum(self.wts)+1e-300)

        if self.cov_mode=='diag':
            self.C_k = [np.diag(self.C_k[k]) for k in range(K)]
            self.C_k = [0.7*np.eye(d_s) if self.bad[k] else np.minimum(self.C_k[k],40) for k in range(K)]
            self.P_k = [np.diag(1.0/np.diag(self.C_k[k])) for k in range(K)]
            self.detC_k = [np.prod(np.diag(self.C_k[k])) for k in range(K)]
        else:
            self.C_k = [0.7*np.eye(d_s) if self.bad[k] else np.minimum(self.C_k[k],40) for k in range(K)]
            self.P_k = [np.linalg.inv(self.C_k[k]) for k in range(K)]
            self.detC_k = [np.linalg.det(self.C_k[k]) for k in range(K)]
        time3 = time.time()
        if verbose:
            print("M-Step: ", time3-time2, " seconds")


    def fit(self,n_init=1,max_it=5,m_it_GMM=100,tol=1e-5,verbose=True):

        if self.iter == 0:
            if verbose:
                print('Starting E-M')

            # Run n_init models for 1* iteration
            best_loss = np.inf
            best_params = {}
            self.iter = 1
            for init in range(n_init):
                self.initialize(m_it_GMM=100)
                self.update(verbose=verbose)# could update multiple times if desired
                if verbose:
                    print('Init '+ str(init)+': loss = '+str(self.loss))
                if self.loss<best_loss:
                    best_loss = self.loss
                    best_params = self.params
            self.params = best_params
            for k,v in best_params.items():
                setattr(self,k,v)
            self.loss=best_loss
            self.save()

        # Run best model for max_it iterations
        while ((self.conv>tol) or (self.iter<10)) and (self.iter<max_it):
            mu_old = np.array(self.mu_k)
            ind_mus_old = np.array(self.ind_mus)
            Ck_old = np.array(self.C_k)
            Q_old = np.array(self.Q)
            wts_old = np.array(self.wts)
            self.update(verbose=verbose)
            self.iter += 1
            self.set_params()
            self.save()

            #convergence check
            c = [np.max(np.abs(ind_mus_old-self.ind_mus)[np.arange(self.N),np.argmax(self.Q,axis=1),:]),
                           np.max(np.abs(mu_old-self.mu_k)[(~np.array(self.bad))*(self.wts>1/(2*self.N))]),
                           np.max(np.abs(Ck_old-self.C_k)[(~np.array(self.bad))*(self.wts>1/(2*self.N))]),
                           np.max(np.abs(Q_old-self.Q)),
                           np.max(np.abs(wts_old-self.wts))]
            self.conv = (self.losses[-2]-self.losses[-1])/self.losses[-1]
            if self.conv<-1e-12:
                print('FAIL!!!!!!!!!')
                if os.path.isfile('fails.npz'):
                    D = np.load('fails.npz',allow_pickle=True)
                    fails = list(D['fails'])
                else:
                    fails = []
                fails.append(self.fname)
                np.savez('fails',fails=fails)

            self.convs.append(c)

            if verbose:
                print('Iteration '+ str(self.iter)+': loss = '+str(self.loss)+', conv = '+str(self.conv))


    def val_neurons(self,verbose=True, val_stim=None,val_spks=None,xval=False):
        self.val_spikes = [np.sum(y) for y in self.ys]
        K=self.K
        N=self.N
        T_dsns=self.T_dsns
        L_dsn=self.L_dsn
        d=self.d
        d_s = self.d_s
        shared_i = self.shared_i
        shared_i2 = self.shared_i2
        d_r = self.d_r
        reg_i = self.reg_i
        reg_i2 = self.reg_i2
        logposts = -np.inf*np.ones((N,K))
        self.Q_val = np.zeros((N,K))
        self.p_val = np.zeros((N,K,L_dsn))
        self.ind_mus_val = np.zeros((N,K,d_s))
        self.ind_sig2s_val = np.zeros((N,K,d_s,d_s))
        self.val_neurons_train_corrs = np.zeros((N,))
        self.val_neurons_train_nnlls = np.zeros((N,))

        time1 = time.time()
        good_Ks = np.arange(K)[~np.array(self.bad)]
        for n in range(N):
            model =  sm.GLM(self.ys[n],self.X_dsns[n],family = self.family)
            for k in good_Ks:
                fun = lambda p: -(model.loglike(p)-(p[shared_i]-self.mu_k[k,:]).dot(self.P_k[k]).dot((p[shared_i]-self.mu_k[k,:]).T)/2)/self.ys[n].size + 1/2*self.l2*np.sum(np.square(p[reg_i]))
                jac = lambda p: -(model.score(p)-self.put(np.zeros_like(p),shared_i,(p[shared_i]-self.mu_k[k,:]).dot(self.P_k[k])))/self.ys[n].size + self.l2*self.put(np.zeros((L_dsn)),reg_i,p[reg_i])
                hess = lambda p: -(model.hessian(p)-(self.put(np.zeros((L_dsn,L_dsn)),shared_i2,self.P_k[k])))/self.ys[n].size + self.l2*self.put(np.zeros((L_dsn,L_dsn)),reg_i2,np.eye(d_r))
                res = minimize(fun,np.random.rand(L_dsn),jac=jac,hess=hess,method=self.hess_alg,options={'maxiter':self.m_it_GLM,'disp':False},tol=1e-6)

                self.ind_mus_val[n,k,:] = res.x[shared_i] #np.maximum(res.x[shared_i],self.W_min)
                self.p_val[n,k,:] = res.x #np.maximum(res.x,self.W_min)
                if self.cov_mode=='diag':
                    C = np.diag(1.0/(np.diag(res.hess)*self.ys[n].size+1e-300))
                else:
                    C = np.linalg.inv(res.hess*self.ys[n].size)+1e-300*np.eye(L_dsn)
                self.ind_sig2s_val[n,k,:,:] = C[shared_i,:][:,shared_i]
                logposts[n,k] = -(res.fun)*self.ys[n].size - 1/2*np.log(self.detC_k[k]+1e-300) + 1/2*np.log(np.linalg.det(C)+1e-300) + np.log(self.wts[k]+1e-300) + 1/2*d_r*np.log(self.l2*self.ys[n].size+1e-300) #steepest descent approx
            Qz = np.exp(logposts[n,:]-np.max(logposts[n,:]))
            Qz[self.bad] = 0
            self.Q_val[n,:] = Qz/(np.sum(Qz)+1e-300)
        logp = np.log(np.sum(np.exp(logposts-np.max(logposts,axis=1,keepdims=True)),axis=1))+np.max(logposts,axis=1)
        self.val_losses = -logp/T_dsns
        self.val_loss = -np.sum(logp)/(np.sum(T_dsns))
        self.p_MAP_val = self.p_val[np.arange(N),np.argmax(self.Q_val,axis=1)]
        self.gmm_TLs = self.gmm_seq.score_samples(self.p_MAP_val[:,shared_i])
        for n in range(N):
            model =  sm.GLM(self.ys[n],self.X_dsns[n],family = self.family)
            self.val_neurons_train_nnlls[n] = -model.loglike(self.p_MAP_val[n,:])/T_dsns[n]
            sim_spks = sim_GLM(self.stim[n][0],self.p_MAP_val[n,:-1-d[1]],self.p_MAP_val[n,-1-d[1]:-1],self.p_MAP_val[n,-1],downsample = self.downsample,n_sims=self.n_sims)[0]
            self.val_neurons_train_corrs[n] = EVratio(sim_spks,self.spks[n],self.sig)

        prefix = 'val_neurons_val'
        out_dict={}

        if val_spks is not None:
            val_nnlls = np.zeros((N,))
            val_corrs = np.zeros((N,))
            for n in range(N):
                vsn = val_stim if self.shared_stim else val_stim[n]
                X_dsn, y = construct_Xdsn(vsn,val_spks[n],d,downsample=self.downsample)
                sim_val_spks = sim_GLM(vsn[0],self.p_MAP_val[n,:-1-d[1]],self.p_MAP_val[n,-1-d[1]:-1],self.p_MAP_val[n,-1],downsample = self.downsample,n_sims=self.n_sims)[0]
                val_model = sm.GLM(y,X_dsn,family = self.family)
                val_nnlls[n] = -val_model.loglike(self.p_MAP_val[n,:])/(y.size)
                val_corrs[n] = EVratio(sim_val_spks,val_spks[n],self.sig)
            out_dict[prefix+'_nnlls'] = val_nnlls
            out_dict[prefix+'_corrs'] = val_corrs

        time2 = time.time()
        if verbose:
            print("Val_neurons: ", time2-time1, " seconds")

        if not xval:
            self.set_params()
        else:
            try:
                out_dict[prefix+'_nnlls'] = np.vstack((getattr(self,prefix+'_nnlls'),val_nnlls))
                out_dict[prefix+'_corrs'] = np.vstack((getattr(self,prefix+'_corrs'),val_corrs))
            except:
                pass
        self.params.update(out_dict)
        self.save()


    def results(self,val_stim=None,val_spks=None,prefix='val'):
        K=self.K
        N=self.N
        T_dsns=self.T_dsns
        L_dsn=self.L_dsn
        d_s = self.d_s
        d=self.d

        p_MAP = np.zeros((N,L_dsn))
        train_nnlls = np.zeros((N,))
        train_corrs = np.zeros((N,))
        for n in range(N):
            model =  sm.GLM(self.ys[n],self.X_dsns[n],family = self.family)
            k = np.argmax(self.Q[n,:])
            p_MAP[n,:] = self.p[n,k,:]
            train_nnlls[n] = -model.loglike(p_MAP[n,:])/T_dsns[n]
            sim_spks = sim_GLM(self.stim[n][0],p_MAP[n,:-1-d[1]],p_MAP[n,-1-d[1]:-1],p_MAP[n,-1],downsample = self.downsample,n_sims=self.n_sims)[0]
            train_corrs[n] = EVratio(sim_spks,self.spks[n],self.sig)


        if self.cov_mode=='full':
            BIC = -self.loss*np.sum(T_dsns)-1/2*(K*(1+d_s+d_s**2)-1)*np.log(N)
        else:
            BIC = -self.loss*np.sum(T_dsns)-1/2*(K*(1+d_s+d_s)-1)*np.log(N)

        out_dict = dict(BIC=BIC,
                        Fs=p_MAP[:,:-1-d[1]],Ws=p_MAP[:,-1-d[1]:-1],bs=p_MAP[:,-1],
                        mu_k=self.mu_k,C_k=self.C_k,Q=self.Q,pi=self.wts,p=self.p,
                        train_nnlls=train_nnlls,train_corrs=train_corrs)

        if val_stim is not None:
            self.val_spikes = [np.sum(y) for y in val_spks]
            val_nnlls = np.zeros((N,))
            val_corrs = np.zeros((N,))
            for n in range(N):
                vsn = val_stim if self.shared_stim else val_stim[n]
                X_dsn, y = construct_Xdsn(vsn,val_spks[n],d,downsample=self.downsample)
                sim_val_spks = sim_GLM(vsn[0],p_MAP[n,:-1-d[1]],p_MAP[n,-1-d[1]:-1],p_MAP[n,-1],downsample = self.downsample,n_sims=self.n_sims)[0]
                val_model = sm.GLM(y,X_dsn,family = self.family)
                val_corrs[n] = EVratio(sim_val_spks,val_spks[n],self.sig)
                val_nnlls[n] = -val_model.loglike(p_MAP[n,:])/(y.size)
                out_dict[prefix+'_nnlls'] = val_nnlls
                out_dict[prefix+'_corrs'] = val_corrs
        self.set_params()
        out_dict.update(self.params)
        np.savez(self.fname,**out_dict)
        return(out_dict)



