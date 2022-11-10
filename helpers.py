import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import axes_grid1

def rms(x):
    return np.sqrt(np.mean(np.square(x)))


def rmst(true,est,thresh=-4): 
    return rms(np.ravel(true)-np.ravel(est)[np.ravel(true)>thresh])


def mrm(x):
    return np.mean(np.sqrt(np.median(x,axis=1)))


def set_size(w,h, ax=None):
    """ w, h: width, height in inches """
    if not ax: ax=plt.gca()
    l = ax.figure.subplotpars.left
    r = ax.figure.subplotpars.right
    t = ax.figure.subplotpars.top
    b = ax.figure.subplotpars.bottom
    figw = float(w)/(r-l)
    figh = float(h)/(t-b)
    ax.figure.set_size_inches(figw, figh)


def add_colorbar(im, aspect=20, pad_fraction=0.5, **kwargs):
    """Add a vertical color bar to an image plot."""
    divider = axes_grid1.make_axes_locatable(im.axes)
    width = axes_grid1.axes_size.AxesY(im.axes, aspect=1./aspect)
    pad = axes_grid1.axes_size.Fraction(pad_fraction, width)
    current_ax = plt.gca()
    cax = divider.append_axes("right", size=width, pad=pad)
    plt.sca(current_ax)
    return im.axes.figure.colorbar(im, cax=cax, **kwargs)


def EV(x1,x2):
    return np.mean((np.var(x1,axis=1)+np.var(x2,axis=-1)-np.var(x1-x2,axis=1))/(np.var(x1,axis=1)+np.var(x2)))


def EVratio(preds,data,sig):
    t_filt = np.arange(-100,101)
    filt = np.exp(-np.square(t_filt)/(sig**2*2))
    filt = filt/np.sum(filt)
    stPSTHm = [np.convolve(x,filt,mode='same') for x in preds]
    stPSTHd = [np.convolve(x,filt,mode='same') for x in data]
    PSTHd = np.mean(stPSTHd,axis=0)
    PSTHm = np.mean(stPSTHm,axis=0)
    
    EVd = EV(stPSTHd,PSTHd)
    EVdm = EV(stPSTHd,PSTHm)
    return EVdm/EVd


