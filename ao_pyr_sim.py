#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 13:32:51 2017

@author: frazin
"""
#%%
import numpy as np
import matplotlib.pyplot as plt
import cPickle as pickle
import scipy.optimize as so
import scipy.interpolate as interp
import Pyr
#import imp  #  this doesn't work.
#imp.load_source('ReadOPD','/users/frazin/MAOS/Py')


# amp is the amplitude of the tilt in lambda/D
def AddTilt(phase, amp=0.25, decim=20):
    npup = phase.shape[0]
    nt = phase.shape[1]
    nnt = nt/decim  # nnt corresponds to fewer time samples
    tilt = amp*np.random.randn(nnt)
    t0 = np.linspace(0, nt-1, nt)
    tt = np.linspace(0, nt-1, nnt)
    cspline = interp.CubicSpline(tt, tilt)
    ttilt = cspline(t0)
    x = np.linspace(-1, 1, npup)
    for n in range(nt):
        ph = ttilt[n]*np.pi*x
        phase[:, n] += ph
    return(phase, ttilt)


# note that a defocus coefficient of 3.31 (i.e, foc = 3.31*x*x)
# contributes a variance of 1 to the phase 
def AddDefocus(phase, amp=1.4, decim=30):
    npup = phase.shape[0]
    nt = phase.shape[1]
    nnt = nt/decim  # nnt corresponds to fewer time samples
    foc = amp*np.random.randn(nnt)
    t0 = np.linspace(0, nt-1, nt)
    tt = np.linspace(0, nt-1, nnt)
    cspline = interp.CubicSpline(tt, foc)
    tfoc = cspline(t0)
    x = np.linspace(-1, 1, npup)
    for n in range(nt):
        ph = tfoc[n]*3.31*x*x
        phase[:, n] += ph
    return(phase, tfoc)

if False:    
    ddir = '/Users/frazin/MAOS/results/PyrSim2'
    # if there is a problem, it is likely because the loc_file needs
    #   to be created by  going into MATLAB.
    (opd, loc) = ReadOPD.ReadOPD(ddir)
    phase = np.array(opd[:, 65, :]*2*np.pi/(0.85e-6))
    del opd, loc
else:
    phase = pickle.load(open("pickles/maos2000.p", "rb"))
nt = phase.shape[1] 
for n in range(nt):
    phase[:,n] -= phase[65,n]

(phase, tilt) = AddTilt(phase)
(phase, foc) = AddDefocus(phase)

phaseH = phase*0.85/1.65  # H band phase
strehl_phase = np.zeros(nt)  # strehl ratios
strehl_phaseH = np.zeros(nt)
for n in range(nt):
    strehl_phase[n] = Pyr.SigmaToStrehl(np.sqrt(np.var(phase[:, n])))
    strehl_phaseH[n] = Pyr.SigmaToStrehl(np.sqrt(np.var(phaseH[:,n])))


pyr = Pyr.Pyr()
regp = .1
MaxIt = 20

pcount = 1.e9  # number of photons
n = 360
y0 = pyr.FieldToIntensity(np.exp(1j*phase[:,n]), FullImage=False)
ym = np.zeros((len(y0), nt))  # noisy measured values
yt = np.zeros(ym.shape)  # true values
yh = np.zeros(ym.shape)  # estimated measurements
wt = np.zeros(ym.shape)  # weights
sol = np.vstack((np.ones((pyr.nc/2, nt)), np.zeros((pyr.nc/2, nt))))  # solutions
cost = np.zeros(nt)
del y0

for n in np.arange(10)+360:
    fieldt = np.exp(1j*phase[:, n])
    yt[:, n] = pyr.FieldToIntensity(fieldt, FullImage=False)
    wt[:, n] = 1 + 0*yt[:, n]
    (s, stats) = pyr.FindMinCost(yt[:, n], wt[:, n], sol[:, n-1], regp)
    sol[:, n] = s
    fieldh = s[0:pyr.nc/2] + 1j*s[pyr.nc/2:]
    yh[:, n] = pyr.FieldToIntensity(fieldh)
    plt.figure(n)
    f, ax = plt.subplots(2, 1, sharex=False, sharey=False)
    ax[0].plot(yt[:, n], 'k.-')
    ax[0].plot(yh[:, n], 'rx')
    ax[0].set_title('n= ' + str(n) + 'data mismatch :' + str(stats['cost']))
    ax[1].plot(np.hstack((np.real(fieldt), np.imag(fieldt))), 'k.-')
    ax[1].plot(s, 'rx')
    ax[1].set_title('solultion quality')




#%%
if False:
    from pyramid_1d import *
    MaxIt = 6
    pcount = 1.e9  # number of photons
    # get pyramid signals
    amp = np.ones(phase.shape[0])
    (y0, pj, aj) = pyramid(g=amp*np.exp(1j*phase[:, 0]), no_derivs=False, pHes=False)
    ym = np.zeros((len(y0), nt))  # noisy measured values
    yt = np.zeros(ym.shape)  # true values
    wt = np.zeros(ym.shape)  # weights
    sol = np.zeros(phase.shape) # solutions
    cost = np.zeros(nt)
    
    
    for n in range(nt):
        (y0, pj, aj) = pyramid(g=amp*np.exp(1j*phase[:,n]), no_derivs=False, pHes=False)
        yt[:, n] = y0
        norm = pcount/np.mean(y0)  # adding photon noise
        ym[:, n] = y0*norm
        ym[:, n] += np.sqrt(ym[:, n])*np.random.randn(len(y0))
        ym[:, n] /= norm
        var = ym[:, n]/norm
    #    wt[:, n] = np.divide(1, var)  #not great for nonlinear problems!
    #    wt[:, n] /= np.mean(wt[:, n])
        wt[:, n] = np.ones(y.shape[0])
    del y0, aj, pj
    
    for n in 360 + np.arange(50):  # 10+np.arange(nt - 80):
        x0 = sol[:, n-1]
        
        #estimate tilt/defocus --- THIS MAKES IT WORSE!
    #    a0 = FitLowOrder(x0)
    #    g0 = np.exp(1j*x0)
    #    fargs = (g0, ym[:,n], wt[:,n], False)
    #    opts = {'disp': True, 'maxiter': MaxIt, 'xtol': 1.e-6, 'return_all': True}
    #    lo_result = so.minimize(WrapPyramidModal, a0, args=fargs, method='Newton-CG',
    #                         options=opts, jac=True, hess=WrapPyramidModalHessian)
    
        x0 += CoefToPhase(lo_result.x)
        fargs = (ym[:, n], wt[:, n], False, False)
        opts = {'disp': True, 'maxiter': MaxIt, 'xtol': 1.e-6, 'return_all': True}
        result = so.minimize(WrapPyramid, x0, args=fargs, method='Newton-CG',
                             options=opts, jac=True, hess=WrapPyramidHessian)
        sol[:, n] = result.x
    
        plt.figure(); plt.plot(phase[:,n],'k.-')
        plt.plot(sol[:,n],'rx'); plt.title(str(n) + ': '+ str(strehl_phase[n]))
        cost[n] = result.fun
