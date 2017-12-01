#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 15:47:09 2017

@author: frazin
"""
import numpy as np
import matplotlib.pyplot as plt
import Pyr as pr
import time

p = pr.Pyr()
c0 = np.zeros(p.nc)

strehl = 0.3
sig = pr.StrehlToSigma(strehl)
pcount = 1.e10
reg = .0000001

ph = sig*np.random.randn(p.nc)
ph -= np.mean(ph)

I, dI = p.Intensity(ph)
Itot = np.sum(I)
y = I*(pcount/Itot)
noise_sig = np.sqrt(y)
y += noise_sig*np.random.randn(len(y))
y *= (Itot/pcount)
noise_sig *= (Itot/pcount)
wt = None

t0 = time.time()
ls_x, var_x = p.LeastSqSol(y, wt, c0, RegParam=reg, ZeroPhaseMean=True)
ls_cost, dls_cost = p.Cost(ls_x, y, wt, RegParam=reg)
print "lsq std: ", np.std(ls_x - ph), "cost : ", ls_cost
print " time: ",(time.time() - t0)/60., " minutes"

MaxIt = [300]

for maxit in MaxIt:
    t0 = time.time()
    nl_x, res = p.FindMinCost(ls_x, y, wt=None, RegParam=reg, method='BFGS', MaxIt=maxit, LsqStart=False, AmpCon=False)
    print maxit, ": std: ", np.std(nl_x - ph), " time: ",(time.time() - t0)/60., " minutes"

