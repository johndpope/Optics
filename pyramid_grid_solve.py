#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 15:47:09 2017

@author: frazin
"""
import numpy as np
import matplotlib.pyplot as plt
import time
from multiprocessing import Pool
import cPickle as pickle
import Pyr as pyr

print "initialzing..."
p = pyr.Pyr()
c0 = np.zeros(p.nc)
print "done"

strehl = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
pcount = [1.e4, 1.e5, 1.e7, 1.e9]
dnoise = [0., 1., 5.]
regp = [1.e-8, 1.e-7, 1.e-6, 1.e-5, 1.e-4, 1.e-3, 0.01, 0.1, 1.]
ntrials = 12
nl_maxit = 1000

exm = []  # list of experiments
for st in strehl:
    for pc in pcount:
        for dn in dnoise:
            grid = dict()
            grid['sigma'] = pyr.StrehlToSigma(st)
            grid['strehl'] = st
            grid['pcount'] = pc
            grid['det_noise'] = dn
            grid['reg_param'] = regp
            grid['lin_score'] = 0*np.array(regp)
            grid['lin_cost'] = 0*np.array(regp)
            grid['best_ls_cost'] = 0.
            grid['best_ls_score'] = 0.
            grid['nl_score'] = 0.  # nl stuff only done for best reg_param
            grid['nl_cost'] = 0
            grid['nl_result'] = None  # only holds result from final trial
            exm.append(grid)

I0, dI0 = p.Intensity(c0)
Itot = np.sum(I0)

t0 = time.time()


def single_gridpoint(index):
    assert index < len(exm), "bad index value"
    t0 = time.time()
    ex = exm[index]
    sigma = ex['sigma']
    pcount = ex['pcount']
    d_noise = ex['det_noise']
    print index, " of ", len(exm)
    for kr in range(len(ex['reg_param'])):
        reg = ex['reg_param'][kr]
        for tr in range(ntrials):
            ph = sigma*np.random.randn(p.nc)
            ph -= np.mean(ph)
            I, dI = p.Intensity(ph)
            y = I*(pcount/Itot)
            noise_sig = np.sqrt(y) + d_noise
            y += noise_sig*np.random.randn(len(y))
            y *= (Itot/pcount)
            noise_sig *= (Itot/pcount)
            wt = None  # for ML set to 1/noise_sig^2 --> precision issues
            ls_x, var_x = p.LeastSqSol(y, wt, c0, RegParam=reg, ZeroPhaseMean=True)
            ls_cost, dls_cost = p.Cost(ls_x, y, wt, RegParam=reg)
            score = np.std(ls_x - ph)
            ex['lin_score'][kr] += score/ntrials
            ex['lin_cost'][kr] += ls_cost/ntrials
    kr = np.argmin(ex['lin_score'])  # get best reg param
    reg = ex['reg_param'][kr]
    for tr in range(ntrials):
        ph = sigma*np.random.randn(p.nc)
        ph -= np.mean(ph)
        I, dI = p.Intensity(ph)
        y = I*(pcount/Itot)
        noise_sig = np.sqrt(y) + d_noise
        y += noise_sig*np.random.randn(len(y))
        y *= (Itot/pcount)
        noise_sig *= (Itot/pcount)
        wt = None  # for ML set to 1/noise_sig^2 --> precision issues
        ls_x, var_x = p.LeastSqSol(y, wt, c0, RegParam=reg, ZeroPhaseMean=True)
        ls_cost, dls_cost = p.Cost(ls_x, y, wt, RegParam=reg)
        ex['best_ls_cost'] += ls_cost/ntrials
        ex['best_ls_score'] += np.std(ls_x - ph)/ntrials
        nl_x, res = p.FindMinCost(ls_x, y, wt=None, RegParam=reg, method='BFGS', MaxIt=nl_maxit, LsqStart=False, AmpCon=False)
        ex['nl_cost'] += res.fun/ntrials
        ex['nl_score'] += np.std(nl_x - ph)/ntrials
        ex['nl_result'] = res
    exm[index] = ex  # ex is a local variable, so place in the global exm
    return((time.time() - t0)/60.)


pool = Pool(40)
pool.map(single_gridpoint, range(len(exm)))

with open('grid_results.p', 'w') as fp:
    pickle.dump(exm, fp)