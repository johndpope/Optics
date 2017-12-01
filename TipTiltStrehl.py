#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 13:55:23 2017

@author: frazin
"""

import numpy as np
import matplotlib.pyplot as plt
import Pyr

def SigmaToStrehl(sigma):
    strehl = np.exp(-sigma*sigma)
    return(strehl)

def StrehlToSigma(strehl):
    sigma = np.sqrt(- np.log(strehl))
    return(sigma)

# amp is the amplitude of the tilt in lambda/D
def AddTilt(phase, amp=0.25, npup=129):
    x = np.linspace(-1, 1, npup)
    return(phase + amp*x)

npup = 129
p = Pyr.Pyr(NDim=1, npup=npup, npad=4096)

strehl = np.linspace(.2, .5, 10)
tip = np.linspace(0, 5., 10)
ntrials = 1000

I0, dI0 = p.Intensity(np.zeros(npup))
stat0 = np.zeros((len(strehl), len(tip)))
stat1 = np.zeros((len(strehl), len(tip)))

for m in range(len(strehl)):
    for k in range(len(tip)):
        for tr in range(ntrials):
            ph = StrehlToSigma(strehl[m])*np.random.randn(npup)
            phk = AddTilt(ph, tip[k], npup)
            I, dI = p.Intensity(phk)
            int0 = I[0:npup]
            int1 = I[npup:]
            s0 = np.sum(int0)
            s1 = np.sum(int1)
            stat0[m, k] += np.var(np.sqrt(int0) + np.sqrt(int1))/ntrials  
            stat1[m, k] += s0/s1/ntrials


