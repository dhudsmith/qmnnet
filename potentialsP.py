#!/usr/bin/env python
# encoding: utf-8
"""
generatePotentials.py

Generate an ensemble of potentials to be used in solving
Schrodinger's equaiton.

Created by Hudson Smith on 2016-06-28.
Copyright (c) 2016 ACT. All rights reserved.
"""

import numpy as np
from numpy.random import normal

V02 = 1
nmax = 20
lam = 5.0 / nmax

ns = np.arange(0, nmax + 1)


def modeVariance(n, lam, V02):
    return V02 * np.exp(-lam * n)
np.vectorize(modeVariance)

Vn2s = modeVariance(ns, lam)

Vns = normal(0, np.sqrt(Vn2s))

print(Vns)
