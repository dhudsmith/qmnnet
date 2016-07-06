#!/usr/bin/env python
# encoding: utf-8
"""
eigenvalues.py

Generate an ensemble of potentials to be used in solving
Schrodinger's equaiton.

Created by Hudson Smith on 2016-06-28.
Copyright (c) 2016 ACT. All rights reserved.
"""

import numpy as np
from scipy.integrate import quad
from scipy.interpolate import interp1d
from numpy.random import uniform


def getFourierCoefficients(Vs, Nb):

    L = 0.5
    Nx = Vs.size

    # The points at which the potential was evaluated
    xs = np.linspace(-L, L, Nx)

    # Calculate the value of sine and cosine for all basis states
    # and over the entire x grid
    ns = range(1, Nb + 1)
    trigargs = 2 * np.pi * np.outer(ns, xs)
    cosines = np.cos(trigargs)
    sines = np.sin(trigargs)

    # Calculate the fourier integrand
    a0Integrand = interp1d(xs, Vs / L, kind="cubic")
    anIntegrand = interp1d(xs, Vs * cosines / L, kind="cubic", axis=1)

    bnIntegrand = interp1d(xs, Vs * sines / L, kind="cubic", axis=1)

    # Calculate fourier series coefficients
    a0 = quad(a0Integrand, -L, L)
    ans = []
    for i in range(0, Nb):
        an = quad(anIntegrand[i], -L, L)
        ans.append(an)
    bns = []
    for i in range(0, Nb):
        bn = quad(bnIntegrand[i], -L, L)
        bns.append(bn)

    # Return the requested coefficients
    return [a0, ans, bns]

if __name__ == '__main__':
    Nb = 3
    Vs = np.asarray(uniform(-0.5, 0.5, 10))
    print(Vs)
    [a0, ans, bns] = getFourierCoefficients(Vs, Nb)

    np.savetxt("./VsTest.csv", Vs, delimiter='')
    np.savetxt("./ansTest.csv", ans, delimiter='')
    np.savetxt("./bnsTest.csv", bns, delimiter='')
