#!/usr/bin/env python
# encoding: utf-8
"""
generatePotentials.py

Generate an ensemble of potentials to be used in solving
Schrodinger's equaiton.

Created by Hudson Smith on 2016-06-28.
Copyright (c) 2016 ACT. All rights reserved.
"""

import sys
from numpy.random import uniform
from numpy import save

# Get the inputs (2 total)
# 1.) Number of potentials to create
# 2.) Number of x pts to generate (evenly spaced from -1/2 to 1/2)
# 3.) The output file full path
nargs = 3
if len(sys.argv) != nargs + 1:
    print("Usage: python generatePotentials.py"
          " <Number of potentials> <Number of x points>")
    sys.exit()

numPotentials = int(sys.argv[1])
numxPts = int(sys.argv[2])
filepath = str(sys.argv[3])

potentials = uniform(-0.5, 0.5, (numPotentials, numxPts))

save(filepath, potentials)
