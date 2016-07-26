[![Binder](http://mybinder.org/badge.svg)](http://mybinder.org:/repo/dhudsmith/quantumml)

# qmnnet
qmnnet uses a neural network to rapidly solve the 1D Schrodinger equation. 

# Structure
qmnnet involves three basic logical steps:

1. Generating an ensemble of random poentials
2. Solving Schrödinger's equation for the eigenvalues and eigenvectors for each potential
3. Train a neural network to predict the eigenvalues for new potentials

The jupyter notebook `qmnnet.ipynb' goes through these steps. Click the Binder badge to run the jupyter notebook in your browser.

