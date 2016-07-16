# QuantumML

QuantumML uses machine learning to rapidly solve the 1D Schrodinger equation. 

# Project structure
The project can be broken down into four logical components. Each component is prototyped in a dedicated jupyter notebook containing the documentation, logic, and visualization. These notebooks and their output can be rendered directly in github. To check them out head on over to `/jupyter`. The project components are

1. [potentials.ipynb](jupyter/potentials.ipynb): generate an ensemble of quantum mechanical potentials to use as the inputs for training the neural network (NN).
2. [eigenvalues.ipynb](jupyter/eigenvalues.ipynb): solve Schrodinger's equation for each potential in the ensemble. The NN can be trained to predict either the eigenvalues, the (TODO:) probability distributions, or both.
3. [learn_sklearn.ipynb](jupyter/learn_sklearn.ipynb): fit a neural network model to the potentials and eigenvalues. This notebook focuses on model selection. (See also [learn_sklearn.ipynb](jupyter/learn_sklearn.ipynb).)
4. TODO: validation
