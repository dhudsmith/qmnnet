# QuantumML

QuantumML uses machine learning to rapidly solve the 1D Schrodinger equation. 

# Project structure
The project can be broken down into four logical components. Each component has a dedicated jupyter notebook containing the documentation, logic, and visualization. The components are

1. Generating an ensemble of random potential-energy curves. This step is performed in  [potentials.ipynb](jupyter/potentials.ipynb).
2. Solving the Schrodinger equation for the ensemble of potentials generated in the first step. The output of this step is a list of eigenvalues. This step is performed in [eigenvalues.ipynb](jupyter/eigenvalues.ipynb)
3. Training a machine learning model to capture the mapping from an input potential to the output eigenvalues. This step is performed in [learn.ipynb](jupyter/learn.ipynb).
4. Validating the model. This step is performed in [validation.ipynb](jupyter/validation.ipynb).
