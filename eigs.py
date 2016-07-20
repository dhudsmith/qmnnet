import numpy as np
import time

# --------------- Parameters ---------------
# Number of basis states in wavefunction:
NBW = 40
nbws = np.arange(1, NBW + 1)
# Number of potentials:
NV = int(1E6)
# Number of basis states in the potential:
NB = 10
ns = np.arange(1, NB + 1)
# lambda (variance of Legendre coefficients):
lam = 0.75
# The variance of the n=0 legendre coefficient V_0:
V20 = 10
# Batch size for computing eigenvalues
bs = int(1E5)

# ########## Input file ###########
filepath = "Data/potentials_NV" + str(NV) \
           + "_NB" + str(NB) + "_lam" \
           + str(lam) + "_V20" + str(V20) + ".npy"
VSns, VCns = np.load(filepath)


# --------------- matrix elements ---------------

# analytic expressions for the matrix elements:
def matelS(indices):
    n, m, k = indices
    if (m == n - 2 * k or m == n + 2 * k or m == -n + 2 * k):
        return 0
    else:
        return (-8 * (-1) ** k * (-1 + (-1) ** (m + n)) * k * m * n) / \
               ((2 * k + m - n) * (2 * k - m + n) * (-2 * k + m + n) *
                (2 * k + m + n) * np.pi)


def matelC(indices):
    n, m, k = indices
    if m == n - 2 * k or m == n + 2 * k:
        return (-1) ** k / 2.
    elif m == -n + 2 * k:
        return -(-1) ** k / 2.
    else:
        return 0.

# Precache all the matrix elements
VSmnk = np.zeros((NBW, NBW, NB))
for n in range(1, NBW + 1):
    for m in range(1, NBW + 1):
        for k in range(1, NB):
            VSmnk[n - 1, m - 1, k - 1] = matelS([n, m, k])

VCmnk = np.zeros((NBW, NBW, NB))
for n in range(1, NBW + 1):
    for m in range(1, NBW + 1):
        for k in range(1, NB + 1):
            VCmnk[n - 1, m - 1, k - 1] = matelC([n, m, k])


# --------------- Hamiltonian matrix elements ---------------
def E0(n):
    return n ** 2 * np.pi ** 2 / 8.


E0ns = np.diag(E0(nbws))


def hamils(start_index, n_batch):
    # Be aware: for tensors, np.dot(A,B) sums over the last index of A
    # and the SECOND-TO-LAST index of B. It appears that you cannot change this

    ind = range(start_index, start_index + n_batch)
    Vmns = np.dot(VSns[ind], np.transpose(VSmnk, axes=(0, 2, 1)))\
        + np.dot(VCns[ind], np.transpose(VCmnk, axes=(0, 2, 1)))

    return E0ns + Vmns

# We define a generator function to serve batches of hamiltonians
# We go to this effort because for very large datasets, we cannot
# store all the hamiltonians in memory simultaneously.


def batches(n_total, n_batch):
    for batch_start_index in range(0, n_total, n_batch):
        yield hamils(batch_start_index, n_batch)


# --------------- Eigenvalues ---------------
totalstart = time.time()
batchno = 0
eigs = []
for batch in batches(NV, bs):
    batchno += 1
    start = time.time()
    eigvals, _ = np.linalg.eig(batch)
    eigvals = np.sort(eigvals, axis=1)
    eigs.append(eigvals)
    end = time.time()

    print("Batch %i completed in %f seconds." % (batchno, end - start))
    print("Progress: %.2f %% completed." % (batchno * bs / NV * 100.))

totalstop = time.time()

eigvals = np.concatenate(eigs)
print("Eigenvalue calculation completed in %f seconds." %
      (totalstop - totalstart))

# --------------- Delta epsilon  ---------------
# We now examine the distribution of the eigenvalues.
# In particular, we plot the distributions of
# $$
# \Delta\epsilon_n = \frac{E_n - E_n^0}{\sqrt{\langle V_0^2 \rangle}}
# $$
# This is the shift in the energy for the $n^{\rm th}$
# eigenvalue scaled by the variance in the first Fourier coefficient.

deltaeps = (eigvals - np.diag(E0ns)) / np.sqrt(V20)
deltaeps_mu = np.mean(deltaeps, axis=0)
deltaeps_std = np.std(deltaeps, axis=0)

print("Mean shift in eigenvals:\n", deltaeps_mu)
print("Standard deviation of shift in eigenvals:\n", deltaeps_std)

# Shift the eigenvalues to have zero mean and unit standard deviation
deltaeps_scaled = (deltaeps - deltaeps_mu) / deltaeps_std
print("Mean shift in scaled eigenvals:\n", np.mean(deltaeps_scaled, axis=0))
print("Standard deviation of shift in scaled eigenvals:\n",
      np.std(deltaeps_scaled, axis=0))

# --------------- Output ---------------
# Output the shifted and scaled eigenvalues along with
# the mean and std. dev. so that we can reproduce the original values

# Output file:
outfilepath = "Data/eigenvalues_NV" + str(NV) \
    + "_NB" + str(NB) + "_lam" \
    + str(lam) + "_V20" + str(V20) + ".npy"
outfilepathScalings = "Data/eigenvaluesScalings_NV" + str(NV) \
    + "_NB" + str(NB) + "_lam" \
    + str(lam) + "_V20" + str(V20) + ".npy"

outData = np.concatenate((VSns, VCns, deltaeps_scaled), axis=1)
scalings = np.asarray((deltaeps_mu, deltaeps_std))

np.save(outfilepath, outData)
np.save(outfilepathScalings, scalings)

print("Scaled eigenvalues written to:\n     %s" % outfilepath)
print("Mean and std. dev. written to:\n     %s" % outfilepathScalings)
