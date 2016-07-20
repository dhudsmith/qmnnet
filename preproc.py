import numpy as np

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
# Number of x grid points
Nx = 100
xs = np.linspace(-1, 1, Nx)
# Number of eigenvalues to keep
n_eig = 10

# --------------- Input file ---------------
filepath = "Data/eigenvalues_NV" + str(NV) \
           + "_NB" + str(NB) + "_lam" \
           + str(lam) + "_V20" + str(V20) + ".npy"
data = np.load(filepath)
VSns = data[:, 0:NB]
VCns = data[:, NB:2 * NB]
eigs = data[:, 2 * NB:2 * NB + n_eig]

# --------------- Output file ---------------
outpath = "Data/datagrid_NV" + str(NV) \
    + "_NB" + str(NB) + "_lam" \
    + str(lam) + "_V20" + str(V20) + ".npy"

# --------------- Convert to grid ---------------


def _eval_sines(n_vals, x_vals):
    """
    Evaluate sin(n * pi * x)
    return a 2D numpy array of shape (Length(n_vals), Length(x_vals))

    Keyword arguments:
    n_vals -- a 1D numpy array of successive Fourier mode integers
    x_vals -- a 1D numpy array of successive x coordinates
    """
    return np.sin(np.pi * np.outer(n_vals, x_vals))


def _eval_cosines(n_vals, x_vals):
    """
    Evaluate cos(n * pi * x)
    return a 2D numpy array of shape (Length(n_vals), Length(x_vals))

    Keyword arguments:
    n_vals -- a 1D numpy array of successive Fourier mode integers
    x_vals -- a 1D numpy array of successive x coordinates
    """
    return np.cos(np.pi * np.outer(n_vals, x_vals))


def potential_grid(n_vals, x_vals, sin_coef, cos_coef):
    """
    Calculate the potential with the input Fourier components at the given x coordinates
    return a 2D numpy array of shape (NumRows(sine_coef) x Length(x_vals))

    Keyword arguments:
    n_vals -- a 1D numpy array of successive Fourier mode integers
    x_vals -- a 1D numpy array of successive x coordinates
    sin_coef -- a 2D numpy array containing the coefficients of the Sine functions in the Fourier sum
                The rows correspond to different potentials, the columns to the different terms in the Fourier sum.
                The number of rows must match cos_coef.
    cos_coef -- a 2D numpy array containing the coefficients of the Cosine functions in the Fourier sum
                The rows correspond to different potentials, the columns to the different terms in the Fourier sum.
                The number of rows must match cos_coef.
    """
    sine_grid = _eval_sines(n_vals, x_vals)
    cosine_grid = _eval_cosines(n_vals, x_vals)

    return np.dot(sin_coef, sine_grid) + np.dot(cos_coef, cosine_grid)


def reflect_then_cat(potentials):
    """
    Create a mirrored grid by column indices of the input potentials.
    return a 2D numpy array of shape (2*NumRows(potentials) , NumCols(potentials))

    Keyword arguments:
    potentials --   a 2D numpy array with rows corresponding to different potentials
                    and columns corresponding to different x coordinates
    """

    flipped_grid = potentials[::, ::-1]
    return np.concatenate((potentials, flipped_grid), axis=0)

if __name__ == "__main__":
    Vgrid = potential_grid(ns, xs, VSns, VCns)
    X = reflect_then_cat(Vgrid)
    y = np.concatenate((eigs, eigs))

    outdata = np.concatenate((X, y), axis=1)

    np.save(outpath, outdata)
