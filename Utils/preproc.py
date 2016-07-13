import numpy as np



def _eval_sines(n_vals, x_vals ):
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

    return np.dot(sin_coef,sine_grid) + np.dot(cos_coef,cosine_grid)

def reflect_then_cat(potentials):
    """
    Create a mirrored grid by column indices of the input potentials.
    return a 2D numpy array of shape (2*NumRows(potentials) , NumCols(potentials))

    Keyword arguments:
    potentials --   a 2D numpy array with rows corresponding to different potentials
                    and columns corresponding to different x coordinates
    """

    flipped_grid = potentials[::, ::-1]
    return np.concatenate((potentials, flipped_grid), axis = 0)

if __name__ == "__main__":
    # print(_eval_cosines.__name__, ":\n",  _eval_sines.__doc__)
    # print(_eval_cosines.__name__, ":\n",  _eval_cosines.__doc__)
    # print(potential_grid.__name__, ":\n",  potential_grid.__doc__)

    from numpy.random import normal
    N = 3
    ns = np.arange(1,N+1)
    xs = np.linspace(-1,1,5)
    As = normal(size = (4,N))
    Bs = normal(size = (4,N))

    pot = potential_grid(ns,xs, As, Bs)
    print("Some random potentials:\n", pot)
    print("Add on the reflected potentials:\n", reflect_then_cat(pot))
