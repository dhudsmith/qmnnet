import numpy as np
from scipy.integrate import simps

def psi0(n, x):
    '''
    Evaluate the nth box state at x
    :param n: array-like, 1-indexed state labels
    :param x: array-like, positions between -1 and 1
    :return: an array of shape (len(n), len(x))
    '''
    return np.sin(np.pi/2*np.outer(n,x+1))

def E0(n):
    '''
    The nth energy level in the box
    :param n: the state label
    :return: the energy
    '''
    return n**2 * np.pi**2 / 8.

def _matel_integrand(m,n,x, V):
    '''
    The n,m matrix element of the V potential evaluated at the x coordinates
    :param n:   the row index
    :param m:   the column index
    :param x:   array-like, a vector of x coordinates
    :param V:   array-like, an array of potential values. The rows correspond to
                the entries in x. The columns correspond to different potentials
    :return:    the integrand of the matrix element
    '''
    return psi0(m+1,x) * psi0(n+1,x) * V

def Vmn(m, n, x_vec, V_vec):
    return simps(x=x_vec, y=_matel_integrand(m,n,x_vec,V_vec), axis=1)

def H(n_basis, x_vec, V_vec):
    print("Calculating Hamiltonian matrices...")
    h = np.zeros((V_vec.shape[0],n_basis, n_basis))
    for m in range(n_basis):
        for n in range(m+1):
            h[:,m,n]=Vmn(m,n,x_vec,V_vec)

            # Print a status
            percent = (n * n_basis + m + 1) / n_basis ** 2 * 100
            print("\rStatus: %0.2f %% complete" % percent, end='')
    return h + np.diag(E0( np.arange(1, n_basis+1) ))

def eigsys(n_basis, x_vec, V_vec):
    return np.linalg.eigh(H(n_basis, x_vec, V_vec), UPLO='L')

def psi(evecs, x_vec):
    basis_vec = np.arange(1, evecs.shape[-1] + 1)
    return np.dot(evecs.transpose((0, 2, 1)), psi0(basis_vec, x_vec))

def prob_dist(evecs, x_vec):
    return psi(evecs, x_vec)**2

if __name__ == "__main__":
    x_vec = np.linspace(-1,1,200)
    V_vec = np.asarray((10*np.sin(10*x_vec)*np.exp(-1*x_vec),np.sin(3*x_vec)))
    V_vec[1]+=1

    evals, evecs = eigsys(20, x_vec, V_vec)

    probs = prob_dist(evecs, x_vec)

    import matplotlib.pyplot as plt
    plt.clf()
    for i in range(5):
        plt.plot(x_vec, V_vec[0], 'k-', lw=2)
        plt.plot(x_vec, evals[0,i]+2*probs[0,i], 'r-', lw=2)
        plt.axhline(evals[0, i], -1, 1, color='k', ls='--', lw=2)
    plt.show()



