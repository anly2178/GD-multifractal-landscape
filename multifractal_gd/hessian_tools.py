import numpy as np


def hessian(x):
    """Calculate the hessian matrix with finite differences
    Args:
        x (ndarray) : N dimensional array.

    Returns:
        ndarray: An array with dimensions (d, d, x.shape) where d is the dimensionality of x.
            E.g., array[i, j, x1, ..., xd] is the second derivative in directions i,j at (x1,...,xd).
    """
    x_grad = np.gradient(x)
    hessian = np.empty((x.ndim, x.ndim) + x.shape, dtype=x.dtype)
    for k, grad_k in enumerate(x_grad):
        tmp_grad = np.gradient(grad_k)
        for l, grad_kl in enumerate(tmp_grad):
            hessian[k, l, :, :] = grad_kl
    return hessian


def diagonalise_hessian(H):
    """Diagonalises the Hessian eigenvalues at each position.

    Args:
        H (ndarray): Hessian output from hessian() function, which has shape (d, d, x.shape),
            where x is the landscape.

    Returns:
        ndarray: Diagonalised Hessian replaces Hessian at each position. 
    """
    H_tmp = H.reshape(H.shape[0], H.shape[1], -1)
    npts = H_tmp.shape[2]
    eigenvalues = np.zeros(shape=(1, H.shape[0], npts))
    for i in range(npts):
        eigval = np.linalg.eigvalsh(H_tmp[:, :, i])
        eigenvalues[:, :, i] = eigval
    eig_shape = [1, H.shape[0]] + [h for h in H.shape[2:]]
    eigenvalues = eigenvalues.reshape(eig_shape)
    return eigenvalues
