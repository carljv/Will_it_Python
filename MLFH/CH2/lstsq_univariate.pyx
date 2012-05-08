from libc.math import fmax
import numpy as np
cimport numpy as np
cimport cython


DTYPE = np.double
ctypedef np.double_t DTYPE_t

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def lstsq_univariate(np.ndarray[DTYPE_t, ndim = 2] exog, 
                     np.ndarray[DTYPE_t, ndim = 1] endog):
    """
    A simple function for computing the least squares solutions for univariate 
    linear regression problem. For use with lowess.

    Parameters:
    -----------
    exog:  an Nx2 array of exogenous (RHS) variables. The first column is the
           the "constant", the second the `x` variable. The first column is not
           assumed to be all 1s, to accommodate weighted regressions.
    endog: an Nx1 array of the exogenous (LHS) variable.


    Returns:
    --------
    pars: a 2x1 array of the intercept and slope parameters (a, b).
    """
    cdef:
        np.ndarray[DTYPE_t, ndim = 1] x0, x1
        np.ndarray[DTYPE_t, ndim = 1] pars = np.empty(2, dtype = DTYPE)
        double ss_x0, ss_x1, ss_x0x1, ss_x0y, ss_x1y
        Py_ssize_t i, 
        Py_ssize_t n = exog.shape[0]

    x0 = exog[:, 0]
    x1 = exog[:, 1]
    
    ss_x0   = 0.0
    ss_x1   = 0.0 
    ss_x0x1 = 0.0
    ss_x0y  = 0.0
    ss_x1y  = 0.0

    for i in xrange(n):
        ss_x0   += x0[i] * x0[i]
        ss_x1   += x1[i] * x1[i]
        ss_x0x1 += x0[i] * x1[i]
        ss_x0y  += x0[i] * endog[i]
        ss_x1y  += x1[i] * endog[i]
        
    det = ss_x0 * ss_x1 - ss_x0x1 * ss_x0x1
    
    pars[0] = ss_x1 * ss_x0y - ss_x0x1 * ss_x1y
    pars[0] /= det
    pars[1] = ss_x0 * ss_x1y - ss_x0x1 * ss_x0y
    pars[1] /= det
   
    return pars


def _lowess_initial_fit(np.ndarray[DTYPE_t, ndim = 1] x_copy, 
                        np.ndarray[DTYPE_t, ndim = 1] y_copy,
                        Py_ssize_t k, Py_ssize_t n):
    cdef:
        np.ndarray[DTYPE_t, ndim = 2] weights = np.zeros((n, k), dtype = DTYPE)
        Py_ssize_t left_end = 0
        Py_ssize_t right_end = k
        np.ndarray[DTYPE_t, ndim = 2] X = np.ones((k, 2), dtype = DTYPE)
        np.ndarray[DTYPE_t, ndim = 2] fitted = np.zeros(n, dtype = DTYPE)
        double left_width, right_width
        
    for i in xrange(n):
        left_width = x_copy[i] - x_copy[left_end]
        right_width = x_copy[right_end] - x_copy[i]
        width = fmax(left_width, right_width)
        _lowess_wt_stdandardize(




def _lowess_wt_standardize(weights, new_entries, x_copy_i, width):
    