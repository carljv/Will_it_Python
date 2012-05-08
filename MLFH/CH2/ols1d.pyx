import numpy as np
cimport numpy as np
cimport cython


DTYPE = np.double
ctypedef np.double_t DTYPE_t

@cython.boundscheck(False)
@cython.cdivision(True)
def ols1d2(np.ndarray[DTYPE_t, ndim = 2] exog, np.ndarray[DTYPE_t, ndim = 1] endog):
    cdef np.ndarray[DTYPE_t, ndim = 1] x0, x1
    cdef np.ndarray[DTYPE_t, ndim = 1] pars = np.empty(2, dtype = DTYPE)
    cdef double ss_x0, ss_x1, ss_x0x1, ss_x0y, ss_x1y
    cdef int n = exog.shape[0]

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
            #  ss_x0 * ss_x1y - ss_x0x1 * ss_x0y
    pars[1] /= det
   
    return pars
