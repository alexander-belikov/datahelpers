# cython: boundscheck=False, wraparound=False

from cython cimport view
# from numpy.math cimport expl, logl, isinf, INFINITY, factorial
from numpy.math cimport expl, logl, isinf, INFINITY
from math import factorial


# ctypedef double dtype_t


# cdef inline int _argmax(dtype_t[:] X) nogil:
#     cdef dtype_t X_max = -INFINITY
#     cdef int pos = 0
#     cdef int i
#     for i in range(X.shape[0]):
#         if X[i] > X_max:
#             X_max = X[i]
#             pos = i
#     return pos


# cdef inline dtype_t _logsumexp(dtype_t[:] X) nogil:
#     cdef dtype_t X_max = _max(X)
#     if isinf(X_max):
#         return -INFINITY

#     cdef dtype_t acc = 0
#     for i in range(X.shape[0]):
#         acc += expl(X[i] - X_max)

#     return logl(acc) + X_max

cpdef double f(int k):
    cdef double ret = factorial(k)
    return ret

cpdef double l(double x):
    cdef double ret = logl(x)
    return ret

# cpdef double poi(double _lambda, int k):

#     return (-_lambda + k*logl(_lambda) - logl(factorial(k)))

