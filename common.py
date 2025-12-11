import numpy as np

# CONSTANTS
BIGTY = np.int16    # to track overflow in calculations
UTYPE = np.uint8    # actual type being analyzed (unsigned)
ITYPE = np.int8     # actual type being analyzed (signed)
N = 1000

URAND_MIN = 0       # inclusive
URAND_MAX = 256     # exclusive
IRAND_MIN = -128    # inclusive
IRAND_MAX = 128     # exclusive

UMIN = np.full(N, np.iinfo(UTYPE).min, dtype=UTYPE)
UMAX = np.full(N, np.iinfo(UTYPE).max, dtype=UTYPE)
IMIN = np.full(N, np.iinfo(ITYPE).min, dtype=ITYPE)
IMAX = np.full(N, np.iinfo(ITYPE).max, dtype=ITYPE)

def u_eval(ll, rr, op):
    result = op(ll, rr)
    over = (result < UMIN[0]) | (result > UMAX[0])
    return result.astype(UTYPE), over

def i_eval(ll, rr, op):
    result = op(ll, rr)
    over = (result < IMIN[0]) | (result > IMAX[0])
    return result.astype(ITYPE), over

def u_intervals():
    lo = np.random.randint(URAND_MIN, URAND_MAX, size=N, dtype=BIGTY)
    hi = np.random.randint(URAND_MIN, URAND_MAX, size=N, dtype=BIGTY)
    return np.minimum(lo, hi), np.maximum(lo, hi)

def i_intervals():
    lo = np.random.randint(IRAND_MIN, IRAND_MAX, size=N, dtype=BIGTY)
    hi = np.random.randint(IRAND_MIN, IRAND_MAX, size=N, dtype=BIGTY)
    return np.minimum(lo, hi), np.maximum(lo, hi)
