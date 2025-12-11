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

def np_i_div(ll, rr):
    q = np.abs(ll) // np.abs(rr)
    return np.where((ll < 0) ^ (rr < 0), -q, q)

# Generates N random unsigned intervals
def u_intervals():
    lo = np.random.randint(URAND_MIN, URAND_MAX, size=N, dtype=BIGTY)
    hi = np.random.randint(URAND_MIN, URAND_MAX, size=N, dtype=BIGTY)
    return np.minimum(lo, hi), np.maximum(lo, hi)

# Generates N random signed intervals
def i_intervals():
    lo = np.random.randint(IRAND_MIN, IRAND_MAX, size=N, dtype=BIGTY)
    hi = np.random.randint(IRAND_MIN, IRAND_MAX, size=N, dtype=BIGTY)
    return np.minimum(lo, hi), np.maximum(lo, hi)

# Masks for leading zeros element-wise on xx
def leading_zero_mask(xx):
    bits = np.unpackbits(xx[:, np.newaxis], axis=1)
    cumulative = np.cumsum(bits, axis=1, dtype=np.uint8)
    mask = np.packbits(cumulative == 0)
    return mask

# Generates known zero and known one masks element-wise
def to_masks(ll, hh):
    diff = ll ^ hh
    leading = leading_zero_mask(diff);
    zero_mask = ~ll & leading
    one_mask = ll & leading
    return zero_mask, one_mask
