import numpy as np
from common import *

def u_add(ll_lo, ll_hi, rr_lo, rr_hi):
    lo, lo_over = u_eval(ll_lo, rr_lo, np.add)
    hi, hi_over = u_eval(ll_hi, rr_hi, np.add)

    out = np.zeros([N, 4], dtype=UTYPE)
    out[:, 0] = np.where(lo_over | ~hi_over, lo, UMIN)
    out[:, 1] = np.where(lo_over | ~hi_over, hi, UMAX)
    out[:, 2] = lo_over
    out[:, 3] = lo_over | hi_over
    return out

def u_sub(ll_lo, ll_hi, rr_lo, rr_hi):
    lo, _ = u_eval(ll_lo, rr_hi, np.subtract)
    hi, _ = u_eval(ll_hi, rr_lo, np.subtract)

    out = np.zeros([N, 4], dtype=UTYPE)
    out[:, 0] = np.where((ll_lo >= rr_hi) | (ll_hi < rr_lo), lo, UMIN)
    out[:, 1] = np.where((ll_lo >= rr_hi) | (ll_hi < rr_lo), hi, UMAX)
    out[:, 2] = ll_hi < rr_lo
    out[:, 3] = ll_lo < rr_hi
    return out

def u_mul(ll_lo, ll_hi, rr_lo, rr_hi):
    res1, over1 = u_eval(ll_lo, rr_lo, np.multiply)
    res2, over2 = u_eval(ll_lo, rr_hi, np.multiply)
    res3, over3 = u_eval(ll_hi, rr_lo, np.multiply)
    res4, over4 = u_eval(ll_hi, rr_hi, np.multiply)
    
    any_over = over1 | over2 | over3 | over4
    lo = np.minimum(np.minimum(res1, res2), np.minimum(res3, res4))
    hi = np.maximum(np.maximum(res1, res2), np.maximum(res3, res4))
    
    out = np.zeros([N, 4], dtype=UTYPE)
    out[:, 0] = np.where(any_over, UMIN, lo)
    out[:, 1] = np.where(any_over, UMAX, hi)
    out[:, 2] = np.zeros(N, dtype=bool)
    out[:, 3] = any_over
    return out

def i_add(ll_lo, ll_hi, rr_lo, rr_hi):
    lo, lo_over = i_eval(ll_lo, rr_lo, np.add)
    hi, hi_over = i_eval(ll_hi, rr_hi, np.add)

    out = np.zeros([N, 4], dtype=ITYPE)
    c1 = lo_over & hi_over & (lo <= hi)
    c2 = ~lo_over & ~hi_over
    out[:, 0] = np.where(c1 | c2, lo, IMIN)
    out[:, 1] = np.where(c1 | c2, hi, IMAX)
    out[:, 2] = c1
    out[:, 3] = c1 | ~c2
    return out

def i_sub(ll_lo, ll_hi, rr_lo, rr_hi):
    lo, lo_over = i_eval(ll_lo, rr_hi, np.subtract)
    hi, hi_over = i_eval(ll_hi, rr_lo, np.subtract)

    out = np.zeros([N, 4], dtype=ITYPE)
    c1 = lo_over & hi_over & (lo <= hi)
    c2 = ~lo_over & ~hi_over
    out[:, 0] = np.where(c1 | c2, lo, IMIN)
    out[:, 1] = np.where(c1 | c2, hi, IMAX)
    out[:, 2] = c1
    out[:, 3] = c1 | ~c2
    return out

def i_mul(ll_lo, ll_hi, rr_lo, rr_hi):
    res1, over1 = i_eval(ll_lo, rr_lo, np.multiply)
    res2, over2 = i_eval(ll_lo, rr_hi, np.multiply)
    res3, over3 = i_eval(ll_hi, rr_lo, np.multiply)
    res4, over4 = i_eval(ll_hi, rr_hi, np.multiply)

    any_over = over1 | over2 | over3 | over4
    lo = np.minimum(np.minimum(res1, res2), np.minimum(res3, res4))
    hi = np.maximum(np.maximum(res1, res2), np.maximum(res3, res4))

    out = np.zeros([N, 4], dtype=ITYPE)
    out[:, 0] = np.where(any_over, IMIN, lo)
    out[:, 1] = np.where(any_over, IMAX, hi)
    out[:, 2] = np.zeros(N, dtype=bool)
    out[:, 3] = any_over
    return out
