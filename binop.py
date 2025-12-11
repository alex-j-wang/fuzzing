import numpy as np
from common import *

# UNSIGNED OPERATIONS
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
    out[:, 2] = np.False_
    out[:, 3] = any_over
    return out

def u_div(ll_lo, ll_hi, rr_lo, rr_hi):
    res1, _ = u_eval(ll_lo, rr_lo, np.floor_divide)
    res2, _ = u_eval(ll_lo, rr_hi, np.floor_divide)
    res3, _ = u_eval(ll_hi, rr_lo, np.floor_divide)
    res4, _ = u_eval(ll_hi, rr_hi, np.floor_divide)

    lo = np.minimum(np.minimum(res1, res2), np.minimum(res3, res4))
    hi = np.maximum(np.maximum(res1, res2), np.maximum(res3, res4))

    out = np.zeros([N, 4], dtype=UTYPE)
    out[:, 0] = np.where(rr_lo == 0, UMIN, lo)
    out[:, 1] = np.where(rr_lo == 0, UMAX, hi)
    return out

def u_rem(ll_lo, ll_hi, rr_lo, rr_hi):
    bound = UTYPE(rr_hi - 1)
    out = np.zeros([N, 4], dtype=UTYPE)
    out[:, 0] = UMIN
    out[:, 1] = np.where(rr_hi == 0, UMAX, bound)
    return out

def u_and(ll_lo, ll_hi, rr_lo, rr_hi):
    ll_lo = ll_lo.astype(UTYPE)
    ll_hi = ll_hi.astype(UTYPE)
    rr_lo = rr_lo.astype(UTYPE)
    rr_hi = rr_hi.astype(UTYPE)

    ll_zero, ll_one = to_masks(ll_lo, ll_hi)
    rr_zero, rr_one = to_masks(rr_lo, rr_hi)

    one = ll_one & rr_one
    zero = ll_zero | rr_zero

    out = np.zeros([N, 4], dtype=UTYPE)
    out[:, 0] = one
    out[:, 1] = one | ~zero
    return out

def u_or(ll_lo, ll_hi, rr_lo, rr_hi):
    ll_lo = ll_lo.astype(UTYPE)
    ll_hi = ll_hi.astype(UTYPE)
    rr_lo = rr_lo.astype(UTYPE)
    rr_hi = rr_hi.astype(UTYPE)

    ll_zero, ll_one = to_masks(ll_lo, ll_hi)
    rr_zero, rr_one = to_masks(rr_lo, rr_hi)

    one = ll_one | rr_one
    zero = ll_zero & rr_zero

    out = np.zeros([N, 4], dtype=UTYPE)
    out[:, 0] = one
    out[:, 1] = one | ~zero
    return out

def u_xor(ll_lo, ll_hi, rr_lo, rr_hi):
    ll_lo = ll_lo.astype(UTYPE)
    ll_hi = ll_hi.astype(UTYPE)
    rr_lo = rr_lo.astype(UTYPE)
    rr_hi = rr_hi.astype(UTYPE)

    ll_zero, ll_one = to_masks(ll_lo, ll_hi)
    rr_zero, rr_one = to_masks(rr_lo, rr_hi)

    one = (ll_one & rr_zero) | (ll_zero & rr_one)
    zero = (ll_zero & rr_zero) | (ll_one & rr_one)

    out = np.zeros([N, 4], dtype=UTYPE)
    out[:, 0] = one
    out[:, 1] = one | ~zero
    return out

# SIGNED OPERATIONS
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
    out[:, 2] = np.False_
    out[:, 3] = any_over
    return out

def i_div(ll_lo, ll_hi, rr_lo, rr_hi):
    res1, _ = i_eval(ll_lo, rr_lo, np_i_div)
    res2, _ = i_eval(ll_lo, rr_hi, np_i_div)
    res3, _ = i_eval(ll_hi, rr_lo, np_i_div)
    res4, _ = i_eval(ll_hi, rr_hi, np_i_div)

    zero_mask = (rr_lo <= 0) & (rr_hi >= 0)
    over_mask = (ll_lo == IMIN) & (rr_hi == -1)
    lo = np.minimum(np.minimum(res1, res2), np.minimum(res3, res4))
    hi = np.maximum(np.maximum(res1, res2), np.maximum(res3, res4))

    out = np.zeros([N, 4], dtype=ITYPE)
    out[:, 0] = np.where(zero_mask | over_mask, IMIN, lo)
    out[:, 1] = np.where(zero_mask | over_mask, IMAX, hi)
    out[:, 2] = np.False_
    out[:, 3] = zero_mask | over_mask # not actually important
    return out

def i_rem(ll_lo, ll_hi, rr_lo, rr_hi):
    top_mask = ((rr_lo == 0) & (rr_hi == 0)) | (rr_lo == IMIN)
    zero = ITYPE(0)
    bound = np.maximum(np.abs(rr_lo), np.abs(rr_hi)) - 1
    min_bound = (-bound).astype(ITYPE)
    max_bound = bound.astype(ITYPE)
    
    lo = np.where(ll_hi < 0, min_bound, np.where(ll_lo >= 0, zero, min_bound))
    hi = np.where(ll_hi < 0, zero, max_bound)

    out = np.zeros([N, 4], dtype=ITYPE)
    out[:, 0] = np.where(top_mask, IMIN, lo)
    out[:, 1] = np.where(top_mask, IMAX, hi)
    return out

# COMPARISON OPERATORS
def eq(ll_lo, ll_hi, rr_lo, rr_hi):
    singleton_mask = (ll_lo == ll_hi) & (rr_lo == rr_hi) & (ll_lo == rr_lo)
    disjoint_mask = (ll_hi < rr_lo) | (ll_lo > rr_hi)
    
    out = np.zeros([N, 4], dtype=UTYPE)
    out[:, 0] = singleton_mask
    out[:, 1] = singleton_mask | ~disjoint_mask
    return out

def ne(ll_lo, ll_hi, rr_lo, rr_hi):
    singleton_mask = (ll_lo == ll_hi) & (rr_lo == rr_hi) & (ll_lo == rr_lo)
    disjoint_mask = (ll_hi < rr_lo) | (ll_lo > rr_hi)
    
    out = np.zeros([N, 4], dtype=UTYPE)
    out[:, 0] = ~singleton_mask & disjoint_mask
    out[:, 1] = ~singleton_mask
    return out

def lt(ll_lo, ll_hi, rr_lo, rr_hi):
    lt_mask = ll_hi < rr_lo
    ge_mask = ll_lo >= rr_hi

    out = np.zeros([N, 4], dtype=UTYPE)
    out[:, 0] = lt_mask
    out[:, 1] = lt_mask | ~ge_mask
    return out

def le(ll_lo, ll_hi, rr_lo, rr_hi):
    le_mask = ll_hi <= rr_lo
    gt_mask = ll_lo > rr_hi

    out = np.zeros([N, 4], dtype=UTYPE)
    out[:, 0] = le_mask
    out[:, 1] = le_mask | ~gt_mask
    return out

def gt(ll_lo, ll_hi, rr_lo, rr_hi):
    gt_mask = ll_lo > rr_hi
    le_mask = ll_hi <= rr_lo

    out = np.zeros([N, 4], dtype=UTYPE)
    out[:, 0] = gt_mask
    out[:, 1] = gt_mask | ~le_mask
    return out

def ge(ll_lo, ll_hi, rr_lo, rr_hi):
    ge_mask = ll_lo >= rr_hi
    lt_mask = ll_hi < rr_lo

    out = np.zeros([N, 4], dtype=UTYPE)
    out[:, 0] = ge_mask
    out[:, 1] = ge_mask | ~lt_mask
    return out
