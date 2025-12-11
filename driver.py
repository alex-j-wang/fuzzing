import pandas as pd
import numpy as np
from itertools import product

from common import *
import binop

# OPERATOR LIST
# (numpy_operator, range_operator)
u_ops = [
    (np.add, binop.u_add),
    (np.subtract, binop.u_sub),
    (np.multiply, binop.u_mul),
    (np.floor_divide, binop.u_div),
    (np.mod, binop.u_rem),
    (np.bitwise_and, binop.u_and),
    (np.bitwise_or, binop.u_or),
    (np.bitwise_xor, binop.u_xor),
    (np.equal, binop.eq),
    (np.not_equal, binop.ne),
    (np.less, binop.lt),
    (np.less_equal, binop.le),
    (np.greater, binop.gt),
    (np.greater_equal, binop.ge),
]

# (numpy_operator, range_operator)
i_ops = [
    (np.add, binop.i_add),
    (np.subtract, binop.i_sub),
    (np.multiply, binop.i_mul),
    (np_i_div, binop.i_div),
    (np.fmod, binop.i_rem),
    (np.equal, binop.eq),
    (np.not_equal, binop.ne),
    (np.less, binop.lt),
    (np.less_equal, binop.le),
    (np.greater, binop.gt),
    (np.greater_equal, binop.ge),
]

# DATAFRAME SETUP
operations = []
for entry in u_ops:
    operations.extend(["U8::" + entry[0].__name__.upper()] * N)
for entry in i_ops:
    operations.extend(["I8::" + entry[0].__name__.upper()] * N)

inputs = list(product(['Left', 'Right'], ['Low', 'High']))
approx_outputs = list(product(['Approx Range', 'Approx Over'], ['Low', 'High']))
true_outputs = list(product(['True Range', 'True Over'], ['Low', 'High']))
columns = pd.MultiIndex.from_tuples(
    [('Operation',)] + inputs + true_outputs[:2] + approx_outputs[:2] + true_outputs[2:] + approx_outputs[2:])

df = pd.DataFrame(columns=columns)
df.Operation = operations
idx = 0

# UNSIGNED TESTING
ll_lo, ll_hi = u_intervals()
rr_lo, rr_hi = u_intervals()
cases = np.stack((ll_lo, ll_hi, rr_lo, rr_hi), axis=1)
df.loc[idx:idx+N*len(u_ops)-1, inputs] = np.tile(cases, [len(u_ops), 1])

unsigned = np.arange(UMIN, int(UMAX) + 1, dtype=BIGTY)
left, right = np.meshgrid(unsigned, unsigned)
l_lo, l_hi, r_lo, r_hi = ll_lo, ll_hi, rr_lo, rr_hi
mat_idx = unsigned.astype(UTYPE)

for numpy_op, range_op in u_ops:
    df.loc[idx:idx+N-1, approx_outputs] = range_op(ll_lo, ll_hi, rr_lo, rr_hi)
    result, over = u_eval(left, right, numpy_op)
    
    row_mask = (mat_idx >= r_lo[:, None]) & (mat_idx <= r_hi[:, None])
    col_mask = (mat_idx >= l_lo[:, None]) & (mat_idx <= l_hi[:, None])
    mask = row_mask[:, :, None] & col_mask[:, None, :]

    out_lo = np.where(mask, result[None, :, :], UMAX).min(axis=(1, 2))
    out_hi = np.where(mask, result[None, :, :], UMIN).max(axis=(1, 2))
    over_lo = np.where(mask, over[None, :, :], True).all(axis=(1, 2))
    over_hi = np.where(mask, over[None, :, :], False).any(axis=(1, 2))

    df.loc[idx:idx+N-1, true_outputs] = np.stack((out_lo, out_hi, over_lo, over_hi), axis=1)
    idx += N
    
# SIGNED TESTING
ll_lo, ll_hi = i_intervals()
rr_lo, rr_hi = i_intervals()
cases = np.stack((ll_lo, ll_hi, rr_lo, rr_hi), axis=1)
df.loc[idx:idx+N*len(i_ops)-1, inputs] = np.tile(cases, [len(i_ops), 1])

signed = np.arange(IMIN, int(IMAX) + 1, dtype=BIGTY)
left, right = np.meshgrid(signed, signed)
l_lo, l_hi, r_lo, r_hi = (cases - IMIN).T

for numpy_op, range_op in i_ops:
    df.loc[idx:idx+N-1, approx_outputs] = range_op(ll_lo, ll_hi, rr_lo, rr_hi)
    result, over = i_eval(left, right, numpy_op)

    row_mask = (mat_idx >= r_lo[:, None]) & (mat_idx <= r_hi[:, None])
    col_mask = (mat_idx >= l_lo[:, None]) & (mat_idx <= l_hi[:, None])
    mask = row_mask[:, :, None] & col_mask[:, None, :]

    out_lo = np.where(mask, result[None, :, :], IMAX).min(axis=(1, 2))
    out_hi = np.where(mask, result[None, :, :], IMIN).max(axis=(1, 2))
    over_lo = np.where(mask, over[None, :, :], True).all(axis=(1, 2))
    over_hi = np.where(mask, over[None, :, :], False).any(axis=(1, 2))

    df.loc[idx:idx+N-1, true_outputs] = np.stack((out_lo, out_hi, over_lo, over_hi), axis=1)
    idx += N
    
# DUMP RESULTS
df.to_csv('results.csv', index=False)
fail_mask = (df[approx_outputs[0]] > df[true_outputs[0]]) | (df[approx_outputs[1]] < df[true_outputs[1]]) | \
            (df[approx_outputs[2]] > df[true_outputs[2]]) | (df[approx_outputs[3]] < df[true_outputs[3]])
df.loc[fail_mask].to_csv('failures.csv', index=False)

# SUMMARY STATISTICS
summary = pd.DataFrame(index=operations[::N], columns=['Valid', 'Actual Range Size', 'Approx Range Size', '% Increase'])
summary.index.name = 'Operation'

for op in summary.index:
    op_df = df[df[('Operation', None)] == op]
    op_df = op_df[approx_outputs + true_outputs].astype(np.float64)
    
    # Soundness
    range_valid = (
        (op_df[('Approx Range', 'Low')] <= op_df[('True Range', 'Low')]) &
        (op_df[('Approx Range', 'High')] >= op_df[('True Range', 'High')])
    ).all()
    over_valid = (
        (op_df[('Approx Over', 'Low')] <= op_df[('True Over', 'Low')]) &
        (op_df[('Approx Over', 'High')] >= op_df[('True Over', 'High')])
    ).all()
    valid = range_valid & over_valid
    
    # Precision
    true_size = (op_df[('True Range', 'High')] - op_df[('True Range', 'Low')] + 1).mean()
    approx_size = (op_df[('Approx Range', 'High')] - op_df[('Approx Range', 'Low')] + 1).mean()
    pct_increase = ((approx_size - true_size) / true_size) * 100
    summary.loc[op, :] = [valid, true_size, approx_size, pct_increase.round(3)]

summary.to_csv('summary.csv', index=True)
