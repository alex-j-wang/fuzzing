import pandas as pd
from itertools import product
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "monospace"

SIZE = 256
LB = -10
UB = 266
OP = "U8::BITWISE_AND"
NAME = "U8::BITAND"
df = pd.read_csv(f"{SIZE}/results.csv", header=[0, 1])

inputs = list(product(['Left', 'Right'], ['Low', 'High']))
approx_outputs = list(product(['Approx Range', 'Approx Over'], ['Low', 'High']))
true_outputs = list(product(['True Range', 'True Over'], ['Low', 'High']))

op_df = df[df[('Operation', 'nan')] == OP]
op_df = op_df[inputs + approx_outputs + true_outputs]

true_size = op_df[('True Range', 'High')] - op_df[('True Range', 'Low')] + 1
approx_size = op_df[('Approx Range', 'High')] - op_df[('Approx Range', 'Low')] + 1

plt.scatter(true_size[::10], approx_size[::10], s=10, linewidth=0, alpha=0.5)
plt.xlabel('True Range Size')
plt.ylabel('Rough Range Size')
plt.title(f'Range Precision ({NAME} {SIZE})')
plt.plot([1, true_size.max()], [1, true_size.max()], 'r--', linewidth=2)
plt.grid(True, which="both", ls="--", linewidth=0.5)
plt.xlim(LB, UB)
plt.ylim(LB, UB)
ax = plt.gca()
ax.set_aspect('equal', adjustable='box')

plt.savefig(f'plots/{NAME.replace("::", "-")}-{SIZE}.svg', bbox_inches='tight')

# ll_lo, ll_hi, rr_lo, rr_hi = op_df[inputs].to_numpy().T
# true_size = op_df[('True Range', 'High')] - op_df[('True Range', 'Low')] + 1
# approx_size = op_df[('Approx Range', 'High')] - op_df[('Approx Range', 'Low')] + 1
# zero_mask = (rr_lo <= 0) & (rr_hi >= 0)
# over_mask = (ll_lo == -128) & rr_hi == -1
# colors = np.where(zero_mask | over_mask, 'orange', 'b')

# plt.scatter(true_size[::10], approx_size[::10], s=10, c=colors[::10])
# plt.xlabel('True Range Size')
# plt.ylabel('Rough Range Size')
# plt.title(f'Range Precision ({NAME} {size})')
# plt.plot([1, true_size.max()], [1, true_size.max()], 'r--', linewidth=2)
# plt.grid(True, which="both", ls="--", linewidth=0.5)
# plt.xlim(lb, ub)
# plt.ylim(lb, ub)
# ax = plt.gca()
# ax.set_aspect('equal', adjustable='box')

# plt.savefig(f'plots/{NAME.replace("::", "-")}-{size}.svg', bbox_inches='tight')

# true_size = op_df[('True Range', 'High')] - op_df[('True Range', 'Low')] + 1
# approx_size = op_df[('Approx Range', 'High')] - op_df[('Approx Range', 'Low')] + 1
# points = np.column_stack([true_size, approx_size])
# uniq, counts = np.unique(points, axis=0, return_counts=True)
# alpha = counts / counts.max()

# plt.scatter(
#     uniq[:, 0],
#     uniq[:, 1],
#     linewidth=0,
#     facecolors=[(0.122, 0.467, 0.706, a) for a in alpha]
# )
# plt.xlabel('True Range Size')
# plt.ylabel('Rough Range Size')
# plt.title(f'Range Precision ({NAME} {size})')
# plt.plot([1, 16], [1, 16], 'r--', linewidth=2)
# plt.grid(True, which="both", ls="--", linewidth=0.5)
# plt.xlim(lb, ub)
# plt.ylim(lb, ub)
# ax = plt.gca()
# ax.set_aspect('equal', adjustable='box')

# plt.savefig(f'plots/{NAME.replace("::", "-")}-{size}.svg', bbox_inches='tight')
