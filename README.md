# Fuzzing-based assessment of binary operator range analysis algorithms

This repository uses Numpy and Pandas to assess the correctness and precision of binary operator range analyses for use in our Rust MIR pass.

Configuration variables can be set in [common.py](/common.py). Note that performance makes heavy use of precomputation optimized for 8-bit types and will worsen dramatically for larger types. To add a new operation, place its vectorized range analysis in [binop.py](/binop.py) and add it to the relevant list in [driver.py](/driver.py).

See [plot.py](/plot.py) for preliminary plotting code.

The driver code will create three `.csv` files: `results.csv` to list all trials, `summary.csv` with summary statistics, and `failures.csv` to filter trials in which the true range is not contained in the approximate range. This file should be empty!
