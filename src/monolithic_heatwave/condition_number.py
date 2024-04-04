import sys
import os
import numpy as np
import scipy
import scipy.sparse

assert len(sys.argv) == 2, "You need to enter the path to the solution!"
path = sys.argv[1]

cycle_number = -1
for p in path.split("/"):
    if "cycle=" in p:
        cycle_number = int(p.strip("cycle="))
print(cycle_number, end="   ")

[data, row, column] = np.loadtxt(path + "/matrix.txt")
matrix = scipy.sparse.csr_matrix(
    (data, (row.astype(int), column.astype(int))))

print(matrix.shape[0], end="   ")

print(np.format_float_scientific(np.linalg.cond(matrix.toarray()), precision=2))
