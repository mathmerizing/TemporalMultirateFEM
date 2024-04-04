# %% imports
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy.sparse

# %% load matrices

MULTIRATE_PATH = os.getcwd()
# print("MULTIRATE:")
#print("IS:", MULTIRATE_PATH)
#print("GOAL:", "/home/ifam/roth/Desktop/Code/dealii_dir/DWR/dwr-temporal-multiscale/deal.II/mandel_dG_2d")

[data, row, column] = np.loadtxt(os.path.join(MULTIRATE_PATH, "matrix.txt"))
multirate_matrix = scipy.sparse.csr_matrix(
    (data, (row.astype(int), column.astype(int))))
print(multirate_matrix.shape)
#multirate_rhs = np.loadtxt(os.path.join(MULTIRATE_PATH, "rhs_00000.txt"))[8:]

multirate_sol = np.loadtxt(os.path.join(
    MULTIRATE_PATH, "solution_00000.txt"))  # [8:]
multirate_sol_2 = np.loadtxt(os.path.join(
    MULTIRATE_PATH, "solution_00001.txt"))  # [8:]

# boundary_dofs = []
# for i in range(22):
#     for j in range(22):
#         if fsi_submatrix[i, j] != 0. and i != j:
#             break
#     else:
#         boundary_dofs.append(i)
# print(boundary_dofs)

plt.title("Multirate Matrix")
plt.spy(multirate_matrix, markersize=1)
plt.show()

plt.title("Multirate Matrix (full)")
plt.imshow(multirate_matrix.toarray(), interpolation='nearest', cmap=cm.Greys)
plt.show()

TIME_STEPPING_PATH = os.path.join(os.path.split(os.getcwd())[
                                  0], "mandel_dG0_2d_1u_2p")
print("TIME-STEPPING:")
print("IS:", TIME_STEPPING_PATH)
#print("GOAL:", "/home/ifam/roth/Desktop/Code/dealii_dir/SpaceTime/SpaceTimeFEM-deal.II/Biot/sequential/tensor_product_space_time/cG_s_dG_r_GaussLobatto_slabs")

[data, row, column] = np.loadtxt(
    os.path.join(TIME_STEPPING_PATH, "matrix.txt"))
space_time_matrix = scipy.sparse.csr_matrix(
    (data, (row.astype(int), column.astype(int))))

#print(multirate_matrix[:5, :5].toarray())
#print(space_time_matrix[:5, :5].toarray())
# quit()

# space_time_rhs = np.loadtxt(os.path.join(TIME_STEPPING_PATH, "rhs_00000.txt")) #[8:]
# space_time_rhs_2 = np.loadtxt(os.path.join(TIME_STEPPING_PATH, "rhs_00001.txt")) #[8:]
space_time_sol = np.loadtxt(os.path.join(
    TIME_STEPPING_PATH, "solution_00000.txt"))  # [8:]
space_time_sol_2 = np.loadtxt(os.path.join(
    TIME_STEPPING_PATH, "solution_00001.txt"))  # [8:]

# boundary_dofs = []
# for i in range(22):
#     for j in range(22):
#         if i != j and space_time_matrix[i, j] != 0.:
#             break
#     else:
#         boundary_dofs.append(i)
# print(boundary_dofs)

# # manually clear columns if DoFs are constrained
# for i in range(22):
#     if i not in boundary_dofs:
#         for j in boundary_dofs:
#             space_time_matrix[i, j] = 0.
#             space_time_matrix[j, i] = 0.

print("Error in solution vectors")
_i = np.argmax(np.abs(multirate_sol-space_time_sol))
print(f"Max error in first solution vector: {np.max(np.abs(multirate_sol-space_time_sol))} (val 1 = {multirate_sol[_i]}; val 2 = {space_time_sol[_i]}")
_i = np.argmax(np.abs(multirate_sol_2-space_time_sol_2))
print(f"Max error in second solution vector: {np.max(np.abs(multirate_sol_2-space_time_sol_2))} (val 1 = {multirate_sol_2[_i]}; val 2 = {space_time_sol_2[_i]}")
_i = np.argmax(np.abs(space_time_matrix.toarray().flatten() - multirate_matrix.toarray().flatten()))
print(f"Max error in matrices: {np.max(np.abs(space_time_matrix.toarray() - multirate_matrix.toarray()))} (val 1 = {space_time_matrix.toarray().flatten()[_i]}; val 2 = {multirate_matrix.toarray().flatten()[_i]})")

plt.title("Space-Time Matrix")
plt.spy(space_time_matrix, markersize=1)
plt.show()

plt.title("Space-Time Matrix (full)")
plt.imshow(space_time_matrix.toarray(), interpolation='nearest', cmap=cm.Greys)
plt.show()

plt.title("Error")
plt.spy(space_time_matrix - multirate_matrix, markersize=1)
plt.show()

plt.title("Error (full)")
plt.imshow(space_time_matrix.toarray() - multirate_matrix.toarray(),
           interpolation='nearest', cmap=cm.Greys)
plt.show()

plt.title("Error (relative)")
plt.imshow((space_time_matrix.toarray() - multirate_matrix.toarray()) /
           (multirate_matrix.toarray() + 1e-20), interpolation='nearest', cmap=cm.Greys)
plt.show()

print("Error per block:")
blocks = [("u(0)", 0, 18), ("u(1)", 18, 36),
          ("p(0)", 36, 40), ("p(1)", 40, 44)]
# print(blocks)

for block_i in blocks:
    for block_j in blocks:
        print(f"Block ({block_i[0]}, {block_j[0]}):")
        max_error = np.max(np.abs(multirate_matrix[block_i[1]:block_i[2], block_j[1]:block_j[2]].toarray(
        )-space_time_matrix[block_i[1]:block_i[2], block_j[1]:block_j[2]].toarray()))
        print("Max error:", max_error)

        if max_error > 0.:
            plt.title(f"Error in block ({block_i[0]}, {block_j[0]})")
            plt.imshow(multirate_matrix[block_i[1]:block_i[2], block_j[1]:block_j[2]].toarray(
            )-space_time_matrix[block_i[1]:block_i[2], block_j[1]:block_j[2]].toarray(), interpolation='nearest', cmap=cm.Greys)
            plt.show()

            plt.title(f"Space-time in block ({block_i[0]}, {block_j[0]})")
            plt.imshow(space_time_matrix[block_i[1]:block_i[2], block_j[1]:block_j[2]].toarray(
            ), interpolation='nearest', cmap=cm.Greys)
            plt.show()

            plt.title(f"Multirate in block ({block_i[0]}, {block_j[0]})")
            plt.imshow(multirate_matrix[block_i[1]:block_i[2], block_j[1]:block_j[2]].toarray(
            ), interpolation='nearest', cmap=cm.Greys)
            plt.show()

            print(" Aborting... Please debug/understand this block!")

            break
    else:
        continue  # if break was not called, continue with for loop

    break

print("OVERALL ERROR:", np.max(
    np.abs(multirate_matrix.toarray()-space_time_matrix.toarray())))

plt.title("Multirate coupling matrix")
plt.imshow(multirate_matrix_coupling.toarray(),
           interpolation='nearest', cmap=cm.Greys)
plt.show()

shifted_multirate_matrix_coupling = np.zeros_like(
    multirate_matrix_coupling.toarray())
space_local_dof_indices = {0: 0, 1: 1, 2: 18, 3: 2, 4: 3, 5: 19, 6: 4, 7: 5, 8: 20, 9: 6,
                           10: 7, 11: 21, 12: 8, 13: 9, 14: 10, 15: 11, 16: 12, 17: 13, 18: 14, 19: 15, 20: 16, 21: 17}
for i in range(44):
    for j in range(44):
        if multirate_matrix_coupling[i, j] != 0.:
            if i >= 22:
                i_h = i - 22
                i_k = 1
            else:
                i_h = i
                i_k = 0

            assert space_local_dof_indices[
                i_h] < 18, f"Invalid i_h: {space_local_dof_indices[i_h]} (>= 18)"
            #print("i_h:", space_local_dof_indices[i_h])

            if j >= 22:
                j_h = j - 22
                j_k = 1
            else:
                j_h = j
                j_k = 0

            #print("j_h:", space_local_dof_indices[j_h]-18)
            assert space_local_dof_indices[j_h] - \
                18 < 4, f"Invalid j_h: {space_local_dof_indices[j_h]-18} (>= 4)"

            shifted_multirate_matrix_coupling[
                space_local_dof_indices[i_h] + i_k*18 + 0,
                (space_local_dof_indices[j_h]-18) + j_k*4 + 36
            ] = multirate_matrix_coupling[i, j]

plt.title("CELL multirate coupling matrix 1")
plt.imshow(multirate_matrix_coupling_1.toarray(),
           interpolation='nearest', cmap=cm.Greys)
plt.colorbar()
plt.show()

plt.title("CELL Space-time coupling matrix 1")
plt.imshow(space_time_matrix_coupling_1.toarray(),
           interpolation='nearest', cmap=cm.Greys)
plt.colorbar()
plt.show()

plt.title("CELL error between coupling matrices 1")
plt.imshow(multirate_matrix_coupling_1.toarray(
)-space_time_matrix_coupling_1.toarray(), interpolation='nearest', cmap=cm.Greys)
plt.colorbar()
plt.show()

plt.title("CELL multirate coupling matrix 2")
plt.imshow(multirate_matrix_coupling_2.toarray(),
           interpolation='nearest', cmap=cm.Greys)
plt.show()

plt.title("CELL Space-time coupling matrix 2")
plt.imshow(space_time_matrix_coupling_2.toarray(),
           interpolation='nearest', cmap=cm.Greys)
plt.show()

plt.title("CELL error between coupling matrices 2")
plt.imshow(multirate_matrix_coupling_2.toarray(
)-space_time_matrix_coupling_2.toarray(), interpolation='nearest', cmap=cm.Greys)
plt.colorbar()
plt.show()

print("Max error in cell-wise interface terms:",
      np.max(np.abs(multirate_matrix_coupling_2.toarray()-space_time_matrix_coupling_2.toarray())))

plt.title("Shifted multirate coupling matrix (p,u)")
plt.imshow(
    shifted_multirate_matrix_coupling[0:18, 36:44], interpolation='nearest', cmap=cm.Greys)
plt.colorbar()
plt.show()

plt.title("Space-Time Matrix (p,u)")
plt.imshow(space_time_matrix.toarray()[
           0:18, 36:44], interpolation='nearest', cmap=cm.Greys)
plt.colorbar()
plt.show()

plt.title("Multirate Matrix (p,u)")
plt.imshow(multirate_matrix.toarray()[
           0:18, 36:44], interpolation='nearest', cmap=cm.Greys)
plt.colorbar()
plt.show()

plt.title("Difference between Multirate and Space-Time Matrix (p,u)")
plt.imshow((multirate_matrix.toarray() - space_time_matrix.toarray())
           [0:18, 36:44], interpolation='nearest', cmap=cm.Greys)
plt.colorbar()
plt.show()

print("Multirate Submatrix:")
with np.printoptions(precision=2, linewidth=999):
    print(multirate_matrix[0:9, 36:44].toarray())

print("Space-Time Matrix:")
with np.printoptions(precision=2, linewidth=999):
    print(space_time_matrix[0:9, 36:44].toarray())

print("Shifted multirate Matrix:")
bc_ids = [0, 1, 3, 4, 8]
shifted_multirate_matrix_coupling[bc_ids, :] = 0.
with np.printoptions(precision=2, linewidth=999):
    print(shifted_multirate_matrix_coupling[0:9, 36:44])

# plt.title("Space-Time Matrix (pressure)")
# plt.imshow(space_time_matrix[-4:, -4:].toarray(), interpolation='nearest', cmap=cm.Greys)
# plt.show()
# with np.printoptions(precision=2):
#     print(space_time_matrix[-4:, -4:].toarray())

# print(fsi_rhs.shape)
# print("FSI RHS:", fsi_rhs)
# print(space_time_rhs.shape)
# print("Space-Time RHS:", space_time_rhs)
# print("Space-Time RHS (Second step):", space_time_rhs_2)

print("TIME STEP 1:")
print("  MultiR solution:", multirate_sol)
print("  S-T solution:   ", space_time_sol)
print("  Error:", multirate_sol - space_time_sol)

print("TIME STEP 2:")
print("  MultiR solution:", multirate_sol_2)
print("  S-T solution:", space_time_sol_2)
print("  Error:", multirate_sol_2 - space_time_sol_2)

# # %% compare each block of the matrix
# #blocks = [("p", 18, 22)]
# blocks = [("v_1", 0, 9), ("v_2", 9, 18), ("p", 18, 22)]
# print(blocks)

# for block_i in blocks:
#     scaling = 1.0e3 if block_i[0] == "v_1" or block_i[0] == "v_2" else 1.0
#     for block_j in blocks:
#         print(f"Block ({block_i[0]}, {block_j[0]}):")
#         print("Scaling", scaling)

#         with np.printoptions(precision=2):
#             print("FSI submatrix:")
#             _fsi_matrix = scaling * fsi_submatrix[block_i[1]:block_i[2], block_j[1]:block_j[2]].toarray()
#             print(_fsi_matrix)

#             print("Space-time matrix:")
#             _space_time_matrix = space_time_matrix[block_i[1]:block_i[2], block_j[1]:block_j[2]].toarray()
#             print(_space_time_matrix)

#             print("Absolut difference of matrices:")
#             diff_matrix = _fsi_matrix - _space_time_matrix
#             print(diff_matrix)

#             print("Relative difference of matrices (in %):")
#             diff_matrix[np.abs(diff_matrix) < 1.0e-10] = 0. # ignore small differences
#             rel_diff_matrix = 100. * diff_matrix / (_fsi_matrix + 1.0e-30)
#             rel_diff_matrix[np.abs(rel_diff_matrix) < 1.0e-4] = 0. # ignore relative differences smaller than 0.0001 %
#             print(rel_diff_matrix)

#             # fig, (ax1, ax2) = plt.subplots(1, 2)
#             # fig.suptitle(f"Block ({block_i[0]}, {block_j[0]}):")
#             # ax1.imshow(_fsi_matrix, interpolation='nearest', cmap=cm.Greys)
#             # ax1.set_title("FSI")
#             # ax2.imshow(_space_time_matrix, interpolation='nearest', cmap=cm.Greys)
#             # ax2.set_title("Space-Time")
#             # plt.show()
#         print("\n\n")


# # %%

# %%
