import sys
import os
import numpy as np
import scipy.interpolate
import matplotlib.pyplot as plt

assert len(sys.argv) == 2, "You need to enter the path to the solution!"
path = sys.argv[1]


def load_all_vectors(path, pattern):
    file_names = [f for f in os.listdir(path) if pattern in f]
    vectors = [np.loadtxt(f"{path}/{file_name}") for file_name in file_names]
    return np.hstack(vectors)


def arrays_with_temporal_discontinuities(times, values):
    new_times = []
    new_values = []
    for i in range(times.shape[0]):
        if i+1 < times.shape[0] and times[i+1] == times[i]:
            new_times += 3*[times[i]]
            new_values += [values[i], np.inf, values[i+1]]
        else:
            new_times.append(times[i])
            new_values.append(values[i])
    return new_times, new_values

def save_vtk(file_name, solution, coordinates):
    lines = [
        "# vtk DataFile Version 3.0",
        "PDE SOLUTION",
        "ASCII",
        "DATASET UNSTRUCTURED_GRID",
        ""
    ]

    coordinates_time = coordinates[:,0]
    coordinates_space = coordinates[:,1] 

    solution_displacement = solution[0,:]
    solution_velocity     = solution[1,:]

    points = []
    cells = []
    displacement_lookup_table = []
    velocity_lookup_table = []

    n_space_dofs = np.sum(coordinates_time == 0.)

    offset = 0
    while offset + 2*n_space_dofs <= coordinates_time.shape[0]:
        n_points = len(points)
        for cell in range(n_space_dofs-1):
            cells.append((n_points+cell*2, n_points+cell*2+(2*(n_space_dofs-2)+2), n_points+cell*2+(2*(n_space_dofs-2)+2)+1, n_points+cell*2+1))

        for i in range(offset,offset+2*n_space_dofs):
            points.append((coordinates_time[i], coordinates_space[i], 0.))
            displacement_lookup_table.append(solution_displacement[i])
            velocity_lookup_table.append(solution_velocity[i])

            if i % n_space_dofs != 0 and (i+1) % n_space_dofs != 0:
                points.append((coordinates_time[i], coordinates_space[i], 0.))
                displacement_lookup_table.append(solution_displacement[i])
                velocity_lookup_table.append(solution_velocity[i])

        offset += 2*n_space_dofs

    lines.append(f"POINTS {len(points)} double")
    for point in points:
        lines.append(" ".join([str(coord) for coord in point]))
    lines.append("")

    lines.append(f"CELLS {len(cells)} {len(cells)*5}")    
    for cell in cells:
        lines.append("\t".join(["4"] + [str(dof) for dof in cell]))
    lines.append("")

    lines.append(f"CELL_TYPES {len(cells)}")
    lines.append(" 9"*len(cells))
    lines.append("")

    lines.append(f"POINT_DATA {len(points)}")
    lines.append(f"SCALARS displacement double 1")
    lines.append(f"LOOKUP_TABLE default")
    lines.append(" ".join([str(val) for val in displacement_lookup_table]))
    lines.append(f"SCALARS velocity double 1")
    lines.append(f"LOOKUP_TABLE default")
    lines.append(" ".join([str(val) for val in velocity_lookup_table]))

    with open(file_name, "w") as file:
        file.write("\n".join(lines))

# space-time solution vectors
# displacement:
solution_fluid_u = load_all_vectors(path, "solution_fluid_u_")
solution_solid_u = load_all_vectors(path, "solution_solid_u_")
# velocity:
solution_fluid_v = load_all_vectors(path, "solution_fluid_v_")
solution_solid_v = load_all_vectors(path, "solution_solid_v_")

# space-time error vectors
# displacement:
error_fluid_u = load_all_vectors(path, "error_fluid_u_")
error_solid_u = load_all_vectors(path, "error_solid_u_")
# velocity:
error_fluid_v = load_all_vectors(path, "error_fluid_v_")
error_solid_v = load_all_vectors(path, "error_solid_v_")

# coordinates
# fluid:
coordinates_fluid_x = np.loadtxt(path + "/coordinates_fluid_x.txt")
fluid_x_min, fluid_x_max = np.min(
    coordinates_fluid_x), np.max(coordinates_fluid_x)
coordinates_fluid_t = load_all_vectors(path, "coordinates_fluid_t_")
# np.min(coordinates_fluid_t), np.max(coordinates_fluid_t)
fluid_t_min, fluid_t_max = 0., 4.
coordinates_fluid = np.vstack((
    np.tensordot(coordinates_fluid_t, np.ones_like(
        coordinates_fluid_x), 0).flatten(),
    np.tensordot(np.ones_like(coordinates_fluid_t),
                 coordinates_fluid_x, 0).flatten()
)).T
fluid_n_dofs = {
    "space": coordinates_fluid_x.shape[0], "time": coordinates_fluid_t.shape[0]}
# solid:
coordinates_solid_x = np.loadtxt(path + "/coordinates_solid_x.txt")
solid_x_min, solid_x_max = np.min(
    coordinates_solid_x), np.max(coordinates_solid_x)
coordinates_solid_t = load_all_vectors(path, "coordinates_solid_t_")
# np.min(coordinates_solid_t), np.max(coordinates_solid_t)
solid_t_min, solid_t_max = 0., 4.
coordinates_solid = np.vstack((
    np.tensordot(coordinates_solid_t, np.ones_like(
        coordinates_solid_x), 0).flatten(),
    np.tensordot(np.ones_like(coordinates_solid_t),
                 coordinates_solid_x, 0).flatten()
)).T
solid_n_dofs = {
    "space": coordinates_solid_x.shape[0], "time": coordinates_solid_t.shape[0]}


fluid_grid_t, fluid_grid_x = np.mgrid[fluid_t_min:fluid_t_max:200j,
                                      fluid_x_min:fluid_x_max:200j]
solid_grid_t, solid_grid_x = np.mgrid[solid_t_min:solid_t_max:200j,
                                      solid_x_min:solid_x_max:200j]

# create vtk file for fluid
save_vtk(path + "/solution_fluid.vtk", np.vstack((solution_fluid_u, solution_fluid_v)), coordinates_fluid)
# create vtk file for solid
save_vtk(path + "/solution_solid.vtk", np.vstack((solution_solid_u, solution_solid_v)), coordinates_solid)

# create vtk file for fluid error
save_vtk(path + "/error_fluid.vtk", np.vstack((error_fluid_u, error_fluid_v)), coordinates_fluid)
# create vtk file for solid
save_vtk(path + "/error_solid.vtk", np.vstack((error_solid_u, error_solid_v)), coordinates_solid)

# displacement - interface
plt.title("Displacement - Interface")
solid_idx = coordinates_solid[:, 1] == 2.
fluid_idx = coordinates_fluid[:, 1] == 2.
plt.plot(*arrays_with_temporal_discontinuities(coordinates_solid[solid_idx][:, 0],
         solution_solid_u[solid_idx]), label="solid", color="red")
plt.plot(*arrays_with_temporal_discontinuities(coordinates_fluid[fluid_idx][:, 0],
         solution_fluid_u[fluid_idx]), label="fluid", color="blue")
plt.legend()
plt.savefig(path + "/plot_u_interface.png")
plt.clf()

# velocity - interface
plt.title("Velocity - Interface")
solid_idx = coordinates_solid[:, 1] == 2.
fluid_idx = coordinates_fluid[:, 1] == 2.
plt.plot(*arrays_with_temporal_discontinuities(
    coordinates_solid[solid_idx][:, 0], solution_solid_v[solid_idx]), label="solid", color="red")
plt.plot(*arrays_with_temporal_discontinuities(
    coordinates_fluid[fluid_idx][:, 0], solution_fluid_v[fluid_idx]), label="fluid", color="blue")
plt.legend()
plt.savefig(path + "/plot_v_interface.png")
plt.clf()

quit()


# fluid & solid displacement
grid_t, grid_x = np.mgrid[min(fluid_t_min, solid_t_min):max(fluid_t_max, solid_t_max):200j, min(
    fluid_x_min, solid_x_min):max(fluid_x_max, solid_x_max):200j]
coordinates = np.vstack((coordinates_solid, coordinates_fluid))
solution_u = np.hstack((solution_solid_u, solution_fluid_u))
grid = scipy.interpolate.griddata(
    coordinates, solution_u, (grid_t, grid_x), method="nearest")
plt.imshow(grid.T, extent=(min(fluid_t_min, solid_t_min), max(fluid_t_max, solid_t_max), min(
    fluid_x_min, solid_x_min), max(fluid_x_max, solid_x_max)), origin="lower")
plt.colorbar()
plt.title("Displacement")
plt.savefig(path + "/plot_u_new.png")
plt.clf()

# fluid & solid velocity
grid_t, grid_x = np.mgrid[min(fluid_t_min, solid_t_min):max(fluid_t_max, solid_t_max):200j, min(
    fluid_x_min, solid_x_min):max(fluid_x_max, solid_x_max):200j]
coordinates = np.vstack((coordinates_solid, coordinates_fluid))
solution_v = np.hstack((solution_solid_v, solution_fluid_v))
grid = scipy.interpolate.griddata(
    coordinates, solution_v, (grid_t, grid_x), method="nearest")
plt.imshow(grid.T, extent=(min(fluid_t_min, solid_t_min), max(fluid_t_max, solid_t_max), min(
    fluid_x_min, solid_x_min), max(fluid_x_max, solid_x_max)), origin="lower")
plt.colorbar()
plt.title("Velocity")
plt.savefig(path + "/plot_v_new.png")
plt.clf()


# fluid & solid displacement error
grid_t, grid_x = np.mgrid[min(fluid_t_min, solid_t_min):max(fluid_t_max, solid_t_max):200j, min(
    fluid_x_min, solid_x_min):max(fluid_x_max, solid_x_max):200j]
coordinates = np.vstack((coordinates_solid, coordinates_fluid))
error_u = np.hstack((error_solid_u, error_fluid_u))
grid = scipy.interpolate.griddata(
    coordinates, error_u, (grid_t, grid_x), method="nearest")
plt.imshow(grid.T, extent=(min(fluid_t_min, solid_t_min), max(fluid_t_max, solid_t_max), min(
    fluid_x_min, solid_x_min), max(fluid_x_max, solid_x_max)), origin="lower")
plt.colorbar()
plt.title("Displacement Error")
plt.savefig(path + "/plot_error_u_new.png")
plt.clf()

# fluid & solid velocity error
grid_t, grid_x = np.mgrid[min(fluid_t_min, solid_t_min):max(fluid_t_max, solid_t_max):200j, min(
    fluid_x_min, solid_x_min):max(fluid_x_max, solid_x_max):200j]
coordinates = np.vstack((coordinates_solid, coordinates_fluid))
error_v = np.hstack((error_solid_v, error_fluid_v))
grid = scipy.interpolate.griddata(
    coordinates, error_v, (grid_t, grid_x), method="nearest")
plt.imshow(grid.T, extent=(min(fluid_t_min, solid_t_min), max(fluid_t_max, solid_t_max), min(
    fluid_x_min, solid_x_min), max(fluid_x_max, solid_x_max)), origin="lower")
plt.colorbar()
plt.title("Velocity Error")
plt.savefig(path + "/plot_error_v_new.png")
plt.clf()

# displacement
plt.title("Displacement")
solid_grid = scipy.interpolate.griddata(
    coordinates_solid, solution_solid_u, (solid_grid_t, solid_grid_x), method="nearest")
fluid_grid = scipy.interpolate.griddata(
    coordinates_fluid, solution_fluid_u, (fluid_grid_t, fluid_grid_x), method="nearest")
plt.imshow(np.hstack((fluid_grid, solid_grid)).T, extent=(
    solid_t_min, solid_t_max, fluid_x_min, solid_x_max), origin='lower')
plt.xlabel("$t$")
plt.ylabel("$x$")
plt.colorbar()
plt.savefig(path + "/plot_u_joint.png")
plt.clf()

# displacement - interface
plt.title("Displacement - Interface")
solid_idx = coordinates_solid[:, 1] == 2.
fluid_idx = coordinates_fluid[:, 1] == 2.
plt.plot(*arrays_with_temporal_discontinuities(coordinates_solid[solid_idx][:, 0],
         solution_solid_u[solid_idx]), label="solid", color="red")
plt.plot(*arrays_with_temporal_discontinuities(coordinates_fluid[fluid_idx][:, 0],
         solution_fluid_u[fluid_idx]), label="fluid", color="blue")
plt.legend()
plt.savefig(path + "/plot_u_interface.png")
plt.clf()

# velocity - interface
plt.title("Velocity - Interface")
solid_idx = coordinates_solid[:, 1] == 2.
fluid_idx = coordinates_fluid[:, 1] == 2.
plt.plot(*arrays_with_temporal_discontinuities(
    coordinates_solid[solid_idx][:, 0], solution_solid_v[solid_idx]), label="solid", color="red")
plt.plot(*arrays_with_temporal_discontinuities(
    coordinates_fluid[fluid_idx][:, 0], solution_fluid_v[fluid_idx]), label="fluid", color="blue")
plt.legend()
plt.savefig(path + "/plot_v_interface.png")
plt.clf()

solid_grid = scipy.interpolate.griddata(
    coordinates_solid, solution_solid_u, (solid_grid_t, solid_grid_x), method="nearest")
fluid_grid = scipy.interpolate.griddata(
    coordinates_fluid, solution_fluid_u, (fluid_grid_t, fluid_grid_x), method="nearest")
plt.imshow(np.hstack((fluid_grid, solid_grid)).T, extent=(
    solid_t_min, solid_t_max, fluid_x_min, solid_x_max), origin='lower')
plt.xlabel("$t$")
plt.ylabel("$x$")
plt.colorbar()
plt.savefig(path + "/plot_u_joint.png")
plt.clf()

# velocity
plt.title("Velocity")
solid_grid = scipy.interpolate.griddata(
    coordinates_solid, solution_solid_v, (solid_grid_t, solid_grid_x), method="nearest")
fluid_grid = scipy.interpolate.griddata(
    coordinates_fluid, solution_fluid_v, (fluid_grid_t, fluid_grid_x), method="nearest")
plt.imshow(np.hstack((fluid_grid, solid_grid)).T, extent=(
    solid_t_min, solid_t_max, fluid_x_min, solid_x_max), origin='lower')
plt.xlabel("$t$")
plt.ylabel("$x$")
plt.colorbar()
plt.savefig(path + "/plot_v_joint.png")
plt.clf()

# displacement
fig, axs = plt.subplots(2, 1)
fig.suptitle("Displacement")

# solid:
solid_grid = scipy.interpolate.griddata(
    coordinates_solid, solution_solid_u, (solid_grid_t, solid_grid_x), method="nearest")
im0 = axs[0].imshow(solid_grid.T, extent=(
    solid_t_min, solid_t_max, solid_x_min, solid_x_max), origin='lower')
axs[0].set_xlabel("$t$")
axs[0].set_ylabel("$x$")
axs[0].set_title("solid")
fig.colorbar(im0, ax=axs[0])

# fluid:
fluid_grid = scipy.interpolate.griddata(
    coordinates_fluid, solution_fluid_u, (fluid_grid_t, fluid_grid_x), method="nearest")
im1 = axs[1].imshow(fluid_grid.T, extent=(
    fluid_t_min, fluid_t_max, fluid_x_min, fluid_x_max), origin='lower')
axs[1].set_xlabel("$t$")
axs[1].set_ylabel("$x$")
axs[1].set_title("fluid")
fig.colorbar(im1, ax=axs[1])

# plt.subplots_adjust(hspace=0.5)
plt.savefig(path + "/plot_u.png")
plt.clf()


# velocity
fig, axs = plt.subplots(2, 1)
fig.suptitle("Velocity")

# solid:
solid_grid = scipy.interpolate.griddata(
    coordinates_solid, solution_solid_v, (solid_grid_t, solid_grid_x), method="nearest")
im0 = axs[0].imshow(solid_grid.T, extent=(
    solid_t_min, solid_t_max, solid_x_min, solid_x_max), origin='lower')
axs[0].set_xlabel("$t$")
axs[0].set_ylabel("$x$")
axs[0].set_title("solid")
fig.colorbar(im0, ax=axs[0])

# fluid:
fluid_grid = scipy.interpolate.griddata(
    coordinates_fluid, solution_fluid_v, (fluid_grid_t, fluid_grid_x), method="nearest")
im1 = axs[1].imshow(fluid_grid.T, extent=(
    fluid_t_min, fluid_t_max, fluid_x_min, fluid_x_max), origin='lower')
axs[1].set_xlabel("$t$")
axs[1].set_ylabel("$x$")
axs[1].set_title("fluid")
fig.colorbar(im1, ax=axs[1])

# plt.subplots_adjust(hspace=0.5)
plt.savefig(path + "/plot_v.png")
