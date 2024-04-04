import dolfin
import numpy as np
import matplotlib.pyplot as plt
from fenics import *
from ufl import replace
import time
import argparse
import logging
import os

set_log_active(False) # turn off FEniCS logging
parameters["reorder_dofs_serial"] = False

# parse command line arguments
parser = argparse.ArgumentParser(description="Run a staggered multirate simulation of the 2D Mandel benchmark of poroelasticity.")
parser.add_argument("--n_time_p", type=int, default=1, help="number of time steps for pressure")
parser.add_argument("--convergence_criterion", type=str, default="residual", choices=["residual", "solution"], help="convergence criterion for the staggered scheme")

# parse args
args = parser.parse_args()
n_time_u = 1
n_time_p = args.n_time_p
assert n_time_p >= 1, "n_time_p must be >= 1"
convergence_criterion = args.convergence_criterion

PROBLEM = "Mandel"

start_time = 0.
end_time = 5.0e6

# spatial degrees
s_u = 2
s_p = 1
# macro time step size: k (for displacement)
# micro time step size: k / q (for pressure)
q = float(n_time_p)
k =  q * 1000.0

# Mandel parameters
# M_biot = Biot's constant
M_biot: float = 1.75e7  # 2.5e+12
c_biot: float = 1.0 / M_biot

# alpha_biot = b_biot = Biot's modulo
alpha_biot: float = 1.0
viscosity_biot: float = 1.0e-3
K_biot: float = 1.0e-13
density_biot: float = 1.0

# Traction
traction_x_biot: float = 0.0
traction_y_biot: float = -1.0e7

# Solid parameters
density_structure: float = 1.0
lame_coefficient_mu: float = 1.0e8
poisson_ratio_nu: float = 0.2
lame_coefficient_lambda: float = (2.0 * poisson_ratio_nu * lame_coefficient_mu) / (
    1.0 - 2.0 * poisson_ratio_nu
)
    
MAX_COUPLING_ITER = 10

# create output directory, if it does not exist yet
FOLDER = f"results/{PROBLEM}/nTimeP_{n_time_p}_convergenceCriterion_{convergence_criterion}"
if not os.path.exists(FOLDER):
    os.makedirs(FOLDER)

# remove old log file, if it exists
if os.path.exists(f"{FOLDER}/output.log"):
    os.remove(f"{FOLDER}/output.log")

# configure logging
logging.basicConfig(
    filename=f"{FOLDER}/output.log", 
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

##############################################
# Start a time marching / time slabbing loop #
##############################################
total_flow_solve_time = 0.
total_mechanics_solve_time = 0.
total_plotting_time = 0.
total_postprocessing_time = 0.
setup_time = 0.
total_convergence_criterion_time = 0.

# start simulation
cpu_start_time = time.time()
setup_start_time = time.time()
logging.info(f"CONFIG: problem = {PROBLEM}, s = ({s_u}/{s_p}), r = (0/0), k = {k}, n_time = ({n_time_u}/{n_time_p}), nonlinear = False")

slabs = [(start_time, start_time+k)]
while slabs[-1][1] < end_time - 1e-8:
    slabs.append((slabs[-1][1], slabs[-1][1]+k))

# create spatial mesh
n_x, n_y = 16, 16
space_mesh = RectangleMesh(Point(0.0, 0.0), Point(100.0, 20.0), n_x, n_y)
plot(space_mesh, title="spatial mesh")
plt.savefig(f"{FOLDER}/spatial_mesh.png")

# get spatial function space
element = {
    "u": VectorElement("Lagrange", space_mesh.ufl_cell(), s_u),
    "p": FiniteElement("Lagrange", space_mesh.ufl_cell(), s_p),
}
V = {
    "u": FunctionSpace(space_mesh, element["u"]),
    "p": FunctionSpace(space_mesh, element["p"])
}
dofs = {
    "u": V["u"].dim(),
    "p": V["p"].dim()
}
logging.info(f"Number of spatial DoFs: {dofs['u']+dofs['p']} ({dofs['u']} + {dofs['p']})")
u = TrialFunction(V["u"])
p = TrialFunction(V["p"])
u_h = Function(V["u"])
p_h = [Function(V["p"]) for _ in range(n_time_p)]
# u_old and p_old are being used for the convergence criterion to check if the solution has converged
u_old = Function(V["u"])
p_old = [Function(V["p"]) for _ in range(n_time_p)]
phi_u = TestFunction(V["u"])
phi_p = TestFunction(V["p"])
    
# boundaries
left = CompiledSubDomain("near(x[0], 0.) && on_boundary")
right = CompiledSubDomain("near(x[0], 100.) && on_boundary")
down = CompiledSubDomain("near(x[1], 0.) && on_boundary")
up = CompiledSubDomain("near(x[1], 20.) && on_boundary")

facet_marker = MeshFunction("size_t", space_mesh, 1)
facet_marker.set_all(0)
left.mark(facet_marker, 1)
right.mark(facet_marker, 2)
down.mark(facet_marker, 3)
up.mark(facet_marker, 4)

# boundary for traction force
ds_up = Measure("ds", subdomain_data=facet_marker, subdomain_id=4)
# boundary for cost functional
ds_down = Measure("ds", subdomain_data=facet_marker, subdomain_id=3)

# initial condition on slab
u0 = Function(V["u"])
p0 = Function(V["p"])
u0 = interpolate(Constant((0.,0.)), V["u"])
p0 = interpolate(Constant(0.), V["p"])

u_h.assign(u0)
for i in range(n_time_p):
    p_h[i].assign(p0)

def stress_lin(u):
    return lame_coefficient_lambda * div(u) * Identity(2) \
        + lame_coefficient_mu * (grad(u) + grad(u).T)

normal = FacetNormal(space_mesh)

# pre-compute all spatial forms
form = {}
# forms for flow problem
form["flow_system"] = Constant(c_biot) * p * phi_p * dx \
    + Constant(k / q) * Constant(K_biot / viscosity_biot) * dot(grad(p), grad(phi_p)) * dx
form["flow_div"] = Constant(alpha_biot) * div(u0 - u_h) * phi_p * dx
form["flow_mass"] = Constant(c_biot) * p0 * phi_p * dx
# forms for mechanics problem
form["mechanics_system"] = Constant(k) * inner(stress_lin(u), grad(phi_u)) * dx
form["mechanics_traction"] = Constant(k) * Constant(traction_y_biot) * phi_u[1] * ds_up
form["mechanics_pressure"] = Constant(-1. * k / q) * (
                            - Constant(alpha_biot) * inner(p_h[0] * Identity(2), grad(phi_u)) * dx
                            + Constant(alpha_biot) * inner(p_h[0] * normal, phi_u) * ds_up
)
for i in range(1, n_time_p):
    form["mechanics_pressure"] += Constant(-1. * k / q) * (
                            - Constant(alpha_biot) * inner(p_h[i] * Identity(2), grad(phi_u)) * dx
                            + Constant(alpha_biot) * inner(p_h[i] * normal, phi_u) * ds_up
    )

# boundary conditions
bc_left = DirichletBC(V["u"].sub(0), Constant(0.0), left)  # left: u_x = 0
bc_right = DirichletBC(V["p"], Constant(0.0), right)       # right:  p = 0
bc_down = DirichletBC(V["u"].sub(1), Constant(0.0), down)  # down: u_y = 0
bc = {"u": [bc_left, bc_down], "p": [bc_right]}

pressure_down_values = []
times_pressure_down = []
displacement_x_down_values = []
times_displacement_x_down = []
total_n_dofs = 0
goal_functional = {"u": 0., "p": 0.}
goal_functional_reference = {"u": 271076105.705716, "p": 84171719545698.2}

special_times = [1000., 5000., 10000., 100000., 500000., 5000000.]
special_x_values = np.linspace(0., 100., n_x+1)
bottom_sol_u_x = {}
bottom_sol_p = {}

def compute_residual(p_h, u_h):
    residual_norm = {
        "l2" :    0., 
        "linfty": 0.
    }
    
    mechanics_residual = assemble(replace(form["mechanics_system"], {u: u_h}) - form["mechanics_traction"] - form["mechanics_pressure"])
    for _bc in bc["u"]:
        _bc.apply(mechanics_residual)
    residual_norm["l2"] = mechanics_residual.norm('l2')**2
    residual_norm["linfty"] = mechanics_residual.norm('linf')
    
    # first flow step
    flow_residual = assemble(replace(form["flow_system"], {p: p_h[0]}) - form["flow_div"] - form["flow_mass"])
    for _bc in bc["p"]:
        _bc.apply(flow_residual)
    residual_norm["l2"] += flow_residual.norm('l2')**2
    residual_norm["linfty"] = max(residual_norm["linfty"], flow_residual.norm('linf'))
    
    # remaining flow steps
    for i in range(1, n_time_p):
        flow_residual = assemble(replace(form["flow_system"], {p: p_h[i]}) - replace(form["flow_mass"], {p0: p_h[i-1]}))
        for _bc in bc["p"]:
            _bc.apply(flow_residual)
        residual_norm["l2"] += flow_residual.norm('l2')**2
        residual_norm["linfty"] = max(residual_norm["linfty"], flow_residual.norm('linf'))

    # finish l2 calculation
    residual_norm["l2"] = np.sqrt(residual_norm["l2"])
    
    # TODO: If this is still too slow, use matrix-vector product to assemble residual for linear poroelasticity
    return residual_norm

def compute_solution_diffs(p_h, u_h, p_old, u_old):
    diffs = {
        "l2" :    0.,
        "linfty": 0.,
        "terminate": False
    }

    _norm = {
        "l2" :    0.,
        "linfty": 0.
    }

    # displacement
    diffs["l2"] = norm(u_h.vector() - u_old.vector(), "l2")**2
    diffs["linfty"] = norm(u_h.vector() - u_old.vector(), "linf")

    _norm["l2"] = norm(u_h.vector(), "l2")**2
    _norm["linfty"] = norm(u_h.vector(), "linf")

    # pressure
    for i in range(n_time_p):
        diffs["l2"] += norm(p_h[i].vector() - p_old[i].vector(), "l2")**2
        diffs["linfty"] = max(diffs["linfty"], norm(p_h[i].vector() - p_old[i].vector(), "linf"))

        _norm["l2"] += norm(p_h[i].vector(), "l2")**2
        _norm["linfty"] = max(_norm["linfty"], norm(p_h[i].vector(), "linf"))

    # finish l2 calculation
    diffs["l2"] = np.sqrt(diffs["l2"])
    _norm["l2"] = np.sqrt(_norm["l2"])

    # make errors relative and check if converged
    diffs["l2"] /= _norm["l2"]
    diffs["linfty"] /= _norm["linfty"]

    if diffs["l2"] < 1e-2 and diffs["linfty"] < 1e-2:
        diffs["terminate"] = True
        
    return diffs


setup_time = time.time() - setup_start_time

#####################
# Time slabbing loop:
for l, slab in enumerate(slabs):
    logging.info(f"Solving on slab_{l} = Ω x ({round(slab[0],5)}, {round(slab[1],5)}) ...")
                                 
    convergence_criterion_start_time = time.time()
    l2_residuals = []
    linfty_residuals = []
    l2_solution_diffs = []
    linfty_solution_diffs = []

    if convergence_criterion == "residual":
        residual = compute_residual(p_h, u_h)
        #print("Initial residual:", residual)
    
        l2_residuals = [residual["l2"]]
        linfty_residuals = [residual["linfty"]]
    elif convergence_criterion == "solution":
        u_old.assign(u_h)
        for i in range(n_time_p):
            p_old[i].assign(p_h[i])
    total_convergence_criterion_time += time.time() - convergence_criterion_start_time
    
    for n in range(MAX_COUPLING_ITER):
        ###################
        # flow problem
        #
        flow_solve_start_time = time.time()
        # first flow step
        solve(form["flow_system"] == form["flow_div"] + form["flow_mass"], p_h[0], bc["p"])
        
        # remaining flow steps
        for i in range(1, n_time_p):
            solve(form["flow_system"] == replace(form["flow_mass"], {p0: p_h[i-1]}), p_h[i], bc["p"])
        total_flow_solve_time += time.time() - flow_solve_start_time
        
        ###################
        # mechanics problem
        #
        mechanics_solve_start_time = time.time()
        solve(form["mechanics_system"] == form["mechanics_traction"] + form["mechanics_pressure"], u_h, bc["u"])
        total_mechanics_solve_time += time.time() - mechanics_solve_start_time
        
        if convergence_criterion == "residual":
            # use l2 and linfty norms of monolithic problem residual as convergence criterion
            convergence_criterion_start_time = time.time()
            residual = compute_residual(p_h, u_h)
            #print(f"{n+1}.th residual:", residual)
            l2_residuals.append(residual["l2"])
            linfty_residuals.append(residual["linfty"])
            
            # If both residuals have not improved by at least 20 % then finish staggered scheme
            relative_l2_reduction = (l2_residuals[-2] - l2_residuals[-1]) / l2_residuals[-1]
            relative_linfty_reduction = (linfty_residuals[-2] - linfty_residuals[-1]) / linfty_residuals[-1]
            total_convergence_criterion_time += time.time() - convergence_criterion_start_time
            if relative_l2_reduction < 0.2 or relative_linfty_reduction < 0.2:
                break # finished staggered scheme
        elif convergence_criterion == "solution":
            # use l2 and linfty norms of solution difference between last two iterations as convergence criterion
            convergence_criterion_start_time = time.time()
            diffs = compute_solution_diffs(p_h, u_h, p_old, u_old)
            #print(f"{n+1}.th solution diff:", diffs)
            l2_solution_diffs.append(diffs["l2"])
            linfty_solution_diffs.append(diffs["linfty"])

            # prepare next iteration
            u_old.assign(u_h)
            for i in range(n_time_p):
                p_old[i].assign(p_h[i])
            
            total_convergence_criterion_time += time.time() - convergence_criterion_start_time

            if diffs["terminate"]:
                break
    else:
        logging.info(f"Reached maximal # of staggered iterations: {MAX_COUPLING_ITER} iterations")

    u0.assign(u_h)
    p0.assign(p_h[-1])
    
    if False: #l % 10 == 0:
        plotting_start_time = time.time()
        # plot final solution on slab
        print(f"t = {slab[1]}:")
        c = plot(sqrt(dot(u0, u0)), title="Displacement")
        plt.colorbar(c, orientation="horizontal")
        plt.show()
        c = plot(p0, title="Pressure")
        plt.colorbar(c, orientation="horizontal")
        plt.show()
        total_plotting_time += time.time() - plotting_start_time

    # compute functional values
    postprocessing_start_time = time.time()
    total_n_dofs += dofs["u"] * n_time_u + dofs["p"] * n_time_p
   
    for i in range(n_time_p):
        pressure_down_values.append(float(assemble(p_h[i] * ds_down)))
        goal_functional["p"] += pressure_down_values[-1] * k / q
        times_pressure_down.append(slab[0] + i*k/q)

    displacement_x_down_values.append(float(assemble(u_h[0] * ds_down)))
    goal_functional["u"] += displacement_x_down_values[-1] * k
    times_displacement_x_down.append(0.5*(slab[0] + slab[1]))
    
    # save solution at botttom for special times
    while len(special_times) > 0 and special_times[0] <= slab[1] + 1e-4:
        _t = special_times.pop(0)

        # get micro time step for pressure that contains _t
        _i = 0
        while _i < n_time_u-1 and _t > slab[0] + _i*k/q + 1e-4:
            _i+=1

        bottom_sol_u_x[_t] = []
        bottom_sol_p[_t] = []

        for i in range(len(special_x_values)):
            x = special_x_values[i]

            tmp = u_h(Point(float(x),0.))
            bottom_sol_u_x[_t].append(tmp[0])

            tmp = p_h[_i](Point(float(x),0.))
            bottom_sol_p[_t].append(tmp)
    total_postprocessing_time += time.time() - postprocessing_start_time

    logging.info("Done.\n")
    
logging.info("------------")
logging.info("| RESULTS: |")
logging.info("------------")
logging.info(f"Space-time Dofs: {total_n_dofs:,}\n\n")

logging.info("ERROR:")
logging.info(f"J(u_h):          {goal_functional['u']:.5}")
logging.info(f"J(u):            {goal_functional_reference['u']:.5}")
logging.info(f"|J(u)-J(u_h)|:   {abs(goal_functional['u']-goal_functional_reference['u']):.5}")
logging.info(f"J(p_h):          {goal_functional['p']:.5}")
logging.info(f"J(p):            {goal_functional_reference['p']:.5}")
logging.info(f"|J(p)-J(p_h)|:   {abs(goal_functional['p']-goal_functional_reference['p']):.5}\n")

logging.info("TIME:")
cpu_time = round(time.time() - cpu_start_time - total_postprocessing_time, 5) # NOTE: postprocessing time is not included in total time
logging.info(f"CPU Time:             {cpu_time} s")
logging.info(f"Setup Time:           {setup_time:.5} s ({100. * setup_time / cpu_time:.5} %)")
logging.info(f"Flow Solve Time:      {total_flow_solve_time:.5} s ({100. * total_flow_solve_time / cpu_time:.5} %)")
logging.info(f"Mechanics Solve Time: {total_mechanics_solve_time:.5} s ({100. * total_mechanics_solve_time / cpu_time:.5} %)")
logging.info(f"Conv. Criterion Time: {total_convergence_criterion_time:.5} s ({100. * total_convergence_criterion_time / cpu_time:.5} %)")
#logging.info(f"Plotting Time:        {total_plotting_time:.5} s ({100. * total_plotting_time / cpu_time:.5} %)") # NOTE: Currently not plotting solution
#logging.info(f"Postprocessing Time:  {total_postprocessing_time:.5} s ({100. * total_postprocessing_time / cpu_time:.5} %) \n\n")

plt.clf()
plt.title("Goal functional pressure")
plt.plot(times_pressure_down, pressure_down_values)
plt.savefig(f"{FOLDER}/goal_functional_pressure.png")

plt.clf()
plt.title("Goal functional x-displacement")
plt.plot(times_displacement_x_down, displacement_x_down_values)
plt.savefig(f"{FOLDER}/goal_functional_displacement_x.png")

plt.clf()
for _t in bottom_sol_u_x.keys():
    plt.plot(special_x_values, bottom_sol_u_x[_t], label=f"t = {_t}")
plt.xlabel("x")
plt.ylabel(r"$u_x(x,0)$")
plt.title("x-Displacement at bottom boundary")
plt.legend()
plt.savefig(f"{FOLDER}/bottom_displacement.png")

plt.clf()
for _t in bottom_sol_p.keys():
    plt.plot(special_x_values, bottom_sol_p[_t], label=f"t = {_t}")
plt.xlabel("x")
plt.ylabel(r"$p(x,0)$")
plt.title("Pressure at bottom boundary")
plt.legend()
plt.savefig(f"{FOLDER}/bottom_pressure.png")
