import os

#SOLID_GTE_FLUID = False
SOLID_GTE_FLUID = True
errors = []
legend = []
for folder in os.listdir(os.path.join(os.path.dirname(__file__), "refinement_analysis")):
    full_path = os.path.join(os.path.dirname(
        __file__), "refinement_analysis", folder)
    # print(folder)
    solid_ref = int(folder[6])
    fluid_ref = int(folder[14])

    if SOLID_GTE_FLUID:
        if fluid_ref > solid_ref:
            continue
    else:
        if solid_ref > fluid_ref:
            continue

    legend.append((fluid_ref, solid_ref))
    # print(folder)

    convergence_lines = []
    convergence_data = False
    with open(os.path.join(full_path, "console_output.log"), "r") as file:
        for line in file:
            if convergence_data:
                convergence_lines.append(line)
            if "Error in goal functional values" in line:
                convergence_data = True

    print(convergence_lines)

    error_points = []
    for i, line in enumerate(convergence_lines):
        #print(line.strip("\n").strip(" ").split(" "))
        error = line.strip("\n").strip().split(" ")[-1]
        dofs = 50 * pow(2, i)
        error_points.append((dofs, error))

    errors.append(error_points)

colors = ["orange", "skyblue", "yellow", "blue", "vermillion"]
for i, error in enumerate(errors):
    print("""
    \\addplot[
    color=""", end="")
    print(colors[i], end=",\n")
    print("""\tmark=otimes,
    style=ultra thick,
    ]
    coordinates {\n\t""", end="")
    for (dof, error_val) in error:
        print(f"({dof},{error_val})", end="")
    print("\n\t};")
    # (17800,6.0871740432649257e-02)(128800,1.9325009333641840e-02)(976000,5.6331752241517430e-03)(7590400,1.5232819460108127e-03)
    # };

print("\\legend{", end="")
for i, l in enumerate(legend):
    print("{} {$|\\mathcal{T}_k^f| : |\\mathcal{T}_k^s|$ = " +
          str(pow(2, l[0])) + ":" + str(pow(2, l[1])) + "}", end=",")

print("}")
# \legend{uniform refinement, adaptive refinement, adaptive ref. + $L^2$ proj., adaptive ref. + $H^1_0$ proj.,,}
