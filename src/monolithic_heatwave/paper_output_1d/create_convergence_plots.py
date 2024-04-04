import os

#SOLID_GTE_FLUID = False 
SOLID_GTE_FLUID = True 
errors = []
errors_fluid = []
errors_solid = []
legend = []
for folder in os.listdir(os.path.join(os.path.dirname(__file__), "refinement_analysis")):
    full_path = os.path.join(os.path.dirname(__file__), "refinement_analysis", folder)
    #print(folder)
    solid_ref = int(folder[6])
    fluid_ref = int(folder[14])

    if SOLID_GTE_FLUID:
        if fluid_ref > solid_ref:
            continue
    else:
        if solid_ref > fluid_ref:
            continue

    legend.append((fluid_ref, solid_ref))
    #print(folder)

    convergence_lines = []
    convergence_data = False
    with open(os.path.join(full_path, "console_output.log"), "r") as file:
        for line in file:
            if "cycle" in line and "L2" in line:
                convergence_data = True

            if convergence_data:
                convergence_lines.append(line)

    index_end_first_half = convergence_lines.index('\n')
    #print(convergence_lines[1:index_end_first_half])

    error_points = []
    fluid_error_points = []
    solid_error_points = []
    for line in convergence_lines[1:index_end_first_half]:
        #print(line.strip("\n").strip(" ").split(" "))
        #print(line.strip("\n").strip().split(" ")[-4:])
        dofs, error, error_fluid, error_solid = line.strip("\n").strip().split(" ")[-4:]
        error_points.append((dofs, error))
        fluid_error_points.append((dofs, error_fluid))
        solid_error_points.append((dofs, error_solid))

    errors.append(error_points)
    errors_fluid.append(fluid_error_points)
    errors_solid.append(solid_error_points)

colors = ["orange", "skyblue", "yellow", "blue", "vermillion"]
for i, error in enumerate(errors):
    if i % 2 == 1:
        continue # skip

    print("""
    \\addplot[
    color=""", end="")
    print(colors[i // 2], end=",\n")
    print("""\tmark=otimes,
    style=ultra thick,
    ]
    coordinates {\n\t""", end="")
    for (dof, error_val) in error:
        print(f"({dof},{error_val})", end="")
    print("\n\t};")
    #(17800,6.0871740432649257e-02)(128800,1.9325009333641840e-02)(976000,5.6331752241517430e-03)(7590400,1.5232819460108127e-03)
    #};

print("\\legend{", end ="")
for i, l in enumerate(legend):
    if i % 2 == 1:
        continue # skip
    print("{} {$|\\mathcal{T}_k^f| : |\\mathcal{T}_k^s|$ = " +str(pow(2, l[0])) + ":"  + str(pow(2, l[1])) + "}", end=",")

print("}")
#\legend{uniform refinement, adaptive refinement, adaptive ref. + $L^2$ proj., adaptive ref. + $H^1_0$ proj.,,}

for _errors in [errors_fluid, errors_solid]:
    for i, error in enumerate(_errors):
        if i % 2 == 1:
            continue # skip

        style = "dashed" if _errors is errors_fluid else "dotted"

        print("""
        \\addplot[
        color=""", end="")
        print(colors[i // 2], end=",\n")
        print(f"""\tmark=otimes,
        style={style},
        ]
        coordinates {{\n\t""", end="")
        for (dof, error_val) in error:
            print(f"({dof},{error_val})", end="")
        print("\n\t};")
