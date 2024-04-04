import numpy as np
import matplotlib.pyplot as plt
import os

results = {"solution": {}, "residual": {}}
# reference values for QoI
J_u = 271076105.705716
J_p = 84171719545698.2

# read all output.log files from results/Mandel/ and plot the results
for folder in os.listdir("results/Mandel/"):
    # get all folder names
    if os.path.isdir("results/Mandel/" + folder):
        # print(folder)
        _, n_time_p, _, convergence_criterion = folder.split("_")
        n_time_p = int(n_time_p)

        results[convergence_criterion][n_time_p] = {}

        # read output.log
        with open("results/Mandel/" + folder + "/output.log", "r") as f:
            # ignore all lines until the first line with "RESULTS"
            is_results = False
            result_lines = []
            for line in f:
                if "RESULTS" in line:
                    is_results = True
                if is_results:
                    result_lines.append(line)

            for line in result_lines:
                #print(line)

                # if "Space-time Dofs" in line: get space-time dofs number
                if "Space-time Dofs" in line:
                    space_time_dofs = int(line.split(":")[-1].strip().replace(",", ""))
                    results[convergence_criterion][n_time_p]["dofs"] = space_time_dofs

                # if "J(u_h)" in line: get J(u_h)
                if "J(u_h):" in line:
                    J_u_h = float(line.split(":")[-1].strip())
                    results[convergence_criterion][n_time_p]["J_u_h"] = J_u_h

                # if "|J(u)-J(u_h)|" in line: get |J(u)-J(u_h)|
                if "|J(u)-J(u_h)|:" in line:
                    J_u_diff = float(line.split(":")[-1].strip())
                    # absolute error
                    results[convergence_criterion][n_time_p]["J_u_diff"] = J_u_diff
                    # relative error
                    results[convergence_criterion][n_time_p]["J_u_diff_rel"] = J_u_diff / J_u

                # if "J(p_h)" in line: get J(p_h)
                if "J(p_h):" in line:
                    J_p_h = float(line.split(":")[-1].strip())
                    results[convergence_criterion][n_time_p]["J_p_h"] = J_p_h

                # if "|J(p)-J(p_h)|" in line: get |J(p)-J(p_h)|
                if "|J(p)-J(p_h)|:" in line:
                    J_p_diff = float(line.split(":")[-1].strip())
                    # absolute error
                    results[convergence_criterion][n_time_p]["J_p_diff"] = J_p_diff
                    # relative error
                    results[convergence_criterion][n_time_p]["J_p_diff_rel"] = J_p_diff / J_p
                           
                # if "CPU Time" in line: get CPU time
                if "CPU Time" in line:
                    cpu_time = float(line.split(":")[-1].strip().split(" ")[0])
                    results[convergence_criterion][n_time_p]["cpu_time"] = cpu_time
                
                # if "Setup Time" in line: get setup time
                if "Setup Time" in line:
                    setup_time = float(line.split(":")[-1].strip().split(" ")[0])
                    results[convergence_criterion][n_time_p]["setup_time"] = setup_time

                # if "Flow Solve Time" in line: get flow solve time
                if "Flow Solve Time" in line:
                    flow_solve_time = float(line.split(":")[-1].strip().split(" ")[0])
                    results[convergence_criterion][n_time_p]["flow_solve_time"] = flow_solve_time

                # if "Mechanics Solve Time" in line: get mechanics solve time
                if "Mechanics Solve Time" in line:
                    mechanics_solve_time = float(line.split(":")[-1].strip().split(" ")[0])
                    results[convergence_criterion][n_time_p]["mechanics_solve_time"] = mechanics_solve_time

                # if "Conv. Criterion Time" in line: get convergence criterion time
                if "Conv. Criterion Time" in line:
                    conv_criterion_time = float(line.split(":")[-1].strip().split(" ")[0])
                    results[convergence_criterion][n_time_p]["conv_criterion_time"] = conv_criterion_time

#################################
# create a lineplot for timings #
#################################
fig, ax = plt.subplots()
ax.set_xlabel("Number of pressure time steps per displacement time step")
ax.set_ylabel("Time [s]")
ax.set_title("CPU Time")
ax.grid(True)

for convergence_criterion in results:
    n_time_p = []
    cpu_time = []
    setup_time = []
    flow_solve_time = []
    mechanics_solve_time = []
    conv_criterion_time = []
    for n_time_p_ in results[convergence_criterion]:
        n_time_p.append(n_time_p_)
        cpu_time.append(results[convergence_criterion][n_time_p_]["cpu_time"])
        setup_time.append(results[convergence_criterion][n_time_p_]["setup_time"])
        flow_solve_time.append(results[convergence_criterion][n_time_p_]["flow_solve_time"])
        mechanics_solve_time.append(results[convergence_criterion][n_time_p_]["mechanics_solve_time"])
        conv_criterion_time.append(results[convergence_criterion][n_time_p_]["conv_criterion_time"])
    # CPU time
    ax.scatter(n_time_p, cpu_time, label=f"{convergence_criterion} (Total)")
    ax.plot(n_time_p, cpu_time)
    # setup time
    #ax.scatter(n_time_p, setup_time, label=f"{convergence_criterion} (Setup)")
    #ax.plot(n_time_p, setup_time)
    # flow solve time
    ax.scatter(n_time_p, flow_solve_time, label=f"{convergence_criterion} (Flow Solve)")
    ax.plot(n_time_p, flow_solve_time)
    # mechanics solve time
    ax.scatter(n_time_p, mechanics_solve_time, label=f"{convergence_criterion} (Mechanics Solve)")
    ax.plot(n_time_p, mechanics_solve_time)
    # convergence criterion time
    #ax.scatter(n_time_p, conv_criterion_time, label=f"{convergence_criterion} (Conv. Criterion)")
    #ax.plot(n_time_p, conv_criterion_time)
    # make axes log log
    ax.set_xscale("log")
    ax.set_yscale("log")

ax.legend()
#fig.savefig("results/Mandel/cpu_time.png")
#plt.close(fig)
plt.show()

#########################################
# create a stacked barchart for timings #
#########################################

# for each convergence criterion, create a stacked bar chart
# the bar chart should show the total time broken up into setup, flow solve, mechanics solve, and convergence criterion

for convergence_criterion in results:
    n_time_p = []
    cpu_time = []
    setup_time = []
    flow_solve_time = []
    mechanics_solve_time = []
    conv_criterion_time = []
    for n_time_p_ in results[convergence_criterion]:
        n_time_p.append(n_time_p_)
        cpu_time.append(results[convergence_criterion][n_time_p_]["cpu_time"])
        setup_time.append(results[convergence_criterion][n_time_p_]["setup_time"])
        flow_solve_time.append(results[convergence_criterion][n_time_p_]["flow_solve_time"])
        mechanics_solve_time.append(results[convergence_criterion][n_time_p_]["mechanics_solve_time"])
        conv_criterion_time.append(results[convergence_criterion][n_time_p_]["conv_criterion_time"])

    fig, ax = plt.subplots()
    ax.set_xlabel(rf"#$p$-steps per $u$-step")
    ax.set_ylabel("CPU Time [s]")
    # ax.set_title(f"CPU Time") #({convergence_criterion})")

    # calculate the width of each bar
    width = [0.75 for _ in range(len(n_time_p))]
    n_time_p_i = [i+1 for i in range(len(n_time_p))]

    # create stacked bar chart with adjusted width
    ax.bar(n_time_p_i, flow_solve_time, width=width, label="Flow Solve", edgecolor='white')
    ax.bar(n_time_p_i, mechanics_solve_time, bottom=np.array(flow_solve_time), width=width, label="Mechanics Solve", edgecolor='white')
    ax.bar(n_time_p_i, conv_criterion_time, bottom=np.array(flow_solve_time)+np.array(mechanics_solve_time), width=width, label="Convergence Criterion", edgecolor='white')
    ax.bar(n_time_p_i, cpu_time-(np.array(flow_solve_time)+np.array(mechanics_solve_time)+np.array(conv_criterion_time)), bottom=np.array(flow_solve_time)+np.array(mechanics_solve_time)+np.array(conv_criterion_time), width=width, label="Other", edgecolor='white')

    # make y-axis log
    #ax.set_yscale("log")

    # set ymax to 350
    ax.set_ylim(ymax=350)

    # set y-ticks
    #yticks = [40,50,60,70,80,90,100,200,300]
    #ax.set_yticks(yticks)
    #ax.set_yticklabels(yticks)

    # set x-ticks and labels
    ax.set_xticks(n_time_p_i)
    ax.set_xticklabels(n_time_p)

    ax.legend()
    # save figure as pdf
    fig.savefig(f"results/Mandel/cpu_time_stacked_{convergence_criterion}.pdf")
    # fig.savefig(f"results/Mandel/cpu_time_stacked_{convergence_criterion}.png")
    # plt.show()
    plt.close(fig)

# create LaTeX table for CPU time
print("CPU TIMES:")
for convergence_criterion in results:
    print(f"Table for {convergence_criterion}:")
    print("\\begin{tabular}{|c|c|c|c|c|}")
    print("\\hline")
    print(" & \\multicolumn{4}{c|}{\\textbf{CPU Time [s]}} \\\\")
    print("\\hline")
    print("\\textbf{Pressure Steps per Mechanics Step} & \\textbf{Flow Solve} & \\textbf{Mechanics Solve} & \\textbf{Converegence Criterion} & \\textbf{Total} \\\\")
    print("\\hline")

    n_time_p = []
    cpu_time = []
    setup_time = []
    flow_solve_time = []
    mechanics_solve_time = []
    conv_criterion_time = []
    for n_time_p_ in results[convergence_criterion]:
        n_time_p.append(n_time_p_)
        cpu_time.append(results[convergence_criterion][n_time_p_]["cpu_time"])
        setup_time.append(results[convergence_criterion][n_time_p_]["setup_time"])
        flow_solve_time.append(results[convergence_criterion][n_time_p_]["flow_solve_time"])
        mechanics_solve_time.append(results[convergence_criterion][n_time_p_]["mechanics_solve_time"])
        conv_criterion_time.append(results[convergence_criterion][n_time_p_]["conv_criterion_time"])

    for i in range(len(n_time_p)):
        print(f"{n_time_p[i]} & {flow_solve_time[i]} & {mechanics_solve_time[i]} & {conv_criterion_time[i]} & {cpu_time[i]:.5} \\\\")
    
    print("\\hline")
    print("\\end{tabular}\n\n")

# plot the error in J(u) and J(p) for each convergence criterion
fig, ax = plt.subplots()
ax.set_xlabel("Number of pressure time steps per displacement time step")
ax.set_ylabel("Relative Error [%]")
ax.set_title("Relative Error in QoI")
ax.grid(True)

for convergence_criterion in results:
    n_time_p = []
    J_u_diff_rel = []
    J_p_diff_rel = []
    for n_time_p_ in results[convergence_criterion]:
        n_time_p.append(n_time_p_)
        J_u_diff_rel.append(100. * results[convergence_criterion][n_time_p_]["J_u_diff_rel"])
        J_p_diff_rel.append(100. * results[convergence_criterion][n_time_p_]["J_p_diff_rel"])
    n_time_p_i = [i+1 for i in range(len(n_time_p))]
    # J(u)
    label_convergence_criterion = "$\ell_2$-norm" if convergence_criterion == "solution" else "residual"
    ax.scatter(n_time_p_i[::-1], J_u_diff_rel, label=rf"$J_u$, {label_convergence_criterion}")
    ax.plot(n_time_p_i[::-1], J_u_diff_rel)
    # J(p)
    ax.scatter(n_time_p_i[::-1], J_p_diff_rel, label=rf"$J_p$, {label_convergence_criterion}")
    ax.plot(n_time_p_i[::-1], J_p_diff_rel)
    # make y-axis log
    #ax.set_xscale("log")
    ax.set_yscale("log")

    # set y-ticks
    yticks = [10**i for i in range(-14, 2)]
    ax.set_yticks(yticks)

    # set x-ticks and labels
    ax.set_xticks(n_time_p_i[::-1])
    ax.set_xticklabels(n_time_p)  #[::-1])

ax.legend()
fig.savefig("results/Mandel/qoi_error.pdf")
plt.close(fig)
# plt.show()

# create LaTeX table for QoI error
print("QOI ERRORS:")

for convergence_criterion in results:
    print(f"Table for {convergence_criterion}:")
    print("\\begin{tabular}{|c|c|c|c|c|c|}")
    print("\\hline")
    print(" & \\multicolumn{2}{c|}{$J_p$} & \\multicolumn{2}{c|}{$J_p$} & \\\\")
    print("\\hline")
    print("\\textbf{Pressure Steps per Mechanics Step} & \\textbf{Absolute Error} & \\textbf{Relative Error [%]} & \\textbf{Absolute Error} & \\textbf{Relative Error [%]} &  #\\textbf{DoFs}  \\\\")
    print("\\hline")

    n_time_p = []
    J_u_diff = []
    J_u_diff_rel = []
    J_p_diff = []
    J_p_diff_rel = []
    dofs = []
    for n_time_p_ in results[convergence_criterion]:
        n_time_p.append(n_time_p_)
        J_u_diff.append(results[convergence_criterion][n_time_p_]["J_u_diff"])
        J_u_diff_rel.append(100. * results[convergence_criterion][n_time_p_]["J_u_diff_rel"])
        J_p_diff.append(results[convergence_criterion][n_time_p_]["J_p_diff"])
        J_p_diff_rel.append(100. * results[convergence_criterion][n_time_p_]["J_p_diff_rel"])
        dofs.append(results[convergence_criterion][n_time_p_]["dofs"])

    for i in range(len(n_time_p)):
        J_u_diff_str = f"{J_u_diff[i]:.3e}"
        J_p_diff_str = f"{J_p_diff[i]:.3e}"
        J_u_diff_str = "$" + J_u_diff_str.replace("e", "\\cdot 10^{").replace("+0", "").replace("-0", "-").replace("+","") + "}$"
        J_p_diff_str = "$" + J_p_diff_str.replace("e", "\\cdot 10^{").replace("+0", "").replace("-0", "-").replace("+","") + "}$"
        print(f"{n_time_p[i]} & {J_u_diff_str} & {J_u_diff_rel[i]:.4f} & {J_p_diff_str} & {J_p_diff_rel[i]:.4f} & {dofs[i]:,} \\\\")

    print("\\hline")
    print("\\end{tabular}\n\n")

