import matplotlib.pyplot as plt
import numpy as np
import sys

#print(sys.argv[1:])
name = None
file_name = None
if len(sys.argv) > 1 and sys.argv[1] == "--name":
    name = sys.argv[2]
    #print(name)

# CYCLE = 3
CYCLE = 0

TIMES = [1_000, 5_000, 10_000, 100_000, 500_000, 5_000_000]


if name is not None:
    #print(name)
    dG = name[:5].replace("(","").replace(")","")
    u,p = name.split("=")[1].strip(" ").split(":")
    file_name = f"{dG}_{u}u_{p}p.png"

logged_times = []
pressure_coordinates = []
pressure_values = []
displacement_coordinates = []
displacement_values = []

# pressure file
with open(f"output/dim=2/cycle={CYCLE}/pressure.txt", "r") as f:
    for i, line in enumerate(f):
        # print(i, line)
        if i == 1:
            # get x-coordinates of DoFs at boundary
            pressure_coordinates = [float(val) for val in line.strip(" ").strip("\n").split(" ")]

        if i > 3:
            # get time point and pressure values at DoFs
            _time, _values = line.strip("\n").split(" | ")
            try:
                logged_times.append(int(_time))
            except ValueError:
                logged_times.append(int(float(_time)))
            pressure_values.append([float(v) for v in _values.split(" ")])

# code for approximate verification of QoI computation in C++
#avg_pressure = 0.
#for p in pressure_values:
#    avg_pressure += logged_times[-1] * (100. * np.sum(p) / len(p)) / (len(logged_times)-1)
#print("avg pressure: ", avg_pressure)

# displacement file
with open(f"output/dim=2/cycle={CYCLE}/x_displacement.txt", "r") as f:
    for i, line in enumerate(f):
        # print(i, line)
        if i == 1:
            # get x-coordinates of DoFs at boundary
            displacement_coordinates = [
                float(val) for val in line.strip(" ").strip("\n").split(" ")
            ]

        if i > 3:
            # get time point and pressure values at DoFs
            _, _values = line.strip("\n").split(" | ")
            displacement_values.append([float(v) for v in _values.split(" ")])

# pressure plot
last_i = 0
for t in TIMES:
    for i in range(last_i, len(logged_times)):
        if np.abs(logged_times[i] - t) < 5.0:
            last_i = i
            # print(logged_times[i], pressure_values[i]) #,
            # displacement_values[i])
            plt.plot(pressure_coordinates, pressure_values[i], label="t = " + str(logged_times[i]))
            break

plt.title("Pressure")
if name is not None:
    plt.title(f"Pressure [{name}]")
plt.legend()
if file_name is None:
    plt.show()
else:
    plt.savefig(f"output/dim=2/cycle={CYCLE}/pressure_{file_name}")
    plt.clf()

# pressure plot
last_i = 0
for t in TIMES:
    for i in range(last_i, len(logged_times)):
        if np.abs(logged_times[i] - t) < 5.0:
            last_i = i
            # print(logged_times[i], pressure_values[i]) #,
            # displacement_values[i])
            str_time = str(logged_times[i])
            plt.scatter(logged_times[i], pressure_values[i][0], label=f"t = {str_time:>7}; p(0) = {pressure_values[i][0]}") #, color="blue")
            break

plt.plot([t for t in logged_times], [p[0] for p in pressure_values])
plt.title("Pressure at x = (0,0)")
if name is not None:
    plt.title(f"Pressure at x = (0,0) [{name}]")
plt.xscale("log")
#plt.yscale("log")
plt.legend()
if file_name is None:
    plt.show()
else:
    plt.savefig(f"output/dim=2/cycle={CYCLE}/pressure_origin_{file_name}")
    plt.clf()

# x-displacement plot
last_i = 0
for t in TIMES:
    for i in range(last_i, len(logged_times)):
        if np.abs(logged_times[i] - t) < 5.0:
            last_i = i
            # print(logged_times[i], pressure_values[i]) #,
            # displacement_values[i])
            plt.plot(
                displacement_coordinates,
                displacement_values[i],
                label="t = " + str(logged_times[i]),
            )
            break

plt.title("x-Displacement")
if name is not None:
    plt.title(f"x-Displacement [{name}]")
plt.legend()
if file_name is None:
    plt.show()
else:
    plt.savefig(f"output/dim=2/cycle={CYCLE}/displacement_{file_name}")
    plt.clf()
