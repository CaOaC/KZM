# KZM Perversion

CUDA simulator for *Dynamics of Symmetry Breaking and Defect Generation in a Single Ideal Motorized Chromosome as It Undergoes Mitosis*.

As condensin activity increases, an idealized chromosome develops left- and right-handed helical domains. The boundaries between domains of opposite handedness are called perversions. The study examines how the rate of condensin activation affects their formation and persistence.

## Environment

The [CUDA simulator](simulation/activation_ramp/) builds on Linux with GNU Make, the NVIDIA CUDA Toolkit including `nvcc` and cuRAND, and a CUDA-compatible host compiler with C++17 support. Running the simulator requires an NVIDIA GPU and a compatible driver.

The build was verified with CUDA 12.2 and GCC 7.3.1. The Makefile targets `sm_80` by default. Set `ARCH_SM` to the compute capability of your GPU without the decimal point. Run `make clean` before rebuilding for a different GPU architecture.

## Build

```bash
git clone git@github.com:CaOaC/KZM_Perversion.git
cd KZM_Perversion/simulation/activation_ramp
make
```

The build produces the executable `kickModel` in the current directory.

## Input

Supply the starting conformation separately as `Input/Interphase.txt` relative to the working directory. For the default simulation, it must contain 5,000 whitespace-delimited rows in polymer contour order. Each row has nine fields and the file has no header.

```text
integer integer x y z integer integer integer integer
```

The simulator reads Cartesian coordinates from columns 3–5. The remaining columns are parsed as integers and discarded. Bead order comes from row order.

## Run

Run the example from the directory containing `kickModel` and `Input/Interphase.txt`.

```bash
mkdir -p Output
./kickModel --ensemble 0 --seed 114514
```

| Option | Meaning |
| --- | --- |
| `--ensemble` | Replicate ID written to the trajectory and activity-log filenames. |
| `--seed` | Random seed for the simulation. |
| `--p10` | Motor detachment parameter. Default is `1`. |
| `--p_slide` | Motor sliding parameter. Default is `0`. |

This CUDA configuration uses harmonic bonds. Condensin-I and condensin-II activities start at zero and increase by $5 \times 10^{-4}$ and $10^{-4}$, respectively, at each scheduled update. They are capped at `0.05` and `0.01`. The schedule is defined in [main.cu](simulation/activation_ramp/main.cu).

Use a separate working directory for each simultaneous run. Some output filenames are shared across replicate IDs and are overwritten at startup.

## Output

| File in `Output/` | Contents |
| --- | --- |
| `trajectory_tau*_v*_ems*.dump` | LAMMPS-style trajectory with simulation timestamps, box bounds, bead IDs and types, Cartesian coordinates, condensin-I and condensin-II kick counts, and periodic image flags. The `tau` and `v` filename fields store the values of `p10` and `p_slide`. |
| `kappavstime_<ensemble>.txt` | Activity log. Each row starts with the kick-cycle count, followed by a condensin-I and condensin-II activity pair for each of the ten contour segments. |
| `kicknumber.txt` | Condensin-I kick count for each bead at each saved output step. |
| `TOPO_kappa_*.data` | Initial coordinates and polymer bond connectivity in LAMMPS data format. |
| `para_kappa*.txt` | Run parameters, including bead counts, model constants, box length, and the simulation time limit. |

The simulator writes trajectory frames, activity records, and kick counts every 100 kick cycles during the active run. Trajectory timestamps use simulation time, and the activity log uses the kick-cycle count.

## Dynamics trajectory

[![Three simulated chromosome copies fold and develop red and blue domains of opposite handedness](assets/dynamics-trajectory.gif)](assets/dynamics-trajectory.mp4)

Three simulated chromosome copies develop domains of opposite handedness during finite-rate activation. Red and blue indicate the two local handedness states. [Download the original MP4](assets/dynamics-trajectory.mp4?raw=true).

## Activation ramp protocols

![Schematic of fast, intermediate, and slow motor-activity ramps crossing the same critical activity at different times](assets/activation-ramp-protocols.png)

The schematic shows three linear ramps of condensin-II activity, $\kappa_{\mathrm{II}}(t) = Rt$. Faster ramps reach the critical activity $\kappa_c$ sooner, at $t_c = \kappa_c/R$.

## Citation

Zhiyu Cao, Chaoqun Du, Zhonghuai Hou, and Peter G. Wolynes. “Dynamics of Symmetry Breaking and Defect Generation in a Single Ideal Motorized Chromosome as It Undergoes Mitosis.” Manuscript, 14 November 2026.
