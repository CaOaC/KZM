# KZM Perversion

CUDA simulation and Python analysis code for *Dynamics of Symmetry Breaking and Defect Generation in a Single Ideal Motorized Chromosome as It Undergoes Mitosis*.

In the model, increasing condensin activity leads to the formation of left- and right-handed helical domains along a single idealized chromosome. A **perversion** is the boundary between domains of opposite handedness. The study examines how quickly these boundaries form as motor activity increases and how they move and persist afterward. For slow activation, the mean perversion count scales approximately as $t_c^{-0.484}$, where $t_c$ is the time to reach the chiral ordering transition. At faster activation rates, the count approaches a plateau.

## Dynamics trajectory

[![Three simulated chromosome copies fold and develop red and blue domains of opposite handedness](assets/dynamics-trajectory.gif)](assets/dynamics-trajectory.mp4)

Three simulated chromosome copies develop domains of opposite handedness during finite-rate activation. Red and blue indicate the two local handedness states. [Download the original MP4](assets/dynamics-trajectory.mp4?raw=true).

## Activation ramp protocols

![Schematic of fast, intermediate, and slow motor-activity ramps crossing the same critical activity at different times](assets/activation-ramp-protocols.png)

The schematic shows three linear ramps of condensin-II activity, $\kappa_{\mathrm{II}}(t) = Rt$. Faster ramps reach the critical activity $\kappa_c$ sooner, at $t_c = \kappa_c/R$.

## Setup

```bash
git clone git@github.com:CaOaC/KZM_Perversion.git
cd KZM_Perversion
python -m pip install numpy scipy matplotlib numba
```

The trajectory and frame analysis scripts require an NVIDIA GPU and a working Numba CUDA environment. To check whether Numba detects CUDA, run `python -c "from numba import cuda; print(cuda.is_available())"`. The plotting scripts can run from existing analysis tables on a CPU. Folder dialogs require Tkinter and a graphical desktop. For headless use, set paths in the scripts and disable the folder dialogs.

## CUDA simulation

[`simulation/activation_ramp/`](simulation/activation_ramp/) contains one 5,000-bead interphase-start configuration. In [`main.cu`](simulation/activation_ramp/main.cu), condensin-I and condensin-II activities start at zero and increase by $5\times10^{-4}$ and $10^{-4}$ per scheduled update, respectively. This configuration uses harmonic bonds and defaults to `p_slide=0`. The `--p_slide` option changes the motor sliding probability.

Build on Linux with GNU Make, a C++17 compiler, and the NVIDIA CUDA Toolkit. The [Makefile](simulation/activation_ramp/Makefile) targets `sm_80` by default. Set `ARCH_SM` for your GPU when needed.

```bash
cd simulation/activation_ramp
make
```

Supply the starting conformation separately as `Input/Interphase.txt` in the simulator directory. The input has 5,000 headerless records in polymer contour order, each with two integers, three Cartesian coordinates, and four trailing integers. Run the simulator from the same directory.

```bash
mkdir -p Output
./kickModel --ensemble 0 --seed 114514
```

The simulator writes `Output/trajectory_tau*_v*_ems*.dump` and `Output/kappavstime_*.txt`. The ramp trajectory filename records the `p10` and `p_slide` arguments in its `tau` and `v` fields. To analyze a ramp with the frame-based workflow below, export the dump frames as numbered coordinate tables and keep the corresponding activity log with them.

## Trajectory inputs for fixed-activity analysis

Supply LAMMPS dump files named with the motor residence time (`tau`), sliding speed (`v`), type-I and type-II motor activities (`kI` and `kII`), and replicate index (`ems`). For example:

```text
trajectory_tau1.0000_v0.0000_kI0.0045_kII0.0009_ems46.dump
```

Both dump readers sort beads by atom ID, so IDs should follow polymer contour order. Use Cartesian coordinates and, when the chain crosses periodic boundaries, provide unwrapped coordinates (`id xu yu zu`) as the coordinate columns so the chain stays continuous. Use a common dump time grid for replicates in the same parameter group.

## Analysis workflows

### Chirality and perversion kinetics

In [`batch_perversion_gpu_only_checked_v2.py`](batch_perversion_gpu_only_checked_v2.py), set `DUMP_DIR` in the `__main__` configuration block or select the trajectory folder when prompted. The default output path is `DUMP_DIR/analysis_out`. Set `FRAME_STRIDE` and `SIM_DT` to match the dump sampling and simulation time units before analyzing wall motion. Run:

```bash
python batch_perversion_gpu_only_checked_v2.py
```

The script groups trajectories by `(tau, v, kI, kII)` and writes `kzm_kinetics_robust.txt` for each replicate. The table contains chirality order parameters, perversion counts, domain sizes, wall widths, diffusion estimates, and nucleation and annihilation counts. Each parameter group also has `ensemble_mean.txt`, `ensemble_std.txt`, and `ensemble_summary.txt`.

Point `ROOT_DIR` in [`plot_perversion_results_nocli.py`](plot_perversion_results_nocli.py) at `analysis_out` to plot trajectory and ensemble time series. Use the same root in [`plot_steady_observables_vs_kII_nocli.py`](plot_steady_observables_vs_kII_nocli.py) to plot steady-state observables versus type-II motor activity. Run each plotter with `python <script-name>.py`.

### Perversion boundary motion

In [`perversion_boundary_msd_gpu_nocli.py`](perversion_boundary_msd_gpu_nocli.py), set `DUMP_DIR` or choose the trajectory folder when prompted. Check `FRAME_STRIDE`, `SIM_DT`, and `LAG_STEPS` for your data. Run:

```bash
python perversion_boundary_msd_gpu_nocli.py
```

The script measures mean squared displacement (MSD) of boundaries along the polymer contour and fits $\mathrm{MSD}(\Delta t) = A\,\Delta t^\alpha$. By default, it writes `boundary_msd_by_lag.txt` for each replicate and `ensemble_boundary_msd.txt` and `ensemble_boundary_msd_fit.txt` for each parameter group under `DUMP_DIR/perversion_boundary_msd_out`.

Point `ROOT_DIR` in [`plot_boundary_msd_alpha_nocli.py`](plot_boundary_msd_alpha_nocli.py) or [`alpha_slope_versus_kII.py`](alpha_slope_versus_kII.py) at `perversion_boundary_msd_out` to summarize the ensemble fits. The companion script [`alpha_slope_versus_kII_II.py`](alpha_slope_versus_kII_II.py) reads the alternate `ensemble_msd_fit.txt` format.

### Individual frame files

The two `chirality_*.py` scripts read numbered `frame*.txt` files in the working directory, with Cartesian coordinates in columns 2–4 by default. The basic script writes `observables.txt` and chirality plots. The `domain wall` variant adds wall kinetics and can use `kappavstime_0.txt` to relate chirality to a motor-activity ramp.

## Citation

Zhiyu Cao, Chaoqun Du, Zhonghuai Hou, and Peter G. Wolynes. “Dynamics of Symmetry Breaking and Defect Generation in a Single Ideal Motorized Chromosome as It Undergoes Mitosis.” Manuscript, 14 November 2026.
