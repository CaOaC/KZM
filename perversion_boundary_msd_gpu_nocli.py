#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Boundary-region perversion MSD scaling analysis, GPU version, no command line needed.

This script computes one-dimensional contour-space diffusion of +/- chirality
boundary regions:
    MSD(Delta t) = <[s(t+Delta t)-s(t)]^2>
and fits:
    MSD = A * Delta_t^alpha

Input filename format:
    trajectory_tau1.0000_v0.0000_kI0.0045_kII0.0009_ems46.dump

Outputs are saved under:
    perversion_boundary_msd_out/
"""

from pathlib import Path
from dataclasses import dataclass
from collections import defaultdict
import re
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.optimize import linear_sum_assignment
from numba import cuda

# ============================================================
# User configuration
# ============================================================
DUMP_DIR = r""      # e.g. r"G:\chromosome\domainwall\dumps"; empty => folder dialog
OUT_DIR = r""       # empty => DUMP_DIR / "perversion_boundary_msd_out"
USE_FOLDER_DIALOG = True

FILE_PATTERN = "trajectory_*.dump"
RECURSIVE = False

# Chirality parameters
T_SCALE = 156.25
SG_WINDOW = 51
SG_POLY = 3

# Boundary detection parameters
PERV_EPS = 0.05
MIN_DOMAIN_SIZE = 10

# Frame sampling
FRAME_STRIDE = 20
SIM_DT = 1.0

# Lag steps in analyzed-frame sequence.
# Example: FRAME_STRIDE=20, SIM_DT=1, lag_step=3 => Delta t = 60.
LAG_STEPS = [1, 2, 3, 4, 5, 8, 10, 15, 20, 30, 40, 50]

# Matching boundary centers between two frames
SEARCH_RADIUS = 80.0
LATTICE_SPACING = 1.0

# Power-law fit range in Delta t
FIT_DT_MIN = None
FIT_DT_MAX = None
MIN_FIT_POINTS = 4

# Plot settings
PLOT_SINGLE_TRAJECTORIES = True
FIG_DPI = 300
SHOW_FIGURES = False


@dataclass(frozen=True)
class ParamKey:
    tau: float
    v: float
    kI: float
    kII: float

    def tag(self):
        return f"tau{self.tau:.5f}_v{self.v:.5f}_kI{self.kI:.5f}_kII{self.kII:.5f}"


@dataclass
class TrajectoryInfo:
    path: Path
    key: ParamKey
    ems: int


# ============================================================
# CUDA chirality kernel
# ============================================================
@cuda.jit(device=True)
def _magnitude(v):
    return math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])


@cuda.jit(device=True)
def _cross(a, b):
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


@cuda.jit(device=True)
def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


@cuda.jit(device=True)
def _sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


@cuda.jit(device=True)
def _add(a, b):
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


@cuda.jit(device=True)
def _div(a, c):
    return (a[0] / c, a[1] / c, a[2] / c)


@cuda.jit
def calculate_phi_kernel(beads, phis, T):
    i = cuda.grid(1)
    max_i = beads.shape[0] - int(1.25 * T)

    if i < max_i:
        A = beads[i]
        B = beads[int(i + 0.5 * T)]
        C = beads[int(i + 0.75 * T)]
        D = beads[int(i + 1.25 * T)]

        E = _div(_add(A, B), 2.0)
        F = _div(_add(C, D), 2.0)

        AB = _sub(B, A)
        CD = _sub(D, C)
        EF = _sub(F, E)

        cp = _cross(CD, AB)
        dp = _dot(EF, cp)

        denom = _magnitude(EF) * _magnitude(CD) * _magnitude(AB)
        if denom > 1e-12:
            phis[i] = dp / denom
        else:
            phis[i] = 0.0


def compute_phi_gpu(beads, T, threadsperblock=128):
    beads = np.ascontiguousarray(beads, dtype=np.float32)
    n_beads = beads.shape[0]
    n_phi = n_beads - int(1.25 * T)
    if n_phi <= 5:
        raise ValueError(f"n_phi={n_phi} too small. Check T_SCALE={T} and bead number={n_beads}.")

    phis = np.zeros(n_phi, dtype=np.float32)
    d_beads = cuda.to_device(beads)
    d_phis = cuda.to_device(phis)
    blocks = (n_phi + threadsperblock - 1) // threadsperblock
    calculate_phi_kernel[blocks, threadsperblock](d_beads, d_phis, float(T))
    cuda.synchronize()
    return d_phis.copy_to_host()


# ============================================================
# UI and IO
# ============================================================
def pick_folder_by_dialog(title):
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        folder = filedialog.askdirectory(title=title)
        root.destroy()
        return folder
    except Exception as e:
        print(f"[WARN] Cannot open folder dialog: {e}")
        return ""


def parse_dump_filename(path):
    name = Path(path).name
    pat = (
        r"trajectory_"
        r"tau(?P<tau>[-+0-9.eE]+)_"
        r"v(?P<v>[-+0-9.eE]+)_"
        r"kI(?P<kI>[-+0-9.eE]+)_"
        r"kII(?P<kII>[-+0-9.eE]+)_"
        r"ems(?P<ems>\d+)\.dump$"
    )
    m = re.match(pat, name)
    if m is None:
        return None
    key = ParamKey(float(m.group("tau")), float(m.group("v")), float(m.group("kI")), float(m.group("kII")))
    return TrajectoryInfo(path=Path(path), key=key, ems=int(m.group("ems")))


def iter_lammps_dump_frames(dump_path):
    with open(dump_path, "r", encoding="utf-8", errors="ignore") as f:
        while True:
            line = f.readline()
            if not line:
                break
            if line.strip() != "ITEM: TIMESTEP":
                continue

            timestep_line = f.readline()
            if not timestep_line:
                break
            timestep = float(timestep_line.strip())

            natoms = None
            atom_fields = None
            while True:
                line = f.readline()
                if not line:
                    break
                s = line.strip()
                if not s:
                    continue
                if s.startswith("ITEM: NUMBER OF ATOMS"):
                    natoms = int(float(f.readline().strip()))
                elif s.startswith("ITEM: BOX BOUNDS"):
                    for _ in range(3):
                        f.readline()
                elif s.startswith("ITEM: ATOMS"):
                    atom_fields = s.split()[2:]
                    break

            if atom_fields is None:
                break
            if natoms is None:
                raise ValueError(f"Missing NUMBER OF ATOMS before ATOMS in {dump_path}")

            try:
                id_col = atom_fields.index("id")
                if "x" in atom_fields:
                    x_col, y_col, z_col = atom_fields.index("x"), atom_fields.index("y"), atom_fields.index("z")
                elif "xu" in atom_fields:
                    x_col, y_col, z_col = atom_fields.index("xu"), atom_fields.index("yu"), atom_fields.index("zu")
                elif "xs" in atom_fields:
                    x_col, y_col, z_col = atom_fields.index("xs"), atom_fields.index("ys"), atom_fields.index("zs")
                else:
                    raise ValueError
            except ValueError as e:
                raise ValueError(f"Cannot find id/x/y/z columns. ATOMS fields = {atom_fields}") from e

            atoms = np.empty((natoms, 4), dtype=np.float64)
            for i in range(natoms):
                parts = f.readline().split()
                atoms[i, 0] = int(float(parts[id_col]))
                atoms[i, 1] = float(parts[x_col])
                atoms[i, 2] = float(parts[y_col])
                atoms[i, 3] = float(parts[z_col])

            atoms = atoms[np.argsort(atoms[:, 0])]
            yield {"timestep": timestep, "natoms": natoms, "atoms": np.ascontiguousarray(atoms)}


# ============================================================
# Boundary-region detection
# ============================================================
def smooth_phi(phi, window=51, polyorder=3):
    phi = np.asarray(phi, dtype=np.float64)
    n = len(phi)
    if n < 5:
        return phi
    w = int(window)
    if w >= n:
        w = (n // 2) * 2 - 1
    if w < 5:
        w = 5
    if w % 2 == 0:
        w += 1
    if w >= n:
        w = n - 1 if (n - 1) % 2 == 1 else n - 2
    if w < 5:
        return phi
    return savgol_filter(phi, w, int(polyorder))


def fill_zero_signs(sign):
    s = np.asarray(sign, dtype=np.int8).copy()
    last = 0
    for i in range(len(s)):
        if s[i] == 0:
            s[i] = last
        else:
            last = s[i]
    if len(s) > 0 and s[0] == 0:
        nz = np.nonzero(s)[0]
        if len(nz) > 0:
            s[:nz[0]] = s[nz[0]]
    return s


def find_boundary_regions(phi_smooth, eps=0.05, min_domain_size=10):
    """
    Identify +/- boundary-region centers along the chain.

    Returns:
      centers: array of boundary centers in contour index units
      details: tuples (left_start, left_end, right_start, right_end, center, transition_width)
    """
    phi_smooth = np.asarray(phi_smooth, dtype=np.float64)
    n = len(phi_smooth)
    raw = np.zeros(n, dtype=np.int8)
    raw[phi_smooth > eps] = 1
    raw[phi_smooth < -eps] = -1
    filled = fill_zero_signs(raw)
    if len(filled) == 0 or np.all(filled == 0):
        return np.array([], dtype=np.float64), []

    domains = []
    start = 0
    for i in range(1, n):
        if filled[i] != filled[start]:
            domains.append((start, i, filled[start]))
            start = i
    domains.append((start, n, filled[start]))
    domains = [(a, b, sg) for (a, b, sg) in domains if sg != 0 and (b - a) >= min_domain_size]

    centers = []
    details = []
    for i in range(len(domains) - 1):
        a1, b1, s1 = domains[i]
        a2, b2, s2 = domains[i + 1]
        if s1 * s2 != -1:
            continue
        center = 0.5 * (b1 - 1 + a2)

        left = b1 - 1
        right = a2
        while left > 0 and abs(phi_smooth[left]) < eps:
            left -= 1
        while right < n - 1 and abs(phi_smooth[right]) < eps:
            right += 1
        transition_width = max(1.0, float(right - left))

        centers.append(center)
        details.append((a1, b1, a2, b2, center, transition_width))
    return np.asarray(centers, dtype=np.float64), details


def compute_boundary_positions_for_dump(traj, outdir):
    times = []
    centers_list = []
    details_list = []

    for raw_frame_idx, frame in enumerate(iter_lammps_dump_frames(traj.path)):
        if raw_frame_idx % FRAME_STRIDE != 0:
            continue
        timestep = float(frame["timestep"])
        atoms = frame["atoms"]
        beads = np.ascontiguousarray(atoms[:, 1:4], dtype=np.float32)
        phi = compute_phi_gpu(beads, T_SCALE)
        phi_s = smooth_phi(phi, SG_WINDOW, SG_POLY)
        centers, details = find_boundary_regions(phi_s, eps=PERV_EPS, min_domain_size=MIN_DOMAIN_SIZE)
        times.append(timestep * SIM_DT)
        centers_list.append(centers)
        details_list.append(details)
        print(f"    frame={raw_frame_idx:<6d} t={timestep:<10.1f} N_boundary={len(centers)}")

    times = np.asarray(times, dtype=np.float64)
    with open(outdir / "boundary_positions.txt", "w", encoding="utf-8") as f:
        f.write("# time local_frame_index n_boundary boundary_centers...\n")
        for i, (t, centers) in enumerate(zip(times, centers_list)):
            f.write(f"{t} {i} {len(centers)}")
            for c in centers:
                f.write(f" {c}")
            f.write("\n")

    with open(outdir / "boundary_details.txt", "w", encoding="utf-8") as f:
        f.write("# time local_frame_index left_start left_end right_start right_end center transition_width\n")
        for i, (t, details) in enumerate(zip(times, details_list)):
            for d in details:
                a1, b1, a2, b2, center, width = d
                f.write(f"{t} {i} {a1} {b1} {a2} {b2} {center} {width}\n")
    return times, centers_list


# ============================================================
# MSD calculation
# ============================================================
def match_positions(pos_a, pos_b, search_radius):
    pos_a = np.asarray(pos_a, dtype=np.float64)
    pos_b = np.asarray(pos_b, dtype=np.float64)
    if len(pos_a) == 0 or len(pos_b) == 0:
        return np.array([], dtype=np.float64)
    cost = np.abs(pos_a[:, None] - pos_b[None, :])
    row_ind, col_ind = linear_sum_assignment(cost)
    valid = cost[row_ind, col_ind] <= float(search_radius)
    if not np.any(valid):
        return np.array([], dtype=np.float64)
    dx = (pos_b[col_ind[valid]] - pos_a[row_ind[valid]]) * float(LATTICE_SPACING)
    return dx * dx


def compute_msd_by_lag(times, positions, lag_steps):
    rows = []
    n = len(times)
    for lag in lag_steps:
        lag = int(lag)
        if lag <= 0 or lag >= n:
            continue
        sq_samples = []
        dt_samples = []
        for i in range(n - lag):
            sq = match_positions(positions[i], positions[i + lag], SEARCH_RADIUS)
            if len(sq) == 0:
                continue
            dt = times[i + lag] - times[i]
            if dt <= 0:
                continue
            sq_samples.extend(sq.tolist())
            dt_samples.extend([dt] * len(sq))
        if len(sq_samples) == 0:
            rows.append([lag, np.nan, np.nan, np.nan, 0, np.nan])
            continue
        sq_samples = np.asarray(sq_samples, dtype=np.float64)
        dt_samples = np.asarray(dt_samples, dtype=np.float64)
        dt_mean = float(np.mean(dt_samples))
        msd = float(np.mean(sq_samples))
        sem = float(np.std(sq_samples, ddof=1) / np.sqrt(len(sq_samples))) if len(sq_samples) > 1 else np.nan
        med = float(np.median(sq_samples))
        rows.append([lag, dt_mean, msd, sem, len(sq_samples), med])
    return np.asarray(rows, dtype=np.float64)


def fit_msd_scaling(msd_table):
    if msd_table.size == 0:
        return None
    dt = msd_table[:, 1]
    msd = msd_table[:, 2]
    n_samples = msd_table[:, 4]
    mask = np.isfinite(dt) & np.isfinite(msd) & (dt > 0) & (msd > 0) & (n_samples > 0)
    if FIT_DT_MIN is not None:
        mask &= dt >= float(FIT_DT_MIN)
    if FIT_DT_MAX is not None:
        mask &= dt <= float(FIT_DT_MAX)
    if np.sum(mask) < MIN_FIT_POINTS:
        return None
    x = np.log(dt[mask])
    y = np.log(msd[mask])
    alpha, logA = np.polyfit(x, y, 1)
    y_pred = alpha * x + logA
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    R2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    nfit = len(x)
    if nfit > 2:
        s2 = ss_res / (nfit - 2)
        sx2 = np.sum((x - np.mean(x)) ** 2)
        alpha_se = math.sqrt(s2 / sx2) if sx2 > 0 else np.nan
        logA_se = math.sqrt(s2 * (1.0 / nfit + np.mean(x) ** 2 / sx2)) if sx2 > 0 else np.nan
    else:
        alpha_se = np.nan
        logA_se = np.nan
    return {
        "alpha": float(alpha),
        "alpha_se": float(alpha_se),
        "A": float(np.exp(logA)),
        "logA": float(logA),
        "logA_se": float(logA_se),
        "R2": float(R2),
        "n_fit": int(nfit),
        "fit_dt_min": float(np.min(dt[mask])),
        "fit_dt_max": float(np.max(dt[mask])),
        "mask": mask,
    }


def save_fit(path, fit):
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Boundary-region MSD scaling fit\n")
        f.write("# Model: MSD(delta_t) = A * delta_t^alpha\n")
        if fit is None:
            f.write("fit_success 0\n")
            return
        f.write("fit_success 1\n")
        for k in ["alpha", "alpha_se", "A", "logA", "logA_se", "R2", "n_fit", "fit_dt_min", "fit_dt_max"]:
            f.write(f"{k} {fit[k]}\n")


def plot_msd(msd_table, fit, title, outpath):
    fig, ax = plt.subplots(figsize=(6.2, 4.8))
    if msd_table.size > 0:
        dt = msd_table[:, 1]
        msd = msd_table[:, 2]
        sem = msd_table[:, 3]
        n_samples = msd_table[:, 4]
        mask = np.isfinite(dt) & np.isfinite(msd) & (dt > 0) & (msd > 0) & (n_samples > 0)
        if np.any(mask):
            yerr = np.where(np.isfinite(sem), sem, 0.0)
            ax.errorbar(dt[mask], msd[mask], yerr=yerr[mask], marker="o", lw=1.8, capsize=3, label="boundary MSD")
        if fit is not None and np.any(mask):
            dt_fit = dt[fit["mask"]]
            if len(dt_fit) > 0:
                xline = np.logspace(np.log10(np.min(dt_fit)), np.log10(np.max(dt_fit)), 100)
                yline = fit["A"] * xline ** fit["alpha"]
                if np.isfinite(fit["alpha_se"]):
                    label = rf"fit: $\alpha={fit['alpha']:.3f}\pm{fit['alpha_se']:.3f}$"
                else:
                    label = rf"fit: $\alpha={fit['alpha']:.3f}$"
                ax.plot(xline, yline, "--", lw=2, label=label)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Lag time $\Delta t$")
    ax.set_ylabel(r"MSD$_s(\Delta t)$")
    ax.set_title(title)
    ax.legend(frameon=False)
    ax.tick_params(direction="in", top=True, right=True)
    ax.grid(alpha=0.25, which="both")
    fig.tight_layout()
    fig.savefig(outpath, dpi=FIG_DPI, bbox_inches="tight")
    if SHOW_FIGURES:
        plt.show()
    else:
        plt.close(fig)


# ============================================================
# Workflow
# ============================================================
def analyze_single_trajectory(traj, group_outdir):
    traj_outdir = group_outdir / f"ems{traj.ems:03d}"
    traj_outdir.mkdir(parents=True, exist_ok=True)
    print(f"  [TRAJ] {traj.path.name}")
    times, centers_list = compute_boundary_positions_for_dump(traj, traj_outdir)
    if len(times) < 2:
        print("    [WARN] Too few analyzed frames.")
        return None
    msd_table = compute_msd_by_lag(times, centers_list, LAG_STEPS)
    header = "lag_step delta_t msd msd_sem n_samples median_sq_disp"
    np.savetxt(traj_outdir / "boundary_msd_by_lag.txt", msd_table, header=header)
    fit = fit_msd_scaling(msd_table)
    save_fit(traj_outdir / "boundary_msd_fit.txt", fit)
    if PLOT_SINGLE_TRAJECTORIES:
        plot_msd(msd_table, fit, f"{traj.key.tag()} ems{traj.ems}", traj_outdir / "boundary_msd_scaling.png")
    return {"traj": traj, "msd_table": msd_table, "fit": fit}


def pool_ensemble_msd(single_results):
    lag_map = defaultdict(list)
    for res in single_results:
        if res is None:
            continue
        tab = res["msd_table"]
        for row in tab:
            lag = int(row[0])
            dt, msd, sem, n_samples, median = row[1], row[2], row[3], row[4], row[5]
            if np.isfinite(msd) and np.isfinite(dt) and n_samples > 0:
                lag_map[lag].append(row)
    rows = []
    for lag in sorted(lag_map.keys()):
        arr = np.asarray(lag_map[lag], dtype=np.float64)
        dt_vals = arr[:, 1]
        msd_vals = arr[:, 2]
        n_vals = arr[:, 4]
        weights = n_vals / np.sum(n_vals)
        dt_mean = float(np.sum(weights * dt_vals))
        msd_weighted = float(np.sum(weights * msd_vals))
        n_total = int(np.sum(n_vals))
        if len(msd_vals) > 1:
            msd_sem_traj = float(np.std(msd_vals, ddof=1) / np.sqrt(len(msd_vals)))
        else:
            msd_sem_traj = np.nan
        rows.append([lag, dt_mean, msd_weighted, msd_sem_traj, n_total, len(msd_vals)])
    return np.asarray(rows, dtype=np.float64)


def analyze_group(key, trajs, out_root):
    group_outdir = out_root / key.tag()
    group_outdir.mkdir(parents=True, exist_ok=True)
    print("=" * 80)
    print(f"[GROUP] {key.tag()} | n_traj = {len(trajs)}")
    single_results = []
    for traj in trajs:
        try:
            res = analyze_single_trajectory(traj, group_outdir)
            if res is not None:
                single_results.append(res)
        except Exception as e:
            print(f"  [ERROR] {traj.path.name}: {e}")
    if len(single_results) == 0:
        print("  [WARN] No successful trajectories in this group.")
        return
    ens = pool_ensemble_msd(single_results)
    header = "lag_step delta_t msd msd_sem_traj n_total_samples n_trajectories_contributing"
    np.savetxt(group_outdir / "ensemble_boundary_msd.txt", ens, header=header)
    fit = fit_msd_scaling(ens)
    save_fit(group_outdir / "ensemble_boundary_msd_fit.txt", fit)
    plot_msd(ens, fit, key.tag(), group_outdir / "ensemble_boundary_msd_scaling.png")
    if fit is not None:
        print(f"  [FIT] alpha = {fit['alpha']:.4f} ± {fit['alpha_se']:.4f}, R2={fit['R2']:.4f}")
    else:
        print("  [FIT] failed: not enough valid MSD points.")


def main():
    if not cuda.is_available():
        raise RuntimeError("Numba CUDA is not available. This script is GPU-only.")
    dump_dir_str = DUMP_DIR.strip()
    if (not dump_dir_str) and USE_FOLDER_DIALOG:
        dump_dir_str = pick_folder_by_dialog("选择包含 trajectory_*.dump 的文件夹")
    if not dump_dir_str:
        raise SystemExit("Please set DUMP_DIR or choose a folder from the dialog.")
    dump_dir = Path(dump_dir_str).expanduser().resolve()
    if not dump_dir.is_dir():
        raise NotADirectoryError(f"DUMP_DIR does not exist: {dump_dir}")
    out_root = Path(OUT_DIR).expanduser().resolve() if OUT_DIR.strip() else dump_dir / "perversion_boundary_msd_out"
    out_root.mkdir(parents=True, exist_ok=True)
    dump_files = sorted(dump_dir.rglob(FILE_PATTERN)) if RECURSIVE else sorted(dump_dir.glob(FILE_PATTERN))
    print(f"[INFO] DUMP_DIR = {dump_dir}")
    print(f"[INFO] OUT_DIR  = {out_root}")
    print(f"[INFO] Found {len(dump_files)} dump files.")
    groups = defaultdict(list)
    skipped = []
    for p in dump_files:
        info = parse_dump_filename(p)
        if info is None:
            skipped.append(p.name)
            continue
        groups[info.key].append(info)
    if skipped:
        print(f"[WARN] Skipped {len(skipped)} files because their names do not match the expected format.")
        for name in skipped[:10]:
            print(f"       skipped: {name}")
        if len(skipped) > 10:
            print("       ...")
    if not groups:
        raise SystemExit("No valid trajectory files found.")
    print(f"[INFO] Found {len(groups)} parameter groups.")
    for key in sorted(groups.keys(), key=lambda k: (k.tau, k.v, k.kII, k.kI)):
        trajs = sorted(groups[key], key=lambda x: x.ems)
        analyze_group(key, trajs, out_root)
    print("=" * 80)
    print(f"[DONE] Boundary-region MSD analysis finished. Output: {out_root}")


if __name__ == "__main__":
    main()
