# GPU-only version generated automatically: CPU fallback branches removed.
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch perversion analysis directly from LAMMPS dump trajectories.

What it does
------------
1. Scan a folder for dump files like
   trajectory_tau1.0000_v0.0000_kI0.0045_kII0.0009_ems46.dump
2. Parse parameters from filename.
3. Group files with the same (tau, v, kI, kII) into one ensemble.
4. For each trajectory:
   - read frames directly from dump
   - compute local chirality phi
   - detect perversions / domain walls
   - measure domain-wall diffusion, width, nucleation, annihilation
   - save trajectory-level time series
5. For each ensemble:
   - compute mean/std over trajectories on the common time grid
   - save ensemble-level summary

This script merges the logic of the user's three uploaded scripts:
- dump -> frame parsing
- chirality/perversion analysis in folder mode
- robust wall tracking with nucleation / annihilation statistics

Notes
-----
- GPU is used if Numba CUDA is available and --device gpu is requested.
- Otherwise the script falls back to a CPU implementation.
- The default chirality definition follows the uploaded code.
"""

from __future__ import annotations

import argparse
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np
from scipy.signal import savgol_filter

try:
    from numba import cuda
    CUDA_AVAILABLE = cuda.is_available()
except Exception:
    cuda = None
    CUDA_AVAILABLE = False


# ============================================================
# 1. Filename parsing and grouping
# ============================================================
FILENAME_RE = re.compile(
    r"trajectory_tau(?P<tau>[-+0-9.eE]+)_v(?P<v>[-+0-9.eE]+)_kI(?P<kI>[-+0-9.eE]+)_kII(?P<kII>[-+0-9.eE]+)_ems(?P<ems>\d+)\.dump$"
)


@dataclass(frozen=True)
class ParamKey:
    tau: float
    v: float
    kI: float
    kII: float

    def tag(self) -> str:
        return f"tau{self.tau:.4f}_v{self.v:.4f}_kI{self.kI:.4f}_kII{self.kII:.4f}"


@dataclass
class TrajectoryInfo:
    path: Path
    key: ParamKey
    ems: int


@dataclass
class AnalysisConfig:
    T: float = 156.25
    sg_window: int = 51
    sg_poly: int = 3
    perv_eps: float = 0.05
    perv_gap: int = 20
    lattice_spacing: float = 1.0
    search_radius: float = 40.0
    frame_stride: int = 20
    sim_dt: float = 1.0
    xyz_mode: str = "auto"  # auto | idxyz | dumpfrac
    use_gpu: bool = False

    @property
    def frame_dt(self) -> float:
        return self.sim_dt * self.frame_stride


def parse_dump_filename(path: Path) -> Optional[TrajectoryInfo]:
    m = FILENAME_RE.match(path.name)
    if not m:
        return None
    return TrajectoryInfo(
        path=path,
        key=ParamKey(
            tau=float(m.group("tau")),
            v=float(m.group("v")),
            kI=float(m.group("kI")),
            kII=float(m.group("kII")),
        ),
        ems=int(m.group("ems")),
    )


# ============================================================
# 2. Dump parsing
# ============================================================
def _infer_atom_columns(atom_header: str, xyz_mode: str) -> Dict[str, int]:
    cols = atom_header.strip().split()[2:]  # after "ITEM: ATOMS"
    name_to_idx = {name: i for i, name in enumerate(cols)}

    if xyz_mode == "idxyz":
        required = ["id", "x", "y", "z"]
    elif xyz_mode == "dumpfrac":
        required = ["id", "xs", "ys", "zs"]
    else:
        # auto
        if all(k in name_to_idx for k in ["id", "x", "y", "z"]):
            required = ["id", "x", "y", "z"]
        elif all(k in name_to_idx for k in ["id", "xs", "ys", "zs"]):
            required = ["id", "xs", "ys", "zs"]
        else:
            raise ValueError(
                f"Cannot infer atom columns from header: {atom_header.strip()}"
            )

    missing = [k for k in required if k not in name_to_idx]
    if missing:
        raise ValueError(f"Missing atom columns {missing} in header: {atom_header.strip()}")
    return {k: name_to_idx[k] for k in required}



def iter_lammps_dump_frames(dump_path):
    """
    Strict LAMMPS dump parser.

    Supports sections like:
      ITEM: TIMESTEP
      0.000000
      ITEM: BOX BOUNDS pp pp pp
      ...
      ITEM: NUMBER OF ATOMS
      3000
      ITEM: ATOMS id type x y z con_I con_II ix iy iz
      ...

    Yields dictionaries:
      {
        "timestep": float,
        "natoms": int,
        "atoms": np.ndarray of shape (natoms, 4) with columns (id, x, y, z)
      }
    """
    import numpy as np

    with open(dump_path, "r", encoding="utf-8", errors="ignore") as f:
        timestep = None
        natoms = None

        while True:
            line = f.readline()
            if not line:
                break

            s = line.strip()
            if not s:
                continue

            if s != "ITEM: TIMESTEP":
                continue

            # ---- TIMESTEP ----
            line = f.readline()
            if not line:
                break
            timestep = float(line.strip())

            # ---- Seek NUMBER OF ATOMS and ATOMS ----
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
                    line = f.readline()
                    if not line:
                        raise ValueError(f"Unexpected EOF after ITEM: NUMBER OF ATOMS in {dump_path}")
                    natoms = int(float(line.strip()))

                elif s.startswith("ITEM: BOX BOUNDS"):
                    # LAMMPS orthorhombic box -> skip next 3 lines
                    for _ in range(3):
                        skip = f.readline()
                        if not skip:
                            raise ValueError(f"Unexpected EOF inside BOX BOUNDS in {dump_path}")

                elif s.startswith("ITEM: ATOMS"):
                    atom_fields = s.split()[2:]
                    break

                elif s.startswith("ITEM: TIMESTEP"):
                    raise ValueError(f"Malformed dump: new TIMESTEP encountered before ATOMS in {dump_path}")

            if atom_fields is None:
                break
            if natoms is None:
                raise ValueError(f"Missing NUMBER OF ATOMS before ATOMS section in {dump_path}")

            # Required columns
            try:
                id_col = atom_fields.index("id")
                x_col = atom_fields.index("x") if "x" in atom_fields else atom_fields.index("xu")
                y_col = atom_fields.index("y") if "y" in atom_fields else atom_fields.index("yu")
                z_col = atom_fields.index("z") if "z" in atom_fields else atom_fields.index("zu")
            except ValueError:
                # fallback for scaled coords
                try:
                    id_col = atom_fields.index("id")
                    x_col = atom_fields.index("xs")
                    y_col = atom_fields.index("ys")
                    z_col = atom_fields.index("zs")
                except ValueError as e:
                    raise ValueError(
                        f"Cannot find id/x/y/z columns in ATOMS header of {dump_path}. "
                        f"Fields are: {atom_fields}"
                    ) from e

            atoms = np.empty((natoms, 4), dtype=np.float64)

            for i in range(natoms):
                line = f.readline()
                if not line:
                    raise ValueError(f"Unexpected EOF while reading atom data in {dump_path}")
                parts = line.split()
                if len(parts) < len(atom_fields):
                    raise ValueError(
                        f"Atom line has too few columns in {dump_path}: {line.strip()}"
                    )

                atoms[i, 0] = int(float(parts[id_col]))
                atoms[i, 1] = float(parts[x_col])
                atoms[i, 2] = float(parts[y_col])
                atoms[i, 3] = float(parts[z_col])

            # sort by atom id so the polymer contour is consistent
            order = np.argsort(atoms[:, 0])
            atoms = np.ascontiguousarray(atoms[order])

            yield {
                "timestep": timestep,
                "natoms": natoms,
                "atoms": np.ascontiguousarray(atoms),
            }

def _magnitude(v: np.ndarray) -> float:
    return float(np.sqrt(np.dot(v, v)))


if cuda is not None:
    @cuda.jit(device=True)
    def magnitude_gpu(vector):
        return math.sqrt(vector[0] ** 2 + vector[1] ** 2 + vector[2] ** 2)

    @cuda.jit(device=True)
    def cross_gpu(v1, v2):
        return (v1[1] * v2[2] - v1[2] * v2[1],
                v1[2] * v2[0] - v1[0] * v2[2],
                v1[0] * v2[1] - v1[1] * v2[0])

    @cuda.jit(device=True)
    def dot_gpu(u, v):
        return u[0] * v[0] + u[1] * v[1] + u[2] * v[2]

    @cuda.jit(device=True)
    def sub_gpu(v1, v2):
        return (v1[0] - v2[0], v1[1] - v2[1], v1[2] - v2[2])

    @cuda.jit(device=True)
    def add_gpu(v1, v2):
        return (v1[0] + v2[0], v1[1] + v2[1], v1[2] + v2[2])

    @cuda.jit(device=True)
    def divide_gpu(v, divisor):
        return (v[0] / divisor, v[1] / divisor, v[2] / divisor)

    @cuda.jit
    def calculate_phi_kernel(beads, phis, T):
        i = cuda.grid(1)
        max_i = beads.shape[0] - int(1.25 * T)
        if i < max_i:
            A = beads[i]
            B = beads[int(i + 0.5 * T)]
            C = beads[int(i + 0.75 * T)]
            D = beads[int(i + 1.25 * T)]

            E = divide_gpu(add_gpu(A, B), 2.0)
            F = divide_gpu(add_gpu(C, D), 2.0)

            AB = sub_gpu(B, A)
            CD = sub_gpu(D, C)
            EF = sub_gpu(F, E)

            cp = cross_gpu(CD, AB)
            dp = dot_gpu(EF, cp)
            mags = magnitude_gpu(EF) * magnitude_gpu(CD) * magnitude_gpu(AB)
            if mags > 1e-12:
                phis[i] = dp / mags
            else:
                phis[i] = 0.0


def _phi_gpu(beads: np.ndarray, T: float, block_size: int = 128) -> np.ndarray:
    n = beads.shape[0]
    n_phi = n - int(1.25 * T)
    if n_phi <= 0:
        raise ValueError(f"Chain too short for T={T}; n_beads={n}")
    phis = np.zeros(n_phi, dtype=np.float32)
    d_beads = cuda.to_device(beads)
    d_phis = cuda.to_device(phis)
    grid_size = (n_phi + block_size - 1) // block_size
    calculate_phi_kernel[grid_size, block_size](d_beads, d_phis, T)
    return d_phis.copy_to_host()


def compute_phi_gpu(beads: np.ndarray, T: float) -> np.ndarray:
    if not CUDA_AVAILABLE:
        raise RuntimeError("GPU requested but CUDA is not available.")
    return _phi_gpu(beads, T)


# ============================================================
# 4. Analysis functions
# ============================================================
def smooth_phi(phi: np.ndarray, window: int, polyorder: int) -> np.ndarray:
    phi = np.asarray(phi, dtype=np.float64)
    w = int(window)
    if w >= phi.size:
        w = max(5, (phi.size // 2) * 2 - 1)
    if w < polyorder + 2:
        w = polyorder + 2
    if w % 2 == 0:
        w += 1
    return savgol_filter(phi, w, polyorder)


def calculate_order_parameters(phi: np.ndarray) -> Tuple[float, float]:
    phi = np.asarray(phi, dtype=np.float64)
    phi_bar = float(np.mean(np.abs(phi)))
    psi = float(np.abs(np.mean(phi)))
    return phi_bar, psi


def find_perversions_and_domains(phi_smooth: np.ndarray, eps: float = 0.05, min_gap: int = 20) -> Tuple[np.ndarray, float]:
    phi_smooth = np.asarray(phi_smooth, dtype=np.float64)
    s = np.zeros_like(phi_smooth, dtype=np.int8)
    s[phi_smooth > eps] = 1
    s[phi_smooth < -eps] = -1

    last = 0
    for i in range(len(s)):
        if s[i] == 0:
            s[i] = last
        else:
            last = s[i]
    if s[0] == 0:
        first_nz = np.nonzero(s)[0]
        if len(first_nz) > 0:
            s[:first_nz[0]] = s[first_nz[0]]
        else:
            return np.array([], dtype=int), float(len(phi_smooth))

    flips = np.where(np.diff(s) != 0)[0]
    merged: List[int] = []
    for f in flips:
        if (not merged) or (f - merged[-1] >= min_gap):
            merged.append(int(f))

    perv_idx = np.array(merged, dtype=int)
    total_len = len(phi_smooth)
    if len(perv_idx) == 0:
        avg_size = float(total_len)
    else:
        boundaries = np.concatenate(([0], perv_idx, [total_len]))
        sizes = np.diff(boundaries)
        avg_size = float(np.mean(sizes))
    return perv_idx, avg_size


def analyze_wall_physics_robust(
    phi: np.ndarray,
    curr_idx: np.ndarray,
    prev_idx: Optional[np.ndarray],
    dt: float,
    lattice_spacing: float = 1.0,
    search_radius: float = 40.0,
) -> Tuple[float, float, int, int]:
    phi = np.asarray(phi)
    amplitude = 2.0 * np.mean(np.abs(phi))
    matched_widths: List[float] = []
    valid_displacements: List[float] = []
    matched_curr_indices = set()
    matched_prev_indices = set()

    if (prev_idx is not None) and (len(prev_idx) > 0) and (len(curr_idx) > 0):
        for i, curr_pos in enumerate(curr_idx):
            dists = np.abs(prev_idx - curr_pos)
            min_arg = int(np.argmin(dists))
            min_dist = float(dists[min_arg])
            if min_dist < search_radius and min_arg not in matched_prev_indices:
                matched_prev_indices.add(min_arg)
                matched_curr_indices.add(i)
                dx2 = (min_dist * lattice_spacing) ** 2
                valid_displacements.append(dx2)

                grad = np.gradient(phi)
                idx = int(curr_pos)
                if 0 < idx < len(phi) - 1:
                    slope = abs(float(grad[idx]))
                    if slope > 1e-9:
                        w = amplitude / slope
                        if w < 400:
                            matched_widths.append(float(w))

    n_nucleation = 0
    n_annihilation = 0
    if prev_idx is not None:
        n_annihilation = len(prev_idx) - len(matched_prev_indices)
        n_nucleation = len(curr_idx) - len(matched_curr_indices)

    avg_width = float(np.mean(matched_widths)) if matched_widths else np.nan
    if valid_displacements:
        msd = float(np.mean(valid_displacements))
        D_inst = msd / (2.0 * dt)
    else:
        D_inst = np.nan
    return avg_width, D_inst, int(n_nucleation), int(n_annihilation)


# ============================================================
# 5. Per-trajectory analysis
# ============================================================
def analyze_single_trajectory(traj: TrajectoryInfo, outdir: Path, cfg: AnalysisConfig) -> Path:
    traj_out = outdir / traj.key.tag() / f"ems{traj.ems:03d}"
    traj_out.mkdir(parents=True, exist_ok=True)

    rows: List[List[float]] = []
    prev_perv_idx: Optional[np.ndarray] = None
    processed = 0

    for iframe, frame in enumerate(iter_lammps_dump_frames(traj.path)):
        timestep = frame["timestep"]
        beads = np.ascontiguousarray(frame["atoms"][:, 1:4])
        if iframe % cfg.frame_stride != 0:
            continue
        phi_raw = compute_phi_gpu(beads, cfg.T)
        phi_s = smooth_phi(phi_raw, cfg.sg_window, cfg.sg_poly)
        phi_bar, psi = calculate_order_parameters(phi_s)
        perv_idx, avg_dom = find_perversions_and_domains(phi_s, cfg.perv_eps, cfg.perv_gap)
        avg_width, D_coeff, n_nuc, n_ann = analyze_wall_physics_robust(
            phi_s,
            perv_idx,
            prev_perv_idx,
            cfg.frame_dt,
            lattice_spacing=cfg.lattice_spacing,
            search_radius=cfg.search_radius,
        )
        prev_perv_idx = perv_idx.copy()
        n_perv = len(perv_idx)
        time = timestep * cfg.sim_dt
        rows.append([
            time, timestep, phi_bar, psi, n_perv, avg_dom,
            avg_width, D_coeff, n_nuc, n_ann,
        ])
        processed += 1

    if not rows:
        raise RuntimeError(f"No frames processed for {traj.path}")

    arr = np.asarray(rows, dtype=np.float64)
    out_file = traj_out / "kzm_kinetics_robust.txt"
    header = (
        "time timestep phi_bar psi n_perv avg_dom_size "
        "avg_width_stable diff_coeff_stable n_nucleation n_annihilation"
    )
    np.savetxt(out_file, arr, header=header)

    # summary for the trajectory
    summary = {
        "n_frames": arr.shape[0],
        "mean_n_perv": np.nanmean(arr[:, 4]),
        "mean_dom": np.nanmean(arr[:, 5]),
        "mean_width": np.nanmean(arr[:, 6]),
        "mean_diff": np.nanmean(arr[:, 7]),
        "total_nucleation": np.nansum(arr[:, 8]),
        "total_annihilation": np.nansum(arr[:, 9]),
        "final_n_perv": arr[-1, 4],
        "final_dom": arr[-1, 5],
    }
    with (traj_out / "trajectory_summary.txt").open("w", encoding="utf-8") as f:
        for k, v in summary.items():
            f.write(f"{k}\t{v}\n")
        f.write(f"source_dump\t{traj.path}\n")

    return out_file


# ============================================================
# 6. Ensemble averaging
# ============================================================
def load_trajectory_table(path: Path) -> np.ndarray:
    data = np.loadtxt(path)
    if data.ndim == 1:
        data = data[None, :]
    return data


def ensemble_average(files: Sequence[Path], outdir: Path, key: ParamKey) -> None:
    if not files:
        return
    arrays = [load_trajectory_table(f) for f in files]
    min_len = min(a.shape[0] for a in arrays)
    arrays = [a[:min_len] for a in arrays]
    stack = np.stack(arrays, axis=0)  # (ntraj, ntime, ncol)

    mean_arr = np.nanmean(stack, axis=0)
    std_arr = np.nanstd(stack, axis=0, ddof=0)

    ens_dir = outdir / key.tag()
    ens_dir.mkdir(parents=True, exist_ok=True)

    header_cols = [
        "time", "timestep", "phi_bar", "psi", "n_perv", "avg_dom_size",
        "avg_width_stable", "diff_coeff_stable", "n_nucleation", "n_annihilation"
    ]
    mean_header = " ".join(header_cols)
    std_header = " ".join([f"std_{c}" for c in header_cols])

    np.savetxt(ens_dir / "ensemble_mean.txt", mean_arr, header=mean_header)
    np.savetxt(ens_dir / "ensemble_std.txt", std_arr, header=std_header)

    # one-line summary over time
    with (ens_dir / "ensemble_summary.txt").open("w", encoding="utf-8") as f:
        f.write(f"n_trajectories\t{len(files)}\n")
        f.write(f"common_length\t{min_len}\n")
        f.write(f"mean_n_perv_over_time\t{np.nanmean(mean_arr[:,4])}\n")
        f.write(f"mean_domain_size_over_time\t{np.nanmean(mean_arr[:,5])}\n")
        f.write(f"mean_width_over_time\t{np.nanmean(mean_arr[:,6])}\n")
        f.write(f"mean_diff_over_time\t{np.nanmean(mean_arr[:,7])}\n")
        f.write(f"total_nucleation_mean\t{np.nansum(mean_arr[:,8])}\n")
        f.write(f"total_annihilation_mean\t{np.nansum(mean_arr[:,9])}\n")


# ============================================================
# 7. Main
# ============================================================
def collect_trajectories(indir: Path) -> Dict[ParamKey, List[TrajectoryInfo]]:
    grouped: Dict[ParamKey, List[TrajectoryInfo]] = defaultdict(list)
    for path in sorted(indir.glob("trajectory_*.dump")):
        info = parse_dump_filename(path)
        if info is None:
            continue
        grouped[info.key].append(info)
    return dict(grouped)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Batch perversion/domain-wall analysis for many trajectory_*.dump files."
    )
    ap.add_argument("indir", type=str, help="Folder containing trajectory_*.dump files")
    ap.add_argument("--outdir", type=str, default="analysis_out", help="Output root folder")
    ap.add_argument("--T", type=float, default=156.25, help="Chirality scale T")
    ap.add_argument("--sg-window", type=int, default=51, help="Savitzky-Golay window")
    ap.add_argument("--sg-poly", type=int, default=3, help="Savitzky-Golay polynomial order")
    ap.add_argument("--perv-eps", type=float, default=0.05, help="Threshold for chirality sign")
    ap.add_argument("--perv-gap", type=int, default=20, help="Minimum gap between nearby perversions")
    ap.add_argument("--frame-stride", type=int, default=20, help="Analyze every Nth dump frame")
    ap.add_argument("--sim-dt", type=float, default=1.0, help="Physical time per dump timestep")
    ap.add_argument("--lattice-spacing", type=float, default=1.0, help="Spatial conversion for wall displacement")
    ap.add_argument("--search-radius", type=float, default=40.0, help="Max match distance for wall tracking")
    ap.add_argument(
        "--xyz-mode",
        choices=["auto", "idxyz", "dumpfrac"],
        default="auto",
        help="Which coordinates to read from dump: x/y/z or xs/ys/zs",
    )
    ap.add_argument(
        "--device",
        choices=["cpu", "gpu"],
        default="cpu",
        help="Compute phi on CPU or GPU",
    )
    args = ap.parse_args()

    indir = Path(args.indir).expanduser().resolve()
    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    cfg = AnalysisConfig(
        T=args.T,
        sg_window=args.sg_window,
        sg_poly=args.sg_poly,
        perv_eps=args.perv_eps,
        perv_gap=args.perv_gap,
        lattice_spacing=args.lattice_spacing,
        search_radius=args.search_radius,
        frame_stride=args.frame_stride,
        sim_dt=args.sim_dt,
        xyz_mode=args.xyz_mode,
        use_gpu=(args.device == "gpu"),
    )

    grouped = collect_trajectories(indir)
    if not grouped:
        raise SystemExit(f"No files matching trajectory_*.dump found in {indir}")

    print(f"Found {sum(len(v) for v in grouped.values())} dump files in {len(grouped)} ensembles.")
    if cfg.use_gpu and not CUDA_AVAILABLE:
        raise SystemExit("--device gpu was requested, but CUDA is not available.")

    for key, trajs in sorted(grouped.items(), key=lambda kv: (kv[0].tau, kv[0].v, kv[0].kI, kv[0].kII)):
        trajs = sorted(trajs, key=lambda x: x.ems)
        print(f"\n=== Ensemble {key.tag()} | n={len(trajs)} ===")
        traj_files: List[Path] = []
        for traj in trajs:
            print(f"Processing {traj.path.name} ...")
            result_file = analyze_single_trajectory(traj, outdir, cfg)
            traj_files.append(result_file)
        ensemble_average(traj_files, outdir, key)
        print(f"Saved ensemble outputs under {outdir / key.tag()}")

    print("\nAll done.")



if __name__ == "__main__":
    # ============================================================
    # 用户配置区：不想用命令行时，只需要修改这里
    # ============================================================
    DUMP_DIR = r""            # 例如：r"D:\my_project\dumps"
    OUT_DIR = r""             # 留空则默认输出到 DUMP_DIR/analysis_out
    USE_FOLDER_DIALOG = True  # 若 DUMP_DIR 为空，是否弹窗选择文件夹

    # 分析参数
    T_SCALE = 156.25
    SG_WINDOW = 51
    SG_POLY = 3
    PERV_EPS = 0.05
    PERV_GAP = 20
    LATTICE_SPACING = 1.0
    SEARCH_RADIUS = 40.0
    FRAME_STRIDE = 20
    SIM_DT = 1.0

    # dump 坐标模式: "auto" | "idxyz" | "dumpfrac"
    XYZ_MODE = "auto"

    # 设备: "cpu" 或 "gpu"
    DEVICE = "gpu"

    # 是否递归搜索子文件夹
    RECURSIVE = False

    # 文件名模式
    FILE_PATTERN = "trajectory_*.dump"

    # ============================================================
    # 下面一般不用改
    # ============================================================
    import sys
    from pathlib import Path
    from collections import defaultdict

    def pick_folder_by_dialog():
        try:
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk()
            root.withdraw()
            folder = filedialog.askdirectory(title="选择包含 trajectory_*.dump 的文件夹")
            root.destroy()
            return folder
        except Exception as e:
            print(f"[WARN] 无法打开文件夹选择窗口: {e}")
            return ""

    dump_dir_str = DUMP_DIR.strip()
    if (not dump_dir_str) and USE_FOLDER_DIALOG:
        dump_dir_str = pick_folder_by_dialog()

    if not dump_dir_str:
        raise SystemExit(
            "请先在脚本顶部把 DUMP_DIR 改成你的 dump 文件夹路径，"
            "或者保持 USE_FOLDER_DIALOG=True 后直接运行脚本。"
        )

    dump_dir = Path(dump_dir_str).expanduser().resolve()
    if not dump_dir.is_dir():
        raise NotADirectoryError(f"输入目录不存在: {dump_dir}")

    if OUT_DIR.strip():
        outdir = Path(OUT_DIR).expanduser().resolve()
    else:
        outdir = dump_dir / "analysis_out"
    outdir.mkdir(parents=True, exist_ok=True)

    if DEVICE.lower() != "gpu":
        raise ValueError('这个版本只支持 GPU，请保持 DEVICE = "gpu"')
    if not CUDA_AVAILABLE:
        raise RuntimeError("当前环境的 Numba CUDA 不可用，无法运行这个 GPU-only 版本。")

    use_gpu = True

    cfg = AnalysisConfig(
        T=T_SCALE,
        sg_window=SG_WINDOW,
        sg_poly=SG_POLY,
        perv_eps=PERV_EPS,
        perv_gap=PERV_GAP,
        lattice_spacing=LATTICE_SPACING,
        search_radius=SEARCH_RADIUS,
        frame_stride=FRAME_STRIDE,
        sim_dt=SIM_DT,
        xyz_mode=XYZ_MODE,
        use_gpu=use_gpu,
    )

    if RECURSIVE:
        dump_paths = sorted(dump_dir.rglob(FILE_PATTERN))
    else:
        dump_paths = sorted(dump_dir.glob(FILE_PATTERN))

    if not dump_paths:
        raise SystemExit(f"在 {dump_dir} 中没有找到 {FILE_PATTERN}")

    print(f"[INFO] 找到 {len(dump_paths)} 个 dump 文件")

    groups = defaultdict(list)
    skipped = []
    for p in dump_paths:
        info = parse_dump_filename(p)
        if info is None:
            skipped.append(p.name)
            continue
        groups[info.key].append(info)

    if skipped:
        print(f"[WARN] 有 {len(skipped)} 个文件名不符合命名格式，已跳过。")
        for name in skipped[:10]:
            print(f"       skipped: {name}")
        if len(skipped) > 10:
            print("       ...")

    if not groups:
        raise SystemExit("没有找到符合命名格式的轨迹文件。")

    print(f"[INFO] 自动分成 {len(groups)} 个系综参数组")

    for key in sorted(groups.keys(), key=lambda k: (k.tau, k.v, k.kI, k.kII)):
        trajs = sorted(groups[key], key=lambda tr: tr.ems)
        print("=" * 80)
        print(f"[GROUP] {key.tag()} | 轨迹数 = {len(trajs)}")

        out_files = []
        for traj in trajs:
            print(f"  [TRAJ] {traj.path.name}")
            try:
                out_file = analyze_single_trajectory(traj, outdir, cfg)
                out_files.append(out_file)
            except Exception as e:
                print(f"  [ERROR] {traj.path.name}: {e}")

        if out_files:
            try:
                ensemble_average(out_files, outdir, key)
                print(f"  [OK] 系综平均已完成: {key.tag()}")
            except Exception as e:
                print(f"  [ERROR] 系综平均失败 {key.tag()}: {e}")
        else:
            print("  [WARN] 这一组没有成功处理的轨迹，跳过系综平均。")

    print("=" * 80)
    print(f"[DONE] 全部完成。输出目录: {outdir}")
