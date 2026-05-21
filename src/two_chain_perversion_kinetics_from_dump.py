#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Two-chain chirality / perversion kinetics analysis with A-B chirality correlation
and A-B same-sign chirality overlay.

Input dump format:
ITEM: ATOMS id mol type x y z con_I con_II ix iy iz

Convention:
    Chain A: atom id <= CHAIN_A_MAX_ID
    Chain B: atom id >= CHAIN_B_MIN_ID

Default:
    Chain A: id <= 2000
    Chain B: id >= 2001

The chirality of A and B is computed separately.
The code never computes chirality across the A/B boundary.

New output:
    AB_chirality_same_sign_overlay.png

Meaning:
    red  : phi_A > eps and phi_B > eps, both positive
    blue : phi_A < -eps and phi_B < -eps, both negative
    white: other cases, including opposite signs or near-zero chirality
"""

import os
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from numba import cuda
from scipy.signal import savgol_filter


# ============================================================
# 0. User settings
# ============================================================

INPUT_DUMP = r"F:\e\C++_Project\sister_chromotids\Output\trajectory_tau1.0000_v0.0000_kI0.0500_kII0.1000_ems1_part1.dump"
OUT_DIR = "two_chain_perversion_chirality_corr_out"

# Dump columns:
# ITEM: ATOMS id mol type x y z con_I con_II ix iy iz
ID_COL = 0
MOL_COL = 1
TYPE_COL = 2
POS_COLS = (3, 4, 5)
IMAGE_COLS = (8, 9, 10)

# Explicit A/B boundary by atom id
CHAIN_A_MAX_ID = 2000
CHAIN_B_MIN_ID = 2001

# Chain labels
MOL_IDS_TO_ANALYZE = [1, 2]
MOL_TO_CHAIN_NAME = {
    1: "chain_A",
    2: "chain_B",
}

# If the chain crosses periodic boundaries, set True to unwrap by ix, iy, iz.
UNWRAP_COORDS = False

# Chirality calculation scale
T_SCALE = 156.25

# Smoothing
SMOOTH_WINDOW = 51
SMOOTH_POLYORDER = 3

# Perversion detection
DEFECT_EPS = 0.05
DEFECT_GAP = 20

# Overlay threshold.
# Only points with phi > OVERLAY_EPS or phi < -OVERLAY_EPS are colored.
OVERLAY_EPS = DEFECT_EPS

# Time settings
SIM_DT = 1.0
STRIDE = 20
FRAME_DT = SIM_DT * STRIDE

# Domain-wall tracking
SEARCH_RADIUS = 40.0

# GPU settings
CUDA_BLOCK_SIZE = 128

# Plot settings
SAVE_DPI = 300


# ============================================================
# 1. CUDA kernels for chirality calculation
# ============================================================

@cuda.jit(device=True)
def magnitude(vector):
    return math.sqrt(vector[0] ** 2 + vector[1] ** 2 + vector[2] ** 2)


@cuda.jit(device=True)
def cross(v1, v2):
    return (
        v1[1] * v2[2] - v1[2] * v2[1],
        v1[2] * v2[0] - v1[0] * v2[2],
        v1[0] * v2[1] - v1[1] * v2[0],
    )


@cuda.jit(device=True)
def dot(u, v):
    return u[0] * v[0] + u[1] * v[1] + u[2] * v[2]


@cuda.jit(device=True)
def sub(v1, v2):
    return (
        v1[0] - v2[0],
        v1[1] - v2[1],
        v1[2] - v2[2],
    )


@cuda.jit(device=True)
def add(v1, v2):
    return (
        v1[0] + v2[0],
        v1[1] + v2[1],
        v1[2] + v2[2],
    )


@cuda.jit(device=True)
def divide(v, divisor):
    return (
        v[0] / divisor,
        v[1] / divisor,
        v[2] / divisor,
    )


@cuda.jit
def calculate_phi_kernel(beads, phis, T):
    i = cuda.grid(1)

    max_i = beads.shape[0] - int(1.25 * T)

    if i < max_i:
        A = beads[i]
        B = beads[int(i + 0.5 * T)]
        C = beads[int(i + 0.75 * T)]
        D = beads[int(i + 1.25 * T)]

        E = divide(add(A, B), 2.0)
        F = divide(add(C, D), 2.0)

        AB = sub(B, A)
        CD = sub(D, C)
        EF = sub(F, E)

        cp = cross(CD, AB)
        dp = dot(EF, cp)

        mags = magnitude(EF) * magnitude(CD) * magnitude(AB)

        if mags > 1e-12:
            phis[i] = dp / mags
        else:
            phis[i] = 0.0


def compute_phi_gpu(beads, T, block_size=128):
    """
    Compute chirality phi(s) using CUDA.

    beads must already be one single chain, ordered along the desired contour direction.
    """

    n_beads = beads.shape[0]
    n_phi = n_beads - int(1.25 * T)

    if n_phi <= 5:
        raise ValueError(
            f"Chain too short for chirality calculation: "
            f"n_beads={n_beads}, T={T}, n_phi={n_phi}"
        )

    beads = np.ascontiguousarray(beads.astype(np.float32))
    phis = np.zeros(n_phi, dtype=np.float32)

    d_beads = cuda.to_device(beads)
    d_phis = cuda.to_device(phis)

    grid_size = (n_phi + block_size - 1) // block_size
    calculate_phi_kernel[grid_size, block_size](d_beads, d_phis, T)

    return d_phis.copy_to_host()


# ============================================================
# 2. Perversion analysis
# ============================================================

def calculate_order_parameters(phi):
    phi = np.asarray(phi, dtype=np.float64)

    phi_bar = float(np.mean(np.abs(phi)))
    psi = float(np.abs(np.mean(phi)))

    return phi_bar, psi


def find_perversions_and_domains(phi_smooth, eps=0.05, min_gap=20):
    """
    Detect sign-changing boundaries of smoothed chirality field.
    """

    phi_smooth = np.asarray(phi_smooth, dtype=np.float64)

    s = np.zeros_like(phi_smooth, dtype=np.int8)
    s[phi_smooth > eps] = 1
    s[phi_smooth < -eps] = -1

    # Fill zero regions with last nonzero sign
    last = 0
    for i in range(len(s)):
        if s[i] == 0:
            s[i] = last
        else:
            last = s[i]

    # If the beginning is still zero, fill by first nonzero sign
    if s[0] == 0:
        first_nz = np.nonzero(s)[0]
        if len(first_nz) > 0:
            s[:first_nz[0]] = s[first_nz[0]]
        else:
            return np.array([], dtype=int), float(len(phi_smooth))

    flips = np.where(np.diff(s) != 0)[0]

    merged = []
    for f in flips:
        if (not merged) or (f - merged[-1] >= min_gap):
            merged.append(int(f))

    perv_idx = np.array(merged, dtype=int)

    total_len = len(phi_smooth)

    if len(perv_idx) == 0:
        avg_size = float(total_len)
    else:
        boundaries = np.concatenate(([0], perv_idx, [total_len]))
        avg_size = float(np.mean(np.diff(boundaries)))

    return perv_idx, avg_size


def analyze_wall_physics_robust(
    phi,
    curr_idx,
    prev_idx,
    dt,
    lattice_spacing=1.0,
    search_radius=40.0,
):
    """
    Track domain walls between neighboring analyzed frames.
    Estimate:
        - stable wall width
        - instantaneous diffusion coefficient
        - nucleation events
        - annihilation events
    """

    phi = np.asarray(phi, dtype=np.float64)

    amplitude = 2.0 * np.mean(np.abs(phi))

    matched_widths = []
    valid_displacements = []
    matched_curr_indices = set()
    matched_prev_indices = set()

    if (prev_idx is not None) and (len(prev_idx) > 0) and (len(curr_idx) > 0):
        grad = np.gradient(phi)

        for i, curr_pos in enumerate(curr_idx):
            dists = np.abs(prev_idx - curr_pos)

            min_arg = int(np.argmin(dists))
            min_dist = float(dists[min_arg])

            if min_dist < search_radius and min_arg not in matched_prev_indices:
                matched_prev_indices.add(min_arg)
                matched_curr_indices.add(i)

                valid_displacements.append((min_dist * lattice_spacing) ** 2)

                idx = int(curr_pos)

                if 0 < idx < len(phi) - 1:
                    slope = abs(grad[idx])
                    if slope > 1e-9:
                        w = amplitude / slope

                        if w < 400:
                            matched_widths.append(w)

    n_nucleation = 0
    n_annihilation = 0

    if prev_idx is not None:
        n_annihilation = len(prev_idx) - len(matched_prev_indices)
        n_nucleation = len(curr_idx) - len(matched_curr_indices)

    avg_width = float(np.mean(matched_widths)) if matched_widths else np.nan

    if valid_displacements:
        D_inst = float(np.mean(valid_displacements)) / (2.0 * dt)
    else:
        D_inst = np.nan

    return avg_width, D_inst, n_nucleation, n_annihilation


# ============================================================
# 3. A-B chirality correlation and overlay
# ============================================================

def chirality_correlation(phi_A, phi_B):
    """
    Compute chirality correlation between two chains at the same frame.

    Returns:
        C_norm:
            <phi_A phi_B> / sqrt(<phi_A^2><phi_B^2>)

        C_pearson:
            Pearson correlation after subtracting spatial mean

        C_sign:
            <sign(phi_A) sign(phi_B)>

    If the two chains have slightly different phi lengths, they are truncated
    to the same minimum length.
    """

    phi_A = np.asarray(phi_A, dtype=np.float64)
    phi_B = np.asarray(phi_B, dtype=np.float64)

    n = min(len(phi_A), len(phi_B))

    if n <= 5:
        return np.nan, np.nan, np.nan

    a = phi_A[:n]
    b = phi_B[:n]

    denom = np.sqrt(np.mean(a * a) * np.mean(b * b))

    if denom > 1e-12:
        C_norm = float(np.mean(a * b) / denom)
    else:
        C_norm = np.nan

    a0 = a - np.mean(a)
    b0 = b - np.mean(b)

    denom_p = np.sqrt(np.mean(a0 * a0) * np.mean(b0 * b0))

    if denom_p > 1e-12:
        C_pearson = float(np.mean(a0 * b0) / denom_p)
    else:
        C_pearson = np.nan

    sa = np.sign(a)
    sb = np.sign(b)

    mask = (sa != 0) & (sb != 0)

    if np.any(mask):
        C_sign = float(np.mean(sa[mask] * sb[mask]))
    else:
        C_sign = np.nan

    return C_norm, C_pearson, C_sign


def build_AB_same_sign_overlay(phi_A, phi_B, eps=0.05):
    """
    Build one row of the A-B same-sign chirality overlay.

    Output values:
        +1 : both positive
        -1 : both negative
         0 : otherwise, including opposite signs or near-zero chirality
    """

    phi_A = np.asarray(phi_A, dtype=np.float64)
    phi_B = np.asarray(phi_B, dtype=np.float64)

    n = min(len(phi_A), len(phi_B))
    if n <= 0:
        return np.array([], dtype=np.int8)

    a = phi_A[:n]
    b = phi_B[:n]

    overlay = np.zeros(n, dtype=np.int8)

    overlay[(a > eps) & (b > eps)] = 1
    overlay[(a < -eps) & (b < -eps)] = -1

    return overlay


# ============================================================
# 4. Dump reader
# ============================================================

def split_dump_into_frames(filename):
    with open(filename, "r") as f:
        lines = f.readlines()

    starts = [
        i for i, line in enumerate(lines)
        if line.strip().startswith("ITEM: TIMESTEP")
    ]

    if not starts:
        raise RuntimeError("No ITEM: TIMESTEP found in dump file.")

    ranges = []

    for k, start in enumerate(starts):
        end = starts[k + 1] if k + 1 < len(starts) else len(lines)
        ranges.append((start, end))

    return lines, ranges


def parse_box_bounds(frame_lines):
    for i, line in enumerate(frame_lines):
        if line.strip().startswith("ITEM: BOX BOUNDS"):
            bounds = []

            for k in range(3):
                parts = frame_lines[i + 1 + k].strip().split()
                lo, hi = float(parts[0]), float(parts[1])
                bounds.append((lo, hi))

            return np.array([hi - lo for lo, hi in bounds], dtype=np.float64)

    return None


def parse_one_frame(frame_lines):
    timestep = float(frame_lines[1].strip())

    atoms_header_idx = None

    for i, line in enumerate(frame_lines):
        if line.strip().startswith("ITEM: ATOMS"):
            atoms_header_idx = i
            break

    if atoms_header_idx is None:
        raise RuntimeError("No ITEM: ATOMS found in frame.")

    header = frame_lines[atoms_header_idx].strip().split()[2:]
    col = {name: idx for idx, name in enumerate(header)}

    id_col = col.get("id", ID_COL)
    mol_col = col.get("mol", MOL_COL)
    type_col = col.get("type", TYPE_COL)

    if all(n in col for n in ["x", "y", "z"]):
        pos_cols = (col["x"], col["y"], col["z"])
    else:
        pos_cols = POS_COLS

    if all(n in col for n in ["ix", "iy", "iz"]):
        image_cols = (col["ix"], col["iy"], col["iz"])
    else:
        image_cols = IMAGE_COLS

    box_lengths = parse_box_bounds(frame_lines)

    atoms = []

    for line in frame_lines[atoms_header_idx + 1:]:
        s = line.strip()

        if not s:
            continue

        if s.startswith("ITEM:"):
            break

        parts = s.split()

        atom_id = int(float(parts[id_col]))
        mol_id = int(float(parts[mol_col]))
        atom_type = int(float(parts[type_col])) if type_col is not None else 1

        x = float(parts[pos_cols[0]])
        y = float(parts[pos_cols[1]])
        z = float(parts[pos_cols[2]])

        if UNWRAP_COORDS:
            if box_lengths is None:
                raise RuntimeError("UNWRAP_COORDS=True but no BOX BOUNDS found.")

            ix = int(float(parts[image_cols[0]]))
            iy = int(float(parts[image_cols[1]]))
            iz = int(float(parts[image_cols[2]]))

            x += ix * box_lengths[0]
            y += iy * box_lengths[1]
            z += iz * box_lengths[2]

        atoms.append((atom_id, mol_id, atom_type, x, y, z))

    if not atoms:
        raise RuntimeError("No atoms read in frame.")

    return timestep, header, np.array(atoms, dtype=np.float64)


def get_chain_coordinates(atoms, mol_id):
    """
    Extract chain A or chain B.

    Required convention:
        chain_A, mol_id=1:
            atom id <= CHAIN_A_MAX_ID

        chain_B, mol_id=2:
            atom id >= CHAIN_B_MIN_ID

    Then sort each chain by increasing atom id.

    Therefore:
        A direction: small id -> large id
        B direction: small id -> large id

    The chirality calculation is performed only inside each separated chain.
    """

    atom_id = atoms[:, 0].astype(int)

    if int(mol_id) == 1:
        sub = atoms[atom_id <= CHAIN_A_MAX_ID]

    elif int(mol_id) == 2:
        sub = atoms[atom_id >= CHAIN_B_MIN_ID]

    else:
        return None

    if sub.shape[0] == 0:
        return None

    sub = sub[np.argsort(sub[:, 0])]

    return np.ascontiguousarray(sub[:, 3:6].astype(np.float64))


def smooth_phi(phi_raw, window, polyorder):
    phi_raw = np.asarray(phi_raw, dtype=np.float64)

    if len(phi_raw) < 5:
        return phi_raw

    win = int(window)

    if win > len(phi_raw):
        win = len(phi_raw) if len(phi_raw) % 2 == 1 else len(phi_raw) - 1

    if win < polyorder + 2:
        win = polyorder + 2
        if win % 2 == 0:
            win += 1

    if win > len(phi_raw):
        win = len(phi_raw) if len(phi_raw) % 2 == 1 else len(phi_raw) - 1

    if win < 5:
        return phi_raw

    if win % 2 == 0:
        win -= 1

    return savgol_filter(phi_raw, win, polyorder)


def process_chain(coords):
    phi_raw = compute_phi_gpu(coords, T_SCALE, CUDA_BLOCK_SIZE)
    phi_s = smooth_phi(phi_raw, SMOOTH_WINDOW, SMOOTH_POLYORDER)

    phi_bar, psi = calculate_order_parameters(phi_s)
    perv_idx, avg_dom = find_perversions_and_domains(
        phi_s,
        DEFECT_EPS,
        DEFECT_GAP,
    )

    return phi_bar, psi, perv_idx, avg_dom, phi_s


# ============================================================
# 5. Save and plot
# ============================================================

def save_chain_results(chain_name, results, out_dir):
    arr = np.array(results, dtype=np.float64)

    path = os.path.join(out_dir, f"kzm_kinetics_{chain_name}.txt")

    header = (
        "time frame_idx mol_id "
        "phi_bar psi n_perv avg_dom_size "
        "avg_width(stable) diff_coeff(stable) "
        "n_nucleation n_annihilation"
    )

    np.savetxt(path, arr, header=header)

    return arr, path


def plot_single_chain(chain_name, data_arr, kymograph_matrix, out_dir):
    if data_arr.size == 0:
        return

    t = data_arr[:, 0]

    phi_bar = data_arr[:, 3]
    psi = data_arr[:, 4]
    n_perv = data_arr[:, 5]
    width = data_arr[:, 7]
    diff = data_arr[:, 8]
    n_nuc = data_arr[:, 9]
    n_ann = data_arr[:, 10]

    # Figure 1: order parameters
    fig1, ax1 = plt.subplots(figsize=(8, 5))

    ax1.plot(t, phi_bar, label=r"$\langle |\phi| \rangle$", lw=2)
    ax1.plot(t, psi, label=r"$|\langle \phi \rangle|$", lw=2, ls="--")

    ax1.set_title(f"{chain_name}: chirality order parameters")
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Order parameter")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    fig1.savefig(
        os.path.join(out_dir, f"{chain_name}_fig1_order_parameters.png"),
        dpi=SAVE_DPI,
        bbox_inches="tight",
    )
    plt.close(fig1)

    # Figure 2: perversion number
    fig2, ax2 = plt.subplots(figsize=(8, 5))

    ax2.plot(t, n_perv, lw=2)

    ax2.set_title(f"{chain_name}: perversion number")
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Number of perversions")
    ax2.grid(True, alpha=0.3)

    fig2.savefig(
        os.path.join(out_dir, f"{chain_name}_fig2_perversion_number.png"),
        dpi=SAVE_DPI,
        bbox_inches="tight",
    )
    plt.close(fig2)

    # Figure 3: wall width, diffusion, nucleation / annihilation
    fig3, (ax3_top, ax3_bot) = plt.subplots(
        2,
        1,
        figsize=(10, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [2, 1]},
    )

    ax3_top.plot(t, width, lw=2, label="Width (stable)")
    ax3_top.set_ylabel("Wall width (beads)")
    ax3_top.grid(True, alpha=0.3)

    ax3_top_r = ax3_top.twinx()
    ax3_top_r.plot(t, diff, lw=1.5, ls="--", label="D (stable)")
    ax3_top_r.set_ylabel("Diffusion coefficient D")

    ax3_top.set_title(f"{chain_name}: stable wall motion vs topological events")

    bar_w = FRAME_DT * 0.8

    ax3_bot.bar(t, n_nuc, width=bar_w, label="Nucleation (+)", alpha=0.6)
    ax3_bot.bar(t, -n_ann, width=bar_w, label="Annihilation (-)", alpha=0.6)

    ax3_bot.axhline(0, color="k", lw=0.5)
    ax3_bot.set_xlabel("Time")
    ax3_bot.set_ylabel("Events / frame")
    ax3_bot.legend()
    ax3_bot.grid(True, alpha=0.3, axis="y")

    fig3.tight_layout()

    fig3.savefig(
        os.path.join(out_dir, f"{chain_name}_fig3_kinetics_events.png"),
        dpi=SAVE_DPI,
        bbox_inches="tight",
    )
    plt.close(fig3)

    # Figure 4: kymograph
    try:
        phi_mat = np.array(kymograph_matrix, dtype=np.float64)

        if phi_mat.size > 0:
            fig4, ax4 = plt.subplots(figsize=(10, 6))

            limit = np.percentile(np.abs(phi_mat), 99)

            if limit <= 0:
                limit = 1.0

            im = ax4.imshow(
                phi_mat,
                aspect="auto",
                cmap="bwr",
                vmin=-limit,
                vmax=limit,
                extent=[0, phi_mat.shape[1], t[-1], t[0]],
            )

            plt.colorbar(im, label=r"Chirality $\phi$")

            ax4.set_title(f"{chain_name}: spatiotemporal chirality evolution")
            ax4.set_xlabel("Genomic position")
            ax4.set_ylabel("Time")

            fig4.savefig(
                os.path.join(out_dir, f"{chain_name}_fig4_kymograph.png"),
                dpi=SAVE_DPI,
                bbox_inches="tight",
            )
            plt.close(fig4)

    except Exception as e:
        print(f"[WARN] Skipping kymograph for {chain_name}: {e}")


def plot_two_chain_comparison(chain_data, out_dir):
    if len(chain_data) < 2:
        return

    fig, axes = plt.subplots(3, 1, figsize=(9, 11), sharex=True)

    for chain_name, arr in chain_data.items():
        if arr.size == 0:
            continue

        t = arr[:, 0]

        axes[0].plot(t, arr[:, 5], lw=2, label=f"{chain_name}: N_perv")
        axes[1].plot(t, arr[:, 3], lw=2, label=f"{chain_name}: <|phi|>")
        axes[1].plot(t, arr[:, 4], lw=1.5, ls="--", label=f"{chain_name}: |<phi>|")
        axes[2].plot(t, arr[:, 7], lw=2, label=f"{chain_name}: width")

    axes[0].set_ylabel("N perversions")
    axes[1].set_ylabel("Order parameters")
    axes[2].set_ylabel("Wall width")
    axes[2].set_xlabel("Time")

    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")

    fig.suptitle("Two-chain perversion kinetics comparison")
    fig.tight_layout()

    fig.savefig(
        os.path.join(out_dir, "compare_two_chains_kinetics.png"),
        dpi=SAVE_DPI,
        bbox_inches="tight",
    )

    plt.close(fig)


def save_combined_results(chain_arrays, out_dir):
    rows = []

    for chain_index, (chain_name, arr) in enumerate(chain_arrays.items(), start=1):
        if arr.size == 0:
            continue

        for row in arr:
            rows.append(np.concatenate(([chain_index], row)))

    if not rows:
        return None

    combined = np.vstack(rows)

    path = os.path.join(out_dir, "kzm_kinetics_two_chains_combined.txt")

    header = (
        "chain_index time frame_idx mol_id "
        "phi_bar psi n_perv avg_dom_size "
        "avg_width(stable) diff_coeff(stable) "
        "n_nucleation n_annihilation"
    )

    np.savetxt(path, combined, header=header)

    return path


def save_chirality_correlation_results(corr_results, out_dir):
    if not corr_results:
        return None, None

    corr_arr = np.array(corr_results, dtype=np.float64)

    path = os.path.join(out_dir, "chirality_correlation_AB.txt")

    header = "time frame_idx C_norm C_pearson C_sign"

    np.savetxt(path, corr_arr, header=header)

    return corr_arr, path


def plot_chirality_correlation_AB(corr_arr, out_dir):
    if corr_arr is None or corr_arr.size == 0:
        return

    t = corr_arr[:, 0]

    C_norm = corr_arr[:, 2]
    C_pearson = corr_arr[:, 3]
    C_sign = corr_arr[:, 4]

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(
        t,
        C_norm,
        lw=2,
        label=r"$C_{\rm norm}$",
    )

    ax.plot(
        t,
        C_pearson,
        lw=1.8,
        ls="--",
        label="Pearson",
    )

    ax.plot(
        t,
        C_sign,
        lw=1.5,
        ls=":",
        label="Sign correlation",
    )

    ax.axhline(0, color="k", lw=0.8)
    ax.axhline(1, color="gray", lw=0.5, ls="--")
    ax.axhline(-1, color="gray", lw=0.5, ls="--")

    ax.set_xlabel("Time")
    ax.set_ylabel("A-B chirality correlation")
    ax.set_ylim(-1.05, 1.05)
    ax.set_title("Chirality correlation between chain A and chain B")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    fig.savefig(
        os.path.join(out_dir, "chirality_correlation_AB.png"),
        dpi=SAVE_DPI,
        bbox_inches="tight",
    )

    plt.close(fig)


def save_AB_same_sign_overlay(overlay_matrix, overlay_times, out_dir):
    """
    Save A-B same-sign chirality overlay matrix.

    Values:
        +1 : both positive
        -1 : both negative
         0 : otherwise
    """

    if len(overlay_matrix) == 0:
        return None

    min_len = min(len(row) for row in overlay_matrix)
    if min_len <= 0:
        return None

    mat = np.array([row[:min_len] for row in overlay_matrix], dtype=np.int8)
    times = np.asarray(overlay_times, dtype=np.float64)

    matrix_path = os.path.join(out_dir, "AB_chirality_same_sign_overlay_matrix.txt")
    time_path = os.path.join(out_dir, "AB_chirality_same_sign_overlay_times.txt")

    np.savetxt(
        matrix_path,
        mat,
        fmt="%d",
        header=(
            "A-B same-sign chirality overlay matrix. "
            "+1 both positive, -1 both negative, 0 otherwise. "
            "Rows are sampled frames, columns are genomic positions."
        ),
    )

    np.savetxt(
        time_path,
        times,
        fmt="%.8f",
        header="Times corresponding to rows of AB_chirality_same_sign_overlay_matrix.txt",
    )

    return mat, times, matrix_path, time_path


def plot_AB_same_sign_overlay(overlay_matrix, overlay_times, out_dir):
    """
    Plot A-B same-sign chirality overlay.

    Values:
        +1 : both positive
        -1 : both negative
         0 : otherwise
    """

    if len(overlay_matrix) == 0:
        return

    min_len = min(len(row) for row in overlay_matrix)
    if min_len <= 0:
        return

    mat = np.array([row[:min_len] for row in overlay_matrix], dtype=np.int8)
    times = np.asarray(overlay_times, dtype=np.float64)

    if mat.size == 0 or len(times) == 0:
        return

    # Color convention:
    # -1: blue, 0: white, +1: red
    cmap = ListedColormap(["blue", "white", "red"])
    norm = BoundaryNorm([-1.5, -0.5, 0.5, 1.5], cmap.N)

    fig, ax = plt.subplots(figsize=(10, 6))

    if len(times) >= 2:
        y_min = times[-1]
        y_max = times[0]
    else:
        y_min = 1.0
        y_max = 0.0

    im = ax.imshow(
        mat,
        aspect="auto",
        cmap=cmap,
        norm=norm,
        interpolation="nearest",
        extent=[0, mat.shape[1], y_min, y_max],
    )

    cbar = plt.colorbar(im, ax=ax, ticks=[-1, 0, 1])
    cbar.ax.set_yticklabels(["both negative", "others", "both positive"])

    ax.set_title("A-B same-sign chirality overlay")
    ax.set_xlabel("Genomic position")
    ax.set_ylabel("Time")

    fig.tight_layout()

    fig.savefig(
        os.path.join(out_dir, "AB_chirality_same_sign_overlay.png"),
        dpi=SAVE_DPI,
        bbox_inches="tight",
    )

    fig.savefig(
        os.path.join(out_dir, "AB_chirality_same_sign_overlay.pdf"),
        bbox_inches="tight",
    )

    plt.close(fig)


# ============================================================
# 6. Main
# ============================================================

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    if not os.path.exists(INPUT_DUMP):
        raise FileNotFoundError(f"Cannot find input dump file: {INPUT_DUMP}")

    print("Two-chain perversion kinetics + A-B chirality correlation + overlay")
    print("===================================================================")
    print(f"Input dump : {INPUT_DUMP}")
    print(f"Output dir : {OUT_DIR}")
    print(f"UNWRAP_COORDS = {UNWRAP_COORDS}")
    print()
    print("A/B chain convention:")
    print(f"  chain_A: atom id <= {CHAIN_A_MAX_ID}")
    print(f"  chain_B: atom id >= {CHAIN_B_MIN_ID}")
    print("  Both chains are sorted by increasing atom id.")
    print()
    print("Overlay convention:")
    print(f"  OVERLAY_EPS = {OVERLAY_EPS}")
    print("  +1 / red  : phi_A > eps and phi_B > eps")
    print("  -1 / blue : phi_A < -eps and phi_B < -eps")
    print("   0 / white: other cases")
    print()

    lines, frame_ranges = split_dump_into_frames(INPUT_DUMP)

    sampled = [
        (idx, rng)
        for idx, rng in enumerate(frame_ranges)
        if idx % STRIDE == 0
    ]

    if not sampled:
        sampled = [
            (idx, rng)
            for idx, rng in enumerate(frame_ranges)
        ]

    print(
        f"Found {len(frame_ranges)} frames. "
        f"Analyzing {len(sampled)} sampled frames with STRIDE={STRIDE}."
    )

    first_start, first_end = sampled[0][1]

    _, header, first_atoms = parse_one_frame(lines[first_start:first_end])

    mol_ids_found = sorted(set(first_atoms[:, 1].astype(int).tolist()))

    mol_ids = [
        m for m in MOL_IDS_TO_ANALYZE
        if m in mol_ids_found
    ]

    if not mol_ids:
        raise RuntimeError(
            f"No requested mol IDs found. "
            f"Found {mol_ids_found}, requested {MOL_IDS_TO_ANALYZE}"
        )

    print(f"Atom header: {header}")
    print(f"Mol IDs found: {mol_ids_found}")
    print(f"Mol IDs analyzed: {mol_ids}")

    atom_ids_first = first_atoms[:, 0].astype(int)

    n_A = int(np.sum(atom_ids_first <= CHAIN_A_MAX_ID))
    n_B = int(np.sum(atom_ids_first >= CHAIN_B_MIN_ID))

    print(f"  chain_A by id <= {CHAIN_A_MAX_ID}: {n_A} beads")
    print(f"  chain_B by id >= {CHAIN_B_MIN_ID}: {n_B} beads")
    print()

    results = {
        MOL_TO_CHAIN_NAME.get(m, f"chain_mol{m}"): []
        for m in mol_ids
    }

    kymographs = {
        MOL_TO_CHAIN_NAME.get(m, f"chain_mol{m}"): []
        for m in mol_ids
    }

    prev_perv = {
        MOL_TO_CHAIN_NAME.get(m, f"chain_mol{m}"): None
        for m in mol_ids
    }

    chirality_corr_results = []

    AB_overlay_kymograph = []
    AB_overlay_times = []

    for sampled_count, (frame_idx, (start, end)) in enumerate(sampled, start=1):
        try:
            timestep, header, atoms = parse_one_frame(lines[start:end])

        except Exception as e:
            print(f"[ERROR] Failed to parse frame {frame_idx}: {e}")

            for mol_id in mol_ids:
                chain_name = MOL_TO_CHAIN_NAME.get(mol_id, f"chain_mol{mol_id}")
                prev_perv[chain_name] = None

            continue

        print(
            f"Frame {frame_idx:6d} / sampled {sampled_count:5d} "
            f"| timestep={timestep:g}"
        )

        phi_this_frame = {}

        for mol_id in mol_ids:
            chain_name = MOL_TO_CHAIN_NAME.get(mol_id, f"chain_mol{mol_id}")

            coords = get_chain_coordinates(atoms, mol_id)

            if coords is None:
                print(f"  [WARN] {chain_name}: no atoms found.")
                prev_perv[chain_name] = None
                continue

            try:
                phi_bar, psi, perv_idx, avg_dom, phi_arr = process_chain(coords)

                phi_this_frame[chain_name] = phi_arr

                avg_width, D_coeff, n_nuc, n_ann = analyze_wall_physics_robust(
                    phi_arr,
                    perv_idx,
                    prev_perv[chain_name],
                    FRAME_DT,
                    search_radius=SEARCH_RADIUS,
                )

                prev_perv[chain_name] = perv_idx

                n_perv = len(perv_idx)

                results[chain_name].append(
                    [
                        timestep,
                        frame_idx,
                        mol_id,
                        phi_bar,
                        psi,
                        n_perv,
                        avg_dom,
                        avg_width,
                        D_coeff,
                        n_nuc,
                        n_ann,
                    ]
                )

                kymographs[chain_name].append(phi_arr)

                p_w = avg_width if not np.isnan(avg_width) else 0.0
                p_d = D_coeff if not np.isnan(D_coeff) else 0.0

                print(
                    f"  {chain_name:<8s} mol={mol_id} "
                    f"| Psi={psi:.3f} "
                    f"| <|phi|>={phi_bar:.3f} "
                    f"| N={n_perv:<3d} "
                    f"| W={p_w:.1f} "
                    f"| D={p_d:.3f} "
                    f"| +{n_nuc}/-{n_ann}"
                )

            except Exception as e:
                print(f"  [ERROR] {chain_name}: {e}")
                prev_perv[chain_name] = None

        # Compute A-B chirality correlation and same-sign overlay for this frame
        if ("chain_A" in phi_this_frame) and ("chain_B" in phi_this_frame):
            phi_A = phi_this_frame["chain_A"]
            phi_B = phi_this_frame["chain_B"]

            C_norm, C_pearson, C_sign = chirality_correlation(phi_A, phi_B)

            chirality_corr_results.append(
                [
                    timestep,
                    frame_idx,
                    C_norm,
                    C_pearson,
                    C_sign,
                ]
            )

            overlay_row = build_AB_same_sign_overlay(
                phi_A,
                phi_B,
                eps=OVERLAY_EPS,
            )

            AB_overlay_kymograph.append(overlay_row)
            AB_overlay_times.append(timestep)

            n_both_pos = int(np.sum(overlay_row == 1))
            n_both_neg = int(np.sum(overlay_row == -1))
            n_other = int(np.sum(overlay_row == 0))

            print(
                f"  Chirality corr A-B "
                f"| C_norm={C_norm:.4f} "
                f"| C_pearson={C_pearson:.4f} "
                f"| C_sign={C_sign:.4f}"
            )

            print(
                f"  A-B overlay "
                f"| both +: {n_both_pos} "
                f"| both -: {n_both_neg} "
                f"| other: {n_other}"
            )

    print()
    print("Saving results...")

    chain_arrays = {}

    for mol_id in mol_ids:
        chain_name = MOL_TO_CHAIN_NAME.get(mol_id, f"chain_mol{mol_id}")

        if not results[chain_name]:
            print(f"[WARN] No valid results for {chain_name}.")
            chain_arrays[chain_name] = np.array([])
            continue

        arr, path = save_chain_results(
            chain_name,
            results[chain_name],
            OUT_DIR,
        )

        chain_arrays[chain_name] = arr

        print(f"  Saved {chain_name}: {path}")

        plot_single_chain(
            chain_name,
            arr,
            kymographs[chain_name],
            OUT_DIR,
        )

    combined_path = save_combined_results(chain_arrays, OUT_DIR)

    if combined_path is not None:
        print(f"  Saved combined table: {combined_path}")

    plot_two_chain_comparison(chain_arrays, OUT_DIR)

    corr_arr, corr_path = save_chirality_correlation_results(
        chirality_corr_results,
        OUT_DIR,
    )

    if corr_path is not None:
        print(f"  Saved A-B chirality correlation: {corr_path}")
        plot_chirality_correlation_AB(corr_arr, OUT_DIR)

    overlay_save = save_AB_same_sign_overlay(
        AB_overlay_kymograph,
        AB_overlay_times,
        OUT_DIR,
    )

    if overlay_save is not None:
        mat, times, matrix_path, time_path = overlay_save
        print(f"  Saved A-B same-sign overlay matrix: {matrix_path}")
        print(f"  Saved A-B same-sign overlay times : {time_path}")

        plot_AB_same_sign_overlay(
            AB_overlay_kymograph,
            AB_overlay_times,
            OUT_DIR,
        )

        print("  Saved A-B same-sign overlay figure:")
        print(f"    {os.path.join(OUT_DIR, 'AB_chirality_same_sign_overlay.png')}")
        print(f"    {os.path.join(OUT_DIR, 'AB_chirality_same_sign_overlay.pdf')}")

    print()
    print("Done.")
    print(f"All outputs are in: {OUT_DIR}")


if __name__ == "__main__":
    main()