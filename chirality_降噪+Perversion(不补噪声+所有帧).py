#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modified analysis script for Motorized Chromosome KZM studies.

Key Additions:
1. Global Order Parameter (Psi): Measures net chirality (symmetry breaking).
2. Average Domain Size: Tracks coarsening dynamics.
3. Kymograph: Visualizes spatiotemporal evolution of chirality.

Outputs:
- observables.txt: Time series of all metrics.
- kymograph_chirality.png: Spacetime heatmap.
- comparison_order_parameters.png: Local vs Global chirality.
- ... other plots.
"""

import os
import re
import glob
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from numba import cuda
from scipy.signal import savgol_filter


# =========================
# GPU helpers + kernel
# =========================
@cuda.jit(device=True)
def magnitude(vector):
    return math.sqrt(vector[0] ** 2 + vector[1] ** 2 + vector[2] ** 2)


@cuda.jit(device=True)
def cross(v1, v2):
    return (v1[1] * v2[2] - v1[2] * v2[1],
            v1[2] * v2[0] - v1[0] * v2[2],
            v1[0] * v2[1] - v1[1] * v2[0])


@cuda.jit(device=True)
def dot(u, v):
    return u[0] * v[0] + u[1] * v[1] + u[2] * v[2]


@cuda.jit(device=True)
def sub(v1, v2):
    return (v1[0] - v2[0], v1[1] - v2[1], v1[2] - v2[2])


@cuda.jit(device=True)
def add(v1, v2):
    return (v1[0] + v2[0], v1[1] + v2[1], v1[2] + v2[2])


@cuda.jit(device=True)
def divide(v, divisor):
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

        E = divide(add(A, B), 2.0)
        F = divide(add(C, D), 2.0)

        AB = sub(B, A)
        CD = sub(D, C)
        EF = sub(F, E)

        cp = cross(CD, AB)
        dp = dot(EF, cp)

        mags = magnitude(EF) * magnitude(CD) * magnitude(AB)
        if mags > 0.0:
            phis[i] = dp / mags
        else:
            phis[i] = 0.0


# =========================
# Analysis Functions
# =========================
def calculate_order_parameters(phi):
    """
    Returns:
    - phi_bar: Local chirality strength <|phi|> (How twisted is it locally?)
    - psi: Global order parameter |<phi>| (Is the whole thing left or right?)
    """
    phi = np.asarray(phi, dtype=np.float64)
    phi_bar = float(np.mean(np.abs(phi)))
    psi = float(np.abs(np.mean(phi)))
    return phi_bar, psi


def autocorr_1d(x):
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    if n < 3:
        return np.ones(n, dtype=np.float64)
    x = x - np.mean(x)
    var = np.mean(x * x)
    if var <= 0:
        return np.ones(n, dtype=np.float64)
    # FFT autocorr
    m = 1 << (2 * n - 1).bit_length()
    fx = np.fft.rfft(x, n=m)
    ac = np.fft.irfft(fx * np.conjugate(fx), n=m)[:n]
    denom = np.arange(n, 0, -1, dtype=np.float64)
    ac = ac / denom
    return ac / ac[0]


def correlation_length_from_C(C, method="1e"):
    C = np.asarray(C, dtype=np.float64)
    n = C.size
    if n < 2: return 0.0
    if method == "1e":
        target = math.exp(-1.0)
        idx = np.where(C <= target)[0]
        if len(idx) == 0: return float(n - 1)
        r1 = int(idx[0])
        if r1 == 0: return 0.0
        r0 = r1 - 1
        y0, y1 = C[r0], C[r1]
        if y0 == y1: return float(r1)
        t = (target - y0) / (y1 - y0)
        return float(r0 + t)
    return 0.0


def find_perversions_and_domains(phi_smooth, eps=0.05, min_gap=20):
    """
    Returns:
    - perv_idx: Indices of domain walls.
    - avg_domain_size: Average size of chiral domains (beads).
    """
    phi_smooth = np.asarray(phi_smooth, dtype=np.float64)
    s = np.zeros_like(phi_smooth, dtype=np.int8)
    s[phi_smooth > eps] = 1
    s[phi_smooth < -eps] = -1

    # Fill zeros forward
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
            return np.array([], dtype=int), float(len(phi_smooth))  # No perversions

    flips = np.where(np.diff(s) != 0)[0]

    # Merge nearby flips
    merged = []
    for f in flips:
        if (not merged) or (f - merged[-1] >= min_gap):
            merged.append(int(f))

    perv_idx = np.array(merged, dtype=int)

    # Calculate Domain Sizes
    total_len = len(phi_smooth)
    if len(perv_idx) == 0:
        avg_size = float(total_len)
    else:
        # Include start(0) and end(total_len) to get segment lengths
        boundaries = np.concatenate(([0], perv_idx, [total_len]))
        sizes = np.diff(boundaries)
        avg_size = np.mean(sizes)

    return perv_idx, avg_size


# =========================
# IO & GPU Compute
# =========================
def list_frame_files(patterns=("frame_*.txt", "frame*.txt")):
    files = []
    for pat in patterns:
        files.extend(glob.glob(pat))

    def extract_index(name):
        m = re.search(r"frame[_-]?(\d+)\.txt$", name)
        return int(m.group(1)) if m else -1

    files = sorted([f for f in files if extract_index(f) != -1], key=extract_index)
    return [(extract_index(f), f) for f in files]


def load_beads_positions(path, xyz_cols=(1, 4)):
    # Note: For huge datasets, consider using pandas.read_csv or binary formats
    data = np.loadtxt(path)
    return np.ascontiguousarray(data[:, xyz_cols[0]:xyz_cols[1]])


def compute_phi_gpu(beads_positions, T, threadsperblock=128):
    n_beads = beads_positions.shape[0]
    n_phi = n_beads - int(1.25 * T)
    phis = np.zeros(n_phi, dtype=np.float32)
    beads_d = cuda.to_device(beads_positions)
    phis_d = cuda.to_device(phis)
    blocks = (n_phi + threadsperblock - 1) // threadsperblock
    calculate_phi_kernel[blocks, threadsperblock](beads_d, phis_d, T)
    return phis_d.copy_to_host()


def smooth_phi(phi, window=101, polyorder=3):
    w = int(window)
    if w >= phi.size: w = max(5, (phi.size // 2) * 2 - 1)
    if w < 5: w = 5
    if w % 2 == 0: w += 1
    return savgol_filter(phi, w, polyorder)


def process_frame(path, T, xyz_cols, sg_window, perv_eps, perv_gap):
    beads = load_beads_positions(path, xyz_cols)
    phi_raw = compute_phi_gpu(beads, T)
    phi_s = smooth_phi(phi_raw, window=sg_window)

    # Metrics
    phi_bar, psi = calculate_order_parameters(phi_s)
    C = autocorr_1d(phi_s)
    xi_1e = correlation_length_from_C(C, method="1e")
    perv_idx, avg_dom_size = find_perversions_and_domains(phi_s, eps=perv_eps, min_gap=perv_gap)

    return phi_bar, psi, xi_1e, len(perv_idx), avg_dom_size, phi_s


# =========================
# Main
# =========================
if __name__ == "__main__":
    # --- Parameters ---
    stride = 20  # Analyze every Nth frame
    T = 125 * 1.25  # Chirality scale
    xyz_cols = (1, 4)  # Column indices for X,Y,Z
    sg_window = 101  # Smoothing window
    perv_eps = 0.05  # Threshold for domain detection
    perv_gap = 20  # Minimum domain wall separation
    dt = 1.0  # Time step per frame index

    # --- Setup ---
    items = list_frame_files()
    sampled = [(j, f) for (j, f) in items if (j % stride == 0)]
    if not sampled: sampled = items[::stride]

    results = []
    kymograph_data = []  # Stores phi arrays for heatmap

    print(f"Processing {len(sampled)} frames...")

    for j, path in sampled:
        try:
            phi_bar, psi, xi, n_perv, avg_dom, phi_arr = process_frame(
                path, T, xyz_cols, sg_window, perv_eps, perv_gap
            )

            t = j * dt
            results.append([t, j, phi_bar, psi, xi, n_perv, avg_dom])
            kymograph_data.append(phi_arr)

            print(f"j={j:<6} Psi={psi:.3f} xi={xi:.1f} N_perv={n_perv:<3} L_dom={avg_dom:.1f}")

        except Exception as e:
            print(f"Error frame {j}: {e}")

    # --- Save Data ---
    out_arr = np.array(results)
    header = "t j phi_bar(local_mag) psi(global_order) xi_1e n_perv avg_domain_size"
    np.savetxt("observables.txt", out_arr, header=header)

    # --- Visualization ---
    t = out_arr[:, 0]
    phi_bar = out_arr[:, 2]
    psi = out_arr[:, 3]
    xi = out_arr[:, 4]
    n_perv = out_arr[:, 5]
    avg_dom = out_arr[:, 6]

    # 1. Local vs Global Order Parameter (KZM Signature)
    fig1, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(t, phi_bar, label=r"Local Magnitude $\langle|\phi|\rangle$", color='#5EBEA3', lw=2)
    ax1.plot(t, psi, label=r"Global Order $|\langle\phi\rangle|$", color='k', linestyle='--', lw=2)
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Order Parameter")
    ax1.set_title("Symmetry Breaking: Local vs Global Chirality")
    ax1.legend()
    fig1.savefig("comparison_order_parameters.png", dpi=300)

    # 2. Perversions and Domain Size (Coarsening)
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    ax2.plot(t, n_perv, color='#AA66CC', label="Number of Perversions", lw=2)
    ax2.set_ylabel(r"$N_{perv}$", color='#AA66CC')
    ax2.tick_params(axis='y', labelcolor='#AA66CC')

    ax3 = ax2.twinx()
    ax3.plot(t, avg_dom, color='#FB6542', label="Avg Domain Size", lw=2, linestyle=':')
    ax3.set_ylabel("Domain Size (beads)", color='#FB6542')
    ax3.tick_params(axis='y', labelcolor='#FB6542')

    ax2.set_title("Defect Dynamics & Coarsening")
    fig2.savefig("perversions_and_domains.png", dpi=300)

    # 3. Kymograph (Spacetime Heatmap)
    # Convert list of arrays to 2D matrix.
    # Warning: Assumes all frames have same polymer length.
    # If length changes, this part needs padding.
    try:
        phi_matrix = np.array(kymograph_data)  # Shape: (Time, Space)

        fig3, ax4 = plt.subplots(figsize=(10, 6))
        # Use a diverging colormap (Blue=Left, Red=Right, White=0)
        limit = np.percentile(np.abs(phi_matrix), 98)  # Auto-scale color
        im = ax4.imshow(phi_matrix, aspect='auto', cmap='bwr',
                        vmin=-limit, vmax=limit, interpolation='nearest',
                        extent=[0, phi_matrix.shape[1], t[-1], t[0]])  # Y-axis: Time (Top to Bottom?) or Bottom to Top

        # Standard physics convention: Time flows upwards or downwards.
        # Here origin='upper' (default) means index 0 is at top.
        # Let's flip extent so time increases upwards if desired, or keep as matrix index.
        # Let's label explicitly.

        cbar = plt.colorbar(im, ax=ax4)
        cbar.set_label(r"Chirality $\phi$")
        ax4.set_xlabel("Genomic Position (beads)")
        ax4.set_ylabel("Time")
        ax4.set_title("Spatiotemporal Evolution of Chirality (Kymograph)")
        fig3.savefig("kymograph_chirality.png", dpi=300)

    except Exception as e:
        print(f"Could not plot kymograph (likely varying chain lengths): {e}")

    plt.show()
    print("Done. Generated observables.txt and 3 PNG plots.")