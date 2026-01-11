#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Time evolution of chirality order parameter and correlation length,
with user-configurable line colors.

Outputs:
- chirality_time_series.txt
- phi_bar_vs_time.png
- xi_vs_time.png
"""

import os
import re
import glob
import math
import numpy as np
import matplotlib.pyplot as plt

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
# Order parameter + corrlen
# =========================
def chirality_order_parameter(phi):
    phi = np.asarray(phi, dtype=np.float64)
    return float(np.mean(np.abs(phi)))


def autocorr_1d(x):
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    if n < 3:
        return np.ones(n, dtype=np.float64)

    x = x - np.mean(x)
    var = np.mean(x * x)
    if var <= 0:
        return np.ones(n, dtype=np.float64)

    m = 1 << (2 * n - 1).bit_length()
    fx = np.fft.rfft(x, n=m)
    ac = np.fft.irfft(fx * np.conjugate(fx), n=m)[:n]

    denom = np.arange(n, 0, -1, dtype=np.float64)  # (n-r)
    ac = ac / denom
    C = ac / ac[0]
    return C


def correlation_length_from_C(C, method="1e"):
    C = np.asarray(C, dtype=np.float64)
    n = C.size
    if n < 2:
        return 0.0

    if method == "1e":
        target = math.exp(-1.0)
        idx = np.where(C <= target)[0]
        if len(idx) == 0:
            return float(n - 1)
        r1 = int(idx[0])
        if r1 == 0:
            return 0.0
        r0 = r1 - 1
        y0, y1 = C[r0], C[r1]
        if y0 == y1:
            return float(r1)
        t = (target - y0) / (y1 - y0)
        return float(r0 + t)

    if method == "int":
        neg = np.where(C[1:] < 0)[0]
        rmax = int(neg[0] + 1) if len(neg) > 0 else (n - 1)
        return float(np.sum(C[:rmax + 1]))

    raise ValueError("Unknown method.")


# =========================
# Frame discovery + reading
# =========================
def extract_frame_index(path):
    name = os.path.basename(path)
    m = re.search(r"frame[_-]?(\d+)\.txt$", name)
    if not m:
        return None
    return int(m.group(1))


def list_frame_files(patterns=("frame_*.txt", "frame*.txt")):
    files = []
    for pat in patterns:
        files.extend(glob.glob(pat))
    items = []
    for f in files:
        j = extract_frame_index(f)
        if j is not None:
            items.append((j, f))
    items.sort(key=lambda x: x[0])
    return items


def load_beads_positions(path, xyz_cols=(1, 4)):
    data = np.loadtxt(path)
    start, end = xyz_cols
    pos = data[:, start:end]
    return np.ascontiguousarray(pos)


# =========================
# Per-frame computation
# =========================
def compute_phi_gpu(beads_positions, T, threadsperblock=128):
    n_beads = beads_positions.shape[0]
    n_phi = n_beads - int(1.25 * T)
    if n_phi <= 5:
        raise ValueError(f"Not enough beads ({n_beads}) for T={T}.")

    phis = np.zeros(n_phi, dtype=np.float32)

    beads_d = cuda.to_device(beads_positions)
    phis_d = cuda.to_device(phis)

    blockspergrid = (n_phi + (threadsperblock - 1)) // threadsperblock
    calculate_phi_kernel[blockspergrid, threadsperblock](beads_d, phis_d, T)
    return phis_d.copy_to_host()


def smooth_phi(phi, window=101, polyorder=3):
    phi = np.asarray(phi, dtype=np.float64)
    if phi.size < 7:
        return phi.copy()
    w = int(window)
    if w >= phi.size:
        w = max(5, (phi.size // 2) * 2 - 1)
    if w < 5:
        w = 5
    if w % 2 == 0:
        w += 1
    return savgol_filter(phi, window_length=w, polyorder=polyorder)


def compute_observables_for_frame(path, T, xyz_cols=(1, 4),
                                  sg_window=101, sg_poly=3,
                                  use_smoothed_for_OP=True):
    beads = load_beads_positions(path, xyz_cols=xyz_cols)
    phi = compute_phi_gpu(beads, T=T)
    phi_s = smooth_phi(phi, window=sg_window, polyorder=sg_poly)

    phi_for = phi_s if use_smoothed_for_OP else phi
    phi_bar = chirality_order_parameter(phi_for)

    C = autocorr_1d(phi_for)
    xi_1e = correlation_length_from_C(C, method="1e")
    xi_int = correlation_length_from_C(C, method="int")

    return phi_bar, xi_1e, xi_int


# =========================
# Main
# =========================
if __name__ == "__main__":
    # -------- Config (edit here) --------
    stride = 20
    T = 125 * 1.25
    xyz_cols = (1, 4)
    sg_window = 101
    sg_poly = 3
    use_smoothed_for_OP = True

    dt = None  # e.g. dt = 1.0 if you want physical time

    # -------- NEW: Line colors (edit here) --------
    # Any matplotlib color is OK: 'k', 'r', 'tab:blue', '#00ff00', etc.
    COLOR_PHI_BAR = '#5EBEA3'
    COLOR_XI_1E = '#FB6542'
    COLOR_XI_INT = '#375E97'

    # Optional styling knobs
    LINEWIDTH = 2.0
    MARKER = None          # e.g. 'o' if you want markers
    MARKERSIZE = 3

    # -------- Find frames --------
    items = list_frame_files()
    if not items:
        raise FileNotFoundError("No frame files found (frame_*.txt / frame*.txt) in current directory.")

    sampled = [(j, f) for (j, f) in items if (j % stride == 0)]
    if not sampled:
        sampled = items[::stride]

    # -------- Loop --------
    out_rows = []
    for j, path in sampled:
        try:
            phi_bar, xi_1e, xi_int = compute_observables_for_frame(
                path, T=T, xyz_cols=xyz_cols,
                sg_window=sg_window, sg_poly=sg_poly,
                use_smoothed_for_OP=use_smoothed_for_OP
            )
            t = (j * dt) if (dt is not None) else float(j)
            out_rows.append([t, j, phi_bar, xi_1e, xi_int])
            print(f"[OK] j={j:8d}  phi_bar={phi_bar:.6g}  xi_1e={xi_1e:.3f}  xi_int={xi_int:.3f}  ({os.path.basename(path)})")
        except Exception as e:
            print(f"[SKIP] j={j} file={path} error={e}")

    if not out_rows:
        raise RuntimeError("No frames successfully processed.")

    out = np.array(out_rows, dtype=np.float64)
    np.savetxt(
        "chirality_time_series.txt",
        out,
        header="t  frame_index_j  phi_bar(mean|phi|)  xi_1e(beads)  xi_int(beads)"
    )

    t = out[:, 0]
    phi_bar_ts = out[:, 2]
    xi_1e_ts = out[:, 3]
    xi_int_ts = out[:, 4]

    xlabel = "time" if dt is not None else "frame index j"

    # -------- Plot: phi_bar(t) --------
    fig1, ax1 = plt.subplots(figsize=(7.0, 4.0))
    ax1.plot(
        t, phi_bar_ts,
        linewidth=LINEWIDTH,
        color=COLOR_PHI_BAR,
        marker=MARKER,
        markersize=MARKERSIZE
    )
    ax1.set_xlabel(xlabel, fontsize=12)
    ax1.set_ylabel(r"$\bar{\phi}=\langle|\phi|\rangle$", fontsize=12)
    ax1.set_title("Chirality order parameter vs time", fontsize=13)
    fig1.tight_layout()
    fig1.savefig("phi_bar_vs_time.png", dpi=300, bbox_inches="tight")

    # -------- Plot: xi(t) --------
    fig2, ax2 = plt.subplots(figsize=(7.0, 4.0))
    ax2.plot(
        t, xi_1e_ts,
        linewidth=LINEWIDTH,
        color=COLOR_XI_1E,
        marker=MARKER,
        markersize=MARKERSIZE,
        label=r"$\xi_{1/e}$"
    )
    ax2.plot(
        t, xi_int_ts,
        linewidth=LINEWIDTH,
        color=COLOR_XI_INT,
        marker=MARKER,
        markersize=MARKERSIZE,
        label=r"$\xi_{\rm int}$"
    )
    ax2.set_xlabel(xlabel, fontsize=12)
    ax2.set_ylabel("Correlation length (beads)", fontsize=12)
    ax2.set_title("Correlation length vs time", fontsize=13)
    ax2.legend()
    fig2.tight_layout()
    fig2.savefig("xi_vs_time.png", dpi=300, bbox_inches="tight")

    plt.show()
