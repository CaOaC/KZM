#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Full Analysis Suite for Motorized Chromosome KZM & Kinetics (Robust Version)
============================================================================
Updates:
- Implemented 'Lagrangian Tracking' to separate stable motion from topological events.
- Excluded nucleation/annihilation artifacts from Diffusion and Width calculations.
- Added explicit counting of Nucleation (+) and Annihilation (-) events.
- [NEW] Kappa(t): Read kappavstime file and plot ONLY the 1st segment (kI1, kII1),
  integrated into Fig.1 (Order Parameters) as twin y-axis. NOTE: kappa time uses (tk/100).
- [NEW] Fig.5: Plot local chirality order parameter(s) vs kappa_II (segment 1)
  by interpolating kappa_II(t) onto the analysis-frame time grid t.
"""

import os
import re
import glob
import math
import numpy as np
import matplotlib.pyplot as plt
from numba import cuda
from scipy.signal import savgol_filter


# ==========================================
# 1. GPU Kernels (CUDA) - 保持不变
# ==========================================
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
        if mags > 1e-12:
            phis[i] = dp / mags
        else:
            phis[i] = 0.0


# ==========================================
# 2. 物理分析模块 (CPU)
# ==========================================

def calculate_order_parameters(phi):
    phi = np.asarray(phi, dtype=np.float64)
    phi_bar = float(np.mean(np.abs(phi)))     # <|phi|>
    psi = float(np.abs(np.mean(phi)))         # |<phi>|
    return phi_bar, psi


def find_perversions_and_domains(phi_smooth, eps=0.05, min_gap=20):
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
        sizes = np.diff(boundaries)
        avg_size = np.mean(sizes)

    return perv_idx, avg_size


def analyze_wall_physics_robust(phi, curr_idx, prev_idx, dt, lattice_spacing=1.0, search_radius=40.0):
    """
    鲁棒版物理分析函数：
    1) 仅对“稳定追踪”到的缺陷计算 D 和 Width，剔除湮灭/产生时的数值尖峰。
    2) 分别统计产生 (Nucleation) 和 湮灭 (Annihilation) 事件。
    """
    phi = np.asarray(phi)
    amplitude = 2.0 * np.mean(np.abs(phi))

    matched_widths = []
    valid_displacements = []

    matched_curr_indices = set()
    matched_prev_indices = set()

    # 1) Tracking
    if (prev_idx is not None) and (len(prev_idx) > 0) and (len(curr_idx) > 0):
        for i, curr_pos in enumerate(curr_idx):
            dists = np.abs(prev_idx - curr_pos)
            min_arg = np.argmin(dists)
            min_dist = dists[min_arg]

            if min_dist < search_radius:
                if min_arg not in matched_prev_indices:
                    matched_prev_indices.add(min_arg)
                    matched_curr_indices.add(i)

                    # A) MSD contribution
                    dx2 = (min_dist * lattice_spacing) ** 2
                    valid_displacements.append(dx2)

                    # B) Width from slope
                    grad = np.gradient(phi)
                    idx = int(curr_pos)
                    if 0 < idx < len(phi) - 1:
                        slope = abs(grad[idx])
                        if slope > 1e-9:
                            w = amplitude / slope
                            if w < 400:
                                matched_widths.append(w)

    # 2) Topological events
    n_nucleation = 0
    n_annihilation = 0
    if prev_idx is not None:
        n_annihilation = len(prev_idx) - len(matched_prev_indices)
        n_nucleation = len(curr_idx) - len(matched_curr_indices)

    # 3) Summaries
    avg_width = np.mean(matched_widths) if len(matched_widths) > 0 else np.nan
    if len(valid_displacements) > 0:
        msd = np.mean(valid_displacements)
        D_inst = msd / (2 * dt)
    else:
        D_inst = np.nan

    return avg_width, D_inst, n_nucleation, n_annihilation


# ==========================================
# 2.5 读 κ(t) 模块（从上传脚本移植）
# ==========================================
def read_kappa_file_first_segment(path):
    """
    File format per row (polymer split into 10 segments):
      t  kI1 kII1  kI2 kII2 ... kI10 kII10
    We only use the 1st segment: kI1 and kII1 (columns 2 and 3).
    Returns sorted arrays: tk, kI1, kII1
    """
    rows_t, rows_kI, rows_kII = [], [], []
    with open(path, "r") as f:
        for line in f:
            s = line.strip()
            if (not s) or s.startswith("#"):
                continue
            parts = s.split()
            if len(parts) < 3:
                continue
            t = float(parts[0])
            kI1 = float(parts[1])
            kII1 = float(parts[2])
            rows_t.append(t)
            rows_kI.append(kI1)
            rows_kII.append(kII1)

    if len(rows_t) == 0:
        raise RuntimeError(f"No valid data read from {path}")

    tk = np.asarray(rows_t, dtype=np.float64)
    kI1 = np.asarray(rows_kI, dtype=np.float64)
    kII1 = np.asarray(rows_kII, dtype=np.float64)

    order = np.argsort(tk)
    return tk[order], kI1[order], kII1[order]


# ==========================================
# 3. 辅助 IO 与处理流程
# ==========================================
def list_frame_files():
    files = glob.glob("frame*.txt")

    def extract_index(name):
        m = re.search(r"(\d+)\.txt$", name)
        return int(m.group(1)) if m else -1

    files = sorted([f for f in files if extract_index(f) != -1], key=extract_index)
    return [(extract_index(f), f) for f in files]


def load_beads(path, cols=(1, 4)):
    try:
        data = np.loadtxt(path)
        return np.ascontiguousarray(data[:, cols[0]:cols[1]])
    except Exception as e:
        print(f"Read error {path}: {e}")
        return None


def compute_phi_gpu(beads, T, block_size=128):
    n_beads = beads.shape[0]
    n_phi = n_beads - int(1.25 * T)
    phis = np.zeros(n_phi, dtype=np.float32)

    d_beads = cuda.to_device(beads)
    d_phis = cuda.to_device(phis)

    grid_size = (n_phi + block_size - 1) // block_size
    calculate_phi_kernel[grid_size, block_size](d_beads, d_phis, T)

    return d_phis.copy_to_host()


def process_frame(path, T, cols, sg_win, eps, gap):
    beads = load_beads(path, cols)
    if beads is None:
        return None

    phi_raw = compute_phi_gpu(beads, T)
    if sg_win > len(phi_raw):
        sg_win = len(phi_raw) // 2 * 2 + 1
    phi_s = savgol_filter(phi_raw, sg_win, 3)

    phi_bar, psi = calculate_order_parameters(phi_s)
    perv_idx, avg_dom = find_perversions_and_domains(phi_s, eps, gap)

    return phi_bar, psi, perv_idx, avg_dom, phi_s


# ==========================================
# 4. 主程序
# ==========================================
if __name__ == "__main__":
    # --- 参数设置 ---
    T_scale = 156.25
    xyz_cols = (1, 4)

    smooth_window = 51
    defect_eps = 0.05
    defect_gap = 20

    sim_dt = 1.0
    stride = 20
    frame_dt = sim_dt * stride

    # --- 初始化 ---
    all_files = list_frame_files()
    sampled_files = [(idx, f) for (idx, f) in all_files if idx % stride == 0]
    if not sampled_files:
        sampled_files = all_files[::stride]

    print(f"Found {len(all_files)} files. Analyzing {len(sampled_files)} frames...")

    results = []
    kymograph_matrix = []
    prev_perv_idx = None

    # --- 循环处理 ---
    for j, path in sampled_files:
        try:
            ret = process_frame(path, T_scale, xyz_cols, smooth_window, defect_eps, defect_gap)
            if ret is None:
                continue

            phi_bar, psi, perv_idx, avg_dom, phi_arr = ret

            avg_width, D_coeff, n_nuc, n_ann = analyze_wall_physics_robust(
                phi_arr, perv_idx, prev_perv_idx, frame_dt
            )
            prev_perv_idx = perv_idx

            time = j * sim_dt
            n_perv = len(perv_idx)

            results.append([time, j, phi_bar, psi, n_perv, avg_dom, avg_width, D_coeff, n_nuc, n_ann])
            kymograph_matrix.append(phi_arr)

            p_w = avg_width if not np.isnan(avg_width) else 0.0
            p_d = D_coeff if not np.isnan(D_coeff) else 0.0
            print(f"T={time:<6.0f} | Psi={psi:.2f} | N={n_perv:<3} | W={p_w:.1f} | D={p_d:.3f} | +{n_nuc}/-{n_ann}")

        except Exception as e:
            print(f"Error processing {path}: {e}")
            prev_perv_idx = None

    # --- 保存结果 ---
    data_arr = np.array(results, dtype=np.float64)
    header = "time frame_idx phi_bar psi n_perv avg_dom_size avg_width(stable) diff_coeff(stable) n_nucleation n_annihilation"
    np.savetxt("kzm_kinetics_robust.txt", data_arr, header=header)
    print("\nData saved to kzm_kinetics_robust.txt")

    # ==========================================
    # 5. 绘图 (Visualization)
    # ==========================================
    t = data_arr[:, 0]
    phi_bar = data_arr[:, 2]
    psi = data_arr[:, 3]
    n_perv = data_arr[:, 4]
    avg_dom = data_arr[:, 5]

    width = data_arr[:, 6]
    diff = data_arr[:, 7]

    n_nuc = data_arr[:, 8]
    n_ann = data_arr[:, 9]

    # --- [NEW] Read kappa file (1st segment only) ---
    kappa_candidates = [
        "kappavstime_0.txt",
        os.path.join("/mnt/data", "kappavstime_0.txt"),
    ]
    have_kappa = False
    tk = kI = kII = None
    kappa_path_used = None

    for kp in kappa_candidates:
        if os.path.exists(kp):
            try:
                tk, kI, kII = read_kappa_file_first_segment(kp)
                have_kappa = True
                kappa_path_used = kp
                break
            except Exception as e:
                print(f"[WARN] Found '{kp}' but failed to read: {e}")

    if have_kappa:
        print(f"Loaded kappa (1st segment) from: {kappa_path_used}")
    else:
        print("[WARN] No kappa file found/read. Skipping kappa plot in Fig1 and Fig5.")

    # --------------------------
    # 图 1: 序参数（左轴） + κ(t)（右轴）
    # --------------------------
    fig1, ax1 = plt.subplots(figsize=(8, 5))

    l1 = ax1.plot(t, phi_bar, label=r'Local Magnitude $\langle|\phi|\rangle$', color='c')[0]
    l2 = ax1.plot(t, psi, label=r'Global Order $|\langle\phi\rangle|$', color='k', ls='--')[0]
    ax1.set_title("Symmetry Breaking & Order Parameters (with Kappa)")
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Order Parameter")

    if have_kappa:
        ax1b = ax1.twinx()
        tk_plot = tk / 100.0  # IMPORTANT: same convention as your uploaded script
        k1 = ax1b.plot(tk_plot, kI, lw=2, label=r"$\kappa_I$ (segment 1)")[0]
        k2 = ax1b.plot(tk_plot, kII, lw=2, ls="--", label=r"$\kappa_{II}$ (segment 1)")[0]
        ax1b.set_ylabel(r"$\kappa$")

        handles = [l1, l2, k1, k2]
        labels = [h.get_label() for h in handles]
        ax1.legend(handles, labels, loc="best")
    else:
        ax1.legend(loc="best")

    fig1.savefig("fig1_order_parameters.png", dpi=300, bbox_inches="tight")

    # --------------------------
    # 图 2: 缺陷密度
    # --------------------------
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    ax2.plot(t, n_perv, color='purple', lw=2)
    ax2.set_title("Defect Density Decay")
    ax2.set_ylabel("Number of Perversions")
    ax2.set_xlabel("Time")
    fig2.savefig("fig2_defect_density.png", dpi=300, bbox_inches="tight")

    # --------------------------
    # 图 3: 动力学与拓扑事件 (双栏图)
    # --------------------------
    fig3, (ax3_top, ax3_bot) = plt.subplots(
        2, 1, figsize=(10, 8), sharex=True,
        gridspec_kw={'height_ratios': [2, 1]}
    )

    color_w = 'tab:blue'
    ax3_top.set_ylabel('Wall Width (beads)', color=color_w, fontweight='bold')
    ax3_top.plot(t, width, color=color_w, lw=2, label='Width (Stable)')
    ax3_top.tick_params(axis='y', labelcolor=color_w)

    ax3_top_r = ax3_top.twinx()
    color_d = 'tab:red'
    ax3_top_r.set_ylabel(r'Diffusion Coeff $D$', color=color_d, fontweight='bold')
    ax3_top_r.plot(t, diff, color=color_d, ls='--', lw=1.5, label='Diffusion (Stable)')
    ax3_top_r.tick_params(axis='y', labelcolor=color_d)

    ax3_top.set_title("Decoupled Kinetics: Stable Motion vs. Topological Events")
    ax3_top.grid(True, alpha=0.3)

    bar_w = frame_dt * 0.8
    ax3_bot.bar(t, n_nuc, width=bar_w, label='Nucleation (+)', color='green', alpha=0.6)
    ax3_bot.bar(t, -n_ann, width=bar_w, label='Annihilation (-)', color='red', alpha=0.6)
    ax3_bot.axhline(0, color='k', lw=0.5)
    ax3_bot.set_ylabel("Events / Frame")
    ax3_bot.set_xlabel("Time")
    ax3_bot.legend(loc='upper right')
    ax3_bot.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    fig3.savefig("fig3_kinetics_events.png", dpi=300, bbox_inches="tight")

    # --------------------------
    # 图 4: Kymograph
    # --------------------------
    try:
        phi_mat = np.array(kymograph_matrix)
        if phi_mat.size > 0:
            fig4, ax4 = plt.subplots(figsize=(10, 6))
            limit = np.percentile(np.abs(phi_mat), 99)
            im = ax4.imshow(
                phi_mat, aspect='auto', cmap='bwr', vmin=-limit, vmax=limit,
                extent=[0, phi_mat.shape[1], t[-1], t[0]]
            )
            plt.colorbar(im, label=r"Chirality $\phi$")
            ax4.set_title("Spatiotemporal Evolution")
            ax4.set_ylabel("Time")
            ax4.set_xlabel("Genomic Position")
            fig4.savefig("fig4_kymograph.png", dpi=300, bbox_inches="tight")
    except Exception as e:
        print(f"Skipping kymograph: {e}")

    # --------------------------
    # [NEW] 图 5: 局部手性序参数 vs κ_II
    # --------------------------
    if have_kappa:
        tk_plot = tk / 100.0  # κ文件时间 -> 与你原脚本一致

        # 只保留 κ 时间与分析时间 t 的重叠区间，避免外插造成误导
        t_min = max(np.min(t), np.min(tk_plot))
        t_max = min(np.max(t), np.max(tk_plot))
        mask = (t >= t_min) & (t <= t_max)

        if np.sum(mask) >= 3:
            # κ_II(t) 插值到分析帧时间 t
            kII_at_t = np.interp(t[mask], tk_plot, kII)

            fig5, ax5 = plt.subplots(figsize=(7.5, 5))

            # 你说“画出局部手性随 kappa_II 的变化”
            # 这里默认画 phi_bar=<|phi|> vs kappa_II；同时也可把 psi 画上（更全面）
            ax5.plot(kII_at_t, phi_bar[mask], marker='o', ms=4, lw=1.5,
                     label=r'$\langle|\phi|\rangle$ vs $\kappa_{II}$')
            ax5.plot(kII_at_t, psi[mask], marker='s', ms=4, lw=1.5, ls='--',
                     label=r'$|\langle\phi\rangle|$ vs $\kappa_{II}$')

            ax5.set_xlabel(r'$\kappa_{II}$ (segment 1)')
            ax5.set_ylabel(r'Chirality Order Parameters')
            ax5.set_title(r'Local Chirality vs Motor Activity $\kappa_{II}$')
            ax5.grid(True, alpha=0.3)
            ax5.legend(loc='best')

            fig5.savefig("fig5_phi_vs_kappaII.png", dpi=300, bbox_inches="tight")
            print("Saved Fig5: fig5_phi_vs_kappaII.png")
        else:
            print("[WARN] Overlap between t and tk/100 is too small; skipping Fig5.")
    else:
        print("[WARN] No kappa -> skipping Fig5.")

    plt.show()
    print("All plots generated with separated event tracking (+ kappa in Fig1, phi vs kII in Fig5).")
