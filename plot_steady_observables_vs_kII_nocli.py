#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot steady-state ensemble-averaged observables versus kII.

适用于前面批处理脚本输出的目录结构，例如：

analysis_out/
  tau1.0000_v0.0000_kI0.0045_kII0.0009/
    ensemble_mean.txt
    ensemble_std.txt
  tau1.0000_v0.0000_kI0.0050_kII0.0010/
    ensemble_mean.txt
    ensemble_std.txt

本脚本会：
1. 递归寻找 ensemble_mean.txt；
2. 从父目录名中解析 tau, v, kI, kII；
3. 对每个参数组取稳态窗口平均；
4. 画出序参量/动力学量随 kII 的变化；
5. 图片和汇总数据保存到 ROOT_DIR 下。
"""

from pathlib import Path
import re
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 用户配置区
# ============================================================
ROOT_DIR = r"H:\phasediagram\totalTraj\analysis_out"   # 例如 r"G:\chromosome\domainwall\analysis_out"
USE_FOLDER_DIALOG = True

# 稳态窗口设置：取最后 STEADY_FRACTION 的数据作为稳态
STEADY_FRACTION = 0.2

# 如果你希望只取最后固定帧数，可设置为整数，例如 20；否则设为 None
STEADY_LAST_N = None

# 可选筛选：只画某个 tau 或 v；不筛选则设为 None
FILTER_TAU = None     # 例如 1.0
FILTER_V = None       # 例如 0.0

# 图片参数
FIG_DPI = 300
SHOW_FIGURES = False


# ============================================================
# 数据列定义
# ============================================================
COLUMNS = {
    "time": 0,
    "frame_idx": 1,
    "phi_bar": 2,
    "psi": 3,
    "n_perv": 4,
    "avg_dom_size": 5,
    "avg_width": 6,
    "diff_coeff": 7,
    "n_nucleation": 8,
    "n_annihilation": 9,
}

PLOT_ITEMS = [
    ("phi_bar", r"$\langle|\phi|\rangle_{\rm ss}$", "steady_phi_bar_vs_kII.png"),
    ("psi", r"$|\langle\phi\rangle|_{\rm ss}$", "steady_psi_vs_kII.png"),
    ("n_perv", r"$N_{\rm perv}^{\rm ss}$", "steady_n_perv_vs_kII.png"),
    ("avg_dom_size", r"$L_{\rm domain}^{\rm ss}$", "steady_domain_size_vs_kII.png"),
    ("avg_width", r"$w_{\rm wall}^{\rm ss}$", "steady_wall_width_vs_kII.png"),
    ("diff_coeff", r"$D_{\rm wall}^{\rm ss}$", "steady_wall_diffusion_vs_kII.png"),
    ("n_nucleation", r"$N_{\rm nuc}^{\rm ss}$", "steady_nucleation_vs_kII.png"),
    ("n_annihilation", r"$N_{\rm ann}^{\rm ss}$", "steady_annihilation_vs_kII.png"),
]


# ============================================================
# 工具函数
# ============================================================
def pick_folder_by_dialog():
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        folder = filedialog.askdirectory(title="选择 analysis_out 文件夹")
        root.destroy()
        return folder
    except Exception as e:
        print(f"[WARN] 无法打开文件夹选择窗口: {e}")
        return ""


def parse_param_dir_name(name):
    """
    Parse directory name:
      tau1.0000_v0.0000_kI0.0045_kII0.0009
    Also supports more digits:
      tau1.00000_v0.00000_kI0.00450_kII0.00090
    """
    pat = (
        r"tau(?P<tau>[-+0-9.eE]+)_"
        r"v(?P<v>[-+0-9.eE]+)_"
        r"kI(?P<kI>[-+0-9.eE]+)_"
        r"kII(?P<kII>[-+0-9.eE]+)"
    )
    m = re.search(pat, name)
    if not m:
        return None
    return {
        "tau": float(m.group("tau")),
        "v": float(m.group("v")),
        "kI": float(m.group("kI")),
        "kII": float(m.group("kII")),
    }


def load_txt(path):
    arr = np.loadtxt(path, comments="#")
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr


def steady_slice(arr):
    n = arr.shape[0]
    if n == 0:
        return arr

    if STEADY_LAST_N is not None:
        m = min(int(STEADY_LAST_N), n)
        return arr[-m:]

    frac = float(STEADY_FRACTION)
    if frac <= 0 or frac > 1:
        raise ValueError("STEADY_FRACTION must be in (0, 1].")

    m = max(1, int(round(frac * n)))
    return arr[-m:]


def nanmean_sem(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan, np.nan
    mean = float(np.mean(x))
    if x.size == 1:
        return mean, np.nan
    sem = float(np.std(x, ddof=1) / np.sqrt(x.size))
    return mean, sem


def set_clean_axis(ax):
    ax.tick_params(direction="in", top=True, right=True)
    ax.grid(alpha=0.25)


def savefig(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
    if SHOW_FIGURES:
        plt.show()
    else:
        plt.close(fig)


def collect_steady_data(root):
    rows = []

    mean_files = sorted(root.rglob("ensemble_mean.txt"))
    print(f"[INFO] 找到 {len(mean_files)} 个 ensemble_mean.txt")

    for mean_path in mean_files:
        group_dir = mean_path.parent
        params = parse_param_dir_name(group_dir.name)
        if params is None:
            print(f"[WARN] 跳过无法解析参数的目录: {group_dir.name}")
            continue

        if FILTER_TAU is not None and not np.isclose(params["tau"], float(FILTER_TAU)):
            continue
        if FILTER_V is not None and not np.isclose(params["v"], float(FILTER_V)):
            continue

        try:
            mean_arr = load_txt(mean_path)
        except Exception as e:
            print(f"[WARN] 读取失败 {mean_path}: {e}")
            continue

        ss = steady_slice(mean_arr)

        row = {
            "tau": params["tau"],
            "v": params["v"],
            "kI": params["kI"],
            "kII": params["kII"],
            "n_steady_points": ss.shape[0],
            "source_dir": str(group_dir),
        }

        for name, idx in COLUMNS.items():
            if idx >= ss.shape[1]:
                row[name] = np.nan
                row[name + "_sem_time"] = np.nan
                continue
            val, sem = nanmean_sem(ss[:, idx])
            row[name] = val
            row[name + "_sem_time"] = sem

        rows.append(row)

    rows = sorted(rows, key=lambda r: (r["tau"], r["v"], r["kII"], r["kI"]))
    return rows


def save_summary_table(rows, out_path):
    names = [
        "tau", "v", "kI", "kII", "n_steady_points",
        "phi_bar", "phi_bar_sem_time",
        "psi", "psi_sem_time",
        "n_perv", "n_perv_sem_time",
        "avg_dom_size", "avg_dom_size_sem_time",
        "avg_width", "avg_width_sem_time",
        "diff_coeff", "diff_coeff_sem_time",
        "n_nucleation", "n_nucleation_sem_time",
        "n_annihilation", "n_annihilation_sem_time",
    ]

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("# steady-state observables obtained from ensemble_mean.txt\n")
        if STEADY_LAST_N is not None:
            f.write(f"# steady window: last {STEADY_LAST_N} points\n")
        else:
            f.write(f"# steady window: last fraction {STEADY_FRACTION}\n")
        f.write("# " + " ".join(names) + "\n")
        for r in rows:
            vals = [r.get(name, np.nan) for name in names]
            f.write(" ".join(str(v) for v in vals) + "\n")


def group_by_tau_v(rows):
    groups = {}
    for r in rows:
        key = (r["tau"], r["v"])
        groups.setdefault(key, []).append(r)
    return groups


def plot_item(rows, item_name, ylabel, filename, root):
    groups = group_by_tau_v(rows)

    fig, ax = plt.subplots(figsize=(7.2, 4.8))

    for (tau, v), subrows in sorted(groups.items()):
        subrows = sorted(subrows, key=lambda r: r["kII"])
        x = np.array([r["kII"] for r in subrows], dtype=float)
        y = np.array([r[item_name] for r in subrows], dtype=float)
        yerr = np.array([r[item_name + "_sem_time"] for r in subrows], dtype=float)

        label = rf"$\tau={tau:g},\ v={v:g}$"

        if np.all(~np.isfinite(yerr)):
            ax.plot(x, y, marker="o", lw=2, label=label)
        else:
            yerr_plot = np.where(np.isfinite(yerr), yerr, 0.0)
            ax.errorbar(x, y, yerr=yerr_plot, marker="o", lw=2, capsize=3, label=label)

    ax.set_xlabel(r"$\kappa_{II}$")
    ax.set_ylabel(ylabel)
    ax.set_title(ylabel + r" vs $\kappa_{II}$")
    ax.legend(frameon=False)
    set_clean_axis(ax)
    savefig(fig, root / filename)


def plot_combined_core(rows, root):
    groups = group_by_tau_v(rows)

    fig, axes = plt.subplots(2, 2, figsize=(10, 7.5), sharex=True)
    axes = axes.ravel()

    items = [
        ("phi_bar", r"$\langle|\phi|\rangle_{\rm ss}$"),
        ("psi", r"$|\langle\phi\rangle|_{\rm ss}$"),
        ("n_perv", r"$N_{\rm perv}^{\rm ss}$"),
        ("avg_dom_size", r"$L_{\rm domain}^{\rm ss}$"),
    ]

    for ax, (item, ylabel) in zip(axes, items):
        for (tau, v), subrows in sorted(groups.items()):
            subrows = sorted(subrows, key=lambda r: r["kII"])
            x = np.array([r["kII"] for r in subrows], dtype=float)
            y = np.array([r[item] for r in subrows], dtype=float)
            yerr = np.array([r[item + "_sem_time"] for r in subrows], dtype=float)
            label = rf"$\tau={tau:g},\ v={v:g}$"

            if np.all(~np.isfinite(yerr)):
                ax.plot(x, y, marker="o", lw=2, label=label)
            else:
                yerr_plot = np.where(np.isfinite(yerr), yerr, 0.0)
                ax.errorbar(x, y, yerr=yerr_plot, marker="o", lw=2, capsize=3, label=label)

        ax.set_ylabel(ylabel)
        set_clean_axis(ax)

    axes[2].set_xlabel(r"$\kappa_{II}$")
    axes[3].set_xlabel(r"$\kappa_{II}$")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle(r"Steady-state observables vs $\kappa_{II}$")
    savefig(fig, root / "steady_core_observables_vs_kII.png")


def plot_combined_kinetics(rows, root):
    groups = group_by_tau_v(rows)

    fig, axes = plt.subplots(2, 2, figsize=(10, 7.5), sharex=True)
    axes = axes.ravel()

    items = [
        ("avg_width", r"$w_{\rm wall}^{\rm ss}$"),
        ("diff_coeff", r"$D_{\rm wall}^{\rm ss}$"),
        ("n_nucleation", r"$N_{\rm nuc}^{\rm ss}$"),
        ("n_annihilation", r"$N_{\rm ann}^{\rm ss}$"),
    ]

    for ax, (item, ylabel) in zip(axes, items):
        for (tau, v), subrows in sorted(groups.items()):
            subrows = sorted(subrows, key=lambda r: r["kII"])
            x = np.array([r["kII"] for r in subrows], dtype=float)
            y = np.array([r[item] for r in subrows], dtype=float)
            yerr = np.array([r[item + "_sem_time"] for r in subrows], dtype=float)
            label = rf"$\tau={tau:g},\ v={v:g}$"

            if np.all(~np.isfinite(yerr)):
                ax.plot(x, y, marker="o", lw=2, label=label)
            else:
                yerr_plot = np.where(np.isfinite(yerr), yerr, 0.0)
                ax.errorbar(x, y, yerr=yerr_plot, marker="o", lw=2, capsize=3, label=label)

        ax.set_ylabel(ylabel)
        set_clean_axis(ax)

    axes[2].set_xlabel(r"$\kappa_{II}$")
    axes[3].set_xlabel(r"$\kappa_{II}$")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle(r"Steady-state wall kinetics vs $\kappa_{II}$")
    savefig(fig, root / "steady_wall_kinetics_vs_kII.png")


# ============================================================
# 主程序
# ============================================================
if __name__ == "__main__":
    root_str = ROOT_DIR.strip()
    if (not root_str) and USE_FOLDER_DIALOG:
        root_str = pick_folder_by_dialog()

    if not root_str:
        raise SystemExit("请在脚本顶部设置 ROOT_DIR，或者使用弹窗选择 analysis_out 文件夹。")

    root = Path(root_str).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"ROOT_DIR 不存在: {root}")

    print(f"[INFO] ROOT_DIR = {root}")

    rows = collect_steady_data(root)
    if not rows:
        raise SystemExit("没有收集到可用的稳态数据。")

    summary_path = root / "steady_observables_vs_kII.txt"
    save_summary_table(rows, summary_path)
    print(f"[OK] 稳态汇总数据已保存: {summary_path}")

    for item, ylabel, filename in PLOT_ITEMS:
        plot_item(rows, item, ylabel, filename, root)

    plot_combined_core(rows, root)
    plot_combined_kinetics(rows, root)

    print("[DONE] 所有 steady-state vs kII 图片已生成。")
