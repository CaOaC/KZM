#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import re
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

# ============================================================
# 用户设置
# ============================================================
ROOT_DIR = r""   # 例如 r"G:\chromosome\domainwall\dumps\perversion_boundary_msd_out"
USE_DIALOG = True
FIG_DPI = 300

SMOOTH_POINTS = 100
SPLINE_ORDER = 4   # 三次样条；点数不足时会自动降低


def pick_folder():
    import tkinter as tk
    from tkinter import filedialog
    root = tk.Tk()
    root.withdraw()
    folder = filedialog.askdirectory(title="选择 perversion_boundary_msd_out 文件夹")
    root.destroy()
    return folder


def parse_dir(name):
    m = re.search(
        r"tau([\d.eE+-]+)_v([\d.eE+-]+)_kI([\d.eE+-]+)_kII([\d.eE+-]+)",
        name,
    )
    if not m:
        return None
    return {
        "tau": float(m.group(1)),
        "v": float(m.group(2)),
        "kI": float(m.group(3)),
        "kII": float(m.group(4)),
    }


def read_fit(path):
    d = {}
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            sp = s.split()
            if len(sp) < 2:
                continue
            try:
                d[sp[0]] = float(sp[1])
            except ValueError:
                pass
    return d


def smooth_curve(x, y, n_points=300, spline_order=3):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if len(x) == 0:
        return x, y

    order = np.argsort(x)
    x = x[order]
    y = y[order]

    # remove duplicate x
    xu, idx = np.unique(x, return_index=True)
    yu = y[idx]

    if len(xu) < 3:
        return xu, yu

    k = min(spline_order, len(xu) - 1)

    try:
        xs = np.linspace(xu.min(), xu.max(), n_points)
        spl = make_interp_spline(xu, yu, k=k)
        ys = spl(xs)
        return xs, ys
    except Exception:
        return xu, yu


def main():
    root_str = ROOT_DIR.strip()
    if not root_str and USE_DIALOG:
        root_str = pick_folder()

    if not root_str:
        raise SystemExit("请设置 ROOT_DIR 或使用弹窗选择文件夹。")

    root = Path(root_str).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"ROOT_DIR 不存在: {root}")

    rows = []

    for fit_file in root.rglob("ensemble_boundary_msd_fit.txt"):
        params = parse_dir(fit_file.parent.name)
        if params is None:
            continue

        fit = read_fit(fit_file)

        if int(fit.get("fit_success", 0)) != 1:
            continue

        alpha = fit.get("alpha", np.nan)
        alpha_se = fit.get("alpha_se", np.nan)

        # A 就是 MSD = A * t^alpha 中的 prefactor k
        k_prefactor = fit.get("A", np.nan)

        rows.append({
            **params,
            "alpha": alpha,
            "alpha_se": alpha_se,
            "k_prefactor": k_prefactor,
            "R2": fit.get("R2", np.nan),
        })

    if len(rows) == 0:
        raise RuntimeError("没有找到有效的 ensemble_boundary_msd_fit.txt")

    rows = sorted(rows, key=lambda r: (r["tau"], r["v"], r["kII"], r["kI"]))

    # 保存 summary
    summary_path = root / "alpha_and_k_prefactor_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("# tau v kI kII alpha alpha_se k_prefactor R2\n")
        for r in rows:
            f.write(
                f"{r['tau']} {r['v']} {r['kI']} {r['kII']} "
                f"{r['alpha']} {r['alpha_se']} {r['k_prefactor']} {r['R2']}\n"
            )

    groups = {}
    for r in rows:
        key = (r["tau"], r["v"])
        groups.setdefault(key, []).append(r)

    # ============================================================
    # 图1：alpha vs kII，原始点 + 平滑曲线
    # ============================================================
    fig, ax = plt.subplots(figsize=(6.4, 4.8))

    for (tau, v), sub in groups.items():
        sub = sorted(sub, key=lambda r: r["kII"])

        kII = np.array([r["kII"] for r in sub], dtype=float)
        alpha = np.array([r["alpha"] for r in sub], dtype=float)
        alpha_se = np.array([r["alpha_se"] for r in sub], dtype=float)
        alpha_se = np.where(np.isfinite(alpha_se), alpha_se, 0.0)

        xs, ys = smooth_curve(kII, alpha, SMOOTH_POINTS, SPLINE_ORDER)

        ax.errorbar(
            kII,
            alpha,
            yerr=alpha_se,
            fmt="o",
            ms=5,
            capsize=3,
            label=rf"data $\tau={tau:g}, v={v:g}$",
        )

        ax.plot(
            xs,
            ys,
            "-",
            lw=2,
            label=rf"smooth $\tau={tau:g}, v={v:g}$",
        )

    ax.axhline(1.0, ls="--", alpha=0.5)
    ax.axhline(0.5, ls=":", alpha=0.7)

    ax.set_xlabel(r"$\kappa_{II}$")
    ax.set_ylabel(r"$\alpha$")
    ax.set_title(r"MSD exponent $\alpha$ vs $\kappa_{II}$")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.3)
    ax.tick_params(direction="in", top=True, right=True)

    fig.tight_layout()
    fig.savefig(root / "alpha_vs_kII_smooth.png", dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)

    # ============================================================
    # 图2：k prefactor vs kII，原始点 + 平滑曲线
    # ============================================================
    fig, ax = plt.subplots(figsize=(6.4, 4.8))

    for (tau, v), sub in groups.items():
        sub = sorted(sub, key=lambda r: r["kII"])

        kII = np.array([r["kII"] for r in sub], dtype=float)
        k_pref = np.array([r["k_prefactor"] for r in sub], dtype=float)

        xs, ys = smooth_curve(kII, k_pref, SMOOTH_POINTS, SPLINE_ORDER)

        ax.plot(
            kII,
            k_pref,
            "s",
            ms=5,
            label=rf"data $\tau={tau:g}, v={v:g}$",
        )

        ax.plot(
            xs,
            ys,
            "-",
            lw=2,
            label=rf"smooth $\tau={tau:g}, v={v:g}$",
        )

    ax.set_xlabel(r"$\kappa_{II}$")
    ax.set_ylabel(r"$k$ in MSD $= k t^\alpha$")
    ax.set_title(r"MSD prefactor $k$ vs $\kappa_{II}$")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.3)
    ax.tick_params(direction="in", top=True, right=True)

    fig.tight_layout()
    fig.savefig(root / "k_prefactor_vs_kII_smooth.png", dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)

    # ============================================================
    # 图3：alpha 和 k 放在一张图，双 y 轴，原始点 + 平滑曲线
    # ============================================================
    fig, ax1 = plt.subplots(figsize=(7.0, 5.0))
    ax2 = ax1.twinx()

    for (tau, v), sub in groups.items():
        sub = sorted(sub, key=lambda r: r["kII"])

        kII = np.array([r["kII"] for r in sub], dtype=float)
        alpha = np.array([r["alpha"] for r in sub], dtype=float)
        k_pref = np.array([r["k_prefactor"] for r in sub], dtype=float)

        xs_a, ys_a = smooth_curve(kII, alpha, SMOOTH_POINTS, SPLINE_ORDER)
        xs_k, ys_k = smooth_curve(kII, k_pref, SMOOTH_POINTS, SPLINE_ORDER)

        ax1.plot(
            kII,
            alpha,
            "o",
            ms=5,
            label=rf"$\alpha$ data, $\tau={tau:g}, v={v:g}$",
        )

        ax1.plot(
            xs_a,
            ys_a,
            "-",
            lw=2,
            label=rf"$\alpha$ smooth, $\tau={tau:g}, v={v:g}$",
        )

        ax2.plot(
            kII,
            k_pref,
            "s",
            ms=5,
            label=rf"$k$ data, $\tau={tau:g}, v={v:g}$",
        )

        ax2.plot(
            xs_k,
            ys_k,
            "--",
            lw=2,
            label=rf"$k$ smooth, $\tau={tau:g}, v={v:g}$",
        )

    ax1.axhline(1.0, ls="--", alpha=0.4)
    ax1.axhline(0.5, ls=":", alpha=0.6)

    ax1.set_xlabel(r"$\kappa_{II}$")
    ax1.set_ylabel(r"$\alpha$")
    ax2.set_ylabel(r"$k$ in MSD $= k t^\alpha$")

    ax1.set_title(r"MSD exponent and prefactor vs $\kappa_{II}$")
    ax1.grid(alpha=0.3)

    ax1.tick_params(direction="in", top=True)
    ax2.tick_params(direction="in", top=True)

    lines = ax1.get_lines() + ax2.get_lines()
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, frameon=False, fontsize=7, loc="best")

    fig.tight_layout()
    fig.savefig(root / "alpha_and_k_prefactor_vs_kII_smooth.png", dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)

    print("[DONE]")
    print("Saved:")
    print(summary_path)
    print(root / "alpha_vs_kII_smooth.png")
    print(root / "k_prefactor_vs_kII_smooth.png")
    print(root / "alpha_and_k_prefactor_vs_kII_smooth.png")


if __name__ == "__main__":
    main()