#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Post-process boundary-region MSD scaling results.

Reads outputs from:
    perversion_boundary_msd_gpu_nocli.py

Expected folder structure:
    perversion_boundary_msd_out/
      tau..._v..._kI..._kII.../
        ensemble_boundary_msd_fit.txt
        ensemble_boundary_msd.txt

Outputs under ROOT_DIR:
    alpha_summary.txt
    alpha_vs_kII.png
    alpha_vs_kI.png
    alpha_vs_kII_over_kI.png
    alpha_kI_kII_heatmap.png
"""

from pathlib import Path
import re
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# User configuration
# ============================================================
ROOT_DIR = r""   # e.g. r"G:\chromosome\domainwall\dumps\perversion_boundary_msd_out"
USE_FOLDER_DIALOG = True

FILTER_TAU = None
FILTER_V = None

FIG_DPI = 300
SHOW_FIGURES = False
GROUP_BY_TAU_V = True


# ============================================================
# Utilities
# ============================================================
def pick_folder_by_dialog():
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        folder = filedialog.askdirectory(title="选择 perversion_boundary_msd_out 文件夹")
        root.destroy()
        return folder
    except Exception as e:
        print(f"[WARN] Cannot open folder dialog: {e}")
        return ""


def parse_param_dir_name(name):
    pat = (
        r"tau(?P<tau>[-+0-9.eE]+)_"
        r"v(?P<v>[-+0-9.eE]+)_"
        r"kI(?P<kI>[-+0-9.eE]+)_"
        r"kII(?P<kII>[-+0-9.eE]+)"
    )
    m = re.search(pat, name)
    if m is None:
        return None

    return {
        "tau": float(m.group("tau")),
        "v": float(m.group("v")),
        "kI": float(m.group("kI")),
        "kII": float(m.group("kII")),
    }


def read_fit_file(path):
    d = {
        "fit_success": 0,
        "alpha": np.nan,
        "alpha_se": np.nan,
        "A": np.nan,
        "logA": np.nan,
        "logA_se": np.nan,
        "R2": np.nan,
        "n_fit": np.nan,
        "fit_dt_min": np.nan,
        "fit_dt_max": np.nan,
    }

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if (not s) or s.startswith("#"):
                continue
            parts = s.split()
            if len(parts) < 2:
                continue

            key = parts[0]
            val = parts[1]

            if key == "fit_success":
                d[key] = int(float(val))
            elif key in d:
                try:
                    d[key] = float(val)
                except ValueError:
                    d[key] = np.nan

    return d


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


def collect_alpha_data(root):
    rows = []

    fit_files = sorted(root.rglob("ensemble_boundary_msd_fit.txt"))
    print(f"[INFO] Found {len(fit_files)} ensemble fit files.")

    for fit_path in fit_files:
        group_dir = fit_path.parent
        params = parse_param_dir_name(group_dir.name)

        if params is None:
            print(f"[WARN] Cannot parse parameter folder: {group_dir.name}")
            continue

        if FILTER_TAU is not None and not np.isclose(params["tau"], float(FILTER_TAU)):
            continue
        if FILTER_V is not None and not np.isclose(params["v"], float(FILTER_V)):
            continue

        fit = read_fit_file(fit_path)

        row = {
            "tau": params["tau"],
            "v": params["v"],
            "kI": params["kI"],
            "kII": params["kII"],
            "fit_success": fit["fit_success"],
            "alpha": fit["alpha"],
            "alpha_se": fit["alpha_se"],
            "A": fit["A"],
            "R2": fit["R2"],
            "n_fit": fit["n_fit"],
            "fit_dt_min": fit["fit_dt_min"],
            "fit_dt_max": fit["fit_dt_max"],
            "source_dir": str(group_dir),
        }

        rows.append(row)

    rows = sorted(rows, key=lambda r: (r["tau"], r["v"], r["kII"], r["kI"]))
    return rows


def save_summary(rows, out_path):
    names = [
        "tau", "v", "kI", "kII",
        "fit_success", "alpha", "alpha_se", "A", "R2",
        "n_fit", "fit_dt_min", "fit_dt_max",
    ]

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("# Alpha summary from ensemble_boundary_msd_fit.txt\n")
        f.write("# Model: MSD(Delta t) = A * Delta t^alpha\n")
        f.write("# " + " ".join(names) + "\n")
        for r in rows:
            vals = [r.get(name, np.nan) for name in names]
            f.write(" ".join(str(v) for v in vals) + "\n")


def valid_rows(rows):
    return [
        r for r in rows
        if r["fit_success"] == 1
        and np.isfinite(r["alpha"])
        and np.isfinite(r["kII"])
        and np.isfinite(r["kI"])
    ]


def group_rows(rows):
    if not GROUP_BY_TAU_V:
        return {("all", "all"): rows}

    groups = {}
    for r in rows:
        key = (r["tau"], r["v"])
        groups.setdefault(key, []).append(r)
    return groups


def plot_alpha_vs_x(rows, xname, xlabel, out_path, use_logx=False):
    rows = valid_rows(rows)

    if not rows:
        print(f"[WARN] No valid rows for {out_path.name}")
        return

    fig, ax = plt.subplots(figsize=(7.0, 4.8))

    groups = group_rows(rows)

    for key, sub in sorted(groups.items(), key=lambda kv: str(kv[0])):
        sub = sorted(sub, key=lambda r: r[xname])

        x = np.array([r[xname] for r in sub], dtype=float)
        y = np.array([r["alpha"] for r in sub], dtype=float)
        yerr = np.array([r["alpha_se"] for r in sub], dtype=float)

        if GROUP_BY_TAU_V:
            tau, v = key
            label = rf"$\tau={tau:g},\ v={v:g}$"
        else:
            label = "all"

        yerr_plot = np.where(np.isfinite(yerr), yerr, 0.0)
        ax.errorbar(x, y, yerr=yerr_plot, marker="o", lw=2, capsize=3, label=label)

    if use_logx:
        ax.set_xscale("log")

    ax.axhline(1.0, ls="--", lw=1, alpha=0.6, label=r"$\alpha=1$")
    ax.axhline(0.5, ls=":", lw=1.5, alpha=0.8, label=r"$\alpha=1/2$")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"MSD exponent $\alpha$")
    ax.set_title(r"Boundary-region diffusion exponent $\alpha$")
    ax.legend(frameon=False)
    set_clean_axis(ax)
    savefig(fig, out_path)


def plot_alpha_vs_ratio(rows, out_path):
    rows2 = []
    for r in valid_rows(rows):
        if r["kI"] != 0 and np.isfinite(r["kII"] / r["kI"]):
            rr = dict(r)
            rr["kII_over_kI"] = rr["kII"] / rr["kI"]
            rows2.append(rr)

    if not rows2:
        print(f"[WARN] No valid rows for {out_path.name}")
        return

    plot_alpha_vs_x(
        rows2,
        xname="kII_over_kI",
        xlabel=r"$\kappa_{II}/\kappa_I$",
        out_path=out_path,
        use_logx=False,
    )


def plot_heatmap(rows, out_path):
    rows = valid_rows(rows)
    if not rows:
        print("[WARN] No valid rows for heatmap.")
        return

    groups = group_rows(rows)

    if len(groups) > 1:
        for key, sub in groups.items():
            tau, v = key
            suffix = f"_tau{tau:.5f}_v{v:.5f}"
            plot_heatmap_single(sub, out_path.with_name(out_path.stem + suffix + out_path.suffix))
    else:
        plot_heatmap_single(rows, out_path)


def plot_heatmap_single(rows, out_path):
    rows = valid_rows(rows)

    kI_vals = sorted(set(float(r["kI"]) for r in rows))
    kII_vals = sorted(set(float(r["kII"]) for r in rows))

    if len(kI_vals) < 2 or len(kII_vals) < 2:
        print(f"[WARN] Not enough kI/kII grid points for heatmap: {out_path.name}")
        return

    Z = np.full((len(kII_vals), len(kI_vals)), np.nan, dtype=float)
    kI_index = {v: i for i, v in enumerate(kI_vals)}
    kII_index = {v: i for i, v in enumerate(kII_vals)}

    for r in rows:
        i = kII_index[float(r["kII"])]
        j = kI_index[float(r["kI"])]
        Z[i, j] = r["alpha"]

    fig, ax = plt.subplots(figsize=(7.0, 5.4))

    im = ax.imshow(
        Z,
        origin="lower",
        aspect="auto",
        extent=[min(kI_vals), max(kI_vals), min(kII_vals), max(kII_vals)],
    )

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(r"$\alpha$")

    ax.set_xlabel(r"$\kappa_I$")
    ax.set_ylabel(r"$\kappa_{II}$")
    ax.set_title(r"Boundary-region MSD exponent $\alpha$")

    set_clean_axis(ax)
    savefig(fig, out_path)


def print_quick_summary(rows):
    rows = valid_rows(rows)

    if not rows:
        print("[WARN] No successful alpha fits.")
        return

    alpha = np.array([r["alpha"] for r in rows], dtype=float)
    print("[SUMMARY]")
    print(f"  n_valid = {len(rows)}")
    print(f"  alpha min / mean / max = {np.nanmin(alpha):.4f} / {np.nanmean(alpha):.4f} / {np.nanmax(alpha):.4f}")

    low = sorted(rows, key=lambda r: r["alpha"])[:5]
    print("  lowest alpha:")
    for r in low:
        print(f"    alpha={r['alpha']:.4f}, kI={r['kI']:.6g}, kII={r['kII']:.6g}, tau={r['tau']}, v={r['v']}")

    high = sorted(rows, key=lambda r: r["alpha"], reverse=True)[:5]
    print("  highest alpha:")
    for r in high:
        print(f"    alpha={r['alpha']:.4f}, kI={r['kI']:.6g}, kII={r['kII']:.6g}, tau={r['tau']}, v={r['v']}")


if __name__ == "__main__":
    root_str = ROOT_DIR.strip()

    if (not root_str) and USE_FOLDER_DIALOG:
        root_str = pick_folder_by_dialog()

    if not root_str:
        raise SystemExit("Please set ROOT_DIR or choose a folder from the dialog.")

    root = Path(root_str).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"ROOT_DIR does not exist: {root}")

    print(f"[INFO] ROOT_DIR = {root}")

    rows = collect_alpha_data(root)
    if not rows:
        raise SystemExit("No alpha data found.")

    summary_path = root / "alpha_summary.txt"
    save_summary(rows, summary_path)
    print(f"[OK] Saved summary: {summary_path}")

    print_quick_summary(rows)

    plot_alpha_vs_x(
        rows,
        xname="kII",
        xlabel=r"$\kappa_{II}$",
        out_path=root / "alpha_vs_kII.png",
        use_logx=False,
    )

    plot_alpha_vs_x(
        rows,
        xname="kI",
        xlabel=r"$\kappa_I$",
        out_path=root / "alpha_vs_kI.png",
        use_logx=False,
    )

    plot_alpha_vs_ratio(
        rows,
        out_path=root / "alpha_vs_kII_over_kI.png",
    )

    plot_heatmap(
        rows,
        out_path=root / "alpha_kI_kII_heatmap.png",
    )

    print("[DONE] Alpha plots generated.")
