#!/usr/bin/env python3
# align_two_chromatin.py
# Usage:
# python align_two_chromatin.py -i 2000_sis_xu_p2.txt -o aligned.txt --mol1 1 --mol2 2 --gap 2.0

import argparse
import numpy as np


def read_atom_file(path):
    rows = []
    comments = []
    with open(path, "r") as f:
        for line in f:
            s = line.strip()
            if not s:
                comments.append(line)
                continue
            if s.startswith("ITEM:"):
                comments.append(line)
                continue
            parts = s.split()
            if len(parts) < 6:
                comments.append(line)
                continue
            rows.append(parts)
    return rows, comments


def write_atom_file(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(" ".join(r) + "\n")


def principal_axis(xyz):
    x = xyz - xyz.mean(axis=0)
    cov = x.T @ x / len(x)
    vals, vecs = np.linalg.eigh(cov)
    u = vecs[:, np.argmax(vals)]
    u /= np.linalg.norm(u)

    # make direction consistent with chain index direction
    end_to_end = xyz[-1] - xyz[0]
    if np.dot(u, end_to_end) < 0:
        u = -u
    return u


def rotation_from_a_to_b(a, b):
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)

    v = np.cross(a, b)
    c = np.dot(a, b)

    if c > 1 - 1e-12:
        return np.eye(3)

    if c < -1 + 1e-12:
        # 180 degree rotation around any axis perpendicular to a
        tmp = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(tmp, a)) > 0.9:
            tmp = np.array([0.0, 1.0, 0.0])
        v = np.cross(a, tmp)
        v /= np.linalg.norm(v)
        K = np.array([
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0]
        ])
        return np.eye(3) + 2 * (K @ K)

    s = np.linalg.norm(v)
    K = np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])
    R = np.eye(3) + K + K @ K * ((1 - c) / s**2)
    return R


def perpendicular_radius(xyz, axis):
    c = xyz.mean(axis=0)
    rel = xyz - c
    para = np.outer(rel @ axis, axis)
    perp = rel - para
    return np.percentile(np.linalg.norm(perp, axis=1), 95)


def choose_side_direction(xyz, axis):
    c = xyz.mean(axis=0)
    rel = xyz - c
    cov = rel.T @ rel / len(rel)
    vals, vecs = np.linalg.eigh(cov)

    # choose second largest PCA direction, then project perpendicular to main axis
    order = np.argsort(vals)[::-1]
    n = vecs[:, order[1]]
    n = n - np.dot(n, axis) * axis

    if np.linalg.norm(n) < 1e-8:
        tmp = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(tmp, axis)) > 0.9:
            tmp = np.array([0.0, 1.0, 0.0])
        n = np.cross(axis, tmp)

    n /= np.linalg.norm(n)
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", default="./2000_sis_xu_p2.txt")
    ap.add_argument("-o", "--output", default="./aligned_chromosome.txt")
    ap.add_argument("--mol1", type=int, default=1)
    ap.add_argument("--mol2", type=int, default=2)
    ap.add_argument("--gap", type=float, default=2.0,
                    help="surface-to-surface gap between two compact chromosomes")
    ap.add_argument("--center-distance", type=float, default=None,
                    help="direct center-to-center distance; if given, overrides gap")
    ap.add_argument("--xcol", type=int, default=3)
    ap.add_argument("--ycol", type=int, default=4)
    ap.add_argument("--zcol", type=int, default=5)
    ap.add_argument("--molcol", type=int, default=1)
    ap.add_argument("--precision", type=int, default=4)
    args = ap.parse_args()

    rows, _ = read_atom_file(args.input)

    mols = np.array([int(float(r[args.molcol])) for r in rows])
    xyz = np.array([[float(r[args.xcol]), float(r[args.ycol]), float(r[args.zcol])] for r in rows])

    idx1 = np.where(mols == args.mol1)[0]
    idx2 = np.where(mols == args.mol2)[0]

    if len(idx1) == 0 or len(idx2) == 0:
        raise ValueError("Cannot find requested mol ids.")

    xyz1 = xyz[idx1]
    xyz2 = xyz[idx2]

    c1 = xyz1.mean(axis=0)
    c2 = xyz2.mean(axis=0)

    u1 = principal_axis(xyz1)
    u2 = principal_axis(xyz2)

    R = rotation_from_a_to_b(u2, u1)

    xyz2_centered = xyz2 - c2
    xyz2_rot = xyz2_centered @ R.T

    n = choose_side_direction(xyz1, u1)

    if args.center_distance is None:
        r1 = perpendicular_radius(xyz1, u1)
        r2 = perpendicular_radius(xyz2_rot + c1, u1)
        sep = r1 + r2 + args.gap
    else:
        sep = args.center_distance

    new_c2 = c1 + sep * n
    xyz2_new = xyz2_rot + new_c2

    xyz_out = xyz.copy()
    xyz_out[idx2] = xyz2_new

    fmt = f"{{:.{args.precision}f}}"
    for i, r in enumerate(rows):
        r[args.xcol] = fmt.format(xyz_out[i, 0])
        r[args.ycol] = fmt.format(xyz_out[i, 1])
        r[args.zcol] = fmt.format(xyz_out[i, 2])

    write_atom_file(args.output, rows)

    print("Done.")
    print(f"mol {args.mol1} fixed.")
    print(f"mol {args.mol2} rotated and translated.")
    print(f"principal axis mol1 = {u1}")
    print(f"principal axis mol2 after alignment = {u1}")
    print(f"center distance = {sep:.6f}")


if __name__ == "__main__":
    main()