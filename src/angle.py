import os
import numpy as np
import argparse
import time

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
DUMP_DIR = os.path.join(PROJECT_ROOT, "dump_files")
OTHER_DIR = os.path.join(PROJECT_ROOT, "other_files")


def unwrap_positions(x, y, z, ix, iy, iz, box):
    lx = box[0][1] - box[0][0]
    ly = box[1][1] - box[1][0]
    lz = box[2][1] - box[2][0]

    xu = x + ix * lx
    yu = y + iy * ly
    zu = z + iz * lz

    return np.column_stack([xu, yu, zu])


def principal_axis(coords):
    coords = coords - coords.mean(axis=0)

    cov = coords.T @ coords / len(coords)

    eigvals, eigvecs = np.linalg.eigh(cov)

    axis = eigvecs[:, np.argmax(eigvals)]
    axis = axis / np.linalg.norm(axis)

    return axis


def angle_between_axes(u, v):
    dot_abs = abs(np.dot(u, v))
    dot_abs = np.clip(dot_abs, -1.0, 1.0)

    angle_rad = np.arccos(dot_abs)
    angle_deg = np.degrees(angle_rad)

    return angle_deg, dot_abs


def read_one_frame(f):
    line = f.readline()

    if not line:
        return None

    if not line.startswith("ITEM: TIMESTEP"):
        raise ValueError("Dump format error: expected ITEM: TIMESTEP")

    timestep = float(f.readline().strip())

    box = None
    n_atoms = None
    columns = None

    while True:
        line = f.readline()
        if not line:
            raise ValueError("Dump format error: unexpected end of frame before ITEM: ATOMS")

        if line.startswith("ITEM: BOX BOUNDS"):
            box = []
            for _ in range(3):
                lo, hi = map(float, f.readline().split()[:2])
                box.append((lo, hi))
            continue

        if line.startswith("ITEM: NUMBER OF ATOMS"):
            n_atoms = int(float(f.readline().strip()))
            continue

        if line.startswith("ITEM: ATOMS"):
            columns = line.strip().split()[2:]
            break

        # ignore other ITEM lines or comments
        continue

    if box is None:
        raise ValueError("Dump format error: expected ITEM: BOX BOUNDS")
    if n_atoms is None:
        raise ValueError("Dump format error: expected ITEM: NUMBER OF ATOMS")

    col = {name: i for i, name in enumerate(columns)}

    if "id" not in col:
        raise ValueError("Missing required column: id")
    if "x" not in col or "y" not in col or "z" not in col:
        raise ValueError("Missing required column: x, y, or z")

    if "mol" in col:
        mol_col = col["mol"]
    elif "type" in col:
        mol_col = col["type"]
    else:
        raise ValueError("Missing required column: mol or type")

    has_image = all(k in col for k in ["ix", "iy", "iz"])

    data = []
    for _ in range(n_atoms):
        parts = f.readline().split()

        if len(parts) < len(columns):
            raise ValueError(
                f"Bad atom line at timestep {timestep}: "
                f"expected {len(columns)} columns, got {len(parts)}"
            )

        data.append(parts)

    data = np.array(data, dtype=object)

    mol = data[:, mol_col].astype(int)
    x = data[:, col["x"]].astype(float)
    y = data[:, col["y"]].astype(float)
    z = data[:, col["z"]].astype(float)

    if has_image:
        ix = data[:, col["ix"]].astype(int)
        iy = data[:, col["iy"]].astype(int)
        iz = data[:, col["iz"]].astype(int)

        coords = unwrap_positions(x, y, z, ix, iy, iz, box)
    else:
        coords = np.column_stack([x, y, z])

    return timestep, mol, coords


def analyze_dump(
    dump_file,
    output_file=os.path.join(OTHER_DIR, "chromosome_angle_vs_time.txt"),
    mol1=1,
    mol2=2,
    follow=False,
    sleep_time=1.0,
):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, "w") as out:
        out.write("# timestep angle_deg dot_abs\n")

        with open(dump_file, "r") as f:
            while True:
                pos = f.tell()

                try:
                    frame = read_one_frame(f)
                except Exception as e:
                    if follow:
                        f.seek(pos)
                        time.sleep(sleep_time)
                        continue
                    else:
                        raise e

                if frame is None:
                    if follow:
                        time.sleep(sleep_time)
                        continue
                    else:
                        break

                timestep, mol, coords = frame

                coords1 = coords[mol == mol1]
                coords2 = coords[mol == mol2]

                if len(coords1) < 3 or len(coords2) < 3:
                    print(
                        f"Skip timestep {timestep:.6f}: "
                        f"mol {mol1} has {len(coords1)} atoms, "
                        f"mol {mol2} has {len(coords2)} atoms."
                    )
                    continue

                axis1 = principal_axis(coords1)
                axis2 = principal_axis(coords2)

                angle_deg, dot_abs = angle_between_axes(axis1, axis2)

                out.write(f"{timestep:.6f} {angle_deg:.8f} {dot_abs:.8f}\n")
                out.flush()

                # print(
                #     f"timestep = {timestep:.6f}, "
                #     f"angle = {angle_deg:.6f} deg, "
                #     f"|dot| = {dot_abs:.8f}"
                # )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calculate the angle between two chromosomes from a LAMMPS-style dump file."
    )

    parser.add_argument(
        "-i",
        "--input",
        # default=os.path.join(DUMP_DIR, "2000_sis.dump"),
        default=r"F:\e\C++_Project\sister_chromotids\Output\trajectory_tau1.0000_v0.0000_kI0.0500_kII0.1000_ems1_part2.dump",
        help="Input dump file",
    )

    parser.add_argument(
        "-o",
        "--output",
        default=os.path.join(OTHER_DIR, "chromosome_angle_vs_time.txt"),
        help="Output text file",
    )

    parser.add_argument(
        "--mol1",
        type=int,
        default=1,
        help="Molecule ID of chromosome 1",
    )

    parser.add_argument(
        "--mol2",
        type=int,
        default=2,
        help="Molecule ID of chromosome 2",
    )

    parser.add_argument(
        "--follow",
        action="store_true",
        help="Follow a growing dump file in real time",
    )

    parser.add_argument(
        "--sleep",
        type=float,
        default=1.0,
        help="Sleep time when using --follow",
    )

    args = parser.parse_args()

    output_file = args.output
    if not os.path.dirname(output_file):
        output_file = os.path.join(OTHER_DIR, output_file)

    analyze_dump(
        dump_file=args.input,
        output_file=output_file,
        mol1=args.mol1,
        mol2=args.mol2,
        follow=args.follow,
        sleep_time=args.sleep,
    )

    import numpy as np
    import matplotlib.pyplot as plt

    data = np.loadtxt(output_file, comments="#")

    if data.ndim == 1:
        data = np.atleast_2d(data)

    if data.size == 0 or data.shape[1] < 2:
        raise RuntimeError(f"No valid data found in {output_file}")

    t = data[:, 0]
    angle = data[:, 1]

    plt.figure(figsize=(6, 4))
    plt.plot(t, angle, marker="o", markersize=3, linewidth=1.5)

    plt.xlabel("Timestep")
    plt.ylabel("Angle between chromosomes (degree)")
    plt.title("Chromosome alignment angle vs time")

    output_dir = os.path.dirname(output_file)
    os.makedirs(output_dir, exist_ok=True)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "chromosome_angle_vs_time.png"), dpi=300)
    plt.savefig(os.path.join(output_dir, "chromosome_angle_vs_time.pdf"))
    plt.show()