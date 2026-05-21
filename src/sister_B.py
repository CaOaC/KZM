import numpy as np

# ============================================================
# Parameters
# ============================================================
twist_period = 100          # one full turn every 100 monomers
sister_distance = 1.0       # half-separation radius around centerline
eps = 1e-8

# Must match Program A if you want identical dump box
N = 2000
target_density = 0.2
sphere_radius = (3.0 * N / (4.0 * np.pi * target_density)) ** (1.0 / 3.0)

box_lo = 0
box_hi = 100

box_center = np.array([
    0.5 * (box_lo + box_hi),
    0.5 * (box_lo + box_hi),
    0.5 * (box_lo + box_hi)
], dtype=np.float64)

# ============================================================
# Load the single chromosome centerline
# ============================================================
data = np.loadtxt("1000_single_config.txt")
R = data[:, 2:5].astype(np.float64)

# In case the actual file length is different from N
N = len(R)

# ============================================================
# Move the original centerline to the box center
# ============================================================
R_original_center = R.mean(axis=0)
R = R - R_original_center + box_center

print("[B] Original centerline center =", R_original_center)
print("[B] New centerline center      =", R.mean(axis=0))
print("[B] Target box center          =", box_center)

# ============================================================
# Build local orthonormal frame along centerline
# We use a simple tangent-normal-binormal-like construction
# ============================================================
def local_frame(R):
    N = len(R)

    t = np.zeros((N, 3), dtype=np.float64)
    n = np.zeros((N, 3), dtype=np.float64)
    b = np.zeros((N, 3), dtype=np.float64)

    # tangent
    for i in range(1, N - 1):
        dr = R[i + 1] - R[i - 1]
        norm = np.linalg.norm(dr) + eps
        t[i] = dr / norm

    # fill boundaries
    t[0] = t[1]
    t[-1] = t[-2]

    # choose a reference vector not parallel to tangent
    for i in range(N):
        ref = np.array([1.0, 0.0, 0.0], dtype=np.float64)

        if abs(np.dot(t[i], ref)) > 0.9:
            ref = np.array([0.0, 1.0, 0.0], dtype=np.float64)

        n_i = np.cross(t[i], ref)
        n_norm = np.linalg.norm(n_i) + eps
        n[i] = n_i / n_norm

        b_i = np.cross(t[i], n[i])
        b_norm = np.linalg.norm(b_i) + eps
        b[i] = b_i / b_norm

    return t, n, b

t, n, b = local_frame(R)

# ============================================================
# Generate two sister chromatids by winding around the same path
#
# Since r1 = R + offset and r2 = R - offset,
# the midpoint of the two sisters is exactly R.
# Therefore, if R is centered at box_center,
# the whole two-chain structure is also centered at box_center.
# ============================================================
r1 = np.zeros_like(R)
r2 = np.zeros_like(R)

for i in range(N):
    theta = 2.0 * np.pi * i / twist_period

    offset = sister_distance * (
        np.cos(theta) * n[i] + np.sin(theta) * b[i]
    )

    r1[i] = R[i] + offset
    r2[i] = R[i] - offset

# ============================================================
# Check structure center and coordinate range
# ============================================================
all_pos = np.vstack([r1, r2])

structure_center = all_pos.mean(axis=0)
coord_min = all_pos.min(axis=0)
coord_max = all_pos.max(axis=0)

print("[B] Structure center =", structure_center)
print("[B] Coordinate min   =", coord_min)
print("[B] Coordinate max   =", coord_max)

if np.any(coord_min < box_lo) or np.any(coord_max > box_hi):
    print("[Warning] Some coordinates are outside the box.")
    print("          You may need a larger box or a smaller sister_distance.")
else:
    print("[B] All coordinates are inside the box.")

# ============================================================
# Dump writer
# ============================================================
def write_sisters_dump(filename, r1, r2, box_lo, box_hi):
    N = len(r1)

    with open(filename, "w") as fh:
        fh.write("ITEM: TIMESTEP\n")
        fh.write("0\n")
        fh.write("ITEM: NUMBER OF ATOMS\n")
        fh.write(f"{2*N}\n")
        fh.write("ITEM: BOX BOUNDS pp pp pp\n")
        fh.write(f"{box_lo:.8f} {box_hi:.8f}\n")
        fh.write(f"{box_lo:.8f} {box_hi:.8f}\n")
        fh.write(f"{box_lo:.8f} {box_hi:.8f}\n")
        fh.write("ITEM: ATOMS id type x y z\n")

        # sister 1 => type 1
        for i in range(N):
            fh.write(
                f"{i+1} 1 "
                f"{r1[i,0]:.8f} {r1[i,1]:.8f} {r1[i,2]:.8f}\n"
            )

        # sister 2 => type 2
        for i in range(N):
            fh.write(
                f"{N+i+1} 2 "
                f"{r2[i,0]:.8f} {r2[i,1]:.8f} {r2[i,2]:.8f}\n"
            )

# ============================================================
# Save files
# ============================================================
dump_filename = "1000_sis.dump"

write_sisters_dump(dump_filename, r1, r2, box_lo, box_hi)

np.save("sister_1.npy", r1)
np.save("sister_2.npy", r2)

print("[B] Finished.")
print("Saved sister_1.npy, sister_2.npy, and", dump_filename)
print(f"[B] Sphere radius = {sphere_radius:.8f}")
print(f"[B] Dump box = [{box_lo:.8f}, {box_hi:.8f}] in x,y,z")