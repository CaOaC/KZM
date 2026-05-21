import numpy as np

# ============================================================
# Parameters
# ============================================================
d = 20.0                                  # distance between two sister chromatids
shift_direction = np.array([1.0, 0.0, 0.0])  # translation direction

N = 2000
target_density = 0.2

# Box
box_lo = 0.0
box_hi = 100.0

box_center = np.array([
    0.5 * (box_lo + box_hi),
    0.5 * (box_lo + box_hi),
    0.5 * (box_lo + box_hi)
], dtype=np.float64)

# ============================================================
# Load the single chromosome centerline
# ============================================================
data = np.loadtxt("chromosome_A.txt")
R = data[:, 2:5].astype(np.float64)

# In case the actual file length is different from N
N = len(R)

# ============================================================
# Center the original chromosome at the box center
# ============================================================
R_center = R.mean(axis=0)
R_centered = R - R_center + box_center

# ============================================================
# Generate two sister chromatids
#
# The center of the two-chain system is at box_center.
# The separation between the two chains is d.
# ============================================================
shift_direction = shift_direction.astype(np.float64)
shift_direction = shift_direction / np.linalg.norm(shift_direction)

shift_vector = d * shift_direction

r1 = R_centered - 0.5 * shift_vector
r2 = R_centered + 0.5 * shift_vector

# ============================================================
# Check coordinate range
# ============================================================
all_pos = np.vstack([r1, r2])

coord_min = all_pos.min(axis=0)
coord_max = all_pos.max(axis=0)

print("[B] Coordinate min =", coord_min)
print("[B] Coordinate max =", coord_max)

if np.any(coord_min < box_lo) or np.any(coord_max > box_hi):
    print("[Warning] Some coordinates are outside the box!")
    print("          You may need a larger box or a smaller d.")
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
dump_filename = "2000_sis_sep.dump"

write_sisters_dump(dump_filename, r1, r2, box_lo, box_hi)

np.save("sister_1.npy", r1)
np.save("sister_2.npy", r2)

print("[B] Finished.")
print("Saved sister_1.npy, sister_2.npy, and", dump_filename)
print(f"[B] Translation vector between sisters = {shift_vector}")
print(f"[B] Box center = {box_center}")
print(f"[B] Dump box = [{box_lo:.8f}, {box_hi:.8f}] in x,y,z")