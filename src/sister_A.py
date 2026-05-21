import numpy as np
from numba import njit, prange

# ============================================================
# Parameters
# ============================================================
N = 1000                        # number of beads
dt = 0.001                      # Langevin timestep
steps = 1000000                   # total number of steps
dump_interval = 1000             # dump every this many steps

gamma = 1.0                     # friction coefficient
kT = 1.0                        # thermal energy

k_bond = 50.0                   # spring constant for bonds
bond_length = 1.0               # equilibrium bond length

# Literature-like density: rho = 0.01 particle / unit_length^3
# R = (3N / (4 pi rho))^(1/3)
target_density = 0.2
sphere_radius = (3.0 * N / (4.0 * np.pi * target_density)) ** (1.0 / 3.0)
# sphere_radius = 10

# Soft repulsion only (topo II effect: chains may pass through each other)
repulsion_strength = 0.2
repulsion_cutoff = 2.0

# Spherical confinement strength
k_confine = 5

# Numerical safety
eps = 1e-6
F_max = 100.0

# Dump box bounds: consistent with program box
box_lo = -sphere_radius
box_hi = +sphere_radius

# ============================================================
# Initialization
# Start from a stretched chain with small noise to avoid overlaps
# ============================================================
r = np.zeros((N, 3), dtype=np.float64)
for i in range(1, N):
    r[i] = r[i - 1] + np.array([bond_length, 0.0, 0.0])
    r[i] += 0.05 * np.random.randn(3)

# Shift initial chain roughly to center
r -= np.mean(r, axis=0)

# ============================================================
# Force computation
# ============================================================
@njit(parallel=True)
def compute_bond_forces(r, k_bond, bond_length, eps):
    N = r.shape[0]
    F = np.zeros_like(r)

    for i in prange(N - 1):
        dr = r[i + 1] - r[i]
        dist = np.sqrt(np.sum(dr * dr)) + eps
        f = k_bond * (dist - bond_length) * dr / dist
        F[i] += f
        F[i + 1] -= f

    return F


@njit(parallel=True)
def compute_soft_repulsion_forces(r, repulsion_strength, repulsion_cutoff, eps):
    N = r.shape[0]
    F = np.zeros_like(r)

    # Soft repulsion only, no hard-core topology preservation
    # This allows effective strand passage (topo II-like)
    for i in prange(N):
        for j in range(i + 2, N):   # exclude bonded neighbors
            dr = r[j] - r[i]
            dist = np.sqrt(np.sum(dr * dr)) + eps

            if dist < repulsion_cutoff:
                # Mild, finite soft force
                overlap = repulsion_cutoff - dist
                f = repulsion_strength * overlap * dr / (dist + 0.01)
                F[i] -= f
                F[j] += f

    return F


@njit(parallel=True)
def compute_confinement_forces(r, sphere_radius, k_confine, eps):
    N = r.shape[0]
    F = np.zeros_like(r)

    for i in prange(N):
        dist = np.sqrt(np.sum(r[i] * r[i])) + eps
        if dist > sphere_radius:
            # harmonic restoring force towards sphere interior
            F[i] -= k_confine * (dist - sphere_radius) * r[i] / dist

    return F


@njit(parallel=True)
def clip_forces(F, F_max, eps):
    N = F.shape[0]
    for i in prange(N):
        norm = np.sqrt(np.sum(F[i] * F[i])) + eps
        if norm > F_max:
            F[i] *= F_max / norm
    return F


@njit(parallel=True)
def add_forces(F1, F2, F3):
    N = F1.shape[0]
    F = np.zeros_like(F1)
    for i in prange(N):
        F[i] = F1[i] + F2[i] + F3[i]
    return F


# ============================================================
# Dump writer (LAMMPS-style dump, OVITO-readable)
# Box size is consistent with program box: [-sphere_radius, sphere_radius]
# ============================================================
def write_dump_frame(fh, step, r, box_lo, box_hi):
    N = len(r)
    fh.write("ITEM: TIMESTEP\n")
    fh.write(f"{step}\n")
    fh.write("ITEM: NUMBER OF ATOMS\n")
    fh.write(f"{N}\n")
    fh.write("ITEM: BOX BOUNDS pp pp pp\n")
    fh.write(f"{box_lo:.8f} {box_hi:.8f}\n")
    fh.write(f"{box_lo:.8f} {box_hi:.8f}\n")
    fh.write(f"{box_lo:.8f} {box_hi:.8f}\n")
    fh.write("ITEM: ATOMS id type x y z\n")
    for i in range(N):
        fh.write(f"{i+1} 1 {r[i,0]:.8f} {r[i,1]:.8f} {r[i,2]:.8f}\n")


# ============================================================
# Langevin dynamics
# overdamped: r(t+dt) = r + (dt/gamma) F + sqrt(2kT dt/gamma) eta
# ============================================================
noise_prefactor = np.sqrt(2.0 * kT * dt / gamma)

with open("1000_sigle.dump", "w") as dump_fh:
    for step in range(steps):
        F_bond = compute_bond_forces(r, k_bond, bond_length, eps)
        F_rep = compute_soft_repulsion_forces(r, repulsion_strength, repulsion_cutoff, eps)
        F_conf = compute_confinement_forces(r, sphere_radius, k_confine, eps)

        F = add_forces(F_bond, F_rep, F_conf)
        F = clip_forces(F, F_max, eps)

        noise = noise_prefactor * np.random.randn(N, 3)
        r += (dt / gamma) * F + noise

        if step % dump_interval == 0:
            write_dump_frame(dump_fh, step, r, box_lo, box_hi)
            print(f"[A] step = {step}/{steps}, sphere_radius = {sphere_radius:.6f}")

        if np.isnan(r).any():
            raise RuntimeError(f"NaN detected at step {step}")

np.save("chromosome_A.npy", r)
print("[A] Finished. Saved chromosome_A.npy and chromosome_A.dump")
print(f"[A] Sphere radius = {sphere_radius:.8f}")
print(f"[A] Dump box = [{box_lo:.8f}, {box_hi:.8f}] in x,y,z")