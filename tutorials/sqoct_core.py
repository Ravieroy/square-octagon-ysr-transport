"""
This modules contains core utilities for square-octagon BdG and YSR calculations.

Conventions
-----------
- NN hopping: t1.
- Intra-cell diagonal NNN hopping: t2.
- Nambu basis: (c_up, c_down, c_up^dagger, c_down^dagger).
- Uniform onsite spin-singlet pairing Delta.
- Classical magnetic impurity polarized along z with exchange J.
- Energies are typically measured in units of t1.

This module contains no plotting code. Figure notebooks should import
these routines and handle plotting separately.
"""

import warnings

import kwant
import numpy as np
import scipy.sparse as sp

from scipy.optimize import minimize_scalar
from scipy.sparse.linalg import ArpackNoConvergence, eigsh, splu


# Pauli matrices and BdG constants

s0 = np.eye(2, dtype=complex)

sx = np.array([
    [0, 1],
    [1, 0]
], dtype=complex)

sy = np.array([
    [0, -1j],
    [1j, 0]
], dtype=complex)

sz = np.array([
    [1, 0],
    [0, -1]
], dtype=complex)

z2 = np.zeros((2, 2), dtype=complex)
I4 = np.eye(4, dtype=complex)

# Magnetic-impurity exchange matrix (M_imp)

MIMP = np.block([
    [sz, z2],
    [z2, -sz]
])


# Square-octagon lattice

sqrt2 = np.sqrt(2.0)
a1 = (1.0 + sqrt2, 0.0)
a2 = (0.0, 1.0 + sqrt2)
_s = 1.0 / sqrt2

basis = [
    (0.0, _s),
    (-_s, 0.0),
    (0.0, -_s),
    (_s, 0.0)
]

lat = kwant.lattice.general(
    [a1, a2], basis=basis,
    norbs=4, name="sqo"
)

A, B, C, D = lat.sublattices

SUBLATTICES = {
    "A": A,
    "B": B,
    "C": C,
    "D": D,
}


# Square-octagon connectivity

NN_HOPS = [
    kwant.builder.HoppingKind((0, 0), A, B),
    kwant.builder.HoppingKind((0, 0), B, C),
    kwant.builder.HoppingKind((0, 0), C, D),
    kwant.builder.HoppingKind((0, 0), D, A),
    kwant.builder.HoppingKind((0, 1), C, A),
    kwant.builder.HoppingKind((1, 0), B, D),
]

NNN_HOPS = [
    kwant.builder.HoppingKind((0, 0), A, C),
    kwant.builder.HoppingKind((0, 0), B, D),
]


# BdG onsite and hopping matrices

def onsite_clean(site, mu, Delta, J):
    h = -mu * s0
    pairing = 1j * Delta * sy

    return np.block([
        [h, pairing],
        [pairing.conj().T, -h.T]
    ])


def onsite_impurity(site, mu, Delta, J):
    h = -mu * s0 + J * sz
    pairing = 1j * Delta * sy

    return np.block([
        [h, pairing],
        [pairing.conj().T, -h.T]
    ])


def hopping_bdg(t):
    he = -t * s0

    return np.block([
        [he, z2],
        [z2, -he.conj()]
    ])


# Finite isolated YSR system

def build_ysr_system(
    Nx, Ny,
    t1=1.0, t2=1.0,
    impurity_sublattice="A",
    impurity_cell=None,
):

    if impurity_sublattice not in SUBLATTICES:
        raise ValueError("impurity_sublattice must be one of A, B, C, D")

    if impurity_cell is None:
        impurity_cell = (Nx // 2, Ny // 2)

    impurity_site = SUBLATTICES[impurity_sublattice](*impurity_cell)
    syst = kwant.Builder()

    for ix in range(Nx):
        for iy in range(Ny):
            syst[A(ix, iy)] = onsite_clean
            syst[B(ix, iy)] = onsite_clean
            syst[C(ix, iy)] = onsite_clean
            syst[D(ix, iy)] = onsite_clean

    syst[impurity_site] = onsite_impurity

    for hop in NN_HOPS:
        syst[hop] = hopping_bdg(t1)

    for hop in NNN_HOPS:
        syst[hop] = hopping_bdg(t2)

    fsyst = syst.finalized()
    impurity_index = fsyst.id_by_site[impurity_site]

    return fsyst, impurity_site, impurity_index


# Low-energy BdG eigensolver

def low_energy_states(
    fsyst, J, mu, Delta,
    k=6, sigma=1e-7,
    return_vectors=False,
    tol=1e-9, maxiter=200000,
):
    H = fsyst.hamiltonian_submatrix(
        sparse=True,
        params={"J": J, "mu": mu, "Delta": Delta}
    ).tocsc()

    ncv = min(max(4 * k + 10, 30), H.shape[0] - 1)

    try:
        vals, vecs = eigsh(
            H, k=k, sigma=sigma, which="LM",
            ncv=ncv, maxiter=maxiter, tol=tol
        )

    except ArpackNoConvergence as err:
        vals = err.eigenvalues
        vecs = err.eigenvectors

        if vals is None or len(vals) < 2:
            raise

        warnings.warn(
            f"ARPACK returned only {len(vals)} converged eigenvalues "
            f"at J={J:.6g}, mu={mu:.6g}.",
            RuntimeWarning,
        )

    order = np.argsort(vals)
    vals = np.asarray(vals)[order]

    if vecs is not None:
        vecs = np.asarray(vecs)[:, order]

    if return_vectors:
        return vals, vecs

    return vals


def ysr_energy(fsyst, J, mu, Delta, k=6):
    vals = low_energy_states(
        fsyst, J=J, mu=mu, Delta=Delta,
        k=k, return_vectors=False
    )
    return float(np.min(np.abs(vals)))


def scan_ysr_vs_J(fsyst, J_values, mu, Delta, k=6):
    return np.array([
        ysr_energy(fsyst, J=J, mu=mu, Delta=Delta, k=k)
        for J in np.asarray(J_values)
    ])


# Wavefunction observables

def state_site_weights(fsyst, psi, normalize=True):
    nsites = len(fsyst.sites)
    psi_site = np.asarray(psi).reshape(nsites, 4)
    weights = np.sum(np.abs(psi_site) ** 2, axis=1).real

    if normalize:
        total = np.sum(weights)
        if total > 0:
            weights = weights / total

    return weights


def state_ipr(weights):
    weights = np.asarray(weights, dtype=float)
    total = weights.sum()

    if total <= 0:
        return np.nan

    w = weights / total
    return float(np.sum(w ** 2))


def impurity_weight(weights, impurity_index):
    weights = np.asarray(weights, dtype=float)
    total = weights.sum()

    if total <= 0:
        return np.nan

    return float(weights[impurity_index] / total)


def sublattice_weights(fsyst, weights):
    weights = np.asarray(weights, dtype=float)
    total = weights.sum()

    if total > 0:
        weights = weights / total

    out = {}

    for label, family in SUBLATTICES.items():
        inds = [
            i for i, site in enumerate(fsyst.sites)
            if site.family == family
        ]
        out[label] = float(np.sum(weights[inds]))

    return out


# Local superconducting Green function

def local_clean_green(
    fsyst, impurity_index,
    mu, Delta,
    E=0.0, eta=1e-8,
):
    H0 = fsyst.hamiltonian_submatrix(
        sparse=True,
        params={"J": 0.0, "mu": mu, "Delta": Delta}
    ).tocsc()

    N = H0.shape[0]
    mat = (E + 1j * eta) * sp.eye(N, dtype=complex, format="csc") - H0
    lu = splu(mat)

    inds = 4 * impurity_index + np.arange(4)
    rhs = np.zeros((N, 4), dtype=complex)

    for a, ind in enumerate(inds):
        rhs[ind, a] = 1.0

    X = lu.solve(rhs)

    return X[inds, :]


def ysr_pole_strength(J, Gii):
    D = I4 - J * (Gii @ MIMP)
    return float(np.linalg.svd(D, compute_uv=False)[-1])


def Jc_from_green(
    Gii,
    Jmax=5.0, nscan=801,
    pole_tol=1e-3, xatol=1e-7,
    return_scan=False,
):
    J_scan = np.linspace(0.0, Jmax, nscan)
    s_scan = np.array([
        ysr_pole_strength(J, Gii)
        for J in J_scan
    ])

    candidate_inds = []

    for i in range(1, len(J_scan) - 1):
        if s_scan[i] <= s_scan[i - 1] and s_scan[i] <= s_scan[i + 1]:
            candidate_inds.append(i)

    refined = []

    for idx in candidate_inds:
        left = J_scan[idx - 1]
        right = J_scan[idx + 1]

        result = minimize_scalar(
            lambda J: ysr_pole_strength(J, Gii),
            bounds=(left, right), method="bounded",
            options={"xatol": xatol}
        )

        refined.append((float(result.x), float(result.fun)))

    good = [
        pair for pair in refined
        if pair[0] > 0 and pair[1] < pole_tol
    ]

    if good:
        Jc, pole_min = sorted(good, key=lambda x: x[0])[0]
    else:
        idx_best = int(np.argmin(s_scan))
        Jc = np.nan
        pole_min = float(s_scan[idx_best])

    if return_scan:
        return Jc, pole_min, J_scan, s_scan

    return Jc, pole_min


def local_green_components(Gii):
    g = 0.5 * (Gii[0, 0] + Gii[1, 1])
    f_s = 0.5 * (Gii[0, 3] - Gii[1, 2])

    return g, f_s


def scan_Jc_vs_mu_green(
    fsyst, impurity_index,
    mu_values, Delta,
    Jmax=5.0, nscan=801,
    pole_tol=1e-3, eta=1e-8,
):

    mu_values = np.asarray(mu_values, dtype=float)

    Jc = np.full(mu_values.shape, np.nan, dtype=float)
    pole_min = np.full(mu_values.shape, np.nan, dtype=float)
    g = np.full(mu_values.shape, np.nan + 0j, dtype=complex)
    f = np.full(mu_values.shape, np.nan + 0j, dtype=complex)

    for i, mu in enumerate(mu_values):
        Gii = local_clean_green(
            fsyst, impurity_index,
            mu=mu, Delta=Delta,
            E=0.0, eta=eta
        )

        Jc[i], pole_min[i] = Jc_from_green(
            Gii, Jmax=Jmax,
            nscan=nscan, pole_tol=pole_tol
        )

        g[i], f[i] = local_green_components(Gii)

    return {
        "mu": mu_values,
        "Jc": Jc,
        "pole_min": pole_min,
        "g": g,
        "f": f,
    }


def near_zero_ysr_state(
    fsyst, impurity_index,
    J, mu, Delta,
    k=8,
):

    vals, vecs = low_energy_states(
        fsyst, J=J, mu=mu, Delta=Delta,
        k=k, return_vectors=True
    )

    idx = int(np.argmin(np.abs(vals)))
    energy = float(vals[idx])
    psi = vecs[:, idx]

    weights = state_site_weights(fsyst, psi, normalize=True)

    return {
        "energy": energy,
        "psi": psi,
        "weights": weights,
        "ipr": state_ipr(weights),
        "impurity_weight": impurity_weight(weights, impurity_index),
        "sublattice_weights": sublattice_weights(fsyst, weights),
    }


# Normal-state spectrum and local DOS

def normal_state_spectrum(fsyst):
    Hbdg = fsyst.hamiltonian_submatrix(
        sparse=False,
        params={"J": 0.0, "mu": 0.0, "Delta": 0.0}
    )

    nsites = len(fsyst.sites)
    e_up = np.arange(0, 4 * nsites, 4)
    Hnormal = Hbdg[np.ix_(e_up, e_up)]

    eps, vecs = np.linalg.eigh(Hnormal)

    return eps, vecs


def normal_local_dos(
    fsyst, impurity_index,
    energies, eta=0.03,
    return_bulk_dos=False,
):
    energies = np.atleast_1d(np.asarray(energies, dtype=float))
    eps, vecs = normal_state_spectrum(fsyst)

    imp_weights = np.abs(vecs[impurity_index, :]) ** 2

    def lorentzian(E):
        return eta / np.pi / ((E - eps) ** 2 + eta ** 2)

    rho_local = np.array([
        np.sum(imp_weights * lorentzian(E))
        for E in energies
    ])

    if not return_bulk_dos:
        return rho_local

    nsites = len(fsyst.sites)
    rho_bulk = np.array([
        np.sum(lorentzian(E)) / nsites
        for E in energies
    ])

    return rho_local, rho_bulk


# Bloch Hamiltonian and clean band structure

def bloch_hamiltonian(k1, k2, t1=1.0, t2=1.0, mu=0.0):
    H = np.zeros((4, 4), dtype=complex)

    # Internal square edges
    H[0, 1] = -t1
    H[1, 2] = -t1
    H[2, 3] = -t1
    H[0, 3] = -t1

    # Diagonal + inter-cell hopping
    H[0, 2] = -(t2 + t1 * np.exp(-1j * k2))
    H[1, 3] = -(t2 + t1 * np.exp(-1j * k1))

    H = H + H.conj().T
    H -= mu * np.eye(4)

    return H


def square_bz_path(nk_segment=150):
    G = np.array([0.0, 0.0])
    X = np.array([np.pi, 0.0])
    M = np.array([np.pi, np.pi])

    vertices = [G, X, M, G]

    kpts = []
    xvals = []
    ticks = [0.0]
    running = 0.0

    for iseg, (p0, p1) in enumerate(zip(vertices[:-1], vertices[1:])):
        ts = np.linspace(
            0.0, 1.0, nk_segment,
            endpoint=(iseg == len(vertices) - 2)
        )

        for t in ts:
            k = (1 - t) * p0 + t * p1

            if kpts:
                running += np.linalg.norm(k - kpts[-1])

            kpts.append(k.copy())
            xvals.append(running)

        ticks.append(running)

    return (
        np.asarray(kpts),
        np.asarray(xvals),
        ticks,
        [r"$\Gamma$", r"$X$", r"$M$", r"$\Gamma$"],
    )


def band_structure(t1=1.0, t2=1.0, mu=0.0, nk_segment=150):
    kpts, x, ticks, labels = square_bz_path(nk_segment=nk_segment)

    bands = np.array([
        np.linalg.eigvalsh(
            bloch_hamiltonian(
                k1=k[0], k2=k[1],
                t1=t1, t2=t2, mu=mu
            )
        )
        for k in kpts
    ])

    return x, bands, ticks, labels


# Small geometry helpers

def site_positions(fsyst):
    return np.asarray([site.pos for site in fsyst.sites], dtype=float)


def bond_type_from_distance(site1, site2, atol=1e-6):
    r1 = np.asarray(site1.pos, dtype=float)
    r2 = np.asarray(site2.pos, dtype=float)
    d = np.linalg.norm(r1 - r2)

    if np.isclose(d, 1.0, atol=atol):
        return "t1"

    if np.isclose(d, np.sqrt(2.0), atol=atol):
        return "t2"

    return "unknown"


__all__ = [
    "lat", "A", "B", "C", "D",
    "a1", "a2", "NN_HOPS", "NNN_HOPS", "SUBLATTICES",
    "s0", "sx", "sy", "sz", "z2", "I4", "MIMP",
    "build_ysr_system",
    "low_energy_states", "ysr_energy", "scan_ysr_vs_J",
    "state_site_weights", "state_ipr", "impurity_weight", "sublattice_weights",
    "local_clean_green", "ysr_pole_strength", "Jc_from_green",
    "local_green_components", "scan_Jc_vs_mu_green", "near_zero_ysr_state",
    "normal_state_spectrum", "normal_local_dos",
    "bloch_hamiltonian", "square_bz_path", "band_structure",
    "site_positions", "bond_type_from_distance",
]
