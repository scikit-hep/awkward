import sys
from pathlib import Path
import awkward as ak
import numpy as np
import cupy as cp

sys.path.insert(0, str(Path(__file__).parent))
from helpers import filter_lists, list_sizes, select_lists, transform_lists  # noqa: E402


# 3 events with different numbers of electrons and muons in each
# (numerical values are made up and aren't physically meaningful)
# ---
# legend:
# - 'pt': transverse momentum
# - 'eta': pseudorapidity (collider detector angle, e.g. CMS detector)
# - 'phi': azimuthal angle (collider detector angle, e.g. CMS detector)
electrons = ak.Array(
    [
        # 2 electrons
        {"pt": [50.0, 60.0], "eta": [2.1, 2.2], "phi": [0.6, 0.7]},
        # 1 electron
        {"pt": [30.0], "eta": [-1.5], "phi": [0.3]},
        # 0 electrons
        {"pt": [], "eta": [], "phi": []},
    ]
)
muons = ak.Array(
    [
        # 1 muon
        {"pt": [45.0], "eta": [2.5], "phi": [0.4]},
        # 2 muons
        {"pt": [25.0, 35.0], "eta": [-2.0, 1.0], "phi": [0.5, 0.6]},
        # 1 muon
        {"pt": [15.0], "eta": [0.0], "phi": [0.7]},
    ]
)

events = ak.zip({"electrons": electrons, "muons": muons}, depth_limit=1)


def invariant_mass(two_particles: ak.Array) -> ak.Array:
    """Compute invariant mass of two particles given their pt, eta, phi."""
    pt1, eta1, phi1 = (
        two_particles[:, 0].pt,
        two_particles[:, 0].eta,
        two_particles[:, 0].phi,
    )
    pt2, eta2, phi2 = (
        two_particles[:, 1].pt,
        two_particles[:, 1].eta,
        two_particles[:, 1].phi,
    )

    # https://en.wikipedia.org/wiki/Invariant_mass#Collider_experiments
    m2 = 2 * pt1 * pt2 * (np.cosh(eta1 - eta2) - np.cos(phi1 - phi2))
    return np.sqrt(m2)


def physics_analysis(events: ak.Array) -> ak.Array:
    """
    A oversimplified physics analysis selecting events with exactly 2 leptons (electrons or muons)
    and computing their invariant mass.
    """
    # select only electrons with pt > 40
    selected_electrons = events.electrons[events.electrons.pt > 40.0]
    # select only muons with pt > 20 and abs(eta) < 2.4
    selected_muons = events.muons[
        (events.muons.pt > 20.0) & (abs(events.muons.eta) < 2.4)
    ]

    # choose exactly 2 leptons (electrons or muons)
    two_electrons = selected_electrons[ak.num(
        selected_electrons.pt, axis=-1) == 2]
    two_muons = selected_muons[ak.num(selected_muons.pt, axis=-1) == 2]

    return {
        "electron": invariant_mass(two_electrons),
        "muon": invariant_mass(two_muons),
    }


def physics_analysis_gpu(events: ak.Array) -> ak.Array:
    """
    A oversimplified physics analysis selecting events with exactly 2 leptons (electrons or muons)
    and computing their invariant mass.
    """
    # select only electrons with pt > 40
    selected_electrons = events.electrons[events.electrons.pt > 40.0]
    # select only muons with pt > 20 and abs(eta) < 2.4
    selected_muons = events.muons[
        (events.muons.pt > 20.0) & (abs(events.muons.eta) < 2.4)
    ]

    # choose exactly 2 leptons (electrons or muons)
    two_electrons = selected_electrons[ak.num(
        selected_electrons.pt, axis=-1) == 2]
    two_muons = selected_muons[ak.num(
        selected_muons.pt, axis=-1) == 2]

    return {
        "electron": invariant_mass(two_electrons),
        "muon": invariant_mass(two_muons),
    }


def physics_analysis_cccl(events: ak.Array) -> ak.Array:
    """
    CCCL-based physics analysis selecting events with exactly 2 leptons (electrons or muons)
    and computing their invariant mass using cuda.compute primitives.
    """
    def cond_muon(x):
        return (x[0] > 20.0) & (abs(x[1]) < 2.4)

    def cond_electron(x):
        return x[0] > 40.0

    selected_muons = filter_lists(events.muons, cond_muon)
    selected_electrons = filter_lists(events.electrons, cond_electron)

    two_muons = select_lists(
        selected_muons, (list_sizes(selected_muons) == 2).astype('int8'))
    two_electrons = select_lists(
        selected_electrons, (list_sizes(selected_electrons) == 2).astype('int8'))

    def invariant_mass(two_particles):
        """Compute invariant mass of two particles given their pt, eta, phi."""
        pt1, eta1, phi1 = (
            two_particles[0][0],
            two_particles[0][1],
            two_particles[0][2],
        )
        pt2, eta2, phi2 = (
            two_particles[1][0],
            two_particles[1][1],
            two_particles[1][2],
        )
        # https://en.wikipedia.org/wiki/Invariant_mass#Collider_experiments
        m2 = 2 * pt1 * pt2 * (np.cosh(eta1 - eta2) - np.cos(phi1 - phi2))
        return m2 ** 0.5

    masses_electrons = cp.zeros(len(two_electrons), dtype=np.float64)
    masses_muons = cp.zeros(len(two_muons), dtype=np.float64)

    transform_lists(two_muons, masses_muons, 2, invariant_mass)
    transform_lists(two_electrons, masses_electrons, 2, invariant_mass)

    return {
        "electron": masses_electrons,
        "muon": masses_muons,
    }


if __name__ == "__main__":
    # ipython -i studies/cccl/playground.py to play around with `events` and `physics_analysis`

    # Run original function
    inv_mass = physics_analysis(events)
    print("Original physics_analysis() results:")
    print("  Electron invariant masses (in GeV):", inv_mass["electron"])
    print("  Muon invariant masses (in GeV):", inv_mass["muon"])

    events_gpu = ak.to_backend(events, "cuda")

    # Run Awkward Array on GPU function
    inv_mass_gpu = physics_analysis_gpu(events_gpu)
    print("\nGPU physics_analysis_gpu() results:")
    print("  Electron invariant masses (in GeV):", inv_mass_gpu["electron"])
    print("  Muon invariant masses (in GeV):", inv_mass_gpu["muon"])

    # Run CCCL-based function
    inv_mass_cccl = physics_analysis_cccl(events_gpu)
    print("\nCCCL physics_analysis_cccl() results:")
    print("  Electron invariant masses (in GeV):", inv_mass_cccl["electron"])
    print("  Muon invariant masses (in GeV):", inv_mass_cccl["muon"])
