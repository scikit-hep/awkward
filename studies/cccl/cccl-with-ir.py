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
    
    print(list_sizes(selected_muons) == 2)
    print((list_sizes(selected_muons) == 2).astype('int8'))
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

    
def physics_analysis_cccl_ir(events: ak.Array) -> ak.Array:
    from awkward._connect.cuda.ir_nodes import (
    Filter,
    SelectLists,
    ListSizes,
    TransformLists,
    )       
    
    """
    CCCL-based physics analysis selecting events with exactly 2 leptons (electrons or muons)
    and computing their invariant mass using cuda.compute primitives.
    """
    def cond_muon(x):
        return (x[0] > 20.0) & (abs(x[1]) < 2.4)

    def cond_electron(x):
        return x[0] > 40.0
    
    #using nodes
    selected_muons = Filter(events.muons, cond_muon)
    selected_electrons = Filter(events.electrons, cond_electron)
        
    # Use lazy wrapper
    # lazy_events = ak.cuda.lazy(events)
    # print(dir(lazy_events))
    # selected_muons_llllllll = lazy_events.filter(cond_muon)
    # # print("selected_muons_llllllll", selected_muons_llllllll)
    
    # TODO: .astype('int8') dooesn't work!
    two_muons = SelectLists(
        selected_muons, (ListSizes(selected_muons) == 2).astype('int8'))
    two_electrons = SelectLists(
        selected_electrons, (ListSizes(selected_electrons) == 2).astype('int8'))

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
    
    masses_electrons = TransformLists(two_muons, 2, invariant_mass)
    masses_muons = TransformLists(two_electrons, 2, invariant_mass)

    return {
        "electron": masses_electrons.compute(),
        "muon": masses_muons.compute(),
    }

if __name__ == "__main__":
    # ipython -i studies/cccl/playground.py to play around with `events` and `physics_analysis`

    events_gpu = ak.to_backend(events, "cuda")

    # Run CCCL-based function
    inv_mass_cccl = physics_analysis_cccl_ir(events_gpu)
    print("\nCCCL physics_analysis_cccl() results:")
    print("  Electron invariant masses (in GeV):", inv_mass_cccl["electron"])
    print("  Muon invariant masses (in GeV):", inv_mass_cccl["muon"])