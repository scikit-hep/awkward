# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401


def test_empty_slice():
    # muon = ak.Array([[{"pt": 1.0}], []], with_name="muon")
    electron = ak.Array([[], [{"pt": 1.0}]], with_name="electron")

    electron = electron[electron.pt > 5]

    id = ak.argsort(electron, axis=1)

    assert ak.to_list(electron[id]) == [[], []]


# from coffea import nanoevents
#
# def test_empty_unions():
#     nanoevents.NanoAODSchema.mixins["FatJetLS"] = "PtEtaPhiMLorentzVector"
#
#     x = nanoevents.NanoEventsFactory.from_root(
#         './nano106Xv8_on_mini106X_2017_mc_NANO_py_NANO_7.root',
#         entry_start=100560,
#         entry_stop=102266
#     )
#
#     events = x.events()
#
#     muons = events.Muon
#     electrons = events.Electron
#     jets = events.Jet
#     fatjets = events.FatJet
#     subjets = events.SubJet
#     fatjetsLS = events.FatJetLS
#     met = events.MET
#
#     goodmuon = (
#         (muons.mediumId)
#         & (muons.miniPFRelIso_all <= 0.2)
#         & (muons.pt >= 27)
#         & (abs(muons.eta) <= 2.4)
#         & (abs(muons.dz) < 0.1)
#         & (abs(muons.dxy) < 0.05)
#         & (muons.sip3d < 4)
#     )
#     good_muons = muons[goodmuon]
#
#     # electrons
#     goodelectron = (
#         (electrons.mvaFall17V2noIso_WP90)
#         & (electrons.pt >= 30)
#         & (abs(electrons.eta) <= 1.479)
#         & (abs(electrons.dz) < 0.1)
#         & (abs(electrons.dxy) < 0.05)
#         & (electrons.sip3d < 4)
#     )
#     good_electrons = electrons[goodelectron]
#
#     leptons = ak.concatenate([muons, electrons], axis=1)
#     good_leptons = ak.concatenate([good_muons, good_electrons], axis=1)
#
#     idx1 = ak.argsort(leptons.pt, axis=1)
#     idx2 = ak.argsort(good_leptons.pt, axis=1)
#
#     leptons[idx1]
#
#     good_leptons[idx2]
