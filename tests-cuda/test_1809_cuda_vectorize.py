# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import numpy as np  # noqa: F401
import pytest  # noqa: F401

import awkward as ak  # noqa: F401

numba = pytest.importorskip("numba")

import math  # noqa: E402

from numba import cuda, vectorize  # noqa: F401, E402


@cuda.jit(device=True)
def polar_to_cartesian(rho, theta):
    x = rho * math.cos(theta)
    y = rho * math.sin(theta)
    return x, y


@vectorize(["float32(float32, float32, float32, float32)"], target="cuda")
def polar_distance(rho1, theta1, rho2, theta2):
    x1, y1 = polar_to_cartesian(rho1, theta1)
    x2, y2 = polar_to_cartesian(rho2, theta2)

    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5


@numba.njit
def polar_to_cartesian_cpu(rho, theta):
    x = rho * math.cos(theta)
    y = rho * math.sin(theta)
    return x, y


@vectorize(["float32(float32, float32, float32, float32)"])  # default target is CPU
def polar_distance_cpu(rho1, theta1, rho2, theta2):
    x1, y1 = polar_to_cartesian_cpu(rho1, theta1)
    x2, y2 = polar_to_cartesian_cpu(rho2, theta2)

    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5


def test_polar_distance():
    n = 1000000
    rho1 = ak.Array(np.random.uniform(0.5, 1.5, size=n).astype(np.float32))
    theta1 = ak.Array(np.random.uniform(-np.pi, np.pi, size=n).astype(np.float32))
    rho2 = ak.Array(np.random.uniform(0.5, 1.5, size=n).astype(np.float32))
    theta2 = ak.Array(np.random.uniform(-np.pi, np.pi, size=n).astype(np.float32))

    np.testing.assert_allclose(
        polar_distance(rho1, theta1, rho2, theta2),
        polar_distance_cpu(rho1, theta1, rho2, theta2),
        rtol=1e-7,
        atol=5e-7,
    )
