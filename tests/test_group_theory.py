import numpy as np
import pytest

from matrixbootstrap.group_theory import SpecialUnitaryGroup, build_N_dim_irrep_of_SU_2


@pytest.mark.parametrize("N", [2, 3, 4, 5])
def test_su2_casimir_is_j_times_j_plus_one(N):
    """
    The Casimir operator J² = Jx² + Jy² + Jz² must equal j(j+1) I,
    where j = (N-1)/2 is the spin of the representation.

    This is the defining property of a spin-j irrep of SU(2): all states in
    the multiplet are eigenstates of J² with the same eigenvalue j(j+1).
    """
    j = (N - 1) / 2
    mats = build_N_dim_irrep_of_SU_2(N)
    Jx, Jy, Jz = mats["Jx"], mats["Jy"], mats["Jz"]
    casimir = Jx @ Jx + Jy @ Jy + Jz @ Jz
    assert np.allclose(casimir, j * (j + 1) * np.eye(N))


@pytest.mark.parametrize("N", [2, 3, 4])
def test_sun_generators_satisfy_lie_algebra(N):
    """
    SU(N) generators must satisfy the Lie algebra relation [T_i, T_j] = i f_{ijk} T_k.

    SpecialUnitaryGroup._validate() checks this relation for every pair (i, j)
    and raises AssertionError on failure. Successful instantiation is therefore
    sufficient as a test.
    """
    SpecialUnitaryGroup(N)  # _validate() is called in __init__
