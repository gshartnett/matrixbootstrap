import numpy as np
import pytest

from matrixbootstrap.brezin import compute_Brezin_energy, compute_Brezin_energy_Han_conventions

# ---------------------------------------------------------------------------
# Reference data: Table 3 from Brezin et al, Commun. Math. Phys. 59, 35-51 (1978).
# Values are given to 3 significant figures, so we test to atol=5e-4.
# ---------------------------------------------------------------------------

TABLE_3 = [
    (0.01, 0.505),
    (0.1,  0.542),
    (0.5,  0.651),
    (1.0,  0.740),
    (50.0, 2.217),
]


@pytest.mark.parametrize("g_value, expected", TABLE_3)
def test_energy_matches_brezin_table3(g_value, expected):
    """
    The large-N ground state energy must reproduce Table 3 of Brezin et al (1978).

    This validates the full numerical pipeline: solving the Fermi level equation
    (Eq. 80b) and evaluating the energy integral (Eq. 80a).
    """
    assert np.isclose(compute_Brezin_energy(g_value), expected, atol=5e-4)


def test_han_brezin_convention_consistency():
    """
    The two convention functions must satisfy E_Han(g) = 2 E_Brezin(g/2).

    Han et al use a rescaled coupling; this tests that the conversion wrapper
    is implemented correctly and that both functions are self-consistent.
    """
    for g in [0.1, 1.0, 10.0]:
        assert np.isclose(
            compute_Brezin_energy_Han_conventions(g),
            2 * compute_Brezin_energy(g / 2),
        )
