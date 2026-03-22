"""
Code for computing the energy in the matrix quantum mechanical model studied in Brezin et al, Sec 5.

    Brézin, E., Itzykson, C., Parisi, G. et al. Planar diagrams. Commun.Math. Phys. 59, 35–51 (1978). https://doi.org/10.1007/BF01614153
"""

import numpy as np
from scipy.integrate import quad
from scipy.optimize import root_scalar


def fermi_level_integrand(x: float, eps: float, g_value: float) -> float:
    """
    Integrand for Eq 80b in Brezin et al.

    Parameters
    ----------
    x : float
        The dummy integration variable, aka lambda in the paper.
    eps : float
        The scaled Fermi level.
    g_value : float
        The quartic coupling constant.

    Returns
    -------
    float
        The value of the integrand.
    """
    return (
        (1 / np.pi)
        * (2 * eps - x**2 - 2 * g_value * x**4) ** (1 / 2)
        * np.heaviside(2 * eps - x**2 - 2 * g_value * x**4, 0.5)
    )


def fermi_level_integral(eps: float, g_value: float) -> float:
    """
    Integrand for Eq 80b in Brezin et al.
    The integration limits are determined by the constraint
    that this expression should be real.

    Parameters
    ----------
    eps : float
        The scaled Fermi level.
    g_value : float
        The quartic coupling constant.

    Returns
    -------
    float
        The value of the integral.
    """
    int_limit_upper = np.sqrt((-1 + np.sqrt(1 + 16 * g_value * eps)) / (4 * g_value))
    int_limit_lower = -int_limit_upper
    result, _ = quad(
        fermi_level_integrand,
        int_limit_lower,
        int_limit_upper,
        args=(
            eps,
            g_value,
        ),
    )
    return result


def energy_integrand(x: float, eps: float, g_value: float) -> float:
    """
    Integrand for Eq 80a in Brezin et al.

    Parameters
    ----------
    x : float
        The dummy integration variable, aka lambda in the paper.
    eps : float
        The scaled Fermi level.
    g_value : float
        The quartic coupling constant.

    Returns
    -------
    float
        The value of the integrand.
    """
    return (
        (1 / (3 * np.pi))
        * (2 * eps - x**2 - 2 * g_value * x**4) ** (3 / 2)
        * np.heaviside(2 * eps - x**2 - 2 * g_value * x**4, 0.5)
    )


def energy_integral(eps: float, g_value: float) -> float:
    """
    Integrand for Eq 80a in Brezin et al.
    The integration limits are determined by the constraint
    that this expression should be real.

    Parameters
    ----------
    eps : float
        The scaled Fermi level.
    g_value : float
        The quartic coupling constant.

    Returns
    -------
    float
        The value of the integral.
    """
    int_limit_upper = np.sqrt((-1 + np.sqrt(1 + 16 * g_value * eps)) / (4 * g_value))
    int_limit_lower = -int_limit_upper
    result, _ = quad(
        energy_integrand,
        int_limit_lower,
        int_limit_upper,
        args=(
            eps,
            g_value,
        ),
    )
    return eps - result


def fermi_level_eqn(eps: float, g_value: float) -> float:
    """
    Eq. 80b in Brezin et al

    Parameters
    ----------
    eps : float
        The scaled Fermi level.
    g_value : float
        The quartic coupling constant.

    Returns
    -------
    float
        The value of the integral.
    """
    return fermi_level_integral(eps, g_value) - 1


def compute_Brezin_energy(g_value: float) -> float:
    """
    The energy (divided by N^2) of the matrix model in the large-N limit
    as calculated in Eq 80a.
    Note that the conventions here are different from Han et al:
        Energy_Han(g) = 2 Energy_Brezin(g/2)

    Parameters
    ----------
    g_value : float
        The quartic coupling constant.

    Returns
    -------
    float
        The energy.
    """
    sol = root_scalar(fermi_level_eqn, args=(g_value), bracket=[0, 10], method="brentq")
    return energy_integral(eps=sol.root, g_value=g_value)


def compute_Brezin_energy_Han_conventions(g_value: float) -> float:
    return 2 * compute_Brezin_energy(g_value / 2)


if __name__ == "__main__":

    # compare against Table 3 in Brezin et al
    table_values = {
        0.01: 0.505,
        0.1: 0.542,
        0.5: 0.651,
        1: 0.740,
        50: 2.217,
        # 1000: 5.915, # for this extreme value the code fails
    }

    print(
        "Compare the values listed in Table 3 of Brezin et al against the values computed using this code:"
    )
    for g, table_val in table_values.items():
        computed_val = compute_Brezin_energy(g)
        error = np.abs(computed_val - table_val)
        print(
            f"  g = {g} | table 3 value = {table_val} | computed value = {computed_val:.03f} | error = {error:.4e}"
        )
