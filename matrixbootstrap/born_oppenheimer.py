import fire
import numpy as np
from scipy.integrate import trapezoid as trapz
from scipy.optimize import minimize
from scipy.optimize._optimize import OptimizeResult


class BornOppenheimer:
    """
    The Born-Oppenheimer model for the 2 matrix model,
    see https://doi.org/10.1103/PhysRevLett.125.041601.
    """
    def __init__(self, m=None, g=None, g2=None, g4=None):
        """
        Support initialization with either m, g (HHK conventions) or g2, g4 (my conventions).
        """
        if m is not None and g is not None and g2 is None and g4 is None:
            self.m = m
            self.g = g
            self.g2 = g2
            self.g4 = g4
            #self.g2 = m**2
            #self.g4 = 4*g**2
        elif m is None and g is None and g2 is not None and g4 is not None:
            self.g2 = g2
            self.g4 = g4
            self.m = np.sqrt(g2)
            self.g = np.sqrt(g4)/2

    def normalization_constraint(self, rho: np.ndarray, x_grid: np.ndarray) -> float:
        """
        The normalization constraint equation, integral(rho) - 1.

        Parameters
        ----------
        rho : np.ndarray
            The collective coordinates.
        x_grid : np.ndarray
            The grid of x-values to consider.

        Returns
        -------
        float
            The RHS of the constraint equation (should be zero).
        """
        integral_rho = trapz(rho, x_grid)
        return integral_rho - 1.0

    def omega_matrix(self, x_grid: np.ndarray) -> np.ndarray:
        """
        The omega(x, y) term as a vectorized function.

        Parameters
        ----------
        x_grid : np.ndaray
            The grid of x-values to consider.

        Returns
        -------
        np.ndarray
            The omega(x, y) values as a 2D numpy array.
        """
        x_i, x_j = np.meshgrid(x_grid, x_grid)
        return np.sqrt(self.m**2 + self.g**2 * (x_i - x_j) ** 2)

    def local_energy_density(self, rho: np.ndarray, x_grid: np.ndarray) -> np.ndarray:
        """
        The local energy term.

        Parameters
        ----------
        rho : np.ndaray
            The collective coordinates.
        x_grid : np.ndaray
            The grid of x-values to consider.

        Returns
        -------
        np.ndarray
            The local energy term (represented as a 1D array).
        """
        return (np.pi**2 / 3) * rho**3 + self.m**2 * x_grid**2 * rho

    def non_local_energy_term(self, rho: np.ndarray, x_grid: np.ndarray) -> np.ndarray:
        """
        Non-local energy term using NumPy vectorization.

        Parameters
        ----------
        rho : np.ndaray
            The collective coordinates.
        x_grid : np.ndaray
            The grid of x-values to consider.

        Returns
        -------
        np.ndarray
            The non-local energy term (represented as a 1D array).
        """
        # compute the omega matrix for all pairs (x_i, x_j)
        omega_mat = self.omega_matrix(x_grid)

        # compute the outer product of rho with itself
        rho_outer = np.outer(rho, rho)

        # element-wise multiply the rho_outer with omega_mat
        delta_x = x_grid[1] - x_grid[0]  # assumes a uniform grid
        non_local_energy = np.sum(rho_outer * omega_mat) * delta_x**2

        return non_local_energy

    def E_BO_discretized(self, rho: np.ndarray, x_grid: np.ndarray) -> float:
        """
        The discretized Born-Oppenheimer energy functional.

        Parameters
        ----------
        rho : np.ndaray
            The collective coordinates.
        x_grid : np.ndaray
            The grid of x-values to consider.

        Returns
        -------
        float
            The energy.
        """
        # local energy: sum over the grid points using trapz
        local_energy = trapz(
            [self.local_energy_density(rho_i, x_i) for rho_i, x_i in zip(rho, x_grid)],
            x_grid,
        )

        # non-local energy: double sum over grid points
        non_local_energy = self.non_local_energy_term(rho, x_grid)

        return local_energy + non_local_energy

    def solve(self, x_grid: np.ndarray) -> OptimizeResult:
        """
        Find the minimium allowable energy of the Born-Oppenheimer functional.

        Parameters
        ----------
        x_grid : np.ndarray
            The grid of x-values to consider.

        Returns
        -------
        OptimizeResult
            The optimization result.
        """

        # bounds for rho (rho >= 0)
        bounds = [(0, None)] * len(x_grid)  # rho(x) >= 0 for all x

        # initial guess for rho (e.g., uniform distribution)
        x_min, x_max = x_grid[0], x_grid[-1]
        initial_rho = np.ones_like(x_grid) * 1 / (x_max - x_min)

        # convert to my notation if necessary by adding a factor of 1/2
        if self.g2 is not None:
            energy_func = lambda rho, grid: self.E_BO_discretized(rho, grid) / 2
        else:
            energy_func = self.E_BO_discretized

        # Minimize the energy functional E_BO with the normalization constraint
        result = minimize(
            energy_func,
            initial_rho,
            args=(x_grid),
            method="SLSQP",
            bounds=bounds,
            constraints={
                "type": "eq",
                "fun": lambda rho: self.normalization_constraint(rho, x_grid),
            },
        )

        return result


def main(m: float=1, g: float=1, npoints: int=100):

    # define the x grid
    x_min, x_max = -3, 3
    x_grid = np.linspace(x_min, x_max, npoints)

    # set-up the BO model (HHK conventions)
    born_oppenheimer = BornOppenheimer(m=m, g=g)
    result = born_oppenheimer.solve(x_grid=x_grid)
    optimal_energy_HHK = result.fun

    # set-up the BO model (my conventions)
    g2 = m**2
    g4 = 4 * g**2
    born_oppenheimer = BornOppenheimer(g2=g2, g4=g4)
    result = born_oppenheimer.solve(x_grid=x_grid)
    optimal_energy_me = result.fun

    print(f"Minimum BO energy for m={m}, g={g}: E={optimal_energy_HHK:.4f} (HHK conventions)")
    print(f"Minimum BO energy for g2={born_oppenheimer.g2}, g4={born_oppenheimer.g4}: E={optimal_energy_me:.4f} (my conventions)")
    print(f"The results should be off by a factor of two: diff={(optimal_energy_me - optimal_energy_HHK / 2):.4e}.")

    # massless case
    g2 = 0
    g4 = 1
    born_oppenheimer = BornOppenheimer(g2=g2, g4=g4)
    result = born_oppenheimer.solve(x_grid=x_grid)
    optimal_energy_me = result.fun
    print(f"\nMinimum BO energy for g2={born_oppenheimer.g2}, g4={born_oppenheimer.g4}: E={optimal_energy_me:.4f} (my conventions)")


if __name__ == "__main__":
    fire.Fire(main)