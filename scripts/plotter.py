import inspect
import json
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cycler import cycler
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from matrixbootstrap.born_oppenheimer import BornOppenheimer
from matrixbootstrap.brezin import compute_Brezin_energy

plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"
plt.rcParams["xtick.major.size"] = 5.0
plt.rcParams["xtick.minor.size"] = 3.0
plt.rcParams["ytick.major.size"] = 5.0
plt.rcParams["ytick.minor.size"] = 3.0
plt.rcParams["lines.linewidth"] = 2
plt.rc("font", family="serif", size=16)
matplotlib.rc("text", usetex=True)
matplotlib.rc("legend", fontsize=16)
matplotlib.rcParams["axes.prop_cycle"] = cycler(
    color=["#E24A33", "#348ABD", "#988ED5", "#777777", "#FBC15E", "#8EBA42", "#FFB5B8"]
    # color=['#9e0142','#d53e4f','#f46d43','#fdae61','#fee08b','#e6f598','#abdda4','#66c2a5','#3288bd','#5e4fa2'][::-1]
)
matplotlib.rcParams.update(
    {"axes.grid": False, "grid.alpha": 0.75, "grid.linewidth": 0.5}
)
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

np.set_printoptions(linewidth=160)


def load_data(datadir, names_in_filename, tol=1e-6, delete_param=True):

    # grab the data files
    names_in_filename.append(".json")
    files = []
    for f in os.listdir(datadir):
        if all(name in f for name in names_in_filename):
            files.append(f)
    print(f"number of files found: {len(files)}")

    if len(files) == 0:
        return

    # build dataframe
    data = []
    for file in files:
        with open(f"{datadir}/{file}") as f:
            result = json.load(f)
        if delete_param:
            del result["param"]  # remove param vector
        # result["energy"] = float(file.split('_')[1][:-5]) # add g4 coupling
        result["filename"] = file

        if (
            (np.abs(result["min_bootstrap_eigenvalue"]) < tol)
            & (np.abs(result["violation_of_linear_constraints"]) < tol)
            & (np.abs(result["quad_constraint_violation_norm"]) < tol)
        ):
            data.append(result)

    df = pd.DataFrame(data)
    if len(df) == 0:
        return df.copy()

    df.sort_values("energy", inplace=True)
    max_violation_linear = df["violation_of_linear_constraints"].max()
    max_violation_quadratic = df["max_quad_constraint_violation"].max()
    max_violation_PSD = df["min_bootstrap_eigenvalue"].abs().max()

    print(f"number of loaded data points: {len(data)}")
    print(f"max violation of linear constraints:{max_violation_linear:.4e}")
    print(f"max violation of PSD constraints:{max_violation_PSD:.4e}")
    print(f"max violation of quadratic constraints:{max_violation_quadratic:.4e}\n")

    return df.copy()


def add_extension(filename, extension):
    if extension not in filename:
        filename = f"{filename}.{extension}"
    return filename


def make_figure_efffective_potential(extension="pdf"):

    this_function_name = inspect.currentframe().f_code.co_name
    print("\n=========================================")
    print(f"running function: {this_function_name}\n")

    roots = []
    nu_values = np.linspace(1, 10, 250)
    for nu in nu_values:
        coeffs = np.array([2, -3, 1, 8 / (3 * nu**3)])
        roots.append(np.roots(coeffs))
    roots = np.asarray(roots)

    fig, ax = plt.subplots(figsize=(6, 5))
    labels = ["SUSY vaccum", "non-SUSY vacuum", "trivial vacuum"]
    for root_idx in [0, 1, 2]:
        ax.plot(
            [nu for idx, nu in enumerate(nu_values) if roots[idx, root_idx].imag == 0],
            [
                roots[idx, root_idx].real
                for idx, nu in enumerate(nu_values)
                if roots[idx, root_idx].imag == 0
            ],
            label=labels[root_idx],
        )

    phi_min = 0.5 - np.sqrt(1 / 6)
    phi_max = 0.5 + np.sqrt(1 / 6)
    ax.axhline(phi_min, color="k", linestyle="-", linewidth=1)
    ax.axhline(phi_max, color="k", linestyle="-", linewidth=1)
    ax.fill_between(nu_values, phi_min, phi_max, color="gray", alpha=0.25)
    ax.set_xlim([nu_values[0], nu_values[-1]])
    ax.axvline(786 ** (1 / 6), color="k", linestyle="--", linewidth=1)
    ax.text(0.5, 0.55, r"Unstable", fontsize=14, transform=ax.transAxes)

    ax.set_xlabel(r"$\nu/\lambda^{1/3}$")
    ax.set_ylabel(r"$\phi_{cl}$")
    # ax.set_xscale('log')
    ax.legend(fontsize=14)

    plt.tight_layout()
    filename = "figures/" + "_".join(this_function_name.split("_")[2:])
    plt.savefig(add_extension(filename=filename, extension=extension))
    plt.show()


def make_figure_HHK_fig_3a(extension="pdf"):

    this_function_name = inspect.currentframe().f_code.co_name
    print("\n=========================================")
    print(f"running function: {this_function_name}\n")

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.grid(True)

    # add HHK extracted data
    for L in [4, 3]:
        df_han_hartnoll_fig3a = pd.read_csv(
            f"data/data_from_papers/han_hartnoll_data/TwoMatrixFig3a_L_{L}.csv",
            header=None,
            names=["x", "y"],
        )
        x = df_han_hartnoll_fig3a["x"]
        y = df_han_hartnoll_fig3a["y"]
        ax.scatter(x, y, label=f"Han et al, L={L}", edgecolor="k", zorder=10, s=30)

    # add Born-Oppeheimer curve (HHK conventions)
    x_min, x_max = -4, 4
    npoints = 200
    x_grid = np.linspace(x_min, x_max, npoints)
    min_energy = []
    for g in np.sqrt(x):
        born_oppenheimer = BornOppenheimer(m=1, g=g)
        result = born_oppenheimer.solve(x_grid=x_grid)
        min_energy.append(result.fun)
    min_energy = np.asarray(min_energy)
    ax.plot(x, min_energy, label="Born-Oppenheimer", color="k")
    ax.set_xlabel(r"$N g^2$")
    ax.set_ylabel(r"$E_0/N$")

    ax.legend()

    plt.tight_layout()
    filename = "figures/" + "_".join(this_function_name.split("_")[2:])
    plt.savefig(add_extension(filename=filename, extension=extension))
    plt.show()


def make_figure_regularization_scan_one_matrix(extension="pdf"):

    this_function_name = inspect.currentframe().f_code.co_name
    print("\n=========================================")
    print(f"running function: {this_function_name}\n")

    # load data
    data = []
    for L in [3, 4]:
        path = f"data/OneMatrix_L_{L}_reg_scan"
        files = [f for f in os.listdir(path) if ".json" in f]
        print(f"L={L}, number of data points found: {len(files)}")

        for file in files:
            with open(f"{path}/{file}") as f:
                result = json.load(f)
            del result["param"]  # remove param vector
            if result["max_quad_constraint_violation"] < 1e-2:
                result["L"] = int(L)
                result["reg"] = float(file.split("_")[3][:-5])
                data.append(result)
    df = pd.DataFrame(data)
    df.sort_values("energy", inplace=True)

    # plot
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    for idx, L in enumerate([3, 4]):

        ax[0].scatter(
            df[df["L"] == L]["reg"],
            np.abs(df[df["L"] == L]["energy"] - compute_Brezin_energy(g_value=1 / 4)),
            label=f"L={L}",
            edgecolor="k",
            zorder=10,
            s=30,
        )
        ax[0].set_xlabel("regularization")
        ax[0].set_ylabel("abs(energy - Brezin energy)")
        ax[0].set_xscale("log")
        ax[0].set_yscale("log")
        ax[0].legend(fontsize=10)

        ax[1].scatter(
            df[df["L"] == L]["reg"],
            np.abs(df[df["L"] == L]["min_bootstrap_eigenvalue"]),
            label=f"min bootstrap eigenvalue, L={L}",
            edgecolor="k",
            zorder=10,
            s=30,
        )
        ax[1].scatter(
            df[df["L"] == L]["reg"],
            np.abs(df[df["L"] == L]["max_quad_constraint_violation"]),
            label=f"max_quad_constraint_violation L={L}",
            edgecolor="k",
            zorder=10,
            s=30,
        )
        ax[1].scatter(
            df[df["L"] == L]["reg"],
            np.abs(df[df["L"] == L]["violation_of_linear_constraints"]),
            label=f"violation_of_linear_constraints L={L}",
            edgecolor="k",
            zorder=10,
            s=30,
        )
        ax[1].set_xlabel("regularization")
        ax[1].set_ylabel("constraint violation")
        ax[1].set_xscale("log")
        ax[1].set_yscale("log")
        ax[1].legend(fontsize=10)
    plt.suptitle(r"OneMatrix model $g2=1$, $g4=1$, $g6=0$, minimizing energy")

    plt.tight_layout()
    filename = "figures/" + "_".join(this_function_name.split("_")[2:])
    plt.savefig(add_extension(filename=filename, extension=extension))
    plt.show()


def make_figure_regularization_scan_two_matrix_massless(extension="pdf"):

    this_function_name = inspect.currentframe().f_code.co_name
    print("=========================================")
    print(f"running function: {this_function_name}\n")

    # load the data
    L = 3
    path = "data/TwoMatrix_L_3_g2_0_g4_1_reg_scan"
    data = []
    files = [f for f in os.listdir(path) if ".json" in f]
    print(f"Number of data points found: {len(files)}")
    for file in files:
        with open(f"{path}/{file}") as f:
            result = json.load(f)
        del result["param"]  # remove param vector
        if result["max_quad_constraint_violation"] < 1e-2:
            result["reg"] = float(file.split("_")[3][:-5])
            data.append(result)
    df = pd.DataFrame(data)
    df.sort_values("energy", inplace=True)

    # plot
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].scatter(
        df["reg"], df["energy"], label=f"L={L}", edgecolor="k", zorder=10, s=30
    )

    energy_asymptotic = df["energy"].max()
    ax[0].axhline(
        energy_asymptotic, color="k", linestyle="--", label="asymptotic energy"
    )
    ax[0].text(
        1e-7,
        energy_asymptotic,
        f"{energy_asymptotic:.4f}",
        fontsize=14,
        verticalalignment="bottom",
    )

    ax[1].scatter(
        df["reg"],
        np.abs(df["min_bootstrap_eigenvalue"]),
        label=f"min bootstrap eigenvalue, L={L}",
        edgecolor="k",
        zorder=10,
        s=30,
    )
    ax[1].scatter(
        df["reg"],
        np.abs(df["max_quad_constraint_violation"]),
        label=f"max_quad_constraint_violation L={L}",
        edgecolor="k",
        zorder=10,
        s=30,
    )
    ax[1].scatter(
        df["reg"],
        np.abs(df["violation_of_linear_constraints"]),
        label=f"violation_of_linear_constraints L={L}",
        edgecolor="k",
        zorder=10,
        s=30,
    )

    # add Born-Oppeheimer curve
    x_min, x_max = -4, 4
    npoints = 200
    x_grid = np.linspace(x_min, x_max, npoints)
    born_oppenheimer = BornOppenheimer(g2=0, g4=1)
    result = born_oppenheimer.solve(x_grid=x_grid)
    bo_energy = result.fun
    ax[0].axhline(bo_energy, color=colors[2], linestyle="--", label="Born-Oppenheimer")

    # plot settings
    ax[0].set_xlabel("regularization")
    ax[0].set_ylabel("energy")
    ax[0].set_xscale("log")
    ax[0].legend(fontsize=10)
    ax[1].set_xlabel("regularization")
    ax[1].set_ylabel("constraint violation")
    ax[1].set_xscale("log")
    ax[1].set_yscale("log")
    ax[1].legend(fontsize=10)
    plt.suptitle(r"TwoMatrix, SO(2) sym, $g2=0$, $g4=1$, minimizing energy")

    plt.tight_layout()
    filename = "figures/" + "_".join(this_function_name.split("_")[2:])
    plt.savefig(add_extension(filename=filename, extension=extension))
    plt.show()


def make_figure_regularization_scan_two_matrix_massive(extension="pdf"):

    this_function_name = inspect.currentframe().f_code.co_name
    print("\n=========================================")
    print(f"running function: {this_function_name}\n")

    # load the data
    L = 3
    path = "data/TwoMatrix_L_3_g2_1_g4_1_reg_scan"
    data = []
    files = [f for f in os.listdir(path) if ".json" in f]
    print(f"Number of data points found: {len(files)}")
    for file in files:
        with open(f"{path}/{file}") as f:
            result = json.load(f)
        del result["param"]  # remove param vector
        if result["max_quad_constraint_violation"] < 1e-2:
            result["reg"] = float(file.split("_")[3][:-5])
            data.append(result)
    df = pd.DataFrame(data)
    df.sort_values("energy", inplace=True)

    # plot
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].scatter(
        df["reg"], df["energy"], label=f"L={L}", edgecolor="k", zorder=10, s=30
    )

    energy_asymptotic = df["energy"].max()
    ax[0].axhline(
        energy_asymptotic, color="k", linestyle="--", label="asymptotic energy"
    )
    ax[0].text(
        1e-7,
        energy_asymptotic,
        f"{energy_asymptotic:.4f}",
        fontsize=14,
        verticalalignment="bottom",
    )

    ax[1].scatter(
        df["reg"],
        np.abs(df["min_bootstrap_eigenvalue"]),
        label=f"min bootstrap eigenvalue, L={L}",
        edgecolor="k",
        zorder=10,
        s=30,
    )
    ax[1].scatter(
        df["reg"],
        np.abs(df["max_quad_constraint_violation"]),
        label=f"max_quad_constraint_violation L={L}",
        edgecolor="k",
        zorder=10,
        s=30,
    )
    ax[1].scatter(
        df["reg"],
        np.abs(df["violation_of_linear_constraints"]),
        label=f"violation_of_linear_constraints L={L}",
        edgecolor="k",
        zorder=10,
        s=30,
    )

    # add Born-Oppeheimer curve
    x_min, x_max = -4, 4
    npoints = 200
    x_grid = np.linspace(x_min, x_max, npoints)
    born_oppenheimer = BornOppenheimer(g2=1, g4=1)
    result = born_oppenheimer.solve(x_grid=x_grid)
    bo_energy = result.fun
    ax[0].axhline(bo_energy, color=colors[2], linestyle="--", label="Born-Oppenheimer")

    # plot settings
    ax[0].set_xlabel("regularization")
    ax[0].set_ylabel("energy")
    ax[0].set_xscale("log")
    ax[0].legend(fontsize=10)
    ax[1].set_xlabel("regularization")
    ax[1].set_ylabel("constraint violation")
    ax[1].set_xscale("log")
    ax[1].set_yscale("log")
    ax[1].legend(fontsize=10)
    plt.suptitle(r"TwoMatrix, SO(2) sym $g2=1$, $g4=1$, minimizing energy")

    plt.tight_layout()
    filename = "figures/" + "_".join(this_function_name.split("_")[2:])
    plt.savefig(add_extension(filename=filename, extension=extension))
    plt.show()


def make_figure_two_matrix_mass_vs_E(extension="pdf", reg=1e-4):

    this_function_name = inspect.currentframe().f_code.co_name
    print("\n=========================================")
    print(f"running function: {this_function_name}, reg={reg}\n")

    # load the data
    L = 3
    path = f"data/TwoMatrix_L_{L}_symmetric_min_energy"
    data = []
    files = [f for f in os.listdir(path) if ".json" in f]
    # print(f"Number of data points found: {len(files)}")
    for file in files:
        with open(f"{path}/{file}") as f:
            result = json.load(f)
        del result["param"]  # remove param vector
        if result["max_quad_constraint_violation"] < 1e-5:
            result["g2"] = float(file.split("_")[1])
            result["reg"] = float(file.split("_")[5][:-5])
            if reg is not None:
                if result["reg"] == reg:
                    data.append(result)
            else:
                data.append(result)
    print(f"Number of data points found: {len(data)}")
    df = pd.DataFrame(data)
    df.sort_values("energy", inplace=True)

    # make the plot (for each reg value)
    fig, ax = plt.subplots(figsize=(6, 5))

    if reg is not None:
        ax.scatter(df["g2"], df["energy"], label=f"L={L}", edgecolor="k", s=30)
    else:
        for tmp_reg in df["reg"].unique():
            df_reg = df[df["reg"] == tmp_reg]
            ax.scatter(
                df_reg["g2"],
                df_reg["energy"],
                label=f"L={L}, reg={tmp_reg:.2e}",
                edgecolor="k",
                s=30,
            )

    # add min energy
    g2_min = df["g2"].iloc[0]
    e_min = df["energy"].iloc[0]
    str = f"g2 = {g2_min:.4f}, E={e_min:.4f}"
    ax.axhline(e_min, color="k", linestyle="--")
    ax.text(0.5, 0.2, str, transform=ax.transAxes, fontsize=16, verticalalignment="top")

    # add Born-Oppenheimer curve
    x_min, x_max = -4, 4
    npoints = 100
    x_grid = np.linspace(x_min, x_max, npoints)
    min_energy = []
    g2_values = [
        df["g2"].iloc[i] for i in range(0, len(df), 15)
    ]  # select every 15th value to speed up the plot
    for g2 in g2_values:
        m = np.sqrt(g2)  # their mass is my sqrt(g2)
        g = 1 / 2  # their g = 1/2 corresponds to g4 = 1 in my notation
        born_oppenheimer = BornOppenheimer(m=m, g=g)
        result = born_oppenheimer.solve(x_grid=x_grid)
        min_energy.append(result.fun)
    min_energy = (
        np.asarray(min_energy) / 2
    )  # divide by 2 to match the energy definition in HHK paper
    ax.plot(g2_values, min_energy, label="Born-Oppenheimer lower-bound", color="k")

    # plot settings
    ax.set_xlabel(r"$g_2$")
    ax.set_ylabel("energy")
    # ax.set_xscale('log')
    # ax.set_yscale('log')
    ax.legend(fontsize=10)

    if reg is not None:
        ax.set_title(f"TwoMatrix, SO(2) sym, min. energy, reg={reg:.2e}")
        filename = (
            "figures/" + "_".join(this_function_name.split("_")[2:]) + f"_reg_{reg:.2e}"
        )
    else:
        ax.set_title("TwoMatrix, SO(2) sym, min. energy")
        filename = "figures/" + "_".join(this_function_name.split("_")[2:]) + "_all_reg"

    plt.tight_layout()

    plt.savefig(add_extension(filename=filename, extension=extension))
    plt.show()


def make_figure_x2_bound_two_matrix(extension="pdf", reg=1e-4):

    this_function_name = inspect.currentframe().f_code.co_name
    print("\n=========================================")
    print(f"running function: {this_function_name}\n")

    # load the data
    L = 3
    path = f"data/TwoMatrix_L_{L}_symmetric_energy_fixed_g2_1.0_reg_{reg:.2e}"
    print(f"reg = {reg:.2e}")

    fig, ax = plt.subplots(figsize=(6, 5))
    ax_inset = inset_axes(ax, width="35%", height="35%", loc="center right")

    # lower bound
    df = load_data(path, names_in_filename=["op_to_min_x_2"], tol=1e-5)
    ax.scatter(
        df["energy"],
        df["x_2"],
        edgecolor="k",
        zorder=10,
        s=30,
        label=f"lower bound, L={L}",
        color=colors[0],
    )
    ax.plot(df["energy"], df["x_2"], color=colors[0])
    ax_inset.scatter(
        df["energy"], df["x_2"], edgecolor="k", zorder=10, s=30, color=colors[0]
    )
    ax_inset.plot(df["energy"], df["x_2"], color=colors[0])
    y_lower_bound = df["x_2"].copy()

    # upper bound
    df = load_data(path, names_in_filename=["op_to_min_neg_x_2"], tol=1e-5)
    ax.plot(df["energy"], df["x_2"], color=colors[0])
    ax.scatter(
        df["energy"],
        df["x_2"],
        edgecolor="k",
        zorder=10,
        s=30,
        label=f"upper bound, L={L}",
        color=colors[0],
    )
    ax_inset.scatter(
        df["energy"], df["x_2"], edgecolor="k", zorder=10, s=30, color=colors[0]
    )
    ax_inset.plot(df["energy"], df["x_2"], color=colors[0])
    y_upper_bound = df["x_2"].copy()

    ax.set_xlabel(r"$\lambda^{-1/3} N^{-2} E$")
    ax.set_ylabel(r"$\lambda^{-2/3} N^{-2}$ Tr$(X^2)$")
    ax.set_title(r"TwoMatrix, SO(2) sym, $g_2=1$, $g_4=1$" + f" reg={reg:.2e}")
    min_energy = df["energy"].min()
    ax.text(
        0.1,
        0.95,
        f"Min energy={min_energy:.4f}",
        transform=ax.transAxes,
        fontsize=16,
        verticalalignment="top",
    )
    ax.axvline(min_energy, color="k", linestyle="--")

    # identify the cusp
    # difference = np.abs(y_upper_bound - y_lower_bound)
    # energy_cusp = df["energy"][difference.idxmin()]
    energy_cusp = 1.099
    ax.axvline(energy_cusp, color="r", linestyle="--")
    ax_inset.axvline(energy_cusp, color="r", linestyle="--")
    ax.text(
        0.1,
        0.85,
        f"Cusp energy={energy_cusp:.4f}",
        transform=ax.transAxes,
        fontsize=16,
        verticalalignment="top",
    )

    ax_inset.set_xlim(0.9, 1.5)  # Adjust as needed
    ax_inset.set_ylim(0.8, 1.2)  # Adjust as needed

    plt.tight_layout()
    filename = (
        "figures/" + "_".join(this_function_name.split("_")[2:]) + f" reg={reg:.2e}"
    )
    plt.savefig(add_extension(filename=filename, extension=extension))
    plt.show()


def make_figure_x2_bound_two_matrix_all_reg(extension="pdf"):

    this_function_name = inspect.currentframe().f_code.co_name
    print("\n=========================================")
    print(f"running function: {this_function_name}\n")

    fig, ax = plt.subplots(figsize=(6, 5))

    L = 3
    for reg in [1e-4, 1e-1, 1e1]:

        # load the data
        path = f"data/TwoMatrix_L_{L}_symmetric_energy_fixed_g2_1.0_reg_{reg:.2e}"

        # lower bound
        df = load_data(path, names_in_filename=["op_to_min_x_2"], tol=1e-5)
        ax.scatter(
            df["energy"],
            df["x_2"],
            edgecolor="k",
            zorder=10,
            s=30,
            label=f"lower bound, reg={reg:.2e}",
            color=colors[0],
        )
        ax.plot(df["energy"], df["x_2"], color=colors[0])

    ax.set_xlabel(r"$\lambda^{-1/3} N^{-2} E$")
    ax.set_ylabel(r"$\lambda^{-2/3} N^{-2}$ Tr$(X^2)$")
    ax.set_title(r"TwoMatrix, SO(2) sym, $g_2=1$, $g_4=1$")
    ax.legend()

    plt.tight_layout()
    filename = "figures/" + "_".join(this_function_name.split("_")[2:])
    plt.savefig(add_extension(filename=filename, extension=extension))
    plt.show()


def make_figure_regularization_scan_bfss(extension="pdf"):

    this_function_name = inspect.currentframe().f_code.co_name
    print("\n=========================================")
    print(f"running function: {this_function_name}\n")

    # load the data
    L = 3
    path = "data/MiniBFSS_L_3_symmetric_reg_scan"
    data = []
    files = [f for f in os.listdir(path) if ".json" in f]
    print(f"Number of data points found: {len(files)}")
    for file in files:
        with open(f"{path}/{file}") as f:
            result = json.load(f)
        del result["param"]  # remove param vector
        if result["max_quad_constraint_violation"] < 1e-2:
            result["reg"] = float(file.split("_")[3][:-5])
            data.append(result)
    df = pd.DataFrame(data)
    df.sort_values("energy", inplace=True)

    # plot
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].scatter(
        df["reg"], df["energy"], label=f"L={L}", edgecolor="k", zorder=10, s=30
    )
    ax[0].set_xlabel("regularization")
    ax[0].set_ylabel("energy")
    ax[0].set_xscale("log")
    ax[0].legend(fontsize=10)

    energy_asymptotic = df["energy"].max()
    ax[0].axhline(
        energy_asymptotic, color="k", linestyle="--", label="asymptotic energy"
    )
    ax[0].text(
        1e-7,
        energy_asymptotic,
        f"{energy_asymptotic:.4f}",
        fontsize=14,
        verticalalignment="bottom",
    )

    ax[1].scatter(
        df["reg"],
        np.abs(df["min_bootstrap_eigenvalue"]),
        label=f"min bootstrap eigenvalue, L={L}",
        edgecolor="k",
        zorder=10,
        s=30,
    )
    ax[1].scatter(
        df["reg"],
        np.abs(df["max_quad_constraint_violation"]),
        label=f"max_quad_constraint_violation L={L}",
        edgecolor="k",
        zorder=10,
        s=30,
    )
    ax[1].scatter(
        df["reg"],
        np.abs(df["violation_of_linear_constraints"]),
        label=f"violation_of_linear_constraints L={L}",
        edgecolor="k",
        zorder=10,
        s=30,
    )
    ax[1].set_xlabel("regularization")
    ax[1].set_ylabel("constraint violation")
    ax[1].set_xscale("log")
    ax[1].set_yscale("log")
    ax[1].legend(fontsize=10)
    plt.suptitle("MiniBFSS, SO(3) sym, minimizing energy")

    plt.tight_layout()
    filename = "figures/" + "_".join(this_function_name.split("_")[2:])
    plt.savefig(add_extension(filename=filename, extension=extension))
    plt.show()


def make_figure_regularization_scan_bmn(extension="pdf"):

    this_function_name = inspect.currentframe().f_code.co_name
    print("\n=========================================")
    print(f"running function: {this_function_name}\n")

    # load the data
    L = 3
    path = "data/MiniBMN_L_3_symmetric_nu_1.0_lamb_1_reg_scan"
    data = []
    files = [f for f in os.listdir(path) if ".json" in f]
    print(f"Number of data points found: {len(files)}")
    for file in files:
        with open(f"{path}/{file}") as f:
            result = json.load(f)
        del result["param"]  # remove param vector
        if result["max_quad_constraint_violation"] < 1e-2:
            result["reg"] = float(file.split("_")[3][:-5])
            data.append(result)
    df = pd.DataFrame(data)
    df.sort_values("energy", inplace=True)

    # plot
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].scatter(
        df["reg"], df["energy"], label=f"L={L}", edgecolor="k", zorder=10, s=30
    )
    ax[0].set_xlabel("regularization")
    ax[0].set_ylabel("energy")
    ax[0].set_xscale("log")
    ax[0].legend(fontsize=10)

    energy_asymptotic = df["energy"].max()
    ax[0].axhline(
        energy_asymptotic, color="k", linestyle="--", label="asymptotic energy"
    )
    ax[0].text(
        1e-7,
        energy_asymptotic,
        f"{energy_asymptotic:.4f}",
        fontsize=14,
        verticalalignment="bottom",
    )

    ax[1].scatter(
        df["reg"],
        np.abs(df["min_bootstrap_eigenvalue"]),
        label=f"min bootstrap eigenvalue, L={L}",
        edgecolor="k",
        zorder=10,
        s=30,
    )
    ax[1].scatter(
        df["reg"],
        np.abs(df["max_quad_constraint_violation"]),
        label=f"max_quad_constraint_violation L={L}",
        edgecolor="k",
        zorder=10,
        s=30,
    )
    ax[1].scatter(
        df["reg"],
        np.abs(df["violation_of_linear_constraints"]),
        label=f"violation_of_linear_constraints L={L}",
        edgecolor="k",
        zorder=10,
        s=30,
    )
    ax[1].set_xlabel("regularization")
    ax[1].set_ylabel("constraint violation")
    ax[1].set_xscale("log")
    ax[1].set_yscale("log")
    ax[1].legend(fontsize=10)
    plt.suptitle(r"MiniBMN, SO(3) sym, $\nu=1.0$, $\lambda=1$, minimizing energy")

    plt.tight_layout()
    filename = "figures/" + "_".join(this_function_name.split("_")[2:])
    plt.savefig(add_extension(filename=filename, extension=extension))
    plt.show()


def make_figure_bmn_nu_vs_E(extension="pdf", reg=1e-4):

    this_function_name = inspect.currentframe().f_code.co_name
    print("\n=========================================")
    print(f"running function: {this_function_name}\n")

    # load the data
    L = 3
    path = f"data/MiniBMN_L_{L}_symmetric_min_energy"
    data = []
    files = [f for f in os.listdir(path) if ".json" in f]
    # print(f"Number of data points found: {len(files)}")
    for file in files:
        with open(f"{path}/{file}") as f:
            result = json.load(f)
        del result["param"]  # remove param vector
        if result["max_quad_constraint_violation"] < 1e-5:
            result["nu"] = float(file.split("_")[1])
            result["reg"] = float(file.split("_")[5][:-5])
            if result["reg"] == reg:
                data.append(result)
    print(f"Number of data points found: {len(data)}")
    df = pd.DataFrame(data)
    df.sort_values("energy", inplace=True)

    # make the plot
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(df["nu"], df["energy"], label=f"L={L}", edgecolor="k", zorder=10, s=30)

    # show min nu value
    nu_min = df["nu"].iloc[0]
    e_min = df["energy"].iloc[0]
    str = rf"$\nu$ = {nu_min:.4f}, E={e_min:.4f}"
    ax.axhline(e_min, color="k", linestyle="--")
    ax.text(
        0.1, 0.95, str, transform=ax.transAxes, fontsize=16, verticalalignment="top"
    )

    # show critical nu value
    ax.axvline((786) ** (1 / 6), color="k", linestyle="--", label="critical nu")

    # plot
    ax.set_xlabel(r"$\nu$")
    ax.set_ylabel("energy")
    ax.legend(fontsize=10)
    ax.set_title(f"MiniBMN, SO(3) sym, min. energy, reg={reg:.2e}")

    plt.tight_layout()
    filename = (
        "figures/" + "_".join(this_function_name.split("_")[2:]) + f"_reg_{reg:.2e}"
    )
    plt.savefig(add_extension(filename=filename, extension=extension))
    plt.show()


def make_figure_x2_bound_bmn(extension="pdf", reg=1e-4):

    this_function_name = inspect.currentframe().f_code.co_name
    print("\n=========================================")
    print(f"running function: {this_function_name}\n")

    # load the data
    L = 3
    path = f"data/MiniBMN_L_{L}_symmetric"

    # I didn't add reg to the path, so I have to enter it manually
    print(f"reg = {reg:.2e}")

    fig, ax = plt.subplots(figsize=(6, 5))

    # lower bound
    df = load_data(path, names_in_filename=["operator_to_minimize_x_2"], tol=1e-5)
    ax.scatter(
        df["energy"],
        df["x_2"],
        edgecolor="k",
        zorder=10,
        s=30,
        label=f"lower bound, L={L}",
        color=colors[0],
    )
    ax.plot(df["energy"], df["x_2"], color=colors[0])

    # upper bound
    df = load_data(path, names_in_filename=["operator_to_minimize_neg_x_2"], tol=1e-5)
    ax.plot(df["energy"], df["x_2"], color=colors[0])
    ax.scatter(
        df["energy"],
        df["x_2"],
        edgecolor="k",
        zorder=10,
        s=30,
        label=f"upper bound, L={L}",
        color=colors[0],
    )

    ax.set_xlabel(r"$\lambda^{-1/3} N^{-2} E$")
    ax.set_ylabel(r"$\lambda^{-2/3} N^{-2}$ Tr$(X^2)$")
    ax.set_title(r"MiniBFSS, SO(3) sym, $\nu=1$, $\lambda=1$" + f" reg={reg:.2e}")
    min_energy = df["energy"].min()
    ax.text(
        0.1,
        0.95,
        f"Min energy={min_energy:.4f}",
        transform=ax.transAxes,
        fontsize=16,
        verticalalignment="top",
    )
    ax.axvline(min_energy, color="k", linestyle="--")

    # identify the cusp
    # difference = np.abs(y_upper_bound - y_lower_bound)
    # energy_cusp = df["energy"][difference.idxmin()]
    energy_cusp = 1.099
    ax.axvline(energy_cusp, color="r", linestyle="--")

    plt.tight_layout()
    filename = (
        "figures/" + "_".join(this_function_name.split("_")[2:]) + f" reg={reg:.2e}"
    )
    plt.savefig(add_extension(filename=filename, extension=extension))
    plt.show()


if __name__ == "__main__":

    extension = "png"

    make_figure_efffective_potential(extension)

    """
    ## HHK fig 3a
    make_figure_HHK_fig_3a(extension)

    ## one matrix
    make_figure_regularization_scan_one_matrix(extension)

    ## two matrix
    make_figure_regularization_scan_two_matrix_massless(extension)
    make_figure_regularization_scan_two_matrix_massive(extension)

    for reg in [1e-4, 1e-1, 1e1]:
        make_figure_x2_bound_two_matrix(extension, reg)

    for reg in [None, 1e-4, 1e-3, 1e-1, 1e0]:
        make_figure_two_matrix_mass_vs_E(extension, reg)

    ## bfss
    make_figure_regularization_scan_bfss(extension)

    ## bmn
    make_figure_regularization_scan_bmn(extension)

    for reg in [1e-1, 1e-3, 1e-4]:
        make_figure_bmn_nu_vs_E(extension, reg)

    make_figure_x2_bound_bmn(extension, reg=1e-1)

    for reg in [None, 1e-4, 1e-3, 1e-1, 1e0]:
        make_figure_two_matrix_mass_vs_E(extension, reg)
    """
