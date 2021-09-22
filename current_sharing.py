"""Simple model of current sharing in a composite conductor, i.e.
a superconductor surrounded by a metal matrix."""

import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt

### Superconductor properties ###
# index [units: dimensionless]
n = 15
# Field for defining critical current [units: volt meter^-1]
E_c = 1e-4
# Critical current density [units: amp meter^-2]
J_c = 100 / (4e-3 * 1e-6)

### Matrix material properties
# Resistivity [units: ohm meter]
# Resistivity of copper at 20 K, https://srd.nist.gov/JPCRD/jpcrd155.pdf
rho_m = 0.00280e-8


def solve_current_sharing(I_t, A_s, A_m):
    """Solve the system of equations:
    I_t = J_s A_s + J_m A_m
    E = E_c (J_s / J_c)^n = rho_m J_m

    and return (E, J_s, J_m)
    """
    def voltage_balance(j_ratio):
        """Solve with fsolve
        j_ratio = J_s / J_c
        """
        error = (-E_c * (j_ratio)**n
                 + rho_m / A_m * (I_t - J_c * A_s * j_ratio))
        return error
    result = optimize.root_scalar(
        voltage_balance, bracket=[0., 5.], method='toms748')
    j_ratio = result.root
    J_s = j_ratio * J_c
    E = E_c * (j_ratio)**n
    J_m = E / rho_m
    return E, J_s, J_m


def main():
    # Superconductor area [units: meter^2]
    A_s = 4e-3 * 1e-6
    # Matrix area [units: meter^2]
    A_m = 4e-3 * 60e-6
    # Critical current [units: amp]
    I_c = J_c * A_s

    # Ratios of I_t / I_c to examine
    i_ratio = np.linspace(0.01, 4.)
    # Electric field [units: volt meter^-1]
    E = np.zeros(len(i_ratio))
    # Superconductor current density [units: amp meter^-2]
    J_s = np.zeros(len(i_ratio))
    # Matrix current density [units: amp meter^-2]
    J_m = np.zeros(len(i_ratio))
    # Superconductor current [units: amp]
    I_s = np.zeros(len(i_ratio))
    # Matrix current [units: amp]
    I_m = np.zeros(len(i_ratio))

    for k in range(len(i_ratio)):
        I_t = i_ratio[k] * I_c
        E[k], J_s[k], J_m[k] = solve_current_sharing(I_t, A_s, A_m)
    I_s = J_s * A_s
    I_m = J_m * A_m

    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(6, 8), sharex=True)
    ax_I, ax_E, ax_power = axes

    ax_I.axhline(y=1, color='grey')
    ax_I.axvline(x=1, color='grey')
    ax_I.plot(i_ratio, I_s / I_c, label='Superconductor $I_s / I_c$')
    ax_I.plot(i_ratio, I_m / I_c, label='Matrix $I_m / I_c$', linestyle='--')
    ax_I.set_ylabel('currents / $I_c$ [-]')
    ax_I.set_ylim([0, 2.])

    ax_I.legend()

    ax_E.axhline(y=1e6 * E_c, color='grey')
    ax_E.axvline(x=1, color='grey')
    ax_E.plot(i_ratio, 1e6 * E, color='black')
    ax_E.set_ylabel('Elec. field $E$ [uV/m]')
    ax_E.set_ylim([0, ax_E.get_ylim()[1]])

    ax_power.axvline(x=1, color='grey')
    ax_power.plot(i_ratio, 1e3 * E * I_s, label='Superconductor')
    ax_power.plot(i_ratio, 1e3 * E * I_m, label='Matrix', linestyle='--')
    ax_power.set_xlabel('Total transport current $I_t / I_c$ [-]')
    ax_power.set_ylabel('Joule dissipation [mW/m]')
    ax_power.set_ylim([0, ax_power.get_ylim()[1]])
    ax_power.legend()

    plt.tight_layout()



if __name__ == '__main__':
    main()
    plt.show()
