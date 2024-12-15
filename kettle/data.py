"""0D transient temperature solver for boiling water"""

from typing import Tuple

import numpy as np

# Constants
g: float = 9.81  # Gravitational acceleration, m/s^2
sigma: float = 0.072  # Surface tension of water, N/m
h_fg: float = 2257e3  # Latent heat of vaporization of water, J/kg
c_p_water: float = 4186  # Specific heat capacity of water, J/kg·K
c_p_metal: float = 500  # Specific heat capacity of stainless steel, J/kg·K
rho_l: float = 958.4  # Density of water at 100°C, kg/m^3
rho_v: float = 0.597  # Density of water vapor, kg/m^3
mu_l: float = 0.000282  # Dynamic viscosity of water at 100°C, Pa·s
k_l: float = 0.67  # Thermal conductivity of water at 100°C, W/m·K
beta: float = 0.00033  # Corrected thermal expansion coefficient of water at 100°C, 1/K
alpha: float = k_l / (rho_l * c_p_water)  # Thermal diffusivity, m^2/s
Pr: float = c_p_water * mu_l / k_l  # Prandtl number
C_s: float = (
    0.006  # Surface-fluid constant for boiling (polished stainless steel and water)
)
H: float = 0.1  # Characteristic length (height of water), m
m_w: float = 0.3  # Mass of water (300 ml), kg
m_metal: float = 0.1  # Mass of stainless steel base (thin kettle bottom), kg
Ra_c: float = 1708  # Critical Rayleigh number for natural convection

# Heater input and initial conditions
A_HEATER: float = 0.01  # Surface area of kettle bottom, m^2
T_AMBIENT: float = 298.15  # Ambient temperature, K (25°C)
T_SAT: float = 373.15  # Saturation temperature of water at 1 atm, K
T_METAL_0: float = T_AMBIENT  # Initial metal temperature, K
T_WATER_0: float = T_AMBIENT  # Initial water temperature, K
DELTA_T_ONB: float = 8.0  # ONB superheat midpoint, K
T_ONB = T_SAT + DELTA_T_ONB  # ONB temperature, K
DELTA_T_CRIT: float = Ra_c * mu_l * alpha / (g * beta * H**3)
HEATING_POWER: float = 2000  # heater power, W
STEEPNESS: float = 1.5  # Controls the steepness of the sigmoid transition


# Heat loss parameters
H_ENV: float = 10.0  # Heat transfer coefficient to environment, W/m^2·K
A_ENV: float = 0.02  # Exposed surface area for water loss, m^2

# Simulation parameters
TURNED_OFF_TIME = 50  # heater turned off at time, s
SIMULATION_TIME = 65  # s


def sigmoid(T: float) -> float:
    """
    Sigmoid function starting at T_SAT and approaching 1 at 2 * (T_SAT + DELTA_T_ONB).

    Parameters:
    - T (float): Temperature value (or array) at which to evaluate the sigmoid.

    Returns:
    - Sigmoid value(s) at T.
    """
    return 1 / (1 + np.exp(-STEEPNESS * (T - T_ONB)))


def heating_function(tt: float) -> float:
    """
    User-defined heater input as a function of time.

    Parameters:
        tt (float): Time in seconds.

    Returns:
        float: Heat input in Watts.
    """
    if tt < TURNED_OFF_TIME:
        return HEATING_POWER  # Constant at first
    return 0  # Turned off afterward


heating_function_vect = np.vectorize(heating_function)


def rayleigh_number(Tsurface: float, Twater: float) -> float:
    """
    Calculate Rayleigh number.

    Parameters:
        Tsurface (float): Surface temperature in Kelvin.
        Twater (float): Water temperature in Kelvin.

    Returns:
        float: Rayleigh number.
    """
    deltaT = Tsurface - Twater
    return g * beta * deltaT * H**3 / (mu_l * alpha)


def nusselt_number_rb(Ra: float) -> Tuple[float, str]:
    """
    Determine Nusselt number based on Rayleigh number.

    Parameters:
        Ra (float): Rayleigh number.

    Returns:
        float: Nusselt number.
    """
    if Ra < 1e7:
        # Laminar natural convection
        return 0.54 * Ra ** (1 / 4), "lam-conv"
    # Turbulent natural convection
    return 0.15 * Ra ** (1 / 3), "turb-conv"


def single_phase_heat_flux(Tmetal: float, Twater: float) -> Tuple[float, str]:
    """
    Calculate heat flux from the metal to the water for single-phase HT.

    Parameters:
        Tmetal (float): Metal temperature in Kelvin.
        Twater (float): Water temperature in Kelvin.

    Returns:
        float: Heat flux in Watts/m**2.
        str: Current HT regime ('conduction', 'lam-conv', or 'turb-conv').
    """
    if Tmetal - Twater < DELTA_T_CRIT:
        regime = "conduction"
        # Simplified conduction Nu
        Nu = 1
    else:
        Ra = rayleigh_number(Tmetal, Twater)
        Nu, regime = nusselt_number_rb(Ra)

    return Nu * k_l / H * (Tmetal - Twater), regime


def boiling_heat_flux(Tmetal) -> Tuple[float, float]:
    """
    Calculate heat flux from the metal to the water for boiling HT.

    Parameters:
        Tmetal (float): Metal temperature in Kelvin.

    Returns:
        float: Heat flux in Watts/m**2.
        float: Fraction of surface occupied by boiling.
    """
    if Tmetal > T_SAT:
        # Excess temperature relative to saturation temperature
        deltaT = Tmetal - T_SAT
        g0 = 1.0  # Force conversion factor (kg·m/N·s^2)

        # Rohsenow correlation for heat flux (q'')
        flux = (
            ((c_p_water * deltaT) / (h_fg * Pr**1.0 * C_s)) ** 3
            * mu_l
            * h_fg
            * ((g * (rho_l - rho_v)) / (g0 * sigma)) ** 0.5
        )

        fraction = sigmoid(Tmetal)
        return flux, fraction
    return 0.0, 0.0


boiling_heat_flux_vect = np.vectorize(boiling_heat_flux)


def heat_flow(Tmetal: float, Twater: float) -> Tuple[float, str]:
    """
    Calculate heat flow from the metal to the water based on the current HT regime.

    Parameters:
        Tmetal (float): Metal temperature in Kelvin.
        Twater (float): Water temperature in Kelvin.

    Returns:
        float: Heat flow in Watts.
        str: Current HT regime.
    """
    sp_flux, regime = single_phase_heat_flux(Tmetal, Twater)
    mp_flux, mp_fraction = boiling_heat_flux(Tmetal)
    if Tmetal > T_SAT:
        regime = "boiling"

    flux = sp_flux * (1 - mp_fraction) + mp_flux * mp_fraction
    return flux * A_HEATER, regime


heat_flow_vect = np.vectorize(heat_flow)


def heat_loss_to_environment(Twater: float) -> float:
    """
    Calculate heat loss from the water to the environment.

    Parameters:
        Twater (float): Water temperature in Kelvin.
        Tambient (float): Ambient temperature in Kelvin.

    Returns:
        float: Heat loss in Watts.
    """
    return H_ENV * A_ENV * (Twater - T_AMBIENT)


def temperature_ode(tt: float, y: list[float]) -> list[float]:
    """
    ODE system for water and metal temperatures.

    Parameters:
        tt (float): Time in seconds.
        y (list[float]): List containing metal and water temperatures [T_metal, T_water].

    Returns:
        list[float]: Time derivatives [dT_metal/dt, dT_water/dt].
    """
    Tmetal, Twater = y
    Q_transfer, _ = heat_flow(Tmetal, Twater)
    Q_loss = heat_loss_to_environment(Twater)
    Q_heater = heating_function(tt)

    dT_water_dt = (Q_transfer - Q_loss) / (m_w * c_p_water)
    if Twater > T_SAT:
        dT_water_dt = 0
    dT_metal_dt = (Q_heater - Q_transfer) / (m_metal * c_p_metal)

    return [dT_metal_dt, dT_water_dt]
