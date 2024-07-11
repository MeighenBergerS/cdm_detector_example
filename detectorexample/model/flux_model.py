# -*- coding: utf-8 -*-
# flux_model.py
# Authors: Stephan Meighen-Berger
# Basic primary and neutrino flux models

# imports
import numpy as np
import pickle as pkl


def primary_model(
        E: np.ndarray,
        a1: float, a2: float, a3: float, a4: float,
        gamma1: float, gamma2: float, gamma3: float, gamma4: float,
        R1: float, R2: float, R3: float, R4: float, Z: float) -> np.ndarray:
    """ Primary model function based on
    Gaisser, T.K., Astroparticle Physics 35, 801 (2012)
    Flux in units of [GeV (m^2 sr s)^{-1}}]

    Parameters
    ----------
    E: np.ndarray
        The primary energies of interest
    a1, a2, a3, a4: float
        Normalization parameters for the different sources
    gamma1, gamma2, gamma3, gamma4: float
        The power-law of each source
    R1, R2, R3, R4: float
        The magnetic cut-off
    Z: int
        Charge of the nucleus

    Returns
    -------
    flux: np.ndarray
        Flux in units of [GeV (m^2 sr s)^{-1}}]
    """
    return (
        a1 * E**(-gamma1) * np.exp(-E / (R1 * Z)) +
        a2 * E**(-gamma2) * np.exp(-E / (R2 * Z)) +
        a3 * E**(-gamma3) * np.exp(-E / (R3 * Z)) +
        a4 * E**(-gamma4) * np.exp(-E / (R4 * Z))
    )

def primary_flux_H3a(E: np.ndarray) -> np.ndarray:
    """ The H3a model for CR in units [GeV (m^2 sr s)^{-1}}]

    Parameters
    ----------
    E: np.ndarray
        The primary energies of interest

    Returns
    -------
    flux: np.ndarray
        Flux in units of [GeV (m^2 sr s)^{-1}}]
    """
    (
        primary_model(E, 7860, 20, 1.7, 0, 1.66, 1.4, 1.4, 0., 4e6, 30e6, 2e9, 60e9, 1) +
        primary_model(E, 3550, 20, 1.7, 0, 1.58, 1.4, 1.4, 1.6, 4e6, 30e6, 2e9, 60e9, 2) +
        primary_model(E, 2200, 13.4, 1.14, 0, 1.63, 1.4, 1.4, 1.6, 4e6, 30e6, 2e9, 60e9, 7) +
        primary_model(E, 1340, 13.4, 1.14, 0, 1.67, 1.4, 1.4, 1.6, 4e6, 30e6, 2e9, 60e9, 12) +
        primary_model(E, 2120, 13.4, 1.14, 0, 1.63, 1.4, 1.4, 1.6, 4e6, 30e6, 2e9, 60e9, 26)
    )

def neutrino_fluxes(path_to_data: str) -> list:
    """ Fetches pre-calculated neutrino flux tables based on MCEq

    https://github.com/mceq-project/MCEq

    Parameters
    ----------
    path_to_data: str
        Path to data files

    Returns
    -------
    fluxes: list
        list of dictionaries [numu, nue], where each dictionary contains pre-calculated neutrino fluxes for
        locations, primary and interaction models, and zenith angles.
    """
    with open(path_to_data + 'mceq_numu_grids.pkl', 'rb') as handle:
        numu_dict = pkl.load(handle)

    with open(path_to_data + 'mceq_nue_grids.pkl', 'rb') as handle:
        nue_dict = pkl.load(handle)

    return [numu_dict, nue_dict]
