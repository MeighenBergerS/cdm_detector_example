# -*- coding: utf-8 -*-
# propagation.py
# Authors: Stephan Meighen-Berger
# Basic propagation and light production

from fennel import Fennel
import numpy as np
from ..utils import find_nearest

def light_production(z_grid, wavelengths, energy_grid, photon_cut=100):
    fl = Fennel()
    for E in energy_grid:
        dc_em, dc_sample_em, _, _, zp_em, _ = fl.auto_yields(E, 11, z_grid=z_grid, wavelengths=wavelengths)
        dc_had, dc_sample_had, _, _, zp_had, _ = fl.auto_yields(E, 211, z_grid=z_grid, wavelengths=wavelengths)
        em_counts = np.trapz(dc_em, wavelengths)
        had_counts = np.trapz(dc_had, wavelengths)
        em_tmp  = zp_em[0] / np.trapz(zp_em[0], x=z_grid) * em_counts
        had_tmp = zp_had[0] / np.trapz(zp_had[0], x=z_grid) * had_counts
        em_lengths.append(z_grid[find_nearest(em_tmp, photon_cut)])
        had_lengths.append(z_grid[find_nearest(had_tmp, photon_cut)])
        electron_photons.append(zp_em[0] / np.trapz(zp_em[0], x=z_grid) * em_counts)
        hadron_photons.append(zp_had[0] / np.trapz(zp_had[0], x=z_grid) * had_counts)

    had_lengths = np.array(had_lengths)[10:22]
    em_lengths = np.array(em_lengths)[10:22]

    electron_photons = np.array(electron_photons)[10:22]
    hadron_photons = np.array(hadron_photons)[10:22]