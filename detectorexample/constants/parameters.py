# -*- coding: utf-8 -*-
# parameters.py
# Authors: Stephan Meighen-Berger
# basic parameters used in the simulation. This are NOT physical constants!

# imports
import numpy as np


energy_grid = np.logspace(np.log10(0.07943282347242814), 11, 122)
# TODO: This needs to become dependent on the refractive index
c_water = 225000 * 1e5