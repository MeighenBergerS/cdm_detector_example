# -*- coding: utf-8 -*-
# detector.py
# Detector Functions

#  simulation parameters
year = 365 * 24 * 60 * 60
time = 10 * year
molecules_per_cm3 = 3.345 * 10**22
molecules_detector = molecules_per_cm3 * volume_detector
nTargets = (16 + 2) * molecules_detector  # H2O