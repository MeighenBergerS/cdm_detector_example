# -*- coding: utf-8 -*-
# detectorexample.py
# Copyright (C) 2024 Stephan Meighen-Berger,
# Interface class to the package

import numpy as np
from typing import Union
from .config import config
from tqdm.auto import tqdm

class DetectorExample(object):
    """Class for unifying injection, energy loss calculation, and photon propagation"""
    def __init__(
        self,
        userconfig: Union[None, dict, str] = None,
    ) -> None:
        """Initializes the DetectorExample class

        params
        ______
        userconfig: Configuration dictionary or path to yaml file 
            which specifies configuration

        raises
        ______

        """
        if userconfig is not None:
            if isinstance(userconfig, dict):
                config.from_dict(userconfig)
            else:
                config.from_yaml(userconfig)
        
        self._rng = np.random.RandomState(config["run"]["seed"])
        #  simulation parameters
        year = 365 * 24 * 60 * 60
        time = 10 * year
        molecules_per_cm3 = 3.345 * 10**22
        molecules_detector = molecules_per_cm3 * volume_detector
        nTargets = (16 + 2) * molecules_detector  # H2O

        detector_factor = nTargets * time * np.pi * 4
        interactions_numu_nc = detector_factor * numu_flux * cross_section_NC(energy_grid) * energy_widths
        interactions_nue_CC = detector_factor * nue_flux * cross_section_CC(energy_grid) * energy_widths
        interactions_nue_NC = detector_factor * nue_flux * cross_section_NC(energy_grid) * energy_widths

    # Sampling function for the interaction
    def spatial_sampling(seflf, nsamples: int, detector_radius: float, rng: np.random.RandomState) -> np.ndarray:
        """ generates an event sample

        Parameters
        ----------
        nsamples: int
            Number of samples
        detector_radius: float
            The radius of the detector
        rng: np.random.RandomState
            The random state generator

        Returns
        -------
        rsamples: np.ndarray
            Sampled radius
        phi_samples: np.ndarray
            Sampled phi (interaction point and outgoing)
        theta_samples: np.ndarray
            Sampled theta (interaction point and outgoing)
        """
        radius_samples = rng.uniform(0, detector_radius, size=nsamples)
        phi_samples = rng.uniform(0., 360., size=(nsamples, 2))
        theta_samples = rng.uniform(0., 180., size=(nsamples, 2))
        return radius_samples, phi_samples, theta_samples

    ns_bins = np.linspace(0, 100, 101)

    def event_generator(
        nsamples: np.ndarray,
        energy_grid: np.ndarray,
        detector_radius: float,
        rng: np.random.RandomState,
        type='CC') -> np.ndarray:
        """ generates particle events within the detector
        """
        # Spatial generation
        spatial_samples = []
        for nsamp in nsamples:
            spatial_samples.append(
                spatial_sampling(int(nsamp), detector_radius, rng)
            )
        
        if type == 'CC':
            lengths = em_lengths
            distro = electron_photons
        else:
            lengths = had_lengths
            distro = hadron_photons
        spatial_cuts = []
        timing_arr = []
        for idE, spatial_sample in enumerate(spatial_samples):
            rsampl, phisamp, thetasampl = spatial_sample
            X_int = rsampl * np.sin(phisamp[:, 0]) * np.cos(thetasampl[:, 0])
            Y_int = rsampl * np.sin(phisamp[:, 0]) * np.sin(thetasampl[:, 0])
            Z_int = rsampl * np.cos(phisamp[:, 0])
            arrow_scale = lengths[idE]
            X_dir = arrow_scale * np.sin(phisamp[:, 1]) * np.cos(thetasampl[:, 1])
            Y_dir = arrow_scale * np.sin(phisamp[:, 1]) * np.sin(thetasampl[:, 1])
            Z_dir = arrow_scale * np.cos(phisamp[:, 1])
            # Spatial Cuts
            event_r = np.sqrt(
                (X_int + X_dir)**2 +
                (Y_int + Y_dir)**2 +
                (Z_int + Z_dir)**2
            )
            event_cut = event_r < detector_radius
            spatial_cuts.append(event_cut)
            # Timing
            energy_sample = []
            for idS, _ in enumerate(X_int):
                hits_binned, _ = np.histogram(
                    ((np.abs(event_r[idS] - detector_radius) + z_grid) / c_water) * 1e9,
                    bins=ns_bins, weights=distro[idE]
                )
                energy_sample.append(gaussian_filter(hits_binned, sigma=2, radius=10))
            timing_arr.append(energy_sample)
        spatial_cuts = np.array(spatial_cuts)
        return spatial_samples, np.array(timing_arr), spatial_cuts

    # Analysis
    def tail_vs_start(pulses: np.ndarray) -> np.ndarray:
        """ takes an array of pulses and checks their likelihood of being a CC event
        """
        idmaxes = np.argmax(pulses, axis=1)
        ratio_arr = np.array([
            np.sum(pulses[idTest][idmaxes[idTest]+1:]) /
            np.sum(pulses[idTest][:idmaxes[idTest]+1])
            for idTest in range(len(pulses))
        ])
        return ratio_arr

    def data_TvsS_test(all_pulses: np.ndarray, cuts:np.ndarray) -> np.ndarray:
        """ helper function to apply analysis to the entire set
        """
        ratio_energy_bins = []
        for idE, energy_bin in enumerate(all_pulses):
            tmp_pulses = np.array(energy_bin)[cuts[idE]]
            ratio_energy_bins.append(tail_vs_start(tmp_pulses))
        return np.concatenate(np.array(ratio_energy_bins, dtype=object))

    def data_TvsS_cut(all_pulses: np.ndarray, cuts:np.ndarray, TvsS_cut: float) -> np.ndarray:
        """ helper function to apply analysis cuts to the entire set
        """
        tmp_bool = []
        for idE, energy_bin in enumerate(all_pulses):
            tmp_pulses = np.array(energy_bin)[cuts[idE]]
            tmp_bool.append(np.less(tail_vs_start(tmp_pulses), TvsS_cut))
        return np.array(tmp_bool, dtype=object)

    def analysis_simulation(nTrials: int, signal: np.ndarray, background: np.ndarray, seed=1337) -> np.ndarray:
        """ entire analysis multiple times
        """
        totals_CC_pre = []
        totals_NC_pre = []
        totals_CC = []
        totals_NC = []
        rng_trial = np.random.RandomState(seed)
        # Offloading some random number work before loop
        signal_sets = rng_trial.poisson(signal[10:22], size=(nTrials, len(signal[10:22])))
        background_sets = rng_trial.poisson(background[10:22], size=(nTrials, len(signal[10:22])))
        for set in tqdm(range(nTrials)):
            _, timing_samples_CC, cuts_CC = event_generator(
                signal_sets[set],  # Sampling the events as well!
                energy_grid[10:22],
                demo_sphere_radius,
                rng_trial,
                type='CC'
            )
            _, timing_samples_NC, cuts_NC = event_generator(
                background_sets[set],
                energy_grid[10:22],
                demo_sphere_radius,
                rng_trial,
                type='NC'
            )
            CC_counts = np.array([
                np.sum(cut_e) for cut_e in cuts_CC
            ])
            NC_counts = np.array([
                np.sum(cut_e) for cut_e in cuts_NC
            ])
            cc_cut_set = data_TvsS_cut(timing_samples_CC, cuts_CC, 1.5023693639498166)
            nc_cut_set = data_TvsS_cut(timing_samples_NC, cuts_NC, 1.5023693639498166)

            # Sum didn't work
            cc_cut_counts = np.array([
                np.sum(elem) for elem in cc_cut_set
            ])
            nc_cut_counts = np.array([
                np.sum(elem) for elem in nc_cut_set
            ])
            totals_CC_pre.append(np.sum(CC_counts))
            totals_NC_pre.append(np.sum(NC_counts))
            totals_CC.append(np.sum(cc_cut_counts))
            totals_NC.append(np.sum(nc_cut_counts))
        return np.array([
            totals_NC, totals_NC_pre,
            totals_CC, totals_CC_pre,
        ])

    results = analysis_simulation(100, interactions_nue_NC, (interactions_numu_nc + interactions_nue_CC))