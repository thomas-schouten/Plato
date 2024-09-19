from typing import Optional, Union, List

import numpy as _numpy
import tqdm as _tqdm

import setup, functions_main

class Points(object):
    def __init__(
            self,
            _settings: object,
        ):
        """
        Class to store and manipulate point data.

        :param _settings:   settings object
        :type _settings:    object
        """
        # Store settings
        self.settings = _settings
    
        # Initialise data
        self.data = {}
    
        # Load or initialise points
        self.data = setup.load_data(
            self.data,
            self.reconstruction,
            self.settings.name,
            self.settings.ages,
            "Points",
            self.settings.cases,
            self.settings.options,
            self.settings.point_cases,
            self.settings.dir_path,
            plates = self.plates,
            resolved_geometries = self.resolved_geometries,
            DEBUG_MODE = self.DEBUG_MODE,
            PARALLEL_MODE = self.PARALLEL_MODE,
        )

    def sample_points(
            self,
            reconstruction_times: Optional[Union[_numpy.ndarray, List, float, int]] = None,
            cases: Optional[Union[List[str], str]] = None,
            PROGRESS_BAR: Optional[bool] = True,    
        ):
        """
        Samples seafloor age at points
        The results are stored in the `points` DataFrame, specifically in the `seafloor_age` field for each case and reconstruction time.

        :param reconstruction_times:    reconstruction times to sample points for
        :type reconstruction_times:     list
        :param cases:                   cases to sample points for (defaults to gpe cases if not specified).
        :type cases:                    list
        :param PROGRESS_BAR:            whether or not to display a progress bar
        :type PROGRESS_BAR:             bool
        """
        # Define reconstruction times if not provided
        if reconstruction_times is None:
            reconstruction_times = self.times

        # Check if reconstruction times is a single value
        if isinstance(reconstruction_times, (int, float, _numpy.integer, _numpy.floating)):
            reconstruction_times = [reconstruction_times]

        # Make iterable
        if cases is None:
            iterable = self.gpe_cases
        else:
            if isinstance(cases, str):
                cases = [cases]
            iterable = {case: [] for case in cases}

        # Loop through valid times
        for reconstruction_time in _tqdm(reconstruction_times, desc="Sampling points", disable=(self.DEBUG_MODE or not PROGRESS_BAR)):
            if self.DEBUG_MODE:
                print(f"Sampling points at {reconstruction_time} Ma")

            for key, entries in iterable.items():
                if self.DEBUG_MODE:
                    print(f"Sampling points for case {key} and entries {entries}...")

                # Select dictionaries
                self.seafloor[reconstruction_time] = self.seafloor[reconstruction_time]
                
                # Sample seafloor age at points
                self.points[reconstruction_time][key]["seafloor_age"] = functions_main.sample_ages(self.points[reconstruction_time][key].lat, self.points[reconstruction_time][key].lon, self.seafloor[reconstruction_time]["seafloor_age"])
                
                # Copy DataFrames to other cases
                if len(entries) > 1 and cases is None:
                    for entry in entries[1:]:
                        self.points[reconstruction_time][entry]["seafloor_age"] = self.points[reconstruction_time][key]["seafloor_age"]

        # Set flag to True
        self.sampled_points = True
    
    def compute_gpe_torque(
            self,
            _ages: Optional[Union[_numpy.ndarray, List, float, int]] = None,
            _cases: Optional[Union[List[str], str]] = None,
            PROGRESS_BAR: Optional[bool] = True,    
        ):
        """
        Function to compute gravitational potential energy (GPE) torque.

        :param _ages:    reconstruction times to compute residual torque for
        :type _ages:     list
        :param cases:                   cases to compute GPE torque for (defaults to GPE cases if not specified).
        :type cases:                    list
        :param PROGRESS_BAR:            whether or not to display a progress bar
        :type PROGRESS_BAR:             bool
        """
        # Define reconstruction times if not provided
        if _ages is None:
            _ages = self.times

        # Check if reconstruction times is a single value
        if isinstance(_ages, (int, float, _numpy.integer, _numpy.floating)):
            _ages = [_ages]

        # Check if points have been sampled
        if self.sampled_points == False:
            self.sample_points(_ages, cases)

        # Make iterable
        if cases is None:
            iterable = self.mantle_drag_cases
        else:
            if isinstance(cases, str):
                cases = [cases]
            iterable = {case: [] for case in cases}

        # Loop through reconstruction times
        for i, _age in _tqdm(enumerate(_ages), desc="Computing GPE torques", disable=(self.DEBUG_MODE or not PROGRESS_BAR)):
            if self.DEBUG_MODE:
                print(f"Computing slab bend torques at {_age} Ma")

            # Loop through gpe cases
            for key, entries in iterable.items():
                if self.DEBUG_MODE:
                    print(f"Computing GPE torque for cases {entries}")

                # Calculate GPE torque
                if self.options[key]["GPE torque"]: 
                    self.points[_age][key] = functions_main.compute_GPE_force(self.points[_age][key], self.seafloor[_age], self.options[key], self.mech)
                    self.plates[_age][key] = functions_main.compute_torque_on_plates(
                        self.plates[_age][key], 
                        self.points[_age][key].lat, 
                        self.points[_age][key].lon, 
                        self.points[_age][key].plateID, 
                        self.points[_age][key].GPE_force_lat, 
                        self.points[_age][key].GPE_force_lon,
                        self.points[_age][key].segment_length_lat, 
                        self.points[_age][key].segment_length_lon,
                        self.constants,
                        torque_variable="GPE_torque"
                    )

                    # Copy DataFrames
                    if len(entries) > 1 and cases is None:
                        [[self.points[_age][entry].update(
                            {"GPE_force_" + coord: self.points[_age][key]["GPE_force_" + coord]}
                        ) for coord in ["lat", "lon", "mag"]] for entry in entries[1:]]
                        [[self.plates[_age][entry].update(
                            {"GPE_torque_" + axis: self.plates[_age][key]["GPE_torque_" + axis]}
                        ) for axis in ["x", "y", "z", "mag"]] for entry in entries[1:]]

    def compute_mantle_drag_torque(
            self,
            _ages: Optional[Union[_numpy.ndarray, List, float, int]] = None,
            cases: Optional[Union[List[str], str]] = None,
            PROGRESS_BAR: Optional[bool] = True,    
        ):
        """
        Function to calculate mantle drag torque

        :param _ages:    reconstruction times to compute residual torque for
        :type _ages:     list
        :param cases:                   cases to compute mantle drag torque for (defaults to mantle drag cases if not specified).
        :type cases:                    list
        :param PROGRESS_BAR:            whether or not to display a progress bar
        :type PROGRESS_BAR:             bool
        """
        # Define reconstruction times if not provided
        if _ages is None:
            _ages = self.times

        # Check if reconstruction times is a single value
        if isinstance(_ages, (int, float, _numpy.integer, _numpy.floating)):
            _ages = [_ages]

        # Make iterable
        if cases is None:
            iterable = self.mantle_drag_cases
        else:
            if isinstance(cases, str):
                cases = [cases]
            iterable = {case: [] for case in cases}

        # Loop through reconstruction times
        for i, _age in _tqdm(enumerate(_ages), desc="Computing mantle drag torques", disable=(self.DEBUG_MODE or not PROGRESS_BAR)):
            if self.DEBUG_MODE:
                print(f"Computing mantle drag torques at {_age} Ma")

            # Loop through mantle drag cases
            for key, entries in iterable.items():
                if self.options[key]["Reconstructed motions"]:
                    if self.DEBUG_MODE:
                        print(f"Computing mantle drag torque from reconstructed motions for cases {entries}")

                    # Calculate Mantle drag torque
                    if self.options[key]["Mantle drag torque"]:
                        # Calculate mantle drag force
                        self.plates[_age][key], self.points[_age][key], self.slabs[_age][key] = functions_main.compute_mantle_drag_force(
                            self.plates[_age][key],
                            self.points[_age][key],
                            self.slabs[_age][key],
                            self.options[key],
                            self.mech,
                            self.constants,
                            self.DEBUG_MODE,
                        )

                        # Calculate mantle drag torque
                        self.plates[_age][key] = functions_main.compute_torque_on_plates(
                            self.plates[_age][key], 
                            self.points[_age][key].lat, 
                            self.points[_age][key].lon, 
                            self.points[_age][key].plateID, 
                            self.points[_age][key].mantle_drag_force_lat, 
                            self.points[_age][key].mantle_drag_force_lon,
                            self.points[_age][key].segment_length_lat,
                            self.points[_age][key].segment_length_lon,
                            self.constants,
                            torque_variable="mantle_drag_torque"
                        )

                        # Enter mantle drag torque in other cases
                        if len(entries) > 1 and cases is None:
                                [[self.points[_age][entry].update(
                                    {"mantle_drag_force_" + coord: self.points[_age][key]["mantle_drag_force_" + coord]}
                                ) for coord in ["lat", "lon", "mag"]] for entry in entries[1:]]
                                [[self.plates[_age][entry].update(
                                    {"mantle_drag_torque_" + coord: self.plates[_age][key]["mantle_drag_torque_" + coord]}
                                ) for coord in ["x", "y", "z", "mag"]] for entry in entries[1:]]

            # Loop through all cases
            for case in self.cases:
                if not self.options[case]["Reconstructed motions"]:
                    if self.DEBUG_MODE:
                        print(f"Computing mantle drag torque using torque balance for case {case}")

                    if self.options[case]["Mantle drag torque"]:
                        # Calculate mantle drag force
                        self.plates[_age][case], self.points[_age][case], self.slabs[_age][case] = functions_main.compute_mantle_drag_force(
                            self.plates[_age][case],
                            self.points[_age][case],
                            self.slabs[_age][case],
                            self.options[case],
                            self.mech,
                            self.constants,
                            self.DEBUG_MODE,
                        )

                        # Calculate mantle drag torque
                        self.plates[_age][case] = functions_main.compute_torque_on_plates(
                            self.plates[_age][case], 
                            self.points[_age][case].lat, 
                            self.points[_age][case].lon, 
                            self.points[_age][case].plateID, 
                            self.points[_age][case].mantle_drag_force_lat, 
                            self.points[_age][case].mantle_drag_force_lon,
                            self.points[_age][case].segment_length_lat,
                            self.points[_age][case].segment_length_lon,
                            self.constants,
                            torque_variable="mantle_drag_torque"
                        )

                        # Compute velocity grid
                        self.velocity[_age][case] = setup.get_velocity_grid(
                            self.points[_age][case], 
                            self.seafloor[_age]
                        )

                        # Compute RMS speeds
                        self.plates[_age][case] = functions_main.compute_rms_velocity(
                            self.plates[_age][case],
                            self.points[_age][case]
                        )
