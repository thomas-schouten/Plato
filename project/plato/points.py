from typing import Dict, List, Optional, Union

import gplately as _gplately
import numpy as _numpy
import tqdm as _tqdm

import utils_data, utils_calc, utils_init
from settings import Settings

class Points:
    def __init__(
            self,
            settings: Optional[Union[None, Settings]]= None,
            reconstruction: Optional[_gplately.PlateReconstruction]= None,
            rotation_file: Optional[str]= None,
            topology_file: Optional[str]= None,
            polygon_file: Optional[str]= None,
            reconstruction_name: Optional[str] = None,
            ages: Optional[Union[_numpy.ndarray, List, float, int]] = None,
            cases_file: Optional[list[str]]= None,
            cases_sheet: Optional[str]= "Sheet1",
            files_dir: Optional[str]= None,
            resolved_geometries: Optional[Dict] = None,
            PARALLEL_MODE: Optional[bool] = False,
            DEBUG_MODE: Optional[bool] = False,
        ):
        """
        Class to store and manipulate point data.

        :param settings: Simulation parameters.
        :type settings: Settings object
        :param reconstruction: Plate reconstruction.
        :type reconstruction: Reconstruction object
        :param plates: Optional dictionary of plate data (default: None).
        :type plates: Optional[Dict]
        :param data: Optional dictionary of point data structured by age and case (default: None).
        :type data: Optional[Dict]
        """
        # Store settings object
        self.settings = utils_init.get_settings(
            settings, 
            ages, 
            cases_file,
            cases_sheet,
            files_dir,
        )
            
        # Store reconstruction object
        self.reconstruction = utils_init.get_reconstruction(
            reconstruction,
            rotation_file,
            topology_file,
            polygon_file,
            reconstruction_name,
        )

        # Load data
        self.data = utils_data.load_data(
            self.reconstruction,
            self.settings.name,
            self.settings.ages,
            "Points",
            self.settings.cases,
            self.settings.options,
            self.settings.point_cases,
            self.settings.dir_path,
            plate_data=plate_data,
            resolved_geometries=resolved_geometries,
            PARALLEL_MODE=self.settings.PARALLEL_MODE,
        )

        # Create a mask for the data to filter out points with an area below the threshold set in the settings
        self.mask = {age: self.data[age][self.settings.cases[0]].area >= self.settings.options["Minimum plate area"] for age in self.settings.ages}

        # Get valid plateIDs
        self.plateIDs = self.data[self.settings.ages[0]][self.settings.cases[0]].plateID.unique()

        # Calculate velocities at points
        self.calculate_velocities()

        # Set flags for computed torques
        self.sampled_points = False
        self.computed_gpe_torque = False
        self.computed_mantle_drag_torque = False

    def calculate_velocities(
            self,
            ages: Optional[Union[_numpy.ndarray, List, float, int]] = None,
            cases: Optional[Union[List[str], str]] = None,
            stage_rotation: Optional[Dict] = None,
        ):
        """
        Function to compute velocities at points.
        """
        # Define ages if not provided
        if ages is None:
            ages = self.settings.ages
        else:
            if isinstance(ages, str):
                ages = [ages]

        # Define cases if not provided
        if cases is None:
            cases = self.settings.cases

        # Define stage rotation if not provided
        if stage_rotation is None:
            _stage_rotation = {}

        # Loop through ages and cases
        for _age in ages:
            # Get stage rotation, if not provided
            if stage_rotation is None:
                for plateID in self.plateIDs[self.mask[_age][cases[0]]]:
                    _stage_rotation = self.reconstruction.rotation_model.get_rotation(
                        to_time =_age,
                        moving_plate_id = plateID,
                        from_time=_age + self.settings.options["Velocity time step"],
                        anchor_plate_id = self.settings.options["Anchor plateID"]
                    )
        
            for _case in cases:
                # Filter points
                selected_points = self.data[_age][_case][self.mask[_age][_case]]

                # Compute velocities
                velocities = utils_calc.compute_velocity(
                    selected_points.lat,
                    selected_points.lon,
                    _stage_rotation[0],
                    _stage_rotation[1],
                    _stage_rotation[2],
                )

                # Store velocities
                self.data[_age][_case]["v_lat"] = velocities[0]
                self.data[_age][_case]["v_lon"] = velocities[1]
                self.data[_age][_case]["v_mag"] = velocities[2]
                self.data[_age][_case]["v_azi"] = velocities[3]
                self.data[_age][_case]["omega"] = velocities[4]

    def sample_points(
            self,
            ages: Optional[Union[_numpy.ndarray, List, float, int]] = None,
            cases: Optional[Union[List[str], str]] = None,
            PROGRESS_BAR: Optional[bool] = True,    
        ):
        """
        Samples seafloor age at points
        The results are stored in the `points` DataFrame, specifically in the `seafloor_age` field for each case and age.

        :param ages:    reconstruction times to sample points for
        :type ages:     list
        :param cases:                   cases to sample points for (defaults to gpe cases if not specified).
        :type cases:                    list
        :param PROGRESS_BAR:            whether or not to display a progress bar
        :type PROGRESS_BAR:             bool
        """
        # Define ages if not provided
        if ages is None:
            ages = self.settings.ages
        else:
            if isinstance(ages, str):
                ages = [ages]

        # Make iterable
        if cases is None:
            iterable = self.settings.gpe_cases
        else:
            if isinstance(cases, str):
                cases = [cases]
            iterable = {case: [] for case in cases}

        # Loop through valid times
        for _age in _tqdm(ages, desc="Sampling points", disable=(self.settings.DEBUG_MODE or not PROGRESS_BAR)):
            if self.settings.DEBUG_MODE:
                print(f"Sampling points at {_age} Ma")

            for key, entries in iterable.items():
                if self.settings.DEBUG_MODE:
                    print(f"Sampling points for case {key} and entries {entries}...")

                # Select dictionaries
                self.seafloor[_age] = self.seafloor[_age]
                
                # Sample seafloor age at points
                self.points[_age][key]["seafloor_age"] = utils_calc.sample_ages(self.points[_age][key].lat, self.points[_age][key].lon, self.seafloor[_age]["seafloor_age"])
                
                # Copy DataFrames to other cases
                if len(entries) > 1 and cases is None:
                    for entry in entries[1:]:
                        self.points[_age][entry]["seafloor_age"] = self.points[_age][key]["seafloor_age"]

        # Set flag to True
        self.sampled_points = True
    
    def compute_gpe_force(
            self,
            ages: Optional[Union[_numpy.ndarray, List, float, int]] = None,
            cases: Optional[Union[List[str], str]] = None,
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
        if ages is None:
            ages = self.settings.ages
        else:
            if isinstance(ages, str):
                ages = [ages]

        # Check if points have been sampled
        if self.sampled_points == False:
            self.sample_points(ages, cases)

        # Make iterable
        if cases is None:
            iterable = self.mantle_drag_cases
        else:
            if isinstance(cases, str):
                cases = [cases]
            iterable = {case: [] for case in cases}

        # Loop through reconstruction times
        for i, _age in _tqdm(enumerate(_ages), desc="Computing GPE torques", disable=(self.settings.DEBUG_MODE or not PROGRESS_BAR)):
            if self.settings.DEBUG_MODE:
                print(f"Computing slab bend torques at {_age} Ma")

            # Loop through gpe cases
            for key, entries in iterable.items():
                if self.settings.DEBUG_MODE:
                    print(f"Computing GPE torque for cases {entries}")

                # Calculate GPE torque
                if self.options[key]["GPE torque"]: 
                    self.points[_age][key] = utils_calc.compute_GPE_force(self.points[_age][key], self.seafloor[_age], self.options[key], self.mech)
                    self.plates[_age][key] = utils_calc.compute_torque_on_plates(
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

    def compute_mantle_drag_force(
            self,
            ages: Optional[Union[_numpy.ndarray, List, float, int]] = None,
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
        if ages is None:
            ages = self.settings.ages
        else:
            if isinstance(ages, str):
                ages = [ages]

        # Make iterable
        if cases is None:
            iterable = self.mantle_drag_cases
        else:
            if isinstance(cases, str):
                cases = [cases]
            iterable = {case: [] for case in cases}

        # Loop through reconstruction times
        for i, _age in _tqdm(enumerate(ages), desc="Computing mantle drag torques", disable=(self.settings.DEBUG_MODE or not PROGRESS_BAR)):
            if self.settings.DEBUG_MODE:
                print(f"Computing mantle drag torques at {_age} Ma")

            # Loop through mantle drag cases
            for key, entries in iterable.items():
                if self.options[key]["Reconstructed motions"]:
                    if self.settings.DEBUG_MODE:
                        print(f"Computing mantle drag torque from reconstructed motions for cases {entries}")

                    # Calculate Mantle drag torque
                    if self.options[key]["Mantle drag torque"]:
                        # Calculate mantle drag force
                        self.plates.data[_age][key], self.points[_age][key], self.slabs[_age][key] = utils_calc.compute_mantle_drag_force(
                            self.plates.data[_age][key],
                            self.points[_age][key],
                            self.slabs[_age][key],
                            self.options[key],
                            self.mech,
                            self.constants,
                            self.settings.DEBUG_MODE,
                        )

                        # Calculate mantle drag torque
                        self.plates.data[_age][key] = utils_calc.compute_torque_on_plates(
                            self.plates.data[_age][key], 
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
                    if self.settings.DEBUG_MODE:
                        print(f"Computing mantle drag torque using torque balance for case {case}")

                    if self.options[case]["Mantle drag torque"]:
                        # Calculate mantle drag force
                        self.plates.data[_age][case], self.points[_age][case], self.slabs[_age][case] = utils_calc.compute_mantle_drag_force(
                            self.plates.data[_age][case],
                            self.points[_age][case],
                            self.slabs[_age][case],
                            self.options[case],
                            self.mech,
                            self.constants,
                            self.settings.DEBUG_MODE,
                        )

                        # Calculate mantle drag torque
                        self.plates.data[_age][case] = utils_calc.compute_torque_on_plates(
                            self.plates.data[_age][case], 
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
                        self.velocity[_age][case] = utils_data.get_velocity_grid(
                            self.points[_age][case], 
                            self.seafloor[_age]
                        )

                        # Compute RMS velocity
                        self.plates.data[_age][case] = utils_calc.compute_rms_velocity(
                            self.plates.data[_age][case],
                            self.points[_age][case]
                        )
