import logging
from typing import Dict, List, Optional, Union

import gplately as _gplately
import numpy as _numpy
import xarray as _xarray
from tqdm import tqdm as _tqdm

import utils_data, utils_calc, utils_init
from settings import Settings

class Slabs:
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

        # Initialise data dictionary
        self.data = {age: {} for age in self.settings.ages}

        # Loop through times
        for _age in _tqdm(ages, desc="Loading data", disable=self.settings.logger.level == logging.INFO):
            # Load available data
            self.data[_age] = utils_data.load_data(
                self.data[_age],
                self.settings.name,
                _age,
                "Slabs",
                self.settings.cases,
                self.settings.point_cases,
                self.settings.dir_path,
                PARALLEL_MODE=self.settings.PARALLEL_MODE,
            )

            # Initialise missing data
            available_case = None
            for key, entries in self.settings.slab_cases.items():
                # Check if data is available
                for entry in entries:
                    if self.data[_age][entry] is not None:
                        available_case = entry
                        break
                
                # If data is available, copy to other cases    
                if available_case:
                    for entry in entries:
                        if entry is not available_case:
                            self.data[_age][entry] = self.data[_age][available_case].copy()

                # If no data is available, initialise new data
                else:
                    # Check if resolved geometries are available
                    if not resolved_geometries or key not in resolved_geometries.keys():
                        resolved_geometries = {}
                        resolved_geometries[_age] = utils_data.get_topology_geometries(
                            self.reconstruction, _age, self.settings.options[self.settings.cases[0]]["Anchor plateID"]
                        )

                    # Initialise missing data
                    self.data[_age][key] = utils_data.get_slab_data(
                        self.reconstruction,
                        _age,
                        resolved_geometries[_age], 
                        self.settings.options[key],
                    )

                    # Copy data to other cases
                    if len(entries) > 1:
                        for entry in entries[1:]:
                            self.data[_age][entry] = self.data[_age][key].copy()

        # Calculate velocities at points
        self.calculate_velocities()

        # Calculate total slab length as a function of age and case
        self.total_slab_length = _numpy.zeros((len(self.settings.ages), len(self.settings.slab_pull_cases)))
        for i, _age in enumerate(self.settings.ages):
            for j, _case in enumerate(self.settings.slab_pull_cases):
                self.total_slab_length[i] = self.data[_age][_case].trench_segment_length.sum()

        # Set flag for sampling slabs and upper plates
        self.sampled_slabs = False
        self.sampled_upper_plates = False

    def calculate_velocities(
            self,
            ages: Optional[Union[_numpy.ndarray, List, float, int]] = None,
            cases: Optional[Union[List[str], str]] = None,
            stage_rotation: Optional[Dict] = None,
        ):
        """
        Function to compute velocities at slabs.
        """
        # Define ages if not provided
        _ages = utils_data.get_ages(ages, self.settings.ages)
        
        # Define cases if not provided
        _cases = utils_data.get_cases(cases, self.settings.cases)

        # Loop through ages and cases
        for _age in _ages:
            for plate in ["upper", "lower", "trench"]:
                for _case in _cases:
                    for plateID in self.data[_age][_case][f"{plate}_plateID"].unique():
                        # Get stage rotation, if not provided
                        if stage_rotation is None:
                            _stage_rotation = self.reconstruction.rotation_model.get_rotation(
                                to_time =_age,
                                moving_plate_id = int(plateID),
                                from_time=_age + self.settings.options[_case]["Velocity time step"],
                                anchor_plate_id = self.settings.options[_case]["Anchor plateID"]
                            ).get_lat_lon_euler_pole_and_angle_degrees()
                        else:
                            # Get stage rotation from the provided DataFrame in the dictionary
                            _stage_rotation = stage_rotation[_age][_case][stage_rotation[_age][_case].plateID == plateID]
                                        
                        # Make mask for plate
                        mask = self.data[_age][_case][f"{plate}_plateID"] == plateID
                                                
                        # Compute velocities
                        velocities = utils_calc.compute_velocity(
                            self.data[_age][_case].lat[mask],
                            self.data[_age][_case].lon[mask],
                            _stage_rotation[0],
                            _stage_rotation[1],
                            _stage_rotation[2],
                            self.settings.constants,
                        )

                        # Store velocities
                        self.data[_age][_case].loc[mask, f"v_{plate}_lat"] = velocities[0]
                        self.data[_age][_case].loc[mask, f"v_{plate}_lon"] = velocities[1]
                        self.data[_age][_case].loc[mask, f"v_{plate}_mag"] = velocities[2]
                        self.data[_age][_case].loc[mask, f"omega_{plate}"] = velocities[3]

    def sample_slabs(
            self,
            ages: Optional[Union[_numpy.ndarray, List, float, int]] = None,
            cases: Optional[Union[List[str], str]] = None,
            seafloor_grid: Optional[_xarray.Dataset] = None,
            PROGRESS_BAR: Optional[bool] = True,    
        ):
        """
        Samples seafloor age (and optionally, sediment thickness) the lower plate along subduction zones
        The results are stored in the `slabs` DataFrame, specifically in the `lower_plate_age`, `sediment_thickness`, and `lower_plate_thickness` fields for each case and reconstruction time.

        :param _ages:    reconstruction times to sample slabs for
        :type _ages:     list
        :param cases:                   cases to sample slabs for (defaults to slab pull cases if not specified).
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
            iterable = self.settings.slab_pull_cases
        else:
            if isinstance(cases, str):
                cases = [cases]
            iterable = {_case: [] for _case in cases}

        # Check options for slabs
        for _age in tqdm(ages, desc="Sampling slabs", disable=(self.DEBUG_MODE or not PROGRESS_BAR)):
            if self.DEBUG_MODE:
                print(f"Sampling slabs at {_age} Ma")

            # Select cases
            for key, entries in iterable.items():
                if self.DEBUG_MODE:
                    print(f"Sampling overriding plate for case {key} and entries {entries}...")
                    
                if self.options[key]["Slab pull torque"] or self.options[key]["Slab bend torque"]:
                    # Sample age and sediment thickness of lower plate from seafloor
                    self.data[_age][key]["lower_plate_age"], self.data[_age][key]["sediment_thickness"] = utils_calc.sample_slabs_from_seafloor(
                        self.data[_age][key].lat, 
                        self.data[_age][key].lon,
                        self.data[_age][key].trench_normal_azimuth,
                        self.seafloor[_age], 
                        self.options[key],
                        "lower plate",
                        sediment_thickness=self.data[_age][key].sediment_thickness,
                        continental_arc=self.data[_age][key].continental_arc,
                    )

                    # Calculate lower plate thickness
                    self.data[_age][key]["lower_plate_thickness"], _, _ = utils_calc.compute_thicknesses(
                        self.data[_age][key].lower_plate_age,
                        self.options[key],
                        crust = False, 
                        water = False
                    )

                    # Calculate slab flux
                    self.plates[_age][key] = utils_calc.compute_subduction_flux(
                        self.plates[_age][key],
                        self.data[_age][key],
                        type="slab"
                    )

                    if self.options[key]["Sediment subduction"]:
                        # Calculate sediment subduction
                        self.plates[_age][key] = utils_calc.compute_subduction_flux(
                            self.plates[_age][key],
                            self.data[_age][key],
                            type="sediment"
                        )

                    if len(entries) > 1:
                        for entry in entries[1:]:
                            self.data[_age][entry]["lower_plate_age"] = self.data[_age][key]["lower_plate_age"]
                            self.data[_age][entry]["sediment_thickness"] = self.data[_age][key]["sediment_thickness"]
                            self.data[_age][entry]["lower_plate_thickness"] = self.data[_age][key]["lower_plate_thickness"]

        # Set flag to True
        self.sampled_slabs = True

    def sample_upper_plates(
            self,
            _ages: Optional[Union[_numpy.ndarray, List, float, int]] = None,
            cases: Optional[Union[List[str], str]] = None,
            PROGRESS_BAR: Optional[bool] = True,    
        ):
        """
        Samples seafloor age the upper plate along subduction zones
        The results are stored in the `slabs` DataFrame, specifically in the `upper_plate_age`, `upper_plate_thickness` fields for each case and reconstruction time.

        :param _ages:    reconstruction times to sample upper plates for
        :type _ages:     list
        :param cases:                   cases to sample upper plates for (defaults to slab pull cases if not specified).
        :type cases:                    list
        :param PROGRESS_BAR:            whether or not to display a progress bar
        :type PROGRESS_BAR:             bool
        """
        # Define reconstruction times if not provided
        if _ages is None:
            _ages = self.settings.ages
        else:
            # Check if reconstruction times is a single value
            if isinstance(_ages, (int, float, _numpy.integer, _numpy.floating)):
                _ages = [_ages]
        
        # Make iterable
        if cases is None:
            iterable = self.slab_pull_cases
        else:
            if isinstance(cases, str):
                cases = [cases]
            iterable = {case: [] for case in cases}

        # Loop through valid times    
        for _age in tqdm(_ages, desc="Sampling upper plates", disable=(self.DEBUG_MODE or not PROGRESS_BAR)):
            if self.DEBUG_MODE:
                print(f"Sampling overriding plate at {_age} Ma")

            # Select cases
            for key, entries in iterable.items():
                if self.DEBUG_MODE:
                    print(f"Sampling overriding plate for case {key} and entries {entries}...")

                # Check whether to output erosion rate and sediment thickness
                if self.options[key]["Sediment subduction"] and self.options[key]["Active margin sediments"] != 0 and self.options[key]["Sample erosion grid"] in self.seafloor[_age].data_vars:
                    # Sample age and arc type, erosion rate and sediment thickness of upper plate from seafloor
                    self.data[_age][key]["upper_plate_age"], self.data[_age][key]["continental_arc"], self.data[_age][key]["erosion_rate"], self.data[_age][key]["sediment_thickness"] = utils_calc.sample_slabs_from_seafloor(
                        self.data[_age][key].lat, 
                        self.data[_age][key].lon,
                        self.data[_age][key].trench_normal_azimuth,  
                        self.seafloor[_age],
                        self.options[key],
                        "upper plate",
                        sediment_thickness=self.data[_age][key].sediment_thickness,
                    )

                elif self.options[key]["Sediment subduction"] and self.options[key]["Active margin sediments"] != 0:
                    # Sample age and arc type of upper plate from seafloor
                    self.data[_age][key]["upper_plate_age"], self.data[_age][key]["continental_arc"] = utils_calc.sample_slabs_from_seafloor(
                        self.data[_age][key].lat, 
                        self.data[_age][key].lon,
                        self.data[_age][key].trench_normal_azimuth,  
                        self.seafloor[_age],
                        self.options[key],
                        "upper plate",
                    )
                
                # Copy DataFrames to other cases
                if len(entries) > 1 and cases is None:
                    for entry in entries[1:]:
                        self.data[_age][entry]["upper_plate_age"] = self.data[_age][key]["upper_plate_age"]
                        self.data[_age][entry]["continental_arc"] = self.data[_age][key]["continental_arc"]
                        if self.options[key]["Sample erosion grid"]:
                            self.data[_age][entry]["erosion_rate"] = self.data[_age][key]["erosion_rate"]
                            self.data[_age][entry]["sediment_thickness"] = self.data[_age][key]["sediment_thickness"]
        
        # Set flag to True
        self.sampled_upper_plates = True

    def compute_slab_pull_force(
            self,
            ages: Optional[Union[_numpy.ndarray, List, float, int]] = None,
            cases: Optional[Union[List[str], str]] = None,
            PROGRESS_BAR: Optional[bool] = True,    
        ):
        """
        Compute slab pull torque.

        :param _ages:    reconstruction times to compute slab pull torque for
        :type _ages:     list
        :param cases:                   cases to compute slab pull torque for (defaults to slab pull cases if not specified).
        :type cases:                    list
        :param PROGRESS_BAR:            whether or not to display a progress bar
        :type PROGRESS_BAR:             bool
        """
        # Define reconstruction times if not provided
        if ages is None:
            ages = self.settings.ages
        else:
            # Check if reconstruction times is a single value
            if isinstance(ages, (int, float, _numpy.integer, _numpy.floating)):
                ages = [ages]

        # Check if upper plates have been sampled already
        if self.sampled_upper_plates == False:
            self.sample_upper_plates(ages, cases)

        # Check if slabs have been sampled already
        if self.sampled_slabs == False:
            self.sample_slabs(ages, cases)

        # Make iterable
        if cases is None:
            iterable = self.settings.slab_pull_cases
        else:
            if isinstance(cases, str):
                cases = [cases]
            iterable = {case: [] for case in cases}

        # Loop through reconstruction times
        for i, _age in tqdm(enumerate(self.times), desc="Computing slab pull torques", disable=(self.DEBUG_MODE or not PROGRESS_BAR)):
            if self.DEBUG_MODE:
                print(f"Computing slab pull torques at {_age} Ma")

            # Loop through slab pull cases
            for key, entries in iterable.items():
                if self.DEBUG_MODE and cases is None:
                    print(f"Computing slab pull torques for cases {entries}")
                elif self.DEBUG_MODE:
                    print(f"Computing slab pull torques for case {key}")

                if self.options[key]["Slab pull torque"]:
                    # Calculate slab pull torque
                    self.data[_age][key] = utils_calc.compute_slab_pull_force(self.data[_age][key], self.options[key], self.mech)
                    
                    # Compute interface term if necessary
                    if self.options[key]["Sediment subduction"]:
                        self.data[_age][key] = utils_calc.compute_interface_term(self.data[_age][key], self.options[key], self.DEBUG_MODE)
                    
                    # Compute torque on plates
                    self.plates[_age][key] = utils_calc.compute_torque_on_plates(
                        self.plates[_age][key], 
                        self.data[_age][key].lat, 
                        self.data[_age][key].lon, 
                        self.data[_age][key].lower_plateID, 
                        self.data[_age][key].slab_pull_force_lat, 
                        self.data[_age][key].slab_pull_force_lon,
                        self.data[_age][key].trench_segment_length,
                        1,
                        self.constants,
                        torque_variable="slab_pull_torque"
                    )

                    # Copy DataFrames
                    if len(entries) > 1 and cases is None:
                        [[self.data[_age][entry].update(
                            {"slab_pull_force_" + coord: self.data[_age][key]["slab_pull_force_" + coord]}
                        ) for coord in ["lat", "lon", "mag"]] for entry in entries[1:]]
                        [[self.plates[_age][entry].update(
                            {"slab_pull_force_" + coord: self.plates[_age][key]["slab_pull_force_" + coord]}
                        ) for coord in ["lat", "lon", "mag"]] for entry in entries[1:]]
                        [[self.plates[_age][entry].update(
                            {"slab_pull_torque_" + axis: self.plates[_age][key]["slab_pull_torque_" + axis]}
                        ) for axis in ["x", "y", "z", "mag"]] for entry in entries[1:]]

    def compute_slab_bend_force(
            self,
            _ages: Optional[Union[_numpy.ndarray, List, float, int]] = None,
            _cases: Optional[Union[List[str], str]] = None,
            PROGRESS_BAR: Optional[bool] = True,    
        ):
        """
        Compute slab bend torque.

        :param _ages:    reconstruction times to compute slab bend torque for
        :type _ages:     list
        :param cases:                   cases to compute slab bend torque for (defaults to slab bend cases if not specified).
        :type cases:                    list
        :param PROGRESS_BAR:            whether or not to display a progress bar
        :type PROGRESS_BAR:             bool
        """
        # Define reconstruction times if not provided
        if _ages is None:
            _ages = self.settings.ages
        else:
            # Check if reconstruction times is a single value
            if isinstance(_ages, (int, float, _numpy.integer, _numpy.floating)):
                _ages = [_ages]

        # Check if slabs have been sampled already
        if self.sampled_slabs == False:
            self.sample_slabs(_ages, _cases)

        # Make iterable
        if _cases is None:
            iterable = self.settings.slab_bend_cases
        else:
            if isinstance(_cases, str):
                cases = [cases]
            iterable = {_case: [] for _case in _cases}

        # Loop through reconstruction times
        for i, _age in tqdm(enumerate(_ages), desc="Computing slab bend torques", disable=(self.DEBUG_MODE or not PROGRESS_BAR)):
            if self.DEBUG_MODE:
                print(f"Computing slab bend torques at {_age} Ma")

            # Loop through slab bend cases
            for key, entries in iterable.items():
                if self.DEBUG_MODE:
                    print(f"Computing slab bend torques for cases {entries}")

                # Calculate slab bending torque
                if self.options[key]["Slab bend torque"]:
                    self.data[_age][key] = utils_calc.compute_slab_bend_force(self.data[_age][key], self.options[key], self.mech, self.constants)
                    self.plates[_age][key] = utils_calc.compute_torque_on_plates(
                        self.plates[_age][key], 
                        self.data[_age][key].lat, 
                        self.data[_age][key].lon, 
                        self.data[_age][key].lower_plateID, 
                        self.data[_age][key].slab_bend_force_lat, 
                        self.data[_age][key].slab_bend_force_lon,
                        self.data[_age][key].trench_segment_length,
                        1,
                        self.constants,
                        torque_variable="slab_bend_torque"
                    )

                    # Copy DataFrames
                    if len(entries) > 1 and cases is None:
                        [self.data[_age][entry].update(
                            {"slab_bend_force_" + coord: self.data[_age][key]["slab_bend_force_" + coord]}
                        ) for coord in ["lat", "lon", "mag"] for entry in entries[1:]]
                        [self.plates[_age][entry].update(
                            {"slab_bend_force_" + coord: self.plates[_age][key]["slab_bend_force_" + coord]}
                        ) for coord in ["lat", "lon", "mag"] for entry in entries[1:]]
                        [self.plates[_age][entry].update(
                            {"slab_bend_torque_" + axis: self.plates[_age][key]["slab_bend_torque_" + axis]}
                        ) for axis in ["x", "y", "z", "mag"] for entry in entries[1:]]