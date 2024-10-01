# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# PLATO
# Algorithm to calculate plate forces from tectonic reconstructions
# Thomas Schouten, 2024
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Import libraries
# Standard libraries
import os
import multiprocessing
import warnings
from typing import List, Optional, Union
from copy import deepcopy

# Third-party libraries
import numpy as _numpy
import matplotlib.pyplot as plt
import geopandas as _geopandas
import gplately as _gplately
import pandas as _pandas
from gplately import pygplates as _pygplates
import cartopy.crs as ccrs
import cmcrameri as cmc
from tqdm import tqdm
import xarray as _xarray

# Local libraries
import setup
import functions_main
import sys

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# SLABS OBJECT
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

class Slabs:
    def __init__(
            self,
            settings: object,
            reconstruction: object,
            plates: dict,
            files_dir: str,
        ):
        """
        Slabs object. Contains all information on slabs
        """
        # Get the slab data
        self.data = {}

        # Load or initialise slabs
        self.data = setup.load_data(
            self.data,
            self.reconstruction,
            self.settings.name,
            self.settings.ages,
            "Slabs",
            self.settings.cases,
            self.settings.options,
            self.settings.slab_cases,
            files_dir,
            plates = self.plates.data,
            resolved_geometries = self.plates.resolved_geometries,
            DEBUG_MODE = self.settings.DEBUG_MODE,
            PARALLEL_MODE = self.settings.PARALLEL_MODE,
        )

        # Calculate total slab length as a function of age and slab tessellation spacing
        self.total_slab_length = _numpy.zeros((len(self.settings.ages), len(self.settings.slab_pull_cases)))
        for i, _age in enumerate(self.settings.ages):
            for j, _case in enumerate(self.settings.slab_pull_cases):
                self.total_slab_length[i] = self.data[_age][_case].trench_segment_length.sum()

        # Set flag for sampling slabs and upper plates
        self.sampled_slabs = False
        self.sampled_upper_plates = False

    def get_slab_data(
        reconstruction: _gplately.PlateReconstruction,
        age: int,
        plates: _pandas.DataFrame,
        topology_geometries: _geopandas.GeoDataFrame,
        options: dict,
        PARALLEL_MODE: Optional[bool] = False,
        ):
        """
        Function to get data on slabs in reconstruction.

        :param reconstruction:      reconstruction
        :type reconstruction:       _gplately.PlateReconstruction
        :param age:                 reconstruction time
        :type age:                  integer
        :param plates:              plates
        :type plates:               pandas.DataFrame
        :param topology_geometries: topology geometries
        :type topology_geometries:  geopandas.GeoDataFrame
        :param options:             options for the case
        :type options:              dict
        :param DEBUG_MODE:          whether to run in debug mode
        :type DEBUG_MODE:           bool
        :param PARALLEL_MODE:       whether to run in parallel mode
        :type PARALLEL_MODE:        bool
        
        :return:                    slabs
        :rtype:                     pandas.DataFrame
        """
        # Set constants
        constants = set_constants()

        # Tesselate subduction zones and get slab pull and bend torques along subduction zones
        slabs = reconstruction.tessellate_subduction_zones(age, ignore_warnings=True, tessellation_threshold_radians=(options["Slab tesselation spacing"]/constants.mean_Earth_radius_km))

        # Convert to _pandas.DataFrame
        slabs = _pandas.DataFrame(slabs)

        # Kick unused columns
        slabs = slabs.drop(columns=[2, 3, 4, 5])

        slabs.columns = ["lon", "lat", "trench_segment_length", "trench_normal_azimuth", "lower_plateID", "trench_plateID"]

        # Convert trench segment length from degree to m
        slabs.trench_segment_length *= constants.equatorial_Earth_circumference / 360

        # Get plateIDs of overriding plates
        sampling_lat, sampling_lon = project_points(
            slabs.lat,
            slabs.lon,
            slabs.trench_normal_azimuth,
            100
        )
        slabs["upper_plateID"] = get_plateIDs(
            reconstruction,
            topology_geometries,
            sampling_lat,
            sampling_lon,
            age,
            PARALLEL_MODE=PARALLEL_MODE
        )

        # Get absolute velocities of upper and lower plates
        for plate in ["upper_plate", "lower_plate", "trench_plate"]:
            # Loop through lower plateIDs to get absolute lower plate velocities
            for plateID in slabs[plate + "ID"].unique():
                # Select all points with the same plateID
                selected_slabs = slabs[slabs[plate + "ID"] == plateID]

                # Get stage rotation for plateID
                selected_plate = plates[plates.plateID == plateID]

                if len(selected_plate) == 0:
                    stage_rotation = reconstruction.rotation_model.get_rotation(
                        to_time=age,
                        moving_plate_id=int(plateID),
                        from_time=age + options["Velocity time step"],
                        anchor_plate_id=options["Anchor plateID"]
                    ).get_lat_lon_euler_pole_and_angle_degrees()
                else:
                    stage_rotation = (
                        selected_plate.pole_lat.values[0],
                        selected_plate.pole_lon.values[0],
                        selected_plate.pole_angle.values[0]
                    )

                # Get plate velocities
                selected_velocities = get_velocities(
                    selected_slabs.lat,
                    selected_slabs.lon,
                    stage_rotation
                )

                # Store in array
                slabs.loc[slabs[plate + "ID"] == plateID, "v_" + plate + "_lat"] = selected_velocities[0]
                slabs.loc[slabs[plate + "ID"] == plateID, "v_" + plate + "_lon"] = selected_velocities[1]
                slabs.loc[slabs[plate + "ID"] == plateID, "v_" + plate + "_mag"] = selected_velocities[2]
                slabs.loc[slabs[plate + "ID"] == plateID, "v_" + plate + "_azi"] = selected_velocities[3]

        # Calculate convergence rates
        slabs["v_convergence_lat"] = slabs.v_lower_plate_lat - slabs.v_trench_plate_lat
        slabs["v_convergence_lon"] = slabs.v_lower_plate_lon - slabs.v_trench_plate_lon
        slabs["v_convergence_mag"] = _numpy.sqrt(slabs.v_convergence_lat**2 + slabs.v_convergence_lon**2)

        # Initialise other columns to store seafloor ages and forces
        # Upper plate
        slabs["upper_plate_thickness"] = 0.
        slabs["upper_plate_age"] = 0.
        slabs["continental_arc"] = False
        slabs["erosion_rate"] = 0.

        # Lower plate
        slabs["lower_plate_age"] = 0.
        slabs["lower_plate_thickness"] = 0.
        slabs["sediment_thickness"] = 0.
        slabs["sediment_fraction"] = 0.
        slabs["slab_length"] = options["Slab length"]

        # Forces
        forces = ["slab_pull", "slab_bend", "residual"]
        coords = ["mag", "lat", "lon"]
        slabs[[force + "_force_" + coord for force in forces for coord in coords]] = [[0] * 9 for _ in range(len(slabs))]
        slabs["residual_force_azi"] = 0.
        slabs["residual_alignment"] = 0.

        # Make sure all the columns are floats
        slabs = slabs.apply(lambda x: x.astype(float) if x.name != "continental_arc" else x)

        return slabs

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
                    self.data[_age][key]["lower_plate_age"], self.data[_age][key]["sediment_thickness"] = functions_main.sample_slabs_from_seafloor(
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
                    self.data[_age][key]["lower_plate_thickness"], _, _ = functions_main.compute_thicknesses(
                        self.data[_age][key].lower_plate_age,
                        self.options[key],
                        crust = False, 
                        water = False
                    )

                    # Calculate slab flux
                    self.plates[_age][key] = functions_main.compute_subduction_flux(
                        self.plates[_age][key],
                        self.data[_age][key],
                        type="slab"
                    )

                    if self.options[key]["Sediment subduction"]:
                        # Calculate sediment subduction
                        self.plates[_age][key] = functions_main.compute_subduction_flux(
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
                    self.data[_age][key]["upper_plate_age"], self.data[_age][key]["continental_arc"], self.data[_age][key]["erosion_rate"], self.data[_age][key]["sediment_thickness"] = functions_main.sample_slabs_from_seafloor(
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
                    self.data[_age][key]["upper_plate_age"], self.data[_age][key]["continental_arc"] = functions_main.sample_slabs_from_seafloor(
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
                    self.data[_age][key] = functions_main.compute_slab_pull_force(self.data[_age][key], self.options[key], self.mech)
                    
                    # Compute interface term if necessary
                    if self.options[key]["Sediment subduction"]:
                        self.data[_age][key] = functions_main.compute_interface_term(self.data[_age][key], self.options[key], self.DEBUG_MODE)
                    
                    # Compute torque on plates
                    self.plates[_age][key] = functions_main.compute_torque_on_plates(
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
                    self.data[_age][key] = functions_main.compute_slab_bend_force(self.data[_age][key], self.options[key], self.mech, self.constants)
                    self.plates[_age][key] = functions_main.compute_torque_on_plates(
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