# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# PLATO
# Algorithm to calculate plate forces from tectonic reconstructions
# Plates object
# Thomas Schouten and Edward Clennett, 2023
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
import geopandas as _gpd
import gplately
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
# PLATES OBJECT
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

class Plates:
    def __init__(
            self,
            settings,
            reconstruction,
        ):
        # Store input variables
        self.settings = settings
        self.reconstruction = reconstruction
        
        # GEOMETRIES
        # Set up plate reconstruction object and initialise dictionaries to store resolved topologies and geometries
        self.resolved_topologies, self.resolved_geometries = {}, {}

        # Load or initialise plate geometries
        for _age in tqdm(self.settings.ages, desc="Loading geometries", disable=self.DEBUG_MODE):
            
            # Load resolved geometries if they are available
            self.resolved_geometries[_age] = setup.GeoDataFrame_from_geoparquet(
                self.settings.dir_path,
                "Geometries",
                _age,
                self.settings.name,
            )

            # Get new topologies if they are unavailable
            if self.resolved_geometries[_age] is None:
                self.resolved_geometries[_age] = setup.get_topology_geometries(
                    self.reconstruction, _age, anchor_plateID=0
                )
            
            # Resolve topologies to use to get plates
            # NOTE: This is done because some information is retrieved from the resolved topologies and some from the resolved geometries
            #       This step could be sped up by extracting all information from the geopandas DataFrame, but so far this has not been the main bottleneck
            # Ignore annoying warnings that the field names are laundered
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    action="ignore",
                    message="Normalized/laundered field name:"
                )
                self.resolved_topologies[_age] = []
                _pygplates.resolve_topologies(
                    self.reconstruction.topologies,
                    self.reconstruction.rotations, 
                    self.resolved_topologies[_age], 
                    _age, 
                    anchor_plate_id=0
                )

        # DATA
        # Load or initialise plate data
        self.data = setup.load_data(
            self.data,
            self.reconstruction,
            self.settings.name,
            self.settings.ages,
            "Plates",
            self.settings.cases,
            self.settings.options,
            self.plate_cases,
            self.settings.dir_path,
            resolved_topologies = self.resolved_topologies,
            resolved_geometries = self.resolved_geometries,
            DEBUG_MODE = self.settings.DEBUG_MODE,
            PARALLEL_MODE = self.settings.PARALLEL_MODE,
        )

    def calculate_rms_velocity(
                self,
            ):
            """
            Function to calculate the root mean square (RMS) velocity of the plates.
            """
            for _age in self.settings.ages:
                # Calculate rms velocity
                for key, entries in self.settings.gpe_cases.items():
                    if self.data[self.settings.ages][key]["v_rms_mag"].mean() == 0:
                        self.data[self.settings.ages][key] = functions_main.compute_rms_velocity(
                            self.data[self.settings.ages][key],
                            self.data[self.settings.ages][key]
                        )

                    self.data[_age] = functions_main.copy_values(
                        self.data[_age], 
                        key, 
                        entries, 
                        ["v_rms_mag", "v_rms_azi", "omega_rms"], 
                    )
                
                    # Copy DataFrames to other cases
                    for entry in entries[1:]:
                        if self.plates[_age][entry]["v_rms_mag"].mean() == 0:
                            self.plates[_age][entry]["v_rms_mag"] = self.plates[_age][key]["v_rms_mag"]

                        if self.plates[_age][entry]["v_rms_azi"].mean() == 0:
                            self.plates[_age][entry]["v_rms_azi"] = self.plates[_age][key]["v_rms_azi"]

                        if self.plates[_age][entry]["omega_rms"].mean() == 0:
                            self.plates[_age][entry]["omega_rms"] = self.plates[_age][key]["omega_rms"]


    def optimise_torques(
            self,
            _ages: Optional[Union[_numpy.ndarray, List, float, int]] = None,
            cases: Optional[Union[List[str], str]] = None,
            plates: Optional[
                Union[
                    int,
                    float,
                    _numpy.floating,
                    _numpy.integer,
                    List[Union[int, float, _numpy.floating, _numpy.integer]],
                    _numpy.ndarray
                ]
            ] = None,
            PROGRESS_BAR: Optional[bool] = True,    
        ):
        """
        Function to optimise torques

        :param _ages:    reconstruction times to compute residual torque for
        :type _ages:     list
        :param cases:                   cases to compute driving torque for
        :type cases:                    list
        :param plates:                  plates to optimise torques for
        :type plates:                   list
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
            slab_iterable = self.slab_pull_cases
            mantle_iterable = self.mantle_drag_cases
        else:
            if isinstance(cases, str):
                cases = [cases]
            slab_iterable = {case: [] for case in cases}
            mantle_iterable = {case: [] for case in cases}

        for i, _age in tqdm(self.settings.ages, desc="Optimising torques", disable=(self.DEBUG_MODE or not PROGRESS_BAR)):
            if self.DEBUG_MODE:
                print(f"Optimising torques at {_age} Ma")            
            
            # Optimise torques for slab pull cases
            for key, entries in slab_iterable.items():
                if self.options[key]["Slab pull torque"]:
                    # Select plates
                    selected_plates = self.plates[_age][key].copy()
                    if plates is not None:
                        if isinstance(plates, (int, float, _numpy.floating, _numpy.integer)):
                            plates = [plates]
                        selected_plates = selected_plates.loc[selected_plates.plateID.isin(plates)].copy()
                    
                    # Optimise torques
                    selected_plates = functions_main.optimise_torques(
                        selected_plates,
                        self.mech,
                        self.options[key],
                    )

                    # Feed back into plates
                    if plates is not None:
                        mask = self.plates[_age][key].plateID.isin(plates)
                        self.plates[_age][key].loc[mask, :] = selected_plates
                    else:
                        self.plates[_age][key] = selected_plates

                    # Copy DataFrames, if necessary
                    if len(entries) > 1 and cases is None
                        columns = ["slab_pull_torque_opt" + axis for axis in ["x", "y", "z", "mag"]]
                        self.data[_age] = functions_main.copy_values(
                            self.data[_age], 
                            key, 
                            entries, 
                            columns, 
                        )

            # Optimise torques for mantle drag cases
            for key, entries in mantle_iterable.items():
                if self.options[key]["Mantle drag torque"]:
                    # Select plates
                    selected_plates = self.plates[_age][key].copy()
                    if plates is not None:
                        if isinstance(plates, (int, float, _numpy.floating, _numpy.integer)):
                            plates = [plates]
                        selected_plates = selected_plates[selected_plates.plateID.isin(plates)].copy()

                    selected_plates = functions_main.optimise_torques(
                        selected_plates,
                        self.mech,
                        self.options[key],
                    )

                    # Feed back into plates
                    if plates is not None:
                        self.plates[_age][key][self.plates[_age][key].plateID.isin(plates)] = selected_plates

                    # Copy DataFrames, if necessary
                    if len(entries) > 1 and cases is None:
                        columns = ["mantle_drag_torque_opt" + axis for axis in ["x", "y", "z", "mag"]]
                        self.data[_age] = functions_main.copy_values(
                            self.data[_age], 
                            key, 
                            entries, 
                            columns, 
                        )

    def compute_driving_torque(
            self,
            _ages: Optional[Union[_numpy.ndarray, List, float, int]] = None,
            cases: Optional[Union[List[str], str]] = None,
            plateIDs: Optional[
                Union[
                    int,
                    float,
                    _numpy.floating,
                    _numpy.integer,
                    List[Union[int, float, _numpy.floating, _numpy.integer]],
                    _numpy.ndarray
                ]
            ] = None,
            PROGRESS_BAR: Optional[bool] = True,
        ):
        """
        Function to calculate driving torque

        :param _ages:    reconstruction times to compute residual torque for
        :type _ages:     list
        :param cases:                   cases to compute driving torque for
        :type cases:                    list
        :param plates:                  plates to compute driving torque for
        :type plates:                   list
        :param PROGRESS_BAR:            whether or not to display a progress bar
        :type PROGRESS_BAR:             bool
        """
        # Define reconstruction times if not provided
        if _ages is None:
            _ages = self.times

        # Check if reconstruction times is a single value
        if isinstance(_ages, (int, float, _numpy.integer, _numpy.floating)):
            _ages = [_ages]

        # Define cases if not provided
        if cases is None:
            cases = self.cases

        # Loop through reconstruction times
        for i, _age in tqdm(enumerate(_ages), desc="Computing driving torques", disable=(self.DEBUG_MODE or not PROGRESS_BAR)):
            if self.DEBUG_MODE:
                print(f"Computing driving torques at {_age} Ma")

            for case in cases:
                # Select plates
                selected_plates = self.plates[_age][case].copy()
                if plates is not None:
                    if isinstance(plates, (int, float, _numpy.floating, _numpy.integer)):
                            plates = [plates]
                    selected_plates = selected_plates.loc[selected_plates.plateID.isin(plates)].copy()

                # Calculate driving torque
                selected_plates = functions_main.sum_torque(selected_plates, "driving", self.constants)

                # Feed back into plates
                if plates is not None:
                    mask = self.plates[_age][case].plateID.isin(plates)
                    self.plates[_age][case].loc[mask, :] = selected_plates
                else:
                    self.plates[_age][case] = selected_plates

    def compute_residual_torque(
            self, 
            _ages: Optional[Union[_numpy.ndarray, List, float, int]] = None,
            cases: Optional[Union[List[str], str]] = None,
            plates: Optional[
                Union[
                    int,
                    float,
                    _numpy.floating,
                    _numpy.integer,
                    List[Union[int, float, _numpy.floating, _numpy.integer]],
                    _numpy.ndarray
                ]
            ] = None,
            PROGRESS_BAR: Optional[bool] = True,            
        ):
        """
        Function to calculate residual torque

        :param _ages:    reconstruction times to compute residual torque for
        :type _ages:     list
        :param cases:                   cases to compute driving torque for
        :type cases:                    str or list
        :param plates:                  plates to compute driving torque for
        :type plates:                   list
        :param PROGRESS_BAR:            whether or not to display a progress bar
        :type PROGRESS_BAR:             bool
        """
        # Define reconstruction times if not provided
        if _ages is None:
            _ages = self.times

        # Check if reconstruction times is a single value
        if isinstance(_ages, (int, float, _numpy.integer, _numpy.floating)):
            _ages = [_ages]

        # Define cases if not provided
        if cases is None:
            cases = self.cases

        # Loop through reconstruction times
        for i, _age in tqdm(enumerate(_ages), desc="Computing residual torques", disable=(self.DEBUG_MODE or not PROGRESS_BAR)):
            if self.DEBUG_MODE:
                print(f"Computing residual torques at {_age} Ma")

            for case in self.cases:
                # Select cases that require residual torque computation
                if self.options[case]["Reconstructed motions"]:
                    # Select plates
                    selected_plates = self.plates[_age][case].copy()
                    if plates is not None:
                        if isinstance(plates, (int, float, _numpy.floating, _numpy.integer)):
                            plates = [plates]
                        selected_plates = selected_plates.loc[selected_plates.plateID.isin(plates)].copy()

                    # Calculate driving torque
                    selected_plates = functions_main.sum_torque(selected_plates, "driving", self.constants)

                    # Feed back into plates
                    if plates is not None:
                        mask = self.plates[_age][case].plateID.isin(plates)
                        self.plates[_age][case].loc[mask, :] = selected_plates
                    else:
                        self.plates[_age][case] = selected_plates

                    # Select slabs
                    selected_slabs = self.slabs[_age][case]
                    if plates is not None:
                        selected_slabs = selected_slabs[selected_slabs.lower_plateID.isin(plates)].copy()

                    # Calculate residual torque along subduction zones
                    selected_slabs = functions_main.compute_residual_along_trench(
                        selected_plates,
                        selected_slabs,
                        self.constants,
                        DEBUG_MODE = self.DEBUG_MODE,
                    )

                    # Feed back into slabs
                    if plates is not None:
                        mask = self.self.slabs[_age][case].lower_plateID.isin(plates)
                        self.slabs[_age][case].loc[mask, :] = selected_slabs
                    else:
                        self.slabs[_age][case] = selected_slabs

                else:
                    # Set residual torque to zero
                    for coord in ["x", "y", "z", "mag"]:
                        self.plates[_age][case]["residual_torque_" + coord] = 0


    def calculate_torque_on_plates(
            self,
            ages: Optional[Union[_numpy.ndarray, List, float, int]] = None,
            cases: Optional[Union[List[str], str]] = None,
            plates: Optional[
                Union[
                    int,
                    float,
                    _numpy.floating,
                    _numpy.integer,
                    List[Union[int, float, _numpy.floating, _numpy.integer]],
                    _numpy.ndarray
                ]
            ] = None,
            PROGRESS_BAR: Optional[bool] = True,
        ):
        """
        Function to calculate the torque on plates
        """"
        # Select plates
        
        # Compute torque on plates
        self.data[_age][key] = functions_main.compute_torque_on_plates(
            self.data[_age][key], 
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