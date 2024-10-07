# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# PLATO
# Algorithm to calculate plate forces from tectonic reconstructions
# Thomas Schouten, 2024
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Import libraries
# Standard libraries
import os as _os
import logging
import warnings
from typing import Dict, List, Optional, Union

# Third-party libraries
import numpy as _numpy
import pandas as _pandas
import gplately as _gplately
from gplately import pygplates as _pygplates
from tqdm import tqdm

# Local libraries
import utils_calc, utils_data, utils_init
from settings import Settings
from points import Points
from slabs import Slabs

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# PLATES OBJECT
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

class Plates:
    """
    Class that contains all information for the plates in a reconstruction.
    """
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
            data: Optional[Union[Dict, str]] = None,
            resolved_geometries: Optional[Dict] = None,
            PARALLEL_MODE: Optional[bool] = False,
            DEBUG_MODE: Optional[bool] = False,
        ):
        """
        Initialise the Plates object with the required objects.

        :param settings:            Settings object (default: None)
        :type settings:             Optional[Settings]
        :param reconstruction:      Reconstruction object (default: None)
        :type reconstruction:       Optional[Reconstruction]
        :param data:                Data object (default: None)
        :type data:                 Optional[Data]
        :param resolved_geometries: Resolved geometries (default: None)
        :type resolved_geometries:  Optional[dict]
        """
        # Store settings object
        self.settings = utils_init.get_settings(
            settings, 
            ages, 
            cases_file,
            cases_sheet,
            files_dir,
            PARALLEL_MODE = PARALLEL_MODE,
            DEBUG_MODE = DEBUG_MODE,
        )
            
        # Store reconstruction object
        self.reconstruction = utils_init.get_reconstruction(
            reconstruction,
            rotation_file,
            topology_file,
            polygon_file,
            reconstruction_name,
        )
        
        # GEOMETRIES
        # Set up plate reconstruction object and initialise dictionaries to store resolved topologies and geometries
        self.resolved_topologies, self.resolved_geometries = {}, {}

        # Define ages if not provided
        _ages = utils_data.get_ages(
            ages,
            self.settings.ages,
        )

        # Load or initialise plate geometries
        for _age in tqdm(_ages, desc="Loading geometries", disable=logging.getLogger().isEnabledFor(logging.INFO)):
            
            # Load resolved geometries if they are available
            if resolved_geometries is not None:
                # Check if resolved geometries are a dictionary
                if not isinstance(resolved_geometries, dict):
                    raise ValueError("Resolved geometries should be a dictionary.")
                
                # Check if the age is in the dictionary
                if _age in resolved_geometries.keys():
                    self.resolved_geometries[_age] = resolved_geometries[_age]
                else:
                    self.resolved_geometries[_age] = utils_data.GeoDataFrame_from_geoparquet(
                        self.settings.dir_path,
                        "Geometries",
                        _age,
                        self.settings.name,
                    )

                    # Get new topologies if they are unavailable
                    if self.resolved_geometries[_age] is None:
                        self.resolved_geometries[_age] = utils_data.get_topology_geometries(
                            self.reconstruction, _age, anchor_plateID=0
                        )
            
            # Resolve topologies to use to get plates
            # NOTE: This is done because some information is retrieved from the resolved topologies and some from the resolved geometries
            #       This step could be sped up by extracting all information from the geopandas DataFrame, but so far this has not been the main bottleneck
            self.resolved_topologies = utils_data.get_resolved_topologies(
                self.reconstruction,
                _age,
            )

        # DATA
        # Load data for all combinations of ages and cases
        self.data = utils_data.load_data(
            self.reconstruction,
            self.settings.name,
            _ages,
            "Plates",
            self.settings.cases,
            self.settings.options,
            self.settings.plate_cases,
            files_dir,
            resolved_topologies = self.resolved_topologies,
            resolved_geometries = self.resolved_geometries,
            PARALLEL_MODE = self.settings.PARALLEL_MODE,
        )

    def calculate_rms_velocity(
            self,
            point_data: Optional[Dict] = None,
            ages: Optional[Union[_numpy.ndarray, List, float, int]] = None,
            cases: Optional[Union[List[str], str]] = None,
        ):
        """
        Function to calculate the root mean square (RMS) velocity of the plates.
        """
        # Define ages if not provided
        _ages = utils_data.get_ages(
            ages,
            self.settings.ages,
        )

        # Check if no points are passed, initialise Points object
        if point_data is None:
            logging.info("No points data provided, initialising Points object.")
            return
        
            # points = Points(
            #     self.settings,
            #     self.reconstruction,
            #     ages,
            #     plate_data = self.data,
            #     resolved_geometries = self.resolved_geometries,
            # )

            # point_data = points.data
        
        # Define cases if not provided, default to GPE cases because it only depends on the grid spacing
        _iterable = utils_data.get_iterable(
            cases,
            self.settings.gpe_cases,
        )

        for _age in _ages:
            # Calculate rms velocity
            for key, entries in _iterable.items():
                print(self.data)
                if self.data[_age][key]["v_rms_mag"].mean() == 0:
                    self.data[_age][key] = utils_calc.compute_rms_velocity(
                        self.data[_age][key],
                        point_data[_age][key]
                    )

                self.data[_age] = utils_calc.copy_values(
                    self.data[_age], 
                    key, 
                    entries, 
                    ["v_rms_mag", "v_rms_azi", "omega_rms"], 
                )
            
                # Copy DataFrames to other cases
                for entry in entries[1:]:
                    if self.data[_age][entry]["v_rms_mag"].mean() == 0:
                        self.data[_age][entry]["v_rms_mag"] = self.data[_age][key]["v_rms_mag"]

                    if self.data[_age][entry]["v_rms_azi"].mean() == 0:
                        self.data[_age][entry]["v_rms_azi"] = self.data[_age][key]["v_rms_azi"]

                    if self.data[_age][entry]["omega_rms"].mean() == 0:
                        self.data[_age][entry]["omega_rms"] = self.data[_age][key]["omega_rms"]

    def optimise_torques(
            self,
            ages: Optional[Union[_numpy.ndarray, List, float, int]] = None,
            cases: Optional[Union[List[str], str]] = None,
            plate_IDs: Optional[
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

        :param ages:    reconstruction times to compute residual torque for
        :type ages:     list
        :param cases:                   cases to compute driving torque for
        :type cases:                    list
        :param plates:                  plates to optimise torques for
        :type plates:                   list
        :param PROGRESS_BAR:            whether or not to display a progress bar
        :type PROGRESS_BAR:             bool
        """
        # Define ages if not provided
        _ages = utils_data.get_ages(
            ages,
            self.settings.ages,
        )

        # Define iterable if cases not provided
        _slab_iterable = utils_data.get_iterable(
            cases,
            self.settings.slab_cases,
        )
        _mantle_iterable = utils_data.get_iterable(
            cases,
            self.settings.mantle_drag_cases,
        )
        if ages is not None:
            # Check if ages is a single value
            if isinstance(ages, (int, float, _numpy.integer, _numpy.floating)):
                ages = [ages]
        else:
            # Otherwise, use all ages from the settings
            ages = self.settings.ages

        # Make iterable
        if cases is None:
            slab_iterable = self.slab_pull_cases
            mantle_iterable = self.mantle_drag_cases
        else:
            if isinstance(cases, str):
                cases = [cases]
            slab_iterable = {case: [] for case in cases}
            mantle_iterable = {case: [] for case in cases}

        for i, _age in tqdm(self.settings.ages, desc="Optimising torques", disable=(self.settings.DEBUG_MODE or not PROGRESS_BAR)):
            if self.settings.DEBUG_MODE:
                print(f"Optimising torques at {_age} Ma")            
            
            # Optimise torques for slab pull cases
            for key, entries in slab_iterable.items():
                if self.options[key]["Slab pull torque"]:
                    # Select plates
                    selected_data = self.plates[_age][key].copy()
                    if plates is not None:
                        if isinstance(plates, (int, float, _numpy.floating, _numpy.integer)):
                            plates = [plates]
                        selected_data = selected_data.loc[selected_data.plateID.isin(plate_IDs)].copy()
                    
                    # Optimise torques
                    selected_plates = utils_calc.optimise_torques(
                        selected_plates,
                        self.mech,
                        self.options[key],
                    )

                    # Feed back into plates
                    if plates is not None:
                        mask = self.plates[_age][key].plateID.isin(plates)
                        self.data[_age][key].loc[mask, :] = selected_plates
                    else:
                        self.data[_age][key] = selected_plates

                    # Copy DataFrames, if necessary
                    if len(entries) > 1 and cases is None:
                        columns = ["slab_pull_torque_opt" + axis for axis in ["x", "y", "z", "mag"]]
                        self.data[_age] = utils_calc.copy_values(
                            self.data[_age], 
                            key, 
                            entries, 
                            columns, 
                        )

            # Optimise torques for mantle drag cases
            for key, entries in mantle_iterable.items():
                if self.options[key]["Mantle drag torque"]:
                    # Select plates
                    selected_data = self.data[_age][key].copy()
                    if plates is not None:
                        if isinstance(plate_IDs, (int, float, _numpy.floating, _numpy.integer)):
                            plates = [plate_IDs]
                        selected_data = selected_data[selected_data.plateID.isin(plate_IDs)].copy()

                    selected_data = utils_calc.optimise_torques(
                        selected_data,
                        self.mech,
                        self.options[key],
                    )

                    # Feed back into plates
                    if plates is not None:
                        self.data[_age][key][self.plates[_age][key].plateID.isin(plates)] = selected_data

                    # Copy DataFrames, if necessary
                    if len(entries) > 1 and cases is None:
                        columns = ["mantle_drag_torque_opt" + axis for axis in ["x", "y", "z", "mag"]]
                        self.data[_age] = utils_calc.copy_values(
                            self.data[_age], 
                            key, 
                            entries, 
                            columns, 
                        )

    def compute_driving_torque(
            self,
            ages: Optional[Union[_numpy.ndarray, List, float, int]] = None,
            cases: Optional[Union[List[str], str]] = None,
            plate_IDs: Optional[
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

        :param ages:    reconstruction times to compute residual torque for
        :type ages:     list
        :param cases:                   cases to compute driving torque for
        :type cases:                    list
        :param plates:                  plates to compute driving torque for
        :type plates:                   list
        :param PROGRESS_BAR:            whether or not to display a progress bar
        :type PROGRESS_BAR:             bool
        """
        # Define ages if not provided
        _ages = utils_data.get_ages(
            ages,
            self.settings.ages,
        )

        # Define cases if not provided
        _cases = utils_data.get_cases(
            cases,
            self.settings.cases,
        )

        # Loop through reconstruction times
        for i, _age in tqdm(enumerate(_ages), desc="Computing driving torques", disable=(self.settings.DEBUG_MODE or not PROGRESS_BAR)):
            if self.settings.DEBUG_MODE:
                print(f"Computing driving torques at {_age} Ma")

            for _case in _cases:
                # Select plates
                selected_plates = self.plates[_age][_case].copy()
                if plates is not None:
                    if isinstance(plates, (int, float, _numpy.floating, _numpy.integer)):
                            plates = [plates]
                    selected_plates = selected_plates.loc[selected_plates.plateID.isin(plates)].copy()

                # Calculate driving torque
                selected_plates = utils_calc.sum_torque(selected_plates, "driving", self.constants)

                # Feed back into plates
                if plates is not None:
                    mask = self.plates[_age][_case].plateID.isin(plates)
                    self.plates[_age][_case].loc[mask, :] = selected_plates
                else:
                    self.plates[_age][_case] = selected_plates

    def compute_residual_torque(
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
        Function to calculate residual torque

        :param ages:    reconstruction times to compute residual torque for
        :type ages:     list
        :param cases:                   cases to compute driving torque for
        :type cases:                    str or list
        :param plates:                  plates to compute driving torque for
        :type plates:                   list
        :param PROGRESS_BAR:            whether or not to display a progress bar
        :type PROGRESS_BAR:             bool
        """
        # Define ages if not provided
        if ages is not None:
            # Check if ages is a single value
            if isinstance(ages, (int, float, _numpy.integer, _numpy.floating)):
                ages = [ages]
        else:
            # Otherwise, use all ages from the settings
            ages = self.settings.ages

        # Define cases if not provided
        if cases is None:
            cases = self.cases

        # Loop through reconstruction times
        for i, _age in tqdm(enumerate(ages), desc="Computing residual torques", disable=(self.settings.DEBUG_MODE or not PROGRESS_BAR)):
            if self.settings.DEBUG_MODE:
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
                    selected_plates = utils_calc.sum_torque(selected_plates, "driving", self.constants)

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
                    selected_slabs = utils_calc.compute_residual_along_trench(
                        selected_plates,
                        selected_slabs,
                        self.constants,
                        DEBUG_MODE = self.settings.DEBUG_MODE,
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
            type: str,
            ages: Optional[Union[_numpy.ndarray, List, float, int]] = None,
            cases: Optional[Union[List[str], str]] = None,
            plate_IDs: Optional[
                Union[
                    int,
                    float,
                    _numpy.floating,
                    _numpy.integer,
                    List[Union[int, float, _numpy.floating, _numpy.integer]],
                    _numpy.ndarray
                ]
            ] = None,
            force_data: Optional[Union[Dict, str]] = None,
        ):
        """
        Function to calculate the torque on plates
        """
        # Define ages if not provided
        _ages = utils_data.get_ages(
            ages,
            self.settings.ages,
        )

        # # Get data if not provided
        # if force_data is None:
        #     if type == "Slab_pull_torque" or type == "Slab_bend_torque":
        #         # Load or initialise slabs
        #         slabs = Slabs(
        #             self.settings,
        #             self.reconstruction,
        #             ages,
        #             resolved_geometries = self.resolved_geometries,
        #         )

        #         # Sample slabs
        #         force_data = slabs.data
            
     
        # Define cases if not provided, default to GPE cases because it only depends on the grid spacing
        if type == "slab_pull_torque" or type == "slab_bend_torque":
            matching_cases = self.settings.slab_cases
        elif type == "GPE_torque":
            matching_cases = self.settings.gpe_cases
        elif type == "mantle_drag_torque":
            matching_cases = self.settings.mantle_drag_cases
        
        _iterable = utils_data.get_iterable(
            cases,
            matching_cases,
        )

        # Select plates
        if plate_IDs is not None:
            if isinstance(plate_IDs, (int, float, _numpy.floating, _numpy.integer)):
                plate_IDs = [plate_IDs]
        
        # Loop through ages
        for _age in tqdm(_ages, desc="Calculating torque on plates"):
            logging.info(f"Calculating torque on plates at {_age} Ma")

            for key in cases:
                # Select plates, if necessary
                selected_data = self.data[_age][key].copy()
                if plate_IDs is not None:
                    selected_data = selected_data.loc[selected_data.plateID.isin(plate_IDs)].copy()

                # Define length of segment
                if type == "slab_pull_torque" or type == "slab_bend_torque":
                    length = force_data[_age][key].trench_segment_length
                    width = 1.
                else:
                    length = force_data[_age][key].segment_length_lat
                    width = force_data[_age][key].segment_length_lat

                # Calculate torques
                selected_data = utils_calc.compute_torque_on_plates(
                    self.data[_age][key], 
                    self.force_data[_age][key].lat, 
                    self.force_data[_age][key].lon, 
                    self.force_data[_age][key].plateID, 
                    self.force_data[_age][key].mantle_drag_force_lat, 
                    self.force_data[_age][key].mantle_drag_force_lon,
                    self.force_data[_age][key].segment_length_lat,
                    self.force_data[_age][key].segment_length_lon,
                    self.constants,
                    torque_variable=type
                )

                # Feed back into plates
                if plate_IDs is not None:
                    mask = self.data[_age][key].plateID.isin(plate_IDs)
                    self.data[_age][key].loc[mask, :] = selected_data
                else:
                    self.data[_age][key] = selected_data