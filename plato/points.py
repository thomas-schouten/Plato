# Standard libraries
import logging
from typing import Dict, List, Optional, Union

# Third-party libraries
import geopandas as _geopandas
import gplately as _gplately
import numpy as _numpy
import pandas as _pandas
import xarray as _xarray
from tqdm import tqdm as _tqdm

# Testing
import matplotlib.pyplot as plt

# Local libraries
from . import utils_data, utils_calc, utils_init
from .settings import Settings

class Points:
    """
    Class that contains all information for the points in a reconstruction.
    A `Slabs` object can be initialised in multiple ways:

    1.  The user can initialise a `Points` object from scratch by providing the reconstruction and the ages of interest.
        The reconstruction can be provided as a file with rotation poles, a file with topologies, and a file with polygons, or as one of the model name string identifiers for the models available on the GPlately DataServer (https://gplates.github.io/gplately/v1.3.0/#dataserver).
        
        Additionally, the user may specify the excel file with a number of different cases (combinations of options) to be considered.

    2.  Alternatively, the user can initialise a `Points` object by providing a `Settings` object and a `Reconstruction` object from a `Globe`, `Grids`, `Plates`, `Points` or `Slabs` object.
        Providing the settings from a `Points` object will allow the user to initialise a new `Points` object with the same settings as the original object.

    :param settings:            `Settings` object (default: None)
    :type settings:             plato.settings.Settings
    :param reconstruction:      `Reconstruction` object (default: None)
    :type reconstruction:       gplately.PlateReconstruction
    :param rotation_file:       filepath to .rot file with rotation poles (default: None)
    :type rotation_file:        str
    :param topology_file:       filepath to .gpml file with topologies (default: None)
    :type topology_file:        str
    :param polygon_file:        filepath to .gpml file with polygons (default: None)
    :type polygon_file:         str
    :param reconstruction_name: model name string identifiers for the GPlately DataServer (default: None)
    :type reconstruction_name:  str
    :param ages:                ages of interest (default: None)
    :type ages:                 float, int, list, numpy.ndarray
    :param cases_file:          filepath to excel file with cases (default: None)
    :type cases_file:           str
    :param cases_sheet:         name of the sheet in the excel file with cases (default: "Sheet1")
    :type cases_sheet:          str
    :param files_dir:           directory to store files (default: None)
    :type files_dir:            str
    :param PARALLEL_MODE:       flag to enable parallel mode (default: False)
    :type PARALLEL_MODE:        bool
    :param DEBUG_MODE:          flag to enable debug mode (default: False)
    :type DEBUG_MODE:           bool
    :param CALCULATE_VELOCITIES: flag to calculate velocities (default: True)
    :type CALCULATE_VELOCITIES: bool
    """
    def __init__(
            self,
            settings: Optional[Settings] = None,
            reconstruction: Optional[_gplately.PlateReconstruction] = None,
            rotation_file: Optional[str] = None,
            topology_file: Optional[str] = None,
            polygon_file: Optional[str] = None,
            reconstruction_name: Optional[str] = None,
            ages: Optional[Union[int, float, List[Union[int, float]], _numpy.ndarray]] = None,
            cases_file: Optional[str] = None,
            cases_sheet: str = "Sheet1",
            files_dir: Optional[str] = None,
            resolved_geometries: Dict[float, Dict[str, _geopandas.GeoDataFrame]] = None,
            PARALLEL_MODE: bool = False,
            DEBUG_MODE: bool = False,
            CALCULATE_VELOCITIES: bool = True,
            PROGRESS_BAR: bool = True,
        ):
        """
        Constructor for the 'Points' object.
        """
        # Store settings object
        self.settings = utils_init.get_settings(
            settings, 
            reconstruction_name,
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

        # Initialise data dictionary
        self.data = {age: {} for age in self.settings.ages}

        # Initialise dictionary to store data that was newly initialised
        self.NEW_DATA = {age: [] for age in self.settings.ages}

        # Loop through times
        for _age in _tqdm(self.settings.ages, desc="Loading point data", disable=self.settings.logger.level==logging.INFO):
            # Load available data
            for key, entries in self.settings.point_cases.items():
                # Make list to store available cases
                available_cases = []

                # Try to load all DataFrames
                for entry in entries:
                    self.data[_age][entry] = utils_data.DataFrame_from_parquet(
                        self.settings.dir_path,
                        "Points",
                        self.settings.name,
                        _age,
                        entry,
                    )
                    # Store the cases for which a DataFrame could be loaded
                    if self.data[_age][entry] is not None:
                        available_cases.append(entry)
                
                # Check if any DataFrames were loaded
                if len(available_cases) > 0:
                    # Copy all DataFrames from the available case        
                    for entry in entries:
                        if entry not in available_cases:
                            self.data[_age][entry] = self.data[_age][available_cases[0]].copy()

                else:
                    logging.info(f"No point data found for age {_age} and key {key}.")
                    # Initialise missing data
                    if not isinstance(resolved_geometries, dict) or not isinstance(resolved_geometries.get(key), _geopandas.GeoDataFrame):
                        resolved_geometries = utils_data.get_resolved_geometries(
                            self.reconstruction,
                            _age,
                            self.settings.options[key]["Anchor plateID"]
                        )

                    # Initialise missing data
                    self.data[_age][key] = utils_data.get_point_data(
                        self.reconstruction,
                        _age,
                        resolved_geometries, 
                        self.settings.options[key],
                    )

                    # Copy data to other cases
                    if len(entries) > 1:
                        for entry in entries[1:]:
                            self.data[_age][entry] = self.data[_age][key].copy()

        # Calculate velocities at points
        if CALCULATE_VELOCITIES:
            for _age in self.settings.ages:
                for _case in self.settings.reconstructed_cases:
                    if _age in self.NEW_DATA.keys() or self.data[_age][_case]["velocity_mag"].mean() == 0:
                        self.calculate_velocities(
                            _age,
                            _case,
                            PROGRESS_BAR = False
                        )

        # Set flags for computed torques
        self.sampled_seafloor = False
        self.sampled_LAB = False
        self.computed_gpe_torque = False
        self.computed_mantle_drag_torque = False

    def __str__(self):
        return f"Points is a class that contains data and methods for working with grid points."
    
    def __repr__(self):
        return self.__str__()

    def calculate_velocities(
            self,
            ages: Optional[Union[int, float, List[Union[int, float]], _numpy.ndarray]] = None,
            cases: Optional[Union[str, List[str]]] = None,
            stage_rotation: Optional[Dict[Union[int, float], Dict[str, _pandas.DataFrame]]] = None,
            PROGRESS_BAR: bool = True,
        ):
        """
        Function to compute velocities at points.

        :param ages:            ages of interest (default: None)
        :type ages:             float, int, list, numpy.ndarray
        :param cases:           cases of interest (default: None)
        :type cases:            str, list
        :param stage_rotation:  stage rotation poles (default: None)
        :type stage_rotation:   dict
        """
        # Define ages if not provided
        _ages = utils_data.select_ages(ages, self.settings.ages)
        
        # Define cases if not provided
        _cases = utils_data.select_cases(cases, self.settings.point_cases)

        # Loop through ages and cases
        for _age in _tqdm(
                _ages,
                desc="Calculating velocities at points",
                disable=(self.settings.logger.level in [logging.INFO, logging.DEBUG] or not PROGRESS_BAR)
            ):
            for _case in _cases:
                for plateID in self.data[_age][_case].plateID.unique():
                    # Get stage rotation, if not provided
                    if (
                        isinstance(stage_rotation, dict)
                        and _age in stage_rotation.keys()
                        and _case in stage_rotation[_age].keys()
                        and isinstance(stage_rotation[_age][_case], _pandas.DataFrame)
                    ):
                        # Get stage rotation from the provided DataFrame in the dictionary
                        _stage_rotation = stage_rotation[_age][_case][stage_rotation[_age][_case].plateID == plateID]
                
                    # Get stage rotation, if not provided
                    else:
                        stage_rotation = self.reconstruction.rotation_model.get_rotation(
                            to_time =_age,
                            moving_plate_id = int(plateID),
                            from_time=_age + self.settings.options[_case]["Velocity time step"],
                            anchor_plate_id = self.settings.options[_case]["Anchor plateID"]
                        ).get_lat_lon_euler_pole_and_angle_degrees()

                        # Organise as DataFrame
                        _stage_rotation = _pandas.DataFrame({
                                "plateID": [plateID],
                                "pole_lat": [stage_rotation[0]],
                                "pole_lon": [stage_rotation[1]],
                                "pole_angle": [stage_rotation[2]],
                            })
                        
                    # Make mask for plate
                    mask = self.data[_age][_case].plateID == plateID
                                            
                    # Compute velocities
                    velocities = utils_calc.compute_velocity(
                        self.data[_age][_case].loc[mask],
                        _stage_rotation,
                        self.settings.constants,
                    )

                    # Store velocities
                    self.data[_age][_case]["velocity_lat"].values[mask] = velocities[0]
                    self.data[_age][_case]["velocity_lon"].values[mask] = velocities[1]
                    self.data[_age][_case]["velocity_mag"].values[mask] = velocities[2]
                    self.data[_age][_case]["velocity_azi"].values[mask] = velocities[3]
                    self.data[_age][_case]["spin_rate_mag"].values[mask] = velocities[4]

    def sample_seafloor_ages(
            self,
            ages: Optional[Union[int, float, List[Union[int, float]], _numpy.ndarray]] = None,
            cases: Optional[Union[str, List[str]]] = None,
            plateIDs: Optional[Union[int, float, List[Union[int, float]], _numpy.ndarray]] = None,
            grids: Optional[Dict[Union[int, float], _xarray.Dataset]] = None,
            vars: Optional[List[str]] = ["seafloor_age"],
            PROGRESS_BAR: bool = True,
        ):
        """
        Samples seafloor age at points.

        :param ages:            ages of interest (default: None)
        :type ages:             float, int, list, numpy.ndarray
        :param cases:           cases of interest (default: None)
        :type cases:            str, list
        :param plateIDs:        plateIDs of interest (default: None)
        :type plateIDs:         list, numpy.ndarray
        :param grids:           seafloor age grids (default: None)
        :type grids:            dict
        :param vars:            variables to sample (default: ["seafloor_age"])
        :type vars:             str, list
        :param PROGRESS_BAR:    flag to enable progress bar (default: True)
        :type PROGRESS_BAR:     bool
        """
        # Sample grid
        self.sample_grid(
            ages,
            cases,
            plateIDs,
            grids,
            vars,
            default_cases = "gpe",
            PROGRESS_BAR = PROGRESS_BAR,
        )

        # Set sampling flag to true
        self.sampled_seafloor = True

    def sample_lab_depths(
            self,
            ages: Optional[Union[int, float, List[Union[int, float]], _numpy.ndarray]] = None,
            cases: Optional[Union[str, List[str]]] = None,
            plateIDs: Optional[Union[int, float, List[Union[int, float]], _numpy.ndarray]] = None,
            grids: Optional[Dict[Union[int, float], _xarray.Dataset]] = None,
            vars: Optional[List[str]] = ["LAB_depth"],
            PROGRESS_BAR: bool = True,
        ):
        """
        Samples the depth of the lithosphere-asthenosphere boundary (LAB) at points.

        :param ages:            ages of interest (default: None)
        :type ages:             float, int, list, numpy.ndarray
        :param cases:           cases of interest (default: None)
        :type cases:            str, list
        :param plateIDs:        plateIDs of interest (default: None)
        :type plateIDs:         list, numpy.ndarray
        :param grids:           grids to sample (default: None)
        :type grids:            dict
        :param vars:            variables to sample (default: ["LAB_depth"])
        :type vars:             str, list
        :param PROGRESS_BAR:    flag to enable progress bar (default: True)
        :type PROGRESS_BAR:     bool
        """
        # Sample grid
        self.sample_grid(
            ages,
            cases,
            plateIDs,
            grids,
            vars,
            default_cases = "mantle drag",
            PROGRESS_BAR = PROGRESS_BAR,
        )

        # Set sampling flag to true
        self.sampled_LAB = True

    def sample_grid(
            self,
            ages: Optional[Union[int, float, List[Union[int, float]], _numpy.ndarray]] = None,
            cases: Optional[Union[str, List[str]]] = None,
            plateIDs: Optional[Union[int, float, List[Union[int, float]], _numpy.ndarray]] = None,
            grids: Dict[Union[int, float], Union[_xarray.Dataset, Dict[str, _xarray.Dataset]]] = None,
            vars: List[str] = ["seafloor_age"],
            sampling_coords: List[str] = ["lat", "lon"],
            cols: Optional[List[str]] = None,
            default_cases: Optional[str] = None,
            PROGRESS_BAR: bool = True,
        ):
        """
        Samples any grid at points.

        :param ages:            ages of interest (default: None)
        :type ages:             float, int, list, numpy.ndarray
        :param cases:           cases of interest (default: None)
        :type cases:            str, list
        :param plateIDs:        plateIDs of interest (default: None)
        :type plateIDs:         list, numpy.ndarray
        :param grids:           grids to sample (default: None)
        :type grids:            dict
        :param vars:            variables to sample (default: ["seafloor_age"])
        :type vars:             str, list
        :param sampling_coords: coordinates to sample (default: ["lat", "lon"])
        :type sampling_coords:  list
        :param cols:            columns to store sampled data (default: ["seafloor_age"])
        :type cols:             str, list
        """
        # Define ages if not provided
        _ages = utils_data.select_ages(ages, self.settings.ages)

        # Define default case if not provided
        if not default_cases:
            default_cases = self.settings.cases
        elif default_cases == "mantle drag":
            default_cases = self.settings.mantle_drag_cases
        elif default_cases == "gpe":
            default_cases = self.settings.gpe_cases
        elif default_cases == "points":
            default_cases = self.settings.points
        
        # Define cases if not provided
        _iterable = utils_data.select_iterable(cases, default_cases)

        # Define variables if not provided
        if vars is not None and isinstance(vars, str):
            _vars = [vars]
        elif vars is not None and isinstance(vars, list):
            _vars = vars
        else:
            _vars = []

        # Loop through valid times
        for _age in _tqdm(
                _ages, 
                desc="Sampling points", 
                disable=(self.settings.logger.level in [logging.INFO, logging.DEBUG] or not PROGRESS_BAR)
            ):
            for key, entries in _iterable.items():
                # Define plateIDs if not provided
                _plateIDs = utils_data.select_plateIDs(plateIDs, self.data[_age][key].plateID.unique())

                # Select points
                _data = self.data[_age][key]
                if plateIDs is not None:
                    _data = _data[_data.plateID.isin(_plateIDs)]

                # Determine the appropriate grid
                _grid = None
                if _age in grids.keys():
                    if isinstance(grids[_age], _xarray.Dataset):
                        _grid = grids[_age]
                    elif key in grids[_age] and isinstance(grids[_age][key], _xarray.Dataset):
                        _grid = grids[_age][key]
                
                if _grid is None:
                    logging.warning(f"No valid grid found for age {_age} and key {key}.")
                    continue  # Skip this iteration if no valid grid is found

                # Set _vars to the grid's data variables if not already defined
                _vars = list(_grid.data_vars) if not _vars else _vars

                # Set columns to _vars if not already defined or if not of the same length
                _cols = _vars if cols is None or len(cols) != len(_vars) else cols

                # Sample grid at points for each variable
                for _var, _col in zip(_vars, _cols):
                    sampled_data = utils_calc.sample_grid(
                        _data[sampling_coords[0]],
                        _data[sampling_coords[1]],
                        _grid[_var],
                    )

                    # Enter sampled data back into the DataFrame
                    self.data[_age][key].loc[_data.index, _col] = sampled_data
                    
                    # Copy to other entries
                    self.data[_age] = utils_data.copy_values(
                        self.data[_age], 
                        key, 
                        entries, 
                        [_col],
                    )

    def calculate_lab_depths(
            self,
            ages: Optional[Union[int, float, List[Union[int, float]], _numpy.ndarray]] = None,
            cases: Optional[Union[str, List[str]]] = None,
            plateIDs: Optional[Union[int, float, List[Union[int, float]], _numpy.ndarray]] = None,
            grids: Optional[Dict[Union[int, float], _xarray.Dataset]] = None,
            PROGRESS_BAR: bool = True,
        ):
        """
        Function to compute the depth of the lithosphere-asthenosphere boundary (LAB) at points.

        :param ages:            ages of interest (default: None)
        :type ages:             float, int, list, numpy.ndarray
        :param cases:           cases of interest (default: None)
        :type cases:            str, list
        """
        # Define ages if not provided
        _ages = utils_data.select_ages(ages, self.settings.ages)

        # Define cases if not provided
        _iterable = utils_data.select_iterable(cases, self.settings.mantle_drag_cases)

        # Sample LAB depths, if not already sampled
        if not self.sampled_LAB:
            self.sample_lab_depths(
                ages,
                cases,
                plateIDs,
                grids,
            )
        
        # Loop through ages and cases
        for _age in _tqdm(
                _ages, 
                desc="Calculating LAB depths at points", 
                disable=(self.settings.logger.level in [logging.INFO, logging.DEBUG] or not PROGRESS_BAR)
            ):
            for key, entries in _iterable.items():
                if self.settings.options[key]["Continental keels"]:
                    # Select points
                    _data = self.data[_age][key].copy()

                    # Remove continents (i.e. points with NaN values)
                    _data = _data[_data["seafloor_age"].notna()]

                    # Calculate LAB depths
                    computed_data = utils_calc.compute_LAB_depth(
                        _data,
                        self.settings.options[key],
                    )

                    # Enter sampled data back into the DataFrame
                    self.data[_age][key].loc[_data.index, "LAB_depth"] = computed_data.copy()
                    
                    # Copy to other entries
                    if len(entries) > 1:
                        self.data[_age] = utils_data.copy_values(
                            self.data[_age], 
                            key, 
                            entries,
                            ["LAB_depth"],
                        )
    
    def calculate_gpe_force(
            self,
            ages: Optional[Union[int, float, List[Union[int, float]], _numpy.ndarray]] = None,
            cases: Optional[Union[str, List[str]]] = None,
            plateIDs: Optional[Union[int, float, List[Union[int, float]], _numpy.ndarray]] = None,
            seafloor_grid: Dict[Union[int, float], _xarray.Dataset] = None,
            PROGRESS_BAR: bool = True,
        ):
        """
        Function to compute gravitational potential energy (GPE) force acting at points.

        :param ages:            ages of interest (default: None)
        :type ages:             float, int, list, numpy.ndarray
        :param cases:           cases of interest (default: None)
        :type cases:            str, list
        :param plateIDs:        plateIDs of interest (default: None)
        :type plateIDs:         list, numpy.ndarray
        :param seafloor_grid:   seafloor age grid (default: None)
        :type seafloor_grid:    dict
        """
        # Define ages if not provided
        _ages = utils_data.select_ages(ages, self.settings.ages)
        
        # Define cases if not provided
        if cases == "reconstructed":
            _iterable = utils_data.select_iterable(None, self.settings.reconstructed_cases)
        elif cases == "synthetic":
            _iterable = utils_data.select_iterable(None, self.settings.synthetic_cases)
        else:
            _iterable = utils_data.select_iterable(cases, self.settings.gpe_cases)

        # Loop through reconstruction times
        for _age in _tqdm(
                _ages, 
                desc="Calculating GPE forces", 
                disable=(self.settings.logger.level in [logging.INFO, logging.DEBUG] or not PROGRESS_BAR)
            ):
            # Loop through gpe cases
            for key, entries in _iterable.items():
                if self.settings.options[key]["GPE torque"]:
                    # Select points
                    _data = self.data[_age][key].copy()

                    # Define plateIDs if not provided
                    _plateIDs = utils_data.select_plateIDs(plateIDs, _data.plateID.unique())

                    # Select points
                    if plateIDs is not None:
                        _data = _data[_data.plateID.isin(_plateIDs)]

                    if _data.empty:
                        logging.info(f"No valid points found for case {key} at age {_age} Ma.")
                        continue
                        
                    # Calculate GPE force
                    computed_data = utils_calc.compute_GPE_force(
                        _data,
                        seafloor_grid[_age].seafloor_age,
                        self.settings.options[key],
                        self.settings.mech,
                    )

                    # Enter sampled data back into the DataFrame
                    self.data[_age][key].loc[_data.index] = computed_data.copy()
                    
                    # Copy to other entries
                    if len(entries) > 1:
                        cols = [
                            "lithospheric_mantle_thickness",
                            "crustal_thickness",
                            "water_depth",
                            "U",
                            "GPE_force_lat",
                            "GPE_force_lon",
                            "GPE_force_mag",
                        ]
                        self.data[_age] = utils_data.copy_values(
                            self.data[_age], 
                            key, 
                            entries,
                            cols,
                        )

    def calculate_mantle_drag_force(
            self,
            ages: Optional[Union[int, float, List[Union[int, float]], _numpy.ndarray]] = None,
            cases: Optional[Union[str, List[str]]] = None,
            plateIDs: Optional[Union[int, float, List[Union[int, float]], _numpy.ndarray]] = None,
            PROGRESS_BAR: bool = True,
        ):
        """
        Function to compute mantle drag force acting at points.

        :param ages:            ages of interest (default: None)
        :type ages:             float, int, list, numpy.ndarray
        :param cases:           cases of interest (default: None)
        :type cases:            str, list
        :param plateIDs:        plateIDs of interest (default: None)
        :type plateIDs:         list, numpy.ndarray
        """
        # Define ages if not provided
        _ages = utils_data.select_ages(ages, self.settings.ages)
        
        # Define cases if not provided
        if cases == "reconstructed":
            _iterable = utils_data.select_iterable(None, self.settings.reconstructed_cases)
        elif cases == "synthetic":
            _iterable = utils_data.select_iterable(None, self.settings.synthetic_cases)
        else:
            _iterable = utils_data.select_iterable(cases, self.settings.mantle_drag_cases)

        # Loop through reconstruction times
        for _age in _tqdm(
                _ages, 
                desc="Calculating mantle drag forces", 
                disable=(self.settings.logger.level in [logging.INFO, logging.DEBUG] or not PROGRESS_BAR)
            ):
            # Loop through gpe cases
            for key, entries in _iterable.items():
                if self.settings.options[key]["Mantle drag torque"] and self.settings.options[key]["Reconstructed motions"]:
                    # Select points
                    _data = self.data[_age][key]

                    # Define plateIDs if not provided
                    _plateIDs = utils_data.select_plateIDs(plateIDs, _data.plateID.unique())

                    # Select points
                    if plateIDs is not None:
                        _data = _data[_data.plateID.isin(_plateIDs)]

                    if _data.empty:
                        logging.info(f"No valid points found for case {key} at age {_age} Ma.")
                        continue
                        
                    # Calculate mantle force
                    computed_data = utils_calc.compute_mantle_drag_force(
                        _data,
                        self.settings.options[key],
                        self.settings.constants,
                    )

                    # Enter sampled data back into the DataFrame
                    self.data[_age][key].loc[_data.index] = computed_data
                    
                    # Copy to other entries
                    cols = [
                        "mantle_drag_force_lat",
                        "mantle_drag_force_lon",
                        "mantle_drag_force_mag",
                    ]
                    self.data[_age] = utils_data.copy_values(
                        self.data[_age], 
                        key, 
                        entries,
                        cols,
                    )

    def calculate_residual_force(
            self,
            ages: Optional[Union[int, float, List[Union[int, float]], _numpy.ndarray]] = None,
            cases: Optional[Union[str, List[str]]] = None,
            plateIDs: Optional[Union[int, float, List[Union[int, float]], _numpy.ndarray]] = None,
            residual_torque: Optional[Dict] = None,
            PROGRESS_BAR: bool = True,
        ):
        """
        Function to calculate residual torque along trenches.

        :param ages:            ages of interest (default: None)
        :type ages:             float, int, list, numpy.ndarray
        :param cases:           cases of interest (default: None)
        :type cases:            str, list
        :param plateIDs:        plateIDs of interest (default: None)
        :type plateIDs:         list, numpy.ndarray
        :param residual_torque: residual torque along trenches (default: None)
        :type residual_torque:  dict
        """
        # Define ages if not provided
        _ages = utils_data.select_ages(ages, self.settings.ages)

        # Define cases if not provided
        if cases == "reconstructed":
            _, _cases = self.settings.reconstructed_cases.items()
        elif cases == "synthetic":
            _, _cases = self.settings.synthetic_cases.items()
        else:
            _cases = utils_data.select_cases(cases, self.settings.cases)

        # Loop through ages and cases
        for _case in _tqdm(
                _cases,
                desc="Calculating residual forces at points",
                disable=(self.settings.logger.level in [logging.INFO, logging.DEBUG] or not PROGRESS_BAR)
            ):
            if self.settings.options[_case]["Reconstructed motions"]:
                for _age in _ages:
                    # Select plateIDs
                    _plateIDs = utils_data.select_plateIDs(plateIDs, self.data[_age][_case]["plateID"].unique())

                    for _plateID in _plateIDs:
                        if (
                            isinstance(residual_torque, dict)
                            and _age in residual_torque.keys()
                            and _case in residual_torque[_age].keys()
                            and isinstance(residual_torque[_age][_case], _pandas.DataFrame)
                        ):
                            # Get stage rotation from the provided DataFrame in the dictionary
                            _residual_torque = residual_torque[_age][_case][residual_torque[_age][_case].plateID == _plateID]
                        
                        # Make mask for plate
                        mask = self.data[_age][_case]["plateID"] == _plateID

                        if mask.sum() == 0:
                            logging.info(f"No valid points found for age {_age}, case {_case}, and plateID {_plateID}.")
                            continue
                                                
                        # Compute velocities
                        forces = utils_calc.compute_residual_force(
                            self.data[_age][_case].loc[mask],
                            _residual_torque,
                            plateID_col = "plateID",
                            weight_col = "segment_area",
                        )

                        # Store velocities
                        self.data[_age][_case].loc[mask, "residual_force_lat"] = forces[0]
                        self.data[_age][_case].loc[mask, "residual_force_lon"] = forces[1]
                        self.data[_age][_case].loc[mask, "residual_force_mag"] = forces[2]
                        self.data[_age][_case].loc[mask, "residual_force_azi"] = forces[3]

    def save(
            self,
            ages: Optional[Union[int, float, List[Union[int, float]], _numpy.ndarray]] = None,
            cases: Optional[Union[str, List[str]]] = None,
            plateIDs: Optional[Union[int, float, List[Union[int, float]], _numpy.ndarray]] = None,
            file_dir: Optional[str] = None,
            PROGRESS_BAR: bool = True,
        ):
        """
        Function to save the 'Points' object.
        Data of the points object is saved to .parquet files.

        :param ages:            ages of interest (default: None)
        :type ages:             float, int, list, numpy.ndarray
        :param cases:           cases of interest (default: None)
        :type cases:            str, list
        :param plateIDs:        plateIDs of interest (default: None)
        :type plateIDs:         list, numpy.ndarray
        :param file_dir:        directory to store files (default: None)
        :type file_dir:         str
        """
        # Define ages if not provided
        _ages = utils_data.select_ages(ages, self.settings.ages)

        # Define cases if not provided
        _cases = utils_data.select_cases(cases, self.settings.cases)
        
        # Get file dir
        _file_dir = self.settings.dir_path if file_dir is None else file_dir

        # Loop through ages
        for _age in _tqdm(
                _ages, 
                desc="Saving Points", 
                disable=(self.settings.logger.level in [logging.INFO, logging.DEBUG] or not PROGRESS_BAR)
            ):            
            # Loop through cases
            for _case in _cases:
                # Define plateIDs if not provided
                _plateIDs = utils_data.select_plateIDs(plateIDs, self.data[_age][_case].plateID.unique())

                # Select data
                _data = self.data[_age][_case][self.data[_age][_case].plateID.isin(_plateIDs)]

                # Save data
                utils_data.DataFrame_to_parquet(
                    _data,
                    "Points",
                    self.settings.name,
                    _age,
                    _case,
                    _file_dir,
                )

        logging.info(f"Points saved to {self.settings.dir_path}")

    def export(
            self,
            ages: Optional[Union[int, float, List[Union[int, float]], _numpy.ndarray]] = None,
            cases: Optional[Union[str, List[str]]] = None,
            plateIDs: Optional[Union[int, float, List[Union[int, float]], _numpy.ndarray]] = None,
            file_dir: Optional[str] = None,
            PROGRESS_BAR: bool = True,
        ):
        """
        Function to export the 'Points' object.
        Data of the points object is saved to .csv files.

        :param ages:            ages of interest (default: None)
        :type ages:             float, int, list, numpy.ndarray
        :param cases:           cases of interest (default: None)
        :type cases:            str, list
        :param plateIDs:        plateIDs of interest (default: None)
        :type plateIDs:         list, numpy.ndarray
        :param file_dir:        directory to store files (default: None)
        :type file_dir:         str
        """
        # Define ages if not provided
        _ages = utils_data.select_ages(ages, self.settings.ages)

        # Define cases if not provided
        _cases = utils_data.select_cases(cases, self.settings.cases)
        
        # Get file dir
        _file_dir = self.settings.dir_path if file_dir is None else file_dir

        # Loop through ages
        for _age in _tqdm(
                _ages, 
                desc="Exporting Points", 
                disable=(self.settings.logger.level in [logging.INFO, logging.DEBUG] or not PROGRESS_BAR)
            ):            
            # Loop through cases
            for _case in _cases:
                # Define plateIDs if not provided
                _plateIDs = utils_data.select_plateIDs(plateIDs, self.data[_age][_case].plateID.unique())

                # Select data
                _data = self.data[_age][_case][self.data[_age][_case].plateID.isin(_plateIDs)]

                # Export data
                utils_data.DataFrame_to_csv(
                    _data,
                    "Points",
                    self.settings.name,
                    _age,
                    _case,
                    _file_dir,
                )

        logging.info(f"Points exported to {self.settings.dir_path}")