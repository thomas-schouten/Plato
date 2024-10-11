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
        Class to store and manipulate data on slabs.
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

        # Loop through times
        for _age in _tqdm(self.settings.ages, desc="Loading data", disable=self.settings.logger.level == logging.INFO):
            # Load available data
            for key, entries in self.settings.slab_cases.items():
                # Make list to store available cases
                available_cases = []

                # Try to load all DataFrames
                for entry in entries:
                    self.data[_age][entry] = utils_data.DataFrame_from_parquet(
                        self.settings.dir_path,
                        "Slabs",
                        self.settings.name,
                        entry,
                        _age
                    )
                    # Store the cases for which a DataFrame could be loaded
                    if self.data[_age][entry] is not None:
                        available_cases.append(entry)
                
                # Check if any DataFrames were loaded
                if len(available_cases) > 0:
                    # Copy all DataFrames from the available case        
                    for entries in entry:
                        if entry not in available_cases:
                            self.data[_age][entry] = self.data[_age][available_cases[0]].copy()
                else:
                    # Initialise missing data
                    if not resolved_geometries or key not in resolved_geometries.keys():
                        resolved_geometries = utils_data.get_topology_geometries(
                            self.reconstruction, _age, self.settings.options[self.settings.cases[0]]["Anchor plateID"]
                        )

                    # Initialise missing data
                    self.data[_age][key] = utils_data.get_slab_data(
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
        self.calculate_velocities()

        # Calculate total slab length as a function of age and case
        self.total_slab_length = _numpy.zeros((len(self.settings.ages), len(self.settings.slab_pull_cases)))
        for i, _age in enumerate(self.settings.ages):
            for j, _case in enumerate(self.settings.slab_pull_cases):
                self.total_slab_length[i] = self.data[_age][_case].trench_segment_length.sum()

        # Set flag for sampling slabs and upper plates
        self.sampled_slabs = False
        self.sampled_arcs = False

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
        _ages = utils_data.select_ages(ages, self.settings.ages)
        
        # Define cases if not provided
        _cases = utils_data.select_cases(cases, self.settings.cases)

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

    def sample_slab_seafloor_ages(
            self,
            ages: Optional[Union[_numpy.ndarray, List, float, int]] = None,
            cases: Optional[Union[List[str], str]] = None,
            plateIDs: Optional[Union[List[int], List[float], _numpy.ndarray]] = None,
            seafloor_grids: Optional[Dict] = None,
        ):
        """
        Samples seafloor age at slabs.
        """
        # Sample grid
        self.sample_grid(
            ages,
            cases,
            plateIDs,
            seafloor_grids,
            plate = "lower",
            vars = ["seafloor_age"],
            cols = ["slab_age"],
        )

        # Set sampling flag to true
        self.sampled_seafloor_at_slabs = True

    def sample_arc_seafloor_ages(
            self,
            ages: Optional[Union[_numpy.ndarray, List, float, int]] = None,
            cases: Optional[Union[List[str], str]] = None,
            plateIDs: Optional[Union[List[int], List[float], _numpy.ndarray]] = None,
            seafloor_grids: Optional[Dict] = None,
        ):
        """
        Samples seafloor age at slabs.
        """
        # Ensure variables is a list
        if isinstance(vars, str):
            vars = [vars]
        
        # Sample grid
        self.sample_grid(
            ages,
            cases,
            plateIDs,
            seafloor_grids,
            plate = "upper",
            vars = ["seafloor_age"],
            cols = ["arc_age"],
        )

        # Set sampling flag to true
        self.sampled_seafloor_at_arcs = True

    def sample_grid(
            self,
            ages: Optional[Union[_numpy.ndarray, List, float, int]] = None,
            cases: Optional[Union[List[str], str]] = None,
            plateIDs: Optional[Union[List[int], List[float], _numpy.ndarray]] = None,
            grids: Optional[Dict] = None,
            plate: Optional[str] = "lower",
            vars: Optional[Union[str, List[str]]] = ["seafloor_age"],
            cols = ["slab_seafloor_age"],
        ):
        """
        Samples any grid at slabs.
        """
        # Define ages if not provided
        _ages = utils_data.select_ages(ages, self.settings.ages)
        
        # Define cases if not provided
        _iterable = utils_data.select_iterable(cases, self.settings.slab_cases)

        # Define variables if not provided
        if vars is not None and isinstance(vars, str):
            _vars = [vars]
        elif vars is not None and isinstance(vars, list):
            _vars = vars

        # Define sampling points
        if plate == "upper":
            sampling_coords = ["arc_sampling_lat", "arc_sampling_lon"]
        else:
            sampling_coords = ["slab_sampling_lat", "slab_sampling_lon"]

        # Loop through valid times
        for _age in _tqdm(_ages, desc="Sampling points", disable=self.settings.logger.level == logging.INFO):
            for key, entries in _iterable.items():
                # Define plateIDs if not provided
                _plateIDs = utils_data.select_plateIDs(plateIDs, self.data[_age][key][f"{plate}_plateID"].unique())

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
                _cols = _vars if not cols or len(cols) != len(_vars) else cols

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
                        [_col]
                    )

    def calculate_slab_pull_force(
            self,
            ages: Optional[Union[_numpy.ndarray, List, float, int]] = None,
            cases: Optional[Union[List[str], str]] = None,
            plateIDs: Optional[Union[List[int], List[float], _numpy.ndarray]] = None,
            seafloor_grid: Optional[Dict] = None,
        ):
        """
        Function to compute slab pull force along trenches.
        """
        # Define ages if not provided
        _ages = utils_data.select_ages(ages, self.settings.ages)
        
        # Define cases if not provided
        _iterable = utils_data.select_iterable(cases, self.settings.slab_pull_cases)

        # Loop through reconstruction times
        for _age in _tqdm(_ages, desc="Computing slab pull forces", disable=(self.settings.logger.level==logging.INFO)):
            # Loop through gpe cases
            for key, entries in _iterable.items():
                if self.settings.options[key]["Slab pull torque"]:
                    # Select points
                    _data = self.data[_age][key]

                    # Define plateIDs if not provided
                    _plateIDs = utils_data.select_plateIDs(plateIDs, _data.plateID.unique())

                    # Select points
                    if plateIDs is not None:
                        _data = _data[_data.plateID.isin(_plateIDs)]
                        
                    # Calculate GPE force
                    _data = utils_calc.compute_slab_pull_force(
                        _data,
                        seafloor_grid[_age].seafloor_age,
                        self.settings.options[key],
                        self.settings.mech,
                    )

                    # Enter sampled data back into the DataFrame
                    self.data[_age][key].loc[_data.index] = _data
                    
                    # Copy to other entries
                    cols = [
                        "slab_lithospheric_thickness",
                        "slab_crustal_thickness",
                        "slab_water_depth",
                        "slab_pull_force_lat",
                        "slab_pull_force_lon",
                    ]
                    self.data[_age] = utils_data.copy_values(
                        self.data[_age], 
                        key, 
                        entries,
                        cols,
                    )

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

    def save(
            self,
            ages: Union[None, List[int], List[float], _numpy.ndarray] = None,
            cases: Union[None, str, List[str]] = None,
            plateIDs: Union[None, List[int], List[float], _numpy.ndarray] = None,
            file_dir: Optional[str] = None,
        ):
        """
        Function to save the 'Slabs' object.
        Data of the 'Slabs' object is saved to .parquet files.
        """
        # Define ages if not provided
        _ages = utils_data.select_ages(ages, self.settings.ages)

        # Define cases if not provided
        _cases = utils_data.select_cases(cases, self.settings.cases)
        
        # Get file dir
        _file_dir = self.settings.dir_path if file_dir is None else file_dir

        # Loop through ages
        for _age in _tqdm(_ages, desc="Saving Slabs", disable=self.settings.logger.level==logging.INFO):
            # Loop through cases
            for _case in _cases:
                utils_data.DataFrame_to_parquet(
                    self.data[_age][_case],
                    "Slabs",
                    self.settings.name,
                    _age,
                    _case,
                    _file_dir,
                )

        logging.info(f"Slabs saved to {self.settings.dir_path}")

    def export(
            self,
            ages: Union[None, List[int], List[float], _numpy.ndarray] = None,
            cases: Union[None, str, List[str]] = None,
            plateIDs: Union[None, List[int], List[float], _numpy.ndarray] = None,
            file_dir: Optional[str] = None,
        ):
        """
        Function to export the 'Slabs' object.
        Data of the 'Slabs' object is exported to .csv files.
        """
        # Define ages if not provided
        _ages = utils_data.select_ages(ages, self.settings.ages)

        # Define cases if not provided
        _cases = utils_data.select_cases(cases, self.settings.cases)
        
        # Get file dir
        _file_dir = self.settings.dir_path if file_dir is None else file_dir

        # Loop through ages
        for _age in _tqdm(_ages, desc="Exporting Slabs", disable=self.settings.logger.level==logging.INFO):
            # Loop through cases
            for _case in _cases:
                utils_data.DataFrame_to_csv(
                    self.data[_age][_case],
                    "Slabs",
                    self.settings.name,
                    _age,
                    _case,
                    _file_dir,
                )

        logging.info(f"Slabs exported to {self.settings.dir_path}")