import logging
from typing import Dict, List, Optional, Union

import gplately as _gplately
import numpy as _numpy
import xarray as _xarray
from tqdm import tqdm as _tqdm

import utils_data, utils_calc, utils_init
from settings import Settings

class Points:
    def __init__(
            self,
            settings: Optional[Settings] = None,
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
                    for entries in entry:
                        if entry not in available_cases:
                            self.data[_age][entry] = self.data[_age][available_cases[0]].copy()
                else:
                    # Initialise missing data
                    if not resolved_geometries or key not in resolved_geometries.keys():
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
        self.calculate_velocities()

        # Set flags for computed torques
        self.sampled_seafloor = False
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
        _ages = utils_data.select_ages(ages, self.settings.ages)
        
        # Define cases if not provided
        _cases = utils_data.select_cases(cases, self.settings.point_cases)

        # Loop through ages and cases
        for _age in _ages:
            for _case in _cases:
                for plateID in self.data[_age][_case].plateID.unique():
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
                    mask = self.data[_age][_case].plateID == plateID
                                            
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
                    self.data[_age][_case].loc[mask, f"v_lat"] = velocities[0]
                    self.data[_age][_case].loc[mask, f"v_lon"] = velocities[1]
                    self.data[_age][_case].loc[mask, f"v_mag"] = velocities[2]
                    self.data[_age][_case].loc[mask, f"omega"] = velocities[3]

    def sample_seafloor_ages(
            self,
            ages: Optional[Union[_numpy.ndarray, List, float, int]] = None,
            cases: Optional[Union[List[str], str]] = None,
            plateIDs: Optional[Union[List[int], List[float], _numpy.ndarray]] = None,
            seafloor_grids: Optional[Dict] = None,
        ):
        """
        Samples seafloor age at points.
        """
        # Sample grid
        self.sample_grid(
            ages,
            cases,
            plateIDs,
            seafloor_grids,
            vars,
        )

        # Set sampling flag to true
        self.sampled_seafloor = True

    def sample_grid(
            self,
            ages: Optional[Union[_numpy.ndarray, List, float, int]] = None,
            cases: Optional[Union[List[str], str]] = None,
            plateIDs: Optional[Union[List[int], List[float], _numpy.ndarray]] = None,
            grids: Optional[Dict] = None,
            vars: Optional[Union[str, List[str]]] = ["seafloor_age"],
            sampling_coords: Optional[List[str]] = ["lat", "lon"],
            cols: Optional[Union[str, List[str]]] = ["seafloor_age"],
        ):
        """
        Samples any grid at points.
        """
        # Define ages if not provided
        _ages = utils_data.select_ages(ages, self.settings.ages)
        
        # Define cases if not provided
        _iterable = utils_data.select_iterable(cases, self.settings.gpe_cases)

        # Define variables if not provided
        if vars is not None and isinstance(vars, str):
            _vars = [vars]
        elif vars is not None and isinstance(vars, list):
            _vars = vars
        else:
            _vars = []

        # Loop through valid times
        for _age in _tqdm(_ages, desc="Sampling points", disable=self.settings.logger.level == logging.INFO):
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
                        [_col],
                    )
    
    def calculate_gpe_force(
            self,
            ages: Optional[Union[_numpy.ndarray, List, float, int]] = None,
            cases: Optional[Union[List[str], str]] = None,
            plateIDs: Optional[Union[List[int], List[float], _numpy.ndarray]] = None,
            seafloor_grid: Optional[Dict] = None,
        ):
        """
        Function to compute gravitational potential energy (GPE) torque.
        """
        # Define ages if not provided
        _ages = utils_data.select_ages(ages, self.settings.ages)
        
        # Define cases if not provided
        _iterable = utils_data.select_iterable(cases, self.settings.gpe_cases)

        # Loop through reconstruction times
        for i, _age in _tqdm(enumerate(_ages), desc="Computing GPE forces", disable=(self.settings.logger.level==logging.INFO)):
            # Loop through gpe cases
            for key, entries in _iterable.items():
                if self.settings.options[key]["GPE torque"]:
                    # Select points
                    _data = self.data[_age][key]

                    # Define plateIDs if not provided
                    _plateIDs = utils_data.select_plateIDs(plateIDs, _data.plateID.unique())

                    # Select points
                    if plateIDs is not None:
                        _data = _data[_data.plateID.isin(_plateIDs)]

                    # Calculate GPE force
                    _data = utils_calc.compute_GPE_force(
                        _data,
                        seafloor_grid[_age].seafloor_age,
                        self.settings.options[key],
                        self.settings.mech,
                    )

                    # Enter sampled data back into the DataFrame
                    self.data[_age][key].loc[_data.index] = _data
                    
                    # Copy to other entries
                    cols = [
                        "lithospheric_mantle_thickness",
                        "crustal_thickness",
                        "water_depth",
                        "U",
                        "GPE_force_lat",
                        "GPE_force_lon",
                    ]
                    self.data[_age] = utils_data.copy_values(
                        self.data[_age], 
                        key, 
                        entries,
                        cols,
                    )

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

    def save(
            self,
            ages: Union[None, List[int], List[float], _numpy.ndarray] = None,
            cases: Union[None, str, List[str]] = None,
            plateIDs: Union[None, List[int], List[float], _numpy.ndarray] = None,
            file_dir: Optional[str] = None,
        ):
        """
        Function to save the 'Points' object.
        """
        # Define ages if not provided
        _ages = utils_data.select_ages(ages, self.settings.ages)

        # Define cases if not provided
        _cases = utils_data.select_cases(cases, self.settings.cases)
        
        # Get file dir
        _file_dir = self.settings.dir_path if file_dir is None else file_dir

        # Loop through ages
        for _age in _tqdm(_ages, desc="Saving Points", disable=self.settings.logger.level==logging.INFO):
            # Loop through cases
            for _case in _cases:
                utils_data.DataFrame_to_parquet(
                    self.data[_age][_case],
                    "Points",
                    self.settings.name,
                    _age,
                    _case,
                    _file_dir,
                )

        logging.info(f"Points saved to {self.settings.dir_path}")

    def export(
            self,
            ages: Union[None, List[int], List[float], _numpy.ndarray] = None,
            cases: Union[None, str, List[str]] = None,
            plateIDs: Union[None, List[int], List[float], _numpy.ndarray] = None,
            file_dir: Optional[str] = None,
        ):
        """
        Function to export the 'Points' object.
        Data of the points object is saved to .csv files.
        """
        # Define ages if not provided
        _ages = utils_data.select_ages(ages, self.settings.ages)

        # Define cases if not provided
        _cases = utils_data.select_cases(cases, self.settings.cases)
        
        # Get file dir
        _file_dir = self.settings.dir_path if file_dir is None else file_dir

        # Loop through ages
        for _age in _tqdm(_ages, desc="Exporting Points", disable=self.settings.logger.level==logging.INFO):
            # Loop through cases
            for _case in _cases:
                utils_data.DataFrame_to_csv(
                    self.data[_age][_case],
                    "Points",
                    self.settings.name,
                    _age,
                    _case,
                    _file_dir,
                )

        logging.info(f"Points exported to {self.settings.dir_path}")