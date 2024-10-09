import logging
from typing import Dict, List, Optional, Union

import numpy as _numpy
import gplately as _gplately
import xarray as _xarray
from tqdm import tqdm as _tqdm

import utils_data, utils_init
from settings import Settings

class Grids():
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
            seafloor_age_grids: Optional[Dict] = None,
            sediment_grids: Optional[Dict] = None,
            continental_grids: Optional[Dict] = None,
            velocity_grids: Optional[Dict] = None,
            DEBUG_MODE: Optional[bool] = False,
            PARALLEL_MODE: Optional[bool] = False,
        ):
        """
        Object to hold gridded data.
        Seafloor grids contain lithospheric age and, optionally, sediment thickness.
        Continental grids contain lithospheric thickness and, optionally, crustal thickness.
        Velocity grids contain plate velocity data.
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

        # Initialise dictionary to store grids
        self.seafloor_age = {_age: None for _age in self.settings.ages}

        # Load seafloor grids
        for _age in _tqdm(self.settings.ages, desc="Loading grids", disable=self.settings.logger.level==logging.INFO):
            if seafloor_age_grids is not None and _age in seafloor_age_grids.keys():
                # If the seafloor is present in the provided dictionary, copy
                self.seafloor_age[_age] = seafloor_age_grids[_age]

                # Make sure that the coordinates and variables are named correctly
                self.rename_coordinates_and_variables(self.seafloor_age[_age], "seafloor_age")

            else:
                self.seafloor_age[_age] = utils_data.get_seafloor_age_grid(
                    self.settings.name,
                    _age,
                )

        # Store sediment, continental and velocity grids, if provided, otherwise initialise empty dictionaries to store them at a later stage.
        self.sediment = sediment_grids if sediment_grids else None
        self.continent = continental_grids if continental_grids else None
        self.velocity = velocity_grids if velocity_grids else None

    def __str__(self):
        return f"Plato grids object with seafloor, continental, and velocity grids."
    
    def __repr__(self):
        return self.__str__()
    
    def rename_coordinates_and_variables(
            grid: _xarray.Dataset,
            variable: Optional["str"],
        ):
        """
        Function to rename coordinates and variables 
        """
        # Clean up the grid a little
        if "lat" in grid.coords:
            grid = grid.rename({"lat": "latitude"})
        if "lon" in grid.coords:
            grid = grid.rename({"lon": "longitude"})
        if variable and "z" in grid.data_vars:
            grid = grid.rename({"z": "seafloor_age"})

    def array2data_array(
            self,
            lats: Union[list, _numpy.ndarray],
            lons: Union[list, _numpy.ndarray],
            data: Union[list, _numpy.ndarray],
            var_name: str,
        ):
        """
        Interpolates data to the resolution seafloor grid.
        
        :param lats: Latitude values for the data.
        :type lats: list or numpy.ndarray
        :param lons: Longitude values for the data.
        :type lons: list or numpy.ndarray
        :param data: Data values to be interpolated.
        :type data: list or numpy.ndarray
        :param var_name: Name of the variable.
        :type var_name: str
        """
        # Convert to numpy arrays
        lats = _numpy.asarray(lats)
        lons = _numpy.asarray(lons)
        data = _numpy.asarray(data)

        # Check if the data is the right shape
        if lats.shape == data.shape:
            lats = _numpy.unique(lats.flatten())
        
        if lons.shape == data.shape:
            lons = _numpy.unique(lons.flatten())

        # Create the grid
        data_array = _xarray.DataArray(
            data,
            coords = {
                "lat": lats,
                "lon": lons
            },
            dims = ["lat", "lon"],
            name = var_name
        )

        return data_array

    def data_arrays2dataset(
            self,
            data_arrays: dict,
            grid_name: str,
        ):
        """
        Creates a grid object from a dictionary of data arrays.

        :param data_arrays: Dictionary of data arrays.
        :type data_arrays: dict
        :param grid_name: Name of the grid.
        :type grid_name: str
        """
        # Create the grid
        grid = _xarray.Dataset(
            data_vars = data_arrays
        )

        # Dynamically assign the grid to an attribute using the var_name
        setattr(self, grid_name, grid)

    def save_all(
        self,
        ages: Union[None, List[int], List[float], _numpy.ndarray] = None,
        cases: Union[None, str, List[str]] = None,
        file_dir: Optional[str] = None,
        ):
        """
        Function to save all the grids
        """
        # Save seafloor grid
        self.save_seafloor_age(ages, file_dir)

        # Save sediment grid
        self.save_sediment(ages, cases, file_dir)

        # Save continental grid
        self.save_continent(ages, file_dir)

        # Save velocity grid
        self.save_velocity(ages, file_dir)

    def save_seafloor_age(
            self,
            ages: Union[None, List[int], List[float], _numpy.ndarray] = None,
            file_dir: Optional[str] = None,
        ):
        """
        Function to save the the seafloor age grid.
        """
        self.save_grid(self.seafloor_age, "Seafloor", ages, None, file_dir)

    def save_sediment(
            self,
            ages: Union[None, List[int], List[float], _numpy.ndarray] = None,
            cases: Union[None, str, List[str]] = None,
            file_dir: Optional[str] = None,
        ):
        """
        Function to save the the sediment grid.
        """
        if self.sediment is not None:
            self.save_grid(self.sediment, "Sediment", ages, cases, file_dir)

    def save_continent(
            self,
            ages: Union[None, List[int], List[float], _numpy.ndarray] = None,
            cases: Union[None, str, List[str]] = None,
            file_dir: Optional[str] = None,
        ):
        """
        Function to save the the continental grid.
        """
        # Check if grids exists
        if self.continent is not None:
            self.save_grid(self.continent, "Continent", ages, cases, file_dir)

    def save_velocity(
            self,
            ages: Union[None, List[int], List[float], _numpy.ndarray] = None,
            cases: Union[None, str, List[str]] = None,
            file_dir: Optional[str] = None,
        ):
        """
        Function to save the the velocity grid.
        """
        # Check if grids exists
        if self.velocity is not None:
            self.save_grid(self.velocity, "Velocity", ages, cases, file_dir)
        
    def save_grid(
            self,
            data: Dict,
            type: str,
            ages: Union[None, List[int], List[float], _numpy.ndarray] = None,
            cases: Union[None, str, List[str]] = None,
            file_dir: Optional[str] = None,
        ):
        """
        Function to save a grid
        """
        # Define ages, if not provided
        _ages = utils_data.get_ages(ages, self.settings.ages)

        # Define cases, if not provided
        _cases = utils_data.get_cases(cases, self.settings.cases)

        # Get file dir
        _file_dir = self.settings.dir_path if file_dir is None else file_dir

        # Loop through ages
        for _age in _tqdm(_ages, desc=f"Saving {type} grids", disable=self.settings.logger.level==logging.INFO):
            if cases is not None:
                # Loop through cases
                for _case in _cases:
                    utils_data.Dataset_to_netcdf(
                        data[_age][_case],
                        type,
                        self.settings.name,
                        _age,
                        file_dir,
                        _case,
                    )
            else:
                utils_data.Dataset_to_netcdf(
                        data[_age],
                        type,
                        self.settings.name,
                        _age,
                        file_dir,
                    )