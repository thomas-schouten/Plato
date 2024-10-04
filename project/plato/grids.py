from typing import Optional, Union

import numpy as _numpy
import xarray as _xarray

import utils_data

class Grids():
    def __init__(
            self,
            settings,
            seafloor_grids: Optional[dict] = None,
            continental_grids: Optional[dict] = None,
            velocity_grids: Optional[dict] = None
        ):
        """
        Object to hold gridded data.
        Seafloor grids contain lithospheric age and, optionally, sediment thickness.
        Continental grids contain lithospheric thickness and, optionally, crustal thickness.
        Velocity grids contain plate velocity data.
        """
        # Store the settings
        self.settings = settings

        # Store seafloor grid
        if seafloor_grids:
            self.seafloor = seafloor_grids
        else:
            self.seafloor = {}

            # Load or initialise seafloor
            self.seafloor = setup.load_grid(
                self.seafloor,
                self.settings.name,
                self.settings.ages,
                "Seafloor",
                self.settings.dir_path,
                DEBUG_MODE = self.settings.DEBUG_MODE
            )

        # Store continental and velocity grids, if provided, otherwise initialise empty dictionaries to store them at a later stage.
        self.continental = continental_grids if continental_grids else {}
        self.velocity = velocity_grids if velocity_grids else {}

    def __str__(self):
        return f"Plato grids object with seafloor, continental, and velocity grids."
    
    def __repr__(self):
        return self.__str__()

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