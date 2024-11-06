import logging
from copy import deepcopy
from typing import Dict, List, Optional, Union

import numpy as _numpy
import gplately as _gplately
import xarray as _xarray
from tqdm import tqdm as _tqdm

from . import utils_data, utils_init
from .settings import Settings

class Grids():
    """
    Class to hold gridded data.
    
    Seafloor grids contain lithospheric age and, optionally, sediment thickness.
    Continental grids contain lithospheric thickness and, optionally, crustal thickness.
    Velocity grids contain plate velocity data.

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
    :param seafloor_age_grids:  seafloor age grids (default: None)
    :type seafloor_age_grids:   dict, xarray.Dataset
    :param sediment_grids:      sediment thickness grids (default: None)
    :type sediment_grids:       dict, xarray.Dataset
    :param continental_grids:   continental crust thickness grids (default: None)
    :type continental_grids:    dict, xarray.Dataset
    :param velocity_grids:      velocity grids (default: None)
    :type velocity_grids:       dict, xarray.Dataset
    :param DEBUG_MODE:          flag to enable debug mode (default: False)
    :type DEBUG_MODE:           bool
    :param PARALLEL_MODE:       flag to enable parallel mode (default: False)
    :type PARALLEL_MODE:        bool
    """
    def __init__(
            self,
            settings = None,
            reconstruction = None,
            rotation_file = None,
            topology_file = None,
            polygon_file = None,
            reconstruction_name = None,
            ages = None,
            cases_file = None,
            cases_sheet = "Sheet1",
            files_dir = None,
            seafloor_age_grids = None,
            sediment_grids = None,
            continental_grids = None,
            velocity_grids = None,
            DEBUG_MODE = False,
            PARALLEL_MODE = False,
        ):
        """
        Constructor for the `Grids` class.
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
        self.data = {_age: None for _age in self.settings.ages}

        # Initialise dictionary to store grids
        self.seafloor_age = {_age: None for _age in self.settings.ages}

        # Load seafloor grids
        for _age in _tqdm(self.settings.ages, desc="Loading grids", disable=self.settings.logger.level==logging.INFO):
            if seafloor_age_grids is not None and _age in seafloor_age_grids.keys() and isinstance(seafloor_age_grids[_age], _xarray.Dataset):
                # If the seafloor is present in the provided dictionary, copy
                logging.info(f"Loading seafloor age grid for {_age} Ma.")
                self.seafloor_age[_age] = seafloor_age_grids[_age]

            else:
                logging.info(f"Downloading seafloor age grid for {_age} Ma.")
                self.seafloor_age[_age] = utils_data.get_seafloor_age_grid(
                    self.settings.name,
                    _age,
                )

            # Make sure that the coordinates and variables are named correctly
            if len(self.seafloor_age[_age].data_vars) == 1 and "z" in self.seafloor_age[_age].data_vars:
                self.seafloor_age[_age] = utils_data.rename_coordinates_and_variables(self.seafloor_age[_age], "z", "seafloor_age")
                
        # Store sediment, continental and velocity grids, if provided, otherwise initialise empty dictionaries to store them at a later stage.
        self.sediment = {_age: None for _age in self.settings.ages}
        if isinstance(sediment_grids, Dict):
            for _age in _tqdm(self.settings.ages, desc="Loading sediment grids", disable=self.settings.logger.level==logging.INFO):
                if _age in sediment_grids.keys() and isinstance(sediment_grids[_age], _xarray.Dataset):
                    # If the sediment is present in the provided dictionary, copy
                    logging.info(f"Loading sediment grid for {_age} Ma.")
                    self.sediment[_age] = sediment_grids[_age]

                    # Make sure that the coordinates and variables are named correctly
                    self.sediment[_age] = utils_data.rename_coordinates_and_variables(self.sediment[_age], "z", "sediment_thickness")

        self.continent = continental_grids if continental_grids else None
        if continental_grids is Dict:
            for _age in _tqdm(self.settings.ages, desc="Loading sediment grids", disable=self.settings.logger.level==logging.INFO):
                if _age in continental_grids.keys() and isinstance(continental_grids[_age], _xarray.Dataset):
                    # If the sediment is present in the provided dictionary, copy
                    logging.info(f"Loading sediment grid for {_age} Ma.")
                    self.continent[_age] = continental_grids[_age]

                    # Make sure that the coordinates and variables are named correctly
                    self.continent[_age] = utils_data.rename_coordinates_and_variables(self.sediment[_age], "z", "continental_thickness")

        # Initialise dictionary to store velocity grids
        # The entries are empty because the velocity is interpolated from point data
        self.velocity = {_age: {_case: None for _case in self.settings.cases} for _age in self.settings.ages}

    def __str__(self):
        return f"Grids is a class that contains data and methods for working with (reconstructed) global grids."
    
    def __repr__(self):
        return self.__str__()

    def add_grid(
            self,
            input_grids,
            variable_name = "new_grid",
            grid_type = "seafloor_age",
            target_variable = "z",
            mask_continents = False,
            interpolate = True,
            prefactor = 1.,
        ):
        """
        Function to add another grid of a variable to the seafloor grid.
        The grids should be organised in a dictionary with each item being an xarray.Dataset with each key being the corresponding reconstruction age, or a single xarray.Dataset, in which case it will be stored without an age.
        'mask_continents' is a boolean that determines whether or not to cut the grids to the seafloor. It should only be used for grids that only describe the seafloor, e.g. marine sediment distributions, and not e.g. continental erosion rate grids.
        """

        # Check if the attribute exists and is initially None
        if getattr(self, grid_type) is None:
            # Initialize to the type of input_grids
            if isinstance(input_grids, dict):
                setattr(self, grid_type, {})
            else:
                setattr(self, grid_type, _xarray.Dataset())

        # Access the storage dictionary directly (not creating a local copy)
        new_grids = getattr(self, grid_type)

        # If input_grids is a single xarray.Dataset, store it directly
        if isinstance(input_grids, _xarray.Dataset):
            if target_variable in input_grids.variables:
                # Copy the input grid to avoid modifying the original
                _input_grids = deepcopy(input_grids)

                # Rename coordinates and variables
                _input_grids = utils_data.rename_coordinates_and_variables(_input_grids, target_variable, variable_name)

                # Store the grid with the prefactor applied
                _input_grids[variable_name] *= prefactor

                # Update the attribute
                new_grids[variable_name] = _input_grids

            else:
                print(f"Target variable '{target_variable}' does not exist in the input grids.")

        # If input_grids is a dictionary of xarray.Datasets, loop through the ages to store the grids
        elif isinstance(input_grids, Dict):
            # Loop through ages to load, interpolate, and store variables in the specified grid
            for age in self.settings.ages:
                # Check if target_variable exists in input grids
                if age in input_grids and target_variable in input_grids[age].variables:
                    # Copy the dataset to avoid modifying the original
                    _input_grid = deepcopy(input_grids[age])

                    # Rename coordinates and variables
                    _input_grid = utils_data.rename_coordinates_and_variables(_input_grid, target_variable, variable_name)

                    if interpolate:
                        # Interpolate input grids to the resolution of the seafloor age grid
                        _input_grid = _input_grid.interp_like(self.seafloor_age[age]["seafloor_age"], method="nearest")
                        _input_grid[variable_name] *= prefactor

                    # Align grids if mask_continents is True
                    if mask_continents and self.seafloor_age[age] is _xarray.Dataset:
                        mask = {}
                        for variable_1 in self.seafloor_age[age].data_vars:
                            mask[variable_1] = _numpy.isnan(self.seafloor_age[age][variable_1].values)

                            # Apply masks to all grids
                            for variable_2 in self.seafloor_age[age].data_vars:
                                _input_grid[variable_2] = self.seafloor_age[age][variable_2].where(~mask[variable_1])

                    # Update the dictionary stored in the attribute
                    new_grids[age] = _input_grid

                else:
                    print(f"Target variable '{target_variable}' does not exist in the input grids for {age} Ma.")
        
        else:
            raise ValueError("Input grids should be either a single xarray.Dataset or a dictionary of xarray.Datasets.")
        
        # Ensure the modified new_grids is saved back to the object's attribute
        setattr(self, grid_type, new_grids)
        logging.info(f"{grid_type} updated:", getattr(self, grid_type))
    
    def generate_velocity_grid(
            self,
            ages: Union[int, float],
            cases: Optional[str],
            point_data: Dict[str, _numpy.ndarray],
            components: Union[str, List[str]] = None,
        ):
        """
        Function to generate a velocity grid.
        """
        # Define ages, if not provided
        _ages = utils_data.select_ages(ages, self.settings.ages)

        # Define cases, if not provided
        _cases = utils_data.select_cases(cases, self.settings.cases)

        # Define components, if not provided
        _components = components if components else ["velocity_lat", "velocity_lon", "velocity_mag", "spin_rate_mag"]
        _components = [_components] if isinstance(_components, str) else _components

        # Loop through the ages
        for _age in _tqdm(_ages, desc="Generating velocity grids", disable=self.settings.logger.level==logging.INFO):
            # Loop through the cases
            for _case in _cases:
                if _age in point_data and _case in point_data[_age]:
                    logging.info(f"Generating velocity grid for {_age} Ma and case {_case}.")

                    # Initialise a dictionary to store DataArrays for each component
                    data_arrays = {}

                    # Interpolate the different components of the velocity to the resolution of the seafloor age grid
                    for _component in _components:
                        if _component in point_data[_age][_case]:
                            logging.info(f"Generating velocity grid for {_age} Ma and case {_case}.")

                            # Get the component longitude and latitude
                            lon = point_data[_age][_case]["lon"].unique()
                            lat = point_data[_age][_case]["lat"].unique()

                            # Reshape the component data to 2D
                            data = point_data[_age][_case][_component].values.reshape(len(lat), len(lon))
                            
                            # Create a DataArray for this component
                            data_arrays[_component] = _xarray.DataArray(
                                data=data,
                                coords={"lat": lat, "lon": lon},
                                dims=["lat", "lon"],
                            )

                    # Create a Dataset from the collected DataArrays
                    dataset = _xarray.Dataset(data_arrays)

                    # Store the dataset in the velocity grid
                    self.velocity[_age][_case] = dataset
    
    def interpolate_data_to_grid(
            self,
            age: Union[int, float],
            lat: Union[float, List[float], _numpy.ndarray],
            lon: Union[float, List[float], _numpy.ndarray],
            data: Union[float, List[float], _numpy.ndarray],
            case: Optional[str] = None,
            grid_type: str = "velocity",
        ):
        """
        Function to interpolate data to the resolution of the seafloor age grid.
        """
        # Convert inputs to numpy arrays if they are lists
        lat = _numpy.asarray(lat)
        lon = _numpy.asarray(lon)
        data = _numpy.asarray(data)
        
        # Get unique values of lat and lon
        lat = _numpy.unique(lat)
        lon = _numpy.unique(lon)

        # Check if the data is 2D
        if len(data.shape) != 2:
            try:
                data = data.reshape(len(lat), len(lon))
            except ValueError:
                raise ValueError("Data should be 2D with dimensions of lat and lon.")

        # Check if the attribute exists and is initially None
        if getattr(self, grid_type) is None:
            if case is not None:
                setattr(self, grid_type, {age: {case: None}})
            else:
                setattr(self, grid_type, {age: {}})

        # Initialise the dictionary to store the new grids
        new_grids = getattr(self, grid_type)

        # Make an xarray.Dataset with the data
        dataset = _xarray.Dataset(
            {
                grid_type: (["lat", "lon"], data),
            },
            coords={
                "lat": (["lat"], lat),
                "lon": (["lon"], lon),
            },
        )

        if case is not None:
            # Store the dataset with the case
            new_grids[age][case] = dataset

        else:
            # Store the dataset without the case
            new_grids[age] = dataset

        # Ensure the modified new_grids is saved back to the object's attribute
        setattr(self, grid_type, new_grids)
        logging.info(f"{grid_type} updated:", getattr(self, grid_type))

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
        self.save_continent(ages, cases, file_dir)

        # Save velocity grid
        self.save_velocity(ages, cases, file_dir)

    def save_seafloor_age(
            self,
            ages: Union[None, List[int], List[float], _numpy.ndarray] = None,
            file_dir: Optional[str] = None,
        ):
        """
        Function to save the the seafloor age grid.
        """
        self.save_grid(self.seafloor_age, "Seafloor_age", ages, None, file_dir)

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
        _ages = utils_data.select_ages(ages, self.settings.ages)

        # Define cases, if not provided
        _cases = utils_data.select_cases(cases, self.settings.cases)

        # Get file dir
        _file_dir = self.settings.dir_path if file_dir is None else file_dir
        
        # Loop through ages
        for _age in _tqdm(_ages, desc=f"Saving {type} grids", disable=self.settings.logger.level==logging.INFO):
            if data[_age] is Dict:
                # Loop through cases
                for _case in _cases:
                    if data[_age][_case] is _xarray.Dataset:
                        utils_data.Dataset_to_netcdf(
                            data[_age][_case],
                            type,
                            self.settings.name,
                            _age,
                            _case,
                            _file_dir,
                        )
    
            else:
                if data[_age] is _xarray.Dataset:
                    utils_data.Dataset_to_netcdf(
                            data[_age],
                            type,
                            self.settings.name,
                            _age,
                            None,
                            _file_dir,
                        )