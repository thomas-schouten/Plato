import logging
from typing import Dict, List, Optional, Union

import geopandas as _geopandas
import gplately as _gplately
import numpy as _numpy
import pandas as _pandas
import xarray as _xarray
from tqdm import tqdm as _tqdm

from . import utils_data, utils_calc, utils_init
from .settings import Settings

class Slabs:
    """
    Class that contains all information for the slabs in a reconstruction.
    A `Slabs` object can be initialised in multiple ways:

    1.  The user can initialise a `Slabs` object from scratch by providing the reconstruction and the ages of interest.
        The reconstruction can be provided as a file with rotation poles, a file with topologies, and a file with polygons, or as one of the model name string identifiers for the models available on the GPlately DataServer (https://gplates.github.io/gplately/v1.3.0/#dataserver).
        
        Additionally, the user may specify the excel file with a number of different cases (combinations of options) to be considered.

    2.  Alternatively, the user can initialise a `Slabs` object by providing a `Settings` object and a `Reconstruction` object from a `Globe`, `Grids`, `Plates`, `Points` or `Slabs` object.
        Providing the settings from a `Slabs` object will allow the user to initialise a new `Slabs` object with the same settings as the original object.

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
            resolved_geometries = None,
            PARALLEL_MODE = False,
            DEBUG_MODE = False,
            CALCULATE_VELOCITIES = True,
        ):
        """
        Constructor for the `Slabs` class.
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
        for _age in _tqdm(self.settings.ages, desc="Loading slab data", disable=self.settings.logger.level == logging.INFO):
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
                    if not isinstance(resolved_geometries, Dict) or not isinstance(resolved_geometries.get(key), _geopandas.GeoDataFrame):
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

        # Calculate velocities along slabs
        if CALCULATE_VELOCITIES:
            self.calculate_velocities()

        # Calculate total slab length as a function of age and case
        self.total_slab_length = _numpy.zeros((len(self.settings.ages), len(self.settings.slab_pull_cases)))
        for i, _age in enumerate(self.settings.ages):
            for j, _case in enumerate(self.settings.slab_pull_cases):
                self.total_slab_length[i] = self.data[_age][_case].trench_segment_length.sum()

        # Set flag for sampling slabs and upper plates
        self.sampled_slabs = False
        self.sampled_arcs = False

    def __str__(self):
        return f"Slabs is a class that contains data and methods for working with (reconstructed) subduction zones."
    
    def __repr__(self):
        return self.__str__()

    def calculate_velocities(
            self,
            ages = None,
            cases = None,
            stage_rotation = None,
        ):
        """
        Function to compute velocities at slabs.

        :param ages:            ages of interest (default: None)
        :type ages:             float, int, list, numpy.ndarray
        :param cases:           cases of interest (default: None)
        :type cases:            str, list
        :param stage_rotation:  stage rotation model (default: None)
        :type stage_rotation:   dict
        """
        # Define ages if not provided
        _ages = utils_data.select_ages(ages, self.settings.ages)
        
        # Define cases if not provided
        _cases = utils_data.select_cases(cases, self.settings.cases)

        # Loop through ages and cases
        for _age in _ages:
            for plate in ["upper_plate", "lower_plate", "trench"]:
                plateID_col = f"{plate}ID" if plate != "trench" else "trench_plateID"
                for _case in _cases:
                    for plateID in self.data[_age][_case][plateID_col].unique():
                        if (
                            isinstance(stage_rotation, Dict)
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
                        mask = self.data[_age][_case][plateID_col] == plateID
                                                
                        # Compute velocities
                        velocities = utils_calc.compute_velocity(
                            self.data[_age][_case].loc[mask],
                            _stage_rotation,
                            self.settings.constants,
                            plateID_col,
                        )

                        # Store velocities
                        self.data[_age][_case].loc[mask, f"{plate}_velocity_lat"] = velocities[0]
                        self.data[_age][_case].loc[mask, f"{plate}_velocity_lon"] = velocities[1]
                        self.data[_age][_case].loc[mask, f"{plate}_velocity_mag"] = velocities[2]
                        self.data[_age][_case].loc[mask, f"{plate}_spin_rate_mag"] = velocities[4]

    def sample_slab_seafloor_ages(
            self,
            ages = None,
            cases = None,
            plateIDs = None,
            grids = None,
        ):
        """
        Samples seafloor age at slabs.
        This is a special case of the sample_grid function, 
        where the variable to sample from the grid is set to "seafloor_age" and the column to store the sampled values is set to "slab_seafloor_age".

        :param ages:            ages of interest (default: None)
        :type ages:             float, int, list, numpy.ndarray
        :param cases:           cases of interest (default: None)
        :type cases:            str, list
        :param plateIDs:        plateIDs of interest (default: None)
        :type plateIDs:         list, numpy.ndarray
        :param grids:           grids to sample from (default: None)
        :type grids:            dict[float, dict[str, xarray.Dataset]]
        """
        # Sample grid
        self.sample_grid(
            ages,
            cases,
            plateIDs,
            grids,
            plate = "lower",
            vars = ["seafloor_age"],
            cols = ["slab_seafloor_age"],
        )

        # Set sampling flag to true
        self.sampled_seafloor_at_slabs = True

    def sample_arc_seafloor_ages(
            self,
            ages = None,
            cases = None,
            plateIDs = None,
            grids = None,
        ):
        """
        Samples seafloor age at arcs.
        This is a special case of the sample_grid function, 
        where the variable to sample from the grid is set to "seafloor_age" and the column to store the sampled values is set to "arc_seafloor_age".

        :param ages:            ages of interest (default: None)
        :type ages:             float, int, list, numpy.ndarray
        :param cases:           cases of interest (default: None)
        :type cases:            str, list
        :param plateIDs:        plateIDs of interest (default: None)
        :type plateIDs:         list, numpy.ndarray
        :param grids:           grids to sample from (default: None)
        :type grids:            dict[float, dict[str, xarray.Dataset]]
        """
        # Sample grid
        self.sample_grid(
            ages,
            cases,
            plateIDs,
            grids,
            plate = "upper",
            vars = ["seafloor_age"],
            cols = ["arc_seafloor_age"],
        )

        # Set sampling flag to true
        self.sampled_seafloor_at_arcs = True

    def sample_slab_sediment_thickness(
            self,
            ages = None,
            cases = None,
            plateIDs = None,
            grids = None,
        ):
        """
        Samples sediment thickness at slabs.
        This function calls several internal functions to sample sediment thickness at slabs:

        1.  A a special case of the sample_grid function, where the variable to sample from the grid is set to None and the column to store the sampled values is set to "sediment_thickness".
            Settings the variable to sample from the grid to None will sample all variables in the grid specified in the case settings.

        2.  The set_continental_arc function, which will call the sample_arc_seafloor_ages function if the seafloor ages at arcs have not been sampled yet,
            and then sets whether a trench has a continental arc with additional manual overrides for known oceanic arcs that are masked on seafloor age grids of the Earthbyte reconstructions available on the GPlately DataServer.

        3.  A special case of the set_values function, where the column to set the values is set to "sediment_thickness" and the value to set is set to the value specified in the case settings.

        :param ages:            ages of interest (default: None)
        :type ages:             float, int, list, numpy.ndarray
        :param cases:           cases of interest (default: None)
        :type cases:            str, list
        :param plateIDs:        plateIDs of interest (default: None)
        :type plateIDs:         list, numpy.ndarray
        :param grids:           grids to sample from (default: None)
        :type grids:            dict[float, dict[str, xarray.Dataset]]
        """
        # Sample grid
        self.sample_grid(
            ages,
            cases,
            plateIDs,
            grids,
            plate = "lower",
            vars = None,
            cols = ["sediment_thickness"],
        )

        # Get continental arcs
        self.set_continental_arc(
            ages,
            cases,
            plateIDs,
        )

        # Set active margin sediment thickness
        self.set_values(
            ages,
            cases,
            plateIDs,
            cols = "sediment_thickness",
        )

        # Set sampling flag to true
        self.sampled_sediment_thickness_at_slabs = True

    def set_continental_arc(
            self,
            ages = None,
            cases = None,
            plateIDs = None,
        ):
        """
        Function to set whether a trench has a continental arc.
        This function calls the sample_arc_seafloor_ages function if the seafloor ages at arcs have not been sampled yet,
        and then sets whether a trench has a continental arc with additional manual overrides for known oceanic arcs that are masked on seafloor age grids of the Earthbyte reconstructions available on the GPlately DataServer.

        :param ages:            ages of interest (default: None)
        :type ages:             float, int, list, numpy.ndarray
        :param cases:           cases of interest (default: None)
        :type cases:            str, list
        :param plateIDs:        plateIDs of interest (default: None)
        :type plateIDs:         list, numpy.ndarray
        """
        # Check whether arc seafloor ages have been sampled
        if not self.sampled_seafloor_at_arcs:
            self.sample_arc_seafloor_ages(ages, cases, plateIDs)

        # Define ages if not provided
        _ages = utils_data.select_ages(ages, self.settings.ages)

        # Define cases if not provided
        _iterable = utils_data.select_iterable(cases, self.settings.slab_cases)

        # Loop through slab cases
        for key, entries in _iterable.items():
            # Loop through ages
            for _age in _ages:
                # Define plateIDs if not provided
                _plateIDs = utils_data.select_plateIDs(plateIDs, self.data[_age][key]["trench_plateID"].unique())

                # Select points
                _data = self.data[_age][key]
                if plateIDs is not None:
                    _data = _data[_data.trench_plateID.isin(_plateIDs)]

                # Set continental arc
                _data.loc[_data.arc_seafloor_age.isna() & ~_data.trench_plateID.isin(self.settings.oceanic_arc_plateIDs), "continental_arc"] = True

                # Enter sampled data back into the DataFrame
                self.data[_age][key].loc[_data.index] = _data

                # Copy to other entries
                if len(entries) > 1:
                    self.data[_age] = utils_data.copy_values(
                        self.data[_age], 
                        key, 
                        entries, 
                        ["continental_arc"],
                    )

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
        _iterable = utils_data.select_iterable(cases, self.settings.cases)

        # Define sampling points
        type = "arc" if plate == "upper" else "slab"

        # Loop through valid cases
        # Order of loops is flipped to skip cases where no grid needs to be sampled
        for key, entries in _tqdm(_iterable.items(), desc="Sampling slabs", disable=self.settings.logger.level == logging.INFO):
            # Skip if sediment grid is not sampled
            if cols == ["sediment_thickness"] and not self.settings.options[key]["Sample sediment grid"]:
                logging.info(f"Skipping sampling of sediment thickness for case {key}.")
                continue
            
            # Skip if erosion grid is not sampled
            if cols == ["erosion_rate"] and not self.settings.options[key]["Sample erosion grid"]:
                logging.info(f"Skipping sampling of erosion rate for case {key}.")
                continue

            # Loop through ages
            for _age in _ages:
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
                    elif isinstance(grids[_age], Dict) and key in grids[_age].keys() and isinstance(grids[_age][key], _xarray.Dataset):
                        _grid = grids[_age][key]
                
                if _grid is None:
                    logging.warning(f"No valid grid found for age {_age} and key {key}.")
                    continue  # Skip this iteration if no valid grid is found

                # Define variables and columns
                if cols == ["sediment_thickness"]:
                    # Specific case for sampling sediment thickness, with the variable name set to the one specified in the settings
                    _cols = [cols]
                    _vars = [self.settings.options[key]["Sample sediment grid"]]
                    logging.info(f"Sampling sediment thickness for case {key}.")

                elif cols == ["erosion_rate"]:
                    # Specific case for sampling erosion rate, with the variable name set to the one specified in the settings
                    _cols = [cols]
                    _vars = [self.settings.options[key]["Sample erosion grid"]]
                    logging.info(f"Sampling erosion rate for case {key}.")

                elif isinstance(cols, str) and isinstance(vars, list):
                    # General case for sampling a single variable with multiple columns
                    _cols = [cols]
                    _vars = vars
                    logging.info(f"Sampling {_vars} for case {key}.")

                elif isinstance(cols, str) and isinstance(vars, str):
                    # General case for sampling a single variable with a single column
                    _cols = [cols]
                    _vars = [vars]
                    logging.info(f"Sampling {_vars} for case {key}.")

                elif isinstance(cols, list) and isinstance(vars, list) and len(cols) == len(vars):
                    # General case for sampling multiple variables with multiple columns
                    _cols = cols
                    _vars = vars
                    logging.info(f"Sampling {_vars} for case {key}.")

                else:
                    # Default case for sampling all variables in the grid and using the variable names as column names
                    _cols = list(_grid.data_vars)
                    _vars = list(_grid.data_vars)
                    logging.info(f"Sampling {_vars} for case {key}.")

                # Sample grid at points for each variable
                for _col in _cols:
                    # Accumulate data if multiple variables are sampled for the same column
                    accumulated_data = _numpy.zeros_like(_data.lat.values)

                    for _var in _vars:
                        sampled_data = utils_calc.sample_grid(
                            _data[f"{type}_sampling_lat"],
                            _data[f"{type}_sampling_lon"],
                            _grid[_var],
                        )
                        accumulated_data += sampled_data

                    # Enter sampled data back into the DataFrame
                    self.data[_age][key].loc[_data.index, _col] = accumulated_data

                    # Copy to other entries
                    if len(entries) > 1:
                        self.data[_age] = utils_data.copy_values(
                            self.data[_age], 
                            key, 
                            entries, 
                            [_col]
                        )

    def set_values(
            self,
            ages: Optional[Union[_numpy.ndarray, List, float, int]] = None,
            cases: Optional[Union[List[str], str]] = None,
            plateIDs: Optional[Union[List[int], List[float], _numpy.ndarray]] = None,
            plate: Optional[str] = "lower",
            cols: Optional[Union[List[str], str]] = None,
            vals: Optional[Union[List[float], float]] = None,
        ):
        """
        Function to add values to the 'Slabs' object.
        """
        # Define ages if not provided
        _ages = utils_data.select_ages(ages, self.settings.ages)

        # Define cases if not provided
        _iterable = utils_data.select_iterable(cases, self.settings.cases)

        # Define whether to add values to slabs or arcs
        type = "arc" if plate == "upper" else "slab"

        # Loop through valid cases
        for key, entries in _tqdm(_iterable.items(), desc="Adding values", disable=(self.settings.logger.level==logging.INFO)):
            # Special case for active margin sediments
            if cols == "sediment_thickness" and vals == None:
                if not self.settings.options[key]["Active margin sediments"]:
                    logging.info(f"Skipping setting active margin sediments for case {key}.")
                    continue
                
                _cols = [cols]
                _vals = [self.settings.options[key]["Active margin sediments"]]
                logging.info(f"Setting active margin sediments for case {key}.")

            elif isinstance(cols, str) and isinstance(vals, float):
                _cols = [cols]
                _vals = [vals]

            elif isinstance(cols, list) and isinstance(vals, list) and len(cols) == len(vals):
                _cols = cols
                _vals = vals

            else:
                logging.warning(f"Invalid input for setting values for case {key}.")
                continue

            # Loop through ages
            for _age in _ages:
                # Define plateIDs if not provided
                _plateIDs = utils_data.select_plateIDs(plateIDs, self.data[_age][key][f"{plate}_plateID"].unique())

                # Select points
                _data = self.data[_age][key]
                if plateIDs is not None:
                    _data = _data[_data.plateID.isin(_plateIDs)]

                # Mask continental overriding plates if setting active margin sediment thickness
                if cols == "sediment_thickness" and vals == None:
                    _data = _data[_data.continental_arc == True]

                for _col, _val in zip(_cols, _vals):
                    _data[_col] = _val

                # Enter sampled data back into the DataFrame
                self.data[_age][key].loc[_data.index] = _data

                # Copy to other entries
                if len(entries) > 1:
                    self.data[_age] = utils_data.copy_values(
                        self.data[_age], 
                        key, 
                        entries, 
                        cols,
                    )

    def calculate_slab_pull_force(
            self,
            ages: Optional[Union[_numpy.ndarray, List, float, int]] = None,
            cases: Optional[Union[List[str], str]] = None,
            plateIDs: Optional[Union[List[int], List[float], _numpy.ndarray]] = None,
        ):
        """
        Function to compute slab pull force along trenches.
        """
        # Define ages if not provided
        _ages = utils_data.select_ages(ages, self.settings.ages)
        
        # Define cases if not provided
        _iterable = utils_data.select_iterable(cases, self.settings.slab_pull_cases)

        # Loop through valid cases
        # Order of loops is flipped to skip cases where no slab pull torque needs to be sampled
        for key, entries in _tqdm(_iterable.items(), desc="Computing slab pull forces", disable=(self.settings.logger.level==logging.INFO)):
            # Skip if slab pull torque is not sampled
            if self.settings.options[key]["Slab pull torque"]:                
                # Loop through ages
                for _age in _ages:
                    # Select points
                    _data = self.data[_age][key].copy()

                    # Define plateIDs if not provided
                    _plateIDs = utils_data.select_plateIDs(plateIDs, _data.lower_plateID.unique())

                    # Select points
                    if plateIDs is not None:
                        _data = _data[_data.lower_plateID.isin(_plateIDs)]
                        
                    # Calculate slab pull force
                    computed_data1 = utils_calc.compute_slab_pull_force(
                        _data,
                        self.settings.options[key],
                        self.settings.mech,
                    )

                    # Compute interface term
                    computed_data2 = utils_calc.compute_interface_term(
                        computed_data1,
                        self.settings.options[key],
                    )

                    # Enter sampled data back into the DataFrame
                    self.data[_age][key].loc[_data.index] = computed_data2
                    
                    # Copy to other entries
                    if len(entries) > 1:
                        cols = [
                            "slab_lithospheric_thickness",
                            "slab_crustal_thickness",
                            "slab_water_depth",
                            "shear_zone_width",
                            "sediment_fraction",
                            "slab_pull_force_lat",
                            "slab_pull_force_lon",
                            "slab_pull_force_mag",
                        ]
                        self.data[_age] = utils_data.copy_values(
                            self.data[_age], 
                            key, 
                            entries,
                            cols,
                        )

            # Inform the user that the slab pull forces have been calculated
            logging.info(f"Calculated slab pull forces for case {key} Ma.")

    def calculate_slab_bend_force(
            self,
            ages: Optional[Union[_numpy.ndarray, List, float, int]] = None,
            cases: Optional[Union[List[str], str]] = None,
            plateIDs: Optional[Union[List[int], List[float], _numpy.ndarray]] = None,
        ):
        """
        Function to compute slab bend force along trenches.
        """
        # Define ages if not provided
        _ages = utils_data.select_ages(ages, self.settings.ages)
        
        # Define cases if not provided
        _iterable = utils_data.select_iterable(cases, self.settings.slab_pull_cases)

        # Loop through valid cases
        # Order of loops is flipped to skip cases where no slab bend torque needs to be sampled
        for key, entries in _tqdm(_iterable.items(), desc="Computing slab bend forces", disable=(self.settings.logger.level==logging.INFO)):
            # Skip if slab pull torque is not sampled
            if self.settings.options[key]["Slab bend torque"]:
                # Loop through ages
                for _age in _ages:
                    # Select points
                    _data = self.data[_age][key]

                    # Define plateIDs if not provided
                    _plateIDs = utils_data.select_plateIDs(plateIDs, _data.lower_plateID.unique())

                    # Select points
                    if plateIDs is not None:
                        _data = _data[_data.lower_plateID.isin(_plateIDs)]
                        
                    # Calculate slab pull force
                    _data = utils_calc.compute_slab_bend_force(
                        _data,
                        self.settings.options[key],
                        self.settings.mech,
                    )

                    # Enter sampled data back into the DataFrame
                    self.data[_age][key].loc[_data.index] = _data
                    
                    # Copy to other entries
                    if len(entries) > 1:
                        cols = [
                            "slab_lithospheric_thickness",
                            "slab_crustal_thickness",
                            "slab_water_depth",
                            "slab_bend_force_lat",
                            "slab_bend_force_lon",
                        ]
                        self.data[_age] = utils_data.copy_values(
                            self.data[_age], 
                            key, 
                            entries,
                            cols,
                        )
            
            # Inform the user that the slab bend forces have been calculated
            logging.info(f"Calculated slab bend forces for case {key}.")

    def calculate_residual_force(
            self,
            ages: Optional[Union[_numpy.ndarray, List, float, int]] = None,
            cases: Optional[Union[List[str], str]] = None,
            plateIDs: Optional[Union[List[int], List[float], _numpy.ndarray]] = None,
            residual_torque: Optional[Dict] = None,
        ):
        """
        Function to calculate residual torque along trenches.
        """
        # Define ages if not provided
        _ages = utils_data.select_ages(ages, self.settings.ages)

        # Define cases if not provided
        _cases = utils_data.select_iterable(cases, self.settings.slab_pull_cases)

        # Loop through ages and cases
        for _age in _ages:
            for _case in _cases:
                # Select plateIDs
                _plateIDs = utils_data.select_plateIDs(plateIDs, self.data[_age][_case]["lower_plateID"].unique())

                for _plateID in _plateIDs:
                    if (
                        isinstance(residual_torque, Dict)
                        and _age in residual_torque.keys()
                        and _case in residual_torque[_age].keys()
                        and isinstance(residual_torque[_age][_case], _pandas.DataFrame)
                    ):
                        # Get stage rotation from the provided DataFrame in the dictionary
                        _residual_torque = residual_torque[_age][_case][residual_torque[_age][_case].plateID == _plateID]
                    
                    # Make mask for plate
                    mask = self.data[_age][_case]["lower_plateID"] == _plateID
                                            
                    # Compute velocities
                    forces = utils_calc.compute_residual_force(
                        self.data[_age][_case].loc[mask],
                        _residual_torque,
                        plateID_col = "lower_plateID",
                        weight_col = "trench_normal_azimuth",
                    )

                    # Store velocities
                    self.data[_age][_case].loc[mask, "residual_force_lat"] = forces[0]
                    self.data[_age][_case].loc[mask, "residual_force_lon"] = forces[1]
                    self.data[_age][_case].loc[mask, "residual_force_mag"] = forces[2]
                    self.data[_age][_case].loc[mask, "residual_force_azi"] = forces[3]

    def extract_data_through_time(
            self,
            ages: Optional[Union[_numpy.ndarray, List, float, int]] = None,
            cases: Optional[Union[List[str], str]] = None,
            plateIDs: Optional[Union[List[int], List[float], _numpy.ndarray]] = None,
            var: Optional[Union[List[str], str]] = "None",
        ):
        """
        Function to extract data on slabs through time as a pandas.DataFrame.
        """
        # Define ages if not provided
        _ages = utils_data.select_ages(ages, self.settings.ages)

        # Define cases if not provided
        _cases = utils_data.select_cases(cases, self.settings.cases)

        # Define plateIDs if not provided
        # Default is to select all major plates in the Müller et al. (2016) reconstruction
        _plateIDs = utils_data.select_plateIDs(
            plateIDs, 
            [101,   # North America
            201,    # South America
            301,    # Eurasia
            501,    # India
            801,    # Australia
            802,    # Antarctica
            901,    # Pacific
            902,    # Farallon
            911,    # Nazca
            919,    # Phoenix
            926,    # Izanagi
            ]
        )

        # Initialise dictionary to store results
        extracted_data = {case: None for case in _cases}

        # Loop through valid cases
        for _case in _cases:
            # Initialise DataFrame
            extracted_data[_case] = _pandas.DataFrame({
                "Age": _ages,
            })
            for _plateID in _plateIDs:
                # Initialise column for each plate
                extracted_data[_case][_plateID] = _numpy.nan

            for i, _age in enumerate(_ages):
                # Select data for the given age and case
                _data = self.data[_age][_case]

                # Loop through plateIDs
                for i, _plateID in enumerate(_plateIDs):
                    if _data.lower_plateID.isin([_plateID]).any():
                        # Hard-coded exception for the Indo-Australian plate for 20-43 Ma (which is defined as 801 in the Müller et al. (2016) reconstruction)
                        _plateID = 801 if _plateID == 501 and _age >= 20 and _age <= 43 else _plateID

                        # Extract data
                        value = _data[_data.lower_plateID == _plateID][var].values[0]

                        # Assign value to DataFrame if not zero
                        # This assumes that any of the variables of interest are never zero
                        if value != 0:
                            extracted_data[_case].loc[i, _plateID] = value

        # Return extracted data
        if len(_cases) == 1:
            # If only one case is selected, return the DataFrame
            return extracted_data[_cases[0]]
        else:
            # If multiple cases are selected, return the dictionary
            return extracted_data
        
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