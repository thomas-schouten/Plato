# Standard libraries
import logging
from typing import Dict, List, Optional, Union

# Third-party libraries
import gplately as _gplately
import numpy as _numpy
import pandas as _pandas
from tqdm import tqdm as _tqdm

# Local libraries
from . import utils_data, utils_calc, utils_init
from .points import Points
from .settings import Settings

class Plates:
    """
    Class that contains all information for the plates in a reconstruction.
    A `Plates` object can be initialised in multiple ways:

    1.  The user can initialise a `Plates` object from scratch by providing the reconstruction and the ages of interest.
        The reconstruction can be provided as a file with rotation poles, a file with topologies, and a file with polygons, or as one of the model name string identifiers for the models available on the GPlately DataServer (https://gplates.github.io/gplately/v1.3.0/#dataserver).
        
        Additionally, the user may specify the excel file with a number of different cases (combinations of options) to be considered.

    2.  Alternatively, the user can initialise a `Plates` object by providing a `Settings` object and a `Reconstruction` object from a `Globe`, `Grids`, `Plates`, `Points` or `Slabs` object.
        Providing the settings from a `Plates` object will allow the user to initialise a new `Plates` object with the same settings as the original object.

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
    :param PROGRESS_BAR:        flag to enable the tqdm progress bar (default: True)
    :type PROGRESS_BAR:         bool
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
            PARALLEL_MODE: bool = False,
            DEBUG_MODE: bool = False,
            PROGRESS_BAR: bool = True,
        ):
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

        # Set shortcut to ages, cases and options
        self.ages = self.settings.ages
        self.cases = self.settings.cases
        self.options = self.settings.options
        
        # GEOMETRIES
        # Set up plate reconstruction object and initialise dictionaries to store resolved topologies and geometries
        self.resolved_topologies = {_age: {} for _age in self.settings.ages}
        self.resolved_geometries = {_age: {} for _age in self.settings.ages}

        # Load or initialise plate geometries
        for _age in _tqdm(
                self.settings.ages,
                desc="Loading plate geometries",
                disable=(self.settings.logger.level in [logging.INFO, logging.DEBUG] or not PROGRESS_BAR)
            ):
            # Load available data
            for key, entries in self.settings.plate_cases.items():
                # Make list to store available cases
                available_cases = []

                # Try to load all DataFrames
                for entry in entries:
                    self.resolved_geometries[_age][entry] = utils_data.GeoDataFrame_from_geoparquet(
                        self.settings.dir_path,
                        "Geometries",
                        self.settings.name,
                        _age,
                        entry
                    )

                    # Store the cases for which a DataFrame could be loaded
                    if self.resolved_geometries[_age][entry] is not None:
                        available_cases.append(entry)
                
                # Check if any DataFrames were loaded
                if len(available_cases) > 0:
                    # Copy all DataFrames from the available case        
                    for entry in entries:
                        if entry not in available_cases:
                            self.resolved_geometries[_age][entry] = self.resolved_geometries[_age][available_cases[0]].copy()
                else:
                    # Initialise missing geometries
                    self.resolved_geometries[_age][key] = utils_data.get_resolved_geometries(
                        self.reconstruction,
                        _age,
                        self.settings.options[key]["Anchor plateID"]
                    )

                    # Resolve topologies to use to get plates
                    # NOTE: This is done because some information is retrieved from the resolved topologies and some from the resolved geometries
                    #       This step could be sped up by extracting all information from the geopandas DataFrame, but so far this has not been the main bottleneck
                    self.resolved_topologies[_age][key] = utils_data.get_resolved_topologies(
                        self.reconstruction,
                        _age,
                        self.settings.options[key]["Anchor plateID"],
                    )

                    # Copy to matching cases
                    if len(entries) > 1:
                        for entry in entries[1:]:
                            self.resolved_geometries[_age][entry] = self.resolved_geometries[_age][key].copy()
                            self.resolved_topologies[_age][entry] = self.resolved_topologies[_age][key].copy()
        
        # DATA
        # Initialise data dictionary
        self.data = {age: {} for age in self.settings.ages}

        # Initialise dictionary to store data that was newly initialised
        self.NEW_DATA = {age: [] for age in self.settings.ages}

        # Loop through times
        for _age in _tqdm(
                self.settings.ages,
                desc="Loading plate data",
                disable=(self.settings.logger.level in [logging.INFO, logging.DEBUG] or not PROGRESS_BAR)
            ):
            # Load available data
            for key, entries in self.settings.plate_cases.items():
                # Make list to store available cases
                available_cases = []

                # Try to load all DataFrames
                for entry in entries:
                    self.data[_age][entry] = utils_data.DataFrame_from_parquet(
                        self.settings.dir_path,
                        "Plates",
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
                    # Initialise missing data
                    self.data[_age][key] = utils_data.get_plate_data(
                        self.reconstruction.rotation_model,
                        _age,
                        self.resolved_topologies[_age][key], 
                        self.settings.options[key],
                    )

                    # Set flag to for newly initialised data
                    self.NEW_DATA[_age].append(key)
                    
                    # Copy to matching cases
                    if len(entries) > 1:
                        for entry in entries[1:]:
                            self.data[_age][entry] = self.data[_age][key].copy()

    def __str__(self):
        return f"Plates is a class that contains data, geometries and methods for working with (reconstructed) plates."
    
    def __repr__(self):
        return self.__str__()
    
    def calculate_continental_fraction(
            self,
            points: Optional[Points] = None,
            seafloor_grids: Optional[Dict[float, Dict[str, _pandas.DataFrame]]] = None,
            ages: Optional[Union[int, float, List[Union[int, float]], _numpy.ndarray]] = None,
            cases: Optional[Union[str, List[str]]] = None,
            plateIDs: Optional[Union[int, float, List[Union[int, float]], _numpy.ndarray]] = None,
            PROGRESS_BAR: bool = True,
        ):
        """
        Function to calculate the continental fraction of the plates.

        This function can be called in multiple ways:

        1.  If no `Points` object is provided, but a `Grids` is, the function will initialise a `Points` object, sample the seafloor ages and calculate the continental fraction

        2.  If a `Points` object is provided, the function will calculate the continental fraction using the data in the `Points` object.

        3.  If ages, cases, and plateIDs are provided, the function will calculate the continental fraction for the specified ages, cases, and plateIDs.
            Otherwise, the function will calculate the continental fraction for all ages, cases, and plateIDs.

        :param points:          `Points` object (default: None)
        :type points:           plato.points.Points
        :param ages:            ages of interest (default: None)
        :type ages:             float, int, list, numpy.ndarray
        :param cases:           cases of interest (default: None)
        :type cases:            str, list
        :param plateIDs:        plateIDs of interest (default: None)
        :type plateIDs:         int, float, list, numpy.ndarray
        :param PROGRESS_BAR:    flag to enable the tqdm progress bar (default: True)
        :type PROGRESS_BAR:     bool
        """
        # Define ages if not provided
        _ages = utils_data.select_ages(ages, self.settings.ages)

        # Check if no points are passed, initialise Points object
        if points is None and seafloor_grids:
            # Initialise a Points object
            points = Points(
                settings = self.settings,
                reconstruction = self.reconstruction,
                resolved_geometries = self.resolved_geometries
            )
        
        # Define cases if not provided, default to GPE cases because it only depends on the grid spacing
        _iterable = utils_data.select_iterable(cases, self.settings.cases)

        # Loop through ages
        for _age in _tqdm(
                _ages, 
                desc="Calculating continental fractions",
                disable=(self.settings.logger.level in [logging.INFO, logging.DEBUG] or not PROGRESS_BAR)
            ):
            # Check if age in point data
            if _age in points.data.keys():
                # Loop through cases
                for key, entries in _iterable.items():
                    # Check if case in point data
                    if key not in points.data[_age].keys():
                        # Initialise a Points object for this age and case
                        points = Points(
                            settings = self.settings,
                            reconstruction = self.reconstruction,
                            ages = _age,
                            cases = key,
                            resolved_geometries = self.resolved_geometries,
                        )
                        logging.info(f"Initialised Points object for case {key} at {_age} Ma to calculate RMS velocities")

                        # Sample seafloor age
                        points.sample_seafloor_age(
                            seafloor_grids,
                            ages = _age,
                            cases = key,
                            PROGRESS_BAR = PROGRESS_BAR,
                        )

                    # Define plateIDs if not provided
                    _plateIDs = utils_data.select_plateIDs(
                        plateIDs,
                        self.data[_age][key].plateID,
                    )

                    # Loop through plateIDs
                    for _plateID in _plateIDs:
                        # Select point data
                        _data = points.data[_age][key]
                        _data = _data[_data.plateID == _plateID]

                        if _data.empty:
                            continue

                        # Calculate continental fraction for plate
                        self.data[_age][key].loc[self.data[_age][key]["plateID"] == _plateID, "continental_fraction"] = _data.loc[_data["seafloor_age"].isna(), "segment_area"].sum() / _data["segment_area"].sum()

                    # Copy values to other cases, if necessary
                    if len(entries) > 1:
                        self.data[_age] = utils_data.copy_values(
                            self.data[_age], 
                            key, 
                            entries, 
                            ["continental_fraction"], 
                        )

    def calculate_mean_lab_depth(
            self,
            points: Optional[Points] = None,
            ages: Optional[Union[int, float, List[Union[int, float]], _numpy.ndarray]] = None,
            cases: Optional[Union[str, List[str]]] = None,
            plateIDs: Optional[Union[int, float, List[Union[int, float]], _numpy.ndarray]] = None,
            PROGRESS_BAR: bool = True,
        ):
        """
        Function to calculate the continental fraction of the plates.

        :param points:          `Points` object (default: None)
        :type points:           plato.points.Points
        :param ages:            ages of interest (default: None)
        :type ages:             float, int, list, numpy.ndarray
        :param cases:           cases of interest (default: None)
        :type cases:            str, list
        :param plateIDs:        plateIDs of interest (default: None)
        :type plateIDs:         int, float, list, numpy.ndarray
        :param PROGRESS_BAR:    flag to enable the tqdm progress bar (default: True)
        :type PROGRESS_BAR:     bool
        """
        # Define ages if not provided
        _ages = utils_data.select_ages(ages, self.settings.ages)
        
        # Define cases if not provided, default to GPE cases because it only depends on the grid spacing
        _iterable = utils_data.select_iterable(cases, self.settings.cases)

        # Loop through ages
        for _age in _tqdm(
                _ages, 
                desc="Calculating mean LAB depths",
                disable=(self.settings.logger.level in [logging.INFO, logging.DEBUG] or not PROGRESS_BAR)
            ):
            # Check if age in point data
            if _age in points.data.keys():
                # Loop through cases
                for key, entries in _iterable.items():
                    # Define plateIDs if not provided
                    _plateIDs = utils_data.select_plateIDs(
                        plateIDs,
                        self.data[_age][key].plateID,
                    )

                    # Loop through plateIDs
                    for _plateID in _plateIDs:
                        # Select point data
                        _data = points.data[_age][key]
                        _data = _data[_data.plateID == _plateID]

                        if _data.empty:
                            continue

                        # Calculate continental fraction for plate
                        self.data[_age][key].loc[self.data[_age][key]["plateID"] == _plateID, "mean_LAB_depth"] = _numpy.sum(_data["LAB_depth"] * _data["segment_area"]) / _data["segment_area"].sum()

                    # Copy values to other cases, if necessary
                    if len(entries) > 1:
                        self.data[_age] = utils_data.copy_values(
                            self.data[_age], 
                            key, 
                            entries, 
                            ["mean_LAB_depth"], 
                        )
                    
    def calculate_rms_velocity(
            self,
            points: Optional[Points] = None,
            ages: Optional[Union[int, float, List[Union[int, float]], _numpy.ndarray]] = None,
            cases: Optional[Union[str, List[str]]] = None,
            plateIDs: Optional[Union[int, float, List[Union[int, float]], _numpy.ndarray]] = None,
            PROGRESS_BAR: bool = True,
        ):
        """
        Function to calculate the root mean square (RMS) velocity of the plates.

        This function can be called in multiple ways:

        1.  If no `Points` object is provided, the function will initialise a `Points` object and calculate the RMS velocity.

        2.  If a `Points` object is provided, the function will calculate the RMS velocity for using the data in the `Points` object.

        3.  If ages, cases, and plateIDs are provided, the function will calculate the RMS velocity for the specified ages, cases, and plateIDs.
            Otherwise, the function will calculate the RMS velocity for all ages, cases, and plateIDs.

        :param points:          `Points` object (default: None)
        :type points:           plato.points.Points
        :param ages:            ages of interest (default: None)
        :type ages:             float, int, list, numpy.ndarray
        :param cases:           cases of interest (default: None)
        :type cases:            str, list
        :param plateIDs:        plateIDs of interest (default: None)
        :type plateIDs:         int, float, list, numpy.ndarray
        :param PROGRESS_BAR:    flag to enable the tqdm progress bar (default: True)
        :type PROGRESS_BAR:     bool
        """
        # Define ages if not provided
        _ages = utils_data.select_ages(ages, self.settings.ages)

        # Check if no points are passed, initialise Points object
        if points is None:
            # Initialise a Points object
            points = Points(
                settings = self.settings,
                reconstruction = self.reconstruction,
                resolved_geometries = self.resolved_geometries
            )
        
        # Define cases if not provided
        if cases == "reconstructed" or cases == ["reconstructed"]:
            _iterable = utils_data.select_iterable(None, self.settings.reconstructed_cases)
        elif cases == "synthetic" or cases == ["synthetic"]:
            _iterable = utils_data.select_iterable(None, self.settings.synthetic_cases)
        else:
            _iterable = utils_data.select_iterable(cases, self.settings.slab_pull_cases)

        # Loop through ages
        for _age in _tqdm(
                _ages, 
                desc="Calculating RMS velocities",
                disable=(self.settings.logger.level in [logging.INFO, logging.DEBUG] or not PROGRESS_BAR)
            ):
            # Check if age in point data
            if _age in points.data.keys():
                # Loop through cases
                for key, entries in _iterable.items():
                    # Check if case in point data
                    if key not in points.data[_age].keys():
                        # Initialise a Points object for this age and case
                        points = Points(
                            settings = self.settings,
                            reconstruction = self.reconstruction,
                            ages = _age,
                            cases = key,
                            resolved_geometries = self.resolved_geometries,
                        )
                        logging.info(f"Initialised Points object for case {key} at {_age} Ma to calculate RMS velocities")

                    # Define plateIDs if not provided
                    _plateIDs = utils_data.select_plateIDs(
                        plateIDs,
                        self.data[_age][key].plateID,
                    )
                    
                    # Loop through plates
                    for _plateID in _plateIDs:
                        # Select points belonging to plate 
                        mask = points.data[_age][key].plateID == _plateID

                        if mask.sum() == 0:
                            logging.warning(f"No points found for plate {_plateID} for case {key} at {_age} Ma")
                            continue

                        # Calculate RMS velocity for plate
                        rms_velocity = utils_calc.compute_rms_velocity(
                            points.data[_age][key].segment_length_lat.values[mask],
                            points.data[_age][key].segment_length_lon.values[mask],
                            points.data[_age][key].velocity_mag.values[mask],
                            points.data[_age][key].velocity_azi.values[mask],
                            points.data[_age][key].spin_rate_mag.values[mask],
                        )

                        # Store RMS velocity components 
                        self.data[_age][key].loc[self.data[_age][key].plateID == _plateID, "velocity_rms_mag"] = rms_velocity[0]
                        self.data[_age][key].loc[self.data[_age][key].plateID == _plateID, "velocity_rms_azi"] = rms_velocity[1]
                        self.data[_age][key].loc[self.data[_age][key].plateID == _plateID, "spin_rate_rms_mag"] = rms_velocity[2]

                    if len(entries) > 1:
                        self.data[_age] = utils_data.copy_values(
                            self.data[_age], 
                            key, 
                            entries, 
                            ["velocity_rms_mag", "velocity_rms_azi", "spin_rate_rms_mag"], 
                        )

    def calculate_torque_on_plates(
            self,
            point_data: Dict[float, Dict[str, _pandas.DataFrame]],
            ages: Optional[Union[int, float, List[Union[int, float]], _numpy.ndarray]] = None,
            cases: Optional[Union[str, List[str]]] = None,
            plateIDs: Optional[Union[int, float, List[Union[int, float]], _numpy.ndarray]] = None,
            torque_var: str = "torque",
            PROGRESS_BAR: bool = True,
        ):
        """
        Function to calculate the torque on plates from the forces acting on a set of points on Earth.

        This function is used in the `PlateTorques` module to calculate the torque on plates arising from slab pull, slab bend, gravitational potential energy (GPE), and mantle drag.
        It can also be used directly if the user has a set of points on Earth with forces acting on them. Thes should be organised in a dictionary with the ages of interest as keys and the cases as subkeys.

        :param point_data:      dictionary with point data
        :type point_data:       dict
        :param ages:            ages of interest (default: None)
        :type ages:             float, int, list, numpy.ndarray
        :param cases:           cases of interest (default: None)
        :type cases:            str, list
        :param plateIDs:        plateIDs of interest (default: None)
        :type plateIDs:         int, float, list, numpy.ndarray
        :param torque_var:      variable to calculate torque for (default: "torque")
        :type torque_var:       str
        :param PROGRESS_BAR:    flag to enable the tqdm progress bar (default: True)
        :type PROGRESS_BAR:     bool
        """
        # Define ages if not provided
        _ages = utils_data.select_ages(ages, self.settings.ages)
     
        # Define cases if not provided, defaulting to the cases that are relevant for the torque variable
        if torque_var == "slab_pull":
            matching_cases = self.settings.cases
        elif torque_var == "slab_bend":
            matching_cases = self.settings.slab_bend_cases
        elif torque_var == "slab_suction":
            matching_cases = self.settings.cases
        elif torque_var == "GPE":
            matching_cases = self.settings.gpe_cases
        elif torque_var == "mantle_drag":
            matching_cases = self.settings.mantle_drag_cases
        
        # Define iterable, if cases not provided
        # Define cases if not provided
        if cases == "reconstructed":
            _iterable = utils_data.select_iterable(None, self.settings.reconstructed_cases)
        elif cases == "synthetic":
            _iterable = utils_data.select_iterable(None, self.settings.synthetic_cases)
        else:
            _iterable = utils_data.select_iterable(cases, matching_cases)

        # Define plateID column of point data
        if torque_var == "slab_pull" or torque_var == "slab_bend":
            point_data_plateID_col = "lower_plateID"
        elif torque_var == "slab_suction":
            point_data_plateID_col = "upper_plateID"
        else:
            "plateID"

        # Define columns to store torque and force components and store them in one list
        torque_cols = [f"{torque_var}_torque_" + axis for axis in ["x", "y", "z", "mag"]]
        force_cols = [f"{torque_var}_force_" + axis for axis in ["lat", "lon", "mag", "azi"]]
        cols = torque_cols + force_cols 

        # Loop through ages
        for _age in _tqdm(
                _ages,
                desc=f"Calculating {torque_var.replace('_', ' ')} torque on plates", 
                disable=(self.settings.logger.level in [logging.INFO, logging.DEBUG] or not PROGRESS_BAR)
            ):
            logging.info(f"Calculating torque on plates at {_age} Ma")
            for key, entries in _iterable.items():
                # Select data
                _plate_data = self.data[_age][key].copy()
                _point_data = point_data[_age][key].copy()

                # Define plateIDs if not provided
                _plateIDs = utils_data.select_plateIDs(plateIDs, _plate_data.plateID.unique())

                # Select points
                if plateIDs is not None:
                    _plate_data = _plate_data[_plate_data.plateID.isin(_plateIDs)]
                    _point_data = _point_data[_point_data[point_data_plateID_col].isin(_plateIDs)]

                if torque_var == "slab_pull" or torque_var == "slab_bend":
                    selected_points_plateID = _point_data.lower_plateID.values
                    selected_points_area = _point_data.trench_segment_length.values
                elif torque_var == "slab_suction":
                    selected_points_plateID = _point_data.upper_plateID.values
                    selected_points_area = _point_data.trench_segment_length.values
                else:
                    selected_points_plateID = _point_data.plateID.values
                    selected_points_area = _point_data.segment_length_lat.values * _point_data.segment_length_lon.values

                # Calculate torques
                computed_data = utils_calc.compute_torque_on_plates(
                    _plate_data,
                    _point_data.lat.values,
                    _point_data.lon.values,
                    selected_points_plateID,
                    _point_data[f"{torque_var}_force_lat"].values, 
                    _point_data[f"{torque_var}_force_lon"].values,
                    selected_points_area,
                    self.settings.constants,
                    torque_var = torque_var,
                )
    
                # Enter sampled data back into the DataFrame
                self.data[_age][key].loc[_plate_data.index] = computed_data.copy()

                # Copy DataFrames, if necessary
                if len(entries) > 1:
                    self.data[_age] = utils_data.copy_values(
                        self.data[_age], 
                        key, 
                        entries, 
                        cols, 
                    )

    def calculate_driving_torque(
            self,
            ages: Optional[Union[int, float, List[Union[int, float]], _numpy.ndarray]] = None,
            cases: Optional[Union[str, List[str]]] = None,
            plateIDs: Optional[Union[int, float, List[Union[int, float]], _numpy.ndarray]] = None,
            PROGRESS_BAR: bool = True,
        ):
        """
        Function to calculate the driving torque acting on each plate.

        The driving torque is the sum of the torques arising from the slab pull and gravitational potential energy (GPE) force.

        :param ages:            ages of interest (default: None)
        :type ages:             float, int, list, numpy.ndarray
        :param cases:           cases of interest (default: None)
        :type cases:            str, list
        :param plateIDs:        plateIDs of interest (default: None)
        :type plateIDs:         int, float, list, numpy.ndarray
        :param torque_var:      variable to calculate torque for (default: "torque")
        :type torque_var:       str
        :param PROGRESS_BAR:    flag to enable the tqdm progress bar (default: True)
        :type PROGRESS_BAR:     bool
        """
        # Define ages if not provided
        _ages = utils_data.select_ages(ages, self.settings.ages)
        
        # Define cases if not provided
        if cases == "reconstructed" or cases == ["reconstructed"]:
            _cases = self.settings.reconstructed_cases
        elif cases == "synthetic" or cases == ["synthetic"]:
            _cases = self.settings.synthetic_cases
        else:
            _cases = utils_data.select_cases(cases, self.settings.cases)

        # Inform the user that the driving torques are being calculated
        logging.info("Computing driving torques...")

        # Loop through ages
        for i, _age in enumerate(_tqdm(
                _ages,
                desc="Calculating driving torque",
                disable=(self.settings.logger.level in [logging.INFO, logging.DEBUG] or not PROGRESS_BAR)
            )):
            # Loop through cases
            for _case in _cases:
                # Select plates
                _data = self.data[_age][_case].copy()
                
                # Select plateIDs and mask
                _plateIDs = utils_data.select_plateIDs(plateIDs, _data.plateID)
                mask = _data.plateID.isin(_plateIDs)

                # Calculate driving torque
                computed_data = utils_calc.sum_torque(_data[mask], "driving", self.settings.constants)

                # Enter sampled data back into the DataFrame
                self.data[_age][_case].loc[mask] = computed_data
        
        # Inform the user that the driving torques have been calculated
        logging.info("Driving torques calculated!")

    def calculate_residual_torque(
            self,
            ages: Optional[Union[int, float, List[Union[int, float]], _numpy.ndarray]] = None,
            cases: Optional[Union[str, List[str]]] = None,
            plateIDs: Optional[Union[int, float, List[Union[int, float]], _numpy.ndarray]] = None,
            PROGRESS_BAR: bool = True,
        ):
        """
        Function to calculate the residual torque acting on each plate.

        The residual torque is the sum of the torques arising from driving (slab pull and gravitational potential energy (GPE) force) and resistive forces (slab bend and mantle drag force).

        :param ages:            ages of interest (default: None)
        :type ages:             float, int, list, numpy.ndarray
        :param cases:           cases of interest (default: None)
        :type cases:            str, list
        :param plateIDs:        plateIDs of interest (default: None)
        :type plateIDs:         int, float, list, numpy.ndarray
        :param torque_var:      variable to calculate torque for (default: "torque")
        :type torque_var:       str
        :param PROGRESS_BAR:    flag to enable the tqdm progress bar (default: True)
        :type PROGRESS_BAR:     bool
        """
        # Define ages if not provided
        _ages = utils_data.select_ages(ages, self.settings.ages)
        
        # Define cases if not provided
        if cases == "reconstructed" or cases == ["reconstructed"]:
            _cases = self.settings.reconstructed_cases
        elif cases == "synthetic" or cases == ["synthetic"]:
            _cases = self.settings.synthetic_cases
        else:
            _cases = utils_data.select_cases(cases, self.settings.cases)

        # Inform the user that the driving torques are being calculated
        logging.info(f"Computing residual torques...")
        
        # Loop through cases
        # Order of loops is flipped to skip cases where no slab pull torque needs to be sampled
        for _case in _tqdm(
                _cases,
                desc="Calculating residual torque",
                disable=(self.settings.logger.level in [logging.INFO, logging.DEBUG] or not PROGRESS_BAR)
            ):
            # Skip if reconstructed motions are enabled
            if not self.settings.options[_case]["Reconstructed motions"]:
                continue

            # Loop through ages
            for _age in _ages:
                # Select plates
                _data = self.data[_age][_case].copy()
                
                # Select plateIDs and mask
                _plateIDs = utils_data.select_plateIDs(plateIDs, _data.plateID)
                mask = _data.plateID.isin(_plateIDs)

                # Calculate driving torque
                computed_data = utils_calc.sum_torque(_data[mask], "residual", self.settings.constants)

                # Enter sampled data back into the DataFrame
                self.data[_age][_case].loc[mask] = computed_data.copy()

        # Inform the user that the driving torques have been calculated
        logging.info(f"Residual torques for case calculated!")
                
    def calculate_synthetic_velocity(
            self,
            ages: Optional[Union[int, float, List[Union[int, float]], _numpy.ndarray]] = None,
            cases: Optional[Union[str, List[str]]] = None,
            plateIDs: Optional[Union[int, float, List[Union[int, float]], _numpy.ndarray]] = None,
            PROGRESS_BAR: bool = True,
            RECONSTRUCTED_CASES: bool = False,
        ):
        """
        Function to calculate synthetic velocity of plates.

        The synthetic velocity is calculated by summing all torques acting on a plate, except for the mantle drag torque.

        :param ages:            ages of interest (default: None)
        :type ages:             float, int, list, numpy.ndarray
        :param cases:           cases of interest (default: None)
        :type cases:            str, list
        :param plateIDs:        plateIDs of interest (default: None)
        :type plateIDs:         int, float, list, numpy.ndarray
        :param torque_var:      variable to calculate torque for (default: "torque")
        :type torque_var:       str
        :param PROGRESS_BAR:    flag to enable the tqdm progress bar (default: True)
        :type PROGRESS_BAR:     bool
        """
        # Define ages if not provided
        _ages = utils_data.select_ages(ages, self.settings.ages)

        # Define cases if not provided
        if cases == "reconstructed" or cases == ["reconstructed"]:
            return
        elif cases == "synthetic" or cases == ["synthetic"]:
            _cases = self.settings.synthetic_cases
        else:
            _cases = utils_data.select_cases(cases, self.settings.cases)

        # Loop through cases
        # Order of loops is flipped to skip cases where no slab pull torque needs to be sampled
        for _case in _tqdm(
                _cases,
                desc="Calculating synthetic velocity", 
                disable=(self.settings.logger.level in [logging.INFO, logging.DEBUG] or not PROGRESS_BAR)
            ):
            # Skip if reconstructed motions are enabled
            if not self.settings.options[_case]["Reconstructed motions"] or RECONSTRUCTED_CASES:
                # Inform the user that the driving torques are being calculated
                logging.info(f"Computing synthetic velocity for case {_case}")

                # Loop through ages
                for _age in _ages:
                    # Select plates
                    _data = self.data[_age][_case].copy()
                    
                    # Select plateIDs and mask
                    _plateIDs = utils_data.select_plateIDs(plateIDs, _data.plateID)
                    
                    if plateIDs is not None:
                        _data = _data[_data.plateID.isin(_plateIDs)]

                    # Calculate synthetic mantle drag torque
                    computed_data1 = utils_calc.sum_torque(_data, "mantle_drag", self.settings.constants)

                    # Calculate synthetic stage rotation
                    computed_data2 = utils_calc.compute_synthetic_stage_rotation(computed_data1, self.settings.options[_case])

                    # Enter sampled data back into the DataFrame
                    self.data[_age][_case].loc[_data.index] = computed_data2.copy()

    def rotate_torque(
            self,
            reference_rotations: _gplately.pygplates.RotationModel,
            reference_plates: Dict[str, Dict[str, _pandas.DataFrame]],
            torque = "slab_pull_torque",
            ages: Optional[Union[int, float, List[Union[int, float]], _numpy.ndarray]] = None,
            cases: Optional[Union[str, List[str]]] = None,
            plateIDs: Optional[Union[int, float, List[Union[int, float]], _numpy.ndarray]] = None,
            reference_case = None,
            PROGRESS_BAR: bool = True,
        ):
        """
        Function to rotate a torque vector stored in another the Plates object to the reference frame of this Plates object.

        :param reference_rotations:     reference rotations to use for rotation
        :type reference_rotations:      dict, xarray.Dataset
        :param reference_plates:        reference plates to use for rotation
        :type reference_plates:         dict, xarray.Dataset
        :param torque:                  torque to rotate (default: "slab_pull_torque")
        :type torque:                   str
        :param ages:                    ages of interest (default: None)
        :type ages:                     float, int, list, numpy.ndarray
        :param cases:                   cases of interest (default: None)
        :type cases:                    str, list
        :param plateIDs:                plateIDs of interest (default: None)
        :type plateIDs:                 int, float, list, numpy.ndarray
        :param torque_var:              variable to calculate torque for (default: "torque")
        :type torque_var:               str
        :param PROGRESS_BAR:            flag to enable the tqdm progress bar (default: True)
        :type PROGRESS_BAR:             bool
        """
        # Define ages if not provided
        _ages = utils_data.select_ages(ages, self.settings.ages)

        # Define cases if not provided
        if cases == "reconstructed" or cases == ["reconstructed"]:
            _cases = self.settings.reconstructed_cases
        elif cases == "synthetic" or cases == ["synthetic"]:
            _cases = self.settings.synthetic_cases
        else:
            _cases = utils_data.select_cases(cases, self.settings.cases)

        # Check if reference case is provided, otherwise default to first case in list
        if reference_case == None:
            reference_case = list(reference_plates.data.keys())[0]

        # Loop through all reconstruction times
        for _age in _tqdm(
                _ages,
                desc="Rotating torques",
                disable=(self.settings.logger.level in [logging.INFO, logging.DEBUG] or not PROGRESS_BAR)
            ):
            # If the case is not given, select the first one from the list in the provided reference plates.
            if reference_case == None:
                reference_case = list(reference_plates.data[_age].keys())[0]

            # Loop through all cases
            for _case in _cases:
                # Select cases that require rotation
                if self.settings.options[_case]["Reconstructed motions"] and self.settings.options[_case]["Mantle drag torque"]:
                    # Select plates
                    _data = self.data[_age][_case].copy()
                    
                    # Select plateIDs and mask
                    _plateIDs = utils_data.select_plateIDs(plateIDs, _data.plateID)
                    
                    if plateIDs is not None:
                        _data = _data[_data.plateID.isin(_plateIDs)]

                    for _plateID in _plateIDs:
                        # Rotate x, y, and z components of torque
                        _data.loc[_data.plateID == _plateID, [torque + "_x", torque + "_y", torque + "_z"]] = utils_calc.rotate_torque(
                            _plateID,
                            reference_plates.data[_age][_case].loc[reference_plates.data[_age][_case].plateID == _plateID, [torque + "_x", torque + "_y", torque + "_z"]].copy(),
                            reference_rotations,
                            self.reconstruction.rotation_model,
                            _age,
                            self.settings.constants,
                        )

                        # Copy magnitude of torque
                        _data.loc[_data.plateID == _plateID, torque + "_mag"] = reference_plates.data[_age][_case].loc[reference_plates.data[_age][_case].plateID == _plateID, torque + "_mag"].values[0]

                        # Enter sampled data back into the DataFrame
                        self.data[_age][_case].loc[_data.index] = _data.copy()

    def extract_data_through_time(
            self,
            ages: Optional[Union[int, float, List[Union[int, float]], _numpy.ndarray]] = None,
            cases: Optional[Union[str, List[str]]] = None,
            plateIDs: Optional[Union[int, float, List[Union[int, float]], _numpy.ndarray]] = None,
            var: str = "velocity_rms_mag",
        ):
        """
        Function to extract data on slabs through time as a pandas.DataFrame.

        :param ages:        ages of interest (default: None)
        :type ages:         float, int, list, numpy.ndarray
        :param cases:       cases of interest (default: None)
        :type cases:        str, list[str]
        :param plateIDs:    plateIDs of interest (default: None)
        :type plateIDs:     int, float, list[int, float], numpy.ndarray
        :param var:         variable to extract (default: "velocity_rms_mag")
        :type var:          str

        :return:            extracted data with age as index and plateID as columns
        :rtype:             dict[str, pandas.DataFrame] or pandas.DataFrame if only one case is selected
        """
        # Define ages if not provided
        _ages = utils_data.select_ages(ages, self.settings.ages)

        # Define cases if not provided
        if cases == "reconstructed" or cases == ["reconstructed"]:
            _cases = self.settings.reconstructed_cases
        elif cases == "synthetic" or cases == ["synthetic"]:
            _cases = self.settings.synthetic_cases
        else:
            _cases = utils_data.select_cases(cases, self.settings.cases)

        # Define plateIDs if not provided
        # Default is to select all major plates in the Müller et al. (2016) reconstruction
        _plateIDs = utils_data.select_plateIDs(
            plateIDs, 
            [
                101,    # North America
                201,    # South America
                301,    # Eurasia
                501,    # India
                701,    # Africa
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
        extracted_data = {_case: None for _case in _cases}

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
                for _plateID in _plateIDs:
                    if _data.plateID.isin([_plateID]).any():
                        # Hard-coded exception for the Indo-Australian plate for 20-43 Ma (which is defined as 801 in the Müller et al. (2016) reconstruction)
                        _plateID = 801 if _plateID == 501 and _age >= 20 and _age <= 43 else _plateID

                        # Extract data
                        value = _data[_data.plateID == _plateID][var].values[0]

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
            ages: Optional[Union[int, float, List[Union[int, float]], _numpy.ndarray]] = None,
            cases: Optional[Union[str, List[str]]] = None,
            plateIDs: Optional[Union[int, float, List[Union[int, float]], _numpy.ndarray]] = None,
            file_dir: str = None,
            PROGRESS_BAR: bool = True,
        ):
        """
        Function to export the `Plates` object.
        All files are saved as .parquet files to reduce file size and enable rapid reloading.
        The `Plates` object can be filtered by age, case, and plateID.
        By default, the files are saved to the directory specified in the settings object.
        The user can specify the directory to store the files using the `file_dir`.

        :param ages:            ages of interest (default: None)
        :type ages:             float, int, list, numpy.ndarray
        :param cases:           cases of interest (default: None)
        :type cases:            str, list[str]
        :param plateIDs:        plateIDs of interest (default: None)
        :type plateIDs:         int, float, list[int, float], numpy.ndarray
        :param file_dir:        directory to store files (default: None)
        :type file_dir:         str
        :param PROGRESS_BAR:    flag to enable the tqdm progress bar (default: True)
        :type PROGRESS_BAR:     bool
        """
        # Define ages if not provided
        _ages = utils_data.select_ages(ages, self.settings.ages)

        # Define cases if not provided
        if cases == "reconstructed" or cases == ["reconstructed"]:
            _cases = self.settings.reconstructed_cases
        elif cases == "synthetic" or cases == ["synthetic"]:
            _cases = self.settings.synthetic_cases
        else:
            _cases = utils_data.select_cases(cases, self.settings.cases)
        
        # Get file dir
        _file_dir = self.settings.dir_path if file_dir is None else file_dir

        # Loop through ages
        for _age in _tqdm(
                _ages, 
                desc="Saving Plates", 
                disable=(self.settings.logger.level in [logging.INFO, logging.DEBUG] or not PROGRESS_BAR)
            ):
            # Loop through cases
            for _case in _cases:
                # Select resolved geometries, if required
                _resolved_geometries = self.resolved_geometries[_age][_case]
                if plateIDs:
                    _resolved_geometries = _resolved_geometries[_resolved_geometries.PLATEID1.isin(plateIDs)]

                # Save resolved_geometries
                utils_data.GeoDataFrame_to_geoparquet(
                    _resolved_geometries,
                    "Geometries",
                    self.settings.name,
                    _age,
                    _case,
                    _file_dir,
                )

                # Select data, if required
                _data = self.data[_age][_case]
                if plateIDs:
                    _data = _data[_data.plateID.isin(plateIDs)]

                # Save data
                utils_data.DataFrame_to_parquet(
                    _data,
                    "Plates",
                    self.settings.name,
                    _age,
                    _case,
                    _file_dir,
                )

        logging.info(f"Plates saved to {self.settings.dir_path}")

    def export(
            self,
            ages: Optional[Union[int, float, List[Union[int, float]], _numpy.ndarray]] = None,
            cases: Optional[Union[str, List[str]]] = None,
            plateIDs: Optional[Union[int, float, List[Union[int, float]], _numpy.ndarray]] = None,
            file_dir: str = None,
            PROGRESS_BAR: bool = True,
        ):
        """
        Function to export the `Plates` object.
        Geometries are exported as shapefiles, data are exported as .csv files.
        The `Plates` object can be filtered by age, case, and plateID.
        By default, the files are saved to the directory specified in the settings object.
        The user can specify the directory to store the files using the `file_dir`.

        :param ages:            ages of interest (default: None)
        :type ages:             float, int, list, numpy.ndarray
        :param cases:           cases of interest (default: None)
        :type cases:            str, list[str]
        :param plateIDs:        plateIDs of interest (default: None)
        :type plateIDs:         int, float, list[int, float], numpy.ndarray
        :param file_dir:        directory to store files (default: None)
        :type file_dir:         str
        :param PROGRESS_BAR:    flag to enable the tqdm progress bar (default: True)
        :type PROGRESS_BAR:     bool
        """
        # Define ages if not provided
        _ages = utils_data.select_ages(ages, self.settings.ages)

        # Define cases if not provided
        if cases == "reconstructed" or cases == ["reconstructed"]:
            _cases = self.settings.reconstructed_cases
        elif cases == "synthetic" or cases == ["synthetic"]:
            _cases = self.settings.synthetic_cases
        else:
            _cases = utils_data.select_cases(cases, self.settings.cases)
        
        # Get file dir
        _file_dir = self.settings.dir_path if file_dir is None else file_dir

        # Loop through ages
        for _age in _tqdm(
                _ages, 
                desc="Exporting Plates", 
                disable=(self.settings.logger.level in [logging.INFO, logging.DEBUG] or not PROGRESS_BAR)
            ):
            # Loop through cases
            for _case in _cases:
                # Select resolved geometries, if required
                _resolved_geometries = self.resolved_geometries[_age][_case]
                if plateIDs:
                    _resolved_geometries = _resolved_geometries[_resolved_geometries.PLATEID1.isin(plateIDs)]

                # Save resolved_geometries
                utils_data.GeoDataFrame_to_shapefile(
                    _resolved_geometries,
                    "Geometries",
                    self.settings.name,
                    _age,
                    _case,
                    _file_dir,
                )

                utils_data.DataFrame_to_csv(
                    self.data[_age][_case],
                    "Plates",
                    self.settings.name,
                    _age,
                    _case,
                    _file_dir,
                )

        logging.info(f"Plates exported to {self.settings.dir_path}")