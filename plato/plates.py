import logging
from typing import Dict, List, Optional, Union

import gplately as _gplately
import numpy as _numpy
from tqdm import tqdm as _tqdm

import utils_data, utils_calc, utils_init
from settings import Settings
from points import Points

class Plates:
    """
    Class that contains all information for the plates in a reconstruction.
    """
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
        
        # GEOMETRIES
        # Set up plate reconstruction object and initialise dictionaries to store resolved topologies and geometries
        self.resolved_topologies = {_age: {} for _age in self.settings.ages}
        self.resolved_geometries = {_age: {} for _age in self.settings.ages}

        # Load or initialise plate geometries
        for _age in _tqdm(self.settings.ages, desc="Loading geometries", disable=self.settings.logger.level==logging.INFO):
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
                    for entries in entry:
                        if entry not in available_cases:
                            self.geometries[_age][entry] = self.geometries[_age][available_cases[0]].copy()
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

        # Initialise data dictionary
        self.data = {age: {} for age in self.settings.ages}

        # Loop through times
        for _age in _tqdm(self.settings.ages, desc="Loading plate data", disable=self.settings.logger.level==logging.INFO):
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
                    for entries in entry:
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
                    
                    # Copy to matching cases
                    if len(entries) > 1:
                        for entry in entries[1:]:
                            self.data[_age][entry] = self.data[_age][key].copy()

    def calculate_rms_velocity(
            self,
            points: Optional['Points'] = None,
            ages: Optional[Union[_numpy.ndarray, List, float, int]] = None,
            cases: Optional[Union[List[str], str]] = None,
            plateIDs: Optional[Union[List, _numpy.ndarray]] = None,
        ):
        """
        Function to calculate the root mean square (RMS) velocity of the plates.
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
        
        # Define cases if not provided, default to GPE cases because it only depends on the grid spacing
        _iterable = utils_data.select_iterable(cases, self.settings.point_cases)

        # Loop through ages
        for _age in _tqdm(_ages, desc="Calculating RMS velocities", disable=self.settings.logger.level==logging.INFO):
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

                    self.data[_age] = utils_data.copy_values(
                        self.data[_age], 
                        key, 
                        entries, 
                        ["velocity_rms_mag", "velocity_rms_azi", "spin_rate_rms_mag"], 
                    )

    def calculate_torque_on_plates(
            self,
            point_data: Dict,
            ages: Optional[Union[_numpy.ndarray, List, float, int]] = None,
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
            torque_var: str = "torque",
        ):
        """
        Function to calculate the torque on plates from the forces acting on a set of points on Earth.
        """
        # Define ages if not provided
        _ages = utils_data.select_ages(ages, self.settings.ages)
     
        # Define cases if not provided, defaulting to the cases that are relevant for the torque variable
        if torque_var == "slab_pull":
            matching_cases = self.settings.slab_pull_cases
        elif torque_var == "slab_bend":
            matching_cases = self.settings.slab_bend_cases
        elif torque_var == "GPE":
            matching_cases = self.settings.gpe_cases
        elif torque_var == "mantle_drag":
            matching_cases = self.settings.mantle_drag_cases
        
        # Define iterable, if cases not provided
        _iterable = utils_data.select_iterable(cases, matching_cases)

        # Define plateID column of point data
        point_data_plateID_col = "lower_plateID" if torque_var == "slab_pull" or torque_var == "slab_bend" else "plateID"

        # Define columns to store torque and force components and store them in one list
        torque_cols = [f"{torque_var}_torque_" + axis for axis in ["x", "y", "z", "mag"]]
        force_cols = [f"{torque_var}_force_" + axis for axis in ["lat", "lon", "mag", "azi"]]
        cols = torque_cols + force_cols 

        # Loop through ages
        for _age in _tqdm(_ages, desc="Calculating torque on plates", disable=(self.settings.logger.level==logging.INFO)):
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
                else:
                    selected_points_plateID = _point_data.plateID.values
                    selected_points_area = _point_data.segment_length_lat.values * _point_data.segment_length_lon.values

                if torque_var == "GPE":
                    print("Force lon:", _point_data[f"{torque_var}_force_lon"].values)

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
                if torque_var == "GPE":
                    print("Torque mag: ", computed_data[f"{torque_var}_torque_mag"].values)

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
            ages: Optional[Union[_numpy.ndarray, List, float, int]] = None,
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
        ):
        """
        Function to calculate driving torque.
        """
        # Define ages if not provided
        _ages = utils_data.select_ages(ages, self.settings.ages)
        
        # Define cases if not provided
        _cases = utils_data.select_cases(cases, self.settings.cases)

        # Loop through ages
        for i, _age in _tqdm(enumerate(_ages), desc="Calculating driving torque", disable=self.settings.logger.level==logging.INFO):
            # Inform the user that the driving torques are being calculated
            logging.info(f"Computing driving torque at {_age} Ma")

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
        
        # Set flag to indicate that the driving torque has been calculated
        self.driving_torque_calculated = True

    def calculate_residual_torque(
            self,
            ages: Optional[Union[_numpy.ndarray, List, float, int]] = None,
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
        ):
        """
        Function to calculate driving torques.
        """
        # Define ages if not provided
        _ages = utils_data.select_ages(ages, self.settings.ages)
        
        # Define cases if not provided
        _cases = utils_data.select_cases(cases, self.settings.cases)

        # Loop through cases
        # Order of loops is flipped to skip cases where no slab pull torque needs to be sampled
        for _case in _tqdm(_cases, desc="Calculating residual torque", disable=self.settings.logger.level==logging.INFO):
            # Skip if reconstructed motions are enabled
            if self.settings.options[_case]["Reconstructed motions"]:
                continue

            # Inform the user that the driving torques are being calculated
            logging.info(f"Computing residual torque at {_case} Ma")

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
                
    def calculate_synthetic_velocity(
            self,
            ages: Optional[Union[_numpy.ndarray, List, float, int]] = None,
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
        ):
        """
        Function to calculate synthetic velocity of plates.
        """
        # Define ages if not provided
        _ages = utils_data.select_ages(ages, self.settings.ages)
        
        # Define cases if not provided
        _cases = utils_data.select_cases(cases, self.settings.cases)

        # Loop through cases
        # Order of loops is flipped to skip cases where no slab pull torque needs to be sampled
        for _case in _tqdm(_cases, desc="Calculating synthetic velocity", disable=self.settings.logger.level==logging.INFO):
            # Skip if reconstructed motions are enabled
            if not self.settings.options[_case]["Reconstructed motions"]:

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

                    # Calculate synthetic velocity
                    computed_data2 = utils_calc.compute_synthetic_stage_rotation(computed_data1, self.settings.options[_case])

                    # Enter sampled data back into the DataFrame
                    self.data[_age][_case].loc[_data.index] = computed_data2.copy()

    def save(
            self,
            ages: Union[None, List[int], List[float], _numpy.ndarray] = None,
            cases: Union[None, str, List[str]] = None,
            plateIDs: Union[None, List[int], List[float], _numpy.ndarray] = None,
            file_dir: Optional[str] = None,
        ):
        """
        Function to save the 'Plates' object.
        """
        # Define ages if not provided
        _ages = utils_data.select_ages(ages, self.settings.ages)

        # Define cases if not provided
        _cases = utils_data.select_cases(cases, self.settings.cases)
        
        # Get file dir
        _file_dir = self.settings.dir_path if file_dir is None else file_dir

        # Loop through ages
        for _age in _tqdm(_ages, desc="Saving Plates", disable=self.settings.logger.level==logging.INFO):
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
            ages: Union[None, List[int], List[float], _numpy.ndarray] = None,
            cases: Union[None, str, List[str]] = None,
            plateIDs: Union[None, List[int], List[float], _numpy.ndarray] = None,
            file_dir: Optional[str] = None,
        ):
        """
        Function to export the 'Plates' object.
        Geometries are saved as shapefiles, data are saved as .csv files.
        """
        # Define ages if not provided
        _ages = utils_data.select_ages(ages, self.settings.ages)

        # Define cases if not provided
        _cases = utils_data.select_cases(cases, self.settings.cases)
        
        # Get file dir
        _file_dir = self.settings.dir_path if file_dir is None else file_dir

        # Loop through ages
        for _age in _tqdm(_ages, desc="Exporting Plates", disable=self.settings.logger.level==logging.INFO):
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


