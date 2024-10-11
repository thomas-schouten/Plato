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
                            points.data[_age][key].v_mag.values[mask],
                            points.data[_age][key].v_azi.values[mask],
                            points.data[_age][key].omega.values[mask],
                        )

                        # Store RMS velocity components 
                        self.data[_age][key].loc[self.data[_age][key].plateID == _plateID, "v_rms_mag"] = rms_velocity[0]
                        self.data[_age][key].loc[self.data[_age][key].plateID == _plateID, "v_rms_azi"] = rms_velocity[1]
                        self.data[_age][key].loc[self.data[_age][key].plateID == _plateID, "omega_rms"] = rms_velocity[2]

                    self.data[_age] = utils_data.copy_values(
                        self.data[_age], 
                        key, 
                        entries, 
                        ["v_rms_mag", "v_rms_azi", "omega_rms"], 
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
     
        # Define cases if not provided, default to GPE cases because it only depends on the grid spacing
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

        # Loop through ages
        for _age in _tqdm(_ages, desc="Calculating torque on plates"):
            logging.info(f"Calculating torque on plates at {_age} Ma")
            for key, entries in _iterable.items():
                # Define plateIDs if not provided
                _plateIDs = utils_data.select_plateIDs(
                    plateIDs,
                    self.data[_age][key].plateID,
                )

                # Define masks
                plates_mask = self.data[_age][key].loc[:, "plateID"].isin(_plateIDs)
                points_mask = point_data[_age][key].loc[:, "plateID"].isin(_plateIDs)

                if torque_var == "slab_pull" or torque_var == "slab_bend":
                    selected_points_plateID = point_data[_age][key].lower_plateID.values[points_mask]
                    selected_points_area = point_data[_age][key].trench_segment_length.values[points_mask]
                else:
                    selected_points_plateID = point_data[_age][key].plateID.values[points_mask]
                    selected_points_area = point_data[_age][key].segment_length_lat.values[points_mask] * point_data[_age][key].segment_length_lon.values[points_mask]

                # Calculate torques
                computed_data = utils_calc.compute_torque_on_plates(
                    self.data[_age][key].loc[plates_mask], 
                    point_data[_age][key].lat.values[points_mask],
                    point_data[_age][key].lon.values[points_mask],
                    selected_points_plateID,
                    point_data[_age][key][f"{torque_var}_force_lat"].values[points_mask], 
                    point_data[_age][key][f"{torque_var}_force_lon"].values[points_mask],
                    selected_points_area,
                    self.settings.constants,
                    torque_var = torque_var,
                )

                # Feed back into plates
                self.data[_age][key].loc[plates_mask].update(computed_data)

                # Copy DataFrames, if necessary
                if len(entries) > 1:
                    columns = [f"{torque_var}_torque" + axis for axis in ["x", "y", "z", "mag"]]
                    self.data[_age] = utils_data.copy_values(
                        self.data[_age], 
                        key, 
                        entries, 
                        columns, 
                    )

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
        _ages = utils_data.select_ages(
            ages,
            self.settings.ages,
        )

        # Define iterable if cases not provided
        _slab_iterable = utils_data.select_iterable(
            cases,
            self.settings.slab_cases,
        )
        _mantle_iterable = utils_data.select_iterable(
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

        for i, _age in tqdm(self.settings.ages, desc="Optimising torques", disable=self.settings.logger.level==logging.INFO):
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
                        self.data[_age] = utils_data.copy_values(
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
                        self.data[_age] = utils_data.copy_values(
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
        _ages = utils_data.select_ages(
            ages,
            self.settings.ages,
        )

        # Define cases if not provided
        _cases = utils_data.select_cases(
            cases,
            self.settings.cases,
        )

        # Loop through reconstruction times
        for i, _age in tqdm(enumerate(_ages), desc="Computing driving torques", disable=self.settings.logger.level==logging.INFO):
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
        for i, _age in tqdm(enumerate(ages), desc="Computing residual torques", disable=self.settings.logger.level==logging.INFO):
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


