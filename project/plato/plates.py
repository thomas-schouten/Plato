# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# PLATO
# Algorithm to calculate plate forces from tectonic reconstructions
# Thomas Schouten, 2024
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Import libraries
# Standard libraries
import os as _os
import warnings
from typing import Dict, List, Optional, Union

# Third-party libraries
import numpy as _numpy
import pandas as _pandas
import gplately as _gplately
from gplately import pygplates as _pygplates
from tqdm import tqdm

# Local libraries
import utils_calc, utils_data, utils_init
from settings import Settings

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# PLATES OBJECT
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

class Plates:
    """
    Class that contains all information for the plates in a reconstruction.
    """
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
            data: Optional[Union[Dict, str]] = None,
            resolved_geometries: Optional[Dict] = None,
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
            ages, 
            cases_file,
            cases_sheet,
            files_dir,
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
        self.resolved_topologies, self.resolved_geometries = {}, {}

        # Define ages if not provided
        _ages = utils_data.get_ages(
            ages,
            self.settings.ages,
        )

        # Load or initialise plate geometries
        for _age in tqdm(_ages, desc="Loading geometries", disable=self.settings.DEBUG_MODE):
            
            # Load resolved geometries if they are available
            if resolved_geometries is not None:
                # Check if resolved geometries are a dictionary
                if not isinstance(resolved_geometries, dict):
                    raise ValueError("Resolved geometries should be a dictionary.")
                
                # Check if the age is in the dictionary
                if _age in resolved_geometries.keys():
                    self.resolved_geometries[_age] = resolved_geometries[_age]
                else:
                    self.resolved_geometries[_age] = utils_data.GeoDataFrame_from_geoparquet(
                        self.settings.dir_path,
                        "Geometries",
                        _age,
                        self.settings.name,
                    )

                    # Get new topologies if they are unavailable
                    if self.resolved_geometries[_age] is None:
                        self.resolved_geometries[_age] = utils_data.get_topology_geometries(
                            self.reconstruction, _age, anchor_plateID=0
                        )
            
            # Resolve topologies to use to get plates
            # NOTE: This is done because some information is retrieved from the resolved topologies and some from the resolved geometries
            #       This step could be sped up by extracting all information from the geopandas DataFrame, but so far this has not been the main bottleneck
            # Ignore annoying warnings that the field names are laundered
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    action="ignore",
                    message="Normalized/laundered field name:"
                )
                self.resolved_topologies[_age] = []
                _pygplates.resolve_topologies(
                    self.reconstruction.topology_features,
                    self.reconstruction.rotation_model, 
                    self.resolved_topologies[_age], 
                    _age, 
                    anchor_plate_id=0
                )

        # DATA
        # Load or initialise plate data
        self.data = utils_init.get_data(
            data,
            _ages,
            self.settings.cases,
            self.settings.plate_cases,
            self.get_plate_data,
            self.reconstruction.rotation_model,
            self.resolved_topologies,
            self.settings,
        )

    def get_plate_data(
        rotations: _pygplates.RotationModel,
        age: int,
        resolved_topologies: list, 
        options: dict,
        ):
        """
        Function to get data on plates in reconstruction.

        :param rotations:           rotation model
        :type rotations:            _pygplates.RotationModel object
        :param age:                 reconstruction age
        :type age:                  integer
        :param resolved_topologies: resolved topologies
        :type resolved_topologies:  list of resolved topologies
        :param options:             options for the case
        :type options:              dict

        :return:                    plates
        :rtype:                     pandas.DataFrame
        """
        # Set constants
        constants = utils_calc.set_constants()

        # Make _pandas.df with all plates
        # Initialise list
        plates = _numpy.zeros([len(resolved_topologies),10])
        
        # Loop through plates
        for n, topology in enumerate(resolved_topologies):

            # Get plateID
            plates[n,0] = topology.get_resolved_feature().get_reconstruction_plate_id()

            # Get plate area
            plates[n,1] = topology.get_resolved_geometry().get_area() * constants.mean_Earth_radius_m**2

            # Get Euler rotations
            stage_rotation = rotations.get_rotation(
                to_time=age,
                moving_plate_id=int(plates[n,0]),
                from_time=age + options["Velocity time step"],
                anchor_plate_id=options["Anchor plateID"]
            )
            pole_lat, pole_lon, pole_angle = stage_rotation.get_lat_lon_euler_pole_and_angle_degrees()
            plates[n,2] = pole_lat
            plates[n,3] = pole_lon
            plates[n,4] = pole_angle

            # Get plate centroid
            centroid = topology.get_resolved_geometry().get_interior_centroid()
            centroid_lat, centroid_lon = centroid.to_lat_lon_array()[0]
            plates[n,5] = centroid_lon
            plates[n,6] = centroid_lat

            # Get velocity [cm/a] at centroid
            centroid_velocity = get_velocities([centroid_lat], [centroid_lon], (pole_lat, pole_lon, pole_angle))
        
            plates[n,7] = centroid_velocity[1]
            plates[n,8] = centroid_velocity[0]
            plates[n,9] = centroid_velocity[2]

        # Convert to DataFrame    
        plates = _pandas.DataFrame(plates)

        # Initialise columns
        plates.columns = ["plateID", "area", "pole_lat", "pole_lon", "pole_angle", "centroid_lon", "centroid_lat", "centroid_v_lon", "centroid_v_lat", "centroid_v_mag"]

        # Merge topological networks with main plate; this is necessary because the topological networks have the same PlateID as their host plate and this leads to computational issues down the road
        main_plates_indices = plates.groupby("plateID")["area"].idxmax()

        # Create new DataFrame with the main plates
        merged_plates = plates.loc[main_plates_indices]

        # Aggregating the area column by summing the areas of all plates with the same plateID
        merged_plates["area"] = plates.groupby("plateID")["area"].sum().values

        # Get plate names
        merged_plates["name"] = _numpy.nan; merged_plates.name = get_plate_names(merged_plates.plateID)
        merged_plates["name"] = merged_plates["name"].astype(str)

        # Sort and index by plate ID
        merged_plates = merged_plates.sort_values(by="plateID")
        merged_plates = merged_plates.reset_index(drop=True)

        # Initialise columns to store other whole-plate properties
        merged_plates["trench_length"] = 0.; merged_plates["zeta"] = 0.
        merged_plates["v_rms_mag"] = 0.; merged_plates["v_rms_azi"] = 0.; merged_plates["omega_rms"] = 0.
        merged_plates["slab_flux"] = 0.; merged_plates["sediment_flux"] = 0.

        # Initialise columns to store whole-plate torques (Cartesian) and force at plate centroid (North-East).
        torques = ["slab_pull", "GPE", "slab_bend", "mantle_drag", "driving", "residual"]
        axes = ["x", "y", "z", "mag"]
        coords = ["lat", "lon", "mag", "azi"]
        
        merged_plates[[torque + "_torque_" + axis for torque in torques for axis in axes]] = [[0.] * len(torques) * len(axes) for _ in range(len(merged_plates.plateID))]
        merged_plates[["slab_pull_torque_opt_" + axis for axis in axes]] = [[0.] * len(axes) for _ in range(len(merged_plates.plateID))]
        merged_plates[["mantle_drag_torque_opt_" + axis for axis in axes]] = [[0.] * len(axes) for _ in range(len(merged_plates.plateID))]
        merged_plates[["driving_torque_opt_" + axis for axis in axes]] = [[0.] * len(axes) for _ in range(len(merged_plates.plateID))]
        merged_plates[["residual_torque_opt_" + axis for axis in axes]] = [[0.] * len(axes) for _ in range(len(merged_plates.plateID))]
        merged_plates[[torque + "_force_" + coord for torque in torques for coord in coords]] = [[0.] * len(torques) * len(coords) for _ in range(len(merged_plates.plateID))]
        merged_plates[["slab_pull_force_opt_" + coord for coord in coords]] = [[0.] * len(coords) for _ in range(len(merged_plates.plateID))]
        merged_plates[["mantle_drag_force_opt_" + coord for coord in coords]] = [[0.] * len(coords) for _ in range(len(merged_plates.plateID))]
        merged_plates[["driving_force_opt_" + coord for coord in coords]] = [[0.] * len(coords) for _ in range(len(merged_plates.plateID))]
        merged_plates[["residual_force_opt_" + coord for coord in coords]] = [[0.] * len(coords) for _ in range(len(merged_plates.plateID))]

        return merged_plates

    def calculate_rms_velocity(
            self,
            ages: Optional[Union[_numpy.ndarray, List, float, int]] = None,
            cases: Optional[Union[List[str], str]] = None,
        ):
        """
        Function to calculate the root mean square (RMS) velocity of the plates.
        """
        # Define ages if not provided
        _ages = utils_data.get_ages(
            ages,
            self.settings.ages,
        )
        
        # Define cases if not provided, default to GPE cases because it only depends on the grid spacing
        _iterable = utils_data.get_iterable(
            cases,
            self.settings.gpe_cases,
        )

        for _age in _ages:
            # Calculate rms velocity
            for key, entries in _iterable.items():
                print(self.data)
                if self.data[_age][key]["v_rms_mag"].mean() == 0:
                    self.data[_age][key] = utils_calc.compute_rms_velocity(
                        self.data[_age][key],
                        self.data[_age][key]
                    )

                self.data[_age] = utils_calc.copy_values(
                    self.data[_age], 
                    key, 
                    entries, 
                    ["v_rms_mag", "v_rms_azi", "omega_rms"], 
                )
            
                # Copy DataFrames to other cases
                for entry in entries[1:]:
                    if self.data[_age][entry]["v_rms_mag"].mean() == 0:
                        self.data[_age][entry]["v_rms_mag"] = self.data[_age][key]["v_rms_mag"]

                    if self.data[_age][entry]["v_rms_azi"].mean() == 0:
                        self.data[_age][entry]["v_rms_azi"] = self.data[_age][key]["v_rms_azi"]

                    if self.data[_age][entry]["omega_rms"].mean() == 0:
                        self.data[_age][entry]["omega_rms"] = self.data[_age][key]["omega_rms"]

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
        _ages = utils_data.get_ages(
            ages,
            self.settings.ages,
        )

        # Define iterable if cases not provided
        _slab_iterable = utils_data.get_iterable(
            cases,
            self.settings.slab_cases,
        )
        _mantle_iterable = utils_data.get_iterable(
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

        for i, _age in tqdm(self.settings.ages, desc="Optimising torques", disable=(self.settings.DEBUG_MODE or not PROGRESS_BAR)):
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
                        self.data[_age] = utils_calc.copy_values(
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
                        self.data[_age] = utils_calc.copy_values(
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
        _ages = utils_data.get_ages(
            ages,
            self.settings.ages,
        )

        # Define cases if not provided
        _cases = utils_data.get_cases(
            cases,
            self.settings.cases,
        )

        # Loop through reconstruction times
        for i, _age in tqdm(enumerate(_ages), desc="Computing driving torques", disable=(self.settings.DEBUG_MODE or not PROGRESS_BAR)):
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
        for i, _age in tqdm(enumerate(ages), desc="Computing residual torques", disable=(self.settings.DEBUG_MODE or not PROGRESS_BAR)):
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


    def calculate_torque_on_plates(
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
        Function to calculate the torque on plates
        """
        # Define ages if not provided
        if ages is not None:
            # Check if ages is a single value
            if isinstance(ages, (int, float, _numpy.integer, _numpy.floating)):
                ages = [ages]
        else:
            # Otherwise, use all ages from the settings
            ages = self.settings.ages

        # Select plates
        if plate_IDs is not None:
            if isinstance(plate_IDs, (int, float, _numpy.floating, _numpy.integer)):
                plate_IDs = [plate_IDs]
        
        # Loop through ages
        for _age in tqdm(ages, desc="Calculating torque on plates", disable=(self.settings.DEBUG_MODE or not PROGRESS_BAR)):
            if self.settings.DEBUG_MODE:
                print(f"Calculating torque on plates at {_age} Ma")

            for key in cases:
                # Select plates, if necessary
                selected_data = self.data[_age][key].copy()
                if plate_IDs is not None:
                    selected_data = selected_data.loc[selected_data.plateID.isin(plate_IDs)].copy()

                
                # Calculate torques
                selected_data = utils_calc.calculate_torque_on_plates(
                    selected_data,
                    self.constants,
                    self.options[key],
                )

                # Feed back into plates
                if plate_IDs is not None:
                    mask = self.data[_age][key].plateID.isin(plate_IDs)
                    self.data[_age][key].loc[mask, :] = selected_data
                else:
                    self.data[_age][key] = selected_data