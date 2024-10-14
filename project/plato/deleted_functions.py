def plot_velocity_map(
            self,
            ax,
            fig,
            reconstruction_time,
            case,
            plotting_options
        ):
        """
        Function to create subplot with plate velocities
            ax:                     axes object
            fig:                    figure
            reconstruction_time:    the time for which to display the map
            case:                   case for which to plot the sediments
            plotting_options:       dictionary with options for plotting
        """
        # Check if reconstruction time is in valid times
        if reconstruction_time not in self.times:
            return print("Invalid reconstruction time")
        
        # Set basemap
        ax, gl = self.plot_basemap(ax)

        # Plot plates and coastlines
        self.plot_reconstruction(ax, reconstruction_time, plotting_options, plates=True, trenches=False)

        # Get data
        plate_vectors = self.plates[reconstruction_time][case].loc[self.plates[reconstruction_time][case].area >= self.options[case]["Minimum plate area"]]
        slab_data = self.slabs[reconstruction_time][case].loc[self.slabs[reconstruction_time][case].lower_plateID.isin(plate_vectors.plateID)]
        slab_vectors = slab_data.iloc[::5]

        # Plot velocity magnitude at trenches
        vels = ax.scatter(
            slab_data.lon,
            slab_data.lat,
            c=slab_data.v_lower_plate_mag,
            s=plotting_options["marker size"],
            transform=ccrs.PlateCarree(),
            cmap=plotting_options["velocity magnitude cmap"],
            vmin=0,
            vmax=plotting_options["velocity max"]
        )

        # Plot velocity at subduction zones
        slab_vectors = ax.quiver(
            x=slab_vectors.lon,
            y=slab_vectors.lat,
            u=slab_vectors.v_lower_plate_lon,
            v=slab_vectors.v_lower_plate_lat,
            transform=ccrs.PlateCarree(),
            # label=vector.capitalize(),
            width=2e-3,
            scale=3e2,
            zorder=4,
            color='black'
        )

        # Plot velocity at centroid
        centroid_vectors = ax.quiver(
            x=plate_vectors.centroid_lon,
            y=plate_vectors.centroid_lat,
            u=plate_vectors.centroid_v_lon,
            v=plate_vectors.centroid_v_lat,
            transform=ccrs.PlateCarree(),
            # label=vector.capitalize(),
            width=5e-3,
            scale=3e2,
            zorder=4,
            color='white',
            edgecolor='black',
            linewidth=1
        )

        # Colourbar
        if plotting_options["cbar"] is True:
            fig.colorbar(vels, ax=ax, label="Velocity [cm/a]", orientation=plotting_options["orientation cbar"], shrink=0.75, aspect=20)
    
        return ax, vels, centroid_vectors, slab_vectors


def optimise_torques(self, sediments=True):
        """
        Function to apply optimised parameters to torques
        Arguments:
            opt_visc
            opt_sp_const
        """
        # Apply to each torque in DataFrame
        axes = ["_x", "_y", "_z", "_mag"]
        for case in self.cases:
            for axis in axes:
                self.plates[case]["slab_pull_torque_opt" + axis] = self.options[case]["Slab pull constant"] * self.plates[case]["slab_pull_torque" + axis]
                if self.options[case]["Reconstructed motions"]:
                    self.plates[case]["mantle_drag_torque_opt" + axis] = self.options[case]["Mantle viscosity"] * self.plates[case]["mantle_drag_torque" + axis]
                
                for reconstruction_time in self.times:
                    if sediments == True:
                        self.plates[reconstruction_time][case]["slab_pull_torque_opt" + axis] = self.options[case]["Slab pull constant"] * self.plates[reconstruction_time][case]["slab_pull_torque" + axis]
                    if self.options[case]["Reconstructed motions"]:
                        self.plates[reconstruction_time][case]["mantle_drag_torque_opt" + axis] = self.options[case]["Mantle viscosity"] * self.plates[reconstruction_time][case]["mantle_drag_torque" + axis]

        # Apply to forces at centroid
        coords = ["lon", "lat"]
        for reconstruction_time in self.times:
            for case in self.cases:
                for coord in coords:
                    self.plates[reconstruction_time][case]["slab_pull_force_opt" + coord] = self.options[case]["Slab pull constant"] * self.plates[reconstruction_time][case]["slab_pull_force" + coord]
                    if self.options[case]["Reconstructed motions"]:
                        self.plates[reconstruction_time][case]["mantle_drag_force_opt" + coord] = self.options[case]["Mantle viscosity"] * self.plates[reconstruction_time][case]["slab_pull_force" + coord]

        self.optimised_torques = True


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# PLATO
# Algorithm to calculate plate forces from tectonic reconstructions
# Initialiser for accelerated parallel initialisation of plate, point and slab data
# Thomas Schouten, 2024
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Import libraries
# Standard libraries
import os as _os
import warnings
from typing import List, Optional, Union

# Third-party libraries
import numpy as _numpy
import matplotlib.pyplot as plt
import geopandas as _gpd
import gplately
from gplately import pygplates as _pygplates
import cartopy.crs as ccrs
import cmcrameri as cmc
from tqdm import tqdm
import xarray as _xarray
from time import time

# Local libraries
import setup
import setup_parallel
import functions_main

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# INITIALISER
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def initialise_plato(
        reconstruction_name: str, 
        reconstruction_times: List[int] or _numpy.array, 
        cases_file: str, 
        cases_sheet: Optional[str] = "Sheet1", 
        files_dir: Optional[str] = None,
        rotation_file: Optional[List[str]] = None,
        topology_file: Optional[List[str]] = None,
        polygon_file: Optional[List[str]] = None,
    ):

    # Store cases and case options
    cases, options = setup.get_options(cases_file, cases_sheet)

    # Group cases for initialisation of slabs and points
    slab_options = ["Slab tesselation spacing"]
    slab_cases = setup.process_cases(cases, options, slab_options)
    point_options = ["Grid spacing"]
    point_cases = setup.process_cases(cases, options, point_options)

    # Convert GPlates file inputs to GPlates objects
    rotations = _pygplates.RotationModel(rotation_file)
    topologies = _pygplates.FeatureCollection(topology_file)
    polygons = _pygplates.FeatureCollection(polygon_file)

    reconstruction = gplately.PlateReconstruction(rotations, topologies, polygons)
    
    for reconstruction_time in tqdm(reconstruction_times, desc="Initialising and saving files"):
        # Get geometries of plates
        resolved_geometries = setup.get_topology_geometries(
            reconstruction, reconstruction_time, anchor_plateID=0
        )

        # Resolve topologies to use to get plates
        # NOTE: This is done because some information is retrieved from the resolved topologies and some from the resolved geometries
        #       This step could be sped up by extracting all information from the resolved geometries, but so far this has not been the main bottleneck
        # Ignore annoying warnings
        with warnings.catch_warnings():
            warnings.filterwarnings(
                action="ignore",
                message="Normalized/laundered field name:"
            )
            resolved_topologies = []
            _pygplates.resolve_topologies(
                topologies,
                rotations, 
                resolved_topologies, 
                reconstruction_time, 
                anchor_plate_id=0
            )

        # Get plates
        plates = setup_parallel.get_plates(
                    reconstruction.rotation_model,
                    reconstruction_time,
                    resolved_topologies,
                    options[cases[0]],
                )
        
        # Get slabs
        slabs = {}
        for key, entries in slab_cases.items():
            slabs[key] = setup_parallel.get_slabs(
                        reconstruction,
                        reconstruction_time,
                        plates,
                        resolved_geometries,
                        options[key],
                    )
            
            # Copy DataFrames to other cases
            for entry in entries[1:]:
                slabs[entry] = slabs[key].copy()

        # Get points
        points = {}
        for key, entries in point_cases.items():
            points[key] = setup_parallel.get_points(
                        reconstruction,
                        reconstruction_time,
                        plates,
                        resolved_geometries,
                        options[key],
                    )

            # Copy DataFrames to other cases
            for entry in entries[1:]:
                points[entry] = points[key].copy()

        # Save all data to files
        for case in cases:
            # Save plates
            setup.DataFrame_to_parquet(
                plates,
                "Plates",
                reconstruction_name,
                reconstruction_time,
                case,
                files_dir,
            )

            # Save slabs
            setup.DataFrame_to_parquet(
                slabs[case],
                "Slabs",
                reconstruction_name,
                reconstruction_time,
                case,
                files_dir,
            )

            # Save points
            setup.DataFrame_to_parquet(
                points[case],
                "Points",
                reconstruction_name,
                reconstruction_time,
                case,
                files_dir,
            )

# if __name__ == "__main__":
#     # Set reconstruction times
#     reconstruction_times = _numpy.arange(31, 41, 1)

#     # Define path to directory containing reconstruction files
#     main_dir = _os.path.join("", "Users", "thomas", "Documents", "_Plato", "Reconstruction_analysis")

#     # Define the parameter settings for the model
#     settings_file = _os.path.join(main_dir, "settings.xlsx")

#     settings_sheet = "Sheet2"
    
#     M2016_model_dir = _os.path.join(main_dir, "M2016")

#     # Get start time
#     start_time = time()

#     # Set up PlateForces object
#     initialise_plato(
#         "Muller2016",
#         reconstruction_times,
#         settings_file,
#         cases_sheet = settings_sheet,
#         files_dir = _os.path.join("Output", M2016_model_dir, "Lr-Hb"),
#         rotation_file = _os.path.join("GPlates_files", "M2016", f"M2016_rotations_Lr-Hb.rot"),
#         topology_file = _os.path.join("GPlates_files", "M2016", "M2016_topologies.gpml"),
#         polygon_file = _os.path.join("GPlates_files", "M2016", "M2016_polygons.gpml"),
#     )

#     # Print the time taken to set up the model
#     set_up_time = time() - start_time
#     print(f"Time taken to set up the model: {set_up_time:.2e} seconds")


def load_data(
        _data: dict,
        _age: Union[float, int, _numpy.floating, _numpy.integer], 
        cases: List[str], 
        matching_cases: dict, 
        type: str,
        rotations: _pygplates.RotationModel,
        resolved_topologies: list,
        options: dict,
        data: Optional[Union[dict, str]] = None,
    ):
    """
    Function to load or initialise data for the current simulation.
    """
    # Load the data if it is available
    if data is not None:
        # Check if data is a dictionary
        if isinstance(data, dict):
            if _age in data.keys():
                # Initialise _age only if there's a matching case
                matching_cases_list = [_case for _case in cases if _case in data[_age].keys()]

                # Only initialise _data[_age] if there are matching cases
                if matching_cases_list:
                    _data[_age] = {}  # Initialise _age for matching cases

                for _case in matching_cases_list:
                    _data[_age][_case] = data[_age][_case]

                # Get missing cases
                existing_cases = set(data[_age].keys())
                missing_cases = _numpy.array([_case for _case in cases if _case not in existing_cases])

                # Loop through missing cases
                for _case in missing_cases:
                    # Check for matching case and copy data
                    matching_key = next((key for key, cases in matching_cases.items() if _case in cases), None)
                    if matching_key:
                        for matching_case in matching_cases[matching_key]:  # Access the cases correctly
                            if matching_case in _data[_age]:
                                # Copy data from the matching case
                                _data[_age][_case] = _data[_age][matching_case].copy()
                                break  # Exit after copying to avoid overwriting

                    else:
                        # Use the provided data retrieval function if no matching case is found
                        if type == "Plates":
                            if type == "Plates":
                                _data[_age][_case] = get_plate_data(
                                    rotations,
                                    _age,
                                    resolved_topologies,
                                    options
                                    )
                            if type == "Slabs":
                                _data[_age][_case] = get_slab_data(
                                    rotations,
                                    _age,
                                    resolved_topologies,
                                    options
                                    )
                            if type == "Points":
                                _data[_age][_case] = get_point_data(
                                    _data,
                                    _age,
                                    resolved_topologies,
                                    options
                                    )

        # TODO: Implement loading data from a file if needed
        if isinstance(data, str):
            pass

        # Initialise data if it is not available
        if _age not in _data.keys():
            _data[_age] = {}
            # Loop through cases
            for _case in cases:
                

    return _data

def get_data(
        rotations: _pygplates.RotationModel,
        resolved_topologies: list,
        options: dict,
        type: str,
    ):
    """
    Function to get data for the current simulation.

    :param rotations:           rotation model
    :type rotations:            _pygplates.RotationModel object
    :param resolved_topologies: resolved topologies
    :type resolved_topologies:  list of resolved topologies
    :param options:             options for the case
    :type options:              dict
    :param type:                type of data
    :type type:                 str

    :return:                    data
    :rtype:                     dict
    """
    if type == "Plates":
        data = get_plate_data(
            rotations,
            _age,
            resolved_topologies,
            options
            )
    if type == "Slabs":
        data = get_slab_data(
            rotations,
            _age,
            resolved_topologies,
            options
            )
    if type == "Points":
        data = get_point_data(
            _data,
            _age,
            resolved_topologies,
            options
            )
    
    return data
    
   def load_grid(
        grid: dict,
        reconstruction_name: str,
        ages: list,
        type: str,
        files_dir: str,
        points: Optional[dict] = None,
        seafloor_grid: Optional[_xarray.Dataset] = None,
        cases: Optional[list] = None,
        DEBUG_MODE: Optional[bool] = False
    ) -> dict:
    """
    Function to load grid from a folder.

    :param grids:                  grids
    :type grids:                   dict
    :param reconstruction_name:    name of reconstruction
    :type reconstruction_name:     string
    :param ages:   reconstruction times
    :type ages:    list or numpy.array
    :param type:                   type of grid
    :type type:                    string
    :param files_dir:              files directory
    :type files_dir:               string
    :param points:                 points
    :type points:                  dict
    :param seafloor_grid:          seafloor grid
    :type seafloor_grid:           xarray.Dataset
    :param cases:                  cases
    :type cases:                   list
    :param DEBUG_MODE:             whether or not to run in debug mode
    :type DEBUG_MODE:              bool

    :return:                       grids
    :rtype:                        xarray.Dataset
    """
    # Loop through times
    for _age in _tqdm(ages, desc=f"Loading {type} grids", disable=(DEBUG_MODE, logging.getLogger().getEffectiveLevel() > logging.INFO)):
        # Check if the grid for the reconstruction time is already in the dictionary
        if _age in grid:
            # Rename variables and coordinates in seafloor age grid for clarity
            if type == "Seafloor":
                if "z" in grid[_age].data_vars:
                    grid[_age] = grid[_age].rename({"z": "seafloor_age"})
                if "lat" in grid[_age].coords:
                    grid[_age] = grid[_age].rename({"lat": "latitude"})
                if "lon" in grid[_age].coords:
                    grid[_age] = grid[_age].rename({"lon": "longitude"})

            continue

        # Load grid if found
        if type == "Seafloor":
            # Load grid if found
            grid[_age] = Dataset_from_netCDF(files_dir, type, _age, reconstruction_name)

            # Download seafloor age grid from GPlately DataServer
            grid[_age] = get_seafloor_grid(reconstruction_name, _age)

        elif type == "Velocity" and cases:
            # Initialise dictionary to store velocity grids for cases
            grid[_age] = {}

            # Loop through cases
            for case in cases:
                # Load grid if found
                grid[_age][_case] = Dataset_from_netCDF(files_dir, type, _age, reconstruction_name, case=case)

                # If not found, initialise a new grid
                if grid[_age][_case] is None:
                
                    # Interpolate velocity grid from points
                    if type == "Velocity":
                        for case in cases:
                            if DEBUG_MODE:
                                print(f"{type} grid for {reconstruction_name} at {_age} Ma not found, interpolating from points...")

                            # Get velocity grid
                            grid[_age][_case] = get_velocity_grid(points[_age][_case], seafloor_grid[_age])

    return grid
    
def load_data(
        data: dict,
        reconstruction_name: str,
        age: Union[list, _numpy.array],
        type: str,
        all_cases: list,
        matching_case_dict: dict,
        files_dir: Optional[str] = None,
        PARALLEL_MODE: Optional[bool] = False,
    ):
    """
    Function to load DataFrames from a folder, or initialise new DataFrames

    :return:                      data
    :rtype:                       dict
    """
    # Initialise list to store available and unavailable cases
    unavailable_cases = all_cases.copy()
    available_cases = []

    # If a file directory is provided, check for the existence of files
    if files_dir:
        for case in all_cases:
            # Load DataFrame if found
            data[case] = DataFrame_from_parquet(files_dir, type, reconstruction_name, case, age)

            if data[case] is not None:
                unavailable_cases.remove(case)
                available_cases.append(case)
            else:
                logging.info(f"DataFrame for {type} for {reconstruction_name} at {age} Ma for case {case} not found, checking for similar cases...")

    # Get available cases
    for unavailable_case in unavailable_cases:
        data[unavailable_case] = get_available_cases(data, unavailable_case, available_cases, matching_case_dict)

        if data[unavailable_case] is not None:
            available_cases.append(unavailable_case)

    return data

def get_available_cases(data, unavailable_case, available_cases, matching_case_dict):
    # Copy dataframes for unavailable cases
    matching_key = None

    # Find dictionary key of list in which unavailable case is located
    for key, matching_cases in matching_case_dict.items():
        for matching_case in matching_cases:
            if matching_case == unavailable_case:
                matching_key = key
                break
        if matching_key:
            break

    # Check if there is an available case in the corresponding list
    for matching_case in matching_case_dict[matching_key]:
        # Copy DataFrame if found
        if matching_case in available_cases:
            # Ensure that matching_case data is not None
            if data[matching_case] is not None:
                data[unavailable_case] = data[matching_case].copy()
                return data[unavailable_case]
            
            else:
                logging.info(f"Data for matching case '{matching_case}' is None; cannot copy to '{unavailable_case}'.")

    # If no matching case is found, return None
    data[unavailable_case] = None

    return data[unavailable_case]
    
    
def sample_slabs(
            self,
            ages: Optional[Union[_numpy.ndarray, List, float, int]] = None,
            cases: Optional[Union[List[str], str]] = None,
            seafloor_grid: Optional[_xarray.Dataset] = None,
            PROGRESS_BAR: Optional[bool] = True,    
        ):
        """
        Samples seafloor age (and optionally, sediment thickness) the lower plate along subduction zones
        The results are stored in the `slabs` DataFrame, specifically in the `lower_plate_age`, `sediment_thickness`, and `lower_plate_thickness` fields for each case and reconstruction time.

        :param _ages:    reconstruction times to sample slabs for
        :type _ages:     list
        :param cases:                   cases to sample slabs for (defaults to slab pull cases if not specified).
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
            iterable = self.settings.slab_pull_cases
        else:
            if isinstance(cases, str):
                cases = [cases]
            iterable = {_case: [] for _case in cases}

        # Check options for slabs
        for _age in _tqdm(ages, desc="Sampling slabs", disable=(self.DEBUG_MODE or not PROGRESS_BAR)):
            if self.DEBUG_MODE:
                print(f"Sampling slabs at {_age} Ma")

            # Select cases
            for key, entries in iterable.items():
                if self.DEBUG_MODE:
                    print(f"Sampling overriding plate for case {key} and entries {entries}...")
                    
                if self.options[key]["Slab pull torque"] or self.options[key]["Slab bend torque"]:
                    # Sample age and sediment thickness of lower plate from seafloor
                    self.data[_age][key]["lower_plate_age"], self.data[_age][key]["sediment_thickness"] = utils_calc.sample_slabs_from_seafloor(
                        self.data[_age][key].lat, 
                        self.data[_age][key].lon,
                        self.data[_age][key].trench_normal_azimuth,
                        self.seafloor[_age], 
                        self.options[key],
                        "lower plate",
                        sediment_thickness=self.data[_age][key].sediment_thickness,
                        continental_arc=self.data[_age][key].continental_arc,
                    )

                    # Calculate lower plate thickness
                    self.data[_age][key]["lower_plate_thickness"], _, _ = utils_calc.compute_thicknesses(
                        self.data[_age][key].lower_plate_age,
                        self.options[key],
                        crust = False, 
                        water = False
                    )

                    # Calculate slab flux
                    self.plates[_age][key] = utils_calc.compute_subduction_flux(
                        self.plates[_age][key],
                        self.data[_age][key],
                        type="slab"
                    )

                    if self.options[key]["Sediment subduction"]:
                        # Calculate sediment subduction
                        self.plates[_age][key] = utils_calc.compute_subduction_flux(
                            self.plates[_age][key],
                            self.data[_age][key],
                            type="sediment"
                        )

                    if len(entries) > 1:
                        for entry in entries[1:]:
                            self.data[_age][entry]["lower_plate_age"] = self.data[_age][key]["lower_plate_age"]
                            self.data[_age][entry]["sediment_thickness"] = self.data[_age][key]["sediment_thickness"]
                            self.data[_age][entry]["lower_plate_thickness"] = self.data[_age][key]["lower_plate_thickness"]

        # Set flag to True
        self.sampled_slabs = True

    def sample_upper_plates(
            self,
            _ages: Optional[Union[_numpy.ndarray, List, float, int]] = None,
            cases: Optional[Union[List[str], str]] = None,
            PROGRESS_BAR: Optional[bool] = True,    
        ):
        """
        Samples seafloor age the upper plate along subduction zones
        The results are stored in the `slabs` DataFrame, specifically in the `upper_plate_age`, `upper_plate_thickness` fields for each case and reconstruction time.

        :param _ages:    reconstruction times to sample upper plates for
        :type _ages:     list
        :param cases:                   cases to sample upper plates for (defaults to slab pull cases if not specified).
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
        
        # Make iterable
        if cases is None:
            iterable = self.slab_pull_cases
        else:
            if isinstance(cases, str):
                cases = [cases]
            iterable = {case: [] for case in cases}

        # Loop through valid times    
        for _age in tqdm(_ages, desc="Sampling upper plates", disable=(self.DEBUG_MODE or not PROGRESS_BAR)):
            if self.DEBUG_MODE:
                print(f"Sampling overriding plate at {_age} Ma")

            # Select cases
            for key, entries in iterable.items():
                if self.DEBUG_MODE:
                    print(f"Sampling overriding plate for case {key} and entries {entries}...")

                # Check whether to output erosion rate and sediment thickness
                if self.options[key]["Sediment subduction"] and self.options[key]["Active margin sediments"] != 0 and self.options[key]["Sample erosion grid"] in self.seafloor[_age].data_vars:
                    # Sample age and arc type, erosion rate and sediment thickness of upper plate from seafloor
                    self.data[_age][key]["upper_plate_age"], self.data[_age][key]["continental_arc"], self.data[_age][key]["erosion_rate"], self.data[_age][key]["sediment_thickness"] = utils_calc.sample_slabs_from_seafloor(
                        self.data[_age][key].lat, 
                        self.data[_age][key].lon,
                        self.data[_age][key].trench_normal_azimuth,  
                        self.seafloor[_age],
                        self.options[key],
                        "upper plate",
                        sediment_thickness=self.data[_age][key].sediment_thickness,
                    )

                elif self.options[key]["Sediment subduction"] and self.options[key]["Active margin sediments"] != 0:
                    # Sample age and arc type of upper plate from seafloor
                    self.data[_age][key]["upper_plate_age"], self.data[_age][key]["continental_arc"] = utils_calc.sample_slabs_from_seafloor(
                        self.data[_age][key].lat, 
                        self.data[_age][key].lon,
                        self.data[_age][key].trench_normal_azimuth,  
                        self.seafloor[_age],
                        self.options[key],
                        "upper plate",
                    )
                
                # Copy DataFrames to other cases
                if len(entries) > 1 and cases is None:
                    for entry in entries[1:]:
                        self.data[_age][entry]["upper_plate_age"] = self.data[_age][key]["upper_plate_age"]
                        self.data[_age][entry]["continental_arc"] = self.data[_age][key]["continental_arc"]
                        if self.options[key]["Sample erosion grid"]:
                            self.data[_age][entry]["erosion_rate"] = self.data[_age][key]["erosion_rate"]
                            self.data[_age][entry]["sediment_thickness"] = self.data[_age][key]["sediment_thickness"]
        
        # Set flag to True
        self.sampled_upper_plates = True

def get_globe_data(
        plates: dict,
        slabs: dict,
        points: dict,
        seafloor_grid: dict,
        ages: _numpy.array,
        case: str,
    ):
    """
    Function to get relevant geodynamic data for the entire globe.

    :param plates:                plates
    :type plates:                 dict
    :param slabs:                 slabs
    :type slabs:                  dict
    :param points:                points
    :type points:                 dict
    :param seafloor_grid:         seafloor grid
    :type seafloor_grid:          dict

    :return:                      globe
    :rtype:                       pandas.DataFrame
    """
    # Initialise empty arrays
    num_plates = _numpy.zeros_like(ages)
    slab_length = _numpy.zeros_like(ages)
    v_rms_mag = _numpy.zeros_like(ages)
    v_rms_azi = _numpy.zeros_like(ages)
    mean_seafloor_age = _numpy.zeros_like(ages)

    for i, age in enumerate(ages):
        # Get number of plates
        num_plates[i] = len(plates[age][case].plateID.values)

        # Get slab length
        slab_length[i] = slabs[age][case].trench_segment_length.sum()

        # Get global RMS velocity
        # Get area for each grid point as well as total area
        areas = points[age][case].segment_length_lat.values * points[age][case].segment_length_lon.values
        total_area = _numpy.sum(areas)

        # Calculate RMS speed
        v_rms_mag[i] = _numpy.sum(points[age][case].v_mag * areas) / total_area

        # Calculate RMS azimuth
        v_rms_sin = _numpy.sum(_numpy.sin(points[age][case]["velocity_lat"]) * areas) / total_area
        v_rms_cos = _numpy.sum(_numpy.cos(points[age][case]["velocity_lat"]) * areas) / total_area
        v_rms_azi[i] = _numpy.rad2deg(
            -1 * (_numpy.arctan2(v_rms_sin, v_rms_cos) + 0.5 * _numpy.pi)
        )

        # Get mean seafloor age
        mean_seafloor_age[i] = _numpy.nanmean(_seafloor_grid[_age].seafloor_age.values)

    # Organise as pd.DataFrame
    globe = _pandas.DataFrame({
        "number_of_plates": num_plates,
        "total_slab_length": slab_length,
        "v_rms_mag": v_rms_mag,
        "v_rms_azi": v_rms_azi,
        "mean_seafloor_age": mean_seafloor_age,
    })
        
    return globe