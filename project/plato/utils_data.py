# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# PLATO
# Algorithm to calculate plate forces from tectonic reconstructions
# Setup
# Thomas Schouten, 2023
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Import libraries
# Standard libraries
import contextlib
import io
import os as _os
import logging as logging
import tempfile
import shutil
import warnings

from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Union

# Third-party libraries
import geopandas as _geopandas

import numpy as _numpy
import gplately as _gplately
import pandas as _pandas
import pygplates as _pygplates
import xarray as _xarray
from tqdm import tqdm as _tqdm

# Local libraries
from utils_calc import set_constants, mag_azi2lat_lon, project_points

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# INITIALISATION 
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

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
        constants = set_constants()

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

def get_slab_data(
        reconstruction: _gplately.PlateReconstruction,
        age: int,
        topology_geometries: _geopandas.GeoDataFrame,
        options: dict,
        PARALLEL_MODE: Optional[bool] = False,
        ):
        """
        Function to get data on slabs in reconstruction.

        :param reconstruction:      reconstruction
        :type reconstruction:       _gplately.PlateReconstruction
        :param age:                 reconstruction time
        :type age:                  integer
        :param topology_geometries: topology geometries
        :type topology_geometries:  geopandas.GeoDataFrame
        :param options:             options for the case
        :type options:              dict
        :param DEBUG_MODE:          whether to run in debug mode
        :type DEBUG_MODE:           bool
        :param PARALLEL_MODE:       whether to run in parallel mode
        :type PARALLEL_MODE:        bool
        
        :return:                    slabs
        :rtype:                     pandas.DataFrame
        """
        # Set constants
        constants = set_constants()

        # Tesselate subduction zones and get slab pull and bend torques along subduction zones
        slabs = reconstruction.tessellate_subduction_zones(age, ignore_warnings=True, tessellation_threshold_radians=(options["Slab tesselation spacing"]/constants.mean_Earth_radius_km))

        # Convert to _pandas.DataFrame
        slabs = _pandas.DataFrame(slabs)

        # Kick unused columns
        slabs = slabs.drop(columns=[2, 3, 4, 5])

        slabs.columns = ["lon", "lat", "trench_segment_length", "trench_normal_azimuth", "lower_plateID", "trench_plateID"]

        # Convert trench segment length from degree to m
        slabs.trench_segment_length *= constants.equatorial_Earth_circumference / 360

        # Get plateIDs of overriding plates
        sampling_lat, sampling_lon = project_points(
            slabs.lat,
            slabs.lon,
            slabs.trench_normal_azimuth,
            100
        )
        slabs["upper_plateID"] = get_plateIDs(
            reconstruction,
            topology_geometries,
            sampling_lat,
            sampling_lon,
            age,
            PARALLEL_MODE=PARALLEL_MODE
        )

        # Initialise columns to store convergence rates
        types = ["upper_plate", "lower_plate", "convergence"]
        coords = ["lat", "lon", "mag"]
        slabs[[f"{type}_v_{coord}" for type in types for coord in coords]] = [[0.] * len(coords) * len(types) for _ in range(len(slabs))]

        # Initialise other columns to store seafloor ages and forces
        # Upper plate
        slabs["upper_plate_thickness"] = 0.
        slabs["upper_plate_age"] = 0.
        slabs["continental_arc"] = False
        slabs["erosion_rate"] = 0.

        # Lower plate
        slabs["lower_plate_age"] = 0.
        slabs["lower_plate_thickness"] = 0.
        slabs["sediment_thickness"] = 0.
        slabs["sediment_fraction"] = 0.
        slabs["slab_length"] = options["Slab length"]

        # Forces
        forces = ["slab_pull", "slab_bend", "residual"]
        coords = ["mag", "lat", "lon"]
        slabs[[force + "_force_" + coord for force in forces for coord in coords]] = [[0.] * len(coords) * len(forces) for _ in range(len(slabs))]
        slabs["residual_force_azi"] = 0.
        slabs["residual_alignment"] = 0.

        # Make sure all the columns are floats
        slabs = slabs.apply(lambda x: x.astype(float) if x.name != "continental_arc" else x)

        return slabs

def get_point_data(
        reconstruction: _gplately.PlateReconstruction,
        age: int,
        topology_geometries: _geopandas.GeoDataFrame,
        options: dict,
        PARALLEL_MODE: Optional[bool] = False,
    ):
    """
    Function to get data on regularly spaced grid points in reconstruction.

    :param reconstruction:      Reconstruction
    :type reconstruction:       gplately.PlateReconstruction
    :param age:                 reconstruction time
    :type age:                  integer
    :param plates:              plates
    :type plates:               pandas.DataFrame
    :param topology_geometries: topology geometries
    :type topology_geometries:  geopandas.GeoDataFrame
    :param options:             options for the case
    :type options:              dict

    :return:                    points
    :rtype:                     pandas.DataFrame    
    """
    # Set constants
    constants = set_constants()
    
    # Define grid spacing and 
    lats = _numpy.arange(-90,91,options["Grid spacing"], dtype=float)
    lons = _numpy.arange(-180,180,options["Grid spacing"], dtype=float)

    # Create a meshgrid of latitudes and longitudes
    lon_grid, lat_grid = _numpy.meshgrid(lons, lats)
    lon_grid, lat_grid = lon_grid.flatten(), lat_grid.flatten()

    # Get plateIDs for points
    plateIDs = get_plateIDs(
        reconstruction,
        topology_geometries,
        lat_grid,
        lon_grid,
        age,
        PARALLEL_MODE=PARALLEL_MODE
    )

    # Convert degree spacing to metre spacing
    segment_length_lat = constants.mean_Earth_radius_m * (_numpy.pi/180) * options["Grid spacing"]
    segment_length_lon = constants.mean_Earth_radius_m * (_numpy.pi/180) * _numpy.cos(_numpy.deg2rad(lat_grid)) * options["Grid spacing"]

    # Organise as DataFrame
    points = _pandas.DataFrame({"lat": lat_grid, 
                           "lon": lon_grid, 
                           "plateID": plateIDs, 
                           "segment_length_lat": segment_length_lat,
                           "segment_length_lon": segment_length_lon,
                           },
                           dtype=float
                        )

    # Add additional columns to store velocities
    components = ["v_lat", "v_lon", "v_mag", "v_azi", "omega"]
    points[[component for component in components]] = [[0.] * len(components) for _ in range(len(points))]

    # Add additional columns to store seafloor properties
    points["seafloor_age"] = 0
    points["lithospheric_thickness"] = 0
    points["crustal_thickness"] = 0
    points["water_depth"] = 0
    points["U"] = 0

    # Add additional columns to store forces
    forces = ["GPE", "mantle_drag"]
    coords = ["lat", "lon", "mag"]
    points[[force + "_force_" + coord for force in forces for coord in coords]] = [[0.] * len(forces) * len(coords) for _ in range(len(points))]
    
    return points

def get_globe_data(
        _plates: dict,
        _slabs: dict,
        _points: dict,
        _seafloor_grid: dict,
        _ages: _numpy.array,
        _case: str,
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
    num_plates = _numpy.zeros_like(_ages)
    slab_length = _numpy.zeros_like(_ages)
    v_rms_mag = _numpy.zeros_like(_ages)
    v_rms_azi = _numpy.zeros_like(_ages)
    mean_seafloor_age = _numpy.zeros_like(_ages)

    for i, _age in enumerate(_ages):
        # Get number of plates
        num_plates[i] = len(_plates[_age][_case].plateID.values)

        # Get slab length
        slab_length[i] = _slabs[_age][_case].trench_segment_length.sum()

        # Get global RMS velocity
        # Get area for each grid point as well as total area
        areas = _points[_age][_case].segment_length_lat.values * _points[_age][_case].segment_length_lon.values
        total_area = _numpy.sum(areas)

        # Calculate RMS speed
        v_rms_mag[i] = _numpy.sum(_points[_age][_case].v_mag * areas) / total_area

        # Calculate RMS azimuth
        v_rms_sin = _numpy.sum(_numpy.sin(_points[_age][_case].v_lat) * areas) / total_area
        v_rms_cos = _numpy.sum(_numpy.cos(_points[_age][_case].v_lat) * areas) / total_area
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

def get_resolved_topologies(
        reconstruction: _gplately.PlateReconstruction,
        ages: Union[int, float, _numpy.floating, _numpy.integer, List, _numpy.array],
        filename: Optional[str] = None,
    ) -> Dict:
    """
    Function to get resolved geometries for all ages.
    """
    if isinstance(ages, (int, float, _numpy.floating, _numpy.integer)):
        ages = _numpy.array([ages])
    elif isinstance(ages, list):
        ages = _numpy.array(ages)

    # Initialise dictionary to store resolved topologies
    _resolved_topologies = {}

    # Initialise dictionary to store resolved topologies
    _resolved_topologies = {_age: [] for _age in ages}

    for _age in ages:
        if filename:
            # Initialise list to store resolved topologies for each age
            resolved_topologies = filename
        else:
            resolved_topologies = []

        # Resolve topologies for the current age
        with warnings.catch_warnings():
            # Ignore warnings about field name laundering
            warnings.filterwarnings(
                action="ignore",
                message="Normalized/laundered field name:"
            )
            _pygplates.resolve_topologies(
                reconstruction.topology_features,
                reconstruction.rotation_model,
                resolved_topologies,
                _age,
                anchor_plate_id=0
            )
        
        # Store the resolved topologies for the current age
        _resolved_topologies[_age] = resolved_topologies
    
    return _resolved_topologies
    
def get_resolved_geometries(
        reconstruction: _gplately.PlateReconstruction,
        ages: _numpy.array,
        resolved_topologies: Optional[Dict] = None,
    ) -> Dict:
    """
    Function to obtain resolved geometries as GeoDataFrames for all ages.

    :param reconstruction:        reconstruction
    :type reconstruction:         gplately.PlateReconstruction
    :param ages:                  ages
    :type ages:                   numpy.array
    :param resolved_topologies:   resolved topologies
    :type resolved_topologies:    dict

    :return:                      resolved_geometries
    :rtype:                       dict
    """
    # Make temporary directory to hold shapefiles
    temp_dir = tempfile.mkdtemp()

    # Initialise dictionary to store resolved geometries
    resolved_geometries = {}

    for _age in ages:
        # Save resolved topologies as shapefiles
        if resolved_topologies is None:
            topology_file = _os.path.join(temp_dir, f"topologies_{_age}.shp")
            resolved_topologies = get_resolved_topologies(reconstruction, _age, topology_file)

        # Load resolved topologies as GeoDataFrames
        resolved_geometries[_age] = _geopandas.read_file(topology_file)
    
    # Remove temporary directory
    shutil.rmtree(temp_dir)

    return resolved_geometries

def extract_geometry_data(topology_geometries):
    """
    Function to extract only the geometry and plateID from topology geometries.

    :param topology_geometries:        topology geometries
    :type topology_geometries:         geopandas.GeoDataFrame

    :return:                           geometries_data
    :rtype:                            list
    """
    return [(geom, plateID) for geom, plateID in zip(topology_geometries.geometry, topology_geometries.PLATEID1)]

def process_plateIDs(
        geometries_data: list,
        lats_chunk: _numpy.array,
        lons_chunk: _numpy.array,
    ) -> list:
    """
    Function to process plateIDs for a chunk of latitudes and longitudes.

    :param geometries_data:        geometry data
    :type geometries_data:         list
    :param lats_chunk:             chunk of latitudes
    :type lats_chunk:              numpy.array
    :param lons_chunk:             chunk of longitudes
    :type lons_chunk:              numpy.array

    :return:                       plateIDs
    :rtype:                        numpy.array
    """
    plateIDs = _numpy.zeros(len(lats_chunk))
    
    for topology_geometry, topology_plateID in geometries_data:
        mask = topology_geometry.contains(_geopandas.points_from_xy(lons_chunk, lats_chunk))
        plateIDs[mask] = topology_plateID

        # Break if all points have been assigned a plate ID
        if plateIDs.all():
            break

    return plateIDs

def get_plateIDs(
        reconstruction: _gplately.PlateReconstruction,
        topology_geometries: _geopandas.GeoDataFrame,
        lats: Union[List, _numpy.array],
        lons: Union[List, _numpy.array],
        age: int,
        PARALLEL_MODE: Optional[bool] = False,
    ):
    """
    Function to get plate IDs for a set of latitudes and longitudes.

    :param reconstruction:             reconstruction
    :type reconstruction:              _gplately.PlateReconstruction
    :param topology_geometries:        topology geometries
    :type topology_geometries:         geopandas.GeoDataFrame
    :param lats:                       latitudes
    :type lats:                        list or _numpy.array
    :param lons:                       longitudes
    :type lons:                        list or _numpy.array
    :param _age:        reconstruction time
    :type _age:         integer

    :return:                           plateIDs
    :rtype:                            list
    """
    # Convert lats and lons to numpy arrays if they are not already
    lats = _numpy.asarray(lats)
    lons = _numpy.asarray(lons)
    
    # Extract geometry data
    geometries_data = extract_geometry_data(topology_geometries)

    # Get plateIDs for the entire dataset
    plateIDs = process_plateIDs(geometries_data, lats, lons)
    
    # Use vectorised operations to find and assign plate IDs for remaining points
    no_plateID_mask = plateIDs == 0
    if no_plateID_mask.any():
        no_plateID_grid = _gplately.Points(
            reconstruction,
            lons[no_plateID_mask],
            lats[no_plateID_mask],
            time=int(age),
        )
        plateIDs[no_plateID_mask] = no_plateID_grid.plate_id

    return plateIDs

def get_velocities(
        lats: Union[List, _numpy.array],
        lons: Union[List, _numpy.array],
        stage_rotation: tuple,
    ) -> Tuple[_numpy.array, _numpy.array, _numpy.array, _numpy.array]:
    """
    Function to get velocities for a set of latitudes and longitudes.
    NOTE: This function is not vectorised yet, but has not been a bottleneck in the code so far.

    :param lats:                     latitudes
    :type lats:                      list or numpy.array
    :param lons:                     longitudes
    :type lons:                      list or numpy.array
    :param stage_rotation:           stage rotation defined by pole latitude, pole longitude and pole angle
    :type stage_rotation:            tuple

    :return:                         velocities_lat, velocities_lon, velocities_mag, velocities_azi
    :rtype:                          numpy.array, numpy.array, numpy.array, numpy.array
    """
    # Convert lats and lons to numpy arrays if they are not already
    lats = _numpy.asarray(lats)
    lons = _numpy.asarray(lons)

    # Initialise empty array to store velocities
    velocities_lat = _numpy.zeros(len(lats))
    velocities_lon = _numpy.zeros(len(lats))
    velocities_mag = _numpy.zeros(len(lats))
    velocities_azi = _numpy.zeros(len(lats))

    # Loop through points to get velocities
    for i, _ in enumerate(lats):
        # Convert to LocalCartesian
        point = _pygplates.PointOnSphere((lats[i], lons[i]))

        # Calculate magnitude and azimuth of velocities at points
        velocity_mag_azi = _numpy.asarray(
            _pygplates.LocalCartesian.convert_from_geocentric_to_magnitude_azimuth_inclination(
                point,
                _pygplates.calculate_velocities(
                    point, 
                    _pygplates.FiniteRotation((stage_rotation[0], stage_rotation[1]), _numpy.deg2rad(stage_rotation[2])), 
                    1.,
                    velocity_units = _pygplates.VelocityUnits.cms_per_yr
                )
            )
        )

        # Get magnitude and azimuth of velocities
        velocities_mag[i] = velocity_mag_azi[0][0]; velocities_azi[i] = velocity_mag_azi[0][1]

    # Convert to lat and lon components
    velocities_lat, velocities_lon = mag_azi2lat_lon(velocities_mag, _numpy.rad2deg(velocities_azi))

    return velocities_lat, velocities_lon, velocities_mag, velocities_azi

def get_topology_geometries(
        reconstruction: _gplately.PlateReconstruction,
        age: int,
        anchor_plateID: int
    ):
    """
    Function to resolve topologies and get geometries as a GeoDataFrame

    :param reconstruction:        reconstruction
    :type reconstruction:         gplately.PlateReconstruction
    :param _age:   reconstruction time
    :type _age:    integer
    :param anchor_plateID:        anchor plate ID
    :type anchor_plateID:         integer
    :return:                      resolved_topologies
    :rtype:                       geopandas.GeoDataFrame
    """
    # Make temporary directory to hold shapefiles
    temp_dir = tempfile.mkdtemp()

    # Resolve topological networks and load as GeoDataFrame
    topology_file = _os.path.join(temp_dir, "topologies.shp")
    with warnings.catch_warnings():
        warnings.filterwarnings(
            action="ignore",
            message="Normalized/laundered field name:"
        )
        _pygplates.resolve_topologies(
            reconstruction.topology_features, 
            reconstruction.rotation_model, 
            topology_file, 
            int(age), 
            anchor_plate_id=anchor_plateID
        )
        if _os.path.exists(topology_file):
            topology_geometries = _geopandas.read_file(topology_file)

    # Remove temporary directory
    shutil.rmtree(temp_dir)

    return topology_geometries

def get_geometric_properties(
        plates: _pandas.DataFrame,
        slabs: _pandas.DataFrame,
    ):
    """
    Function to get geometric properties of plates.

    :param plates:                plates
    :type plates:                 pandas.DataFrame
    :param slabs:                 slabs
    :type slabs:                  pandas.DataFrame

    :return:                      plates
    :rtype:                       pandas.DataFrame
    """
    # Calculate trench length and omega
    for plateID in plates.plateID:
        if plateID in slabs.lower_plateID.unique():
            plates.loc[plates.plateID == plateID, "trench_length"] = slabs[slabs.lower_plateID == plateID].trench_segment_length.sum()
            plates.loc[plates.plateID == plateID, "zeta"] = plates[plates.plateID == plateID].area.values[0] / plates[plates.plateID == plateID].trench_length.values[0]

    return plates

def get_plate_names(
        plate_id_list: Union[list or _numpy.array],
    ):
    """
    Function to get plate names corresponding to plate ids

    :param plate_id_list:        list of plate ids
    :type plate_id_list:         list or numpy.array

    :return:                     plate_names
    :rtype:                      list
    """
    plate_name_dict = {
        101: "N America",
        201: "S America",
        301: "Eurasia",
        302: "Baltica",
        501: "India",
        503: "Arabia",
        511: "Capricorn",
        701: "S Africa",
        702: "Madagascar",
        709: "Somalia",
        714: "NW Africa",
        715: "NE Africa",
        801: "Australia",
        802: "Antarctica",
        901: "Pacific",
        902: "Farallon",
        904: "Aluk",
        909: "Cocos",
        911: "Nazca",
        918: "Kula",
        919: "Phoenix",
        926: "Izanagi",
        5400: "Burma",
        5599: "Tethyan Himalaya",
        7520: "Argoland",
        9002: "Farallon",
        9006: "Izanami",
        9009: "Izanagi",
        9010: "Pontus"
    } 

    # Create a defaultdict with the default value as the plate ID
    default_plate_name = defaultdict(lambda: "Unknown", plate_name_dict)

    # Retrieve the plate names based on the plate IDs
    plate_names = [default_plate_name[plate_id] for plate_id in plate_id_list]

    return plate_names

def get_options(
        file_name: Optional[str] = None,
        sheet_name: Optional[str] = None
    ) -> Tuple[List[str], Dict[str, Dict[str, Optional[str]]]]:
    """
    Function to get options from excel file. If no arguments are provided,
    returns the default options and assigns 'ref' to the case.

    :param file_name:            file name (optional)
    :type file_name:             str, optional
    :param sheet_name:           sheet name (optional)
    :type sheet_name:            str, optional

    :return:                     cases, options
    :rtype:                      list, dict
    """
    # Define all options
    all_options = ["Slab pull torque",
                   "GPE torque",
                   "Mantle drag torque",
                   "Slab bend torque",
                   "Slab bend mechanism",
                   "Reconstructed motions",
                   "Continental crust",
                   "Seafloor age variable",
                   "Seafloor age profile",
                   "Sample sediment grid", 
                   "Active margin sediments",
                   "Sample erosion grid", 
                   "Erosion to sediment ratio",
                   "Sediment subduction",
                   "Shear zone width",
                   "Slab length",
                   "Strain rate",
                   "Slab pull constant",
                   "Mantle viscosity",
                   "Slab tesselation spacing",
                   "Grid spacing",
                   "Minimum plate area",
                   "Anchor plateID",
                   "Velocity time step"
                   ]
    
    # Define default values
    default_values = [True,
                      True,
                      True,
                      False,
                      "viscous",
                      True,
                      False,
                      "z",
                      "half space cooling",
                      False,
                      0,
                      False,
                      2,
                      False,
                      2e3,
                      700e3,
                      1e-12,
                      0.0316,
                      1.22e20,
                      250,
                      1,
                      7.5e12,
                      0,
                      1,
                      ]

    # Adjust TRUE/FALSE values in excel file to boolean
    boolean_options = ["Slab pull torque",
                       "GPE torque",
                       "Mantle drag torque",
                       "Slab bend torque",
                       "Reconstructed motions",
                       "Continental crust",
                       "Randomise trench orientation",
                       "Randomise slab age"]

    # If no file_name is provided, return default values with case "ref"
    if not file_name:
        cases = ["ref"]
        options = {"ref": {option: default_values[i] for i, option in enumerate(all_options)}}
        return cases, options

    # Read file
    case_options = _pandas.read_excel(file_name, sheet_name=sheet_name, comment="#")

    # Initialise list of cases
    cases = []

    # Initialise options dictionary
    options = {}

    # Loop over rows to obtain options from excel file
    for _, row in case_options.iterrows():
        _case = row.get("Name", "ref")  # Assign "ref" if no Name column or no case name
        cases.append(_case)
        options[_case] = {}
        for i, option in enumerate(all_options):
            if option in case_options.columns:
                # Handle boolean conversion
                if option in boolean_options and row[option] == 1:
                    row[option] = True
                elif option in boolean_options and row[option] == 0:
                    row[option] = False
                options[_case][option] = row[option]
            else:
                options[_case][option] = default_values[i]

    # If no cases were found, use the default "ref" case
    if not cases:
        cases = ["ref"]
        options["ref"] = {option: default_values[i] for i, option in enumerate(all_options)}

    return cases, options


def get_seafloor_grid(
        reconstruction_name: str,
        _age: int,
        DEBUG_MODE: bool = False
    ) -> _xarray.Dataset:
    """
    Function to obtain seafloor grid from GPlately DataServer.
    
    :param reconstruction_name:    name of reconstruction
    :type reconstruction_name:     string
    :param ages:   reconstruction times
    :type ages:    list or numpy.array
    :param DEBUG_MODE:             whether to run in debug mode
    :type DEBUG_MODE:              bool

    :return:                       seafloor_grids
    :rtype:                        xarray.Dataset
    """
    # Call _gplately"s DataServer from the download.py module
    gdownload = _gplately.download.DataServer(reconstruction_name)

    # Inform the user of the ongoing process if in debug mode
    logger.debug(f"Downloading age grid for {reconstruction_name} at {_age} Ma")

    # Download the age grid, suppressing stdout output if not in debug mode
    if DEBUG_MODE:
        age_raster = gdownload.get_age_grid(time=_age)
    else:
        with contextlib.redirect_stdout(io.StringIO()):
            age_raster = gdownload.get_age_grid(time=_age)

    # Convert the data to a masked array
    seafloor_ages_ma = _numpy.ma.masked_invalid(age_raster.data)
    
    # Convert the masked array to a regular numpy array with NaN for masked values
    seafloor_ages = seafloor_ages_ma.filled(_numpy.nan)

    lon = age_raster.lons
    lat = age_raster.lats

    # Create a xarray dataset
    age_grid = _xarray.Dataset(
        {
            "seafloor_age": (["latitude", "longitude"], seafloor_ages.astype(_numpy.float64)),
        },
        coords={
            "latitude": lat,
            "longitude": lon,
        },
    )
    
    return age_grid

def get_velocity_grid(
        points: _pandas.DataFrame,
        seafloor_grid: _xarray.DataArray,
    ):
    """
    Function to obtain velocity grid from the velocity sampled at the points interpolated to the resolution of the seafloor grid.

    :param reconstruction_name:    name of reconstruction
    :type reconstruction_name:     string
    :param _age:    reconstruction time
    :type _age:     integer
    :param seafloor_grid:          seafloor ages
    :type seafloor_grid:           xarray.DataArray

    :return:                       velocity_grid
    :rtype:                        xarray.Dataset
    """
    # Make xarray velocity grid
    velocity_grid = _xarray.Dataset(
            {
                "velocity_magnitude": (["latitude", "longitude"], points.v_mag.values.reshape(points.lat.unique().size, points.lon.unique().size)),
                "velocity_latitude": (["latitude", "longitude"], points.v_lat.values.reshape(points.lat.unique().size, points.lon.unique().size)),
                "velocity_longitude": (["latitude", "longitude"], points.v_lon.values.reshape(points.lat.unique().size, points.lon.unique().size)),
            },
            coords={
                "latitude": points.lat.unique(),
                "longitude": points.lon.unique(),
            },
        )
    
    # Interpolate to resolution of seafloor grid
    velocity_grid = velocity_grid.interp(latitude=seafloor_grid.latitude, longitude=seafloor_grid.longitude, method="linear")

    # Interpolate NaN values along the dateline
    velocity_grid = velocity_grid.interpolate_na()

    return velocity_grid

def get_ages(
        ages: Union[None, int, float, list, _numpy.integer, _numpy.floating, _numpy.ndarray],
        default_ages: _numpy.ndarray,
    ) -> _numpy.ndarray:
    """
    Function to check and get ages.

    :param ages:            ages
    :type ages:             None, int, float, list, numpy.integer, numpy.floating, numpy.ndarray
    :param default_ages:    settings ages
    :type default_ages:     numpy.ndarray

    :return:                ages
    :rtype:                 numpy.ndarray
    """
    # Define ages
    if ages is None:
        # If no ages are provided, use default ages
        _ages = default_ages

    elif isinstance(ages, (int, float, _numpy.integer, _numpy.floating)):
        # If a single value is provided, convert to numpy array
        _ages = _numpy.array([ages])

    elif isinstance(ages, list):
        # If a list is provided, convert to numpy array
        _ages = _numpy.array(ages)

    elif isinstance(ages, _numpy.ndarray):
        # If a numpy array is provided, use as is
        _ages = ages

    return _ages

def get_cases(
    cases: Union[None, str, List[str]],
    default_cases: List[str],
    ) -> List[str]:
    """
    Function to check and get cases.

    :param cases:           cases (can be None, a single case as a string, or a list of cases)
    :type cases:            None, str, or list of strings
    :param default_cases:   default cases to use if cases is not provided
    :type default_cases:    list of strings

    :return:                 a list of cases
    :rtype:                  list of strings
    """
    # Define cases
    if cases is None:
        # If no cases are provided, use default cases
        _cases = default_cases

    else:
        # Check if cases is a single value (str), convert to list
        if isinstance(cases, str):
            _cases = [cases]

    return _cases

def get_iterable(
        cases: Union[None, str, List[str]],
        default_iterable: List[str],
    ) -> Dict[str, List[str]]:
    """
    Function to check and get iterable.

    :param cases:               cases (can be None, a single case as a string, or a list of cases)
    :type cases:                None, str, or list of strings
    :param default_iterable:    default iterable to use if cases is not provided
    :type default_iterable:     list of strings

    :return:                 iterable
    :rtype:                  dict
    """
    # Define iterable
    if cases is None:
        # If no cases are provided, use the default iterable
        _iterable = default_iterable

    else:
        # Check if cases is a single value (str), convert to list
        if isinstance(cases, str):
            cases = [cases]
        
        # Make dictionary of iterable
        _iterable = {case: [] for case in cases}

    return _iterable

def get_plates(
        plate_IDs: Union[None, int, list, _numpy.integer, _numpy.floating, _numpy.ndarray],
    ) -> _numpy.ndarray:
    """
    Function to check and get plate IDs.

    :param plate_IDs:        plate IDs
    :type plate_IDs:         None, int, list, numpy.integer, numpy.floating, numpy.ndarray

    :return:                 plate IDs
    :rtype:                  numpy.ndarray
    """
    # Define plate IDs
    if isinstance(plates, (int, float, _numpy.floating, _numpy.integer)):
            plates = [plates]

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# PROCESS CASES 
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- 

def process_cases(cases, options, target_options):
    """
    Function to process cases and options to accelerate computation. Each case is assigned a dictionary of identical cases for a given set of target options.
    The goal here is that if these target options are identical, the computation is only peformed once and the results are copied to the other cases.

    :param cases:           cases
    :type cases:            list
    :param options:         options
    :type options:          dict
    :param target_options:  target options
    :type target_options:   list

    :return:                case_dict
    :rtype:                 dict
    """
    # Initialise dictionary to store processed cases
    processed_cases = set()
    case_dict = {}

    # Loop through cases to process
    for _case in cases:
        # Ignore processed cases
        if _case in processed_cases:
            continue
        
        # Initialise list to store similar cases
        case_dict[_case] = [_case]

        # Add case to processed cases
        processed_cases.add(_case)

        # Loop through other cases to find similar cases
        for other_case in cases:
            # Ignore if it is the same case
            if _case == other_case:
                continue
            
            # Add case to processed cases if it is similar
            if all(options[_case][opt] == options[other_case][opt] for opt in target_options):
                case_dict[_case].append(other_case)
                processed_cases.add(other_case)

    return case_dict

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# SAVING 
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def DataFrame_to_parquet(
        data: _pandas.DataFrame,
        data_name: str,
        reconstruction_name: str,
        _age: int,
        case: str,
        folder: str,
    ):
    """
    Function to save DataFrame to a Parquet file in a folder efficiently.

    :param data:                  data
    :type data:                   pandas.DataFrame
    :param data_name:             name of dataset
    :type data_name:              string
    :param reconstruction_name:   name of reconstruction
    :type reconstruction_name:    string
    :param _age:   reconstruction time
    :type _age:    integer
    :param case:                  case
    :type case:                   string
    :param folder:                folder name
    :type folder:                 string
    :param DEBUG_MODE:            whether to run in debug mode
    :type DEBUG_MODE:             bool
    """
    # Construct the file path
    target_dir = folder if folder else _os.getcwd()
    file_name = f"{data_name}_{reconstruction_name}_{case}_{_age}Ma.parquet"
    file_path = _os.path.join(target_dir, data_name, file_name)
    
    # Debug information
    logger.info(f"Saving {data_name} to {file_path}")

    # Ensure the directory exists
    _os.makedirs(_os.path.dirname(file_path), exist_ok=True)
    
    # Delete old file if it exists
    try:
        _os.remove(file_path)
    except FileNotFoundError:
        pass  # No need to remove if file does not exist

    # Save the data to Parquet
    data.to_parquet(file_path, index=False)

def DataFrame_to_csv(
        data: _pandas.DataFrame,
        data_name: str,
        reconstruction_name: str,
        _age: int,
        case: str,
        folder: str,
        DEBUG_MODE: bool = False
    ):
    """
    Function to save DataFrame to a folder efficiently.

    :param data:                  data
    :type data:                   pandas.DataFrame
    :param data_name:             name of dataset
    :type data_name:              string
    :param reconstruction_name:   name of reconstruction
    :type reconstruction_name:    string
    :param _age:   reconstruction time
    :type _age:    integer
    :param case:                  case
    :type case:                   string
    :param folder:                folder name
    :type folder:                 string
    :param DEBUG_MODE:            whether to run in debug mode
    :type DEBUG_MODE:             bool
    """
    # Construct the file path
    target_dir = folder if folder else _os.getcwd()
    file_name = f"{data_name}_{reconstruction_name}_{case}_{_age}Ma.csv"
    file_path = _os.path.join(target_dir, data_name, file_name)
    
    # Debug information
    logger.info(f"Saving {data_name} to {file_path}")

    # Ensure the directory exists
    _os.makedirs(_os.path.dirname(file_path), exist_ok=True)
    
    # Delete old file if it exists
    try:
        _os.remove(file_path)
    except FileNotFoundError:
        pass  # No need to remove if file does not exist

    # Save the data to CSV
    data.to_csv(file_path, index=False)
    
def GeoDataFrame_to_geoparquet(
        data: _geopandas.GeoDataFrame,
        data_name: str,
        reconstruction_name: str,
        _age: int,
        folder: str,
        DEBUG_MODE: bool = False
    ):
    """
    Function to save GeoDataFrame to a GeoParquet file in a folder efficiently.

    :param data:                  data
    :type data:                   geopandas.GeoDataFrame
    :param data_name:             name of dataset
    :type data_name:              string
    :param reconstruction_name:   name of reconstruction
    :type reconstruction_name:    string
    :param _age:   age of reconstruction in Ma
    :type _age:    int
    :param folder:                folder name
    :type folder:                 string
    :param DEBUG_MODE:            whether to run in debug mode
    :type DEBUG_MODE:             bool
    """
    # Construct the target directory and file path
    target_dir = _os.path.join(folder if folder else _os.getcwd(), data_name)
    file_name = f"{data_name}_{reconstruction_name}_{_age}Ma.parquet"
    file_path = _os.path.join(target_dir, file_name)
    
    # Debug information
    if DEBUG_MODE:
        print(f"Target directory for {data_name}: {target_dir}")
        print(f"File path for {data_name} at {_age}: {file_path}")

    # Ensure the directory exists
    _os.makedirs(target_dir, exist_ok=True)
    
    # Delete old file if it exists
    try:
        _os.remove(file_path)
        if DEBUG_MODE:
            print(f"Deleted old file {file_path}")
    except FileNotFoundError:
        pass  # File does not exist, no need to remove

    # Save the data to a GeoParquet file
    data.to_parquet(file_path)

def GeoDataFrame_to_shapefile(
        data: _geopandas.GeoDataFrame,
        data_name: str,
        reconstruction_name: str,
        _age: int,
        folder: str,
        DEBUG_MODE: bool = False
    ):
    """
    Function to save GeoDataFrame to a folder efficiently.

    :param data:                  data
    :type data:                   geopandas.GeoDataFrame
    :param data_name:             name of dataset
    :type data_name:              string
    :param reconstruction_name:   name of reconstruction
    :type reconstruction_name:    string
    :param _age:   age of reconstruction in Ma
    :type _age:    int
    :param folder:                folder
    :type folder:                 string
    :param DEBUG_MODE:            whether to run in debug mode
    :type DEBUG_MODE:             bool
    """
    # Construct the target directory and file path
    target_dir = _os.path.join(folder if folder else _os.getcwd(), data_name)
    file_name = f"{data_name}_{reconstruction_name}_{_age}Ma.shp"
    file_path = _os.path.join(target_dir, file_name)
    
    # Debug information
    if DEBUG_MODE:
        print(f"Target directory for {data_name}: {target_dir}")
        print(f"File path for {data_name} at {_age}: {file_path}")

    # Ensure the directory exists
    _os.makedirs(target_dir, exist_ok=True)
    
    # Delete old file if it exists
    try:
        _os.remove(file_path)
        if DEBUG_MODE:
            print(f"Deleted old file {file_path}")
    except FileNotFoundError:
        pass  # File does not exist, no need to remove

    # Save the data to a shapefile
    data.to_file(file_path)

def Dataset_to_netcdf(
        data: _xarray.Dataset,
        data_name: str,
        reconstruction_name: str,
        _age: int,
        folder: str,
        case: str = None,
        DEBUG_MODE: bool = False
    ):
    """
    Function to save Dataset to a NetCDF file in a folder efficiently.

    :param data:                  data
    :type data:                   xarray.Dataset
    :param data_name:             name of dataset
    :type data_name:              string
    :param reconstruction_name:   name of reconstruction
    :type reconstruction_name:    string
    :param _age:   age of reconstruction in Ma
    :type _age:    int
    :param folder:                folder
    :type folder:                 string
    :param DEBUG_MODE:            whether to run in debug mode
    :type DEBUG_MODE:             bool
    """
    # Construct the target directory and file path
    target_dir = _os.path.join(folder if folder else _os.getcwd(), data_name)
    if case:
        file_name = f"{data_name}_{reconstruction_name}_{case}_{_age}Ma.nc"
    else:
        file_name = f"{data_name}_{reconstruction_name}_{_age}Ma.nc"
    file_path = _os.path.join(target_dir, file_name)

    # Debug information
    if DEBUG_MODE:
        print(f"Target directory for {data_name}: {target_dir}")
        print(f"File path for {data_name} at {_age}: {file_path}")

    # Ensure the directory exists
    _os.makedirs(target_dir, exist_ok=True)

    # Delete old file if it exists
    try:
        _os.remove(file_path)
        if DEBUG_MODE:
            print(f"Deleted old file {file_path}")
    except FileNotFoundError:
        pass

    # Save the data to a NetCDF file
    data.to_netcdf(file_path)

def check_dir(target_dir):
    """
    Function to check if a directory exists, and create it if it doesn't
    """
    # Check if a directory exists, and create it if it doesn't
    if not _os.path.exists(target_dir):
        _os.makedirs(target_dir)

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# LOADING 
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

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
    for _age in tqdm(ages, desc=f"Loading {type} grids", disable=DEBUG_MODE):
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

def DataFrame_from_parquet(
        folder: str,
        type: str,
        reconstruction_name: str,
        case: str,
        _age: int
    ) -> _pandas.DataFrame:
    """
    Function to load DataFrames from a folder efficiently.

    :param folder:               folder
    :type folder:                str
    :param type:                 type of data
    :type type:                  str
    :param reconstruction_name:  name of reconstruction
    :type reconstruction_name:   str
    :param case:                 case
    :type case:                  str
    :param _age:  reconstruction time
    :type _age:   int
    
    :return:                     data
    :rtype:                      pandas.DataFrame or None
    """
    # Construct the target file path
    target_file = _os.path.join(
        folder if folder else _os.getcwd(),
        type,
        f"{type}_{reconstruction_name}_{case}_{_age}Ma.parquet"
    )

    # Check if target file exists and load data
    if _os.path.exists(target_file):
        return _pandas.read_parquet(target_file)
    else:
        return None

def DataFrame_from_csv(
        folder: str,
        type: str,
        reconstruction_name: str,
        case: str,
        _age: int
    ) -> _pandas.DataFrame:
    """
    Function to load DataFrames from a folder efficiently.

    :param folder:               folder
    :type folder:                str
    :param type:                 type of data
    :type type:                  str
    :param reconstruction_name:  name of reconstruction
    :type reconstruction_name:   str
    :param case:                 case
    :type case:                  str
    :param _age:  reconstruction time
    :type _age:   int
    
    :return:                     data
    :rtype:                      pandas.DataFrame or None
    """
    # Construct the target file path
    target_file = _os.path.join(
        folder if folder else _os.getcwd(),
        type,
        f"{type}_{reconstruction_name}_{case}_{_age}Ma.csv"
    )

    # Check if target file exists and load data
    if _os.path.exists(target_file):
        return _pandas.read_csv(target_file)
    else:
        return None

def Dataset_from_netCDF(
        folder: str,
        type: str,
        _age: int,
        reconstruction_name: str,
        case: Optional[str] = None
    ) -> _xarray.Dataset:
    """
    Function to load xarray Dataset from a folder efficiently.

    :param folder:               folder
    :type folder:                str
    :param _age:  reconstruction time
    :type _age:   int
    :param reconstruction_name:  name of reconstruction
    :type reconstruction_name:   str
    :param case:                 optional case
    :type case:                  str, optional

    :return:                     data
    :rtype:                      xarray.Dataset or None
    """
    # Construct the file name based on whether a case is provided
    file_name = f"{type}_{reconstruction_name}_{case + '_' if case else ''}{_age}Ma.nc"

    # Construct the full path to the target file
    target_file = _os.path.join(folder if folder else _os.getcwd(), type, file_name)

    # Check if the target file exists and load the dataset
    if _os.path.exists(target_file):
        return _xarray.open_dataset(target_file)
    else:
        return None
    
def GeoDataFrame_from_geoparquet(
        folder: str,
        type: str,
        _age: int,
        reconstruction_name: str
    ) -> _geopandas.GeoDataFrame:
    """
    Function to load GeoDataFrame from a folder efficiently.

    :param folder:               folder
    :type folder:                str
    :param _age:  reconstruction time
    :type _age:   int
    :param reconstruction_name:  name of reconstruction
    :type reconstruction_name:   str

    :return:                     data
    :rtype:                      geopandas.GeoDataFrame or None
    """
    # Construct the target file path
    target_file = _os.path.join(
        folder if folder else _os.getcwd(),
        type,
        f"{type}_{reconstruction_name}_{_age}Ma.parquet"
    )

    # Check if target file exists and load data
    if _os.path.exists(target_file):
        return _geopandas.read_parquet(target_file)
    else:
        return None
    
def GeoDataFrame_from_shapefile(
        folder: str,
        type: str,
        _age: int,
        reconstruction_name: str
    ) -> _geopandas.GeoDataFrame:
    """
    Function to load GeoDataFrame from a folder efficiently.

    :param folder:               folder
    :type folder:                str
    :param _age:  reconstruction time
    :type _age:   int
    :param reconstruction_name:  name of reconstruction
    :type reconstruction_name:   str

    :return:                     data
    :rtype:                      geopandas.GeoDataFrame or None
    """
    # Construct the target file path
    target_file = _os.path.join(
        folder if folder else _os.getcwd(),
        type,
        f"{type}_{reconstruction_name}_{_age}Ma.shp"
    )

    # Check if target file exists and load data
    if _os.path.exists(target_file):
        return _geopandas.read_file(target_file)
    else:
        return None