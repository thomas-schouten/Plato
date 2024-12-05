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
import inspect
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
from .utils_calc import set_constants, mag_azi2lat_lon, project_points, compute_velocity

def get_plate_data(
        rotations: _pygplates.RotationModel,
        age: int,
        resolved_topologies: list, 
        options: dict,
    ) -> _pandas.DataFrame:
    """
    Function to get data on plates in reconstruction.
    """
    # Set constants
    constants = set_constants()

    # Make _pandas.df with all plates
    # Initialise list
    plates = _numpy.zeros([len(resolved_topologies),7])

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

    # Convert to DataFrame    
    plates = _pandas.DataFrame(plates)

    # Initialise columns
    plates.columns = ["plateID", "area", "pole_lat", "pole_lon", "pole_angle", "centroid_lon", "centroid_lat"]

    # Merge topological networks with main plate
    # This is necessary because the topological networks have the same PlateID as their host plate and this leads to computational issues down the road
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

    # Initialise columns to store other whole-plate properties
    merged_plates["trench_length"] = 0.; merged_plates["zeta"] = 0.
    merged_plates["velocity_rms_mag"] = 0.; merged_plates["velocity_rms_azi"] = 0.; merged_plates["spin_rate_rms"] = 0.
    merged_plates["slab_flux"] = 0.; merged_plates["sediment_flux"] = 0.

    # Initialise columns to store whole-plate torques (Cartesian) and force at plate centroid (North-East).
    torques = ["slab_pull", "GPE", "slab_suction", "slab_bend", "mantle_drag", "driving", "residual"]
    axes = ["x", "y", "z", "mag"]
    coords = ["lat", "lon", "mag", "azi"]
    
    merged_plates[[torque + "_torque_" + axis for torque in torques for axis in axes]] = [[0.] * len(torques) * len(axes) for _ in range(len(merged_plates.plateID))]
    merged_plates[[torque + "_force_" + coord for torque in torques for coord in coords]] = [[0.] * len(torques) * len(coords) for _ in range(len(merged_plates.plateID))]
    
    return merged_plates

def get_slab_data(
        reconstruction: _gplately.PlateReconstruction,
        age: int,
        topology_geometries: _geopandas.GeoDataFrame,
        options: dict,
        PARALLEL_MODE: Optional[bool] = False,
    ) -> _pandas.DataFrame:
    """
    Function to get data on slabs in reconstruction.
    To this end, the subduction zones for the set age are split into points.
    """
    # Set constants
    constants = set_constants()

    # Discretise subduction zones into points
    slabs = reconstruction.tessellate_subduction_zones(
        age,
        ignore_warnings=True,
        tessellation_threshold_radians=(options["Slab tesselation spacing"]/constants.mean_Earth_radius_km)
    )

    # Convert to _pandas.DataFrame
    slabs = _pandas.DataFrame(slabs)

    # Kick unused columns and rename the rest
    slabs = slabs.drop(columns=[2, 3, 4, 5])
    slabs.columns = ["lon", "lat", "trench_segment_length", "trench_normal_azimuth", "lower_plateID", "trench_plateID"]

    # Convert trench segment length from degree to m
    slabs.trench_segment_length *= constants.equatorial_Earth_circumference / 360

    # Get slab sampling points
    slabs["slab_sampling_lat"], slabs["slab_sampling_lon"] = project_points(
        slabs.lat,
        slabs.lon,
        slabs.trench_normal_azimuth,
        -30,
    )

    # Get arc sampling points
    slabs["arc_sampling_lat"], slabs["arc_sampling_lon"] = project_points(
        slabs.lat,
        slabs.lon,
        slabs.trench_normal_azimuth,
        200,
    )

    # Get plateIDs for upper plates
    slabs["upper_plateID"] = get_plateIDs(
        reconstruction,
        topology_geometries,
        slabs["arc_sampling_lat"],
        slabs["arc_sampling_lon"],
        age,
        PARALLEL_MODE = PARALLEL_MODE,
    )

    # Initialise columns to store convergence rates
    types = ["upper_plate", "lower_plate", "trench", "convergence"]
    coords = ["lat", "lon", "mag"]
    slabs[[f"{type}_velocity_{coord}" for type in types for coord in coords]] = [[0.] * len(coords) * len(types) for _ in range(len(slabs))]

    # Initialise other columns to store seafloor ages and forces
    # Upper plate
    slabs["arc_thickness"] = 0.
    slabs["arc_seafloor_age"] = 0.
    slabs["continental_arc"] = False
    slabs["erosion_rate"] = 0.

    # Lower plate
    slabs["slab_seafloor_age"] = 0.
    slabs["slab_lithospheric_thickness"] = 0.
    slabs["slab_crustal_thickness"] = 0.
    slabs["slab_water_depth"] = 0.
    slabs["shear_zone_width"] = 0.
    slabs["sediment_thickness"] = 0.
    slabs["sediment_fraction"] = 0.
    slabs["slab_length"] = 0.

    # Forces
    forces = ["slab_pull", "slab_bend", "slab_suction", "residual"]
    coords = ["mag", "lat", "lon"]
    slabs[[force + "_force_" + coord for force in forces for coord in coords]] = [[0.] * len(coords) * len(forces) for _ in range(len(slabs))]
    slabs["slab_pull_constant"] = options["Slab pull constant"]
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
    ) -> _pandas.DataFrame:
    """
    Function to get data on regularly spaced grid points in reconstruction.
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
    segment_area = segment_length_lat * segment_length_lon

    # Organise as DataFrame
    points = _pandas.DataFrame({"lat": lat_grid, 
                           "lon": lon_grid, 
                           "plateID": plateIDs, 
                           "segment_length_lat": segment_length_lat,
                           "segment_length_lon": segment_length_lon,
                            "segment_area": segment_area,
                           },
                           dtype=float
                        )

    # Add additional columns to store velocities
    components = ["velocity_lat", "velocity_lon", "velocity_mag", "velocity_azi", "spin_rate_mag"]
    points[[component for component in components]] = [[0.] * len(components) for _ in range(len(points))]

    # Add additional columns to store seafloor properties
    points["seafloor_age"] = 0.
    points["lithospheric_mantle_thickness"] = 0.
    points["crustal_thickness"] = 0.
    points["water_depth"] = 0.
    points["U"] = 0.

    # Add additional columns to store forces
    forces = ["GPE", "mantle_drag", "residual"]
    coords = ["lat", "lon", "mag", "azi"]
    points[[force + "_force_" + coord for force in forces for coord in coords]] = [[0.] * len(forces) * len(coords) for _ in range(len(points))]
    points["mantle_viscosity"] = options["Mantle viscosity"]
    
    return points

def get_resolved_topologies(
        reconstruction: _gplately.PlateReconstruction,
        age: Union[int, float, _numpy.floating, _numpy.integer],
        anchor_plateID: Optional[Union[int, float, _numpy.integer, _numpy.floating]] = 0,
        filename: Optional[str] = None,
    ) -> Dict:
    """
    Function to get resolved geometries for all ages.
    """
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
            age,
            anchor_plate_id=int(anchor_plateID)
        )
    
    return resolved_topologies
    
def get_resolved_geometries(
        reconstruction: _gplately.PlateReconstruction,
        age: Union[int, float, _numpy.floating, _numpy.integer],
        anchor_plateID: Optional[Union[int, float, _numpy.integer, _numpy.floating]]
    ) -> Dict:
    """
    Function to obtain resolved geometries as GeoDataFrames for all ages.
    """
    # Make temporary directory to hold shapefiles
    temp_dir = tempfile.mkdtemp()

    # Initialise dictionary to store resolved geometries
    resolved_geometries = {}

    topology_file = _os.path.join(temp_dir, f"topologies_{age}.shp")
    get_resolved_topologies(reconstruction, age, anchor_plateID, topology_file)

    # Load resolved topologies as GeoDataFrames
    resolved_geometries = _geopandas.read_file(topology_file)
    
    # Remove temporary directory
    shutil.rmtree(temp_dir)

    return resolved_geometries

def extract_geometry_data(
        topology_geometries: _geopandas.GeoDataFrame
    ) -> list:
    """
    Function to extract only the geometry and plateID from topology geometries.
    """
    return [(geom, plateID) for geom, plateID in zip(topology_geometries.geometry, topology_geometries.PLATEID1)]

def process_plateIDs(
        geometries_data: list,
        lats_chunk: _numpy.ndarray,
        lons_chunk: _numpy.ndarray,
    ) -> list:
    """
    Function to process plateIDs for a chunk of latitudes and longitudes.
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
        lats: Union[List, _numpy.ndarray],
        lons: Union[List, _numpy.ndarray],
        age: int,
        PARALLEL_MODE: Optional[bool] = False,
    ) -> _numpy.ndarray:
    """
    Function to get plate IDs for a set of latitudes and longitudes.
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

# def get_velocities(
#         lats: Union[List, _numpy.array],
#         lons: Union[List, _numpy.array],
#         stage_rotation: tuple,
#     ) -> Tuple[_numpy.array, _numpy.array, _numpy.array, _numpy.array]:
#     """
#     Function to get velocities for a set of latitudes and longitudes.
#     NOTE: This function is not vectorised yet, but has not been a bottleneck in the code so far.
#     """
#     # Convert lats and lons to numpy arrays if they are not already
#     lats = _numpy.asarray(lats)
#     lons = _numpy.asarray(lons)

#     # Initialise empty array to store velocities
#     velocities_lat = _numpy.zeros(len(lats))
#     velocities_lon = _numpy.zeros(len(lats))
#     velocities_mag = _numpy.zeros(len(lats))
#     velocities_azi = _numpy.zeros(len(lats))

#     # Loop through points to get velocities
#     for i, _ in enumerate(lats):
#         # Convert to LocalCartesian
#         point = _pygplates.PointOnSphere((lats[i], lons[i]))

#         # Calculate magnitude and azimuth of velocities at points
#         velocity_mag_azi = _numpy.asarray(
#             _pygplates.LocalCartesian.convert_from_geocentric_to_magnitude_azimuth_inclination(
#                 point,
#                 _pygplates.calculate_velocities(
#                     point, 
#                     _pygplates.FiniteRotation((stage_rotation[0], stage_rotation[1]), _numpy.deg2rad(stage_rotation[2])), 
#                     1.,
#                     velocity_units = _pygplates.VelocityUnits.cms_per_yr
#                 )
#             )
#         )

#         # Get magnitude and azimuth of velocities
#         velocities_mag[i] = velocity_mag_azi[0][0]; velocities_azi[i] = velocity_mag_azi[0][1]

#     # Convert to lat and lon components
#     velocities_lat, velocities_lon = mag_azi2lat_lon(velocities_mag, _numpy.rad2deg(velocities_azi))

#     return velocities_lat, velocities_lon, velocities_mag, velocities_azi

def get_topology_geometries(
        reconstruction: _gplately.PlateReconstruction,
        age: int,
        anchor_plateID: int
    ) -> _geopandas.GeoDataFrame:
    """
    Function to resolve topologies and get geometries as a GeoDataFrame
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
    ) -> _pandas.DataFrame:
    """
    Function to get geometric properties of plates.
    """
    # Calculate trench length and omega
    for plateID in plates.plateID:
        if plateID in slabs.lower_plateID.unique():
            plates.loc[plates.plateID == plateID, "trench_length"] = slabs[slabs.lower_plateID == plateID].trench_segment_length.sum()
            plates.loc[plates.plateID == plateID, "zeta"] = plates[plates.plateID == plateID].area.values[0] / plates[plates.plateID == plateID].trench_length.values[0]

    return plates

def get_plate_names(
        plate_id_list: Union[list, _numpy.ndarray],
    ) -> list[str]:
    """
    Function to get plate names corresponding to plate ids
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
    """
    # Define all options
    all_options = [
        "Slab pull torque",
        "GPE torque",
        "Mantle drag torque",
        "Slab bend torque",
        "Slab suction torque",
        "Slab bend mechanism",
        "Reconstructed motions",
        "Continental crust",
        "Continental keels",
        "Lateral viscosity variation",
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
        "Slab suction constant",
        "Mantle viscosity",
        "Slab tesselation spacing",
        "Grid spacing",
        "Minimum plate area",
        "Anchor plateID",
        "Velocity time step",
        "Grid resolution",
    ]
    
    # Define default values
    default_values = [
        True,
        True,
        True,
        False,
        False,
        "viscous",
        True,
        False,
        False,
        1,
        "half space cooling",
        False,
        0,
        False,
        2,
        False,
        2e3,
        700e3,
        1e-12,
        0.2225,
        1/3,
        1.44e20,
        250,
        1,
        0,
        0,
        1,
        0.1,
    ]

    # Adjust TRUE/FALSE values in excel file to boolean
    boolean_options = [
        "Slab pull torque",
        "GPE torque",
        "Mantle drag torque",
        "Slab bend torque",
        "Reconstructed motions",
        "Continental crust",
        "Continental keels",
        "Randomise trench orientation",
        "Randomise slab age",
    ]

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

def get_seafloor_age_grid(
        reconstruction_name: str,
        age: int,
        DEBUG_MODE: bool = False
    ) -> _xarray.Dataset:
    """
    Function to obtain seafloor grid from GPlately DataServer.
    """
    # Call _gplately"s DataServer from the download.py module
    gdownload = _gplately.download.DataServer(reconstruction_name)

    # Inform the user of the ongoing process if in debug mode
    logging.info(f"Downloading age grid for {reconstruction_name} at {age} Ma")

    # Download the age grid, suppressing stdout output if not in debug mode
    if DEBUG_MODE:
        age_raster = gdownload.get_age_grid(time=age)
    else:
        with contextlib.redirect_stdout(io.StringIO()):
            age_raster = gdownload.get_age_grid(time=age)

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
    """
    # Make xarray velocity grid
    velocity_grid = _xarray.Dataset(
            {
                "velocity_magnitude": (["latitude", "longitude"], points.velocity_mag.values.reshape(points.lat.unique().size, points.lon.unique().size)),
                "velocity_latitude": (["latitude", "longitude"], points.velocity_lat.values.reshape(points.lat.unique().size, points.lon.unique().size)),
                "velocity_longitude": (["latitude", "longitude"], points.velocity_lon.values.reshape(points.lat.unique().size, points.lon.unique().size)),
                "spin_rate_magnitude": (["latitude", "longitude"], points.spin_rate_mag.values.reshape(points.lat.unique().size, points.lon.unique().size)),
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

def select_ages(
        ages: Union[None, int, float, list, _numpy.integer, _numpy.floating, _numpy.ndarray],
        default_ages: _numpy.ndarray,
    ) -> _numpy.ndarray:
    """
    Function to check and get ages.
    """
    # Define ages
    if ages is None:
        # If no ages are provided, use default ages
        return default_ages
    
    elif isinstance(ages, _numpy.ndarray):
        # If a numpy array is provided, use as is
        return ages
    
    elif isinstance(ages, list):
        # If a list is provided, convert to numpy array
        return _numpy.array(ages)

    elif isinstance(ages, (int, float, _numpy.integer, _numpy.floating)):
        # If a single value is provided, convert to numpy array
        return _numpy.array([ages])

def select_cases(
    cases: Union[None, str, List[str]],
    default_cases: List[str],
    ) -> List[str]:
    """
    Function to check and get cases.
    """
    # Define cases
    if cases is None:
        # If no cases are provided, use default cases
        return default_cases

    elif isinstance(cases, List):
        # If cases is a list, return it
        return cases

    elif isinstance(cases, str):
        # If cases is a string, put it in a list
       return [cases]

def select_iterable(
        cases: Union[None, str, List[str]],
        default_iterable: List[str],
    ) -> Dict[str, List[str]]:
    """
    Function to check and get iterable.
    """
    # Define iterable
    if cases is None:
        # If no iterable is provided, use the default iterable
        if isinstance(default_iterable, list):
            # If the default iterable is a list, return a dictionary with the default iterable
            return {case: [case] for case in default_iterable}
        else:
            return default_iterable

    elif isinstance(cases, str):
        # If cases is a single value, put it in a list
        cases = [cases]

    # Make dictionary of iterable
    _iterable = {case: [] for case in cases}

    return _iterable

def select_plateIDs(
        plateIDs: Union[None, int, list, _numpy.integer, _numpy.floating, _numpy.ndarray],
        default_plates: List,
    ) -> Union[List, _numpy.ndarray]:
    """
    Function to check and get plate IDs.
    """
    # Define iterable
    if plateIDs is None:
        # If no plateIDs are provided, use the default iterable
        return default_plates
    
    elif isinstance(plateIDs, (List, _numpy.ndarray)):
        # If plateIDs is already a list or a numpy array, simply return it
        return plateIDs

    elif isinstance(plateIDs, (int, float, _numpy.floating, _numpy.integer)):
        # If plateIDs is a single value, put it in a list
        return [plateIDs]
    
def copy_values(
        data: Dict[str, _pandas.DataFrame],
        key: str,
        entries: List[str],
        cols: Optional[Union[None, str, List[str]]] = None,
        check: bool = False
    ) -> Dict[str, _pandas.DataFrame]:
    """
    Function to copy values from one dataframe in a dictionary to another.
    """
    # Ensure cols is a list for easier iteration
    if isinstance(cols, str):
        cols = [cols]

    for entry in entries[1:]:
        # Loop through columns
        for col in cols:
            # Copy column data as a deep copy
            data[entry].loc[:, col] = data[key][col].copy(deep=True)

    return data

def get_variables(
        variables: Union[None, str, List[str]],
        default_variables: List,
    ) -> List[str]:
    """
    Function to get variables
    """
    if variables is None:
        return default_variables
    
    elif isinstance(variables, List(str)):
        return variables
    
    elif isinstance(variables, str):
        return [variables]

def process_cases(
        cases: List[str],
        options: Dict[str, Dict[str, Optional[str]]],
        target_options: List[str],
    ) -> Dict[str, List[str]]:
    """
    Function to process cases and options to accelerate computation. Each case is assigned a dictionary of identical cases for a given set of target options.
    The goal here is that if these target options are identical, the computation is only peformed once and the results are copied to the other cases.
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

def DataFrame_to_parquet(
        data: _pandas.DataFrame,
        data_name: str,
        reconstruction_name: str,
        age: int,
        case: str,
        folder: str,
    ) -> None:
    """
    Function to save DataFrame to a Parquet file in a folder efficiently.
    """
    # Get the file path
    file_path = get_file_path(data_name, "parquet", reconstruction_name, age, case, folder)

    # Delete the old file if it exists
    try:
        _os.remove(file_path)
        logging.info(f"Deleted old file {file_path}")
        
    except FileNotFoundError:
        pass

    # Save the data to Parquet
    data.to_parquet(file_path, index=False)

def DataFrame_to_csv(
        data: _pandas.DataFrame,
        data_name: str,
        reconstruction_name: str,
        age: int,
        case: str,
        folder: str,
    ) -> None:
    """
    Function to save DataFrame to a folder efficiently.
    """
    # Get the file path
    file_path = get_file_path(data_name, "csv", reconstruction_name, age, case, folder)

    # Delete the old file if it exists
    try:
        _os.remove(file_path)
        logging.info(f"Deleted old file {file_path}")
        
    except FileNotFoundError:
        pass

    # Save the data to CSV
    data.to_csv(file_path, index=False)
    
def GeoDataFrame_to_geoparquet(
        data: _geopandas.GeoDataFrame,
        data_name: str,
        reconstruction_name: str,
        age: int,
        case: str,
        folder: str,
    ) -> None:
    """
    Function to save GeoDataFrame to a GeoParquet file in a folder efficiently.
    """
    # Get the file path
    file_path = get_file_path(data_name, "parquet", reconstruction_name, age, case, folder)

    # Delete the old file if it exists
    try:
        _os.remove(file_path)
        logging.info(f"Deleted old file {file_path}")
        
    except FileNotFoundError:
        pass

    # Save the data to a GeoParquet file
    data.to_parquet(file_path)

def GeoDataFrame_to_shapefile(
        data: _geopandas.GeoDataFrame,
        data_name: str,
        reconstruction_name: str,
        age: int,
        case: str,
        folder: str,
    ) -> None:
    """
    Function to save GeoDataFrame to a folder efficiently.
    """
    # Get the file path
    file_path = get_file_path(data_name, "shp", reconstruction_name, age, case, folder)

    # Delete the old file if it exists
    try:
        _os.remove(file_path)
        logging.info(f"Deleted old file {file_path}")
        
    except FileNotFoundError:
        pass

    # Save the data to a shapefile
    data.to_file(file_path)

def Dataset_to_netcdf(
        data: _xarray.Dataset,
        data_name: str,
        reconstruction_name: str,
        age: int,
        case: Optional[str] = None,
        folder: Optional[str] = None,
    ) -> None:
    """
    Function to save Dataset to a NetCDF file in a folder efficiently.
    """
    # Get the file path
    file_path = get_file_path(data_name, "nc", reconstruction_name, age, case, folder)

    # Delete the old file if it exists
    try:
        _os.remove(file_path)
        logging.info(f"Deleted old file {file_path}")
        
    except FileNotFoundError:
        pass

    # Save the data to a NetCDF file
    data.to_netcdf(file_path)

def DataFrame_from_parquet(
        folder: str,
        data_name: str,
        reconstruction_name: str,
        age: int,
        case: str,
    ) -> _pandas.DataFrame:
    """
    Function to load DataFrames from a folder efficiently.
    """
    # Construct the target file path
    file_path = get_file_path(data_name, "parquet", reconstruction_name, age, case, folder)

    # Check if target file exists and if so, load data
    if _os.path.exists(file_path):
        return _pandas.read_parquet(file_path)
    else:
        return None
    
def DataFrame_from_csv(
        folder: str,
        data_name: str,
        reconstruction_name: str,
        age: int,
        case: str,
    ) -> _pandas.DataFrame:
    """
    Function to load DataFrames from a folder efficiently.
    """
    # Construct the target file path
    file_path = get_file_path(data_name, "csv", reconstruction_name, age, case, folder)

    # Check if target file exists and if so, load data
    if _os.path.exists(file_path):
        return _pandas.read_csv(file_path)
    else:
        return None
    
def GeoDataFrame_from_geoparquet(
        folder: str,
        type: str,
        reconstruction_name: str,
        age: int,
        case: str,
    ) -> _geopandas.GeoDataFrame:
    """
    Function to load GeoDataFrame from a folder efficiently.
    """
    # Construct the target file path
    file_path = get_file_path(type, "parquet", reconstruction_name, age, case, folder)

    # Check if target file exists and load data
    if _os.path.exists(file_path):
        return _geopandas.read_parquet(file_path)
    else:
        return None
    
def GeoDataFrame_from_shapefile(
        folder: str,
        type: str,
        reconstruction_name: str,
        age: int,
        case: str,
    ) -> _geopandas.GeoDataFrame:
    """
    Function to load GeoDataFrame from a folder efficiently.
    """
    # Construct the target file path
    file_path = get_file_path(type, "shp", reconstruction_name, age, case, folder)

    # Check if target file exists and load data
    if _os.path.exists(file_path):
        return _geopandas.read_file(file_path)
    else:
        return None
    
def Dataset_from_netCDF(
        folder: str,
        data_name: str,
        reconstruction_name: str,
        age: int,
        case: str,
    ) -> _xarray.Dataset:
    """
    Function to load xarray Dataset from a folder efficiently.
    """
    # Construct the file name based on whether a case is provided
    file_path = get_file_path(data_name, "nc", reconstruction_name, age, case, folder)

    # Check if the target file exists and load the dataset
    if _os.path.exists(file_path):
        return _xarray.open_dataset(file_path)
    else:
        return None
    
def check_dir(
        target_dir: str,
    ) -> None:
    """
    Function to check if a directory exists, and create it if it doesn't
    """
    # Check if a directory exists, and create it if it doesn't
    if not _os.path.exists(target_dir):
        _os.makedirs(target_dir)

def get_file_path(
        data_name: str,
        file_extension: str,
        reconstruction_name: str,
        age: int,
        case: Optional[str] = None,
        folder: Optional[str] = None,
    ) -> str:
    """
    Function to save a file
    """
    # Construct the file path
    target_dir = folder if folder else _os.getcwd()
    if age is not None and case is not None:
        file_name = f"{data_name}_{reconstruction_name}_{case}_{age}Ma.{file_extension}"
    elif age is None and case is not None:
        file_name = f"{data_name}_{reconstruction_name}_{case}.{file_extension}"
    elif age is not None and case is None:
        file_name = f"{data_name}_{reconstruction_name}_{age}Ma.{file_extension}"
    elif age is None and case is None:
        file_name = f"{data_name}_{reconstruction_name}.{file_extension}"
    file_path = _os.path.join(target_dir, data_name, file_name)

    # Ensure the directory exists
    _os.makedirs(_os.path.join(target_dir, data_name), exist_ok=True)
    
    return file_path
    
def get_variable_name(
        var: str,
    ) -> str:
    """
    Function to get the name of a python variable
    """
    # Get the frame of the caller function
    callers_local_vars = inspect.currentframe().f_back.f_locals.items()

    # Find the variable name that matches the passed object
    return [var_name for var_name, var_val in callers_local_vars if var_val is var][0]

def rename_coordinates_and_variables(
        grid: _xarray.Dataset,
        var_old_name: Optional[str] = "z",
        var_new_name: Optional["str"] = "z",
    ):
    """
    Function to rename coordinates and variables 
    """
    if var_old_name in grid.data_vars:
        grid = grid.rename({"z": var_new_name})

    if "lat" in grid.coords:
        grid = grid.rename({"lat": "latitude"})

    if "lon" in grid.coords:
        grid = grid.rename({"lon": "longitude"})

    return grid

def array2data_array(
        lats: Union[list, _numpy.ndarray],
        lons: Union[list, _numpy.ndarray],
        data: Union[list, _numpy.ndarray],
        var_name: str,
    ) -> _xarray.DataArray:
    """
    Interpolates data to the resolution seafloor grid.
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
        data_arrays: dict,
        grid_name: str,
    ) -> _xarray.Dataset:
    """
    Creates a grid object from a dictionary of data arrays.
    """
    # Create the grid
    grid = _xarray.Dataset(
        data_vars = data_arrays
    )

    # Dynamically assign the grid to an attribute using the var_name
    setattr(grid_name, grid)