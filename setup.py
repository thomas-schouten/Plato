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
import os
import tempfile
import shutil
import warnings

from collections import defaultdict
from typing import Optional
from typing import Union

# Third-party libraries
import geopandas as _geopandas
import matplotlib.pyplot as plt
import numpy as _numpy
import gplately as _gplately
import pandas as _pandas
import pygplates as _pygplates
import xarray as _xarray

from shapely.geometry import Point
from tqdm import tqdm

# Local libraries
from functions_main import set_constants
from functions_main import mag_azi2lat_lon
from functions_main import project_points

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# INITIALISATION 
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def get_plates(
        rotations: _pygplates.RotationModel,
        reconstruction_time: int,
        resolved_topologies: list, 
        options: dict,
    ):
    """
    Function to get data on plates in reconstruction.

    :param rotations:             rotation model
    :type rotations:              _pygplates.RotationModel object
    :param reconstruction_time:   reconstruction time
    :type reconstruction_time:    integer
    :param resolved_topologies:   resolved topologies
    :type resolved_topologies:    list of resolved topologies
    :param options:               options for the case
    :type options:                dict

    :return:                      plates
    :rtype:                       pandas.DataFrame
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
            to_time=reconstruction_time,
            moving_plate_id=int(plates[n,0]),
            from_time=reconstruction_time + options["Velocity time step"],
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

    # Initialise columns to store whole-plate torques (Cartesian) and force at plate centroid (North-East).
    torques = ["slab_pull", "GPE", "slab_bend", "mantle_drag", "driving", "residual"]
    axes = ["x", "y", "z", "mag"]
    coords = ["lat", "lon", "mag"]
    
    merged_plates[[torque + "_torque_" + axis for torque in torques for axis in axes]] = [[0] * len(torques) * len(axes) for _ in range(len(merged_plates.plateID))]
    merged_plates[["slab_pull_torque_opt_" + axis for axis in axes]] = [[0] * len(axes) for _ in range(len(merged_plates.plateID))]
    merged_plates[["mantle_drag_torque_opt_" + axis for axis in axes]] = [[0] * len(axes) for _ in range(len(merged_plates.plateID))]
    merged_plates[["driving_torque_opt_" + axis for axis in axes]] = [[0] * len(axes) for _ in range(len(merged_plates.plateID))]
    merged_plates[["residual_torque_opt_" + axis for axis in axes]] = [[0] * len(axes) for _ in range(len(merged_plates.plateID))]
    merged_plates[[torque + "_force_" + coord for torque in torques for coord in coords]] = [[0] * len(torques) * len(coords) for _ in range(len(merged_plates.plateID))]
    merged_plates[["slab_pull_force_opt_" + coord for coord in coords]] = [[0] * len(coords) for _ in range(len(merged_plates.plateID))]
    merged_plates[["mantle_drag_force_opt_" + coord for coord in coords]] = [[0] * len(coords) for _ in range(len(merged_plates.plateID))]
    merged_plates[["driving_force_opt_" + coord for coord in coords]] = [[0] * len(coords) for _ in range(len(merged_plates.plateID))]
    merged_plates[["residual_force_opt_" + coord for coord in coords]] = [[0] * len(coords) for _ in range(len(merged_plates.plateID))]

    return merged_plates

def get_slabs(
        reconstruction: _gplately.PlateReconstruction,
        reconstruction_time: int,
        plates: _pandas.DataFrame,
        topology_geometries: _geopandas.GeoDataFrame,
        options: dict,
    ):
    """
    Function to get data on slabs in reconstruction.

    :param reconstruction:        reconstruction
    :type reconstruction:         _gplately.PlateReconstruction
    :param reconstruction_time:   reconstruction time
    :type reconstruction_time:    integer
    :param plates:                plates
    :type plates:                 pandas.DataFrame
    :param topology_geometries:   topology geometries
    :type topology_geometries:    geopandas.GeoDataFrame
    :param options:               options for the case
    :type options:                dict
    
    :return:                      slabs
    :rtype:                       pandas.DataFrame
    """
    # Set constants
    constants = set_constants()

    # Tesselate subduction zones and get slab pull and bend torques along subduction zones
    slabs = reconstruction.tessellate_subduction_zones(reconstruction_time, ignore_warnings=True, tessellation_threshold_radians=(options["Slab tesselation spacing"]/constants.mean_Earth_radius_km))

    # Convert to _pandas.DataFrame
    slabs = _pandas.DataFrame(slabs)

    # Kick unused columns
    slabs = slabs.drop(columns=[2, 3, 4, 5])

    slabs.columns = ["lon", "lat", "trench_segment_length", "trench_normal_azimuth", "lower_plateID", "trench_plateID"]

    # Convert trench segment length from degree to m
    slabs.trench_segment_length *= constants.equatorial_Earth_circumference / 360

    # Get plateIDs of overriding plates
    sampling_lat, sampling_lon = project_points(slabs.lat, slabs.lon, slabs.trench_normal_azimuth, 100)
    slabs["upper_plateID"] = get_plateIDs(reconstruction, topology_geometries, sampling_lat, sampling_lon, reconstruction_time)

    # Get absolute velocities of upper and lower plates
    for plate in ["upper_plate", "lower_plate", "trench_plate"]:
        # Loop through lower plateIDs to get absolute lower plate velocities
        for plateID in slabs[plate + "ID"].unique():
            # Select all points with the same plateID
            selected_slabs = slabs[slabs[plate + "ID"] == plateID]

            # Get stage rotation for plateID
            selected_plate = plates[plates.plateID == plateID]

            if len(selected_plate) == 0:
                stage_rotation = reconstruction.rotation_model.get_rotation(
                    to_time=reconstruction_time,
                    moving_plate_id=int(plateID),
                    from_time=reconstruction_time + options["Velocity time step"],
                    anchor_plate_id=options["Anchor plateID"]
                ).get_lat_lon_euler_pole_and_angle_degrees()
            else:
                stage_rotation = (selected_plate.pole_lat.values[0], selected_plate.pole_lon.values[0], selected_plate.pole_angle.values[0])

            # Get plate velocities
            selected_velocities = get_velocities(selected_slabs.lat, selected_slabs.lon, stage_rotation)

            # Store in array
            slabs.loc[slabs[plate + "ID"] == plateID, "v_" + plate + "_lat"] = selected_velocities[0]
            slabs.loc[slabs[plate + "ID"] == plateID, "v_" + plate + "_lon"] = selected_velocities[1]
            slabs.loc[slabs[plate + "ID"] == plateID, "v_" + plate + "_mag"] = selected_velocities[2]
            slabs.loc[slabs[plate + "ID"] == plateID, "v_" + plate + "_azi"] = selected_velocities[3]

    # Calculate convergence rates
    slabs["v_convergence_lat"] = slabs.v_lower_plate_lat - slabs.v_trench_plate_lat
    slabs["v_convergence_lon"] = slabs.v_lower_plate_lon - slabs.v_trench_plate_lon
    slabs["v_convergence_mag"] = _numpy.sqrt(slabs.v_convergence_lat**2 + slabs.v_convergence_lon**2)

    # Initialise other columns to store seafloor ages and forces
    # Upper plate
    slabs["upper_plate_thickness"] = 0
    slabs["upper_plate_age"] = 0   
    slabs["continental_arc"] = False
    slabs["erosion_rate"] = 0

    # Lower plate
    slabs["lower_plate_age"] = 0
    slabs["lower_plate_thickness"] = 0
    slabs["sediment_thickness"] = 0
    slabs["sediment_fraction"] = 0.
    slabs["slab_length"] = options["Slab length"]

    # Forces
    forces = ["slab_pull", "slab_bend"]
    coords = ["mag", "lat", "lon"]
    slabs[[force + "_force_" + coord for force in forces for coord in coords]] = [[0] * 6 for _ in range(len(slabs))] 

    # Make sure all the columns are floats
    slabs = slabs.astype(float)

    return slabs

def get_points(
        reconstruction: _gplately.PlateReconstruction,
        reconstruction_time: int,
        plates: _pandas.DataFrame,
        topology_geometries: _geopandas.GeoDataFrame,
        options: dict,
    ):
    """
    Function to get data on regularly spaced grid points in reconstruction.

    :param reconstruction:        reconstruction
    :type reconstruction:         _gplately.PlateReconstruction
    :param reconstruction_time:   reconstruction time
    :type reconstruction_time:    integer
    :param plates:                plates
    :type plates:                 pandas.DataFrame
    :param topology_geometries:   topology geometries
    :type topology_geometries:    geopandas.GeoDataFrame
    :param options:               options for the case
    :type options:                dict

    :return:                      points
    :rtype:                       pandas.DataFrame    
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
    plateIDs = get_plateIDs(reconstruction, topology_geometries, lat_grid, lon_grid, reconstruction_time)

    # Initialise empty array to store velocities
    velocity_lat, velocity_lon = _numpy.zeros_like(lat_grid), _numpy.zeros_like(lat_grid)
    velocity_mag, velocity_azi = _numpy.zeros_like(lat_grid), _numpy.zeros_like(lat_grid)

    # Loop through plateIDs to get velocities
    for plateID in _numpy.unique(plateIDs):
        # Your code here
        # Select all points with the same plateID
        selected_lon, selected_lat = lon_grid[plateIDs == plateID], lat_grid[plateIDs == plateID]

        # Get stage rotation for plateID
        selected_plate = plates[plates.plateID == plateID]

        if len(selected_plate) == 0:
            stage_rotation = reconstruction.rotation_model.get_rotation(
                to_time=reconstruction_time,
                moving_plate_id=int(plateID),
                from_time=reconstruction_time + options["Velocity time step"],
                anchor_plate_id=options["Anchor plateID"]
            ).get_lat_lon_euler_pole_and_angle_degrees()
        else:
            stage_rotation = (selected_plate.pole_lat.values[0], selected_plate.pole_lon.values[0], selected_plate.pole_angle.values[0])

        # Get plate velocities
        selected_velocities = get_velocities(selected_lat, selected_lon, stage_rotation)

        # Store in array
        velocity_lat[plateIDs == plateID] = selected_velocities[0]
        velocity_lon[plateIDs == plateID] = selected_velocities[1]
        velocity_mag[plateIDs == plateID] = selected_velocities[2]
        velocity_azi[plateIDs == plateID] = selected_velocities[3]

    # Convert degree spacing to metre spacing
    segment_length_lat = constants.mean_Earth_radius_m * (_numpy.pi/180) * options["Grid spacing"]
    segment_length_lon = constants.mean_Earth_radius_m * (_numpy.pi/180) * _numpy.cos(_numpy.deg2rad(lat_grid)) * options["Grid spacing"]

    # Organise as DataFrame
    points = _pandas.DataFrame({"lat": lat_grid, 
                           "lon": lon_grid, 
                           "plateID": plateIDs, 
                           "segment_length_lat": segment_length_lat,
                           "segment_length_lon": segment_length_lon,
                           "v_lat": velocity_lat, 
                           "v_lon": velocity_lon,
                           "v_mag": velocity_mag,
                           "v_azi": velocity_azi,},
                           dtype=float
                        )

    # Add additional columns to store seafloor ages and forces
    points["seafloor_age"] = 0
    points["lithospheric_thickness"] = 0
    points["crustal_thickness"] = 0
    points["water_depth"] = 0
    points["U"] = 0
    forces = ["GPE", "mantle_drag"]
    coords = ["lat", "lon", "mag"]

    points[[force + "_force_" + coord for force in forces for coord in coords]] = [[0] * len(forces) * len(coords) for _ in range(len(points))]
    
    return points

def get_plateIDs(
        reconstruction: _gplately.PlateReconstruction,
        topology_geometries: _geopandas.GeoDataFrame,
        lats: Union[list or _numpy.array],
        lons: Union[list or _numpy.array],
        reconstruction_time: int,
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
    :param reconstruction_time:        reconstruction time
    :type reconstruction_time:         integer

    :return:                           plateIDs
    :rtype:                            list
    """
    # Convert lats and lons to numpy arrays if they are not already
    lats = _numpy.array(lats)
    lons = _numpy.array(lons)

    # Create a GeoDataFrame with grid
    grid = _geopandas.GeoDataFrame({"geometry": [Point(lon, lat) for lon, lat in zip(lons, lats)]})

    # Initialise empty array to store plateIDs
    plateIDs = _numpy.zeros(len(lons))

    # Loop through points to get plateIDs
    for topology_geometry, topology_plateID in zip(topology_geometries.geometry, topology_geometries.PLATEID1):
        # Try to get plateIDs for points within topology geometries
        try:
            inside_points = grid[grid.geometry.within(topology_geometry)]
            plateIDs[inside_points.index] = topology_plateID
            
        # If there are no points within the topology geometry or this throws an error, pass
        except:
            pass

    # Get plateIDs for points for which no plateID was found
    no_plateID = _numpy.where(plateIDs == 0)
    
    if len(no_plateID[0]) != 0:
        no_plateID_lat = lats[no_plateID]
        no_plateID_lon = lons[no_plateID]

        # Use _pygplates to fill in remaining plate IDs
        no_plateID_grid = _gplately.Points(reconstruction, no_plateID_lon, no_plateID_lat, time=reconstruction_time)

        # Insert plate IDs into array
        plateIDs[no_plateID] = no_plateID_grid.plate_id
    
    return plateIDs

def get_velocities(
        lats: Union[list or _numpy.array],
        lons: Union[list or _numpy.array],
        stage_rotation: tuple,
    ):
    """
    Function to get velocities for a set of latitudes and longitudes.

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
    lats = _numpy.array(lats)
    lons = _numpy.array(lons)

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
        reconstruction_time: int,
        anchor_plateID: int
    ):
    """
    Function to resolve topologies and get geometries as a GeoDataFrame

    :param reconstruction:        reconstruction
    :type reconstruction:         gplately.PlateReconstruction
    :param reconstruction_time:   reconstruction time
    :type reconstruction_time:    integer
    :param anchor_plateID:        anchor plate ID
    :type anchor_plateID:         integer
    :return:                      resolved_topologies
    :rtype:                       geopandas.GeoDataFrame
    """
    # Make temporary directory to hold shapefiles
    temp_dir = tempfile.mkdtemp()

    # Resolve topological networks and load as GeoDataFrame
    topology_file = os.path.join(temp_dir, "topologies.shp")
    with warnings.catch_warnings():
        warnings.filterwarnings(
            action="ignore",
            message="Normalized/laundered field name:"
        )
        _pygplates.resolve_topologies(
            reconstruction.topology_features, 
            reconstruction.rotation_model, 
            topology_file, 
            reconstruction_time, 
            anchor_plate_id=anchor_plateID
        )
        if os.path.exists(topology_file):
            topology_geometries = _geopandas.read_file(topology_file)

    # Remove temporary directory
    shutil.rmtree(temp_dir)

    return topology_geometries
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
        file_name: str,
        sheet_name: Optional[str]
    ):
    """
    Function to get options from excel file

    :param file_name:            file name
    :type file_name:             string
    :param sheet_name:           sheet name
    :type sheet_name:            string

    :return:                     cases, options
    :rtype:                      list, dict
    """
    # Read file
    case_options = _pandas.read_excel(file_name, sheet_name=sheet_name, comment="#")

    # Initialise list of cases
    cases = []

    # Initialise options dictionary
    options = {}

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
                      0.0301,
                      8.97e18,
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

    # Loop over rows to obtain options from excel file
    for _, row in case_options.iterrows():
        case = row["Name"]
        cases.append(case)
        options[case] = {}
        for i, option in enumerate(all_options):
            if option in case_options:
                if option in boolean_options and row[option] == 1:
                    row[option] = True
                elif option in boolean_options and row[option] == 0:
                    row[option] = False
                options[case][option] = row[option]
            else:
                options[case][option] = default_values[i]

    return cases, options

def get_seafloor_grid(
        reconstruction_name: str,
        reconstruction_time: int,
        DEBUG_MODE: bool = False
    ):
    """
    Function to obtain seafloor grid from GPlately DataServer
    
    :param reconstruction_name:    name of reconstruction
    :type reconstruction_name:     string
    :param reconstruction_times:   reconstruction times
    :type reconstruction_times:    list or numpy.array
    :param DEBUG_MODE:             whether to run in debug mode
    :type DEBUG_MODE:              bool

    :return:                       seafloor_grids
    :rtype:                        xarray.Dataset
    """
    # Call _gplately"s DataServer from the download.py module
    gdownload = _gplately.download.DataServer(reconstruction_name)

    if DEBUG_MODE:
        # Let the user know what is happening
        print(f"Downloading age grid for {reconstruction_name} at {reconstruction_time} Ma")
        age_raster = gdownload.get_age_grid(time=reconstruction_time)
    else:
        # Suppress print statements if not in debug mode
        with contextlib.redirect_stdout(io.StringIO()):
            age_raster = gdownload.get_age_grid(time=reconstruction_time)

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
    :param reconstruction_time:    reconstruction time
    :type reconstruction_time:     integer
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
    velocity_grid = velocity_grid.interp_like(seafloor_grid)

    # Interpolate NaN values along the dateline
    velocity_grid = velocity_grid.interpolate_na()

    return velocity_grid

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
    for case in cases:
        # Ignore processed cases
        if case in processed_cases:
            continue
        
        # Initialise list to store similar cases
        case_dict[case] = [case]

        # Add case to processed cases
        processed_cases.add(case)

        # Loop through other cases to find similar cases
        for other_case in cases:
            # Ignore if it is the same case
            if case == other_case:
                continue
            
            # Add case to processed cases if it is similar
            if all(options[case][opt] == options[other_case][opt] for opt in target_options):
                case_dict[case].append(other_case)
                processed_cases.add(other_case)

    return case_dict

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# SAVING 
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def DataFrame_to_csv(data, data_name, reconstruction_name, reconstruction_time, case, folder, DEBUG_MODE=False):
    """
    Function to save DataFrame to a folder

    :param data:                  data
    :type data:                   pandas.DataFrame
    :param data_name:             name of dataset
    :type data_name:              string
    :param reconstruction_name:   name of reconstruction
    :type reconstruction_name:    string
    :param reconstruction_time:   reconstruction time
    :type reconstruction_time:    integer
    :param case:                  case
    :type case:                   string
    :param folder:                folder
    :type folder:                 string
    :param DEBUG_MODE:                 whether to run in debug mode
    :type DEBUG_MODE:                  bool
    """
    if DEBUG_MODE:
        if folder:
            print(f"Saving {data_name} to {folder}/{data_name}")
        else:
            print(f"Saving {data_name} to {data_name}")

    # Determine the target directory
    target_dir = os.path.join(folder if folder else os.getcwd(), data_name)
    
    # Ensure the directory exists
    check_dir(target_dir)
    
    # Make file name
    file_name = f"{data_name}_{reconstruction_name}_{case}_{reconstruction_time}Ma.csv"
    file_path = os.path.join(target_dir, file_name)

    # Delete old file if it exists to prevent "Permission denied" error
    if os.path.exists(file_path):
        os.remove(file_path)

    # Save the data to CSV
    data.to_csv(file_path, index=False)

def Dataset_to_netCDF(data, data_name, reconstruction_name, reconstruction_time, folder, DEBUG_MODE=False):
    """
    Function to save xarray Dataset to a folder

    :param data:                  data
    :type data:                   xarray.Dataset
    :param data_name:             name of dataset
    :type data_name:              string
    :param reconstruction_name:   name of reconstruction
    :type reconstruction_name:    string
    :param reconstruction_time:   age of reconstruction in Ma
    :type reconstruction_time:    int
    :param folder:                folder
    :type folder:                 string
    """
    # Determine the target directory
    target_dir = os.path.join(folder if folder else os.getcwd(), data_name)
    
    # Ensure the directory exists
    check_dir(target_dir)
    
    # Make file name
    file_name = f"{data_name}_{reconstruction_name}_{reconstruction_time}Ma.nc"
    file_path = os.path.join(target_dir, file_name)

    # Print target directory and file path if in debug mode
    if DEBUG_MODE:
        print(f"Target directory for {data_name}: {target_dir}")
        print(f"File path for {data_name} at {reconstruction_time}: {file_path}")

    # Delete old file if it exists to prevent "Permission denied" error
    if os.path.exists(file_path):
        os.remove(file_path)
        if DEBUG_MODE:
            print(f"Deleted old file {file_path}")

    # Save the data to NetCDF
    data.to_netcdf(file_path)
    
def GeoDataFrame_to_shapefile(data, data_name, reconstruction_name, reconstruction_time, folder, DEBUG_MODE=False):
    """
    Function to save GeoDataFrame to a folder

    :param data:                  data
    :type data:                   geopandas.GeoDataFrame
    :param data_name:             name of dataset
    :type data_name:              string
    :param reconstruction_name:   name of reconstruction
    :type reconstruction_name:    string
    :param reconstruction_time:   age of reconstruction in Ma
    :type reconstruction_time:    int
    :param folder:                folder
    :type folder:                 string
    """
    # Determine the target directory
    target_dir = os.path.join(folder if folder else os.getcwd(), data_name)
    
    # Define target dir and check if it exists
    target_dir = os.path.join(folder, data_name)
    check_dir(target_dir)

    # Make file name
    file_name = f"{data_name}_{reconstruction_name}_{reconstruction_time}Ma.shp"
    file_path = os.path.join(target_dir, file_name)

    # Print target directory and file path if in debug mode
    if DEBUG_MODE:
        print(f"Target directory for {data_name}: {target_dir}")
        print(f"File path for {data_name} at {reconstruction_time}: {file_path}")

    # Delete old file if it exists to prevent "Permission denied" error
    if os.path.exists(file_path):
        os.remove(file_path)
        if DEBUG_MODE:
            print(f"Deleted old file {file_path}")

    # Save the data to a shapefile
    data.to_file(file_path)

def check_dir(target_dir):
    """
    Function to check if a directory exists, and create it if it doesn't
    """
    # Check if a directory exists, and create it if it doesn't
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# LOADING 
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def load_data(
        data: dict,
        reconstruction: _gplately.PlateReconstruction,
        reconstruction_name: str,
        reconstruction_times: list,
        type: str,
        all_cases: list,
        all_options: dict,
        matching_case_dict: dict,
        files_dir: Optional[str] = None,
        plates = None,
        resolved_topologies = None,
        resolved_geometries = None,
        DEBUG_MODE: Optional[bool] = False
    ):
    """
    Function to load DataFrames from a folder, or initialise new DataFrames
    
    :param data:                  data
    :type data:                   dict
    :param reconstruction:        reconstruction
    :type reconstruction:         gplately.PlateReconstruction
    :param reconstruction_name:   name of reconstruction
    :type reconstruction_name:    string
    :param reconstruction_times:  reconstruction times
    :type reconstruction_times:   list or _numpy.array

    :return:                      data
    :rtype:                       dict
    """
    # Loop through times
    for reconstruction_time in tqdm(reconstruction_times, desc=f"Loading {type} DataFrames", disable=DEBUG_MODE):
        
        # Initialise dictionary to store data for reconstruction time
        data[reconstruction_time] = {}

        # Initialise list to store available and unavailable cases
        unavailable_cases = all_cases.copy()
        available_cases = []

        # If a file directory is provided, check for the existence of files
        if files_dir:
            for case in all_cases:
                # Load DataFrame if found
                data[reconstruction_time][case] = DataFrame_from_csv(files_dir, type, reconstruction_name, case, reconstruction_time)

                if data[reconstruction_time][case] is not None:
                    unavailable_cases.remove(case)
                    available_cases.append(case)
                else:
                    if DEBUG_MODE:
                        print(f"DataFrame for {type} for {reconstruction_name} at {reconstruction_time} Ma for case {case} not found, checking for similar cases...")

        # Copy dataframes for unavailable cases
        for unavailable_case in unavailable_cases:
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
                    data[reconstruction_time][unavailable_case] = data[reconstruction_time][matching_case].copy()
                    available_cases.append(unavailable_case)
                    break
                
                # Initialise new DataFrame if not found
                if data[reconstruction_time][unavailable_case] is None:
                    if DEBUG_MODE:
                        # Let the user know you're busy
                        print(f"Initialising new DataFrame for {type} for {reconstruction_name} at {reconstruction_time} Ma for case {unavailable_case}...")

                    if type == "Plates":
                        data[reconstruction_time][unavailable_case] = get_plates(reconstruction.rotation_model, reconstruction_time, resolved_topologies[reconstruction_time], all_options[unavailable_case])
                    if type == "Slabs":
                        data[reconstruction_time][unavailable_case] = get_slabs(reconstruction, reconstruction_time, plates[reconstruction_time][unavailable_case], resolved_geometries[reconstruction_time], all_options[unavailable_case])
                    if type == "Points":
                        data[reconstruction_time][unavailable_case] = get_points(reconstruction, reconstruction_time, plates[reconstruction_time][unavailable_case], resolved_geometries[reconstruction_time], all_options[unavailable_case])

                    # Append case to available cases
                    available_cases.append(unavailable_case)

    return data

def load_torques(
        torques: dict,
        reconstruction_times: list,
        cases: list,
        plates: dict,
        plates_of_interest: list,
        DEBUG_MODE: Optional[bool] = False
    ):
    """
    Function to load torques DataFrames.

    :param torques:                 dictionary to store the torques DataFrames.
    :type torques:                  dict
    :param reconstruction_times:    reconstruction times.
    :type reconstruction_times:     list or numpy.array of ints
    :param cases:                   list of cases to process.
    :type cases:                    list of str
    :param plates:                  dictionary of plates DataFrames indexed by reconstruction time and case.
    :type plates:                   dict
    :param plates_of_interest:      plates to process.
    :type plates_of_interest:       list of int
    :param DEBUG_MODE:              whether or not to run in debug mode
    :type DEBUG_MODE:               bool

    :return:                        dictionary containing the torques DataFrames.
    :rtype:                         dict

    This function always reinitialises the torques DataFrames, as the information in them is also stored in the Plates DataFrames and can be recalculated from there.
    """
    # Define torque types
    torque_types = [
        "slab_pull_torque_mag", 
        "slab_pull_torque_opt_mag", 
        "GPE_torque_mag", 
        "slab_bend_torque_mag", 
        "mantle_drag_torque_mag", 
        "mantle_drag_torque_opt_mag", 
        "driving_torque_mag",
        "driving_torque_opt_mag",
        "residual_torque_mag",
        "residual_torque_opt_mag",
    ]

    # Loop through cases
    for case in tqdm(cases, desc="Loading torques", disable=DEBUG_MODE):
        if DEBUG_MODE:
            print(f"Loading torques for case: {case}")

        # Initialise dictionary to store torques for case
        torques[case] = {}

        # Loop through plates of interest
        for plate in plates_of_interest:
            if DEBUG_MODE:
                print(f"Loading torques for plate: {plate}")

            # Initialise array to store torques for plate
            torque_data = _numpy.zeros((len(reconstruction_times), 11))

            # Loop through reconstruction times
            for i, reconstruction_time in enumerate(reconstruction_times):
                if DEBUG_MODE:
                    print(f"Loading torques for {reconstruction_time} Ma...")

                # Store reconstruction time in first column of array
                torque_data[i, 0] = reconstruction_time

                # Check if plate is in plates DataFrame
                if plate in plates[reconstruction_time][case].plateID.values:
                    torque_data[i, 1:11] = plates[reconstruction_time][case].loc[
                        plates[reconstruction_time][case].plateID == plate, 
                        torque_types
                    ].values[0]

            # Convert to pandas DataFrame
            torques[case][plate] = _pandas.DataFrame(
                torque_data, 
                columns=[
                    "age", 
                    "slab_pull_torque", 
                    "slab_pull_torque_opt", 
                    "GPE_torque", 
                    "slab_bend_torque", 
                    "mantle_drag_torque", 
                    "mantle_drag_torque_opt", 
                    "driving_torque",
                    "driving_torque_opt",
                    "residual_torque",
                    "residual_torque_opt"
                ]
            )

    return torques

def load_grid(
        grid: dict,
        reconstruction_name: str,
        reconstruction_times: list,
        type: str,
        files_dir: str,
        points: Optional[dict] = None,
        seafloor_grid: Optional[_xarray.Dataset] = None,
        cases: Optional[list] = None,
        DEBUG_MODE: Optional[bool] = False
    ):
    """
    Function to load grid from a folder.

    :param grids:                  grids
    :type grids:                   dict
    :param reconstruction_name:    name of reconstruction
    :type reconstruction_name:     string
    :param reconstruction_times:   reconstruction times
    :type reconstruction_times:    list or numpy.array
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
    for reconstruction_time in tqdm(reconstruction_times, desc=f"Loading {type} grids", disable=DEBUG_MODE):
        # Check if the grid for the reconstruction time is already in the dictionary
        if reconstruction_time in grid:
            # Rename variables and coordinates in seafloor age grid for clarity
            if type == "Seafloor":
                if "z" in grid[reconstruction_time].data_vars:
                    grid[reconstruction_time] = grid[reconstruction_time].rename({"z": "seafloor_age"})
                if "lat" in grid[reconstruction_time].coords:
                    grid[reconstruction_time] = grid[reconstruction_time].rename({"lat": "latitude"})
                if "lon" in grid[reconstruction_time].coords:
                    grid[reconstruction_time] = grid[reconstruction_time].rename({"lon": "longitude"})

            continue

        # Load grid if found
        if type == "Seafloor":
            # Load grid if found
            grid[reconstruction_time] = Dataset_from_netCDF(files_dir, type, reconstruction_time, reconstruction_name)

            # Download seafloor age grid from GPlately DataServer
            grid[reconstruction_time] = get_seafloor_grid(reconstruction_name, reconstruction_time)

        elif type == "Velocity" and cases:
            # Initialise dictionary to store velocity grids for cases
            grid[reconstruction_time] = {}

            # Loop through cases
            for case in cases:
                # Load grid if found
                grid[reconstruction_time][case] = Dataset_from_netCDF(files_dir, type, reconstruction_time, reconstruction_name, case=case)

            # If not found, initialise a new grid
            if grid[reconstruction_time][case] is None:
                
                # Interpolate velocity grid from points
                if type == "Velocity":
                    for case in cases:
                        if DEBUG_MODE:
                            print(f"{type} grid for {reconstruction_name} at {reconstruction_time} Ma not found, interpolating from points...")

                        # Get velocity grid
                        grid[reconstruction_time][case] = get_velocity_grid(points[reconstruction_time][case], seafloor_grid[reconstruction_time])

    return grid

def DataFrame_from_csv(
        folder: str,
        type: str,
        reconstruction_name: str,
        case: str,
        reconstruction_time: int,
    ):
    """
    Function to load DataFrames from a folder

    :param folder:               folder
    :type folder:                str
    :param type:                 type of data
    :type type:                  str
    :param reconstruction_name:  name of reconstruction
    :type reconstruction_name:   str
    :param case:                 case
    :type case:                  str
    :param reconstruction_time:  reconstruction time
    :type reconstruction_time:   inte
    
    :return:                     data
    :rtype:                      pandas.DataFrame
    """
    # Get target folder
    if folder:
        target_file = os.path.join(folder, type, f"{type}_{reconstruction_name}_{case}_{reconstruction_time}Ma.csv")
    else:
        target_file = os.getcwd(type, f"{type}_{reconstruction_name}_{case}_{reconstruction_time}Ma.csv")  # Use the current working directory

    # Check if target file exists
    if os.path.exists(target_file):
        # Load data
        data = _pandas.read_csv(os.path.join(target_file))

        return data
    else:
        return None

def Dataset_from_netCDF(
        folder: str,
        type: str,
        reconstruction_time: int,
        reconstruction_name: str,
        case: Optional[str] = None,
    ):
    """
    Function to load xarray Dataset from a folder

    :param folder:               folder
    :type folder:                string
    :param reconstruction_times: reconstruction times
    :type reconstruction_times:  list or numpy.array
    :param reconstruction_name:  name of reconstruction
    :type reconstruction_name:   str
    :param case:                 case
    :type case:                  str

    :return:                     data
    :rtype:                      xarray.Dataset
    """
    # Make file name
    if case:
        file_name = f"{type}_{reconstruction_name}_{case}_{reconstruction_time}Ma.nc"
    else:
        file_name = f"{type}_{reconstruction_name}_{reconstruction_time}Ma.nc"

    # Get target file
    target_file = os.path.join(folder if folder else os.getcwd(), type, file_name)

    # Check if target folder exists
    if os.path.exists(target_file):
        # Load data
        data = _xarray.open_dataset(os.path.join(target_file), cache=False)

        return data
    else:
        return None
    
def GeoDataFrame_from_shapefile(
        folder: str,
        type: str,
        reconstruction_time: int,
        reconstruction_name: str,
    ):
    """
    Function to load GeoDataFrame from a folder

    :param folder:               folder
    :type folder:                string
    :param reconstruction_times: reconstruction times
    :type reconstruction_times:  list or numpy.array
    :param reconstruction_name:  name of reconstruction
    :type reconstruction_name:   string

    :return:                     data
    :rtype:                      geopandas.GeoDataFrame
    """
    # Get target folder
    if folder:
        target_file = os.path.join(folder, type, f"{type}_{reconstruction_name}_{reconstruction_time}Ma.shp")
    else:
        target_file = os.getcwd(type, f"{type}_{reconstruction_name}_{reconstruction_time}Ma.shp")  # Use the current working directory
    
    # Check if target folder exists
    if os.path.exists(target_file):
        # Load data
        data = _geopandas.read_file(os.path.join(target_file))

        return data
    else:
        return None