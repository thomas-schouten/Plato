# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# PLATEFO
# Algorithm to calculate plate forces from tectonic reconstructions
# Thomas Schouten and Edward Clennett, 2023
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Import libraries
# Standard libraries
import numpy as _numpy
import pandas as _pandas
import pygplates
from scipy.optimize import newton
import xarray as _xarray

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# MECHANICAL PARAMETERS AND CONSTANTS
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

class set_mech_params:
    """
    Class containing mechanical parameters used in calculations.
    """
    def __init__(self):
        # Mechanical and rheological parameters:
        self.g = 9.81                                       # gravity [m/s2]
        self.dT = 1200                                      # mantle-surface T contrast [K]
        self.rho0 = 3300                                    # reference mantle density  [kg/m3]
        self.rho_w = 1000                                   # water density [kg/m3]
        self.rho_sw = 1020                                  # water density for plate model
        self.rho_s = 2650                                   # density of sediments (quartz sand)
        self.rho_c = 2868                                   # density of continental crust
        self.rho_l = 3412                                   # lithosphere density
        self.rho_a = 3350                                   # asthenosphere density 
        self.alpha = 3e-5                                   # thermal expansivity [K-1]
        self.kappa = 1e-6                                   # thermal diffusivity [m2/s]
        self.depth = 700e3                                  # slab depth [m]
        self.rad_curv = 390e3                               # slab curvature [m]
        self.L = 130e3                                      # compensation depth [m]
        self.L0 = 100e3                                     # lithospheric shell thickness [m]
        self.La = 200e3                                     # asthenospheric thickness [m]
        self.visc_a = 1e20                                  # reference astheospheric viscosity [Pa s]
        self.lith_visc = 500e20                             # lithospheric viscosity [Pa s]
        self.lith_age_RP = 60                               # age of oldest sea-floor in approximate ridge push calculation  [Ma]
        self.yield_stress = 1050e6                          # Byerlee yield strength at 40km, i.e. 60e6 + 0.6*(3300*10.0*40e3) [Pa]
        self.cont_lith_thick = 100e3                        # continental lithospheric thickness (where there is no age) [m]
        self.cont_crust_thick = 33e3                        # continental crustal thickness (where there is no age) [m]
        self.island_arc_lith_thick = 50e3                   # island arc lithospheric thickness (where there is an age) [m]
        self.ocean_crust_thick = 8e3                        # oceanic crustal thickness [m]

        # Derived parameters
        self.drho_slab = self.rho0 * self.alpha * self.dT   # Density contrast between slab and surrounding mantle [kg/m3]
        self.drho_sed = self.rho_s - self.rho0              # Density contrast between sediments (quartz sand) and surrounding mantle [kg/m3]

# Create instance of mech
mech = set_mech_params()

class set_constants:
    """
    Class containing constants and conversions used calculations.
    """
    def __init__(self):
        # Constants
        self.mean_Earth_radius_km = 6371                # mean Earth radius [km]
        self.mean_Earth_radius_m = 6371e3               # mean Earth radius [m]
        self.equatorial_Earth_radius_m = 6378.1e3       # Earth radius at equator
        self.equatorial_Earth_circumference = 40075e3   # Earth circumference at equator [m]
        
        # Conversions
        self.ma2s = 1e6 * 365.25 * 24 * 60 * 60         # Ma to s
        self.s2ma = 1 / self.ma2s                       # s to Ma
        self.m_s2cm_a = 1e2 * (365.25 * 24 * 60 * 60)   # m/s to cm/a 
        self.cm_a2m_s = 1 / self.m_s2cm_a               # cm/a to m/s
        self.rad_a2m_s =  self.mean_Earth_radius_m * \
            _numpy.pi/180 / (365.25 * 24 * 60 * 60)     # rad/a to m/s
        self.m_s2rad_a = 1 / self.rad_a2m_s             # m/s to rad/a
        self.cm2in = 0.3937008

# Create instance of mech
constants = set_constants()

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# SUBDUCTION ZONES
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def compute_slab_pull_force(slabs, options, mech):
    """
    Function to optimise slab pull force at subduction zones

    :param slabs:       subduction zone data
    :type slabs:        pandas.DataFrame
    :param options:     options
    :type options:      dict
    :param mech:        mechanical parameters used in calculations
    :type mech:         class

    :return:            slabs
    :rtype:             pandas.DataFrame
    """
    # Calculate slab pull force acting on point along subduction zone
    slabs["slab_pull_force_mag"] = _numpy.where(
        _numpy.isnan(slabs.lower_plate_age),
        0,
        slabs["lower_plate_thickness"] * slabs.slab_length * mech.drho_slab * mech.g * 1/_numpy.sqrt(_numpy.pi)
        )

    if options["Sediment subduction"]:
        # Add positive buoyancy of sediments
        slabs.slab_pull_force_mag = _numpy.where(
            _numpy.isnan(slabs.lower_plate_age), 
            slabs.slab_pull_force_mag,
            slabs.slab_pull_force_mag + slabs.sediment_thickness * slabs.slab_length * mech.drho_sed * mech.g * 1/_numpy.sqrt(_numpy.pi)
        )

    # Decompose into latitudinal and longitudinal components
    slabs["slab_pull_force_lat"], slabs["slab_pull_force_lon"] = mag_azi2lat_lon(slabs.slab_pull_force_mag, slabs.trench_normal_azimuth)

    return slabs

def compute_interface_term(slabs, options):
    """
    Function to calculate the interface term that accounts for resisting forces at the subduction interface.
    These forces are i) shearing along the plate interface, ii) bending of the slab, and iii) vertical resistance to slab sinking.

    :param slabs:       subduction zone data
    :type slabs:        pandas.DataFrame
    :param options:     options
    :type options:      dict

    :return:            slabs
    :rtype:             pandas.DataFrame
    """
    # Determine sediment fraction
    if options["Sediment subduction"]:
        # Determine shear zone width
        if options["Shear zone width"] == "variable":
            slabs["shear_zone_width"] = slabs["v_convergence_mag"] * constants.cm_a2m_s / options["Strain rate"]
        else:
            slabs["shear_zone_width"] = options["Shear zone width"]

        # Calculate sediment fraction using sediment thickness and shear zone width
        slabs["sediment_fraction"] = _numpy.where(_numpy.isnan(slabs.lower_plate_age), 0, slabs["sediment_thickness"] / slabs["shear_zone_width"])
        slabs["sediment_fraction"] = _numpy.where(slabs["sediment_thickness"] <= slabs["shear_zone_width"], slabs["sediment_fraction"],  1)
        slabs["sediment_fraction"] = _numpy.nan_to_num(slabs["sediment_fraction"])

    # Calculate interface term for all components of the slab pull force
    interface_term = 10 * options["Slab pull constant"] - options["Slab pull constant"] * 10**(1-slabs["sediment_fraction"]) * options["Slab pull constant"]

    # Apply interface term to slab pull force
    slabs["slab_pull_force_opt_mag"] = slabs["slab_pull_force_mag"] * interface_term
    slabs["slab_pull_force_opt_lat"] = slabs["slab_pull_force_lat"] * interface_term
    slabs["slab_pull_force_opt_lon"] = slabs["slab_pull_force_lon"] * interface_term

    return slabs

def compute_slab_bending_force(slabs, options, mech, constants):
    """
    Function to calculate the slab bending force

    :param slabs:       subduction zone data
    :type slabs:        pandas.DataFrame
    :param options:     options
    :type options:      dict
    :param mech:        mechanical parameters used in calculations
    :type mech:         class

    :return:            slabs
    :rtype:             pandas.DataFrame
    """
    # Calculate slab bending torque
    if options["Bending mechanism"] == "viscous":
        bending_force = (-2. / 3.) * ((slabs.lower_plate_thickness) / (mech.rad_curv)) ** 3 * mech.lith_visc * slabs.v_convergence * constants.cm_a2m_s # [n-s , e-w], [N/m]
    elif options["Bending mechanism"] == "plastic":
        bending_force = (-1. / 6.) * ((slabs.lower_plate_thickness ** 2) / mech.rad_curv) * mech.yield_stress * _numpy.array(
            (_numpy.cos(slabs.trench_normal_vector + slabs.obliquity_convergence), _numpy.sin(slabs.trench_normal_vector + slabs.obliquity_convergence)))  # [n-s, e-w], [N/m]
        
    slabs["bending_force_lat"], slabs["bending_force_lon"] = mag_azi2lat_lon(bending_force, slabs.trench_normal_vector + slabs.obliquity_convergence)
    
    return slabs

def sample_slabs_from_seafloor(
        lat,
        lon,
        trench_normal_azimuth, 
        seafloor, 
        options, 
        plate, 
        age_variable="seafloor_age", 
        coords=["latitude", "longitude"], 
        continental_arc=None, 
        sediment_thickness=None, 
    ):
    """
    Function to obtain relevant upper or lower plate data from tesselated subduction zones.

    :param lat:                     column of pandas.DataFrame containing latitudes.
    :type lat:                      numpy.array
    :param lon:                     column of pandas.DataFrame containing longitudes.
    :type lon:                      numpy.array
    :param trench_normal_azimuth:   column of pandas.DataFrame containing trench normal azimuth.
    :type trench_normal_azimuth:    numpy.array
    :param seafloor:                xarray.Dataset containing seafloor age data
    :type seafloor:                 xarray.Dataset
    :param options:                 dictionary with options
    :type options:                  dict
    :param plate:                   plate type
    :type plate:                    str
    :param age_variable:            name of variable in xarray.dataset containing seafloor ages
    :type age_variable:             str
    :param coords:                  coordinates of seafloor data
    :type coords:                   list
    :param continental_arc:         column of pandas.DataFrame containing boolean values indicating whether arc is continental or not
    :type continental_arc:          numpy.array
    :param sediment_thickness:      column of pandas.DataFrame containing sediment thickness
    :type sediment_thickness:       numpy.array

    :return:                        slabs
    :rtype:                         pandas.DataFrame
    """
    # Load seafloor into memory to decrease computation time
    seafloor = seafloor.load()

    # Define sampling distance [km]
    if plate == "lower plate":
        initial_sampling_distance = -30
    if plate == "upper plate":
        initial_sampling_distance = 200

    # Sample plate
    sampling_lat, sampling_lon = project_points(lat, lon, trench_normal_azimuth, initial_sampling_distance)

    # Extract latitude and longitude values from slabs and convert to xarray DataArrays
    sampling_lat_da = _xarray.DataArray(sampling_lat, dims="point")
    sampling_lon_da = _xarray.DataArray(sampling_lon, dims="point")

    # Interpolate age value at point
    ages = seafloor[age_variable].interp({coords[0]: sampling_lat_da, coords[1]: sampling_lon_da}).values.tolist()

    # Find problematic indices to iteratively find age of lower plate
    initial_mask = _numpy.isnan(ages)
    mask = initial_mask

    # Define sampling distance [km] and number of iterations
    if plate == "lower plate":
        current_sampling_distance = initial_sampling_distance - 30
        iterations = 12
    if plate == "upper plate":
        current_sampling_distance = initial_sampling_distance + 200
        iterations = 8

    for i in range(iterations):
        sampling_lat[mask], sampling_lon[mask] = project_points(lat[mask], lon[mask], trench_normal_azimuth[mask], current_sampling_distance)
        sampling_lat_da = _xarray.DataArray(sampling_lat, dims="point")
        sampling_lon_da = _xarray.DataArray(sampling_lon, dims="point")
        ages = _numpy.where(mask, seafloor[age_variable].interp({coords[0]: sampling_lat_da, coords[1]: sampling_lon_da}).values.tolist(), ages)
        mask = _numpy.isnan(ages)

        # Define new sampling distance
        if plate == "lower plate":
            if i <= 1:
                current_sampling_distance -= 30
            elif i % 2 == 0:
                current_sampling_distance -= 30 * (2 ** (i // 2))

                if plate == "upper plate":
                        current_sampling_distance += 100

    # Check whether arc is continental or not
    if plate == "upper plate":
        # Set continental arc to True where there is no age
        continental_arc = _numpy.isnan(ages)

        # Manual overrides for island arcs that do not appear in the seafloor age grid. Hardcoded by lack of a better method for now.
        # island_arc_plateIDs = []

        # Sample erosion rate, if applicable
        if options["Sediment subduction"] and options["Sample erosion grid"] in seafloor.data_vars:
            # Reset sediment thickness to avoid adding double the sediment
            sediment_thickness = _numpy.zeros(len(ages))

            # Set new sampling points 100 km inboard of the trench
            sampling_lat, sampling_lon = project_points(lat, lon, trench_normal_azimuth, 300)

            # Convert to xarray DataArrays
            sampling_lat_da = _xarray.DataArray(sampling_lat, dims="point")
            sampling_lon_da = _xarray.DataArray(sampling_lon, dims="point")

            # Interpolate elevation change at sampling points
            erosion_rate = seafloor[options["Sample erosion grid"]].interp({coords[0]: sampling_lat_da, coords[1]: sampling_lon_da}).values.tolist()

            # For NaN values, sample 100 km further inboard
            current_sampling_distance = 250
            for i in range(3):
                # Find problematic indices to iteratively find erosion/deposition rate of upper plate
                mask = _numpy.isnan(erosion_rate)

                # Define new sampling points
                sampling_lat[mask], sampling_lon[mask] = project_points(lat[mask], lon[mask], trench_normal_azimuth[mask], current_sampling_distance)

                # Convert to xarray DataArrays
                sampling_lat_da = _xarray.DataArray(sampling_lat, dims="point")
                sampling_lon_da = _xarray.DataArray(sampling_lon, dims="point")

                # Interpolate elevation change at sampling points
                erosion_rate = _numpy.where(mask, seafloor[options["Sample erosion grid"]].interp({coords[0]: sampling_lat_da, coords[1]: sampling_lon_da}).values.tolist(), erosion_rate)

                # Define new sampling distance
                current_sampling_distance += 50

            # Close the seafloor to free memory space
            seafloor.close()
            
            # Convert erosion rate to sediment thickness
            sediment_thickness += erosion_rate * options["Erosion to sediment ratio"]
            
            return ages, continental_arc, erosion_rate, sediment_thickness
        
        else:
            # Close the seafloor to free memory space
            seafloor.close()

            return ages, continental_arc
 
    if plate == "lower plate":
        # Reset sediment thickness to avoid adding double the sediment
        sediment_thickness = _numpy.zeros(len(ages))

        # Add active margin sediments
        if options["Sediment subduction"] and options["Active margin sediments"] != 0 and not options["Sample erosion grid"]:
            sediment_thickness = _numpy.where(continental_arc == True, sediment_thickness+options["Active margin sediments"], sediment_thickness)

        # Sample sediment grid
        if options["Sediment subduction"] and options["Sample sediment grid"] != 0:
            # Extract latitude and longitude values from slabs and convert to xarray DataArrays
            sampling_lat_da = _xarray.DataArray(sampling_lat, dims="point")
            sampling_lon_da = _xarray.DataArray(sampling_lon, dims="point")
            sediment_thickness += seafloor[options["Sample sediment grid"]].interp({coords[0]: sampling_lat_da, coords[1]: sampling_lon_da}).values.tolist()

        # Close the seafloor to free memory space
        seafloor.close()

        return ages, sediment_thickness

def project_points(lat, lon, azimuth, distance):
    """
    Function to calculate coordinates of sampling points

    :param lat:         column of _pandas.DataFrame containing latitudes.
    :type lat:          numpy.array
    :param lon:         column of _pandas.DataFrame containing longitudes.
    :type lon:          numpy.array
    :param azimuth:     column of _pandas.DataFrame containing trench normal azimuth.
    :type azimuth:      numpy.array
    :param distance:    distance to project points [km].
    :type distance:     float

    :return:            sampling_lat, sampling_lon
    :rtype:             numpy.array, numpy.array
    """
    # Set constants
    constants = set_constants()

    # Convert to radians
    lon_radians = _numpy.deg2rad(lon)
    lat_radians = _numpy.deg2rad(lat)
    azimuth_radians = _numpy.deg2rad(azimuth)

    # Angular distance in km
    angular_distance = distance / constants.mean_Earth_radius_km

    # Calculate sample points
    new_lat_radians = _numpy.arcsin(_numpy.sin(lat_radians) * _numpy.cos(angular_distance) + _numpy.cos(lat_radians) * _numpy.sin(angular_distance) * _numpy.cos(azimuth_radians))
    new_lon_radians = lon_radians + _numpy.arctan2(_numpy.sin(azimuth_radians) * _numpy.sin(angular_distance) * _numpy.cos(lat_radians), _numpy.cos(angular_distance) - _numpy.sin(lat_radians) * _numpy.sin(new_lat_radians))
    new_lon = _numpy.degrees(new_lon_radians)
    new_lat = _numpy.degrees(new_lat_radians)

    return new_lat, new_lon

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# GRAVITATIONAL POTENTIAL ENERGY
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def compute_GPE_force(points, seafloor, options, mech, age_variable="seafloor_age"):
    """
    Function to calculate GPE force at points

    :param points:                  pandas.DataFrame containing data of points including columns with latitude, longitude and plateID
    :type points:                   pandas.DataFrame
    :param seafloor:                xarray.Dataset containing seafloor age data
    :type seafloor:                 xarray.Dataset
    :param options:                 dictionary with options
    :type options:                  dict
    :param age_variable:            name of variable in _xarray.dataset containing seafloor ages
    :type age_variable:             str

    :return:                        points
    :rtype:                         pandas.DataFrame
    """
    # Get grid spacing
    grid_spacing_deg = options["Grid spacing"]

    # Get nearby points
    # Longitude
    dx_lon = points.lon + 0.5 * grid_spacing_deg
    minus_dx_lon = points.lon - 0.5 * grid_spacing_deg

    # Adjust for dateline
    dx_lon = _numpy.where(dx_lon > 180, dx_lon - 360, dx_lon)
    minus_dx_lon = _numpy.where(minus_dx_lon < -180, minus_dx_lon + 360, minus_dx_lon)

    # Latitude
    dy_lat = points.lat + 0.5 * grid_spacing_deg
    minus_dy_lat = points.lat - 0.5 * grid_spacing_deg

    # Adjust for poles
    dy_lat = _numpy.where(dy_lat > 90, 90 - 2 * grid_spacing_deg, dy_lat)
    dy_lon = _numpy.where(dy_lat > 90, points.lon + 180, points.lon)
    dy_lon = _numpy.where(dy_lon > 180, dy_lon - 360, dy_lon)
    minus_dy_lat = _numpy.where(minus_dy_lat < -90, -90 + 2 * grid_spacing_deg, minus_dy_lat)
    minus_dy_lon = _numpy.where(minus_dy_lat < -90, points.lon + 180, points.lon)
    minus_dy_lon = _numpy.where(minus_dy_lon > 180, minus_dy_lon - 360, minus_dy_lon)

    # Sample ages and compute crustal thicknesses at points
    points["lithospheric_mantle_thickness"], points["crustal_thickness"], points["water_depth"] = compute_thicknesses(
                points["seafloor_age"],
                options
    )

    # Height of layers for integration
    zw = mech.L - points.water_depth
    zc = mech.L - (points.water_depth + points.crustal_thickness)
    zl = mech.L - (points.water_depth + points.crustal_thickness + points.lithospheric_mantle_thickness)

    # Calculate U
    points["U"] = 0.5 * mech.g * (
        mech.rho_a * (zl) ** 2 +
        mech.rho_l * (zc) ** 2 -
        mech.rho_l * (zl) ** 2 +
        mech.rho_c * (zw) ** 2 -
        mech.rho_c * (zc) ** 2 +
        mech.rho_sw * (mech.L) ** 2 -
        mech.rho_sw * (zw) ** 2
    )
    
    # Sample ages and compute crustal thicknesses at nearby points
    # dx_lon
    ages = {}
    for i in range(0,4):
        if i == 0:
            sampling_lat = points.lat; sampling_lon = dx_lon
        if i == 1:
            sampling_lat = points.lat; sampling_lon = minus_dx_lon
        if i == 2:
            sampling_lat = dy_lat; sampling_lon = dy_lon
        if i == 3:
            sampling_lat = minus_dy_lat; sampling_lon = minus_dy_lon

        ages[i] = sample_ages(sampling_lat, sampling_lon, seafloor[age_variable])
        lithospheric_mantle_thickness, crustal_thickness, water_depth = compute_thicknesses(
                    ages[i],
                    options
        )

        # Height of layers for integration
        zw = mech.L - water_depth
        zc = mech.L - (water_depth + crustal_thickness)
        zl = mech.L - (water_depth + crustal_thickness + lithospheric_mantle_thickness)

        # Calculate U
        U = 0.5 * mech.g * (
            mech.rho_a * (zl) ** 2 +
            mech.rho_l * (zc) ** 2 -
            mech.rho_l * (zl) ** 2 +
            mech.rho_c * (zw) ** 2 -
            mech.rho_c * (zc) ** 2 +
            mech.rho_sw * (mech.L) ** 2 -
            mech.rho_sw * (zw) ** 2
        )

        if i == 0:
            dx_U = U
        if i == 1:
            minus_dx_U = U
        if i == 2:
            dy_U = U
        if i == 3:
            minus_dy_U = U

    # Calculate force
    points["GPE_force_lat"] = (-mech.L0 / mech.L) * (dy_U - minus_dy_U) / points["segment_length_lat"]
    points["GPE_force_lon"] = (-mech.L0 / mech.L) * (dx_U - minus_dx_U) / points["segment_length_lon"]

    # Eliminate passive continental margins
    if not options["Continental crust"]:
        points["GPE_force_lat"] = _numpy.where(points["seafloor_age"].isna(), 0, points["GPE_force_lat"])
        points["GPE_force_lon"] = _numpy.where(points["seafloor_age"].isna(), 0, points["GPE_force_lon"])
        for i in range(0,4):
            points["GPE_force_lat"] = _numpy.where(_numpy.isnan(ages[i]), 0, points["GPE_force_lat"])
            points["GPE_force_lon"] = _numpy.where(_numpy.isnan(ages[i]), 0, points["GPE_force_lon"])

    return points

def sample_ages(lat, lon, seafloor, coords=["latitude", "longitude"]):
    # Load seafloor into memory to decrease computation time
    seafloor = seafloor.load()

    # Extract latitude and longitude values from points and convert to xarray DataArrays
    lat_da = _xarray.DataArray(lat, dims="point")
    lon_da = _xarray.DataArray(lon, dims="point")

    # Interpolate age value at point
    ages = _numpy.array(seafloor.interp({coords[0]: lat_da, coords[1]: lon_da}, method="nearest").values.tolist())

    # Close the seafloor to free memory space
    seafloor.close()

    return ages

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# BASAL TRACTIONS
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def compute_mantle_drag_force(torques, points, slabs, options, mech, constants):
    """
    Function to calculate mantle drag force at points

    :param torques:                 pandas.DataFrame containing
    :type torques:                  pandas.DataFrame
    :param points:                  pandas.DataFrame containing data of points including columns with latitude, longitude and plateID
    :type points:                   pandas.DataFrame
    :param options:                 dictionary with options
    :type options:                  dict
    :param mech:                    mechanical parameters used in calculations
    :type mech:                     class
    :param constants:               constants used in calculations
    :type constants:                class

    :return:                        torques, points
    :rtype:                         pandas.DataFrame, pandas.DataFrame
    """
    # Get velocities at points
    if options["Reconstructed motions"]:
        # Calculate mantle drag force
        points["mantle_drag_force_lat"] = -1 * points.v_lat * constants.cm_a2m_s * options["Mantle viscosity"] / mech.La
        points["mantle_drag_force_lon"] = -1 * points.v_lon * constants.cm_a2m_s * options["Mantle viscosity"] / mech.La

    else:
        # Calculate residual torque
        for axis in ["_x", "_y", "_z"]:
            torques["mantle_drag_torque" + axis] = (torques["slab_pull_torque_opt" + axis] + torques["GPE_torque" + axis] + torques["slab_bend_torque" + axis]) * -1
        torques["mantle_drag_torque_opt_mag"] = xyz2mag(torques["mantle_drag_torque_x"], torques["mantle_drag_torque_y"], torques["mantle_drag_torque_z"])
        
        # Convert to centroid
        centroid_position = lat_lon2xyz(torques.centroid_lat, torques.centroid_lon, constants)
        centroid_unit_position = centroid_position / constants.mean_Earth_radius_m
        
        # Calculate force from cross product of torques with centroid position
        summed_torques_cartesian = _numpy.array([torques["mantle_drag_torque_x"], torques["mantle_drag_torque_y"], torques["mantle_drag_torque_z"]])
        summed_torques_cartesian_normalised = summed_torques_cartesian / (_numpy.repeat(_numpy.array(torques.area)[_numpy.newaxis, :], 3, axis=0) * options["Mantle viscosity"]/mech.La)
        force_at_centroid = _numpy.cross(summed_torques_cartesian, centroid_unit_position, axis=0)
        velocity_at_centroid = _numpy.cross(-1 * summed_torques_cartesian_normalised, centroid_unit_position, axis=0)

        # Calculate force at centroid
        torques["mantle_drag_force_lat"], torques["mantle_drag_force_lon"], torques["mantle_drag_force_mag"], torques["mantle_drag_force_azi"] = vector_xyz2lat_lon(
            torques.centroid_lat, torques.centroid_lon, force_at_centroid, constants
        )

        # Calculate velocity at centroid and convert to cm/a
        torques["centroid_v_lat"], torques["centroid_v_lon"], torques["centroid_v_mag"], torques["centroid_v_azi"] = vector_xyz2lat_lon(torques.centroid_lat, torques.centroid_lon, velocity_at_centroid, constants)
        torques["centroid_v_lat"] *= constants.m_s2cm_a; torques["centroid_v_lon"] *= constants.m_s2cm_a; torques["centroid_v_mag"] *= constants.m_s2cm_a

        # Loop through slabs
        for j in range(len(slabs.lat)):
            # Check if upper plate is in torques
            if slabs.lower_plateID[j] in torques.plateID.values:
                # Get the index of the lower plate in the torques DataFrame
                n = _numpy.where(torques.plateID.values == slabs.lower_plateID[j])[0][0]

                # Check if the area condition is satisfied
                if torques.area.values[n] >= options["Minimum plate area"]:
                    # Calculate the velocity of the lower plate as the cross product of the torque and the unit position vector
                    slab_velocity = vector_xyz2lat_lon(
                        [slabs.loc[j, "lat"]],
                        [slabs.loc[j, "lon"]],
                        _numpy.array(
                            [_numpy.cross(
                            -1 * summed_torques_cartesian_normalised[:,n], lat_lon2xyz(
                                slabs.loc[j, "lat"], slabs.loc[j, "lon"], constants
                                ) / constants.mean_Earth_radius_m,
                            axis=0
                            )]
                        ).T,
                        constants
                    )

                    # Assign the velocity to the respective columns in the slabs DataFrame
                    slabs.loc[j, "v_lower_plate_lat"] = slab_velocity[0]
                    slabs.loc[j, "v_lower_plate_lon"] = slab_velocity[1]
                    slabs.loc[j, "v_lower_plate_mag"] = slab_velocity[2]
                    slabs.loc[j, "v_lower_plate_azi"] = slab_velocity[3]
                else:
                    # If the area condition is not satisfied, set the velocity to zero
                    slabs.loc[j, "v_lower_plate_lat"] = 0
                    slabs.loc[j, "v_lower_plate_lon"] = 0
                    slabs.loc[j, "v_lower_plate_mag"] = 0
                    slabs.loc[j, "v_lower_plate_azi"] = _numpy.nan

            # If the torque balance is not known for the lower plate, set the velocity to zero
            else:
                slabs.loc[j, "v_lower_plate_lat"] = 0
                slabs.loc[j, "v_lower_plate_lon"] = 0
                slabs.loc[j, "v_lower_plate_mag"] = 0
                slabs.loc[j, "v_lower_plate_azi"] = _numpy.nan

            # Check if the torque balance is known for the upper plate
            if slabs.upper_plateID[j] in torques.plateID.values:
                # Get the index of the plate in the torques DataFrame
                m = _numpy.where(torques.plateID.values == slabs.upper_plateID[j])[0][0]

                # Check if the area condition is satisfied
                if torques.area.values[m] >= options["Minimum plate area"]:
                    # Calculate the velocity of the lower plate as the cross product of the torque and the unit position vector
                    trench_velocity = vector_xyz2lat_lon(
                        [slabs.lat[j]],
                        [slabs.lon[j]],
                        _numpy.array(
                            [_numpy.cross(
                            -1 * summed_torques_cartesian_normalised[:,m], lat_lon2xyz(
                                slabs.lat[j], slabs.lon[j], constants
                                ) / constants.mean_Earth_radius_m,
                            axis=0
                            )]
                        ).T,
                        constants
                    )
                    
                    # Assign the velocity to the respective columns in the slabs DataFrame
                    slabs.loc[j, "v_upper_plate_lat"] = trench_velocity[0]
                    slabs.loc[j, "v_upper_plate_lon"] = trench_velocity[1]
                    slabs.loc[j, "v_upper_plate_mag"] = trench_velocity[2]
                    slabs.loc[j, "v_upper_plate_azi"] = trench_velocity[3]

                else:
                    # If the area condition is not satisfied, set the velocity to zero
                    slabs.loc[j, "v_upper_plate_lat"] = 0
                    slabs.loc[j, "v_upper_plate_lon"] = 0
                    slabs.loc[j, "v_upper_plate_mag"] = 0
                    slabs.loc[j, "v_upper_plate_azi"] = _numpy.nan

            # If the torque balance is not known for the upper plate, set the velocity to zero
            else:
                slabs.loc[j, "v_upper_plate_lat"] = 0
                slabs.loc[j, "v_upper_plate_lon"] = 0
                slabs.loc[j, "v_upper_plate_mag"] = 0
                slabs.loc[j, "v_upper_plate_azi"] = _numpy.nan

        # Convert to cm/a
        slabs["v_lower_plate_lat"] *= constants.m_s2cm_a; slabs["v_lower_plate_lon"] *= constants.m_s2cm_a; slabs["v_lower_plate_mag"] *= constants.m_s2cm_a
        slabs["v_upper_plate_lat"] *= constants.m_s2cm_a; slabs["v_upper_plate_lon"] *= constants.m_s2cm_a; slabs["v_upper_plate_mag"] *= constants.m_s2cm_a
        
    return torques, points, slabs

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# GENERAL FUNCTIONS
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def compute_thicknesses(ages, options, crust=True, water=True):
    """
    Calculate lithospheric mantle thickness, crustal thickness, and water depth based on seafloor age profiles.

    :param ages:                Seafloor ages for which thicknesses are calculated.
    :type ages:                 array-like
    :param options:             Options for controlling the calculation, including the seafloor age profile.
    :type options:              dict
    :param crust:               Flag to calculate crustal thickness. Defaults to True.
    :type crust:                bool
    :param water:               Flag to calculate water depth. Defaults to True.
    :type water:                bool

    :return:                    Calculated lithospheric mantle thickness.
    :rtype:                     numpy.array
    :return crustal_thickness:  Calculated crustal thickness if crust is True, otherwise None.
    :rtype crustal_thickness:   numpy.array or None
    :return water_depth:        Calculated water depth if water is True, otherwise None.
    :rtype water_depth:         numpy.array or None

    This function calculates lithospheric mantle thickness, crustal thickness, and water depth based on seafloor age profiles.
    The calculation depends on options["Seafloor age profile"]:
        - If "half space cooling", lithospheric_mantle_thickness is calculated from half space cooling theory.
        - If "plate model", lithospheric_mantle_thickness is calculated from a plate model.
    
    Crustal thickness and water depth are optional and depend on the values of the 'crust' and 'water' parameters, respectively.
    """
    # Set mechanical parameters and constants
    mech = set_mech_params()
    constants = set_constants()

    # Thickness of oceanic lithosphere from half space cooling and water depth from isostasy
    if options["Seafloor age profile"] == "half space cooling":
        lithospheric_mantle_thickness = _numpy.where(_numpy.isnan(ages), 
                                                mech.cont_lith_thick, 
                                                2.32 * _numpy.sqrt(mech.kappa * ages * constants.ma2s))
        
        if crust:
            crustal_thickness = _numpy.where(_numpy.isnan(ages), 
                                        mech.cont_crust_thick, 
                                        mech.ocean_crust_thick)
        else:
            crustal_thickness = None
            
        if water:
            water_depth = _numpy.where(_numpy.isnan(ages), 
                                0,
                                (lithospheric_mantle_thickness * ((mech.rho_a - mech.rho_l) / (mech.rho_sw - mech.rho_a))) + 2600)
        else:
            water_depth = None
        
    # Water depth from half space cooling and lithospheric thickness from isostasy
    elif options["Seafloor age profile"] == "plate model":
        hw = _numpy.where(ages > 81, 6586 - 3200 * _numpy.exp((-ages / 62.8)), ages)
        hw = _numpy.where(hw <= 81, 2600 + 345 * _numpy.sqrt(hw), hw)
        lithospheric_mantle_thickness = (hw - 2600) * ((mech.rho_sw - mech.rho_a) / (mech.rho_a - mech.rho_l))

        if crust:
            crustal_thickness = _numpy.where(_numpy.isnan(ages), 
                                        mech.cont_crust_thick, 
                                        mech.ocean_crust_thick)
        else:
            crustal_thickness = None
        
        if water:
            water_depth = hw
        else:
            water_depth = None

    return lithospheric_mantle_thickness, crustal_thickness, water_depth

def compute_torque_on_plates(torques, lat, lon, plateID, force_lat, force_lon, segment_length_lat, segment_length_lon, constants, torque_variable="torque"):
    """
    Calculate and update torque information on plates based on latitudinal, longitudinal forces, and segment dimensions.

    :param torques:             Torque data with columns 'plateID', 'centroid_lat', 'centroid_lon', and torque components.
    :type torques:              pd.DataFrame
    :param lat:                 Latitude of plate points in degrees.
    :type lat:                  float or array-like
    :param lon:                 Longitude of plate points in degrees.
    :type lon:                  float or array-like
    :param plateID:             Plate IDs corresponding to each point.
    :type plateID:              float or array-like
    :param force_lat:           Latitudinal component of the applied force.
    :type force_lat:            float or array-like
    :param force_lon:           Longitudinal component of the applied force.
    :type force_lon:            float or array-like
    :param segment_length_lat:  Length of the segment in the latitudinal direction.
    :type segment_length_lat:   float or array-like
    :param segment_length_lon:  Length of the segment in the longitudinal direction.
    :type segment_length_lon:   float or array-like
    :param constants:           Constants used in coordinate conversions and calculations.
    :type constants:            class
    :param torque_variable:     Name of the torque variable. Defaults to "torque".
    :type torque_variable:      str

    :return: Updated torques DataFrame with added columns for torque components at the centroid, force components at the centroid, and latitudinal and longitudinal components of the force.
    :rtype: pd.DataFrame

    This function calculates torques in Cartesian coordinates based on latitudinal, longitudinal forces, and segment dimensions.
    It then sums the torque components for each plate, calculates the torque vector at the centroid, and updates the torques DataFrame.
    Finally, it calculates the force components at the centroid, converts them to latitudinal and longitudinal components, and adds these to the torques DataFrame.
    """
    # Initialize dataframes and sort plateIDs
    data = _pandas.DataFrame({"plateID": plateID})

    # Convert points to Cartesian coordinates
    position = lat_lon2xyz(lat, lon, constants)
    
    # Calculate torques in Cartesian coordinates
    torques_cartesian = torques2xyz(position, lat, lon, force_lat, force_lon, segment_length_lat, segment_length_lon)
    
    # Assign the calculated torques to the new torque_variable columns
    data[torque_variable + "_x"] = torques_cartesian[0]
    data[torque_variable + "_y"] = torques_cartesian[1]
    data[torque_variable + "_z"] = torques_cartesian[2]

    # Sum components of plates based on plateID
    summed_data = data.groupby("plateID", as_index=True).sum().fillna(0)

    # Set "plateID" as the index of the torques DataFrame
    torques.set_index("plateID", inplace=True)

    # Update the torques DataFrame with values from summed_data
    torques.update(summed_data)

    # Reset the index of torques and keep "plateID" as a column
    torques.reset_index(inplace=True)

    # Calculate the position vector of the centroid of the plate in Cartesian coordinates
    centroid_position = lat_lon2xyz(torques.centroid_lat, torques.centroid_lon, constants)

    # Calculate the torque vector as the cross product of the Cartesian torque vector (x, y, z) with the position vector of the centroid
    summed_torques_cartesian = _numpy.array([torques[torque_variable + "_x"], torques[torque_variable + "_y"], torques[torque_variable + "_z"]])
    force_at_centroid = _numpy.cross(summed_torques_cartesian, centroid_position, axis=0) 

    # Compute force magnitude at centroid
    force_variable = torque_variable.replace("torque", "force")
    torques[force_variable + "_lat"], torques[force_variable + "_lon"], torques[force_variable + "_mag"], torques[force_variable + "_azi"] = vector_xyz2lat_lon(
        torques.centroid_lat, torques.centroid_lon, force_at_centroid, constants
    )
    
    return torques

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# CONVERSIONS
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def vector_xyz2lat_lon(lats, lons, vector, constants):
    """
    Function to convert a 3D vector into latitudinal and longitudinal components at a point.

    :param lats:     Latitude in degrees.
    :type lats:      float, int, list, numpy.array, pandas.Series
    :param lons:     Longitude in degrees.
    :type lons:      float, int, list, numpy.array, pandas.Series
    :param vector:   3D vector in Cartesian coordinates.
    :type vector:    numpy.array

    :return:         Latitudinal and longitudinal components of the vector.
    :rtype:          numpy.array, numpy.array

    NOTE: This function uses the pygplates library to convert the vector from Cartesian to magnitude, azimuth, and inclination. It could be optimised using vectorised operations, but so far it has not impacted performance in its current form.
    """
    # Convert lats and lons to numpy arrays, if not already
    lats = _numpy.array(lats); lons = _numpy.array(lons)

    # Initialize dataframes
    vector_mags = _numpy.zeros(len(lats)); vector_azis = _numpy.zeros(len(lats))
    vector_lats = _numpy.zeros(len(lats)); vector_lons = _numpy.zeros(len(lons))

    # Loop through points and convert vector to latitudinal and longitudinal components
    for i, (lat, lon) in enumerate(zip(lats, lons)):
        point = pygplates.PointOnSphere(lat, lon)
        vector_mags[i], vector_azis[i], _ = _numpy.asarray(
            pygplates.LocalCartesian.convert_from_geocentric_to_magnitude_azimuth_inclination(
                point, 
                (vector[0][i], vector[1][i], vector[2][i])
            )
        )

    # Divide by radius of Earth because pygplates.LocalCartesian assumes a unit sphere and therefore multiplies the result by the radius of the Earth.
    vector_mags /= constants.mean_Earth_radius_m; vector_azis = _numpy.rad2deg(vector_azis)
    
    # Convert to latitudinal and longitudinal components
    vector_lats, vector_lons = mag_azi2lat_lon(vector_mags, vector_azis)

    return vector_lats, vector_lons, vector_mags, vector_azis

def lat_lon2xyz(lat, lon, constants):
    """
    Convert latitude and longitude to Cartesian coordinates.

    :param lat:         Latitude in degrees.
    :type lat:          float, int, list, numpy.array, pandas.Series
    :param lon:         Longitude in degrees.
    :type lon:          float, int, list, numpy.array, pandas.Series
    :param constants:   Constants used in the calculation.
    :type constants:    class

    :return:            Position vector in Cartesian coordinates.
    :rtype:             numpy.array
    """
    # Convert to radians
    lat_rads = _numpy.deg2rad(lat)
    lon_rads = _numpy.deg2rad(lon)

    # Calculate position vectors
    position = constants.mean_Earth_radius_m * _numpy.array([_numpy.cos(lat_rads) * _numpy.cos(lon_rads), _numpy.cos(lat_rads) * _numpy.sin(lon_rads), _numpy.sin(lat_rads)])

    return position

def torques2xyz(position, lat, lon, force_lat, force_lon, segment_length_lat, segment_length_lon):
    """
    Calculate torque vectors in Cartesian coordinates.

    :param position:            Position vector in Cartesian coordinates.
    :type position:             numpy.array
    :param lat:                 Latitude in degrees.
    :type lat:                  float, int, list, numpy.array, pandas.Series
    :param lon:                 Longitude in degrees.
    :type lon: float,           int, list, numpy.array, pandas.Series
    :param force_lat:           Latitudinal component of force.
    :type force_lat:            float
    :param force_lon:           Longitudinal component of force.
    :type force_lon:            float
    :param segment_length_lat:  Length of the segment in the latitudinal direction.
    :type segment_length_lat:   float
    :param segment_length_lon:  Length of the segment in the longitudinal direction.
    :type segment_length_lon:   float

    :return:                    Torque vectors in Cartesian coordinates.
    :rtype:                     numpy.array
    """
    # Convert lon, lat to radian
    lon_rads = _numpy.deg2rad(lon)
    lat_rads = _numpy.deg2rad(lat)

    # Calculate force_magnitude
    force_magnitude = _numpy.sqrt((force_lat*segment_length_lat*segment_length_lon)**2 + (force_lon*segment_length_lat*segment_length_lon)**2)

    theta = _numpy.where(
        (force_lon >= 0) & (force_lat >= 0),                     # Condition 1
        _numpy.arctan(force_lat/force_lon),                          # Value when Condition 1 is True
        _numpy.where(
            (force_lon < 0) & (force_lat >= 0) | (force_lon < 0) & (force_lat < 0),    # Condition 2
            _numpy.pi + _numpy.arctan(force_lat/force_lon),              # Value when Condition 2 is True
            (2*_numpy.pi) + _numpy.arctan(force_lat/force_lon)           # Value when Condition 3 is True
        )
    )

    force_x = force_magnitude * _numpy.cos(theta) * (-1.0 * _numpy.sin(lon_rads))
    force_y = force_magnitude * _numpy.cos(theta) * _numpy.cos(lon_rads)
    force_z = force_magnitude * _numpy.sin(theta) * _numpy.cos(lat_rads)

    force = _numpy.array([force_x, force_y, force_z])

    # Calculate torque
    torque = _numpy.cross(position, force, axis=0)

    return torque

def mag_azi2lat_lon(magnitude, azimuth):
    """
    Decompose a vector defined by magnitude and azimuth into latitudinal and longitudinal components.

    :param magnitude:   Magnitude of vector.
    :type magnitude:    float, int, list, numpy.array, pandas.Series
    :param azimuth:     Azimuth of vector in degrees.
    :type azimuth:      float, int, list, numpy.array, pandas.Series

    :return:            Latitudinal and longitudinal components.
    :rtype:             float or numpy.array, float or numpy.array
    """
    # Convert azimuth from degrees to radians
    azimuth_rad = _numpy.deg2rad(azimuth)

    # Calculate components
    component_lat = _numpy.cos(azimuth_rad) * magnitude
    component_lon = _numpy.sin(azimuth_rad) * magnitude

    return component_lat, component_lon

def lat_lon2mag_azi(component_lat, component_lon):
    """
    Function to convert a 2D vector into magnitude and azimuth [degrees from north]

    :param component_lat:   latitudinal component of vector
    :param component_lon:   latitudinal component of vector

    :return:                magnitude, azimuth
    :rtype:                 float or numpy.array, float or numpy.array
    """
    # Calculate magnitude
    magnitude = _numpy.sqrt(component_lat**2 + component_lon**2)

    # Calculate azimuth in radians
    azimuth_rad = _numpy.arctan2(component_lon, component_lat)

    # Convert azimuth from radians to degrees
    azimuth_deg = _numpy.rad2deg(azimuth_rad)

    return magnitude, azimuth_deg

def xyz2lat_lon(position, constants):
    """
    Function to convert a 2D vector into magnitude and azimuth [degrees from north]

    :param position:    Position vector in Cartesian coordinates.
    :type position:     tuple
    :param constants:   Constants used in the calculation.
    :type constants:    class

    :return:            Latitude and longitude in degrees.
    :rtype:             float or numpy.array, float or numpy.array
    """
    # Unpack coordinates
    x, y, z = position

    # Calculate latitude (phi) and longitude (lambda) in radians
    lat_rads = _numpy.arcsin(z)
    lon_rads = _numpy.arctan2(y, x)

    # Convert to degrees
    lat = _numpy.rad2deg(lat_rads)
    lon = _numpy.rad2deg(lon_rads)

    return lat, lon

def xyz2mag(x, y, z):
    """
    Calculate the magnitude of a vector from its Cartesian components.

    :param x:   X-coordinate of the vector.
    :type x:    float or numpy.array
    :param y:   Y-coordinate of the vector.
    :type y:    float or numpy.array
    :param z:   Z-coordinate of the vector.
    :type z:    float or numpy.array

    :return:    Magnitude of the vector.
    :rtype:     float or numpy.array
    """
    return _numpy.sqrt(x**2 + y**2 + z**2)