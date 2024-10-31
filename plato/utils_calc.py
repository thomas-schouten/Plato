# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# PLATO
# Algorithm to calculate plate forces from tectonic reconstructions
# Thomas Schouten and Edward Clennett, 2024
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Import libraries
# Standard libraries
import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as _numpy
import pandas as _pandas
import pygplates
import xarray as _xarray

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
        self.mean_Earth_radius_km = 6371                            # mean Earth radius [km]
        self.mean_Earth_radius_m = 6371e3                           # mean Earth radius [m]
        self.equatorial_Earth_radius_m = 6378.1e3                   # Earth radius at equator
        self.equatorial_Earth_circumference = 40075e3               # Earth circumference at equator [m]
        
        # Conversions
        self.a2s = 365.25 * 24 * 60 * 60                           # a to s
        self.s2a = 1 / self.a2s                                     # s to a

        self.m_s2cm_a = 1e2 / self.s2a  # m/s to cm/a
        self.cm_a2m_s = 1 / self.m_s2cm_a  # cm/a to m/s

        self.rad_a2m_s = self.mean_Earth_radius_m * self.a2s  # rad/a to m/s
        self.m_s2rad_a = 1 / self.rad_a2m_s  # m/s to rad/a

        self.m_s2deg_Ma = _numpy.rad2deg(self.m_s2rad_a) * 1e6  # m/s to deg/Ma
        self.rad_a2cm_a = self.mean_Earth_radius_m * 1e2  # rad/a to cm/a

        self.deg_a2cm_a = _numpy.deg2rad(self.rad_a2cm_a) # deg/a to m/s

        self.cm2in = 0.3937008

# Create instance of mech
constants = set_constants()

def compute_slab_pull_force(
        slabs,
        options,
        mech
    ):
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
    # Calculate thicknesses
    slabs["slab_lithospheric_thickness"], slabs["slab_crustal_thickness"], slabs["slab_water_depth"] = compute_thicknesses(slabs.slab_seafloor_age, options)

    # Calculate length of slab
    slabs["slab_length"] = options["Slab length"]

    # Calculate slab pull force acting on point along subduction zone where there is a seafloor age, and set to 0 where there is no seafloor age
    mask = slabs["slab_seafloor_age"].isna()
    slabs.loc[mask, "slab_pull_force_mag"] = 0
    slabs.loc[~mask, "slab_pull_force_mag"] = (
        slabs.loc[~mask, "slab_lithospheric_thickness"] * slabs.loc[~mask, "slab_length"] * mech.drho_slab * mech.g * 1/_numpy.sqrt(_numpy.pi)
    )

    if options["Sediment subduction"]:
        slabs.loc[~mask, "slab_pull_force_mag"] += (
            slabs.loc[~mask, "sediment_thickness"] * slabs.loc[~mask, "slab_length"] * mech.drho_sed * mech.g * 1/_numpy.sqrt(_numpy.pi)
        )

    # Decompose into latitudinal and longitudinal components
    slabs["slab_pull_force_lat"], slabs["slab_pull_force_lon"] = mag_azi2lat_lon(slabs["slab_pull_force_mag"], slabs["trench_normal_azimuth"])

    return slabs

def compute_interface_term(slabs, options):
    """
    Function to calculate the interface term that accounts for resisting forces at the subduction interface.
    These forces are i) shearing along the plate interface, ii) bending of the slab, and iii) vertical resistance to slab sinking.
    """
    # Calculate the interface term as a function of the sediment fraction, if enabled
    if options["Sediment subduction"]:
        # Determine shear zone width
        if options["Shear zone width"] == "variable":
            slabs["shear_zone_width"] = slabs["v_convergence_mag"] * constants.cm_a2m_s / options["Strain rate"]
        else:
            slabs["shear_zone_width"] = options["Shear zone width"]

        # Calculate sediment fraction using sediment thickness and shear zone width
        # Step 1: Calculate sediment_fraction based on conditions
        slabs["sediment_fraction"] = _numpy.where(
            slabs["slab_seafloor_age"].isna(), 
            0, 
            slabs["sediment_thickness"].fillna(0) / slabs["shear_zone_width"]
        )

        # Step 2: Cap values at 1 (ensure fraction does not exceed 1)
        slabs["sediment_fraction"] = slabs["sediment_fraction"].clip(upper=1)

        # Step 3: Replace NaNs with 0 (if needed)
        slabs["sediment_fraction"] = slabs["sediment_fraction"].fillna(0)
            
        # Calculate interface term
        interface_term = 11 - 10**(1-slabs["sediment_fraction"])
        logging.info(f"Mean, min and max of interface terms: {interface_term.mean()}, {interface_term.min()}, {interface_term.max()}")
    else:
        interface_term = 1

    interface_term *= options["Slab pull constant"]

    # Apply interface term to slab pull force
    slabs["slab_pull_force_mag"] *= interface_term
    slabs["slab_pull_force_lat"] *= interface_term
    slabs["slab_pull_force_lon"] *= interface_term

    return slabs

def compute_slab_bend_force(slabs, options, mech, constants):
    """
    Function to calculate the slab bending force.
    """
    # Calculate slab bending torque
    if options["Bending mechanism"] == "viscous":
        bending_force = (-2. / 3.) * ((slabs.lower_plate_thickness) / (mech.rad_curv)) ** 3 * mech.lith_visc * slabs.v_convergence * constants.cm_a2m_s # [n-s , e-w], [N/m]
    elif options["Bending mechanism"] == "plastic":
        bending_force = (-1. / 6.) * ((slabs.lower_plate_thickness ** 2) / mech.rad_curv) * mech.yield_stress * _numpy.asarray(
            (_numpy.cos(slabs.trench_normal_vector + slabs.obliquity_convergence), _numpy.sin(slabs.trench_normal_vector + slabs.obliquity_convergence))
        )  # [n-s, e-w], [N/m]
        
    slabs["bend_force_lat"], slabs["bend_force_lon"] = mag_azi2lat_lon(bending_force, slabs.trench_normal_vector + slabs.obliquity_convergence)
    
    return slabs

def sample_slabs_from_seafloor(
        lat,
        lon,
        trench_normal_azimuth, 
        seafloor, 
        options, 
        plate, 
        age_variable="seafloor_age", 
        coords=["lat", "lon"], 
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
        current_sampling_distance = initial_sampling_distance + 100
        iterations = 4

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

def compute_GPE_force(
        points: _pandas.DataFrame,
        seafloor_grid: _xarray.DataArray,
        options: Dict,
        mech: Dict,
    ):
    """
    Function to calculate GPE force at points.
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

        ages[i] = sample_grid(sampling_lat, sampling_lon, seafloor_grid)
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

    points["GPE_force_mag"] = _numpy.linalg.norm([points["GPE_force_lat"].values, points["GPE_force_lon"].values], axis=0)

    return points

def sample_grid(
        lat: _numpy.ndarray,
        lon: _numpy.ndarray,
        grid: _xarray.Dataset,
        coords = ["lat", "lon"],
    ):
    """
    Function to sample a grid
    """
    # Load grid into memory to decrease computation time
    grid = grid.load()

    # Extract latitude and longitude values from points and convert to xarray DataArrays
    lat_da = _xarray.DataArray(lat, dims="point")
    lon_da = _xarray.DataArray(lon, dims="point")

    # Interpolate age value at point
    sampled_values = _numpy.asarray(grid.interp({coords[0]: lat_da, coords[1]: lon_da}, method="nearest").values.tolist())

    # Close the grid to free memory space
    grid.close()

    return sampled_values

def compute_mantle_drag_force(
        points: _pandas.DataFrame,
        options: Dict, 
        constants: Dict, 
    ):
    """
    Function to calculate mantle drag force at points
    """
    # Get velocities at points
    if options["Reconstructed motions"]:
        # Calculate mantle drag force
        points["mantle_drag_force_lat"] = -1 * points["velocity_lat"] * constants.cm_a2m_s * options["Mantle viscosity"] / mech.La
        points["mantle_drag_force_lon"] = -1 * points["velocity_lon"] * constants.cm_a2m_s * options["Mantle viscosity"] / mech.La
        points["mantle_drag_force_mag"] = _numpy.linalg.norm([points["mantle_drag_force_lat"], points["mantle_drag_force_lon"]], axis=0)

    return points

def compute_synthetic_stage_rotation(
        plates: _pandas.DataFrame,
        options: Dict,
    ) -> _pandas.DataFrame:
    """
    Function to compute stage rotations.
    """
    # logging.info(f"Mean, min and max of reconstructed stage rotation angles: {plates.pole_angle.mean()}, {plates.pole_angle.min()}, {plates.pole_angle.max()}")

    # Sum the torque vectors (in Cartesian coordinates and in Newton metres)
    mantle_drag_torque_xyz = _numpy.column_stack((plates.mantle_drag_torque_x, plates.mantle_drag_torque_y, plates.mantle_drag_torque_z))

    # Get rotation vector in radians per year by dividing by flipping the sign of the mantle drag torque and dividing by the area of the plate and the drag coefficient (i.e. mantle viscosity / asthenosphere thickness)
    stage_rotations_xyz = -1 * mantle_drag_torque_xyz 

    # Get the rotation poles in spherical coordinates
    stage_rotation_poles_lat, stage_rotation_poles_lon, stage_rotation_poles_mag, _ = geocentric_cartesian2spherical(
        stage_rotations_xyz[:, 0], stage_rotations_xyz[:, 1], stage_rotations_xyz[:, 2]
    )

    # Convert any NaN values to 0
    stage_rotation_poles_mag = _numpy.nan_to_num(stage_rotation_poles_mag)

    # Normalise the rotation poles by the drag coefficient and the square of the Earth's radius
    stage_rotation_poles_mag /= options["Mantle viscosity"] / mech.La * constants.mean_Earth_radius_m**2

    # Convert to degrees because the 'geocentric_cartesian2spherical' does not convert the magnitude to degrees
    stage_rotation_poles_mag = _numpy.rad2deg(stage_rotation_poles_mag)
    
    # Assign to DataFrame
    plates["pole_lat"] = stage_rotation_poles_lat
    plates["pole_lon"] = stage_rotation_poles_lon 
    plates["pole_angle"] = stage_rotation_poles_mag

    return plates
    
def compute_velocity(
        point_data: _pandas.DataFrame,
        plate_data: _pandas.DataFrame,
        constants,
        plateID_col: str = "plateID",
    ) -> Tuple[_numpy.ndarray, _numpy.ndarray, _numpy.ndarray, _numpy.ndarray, _numpy.ndarray]:
    """
    Function to compute lat, lon, magnitude and azimuth of velocity at a set of locations from a Cartesian torque vector.
    """
    # Initialise arrays to store velocities
    v_lats = _numpy.zeros_like(point_data.lat); v_lons = _numpy.zeros_like(point_data.lat)
    v_mags = _numpy.zeros_like(point_data.lat); v_azis = _numpy.zeros_like(point_data.lat)
    spin_rates = _numpy.zeros_like(point_data.lat)

    # Loop through plates more efficiently
    for _, plate in plate_data.iterrows():
        # Mask points belonging to the current plate
        mask = point_data[plateID_col] == plate.plateID

        # Calculate position vectors in Cartesian coordinates (bulk operation) on the unit sphere (i.e. in radians)
        # The shape of the position vectors is (n, 3)
        positions_x, positions_y, positions_z = geocentric_spherical2cartesian(
            point_data[mask].lat, 
            point_data[mask].lon,
        )
        positions_xyz = _numpy.column_stack((positions_x, positions_y, positions_z))

        # Calculate rotation pole in radians per year in Cartesian coordinates
        # The shape of the rotation pole vector is (3,) and the rotation pole is stored in the DataFrame in degrees per million years
        rotation_pole_xyz = _numpy.array(geocentric_spherical2cartesian(
            plate.pole_lat, 
            plate.pole_lon, 
            plate.pole_angle * 1e-6,
        ))

        # Calculate the velocity in degrees per year as the cross product of the rotation and the position vectors
        # The shape of the velocity vectors is (n, 3)
        velocities_xyz = _numpy.cross(rotation_pole_xyz[None, :], positions_xyz)

        # Convert velocity components to latitudinal and longitudinal components
        v_lats[mask], v_lons[mask], v_mags[mask], v_azis[mask] = tangent_cartesian2spherical(
            velocities_xyz,
            point_data[mask].lat.values,
            point_data[mask].lon.values,
        )
        
        # Convert velocity components to cm/a
        v_mags[mask] *= constants.deg_a2cm_a
        v_lats[mask] *= constants.deg_a2cm_a
        v_lons[mask] *= constants.deg_a2cm_a

        # Calculate the spin rate in degrees per million years as the dot product of the velocity and the unit position vector
        spin_rates[mask] = (positions_xyz[:,0] * rotation_pole_xyz[0] + positions_xyz[:,1] * rotation_pole_xyz[1] + positions_xyz[:,2] * rotation_pole_xyz[2]) * 1e6
        
    return v_lats, v_lons, v_mags, v_azis, spin_rates

def compute_rms_velocity(
        segment_length_lat: Union[_numpy.ndarray, _pandas.Series],
        segment_length_lon: Union[_numpy.ndarray, _pandas.Series],
        v_mag: Union[_numpy.ndarray, _pandas.Series],
        v_azi: Union[_numpy.ndarray, _pandas.Series],
        omega: Union[_numpy.ndarray, _pandas.Series],
    ) -> Tuple[float, float, float]:
    """
    Function to calculate area-weighted root mean square (RMS) velocity for a given plate.
    """
    # Precompute segment areas to avoid repeated calculation
    segment_areas = segment_length_lat * segment_length_lon
    total_area = _numpy.sum(segment_areas)

    # Convert azimuth to radians
    v_azi = _numpy.deg2rad(v_azi)

    # Calculate RMS velocity magnitude
    v_rms_mag = _numpy.sum(v_mag * segment_areas) / total_area

    # Calculate RMS velocity azimuth (in radians)
    sin_azi = _numpy.sum(_numpy.sin(v_azi) * segment_areas) / total_area
    cos_azi = _numpy.sum(_numpy.cos(v_azi) * segment_areas) / total_area

    v_rms_azi = _numpy.rad2deg(
        -1 * (_numpy.arctan2(sin_azi, cos_azi) + 0.5 * _numpy.pi)
    )
    # Ensure azimuth is within the range [0, 360]
    v_rms_azi = _numpy.where(v_rms_azi < 0, v_rms_azi + 360, v_rms_azi)
    
    # Calculate spin rate
    omega_rms = _numpy.sum(omega * segment_areas) / total_area

    return v_rms_mag, v_rms_azi, omega_rms

def compute_net_rotation(
        plate_data: _pandas.DataFrame,
        point_data: _pandas.DataFrame,
    ):
    """
    Function to calculate net rotation of the entire lithosphere.
    """
    # Initialise array to store net rotation vector
    net_rotation_xyz = _numpy.zeros(3)

    # Loop through plates more efficiently
    for _, plate in plate_data.iterrows():
        # Select points belonging to the current plate
        selected_points = point_data[point_data.plateID == plate.plateID]

        # Calculate position vectors in Cartesian coordinates (bulk operation) on the unit sphere
        # The shape of the position vectors is (n, 3)
        positions_x, positions_y, positions_z = geocentric_spherical2cartesian(
            selected_points.lat, 
            selected_points.lon, 
        )
        positions_xyz = _numpy.column_stack((positions_x, positions_y, positions_z))

        # Calculate rotation pole in Cartesian coordinates
        # The shape of the rotation pole vector is (3,)
        rotation_pole_xyz = _numpy.array(geocentric_spherical2cartesian(
            plate.pole_lat, 
            plate.pole_lon, 
            plate.pole_angle,
        ))

        # Calculate the double cross product of the position vector and the velocity vector (see Torsvik et al. (2010), https://doi.org/10.1016/j.epsl.2009.12.055)
        # The shape of the rotation pole vector is modified to (1, 3) to allow broadcasting
        point_rotations_xyz = _numpy.cross(_numpy.cross(rotation_pole_xyz[None, :], positions_xyz), positions_xyz)

        # Weight the rotations by segment area (broadcasted multiplication)
        segment_area = (selected_points.segment_length_lat * selected_points.segment_length_lon).values[:, None]
        point_rotations_xyz *= segment_area

        # Accumulate the net rotation vector by summing across all points
        net_rotation_xyz += point_rotations_xyz.sum(axis=0)

    # Normalise the net rotation vector by the total area of the lithosphere
    net_rotation_xyz /= plate_data.area.sum()

    # Convert the net rotation vector to spherical coordinates
    net_rotation_pole_lat, net_rotation_pole_lon, _, _ = geocentric_cartesian2spherical(
        net_rotation_xyz[0], net_rotation_xyz[1], net_rotation_xyz[2],
    )

    # Calculate the magnitude of the net rotation vector
    net_rotation_rate = _numpy.linalg.norm(net_rotation_xyz)

    return net_rotation_pole_lat, net_rotation_pole_lon, net_rotation_rate

def sum_torque(
        plates: _pandas.DataFrame,
        torque_type: str,
        constants: Dict,
    ):
    """
    Function to calculate driving and residual torque on plates.
    """
    # Determine torque components based on torque type
    if torque_type == "driving" or torque_type == "mantle_drag":
        torque_components = ["slab_pull_torque", "GPE_torque"]
    elif torque_type == "mantle_drag":
        torque_components = ["slab_pull_torque", "GPE_torque", "slab_bend_torque"]
    elif torque_type == "residual":
        torque_components = ["slab_pull_torque", "GPE_torque", "slab_bend_torque", "mantle_drag_torque"]
    else:
        raise ValueError("Invalid torque_type, must be 'driving' or 'residual' or 'mantle_drag'!")

    # Calculate torque in Cartesian coordinates
    for axis in ["_x", "_y", "_z"]:
        plates[f"{torque_type}_torque{axis}"] = _numpy.sum(
            [_numpy.nan_to_num(plates[component + axis]) for component in torque_components], axis=0
        )
    
    if torque_type == "mantle_drag":
        for axis in ["_x", "_y", "_z"]:
            torque_values = plates[f"{torque_type}_torque{axis}"].values
            if not _numpy.allclose(torque_values, 0):  # Only flip if non-zero
                plates[f"{torque_type}_torque{axis}"] *= -1
    
    # Organise torque in an array
    summed_torques_cartesian = _numpy.asarray([
        plates[f"{torque_type}_torque_x"], 
        plates[f"{torque_type}_torque_y"], 
        plates[f"{torque_type}_torque_z"]
    ])

    # Calculate torque magnitude
    plates[f"{torque_type}_torque_mag"] = _numpy.linalg.norm(summed_torques_cartesian, axis=0)

    # Calculate the position vector of the centroid of the plate in Cartesian coordinates
    centroid_position = geocentric_spherical2cartesian(plates.centroid_lat, plates.centroid_lon, constants.mean_Earth_radius_m)

    # Calculate the torque vector as the cross product of the Cartesian torque vector (x, y, z) with the position vector of the centroid
    force_at_centroid = _numpy.cross(summed_torques_cartesian, centroid_position, axis=0)

    # Compute force magnitude at centroid
    plates[f"{torque_type}_force_lat"], plates[f"{torque_type}_force_lon"], plates[f"{torque_type}_force_mag"], plates[f"{torque_type}_force_azi"] = geocentric_cartesian2spherical(
        force_at_centroid[0], force_at_centroid[1], force_at_centroid[2]
    )

    return plates

def compute_residual_force(
        point_data: _pandas.DataFrame,
        plate_data: _pandas.DataFrame,
        plateID_col: str = "plateID",
        weight_col: str = "segment_area",
    ) -> _pandas.DataFrame:
    """
    Function to calculate residual torque along trench.
    """
    # Initialise arrays to store velocities
    F_lats = _numpy.zeros_like(point_data.lat); F_lons = _numpy.zeros_like(point_data.lat)
    F_mags = _numpy.zeros_like(point_data.lat); F_azis = _numpy.zeros_like(point_data.lat)

    # Loop through plates more efficiently
    for _, plate in plate_data.iterrows():
        # Mask points belonging to the current plate
        mask = point_data[plateID_col] == plate.plateID

        # Calculate position vectors in Cartesian coordinates (bulk operation) on the unit sphere (i.e. in radians)
        # The shape of the position vectors is (n, 3)
        positions_x, positions_y, positions_z = geocentric_spherical2cartesian(
            point_data[mask].lat, 
            point_data[mask].lon,
        )
        positions_xyz = _numpy.column_stack((positions_x, positions_y, positions_z))

        # Get the torque vector in Cartesian coordinates
        # The shape of the torque vector is (3,) and the torque vector is stored in the DataFrame in Nm
        torques_xyz = _numpy.array([
            plate.residual_torque_x, 
            plate.residual_torque_y, 
            plate.residual_torque_z, 
        ])

        # Calculate the force in N as the cross product of the rotation and the position vectors
        # The shape of the velocity vectors is (n, 3)
        forces_xyz = _numpy.cross(torques_xyz[None, :], positions_xyz)

        # Convert velocity components to latitudinal and longitudinal components
        F_lats[mask], F_lons[mask], F_mags[mask], F_azis[mask] = tangent_cartesian2spherical(
            forces_xyz,
            point_data[mask].lat.values,
            point_data[mask].lon.values,
        )
        
        # Normalise the force components by the weight of the segment
        F_mags[mask] /= point_data[mask][weight_col].values * constants.mean_Earth_radius_m**2
        F_lats[mask] /= point_data[mask][weight_col].values * constants.mean_Earth_radius_m**2
        F_lons[mask] /= point_data[mask][weight_col].values * constants.mean_Earth_radius_m**2

    return F_lats, F_lons, F_mags, F_azis

def optimise_torques(
        plates: _pandas.DataFrame,
        mech: Dict,
        options: Dict,
    ) -> _pandas.DataFrame:
    """
    Function to optimise torques.
    """
    for axis in ["_x", "_y", "_z", "_mag"]:
        plates["slab_pull_torque_opt" + axis] = plates["slab_pull_torque" + axis].values * options["Slab pull constant"]
        plates["mantle_drag_torque_opt" + axis] = plates["mantle_drag_torque" + axis].values * options["Mantle viscosity"] / mech.La

    return plates

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# GENERAL FUNCTIONS
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def compute_thicknesses(seafloor_ages, options, crust=True, water=True):
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
        lithospheric_mantle_thickness = _numpy.where(_numpy.isnan(seafloor_ages), 
                                                mech.cont_lith_thick, 
                                                2.32 * _numpy.sqrt(mech.kappa * seafloor_ages * constants.a2s * 1e6))
        
        if crust:
            crustal_thickness = _numpy.where(_numpy.isnan(seafloor_ages), 
                                        mech.cont_crust_thick, 
                                        mech.ocean_crust_thick)
        else:
            crustal_thickness = _numpy.nan
            
        if water:
            water_depth = _numpy.where(_numpy.isnan(seafloor_ages), 
                                0.,
                                (lithospheric_mantle_thickness * ((mech.rho_a - mech.rho_l) / (mech.rho_sw - mech.rho_a))) + 2600)
        else:
            water_depth = _numpy.nan
        
    # Water depth from half space cooling and lithospheric thickness from isostasy
    elif options["Seafloor age profile"] == "plate model":
        hw = _numpy.where(seafloor_ages > 81, 6586 - 3200 * _numpy.exp((-seafloor_ages / 62.8)), seafloor_ages)
        hw = _numpy.where(hw <= 81, 2600 + 345 * _numpy.sqrt(hw), hw)
        lithospheric_mantle_thickness = (hw - 2600) * ((mech.rho_sw - mech.rho_a) / (mech.rho_a - mech.rho_l))

        if crust:
            crustal_thickness = _numpy.where(_numpy.isnan(seafloor_ages), 
                                        mech.cont_crust_thick, 
                                        mech.ocean_crust_thick)
        else:
            crustal_thickness = _numpy.nan
        
        if water:
            water_depth = hw
        else:
            water_depth = _numpy.nan

    return lithospheric_mantle_thickness, crustal_thickness, water_depth

def compute_torque_on_plates(
        plate_data,
        lats,
        lons,
        plateIDs,
        forces_lat,
        forces_lon,
        areas,
        constants,
        torque_var="torque",
    ):
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
    :param torque_var:     Name of the torque variable. Defaults to "torque".
    :type torque_var:      str

    :return: Updated torques DataFrame with added columns for torque components at the centroid, force components at the centroid, and latitudinal and longitudinal components of the force.
    :rtype: pd.DataFrame

    This function calculates torques in Cartesian coordinates based on latitudinal, longitudinal forces, and segment dimensions.
    It then sums the torque components for each plate, calculates the torque vector at the centroid, and updates the torques DataFrame.
    Finally, it calculates the force components at the centroid, converts them to latitudinal and longitudinal components, and adds these to the torques DataFrame.
    """
    # Initialise dataframes and sort plateIDs
    point_data = _pandas.DataFrame({"plateID": plateIDs})

    # Convert points to Cartesian coordinates
    positions_xyz = geocentric_spherical2cartesian(lats, lons, constants.mean_Earth_radius_m)
    
    # Calculate torques in Cartesian coordinates
    torques_xyz = forces2torques(positions_xyz, lats, lons, forces_lat, forces_lon, areas)
    
    # Assign the calculated torques to the new torque_var columns
    point_data[torque_var + "_torque_x"] = torques_xyz[0]
    point_data[torque_var + "_torque_y"] = torques_xyz[1]
    point_data[torque_var + "_torque_z"] = torques_xyz[2]
    point_data[torque_var + "_torque_mag"] = _numpy.linalg.norm(_numpy.array([torques_xyz[0], torques_xyz[1], torques_xyz[2]]))

    # Sum components of plates based on plateID and fill NaN values with 0
    summed_data = point_data.groupby("plateID", as_index=False).sum().fillna(0)

    # Sort by plateID
    summed_data.sort_values("plateID", inplace=True)

    # Set indices of plateId for both dataframes but keep a copy of the old index
    old_index = plate_data.index
    plate_data.set_index("plateID", inplace=True)

    # Update the plate data with the summed torque components
    plate_data.update(summed_data.set_index("plateID"))

    # Reset the index of the plate data while keeping the old index
    plate_data.reset_index(drop=False, inplace=True)

    # Restore the old index
    plate_data.index = old_index

    # Calculate the position vector of the centroid of the plate in Cartesian coordinates
    centroid_position_xyz = geocentric_spherical2cartesian(plate_data.centroid_lat, plate_data.centroid_lon, constants.mean_Earth_radius_m)

    # Calculate the torque vector as the cross product of the Cartesian torque vector (x, y, z) with the position vector of the centroid
    summed_torques_xyz = _numpy.asarray([
        plate_data[f"{torque_var}_torque_x"], plate_data[f"{torque_var}_torque_y"], plate_data[f"{torque_var}_torque_z"]
    ])
    centroid_force_xyz = _numpy.cross(summed_torques_xyz, centroid_position_xyz, axis=0)

    # Compute force magnitude at centroid
    centroid_force_sph = geocentric_cartesian2spherical(centroid_force_xyz[0], centroid_force_xyz[1], centroid_force_xyz[2])

    # Store values in the torques DataFrame
    plate_data[f"{torque_var}_force_lat"] = centroid_force_sph[0]
    plate_data[f"{torque_var}_force_lon"] = centroid_force_sph[1]
    plate_data[f"{torque_var}_force_mag"] = centroid_force_sph[2]
    plate_data[f"{torque_var}_force_azi"] = centroid_force_sph[3]
    
    return plate_data

def compute_subduction_flux(
        plates,
        slabs,
        type: str = "slab" or "sediment",
    ):
    """
    Function to calculate subduction flux at trench points.

    :param plates:                  plate data
    :type plates:                   pandas.DataFrame
    :param slabs:                   slab data
    :type slabs:                    pandas.DataFrame
    :param type:                    type of subduction flux to calculate
    :type type:                     str

    :return:                        plates
    :rtype:                         pandas.DataFrame
    """
    # Calculate subduction flux
    for plateID in plates.plateID.values:
        selected_slabs = slabs[slabs.lower_plateID == plateID]
        if type == "slab":
            plates.loc[plates.plateID == plateID, "slab_flux"] = (selected_slabs.lower_plate_thickness * selected_slabs.v_lower_plate_mag * selected_slabs.trench_segment_length).sum()
        
        elif type == "sediment":
            plates.loc[plates.plateID == plateID, "sediment_flux"] = (selected_slabs.sediment_thickness * selected_slabs.v_lower_plate_mag * selected_slabs.trench_segment_length).sum()

    return plates

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# CONVERSIONS
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def geocentric_cartesian2spherical(
        x: _numpy.ndarray,
        y: Optional[_numpy.ndarray] = None,
        z: Optional[_numpy.ndarray] = None,
    ):
    """
    Convert Cartesian coordinates to latitude, longitude, magnitude, and azimuth.
    """
    # If only x is provided as a 1D array, unpack it into x, y, z
    if y is None and z is None:
        x, y, z = x[0], x[1], x[2]

    # Convert integers and floats to lists
    if isinstance(x, (int, float, _numpy.integer, _numpy.floating)):
        x = [x]
    if isinstance(y, (int, float, _numpy.integer, _numpy.floating)):
        y = [y]
    if isinstance(z, (int, float, _numpy.integer, _numpy.floating)):
        z = [z]

    # Make sure x, y, z are numpy arrays
    x = _numpy.asarray(x)
    y = _numpy.asarray(y)
    z = _numpy.asarray(z)

    # Stack coordinates to handle multiple points
    coords = _numpy.column_stack((x, y, z))

    # Calculate magnitude (norm)
    mags = _numpy.linalg.norm(coords, axis=1)

    # Mask for zero or NaN magnitudes
    valid_mask = (mags > 0) & (~_numpy.isnan(mags))

    # Initialise result arrays
    lats = _numpy.zeros_like(mags)
    lons = _numpy.zeros_like(mags)
    azis = _numpy.zeros_like(mags)

    # Calculate latitude (in degrees)
    lats[valid_mask] = _numpy.rad2deg(_numpy.arcsin(z[valid_mask] / mags[valid_mask]))

    # Calculate longitude (in degrees)
    lons[valid_mask] = _numpy.rad2deg(_numpy.arctan2(y[valid_mask], x[valid_mask]))

    # Calculate azimuth (in degrees, measured from North in XY plane)
    azis[valid_mask] = _numpy.rad2deg(_numpy.arctan2(x[valid_mask], y[valid_mask]))

    return lats, lons, mags, azis

def geocentric_spherical2cartesian(
        lat,
        lon,
        mag = 1):
    """
    Convert latitude and longitude to Cartesian coordinates.
    """
    # Ensure inputs are NumPy arrays for vectorized operations
    lats = _numpy.asarray(lat)
    lons = _numpy.asarray(lon)
    mags = _numpy.asarray(mag)

    # Convert to radians
    lats_rad = _numpy.deg2rad(lat)
    lons_rad = _numpy.deg2rad(lon)

    # Calculate x, y, z
    x = mag * _numpy.cos(lats_rad) * _numpy.cos(lons_rad)
    y = mag * _numpy.cos(lats_rad) * _numpy.sin(lons_rad)
    z = mag * _numpy.sin(lats_rad)

    return x, y, z

def tangent_cartesian2spherical(
        vectors_xyz: _numpy.ndarray,
        points_lat: _numpy.ndarray,
        points_lon: _numpy.ndarray,
    ):
    """
    Convert a vector that is tangent to the surface of a sphere to spherical coordinates.
    """
    # Initialise result arrays
    vectors_mag = _numpy.zeros_like(points_lat)
    vectors_azi = _numpy.zeros_like(points_lat)

    # Loop through points and convert vector to latitudinal and longitudinal components
    for i, (point_lat, point_lon) in enumerate(zip(points_lat, points_lon)):
        # Make PointonSphere
        point = pygplates.PointOnSphere(point_lat, point_lon)

        # Convert vector to magnitude, azimuth, and inclination
        vectors_mag[i], vectors_azi[i], _ = _numpy.asarray(
            pygplates.LocalCartesian.convert_from_geocentric_to_magnitude_azimuth_inclination(
                point, 
                (vectors_xyz[i,0], vectors_xyz[i,1], vectors_xyz[i,2])
            )
        )
    
    # Convert azimuth from radians to degrees
    vectors_azi = _numpy.rad2deg(vectors_azi)
    
    # Convert to latitudinal and longitudinal components
    vectors_lat, vectors_lon = mag_azi2lat_lon(vectors_mag, vectors_azi)

    return vectors_lat, vectors_lon, vectors_mag, vectors_azi

def forces2torques(
        positions_xyz, 
        lats, 
        lons, 
        forces_lat, 
        forces_lon, 
        areas,
    ):
    """
    Calculate plate torque vector from force vectors.

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
    lons_rad = _numpy.deg2rad(lons)
    lats_rad = _numpy.deg2rad(lats)

    # Calculate force_magnitude
    forces_mag = _numpy.linalg.norm([forces_lat*areas, forces_lon*areas], axis=0)

    # Calculate theta
    theta = _numpy.empty_like(forces_lon)
    mask = ~_numpy.logical_or(forces_lon == 0, _numpy.isnan(forces_lon), _numpy.isnan(forces_lat))
    theta[mask] = _numpy.where(
        (forces_lon[mask] > 0) & (forces_lat[mask] >= 0),  
        _numpy.arctan(forces_lat[mask] / forces_lon[mask]),                          
        _numpy.where(
            (forces_lon[mask] < 0) & (forces_lat[mask] >= 0) | (forces_lon[mask] < 0) & (forces_lat[mask] < 0),    
            _numpy.pi + _numpy.arctan(forces_lat[mask] / forces_lon[mask]),              
            (2*_numpy.pi) + _numpy.arctan(forces_lat[mask] / forces_lon[mask])           
        )
    )

    # Calculate force in Cartesian coordinates
    forces_x = forces_mag * _numpy.cos(theta) * (-1.0 * _numpy.sin(lons_rad))
    forces_y = forces_mag * _numpy.cos(theta) * _numpy.cos(lons_rad)
    forces_z = forces_mag * _numpy.sin(theta) * _numpy.cos(lats_rad)
    forces_xyz = _numpy.asarray([forces_x, forces_y, forces_z])

    # Calculate torque
    torques = _numpy.cross(positions_xyz, forces_xyz, axis=0)

    return torques   

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
    magnitude = _numpy.linalg.norm([component_lat, component_lon**2], axis=0)

    # Calculate azimuth in radians
    azimuth_rad = _numpy.arctan2(component_lon, component_lat)

    # Convert azimuth from radians to degrees
    azimuth_deg = _numpy.rad2deg(azimuth_rad)

    return magnitude, azimuth_deg

def rotate_torque(plateID, torque, rotations_a, rotations_b, reconstruction_time, constants):
    """
    Function to rotate a torque vector in Cartesian coordinates between two reference frames.

    :param plateID:             PlateID for which the torque is rotated.
    :type plateID:              int, float
    :param torque:              Torque vector in Cartesian coordinates.
    :type torque:               numpy.array of length 3
    :param rotations_a:         Rotation model A.
    :type rotations_a:          numpy.array
    :param rotations_b:         Rotation model B.
    :type rotations_b:          numpy.array
    :param reconstruction_time: Time of reconstruction.
    :type reconstruction_time:  float
    
    :return:                    Rotated torque vector in Cartesian coordinates.
    :rtype:                     numpy.array
    """
    # Get equivalent total rotations for the plateID in both rotation models
    relative_rotation_pole = get_relative_rotaton_pole(plateID, rotations_a, rotations_b, reconstruction_time)

    # Rotate torque vector
    rotated_torque = rotate_vector(torque, relative_rotation_pole, constants)

    return rotated_torque

def get_relative_rotaton_pole(plateID, rotations_a, rotations_b, reconstruction_time):
    """
    Function to get the relative rotation pole between two reference frames for any plateID.

    :param plateID:         PlateID for which the relative rotation pole is calculated.
    :type plateID:          int, float
    :param rotations_a:     Rotation model A.
    :type rotations_a:      numpy.array
    :param rotations_b:     Rotation model B.
    :type rotations_b:      numpy.array
    """
    # Make sure the plateID is an integer
    plateID = int(plateID)

    # Make sure the reconstruction time is an integer
    reconstruction_time = int(reconstruction_time)

    # Get equivalent total rotations for the plateID in both rotation models
    rotation_a = rotations_a.get_rotation(
        to_time=reconstruction_time,
        moving_plate_id=plateID,
    )
    rotation_b = rotations_b.get_rotation(
        to_time=reconstruction_time,
        moving_plate_id=plateID,
    )

    # Calculate relative rotation pole
    relative_rotation_pole = rotation_a * rotation_b.get_inverse()

    return relative_rotation_pole.get_lat_lon_euler_pole_and_angle_degrees()

def rotate_vector(vector, rotation):
    """
    Function to rotate a vector in Cartesian coordinates with a given Euler rotation.

    :param vector:      Vector in 3D Cartesian coordinates.
    :type vector:       numpy.array of length 3
    :param rotation:    Rotation pole latitude, rotation pole longitude, and rotation angle in degrees.
    :type rotation:     numpy.array of length 3

    :return:            Rotated vector in Cartesian coordinates.
    :rtype:             numpy.array
    """
    # Convert rotation axis to Cartesian coordinates
    rotation_axis = geocentric_spherical2cartesian(rotation[0], rotation[1], 1)

    # Calculate Euler parameters
    a = _numpy.cos(_numpy.deg2rad(rotation[2]) / 2)
    b = rotation_axis[0] * _numpy.sin(_numpy.deg2rad(rotation[2]) / 2)
    c = rotation_axis[1] * _numpy.sin(_numpy.deg2rad(rotation[2]) / 2)
    d = rotation_axis[2] * _numpy.sin(_numpy.deg2rad(rotation[2]) / 2)

    # Check if squares of Euler parameters is 1
    if not _numpy.isclose(a**2 + b**2 + c**2 + d**2, 1):
        raise ValueError("Euler parameters do not sum to 1")
    
    # Calculate rotation matrix
    rotation_matrix = _numpy.asarray([
        [a**2 + b**2 - c**2 - d**2, 2 * (b * c - a * d), 2 * (a * c + b * d)],
        [2 * (b * c + a * d), a**2 - b**2 + c**2 - d**2, 2 * (c * d - a * b)],
        [2 * (b * d - a * c), 2 * (a * b + c * d), a**2 - b**2 - c**2 + d**2]
    ])

    # Rotate vector
    rotated_vector = _numpy.dot(rotation_matrix, vector.values.T)

    return rotated_vector.T