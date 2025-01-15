# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# PLATO
# Algorithm to calculate plate forces from tectonic reconstructions
# Thomas Schouten and Edward Clennett, 2021-2024
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Import libraries
# Standard libraries
import logging
from typing import Dict, List, Optional, Tuple, Union, Any

# Third-party libraries
import numpy as _numpy
import pandas as _pandas
import pygplates as _pygplates
import xarray as _xarray

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# VALUES
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
        self.za = 300e3                                     # maximum depth of asthenosphere [m]
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
        self.cm_a2deg_a = 1 / self.deg_a2cm_a # cm/a to deg/a

        self.cm2in = 0.3937008

# Create instance of constants
constants = set_constants()

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# FORCE CALCULATIONS
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def compute_slab_pull_force(
        slab_data: _pandas.DataFrame,
        options: Dict[str, Any],
        mech, # Is this really needed?
    ) -> _pandas.DataFrame:
    """
    Function to calculate slab pull force along subduction zones.

    :param slab_data:   subduction zone data
    :type slab_data:    pandas.DataFrame
    :param options:     options
    :type options:      dict
    :param mech:        mechanical parameters used in calculations
    :type mech:         class

    :return:            slabs
    :rtype:             pandas.DataFrame
    """
    # Calculate thicknesses
    slab_data["slab_lithospheric_thickness"], slab_data["slab_crustal_thickness"], slab_data["slab_water_depth"] = compute_thicknesses(slab_data.slab_seafloor_age, options)

    # Calculate length of slab
    # TODO: Implement variable slab length based on some proxy?
    slab_data["slab_length"] = options["Slab length"]

    # Calculate slab pull force acting on point along subduction zone where there is a seafloor age, and set to 0 where there is no seafloor age
    mask = slab_data["slab_seafloor_age"].isna()
    slab_data.loc[mask, "slab_pull_force_mag"] = 0
    slab_data.loc[~mask, "slab_pull_force_mag"] = (
        slab_data.loc[~mask, "slab_lithospheric_thickness"] * slab_data.loc[~mask, "slab_length"] * \
        mech.drho_slab * mech.g * 1/_numpy.sqrt(_numpy.pi)
    )

    # Add the sediments, if necessary
    if options["Sediment subduction"]:
        slab_data.loc[~mask, "slab_pull_force_mag"] += (
            slab_data.loc[~mask, "sediment_thickness"] * slab_data.loc[~mask, "slab_length"] * mech.drho_sed * mech.g * 1/_numpy.sqrt(_numpy.pi)
        )

    # Decompose into latitudinal and longitudinal components
    slab_data["slab_pull_force_lat"], slab_data["slab_pull_force_lon"] = mag_azi2lat_lon(
        slab_data["slab_pull_force_mag"], 
        slab_data["trench_normal_azimuth"]
    )

    return slab_data

def compute_interface_term(
        slab_data: _pandas.DataFrame,
        options: Dict[str, Any],
    ) -> _pandas.DataFrame:
    """
    Function to calculate the interface term that accounts for resisting forces at the subduction interface.
    These forces are i) shearing along the plate interface, ii) bending of the slab, and iii) vertical resistance to slab sinking.

    :param slab_data:   subduction zone data
    :type slab_data:    pandas.DataFrame
    :param options:     options
    :type options:      dict
    """
    # Calculate the interface term as a function of the sediment fraction, if enabled
    if options["Sediment subduction"]:
        # Determine shear zone width
        if options["Shear zone width"] == "variable":
            slab_data["shear_zone_width"] = slab_data["v_convergence_mag"] * constants.cm_a2m_s / options["Strain rate"]
        else:
            slab_data["shear_zone_width"] = options["Shear zone width"]

        # Calculate sediment fraction using sediment thickness and shear zone width
        # Step 1: Calculate sediment_fraction based on conditions
        slab_data["sediment_fraction"] = _numpy.where(
            slab_data["slab_seafloor_age"].isna(), 
            0, 
            slab_data["sediment_thickness"].fillna(0) / slab_data["shear_zone_width"]
        )

        # Step 2: Cap values at 1 (ensure fraction does not exceed 1)
        slab_data["sediment_fraction"] = slab_data["sediment_fraction"].clip(upper=1)

        # Step 3: Replace NaNs with 0 (if needed)
        slab_data["sediment_fraction"] = slab_data["sediment_fraction"].fillna(0)
            
        # Calculate interface term
        interface_term = 11 - 10**(1-slab_data["sediment_fraction"])
        logging.info(f"Mean, min and max of interface terms: {interface_term.mean()}, {interface_term.min()}, {interface_term.max()}")
    else:
        interface_term = 1.
    
    # Multiply by slab pull constant
    interface_term *= slab_data["slab_pull_constant"]

    # Apply interface term to slab pull force
    slab_data["slab_pull_force_mag"] *= interface_term
    slab_data["slab_pull_force_lat"] *= interface_term
    slab_data["slab_pull_force_lon"] *= interface_term

    return slab_data

def compute_slab_suction_force(
        slab_data: _pandas.DataFrame,
        options: Dict[str, Any],
    ) -> _pandas.DataFrame:
    """
    Function to calculate slab suction force at subduction zones.

    :param slab_data:   subduction zone data
    :type slab_data:    pandas.DataFrame
    :param options:     options
    :type options:      dict

    :return:            slab_data
    :rtype:             pandas.DataFrame

    Slab suction here is assumed to be proportional to the slab pull force, because the faster a slab subducts, 
    the faster the overriding plate will be sucked towards the trench.

    NOTE: The magnitude of the slab suction constant needs to be optimised in some way.
    """
    # Mask entries with seafloor age (and thus a nonzero magnitude of the slab pull force)
    mask = ~slab_data["slab_seafloor_age"].isna()

    # Set magnitude of entries with no seafloor age to zero
    slab_data.loc[~mask, "slab_suction_force_mag"] = 0

    # Set magnitude of entries with seafloor age proportional to the slab pull force magnitude
    slab_data.loc[mask, "slab_suction_force_mag"] = options["Slab suction constant"] * slab_data.loc[mask, "slab_pull_force_mag"]

    # Decompose all entries into latitudinal and longitudinal components, with the vector pointing in the opposite direction of the trench normal vector
    slab_data["slab_suction_force_lat"], slab_data["slab_suction_force_lon"] = mag_azi2lat_lon(
        slab_data["slab_suction_force_mag"],
        slab_data["trench_normal_azimuth"]-180 % 360
    )

    return slab_data

def compute_slab_bend_force(
        slab_data: _pandas.DataFrame,
        options: Dict[str, Any],
        mech,  # Is this needed?
        constants  # Is this needed?
    ) -> _pandas.DataFrame:
    """
    Function to calculate the slab bending force.

    :param slab_data:   subduction zone data
    :type slab_data:    pandas.DataFrame
    :param options:     options
    :type options:      dict

    NOTE: This function may need a look to see if it actually works.
    """
    # Calculate slab bending torque
    if options["Bending mechanism"] == "viscous":
        bending_force = (-2. / 3.) * ((slab_data.lower_plate_thickness) / \
                        (mech.rad_curv)) ** 3 * mech.lith_visc * slab_data.v_convergence * constants.cm_a2m_s # [n-s , e-w], [N/m]
    elif options["Bending mechanism"] == "plastic":
        bending_force = (-1. / 6.) * ((slab_data.lower_plate_thickness ** 2) / \
                        mech.rad_curv) * mech.yield_stress * _numpy.asarray(
                            (_numpy.cos(slab_data.trench_normal_vector + slab_data.obliquity_convergence),
                             _numpy.sin(slab_data.trench_normal_vector + slab_data.obliquity_convergence))
                        )  # [n-s, e-w], [N/m]
        
    slab_data["bend_force_lat"], slab_data["bend_force_lon"] = mag_azi2lat_lon(
        bending_force,
        slab_data.trench_normal_vector + slab_data.obliquity_convergence
    )
    
    return slab_data

def compute_GPE_force(
        points: _pandas.DataFrame,
        seafloor_grid: _xarray.DataArray,
        options: Dict[str, Any],
        mech: Dict, # Is this necessary?
    ) -> _pandas.DataFrame:
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

def compute_mantle_drag_force(
        points: _pandas.DataFrame,
        options: Dict[str, Any], 
        constants: Dict, 
    ) -> _pandas.DataFrame:
    """
    Function to calculate mantle drag force at points.

    NOTE: The implementation of lateral viscosity variations may need to be checked.
    """
    # Get velocities at points
    if options["Reconstructed motions"]:
        # if options["Continental keels"]:
        #     # Mask points with seafloor age
        #     mask = ~points["seafloor_age"].isna()

        #     # Calculate mantle drag force under continents
        #     points.loc[~mask, "mantle_drag_force_lat"] = -1 * points.loc[~mask, "velocity_lat"] * constants.cm_a2m_s * options["Mantle viscosity"] * options["Lateral viscosity variation"] / mech.La
        #     points.loc[~mask, "mantle_drag_force_lon"] = -1 * points.loc[~mask, "velocity_lon"] * constants.cm_a2m_s * options["Mantle viscosity"] * options["Lateral viscosity variation"] / mech.La

        # else:
        #     # Select all points
        #     mask = points.index

        # # Calculate mantle drag force
        # points.loc[mask, "mantle_drag_force_lat"] = -1 * points.loc[mask, "velocity_lat"] * constants.cm_a2m_s * options["Mantle viscosity"] / mech.La
        # points.loc[mask, "mantle_drag_force_lon"] = -1 * points.loc[mask, "velocity_lon"] * constants.cm_a2m_s * options["Mantle viscosity"] / mech.La
        # points.loc[:, "mantle_drag_force_mag"] = _numpy.linalg.norm([points["mantle_drag_force_lat"], points["mantle_drag_force_lon"]], axis=0)

        if options["Continental keels"]:
            points["mantle_drag_force_lat"] = -1 * points["velocity_lat"] * constants.cm_a2m_s * options["Mantle viscosity"] / (300e3 - points["LAB_depth"])
            points["mantle_drag_force_lon"] = -1 * points["velocity_lon"] * constants.cm_a2m_s * options["Mantle viscosity"] / (300e3 - points["LAB_depth"])
        else:
            points["mantle_drag_force_lat"] = -1 * points["velocity_lat"] * constants.cm_a2m_s * options["Mantle viscosity"] / mech.La
            points["mantle_drag_force_lon"] = -1 * points["velocity_lon"] * constants.cm_a2m_s * options["Mantle viscosity"] / mech.La

        points["mantle_drag_force_mag"] = _numpy.linalg.norm([points["mantle_drag_force_lat"], points["mantle_drag_force_lon"]], axis=0)
        
    return points

def compute_residual_force(
        point_data: _pandas.DataFrame,
        plate_data: _pandas.DataFrame,
        plateID_col: str = "plateID",
        weight_col: str = "segment_area",
    ) -> _pandas.DataFrame:
    """
    Function to calculate residual torque at a series of points.
    """
    # Initialise arrays to store velocities
    forces_lat = _numpy.zeros_like(point_data.lat); forces_lon = _numpy.zeros_like(point_data.lat)
    forces_mag = _numpy.zeros_like(point_data.lat); forces_azi = _numpy.zeros_like(point_data.lat)

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
        forces_lat[mask], forces_lon[mask], forces_mag[mask], forces_azi[mask] = tangent_cartesian2spherical(
            forces_xyz,
            point_data[mask].lat.values,
            point_data[mask].lon.values,
        )
        
        # Normalise the force components by the weight of the segment
        forces_mag[mask] /= point_data[mask][weight_col].values * constants.mean_Earth_radius_m**2
        forces_lat[mask] /= point_data[mask][weight_col].values * constants.mean_Earth_radius_m**2
        forces_lon[mask] /= point_data[mask][weight_col].values * constants.mean_Earth_radius_m**2

    return forces_lat, forces_lon, forces_mag, forces_azi

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# TORQUE CALCULATIONS
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def compute_torque_on_plates(
        plate_data: _pandas.DataFrame,
        points_lat: Union[_numpy.ndarray, _pandas.DataFrame],
        points_lon: Union[_numpy.ndarray, _pandas.DataFrame],
        plateIDs: Union[_numpy.ndarray, _pandas.DataFrame],
        forces_lat: Union[_numpy.ndarray, _pandas.DataFrame],
        forces_lon: Union[_numpy.ndarray, _pandas.DataFrame],
        areas: Union[_numpy.ndarray, _pandas.DataFrame],
        constants: Dict,
        torque_var: str = "slab_pull",
    ) -> _pandas.DataFrame:
    """
    Calculate and update torque information on plates based on latitudinal, longitudinal forces, and segment dimensions.

    :param torques:             torque data with columns 'plateID', 'centroid_lat', 'centroid_lon', and x, y, z and mag torque components.
    :type torques:              pd.DataFrame
    :param points_lat:          latitude of force vector origins in degrees.
    :type points_lat:           array-like
    :param points_lon:          longitude of force vector origins in degrees.
    :type points_lon:           array-like
    :param plateID:             plate IDs corresponding to each point.
    :type plateID:              array-like
    :param force_lat:           latitudinal component of the applied force.
    :type force_lat:            array-like
    :param force_lon:           longitudinal component of the applied force.
    :type force_lon:            array-like
    :param segment_length_lat:  length of the segment in the latitudinal direction.
    :type segment_length_lat:   array-like
    :param segment_length_lon:  length of the segment in the longitudinal direction.
    :type segment_length_lon:   array-like
    :param constants:           constants used in coordinate conversions and calculations.
    :type constants:            class
    :param torque_var:          name of the torque variable (default = "slab_pull")
    :type torque_var:           str

    :return: Updated torques DataFrame with added columns for torque components at the centroid, force components at the centroid, and latitudinal and longitudinal components of the force.
    :rtype: pd.DataFrame

    This function calculates torques in Cartesian coordinates based on latitudinal, longitudinal forces, and segment dimensions.
    It then sums the torque components for each plate, calculates the torque vector at the centroid, and updates the torques DataFrame.
    Finally, it calculates the force components at the centroid, converts them to latitudinal and longitudinal components, and adds these to the torques DataFrame.
    """
    # Initialise dataframes and sort plateIDs
    point_data = _pandas.DataFrame({"plateID": plateIDs})

    # Convert points to Cartesian coordinates
    positions_xyz = geocentric_spherical2cartesian(
        points_lat,
        points_lon,
        constants.mean_Earth_radius_m
    )
    
    # Convert forces to Cartesian coordinates
    forces_xyz = tangent_spherical2cartesian(
        points_lat,
        points_lon,
        forces_lat * areas,
        forces_lon * areas,
    )

    # Calculate torques in Cartesian coordinates
    torques_xyz = _numpy.cross(positions_xyz, forces_xyz, axis=0)    

    # Assign the calculated torques to the respective columns in the point_data dataframe
    point_data[torque_var + "_torque_x"] = torques_xyz[0]
    point_data[torque_var + "_torque_y"] = torques_xyz[1]
    point_data[torque_var + "_torque_z"] = torques_xyz[2]

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

    # Calculate torque magnitude
    plate_data[torque_var + "_torque_mag"] = _numpy.sqrt(
        plate_data[torque_var + "_torque_x"]**2 + plate_data[torque_var + "_torque_y"]**2 + plate_data[torque_var + "_torque_z"]**2
    )

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

def sum_torque(
        plate_data: _pandas.DataFrame,
        torque_type: str,
        constants: Dict,
    ) -> _pandas.DataFrame:
    """
    Function to calculate driving and residual torque on plates.
    """
    # Determine torque components based on torque type
    if torque_type == "driving" or torque_type == "mantle_drag":
        torque_components = ["slab_pull_torque", "GPE_torque", "slab_suction_torque"]
    elif torque_type == "mantle_drag":
        torque_components = ["slab_pull_torque", "GPE_torque", "slab_bend_torque", "slab_suction_torque"]
    elif torque_type == "residual":
        torque_components = ["slab_pull_torque", "GPE_torque", "slab_bend_torque", "slab_suction_torque", "mantle_drag_torque"]
    else:
        raise ValueError("Invalid torque_type, must be 'driving' or 'residual' or 'mantle_drag'!")

    # Calculate torque in Cartesian coordinates
    for axis in ["_x", "_y", "_z"]:
        plate_data.loc[:, f"{torque_type}_torque{axis}"] = _numpy.sum(
            [_numpy.nan_to_num(plate_data[component + axis]) for component in torque_components], axis=0
        )
    
    if torque_type == "mantle_drag":
        for axis in ["_x", "_y", "_z"]:
            torque_values = plate_data[f"{torque_type}_torque{axis}"].values
            if not _numpy.allclose(torque_values, 0):  # Only flip if non-zero
                plate_data.loc[:, f"{torque_type}_torque{axis}"] *= -1
    
    # Organise torque in an array
    summed_torques_cartesian = _numpy.asarray([
        plate_data[f"{torque_type}_torque_x"], 
        plate_data[f"{torque_type}_torque_y"], 
        plate_data[f"{torque_type}_torque_z"]
    ])

    # Calculate torque magnitude
    plate_data.loc[:, f"{torque_type}_torque_mag"] = _numpy.linalg.norm(summed_torques_cartesian, axis=0)

    # Calculate the position vector of the centroid of the plate in Cartesian coordinates
    centroid_position = geocentric_spherical2cartesian(plate_data.centroid_lat, plate_data.centroid_lon, constants.mean_Earth_radius_m)

    # Calculate the torque vector as the cross product of the Cartesian torque vector (x, y, z) with the position vector of the centroid
    force_at_centroid_xyz = _numpy.cross(summed_torques_cartesian, centroid_position, axis=0)

    # Compute force magnitude at centroid
    force_at_centroid_sph = tangent_cartesian2spherical(
        force_at_centroid_xyz.T, plate_data.centroid_lat, plate_data.centroid_lon
    )

    # Assign force components to DataFrame
    plate_data.loc[:, f"{torque_type}_force_lat"] = force_at_centroid_sph[0]
    plate_data.loc[:, f"{torque_type}_force_lon"] = force_at_centroid_sph[1]
    plate_data.loc[:, f"{torque_type}_force_mag"] = force_at_centroid_sph[2]
    plate_data.loc[:, f"{torque_type}_force_azi"] = force_at_centroid_sph[3]

    return plate_data

def optimise_torques(
        plate_data: _pandas.DataFrame,
        mech: Dict,
        options: Dict,
    ) -> _pandas.DataFrame:
    """
    Function to optimise torques.

    NOTE: Is this function redundant?
    """
    for axis in ["_x", "_y", "_z", "_mag"]:
        plate_data["slab_pull_torque_opt" + axis] = plate_data["slab_pull_torque" + axis].values * options["Slab pull constant"]
        plate_data["mantle_drag_torque_opt" + axis] = plate_data["mantle_drag_torque" + axis].values * options["Mantle viscosity"] / mech.La

    return plate_data

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# GRID SAMPLING
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def sample_grid(
        points_lat: _numpy.ndarray,
        points_lon: _numpy.ndarray,
        grid: _xarray.Dataset,
        coords: List[str] = ["latitude", "longitude"],
    ) -> _numpy.ndarray:
    """
    Function to sample a grid.
    """
    # Load grid into memory to decrease computation time
    grid = grid.load()

    # Extract latitude and longitude values from points and convert to xarray DataArrays
    points_lat_da = _xarray.DataArray(points_lat, dims="point")
    points_lon_da = _xarray.DataArray(points_lon, dims="point")

    # Interpolate age value at point
    sampled_values = _numpy.asarray(
        grid.interp({coords[0]: points_lat_da, coords[1]: points_lon_da}, method="nearest").values.tolist()
    )

    # Close the grid to free memory space
    grid.close()

    return sampled_values

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# GENERAL FUNCTIONS
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def compute_thicknesses(
        seafloor_ages: Union[List[Union[int, float]], _numpy.ndarray, _pandas.Series],
        options: Dict[str, Any],
        crust: bool = True,
        water: bool = True
    ) -> Tuple[_numpy.ndarray, Union[_numpy.ndarray, None], Union[_numpy.ndarray, None]]:
    """
    Calculate lithospheric mantle thickness, crustal thickness, and water depth based on seafloor age profiles.

    :param seafloor_ages:   seafloor ages for which thicknesses are calculated.
    :type seafloor_ages:    array-like
    :param options:         options for controlling the calculation, including the seafloor age profile.
    :type options:          dict
    :param crust:           flag to calculate crustal thickness. Defaults to True.
    :type crust:            bool
    :param water:           flag to calculate water depth. Defaults to True.
    :type water:            bool

    :return:                calculated lithospheric mantle thickness.
    :rtype:                 tuple(array-like, array-like or None, array-like or None)

    This function calculates lithospheric mantle thickness, crustal thickness, and water depth based on seafloor age profiles.
    The calculation depends on options["Seafloor age profile"]:
        - If "half space cooling", lithospheric_mantle_thickness is calculated from half space cooling theory.
        - If "plate model", lithospheric_mantle_thickness is calculated from a plate model.
    
    Crustal thickness and water depth are optional and depend on the values of the 'crust' and 'water' parameters, respectively.
    """
    # Set mechanical parameters and constants
    # NOTE: Is this necessary??
    mech = set_mech_params()
    constants = set_constants()

    # Thickness of oceanic lithosphere from half space cooling and water depth from isostasy
    if options["Seafloor age profile"] == "half space cooling":
        lithospheric_mantle_thickness = _numpy.where(
            _numpy.isnan(seafloor_ages), 
            mech.cont_lith_thick, 
            2.32 * _numpy.sqrt(mech.kappa * seafloor_ages * constants.a2s * 1e6)
        )
        
        if crust:
            crustal_thickness = _numpy.where(
                _numpy.isnan(seafloor_ages), 
                mech.cont_crust_thick, 
                mech.ocean_crust_thick
            )
        else:
            crustal_thickness = _numpy.nan
            
        if water:
            water_depth = _numpy.where(
                _numpy.isnan(seafloor_ages), 
                0.,
                (lithospheric_mantle_thickness * ((mech.rho_a - mech.rho_l) / (mech.rho_sw - mech.rho_a))) + 2600
            )
        else:
            water_depth = _numpy.nan
        
    # Water depth from half space cooling and lithospheric thickness from isostasy
    elif options["Seafloor age profile"] == "plate model":
        hw = _numpy.where(seafloor_ages > 81, 6586 - 3200 * _numpy.exp((-seafloor_ages / 62.8)), seafloor_ages)
        hw = _numpy.where(hw <= 81, 2600 + 345 * _numpy.sqrt(hw), hw)
        lithospheric_mantle_thickness = (hw - 2600) * ((mech.rho_sw - mech.rho_a) / (mech.rho_a - mech.rho_l))

        if crust:
            crustal_thickness = _numpy.where(
                _numpy.isnan(seafloor_ages), 
                mech.cont_crust_thick, 
                mech.ocean_crust_thick
            )
        else:
            crustal_thickness = _numpy.nan
        
        if water:
            water_depth = hw
        else:
            water_depth = _numpy.nan

    return lithospheric_mantle_thickness, crustal_thickness, water_depth

def compute_LAB_depth(
        point_data: _pandas.DataFrame,
        options: Dict[str, Any],
    ) -> _pandas.DataFrame:
    """
    Function to calculate LAB depths based on seafloor ages.

    :param point_data:      point data.
    :type point_data:       pandas.DataFrame
    :param seafloor_ages:   seafloor ages for which LAB depths are calculated.
    :type seafloor_ages:    array-like
    :param options:         options for controlling the calculation, including the seafloor age profile.
    :type options:          dict

    :return:                updated point data with LAB depths.
    :rtype:                 pandas.DataFrame
    """
    # Mask entries with seafloor age
    seafloor_mask = ~point_data["seafloor_age"].isna()

    # Compute lithospheric mantle thickness, crustal thickness, and water depth
    point_data.loc[seafloor_mask, "lithospheric_mantle_thickness"], point_data.loc[seafloor_mask, "crustal_thickness"], point_data.loc[seafloor_mask, "water_depth"] = compute_thicknesses(
        point_data.loc[seafloor_mask, "seafloor_age"],
        options
    )

    # Calculate LAB depth
    point_data.loc[seafloor_mask, "LAB_depth"] = point_data.loc[seafloor_mask, "lithospheric_mantle_thickness"] + point_data.loc[seafloor_mask, "crustal_thickness"] + point_data.loc[seafloor_mask, "water_depth"]

    # Mask entries with no LAB depth
    nan_mask = point_data["LAB_depth"].isna()

    # Fill NaN values with 0
    point_data.loc[nan_mask, "LAB_depth"] = mech.cont_crust_thick + mech.cont_lith_thick

    return point_data

def compute_subduction_flux(
        plate_data: _pandas.DataFrame,
        slab_data: _pandas.DataFrame,
        type: str = "slab" or "sediment",
    ) -> _pandas.DataFrame:
    """
    Function to calculate subduction flux along subduction zones and sum these for plates.

    :param plate_data:  plate data
    :type plate_data:   pandas.DataFrame
    :param slab_data:   slab data
    :type slab_data:    pandas.DataFrame
    :param type:        type of subduction flux to calculate ()
    :type type:         str

    :return:            plate data
    :rtype:             pandas.DataFrame
    """
    # Calculate subduction flux
    for plateID in plate_data.plateID.values:
        selected_slabs = slab_data[slab_data.lower_plateID == plateID]
        if type == "slab":
            plate_data.loc[plate_data.plateID == plateID, "slab_flux"] = (selected_slabs.lower_plate_thickness * selected_slabs.v_lower_plate_mag * selected_slabs.trench_segment_length).sum()
        
        elif type == "sediment":
            plate_data.loc[plate_data.plateID == plateID, "sediment_flux"] = (selected_slabs.sediment_thickness * selected_slabs.v_lower_plate_mag * selected_slabs.trench_segment_length).sum()

    return plate_data

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# CONVERSIONS
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def geocentric_cartesian2spherical(
        vectors_x: Union[int, float, List[Union[int, float]], _numpy.ndarray, _pandas.Series],
        vectors_y: Optional[Union[int, float, List[Union[int, float]], _numpy.ndarray, _pandas.Series]] = None,
        vectors_z: Optional[Union[int, float, List[Union[int, float]], _numpy.ndarray, _pandas.Series]] = None,
    ) -> Tuple[_numpy.ndarray, _numpy.ndarray, _numpy.ndarray, _numpy.ndarray]:
    """
    Convert Cartesian coordinates to latitude, longitude, magnitude, and azimuth.
    """
    # If only x is provided as a 1D array, unpack it into x, y, z
    # This makes the function more flexible.
    if vectors_y is None and vectors_z is None:
        vectors_x, vectors_y, vectors_z = vectors_x[0], vectors_x[1], vectors_x[2]

    # Convert integers and floats to lists
    if isinstance(vectors_x, (int, float, _numpy.integer, _numpy.floating)):
        vectors_x = [vectors_x]
    if isinstance(vectors_y, (int, float, _numpy.integer, _numpy.floating)):
        vectors_y = [vectors_y]
    if isinstance(vectors_z, (int, float, _numpy.integer, _numpy.floating)):
        vectors_z = [vectors_z]

    # Ensure x, y, z are NumPy arrays
    vectors_x = _numpy.asarray(vectors_x)
    vectors_y = _numpy.asarray(vectors_y)
    vectors_z = _numpy.asarray(vectors_z)

    # Stack coordinates to handle multiple points
    vectors_xyz = _numpy.column_stack((vectors_x, vectors_y, vectors_z))

    # Calculate magnitude (norm)
    vectors_mags = _numpy.linalg.norm(vectors_xyz, axis=1)

    # Mask for zero or NaN magnitudes
    valid_mask = (vectors_mags > 0) & (~_numpy.isnan(vectors_mags))

    # Initialise result arrays
    vectors_lats = _numpy.zeros_like(vectors_mags)
    vectors_lons = _numpy.zeros_like(vectors_mags)
    vectors_azis = _numpy.zeros_like(vectors_mags)

    # Calculate latitude (in degrees)
    vectors_lats[valid_mask] = _numpy.rad2deg(_numpy.arcsin(vectors_z[valid_mask] / vectors_mags[valid_mask]))

    # Calculate longitude (in degrees)
    vectors_lons[valid_mask] = _numpy.rad2deg(_numpy.arctan2(vectors_y[valid_mask], vectors_x[valid_mask]))

    # Calculate azimuth (in degrees, measured from North in XY plane)
    vectors_azis[valid_mask] = _numpy.rad2deg(_numpy.arctan2(vectors_x[valid_mask], vectors_y[valid_mask]))

    return vectors_lats, vectors_lons, vectors_mags, vectors_azis

def geocentric_spherical2cartesian(
        points_lat: Union[List[Union[int, float]], _numpy.ndarray, _pandas.Series],
        points_lon: Union[List[Union[int, float]], _numpy.ndarray, _pandas.Series],
        vectors_mag: Union[int, float, List[Union[int, float]], _numpy.ndarray, _pandas.Series] = 1
    ):
    """
    Convert latitude and longitude to Cartesian coordinates.

    

    By default, the Cartesian coordinates are calculated on the unit sphere.
    """
    # Ensure inputs are NumPy arrays
    points_lat = _numpy.asarray(points_lat)
    points_lon = _numpy.asarray(points_lon)
    vectors_mag = _numpy.asarray(vectors_mag)

    # Convert to radians
    points_lat_rad = _numpy.deg2rad(points_lat)
    points_lon_rad = _numpy.deg2rad(points_lon)

    # Calculate x, y, z
    x = vectors_mag * _numpy.cos(points_lat_rad) * _numpy.cos(points_lon_rad)
    y = vectors_mag * _numpy.cos(points_lat_rad) * _numpy.sin(points_lon_rad)
    z = vectors_mag * _numpy.sin(points_lat_rad)

    return x, y, z

def tangent_cartesian2spherical(
        vectors_xyz: _numpy.ndarray,
        points_lat: Union[List[Union[int, float]], _numpy.ndarray, _pandas.Series],
        points_lon: Union[List[Union[int, float]], _numpy.ndarray, _pandas.Series],
    ):
    """
    Convert a vector that is tangent to the surface of a sphere to spherical coordinates.

    

    NOTE:   This function is probably the slowest in Plato.
            It could be sped up by actually implementing the mathematics using numpy.
    """
    # Ensure inputs are NumPy arrays
    points_lat = _numpy.asarray(points_lat)
    points_lon = _numpy.asarray(points_lon)

    # Initialise result arrays
    vectors_mag = _numpy.zeros_like(points_lat)
    vectors_azi = _numpy.zeros_like(points_lat)

    # Loop through points and convert vector to latitudinal and longitudinal components
    for i, (point_lat, point_lon) in enumerate(zip(points_lat, points_lon)):
        # Make PointonSphere
        point = _pygplates.PointOnSphere(point_lat, point_lon)

        # Convert vector to magnitude, azimuth, and inclination
        vectors_mag[i], vectors_azi[i], _ = _numpy.asarray(
            _pygplates.LocalCartesian.convert_from_geocentric_to_magnitude_azimuth_inclination(
                point, 
                (vectors_xyz[i,0], vectors_xyz[i,1], vectors_xyz[i,2])
            )
        )
    
    # Convert azimuth from radians to degrees
    vectors_azi = _numpy.rad2deg(vectors_azi)
    
    # Convert to latitudinal and longitudinal components
    vectors_lat, vectors_lon = mag_azi2lat_lon(vectors_mag, vectors_azi)

    return vectors_lat, vectors_lon, vectors_mag, vectors_azi

def tangent_spherical2cartesian(
        points_lat: Union[List[Union[int, float]], _numpy.ndarray, _pandas.Series], 
        points_lon: Union[List[Union[int, float]], _numpy.ndarray, _pandas.Series], 
        vectors_lat: Union[List[Union[int, float]], _numpy.ndarray, _pandas.Series], 
        vectors_lon: Union[List[Union[int, float]], _numpy.ndarray, _pandas.Series], 
    ):
    """
    Convert a vector tangent to the surface of a sphere to Cartesian coordinates.

    :param points_lat:  latitude of the vector origins in degrees.
    :type points_lat:   list, numpy.array, pandas.Series
    :param points_lon:  longitude in the vector origin in degrees.
    :type points_lon:   list, numpy.array, pandas.Series
    :param vectors_lat: latitudinal vector components.
    :type vectors_lat:  numpy.array, pandas.Series
    :param vectors_lon: longitudinal vector components.
    :type vectors_lon:  numpy.array, pandas.Series

    :return:            vectors in Cartesian coordinates.
    :rtype:             numpy.array
    """
    # Ensure inputs are NumPy arrays
    points_lat = _numpy.asarray(points_lat)
    points_lon = _numpy.asarray(points_lon)

    # Convert lon, lat to radian
    points_lon_rad = _numpy.deg2rad(points_lon)
    points_lat_rad = _numpy.deg2rad(points_lat)

    # Calculate force_magnitude
    vectors_mag = _numpy.linalg.norm([vectors_lat, vectors_lon], axis=0)

    # Calculate theta
    theta = _numpy.empty_like(vectors_lon)
    mask = ~_numpy.logical_or(vectors_lon == 0, _numpy.isnan(vectors_lon), _numpy.isnan(vectors_lat))
    theta[mask] = _numpy.where(
        (vectors_lon[mask] > 0) & (vectors_lat[mask] >= 0),  
        _numpy.arctan(vectors_lat[mask] / vectors_lon[mask]),                          
        _numpy.where(
            (vectors_lon[mask] < 0) & (vectors_lat[mask] >= 0) | (vectors_lon[mask] < 0) & (vectors_lat[mask] < 0),    
            _numpy.pi + _numpy.arctan(vectors_lat[mask] / vectors_lon[mask]),              
            (2*_numpy.pi) + _numpy.arctan(vectors_lat[mask] / vectors_lon[mask])           
        )
    )

    # Calculate force in Cartesian coordinates
    vectors_x = vectors_mag * _numpy.cos(theta) * (-1.0 * _numpy.sin(points_lon_rad))
    vectors_y = vectors_mag * _numpy.cos(theta) * _numpy.cos(points_lon_rad)
    vectors_z = vectors_mag * _numpy.sin(theta) * _numpy.cos(points_lat_rad)

    # Convert to numpy array
    vectors_xyz = _numpy.asarray([vectors_x, vectors_y, vectors_z])

    return vectors_xyz   

def mag_azi2lat_lon(
        magnitude: Union[List[Union[int, float]], _numpy.ndarray, _pandas.Series],
        azimuth: Union[List[Union[int, float]], _numpy.ndarray, _pandas.Series],
    ):
    """
    Decompose a vector defined by magnitude and azimuth into latitudinal and longitudinal components.

    :param magnitude:   Magnitude of vector.
    :type magnitude:    list, numpy.array, pandas.Series
    :param azimuth:     Azimuth of vector in degrees.
    :type azimuth:      numpy.array, pandas.Series

    :return:            Latitudinal and longitudinal components.
    :rtype:             float or numpy.array, float or numpy.array
    """
    # Ensure inputs are NumPy arrays
    magnitude = _numpy.asarray(magnitude)
    azimuth = _numpy.array(azimuth)

    # Convert azimuth from degrees to radians
    azimuth_rad = _numpy.deg2rad(azimuth)

    # Calculate components
    component_lat = _numpy.cos(azimuth_rad) * magnitude
    component_lon = _numpy.sin(azimuth_rad) * magnitude

    return component_lat, component_lon

def lat_lon2mag_azi(
        component_lat: Union[List[Union[int, float]], _numpy.ndarray, _pandas.Series],
        component_lon: Union[List[Union[int, float]], _numpy.ndarray, _pandas.Series],
    ):
    """
    Function to convert a 2D vector into magnitude and azimuth [in degrees from north].

    :param component_lat:   latitudinal component of vector
    :type component_lat:    list, numpy.array, pandas.Series
    :param component_lon:   latitudinal component of vector
    :type component_lon:    list, numpy.array, pandas.Series

    :return:                magnitude, azimuth
    :rtype:                 float or numpy.array, float or numpy.array
    """
    # Esure inputs are NumPy arrays
    component_lat = _numpy.asarray(component_lat)
    component_lon = _numpy.asarray(component_lon)

    # Calculate magnitude
    magnitude = _numpy.linalg.norm([component_lat, component_lon**2], axis=0)

    # Calculate azimuth in radians
    azimuth_rad = _numpy.arctan2(component_lon, component_lat)

    # Convert azimuth from radians to degrees
    azimuth_deg = _numpy.rad2deg(azimuth_rad)

    return magnitude, azimuth_deg

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ROTATIONS & VELOCITIES
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

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

    # Normalise the rotation poles by the drag coefficient and the area of a plate
    if options["Continental keels"]:
        stage_rotation_poles_mag /= options["Mantle viscosity"] / (300e3 - plates.mean_LAB_depth) * plates.area
    else:
        stage_rotation_poles_mag /= options["Mantle viscosity"] / mech.La * plates.area

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
    Function to compute lat, lon, magnitude and azimuth of velocity at a set of locations from an Euler rotation.
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
    v_rms_mag = _numpy.sqrt(_numpy.sum(v_mag**2 * segment_areas) / total_area)

    # Calculate RMS velocity azimuth (in radians)
    sin_azi = _numpy.sum(_numpy.sin(v_azi) * segment_areas) / total_area
    cos_azi = _numpy.sum(_numpy.cos(v_azi) * segment_areas) / total_area

    # v_rms_azi = _numpy.rad2deg(
    #     -1 * (_numpy.arctan2(sin_azi, cos_azi) + 0.5 * _numpy.pi)
    # )
    v_rms_azi = _numpy.rad2deg(
        _numpy.arctan2(sin_azi, cos_azi)
    )
    # Ensure azimuth is within the range [0, 360]
    v_rms_azi = _numpy.where(v_rms_azi < 0, v_rms_azi + 360, v_rms_azi)
    
    # Calculate spin rate
    omega_rms = _numpy.sqrt(_numpy.sum(omega **2 * segment_areas) / total_area)

    return v_rms_mag, v_rms_azi, omega_rms

def compute_net_rotation(
        plate_data: _pandas.DataFrame,
        point_data: _pandas.DataFrame,
    ) -> Tuple[_numpy.ndarray, _numpy.ndarray, _numpy.ndarray]:
    """
    Function to calculate net rotation of the entire lithosphere relative to the lower mantle.
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

def compute_no_net_rotation(
        plate_data: _pandas.DataFrame,
        point_data: _pandas.DataFrame,
    ) -> _pandas.DataFrame:
    """
    Function to remove the net rotation of the entire lithosphere relative to the lower mantle from the stage rotations of individual plates.
    """
    # Calculate net rotation in spherical coordinates
    net_rotation_lat, net_rotation_lon, net_rotation_mag = compute_net_rotation(plate_data, point_data)

    # Convert net rotation to Cartesian coordinates
    net_rotation_xyz = _numpy.column_stack(geocentric_spherical2cartesian(
        net_rotation_lat, 
        net_rotation_lon, 
        net_rotation_mag,
    ))

    # Loop through plates
    for index, plate in plate_data.iterrows():
        # Calculate rotation pole in Cartesian coordinates
        rotation_pole_xyz = _numpy.column_stack(geocentric_spherical2cartesian(
            plate.pole_lat, 
            plate.pole_lon, 
            plate.pole_angle,
        ))

        # Subtract the net rotation from the stage rotation
        new_rotation_pole_xyz = rotation_pole_xyz + net_rotation_xyz

        # Convert the new rotation pole to spherical coordinates
        new_rotation_pole_lat, new_rotation_pole_lon, _, _ = geocentric_cartesian2spherical(
            new_rotation_pole_xyz[:, 0], new_rotation_pole_xyz[:, 1], new_rotation_pole_xyz[:, 2],
        )

        # Calculate the magnitude of the new rotation pole
        new_rotation_rate = _numpy.linalg.norm(new_rotation_pole_xyz)

        # Assign new rotation pole to DataFrame
        plate_data.loc[index, "pole_lat"] = new_rotation_pole_lat[0]
        plate_data.loc[index, "pole_lon"] = new_rotation_pole_lon[0]
        plate_data.loc[index, "pole_angle"] = new_rotation_rate

    return plate_data

def compute_trench_migration(
        slab_data: _pandas.DataFrame,
        options: Dict,
        type: str = None
    ) -> Tuple[_numpy.ndarray, _numpy.ndarray, _numpy.ndarray]:
    """
    Function to calculate global trench migration.
    

    Trench migration is calculated as Euler rotations that define the rotation of the entire subduction geometry relative to the lower mantle.
    """
    # Calculate position vectors in Cartesian coordinates on the unit sphere
    # The shape of the position vectors is (n, 3)
    positions_x, positions_y, positions_z = geocentric_spherical2cartesian(
        slab_data.lat, 
        slab_data.lon, 
    )
    positions_xyz = _numpy.column_stack((positions_x, positions_y, positions_z))
    
    # Define which plate motion to use
    plate = "trench" if options["Reconstructed motions"] else "upper_plate"
    
    # Get lat and lon components of trench normal and trench parallel unit vectors
    if type == "normal" or type == "parallel":
        # Use correct azimuth
        azimuth = slab_data["trench_normal_azimuth"] if type == "normal" else slab_data["trench_normal_azimuth"] + 90 % 360

        # Get trench unit vectors
        trench_vectors_lat, trench_vectors_lon = mag_azi2lat_lon(
            1.,
            azimuth
        )

        # Convert the velocity to degrees/Ma
        velocity_vectors_lat = slab_data[f"{plate}_velocity_lat"] * constants.cm_a2deg_a * 1e6
        velocity_vectors_lon = slab_data[f"{plate}_velocity_lon"] * constants.cm_a2deg_a * 1e6
    
        # Get magnitude of trench normal migration from the dot product of the velocity vector and the trench unit vectors
        trench_migration_mag = trench_vectors_lat * velocity_vectors_lat + trench_vectors_lon * velocity_vectors_lon

        # Decompose into latitudinal and longitudinal components
        trench_migration_lat, trench_migration_lon = mag_azi2lat_lon(
            trench_migration_mag, azimuth
        )

    else:
        # Otherwise, simply use the trench velocity
        trench_migration_lat = slab_data[f"{plate}_velocity_lat"] * constants.cm_a2deg_a * 1e6
        trench_migration_lon = slab_data[f"{plate}_velocity_lon"] * constants.cm_a2deg_a * 1e6

    # Convert to Cartesian coordinates
    trench_migration_xyz = tangent_spherical2cartesian(
        slab_data.lat,
        slab_data.lon,
        trench_migration_lat,
        trench_migration_lon,
    )
    
    # Calculate the cross product of the position vector and the velocity vector 
    # This is essentially the same as the double cross product in the net rotation calculation (see Torsvik et al. (2010), https://doi.org/10.1016/j.epsl.2009.12.055)
    # The shape of the rotation pole vector is modified to (N, 3) to allow broadcasting
    trench_rotations_xyz = _numpy.cross(trench_migration_xyz.T, positions_xyz)

    # Sum trench rotations
    net_trench_rotation_xyz = _numpy.sum(trench_rotations_xyz * slab_data["trench_segment_length"].values[:, None], axis=0)

    # Normalise by the total trench length
    net_trench_rotation_xyz /= slab_data["trench_segment_length"].sum()

    # Get the latitude and longitude of the rotation poles
    net_trench_rotation_lat, net_trench_rotation_lon, _, _ = geocentric_cartesian2spherical(
            net_trench_rotation_xyz[0], net_trench_rotation_xyz[1], net_trench_rotation_xyz[2],
        )
    
    # Get the magnitude of the rotation poles
    net_trench_rotation_mag = _numpy.linalg.norm(net_trench_rotation_xyz)

    return net_trench_rotation_lat, net_trench_rotation_lon, net_trench_rotation_mag

def compute_no_net_trench_migration(
        plate_data: _pandas.DataFrame,
        slab_data: _pandas.DataFrame,
        options: Dict[str, Any],
        type: str = None,
    ) -> _pandas.DataFrame:
    """
    Function to remove the net rotation of the subduction geometry relative to the lower mantle from the stage rotations of individual plates.
    """
    # Calculate net rotation in spherical coordinates
    net_rotation_lat, net_rotation_lon, net_rotation_mag = compute_trench_migration(
        slab_data,
        options,
        type
    )

    # Convert net rotation to Cartesian coordinates
    net_rotation_xyz = _numpy.column_stack(geocentric_spherical2cartesian(
        net_rotation_lat, 
        net_rotation_lon, 
        net_rotation_mag,
    ))

    # Loop through plates
    for index, plate in plate_data.iterrows():
        # Calculate rotation pole in Cartesian coordinates
        rotation_pole_xyz = _numpy.column_stack(geocentric_spherical2cartesian(
            plate.pole_lat, 
            plate.pole_lon, 
            plate.pole_angle,
        ))

        # Subtract the net rotation from the stage rotation
        new_rotation_pole_xyz = rotation_pole_xyz + net_rotation_xyz

        # Convert the new rotation pole to spherical coordinates
        new_rotation_pole_lat, new_rotation_pole_lon, _, _ = geocentric_cartesian2spherical(
            new_rotation_pole_xyz[:, 0], new_rotation_pole_xyz[:, 1], new_rotation_pole_xyz[:, 2],
        )

        # Calculate the magnitude of the new rotation pole
        new_rotation_rate = _numpy.linalg.norm(new_rotation_pole_xyz)

        # Assign new rotation pole to DataFrame
        plate_data.loc[index, "pole_lat"] = new_rotation_pole_lat[0]
        plate_data.loc[index, "pole_lon"] = new_rotation_pole_lon[0]
        plate_data.loc[index, "pole_angle"] = new_rotation_rate

    return plate_data

def rotate_torque(
        plateID: Union[int, float],
        torque: _numpy.ndarray,
        rotations_a: _pygplates.RotationModel,
        rotations_b: _pygplates.RotationModel,
        age: Union[int, float],
    ) -> _numpy.ndarray:
    """
    Function to rotate a torque vector in Cartesian coordinates between two reference frames.

    :param plateID:         plateID for which the relative rotation pole is calculated.
    :type plateID:          int, float
    :param torque:          torque vector in Cartesian coordinates.
    :type torque:           numpy.array of length 3,N or N,3 (WHICH ONE?)
    :param rotations_a:     rotation model A
    :type rotations_a:      pygplates.RotationModel
    :param rotations_b:     rotation model B
    :type rotations_b:      pygplates.RotationModel
    :param age:             age to at which to get the relative rotation.
    :type age:              int, float
    
    :return:                rotated torque vector in Cartesian coordinates.
    :rtype:                 numpy.array of length 3,N or N,3 (WHICH ONE?)
    """
    # Get equivalent total rotations for the plateID in both rotation models
    relative_rotation_pole = get_relative_rotaton_pole(plateID, rotations_a, rotations_b, age)

    # Rotate torque vector
    rotated_torque = rotate_vector(torque, relative_rotation_pole)

    return rotated_torque

def get_relative_rotaton_pole(
        plateID: Union[int, float],
        rotations_a: _pygplates.RotationModel,
        rotations_b: _pygplates.RotationModel,
        age: Union[int, float],
    ) -> _numpy.ndarray:
    """
    Function to get the relative rotation pole between two reference frames (A and B) for any plateID.

    :param plateID:         plateID for which the relative rotation pole is calculated.
    :type plateID:          int, float
    :param rotations_a:     rotation model A
    :type rotations_a:      pygplates.RotationModel
    :param rotations_b:     rotation model B
    :type rotations_b:      pygplates.RotationModel
    :param age:             age to at which to get the relative rotation.
    :type age:              int, float

    :return:                relative rotation from reference frame A to reference frame B.
    :rtype:                 NOTE: Not sure, should check. Either a tuple or an array.
    """
    # Make sure the plateID is an integer
    plateID = int(plateID)

    # Make sure the reconstruction time is an integer
    # NOTE: Why should this be an integer?
    age = int(age)

    # Get equivalent total rotations for the plateID in both rotation models
    rotation_a = rotations_a.get_rotation(
        to_time=age,
        moving_plate_id=plateID,
    )
    rotation_b = rotations_b.get_rotation(
        to_time=age,
        moving_plate_id=plateID,
    )

    # Calculate relative rotation pole
    relative_rotation_pole = rotation_a * rotation_b.get_inverse()

    return relative_rotation_pole.get_lat_lon_euler_pole_and_angle_degrees()

def rotate_vector(
        vector: _numpy.ndarray,
        rotation: _numpy.ndarray,
    ) -> _numpy.ndarray:
    """
    Function to rotate a vector in Cartesian coordinates with a given Euler rotation.

    :param vector:      vector in 3D Cartesian coordinates.
    :type vector:       numpy.array of length 3
    :param rotation:    rotation pole latitude, rotation pole longitude, and rotation angle in degrees.
    :type rotation:     numpy.array of length 3

    :return:            rotated vector in Cartesian coordinates.
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

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# MISCELLANEOUS
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def project_points(
        lat: Union[int, float, list, _numpy.ndarray, _pandas.Series],
        lon: Union[int, float, list, _numpy.ndarray, _pandas.Series],
        azimuth: Union[int, float, list, _numpy.ndarray, _pandas.Series],
        distance: Union[int, float]
    ) -> Tuple[_numpy.ndarray, _numpy.ndarray]:
    """
    Function to calculate coordinates of sampling points.

    :param lat:         latitude of the point in degrees.
    :type lat:          int, float, list, numpy.ndarray, pandas.Series
    :param lon:         longitude of the point in degrees.
    :type lon:          int, float, list, numpy.ndarray, pandas.Series
    :param azimuth:     azimuth of the point in degrees.
    :type azimuth:      int, float, list, numpy.ndarray, pandas.Series
    :param distance:    distance from the point in km.

    :return:            latitude and longitude of the sampling points.
    :rtype:             numpy.ndarray, numpy.ndarray
    """
    # Set constants
    constants = set_constants()

    # Convert to numpy arrays
    lat = _numpy.asarray(lat); lon = _numpy.asarray(lon); azimuth = _numpy.asarray(azimuth)

    # Convert to radians
    lat_rad = _numpy.deg2rad(lat); lon_rad = _numpy.deg2rad(lon); azimuth_rad = _numpy.deg2rad(azimuth)

    # Angular distance in km
    angular_distance = distance / constants.mean_Earth_radius_km

    # Calculate sample points
    new_lat_rad = _numpy.arcsin(
        _numpy.sin(lat_rad) * _numpy.cos(angular_distance) + \
        _numpy.cos(lat_rad) * _numpy.sin(angular_distance) * _numpy.cos(azimuth_rad)
    )
    new_lon_rad = lon_rad + _numpy.arctan2(
        _numpy.sin(azimuth_rad) * _numpy.sin(angular_distance) * _numpy.cos(lat_rad), 
        _numpy.cos(angular_distance) - _numpy.sin(lat_rad) * _numpy.sin(new_lat_rad)
    )

    # Convert to degrees
    new_lon = _numpy.rad2deg(new_lon_rad)
    new_lat = _numpy.rad2deg(new_lat_rad)

    return new_lat, new_lon

def haversine(
        lat1: Union[List[Union[int, float]], _numpy.ndarray, _pandas.Series],
        lon1: Union[List[Union[int, float]], _numpy.ndarray, _pandas.Series], 
        lat2: Union[List[Union[int, float]], _numpy.ndarray, _pandas.Series], 
        lon2: Union[List[Union[int, float]], _numpy.ndarray, _pandas.Series]
    ) -> _numpy.ndarray:
    """
    Calculate the Haversine distance between two sets of points on Earth.

    :param lat1:    latitude of the first point in degrees.
    :type lat1:     list, numpy.ndarray, pandas.Series
    :param lon1:    longitude of the first point in degrees.
    :type lon1:     list, numpy.ndarray, pandas.Series
    :param lat2:    latitude of the second point in degrees.
    :type lat2:     list, numpy.ndarray, pandas.Series
    :param lon2:    longitude of the second point in degrees.
    :type lon2:     list, numpy.ndarray, pandas.Series

    :return:        Haversine distance between the two points in degrees (?).
    :rtype:         numpy.ndarray

    NOTE: Need to check the units of the value returned.
    """
    # Make sure the entries are numpy arrays
    lat1 = _numpy.asarray(lat1); lon1 = _numpy.asarray(lat1)
    lat2 = _numpy.asarray(lat2); lon2 = _numpy.asarray(lon2)

    # Convert latitude and longitude from degrees to radians
    lat1 = _numpy.deg2rad(lat1); lon1 = _numpy.deg2rad(lon1)
    lat2 = _numpy.deg2rad(lat2); lon2 = _numpy.deg2rad(lon2)
    
    # Calculate latitudinal and longitudinal differences
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    # Haversine formula
    a = _numpy.sin(dlat / 2) ** 2 + _numpy.cos(lat1) * _numpy.cos(lat2) * _numpy.sin(dlon / 2) ** 2
    c = 2 * _numpy.arctan2(_numpy.sqrt(a), _numpy.sqrt(1 - a))

    return c

def propose_value(
        existing_values: Union[List[Union[int, float]], _numpy.ndarray],
        existing_scores: Union[List[Union[int, float]], _numpy.ndarray],
        lower_bound: Union[int, float] = 1e-8,
        upper_bound: Union[int, float] = 1e-13,
        exploration_prob: Union[int, float] = 0.0,
    ):
    """
    Propose a new value, either focusing on the current best or performing a random excursion.
    
    :param existing_values:     list of existing values.
    :type existing_values:      list, numpy.ndarray
    :param existing_scores:     list of existing scores.
    :type existing_scores:      list, numpy.ndarray
    :param lower_bound:         lower bound of the parameter space.
    :type lower_bound:          int, float
    :param upper_bound:         upper bound of the parameter space.
    :type upper_bound:          int, float
    :param exploration_prob:    probability of exploration.

    :return:                    proposed value.
    :rtype:                     int, float
    """
    # Decide between exploitation and exploration
    if _numpy.random.rand() < exploration_prob:
        # Random exploration: propose a completely random value in the parameter space
        proposed_value = 10 ** _numpy.random.uniform(_numpy.log10(lower_bound), _numpy.log10(upper_bound))
        
    else:
        # Exploitation: refine around the best observed value
        best_index = _numpy.argmin(existing_scores)
        best_value = existing_values[best_index]
        
        # Generate a small perturbation around the best value in log space
        perturbation = 10 ** (_numpy.log10(best_value) + _numpy.random.uniform(-0.5, 0.5))
        proposed_value = perturbation
    
    return proposed_value