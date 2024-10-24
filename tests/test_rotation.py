# %%
from typing import Optional, Dict
import numpy as _numpy
import pandas as _pandas
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R


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
        self.s2a = 365.25 * 24 * 60 * 60                           # a to s
        self.a2s = 1 / self.s2a                                     # s to a

        self.m_s2cm_a = 1e2 * self.s2a  # m/s to cm/a
        self.cm_a2m_s = 1 / self.m_s2cm_a  # cm/a to m/s

        self.rad_a2m_s = self.mean_Earth_radius_m * self.a2s  # rad/a to m/s
        self.m_s2rad_a = 1 / self.rad_a2m_s  # m/s to rad/a

        self.m_s2deg_Ma = _numpy.rad2deg(self.m_s2rad_a) * 1e6  # m/s to deg/Ma
        self.rad_a2cm_a = self.mean_Earth_radius_m * 1e2  # rad/a to cm/a

        self.deg_a2cm_a = _numpy.deg2rad(self.rad_a2cm_a) # deg/a to m/s

        self.cm2in = 0.3937008

# Create instance of mech
constants = set_constants()

def cartesian2spherical(
        x: _numpy.ndarray,
        y: Optional[_numpy.ndarray] = None,
        z: Optional[_numpy.ndarray] = None,
    ):
    """
    Convert Cartesian coordinates to latitude, longitude, magnitude, and azimuth.

    :param x:           X coordinate.
    :type x:            float, int, list, numpy.array, pandas.Series
    :param y:           Y coordinate.
    :type y:            float, int, list, numpy.array, pandas.Series
    :param z:           Z coordinate.
    :type z:            float, int, list, numpy.array, pandas.Series

    :return:            Latitude (degrees), Longitude (degrees), Magnitude, Azimuth (degrees).
    :rtype:             tuple (numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray)
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

def spherical2cartesian(
        lat,
        lon,
        mag = 1
    ):
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

# Define functions to test
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
    centroid_position = spherical2cartesian(plates.centroid_lat, plates.centroid_lon, constants.mean_Earth_radius_m)

    # Calculate the torque vector as the cross product of the Cartesian torque vector (x, y, z) with the position vector of the centroid
    force_at_centroid = _numpy.cross(summed_torques_cartesian, centroid_position, axis=0)

    # Compute force magnitude at centroid
    plates[f"{torque_type}_force_lat"], plates[f"{torque_type}_force_lon"], plates[f"{torque_type}_force_mag"], plates[f"{torque_type}_force_azi"] = cartesian2spherical(
        force_at_centroid[0], force_at_centroid[1], force_at_centroid[2]
    )

    return plates

def compute_synthetic_stage_rotation(
        plates: _pandas.DataFrame,
        options: Dict,
    ) -> _pandas.DataFrame:
    """
    Function to compute stage rotations.
    """
    # Sum the torque vectors (in Cartesian coordinates and in Newton metres)
    mantle_drag_torque_xyz = _numpy.column_stack((plates.mantle_drag_torque_x, plates.mantle_drag_torque_y, plates.mantle_drag_torque_z))

    # Get the centroid position on the unit sphere
    centroid_positions_before_xyz = _numpy.column_stack(spherical2cartesian(
        plates.centroid_lat, 
        plates.centroid_lon, 
    ))

    # logging.info(f"Mean, min and max of reconstructed stage rotation angles: {plates.pole_angle.mean()}, {plates.pole_angle.min()}, {plates.pole_angle.max()}")
    # 1. Normalize the centroid positions to ensure they are on the unit sphere
    # centroid_positions_before_xyz /= _numpy.linalg.norm(centroid_positions_before_xyz, axis=1)[:, _numpy.newaxis]

    # 2. Calculate the angular velocities (in radians per year) from torque
    # Convert the mantle drag torque to radians per year for each plate
    # stage_rotations_rad = -1 * mantle_drag_torque_xyz / (
    #     _numpy.repeat(_numpy.asarray(plates.area)[:, _numpy.newaxis], 3, axis=1) )
    # #     options["Mantle viscosity"] / mech.La
    # # )

    # Invert the torque to get the rotation
    stage_rotations_xyz = -1 * mantle_drag_torque_xyz / (options["Mantle viscosity"] / mech.La)

    # Convert to spherical coordinates
    stage_rotations_lat, stage_rotations_lon, stage_rotations_mag, _ = cartesian2spherical(
        stage_rotations_xyz[:, 0], stage_rotations_xyz[:, 1], stage_rotations_xyz[:, 2]
    )

    # Normalise magnitudes
    # stage_rotations_mag /= _numpy.linalg.norm(plates.area)

    # Assign the stage rotation angles to the plates DataFrame
    plates["pole_lat"] = stage_rotations_lat
    plates["pole_lon"] = stage_rotations_lon
    plates["pole_angle"] = _numpy.rad2deg(stage_rotations_mag) / plates.area * 1e6

    # Get the cross product of the rotation vector and the centroid position
    centroid_velocities_xyz = _numpy.cross(stage_rotations_xyz, centroid_positions_before_xyz)

    # Convert to spherical coordinates
    centroid_velocities_lat, centroid_velocities_lon, centroid_velocities_mag, _ = cartesian2spherical(
        centroid_velocities_xyz[:, 0], centroid_velocities_xyz[:, 1], centroid_velocities_xyz[:, 2]
    )

    # Assign the stage rotation angles to the plates DataFrame
    plates["centroid_velocity_mag"] = centroid_velocities_mag * constants.deg_a2cm_a

    return plates

    # # 3. Use the cross product to calculate the velocity vector in Cartesian coordinates
    # centroid_velocities_xyz = _numpy.cross(stage_rotations_rad, centroid_positions_before_xyz)

    # # Step 3: Create rotation vectors by scaling velocities for 1 million years
    # rotation_vectors = centroid_velocities_xyz * 1e6  # Radians for 1 million years

    # # Step 4: Use scipy's Rotation.from_rotvec to apply the rotations
    # rotation = R.from_rotvec(rotation_vectors)

    # # Step 5: Rotate the initial positions
    # centroid_positions_after_xyz = rotation.apply(centroid_positions_before_xyz)

    # # Step 6: Normalize to ensure the points remain on the unit sphere
    # centroid_positions_after_xyz /= _numpy.linalg.norm(centroid_positions_after_xyz, axis=1)[:, _numpy.newaxis]

    # # Step 7: Compute the dot product of the initial and final positions
    # dot_products = _numpy.einsum('ij,ij->i', centroid_positions_before_xyz, centroid_positions_after_xyz)

    # # Step 8: Clamp the dot product values to the range [-1, 1] to avoid NaNs
    # dot_products = _numpy.clip(dot_products, -1.0, 1.0)

    # # Step 9: Compute the rotation angles in degrees
    # stage_rotation_angles = _numpy.rad2deg(_numpy.arccos(dot_products))

    # plates["pole_angle"] = stage_rotation_angles

    # return plates

# %%
# Load data
plate_data_ref = _pandas.read_parquet("/Users/thomas/Documents/_Plato/Reconstruction_analysis/Output/M2016/Lr-Hb/Plates/Plates_Muller2016_ref_0Ma.parquet")
plate_data_syn = _pandas.read_parquet("/Users/thomas/Documents/_Plato/Reconstruction_analysis/Output/M2016/Lr-Hb/Plates/Plates_Muller2016_syn_0Ma.parquet")
plate_data = _pandas.read_parquet("/Users/thomas/Documents/_Plato/Plato/project/test/output/Plates/Plates_Muller2016_test_0Ma.parquet")
options = {"Mantle viscosity": 1.72e20}
plate_data_syn.sort_values("plateID", inplace=True, ignore_index=True)
plate_data_after = compute_synthetic_stage_rotation(plate_data.copy(), options)

plate_data_after.sort_values("plateID", inplace=True, ignore_index=True)
plt.scatter(plate_data_after["centroid_velocity_mag"], plate_data["centroid_velocity_mag"])
# plt.show()
# plt.scatter(plate_data_after["centroid_velocity_mag"], plate_data_syn["centroid_v_mag"])

plt.show()
# plt.scatter(plate_data_ref["plateID"].index, plate_data_after["pole_lat"]-plate_data_ref["pole_lat"])
# plt.scatter(plate_data_after["pole_lat"], antipode_pole_lat)
# plt.show()
# plt.scatter(
#     plate_data["mantle_drag_torque_x"]*(options["Mantle viscosity"]/200e3)/plate_data_ref["mantle_drag_torque_x"],
#     plate_data["area"]
#     )

# print(plate_data_after["pole_angle"]/plate_data_ref["pole_angle"])
# %%
