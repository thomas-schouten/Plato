# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from scipy.optimize import lsq_linear

def spherical2cartesian(lat, lon, mag):
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
    lat_rads = np.deg2rad(lat)
    lon_rads = np.deg2rad(lon)

    # Calculate position vectors
    position = mag * np.asarray([np.cos(lat_rads) * np.cos(lon_rads), np.cos(lat_rads) * np.sin(lon_rads), np.sin(lat_rads)])

    return position

def force2torque(position, lat, lon, force_lat, force_lon, segment_length_lat, segment_length_lon):
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
    lon_rads = np.deg2rad(lon)
    lat_rads = np.deg2rad(lat)

    # Calculate force_magnitude
    force_magnitude = np.sqrt((force_lat*segment_length_lat*segment_length_lon)**2 + (force_lon*segment_length_lat*segment_length_lon)**2)

    theta = np.where(
        (force_lon >= 0) & (force_lat >= 0),                     
        np.arctan(force_lat/force_lon),                          
        np.where(
            (force_lon < 0) & (force_lat >= 0) | (force_lon < 0) & (force_lat < 0),    
            np.pi + np.arctan(force_lat/force_lon),              
            (2*np.pi) + np.arctan(force_lat/force_lon)           
        )
    )

    force_x = force_magnitude * np.cos(theta) * (-1.0 * np.sin(lon_rads))
    force_y = force_magnitude * np.cos(theta) * np.cos(lon_rads)
    force_z = force_magnitude * np.sin(theta) * np.cos(lat_rads)

    force = np.asarray([force_x, force_y, force_z])

    # Calculate torque
    torque = np.cross(position, force, axis=0)

    return torque 

def compute_ksi(omega, GPE_torque, slab_pull_force, r_positions):
    """
    Computes the ksi vector based on the given omega, GPE torque, and slab pull force.
    
    Args:
    omega: np.array of shape (3,) representing the angular velocity vector.
    GPE_torque: np.array of shape (3,) representing the gravitational potential energy torque vector.
    slab_pull_force: np.array of shape (3, N), where N is the number of discretized points.
    r_positions: np.array of shape (3, N) representing the position vectors for each point on Earth.
    
    Returns:
    ksi: np.array of shape (N,) representing the ksi vector.
    """
    # Ensure that the inputs are numpy arrays
    omega = np.array(omega)
    GPE_torque = np.array(GPE_torque)
    slab_pull_force = np.array(slab_pull_force)
    r_positions = np.array(r_positions)

    # Check dimensions of the slab pull force and position vectors
    if slab_pull_force.shape[0] != 3 or r_positions.shape[0] != 3:
        raise ValueError("Both slab pull force and position vectors must have shape (3, N)")
    
    N = slab_pull_force.shape[1]  # Number of discretized points

    # Compute the residual term (omega - GPE_torque)
    residual = omega - GPE_torque  # Shape (3,)

    # Prepare to store the cross product results
    final_term = np.zeros((3, N))

    # Compute the double cross product for each column
    for i in range(N):
        cross1 = np.cross(slab_pull_force[:, i], r_positions[:, i])  # F_sp x r
        final_term[:, i] = np.cross(cross1, r_positions[:, i])  # (F_sp x r) x r

    # Set up the least squares system: we want to solve residual = A * ksi
    A = final_term  # Shape (N, 3)

    print("A shape:", A.shape)
    print("Residual shape:", residual.shape)

    # Solve for ksi using least squares
    result = lsq_linear(A, residual, bounds=(0.025, 1))  # Solve non-negative least squares

    ksi = result.x
    
    return ksi

# %%
# Load data for slabs and plates at 60 Ma
path = "/Users/thomas/Documents/_Plato/Reconstruction_analysis/Output/M2016/Lr-Hb/"

slabs = pd.read_parquet(path + "Slabs/Slabs_Muller2016_ref_80Ma.parquet")
plates = pd.read_parquet(path + "Plates/Plates_Muller2016_ref_80Ma.parquet")

# Select a plate
plateID = 902
plates = plates[plates.plateID == plateID]
slabs = slabs[slabs.lower_plateID == plateID]

r_positions = spherical2cartesian(slabs.lat, slabs.lon, 6.371e3)

slab_pull_force = force2torque(
    r_positions, 
    slabs.lat, 
    slabs.lon, 
    slabs.slab_pull_force_lat, 
    slabs.slab_pull_force_lon, 
    slabs.trench_segment_length, 
    1
)

gpe_torque = np.array(
    [plates.GPE_torque_x.values[0], 
        plates.GPE_torque_y.values[0], 
        plates.GPE_torque_z.values[0]]
    )

mantle_drag_torque = np.array(
    [plates.mantle_drag_torque_x.values[0],
        plates.mantle_drag_torque_y.values[0],
        plates.mantle_drag_torque_z.values[0]]
)

omega = spherical2cartesian(
    plates.pole_lat.values[0], 
    plates.pole_lat.values[0], 
    plates.pole_angle.values[0]
)

# %%
ksi_values = np.linspace(0.05, 1, 25)
# residual = np.zeros((len(ksi_values), len(plates), len(trench_plateID)))
for i, ksi_value in enumerate(ksi_values):
    for j, trench_plateID in enumerate(slabs.trench_plateID.unique()):
        for k, _ in enumerate(slabs[slabs.trench_plateID == trench_plateID].lat.values):
            print(i, j, k)
        

# %%
# For a range of ksi values, compute the residual

residuals = []

sp_torque = np.zeros((3, len(ksi_values)))
for i, ksi_value in enumerate(ksi_values):
# Loop through each subduction segment
    for lat, lon, force_lat, force_lon, segment_length_lat in zip(
        slabs.lat, 
        slabs.lon, 
        slabs.slab_pull_force_lat, 
        slabs.slab_pull_force_lon, 
        slabs.trench_segment_length
        ):
        # Calculate the force vector
        torque = force2torque(
            spherical2cartesian(lat, lon, 6.371e3), 
            lat, 
            lon, 
            force_lat, 
            force_lon, 
            segment_length_lat, 
            1
        )

        # Calculate the torque vector
        torque *= ksi_value

        sp_torque[:, i] += torque

    # Driving torque
    driving_torque = gpe_torque + sp_torque[:, i]

    # Calculate the residual
    residual = mantle_drag_torque - driving_torque

    print(residual/driving_torque)

    residuals.append(np.linalg.norm(residual))

fig, ax = plt.subplots()
ax.plot(ksi_values, residuals)
ax.set_xlabel("ksi")
ax.set_ylabel("Residual")
plt.show()

# # Example usage
# ksi = compute_ksi(omega, gpe_torque, slab_pull_force, r_positions)
# print("Computed ksi:", ksi)

fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()})
p = ax.scatter(
    slabs.lon, 
    slabs.lat, 
    c=ksi, 
)
fig.colorbar(p)
# %%
# Inspect data
plt.scatter(
    slabs.lon, 
    slabs.lat, 
    c=slab_pull_force[2], 
)

# %%