# %%
import gplately
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd
import warnings
from typing import Dict, Optional, Union
import cmcrameri as cmc
import cartopy.crs as ccrs

def great_circle_distance(lat1, lon1, lat2, lon2, radius=6371e3):
    """
    Calculate the great circle distance between two points on the Earth.
    
    Parameters:
        lat1, lon1: Latitude and Longitude of the first point (degrees)
        lat2, lon2: Latitude and Longitude of the second point (degrees)
        radius: Radius of the Earth (default: 6371 km)
        
    Returns:
        Distance in kilometers.
    """
    # Convert degrees to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    
    # Differences in coordinates
    delta_lat = lat2 - lat1
    delta_lon = lon2 - lon1
    
    # Haversine formula
    a = np.sin(delta_lat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(delta_lon / 2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    
    # Distance
    distance = radius * c

    return distance

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
    lon_radians = np.deg2rad(lon)
    lat_radians = np.deg2rad(lat)
    azimuth_radians = np.deg2rad(azimuth)

    # Angular distance in km
    angular_distance = distance / constants.mean_Earth_radius_km

    # Calculate sample points
    new_lat_radians = np.arcsin(np.sin(lat_radians) * np.cos(angular_distance) + np.cos(lat_radians) * np.sin(angular_distance) * np.cos(azimuth_radians))
    new_lon_radians = lon_radians + np.arctan2(np.sin(azimuth_radians) * np.sin(angular_distance) * np.cos(lat_radians), np.cos(angular_distance) - np.sin(lat_radians) * np.sin(new_lat_radians))
    new_lon = np.degrees(new_lon_radians)
    new_lat = np.degrees(new_lat_radians)

    return new_lat, new_lon

def calculate_azimuth(lat1, lon1, lat2, lon2):
    # Convert degrees to radians
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)
    
    # Compute the difference in longitude
    delta_lon = lon2 - lon1
    
    # Calculate the azimuth
    x = np.sin(delta_lon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(delta_lon)
    
    azimuth = np.arctan2(x, y)
    
    # Convert azimuth from radians to degrees
    azimuth = np.degrees(azimuth)
    
    # Normalize to 0°–360°
    azimuth = (azimuth + 360) % 360
    
    return azimuth

def great_circle_midpoint(lat1, lon1, lat2, lon2):
    # Convert degrees to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    
    # Cartesian coordinates for the two points
    x1, y1, z1 = np.cos(lat1) * np.cos(lon1), np.cos(lat1) * np.sin(lon1), np.sin(lat1)
    x2, y2, z2 = np.cos(lat2) * np.cos(lon2), np.cos(lat2) * np.sin(lon2), np.sin(lat2)
    
    # Midpoint in Cartesian coordinates
    x_m, y_m, z_m = (x1 + x2) / 2, (y1 + y2) / 2, (z1 + z2) / 2
    
    # Normalize to ensure it lies on the unit sphere
    norm = np.sqrt(x_m**2 + y_m**2 + z_m**2)
    x_m, y_m, z_m = x_m / norm, y_m / norm, z_m / norm
    
    # Convert back to latitude and longitude
    lat_m = np.arcsin(z_m)  # Latitude
    lon_m = np.arctan2(y_m, x_m)  # Longitude
    
    # Convert radians back to degrees
    lat_m, lon_m = map(np.degrees, [lat_m, lon_m])
    
    return lat_m, lon_m

def get_resolved_topologies(
        reconstruction: gplately.PlateReconstruction,
        age: Union[int, float, np.floating, np.integer],
        anchor_plateID: Optional[Union[int, float, np.integer, np.floating]] = 0,
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
        gplately.pygplates.resolve_topologies(
            reconstruction.topology_features,
            reconstruction.rotation_model,
            resolved_topologies,
            age,
            anchor_plate_id=int(anchor_plateID)
        )
    
    return resolved_topologies

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

        self.m_s2deg_Ma = np.rad2deg(self.m_s2rad_a) * 1e6  # m/s to deg/Ma
        self.rad_a2cm_a = self.mean_Earth_radius_m * 1e2  # rad/a to cm/a

        self.deg_a2cm_a = np.deg2rad(self.rad_a2cm_a) # deg/a to m/s

        self.cm2in = 0.3937008

# %%
path = "/Users/thomas/Documents/_Plato/Reconstruction_analysis/GPlates_files/"
rotations = f"{path}M2016/M2016_rotations_Lr-Hb.rot"
topology = f"{path}M2016/M2016_topologies.gpml"
coastlines = f"{path}M2016/M2016_coastlines.gpml"
constants = set_constants()

M2016 = gplately.PlateReconstruction(rotations, topology)
plot_M2016 = gplately.PlotTopologies(M2016, coastlines)

# ridges = M2016.tessellate_mid_ocean_ridges(time=0, tessellation_threshold_radians=250/constants.mean_Earth_radius_km)
# ridges = pd.DataFrame(ridges)
# ridges.columns = ["lon", "lat", "spreading_velocity_mag", "ridge_segment_length"]

# ridges.ridge_segment_length = constants.equatorial_Earth_circumference / 360

# %%
time = 0
resolved_topologies = get_resolved_topologies(M2016, time)
plot_M2016.time = time

lat = []; lon = []; type = []; azi = []; pol = []; length = []
left_sampling_lat, left_sampling_lon = [], []; right_sampling_lat = []; right_sampling_lon = []
for topology in resolved_topologies:
    sub_segments = topology.get_boundary_sub_segments()
    for sub_segment in sub_segments:
        coordinates = sub_segment.get_geometry().to_lat_lon_array()
        feature_type = sub_segment.get_feature().get_feature_type()
        if feature_type == gplately.pygplates.FeatureType.gpml_subduction_zone:
            polarity = sub_segment.get_feature().get_enumeration(gplately.pygplates.PropertyName.gpml_subduction_polarity)

        if len (coordinates) <=2:
            continue

        for i, coord in enumerate(coordinates[:-1]):
            # Get midpoint between coordinates
            midpoint = great_circle_midpoint(
                coord[0], coord[1],
                coordinates[i+1][0], coordinates[i+1][1]
            )
            lat.append(midpoint[0]); lon.append(midpoint[1])

            # Get great circle length between coordinates
            length.append(great_circle_distance(
                coord[0], coord[1],
                coordinates[i+1][0], coordinates[i+1][1]
            ))

            # Get azimuth between coordinates
            azimuth = calculate_azimuth(
                coord[0], coord[1],
                coordinates[i+1][0], coordinates[i+1][1]
            )
            if feature_type == gplately.pygplates.FeatureType.gpml_subduction_zone:
                if polarity == "Left":
                    azimuth = (azimuth - 90) % 360
                else:
                    azimuth = (azimuth + 90) % 360

            azi.append(azimuth)

            left_sampling_points = project_points(
                midpoint[0], midpoint[1], azimuth, -1e3
            )
            left_sampling_lat.append(left_sampling_points[0]); left_sampling_lon.append(left_sampling_points[1])

            right_sampling_points = project_points(
                midpoint[0], midpoint[1], azimuth, 1e3
            )
            right_sampling_lat.append(right_sampling_points[0]); right_sampling_lon.append(right_sampling_points[1])

            # Get feature type
            if feature_type == gplately.pygplates.FeatureType.gpml_subduction_zone:
                if polarity == "Left":
                    type.append(1)
                else:
                    type.append(0)
            elif feature_type == gplately.pygplates.FeatureType.gpml_mid_ocean_ridge:
                type.append(2)
            elif feature_type == gplately.pygplates.FeatureType.gpml_transform:
                type.append(3)
            elif feature_type == gplately.pygplates.FeatureType.gpml_fault:
                type.append(4)
            elif feature_type == gplately.pygplates.FeatureType.gpml_orogenic_belt:
                type.append(5)
            else:
                type.append(6)

# print(lat)
lat = np.array(lat); lon = np.array(lon); type = np.array(type); azi = np.array(azi); length = np.array(length); left_sampling_lat = np.array(left_sampling_lat); left_sampling_lon = np.array(left_sampling_lon); right_sampling_lat = np.array(right_sampling_lat); right_sampling_lon = np.array(right_sampling_lon)
# %%
mask = lat == lat
fig, ax = plt.subplots(subplot_kw={"projection": ccrs.Robinson()}, dpi=300)
# Define color normalization and levels
boundaries = [0, 1, 2, 3, 4, 5, 6, 7]  # Include an extra boundary for color mapping
tick_positions = [(boundaries[i] + boundaries[i + 1]) / 2 for i in range(len(boundaries) - 1)]  # Midpoints

# Use a discrete colormap with the same number of colors as levels
cmap = plt.cm.get_cmap("viridis", len(boundaries) - 1)  # Discrete colormap
norm = mcolors.BoundaryNorm(boundaries=boundaries, ncolors=cmap.N, clip=True)

# Add coastlines
ax.set_global()
plot_M2016.plot_coastlines(ax=ax, color="lightgrey")

# Plot scatter points
p = ax.scatter(
    lon[mask], lat[mask], 
    c=type[mask], norm=norm,
    cmap=cmap,
    transform=ccrs.PlateCarree(),
    s=1,
)

# Add a horizontal colorbar with centered ticks
cbar = fig.colorbar(
    p, ax=ax, orientation="horizontal", ticks=tick_positions, label="Type of plate boundary"
)
cbar.ax.set_xticklabels(["Right-\ndipping", "Left-\ndipping", "Ridge", "Transform", "Fault", "Collision", "Other"])

plt.show()
# %%
mask = type == 3
fig, ax = plt.subplots(subplot_kw={"projection": ccrs.Robinson()}, dpi=300)
# # Define color normalization and levels
# boundaries = [0, 1, 2, 3, 4, 5, 6]  # Include an extra boundary for color mapping
# tick_positions = [(boundaries[i] + boundaries[i + 1]) / 2 for i in range(len(boundaries) - 1)]  # Midpoints

# # Use a discrete colormap with the same number of colors as levels
# cmap = plt.cm.get_cmap("viridis", len(boundaries) - 1)  # Discrete colormap
# norm = mcolors.BoundaryNorm(boundaries=boundaries, ncolors=cmap.N, clip=True)

# Plot scatter points
p = ax.scatter(
    lon[mask], lat[mask], 
    c="k",
    transform=ccrs.PlateCarree(),
    s=1,
)

# Plot scatter points
p = ax.scatter(
    left_sampling_lon[mask], left_sampling_lat[mask], 
    c="r",
    transform=ccrs.PlateCarree(),
    s=1,
)

# Plot scatter points
p = ax.scatter(
    right_sampling_lon[mask], right_sampling_lat[mask], 
    c="b",
    transform=ccrs.PlateCarree(),
    s=1,
)

# Add coastlines
ax.set_global()
ax.coastlines(lw=0.5)

# Add a horizontal colorbar with centered ticks
# cbar = fig.colorbar(
#     p, ax=ax, orientation="horizontal", ticks=tick_positions, label="Type of plate boundary"
# )
# cbar.ax.set_xticklabels(["Right-dipping", "Left-dipping", "Ridge", "Transform", "Fault", "Other"])

plt.show()

# fig.colorbar(p, orientation="horizontal")

# %%
transforms_mask = type == 3
transforms = pd.DataFrame({
    "lat": lat[transforms_mask],
    "lon": lon[transforms_mask],
    "azi": azi[transforms_mask],
    "length": length[transforms_mask],
    "left_lat": left_sampling_lat[transforms_mask],
    "left_lon": left_sampling_lon[transforms_mask],
    "right_lat": right_sampling_lat[transforms_mask],
    "right_lon": right_sampling_lon[transforms_mask]
})

plt.scatter(
    transforms.lon, transforms.lat,
    c="k", s=1
)
plt.scatter(
    transforms.left_lon, transforms.left_lat,
    c="r", s=1
)
plt.scatter(
    transforms.right_lon, transforms.right_lat,
    c="b", s=1
)