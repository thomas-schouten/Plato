import numpy as np

def compute_torque_on_plates(torques, lat, lon, plateID, force_lat, force_lon, segment_length_lat, segment_length_lon, constants, torque_variable="torque", DEBUG_MODE=False):
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
    data[torque_variable + "_mag"] = np.sqrt(
        torques_cartesian[0] ** 2 + torques_cartesian[1] ** 2 + torques_cartesian[2] ** 2
    )

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
    summed_torques_cartesian = np.asarray([torques[torque_variable + "_x"], torques[torque_variable + "_y"], torques[torque_variable + "_z"]])
    force_at_centroid = np.cross(summed_torques_cartesian, centroid_position, axis=0) 

    if DEBUG_MODE:
        print(f"Computing torque at centroid: {force_at_centroid}")

    # Compute force magnitude at centroid
    force_variable = torque_variable.replace("torque", "force")
    torques[force_variable + "_lat"], torques[force_variable + "_lon"], torques[force_variable + "_mag"], torques[force_variable + "_azi"] = vector_xyz2lat_lon(
        torques.centroid_lat, torques.centroid_lon, force_at_centroid, constants
    )
    
    return torques

def compute_torque(plate, plate_slabs, selected_slabs, sp_consts, i, constants):
    # Calculate forces for all segments
    force_lat = plate_slabs.slab_pull_force_lat.values * sp_consts[i]
    force_lon = plate_slabs.slab_pull_force_lon.values * sp_consts[i]

    # Compute total torque
    computed_plates = functions_main.compute_torque_on_plates(
        plate, 
        selected_slabs.lat, 
        selected_slabs.lon, 
        selected_slabs.lower_plateID, 
        force_lat, 
        force_lon, 
        selected_slabs.trench_segment_length,
        1,
        constants,
        torque_variable="slab_pull_torque"
    )

    return (
        computed_plates.slab_pull_torque_x, 
        computed_plates.slab_pull_torque_y, 
        computed_plates.slab_pull_torque_z
    )

def compute_torque_without_segment(j, plate, plate_slabs, selected_slabs, sp_consts, i, constants):
    # Exclude segment j's contribution by zeroing its force
    force_lat = plate_slabs.slab_pull_force_lat.values * sp_consts[i]
    force_lon = plate_slabs.slab_pull_force_lon.values * sp_consts[i]

    # Set forces for the j-th segment to 0
    force_lat[j] = 0
    force_lon[j] = 0

    # Compute torque without segment j
    computed_plates = functions_main.compute_torque_on_plates(
        plate, 
        selected_slabs.lat, 
        selected_slabs.lon, 
        selected_slabs.lower_plateID, 
        force_lat, 
        force_lon, 
        selected_slabs.trench_segment_length,
        1,
        constants,
        torque_variable="slab_pull_torque"
    )

    return (
        j, 
        computed_plates.slab_pull_torque_x, 
        computed_plates.slab_pull_torque_y, 
        computed_plates.slab_pull_torque_z
    )

# Calculate total torque for a given coefficient i
total_torque_x, total_torque_y, total_torque_z = compute_torque(
    plate, plate_slabs, selected_slabs, sp_consts, i, self.constants
)

# Parallelized calculation of torques without each segment
results = Parallel(n_jobs=-1)(
    delayed(compute_torque_without_segment)(
        j, plate, plate_slabs, selected_slabs, sp_consts, i, self.constants
    )
    for j in range(len(plate_slabs))
)

# Calculate relative contributions
for j, partial_x, partial_y, partial_z in results:
    contribution_x = total_torque_x - partial_x
    contribution_y = total_torque_y - partial_y
    contribution_z = total_torque_z - partial_z

    print(f"Segment {j} contribution to torque_x: {contribution_x / total_torque_x:.2%}")
    print(f"Segment {j} contribution to torque_y: {contribution_y / total_torque_y:.2%}")
    print(f"Segment {j} contribution to torque_z: {contribution_z / total_torque_z:.2%}")
