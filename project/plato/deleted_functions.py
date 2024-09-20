def plot_velocity_map(
            self,
            ax,
            fig,
            reconstruction_time,
            case,
            plotting_options
        ):
        """
        Function to create subplot with plate velocities
            ax:                     axes object
            fig:                    figure
            reconstruction_time:    the time for which to display the map
            case:                   case for which to plot the sediments
            plotting_options:       dictionary with options for plotting
        """
        # Check if reconstruction time is in valid times
        if reconstruction_time not in self.times:
            return print("Invalid reconstruction time")
        
        # Set basemap
        ax, gl = self.plot_basemap(ax)

        # Plot plates and coastlines
        self.plot_reconstruction(ax, reconstruction_time, plotting_options, plates=True, trenches=False)

        # Get data
        plate_vectors = self.plates[reconstruction_time][case].loc[self.plates[reconstruction_time][case].area >= self.options[case]["Minimum plate area"]]
        slab_data = self.slabs[reconstruction_time][case].loc[self.slabs[reconstruction_time][case].lower_plateID.isin(plate_vectors.plateID)]
        slab_vectors = slab_data.iloc[::5]

        # Plot velocity magnitude at trenches
        vels = ax.scatter(
            slab_data.lon,
            slab_data.lat,
            c=slab_data.v_lower_plate_mag,
            s=plotting_options["marker size"],
            transform=ccrs.PlateCarree(),
            cmap=plotting_options["velocity magnitude cmap"],
            vmin=0,
            vmax=plotting_options["velocity max"]
        )

        # Plot velocity at subduction zones
        slab_vectors = ax.quiver(
            x=slab_vectors.lon,
            y=slab_vectors.lat,
            u=slab_vectors.v_lower_plate_lon,
            v=slab_vectors.v_lower_plate_lat,
            transform=ccrs.PlateCarree(),
            # label=vector.capitalize(),
            width=2e-3,
            scale=3e2,
            zorder=4,
            color='black'
        )

        # Plot velocity at centroid
        centroid_vectors = ax.quiver(
            x=plate_vectors.centroid_lon,
            y=plate_vectors.centroid_lat,
            u=plate_vectors.centroid_v_lon,
            v=plate_vectors.centroid_v_lat,
            transform=ccrs.PlateCarree(),
            # label=vector.capitalize(),
            width=5e-3,
            scale=3e2,
            zorder=4,
            color='white',
            edgecolor='black',
            linewidth=1
        )

        # Colourbar
        if plotting_options["cbar"] is True:
            fig.colorbar(vels, ax=ax, label="Velocity [cm/a]", orientation=plotting_options["orientation cbar"], shrink=0.75, aspect=20)
    
        return ax, vels, centroid_vectors, slab_vectors


def optimise_torques(self, sediments=True):
        """
        Function to apply optimised parameters to torques
        Arguments:
            opt_visc
            opt_sp_const
        """
        # Apply to each torque in DataFrame
        axes = ["_x", "_y", "_z", "_mag"]
        for case in self.cases:
            for axis in axes:
                self.plates[case]["slab_pull_torque_opt" + axis] = self.options[case]["Slab pull constant"] * self.plates[case]["slab_pull_torque" + axis]
                if self.options[case]["Reconstructed motions"]:
                    self.plates[case]["mantle_drag_torque_opt" + axis] = self.options[case]["Mantle viscosity"] * self.plates[case]["mantle_drag_torque" + axis]
                
                for reconstruction_time in self.times:
                    if sediments == True:
                        self.plates[reconstruction_time][case]["slab_pull_torque_opt" + axis] = self.options[case]["Slab pull constant"] * self.plates[reconstruction_time][case]["slab_pull_torque" + axis]
                    if self.options[case]["Reconstructed motions"]:
                        self.plates[reconstruction_time][case]["mantle_drag_torque_opt" + axis] = self.options[case]["Mantle viscosity"] * self.plates[reconstruction_time][case]["mantle_drag_torque" + axis]

        # Apply to forces at centroid
        coords = ["lon", "lat"]
        for reconstruction_time in self.times:
            for case in self.cases:
                for coord in coords:
                    self.plates[reconstruction_time][case]["slab_pull_force_opt" + coord] = self.options[case]["Slab pull constant"] * self.plates[reconstruction_time][case]["slab_pull_force" + coord]
                    if self.options[case]["Reconstructed motions"]:
                        self.plates[reconstruction_time][case]["mantle_drag_force_opt" + coord] = self.options[case]["Mantle viscosity"] * self.plates[reconstruction_time][case]["slab_pull_force" + coord]

        self.optimised_torques = True


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# PLATO
# Algorithm to calculate plate forces from tectonic reconstructions
# Initialiser for accelerated parallel initialisation of plate, point and slab data
# Thomas Schouten, 2024
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Import libraries
# Standard libraries
import os as _os
import warnings
from typing import List, Optional, Union

# Third-party libraries
import numpy as _numpy
import matplotlib.pyplot as plt
import geopandas as _gpd
import gplately
from gplately import pygplates as _pygplates
import cartopy.crs as ccrs
import cmcrameri as cmc
from tqdm import tqdm
import xarray as _xarray
from time import time

# Local libraries
import setup
import setup_parallel
import functions_main

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# INITIALISER
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def initialise_plato(
        reconstruction_name: str, 
        reconstruction_times: List[int] or _numpy.array, 
        cases_file: str, 
        cases_sheet: Optional[str] = "Sheet1", 
        files_dir: Optional[str] = None,
        rotation_file: Optional[List[str]] = None,
        topology_file: Optional[List[str]] = None,
        polygon_file: Optional[List[str]] = None,
    ):

    # Store cases and case options
    cases, options = setup.get_options(cases_file, cases_sheet)

    # Group cases for initialisation of slabs and points
    slab_options = ["Slab tesselation spacing"]
    slab_cases = setup.process_cases(cases, options, slab_options)
    point_options = ["Grid spacing"]
    point_cases = setup.process_cases(cases, options, point_options)

    # Convert GPlates file inputs to GPlates objects
    rotations = _pygplates.RotationModel(rotation_file)
    topologies = _pygplates.FeatureCollection(topology_file)
    polygons = _pygplates.FeatureCollection(polygon_file)

    reconstruction = gplately.PlateReconstruction(rotations, topologies, polygons)
    
    for reconstruction_time in tqdm(reconstruction_times, desc="Initialising and saving files"):
        # Get geometries of plates
        resolved_geometries = setup.get_topology_geometries(
            reconstruction, reconstruction_time, anchor_plateID=0
        )

        # Resolve topologies to use to get plates
        # NOTE: This is done because some information is retrieved from the resolved topologies and some from the resolved geometries
        #       This step could be sped up by extracting all information from the resolved geometries, but so far this has not been the main bottleneck
        # Ignore annoying warnings
        with warnings.catch_warnings():
            warnings.filterwarnings(
                action="ignore",
                message="Normalized/laundered field name:"
            )
            resolved_topologies = []
            _pygplates.resolve_topologies(
                topologies,
                rotations, 
                resolved_topologies, 
                reconstruction_time, 
                anchor_plate_id=0
            )

        # Get plates
        plates = setup_parallel.get_plates(
                    reconstruction.rotation_model,
                    reconstruction_time,
                    resolved_topologies,
                    options[cases[0]],
                )
        
        # Get slabs
        slabs = {}
        for key, entries in slab_cases.items():
            slabs[key] = setup_parallel.get_slabs(
                        reconstruction,
                        reconstruction_time,
                        plates,
                        resolved_geometries,
                        options[key],
                    )
            
            # Copy DataFrames to other cases
            for entry in entries[1:]:
                slabs[entry] = slabs[key].copy()

        # Get points
        points = {}
        for key, entries in point_cases.items():
            points[key] = setup_parallel.get_points(
                        reconstruction,
                        reconstruction_time,
                        plates,
                        resolved_geometries,
                        options[key],
                    )

            # Copy DataFrames to other cases
            for entry in entries[1:]:
                points[entry] = points[key].copy()

        # Save all data to files
        for case in cases:
            # Save plates
            setup.DataFrame_to_parquet(
                plates,
                "Plates",
                reconstruction_name,
                reconstruction_time,
                case,
                files_dir,
            )

            # Save slabs
            setup.DataFrame_to_parquet(
                slabs[case],
                "Slabs",
                reconstruction_name,
                reconstruction_time,
                case,
                files_dir,
            )

            # Save points
            setup.DataFrame_to_parquet(
                points[case],
                "Points",
                reconstruction_name,
                reconstruction_time,
                case,
                files_dir,
            )

# if __name__ == "__main__":
#     # Set reconstruction times
#     reconstruction_times = _numpy.arange(31, 41, 1)

#     # Define path to directory containing reconstruction files
#     main_dir = _os.path.join("", "Users", "thomas", "Documents", "_Plato", "Reconstruction_analysis")

#     # Define the parameter settings for the model
#     settings_file = _os.path.join(main_dir, "settings.xlsx")

#     settings_sheet = "Sheet2"
    
#     M2016_model_dir = _os.path.join(main_dir, "M2016")

#     # Get start time
#     start_time = time()

#     # Set up PlateForces object
#     initialise_plato(
#         "Muller2016",
#         reconstruction_times,
#         settings_file,
#         cases_sheet = settings_sheet,
#         files_dir = _os.path.join("Output", M2016_model_dir, "Lr-Hb"),
#         rotation_file = _os.path.join("GPlates_files", "M2016", f"M2016_rotations_Lr-Hb.rot"),
#         topology_file = _os.path.join("GPlates_files", "M2016", "M2016_topologies.gpml"),
#         polygon_file = _os.path.join("GPlates_files", "M2016", "M2016_polygons.gpml"),
#     )

#     # Print the time taken to set up the model
#     set_up_time = time() - start_time
#     print(f"Time taken to set up the model: {set_up_time:.2e} seconds")