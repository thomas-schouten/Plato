# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# PLATO
# Algorithm to calculate plate forces from tectonic reconstructions
# PlateForces object
# Thomas Schouten and Edward Clennett, 2023
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Import libraries
# Standard libraries
import os
import multiprocessing
import warnings
from typing import List, Optional, Union
from copy import deepcopy

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

# Local libraries
import setup
import functions_main
import shutil
import sys
import tempfile

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# PLATE FORCES OBJECT
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

class PlateForces():
    def __init__(
            self,
            reconstruction_name: str, 
            reconstruction_times: List[int] or _numpy.array, 
            cases_file: str, 
            cases_sheet: Optional[str] = "Sheet1", 
            files_dir: Optional[str] = None,
            rotation_file: Optional[List[str]] = None,
            topology_file: Optional[List[str]] = None,
            polygon_file: Optional[List[str]] = None,
            coastline_file: Optional[str] = None,
            seafloor_grids: Optional[dict] = None,
            plates_of_interest: Optional[List[int]] = None,
            DEBUG_MODE: Optional[bool] = False,
            PARALLEL_MODE: Optional[bool] = False,
        ):
        """
        PlateForces object.

        This object can be instantiated in two ways: either by loading previously stored files, or by generating new files.

        :param reconstruction_name:     Name of the plate reconstruction.
        :type reconstruction_name:      str
        :param reconstruction_times:    List or array of reconstruction times.
        :type reconstruction_times:     list or numpy.array
        :param cases_file:              Path to the file containing cases data.
        :type cases_file:               str
        :param cases_sheet:             Sheet name in the cases file (default is "Sheet1").
        :type cases_sheet:              str
        :param files_dir:               Directory for storing/loading files (default is None).
        :type files_dir:                str
        :param rotation_file:           Path to the rotation file (default is None).
        :type rotation_file:            str
        :param topology_file:           Path to the topology file (default is None).
        :type topology_file:            str
        :param polygon_file:            Path to the polygon file (default is None).
        :type polygon_file:             str
        :param coastline_file:          Path to the coastline file (default is None).
        :type coastline_file:           str
        :param seafloor_grids:          Dictionary of seafloor grids (default is None).
        :type seafloor_grids:           dict
        """
        # If no seafloor age grids are provided, check if they are available on the GPLately DataServer
        supported_models = ["Seton2012", "Muller2016", "Muller2019", "Clennett2020"]
        if reconstruction_name not in supported_models and not seafloor_grids:
            print(f"No seafloor grid provided, and no grids for the {reconstruction_name} reconstruction are available from GPlately. Exiting now...")
            sys.exit()

        # Set flag for debugging mode
        self.DEBUG_MODE = DEBUG_MODE

        # Set flag for parallel mode
        self.PARALLEL_MODE = PARALLEL_MODE

        # Set files directory
        self.dir_path = os.path.join(os.getcwd(), files_dir)
        if not os.path.exists(self.dir_path):
            os.makedirs(self.dir_path)

        # Store reconstruction name and valid reconstruction times
        self.name = reconstruction_name
        self.times = _numpy.array(reconstruction_times)

        # Let the user know you're busy setting up the plate reconstruction
        print("Setting up plate reconstruction...")

        # Set connection to GPlately DataServer if one of the required files is missing
        if not rotation_file or not topology_file or not polygon_file or not coastline_file or not seafloor_grids:
            gdownload = gplately.DataServer(reconstruction_name)

        # Download reconstruction files if rotation or topology file not provided
        if not rotation_file or not topology_file or not polygon_file:
            print(f"Downloading {reconstruction_name} reconstruction files from the GPlately DataServer...")
            reconstruction_files = gdownload.get_plate_reconstruction_files()
        
        # Initialise or download RotationModel object
        if rotation_file:
            self.rotations = _pygplates.RotationModel(rotation_file)
        else: 
            self.rotations = reconstruction_files[0]

        # Initialise or download topology FeatureCollection object
        if topology_file:
            self.topologies = _pygplates.FeatureCollection(topology_file)
        else:
            self.topologies = reconstruction_files[1]

        # Initialise or download static polygons
        if polygon_file:
            self.polygons = _pygplates.FeatureCollection(polygon_file)
        else:
            self.polygons = reconstruction_files[2]
        
        # Initialise or download coastlines
        if coastline_file:
            self.coastlines = _pygplates.FeatureCollection(coastline_file)
        else:
            self.coastlines, _, _ = gdownload.get_topology_geometries()

        # Set up plate reconstruction object and initialise dictionaries to store resolved topologies and geometries
        self.reconstruction = gplately.PlateReconstruction(self.rotations, self.topologies, self.polygons)
        self.resolved_topologies, self.resolved_geometries = {}, {}

        # Load or initialise plate geometries
        for reconstruction_time in tqdm(self.times, desc="Loading geometries", disable=self.DEBUG_MODE):
            
            # Load resolved geometries if they are available
            self.resolved_geometries[reconstruction_time] = setup.GeoDataFrame_from_shapefile(
                self.dir_path,
                "Geometries",
                reconstruction_time,
                self.name,
            )

            # Get new topologies if they are unavailable
            if self.resolved_geometries[reconstruction_time] is None:
                self.resolved_geometries[reconstruction_time] = setup.get_topology_geometries(
                    self.reconstruction, reconstruction_time, anchor_plateID=0
                )
            
            # Resolve topologies to use to get plates
            # NOTE: This is done because some information is retrieved from the resolved topologies and some from the resolved geometries
            #       This step could be sped up by extracting all information from the resolved geometries
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    action="ignore",
                    message="Normalized/laundered field name:"
                )
                self.resolved_topologies[reconstruction_time] = []
                _pygplates.resolve_topologies(
                    self.topologies,
                    self.rotations, 
                    self.resolved_topologies[reconstruction_time], 
                    reconstruction_time, 
                    anchor_plate_id=0
                )
            
        print("Plate reconstruction ready!")

        # Set plates of interest
        if plates_of_interest:
            self.plates_of_interest = plates_of_interest
        else:
            self.plates_of_interest = [101, 201, 301, 501, 511, 801, 901, 909, 911, 918, 926, 902]

        # Store cases and case options
        self.cases, self.options = setup.get_options(cases_file, cases_sheet)

        # Set mechanical parameters and constants
        self.mech = functions_main.set_mech_params()
        self.constants = functions_main.set_constants()

        # Subdivide cases to accelerate computation
        # Group cases for initialisation of plates, slabs, and points
        plate_options = ["Minimum plate area"]
        self.plate_cases = setup.process_cases(self.cases, self.options, plate_options)
        slab_options = ["Slab tesselation spacing"]
        self.slab_cases = setup.process_cases(self.cases, self.options, slab_options)
        point_options = ["Grid spacing"]
        self.point_cases = setup.process_cases(self.cases, self.options, point_options)

        # Group cases for torque computation
        slab_pull_options = [
            "Slab pull torque",
            "Seafloor age profile",
            "Sample sediment grid",
            "Active margin sediments",
            "Sediment subduction",
            "Sample erosion grid",
            "Slab pull constant",
            "Shear zone width",
            "Slab length"
        ]
        self.slab_pull_cases = setup.process_cases(self.cases, self.options, slab_pull_options)

        slab_bend_options = ["Slab bend torque", "Seafloor age profile"]
        self.slab_bend_cases = setup.process_cases(self.cases, self.options, slab_bend_options)

        gpe_options = ["Continental crust", "Seafloor age profile", "Grid spacing"]
        self.gpe_cases = setup.process_cases(self.cases, self.options, gpe_options)

        mantle_drag_options = ["Reconstructed motions", "Grid spacing"]
        self.mantle_drag_cases = setup.process_cases(self.cases, self.options, mantle_drag_options)

        # Load or initialise dictionaries with DataFrames for plates, slabs, points and torque information.
        self.plates = {}
        self.slabs = {}
        self.points = {}
        self.torques = {}
        
        # Set up dictionaries for seafloor and velocity grids
        if not seafloor_grids:
            self.seafloor = {}
        else:
            self.seafloor = seafloor_grids
        self.velocity = {}

        # Load or initialise plates
        self.plates = setup.load_data(
            self.plates,
            self.reconstruction,
            self.name,
            self.times,
            "Plates",
            self.cases,
            self.options,
            self.plate_cases,
            files_dir,
            resolved_topologies = self.resolved_topologies,
            resolved_geometries = self.resolved_geometries,
            DEBUG_MODE = self.DEBUG_MODE,
            PARALLEL_MODE = self.PARALLEL_MODE,
        )

        # Load or initialise slabs
        self.slabs = setup.load_data(
            self.slabs,
            self.reconstruction,
            self.name,
            self.times,
            "Slabs",
            self.cases,
            self.options,
            self.slab_cases,
            files_dir,
            plates = self.plates,
            resolved_geometries = self.resolved_geometries,
            DEBUG_MODE = self.DEBUG_MODE,
            PARALLEL_MODE = self.PARALLEL_MODE,
        )

        # Load or initialise points
        self.points = setup.load_data(
            self.points,
            self.reconstruction,
            self.name,
            self.times,
            "Points",
            self.cases,
            self.options,
            self.point_cases,
            files_dir,
            plates = self.plates,
            resolved_geometries = self.resolved_geometries,
            DEBUG_MODE = self.DEBUG_MODE,
            PARALLEL_MODE = self.PARALLEL_MODE,
        )

        # Load torques
        self.torques = setup.load_torques(
            self.torques,
            self.times,
            self.cases,
            self.plates,
            self.plates_of_interest,
            DEBUG_MODE = self.DEBUG_MODE,
        )

        # Load or initialise seafloor
        self.seafloor = setup.load_grid(
            self.seafloor,
            self.name,
            self.times,
            "Seafloor",
            files_dir,
            DEBUG_MODE = self.DEBUG_MODE
        )

        # Load or initialise velocity grid
        self.velocity = setup.load_grid(
            self.velocity,
            self.name,
            self.times,
            "Velocity",
            files_dir,
            points = self.points,
            seafloor_grid = self.seafloor,
            cases = self.cases,
            DEBUG_MODE = self.DEBUG_MODE,
        )

        # Set sampling flags to False:
        self.sampled_points = False
        self.sampled_upper_plates = False
        self.sampled_slabs = False
        self.optimised_torques = False

        # Initialise dictionaries to store calibration parameters
        self.residual_torque = {}; self.residual_torque_normalised = {}
        self.driving_torque = {};  self.driving_torque_normalised = {}
        self.opt_sp_const = {}; self.opt_visc = {}
        self.opt_i = {}; self.opt_j = {}

        print("PlateForces object successfully instantiated!")

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# RESETTING OBJECT
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    def reset(self):
        """
        Function to reset the object
        """
        # Reset plates, slabs, points, and seafloor

        # Initialise other columns to store seafloor ages and forces
        torques = ["slab_pull", "GPE", "slab_bend", "mantle_drag"]
        slab_forces = ["slab_pull", "slab_bend", "interface_shear"]
        point_forces = ["GPE", "mantle_drag"]
        axes = ["x", "y", "z", "mag"]
        coords = ["lat", "lon", "mag"]
        
        # Upper plate
        for reconstruction_time in tqdm(self.times, desc="Resetting plates, slabs, and points", disable=self.DEBUG_MODE):
            for case in self.cases:
                # Reset plates
                self.plates[reconstruction_time][case][[torque + "_torque_" + axis for torque in torques for axis in axes]] = [[_numpy.nan] * len(torques) * len(axes) for _ in range(len(self.plates[reconstruction_time][case].plateID))]
                self.plates[reconstruction_time][case][["slab_pull_torque_opt_" + axis for axis in axes]] = [[_numpy.nan] * len(axes) for _ in range(len(self.plates[reconstruction_time][case].plateID))]
                self.plates[reconstruction_time][case][[torque + "_force_" + coord for torque in torques for coord in coords]] = [[_numpy.nan] * len(torques) * len(coords) for _ in range(len(self.plates[reconstruction_time][case].plateID))]
                self.plates[reconstruction_time][case][["slab_pull_force_opt_" + coord for coord in coords]] = [[_numpy.nan] * len(coords) for _ in range(len(self.plates[reconstruction_time][case].plateID))]

                # Reset slabs
                self.slabs[reconstruction_time][case]["upper_plate_thickness"] = _numpy.nan
                self.slabs[reconstruction_time][case]["upper_plate_age"] = _numpy.nan   
                self.slabs[reconstruction_time][case]["continental_arc"] = False
                self.slabs[reconstruction_time][case]["erosion_rate"] = _numpy.nan
                self.slabs[reconstruction_time][case]["lower_plate_age"] = _numpy.nan
                self.slabs[reconstruction_time][case]["lower_plate_thickness"] = _numpy.nan
                self.slabs[reconstruction_time][case]["sediment_thickness"] = _numpy.nan
                self.slabs[reconstruction_time][case]["sediment_fraction"] = 0.
                self.slabs[reconstruction_time][case][[force + "_force_" + coord for force in slab_forces for coord in coords]] = [[_numpy.nan] * 9 for _ in range(len(self.slabs[reconstruction_time][case]))] 

                # Reset points
                self.points[reconstruction_time][case]["seafloor_age"] = _numpy.nan
                self.points[reconstruction_time][case][[force + "_force_" + coord for force in point_forces for coord in coords]] = [[_numpy.nan] * 12 for _ in range(len(self.points[reconstruction_time][case]))]

        # Reset flags
        self.sampled_points = False
        self.sampled_upper_plates = False
        self.sampled_slabs = False
        self.optimised_torques = False

        # Reset optimisations
        self.residual_torque = {}; self.residual_torque_normalised = {}
        self.driving_torque = {};  self.driving_torque_normalised = {}
        self.opt_sp_const = {}; self.opt_visc = {}
        self.opt_i = {}; self.opt_j = {}

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ADDING GRIDS 
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    def add_grid(self, input_grids, variable_name, target_variable="z", cut_to_seafloor=True, prefactor=1):
        """
        Function to add another grid of a variable to the seafloor grid.
        The grids should be organised in a dictionary with each item being an xarray.Dataset with each key being the corresponding reconstruction time.
        Cut_to_seafloor is a boolean that determines whether or not to cut the grids to the seafloor. It should not be used for continental erosion rate grids.

        :param target_grid:             target grid to add the variable to
        :type target_grid:              xarray.DataArray
        :param input_grids:             dictionary of input grids
        :type input_grids:              dict
        :param variable_name:           name of the variable to add
        :type variable_name:            str
        :param target_variable:         name of the variable to add
        :type target_variable:          str
        :param cut_to_seafloor:         whether or not to cut the grids to the seafloor
        :type cut_to_seafloor:          bool
        """
        
        # Loop through times to load, interpolate, and store variables in seafloor grid
        input_grids_interpolated = {}
        for reconstruction_time in tqdm(self.times, desc=f"Adding {variable_name} grid", disable=self.DEBUG_MODE):
            if self.DEBUG_MODE:
                print(f"Adding {variable_name} grid for {reconstruction_time} Ma...")

            # Rename latitude and longitude if necessary
            if "lat" in list(input_grids[reconstruction_time].coords.keys()):
                input_grids[reconstruction_time] = input_grids[reconstruction_time].rename({"lat": "latitude"})
            if "lon" in list(input_grids[reconstruction_time].coords.keys()):
                input_grids[reconstruction_time] = input_grids[reconstruction_time].rename({"lon": "longitude"})
            
            # Check if target_variable exists in input grids
            if target_variable in input_grids[reconstruction_time].variables:
                # Interpolate input grids to seafloor grid
                input_grids_interpolated[reconstruction_time] = input_grids[reconstruction_time].interp_like(self.seafloor[reconstruction_time]["seafloor_age"], method="nearest")
                self.seafloor[reconstruction_time][variable_name] = input_grids_interpolated[reconstruction_time][target_variable] * prefactor

                # Align grids in seafloor
                if cut_to_seafloor:
                    mask = {}
                    for variable_1 in self.seafloor[reconstruction_time].data_vars:
                        mask[variable_1] = _numpy.isnan(self.seafloor[reconstruction_time][variable_1].values)

                        # Apply masks to all grids
                        for variable_2 in self.seafloor[reconstruction_time].data_vars:
                            self.seafloor[reconstruction_time][variable_2] = self.seafloor[reconstruction_time][variable_2].where(~mask[variable_1])

            else:
                if self.DEBUG_MODE:
                    print(f"Target variable '{target_variable}' does not exist in the input grids for {reconstruction_time} Ma.")

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# SAMPLING GRIDS 
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    def sample_slabs(self):
        """
        Samples seafloor age (and optionally, sediment thickness) the lower plate along subduction zones
        The results are stored in the `slabs` DataFrame, specifically in the `lower_plate_age`, `sediment_thickness`, and `lower_plate_thickness` fields for each case and reconstruction time.
        """
        # Check options for slabs
        for reconstruction_time in tqdm(self.times, desc="Sampling slabs", disable=self.DEBUG_MODE):
            if self.DEBUG_MODE:
                print(f"Sampling slabs at {reconstruction_time} Ma")

            # Select cases
            for key, entries in self.slab_pull_cases.items():
                if self.options[key]["Slab pull torque"] or self.options[key]["Slab bend torque"]:
                    # Sample age and sediment thickness of lower plate from seafloor
                    self.slabs[reconstruction_time][key]["lower_plate_age"], self.slabs[reconstruction_time][key]["sediment_thickness"] = functions_main.sample_slabs_from_seafloor(
                        self.slabs[reconstruction_time][key].lat, 
                        self.slabs[reconstruction_time][key].lon,
                        self.slabs[reconstruction_time][key].trench_normal_azimuth,
                        self.seafloor[reconstruction_time], 
                        self.options[key],
                        "lower plate",
                        sediment_thickness=self.slabs[reconstruction_time][key].sediment_thickness,
                        continental_arc=self.slabs[reconstruction_time][key].continental_arc,
                    )

                    # Calculate lower plate thickness
                    self.slabs[reconstruction_time][key]["lower_plate_thickness"], _, _ = functions_main.compute_thicknesses(
                        self.slabs[reconstruction_time][key].lower_plate_age,
                        self.options[key],
                        crust = False, 
                        water = False
                    )
                
                    for entry in entries[1:]:
                        self.slabs[reconstruction_time][entry]["lower_plate_age"] = self.slabs[reconstruction_time][key]["lower_plate_age"]
                        self.slabs[reconstruction_time][entry]["sediment_thickness"] = self.slabs[reconstruction_time][key]["sediment_thickness"]
                        self.slabs[reconstruction_time][entry]["lower_plate_thickness"] = self.slabs[reconstruction_time][key]["lower_plate_thickness"]

        self.sampled_slabs = True

    def sample_upper_plate(self):
        """
        Samples seafloor age the upper plate along subduction zones
        The results are stored in the `slabs` DataFrame, specifically in the `upper_plate_age`, `upper_plate_thickness` fields for each case and reconstruction time.
        """
        # Loop through valid times    
        for reconstruction_time in tqdm(self.times, desc="Sampling upper plates", disable=self.DEBUG_MODE):
            if self.DEBUG_MODE:
                print(f"Sampling overriding plate at {reconstruction_time} Ma")

            # Select cases
            for key, entries in self.slab_pull_cases.items():
                # Check whether to output erosion rate and sediment thickness
                if self.options[key]["Sediment subduction"] and self.options[key]["Sample erosion grid"] in self.seafloor[reconstruction_time].data_vars:
                    # Sample age and arc type, erosion rate and sediment thickness of upper plate from seafloor
                    self.slabs[reconstruction_time][key]["upper_plate_age"], self.slabs[reconstruction_time][key]["continental_arc"], self.slabs[reconstruction_time][key]["erosion_rate"], self.slabs[reconstruction_time][key]["sediment_thickness"] = functions_main.sample_slabs_from_seafloor(
                        self.slabs[reconstruction_time][key].lat, 
                        self.slabs[reconstruction_time][key].lon,
                        self.slabs[reconstruction_time][key].trench_normal_azimuth,  
                        self.seafloor[reconstruction_time],
                        self.options[key],
                        "upper plate",
                        sediment_thickness=self.slabs[reconstruction_time][key].sediment_thickness,
                    )
                elif self.options[key]["Sediment subduction"]:
                    # Sample age and arc type of upper plate from seafloor
                    self.slabs[reconstruction_time][key]["upper_plate_age"], self.slabs[reconstruction_time][key]["continental_arc"] = functions_main.sample_slabs_from_seafloor(
                        self.slabs[reconstruction_time][key].lat, 
                        self.slabs[reconstruction_time][key].lon,
                        self.slabs[reconstruction_time][key].trench_normal_azimuth,  
                        self.seafloor[reconstruction_time],
                        self.options[key],
                        "upper plate",
                    )
                
                # Copy DataFrames to other cases
                for entry in entries[1:]:
                    self.slabs[reconstruction_time][entry]["upper_plate_age"] = self.slabs[reconstruction_time][key]["upper_plate_age"]
                    self.slabs[reconstruction_time][entry]["continental_arc"] = self.slabs[reconstruction_time][key]["continental_arc"]
                    if self.options[key]["Sample erosion grid"]:
                        self.slabs[reconstruction_time][entry]["erosion_rate"] = self.slabs[reconstruction_time][key]["erosion_rate"]
                        self.slabs[reconstruction_time][entry]["sediment_thickness"] = self.slabs[reconstruction_time][key]["sediment_thickness"]
        
        self.sampled_upper_plates = True

    def sample_points(self):
        """
        Samples seafloor age at points
        The results are stored in the `points` DataFrame, specifically in the `seafloor_age` field for each case and reconstruction time.
        """
        # Loop through valid times
        for reconstruction_time in tqdm(self.times, desc="Sampling points", disable=self.DEBUG_MODE):
            if self.DEBUG_MODE:
                print(f"Sampling points at {reconstruction_time} Ma")

            for key, entries in self.gpe_cases.items():
                # Select dictionaries
                self.seafloor[reconstruction_time] = self.seafloor[reconstruction_time]
                
                self.points[reconstruction_time][key]["seafloor_age"] = functions_main.sample_ages(self.points[reconstruction_time][key].lat, self.points[reconstruction_time][key].lon, self.seafloor[reconstruction_time]["seafloor_age"])
                for entry in entries[1:]:
                    self.points[reconstruction_time][entry]["seafloor_age"] = self.points[reconstruction_time][key]["seafloor_age"]

        self.sampled_points = True

    def sample_all(self):
        """
        Samples all relevant data from the seafloor to perform torque computation
        """
        self.sample_slabs()
        self.sample_upper_plate()
        self.sample_points()

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# COMPUTING TORQUES
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    def compute_slab_pull_torque(self):
        """
        Compute slab pull torque
        """
        # Check if upper plates have been sampled already
        if self.sampled_upper_plates == False:
            self.sample_upper_plate()

        # Check if slabs have been sampled already
        if self.sampled_slabs == False:
            self.sample_slabs()

        # Loop through reconstruction times
        for i, reconstruction_time in tqdm(enumerate(self.times), desc="Computing slab pull torques", disable=self.DEBUG_MODE):
            if self.DEBUG_MODE:
                print(f"Computing slab pull torques at {reconstruction_time} Ma")

            # Loop through slab pull cases
            for key, entries in self.slab_pull_cases.items():
                if self.DEBUG_MODE:
                    print(f"Computing slab pull torques for cases {entries}")

                # Calculate slab pull torque
                if self.options[key]["Slab pull torque"]:
                    self.slabs[reconstruction_time][key] = functions_main.compute_slab_pull_force(self.slabs[reconstruction_time][key], self.options[key], self.mech)
                    self.plates[reconstruction_time][key] = functions_main.compute_torque_on_plates(
                        self.plates[reconstruction_time][key], 
                        self.slabs[reconstruction_time][key].lat, 
                        self.slabs[reconstruction_time][key].lon, 
                        self.slabs[reconstruction_time][key].lower_plateID, 
                        self.slabs[reconstruction_time][key].slab_pull_force_lat, 
                        self.slabs[reconstruction_time][key].slab_pull_force_lon,
                        self.slabs[reconstruction_time][key].trench_segment_length,
                        1,
                        self.constants,
                        torque_variable="slab_pull_torque"
                    )

                    # Compute interface term
                    self.slabs[reconstruction_time][key] = functions_main.compute_interface_term(self.slabs[reconstruction_time][key], self.options[key])
                    self.plates[reconstruction_time][key] = functions_main.compute_torque_on_plates(
                        self.plates[reconstruction_time][key], 
                        self.slabs[reconstruction_time][key].lat, 
                        self.slabs[reconstruction_time][key].lon, 
                        self.slabs[reconstruction_time][key].lower_plateID, 
                        self.slabs[reconstruction_time][key].slab_pull_force_opt_lat, 
                        self.slabs[reconstruction_time][key].slab_pull_force_opt_lon,
                        self.slabs[reconstruction_time][key].trench_segment_length,
                        1,
                        self.constants,
                        torque_variable="slab_pull_torque_opt"
                    )

                    # Copy DataFrames
                    if len(entries) > 1:
                        [[self.slabs[reconstruction_time][entry].update(
                            {"slab_pull_force_" + coord: self.slabs[reconstruction_time][key]["slab_pull_force_" + coord]}
                        ) for coord in ["lat", "lon", "mag"]] for entry in entries[1:]]
                        [[self.slabs[reconstruction_time][entry].update(
                            {"slab_pull_force_opt_" + coord: self.slabs[reconstruction_time][key]["slab_pull_force_opt_" + coord]}
                        ) for coord in ["lat", "lon", "mag"]] for entry in entries[1:]]
                        [[self.plates[reconstruction_time][entry].update(
                            {"slab_pull_force_" + coord: self.plates[reconstruction_time][key]["slab_pull_force_" + coord]}
                        ) for coord in ["lat", "lon", "mag"]] for entry in entries[1:]]
                        [[self.plates[reconstruction_time][entry].update(
                            {"slab_pull_torque_" + axis: self.plates[reconstruction_time][key]["slab_pull_torque_" + axis]}
                        ) for axis in ["x", "y", "z", "mag"]] for entry in entries[1:]]
                        [[self.plates[reconstruction_time][entry].update(
                            {"slab_pull_torque_opt_" + axis: self.plates[reconstruction_time][key]["slab_pull_torque_opt_" + axis]}
                        ) for axis in ["x", "y", "z", "mag"]] for entry in entries[1:]]

                    # Enter computed slab pull values into torque dictionary
                    for plate in self.plates_of_interest:
                        if self.DEBUG_MODE:
                            print(f"Updating torques for plate {plate}")

                        # Check if plate is in DataFrame
                        if float(plate) in self.plates[reconstruction_time][key].plateID.values:
                            # Check if value is not NaN
                            torque_value = self.plates[reconstruction_time][key][self.plates[reconstruction_time][key].plateID == float(plate)]["slab_pull_torque_mag"].values[0]

                            if self.DEBUG_MODE:
                                    print(f"Slab pull torque value for {plate} is {torque_value}!")

                            if torque_value != 0. and torque_value != _numpy.nan:
                                # Enter data into DataFrame
                                self.torques[key][plate].loc[i, "slab_pull_torque"] = torque_value
                                self.torques[key][plate].loc[i, "slab_pull_torque_opt"] = self.plates[reconstruction_time][key][self.plates[reconstruction_time][key].plateID == float(plate)]["slab_pull_torque_opt_mag"].values[0]

    def compute_slab_bend_torque(self):
        """
        Compute slab bend torque
        """
        # Check if slabs have been sampled already
        if self.sampled_slabs == False:
            self.sample_slabs()

        # Loop through reconstruction times
        for i, reconstruction_time in tqdm(enumerate(self.times), desc="Computing slab bend torques", disable=self.DEBUG_MODE):
            if self.DEBUG_MODE:
                print(f"Computing slab bend torques at {reconstruction_time} Ma")

            # Loop through slab bend cases
            for key, entries in self.slab_bend_cases.items():
                if self.DEBUG_MODE:
                    print(f"Computing slab bend torques for cases {entries}")

                # Calculate slab bending torque
                if self.options[key]["Slab bend torque"]:
                    self.slabs[reconstruction_time][key] = functions_main.compute_slab_bend_force(self.slabs[reconstruction_time][key], self.options[key], self.mech, self.constants)
                    self.plates[reconstruction_time][key] = functions_main.compute_torque_on_plates(
                        self.plates[reconstruction_time][key], 
                        self.slabs[reconstruction_time][key].lat, 
                        self.slabs[reconstruction_time][key].lon, 
                        self.slabs[reconstruction_time][key].lower_plateID, 
                        self.slabs[reconstruction_time][key].slab_bend_force_lat, 
                        self.slabs[reconstruction_time][key].slab_bend_force_lon,
                        self.slabs[reconstruction_time][key].trench_segment_length,
                        1,
                        self.constants,
                        torque_variable="slab_bend_torque"
                    )

                    # Copy DataFrames
                    if len(entries) > 1:
                        [self.slabs[reconstruction_time][entry].update(
                            {"slab_bend_force_" + coord: self.slabs[reconstruction_time][key]["slab_bend_force_" + coord]}
                        ) for coord in ["lat", "lon", "mag"] for entry in entries[1:]]
                        [self.plates[reconstruction_time][entry].update(
                            {"slab_bend_force_" + coord: self.plates[reconstruction_time][key]["slab_bend_force_" + coord]}
                        ) for coord in ["lat", "lon", "mag"] for entry in entries[1:]]
                        [self.plates[reconstruction_time][entry].update(
                            {"slab_bend_torque_" + axis: self.plates[reconstruction_time][key]["slab_bend_torque_" + axis]}
                        ) for axis in ["x", "y", "z", "mag"] for entry in entries[1:]]

                    # Enter computed slab pull values into torque dictionary
                    for plate in self.plates_of_interest:
                        if self.DEBUG_MODE:
                            print(f"Updating torques for plate {plate}")

                        # Check if plate is in DataFrame
                        if float(plate) in self.plates[reconstruction_time][key].plateID.values:
                            # Check if value is not NaN
                            torque_value = self.plates[reconstruction_time][key][self.plates[reconstruction_time][key].plateID == float(plate)]["slab_bend_torque_mag"].values[0]

                            if self.DEBUG_MODE:
                                    print(f"Slab bend torque value for {plate} is {torque_value}!")

                            if torque_value != 0. and torque_value != _numpy.nan:
                                # Enter data into DataFrame
                                self.torques[key][plate].loc[i, "slab_bend_torque"] = torque_value

    def compute_gpe_torque(self):
        """
        Function to compute gravitational potential energy (GPE) torque
        """
        # Check if points have been sampled
        if self.sampled_points == False:
            self.sample_points()

        # Loop through reconstruction times
        for i, reconstruction_time in tqdm(enumerate(self.times), desc="Computing GPE torques", disable=self.DEBUG_MODE):
            if self.DEBUG_MODE:
                print(f"Computing slab bend torques at {reconstruction_time} Ma")

            # Loop through gpe cases
            for key, entries in self.gpe_cases.items():
                if self.DEBUG_MODE:
                    print(f"Computing GPE torque for cases {entries}")

                # Calculate GPE torque
                if self.options[key]["GPE torque"]: 
                    self.points[reconstruction_time][key] = functions_main.compute_GPE_force(self.points[reconstruction_time][key], self.seafloor[reconstruction_time], self.options[key], self.mech)
                    self.plates[reconstruction_time][key] = functions_main.compute_torque_on_plates(
                        self.plates[reconstruction_time][key], 
                        self.points[reconstruction_time][key].lat, 
                        self.points[reconstruction_time][key].lon, 
                        self.points[reconstruction_time][key].plateID, 
                        self.points[reconstruction_time][key].GPE_force_lat, 
                        self.points[reconstruction_time][key].GPE_force_lon,
                        self.points[reconstruction_time][key].segment_length_lat, 
                        self.points[reconstruction_time][key].segment_length_lon,
                        self.constants,
                        torque_variable="GPE_torque"
                    )

                    # Copy DataFrames
                    if len(entries) > 1:
                        [[self.points[reconstruction_time][entry].update(
                            {"GPE_force_" + coord: self.points[reconstruction_time][key]["GPE_force_" + coord]}
                        ) for coord in ["lat", "lon", "mag"]] for entry in entries[1:]]
                        [[self.plates[reconstruction_time][entry].update(
                            {"GPE_torque_" + axis: self.plates[reconstruction_time][key]["GPE_torque_" + axis]}
                        ) for axis in ["x", "y", "z", "mag"]] for entry in entries[1:]]

                    # Enter computed slab pull values into torque dictionary
                    for plate in self.plates_of_interest:
                        if self.DEBUG_MODE:
                            print(f"Updating torques for plate {plate}")

                        # Check if plate is in DataFrame
                        if float(plate) in self.plates[reconstruction_time][key].plateID.values:
                            # Check if value is not NaN
                            torque_value = self.plates[reconstruction_time][key][self.plates[reconstruction_time][key].plateID == float(plate)]["GPE_torque_mag"].values[0]

                            if self.DEBUG_MODE:
                                    print(f"GPE torque value for {plate} is {torque_value}!")

                            if torque_value != 0. and torque_value != _numpy.nan:
                                # Enter data into DataFrame
                                self.torques[key][plate].loc[i, "GPE_torque"] = torque_value

    def compute_mantle_drag_torque(self):
        """
        Function to calculate mantle drag torque
        """
        # Loop through reconstruction times
        for i, reconstruction_time in tqdm(enumerate(self.times), desc="Computing mantle drag torques", disable=self.DEBUG_MODE):
            if self.DEBUG_MODE:
                print(f"Computing mantle drag torques at {reconstruction_time} Ma")

            # Loop through mantle drag cases
            for key, entries in self.mantle_drag_cases.items():
                if self.options[key]["Reconstructed motions"]:
                    if self.DEBUG_MODE:
                        print(f"Computing mantle drag torque from reconstructed motions for cases {entries}")

                    # Calculate Mantle drag torque
                    if self.options[key]["Mantle drag torque"]:
                        # Calculate mantle drag force
                        self.plates[reconstruction_time][key], self.points[reconstruction_time][key], self.slabs[reconstruction_time][key] = functions_main.compute_mantle_drag_force(
                            self.plates[reconstruction_time][key],
                            self.points[reconstruction_time][key],
                            self.slabs[reconstruction_time][key],
                            self.options[key],
                            self.mech,
                            self.constants,
                            self.DEBUG_MODE,
                        )

                        # Calculate mantle drag torque
                        self.plates[reconstruction_time][key] = functions_main.compute_torque_on_plates(
                            self.plates[reconstruction_time][key], 
                            self.points[reconstruction_time][key].lat, 
                            self.points[reconstruction_time][key].lon, 
                            self.points[reconstruction_time][key].plateID, 
                            self.points[reconstruction_time][key].mantle_drag_force_lat, 
                            self.points[reconstruction_time][key].mantle_drag_force_lon,
                            self.points[reconstruction_time][key].segment_length_lat,
                            self.points[reconstruction_time][key].segment_length_lon,
                            self.constants,
                            torque_variable="mantle_drag_torque"
                        )

                        # Enter mantle drag torque in other cases
                        if len(entries) > 1:
                            [[self.points[reconstruction_time][entry].update(
                                {"mantle_drag_force_" + coord: self.points[reconstruction_time][key]["mantle_drag_force_" + coord]}
                            ) for coord in ["lat", "lon", "mag"]] for entry in entries[1:]]
                            [[self.plates[reconstruction_time][entry].update(
                                {"mantle_drag_torque_" + coord: self.plates[reconstruction_time][key]["mantle_drag_torque_" + coord]}
                            ) for coord in ["x", "y", "z", "mag"]] for entry in entries[1:]]

                        # Enter computed slab pull values into torque dictionary
                        for plate in self.plates_of_interest:
                            if self.DEBUG_MODE:
                                print(f"Updating torques for plate {plate}")

                            # Check if plate is in DataFrame
                            if float(plate) in self.plates[reconstruction_time][key].plateID.values:
                                # Check if value is not NaN
                                torque_value = self.plates[reconstruction_time][key][self.plates[reconstruction_time][key].plateID == float(plate)]["mantle_drag_torque_mag"].values[0]

                                if self.DEBUG_MODE:
                                    print(f"Torque value for {plate} is {torque_value}!")

                                if torque_value != 0. or torque_value != _numpy.nan:
                                    # Enter data into DataFrame
                                    self.torques[key][plate].loc[i, "mantle_drag_torque"] = torque_value

            # Loop through all cases
            for case in self.cases:
                if not self.options[case]["Reconstructed motions"]:
                    if self.DEBUG_MODE:
                        print(f"Computing mantle drag torque using torque balance for case {case}")

                    if self.options[case]["Mantle drag torque"]:
                        # Calculate mantle drag force
                        self.plates[reconstruction_time][case], self.points[reconstruction_time][case], self.slabs[reconstruction_time][case] = functions_main.compute_mantle_drag_force(
                            self.plates[reconstruction_time][case],
                            self.points[reconstruction_time][case],
                            self.slabs[reconstruction_time][case],
                            self.options[case],
                            self.mech,
                            self.constants,
                            self.DEBUG_MODE,
                        )

                        # Calculate mantle drag torque
                        self.plates[reconstruction_time][case] = functions_main.compute_torque_on_plates(
                            self.plates[reconstruction_time][case], 
                            self.points[reconstruction_time][case].lat, 
                            self.points[reconstruction_time][case].lon, 
                            self.points[reconstruction_time][case].plateID, 
                            self.points[reconstruction_time][case].mantle_drag_force_lat, 
                            self.points[reconstruction_time][case].mantle_drag_force_lon,
                            self.points[reconstruction_time][case].segment_length_lat,
                            self.points[reconstruction_time][case].segment_length_lon,
                            self.constants,
                            torque_variable="mantle_drag_torque"
                        )

                        # Compute velocity grid
                        self.velocity[reconstruction_time][case] = setup.get_velocity_grid(self.points[reconstruction_time][case], self.seafloor[reconstruction_time])

                        # Enter computed slab pull values into torque dictionary
                        for plate in self.plates_of_interest:
                            if self.DEBUG_MODE:
                                print(f"Updating torques for plate {plate}")

                            # Check if plate is in DataFrame
                            if float(plate) in self.plates[reconstruction_time][key].plateID.values:
                                # Check if value is not NaN
                                torque_value = self.plates[reconstruction_time][key][self.plates[reconstruction_time][key].plateID == float(plate)]["mantle_drag_torque_mag"].values[0]

                                if self.DEBUG_MODE:
                                    print(f"Mantle drag torque value is {torque_value}!")

                                # Enter data into DataFrame
                                self.torques[key][plate].loc[i, "mantle_drag_torque"] = torque_value
                                self.torques[key][plate].loc[i, "mantle_drag_torque_opt"] = self.plates[reconstruction_time][key][self.plates[reconstruction_time][key].plateID == float(plate)]["mantle_drag_torque_opt_mag"].values[0]

    def optimise_torques(self):
        """
        Function to optimise torques
        """
        for i, reconstruction_time in tqdm(enumerate(self.times), desc="Optimising torques", disable=self.DEBUG_MODE):
            if self.DEBUG_MODE:
                print(f"Optimising torques at {reconstruction_time} Ma")
            
            for key, entries in self.slab_pull_cases.items():
                if self.options[key]["Slab pull torque"]:
                    self.plates[reconstruction_time][key] = functions_main.optimise_torques(
                        self.plates[reconstruction_time][key],
                        self.mech,
                        self.options[key],
                    )

                    # Copy DataFrames
                    if len(entries) > 1:
                        [[self.plates[reconstruction_time][entry].update(
                            {"slab_pull_torque_opt_" + axis: self.plates[reconstruction_time][key]["slab_pull_torque_opt_" + axis]}
                        ) for axis in ["x", "y", "z", "mag"]] for entry in entries[1:]]

                    # Enter computed slab pull values into torque dictionary
                    for plate in self.plates_of_interest:
                        if self.DEBUG_MODE:
                            print(f"Updating torques for plate {plate}")

                        # Check if plate is in DataFrame
                        if float(plate) in self.plates[reconstruction_time][key].plateID.values:
                            # Check if value is not NaN
                            torque_value = self.plates[reconstruction_time][key][self.plates[reconstruction_time][key].plateID == float(plate)]["slab_pull_torque_opt_mag"].values[0]

                            if self.DEBUG_MODE:
                                print(f"Optimised slab pull torque value for {plate} is {torque_value}!")

                            if torque_value != 0 or torque_value != _numpy.nan:
                                # Enter data into DataFrame
                                self.torques[key][plate].loc[i, "slab_pull_torque_opt"] = torque_value

                            # Check if value is not NaN
                            torque_value = self.plates[reconstruction_time][key][self.plates[reconstruction_time][key].plateID == float(plate)]["mantle_drag_torque_opt_mag"].values[0]

                            if self.DEBUG_MODE:
                                print(f"Optimised mantle drag torque value for {plate} is {torque_value}!")

                            if torque_value != 0 and torque_value != _numpy.nan:
                                # Enter data into DataFrame
                                self.torques[key][plate].loc[i, "mantle_drag_torque_opt"] = torque_value
            
            # Copy torques to other cases
            for key, entries in self.mantle_drag_cases.items():
                if self.options[key]["Mantle drag torque"]:
                    self.plates[reconstruction_time][key] = functions_main.optimise_torques(
                        self.plates[reconstruction_time][key],
                        self.mech,
                        self.options[key],
                    )

                    # Copy DataFrames
                    if len(entries) > 1:
                        [[self.plates[reconstruction_time][entry].update(
                            {"mantle_drag_torque_opt_" + axis: self.plates[reconstruction_time][key]["mantle_drag_torque_opt_" + axis]}
                        ) for axis in ["x", "y", "z", "mag"]] for entry in entries[1:]]

                    # Enter computed slab pull values into torque dictionary
                    for plate in self.plates_of_interest:
                        if self.DEBUG_MODE:
                            print(f"Updating torques for plate {plate}")

                        # Check if plate is in DataFrame
                        if float(plate) in self.plates[reconstruction_time][key].plateID.values:
                            # Check if value is not NaN
                            torque_value = self.plates[reconstruction_time][key][self.plates[reconstruction_time][key].plateID == float(plate)]["mantle_drag_torque_opt_mag"].values[0]

                            if self.DEBUG_MODE:
                                print(f"Mantle drag torque magnitude for {plate} is {torque_value}!")

                            if torque_value != 0 and torque_value != _numpy.nan:
                                # Enter data into DataFrame
                                self.torques[key][plate].loc[i, "mantle_drag_torque_opt"] = torque_value
    
    def compute_driving_torque(self):
        """
        Function to calculate driving torque
        """
        # Loop through reconstruction times
        for i, reconstruction_time in tqdm(enumerate(self.times), desc="Computing driving torques", disable=self.DEBUG_MODE):
            if self.DEBUG_MODE:
                print(f"Computing driving torques at {reconstruction_time} Ma")

            for case in self.cases:
                # Calculate driving torque
                self.plates[reconstruction_time][case] = functions_main.sum_torque(self.plates[reconstruction_time][case], "driving", self.constants)

                # Enter computed slab pull values into torque dictionary
                for plate in self.plates_of_interest:
                    if self.DEBUG_MODE:
                        print(f"Updating driving torques for plate {plate}")

                    # Check if plate is in DataFrame
                    if float(plate) in self.plates[reconstruction_time][case].plateID.values:
                        # Check if value is not NaN
                        torque_value = self.plates[reconstruction_time][case][self.plates[reconstruction_time][case].plateID == float(plate)]["driving_torque_mag"].values[0]
                    
                        if self.DEBUG_MODE:
                            print(f"Driving torque magnitude for {plate} is {torque_value}!")

                        if torque_value != 0 and torque_value != _numpy.nan:
                            # Enter data into DataFrame
                            self.torques[case][plate].loc[i, "driving_torque"] = torque_value
                            self.torques[case][plate].loc[i, "driving_torque_opt"] = self.plates[reconstruction_time][case][self.plates[reconstruction_time][case].plateID == float(plate)]["driving_torque_opt_mag"].values[0]

    def compute_residual_torque(self):
        """
        Function to calculate residual torque
        """
        # Loop through reconstruction times
        for i, reconstruction_time in tqdm(enumerate(self.times), desc="Computing residual torques", disable=self.DEBUG_MODE):
            if self.DEBUG_MODE:
                print(f"Computing residual torques at {reconstruction_time} Ma")

            for case in self.cases:
                # Select cases that require residual torque computation
                if self.options[case]["Reconstructed motions"]:
                    # Calculate residual torque
                    self.plates[reconstruction_time][case] = functions_main.sum_torque(self.plates[reconstruction_time][case], "residual", self.constants)

                    # Enter computed slab pull values into torque dictionary
                    for plate in self.plates_of_interest:
                        if self.DEBUG_MODE:
                            print(f"Updating residual torques for plate {plate}")

                        # Check if plate is in DataFrame
                        if float(plate) in self.plates[reconstruction_time][case].plateID.values:
                            # Check if value is not NaN
                            torque_value = self.plates[reconstruction_time][case][self.plates[reconstruction_time][case].plateID == float(plate)]["residual_torque_mag"].values[0]

                            if self.DEBUG_MODE:
                                print(f"Residual torque magnitude for {plate} is {torque_value}!")

                            if torque_value != 0 and torque_value != _numpy.nan:
                                # Enter data into DataFrame
                                self.torques[case][plate].loc[i, "residual_torque"] = torque_value
                                self.torques[case][plate].loc[i, "residual_torque_opt"] = self.plates[reconstruction_time][case][self.plates[reconstruction_time][case].plateID == float(plate)]["residual_torque_opt_mag"].values[0]

                else:
                    # Set residual torque to zero
                    for coord in ["x", "y", "z", "mag"]:
                        self.plates[reconstruction_time][case]["residual_torque_" + coord] = 0

                    # Enter computed slab pull values into torque dictionary
                    for plate in self.plates_of_interest:
                        if self.DEBUG_MODE:
                            print(f"Updating residual torques for plate {plate}")

                        # Check if plate is in DataFrame
                        if float(plate) in self.plates[reconstruction_time][case].plateID.values:
                            if self.DEBUG_MODE:
                                print(f"Plate {plate} is in DataFrame!")

                            # Enter data into DataFrame
                            self.torques[case][plate].loc[i, "residual_torque"] = 0

    def compute_all_torques(self):
        """
        Computes all torques 
        """
        # Calculate slab pull torque
        self.compute_slab_pull_torque()

        # Calculate slab bend torque
        self.compute_slab_bend_torque()

        # Calculate GPE torque
        self.compute_gpe_torque()

        # Calculate mantle drag torque
        self.compute_mantle_drag_torque()

        # Calculate driving torque
        self.compute_driving_torque()
        
        # Calculate residual torque
        self.compute_residual_torque()

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ROTATION 
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    def rotate_torque(
            self,
            torque: str,
            reference_rotations: _pygplates.RotationModel,
            reference_plates: dict,
            reference_case: Optional[str] = None,
            case: Optional[str] = None,
        ):
        """
        Function to rotate torques all plates to a new reference frame

        :param torque:                  torque to rotate
        :type torque:                   str
        :param reference_rotations:     reference rotations
        :type reference_rotations:      pygplates.RotationModel
        :param reference_plates:        reference plates
        :type reference_plates:         dict
        :param case:                    case to rotate
        :type case:                     str or None
        """
        # Check if the torque is valid
        if torque not in ["slab_pull_torque", "slab_pull_torque_opt", "GPE_torque", "slab_bend_torque", "mantle_drag_torque", "mantle_drag_torque_opt"]:
            raise ValueError(f"Invalid torque '{torque}' Please select one of slab_pull_torque, GPE_torque, slab_bend_torque or mantle_drag_torque.")
        
        # Check for which cases to rotate the torques
        if case == None:
            rotate_cases = self.cases
        else:
            rotate_cases = [case]

        # Check if reference case is provided, otherwise default to first case in list
        if reference_case == None:
            reference_case = list(reference_plates.keys())[0]
    
        # Loop through all reconstruction times
        for i, reconstruction_time in tqdm(enumerate(self.times), desc="Rotating torques", disable=self.DEBUG_MODE):
            # Check if times in reference_plates dictionary
            if reference_case in reference_plates.keys():
                # Loop through all cases
                for case in rotate_cases:
                    # Select cases that require rotation
                    if self.options[case]["Reconstructed motions"] and self.options[case]["Mantle drag torque"]:
                        for plateID in self.plates[reconstruction_time][case].plateID.values:
                            # Rotate x, y, and z components of torque
                            self.plates[reconstruction_time][case].loc[self.plates[reconstruction_time][case].plateID == plateID, [torque + "_x", torque + "_y", torque + "_z"]] = functions_main.rotate_torque(
                                plateID,
                                reference_plates[reconstruction_time][case].loc[reference_plates[reconstruction_time][case].plateID == plateID, [torque + "_x", torque + "_y", torque + "_z"]].copy(),
                                reference_rotations,
                                self.rotations,
                                reconstruction_time,
                                self.constants,
                            )

                            # Copy magnitude of torque
                            self.plates[reconstruction_time][case].loc[self.plates[reconstruction_time][case].plateID == plateID, torque + "_mag"] = reference_plates[reconstruction_time][case].loc[reference_plates[reconstruction_time][case].plateID == plateID, torque + "_mag"].values[0]

                            if int(plateID) in self.torques[case].keys():
                                # Print if plateID is in torques dictionary
                                if self.DEBUG_MODE:
                                    print(f"{plateID} is in torques dictionary")
                                    print(self.plates[reconstruction_time][case][self.plates[reconstruction_time][case].plateID == plateID][f"{torque}_mag"].values[0])

                                # Enter data into DataFrame
                                self.torques[case][int(plateID)].loc[i, torque] = self.plates[reconstruction_time][case][self.plates[reconstruction_time][case].plateID == plateID][f"{torque}_mag"].values[0]
        
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# OPTIMISATION 
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    def minimise_residual_torque(
            self,
            opt_time,
            opt_case,
            plates_of_interest=None,
            grid_size=500,
            visc_range=[5e18, 5e20],
            plot=True,
            weight_by_area=True,
            minimum_plate_area=None
        ):
        """
        Function to find optimised coefficients to match plate motions using a grid search

        :param opt_time:                reconstruction time to optimise
        :type opt_time:                 int
        :param opt_case:                case to optimise
        :type opt_case:                 str
        :param plates_of_interest:      plate IDs to include in optimisation
        :type plates_of_interest:       list of integers or None
        :param grid_size:               size of the grid to find optimal viscosity and slab pull coefficient
        :type grid_size:                int
        :param plot:                    whether or not to plot the grid
        :type plot:                     boolean
        :param weight_by_area:          whether or not to weight the residual torque by plate area
        :type weight_by_area:           boolean

        :return:                        None
        """
        # Generate grid of viscosities and slab pull coefficients
        viscs = _numpy.linspace(visc_range[0],visc_range[1],grid_size)
        sp_consts = _numpy.linspace(1e-5,1,grid_size)
        visc_grid, sp_const_grid = _numpy.meshgrid(viscs, sp_consts)
        ones_grid = _numpy.ones_like(visc_grid)

        # Filter plates
        selected_plates = self.plates[opt_time][opt_case].copy()
        if plates_of_interest:
            selected_plates = selected_plates[selected_plates["plateID"].isin(plates_of_interest)]
            selected_plates = selected_plates.reset_index(drop=True)
        else:
            plates_of_interest = selected_plates["plateID"]

        # Filter plates by minimum area
        if minimum_plate_area is None:
            minimum_plate_area = self.options[opt_case]["Minimum plate area"]
        selected_plates = selected_plates[selected_plates["area"] > minimum_plate_area]
        selected_plates = selected_plates.reset_index(drop=True)
        plates_of_interest = selected_plates["plateID"]

        # Get total area
        total_area = selected_plates["area"].sum()

        # Initialise dictionaries and arrays to store driving and residual torques
        if opt_time not in self.driving_torque:
            self.driving_torque[opt_time] = {}
        if opt_time not in self.driving_torque_normalised:
            self.driving_torque_normalised[opt_time] = {}
        if opt_time not in self.residual_torque:
            self.residual_torque[opt_time] = {}
        if opt_time not in self.residual_torque_normalised:
            self.residual_torque_normalised[opt_time] = {}
            
        self.driving_torque[opt_time][opt_case] = _numpy.zeros_like(sp_const_grid); self.driving_torque_normalised[opt_time][opt_case] = _numpy.zeros_like(sp_const_grid)
        self.residual_torque[opt_time][opt_case] = _numpy.zeros_like(sp_const_grid); self.residual_torque_normalised[opt_time][opt_case] = _numpy.zeros_like(sp_const_grid)

        # Initialise dictionaries to store optimal coefficients
        if opt_time not in self.opt_i:
            self.opt_i[opt_time] = {}
        if opt_time not in self.opt_j:
            self.opt_j[opt_time] = {}
        if opt_time not in self.opt_sp_const:
            self.opt_sp_const[opt_time] = {}
        if opt_time not in self.opt_visc:
            self.opt_visc[opt_time] = {}

        # Get torques
        for k, _ in enumerate(plates_of_interest):
            residual_x = _numpy.zeros_like(sp_const_grid); residual_y = _numpy.zeros_like(sp_const_grid); residual_z = _numpy.zeros_like(sp_const_grid)
            if self.options[opt_case]["Slab pull torque"] and "slab_pull_torque_x" in selected_plates.columns:
                residual_x -= selected_plates.slab_pull_torque_x.iloc[k] * sp_const_grid
                residual_y -= selected_plates.slab_pull_torque_y.iloc[k] * sp_const_grid
                residual_z -= selected_plates.slab_pull_torque_z.iloc[k] * sp_const_grid

            # Add GPE torque
            if self.options[opt_case]["GPE torque"] and "GPE_torque_x" in selected_plates.columns:
                residual_x -= selected_plates.GPE_torque_x.iloc[k] * ones_grid
                residual_y -= selected_plates.GPE_torque_y.iloc[k] * ones_grid
                residual_z -= selected_plates.GPE_torque_z.iloc[k] * ones_grid
            
            # Compute magnitude of driving torque
            if weight_by_area:
                self.driving_torque[opt_time][opt_case] += _numpy.sqrt(residual_x**2 + residual_y**2 + residual_z**2) * selected_plates.area.iloc[k] / total_area
            else:
                self.driving_torque[opt_time][opt_case] += _numpy.sqrt(residual_x**2 + residual_y**2 + residual_z**2) / selected_plates.area.iloc[k]

            # Add slab bend torque
            if self.options[opt_case]["Slab bend torque"] and "slab_bend_torque_x" in selected_plates.columns:
                residual_x -= selected_plates.slab_bend_torque_x.iloc[k] * ones_grid
                residual_y -= selected_plates.slab_bend_torque_y.iloc[k] * ones_grid
                residual_z -= selected_plates.slab_bend_torque_z.iloc[k] * ones_grid

            # Add mantle drag torque
            if self.options[opt_case]["Mantle drag torque"] and "mantle_drag_torque_x" in selected_plates.columns:
                residual_x -= selected_plates.mantle_drag_torque_x.iloc[k] * visc_grid / self.mech.La
                residual_y -= selected_plates.mantle_drag_torque_y.iloc[k] * visc_grid / self.mech.La
                residual_z -= selected_plates.mantle_drag_torque_z.iloc[k] * visc_grid / self.mech.La

            # Compute magnitude of residual
            if weight_by_area:
                self.residual_torque[opt_time][opt_case] += _numpy.sqrt(residual_x**2 + residual_y**2 + residual_z**2) * selected_plates.area.iloc[k] / total_area
            else:
                self.residual_torque[opt_time][opt_case] += _numpy.sqrt(residual_x**2 + residual_y**2 + residual_z**2) / selected_plates.area.iloc[k]
    
            # Divide residual by driving torque
            self.residual_torque_normalised[opt_time][opt_case] = _numpy.log10(self.residual_torque[opt_time][opt_case] / self.driving_torque[opt_time][opt_case])

        # Find the indices of the minimum value directly using _numpy.argmin
        self.opt_i[opt_time][opt_case], self.opt_j[opt_time][opt_case] = _numpy.unravel_index(_numpy.argmin(self.residual_torque_normalised[opt_time][opt_case]), self.residual_torque_normalised[opt_time][opt_case].shape)
        self.opt_visc[opt_time][opt_case] = visc_grid[self.opt_i[opt_time][opt_case], self.opt_j[opt_time][opt_case]]
        self.opt_sp_const[opt_time][opt_case] = sp_const_grid[self.opt_i[opt_time][opt_case], self.opt_j[opt_time][opt_case]]

        # Plot
        if plot == True:
            fig, ax = plt.subplots(figsize=(15*self.constants.cm2in, 12*self.constants.cm2in))
            im = ax.imshow(self.residual_torque_normalised[opt_time][opt_case], cmap="cmc.lapaz_r", vmin=-1.5, vmax=1.5)
            ax.set_yticks(_numpy.linspace(0, grid_size - 1, 5))
            ax.set_xticks(_numpy.linspace(0, grid_size - 1, 5))
            ax.set_xticklabels(["{:.2e}".format(visc) for visc in _numpy.linspace(visc_range[0], visc_range[1], 5)])
            ax.set_yticklabels(["{:.2f}".format(sp_const) for sp_const in _numpy.linspace(sp_consts.min(), sp_consts.max(), 5)])
            ax.set_xlabel("Mantle viscosity [Pa s]")
            ax.set_ylabel("Slab pull reduction factor")
            ax.scatter(self.opt_j[opt_time][opt_case], self.opt_i[opt_time][opt_case], marker="*", facecolor="none", edgecolor="k", s=30)  # Adjust the marker style and size as needed
            fig.colorbar(im, label = "Log(residual torque/driving torque)")
            plt.show()

        # Print results
        print(f"Optimal coefficients for ", ", ".join(selected_plates.name.astype(str)), " plate(s), (PlateIDs: ", ", ".join(selected_plates.plateID.astype(str)), ")")
        print("Minimum residual torque: {:.2%} of driving torque".format(10**(_numpy.amin(self.residual_torque_normalised[opt_time][opt_case]))))
        print("Optimum viscosity [Pa s]: {:.2e}".format(self.opt_visc[opt_time][opt_case]))
        print("Optimum Drag Coefficient [Pa s/m]: {:.2e}".format(self.opt_visc[opt_time][opt_case] / self.mech.La))
        print("Optimum Slab Pull constant: {:.2%}".format(self.opt_sp_const[opt_time][opt_case]))

        return self.opt_sp_const[opt_time][opt_case], self.opt_visc[opt_time][opt_case], self.residual_torque_normalised[opt_time][opt_case]
    
    def find_slab_pull_coefficient(self, opt_time, opt_case, plates_of_interest=None, grid_size=500, viscosity=1e19, plot=True, weight_by_area=True):
        """
        Function to find optimised slab pull coefficient for a given (set of) plates using a grid search.

        :param opt_time:                reconstruction time to optimise
        :type opt_time:                 int
        :param opt_case:                case to optimise
        :type opt_case:                 str
        :param plates_of_interest:      plate IDs to include in optimisation
        :type plates_of_interest:       list of integers or None
        :param grid_size:               size of the grid to find optimal slab pull coefficient
        :type grid_size:                int
        :param plot:                    whether or not to plot the grid
        :type plot:                     boolean
        :param weight_by_area:          whether or not to weight the residual torque by plate area
        :type weight_by_area:           boolean
        
        :return:                        The optimal slab pull coefficient
        :rtype:                         float
        """
        # Generate range of possible slab pull coefficients
        sp_consts = _numpy.linspace(1e-5,1,grid_size)
        ones = _numpy.ones_like(sp_consts)

        # Filter plates
        selected_plates = self.plates[opt_time][opt_case].copy()
        if plates_of_interest:
            selected_plates = selected_plates[selected_plates["plateID"].isin(plates_of_interest)]
            if selected_plates.empty:
                return _numpy.nan
            
            selected_plates = selected_plates.reset_index(drop=True)
        else:
            plates_of_interest = selected_plates["plateID"]

        # Initialise dictionary to store optimal slab pull coefficient per plate
        opt_sp_consts = {None for _ in plates_of_interest}
        
        # Loop through plates
        for k, plateID in enumerate(plates_of_interest):
            residual_x = _numpy.zeros_like(sp_consts)
            residual_y = _numpy.zeros_like(sp_consts)
            residual_z = _numpy.zeros_like(sp_consts)

            if self.options[opt_case]["Slab pull torque"] and "slab_pull_torque_x" in selected_plates.columns:
                residual_x -= selected_plates.slab_pull_torque_x.iloc[k] * sp_consts
                residual_y -= selected_plates.slab_pull_torque_y.iloc[k] * sp_consts
                residual_z -= selected_plates.slab_pull_torque_z.iloc[k] * sp_consts

            # Add GPE torque
            if self.options[opt_case]["GPE torque"] and "GPE_torque_x" in selected_plates.columns:
                residual_x -= selected_plates.GPE_torque_x.iloc[k] * ones
                residual_y -= selected_plates.GPE_torque_y.iloc[k] * ones
                residual_z -= selected_plates.GPE_torque_z.iloc[k] * ones

            # Compute magnitude of driving torque
            driving_mag = _numpy.sqrt(residual_x**2 + residual_y**2 + residual_z**2)
            
            # Add slab bend torque
            if self.options[opt_case]["Slab bend torque"] and "slab_bend_torque_x" in selected_plates.columns:
                residual_x -= selected_plates.slab_bend_torque_x.iloc[k] * ones
                residual_y -= selected_plates.slab_bend_torque_y.iloc[k] * ones
                residual_z -= selected_plates.slab_bend_torque_z.iloc[k] * ones

            # Add mantle drag torque
            if self.options[opt_case]["Mantle drag torque"] and "mantle_drag_torque_x" in selected_plates.columns:
                residual_x -= selected_plates.mantle_drag_torque_x.iloc[k] * viscosity
                residual_y -= selected_plates.mantle_drag_torque_y.iloc[k] * viscosity
                residual_z -= selected_plates.mantle_drag_torque_z.iloc[k] * viscosity

            # Compute magnitude of residual
            residual_mag = _numpy.sqrt(residual_x**2 + residual_y**2 + residual_z**2)

            # Find optimal slab pull coefficient
            opt_sp_const = sp_consts[_numpy.argmin(_numpy.log10(residual_mag/driving_mag))]

        return opt_sp_const
    
    def minimise_residual_velocity(self, opt_time, opt_case, plates_of_interest=None, grid_size=10, visc_range=[1e19, 5e20], plot=True, weight_by_area=True, ref_case=None):
        """
        Function to find optimised coefficients to match plate motions using a grid search.

        :param opt_time:                reconstruction time to optimise
        :type opt_time:                 int
        :param opt_case:                case to optimise
        :type opt_case:                 str
        :param plates_of_interest:      plate IDs to include in optimisation
        :type plates_of_interest:       list of integers or None
        :param grid_size:               size of the grid to find optimal viscosity and slab pull coefficient
        :type grid_size:                int
        :param plot:                    whether or not to plot the grid
        :type plot:                     boolean
        :param weight_by_area:          whether or not to weight the residual torque by plate area
        :type weight_by_area:           boolean
        
        :return:                        The optimal slab pull coefficient, the optimal viscosity, the residual plate velocity, and the residual slab velocity.
        :rtype:                         float, float, float, float
        """
        if self.options[opt_case]["Reconstructed motions"]:
            print("Optimisation method designed for synthetic plate velocities only!")
            return
        
        # Get "true" plate velocities
        true_slabs = self.slabs[opt_time][ref_case].copy()

        # Generate grid
        viscs = _numpy.linspace(visc_range[0],visc_range[1],grid_size)
        sp_consts = _numpy.linspace(1e-4,1,grid_size)
        v_upper_plate_residual = _numpy.zeros((grid_size, grid_size))
        v_lower_plate_residual = _numpy.zeros((grid_size, grid_size))
        v_convergence_residual = _numpy.zeros((grid_size, grid_size))

        # Filter plates and slabs
        selected_plates = self.plates[opt_time][opt_case].copy()
        selected_slabs = self.slabs[opt_time][opt_case].copy()
        selected_points = self.points[opt_time][opt_case].copy()

        if plates_of_interest:
            selected_plates = selected_plates[selected_plates["plateID"].isin(plates_of_interest)]
            selected_plates = selected_plates.reset_index(drop=True)
            selected_slabs = selected_slabs[selected_slabs["lower_plateID"].isin(plates_of_interest)]
            selected_slabs = selected_slabs.reset_index(drop=True)
            selected_points = selected_points[selected_points["plateID"].isin(plates_of_interest)]
            selected_points = selected_points.reset_index(drop=True)
            selected_options = self.options[opt_case].copy()
        else:
            plates_of_interest = selected_plates["plateID"]

        # Initialise starting old_plates, old_points, old_slabs by copying self.plates[reconstruction_time][key], self.points[reconstruction_time][key], self.slabs[reconstruction_time][key]
        old_plates = selected_plates.copy(); old_points = selected_points.copy(); old_slabs = selected_slabs.copy()

        # Delete self.slabs[reconstruction_time][key], self.points[reconstruction_time][key], self.plates[reconstruction_time][key]
        del selected_plates, selected_points, selected_slabs
        
        # Loop through plates and slabs and calculate residual velocity
        for i, visc in enumerate(viscs):
            for j, sp_const in enumerate(sp_consts):
                print(i, j)
                # Assign current visc and sp_const to options
                selected_options["Mantle viscosity"] = visc
                selected_options["Slab pull constant"] = sp_const

                # Optimise slab pull force
                [old_plates.update({"slab_pull_torque_opt_" + axis: old_plates["slab_pull_torque_" + axis] * selected_options["Slab pull constant"]}) for axis in ["x", "y", "z"]]

                for k in range(100):
                    # Delete new DataFrames
                    if k != 0:
                        del new_slabs, new_points, new_plates
                    else:
                        old_slabs["v_convergence_mag"] = 0

                    print(_numpy.mean(old_slabs["v_convergence_mag"].values))
                    # Compute interface shear force
                    if self.options[opt_case]["Interface shear torque"]:
                        new_slabs = functions_main.compute_interface_shear_force(old_slabs, self.options[opt_case], self.mech, self.constants)
                    else:
                        new_slabs = old_slabs.copy()

                    # Compute interface shear torque
                    new_plates = functions_main.compute_torque_on_plates(
                        old_plates,
                        new_slabs.lat,
                        new_slabs.lon,
                        new_slabs.lower_plateID,
                        new_slabs.interface_shear_force_lat,
                        new_slabs.interface_shear_force_lon,
                        new_slabs.trench_segment_length,
                        1,
                        self.constants,
                        torque_variable="interface_shear_torque"
                    )

                    # Compute mantle drag force
                    new_plates, new_points, new_slabs = functions_main.compute_mantle_drag_force(old_plates, old_points, new_slabs, self.options[opt_case], self.mech, self.constants)

                    # Compute mantle drag torque
                    new_plates = functions_main.compute_torque_on_plates(
                        new_plates, 
                        new_points.lat, 
                        new_points.lon, 
                        new_points.plateID, 
                        new_points.mantle_drag_force_lat, 
                        new_points.mantle_drag_force_lon,
                        new_points.segment_length_lat,
                        new_points.segment_length_lon,
                        self.constants,
                        torque_variable="mantle_drag_torque"
                    )

                    # Calculate convergence rates
                    v_convergence_lat = new_slabs["v_lower_plate_lat"].values - new_slabs["v_upper_plate_lat"].values
                    v_convergence_lon = new_slabs["v_lower_plate_lon"].values - new_slabs["v_upper_plate_lon"].values
                    v_convergence_mag = _numpy.sqrt(v_convergence_lat**2 + v_convergence_lon**2)

                    # Calculate convergence rates
                    v_convergence_lat = new_slabs["v_lower_plate_lat"].values - new_slabs["v_upper_plate_lat"].values
                    v_convergence_lon = new_slabs["v_lower_plate_lon"].values - new_slabs["v_upper_plate_lon"].values
                    v_convergence_mag = _numpy.sqrt(v_convergence_lat**2 + v_convergence_lon**2)

                    # Check convergence rates
                    if _numpy.max(abs(v_convergence_mag - old_slabs["v_convergence_mag"].values)) < 1e-2: # and _numpy.max(v_convergence_mag) < 25:
                        print(f"Convergence rates converged after {k} iterations")
                        break
                    else:
                        # Assign new values to latest slabs DataFrame
                        new_slabs["v_convergence_lat"], new_slabs["v_convergence_lon"] = functions_main.mag_azi2lat_lon(v_convergence_mag, new_slabs.trench_normal_azimuth); new_slabs["v_convergence_mag"] = v_convergence_mag
                        
                        # Delecte old DataFrames
                        del old_plates, old_points, old_slabs
                        
                        # Overwrite DataFrames
                        old_plates = new_plates.copy(); old_points = new_points.copy(); old_slabs = new_slabs.copy()

                # Calculate residual of plate velocities
                v_upper_plate_residual[i,j] = _numpy.max(abs(new_slabs.v_upper_plate_mag - true_slabs.v_upper_plate_mag))
                print("upper_plate_residual: ", v_upper_plate_residual[i,j])
                v_lower_plate_residual[i,j] = _numpy.max(abs(new_slabs.v_lower_plate_mag - true_slabs.v_lower_plate_mag))
                print("lower_plate_residual: ", v_lower_plate_residual[i,j])
                v_convergence_residual[i,j] = _numpy.max(abs(new_slabs.v_convergence_mag - true_slabs.v_convergence_mag))
                print("convergence_rate_residual: ", v_convergence_residual[i,j])

        # Find the indices of the minimum value directly using _numpy.argmin
        opt_upper_plate_i, opt_upper_plate_j = _numpy.unravel_index(_numpy.argmin(v_upper_plate_residual), v_upper_plate_residual.shape)
        opt_upper_plate_visc = viscs[opt_upper_plate_i]
        opt_upper_plate_sp_const = sp_consts[opt_upper_plate_j]

        opt_lower_plate_i, opt_lower_plate_j = _numpy.unravel_index(_numpy.argmin(v_lower_plate_residual), v_lower_plate_residual.shape)
        opt_lower_plate_visc = viscs[opt_lower_plate_i]
        opt_lower_plate_sp_const = sp_consts[opt_lower_plate_j]

        opt_convergence_i, opt_convergence_j = _numpy.unravel_index(_numpy.argmin(v_convergence_residual), v_convergence_residual.shape)
        opt_convergence_visc = viscs[opt_convergence_i]
        opt_convergence_sp_const = sp_consts[opt_convergence_j]

        # Plot
        for i, j, visc, sp_const, residual in zip([opt_upper_plate_i, opt_lower_plate_i, opt_convergence_i], [opt_upper_plate_j, opt_lower_plate_j, opt_convergence_j], [opt_upper_plate_visc, opt_lower_plate_visc, opt_convergence_visc], [opt_upper_plate_sp_const, opt_lower_plate_sp_const, opt_convergence_sp_const], [v_upper_plate_residual, v_lower_plate_residual, v_convergence_residual]):
            if plot == True:
                fig, ax = plt.subplots(figsize=(15*self.constants.cm2in, 12*self.constants.cm2in))
                im = ax.imshow(residual, cmap="cmc.davos_r")#, vmin=-1.5, vmax=1.5)
                ax.set_yticks(_numpy.linspace(0, grid_size - 1, 5))
                ax.set_xticks(_numpy.linspace(0, grid_size - 1, 5))
                ax.set_xticklabels(["{:.2e}".format(visc) for visc in _numpy.linspace(visc_range[0], visc_range[1], 5)])
                ax.set_yticklabels(["{:.2f}".format(sp_const) for sp_const in _numpy.linspace(sp_consts.min(), sp_consts.max(), 5)])
                ax.set_xlabel("Mantle viscosity [Pa s]")
                ax.set_ylabel("Slab pull reduction factor")
                ax.scatter(j, i, marker="*", facecolor="none", edgecolor="k", s=30)
                fig.colorbar(im, label = "Residual velocity magnitude [cm/a]")
                plt.show()

            print(f"Optimal coefficients for ", ", ".join(new_plates.name.astype(str)), " plate(s), (PlateIDs: ", ", ".join(new_plates.plateID.astype(str)), ")")
            print("Minimum residual torque: {:.2e} cm/a".format(_numpy.amin(residual)))
            print("Optimum viscosity [Pa s]: {:.2e}".format(visc))
            print("Optimum Drag Coefficient [Pa s/m]: {:.2e}".format(visc / self.mech.La))
            print("Optimum Slab Pull constant: {:.2%}".format(sp_const))

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# SAVING 
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    def save_all(self):
        for reconstruction_time in tqdm(self.times, desc="Saving data", disable=self.DEBUG_MODE):
            for case in self.cases:
                setup.DataFrame_to_csv(self.plates[reconstruction_time][case], "Plates", self.name, reconstruction_time, case, self.dir_path, DEBUG_MODE=self.DEBUG_MODE)
                setup.DataFrame_to_csv(self.slabs[reconstruction_time][case], "Slabs", self.name, reconstruction_time, case, self.dir_path, DEBUG_MODE=self.DEBUG_MODE)
                setup.DataFrame_to_csv(self.points[reconstruction_time][case], "Points", self.name, reconstruction_time, case, self.dir_path, DEBUG_MODE=self.DEBUG_MODE)
            setup.GeoDataFrame_to_shapefile(self.resolved_geometries[reconstruction_time], "Geometries", self.name, reconstruction_time, self.dir_path, DEBUG_MODE=self.DEBUG_MODE)
            setup.Dataset_to_netCDF(self.seafloor[reconstruction_time], "Seafloor", self.name, reconstruction_time, self.dir_path, DEBUG_MODE=self.DEBUG_MODE)
            setup.Dataset_to_netCDF(self.velocity[reconstruction_time], "Velocity", self.name, reconstruction_time, self.dir_path, DEBUG_MODE=self.DEBUG_MODE)

        print(f"All data saved to {self.dir_path}!")

    def save_plates(self):
        for reconstruction_time in tqdm(self.times, desc="Saving plates", disable=self.DEBUG_MODE):
            for case in self.cases:
                setup.DataFrame_to_csv(self.plates[reconstruction_time][case], "Plates", self.name, reconstruction_time, case, self.dir_path, DEBUG_MODE=self.DEBUG_MODE)

        print(f"Plates data saved to {self.dir_path}!")

    def save_slabs(self):
        for reconstruction_time in tqdm(self.times, desc="Saving slabs", disable=self.DEBUG_MODE):
            for case in self.cases:
                setup.DataFrame_to_csv(self.slabs[reconstruction_time][case], "Slabs", self.name, reconstruction_time, case, self.dir_path, DEBUG_MODE=self.DEBUG_MODE)

        print(f"Slabs data saved to {self.dir_path}!")
    
    def save_points(self):
        for reconstruction_time in tqdm(self.times, desc="Saving points", disable=self.DEBUG_MODE):
            for case in self.cases:
                setup.DataFrame_to_csv(self.points[reconstruction_time][case], "Points", self.name, reconstruction_time, case, self.dir_path, DEBUG_MODE=self.DEBUG_MODE)

        print(f"Points data saved to {self.dir_path}!")

    def save_geometries(self):
        for reconstruction_time in tqdm(self.times, desc="Saving geometries", disable=self.DEBUG_MODE):
            for case in self.cases:
                setup.GeoDataFrame_to_shapefile(self.resolved_geometries[reconstruction_time], "Geometries", self.name, reconstruction_time, self.dir_path, DEBUG_MODE=self.DEBUG_MODE)

        print(f"Geometries data saved to {self.dir_path}!")

    def save_seafloor(self):
        for reconstruction_time in tqdm(self.times, desc="Saving seafloor", disable=self.DEBUG_MODE):
            setup.Dataset_to_netCDF(self.seafloor[reconstruction_time], "Seafloor", self.name, reconstruction_time, self.dir_path, DEBUG_MODE=self.DEBUG_MODE)

        print(f"Seafloor data saved to {self.dir_path}!")

    def save_velocity(self):
        for reconstruction_time in tqdm(self.times, desc="Saving velocity", disable=self.DEBUG_MODE):
            setup.Dataset_to_netCDF(self.velocity[reconstruction_time], "Velocity", self.name, reconstruction_time, self.dir_path, DEBUG_MODE=self.DEBUG_MODE)

        print(f"Velocity data saved to {self.dir_path}!")

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# PLOTTING 
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    
    def plot_age_map(
            self,
            ax,
            reconstruction_time: int,
            plotting_options: dict
        ):
        """
        Function to create subplot with global seafloor age.

        :param ax:                  axes object
        :type ax:                   matplotlib.axes.Axes
        :param fig:                 figure
        :type fig:                  matplotlib.figure.Figure
        :param reconstruction_time: the time for which to display the map
        :type reconstruction_time:  int
        :param plotting_options:    options for plotting
        :type plotting_options:     dict

        :return:                    axes object and image object
        :rtype:                     matplotlib.axes.Axes, matplotlib.image.AxesImage
        """
        # Check if reconstruction time is in valid times
        if reconstruction_time not in self.times:
            return print("Invalid reconstruction time")
        
        # Set basemap
        ax, gl = self.plot_basemap(ax)

        # NOTE: We need to explicitly turn of top and right labels here, otherwise they will still show up sometimes
        gl.top_labels = False
        gl.right_labels = False

        # Plot age
        im = ax.imshow(
            self.seafloor[reconstruction_time].seafloor_age.values,
            cmap=plotting_options["age cmap"],
            transform=ccrs.PlateCarree(), 
            zorder=1, 
            vmin=0, 
            vmax=plotting_options["age max"], 
            origin="lower"
        )

        # Plot plates and coastlines
        self.plot_reconstruction(ax, reconstruction_time, plotting_options, plates=True, trenches=True, coastlines="fill")

        return im

    def plot_sediment_map(
            self,
            ax,
            reconstruction_time: int,
            case,
            plotting_options: dict,
        ):
        """
        Function to create subplot with global sediment thicknesses.
        
        :param ax:                  axes object
        :type ax:                   matplotlib.axes.Axes
        :param fig:                 figure
        :type fig:                  matplotlib.figure.Figure
        :param reconstruction_time: the time for which to display the map
        :type reconstruction_time:  int
        :param case:                case for which to plot the sediments
        :type case:                 str
        :param plotting_options:    dictionary with options for plotting
        :type plotting_options:     dict
        :param vmin:                minimum value for the colormap
        :type vmin:                 float
        :param vmax:                maximum value for the colormap
        :type vmax:                 float
        :param cmap:                colormap to use for plotting
        :type cmap:                 str
        """
        # Check if reconstruction time is in valid times
        if reconstruction_time not in self.times:
            return print("Invalid reconstruction time")
        
        # Set basemap
        ax, gl = self.plot_basemap(ax)

        if self.options[case]["Sample sediment grid"] !=0:
            raster = self.seafloor[reconstruction_time][self.options[case]["Sample sediment grid"]].copy()
        else:
            raster = _numpy.where(_numpy.isnan(self.seafloor[reconstruction_time].seafloor_age.values), _numpy.nan, 0)

        if plotting_options["sediment log scale"] is True:
            raster = _numpy.log10(raster + 1)

        # Plot sediment
        im = ax.imshow(
            raster,
            cmap=plotting_options["sediment cmap"],
            transform=ccrs.PlateCarree(), 
            zorder=1, 
            vmin=plotting_options["sediment vmin"], 
            vmax=plotting_options["sediment vmax"], 
            origin="lower"
        )

        if self.options[case]["Active margin sediments"] != 0 or self.options[case]["Sample erosion grid"]:
            data = self.slabs[reconstruction_time][case].copy()
            
            if plotting_options["sediment log scale"] is True:
                data["sediment_thickness"] = _numpy.log10(data["sediment_thickness"] + 1)

            slab_data = ax.scatter(
                data.lon,
                data.lat,
                c=data.sediment_thickness,
                s=plotting_options["marker size"],
                transform=ccrs.PlateCarree(),
                cmap=plotting_options["sediment cmap"],
                vmin=plotting_options["sediment vmin"],
                vmax=plotting_options["sediment vmax"],
            )

        # Plot plates and coastlines
        self.plot_reconstruction(ax, reconstruction_time, plotting_options, plates=True, trenches=True)
            
        return im
    
    def plot_erosion_rate_map(
            self,
            ax,
            reconstruction_time: int,
            plotting_options: dict
        ):
        """
        Function to create subplot with global sediment thicknesses
            case:               case for which to plot the sediments
            plotting_options:   dictionary with options for plotting
        """
        # Check if reconstruction time is in valid times
        if reconstruction_time not in self.times:
            return print("Invalid reconstruction time")
        
        # Set basemap
        ax, gl = self.plot_basemap(ax)

        # Plot sediment
        im = ax.imshow(
            self.seafloor[reconstruction_time].erosion_rate.values,
            cmap=plotting_options["erosion cmap"],
            transform=ccrs.PlateCarree(), 
            zorder=1, 
            vmin=0, 
            vmax=plotting_options["erosion max"], 
            origin="lower"
        )

        # Plot plates and coastlines
        ax = self.plot_reconstruction(ax, reconstruction_time, plotting_options, plates=True, trenches=True, coastlines=False)
            
        return im
    
    def plot_velocity_map(
            self,
            ax,
            reconstruction_time,
            case,
            plotting_options
        ):
        """
        Function to plot plate velocities on an axes object
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

        # Plot velocity grid
        im = ax.imshow(
            self.velocity[reconstruction_time][case].velocity_magnitude.values,
            cmap = plotting_options["velocity cmap"],
            transform=ccrs.PlateCarree(), 
            zorder=1, 
            vmin=0, 
            vmax=plotting_options["velocity max"], 
            origin="lower"
        )

        # Subsample velocity vectors
        velocity_vectors = self.points[reconstruction_time][case].iloc[::209].copy()

        # Normalise, if necessary
        if plotting_options["normalise velocity"] is True:
            velocity_vectors["v_lon"] = velocity_vectors["v_lon"] / velocity_vectors["v_mag"]
            velocity_vectors["v_lat"] = velocity_vectors["v_lat"] / velocity_vectors["v_mag"]

        # Plot velocity vectors
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            qu = ax.quiver(
                x=velocity_vectors.lon,
                y=velocity_vectors.lat,
                u=velocity_vectors.v_lon,
                v=velocity_vectors.v_lat,
                transform=ccrs.PlateCarree(),
                width=4e-3,
                scale=3e2,
                zorder=4,
                color="k",
                alpha=0.5
            )

        # Plot plates and coastlines
        ax = self.plot_reconstruction(ax, reconstruction_time, plotting_options, plates=True, trenches=True, coastlines="edge")

        return im, qu

    def plot_velocity_difference_map(
            self,
            ax,
            reconstruction_time,
            case1,
            case2,
            plotting_options
        ):
        """
        Function to create subplot with difference between plate velocity at trenches between two cases
        
        :param ax:                  axes object
        :type ax:                   matplotlib.axes.Axes
        :param fig:                 figure
        :type fig:                  matplotlib.figure.Figure
        :param reconstruction_time: the time for which to display the map
        :type reconstruction_time:  int
        :param case1:               case 1 for which to use the velocities
        :type case1:                str
        :param case2:               case 2 to subtract from case 1
        :type case2:                str
        :param plotting_options:    dictionary with options for plotting
        :type plotting_options:     dict

        :return:                    image object and quiver object
        :rtype:                     matplotlib.image.AxesImage and matplotlib.quiver.Quiver
        """

        # Check if reconstruction time is in valid times
        if reconstruction_time not in self.times:
            return print("Invalid reconstruction time")
        
        # Set basemap
        ax, gl = self.plot_basemap(ax)

        # Plot velocity grid
        im = ax.imshow(
            self.velocity[reconstruction_time][case1].velocity_magnitude.values-self.velocity[reconstruction_time][case2].velocity_magnitude.values,
            cmap = plotting_options["velocity difference cmap"],
            transform=ccrs.PlateCarree(), 
            zorder=1, 
            vmin=-0.5*plotting_options["velocity max"], 
            vmax=0.5*plotting_options["velocity max"], 
            origin="lower"
        )

        # Subsample velocity vectors
        velocity_vectors1 = self.points[reconstruction_time][case1].iloc[::209].copy()
        velocity_vectors2 = self.points[reconstruction_time][case2].iloc[::209].copy()

        # Plot velocity vectors
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            qu = ax.quiver(
                x=velocity_vectors1.lon,
                y=velocity_vectors1.lat,
                u=velocity_vectors1.v_lon-velocity_vectors2.v_lon,
                v=velocity_vectors1.v_lat-velocity_vectors2.v_lat,
                transform=ccrs.PlateCarree(),
                width=4e-3,
                scale=3e2,
                zorder=4,
                color="k",
                alpha=0.5
            )

        # Plot plates and coastlines
        ax = self.plot_reconstruction(ax, reconstruction_time, plotting_options, plates=True, trenches=True, coastlines="edge")

        return im, qu
    
    def plot_relative_velocity_difference_map(self, ax, reconstruction_time, case1, case2, plotting_options):
        """
        Function to create subplot with difference between plate velocity at trenches between two cases
            case:               case for which to plot the sediments
            plotting_options:   dictionary with options for plotting
        """

        # Check if reconstruction time is in valid times
        if reconstruction_time not in self.times:
            return print("Invalid reconstruction time")
        
        # Set basemap
        ax, gl = self.plot_basemap(ax)

        # Plot velocity grid
        im = ax.imshow(
            _numpy.where(
                (self.velocity[reconstruction_time][case2].velocity_magnitude.values == 0) | (_numpy.isnan(self.velocity[reconstruction_time][case2].velocity_magnitude.values)),
                0,
                self.velocity[reconstruction_time][case1].velocity_magnitude.values/self.velocity[reconstruction_time][case2].velocity_magnitude.values,
            ),
            cmap = plotting_options["relative velocity difference cmap"],
            transform=ccrs.PlateCarree(), 
            zorder=1, 
            vmin=0, 
            vmax=plotting_options["relative velocity max"], 
            origin="lower"
        )

        # Subsample velocity vectors
        velocity_vectors1 = self.points[reconstruction_time][case1].iloc[::209].copy()
        velocity_vectors2 = self.points[reconstruction_time][case2].iloc[::209].copy()

        # Plot velocity vectors
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            qu = ax.quiver(
                x=velocity_vectors1.lon,
                y=velocity_vectors1.lat,
                u=_numpy.where(
                    (velocity_vectors2.v_lon.values == 0) | (velocity_vectors1.v_lon.values == 0) | (_numpy.isnan(velocity_vectors2.v_lon.values)) | (_numpy.isnan(velocity_vectors1.v_lon.values)),
                    0,
                    (velocity_vectors1.v_lon.values - velocity_vectors2.v_lon.values) / velocity_vectors2.v_mag.values * 10,
                ),
                v=_numpy.where(
                    (velocity_vectors2.v_lat.values == 0) | (velocity_vectors1.v_lat.values == 0) | (_numpy.isnan(velocity_vectors2.v_lat.values)) | (_numpy.isnan(velocity_vectors1.v_lat.values)),
                    0,
                    (velocity_vectors1.v_lat.values - velocity_vectors2.v_lat.values) / velocity_vectors2.v_mag.values * 10,
                ),
                transform=ccrs.PlateCarree(),
                width=4e-3,
                scale=3e2,
                zorder=4,
                color="k",
                alpha=0.5
            )

        # Plot plates and coastlines
        ax = self.plot_reconstruction(ax, reconstruction_time, plotting_options, plates=True, trenches=True, coastlines="edge")

        return im, qu

    def plot_basemap(self, ax):
        # Set labels
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")

        # Set global extent
        ax.set_global()

        # Set gridlines
        gl = ax.gridlines(
            crs=ccrs.PlateCarree(), 
            draw_labels=True, 
            linewidth=0.5, 
            color="gray", 
            alpha=0.5, 
            linestyle="--", 
            zorder=5
        )

        # Turn off gridlabels for top and right
        gl.top_labels = False
        gl.right_labels = False  

        return ax, gl
    
    def plot_reconstruction(
            self,
            ax,
            reconstruction_time: int, 
            plotting_options: dict, 
            coastlines=True, 
            plates=False, 
            trenches=False,
            velocities=None,
        ):
        """
        Function to plot reconstructed features: coastlines, plates and trenches

        :param ax:                      axes object
        :type ax:                       matplotlib.axes.Axes
        :param reconstruction_time:     the time for which to display the map
        :type reconstruction_time:      int
        :param plotting_options:        options for plotting
        :type plotting_options:         dict
        :param coastlines:              whether or not to plot coastlines
        :type coastlines:               boolean
        :param plates:                  whether or not to plot plates
        :type plates:                   boolean
        :param trenches:                whether or not to plot trenches
        :type trenches:                 boolean
        :param default_frame:           whether or not to use the default reconstruction
        :type default_frame:            boolean

        :return:                        axes object with plotted features
        :rtype:                         matplotlib.axes.Axes
        """
        # Set gplot object
        gplot = gplately.PlotTopologies(self.reconstruction, time=reconstruction_time, coastlines=self.coastlines)

        # Plot coastlines
        # NOTE: Some reconstructions on the GPlately DataServer do not have polygons for coastlines, that's why we need to catch the exception
        if coastlines == "fill":
            try:
                gplot.plot_coastlines(ax, facecolor="lightgrey", zorder=-5)
            except:
                pass

        if coastlines == "edge":
            try:
                gplot.plot_coastlines(ax, edgecolor="black", facecolor="none", zorder=2, lw=0.1)
            except:
                pass
        
        # Plot plates 
        if plates:
            gplot.plot_all_topologies(ax, lw=plotting_options["linewidth plate boundaries"], zorder=4)
            
        # Plot trenches
        if plates and trenches:
            gplot.plot_subduction_teeth(ax, zorder=4)

        return ax
    
    def plot_torque_through_time(
            self,
            ax,
            torque: str,
            plate: int,
            normalise: Union[str, bool] = "driving_torque_opt",
            case: Optional[str] = None,
            **kwargs,
        ):
        """
        Function to plot torque through time

        :param ax:                  axes object
        :type ax:                   matplotlib.axes.Axes
        :param torque:              torque to plot
        :type torque:               str
        :param plate:               plate for which to plot the torque
        :type plate:                int
        :param case:                case for which to plot the torque
        :type case:                 str
        :param kwargs:              additional keyword arguments
        :type kwargs:               dict

        :return:                    axes object with plotted torque
        :rtype:                     matplotlib.axes.Axes
        """
        # If case is not provided, just use the first case
        if case is None:
            case = self.cases[0]

        # Check if plate is in torques dictionary
        if plate not in self.torques[case].keys():
            return print("Plate not in torques dictionary")
        
        # Check if torque is in columns
        if torque not in self.torques[case][plate].columns:
            return print("Torque not in columns, please choose from: ", ", ".join(self.torques[case][plate].columns))
        
        # Get plot data
        plot_data = self.torques[case][plate][torque]

        if self.DEBUG_MODE:
            print(f"{plate} before normalisation, {plot_data}")

        # Normalise torque
        if normalise is not False:
            plot_data = _numpy.where(
                self.torques[case][plate][torque] != 0.,
                _numpy.where(
                    self.torques[case][plate][normalise] != 0.,
                    self.torques[case][plate][torque] / self.torques[case][plate][normalise],            
                    0.
                ),
                0.
            )

            if self.DEBUG_MODE:
                print(f"{plate} after normalisation, {plot_data}")

        # Mask zeros
        plot_data = _numpy.ma.masked_where(plot_data == 0., plot_data)

        # Plot torque
        pl = ax.plot(
            self.times,
            plot_data,
            **kwargs,
        )
        
        return pl

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# RANDOMISATION
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    # def randomise_trench_azimuth(plateID):
    #     random_value = _numpy.random.normal(0, 2.5)
    #     return plateID, random_value

    # def randomise_slab_age(plateID):
        
    #     return plateID, random_value

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# PARALLELISATION
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    def run_parallel(self, function_to_run):
        num_processes = multiprocessing.cpu_count()
        pool = multiprocessing.Pool(processes=num_processes)

        for reconstruction_time in tqdm(self.times, desc="Reconstruction times"):
            print(f"Running {function_to_run.__name__} at {reconstruction_time} Ma")
            for case in self.cases:
                pool.apply_async(function_to_run, args=(reconstruction_time, case))

        pool.close()
        pool.join()