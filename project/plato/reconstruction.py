# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# PLATO
# Algorithm to calculate plate forces from tectonic reconstructions
# Reconstruction object
# Thomas Schouten, 2024
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
import sys

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# RECONSTRUCTION OBJECT
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

class Reconstruction:
    def __init__(
            self,
            name: str,
            ages: List[int],
            dir_path: str,
            rotation_file: Optional[str] = None,
            topology_file: Optional[str] = None,
            polygon_file: Optional[str] = None,
            coastline_file: Optional[str] = None,
        ):

        # Let the user know you're busy setting up the plate reconstruction
        print("Setting up plate reconstruction...")

        # Set connection to GPlately DataServer if one of the required files is missing
        if not rotation_file or not topology_file or not polygon_file or not coastline_file:
            gdownload = gplately.DataServer(name)

        # Download reconstruction files if rotation or topology file not provided
        if not rotation_file or not topology_file or not polygon_file:
            # Check if the reconstruction has topology features
            valid_reconstructions = [
                "Muller2019", 
                "Muller2016", 
                "Merdith2021", 
                "Cao2020", 
                "Clennett2020", 
                "Seton2012", 
                "Matthews2016", 
                "Merdith2017", 
                "Li2008", 
                "Pehrsson2015", 
                "Young2019", 
                "Scotese2008",
                "Clennett2020_M19",
                "Clennett2020_S13",
                "Muller2020",
                "Shephard2013",
            ]
            
            if name in valid_reconstructions:
                print(f"Downloading {name} reconstruction files from the GPlately DataServer...")
                reconstruction_files = gdownload.get_plate_reconstruction_files()
            
        # Initialise RotationModel object
        if rotation_file:
            self.rotations = _pygplates.RotationModel(rotation_file)
        else: 
            self.rotations = reconstruction_files[0]

        # Initialise topology FeatureCollection object
        if topology_file:
            self.topologies = _pygplates.FeatureCollection(topology_file)
        else:
            self.topologies = reconstruction_files[1]

        # Initialise static polygons FeatureCollection object
        if polygon_file:
            self.polygons = _pygplates.FeatureCollection(polygon_file)
        else:
            self.polygons = reconstruction_files[2]
        
        # Initialise coastlines FeatureCollection object
        if coastline_file:
            self.coastlines = _pygplates.FeatureCollection(coastline_file)
        else:
            self.coastlines, _, _ = gdownload.get_topology_geometries()

        self.reconstruction = gplately.PlateReconstruction(self.rotations, self.topologies, self.polygons)
            
        print("Plate reconstruction ready!")