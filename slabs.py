# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# PLATO
# Algorithm to calculate plate forces from tectonic reconstructions
# Slabs object
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
import sys

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# SLABS OBJECT
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

class Slabs:
    def __init__(
            self,
            _reconstruction: object,
            _ages: List or _numpy.ndarray,
            _cases: List[str],
            _options: dict,
            _files_dir: str,
        ):
        """
        Slabs object. Contains all information on slabs
        """
        # Store valid ages
        self.ages = _ages

        # Group cases for torque computation
        # Slab pull cases
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
        self.slab_pull_cases = setup.process_cases(_cases, _options, slab_pull_options)

        # Slab bend cases
        slab_bend_options = ["Slab bend torque", "Seafloor age profile"]
        self.slab_bend_cases = setup.process_cases(_cases, _options, slab_bend_options)

        # Get the slab data
        self.data = {}

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
            _files_dir,
            plates = self.plates,
            resolved_geometries = self.resolved_geometries,
            DEBUG_MODE = self.DEBUG_MODE,
            PARALLEL_MODE = self.PARALLEL_MODE,
        )

    def sample_seafloor(
          self,

    ):
        """
        Sample the slab data
        """
        # Get the slab data
        self.data = functions_main.get_slab_data()