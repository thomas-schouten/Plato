# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# PLATO
# Algorithm to calculate plate forces from tectonic reconstructions
# Plates object
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
# PLATES OBJECT
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

class Plates:
    def __init__(
            self,
            _settings,
            _reconstruction,
        ):
        # Store input variables
        self.settings = _settings
        self.reconstruction = _reconstruction
        
        # GEOMETRIES
        # Set up plate reconstruction object and initialise dictionaries to store resolved topologies and geometries
        self.resolved_topologies, self.resolved_geometries = {}, {}

        # Load or initialise plate geometries
        for _age in tqdm(self.settings.ages, desc="Loading geometries", disable=self.DEBUG_MODE):
            
            # Load resolved geometries if they are available
            self.resolved_geometries[_age] = setup.GeoDataFrame_from_geoparquet(
                self.settings.dir_path,
                "Geometries",
                _age,
                self.settings.name,
            )

            # Get new topologies if they are unavailable
            if self.resolved_geometries[_age] is None:
                self.resolved_geometries[_age] = setup.get_topology_geometries(
                    self.reconstruction, _age, anchor_plateID=0
                )
            
            # Resolve topologies to use to get plates
            # NOTE: This is done because some information is retrieved from the resolved topologies and some from the resolved geometries
            #       This step could be sped up by extracting all information from the geopandas DataFrame, but so far this has not been the main bottleneck
            # Ignore annoying warnings that the field names are laundered
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    action="ignore",
                    message="Normalized/laundered field name:"
                )
                self.resolved_topologies[_age] = []
                _pygplates.resolve_topologies(
                    self.reconstruction.topologies,
                    self.reconstruction.rotations, 
                    self.resolved_topologies[_age], 
                    _age, 
                    anchor_plate_id=0
                )

        # DATA
        # Load or initialise plate data
        self.data = setup.load_data(
            self.data,
            self.reconstruction,
            self.settings.name,
            self.settings.ages,
            "Plates",
            self.settings.cases,
            self.settings.options,
            self.plate_cases,
            self.settings.dir_path,
            resolved_topologies = self.resolved_topologies,
            resolved_geometries = self.resolved_geometries,
            DEBUG_MODE = self.settings.DEBUG_MODE,
            PARALLEL_MODE = self.settings.PARALLEL_MODE,
        )

    def calculate_rms_velocity(
                self,
            ):
            """
            Function to calculate the root mean square (RMS) velocity of the plates.
            """
            for _age in self.settings.ages:
                # Calculate rms velocity
                for key, entries in self.settings.gpe_cases.items():
                    if self.plates[self.settings.ages][key]["v_rms_mag"].mean() == 0:
                        self.plates[self.settings.ages][key] = functions_main.compute_rms_velocity(
                            self.plates[self.settings.ages][key],
                            self.points[self.settings.ages][key]
                        )
                
                    # Copy DataFrames to other cases
                    for entry in entries[1:]:
                        if self.plates[_age][entry]["v_rms_mag"].mean() == 0:
                            self.plates[_age][entry]["v_rms_mag"] = self.plates[_age][key]["v_rms_mag"]

                        if self.plates[_age][entry]["v_rms_azi"].mean() == 0:
                            self.plates[_age][entry]["v_rms_azi"] = self.plates[_age][key]["v_rms_azi"]

                        if self.plates[_age][entry]["omega_rms"].mean() == 0:
                            self.plates[_age][entry]["omega_rms"] = self.plates[_age][key]["omega_rms"]