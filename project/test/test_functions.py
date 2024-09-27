# IMPORT MODULES
# Standard library imports
import os
import sys

# Third party imports
import numpy as np
import pandas as pd

# Local application imports
# Add the path to the plato directory
new_path = os.path.abspath(os.path.join(os.getcwd(), "..", "plato"))

# Add the path to the plato directory
sys.path.append(new_path)

if new_path not in sys.path:
    print("Error: Path not added")

from settings import Settings
from reconstruction import Reconstruction
from plates import Plates
from points import Points
from slabs import Slabs
from platetorques import PlateTorques

# Define the test functions
def test_settings(print_results=True):
    """
    Test the settings module of the plato package.
    """
    if print_results:
        print("Testing settings module")

    settings_test = Settings(
        name="test",
        ages=[1, 2, 3],
        cases_file="test_cases.xlsx",
        cases_sheet="Sheet1",
        files_dir="output",
        PARALLEL_MODE=False,
        DEBUG_MODE=False,
    )

    # Print atrributes
    if print_results:
        print(f"Reconstruction name: {settings_test.name}")
        print(f"Valid reconstruction ages: {settings_test.ages}")
        print(f"Cases: {settings_test.cases}")
        print(f"Options: {settings_test.options}")
        print(f"Directory path: {settings_test.dir_path}")
        print(f"Debug mode: {settings_test.DEBUG_MODE}")
        print(f"Parallel mode: {settings_test.PARALLEL_MODE}")
        # print(f"Plate cases: {settings_test.plate_cases}")
        print(f"Slab cases: {settings_test.slab_cases}")
        print(f"Point cases: {settings_test.point_cases}")
        print("Settings module test complete")

    return settings_test

def test_reconstruction(reconstruction_files=None, model_name="Muller2016", print_results=True):
    """
    Test the reconstruction module of the plato package.
    """
    if print_results:
        print("Testing reconstruction module")

    # Test the reconstruction module
    if reconstruction_files is not None:
        reconstruction_test = Reconstruction(
            model_name,
            rotation_file=reconstruction_files[0],
            topology_file=reconstruction_files[1],
            polygon_file=reconstruction_files[2],
            coastline_file=reconstruction_files[3],
        )
    else:
        reconstruction_test = Reconstruction(
            model_name,
        )
    
    # Print attributes
    if print_results:
        print(f"PlateReconstruction object: {reconstruction_test.plate_reconstruction}")
        print(f"Polygons object: {reconstruction_test.polygons}")
        print(f"Topologies object: {reconstruction_test.topology_features}")
        print(f"Coastlines object: {reconstruction_test.coastlines}")

    return reconstruction_test

def test_plates(settings=None, reconstruction=None, print_results=True):
    """
    Test the plates module of the plato package.
    """
    if print_results:
        print("Testing plates module")
        
    # First get the settings and the reconstruction
    if settings is None:
        settings = test_settings(print_results=False)
    
    if reconstruction is None:
        reconstruction = test_reconstruction(print_results=False)

    # Test the plates module
    plates_test = Plates(
        settings=settings,
        reconstruction=reconstruction,
    )

    # Print attributes
    if print_results:
        # Select first entry in the ages list
        test_age = plates_test.settings.ages[0]
        test_case = plates_test.settings.cases[0]
        print(f"Plates for reconstruction: {plates_test.settings.name}")
        print(f"Plate data at {test_age} Ma for case {test_case}:\n{plates_test.data[test_age][test_case]}")
        print(f"Plate geometry at {test_age} for case {test_case}:\n{plates_test.resolved_geometries[test_age]}")

    return plates_test
# %%
