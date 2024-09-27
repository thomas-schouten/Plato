# IMPORT MODULES
# Standard library imports
import os
import sys

# Third-party imports
import numpy as np
import pandas as pd

# Local application imports
# Add the path to the plato directory
new_path = os.path.abspath(os.path.join(os.getcwd(), "..", "plato"))

# Check if the path is successfully added
if new_path not in sys.path:
    sys.path.append(new_path)
    if new_path not in sys.path:
        raise RuntimeError("Error: Path not added")

from settings import Settings
from reconstruction import Reconstruction
from plates import Plates
from points import Points
from slabs import Slabs
from platetorques import PlateTorques
from globe import Globe

def test_settings(print_results=True):
    """Test the settings module of the plato package."""
    if print_results:
        print("Testing settings module...")

    try:
        settings_test = Settings(
            name="test",
            ages=[0, 1],
            cases_file="test_cases.xlsx",
            cases_sheet="Sheet1",
            files_dir="output",
            PARALLEL_MODE=False,
            DEBUG_MODE=False,
        )

        if print_results:
            print(f"Reconstruction name: {settings_test.name}")
            print(f"Valid reconstruction ages: {settings_test.ages}")
            print(f"Cases: {settings_test.cases}")
            print(f"Options: {settings_test.options}")
            print(f"Directory path: {settings_test.dir_path}")
            print(f"Debug mode: {settings_test.DEBUG_MODE}")
            print(f"Parallel mode: {settings_test.PARALLEL_MODE}")
            print(f"Slab cases: {settings_test.slab_cases}")
            print(f"Point cases: {settings_test.point_cases}")
            print("Settings module test complete.")

    except Exception as e:
        print(f"An error occurred during settings testing: {e}")

    return settings_test

def test_reconstruction(reconstruction_files=None, model_name="Muller2016", print_results=True):
    """Test the reconstruction module of the plato package."""
    if print_results:
        print("Testing reconstruction module...")

    try:
        reconstruction_test = Reconstruction(
            model_name,
            rotation_file=reconstruction_files[0] if reconstruction_files else None,
            topology_file=reconstruction_files[1] if reconstruction_files else None,
            polygon_file=reconstruction_files[2] if reconstruction_files else None,
            coastline_file=reconstruction_files[3] if reconstruction_files else None,
        )
        
        if print_results:
            print(f"PlateReconstruction object: {reconstruction_test.plate_reconstruction}")
            print(f"Polygons object: {reconstruction_test.polygons}")
            print(f"Topologies object: {reconstruction_test.topology_features}")
            print(f"Coastlines object: {reconstruction_test.coastlines}")

    except Exception as e:
        print(f"An error occurred during reconstruction testing: {e}")

    return reconstruction_test

def test_plates(settings=None, reconstruction_files=None, plates_files=None, print_results=True):
    """Test the plates module of the plato package."""
    if print_results:
        print("Testing plates module...")

    if settings is None:
        settings = test_settings(print_results=False)
    
    if reconstruction_files:
        reconstruction = test_reconstruction(reconstruction_files=reconstruction_files, print_results=False)
    else:
        reconstruction = test_reconstruction(print_results=False)

    try:
        plates_test = Plates(
            settings=settings,
            reconstruction=reconstruction,
        )

        # Select first entry in the ages list
        test_age = plates_test.settings.ages[0]
        test_case = plates_test.settings.cases[0]
        if print_results:
            print(f"Plates for reconstruction: {plates_test.settings.name}")
            print(f"Plate data at {test_age} Ma for case {test_case}:\n{plates_test.data[test_age][test_case]}")
            print(f"Plate geometry at {test_age} for case {test_case}:\n{plates_test.resolved_geometries[test_age]}")

        # Test various functions of the Plates class
        plates_test.calculate_rms_velocity()
        if print_results:
            print(f"RMS velocities: {plates_test.rms_velocities}")

        plates_test.calculate_plate_torques()
        if print_results:
            print(f"Plate torques: {plates_test.plate_torques}")

        plates_test.calculate_driving_torques()
        if print_results:
            print(f"Driving torques: {plates_test.driving_torques}")

        plates_test.calculate_residual_torques()
        if print_results:
            print(f"Residual torques: {plates_test.residual_torques}")

        plates_test.optimise_torques()
        if print_results:
            print(f"Optimised torques: {plates_test.optimised_torques}")

    except Exception as e:
        print(f"An error occurred during plates testing: {e}")

    return plates_test

def test_points(data=None, print_results=True):
    """Test the points module of the plato package."""
    if print_results:
        print("Testing points module...")

    try:
        if data:
            points_test = Points(
                data=data,
            )
        points_test = Points()
        print(f"Points test complete.")

    except Exception as e:
        print(f"An error occurred during points testing: {e}")

    return points_test

def test_slabs(data=None, print_results=True):
    """Test the slabs module of the plato package."""
    if print_results:
        print("Testing slabs module...")

    try:
        if data:
            slabs_test = Slabs(
                data=data,
            )
        slabs_test = Slabs()
        print(f"Slabs test complete.")

    except Exception as e:
        print(f"An error occurred during slabs testing: {e}")

    return slabs_test

def test_globe(settings=None, reconstruction=None, reconstruction_files=None, print_results=True):
    """Test the globe module of the plato package."""
    if print_results:
        print("Testing globe module...")

    if settings is None:
        settings = test_settings(print_results=False)
    
    if reconstruction is None:
        reconstruction = test_reconstruction(reconstruction_files=reconstruction_files, print_results=False)

    try:
        # Create a Globe object
        globe_test = Globe(settings=settings, reconstruction=reconstruction)
        if print_results:
            print(f"Globe data:\n{globe_test.data[settings.cases[0]]}")

        # Test calculation of number of plates
        globe_test.calculate_number_of_plates()
        if print_results:
            print(f"Number of plates: {globe_test.number_of_plates}")

        # Test calculation of total subduction zone length
        globe_test.calculate_total_subduction_length()
        if print_results:
            print(f"Total subduction zone length: {globe_test.total_subduction_length}")
        
        print(f"Globe test complete.")

    except Exception as e:
        print(f"An error occurred during globe testing: {e}")

    return globe_test

if __name__ == "__main__":
    # Run tests
    test_settings()
    test_reconstruction()
    test_plates()