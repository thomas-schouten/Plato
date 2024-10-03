# IMPORT MODULES
# Standard library imports
import os
import logging
import sys
import traceback

# Third-party imports
import numpy as np
import gplately
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
from plates import Plates
from points import Points
from slabs import Slabs
from grids import Grids
from platetorques import PlateTorques
from globe import Globe
from optimisation import Optimisation
from plot import Plot

def test_settings(settings_file=None):
    """Test the settings module of the plato package."""
    logging.info("Testing settings module...")
    
    # Test initialisation of the Settings object
    try:
        settings_test = Settings(
            name="test",
            ages=[0, 1],
            cases_file=settings_file,
            files_dir="output",
            PARALLEL_MODE=False,
            DEBUG_MODE=False,
        )
        logging.info("Successfully initialised 'Settings' object.")

    except Exception as e:
        logging.error(f"Settings test failed: {e}")
        traceback.print_exc()

    logging.info("Successfully tested settings module.")

    return settings_test

def test_plates(settings=None, settings_file=None, reconstruction_files=None, plates_files=None):
    """Test the plates module of the plato package."""
    logging.info("Testing plates module...")

    # Make a PlateReconstruction object if files are provided
    if reconstruction_files:
        reconstruction = gplately.PlateReconstruction(reconstruction_files[0], reconstruction_files[1])

    # Test initialisation of Plates object
    try:
        plates_test = Plates(
            settings=settings,
            ages=[0, 1], 
            cases_file=settings_file,
            reconstruction=reconstruction,
            files_dir="output",
            data=plates_files,
        )
        logging.info("Successfully initialised 'Plates' object.")

    except Exception as e:
        logging.error(f"An error occurred during initialisation of the 'Plates' object: {e}")
        traceback.print_exc()

        # Set plates_test to None if an error occurs
        plates_test = None

    # Test various functions of the Plates class
    if plates_test is not None:
        # Test calculation of RMS plate velocities
        try:
            plates_test.calculate_rms_velocity()
            logging.info("Successfully calculated RMS velocities.")

        except Exception as e:
            logging.error(f"An error occurred during RMS velocity calculation: {e}")
            traceback.print_exc()

        # Test calculation of plate torques
        try:
            plates_test.calculate_plate_torques()
            logging.info("Successfully calculated plate torques.")

        except Exception as e:
            logging.error(f"An error occurred during plate torque calculation: {e}")
            traceback.print_exc()

        # Test calculation of plate driving torques
        try:
            plates_test.calculate_driving_torques()
            logging.info("Successfully calculated driving torques.")

        except Exception as e:
            logging.error(f"An error occurred during driving torque calculation: {e}")
            traceback.print_exc()

        # Test calculation of plate residual torques
        try:
            plates_test.calculate_residual_torques()
            logging.info("Successfully calculated residual torques.")

        except Exception as e:
            print(f"An error occurred during residual torque calculation: {e}")
            traceback.print_exc()
        
        # Test optimisation of plate torques (this should probably moved to the optimisation module)
        # try:
        #     plates_test.optimise_torques()
        #     if print_results:
        #         print(f"Optimised torques: {plates_test.optimised_torques}")
        # except Exception as e:
        #     print(f"An error occurred during torque optimisation: {e}")
        #     traceback.print_exc()

    logging.info("Successfully completed plates test.")
    
    return plates_test

def test_points(data=None, seafloor_grid=None, print_results=False):
    """Test the points module of the plato package."""
    if print_results:
        print("Testing 'points' module...")

    # Test initialisation of the Points object
    try:
        if data:
            points_test = Points(
                data=data,
            )
        points_test = Points()
        logging.info("Points object initialised successfully.")

    except Exception as e:
        logging.error(f"An error occurred during initialisation of the 'Points' object: {e}")
        traceback.print_exc()
    
    # Test functions of the Points class
    if points_test in locals():
        # Test sampling of seafloor age grid at points
        if seafloor_grid:
            try:
                points_test.sample_points(seafloor_grid = seafloor_grid)

            except Exception as e:
                logging.error(f"An error occurred during testing of the 'sample_points' function: {e}")
                traceback.print_exc()
        else:
            logging.info("No seafloor grid provided for sampling. Testing of 'sample_points()' function skipped.")

        # Test computation of GPE force
        try:
            points_test.compute_gpe_force()

        except Exception as e:
            logging.error(f"An error occurred during testing of the 'compute_gpe_force' function: {e}")
            traceback.print_exc()

        # Test computation of mantle drag force
        try:
            points_test.compute_mantle_drag_force()
            logging.info("Successfully computed mantle drag force.")

        except Exception as e:
            print(f"An error occurred during testing of the 'compute_mantle_drag_force' function: {e}")
            traceback.print_exc()

    logging.info("Testing of the 'points' module complete.")

    return points_test

def test_slabs(data=None, seafloor_grid=None, print_results=False):
    """Test the slabs module of the plato package."""
    if print_results:
        print("Testing slabs module...")

    # Test initialisation of Slabs object
    try:
        if data:
            slabs_test = Slabs(
                data=data,
            )
        slabs_test = Slabs()

    except Exception as e:
        print(f"An error occurred during initialisation of the 'Slabs' object: {e}")
        traceback.print_exc()

    # Test functions of the Slabs object
    if slabs_test in locals():
        # Test sampling of seafloor age grid at slabs and upper plate
        if seafloor_grid:
            try:
                slabs_test.sample_slabs(seafloor_grid = seafloor_grid)
            except Exception as e:
                print(f"An error occurred during 'sample_slabs' function: {e}")
                traceback.print_exc()
            try:
                slabs_test.sample_upper_plates(seafloor_grid = seafloor_grid)
            except Exception as e:
                print(f"An error occurred during 'sample_upper_plates' function: {e}")
                traceback.print_exc()
        else:
            print("No seafloor grid provided for sampling.")

        # Test computation of slab pull force
        try:
            slabs_test.compute_slab_pull_force()

        except Exception as e:
            print(f"An error occurred during testing the 'compute_slab_pull_force' function: {e}")
            traceback.print_exc()

        # Test computation of slab bend force
        try:
            slabs_test.compute_slab_bend_force()

        except Exception as e:
            print(f"An error occurred during testing the 'compute_slab_bend_force' function: {e}")
            traceback.print_exc()

    if print_results:
        print("Testing of the 'slabs' module complete.")

    return slabs_test

def grids_test(print_results=False):
    """Test the grids module of the plato package."""
    if print_results:
        print("Testing grids module...")

    # Test initialisation of Grids object
    try:
        grids_test = Grids()
        
    except Exception as e:
        print(f"An error occurred during initialisation of the 'Grids' object: {e}")
        traceback.print_exc()

    # Test functions of the Grids object
    if grids_test in locals():
        # Test making an xarray dataset from an a series of xarray data arrays
        try:
            grids_test.data_arrays2dataset()

        except Exception as e:
            print(f"An error occurred during testing of the 'data_arrays2dataset' function: {e}")
            traceback.print_exc()

        # Test interpolation of data to the resolution of the seafloor grid
        try:
            grids_test.array2data_array()

        except Exception as e:
            print(f"An error occurred during testing of the 'array2data_array' function: {e}")
            traceback.print_exc()

    if print_results:
        print("Testing of the 'grids' module complete.")

    return grids_test

def test_globe(settings=None, reconstruction=None, reconstruction_files=None, print_results=False):
    """Test the globe module of the plato package."""
    if print_results:
        print("Testing globe module...")

    if settings is None:
        settings = test_settings(print_results=False)
    
    if reconstruction is None:
        reconstruction = test_reconstruction(reconstruction_files=reconstruction_files, print_results=False)
    
    # Test initialisation of the Globe object
    try:
        globe_test = Globe(settings=settings, reconstruction=reconstruction)
        if print_results:
            print(f"Globe data:\n{globe_test.data[settings.cases[0]]}")

    except Exception as e:
        print(f"An error occurred during globe testing: {e}")
        traceback.print_exc()
    
    # Test several functions of the Globe class
    # Test calculation of number of plates
    try:
        globe_test.calculate_number_of_plates()
        if print_results:
            print(f"Number of plates: {globe_test.number_of_plates}")
    except Exception as e:
        print(f"An error occurred during calculation of number of plates: {e}")
        traceback.print_exc()

    # Test calculation of total subduction zone length
    try:
        globe_test.calculate_total_subduction_length()
        if print_results:
            print(f"Total subduction zone length: {globe_test.total_subduction_length}")
    except Exception as e:
        print(f"An error occurred during calculation of total subduction zone length: {e}")
        traceback.print_exc()

    if print_results:
        print(f"Globe test complete.")

    return globe_test

def plot_test(plate_torques=None, print_results=False):
    """Test the plot module of the plato package."""
    if print_results:
        print("Testing plot module...")
    
    # Test initialisation of Plot object
    try:
        if plate_torques:
            plot_test = Plot(plate_torques)
            print(f"Plot test complete.")
        else:
            print("No PlateTorques object provided for plotting.")

    except Exception as e:
        print(f"An error occurred during plot testing: {e}")
        traceback.print_exc()

    # Test several plotting functions of the Plot object
    try:
        pass
    except Exception as e:
        print(f"An error occurred during plotting: {e}")
        traceback.print_exc()

    return plot_test

if __name__ == "__main__":
    # Run tests
    test_settings()
    test_reconstruction()
    test_plates()