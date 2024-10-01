# %%
# IMPORT MODULES
# Standard library imports
import os
import sys
import logging

# Import the test functions
import test_functions

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# SET TESTS
# Test configurations
TEST_CONFIGS = {
    "TEST_SETTINGS": False,
    "TEST_RECONSTRUCTION": False,
    "TEST_LOCAL_FILES": True,
    "TEST_PLATES": True,
    "TEST_POINTS": True,
    "TEST_SLABS": True,
    "TEST_GLOBE": True,
    "TEST_PLATE_TORQUES": True,
}

# Define whether to print results
PRINT_RESULTS = False

# Define reconstruction files
reconstruction_files = (
    os.path.join("data", "Global_EarthByte_230-0Ma_GK07_AREPS.rot"),
    os.path.join("data", "Global_EarthByte_230-0Ma_GK07_AREPS_Topologies.gpml"),
    os.path.join("data", "Global_EarthByte_230-0Ma_GK07_AREPS_Polygons.gpml"),
    os.path.join("data", "Global_EarthByte_230-0Ma_GK07_AREPS_Coastlines.gpml"),
)

def run_tests():
    """Run all specified tests based on the TEST_CONFIGS dictionary."""
    # Test Settings
    if TEST_CONFIGS["TEST_SETTINGS"]:
        try:
            logging.info("Running settings test...")
            test_functions.test_settings(print_results=PRINT_RESULTS)
            logging.info("Settings test completed successfully.")
        except Exception as e:
            logging.error(f"Settings test failed: {e}")

    # Test Reconstruction
    if TEST_CONFIGS["TEST_RECONSTRUCTION"]:
        try:
            logging.info("Running reconstruction test...")
            if TEST_CONFIGS["TEST_LOCAL_FILES"]:
                test_functions.test_reconstruction(reconstruction_files=reconstruction_files, print_results=PRINT_RESULTS)
            else:
                test_functions.test_reconstruction(print_results=PRINT_RESULTS)
            logging.info("Reconstruction test completed successfully.")
        except Exception as e:
            logging.error(f"Reconstruction test failed: {e}")

    # Test Plates
    if TEST_CONFIGS["TEST_PLATES"]:
        try:
            logging.info("Running plates test...")
            if TEST_CONFIGS["TEST_LOCAL_FILES"]:
                test_functions.test_plates(reconstruction_files=reconstruction_files, print_results=PRINT_RESULTS)
            else:
                test_functions.test_plates(print_results=PRINT_RESULTS)
            logging.info("Plates test completed successfully.")
        except Exception as e:
            logging.error(f"Plates test failed: {e}")

    # Test Points (if implemented)
    if TEST_CONFIGS["TEST_POINTS"]:
        try:
            logging.info("Running points test...")
            test_functions.test_points(print_results=PRINT_RESULTS)
            logging.info("Points test completed successfully.")
        except Exception as e:
            logging.error(f"Points test failed: {e}")

    # Test Slabs
    if TEST_CONFIGS["TEST_SLABS"]:
        try:
            logging.info("Running slabs test...")
            test_functions.test_slabs(print_results= PRINT_RESULTS)
            logging.info("Slabs test completed successfully.")
        except Exception as e:
            logging.error(f"Slabs test failed: {e}")

    # Test Plate Torques
    if TEST_CONFIGS["TEST_PLATE_TORQUES"]:
        try:
            logging.info("Running plate torques test...")
            test_functions.test_plate_torques(print_results=PRINT_RESULTS)
            logging.info("Plate torques test completed successfully.")
        except Exception as e:
            logging.error(f"Plate torques test failed: {e}")

    # Test Globe
    if TEST_CONFIGS["TEST_GLOBE"]:
        try:
            logging.info("Running globe test...")
            if TEST_CONFIGS["TEST_LOCAL_FILES"]:
                test_functions.test_globe(reconstruction_files=reconstruction_files, print_results=PRINT_RESULTS)
            else:
                test_functions.test_globe(print_results=PRINT_RESULTS)
            logging.info("Globe test completed successfully.")
        except Exception as e:
            logging.error(f"Globe test failed: {e}")

# RUN TESTS
if __name__ == "__main__":
    run_tests()

# %%
