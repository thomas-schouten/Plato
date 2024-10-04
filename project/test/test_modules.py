# %%
# TEST PLATO MODULES

# Import the functions to test
# Standard library imports
import os
import sys
import logging

# Import the test functions
import functions_test

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# SET TESTS
# Test configurations
TEST_CONFIGS = {
    "TEST_SETTINGS": False,
    "TEST_LOCAL_FILES": True,
    "TEST_PLATES": False,
    "TEST_POINTS": True,
    "TEST_SLABS": False,
    "TEST_GLOBE": False,
    "TEST_PLATE_TORQUES": False,
}

# Define settings file
settings_file = os.path.join("cases_test.xlsx")

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
        if TEST_CONFIGS["TEST_LOCAL_FILES"]:
            functions_test.test_settings(settings_file=settings_file)
        else:
            functions_test.test_settings()

    # Test Plates
    if TEST_CONFIGS["TEST_PLATES"]:
        if TEST_CONFIGS["TEST_LOCAL_FILES"]:
            functions_test.test_plates(settings_file=settings_file, reconstruction_files=reconstruction_files, )
        else:
            functions_test.test_plates(settings_file=settings_file)

    # Test Points (if implemented)
    if TEST_CONFIGS["TEST_POINTS"]:
        if TEST_CONFIGS["TEST_LOCAL_FILES"]:
            functions_test.test_points(settings_file=settings_file, reconstruction_files=reconstruction_files, )
        else:
            functions_test.test_points(settings_file=settings_file)

    # Test Slabs
    if TEST_CONFIGS["TEST_SLABS"]:
        try:
            logging.info("Running slabs test...")
            functions_test.test_slabs(print_results= PRINT_RESULTS)
            logging.info("Slabs test completed successfully.")
        except Exception as e:
            logging.error(f"Slabs test failed: {e}")

    # Test Plate Torques
    if TEST_CONFIGS["TEST_PLATE_TORQUES"]:
        try:
            logging.info("Running plate torques test...")
            functions_test.test_plate_torques(print_results=PRINT_RESULTS)
            logging.info("Plate torques test completed successfully.")
        except Exception as e:
            logging.error(f"Plate torques test failed: {e}")

    # Test Globe
    if TEST_CONFIGS["TEST_GLOBE"]:
        try:
            logging.info("Running globe test...")
            if TEST_CONFIGS["TEST_LOCAL_FILES"]:
                functions_test.test_globe(reconstruction_files=reconstruction_files, print_results=PRINT_RESULTS)
            else:
                functions_test.test_globe(print_results=PRINT_RESULTS)
            logging.info("Globe test completed successfully.")
        except Exception as e:
            logging.error(f"Globe test failed: {e}")

# RUN TESTS
if __name__ == "__main__":
    run_tests()

# %%
