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
    "TEST_LOCAL_FILES": False,
    "TEST_FUNCTIONS": False,
    "TEST_PLATES": False,
    "TEST_POINTS": False,
    "TEST_SLABS": False,
    "TEST_GRIDS": True,
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

# Define test ages
test_ages = [0, 50]

# # Define seafloor files
# seafloor_files = {}
# for age in test_ages:
#     seafloor_files[age] = os.path.join("data", f"age_{age}.nc")

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
            functions_test.test_plates(settings_file=settings_file, reconstruction_files=reconstruction_files, test_functions=TEST_CONFIGS["TEST_FUNCTIONS"])
        else:
            functions_test.test_plates(settings_file=settings_file, test_functions=TEST_CONFIGS["TEST_FUNCTIONS"])

    # Test Points (if implemented)
    if TEST_CONFIGS["TEST_POINTS"]:
        if TEST_CONFIGS["TEST_LOCAL_FILES"]:
            functions_test.test_points(settings_file=settings_file, reconstruction_files=reconstruction_files, test_functions=TEST_CONFIGS["TEST_FUNCTIONS"])
        else:
            functions_test.test_points(settings_file=settings_file, test_functions=TEST_CONFIGS["TEST_FUNCTIONS"])

    # Test Slabs
    if TEST_CONFIGS["TEST_SLABS"]:
        if TEST_CONFIGS["TEST_LOCAL_FILES"]:
            functions_test.test_slabs(settings_file=settings_file, reconstruction_files=reconstruction_files, test_functions=TEST_CONFIGS["TEST_FUNCTIONS"])
        else:
            functions_test.test_slabs(settings_file=settings_file, test_functions=TEST_CONFIGS["TEST_FUNCTIONS"])
    
    # Test Globe
    if TEST_CONFIGS["TEST_GLOBE"]:
        if TEST_CONFIGS["TEST_LOCAL_FILES"]:
            functions_test.test_globe(settings_file=settings_file, reconstruction_files=reconstruction_files, test_functions=TEST_CONFIGS["TEST_FUNCTIONS"])
        else:
            functions_test.test_globe(settings_file=settings_file, test_functions=TEST_CONFIGS["TEST_FUNCTIONS"])

    # Test Grids
    if TEST_CONFIGS["TEST_GRIDS"]:
        if TEST_CONFIGS["TEST_LOCAL_FILES"]:
            functions_test.test_grids(seafloor_files=seafloor_files, test_functions=TEST_CONFIGS["TEST_FUNCTIONS"])
        else:
            functions_test.test_grids(test_functions=TEST_CONFIGS["TEST_FUNCTIONS"])

    # Test Plate Torques
    if TEST_CONFIGS["TEST_PLATE_TORQUES"]:
        if TEST_CONFIGS["TEST_LOCAL_FILES"]:
            functions_test.test_plate_torques(settings_file=settings_file, reconstruction_files=reconstruction_files, test_functions=TEST_CONFIGS["TEST_FUNCTIONS"])
        else:
            functions_test.test_plate_torques(settings_file=settings_file, test_functions=TEST_CONFIGS["TEST_FUNCTIONS"])

# RUN TESTS
if __name__ == "__main__":
    run_tests()

# %%