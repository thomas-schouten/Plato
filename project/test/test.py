# %%
# IMPORT MODULES
# Standard library imports
import os
import sys

# Import the test functions
import test_functions

# SET TESTS
# Test the settings module
# Last successful test: 2024-09-27
TEST_SETTINGS = True

# Test the reconstruction module
# Last successful test: 2024-09-27
TEST_RECONSTRUCTION = True; TEST_LOCAL_FILES = True
reconstruction_files = (
    os.path.join("data", "Global_EarthByte_230-0Ma_GK07_AREPS.rot"),
    os.path.join("data", "Global_EarthByte_230-0Ma_GK07_AREPS_Topologies.gpml"),
    os.path.join("data", "Global_EarthByte_230-0Ma_GK07_AREPS_Polygons.gpml"),
    os.path.join("data", "Global_EarthByte_230-0Ma_GK07_AREPS_Coastlines.gpml"),
)

# Test the plates module
# Last successful test:
TEST_PLATES = False

# Test the points module
# Last successful test:
TEST_POINTS = False

# Test the slabs module
# Last successful test:
TEST_SLABS = True

# Test the plate torques module
# Last successful test:
TEST_PLATE_TORQUES = True

# RUN TESTS
if TEST_SETTINGS:
    test_functions.test_settings()

# TEST RECONSTRUCTION
if TEST_RECONSTRUCTION:
    if TEST_LOCAL_FILES:
        test_functions.test_reconstruction(reconstruction_files=reconstruction_files)
    else:
        test_functions.test_reconstruction()

# TEST PLATES
if TEST_PLATES:
    test_functions.test_plates()

# TEST POINTS
if TEST_POINTS:
    test_functions.test_points()

# %%
