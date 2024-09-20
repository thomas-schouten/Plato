# %%
# DEFINE TESTS
# Test the settings module
TEST_SETTINGS = True

# Test the reconstruction module
TEST_RECONSTRUCTION = True

# Test the plates module
TEST_PLATES = True

# Test the points module
TEST_POINTS = False

# Test the slabs module
TEST_SLABS = False

# Test the plate torques module
TEST_PLATE_TORQUES = False

# %%
# LOAD PLATO
import os
import sys

# Add the path to the plato directory
new_path = os.path.abspath(os.path.join(os.getcwd(), "..", "plato"))

# Add the path to the plato directory
sys.path.append(new_path)

if new_path not in sys.path:
    print("Error: Path not added")

import settings

# %%
# TEST SETTINGS
if TEST_SETTINGS:
    # Import module
    from settings import Settings

    # Test the settings module
    print("Testing settings module")
    test_settings = Settings(
        name="test",
        ages=[1, 2, 3],
        cases_file="test_cases.xlsx",
        cases_sheet="Sheet1",
        files_dir="output",
        PARALLEL_MODE=False,
        DEBUG_MODE=False,
    )

    # Print atrributes
    print(f"Reconstruction name: {test_settings.name}")
    print(f"Valid reconstruction ages: {test_settings.ages}")
    print(f"Cases: {test_settings.cases}")
    print(f"Options: {test_settings.options}")
    print(f"Directory path: {test_settings.dir_path}")
    print(f"Debug mode: {test_settings.DEBUG_MODE}")
    print(f"Parallel mode: {test_settings.PARALLEL_MODE}")
    print(f"Plate cases: {test_settings.plate_cases}")
    print(f"Slab cases: {test_settings.slab_cases}")
    print(f"Point cases: {test_settings.point_cases}")
    print("Settings module test complete")

# %%
# TEST RECONSTRUCTION
if TEST_RECONSTRUCTION:
    # Import module
    from reconstruction import Reconstruction

    # Test the reconstruction module
    test_reconstruction = Reconstruction(
        "Muller2019",
    )
    
    # Print attributes
    print(f"PlateReconstruction object: {test_reconstruction.plate_reconstruction}")
    print(f"Polygons object: {test_reconstruction.polygons}")
    print(f"Topologies object: {test_reconstruction.topologies}")
    print(f"Coastlines object: {test_reconstruction.coastlines}")

# %%
# TEST PLATES
if TEST_PLATES:
    # Import module
    from plates import Plates

    # Test the plates module
    test_plates = Plates(
        settings=test_settings,

    )

    # Print attributes
    print(f"Plate names: {test_plates.names}")