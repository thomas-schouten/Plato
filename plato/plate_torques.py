# Standard libraries
import os
import logging
import sys

# Local libraries
from . import utils_data, utils_init
from .globe import Globe
from .grids import Grids
from .plates import Plates
from .points import Points
from .slabs import Slabs
from .settings import Settings

class PlateTorques():
    """
    PlateTorques class.

    :param settings:            `Settings` object (default: None)
    :type settings:             plato.settings.Settings
    :param reconstruction:      `Reconstruction` object (default: None)
    :type reconstruction:       gplately.PlateReconstruction
    :param rotation_file:       filepath to .rot file with rotation poles (default: None)
    :type rotation_file:        str
    :param topology_file:       filepath to .gpml file with topologies (default: None)
    :type topology_file:        str
    :param polygon_file:        filepath to .gpml file with polygons (default: None)
    :type polygon_file:         str
    :param reconstruction_name: model name string identifiers for the GPlately DataServer (default: None)
    :type reconstruction_name:  str
    :param ages:                ages of interest (default: None)
    :type ages:                 float, int, list, numpy.ndarray
    :param cases_file:          filepath to excel file with cases (default: None)
    :type cases_file:           str
    :param cases_sheet:         name of the sheet in the excel file with cases (default: "Sheet1")
    :type cases_sheet:          str
    :param files_dir:           directory to store files (default: None)
    :type files_dir:            str
    :param seafloor_age_grids:  seafloor age grids (default: None)
    :type seafloor_age_grids:   dict, xarray.Dataset
    :param sediment_grids:      sediment thickness grids (default: None)
    :type sediment_grids:       dict, xarray.Dataset
    :param continental_grids:   continental crust thickness grids (default: None)
    :type continental_grids:    dict, xarray.Dataset
    :param velocity_grids:      velocity grids (default: None)
    :type velocity_grids:       dict, xarray.Dataset
    :param plates:              `Plates` object (default: None)
    :type plates:               plato.plates.Plates
    :param points:              `Points` object (default: None)
    :type points:               plato.points.Points
    :param slabs:               `Slabs` object (default: None)
    :type slabs:                plato.slabs.Slabs
    :param grids:               `Grids` object (default: None)
    :type grids:                plato.grids.Grids
    :param globe:               `Globe` object (default: None)
    :type globe:                plato.globe.Globe
    :param DEBUG_MODE:          debug mode flag (default: False)
    :type DEBUG_MODE:           bool
    :param PARALLEL_MODE:       parallel mode flag (default: False)
    :type PARALLEL_MODE:        bool
    """
    def __init__(
            self,
            settings = None,
            reconstruction = None,
            rotation_file = None,
            topology_file = None,
            polygon_file = None,
            reconstruction_name = None,
            ages = None,
            cases_file = None,
            cases_sheet = "Sheet1",
            files_dir = None,
            seafloor_age_grids = None,
            sediment_grids = None,
            continental_grids = None,
            velocity_grids = None,
            plates = None,
            points = None,
            slabs = None,
            grids = None,
            globe = None,
            DEBUG_MODE = False,
            PARALLEL_MODE = False
        ):
        # Store settings object
        self.settings = utils_init.get_settings(
            settings, 
            reconstruction_name,
            ages, 
            cases_file,
            cases_sheet,
            files_dir,
            PARALLEL_MODE = PARALLEL_MODE,
            DEBUG_MODE = DEBUG_MODE,
        )
            
        # Store reconstruction object
        self.reconstruction = utils_init.get_reconstruction(
            reconstruction,
            rotation_file,
            topology_file,
            polygon_file,
            reconstruction_name,
        )

        # Get plates object
        if plates:
            self.plates = plates
        else:
            self.plates = Plates(
                self.settings,
                self.reconstruction,
            )

        # Get points object
        if points:
            self.points = points
        else:
            self.points = Points(
                self.settings,
                self.reconstruction,
                resolved_geometries = self.plates.resolved_geometries
            )

        # Get slabs object
        if slabs:
            self.slabs = slabs
        else:
            self.slabs = Slabs(
                self.settings,
                self.reconstruction,
                resolved_geometries = self.plates.resolved_geometries,
            )

        # Get grids object
        if grids:
            self.grids = grids
        else:
            self.grids = Grids(
                self.settings,
                self.reconstruction,
                seafloor_age_grids = seafloor_age_grids,
                sediment_grids = sediment_grids,
                continental_grids = continental_grids,
                velocity_grids = velocity_grids,
            )

        # Get globe object
        if globe:
            self.globe = globe
        else:
            self.globe = Globe(
                self.settings,
                self.reconstruction,
                plates = self.plates,
                points = self.points,
                slabs = self.slabs,
            )

        # Calculate RMS plate velocities
        self.calculate_rms_velocity()

        logging.info("PlateTorques object successfully instantiated!")

    def add_grid(
            self,
            input_grids,
            variable_name = "new_grid",
            grid_type = "seafloor_age",
            target_variable = "z",
            mask_continents = False,
            prefactor = 1.
        ):
        """
        Function to add a grid to the grids object. This calls the add_grid method in the Grids class.

        :param input_grids:         input grids to add to the grids object
        :type input_grids:          dict, xarray.Dataset
        :param variable_name:       name of the variable to add (default: "new_grid")
        :type variable_name:        str
        :param grid_type:           type of grid to add (default: "seafloor_age")
        :type grid_type:            str
        :param target_variable:     target variable to use (default: "z")
        :type target_variable:      str
        :param mask_continents:     mask continents flag (default: False)
        :type mask_continents:      bool
        :param prefactor:           prefactor to apply to the grid (default: 1.)
        :type prefactor:            float
        """
        # Add grid
        self.grids.add_grid(
            input_grids,
            variable_name,
            grid_type,
            target_variable,
            mask_continents,
            prefactor,
        )

    def calculate_rms_velocity(
            self,
            ages = None,
            cases = None,
        ):
        """
        Function to calculate root mean square velocities for plates.
        """
        # Calculate rms velocity
        self.plates.calculate_rms_velocity(
            self.points,
            ages,
            cases,
        )

    def calculate_net_rotation(
            self,
            ages = None,
            cases = None,
            plateIDs = None,
        ):
        """
        Function to calculate net rotation of the entire lithosphere.
        This calls the calculate_net_rotation method in the Globe class.

        :param ages:        ages of interest (default: None)
        :type ages:         float, int, list, numpy.ndarray
        :param cases:       cases of interest (default: None)
        :type cases:        str, list
        :param plateIDs:    plateIDs of interest (default: None)
        :type plateIDs:     int, float, list, numpy.ndarray
        """
        # Calculate net rotation
        self.globe.calculate_net_rotation(
            self.plates,
            self.points,
            ages,
            cases,
            plateIDs,
        )

    def sample_all(
            self,
            ages = None,
            cases = None,
            plateIDs = None,
        ):
        """
        Function to sample all variables relevant to the plate torques calculation.
        This calls three methods from the slabs class: sample_seafloor_ages, sample_slab_sediment_thicknesses, and sample_slab_sediment_thicknesses.

        :param ages:        ages of interest (default: None)
        :type ages:         float, int, list, numpy.ndarray
        :param cases:       cases of interest (default: None)
        :type cases:        str, list
        :param plateIDs:    plateIDs of interest (default: None)
        :type plateIDs:     int, float, list, numpy.ndarray
        """
        # Sample point seafloor ages
        self.sample_point_seafloor_ages(ages, cases, plateIDs)

        # Sample slab seafloor ages
        self.sample_slab_seafloor_ages(ages, cases, plateIDs)

        # Sample arc seafloor ages
        self.sample_arc_seafloor_ages(ages, cases, plateIDs)

        # Sample slab sediment thicknesses
        self.sample_slab_sediment_thicknesses(ages, cases, plateIDs)

    def sample_seafloor_ages(
            self,
            ages = None,
            cases = None,
            plateIDs = None,
        ):
        """
        Function to sample the seafloor ages and other variables (if available).
        This calls the sample_seafloor_ages method for the Points and Slabs class.

        :param ages:        ages of interest (default: None)
        :type ages:         float, int, list, numpy.ndarray
        :param cases:       cases of interest (default: None)
        :type cases:        str, list
        :param plateIDs:    plateIDs of interest (default: None)
        :type plateIDs:     int, float, list, numpy.ndarray
        """
        # Sample points
        self.sample_point_seafloor_ages(ages, cases, plateIDs)
    
        # Sample slabs
        self.sample_slab_seafloor_ages(ages, cases, plateIDs)

    def sample_point_seafloor_ages(
            self,
            ages = None,
            cases = None,
            plateIDs = None,
        ):
        """
        Function to sample the seafloor ages and other variables (if available).

        :param ages:        ages of interest (default: None)
        :type ages:         float, int, list, numpy.ndarray
        :param cases:       cases of interest (default: None)
        :type cases:        str, list
        :param plateIDs:    plateIDs of interest (default: None)
        :type plateIDs:     int, float, list, numpy.ndarray
        """
        self.points.sample_seafloor_ages(ages, cases, plateIDs, self.grids.seafloor_age)

    def sample_slab_seafloor_ages(
            self,
            ages = None,
            cases = None,
            plateIDs = None,
        ):
        """
        Function to sample the seafloor ages and other variables (if available).

        :param ages:        ages of interest (default: None)
        :type ages:         float, int, list, numpy.ndarray
        :param cases:       cases of interest (default: None)
        :type cases:        str, list
        :param plateIDs:    plateIDs of interest (default: None)
        :type plateIDs:     int, float, list, numpy.ndarray
        """
        self.slabs.sample_slab_seafloor_ages(ages, cases, plateIDs, self.grids.seafloor_age)

    def sample_arc_seafloor_ages(
            self,
            ages = None,
            cases = None,
            plateIDs = None,
        ):
        """
        Function to sample the seafloor ages and other variables (if available).

        :param ages:        ages of interest (default: None)
        :type ages:         float, int, list, numpy.ndarray
        :param cases:       cases of interest (default: None)
        :type cases:        str, list
        :param plateIDs:    plateIDs of interest (default: None)
        :type plateIDs:     int, float, list, numpy.ndarray
        """
        self.slabs.sample_arc_seafloor_ages(ages, cases, plateIDs, self.grids.seafloor_age)

    def sample_slab_sediment_thicknesses(
            self,
            ages = None,
            cases = None,
            plateIDs = None,
        ):
        """
        Function to sample the seafloor ages and other variables (if available).

        :param ages:        ages of interest (default: None)
        :type ages:         float, int, list, numpy.ndarray
        :param cases:       cases of interest (default: None)
        :type cases:        str, list
        :param plateIDs:    plateIDs of interest (default: None)
        :type plateIDs:     int, float, list, numpy.ndarray
        """
        # Sample arc seafloor ages
        self.sample_arc_seafloor_ages(ages, cases, plateIDs)

        # Sample slab sediment thickness
        self.slabs.sample_slab_sediment_thickness(ages, cases, plateIDs, self.grids.sediment)

    def calculate_all_torques(
            self,
            ages = None,
            cases = None,
            plateIDs = None,
        ):
        """
        Function to calculate all torques.

        :param ages:        ages of interest (default: None)
        :type ages:         float, int, list, numpy.ndarray
        :param cases:       cases of interest (default: None)
        :type cases:        str, list
        :param plateIDs:    plateIDs of interest (default: None)
        :type plateIDs:     int, float, list, numpy.ndarray
        """
        # Calculate slab pull torque
        self.calculate_slab_pull_torque(ages, cases, plateIDs)

        # Calculate GPE torque
        self.calculate_gpe_torque(ages, cases, plateIDs)

        # Calculate mantle drag torque
        self.calculate_mantle_drag_torque(ages, cases, plateIDs)

        # Calculate slab bend torque
        self.calculate_slab_bend_torque(ages, cases, plateIDs)

        # Calculate synthetic velocity
        self.calculate_synthetic_velocity(ages, cases, plateIDs)

        # Calculate driving torque
        self.calculate_driving_torque(ages, cases, plateIDs)

        # Calculate residual torque
        self.calculate_residual_torque(ages, cases, plateIDs)

        logging.info("Calculated all torques!")

    def calculate_gpe_torque(
            self,
            ages = None,
            cases = None,
            plateIDs = None,
        ):
        """
        Function to calculate the GPE torque.

        :param ages:        ages of interest (default: None)
        :type ages:         float, int, list, numpy.ndarray
        :param cases:       cases of interest (default: None)
        :type cases:        str, list
        :param plateIDs:    plateIDs of interest (default: None)
        :type plateIDs:     int, float, list, numpy.ndarray
        """
        # Calculate GPE force at points
        self.points.calculate_gpe_force(
            ages,
            cases,
            plateIDs,
            self.grids.seafloor_age,
        )

        # Calculate GPE torque acting on plate
        self.plates.calculate_torque_on_plates(
            self.points.data,
            ages,
            cases,
            plateIDs,
            torque_var = "GPE",
        )

        logging.info("Calculated GPE torque!")

    def calculate_slab_pull_torque(
            self,
            ages = None,
            cases = None,
            plateIDs = None,
        ):
        """
        Function to calculate the slab pull torque.

        :param ages:        ages of interest (default: None)
        :type ages:         float, int, list, numpy.ndarray
        :param cases:       cases of interest (default: None)
        :type cases:        str, list
        :param plateIDs:    plateIDs of interest (default: None)
        :type plateIDs:     int, float, list, numpy.ndarray
        """
        # Calculate slab pull force along the trenches
        self.slabs.calculate_slab_pull_force(
            ages,
            cases,
            plateIDs,
        )

        # Calculate slab pull force on plates
        self.plates.calculate_torque_on_plates(
            self.slabs.data,
            ages,
            cases,
            plateIDs,
            torque_var = "slab_pull",
        )

        logging.info("Calculated slab pull torque!")

    def calculate_mantle_drag_torque(
            self,
            ages = None,
            cases = None,
            plateIDs = None,
        ):
        """
        Function to calculate the mantle drag torque.

        :param ages:        ages of interest (default: None)
        :type ages:         float, int, list, numpy.ndarray
        :param cases:       cases of interest (default: None)
        :type cases:        str, list
        :param plateIDs:    plateIDs of interest (default: None)
        :type plateIDs:     int, float, list, numpy.ndarray
        """
        # Calculate mantle drag force
        self.points.calculate_mantle_drag_force(
            ages,
            cases,
            plateIDs,
        )

        # Calculate mantle drag force on plates
        self.plates.calculate_torque_on_plates(
            self.points.data,
            ages,
            cases,
            plateIDs,
            torque_var = "mantle_drag",
        )

        logging.info("Calculated mantle drag torque!")

    def calculate_slab_bend_torque(
            self,
            ages = None,
            cases = None,
            plateIDs = None,
        ):
        """
        Function to calculate the slab bend torque.

        :param ages:        ages of interest (default: None)
        :type ages:         float, int, list, numpy.ndarray
        :param cases:       cases of interest (default: None)
        :type cases:        str, list
        :param plateIDs:    plateIDs of interest (default: None)
        :type plateIDs:     int, float, list, numpy.ndarray
        """
        # Calculate the slab bend force along the trenches
        self.slabs.calculate_slab_bend_force(ages, cases, plateIDs)

        # Calculate the torque on 
        self.plates.calculate_torque_on_plates(
            self.slabs.data,
            ages,
            cases,
            plateIDs,
            torque_var = "slab_bend",
        )

        logging.info("Calculated slab bend torque!")

    def calculate_driving_torque(
            self,
            ages = None,
            cases = None,
            plateIDs = None,
        ):
        """
        Function to calculate the driving torque.

        :param ages:        ages of interest (default: None)
        :type ages:         float, int, list, numpy.ndarray
        :param cases:       cases of interest (default: None)
        :type cases:        str, list
        :param plateIDs:    plateIDs of interest (default: None)
        :type plateIDs:     int, float, list, numpy.ndarray
        """
        # Calculate driving torque
        self.plates.calculate_driving_torque(
            ages,
            cases,
            plateIDs,
        )

    def calculate_residual_torque(
            self,
            ages = None,
            cases = None,
            plateIDs = None,
        ):
        """
        Function to calculate the driving torque.

        :param ages:        ages of interest (default: None)
        :type ages:         float, int, list, numpy.ndarray
        :param cases:       cases of interest (default: None)
        :type cases:        str, list
        :param plateIDs:    plateIDs of interest (default: None)
        :type plateIDs:     int, float, list, numpy.ndarray
        """
        # Calculate residual torque
        self.plates.calculate_residual_torque(
            ages,
            cases,
            plateIDs,
        )

        # Calculate residual torque at slabs
        self.calculate_residual_force(
            ages,
            cases,            
            plateIDs,
            type = "slabs",
        )

        # Calculate residual torque at points
        self.calculate_residual_force(
            ages,
            cases,
            plateIDs,
            type = "points",
        )

    def calculate_residual_force(
            self,
            ages = None,
            cases = None,
            plateIDs = None,
            type = "slabs",
        ):
        """
        Function to calculate the residual forces.

        :param ages:        ages of interest (default: None)
        :type ages:         float, int, list, numpy.ndarray
        :param cases:       cases of interest (default: None)
        :type cases:        str, list
        :param plateIDs:    plateIDs of interest (default: None)
        :type plateIDs:     int, float, list, numpy.ndarray
        """
        if type == "slabs":
            # Calculate residual forces at slabs
            self.slabs.calculate_residual_force(
                ages,
                cases,
                plateIDs,
                self.plates.data,
            )

        elif type == "points":
            # Calculate residual forces at points
            self.points.calculate_residual_force(
                ages,
                cases,
                plateIDs,
                self.plates.data,
            )

        else:
            logging.error("Invalid type provided! Please choose from 'slabs' or 'points'.")

    def calculate_synthetic_velocity(
            self,
            ages = None,
            cases = None,
            plateIDs = None,
        ):
        """
        Function to compute synthetic velocities.

        :param ages:        ages of interest (default: None)
        :type ages:         float, int, list, numpy.ndarray
        :param cases:       cases of interest (default: None)
        :type cases:        str, list
        :param plateIDs:    plateIDs of interest (default: None)
        :type plateIDs:     int, float, list, numpy.ndarray
        """
        # Define ages if not provided
        _ages = utils_data.select_ages(ages, self.settings.ages)

        # Define cases if not provided
        _cases = utils_data.select_cases(cases, self.settings.cases)

        # Calculate synthetic velocities using driving torques
        self.plates.calculate_synthetic_velocity(
            _ages,
            _cases,
            plateIDs,
        )

        # Calculate net rotation
        self.calculate_net_rotation(_ages, _cases)

        # Calculate velocities at points
        self.points.calculate_velocities(
            _ages,
            _cases,
            self.plates.data,
        )

        # Calculate velocities at slabs
        self.slabs.calculate_velocities(
            _ages,
            _cases,
            self.plates.data,
        )

        # Calculate RMS velocity of plates
        self.plates.calculate_rms_velocity(
            self.points,
            _ages,
            _cases,
            plateIDs,
        )

    def extract_data_through_time(
            self,
            ages = None,
            cases = None,
            plateIDs = None,
            type = "plates",
            var = "residual_torque_mag",
        ):
        """
        Function to extract data through time.

        :param ages:        ages of interest (default: None)
        :type ages:         float, int, list, numpy.ndarray
        :param cases:       cases of interest (default: None)
        :type cases:        str, list
        :param plateIDs:    plateIDs of interest (default: None)
        :type plateIDs:     int, float, list, numpy.ndarray
        """
        if type == "plates":
            self.plates.extract_data_through_time(ages, cases, plateIDs, var)

        elif type == "points":  
            self.points.extract_data_through_time(ages, cases, plateIDs, var)

        elif type == "slabs":
            self.slabs.extract_data_through_time(ages, cases, plateIDs, var)

        else:
            logging.error("Invalid type provided! Please choose from 'plates', 'points', or 'slabs'.")

    def save_all(
            self,
            ages = None,
            cases = None,
            plateIDs = None,
            file_dir = None,
        ):
        """
        Function to save all classes within the PlateTorques object.

        :param ages:        ages of interest (default: None)
        :type ages:         float, int, list, numpy.ndarray
        :param cases:       cases of interest (default: None)
        :type cases:        str, list
        :param plateIDs:    plateIDs of interest (default: None)
        :type plateIDs:     int, float, list, numpy.ndarray
        :param file_dir:    directory to store files (default: None)
        :type file_dir:     str
        """
        # Save plates
        self.save_plates(ages, cases, plateIDs, file_dir)

        # Save points
        self.save_points(ages, cases, plateIDs, file_dir)

        # Save slabs
        self.save_slabs(ages, cases, plateIDs, file_dir)

        # Save grids
        self.save_grids(ages, cases, file_dir)

        # Save globe
        self.save_globe(cases, file_dir)

    def save_plates(
            self,
            ages = None,
            cases = None,
            plateIDs = None,
            file_dir = None,
        ):
        """
        Function to save plates.

        :param ages:        ages of interest (default: None)
        :type ages:         float, int, list, numpy.ndarray
        :param cases:       cases of interest (default: None)
        :type cases:        str, list
        :param plateIDs:    plateIDs of interest (default: None)
        :type plateIDs:     int, float, list, numpy.ndarray
        :param file_dir:    directory to store files (default: None)
        :type file_dir:     str
        """
        # Save plates
        self.plates.save(ages, cases, plateIDs, file_dir)

    def save_points(
            self,
            ages = None,
            cases = None,
            plateIDs = None,
            file_dir = None,
        ):
        """
        Function to save points.

        :param ages:        ages of interest (default: None)
        :type ages:         float, int, list, numpy.ndarray
        :param cases:       cases of interest (default: None)
        :type cases:        str, list
        :param plateIDs:    plateIDs of interest (default: None)
        :type plateIDs:     int, float, list, numpy.ndarray
        :param file_dir:    directory to store files (default: None)
        :type file_dir:     str
        """
        # Save points
        self.points.save(ages, cases, plateIDs, file_dir)

    def save_slabs(
            self,
            ages = None,
            cases = None,
            plateIDs = None,
            file_dir = None,
        ):
        """
        Function to save slabs.

        :param ages:        ages of interest (default: None)
        :type ages:         float, int, list, numpy.ndarray
        :param cases:       cases of interest (default: None)
        :type cases:        str, list
        :param plateIDs:    plateIDs of interest (default: None)
        :type plateIDs:     int, float, list, numpy.ndarray
        :param file_dir:    directory to store files (default: None)
        :type file_dir:     str
        """
        # Save slabs
        self.slabs.save(ages, cases, plateIDs, file_dir)

    def save_grids(
            self,
            ages = None,
            cases = None,
            file_dir = None,
        ):
        """
        Function to save grids.

        :param ages:        ages of interest (default: None)
        :type ages:         float, int, list, numpy.ndarray
        :param cases:       cases of interest (default: None)
        :type cases:        str, list
        :param plateIDs:    plateIDs of interest (default: None)
        :type plateIDs:     int, float, list, numpy.ndarray
        :param file_dir:    directory to store files (default: None)
        :type file_dir:     str
        """
        # Save grids
        self.grids.save_all(ages, cases, file_dir)

    def save_globe(
            self,
            cases = None,
            file_dir = None,
        ):
        """
        Function to save globe.

        :param ages:        ages of interest (default: None)
        :type ages:         float, int, list, numpy.ndarray
        :param cases:       cases of interest (default: None)
        :type cases:        str, list
        :param plateIDs:    plateIDs of interest (default: None)
        :type plateIDs:     int, float, list, numpy.ndarray
        :param file_dir:    directory to store files (default: None)
        :type file_dir:     str
        """
        # Save globe
        self.globe.save(cases, file_dir)

    def export_all(
            self,
            ages = None,
            cases = None,
            plateIDs = None,
            file_dir = None,
        ):
        """
        Function to export all classes within the PlateTorques object.

        :param ages:        ages of interest (default: None)
        :type ages:         float, int, list, numpy.ndarray
        :param cases:       cases of interest (default: None)
        :type cases:        str, list
        :param plateIDs:    plateIDs of interest (default: None)
        :type plateIDs:     int, float, list, numpy.ndarray
        :param file_dir:    directory to store files (default: None)
        :type file_dir:     str
        """
        # Save plates
        self.export_plates(ages, cases, plateIDs, file_dir)

        # Save points
        self.export_points(ages, cases, plateIDs, file_dir)

        # Save slabs
        self.export_slabs(ages, cases, plateIDs, file_dir)

        # Save grids
        self.save_grids(ages, cases, file_dir)

        # Save globe
        self.export_globe(cases, file_dir)

    def export_plates(
            self,
            ages = None,
            cases = None,
            plateIDs = None,
            file_dir = None,
        ):
        """
        Function to export plates.

        :param ages:        ages of interest (default: None)
        :type ages:         float, int, list, numpy.ndarray
        :param cases:       cases of interest (default: None)
        :type cases:        str, list
        :param plateIDs:    plateIDs of interest (default: None)
        :type plateIDs:     int, float, list, numpy.ndarray
        :param file_dir:    directory to store files (default: None)
        :type file_dir:     str
        """
        # Save plates
        self.plates.export(ages, cases, plateIDs, file_dir)

    def export_points(
            self,
            ages = None,
            cases = None,
            plateIDs = None,
            file_dir = None,
        ):
        """
        Function to export points.

        :param ages:        ages of interest (default: None)
        :type ages:         float, int, list, numpy.ndarray
        :param cases:       cases of interest (default: None)
        :type cases:        str, list
        :param plateIDs:    plateIDs of interest (default: None)
        :type plateIDs:     int, float, list, numpy.ndarray
        :param file_dir:    directory to store files (default: None)
        :type file_dir:     str
        """
        # Save points
        self.points.export(ages, cases, plateIDs, file_dir)

    def export_slabs(
            self,
            ages = None,
            cases = None,
            plateIDs = None,
            file_dir = None,
        ):
        """
        Function to export slabs.

        :param ages:        ages of interest (default: None)
        :type ages:         float, int, list, numpy.ndarray
        :param cases:       cases of interest (default: None)
        :type cases:        str, list
        :param plateIDs:    plateIDs of interest (default: None)
        :type plateIDs:     int, float, list, numpy.ndarray
        :param file_dir:    directory to store files (default: None)
        :type file_dir:     str
        """
        # Save slabs
        self.slabs.export(ages, cases, plateIDs, file_dir)

    def export_grids(
            self,
            ages = None,
            cases = None,
            file_dir = None,
        ):
        """
        Function to export grids.

        :param ages:        ages of interest (default: None)
        :type ages:         float, int, list, numpy.ndarray
        :param cases:       cases of interest (default: None)
        :type cases:        str, list
        :param file_dir:    directory to store files (default: None)
        :type file_dir:     str
        """
        # Save grids
        self.grids.export(ages, cases, file_dir)

    def export_globe(
            self,
            cases = None,
            file_dir = None,
        ):
        """
        Function to expprt globe.

        :param cases:       cases of interest (default: None)
        :type cases:        str, list
        :param file_dir:    directory to store files (default: None)
        :type file_dir:     str
        """
        # Save globe
        self.globe.export(cases, file_dir)