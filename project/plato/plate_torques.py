# Standard libraries
import os
import logging
import sys
from typing import Dict, List, Optional, Union

# Third-party libraries
import gplately as _gplately
import numpy as _numpy
import pandas as _pandas

# Local libraries
import utils_data, utils_init
from globe import Globe
from grids import Grids
from plates import Plates
from points import Points
from slabs import Slabs
from settings import Settings

class PlateTorques():
    """
    PlateTorques class.
    """
    def __init__(
            self,
            settings: Optional[Settings] = None,
            reconstruction_name: str = None, 
            ages: Union[List[int], _numpy.array] = None, 
            cases_file: str = None, 
            cases_sheet: Optional[str] = "Sheet1", 
            files_dir: Optional[str] = None,
            reconstruction: Optional[_gplately.PlateReconstruction] = None,
            rotation_file: Optional[List[str]] = None,
            topology_file: Optional[List[str]] = None,
            polygon_file: Optional[List[str]] = None,
            coastline_file: Optional[str] = None,
            seafloor_age_grids: Optional[Dict] = None,
            sediment_grids: Optional[Dict] = None,
            continental_grids: Optional[Dict] = None,
            velocity_grids: Optional[Dict] = None,
            plates: Optional[Plates] = None,
            points: Optional[Points] = None,
            slabs: Optional[Slabs] = None,
            grids: Optional[Grids] = None,
            globe: Optional[Globe] = None,
            DEBUG_MODE: Optional[bool] = False,
            PARALLEL_MODE: Optional[bool] = False,
        ):
        """
        Set up the PlateTorques class.
        """
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

        logging.info("PlateTorques object successfully instantiated!")

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
        """
        # Calculate net rotation
        self.globe.calculate_net_rotation(
            self.plates,
            self.points,
            ages,
            cases,
            plateIDs,
        )

    def sample_seafloor_ages(
            self,
            ages: Optional[Union[int, float, _numpy.integer, _numpy.floating, List, _numpy.ndarray]] = None,
            cases: Optional[Union[str, List[str]]] = None,
            plateIDs: Optional[Union[int, float, _numpy.integer, _numpy.floating, List, _numpy.ndarray]] = None,
        ):
        """
        Function to sample the seafloor ages and other variables (if available)
        """
        # Sample points
        self.sample_point_seafloor_ages(ages, cases, plateIDs)
    
        # Sample slabs
        self.sample_slab_seafloor_ages(ages, cases, plateIDs)

    def sample_point_seafloor_ages(
            self,
            ages: Optional[Union[int, float, _numpy.integer, _numpy.floating, List, _numpy.ndarray]] = None,
            cases: Optional[Union[str, List[str]]] = None,
            plateIDs: Optional[Union[int, float, _numpy.integer, _numpy.floating, List, _numpy.ndarray]] = None,
        ):
        """
        Function to sample the seafloor ages and other variables (if available)
        """
        self.points.sample_seafloor_ages(ages, cases, plateIDs, self.grids.seafloor_age)

    def sample_slab_seafloor_ages(
            self,
            ages: Optional[Union[int, float, _numpy.integer, _numpy.floating, List, _numpy.ndarray]] = None,
            cases: Optional[Union[str, List[str]]] = None,
            plateIDs: Optional[Union[int, float, _numpy.integer, _numpy.floating, List, _numpy.ndarray]] = None,
        ):
        """
        Function to sample the seafloor ages and other variables (if available)
        """
        self.slabs.sample_slab_seafloor_ages(ages, cases, plateIDs, self.grids.seafloor_age)

    def sample_slab_sediment_thickness(
            self,
            ages: Optional[Union[int, float, _numpy.integer, _numpy.floating, List, _numpy.ndarray]] = None,
            cases: Optional[Union[str, List[str]]] = None,
            plateIDs: Optional[Union[int, float, _numpy.integer, _numpy.floating, List, _numpy.ndarray]] = None,
        ):
        """
        Function to sample the seafloor ages and other variables (if available)
        """
        self.slabs.sample_slab_sediment_thickness(ages, cases, plateIDs, self.grids.sediment)

    # def sample_seafloor_age_at_upper_plates(
    #         self,
    #         ages: Optional[Union[int, float, _numpy.integer, _numpy.floating, List, _numpy.ndarray]],
    #         cases: Optional[Union[str, List]],
    #         plateIDs: Optional[Union[int, float, _numpy.integer, _numpy.floating, List, _numpy.ndarray]],
    #     ):
    #     """
    #     Function to sample the seafloor ages and other variables (if available)
    #     """
    #     self.slabs.sample_upper_plates(ages, cases, plateIDs, self.grids.seafloor_age)

    def calculate_all_torques(
            self,
            ages: Optional[Union[int, float, _numpy.integer, _numpy.floating, List, _numpy.ndarray]],
            cases: Optional[Union[str, List]],
            plateIDs: Optional[Union[int, float, _numpy.integer, _numpy.floating, List, _numpy.ndarray]],
        ):
        """
        Function to calculate all torques
        """
        # Calculate slab pull torque
        self.calculate_slab_pull_torque(ages, cases, plateIDs)

        # Calculate GPE torque
        self.calculate_gpe_torque(ages, cases, plateIDs)

        # Calculate mantle drag torque
        self.calculate_mantle_drag_torque(ages, cases, plateIDs)

        # Calculate slab bend torque
        self.calculate_slab_bend_torque(ages, cases, plateIDs)

        # Calculate driving torque
        self.calculate_driving_torque(ages, cases, plateIDs)

        # Calculate residual torque
        self.calculate_residual_torque(ages, cases, plateIDs)

        logging.info("Calculated all torques!")

    def calculate_gpe_torque(
            self,
            ages: Optional[Union[int, float, _numpy.integer, _numpy.floating, List, _numpy.ndarray]] = None,
            cases: Optional[Union[str, List]] = None,
            plateIDs: Optional[Union[int, float, _numpy.integer, _numpy.floating, List, _numpy.ndarray]] = None,
        ):
        """
        Function to calculate the GPE torque
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
            ages: Optional[Union[int, float, _numpy.integer, _numpy.floating, List, _numpy.ndarray]] = None,
            cases: Optional[Union[str, List]] = None,
            plateIDs: Optional[Union[int, float, _numpy.integer, _numpy.floating, List, _numpy.ndarray]] = None,
        ):
        """
        Function to calculate the slab pull torque
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
            ages: Optional[Union[int, float, _numpy.integer, _numpy.floating, List, _numpy.ndarray]] = None,
            cases: Optional[Union[str, List]] = None,
            plateIDs: Optional[Union[int, float, _numpy.integer, _numpy.floating, List, _numpy.ndarray]] = None,
        ):
        """
        Function to calculate the mantle drag torque
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
            ages: Optional[Union[int, float, _numpy.integer, _numpy.floating, List, _numpy.ndarray]] = None,
            cases: Optional[Union[str, List]] = None,
            plateIDs: Optional[Union[int, float, _numpy.integer, _numpy.floating, List, _numpy.ndarray]] = None,
        ):
        """
        Function to calculate the slab bend torque
        """
        # Calculate the slab bend force along the trenches
        self.slabs.calculate_slab_bend_force(ages, cases, plateIDs)

        # Calculate the torque on 
        self.plates.calculate_torque_on_plates(
            ages,
            cases,
            plateIDs,
            self.plates.data,
            self.slabs.data,
            torque_var = "slab_bend",
        )

        logging.info("Calculated slab bend torque!")

    def calculate_driving_torque(
            self,
            ages: Optional[Union[int, float, _numpy.integer, _numpy.floating, List, _numpy.ndarray]] = None,
            cases: Optional[Union[str, List]] = None,
            plateIDs: Optional[Union[int, float, _numpy.integer, _numpy.floating, List, _numpy.ndarray]] = None,
        ):
        """
        Function to calculate the driving torque.
        """
        # Calculate driving torque
        self.plates.calculate_driving_torque(
            ages,
            cases,
            plateIDs,
        )

    def calculate_residual_torque(
            self,
            ages: Optional[Union[int, float, _numpy.integer, _numpy.floating, List, _numpy.ndarray]] = None,
            cases: Optional[Union[str, List]] = None,
            plateIDs: Optional[Union[int, float, _numpy.integer, _numpy.floating, List, _numpy.ndarray]] = None,
        ):
        """
        Function to calculate the driving torque.
        """
        # Calculate residual torque
        self.plates.calculate_residual_torque(
            ages,
            cases,
            plateIDs,
        )

    def calculate_synthetic_velocity(
            self,
            ages: Optional[Union[int, float, _numpy.integer, _numpy.floating, List, _numpy.ndarray]] = None,
            cases: Optional[Union[str, List]] = None,
            plateIDs: Optional[Union[int, float, _numpy.integer, _numpy.floating, List, _numpy.ndarray]] = None,
        ):
        """
        Function to compute synthetic velocities.
        """
        # Define ages if not provided
        _ages = utils_data.select_ages(ages, self.settings.ages)

        # Define cases if not provided
        _cases = utils_data.select_cases(cases, self.settings.cases)

        # Get driving torque
        self.calculate_driving_torque(_ages, _cases, plateIDs)

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

    def save_all(
            self,
            ages=None,
            cases=None,
            plateIDs=None,
            file_dir=None,
        ):
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
            ages,
            cases,
            plateIDs,
            file_dir,
        ):
        """
        Function to save plates.
        """
        # Save plates
        self.plates.save(ages, cases, plateIDs, file_dir)

    def save_points(
            self,
            ages,
            cases,
            plateIDs,
            file_dir,
        ):
        """
        Function to save points.
        """
        # Save points
        self.points.save(ages, cases, plateIDs, file_dir)

    def save_slabs(
            self,
            ages,
            cases,
            plateIDs,
            file_dir,
        ):
        """
        Function to save slabs.
        """
        # Save slabs
        self.slabs.save(ages, cases, plateIDs, file_dir)

    def save_grids(
            self,
            ages,
            cases,
            file_dir,
        ):
        """
        Function to save grids.
        """
        # Save grids
        self.grids.save_all(ages, cases, file_dir)

    def save_globe(
            self,
            cases,
            file_dir,
        ):
        """
        Function to save globe.
        """
        # Save globe
        self.globe.save(cases, file_dir)

    def export_all(
            self,
            ages=None,
            cases=None,
            plateIDs=None,
            file_dir=None,
        ):
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
            ages,
            cases,
            plateIDs,
            file_dir,
        ):
        """
        Function to save plates.
        """
        # Save plates
        self.plates.export(ages, cases, plateIDs, file_dir)

    def export_points(
            self,
            ages,
            cases,
            plateIDs,
            file_dir,
        ):
        """
        Function to save points.
        """
        # Save points
        self.points.export(ages, cases, plateIDs, file_dir)

    def export_slabs(
            self,
            ages,
            cases,
            plateIDs,
            file_dir,
        ):
        """
        Function to export slabs.
        """
        # Save slabs
        self.slabs.export(ages, cases, plateIDs, file_dir)

    def export_grids(
            self,
            ages,
            cases,
            file_dir,
        ):
        """
        Function to export grids.
        """
        # Save grids
        self.grids.export(ages, cases, file_dir)

    def export_globe(
            self,
            cases,
            file_dir,
        ):
        """
        Function to expprt globe.
        """
        # Save globe
        self.globe.export(cases, file_dir)