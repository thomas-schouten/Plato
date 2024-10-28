# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# PLATO
# Algorithm to calculate plate forces from tectonic reconstructions
# Thomas Schouten, 2024
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

import os
import logging
from typing import List, Optional

import numpy as _numpy

from . import utils_data, utils_calc

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# SETTINGS OBJECT
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

class Settings:
    def __init__(
        self,
        name: str,
        ages: List[float],
        cases_file: Optional[str] = None,
        cases_sheet: Optional[str] = "Sheet1",
        files_dir: Optional[str] = None,
        PARALLEL_MODE: bool = False,
        DEBUG_MODE: bool = False,
        ):
        """
        Object to store the settings of a plato simulation.

        :param name: Reconstruction name.
        :type name: str
        :param ages: List of valid reconstruction times.
        :type ages: List[float]
        :param cases_file: Path to the cases file.
        :type cases_file: str
        :param cases_sheet: Sheet name in the cases file (default: "Sheet1").
        :type cases_sheet: str
        :param files_dir: Directory path for output files (default: None, current working directory will be used).
        :type files_dir: Optional[str]
        :param PARALLEL_MODE: Flag to enable parallel computation mode (default: False).
        :type PARALLEL_MODE: bool
        :param DEBUG_MODE: Flag to enable debugging mode (default: False).
        :type DEBUG_MODE: bool

        :raises ValueError: If the ages list is empty.
        :raises FileNotFoundError: If the cases file is not found.
        :raises Exception: If an error occurs during cases loading.
        """
        # Set up logging configuration
        self.configure_logger(DEBUG_MODE)

        logging.info(f"Initialising settings for simulation: {name}")

        # Store reconstruction name and valid reconstruction times
        self.name = name
        self.ages = _numpy.array(ages)

        # Validate ages
        if not self.ages.size:
            logging.error("Ages list cannot be empty.")
            raise ValueError("Ages list cannot be empty.")
        logging.debug(f"Valid ages: {self.ages}")

        # Store cases and case options
        try:
            self.cases, self.options = utils_data.get_options(cases_file, cases_sheet)
            logging.info(f"Cases loaded successfully from {cases_file}, sheet: {cases_sheet}")
        except FileNotFoundError as e:
            logging.error(f"Cases file not found: {cases_file} - {e}")
            raise
        except Exception as e:
            logging.error(f"Error loading cases from file: {cases_file}, sheet: {cases_sheet} - {e}")
            raise

        # Set files directory
        self.dir_path = os.path.join(os.getcwd(), files_dir or "")
        try:
            os.makedirs(self.dir_path, exist_ok=True)
            logging.info(f"Output directory set to: {self.dir_path}")
        except OSError as e:
            logging.error(f"Error creating directory: {self.dir_path} - {e}")
            raise

        # Set debug and parallel modes
        self.DEBUG_MODE = DEBUG_MODE
        self.PARALLEL_MODE = PARALLEL_MODE
        logging.debug(f"DEBUG_MODE: {self.DEBUG_MODE}, PARALLEL_MODE: {self.PARALLEL_MODE}")

        # Process case groups
        self.plate_cases = self.process_cases(["Minimum plate area", "Anchor plateID"])
        self.slab_cases = self.process_cases(["Slab tesselation spacing"])
        self.point_cases = self.process_cases(["Grid spacing"])

        # Process torque computation groups
        self.slab_pull_cases = self.process_cases([
            "Slab pull torque", "Seafloor age profile", "Sample sediment grid", 
            "Active margin sediments", "Sediment subduction", "Sample erosion grid", 
            "Slab pull constant", "Shear zone width", "Slab length"
        ])
        self.slab_bend_cases = self.process_cases(["Slab bend torque", "Seafloor age profile"])
        self.gpe_cases = self.process_cases(["Continental crust", "Seafloor age profile", "Grid spacing"])
        self.mantle_drag_cases = self.process_cases(["Reconstructed motions", "Grid spacing"])

        # Store constants and mechanical parameters
        self.constants = utils_calc.set_constants()
        self.mech = utils_calc.set_mech_params()

        # Store plateIDs of oceanic arcs of the Earthbyte reconstructions^* that are masked (i.e. value is NaN) on the seafloor age grid
        # This is to make sure that when adding active margin sediments, these arcs are not included
        # NOTE: this is hardcoded for lack of a better alternative
        # ^* Earthbyte reconstructions: Seton et al. (2012), Müller et al. (2016, 2019), Matthews et al. (2016), Clennett et al. (2020)
        self.oceanic_arc_plateIDs = [
            529, # Kohistan-Ladakh
            608, 659, 699, # Izu-Bonin-Marianas
            612, # Luzon
            645, # East Sunda
            678, # East Philippine
            679, # Halmahera
            673, # Woyla
            688, # Proto-Caroline
            806, # Hikurangi
            821, # Tonga-Kermadec
            827, # New Hebrides
            847, # Vityaz
            853, # West Solomon Sea
            844, 841, 865, 943, # Junction
            1072, 1073, 1080, # Insular
            2007, # Antilles
            9052, 95104, # Central America
            9022, # Cascadia root
            9040, # Angayucham
        ]

        logging.info("Settings initialisation complete.")

    def process_cases(
            self,
            option_keys: List[str]
        ) -> List:
        """
        Process and return cases based on given option keys.

        :param option_keys: List of case option keys to group cases.
        :type option_keys: List[str]

        :return: Processed cases based on the provided option keys.
        :rtype: List

        :raises: Exception if case processing fails.
        """
        try:
            processed_cases = utils_data.process_cases(self.cases, self.options, option_keys)
            logging.debug(f"Processed cases for options: {option_keys}")
            return processed_cases
        except KeyError as e:
            logging.error(f"Option key not found: {e}")
            raise
        except Exception as e:
            logging.error(f"Error processing cases for options: {option_keys} - {e}")
            raise

    def configure_logger(
            self,
            DEBUG_MODE: bool = False
        ):
        """
        Configures the logger for a module.
        
        :param DEBUG_MODE: Whether to set the logging level to DEBUG.
        :type DEBUG_MODE: bool
        """
        # Get the logger
        self.logger = logging.getLogger("plato")

        # Set the logging level
        if DEBUG_MODE:
            self.logger.setLevel(logging.DEBUG)
        else:
            self.logger.setLevel(logging.INFO)

        # Add a console handler if no handlers exist
        if not self.logger.hasHandlers():
            handler = logging.StreamHandler()
            handler.setLevel(logging.DEBUG if DEBUG_MODE else logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def run_parallel_mode(self):
        """
        Placeholder method to implement parallel mode.
        Logs a warning if parallel mode is enabled but not yet implemented.
        """
        if self.PARALLEL_MODE:
            logging.warning("Parallel mode is enabled but not yet implemented.")
        else:
            logging.info("Parallel mode is disabled.")
