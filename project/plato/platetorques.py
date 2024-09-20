# Standard libraries
import os
import sys
from typing import List, Optional, Tuple, Dict

# Third-party libraries
import numpy as _numpy
import pandas as _pandas

# Local libraries
from grids import Grids
from plates import Plates
from points import Points
from reconstruction import Reconstruction
from settings import Settings
from slabs import Slabs


class PlateTorques():
    """
    PlateTorques class.
    """
    def __init__(
            self,
            reconstruction_name: str, 
            ages: List[int] or _numpy.array, 
            cases_file: str, 
            cases_sheet: Optional[str] = "Sheet1", 
            files_dir: Optional[str] = None,
            rotation_file: Optional[List[str]] = None,
            topology_file: Optional[List[str]] = None,
            polygon_file: Optional[List[str]] = None,
            coastline_file: Optional[str] = None,
            seafloor_grids: Optional[dict] = None,
            DEBUG_MODE: Optional[bool] = False,
            PARALLEL_MODE: Optional[bool] = False,
        ):
        """
        Set up the PlateTorques class.
        """
        # Initialise the settings
        self.settings = Settings(
            reconstruction_name=reconstruction_name,
            cases_file=cases_file,
            cases_sheet=cases_sheet,
            files_dir=files_dir,
            rotation_file=rotation_file,
            topology_file=topology_file,
            polygon_file=polygon_file,
            coastline_file=coastline_file,
            seafloor_grids=seafloor_grids,
            DEBUG_MODE=DEBUG_MODE,
            PARALLEL_MODE=PARALLEL_MODE,
        )

        # Initialise reconstruction
        self.reconstruction = Reconstruction(
            settings=self.settings,
        )

        # Initialise plates, slabs and points
        self.plates = Plates(
            settings=self.settings,
            reconstruction=self.reconstruction,
        )

        self.slabs = Slabs(
            settings=self.settings,
            reconstruction=self.reconstruction,
        )

        self.points = Points(
            settings=self.settings,
            reconstruction=self.reconstruction,
        )

        # Initialise grids
        self.grids = Grids(
            settings=self.settings,
            reconstruction=self.reconstruction,
        )