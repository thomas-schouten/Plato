from typing import Optional

import setup

class Grids():
    def __init__(
            self,
            _settings,
            seafloor_grids: Optional[dict] = None,
            continental_grids: Optional[dict] = None,
            velocity_grids: Optional[dict] = None
        ):
        """
        Object to hold gridded data.
        Seafloor grids contain lithospheric age and, optionally, sediment thickness.
        Continental grids contain lithospheric thickness and, optionally, crustal thickness.
        Velocity grids contain plate velocity data.
        """
        # Store the settings
        self.settings = _settings

        # Store seafloor grid
        if seafloor_grids:
            self.seafloor = seafloor_grids
        else:
            self.seafloor = {}

            # Load or initialise seafloor
            self.seafloor = setup.load_grid(
                self.seafloor,
                self.name,
                self.times,
                "Seafloor",
                self.settings.dir_path,
                DEBUG_MODE = self.DEBUG_MODE
            )

        # Store continental grid. This is not central to the object, so it is not loaded by default.
        if continental_grids:
            self.continental = continental_grids
        else:
            pass

        # Store velocity grid. This is not central to the object, so it is not loaded by default.
        if velocity_grids:
            self.velocity = velocity_grids
        else:
            pass
            
