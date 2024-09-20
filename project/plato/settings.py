import os

import numpy as _numpy

import setup

class Settings():
    def __init__(
            self,
            name,
            ages,
            cases_file,
            cases_sheet="Sheet1",
            files_dir=None,
            PARALLEL_MODE = False,
            DEBUG_MODE = False,
        ):
        """
        Object to store the settings of the plato simulation
        """
        # Store reconstruction name and valid reconstruction times
        self.name = name
        self.ages = _numpy.array(ages)

        # Store cases and case options
        self.cases, self.options = setup.get_options(cases_file, cases_sheet)
        
        # Set files directory
        self.dir_path = os.path.join(os.getcwd(), files_dir)
        if not os.path.exists(self.dir_path):
            os.makedirs(self.dir_path)

        # Set flag for debugging mode
        self.DEBUG_MODE = DEBUG_MODE

        # Set flag for parallel mode
        # TODO: Actually implement parallel mode
        self.PARALLEL_MODE = PARALLEL_MODE

        # Subdivide cases to accelerate computation
        # Group cases for initialisation of plates, slabs, and points
        plate_options = ["Minimum plate area"]
        self.plate_cases = setup.process_cases(self.cases, self.options, plate_options)
        slab_options = ["Slab tesselation spacing"]
        self.slab_cases = setup.process_cases(self.cases, self.options, slab_options)
        point_options = ["Grid spacing"]
        self.point_cases = setup.process_cases(self.cases, self.options, point_options)

        # Group cases for torque computation
        slab_pull_options = [
            "Slab pull torque",
            "Seafloor age profile",
            "Sample sediment grid",
            "Active margin sediments",
            "Sediment subduction",
            "Sample erosion grid",
            "Slab pull constant",
            "Shear zone width",
            "Slab length"
        ]
        self.slab_pull_cases = setup.process_cases(self.cases, self.options, slab_pull_options)

        slab_bend_options = ["Slab bend torque", "Seafloor age profile"]
        self.slab_bend_cases = setup.process_cases(self.cases, self.options, slab_bend_options)

        gpe_options = ["Continental crust", "Seafloor age profile", "Grid spacing"]
        self.gpe_cases = setup.process_cases(self.cases, self.options, gpe_options)

        mantle_drag_options = ["Reconstructed motions", "Grid spacing"]
        self.mantle_drag_cases = setup.process_cases(self.cases, self.options, mantle_drag_options)
