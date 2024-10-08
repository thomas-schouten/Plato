import logging
from typing import Dict, List, Optional, Union

import gplately as _gplately
import numpy as _numpy
import pandas as _pandas
from tqdm import tqdm as _tqdm

import utils_data, utils_calc, utils_init
from settings import Settings

class Globe:
    """
    Class to store information on global plate tectonic properties of the Earth.
    """
    def __init__(
            self,
            settings: Optional['Settings'] = None,
            cases_file: Optional[str] = None,
            cases_sheet: Optional[str] = None,
            reconstruction: Optional[_gplately.PlateReconstruction] = None,
            reconstruction_name: Optional[str] = None,
            rotation_file: Optional[str] = None,
            topology_file: Optional[str] = None,
            polygon_file: Optional[str] = None,
            ages: Optional[list] = None,
            files_dir: Optional[str] = None,
            plate_data: Optional[Dict] = None,
            point_data: Optional[Dict] = None,
            slab_data: Optional[Dict] = None,
            DEBUG_MODE: Optional[bool] = False,
            PARALLEL_MODE: Optional[bool] = False,
        ):
        """
        Initialize the Globe class with the required objects.

        :param reconstruction: Reconstruction object (default: None)
        :type reconstruction: Optional[Reconstruction]
        :param settings: Settings object (default: None)
        :type settings: Optional[Settings]
        :param plates: Plates object (default: None)
        :type plates: Optional[Plates]
        :param slabs: Slabs object (default: None)
        :type slabs: Optional[Slabs]
        :param points: Points object (default: None)
        :type points: Optional[Points]
        """
        # Store settings object
        self.settings = utils_init.get_settings(
            settings, 
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

        # Store constants
        self.constants = utils_calc.set_constants()

        # Store mechanical parameters
        self.mech = utils_calc.set_mech_params()

        # Initialise dataframe to store global properties for each case
        self.data = {case: _pandas.DataFrame(
            {'age': self.settings.ages,
            "number_of_plates": 0,
            "world_uncertainty": 0.,
            "subduction_length": 0.,
            "net_rotation_pole_lat": 0.,
            "net_rotation_pole_lon": 0.,
            "net_rotation_rate": 0.,}
        ) for case in self.settings.cases}

        # Get the number of plates
        self.calculate_number_of_plates(plate_data)

        # Get the subduction length
        self.calculate_subduction_length(slab_data)

        # Get the net rotation
        self.calculate_net_rotation(plate_data)

    def calculate_number_of_plates(
            self,
            ages = None,
            plates_data = None
        ):
        """
        Calculate the number of plates for each time step.

        :param ages: List of ages for which to calculate the number of plates (default: None)
        :type ages: Optional[int, float, numpy.integer, numpy.floating, list, numpy.ndarray]
        :param plates_data: Optional plate data for each case and age (default: None)
        :type plates_data: Optional[dict]
        """
        # Define ages if not provided
        _ages = utils_data.get_ages(ages, self.settings.ages)
        
        # Calculate the number of plates for each time step
        for i, _age in enumerate(_ages):
            for _case in self.settings.cases:
                if plates_data and if _age in plates_data.keys() and if _case in plates_data[_age].keys():
                    logging.info(f"Calculating number of plates for case {_case} at age {_age} using provided data")
                    self.data[_case].loc[i, "number_of_plates"] = len(plates_data[_case][_age].plate_id.unique())
                else:
                    logging.info(f"Calculating number of plates for case {_case} at age {_age} using resolved topologies")
                    resolved_topologies = utils_data.get_resolved_topologies(
                        self.reconstruction,
                        [_age],
                    )
                    self.data[_case].loc[i, "number_of_plates"] = len(resolved_topologies[_age])

    def calculate_subduction_length(
            self,
            ages: Optional[Union[int, float, _numpy.integer, _numpy.floating, list, _numpy.ndarray]] = None,
            cases: Optional[List[str]] = None,
            slab_data: Optional[Dict] = None
        ):
        """
        Calculate the subduction length for each time step.

        :param ages: List of ages for which to calculate subduction length (default: None)
        :type ages: Optional[int, float, numpy.integer, numpy.floating, list, numpy.ndarray]
        :param slabs_data: Optional slab data for each age and case (default: None)
        :type slabs_data: Optional[dict]
        """
        # Define ages if not provided
        _ages = utils_data.get_ages(ages, self.settings.ages)

        # Define cases if not provided
        _cases = utils_data.get_cases(cases, self.settings.cases)

        # Calculate the subduction length for each time step
        for i, _age in enumerate(_ages):
            for _case in _cases:
                # Check if slab data is provided
                if slab_data and _age in slab_data.keys() and _case in slab_data[_age].keys():
                    logging.info(f"Calculating subduction length for case {_case} at age {_age} using provided data")
                    self.data[_case].loc[i, "subduction_length"] = slab_data[_case][_age].trench_segment_length.sum()
                else:
                    logging.info(f"Calculating subduction length for case {_case} at age {_age} by tesselating subduction zones")
                    slabs = self.reconstruction.tessellate_subduction_zones(
                        _age,
                        ignore_warnings=True,
                        tessellation_threshold_radians=(
                            self.settings.options[_case]["Slab tesselation spacing"]/self.constants.mean_Earth_radius_km
                        )
                    )
                    # Convert to _pandas.DataFrame
                    slabs = _pandas.DataFrame(slabs)

                    # Kick unused columns and rename the rest
                    slabs = slabs.drop(columns=[2, 3, 4, 5])
                    slabs.columns = ["lon", "lat", "trench_segment_length", "trench_normal_azimuth", "lower_plateID", "trench_plateID"]

                    # Convert trench segment length from degree to m
                    slabs.trench_segment_length *= self.constants.equatorial_Earth_circumference / 360

                    self.data[_case].loc[i, "subduction_length"] = slabs.trench_segment_length.sum()

    def calculate_net_rotation(
            self,
            ages = None,
            cases = None,
            plate_data = None
        ):
        """
        Calculate the net rotation of the Earth's lithosphere.

        :param ages: List of ages for which to calculate net rotation (default: None)
        :type ages: Optional[int, float, numpy.integer, numpy.floating, list, numpy.ndarray]
        :param plates_data: Optional plate data for each case and age (default: None)
        :type plates_data: Optional[dict]
        """
        # Define ages if not provided
        _ages = utils_data.get_ages(ages, self.settings.ages)

        # Define cases if not provided
        _cases = utils_data.get_cases(cases, self.settings.cases)
        
        # Calculate the net rotation of the Earth's lithosphere
        for i, _age in enumerate(_ages):
            for _case in _cases:
                if plate_data and _age in plate_data.keys() and _case in plate_data[_age].keys():
                    logging.info(f"Calculating net rotation for case {_case} at age {_age} using provided data")
                    pass
                else:
                    logging.info(f"Calculating net rotation for case {_case} at age {_age} by resolving plate velocities")
                    resolved_topologies = utils_data.get_resolved_topologies(
                        self.reconstruction,
                        [_age],
                    )
                    for _topology in resolved_topologies[_age]:
                        pass

    def calculate_world_uncertainty(
            self,
            ages = None,
            polygons = None,
            reconstructed_polygons = None
        ):
        """
        Calculate the fraction of the Earth's surface that has been lost to subduction.

        :param ages: List of ages for which to calculate world uncertainty (default: None)
        :type ages: Optional[int, float, numpy.integer, numpy.floating, list, numpy.ndarray]
        :param polygons: Static polygons object or path (default: None)
        :type polygons: Optional[Union[str, _pygplates.FeatureCollection]]
        :param reconstructed_polygons: Optional pre-reconstructed polygons (default: None)
        :type reconstructed_polygons: Optional[dict]
        """
        # Define ages if not provided
        if ages is None:
            ages = self.settings.ages
        elif isinstance(ages, (int, float, _numpy.integer, _numpy.floating)):
            ages = [ages]

        # Check that the polygons are provided
        if not reconstructed_polygons:
            if polygons:
                if isinstance(polygons, str):
                    self.polygons = _pygplates.FeatureCollection(polygons)
                elif isinstance(polygons, _pygplates.FeatureCollection):
                    self.polygons = polygons
                else:
                    if hasattr(self, 'polygons'):
                        polygons = self.polygons
                    else:
                        raise ValueError("No static polygons object provided")

        # Reconstruct the polygons for each time step if necessary
        if reconstructed_polygons is None:
            reconstructed_polygons = {}
            for age in self.settings.ages:
                reconstructed_polygons[age] = _pygplates.reconstruct(
                    self.polygons,
                    self.reconstruction.rotation_model
                )

        # Calculate the fraction of the Earth's surface that has been lost to subduction
        for i, age in enumerate(ages):
            if age in reconstructed_polygons:
                self.data[self.settings.cases[0]].loc[i, "world_uncertainty"] = 1 - reconstructed_polygons[age]
            else:
                self.data[self.settings.cases[0]].loc[i, "world_uncertainty"] = 0  # Handle case when age not found
