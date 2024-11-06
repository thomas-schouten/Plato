import logging
from typing import Dict, List, Optional, Union

import gplately as _gplately
from gplately import pygplates as _pygplates
import numpy as _numpy
import pandas as _pandas
from tqdm import tqdm as _tqdm

from . import utils_data, utils_calc, utils_init
from .settings import Settings
from .plates import Plates
from .points import Points
from .slabs import Slabs

class Globe:
    """
    Class to store information on global plate tectonic properties of the Earth.
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
            plates: Optional[Plates] = None,
            points: Optional[Points] = None,
            slabs: Optional[Slabs] = None,
            PARALLEL_MODE = False,
            DEBUG_MODE = False,
            CALCULATE_VELOCITIES = True,
        ):
        """
        Initialie the Globe class with the required objects.
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

        # Store constants
        self.constants = utils_calc.set_constants()

        # Store mechanical parameters
        self.mech = utils_calc.set_mech_params()

        # Store plates, points and slabs objects, if provided
        self.plates = plates
        self.points = points
        self.slabs = slabs

        # Initialise dataframe to store global properties for each case
        self.data = {case: _pandas.DataFrame(
            {'age': self.settings.ages,
            "number_of_plates": 0,
            "subduction_length": 0.,
            "world_uncertainty": 0.,
            "net_rotation_pole_lat": 0.,
            "net_rotation_pole_lon": 0.,
            "net_rotation_rate": 0.}
        ) for case in self.settings.cases}

        # Get the number of plates
        self.calculate_number_of_plates()

        # Get the subduction length
        self.calculate_subduction_length()

        # Get the world uncertainty
        self.calculate_world_uncertainty()

        # Get the net rotation
        if CALCULATE_VELOCITIES:
            self.calculate_net_rotation()
    
    def __str__(self):
        return f"Globe is a class that contains data and methods for to characterise the global geodynamic state of a reconstruction."
    
    def __repr__(self):
        return self.__str__()
    

    def calculate_number_of_plates(
            self,
            plates: Plates = None,
            ages = None,
            cases = None,
        ):
        """
        Calculate the number of plates for each time step.
        """
        # Define ages if not provided
        _ages = utils_data.select_ages(ages, self.settings.ages)
        
        # Define cases if not provided
        _cases = utils_data.select_cases(cases, self.settings.cases)

        # Calculate the number of plates for each time step
        for i, _age in enumerate(_ages):
            for _case in _cases:
                if isinstance(plates, Plates) and _age in plates.data.keys() and _case in plates.data[_age].keys():
                    logging.info(f"Calculating number of plates for case {_case} at age {_age} using provided data")
                    self.data[_case].loc[i, "number_of_plates"] = len(plates.data[_age][_case].plateID.unique())
                else:
                    logging.info(f"Calculating number of plates for case {_case} at age {_age} using resolved topologies")
                    resolved_topologies = utils_data.get_resolved_topologies(
                        self.reconstruction,
                        _age,
                    )
                    self.data[_case].loc[i, "number_of_plates"] = len(resolved_topologies)

    def calculate_subduction_length(
            self,
            slabs: 'Slabs' = None,
            ages: Optional[Union[int, float, _numpy.integer, _numpy.floating, list, _numpy.ndarray]] = None,
            cases: Optional[List[str]] = None,
        ):
        """
        Calculate the subduction length for each time step.
        """
        # Define ages if not provided
        _ages = utils_data.select_ages(ages, self.settings.ages)

        # Define cases if not provided
        _cases = utils_data.select_cases(cases, self.settings.cases)

        # Calculate the subduction length for each time step
        for i, _age in enumerate(_ages):
            for _case in _cases:
                # Check if slab data is provided
                if utils_init.check_object_data(slabs, Slabs, _age, _case):
                    logging.info(f"Calculating subduction length for case {_case} at age {_age} using provided data")
                    self.data[_case].loc[i, "subduction_length"] = slabs.data[_age][_case].trench_segment_length.sum()
                    slab_data = slabs.data[_age][_case]
                elif utils_init.check_object_data(self.slabs, Slabs, _age, _case):
                    logging.info(f"Calculating subduction length for case {_case} at age {_age} using stored data")
                    slab_data = self.slabs.data[_age][_case]
                    self.data[_case].loc[i, "subduction_length"] = self.slabs.data[_age][_case].trench_segment_length.sum()
                else:
                    logging.info(f"Calculating subduction length for case {_case} at age {_age} by tesselating subduction zones")
                    slab_data = self.reconstruction.tessellate_subduction_zones(
                        _age,
                        ignore_warnings=True,
                        tessellation_threshold_radians=(
                            self.settings.options[_case]["Slab tesselation spacing"]/self.constants.mean_Earth_radius_km
                        )
                    )
                    # Convert to _pandas.DataFrame
                    slab_data = _pandas.DataFrame(slab_data)

                    # Kick unused columns and rename the rest
                    slab_data = slab_data.drop(columns=[2, 3, 4, 5])
                    slab_data.columns = ["lon", "lat", "trench_segment_length", "trench_normal_azimuth", "lower_plateID", "trench_plateID"]

                    # Convert trench segment length from degree to m
                    slab_data.trench_segment_length *= self.constants.equatorial_Earth_circumference / 360

                # Store subduction length
                self.data[_case].loc[i, "subduction_length"] = slab_data.trench_segment_length.sum()

    def calculate_net_rotation(
            self,
            plates: Plates = None,
            points: Points = None,
            ages = None,
            cases = None,
            plateIDs = None,
        ):
        """
        Calculate the net rotation of the Earth's lithosphere.
        """
        # Define ages if not provided
        _ages = utils_data.select_ages(ages, self.settings.ages)

        # Define cases if not provided
        _cases = utils_data.select_cases(cases, self.settings.cases)

        # Calculate the net rotation of the Earth's lithosphere
        for i, _age in enumerate(_ages):
            for _case in _cases:
                # Check if plate data is provided
                if utils_init.check_object_data(plates, Plates, _age, _case):
                    logging.info(f"Calculating net rotation for case {_case} at age {_age} using provided Plates data")
                    _plates = plates
                elif utils_init.check_object_data(self.plates, Plates, _age, _case):
                    logging.info(f"Calculating net rotation for case {_case} at age {_age} using stored Plates data")
                    _plates = self.plates
                else:
                    logging.info(f"Instantiating Plates object for case {_case} at age {_age} to calculate net rotation")
                    # Get a new plates object if not provided
                    _plates = Plates(
                        self.settings,
                        self.reconstruction,
                        ages = _age,
                    )

                # Check if plate data is provided
                if utils_init.check_object_data(plates, Plates, _age, _case):
                    logging.info(f"Calculating net rotation for case {_case} at age {_age} using provided Points data")
                    _points = points
                elif utils_init.check_object_data(self.plates, Plates, _age, _case):
                    logging.info(f"Calculating net rotation for case {_case} at age {_age} using stored Points data")
                    _points = self.points
                else:
                    logging.info(f"Instantiating Plates object for case {_case} at age {_age} to calculate net rotation")
                    # Get a new plates object if not provided
                    _points = Points(
                        self.settings,
                        self.reconstruction,
                        ages = _age,
                        CALCULATE_VELOCITIES = False,
                    )

                # Check if plates and points are provided
                _plateIDs = utils_data.select_plateIDs(plateIDs, _plates.data[_age][_case].plateID.unique())

                # Select plates and points data
                selected_plates = _plates.data[_age][_case]
                selected_points = _points.data[_age][_case]
                if plateIDs is not None:
                    selected_plates = _plates.data[_age][_case][_plates.data[_age][_case].plateID.isin(_plateIDs)]
                    selected_points = _points.data[_age][_case][_points.data[_age][_case].plateID.isin(_plateIDs)]

                # Calculate net rotation
                net_rotation_pole = utils_calc.compute_net_rotation(
                    selected_plates,
                    selected_points,
                )

                # Store net rotation
                self.data[_case].loc[i, "net_rotation_pole_lat"] = net_rotation_pole[0]
                self.data[_case].loc[i, "net_rotation_pole_lon"] = net_rotation_pole[1]
                self.data[_case].loc[i, "net_rotation_rate"] = net_rotation_pole[2]

                logging.info(f"Net rotation for case {_case} at age {_age} has been calculated!")

    def calculate_world_uncertainty(
            self,
            ages = None,
            polygons = None,
            reconstructed_polygons = None
        ):
        """
        Calculate the fraction of the Earth's surface that has been lost to subduction.
        """
        # Define ages if not provided
        _ages = utils_data.select_ages(ages, self.settings.ages)

        # Check that the polygons are provided
        if not reconstructed_polygons:
            reconstructed_polygons = {}
            if polygons:
                if isinstance(polygons, str):
                    self.polygons = _pygplates.FeatureCollection(polygons)
                elif isinstance(polygons, _pygplates.FeatureCollection):
                    self.polygons = polygons
                else:
                    if hasattr(self.reconstruction, 'polygons'):
                        polygons = self.polygons
                    else:
                        raise ValueError("No static polygons provided!")
                
        # Loop through ages
        for i, _age in enumerate(_ages):
            if _age not in reconstructed_polygons.keys():
                # If the age is not present, try to reconstruct the polygons
                try:
                    reconstructed_polygons[_age] = _pygplates.reconstruct(
                        self.polygons,
                        self.reconstruction.rotation_model,
                        time = _age,
                    )
                except:
                    # If that doesn't work, assign NaN to this age
                    self.data[self.settings.cases[0]].loc[i, "world_uncertainty"] = _numpy.nan  
                    break    
            
            area = 0
            for polygon in reconstructed_polygons[_age]:
                area += polygon.get_geometry.get_area()

            self.data[self.settings.cases[0]].loc[i, "world_uncertainty"] = 1 - area        

            # Copy values to other cases
            for _case in self.settings.cases[1:]:
                self.data[_case] = self.data[self.settings.cases[0]]

    def save(
            self,
            cases: Optional[Union[str, List[str]]] = None,
            file_dir: Optional[str] = None,
        ):
        """
        Function to save 'Globe' object.
        Data of the 'Globe' object is saved to .parquet files.
        """
        # Define cases if not provided
        _cases = utils_data.select_cases(cases, self.settings.cases)
        
        # Get file dir
        _file_dir = self.settings.dir_path if file_dir is None else file_dir

        # Loop through ages
        for _case in _tqdm(_cases, desc="Saving Globe", disable=self.settings.logger.level==logging.INFO):
            utils_data.DataFrame_to_parquet(
                self.data[_case],
                "Globe",
                self.settings.name,
                None,
                _case,
                _file_dir,
                )
            
    def export(
            self,
            cases: Optional[Union[str, List[str]]] = None,
            file_dir: Optional[str] = None,
        ):
        """
        Function to export 'Globe' object.
        Data of the 'Globe' object is exported to .csv files.
        """
        # Define cases if not provided
        _cases = utils_data.select_cases(cases, self.settings.cases)
        
        # Get file dir
        _file_dir = self.settings.dir_path if file_dir is None else file_dir

        # Loop through ages
        for _case in _tqdm(_cases, desc="Saving Globe", disable=self.settings.logger.level==logging.INFO):
            utils_data.DataFrame_to_csv(
                self.data[_case],
                "Globe",
                self.settings.name,
                None,
                _case,
                _file_dir,
                )        