import logging
from typing import Dict, List, Optional, Union

import gplately as _gplately
from gplately import pygplates as _pygplates
import numpy as _numpy
import pandas as _pandas
from tqdm import tqdm as _tqdm

import utils_data, utils_calc, utils_init
from settings import Settings
from plates import Plates
from points import Points
from slabs import Slabs

class Globe:
    """
    Class to store information on global plate tectonic properties of the Earth.
    """
    def __init__(
            self,
            settings: Optional[Settings] = None,
            reconstruction: Optional[_gplately.PlateReconstruction]= None,
            rotation_file: Optional[str]= None,
            topology_file: Optional[str]= None,
            polygon_file: Optional[str]= None,
            reconstruction_name: Optional[str] = None,
            ages: Optional[Union[_numpy.ndarray, List, float, int]] = None,
            cases_file: Optional[list[str]]= None,
            cases_sheet: Optional[str]= "Sheet1",
            files_dir: Optional[str]= None,
            resolved_geometries: Optional[Dict] = None,
            plates: Optional['Plates'] = None,
            points: Optional['Points'] = None,
            slabs: Optional['Slabs'] = None,
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
            "world_uncertainty": 0.,
            "subduction_length": 0.,
            "net_rotation_pole_lat": 0.,
            "net_rotation_pole_lon": 0.,
            "net_rotation_rate": 0.}
        ) for case in self.settings.cases}

        # Get the number of plates
        self.calculate_number_of_plates()

        # Get the subduction length
        self.calculate_subduction_length()

        # Get the net rotation
        self.calculate_net_rotation()

        # Get the world uncertainty
        self.calculate_world_uncertainty()

    def calculate_number_of_plates(
            self,
            plates: 'Plates' = None,
            ages = None,
            cases = None,
        ):
        """
        Calculate the number of plates for each time step.

        :param ages: List of ages for which to calculate the number of plates (default: None)
        :type ages: Optional[int, float, numpy.integer, numpy.floating, list, numpy.ndarray]
        :param plates_data: Optional plate data for each case and age (default: None)
        :type plates_data: Optional[dict]
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
            slabs: Optional[Slabs] = None,
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
                if isinstance(slabs, Slabs) and _age in slabs.data.keys() and _case in slabs.data[_age].keys():
                    logging.info(f"Calculating subduction length for case {_case} at age {_age} using provided data")
                    self.data[_case].loc[i, "subduction_length"] = slabs.data[_age][_case].trench_segment_length.sum()
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

                    # Store subduction length
                    self.data[_case].loc[i, "subduction_length"] = slabs.trench_segment_length.sum()

    def calculate_net_rotation(
            self,
            plates: 'Plates' = None,
            points: 'Points' = None,
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
                if not isinstance(plates, Plates) or _age not in plates.data.keys() or _case not in plates.data[_age].keys():
                    logging.info(f"Instantiating Plates object for case {_case} at age {_age} to calculate net rotation")
                    # Get a new plates object if not provided
                    plates = Plates(
                        self.settings,
                        self.reconstruction,
                        ages = _age,
                    )

                # Check if point data is provided
                if not isinstance(plates, Points) or _age not in points.data.keys() or _case not in points.data[_age].keys():
                    logging.info(f"Instantiating Points object for case {_case} at age {_age} to calculate net rotation")
                    # Get a new points object if not provided
                    points = Points(
                        self.settings,
                        self.reconstruction,
                        ages = _age,
                        CALCULATE_VELOCITIES = False,
                    )

                # Check if plates and points are provided
                _plateIDs = utils_data.select_plateIDs(plateIDs, plates.data[_age][_case].plateID.unique())

                # Select plates and points data
                selected_plates = plates.data[_age][_case]
                selected_points = points.data[_age][_case]
                if plateIDs is not None:
                    selected_plates = plates.data[_age][_case][plates.data[_age][_case].plateID.isin(_plateIDs)]
                    selected_points = points.data[_age][_case][points.data[_age][_case].plateID.isin(_plateIDs)]

                # Calculate net rotation
                net_rotation_pole = utils_calc.compute_net_rotation(
                    selected_plates,
                    selected_points,
                )

                # Store net rotation
                self.data[_case].loc[i, "net_rotation_pole_lat"] = net_rotation_pole[0]
                self.data[_case].loc[i, "net_rotation_pole_lon"] = net_rotation_pole[1]
                self.data[_case].loc[i, "net_rotation_rate"] = net_rotation_pole[2]

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
        Data of the 'Globe' object is export to .csv files.
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