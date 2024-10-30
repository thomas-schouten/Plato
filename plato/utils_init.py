import logging
from typing import Dict, List, Optional, Union

import numpy as _numpy
import gplately as _gplately
from gplately import pygplates as _pygplates
from tqdm import tqdm

from .settings import Settings
# from plates import Plates
# from points import Points
# from slabs import Slabs

def get_settings(
        settings: Optional['Settings'] = None,
        reconstruction_name: Optional[str] = None,
        ages: Optional[Union[int, float, list, _numpy.integer, _numpy.floating, _numpy.ndarray]] = None,
        cases_file: Optional[str] = None,
        cases_sheet: Optional[str] = "Sheet1",
        files_dir: Optional[str] = None,
        PARALLEL_MODE: Optional[bool] = False,
        DEBUG_MODE: Optional[bool] = False,
    ):
    """
    Function to set settings or initialise a new object.
    """
    if settings:
            _settings = settings
    else:
        if ages is not None and cases_file is not None:
            if reconstruction_name:
                name = reconstruction_name
            else:
                name = "Reconstruction"
            _settings = Settings(
                name = name,
                ages = ages,
                cases_file = cases_file,
                cases_sheet = cases_sheet,
                files_dir = files_dir,
                PARALLEL_MODE = False,
                DEBUG_MODE = False,
            )
        else:
            raise ValueError("Settings object or ages and cases should be provided.")
        
    return _settings

def get_reconstruction(
        reconstruction: Optional[_gplately.PlateReconstruction] = None,
        rotation_file: Optional[str] = None,
        topology_file: Optional[str] = None,
        polygon_file: Optional[str] = None,
        name: Optional[str] = None,
    ):
    """
    Function to set up a plate reconstruction using gplately.

    :param reconstruction: Reconstruction object.
    :type reconstruction: Optional[Union[_gplately.Reconstruction, 'Reconstruction']]
    :param rotation_file: Path to the rotation file (default: None).
    :type rotation_file: Optional[str]
    :param topology_file: Path to the topology file (default: None).
    :type topology_file: Optional[str]
    :param polygon_file: Path to the polygon file (default: None).
    :type polygon_file: Optional[str]
    :param name: Name of the reconstruction model.
    :type name: Optional[str]
    """
    # If a reconstruction object is provided, return it
    if reconstruction and reconstruction.static_polygons is not None:
        return reconstruction

    # Establish a connection to gplately DataServer if any file is missing
    gdownload = None
    if not rotation_file:
        logging.info(f"Missing rotation file for {name} plate reconstruction from GPlately DataServer.")
    if not topology_file:
        logging.info(f"Missing topology file for {name} plate reconstruction from GPlately DataServer.")
    if not polygon_file:
        logging.info(f"Missing polygon file for {name} plate reconstruction from GPlately DataServer.")

    if not rotation_file or not topology_file or not polygon_file:
        gdownload = _gplately.DataServer(name)

        valid_reconstructions = [
            "Muller2019", "Muller2016", "Merdith2021", "Cao2020", "Clennett2020", 
            "Seton2012", "Matthews2016", "Merdith2017", "Li2008", "Pehrsson2015", 
            "Young2019", "Scotese2008", "Clennett2020_M19", "Clennett2020_S13", 
            "Muller2020", "Shephard2013"
        ]
        
        if name in valid_reconstructions:
            logging.info(f"Downloading {name} reconstruction files from the _gplately DataServer...")
            reconstruction_files = gdownload.get_plate_reconstruction_files()
        else:
            logging.error(f"Invalid reconstruction name '{name}' provided. Valid reconstructions are: {valid_reconstructions}")
            raise ValueError(f"Please provide rotation and topology files or select a reconstruction from the following list: {valid_reconstructions}")
        
    # Initialise rotation model, topology features, polygons, and coastlines
    rotation_model = _pygplates.RotationModel(rotation_file) if rotation_file else reconstruction_files[0]
    topology_features = _pygplates.FeatureCollection(topology_file) if topology_file else reconstruction_files[1]
    polygons = _pygplates.FeatureCollection(polygon_file) if polygon_file else reconstruction_files[2]

    # Create plate reconstruction object
    reconstruction = _gplately.PlateReconstruction(rotation_model, topology_features, polygons)

    # Inform user that the setup is complete
    logging.info("Plate reconstruction ready!")  
    
    return reconstruction

def get_coastlines(
        coastlines: Optional[_pygplates.FeatureCollection] = None,
        settings: Optional['Settings'] = None,
    ):

    if isinstance(coastlines, _pygplates.FeatureCollection):
        return coastlines
    
    elif isinstance(coastlines, str):
        coastlines = _pygplates.FeatureCollection(coastlines)
        return coastlines
    
    elif isinstance(settings, Settings):
        valid_reconstructions = [
            "Muller2019", "Muller2016", "Merdith2021", "Cao2020", "Clennett2020", 
            "Seton2012", "Matthews2016", "Merdith2017", "Li2008", "Pehrsson2015", 
            "Young2019", "Scotese2008", "Clennett2020_M19", "Clennett2020_S13", 
            "Muller2020", "Shephard2013"
        ]
        
        if settings.name in valid_reconstructions:
            logging.info(f"Downloading {settings.name} reconstruction files from the _gplately DataServer...")
            gdownload = _gplately.DataServer(settings.name)
            coastlines, _, _ = gdownload.get_topology_geometries()

    else:
        raise Warning("No coastlines provided. Plotting maps without coastlines.")

    return coastlines

def check_object_data(
        obj,
        type,
        age,
        case
    ) -> bool:
    """
    Check if the given object has the required data for a specific age and case.
    """
    # Check if the object is of the required type
    if isinstance(obj, type):
        try:
            # Check if the object has the required age and case in its data attribute
            return age in obj.data and case in obj.data[age]
        except AttributeError:
            # If 'data' attribute is missing, return False
            return False
    else:
        # If the object is not of the required type, return False
        return False

# def get_plates(
#         plates: Optional['Plates'] = None,
#         settings: Optional[Union[None, Settings]]= None,
#         reconstruction: Optional[_gplately.PlateReconstruction]= None,
#         rotation_file: Optional[str]= None,
#         topology_file: Optional[str]= None,
#         polygon_file: Optional[str]= None,
#         reconstruction_name: Optional[str] = None,
#         ages: Optional[Union[_numpy.ndarray, List, float, int]] = None,
#         cases_file: Optional[list[str]]= None,
#         cases_sheet: Optional[str]= "Sheet1",
#         files_dir: Optional[str]= None,
#         resolved_geometries: Optional[Dict] = None,
#         PARALLEL_MODE: Optional[bool] = False,
#         DEBUG_MODE: Optional[bool] = False,
#     ):
#     """
#     Function to initialise a Plates object
    
#     """
#     # If a Plates object is provided, return it
#     if plates is 'Plates':
#         return plates
    
#     # Else, initialise one
#     plates = Plates(
#         settings,
#         reconstruction,
#         rotation_file,
#         topology_file,
#         polygon_file,
#         reconstruction_name,
#         ages,
#         cases_file,
#         cases_sheet,
#         files_dir,
#         resolved_geometries,
#     )

# def get_points(
        
#     ):
#     """
#     Function to initialise a Points object
#     """
#     pass

# def get_slabs(
        
#     ):
#     """
#     Function to initialise a Slabs object
#     """
#     pass

# def get_grids(
#         self,
#         settings: Optional['Settings'] = None,
#         cases_file: Optional[str] = None,
#         cases_sheet: Optional[str] = None,
#         reconstruction: Optional[_gplately.PlateReconstruction] = None,
#         reconstruction_name: Optional[str] = None,
#         rotation_file: Optional[str] = None,
#         topology_file: Optional[str] = None,
#         polygon_file: Optional[str] = None,
#         ages: Optional[list] = None,
#         files_dir: Optional[str] = None,
#         seafloor_grids: Optional[dict] = None,
#         continental_grids: Optional[dict] = None,
#         velocity_grids: Optional[dict] = None,
#         DEBUG_MODE: Optional[bool] = False,
#         PARALLEL_MODE: Optional[bool] = False,
#     ):
#     """
#     Function to initialise a Grids object
#     """
#     pass