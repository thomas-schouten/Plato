import logging
from typing import Optional, List, Union

import numpy as _numpy
import gplately as _gplately
from gplately import pygplates as _pygplates
from tqdm import tqdm

from settings import Settings

def get_settings(
        settings: Optional['Settings'] = None,
        ages: Optional[Union[int, float, list, _numpy.integer, _numpy.floating, _numpy.ndarray]] = None,
        cases_file: Optional[str] = None,
        cases_sheet: Optional[str] = "Sheet1",
        files_dir: Optional[str] = None,
        reconstruction_name: Optional[str] = None,
        PARALLEL_MODE: Optional[bool] = False,
        DEBUG_MODE: Optional[bool] = False,
    ):
    """
    Function to set settings or initialise a new object.
    """
    if settings:
            _settings = settings
    else:
        if ages and cases_file:
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
    # Inform the user
    logging.info("Setting up plate reconstruction...")

    # If a reconstruction object is provided, return it
    if reconstruction:
        return reconstruction

    # Establish a connection to gplately DataServer if any file is missing
    gdownload = None
    if not rotation_file or not topology_file or not polygon_file:
        gdownload = _gplately.DataServer(name)

    # Download reconstruction files if any are missing
    if not rotation_file or not topology_file or not polygon_file:
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

def get_data(
        self,
        data,
        ages, 
        cases, 
        matching_cases, 
        function_init,
        *args, **kwargs
    ):
    """
    Function to load or initialise data for the current simulation.
    """
    
    # Load or initialize plate data
    _data = {}

    # Load the data if it is available
    if data is not None:
        for _age in tqdm(ages, desc="Loading data"):
            # Check if data is a dictionary
            if isinstance(data, dict):
                if _age in data.keys():
                    # Initialize _age only if there's a matching case
                    matching_cases_list = [_case for _case in cases if _case in data[_age].keys()]

                    # Only initialise _data[_age] if there are matching cases
                    if matching_cases_list:
                        _data[_age] = {}  # Initialise _age for matching cases

                    for _case in matching_cases_list:
                        _data[_age][_case] = data[_age][_case]

            # TODO: Implement loading data from a file if needed
            if isinstance(data, str):
                pass

    # Get missing ages
    existing_keys = set(_data.keys())
    missing_ages = _numpy.array([_age for _age in ages if _age not in existing_keys])

    # Loop through missing ages
    for _age in tqdm(missing_ages, desc="Initialising data"):
        # Get existing cases for the current age
        existing_cases = set(_data.get(_age, {}).keys())
        missing_cases = _numpy.array([_case for _case in cases if _case not in existing_cases])

        # Loop through missing cases
        for _case in missing_cases:
            # Check for matching case and copy data
            matching_key = next((key for key, cases in matching_cases.items() if _case in cases), None)
            if matching_key:
                for matching_case in matching_cases[matching_key]:  # Access the cases correctly
                    if matching_case in _data[_age]:
                        # Copy data from the matching case
                        _data[_age][_case] = _data[_age][matching_case].copy()
                        break  # Exit after copying to avoid overwriting

            else:
                # Use the provided data retrieval function if no matching case is found
                _data[_age][_case] = function_init(
                    _age,  # Pass the age parameter
                    *args,  # Additional positional arguments
                    **kwargs  # Additional keyword arguments
                )

    return _data
