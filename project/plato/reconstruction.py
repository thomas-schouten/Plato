# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# PLATO
# Algorithm to calculate plate forces from tectonic reconstructions
# Thomas Schouten, 2024
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Import libraries
import gplately as _gplately
from gplately import pygplates as _pygplates
import logging
from typing import Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# RECONSTRUCTION OBJECT
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

class Reconstruction:
    def __init__(
            self,
            name: str,
            rotation_file: Optional[str] = None,
            topology_file: Optional[str] = None,
            polygon_file: Optional[str] = None,
            coastline_file: Optional[str] = None,
        ):
        """
        Class to set up a plate reconstruction using gplately.

        :param name: Name of the reconstruction model.
        :type name: str
        :param rotation_file: Path to the rotation file (default: None).
        :type rotation_file: Optional[str]
        :param topology_file: Path to the topology file (default: None).
        :type topology_file: Optional[str]
        :param polygon_file: Path to the polygon file (default: None).
        :type polygon_file: Optional[str]
        :param coastline_file: Path to the coastline file (default: None).
        :type coastline_file: Optional[str]

        :raises ValueError: If the name of the reconstruction model is not valid.
        """

        logging.info("Setting up plate reconstruction...")  # Inform the user

        # Establish a connection to gplately DataServer if any file is missing
        gdownload = None
        if not rotation_file or not topology_file or not polygon_file or not coastline_file:
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
                raise ValueError(f"Please provide rotation and topology files or select a reconstruction from the following list: {valid_reconstructions}")

        # Initialise rotation model, topology features, polygons, and coastlines
        self.rotation_model = _pygplates.RotationModel(rotation_file) if rotation_file else reconstruction_files[0]
        self.topology_features = _pygplates.FeatureCollection(topology_file) if topology_file else reconstruction_files[1]
        self.polygons = _pygplates.FeatureCollection(polygon_file) if polygon_file else reconstruction_files[2]
        self.coastlines = _pygplates.FeatureCollection(coastline_file) if coastline_file else gdownload.get_topology_geometries()[0]

        # Create plate reconstruction object
        self.plate_reconstruction = _gplately.PlateReconstruction(self.rotation_model, self.topology_features, self.polygons)
            
        logging.info("Plate reconstruction ready!")  # Inform user that the setup is complete