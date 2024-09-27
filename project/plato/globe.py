import pandas as _pandas
import pygplates as _pygplates
import numpy as _numpy

class Globe:
    """
    Class to store information on global plate tectonic properties of the Earth.
    """
    
    def __init__(
        self,
        reconstruction = None,
        settings = None,
        plates = None,
        slabs = None,
        points = None,
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
        # Store the settings object
        if settings:
            self.settings = settings
        elif plates:
            self.settings = plates.settings
        elif slabs:
            self.settings = slabs.settings
        elif points:
            self.settings = points.settings
        else:
            raise ValueError("No settings object provided")

        # Store the reconstruction object
        if reconstruction:
            self.reconstruction = reconstruction
        elif plates:
            self.reconstruction = plates.reconstruction
        elif slabs:
            self.reconstruction = slabs.reconstruction
        elif points:
            self.reconstruction = points.reconstruction
        else:
            raise ValueError("No reconstruction object provided")

        # Initialise dataframe to store global properties for each case
        self.data = {case: _pandas.DataFrame({'Age': self.settings.ages}) for case in self.settings.cases}

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
        if ages is None:
            ages = self.settings.ages
        elif isinstance(ages, (int, float, _numpy.integer, _numpy.floating)):
            ages = [ages]
            
        # Get the plate data
        if plates_data is None:
            if hasattr(self, 'plates'):
                plates_data = self.plates.data
            else:
                raise ValueError("No plates object provided")
        
        # Calculate the number of plates for each time step
        for i, age in enumerate(self.settings.ages):
            for case in self.settings.cases:
                self.data[case].loc[i, "number_of_plates"] = len(plates_data[case][age])

    def calculate_subduction_length(
            self,
            ages = None,
            slabs_data = None
        ):
        """
        Calculate the subduction length for each time step.

        :param ages: List of ages for which to calculate subduction length (default: None)
        :type ages: Optional[int, float, numpy.integer, numpy.floating, list, numpy.ndarray]
        :param slabs_data: Optional slab data for each age and case (default: None)
        :type slabs_data: Optional[dict]
        """
        # Define ages if not provided
        if ages is None:
            ages = self.settings.ages
        elif isinstance(ages, (int, float, _numpy.integer, _numpy.floating)):
            ages = [ages]
        
        # Get the slabs data
        if slabs_data is None:
            if hasattr(self, 'slabs'):
                slabs_data = self.slabs.data
            else:
                raise ValueError("No slabs object provided")

        # Calculate the subduction length for each time step
        for i, age in enumerate(self.settings.ages):
            for case in self.settings.cases:
                self.data[case].loc[i, "subduction_length"] = slabs_data[case][age].trench_segment_length.sum()

    def calculate_net_rotation(
            self,
            ages = None,
            plates_data = None
        ):
        """
        Calculate the net rotation of the Earth's lithosphere.

        :param ages: List of ages for which to calculate net rotation (default: None)
        :type ages: Optional[int, float, numpy.integer, numpy.floating, list, numpy.ndarray]
        :param plates_data: Optional plate data for each case and age (default: None)
        :type plates_data: Optional[dict]
        """
        # Define ages if not provided
        if ages is None:
            ages = self.settings.ages
        elif isinstance(ages, (int, float, _numpy.integer, _numpy.floating)):
            ages = [ages]

        # Get the plate data
        if plates_data is None:
            if hasattr(self, 'plates'):
                plates_data = self.plates.data
            else:
                raise ValueError("No plates object provided")
        
        # Calculate the net rotation of the Earth's lithosphere
        # for i, age in enumerate(self.settings.ages):
        # TODO: Implement this function

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
