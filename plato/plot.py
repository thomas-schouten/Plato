# Standard libraries
import logging
import warnings
from typing import Optional, Union

# Third-party libraries
import gplately as _gplately
from gplately import pygplates as _pygplates
import cartopy.crs as ccrs
import numpy as _numpy
import xarray as _xarray

# Plato libraries
from . import utils_init
from .settings import Settings
from .plates import Plates
from .slabs import Slabs
from .points import Points
from .grids import Grids
from .globe import Globe
from .plate_torques import PlateTorques

class PlotReconstruction():
    """
    A class to make standardised plots of reconstructions.
    """
    def __init__(
            self,
            plate_torques: Optional[PlateTorques] = None,
            settings: Optional[Settings] = None,
            reconstruction: Optional[_gplately.PlateReconstruction] = None,
            plates: Optional[Plates] = None,
            slabs: Optional[Slabs] = None,
            points: Optional[Points] = None,
            grids: Optional[Grids] = None,
            globe: Optional[Globe] = None,
            coastlines: Optional[_pygplates.FeatureCollection] = None,
        ):
        """
        Constructor for the Plot class.

        :param settings:        settings object
        :type settings:         Settings
        :param plates:          plates object
        :type plates:           Plates
        :param slabs:           slabs object
        :type slabs:            Slabs
        :param points:          points object
        :type points:           Points
        :param grids:           grids object
        :type grids:            Grids
        :param globe:           globe object
        :type globe:            Globe
        :param plate_torques:   plate torques object
        """
        # Store the input data, if provided
        if isinstance(settings, Settings):
            self.settings = settings
        else:
            self.settings = None

        if isinstance(reconstruction, _gplately.PlateReconstruction):
            self.reconstruction = reconstruction
        else:
            self.reconstruction = None

        if isinstance(plates, Plates):
            self.plates = plates
            if self.settings is None:
                self.settings = self.plates.settings
        else:
            self.plates = None

        if isinstance(slabs, Slabs):
            self.slabs = slabs
            if self.settings is None:
                self.settings = self.slabs.settings
        else:
            self.slabs = None

        if isinstance(points, Points):
            self.points = points
            if self.settings is None:
                self.settings = self.points.settings
        else:
            self.points = None

        if isinstance(grids, Grids):
            self.grids = grids
            if self.settings is None:
                self.settings = self.grids.settings
        else:
            self.grids = None

        if isinstance(globe, Globe):
            self.globe = globe
            if self.settings is None:
                self.settings = self.globe.settings
        else:
            self.globe = None

        if isinstance(plate_torques, PlateTorques):
            self.settings = plate_torques.settings
            self.reconstruction = plate_torques.reconstruction
            self.plates = plate_torques.plates
            self.slabs = plate_torques.slabs
            self.points = plate_torques.points
            self.grids = plate_torques.grids
            self.globe = plate_torques.globe

        # Get coastlines if not provided
        self.coastlines = utils_init.get_coastlines(coastlines, self.settings)

    def plot_seafloor_age_map(
            self,
            ax: object,
            age: int,
            cmap = "cmc.lajolla_r",
            vmin: Optional[Union[int, float, _numpy.integer, _numpy.floating]] = 0,
            vmax: Optional[Union[int, float, _numpy.integer, _numpy.floating]] = 250,
            log_scale: bool = False,
            coastlines_facecolour: Optional[str] = "lightgrey",
            coastlines_edgecolour: Optional[str] = "lightgrey",
            coastlines_linewidth: Optional[Union[int, float, _numpy.integer, _numpy.floating]] = 0,
            plate_boundaries_linewidth: Optional[Union[int, float, _numpy.integer, _numpy.floating]] = 0.5,
        ) -> object:
        """
        Function to create subplot with global seafloor age.

        :param ax:                      axes object
        :type ax:                       matplotlib.axes.Axes
        :param age:                     the time for which to display the map
        :type age:                      int
        :param cmap:                    colormap to use
        :type cmap:                     str
        :param vmin:                    minimum value for colormap
        :type vmin:                     int, float
        :param vmax:                    maximum value for colormap
        :type vmax:                     int, float
        :param log_scale:               whether or not to use log scale
        :type log_scale:                bool
        :param coastlines_facecolour:   facecolour for coastlines
        :type coastlines_facecolour:    str
        :param coastlines_edgecolour:   edgecolour for coastlines
        :type coastlines_edgecolour:    str
        :param coastlines_linewidth:    linewidth for coastlines

        :return:                        image object
        :rtype:                         matplotlib.image.AxesImage
        """
        # Set age to first in list if not provided
        if age is None or age not in self.settings.ages:
            warnings.warn("Invalid reconstruction age, using youngest age.")
            age = self.settings.ages[0]
        
        # Set case to first in list if not provided
        if case is None or case not in self.settings.cases:
            warnings.warn("Invalid case, using first case.")
            case = self.settings.cases[0]
        
        # Set basemap
        gl = self.plot_basemap(ax)

        # NOTE: We need to explicitly turn of top and right labels here, otherwise they will still show up sometimes
        gl.top_labels = False
        gl.right_labels = False

        if age in self.grids.seafloor_age and "seafloor_age" in self.grids.seafloor_age[age].data_vars:
            ax.imshow(self.grids.seafloor_age[age].seafloor_age,)
            
            # Plot seafloor age grid
            im = self.plot_grid(
                ax,
                self.grids.seafloor_age[age].seafloor_age,
                cmap = cmap,
                vmin = vmin,
                vmax = vmax,
                log_scale = log_scale
            )

        else:
            warnings.warn("No seafloor age grid available, only plotting the reconstruction")
            im = None

        # Plot plates and coastlines
        self.plot_reconstruction(
            ax,
            age,
            coastlines_facecolour = coastlines_facecolour,
            coastlines_edgecolour = coastlines_edgecolour,
            coastlines_linewidth = coastlines_linewidth,
            plate_boundaries_linewidth = plate_boundaries_linewidth,
        )

        return im

    def plot_sediment_map(
            self,
            ax,
            age: int,
            case,
            cmap = "cmc.imola",
            vmin = 1e0,
            vmax = 1e4,
            log_scale = True,
            coastlines_facecolour = "lightgrey",
            coastlines_edgecolour = "lightgrey",
            coastlines_linewidth = 0,
            plate_boundaries_linewidth = 0.5,
            marker_size = 20,
        ):
        """
        Function to create subplot with global sediment thicknesses.
        """
        # Set age to first in list if not provided
        if age is None or age not in self.settings.ages:
            warnings.warn("Invalid reconstruction age, using youngest age.")
            age = self.settings.ages[0]
        
        # Set case to first in list if not provided
        if case is None or case not in self.settings.cases:
            warnings.warn("Invalid case, using first case.")
            case = self.settings.cases[0]
        
        # Set basemap
        gl = self.plot_basemap(ax)

        # NOTE: We need to explicitly turn of top and right labels here, otherwise they will still show up sometimes
        gl.top_labels = False
        gl.right_labels = False

        # Get sediment thickness grid
        if age in self.grids.sediment and self.settings.options[case]["Sample sediment grid"] in self.grids.sediment[age].data_vars:           
            grid = self.seafloor[age][self.options[case]["Sample sediment grid"]].values
        else:
            grid = _numpy.where(_numpy.isnan(self.seafloor[age].seafloor_age.values), _numpy.nan, vmin)

        # Plot sediment thickness grid
        im = self.plot_grid(
            ax,
            grid,
            cmap = cmap,
            vmin = vmin,
            vmax = vmax,
            log_scale = log_scale
            )

        # Plot active margin sediments, if activated
        if self.settings.options[case]["Active margin sediments"] != 0 or self.settings.options[case]["Sample erosion grid"]:
            lat = self.slabs.data[age][case].lat.values
            lon = self.slabs.data[age][case].lon.values
            data = self.slabs.data[age][case].sediment_thickness.values
            
            if log_scale is True:
                if vmin == 0:
                    vmin = 1e-3
                if vmax == 0:
                    vmax = 1e3
                vmin = _numpy.log10(vmin)
                vmax = _numpy.log10(vmax)

                data = _numpy.where(
                    data == 0,
                    vmin,
                    _numpy.log10(data),
                )

            sc = ax.scatter(
                lon,
                lat,
                c=data,
                s=marker_size,
                transform=ccrs.PlateCarree(),
                cmap = cmap,
                vmin = vmin,
                vmax = vmax,
            )
        
        else:  
            sc = None

        # Plot plates and coastlines
        ax = self.plot_reconstruction(
            ax,
            age,
            coastlines_facecolour = coastlines_facecolour,
            coastlines_edgecolour = coastlines_edgecolour,
            coastlines_linewidth = coastlines_linewidth,
            plate_boundaries_linewidth = plate_boundaries_linewidth,
        )
            
        return im, sc
    
    def plot_erosion_rate_map(
            self,
            ax,
            age: int = None,
            case: str = None,
            cmap = "cmc.davos_r",
            vmin = 1e0,
            vmax = 1e3,
            log_scale = True,
            coastlines_facecolour = "none",
            coastlines_edgecolour = "none",
            coastlines_linewidth = 0,
            plate_boundaries_linewidth = 0.5,
        ):
        """
        Function to create subplot with global sediment thicknesses
            case:               case for which to plot the sediments
            plotting_options:   dictionary with options for plotting
        """
        # Set age to first in list if not provided
        if age is None or age not in self.settings.ages:
            warnings.warn("Invalid reconstruction age, using youngest age.")
            age = self.settings.ages[0]
        
        # Set case to first in list if not provided
        if case is None or case not in self.settings.cases:
            warnings.warn("Invalid case, using first case.")
            case = self.settings.cases[0]
        
        # Set basemap
        gl = self.plot_basemap(ax)

        # NOTE: We need to explicitly turn of top and right labels here, otherwise they will still show up sometimes
        gl.top_labels = False
        gl.right_labels = False

        # Get erosion rate grid
        if age in self.grids.continent and self.settings.options[case]["Sample erosion rate"] in self.grids.continent[age].data_vars:           
            grid = self.grids.continent[age].erosion_rate.values
            # Get erosion rate grid
            im = self.plot_grid(
                ax,
                grid,
                cmap = cmap,
                vmin = vmin,
                vmax = vmax,
                log_scale = log_scale
            )

        else:
            warnings.warn("No erosion rate grid available, only plotting the reconstruction")
            im = None
        
        # Plot plates and coastlines
        ax = self.plot_reconstruction(
            ax,
            age,
            coastlines_facecolour = coastlines_facecolour,
            coastlines_edgecolour = coastlines_edgecolour,
            coastlines_linewidth = coastlines_linewidth,
            plate_boundaries_linewidth = plate_boundaries_linewidth,
        )
            
        return im
    
    def plot_velocity_map(
            self,
            ax,
            age,
            case = None,
            cmap = "cmc.bilbao_r",
            velocity_component = "velocity_mag",
            vmin = 0,
            vmax = 25,
            normalise_vectors = True,
            log_scale = False,
            coastlines_facecolour = "none",
            coastlines_edgecolour = "black",
            coastlines_linewidth = 0.1,
            plate_boundaries_linewidth = 0.5,
            vector_width = 4e-3,
            vector_scale = 3e2,
            vector_scale_units = "width",
            vector_color = "k",
            vector_alpha = 0.5,
        ):
        """
        Function to plot plate velocities on an axes object
        """
        # Set age to first in list if not provided
        if age is None or age not in self.settings.ages:
            warnings.warn("Invalid reconstruction age, using youngest age.")
            age = self.settings.ages[0]
        
        # Set case to first in list if not provided
        if case is None or case not in self.settings.cases:
            warnings.warn("Invalid case, using first case.")
            case = self.settings.cases[0]
        
        # Set basemap
        gl = self.plot_basemap(ax)

        # NOTE: We need to explicitly turn of top and right labels here, otherwise they will still show up sometimes
        gl.top_labels = False
        gl.right_labels = False

        # Interpolate data from points to grid if not available
        if self.grids.velocity[age][case] is None or velocity_component not in self.grids.velocity[age][case]:
            print("Interpolating data to grid")
            self.grids.generate_velocity_grid(
                age,
                case,
                self.points.data,
                velocity_component,
            )
        
        # Get velocity grid
        grid = self.grids.velocity[age][case][velocity_component].values
        
        # Plot velocity difference grid
        im = self.plot_grid(
            ax,
            grid,
            cmap = cmap,
            vmin = vmin,
            vmax = vmax,
            log_scale = log_scale
        )

        # Get velocity vectors
        velocity_vectors = self.points.data[age][case].iloc[::209].copy()

        # Plot velocity vectors
        qu = self.plot_vectors(
            ax,
            velocity_vectors.lat.values,
            velocity_vectors.lon.values,
            velocity_vectors.velocity_lat.values,
            velocity_vectors.velocity_lon.values,
            velocity_vectors.velocity_mag.values,
            normalise_vectors = normalise_vectors,
            width = vector_width,
            scale = vector_scale,
            scale_units = vector_scale_units,
            facecolour = vector_color,
            alpha = vector_alpha
        )

        # Plot plates and coastlines
        ax = self.plot_reconstruction(
            ax,
            age,
            coastlines_facecolour = coastlines_facecolour,
            coastlines_edgecolour = coastlines_edgecolour,
            coastlines_linewidth = coastlines_linewidth,
            plate_boundaries_linewidth = plate_boundaries_linewidth,
        )

        return im, qu

    def plot_velocity_difference_map(
            self,
            ax,
            _age,
            case1,
            case2,
            cmap = "cmc.vik",
            vmin = -10,
            vmax = 10,
            normalise_vectors = False,
            log_scale = False,
            coastlines_facecolour = "none",
            coastlines_edgecolour = "black",
            coastlines_linewidth = 0.1,
            plate_boundaries_linewidth = 0.5,
            vector_width = 4e-3,
            vector_scale = 3e2,
            vector_color = "k",
            vector_alpha = 0.5,
        ):
        """
        Function to create subplot with difference between plate velocity at trenches between two cases
        
        :param ax:                  axes object
        :type ax:                   matplotlib.axes.Axes
        :param fig:                 figure
        :type fig:                  matplotlib.figure.Figure
        :param _age: the time for which to display the map
        :type _age:  int
        :param case1:               case 1 for which to use the velocities
        :type case1:                str
        :param case2:               case 2 to subtract from case 1
        :type case2:                str
        :param plotting_options:    dictionary with options for plotting
        :type plotting_options:     dict

        :return:                    image object and quiver object
        :rtype:                     matplotlib.image.AxesImage and matplotlib.quiver.Quiver
        """

        # Check if reconstruction time is in valid times
        if _age not in self.times:
            return print("Invalid reconstruction time")
        
        # Set basemap
        gl = self.plot_basemap(ax)

        # Get velocity difference grid
        grid = self.velocity[_age][case1].velocity_magnitude.values-self.velocity[_age][case2].velocity_magnitude.values

        # Plot velocity difference grid
        im = self.plot_grid(
            ax,
            grid,
            cmap = cmap,
            vmin = vmin,
            vmax = vmax,
            log_scale = log_scale
        )

        # Subsample velocity vectors
        velocity_vectors1 = self.points[_age][case1].iloc[::209].copy()
        velocity_vectors2 = self.points[_age][case2].iloc[::209].copy()

        # Plot velocity vectors
        qu = self.plot_vectors(
            ax,
            velocity_vectors1.lat.values,
            velocity_vectors1.lon.values,
            velocity_vectors1.v_lat.values - velocity_vectors2.v_lat.values,
            velocity_vectors1.v_lon.values - velocity_vectors2.v_lon.values,
            velocity_vectors1.v_mag.values - velocity_vectors2.v_mag.values,
            normalise_vectors = normalise_vectors,
            width = vector_width,
            scale = vector_scale,
            facecolour = vector_color,
            alpha = vector_alpha
        )

        # Plot plates and coastlines
        ax = self.plot_reconstruction(
            ax,
            _age,
            coastlines_facecolour = coastlines_facecolour,
            coastlines_edgecolour = coastlines_edgecolour,
            coastlines_linewidth = coastlines_linewidth,
            plate_boundaries_linewidth = plate_boundaries_linewidth,
        )

        return im, qu
    
    def plot_relative_velocity_difference_map(
            self,
            ax,
            _age,
            case1,
            case2,
            cmap = "cmc.cork",
            vmin = 1e-1,
            vmax = 1e1,
            log_scale = True,
            coastlines_facecolour = "none",
            coastlines_edgecolour = "black",
            coastlines_linewidth = 0.1,
            plate_boundaries_linewidth = 0.5,
            vector_width = 4e-3,
            vector_scale = 3e2,
            vector_color = "k",
            vector_alpha = 0.5,
        ):
        """
        Function to create subplot with difference between plate velocity at trenches between two cases
            case:               case for which to plot the sediments
            plotting_options:   dictionary with options for plotting
        """
        # Check if reconstruction time is in valid times
        if _age not in self.times:
            return print("Invalid reconstruction time")
        
        # Set basemap
        gl = self.plot_basemap(ax)

        # Get relative velocity difference grid
        # Ignore annoying warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            grid = _numpy.where(
                (self.velocity[_age][case1].velocity_magnitude.values == 0) | 
                (self.velocity[_age][case2].velocity_magnitude.values == 0) | 
                (_numpy.isnan(self.velocity[_age][case1].velocity_magnitude.values)) | 
                (_numpy.isnan(self.velocity[_age][case2].velocity_magnitude.values)),
                _numpy.nan,
                (self.velocity[_age][case1].velocity_magnitude.values / 
                _numpy.where(
                    self.velocity[_age][case2].velocity_magnitude.values == 0,
                    1e-10,
                    self.velocity[_age][case2].velocity_magnitude.values)
                )
            )

        # Set velocity grid
        im = self.plot_grid(
            ax,
            grid,
            cmap = cmap,
            vmin = vmin,
            vmax = vmax,
            log_scale = log_scale
        )

        # Get velocity vectors
        velocity_vectors1 = self.points[_age][case1].iloc[::209].copy()
        velocity_vectors2 = self.points[_age][case2].iloc[::209].copy()

        vector_lat = velocity_vectors1.v_lat.values - velocity_vectors2.v_lat.values
        vector_lon = velocity_vectors1.v_lon.values - velocity_vectors2.v_lon.values
        vector_mag = _numpy.sqrt(vector_lat**2 + vector_lon**2)

        # Plot velocity vectors
        qu = self.plot_vectors(
            ax,
            velocity_vectors1.lat.values,
            velocity_vectors1.lon.values,
            vector_lat,
            vector_lon,
            vector_mag,
            normalise_vectors = True,
            width = vector_width,
            scale = vector_scale,
            facecolour = vector_color,
            alpha = vector_alpha
        )

        # Plot plates and coastlines
        ax = self.plot_reconstruction(
            ax,
            _age,
            coastlines_facecolour = coastlines_facecolour,
            coastlines_edgecolour = coastlines_edgecolour,
            coastlines_linewidth = coastlines_linewidth,
            plate_boundaries_linewidth = plate_boundaries_linewidth,
        )

        return im, qu
    
    def plot_residual_force_map(
            self,
            ax,
            _age,
            case = None,
            trench_means = False,
            cmap = "cmc.lipari_r",
            vmin = 1e-2,
            vmax = 1e-1,
            normalise_data = True,
            normalise_vectors = True,
            log_scale = True,
            marker_size = 30,
            coastlines_facecolour = "lightgrey",
            coastlines_edgecolour = "lightgrey",
            coastlines_linewidth = 0,
            plate_boundaries_linewidth = 0.5,
            slab_vector_width = 2e-3,
            slab_vector_scale = 3e2,
            slab_vector_colour = "k",
            slab_vector_alpha = 1,
            plate_vector_width = 5e-3,
            plate_vector_scale = 3e2,
            plate_vector_facecolour = "white",
            plate_vector_edgecolour = "k",
            plate_vector_linewidth = 1,
            plate_vector_alpha = 1,
        ):
        """
        Function to plot plate velocities on an axes object
            ax:                     axes object
            fig:                    figure
            _age:    the time for which to display the map
            case:                   case for which to plot the sediments
            plotting_options:       dictionary with options for plotting
        """
        # Check if reconstruction time is in valid times
        if _age not in self.times:
            return print("Invalid reconstruction time")
        
        # Set case to first case in cases list if not specified
        if case is None:
            case = self.cases[0]
        
        # Set basemap
        gl = self.plot_basemap(ax)

        # Copy dataframe
        plot_slabs = self.slabs[_age][case].copy()
        plot_plates = self.plates[_age][case].copy()

        # Calculate means at trenches for the "residual_force_mag" column
        if trench_means is True:
            # Calculate the mean of "residual_force_mag" for each trench_plateID
            mean_values = plot_slabs.groupby("trench_plateID")["residual_force_mag"].transform("mean")
            
            # Assign the mean values back to the original "residual_force_mag" column
            plot_slabs["residual_force_mag"] = mean_values

        # Reorder entries to make sure the largest values are plotted on top
        plot_slabs = plot_slabs.sort_values("residual_force_mag", ascending=True)

        # Normalise data by dividing by the slab pull force magnitude
        slab_data = plot_slabs.residual_force_mag.values
        plate_data_lat = plot_plates.residual_force_lat.values
        plate_data_lon = plot_plates.residual_force_lon.values
        plate_data_mag = plot_plates.residual_force_mag.values

        if normalise_data is True:
            slab_data = _numpy.where(
                plot_slabs.slab_pull_force_mag.values == 0 | _numpy.isnan(plot_slabs.slab_pull_force_mag.values),
                0,
                slab_data / plot_slabs.slab_pull_force_mag.values
            )

            plate_data_lat = _numpy.where(
                plot_plates.slab_pull_force_mag.values == 0 | _numpy.isnan(plot_plates.slab_pull_force_mag.values),
                0,
                plate_data_lat / plot_plates.slab_pull_force_mag.values
            )

            plate_data_lon = _numpy.where(
                plot_plates.slab_pull_force_mag.values == 0 | _numpy.isnan(plot_plates.slab_pull_force_mag.values),
                0,
                plate_data_lon / plot_plates.slab_pull_force_mag.values
            )

            plate_data_mag = _numpy.where(
                plot_plates.slab_pull_force_mag.values == 0 | _numpy.isnan(plot_plates.slab_pull_force_mag.values),
                0,
                plate_data_mag / plot_plates.slab_pull_force_mag.values
            )
            
        # Convert to log scale, if needed
        if log_scale is True:
            if vmin == 0:
                vmin = 1e-3
            if vmax == 0:
                vmax = 1e3
            vmin = _numpy.log10(vmin)
            vmax = _numpy.log10(vmax)

            slab_data = _numpy.where(
                slab_data == 0 | _numpy.isnan(slab_data),
                vmin,
                _numpy.log10(slab_data),
            )

            # plate_data_lat = _numpy.where(
            #     plate_data_lat == 0 | _numpy.isnan(plate_data_lat),
            #     0,
            #     _numpy.log10(plate_data_lat),
            # )

            # plate_data_lon = _numpy.where(
            #     plate_data_lon == 0 | _numpy.isnan(plate_data_lon),
            #     0,
            #     _numpy.log10(plate_data_lon),
            # )

        # Plot velocity difference grid
        sc = ax.scatter(
                plot_slabs.lon.values,
                plot_slabs.lat.values,
                c = slab_data,
                s = marker_size,
                transform = ccrs.PlateCarree(),
                cmap = cmap,
                vmin = vmin,
                vmax = vmax,
            )

        # Get velocity vectors
        force_vectors = self.slabs[_age][case].iloc[::5].copy()

        # Plot velocity vectors
        slab_qu = self.plot_vectors(
            ax,
            force_vectors.lat.values,
            force_vectors.lon.values,
            force_vectors.residual_force_lat.values,
            force_vectors.residual_force_lon.values,
            force_vectors.residual_force_mag.values,
            normalise_vectors = normalise_vectors,
            width = slab_vector_width,
            scale = slab_vector_scale,
            facecolour = slab_vector_colour,
            alpha = slab_vector_alpha
        )

        # Plot residual torque vectors at plate centroids
        plate_qu = self.plot_vectors(
            ax,
            plot_plates.centroid_lat.values,
            plot_plates.centroid_lon.values,
            plate_data_lat,
            plate_data_lon,
            plate_data_mag,
            normalise_vectors = normalise_vectors,
            width = plate_vector_width,
            scale = plate_vector_scale,
            facecolour = plate_vector_facecolour,
            edgecolour = plate_vector_edgecolour,
            linewidth = plate_vector_linewidth,
            alpha = plate_vector_alpha,
            zorder = 10
        )

        # Plot plates and coastlines
        ax = self.plot_reconstruction(
            ax,
            _age,
            coastlines_facecolour = coastlines_facecolour,
            coastlines_edgecolour = coastlines_edgecolour,
            coastlines_linewidth = coastlines_linewidth,
            plate_boundaries_linewidth = plate_boundaries_linewidth,
        )

        return sc, slab_qu, plate_qu
    
    def plot_basemap(self, ax):
        """
        Function to plot a basemap on an axes object.

        :param ax:      axes object
        :type ax:       matplotlib.axes.Axes

        :return:        gridlines object
        :rtype:         cartopy.mpl.gridliner.Gridliner
        """
        # Set labels
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")

        # Set global extent
        ax.set_global()

        # Set gridlines
        gl = ax.gridlines(
            crs=ccrs.PlateCarree(), 
            draw_labels=True, 
            linewidth=0.5, 
            color="gray", 
            alpha=0.5, 
            linestyle="--", 
            zorder=5
        )

        # Turn off gridlabels for top and right
        gl.top_labels = False
        gl.right_labels = False  

        return gl
    
    def plot_grid(
            self,
            ax,
            grid,
            log_scale=False,
            vmin=0,
            vmax=1e3,
            cmap="viridis",
        ):
        """
        Function to plot a grid.

        :param ax:          axes object
        :type ax:           matplotlib.axes.Axes
        :param data:        data to plot
        :type data:         numpy.ndarray
        :param log_scale:   whether or not to use log scale
        :type log_scale:    bool
        :param vmin:        minimum value for colormap
        :type vmin:         float
        :param vmax:        maximum value for colormap
        :type vmax:         float
        :param cmap:        colormap to use
        :type cmap:         str

        :return:            image object
        :rtype:             matplotlib.image.AxesImage
        """
        # Set log scale
        if log_scale:
            if vmin == 0:
                vmin = 1e-3
            if vmax == 0:
                vmax = 1e3
            vmin = _numpy.log10(vmin)
            vmax = _numpy.log10(vmax)

            # Ignore annoying warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                grid = _numpy.where(
                    (_numpy.isnan(grid)) | (grid <= 0),
                    _numpy.nan,
                    _numpy.log10(grid),
                )

        transform = None if ax.projection == ccrs.PlateCarree() else ccrs.PlateCarree()

        # Plot grid    
        im = ax.imshow(
            grid,
            cmap = cmap,
            transform = transform, 
            zorder = 1, 
            vmin = vmin, 
            vmax = vmax, 
            origin = "lower",
            extent = [-180, 180, -90, 90]
        )

        return im
    
    def plot_vectors(
            self,
            ax,
            lat,
            lon,
            vector_lat,
            vector_lon,
            vector_mag = None,
            normalise_vectors = False,
            width = 4e-3,
            scale = 3e2,
            scale_units = "width",
            facecolour = "k",
            edgecolour = "none",
            linewidth = 1,
            alpha = 0.5,
            zorder = 4,
        ):
        """
        Function to plot vectors on an axes object.

        :param ax:                  axes object
        :type ax:                   matplotlib.axes.Axes
        :param lat:                 latitude of the vectors
        :type lat:                  numpy.ndarray
        :param lon:                 longitude of the vectors
        :type lon:                  numpy.ndarray
        :param vector_lat:          latitude component of the vectors
        :type vector_lat:           numpy.ndarray
        :param vector_lon:          longitude component of the vectors
        :type vector_lon:           numpy.ndarray
        :param vector_mag:          magnitude of the vectors
        :type vector_mag:           numpy.ndarray
        :param normalise_vectors:   whether or not to normalise the vectors
        :type normalise_vectors:    bool
        :param width:               width of the vectors
        :type width:                float
        :param scale:               scale of the vectors
        :type scale:                float
        :param zorder:              zorder of the vectors
        :type zorder:               int
        :param color:               color of the vectors
        :type color:                str
        :param alpha:               transparency of the vectors
        :type alpha:                float

        :return:                    quiver object
        :rtype:                     matplotlib.quiver.Quiver
        """
        # Normalise vectors, if necessary
        if normalise_vectors and vector_mag is not None:
            # Normalise by dividing by the magnitude of the vectors
            # Multiply by 10 to make the vectors more visible
            vector_lon = vector_lon / vector_mag * 10
            vector_lat = vector_lat / vector_mag * 10

            print(_numpy.nanmean(_numpy.sqrt(vector_lon**2 + vector_lat**2)))

            print("Normalised vectors!")

        # Plot vectors
        # Ignore annoying warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            qu = ax.quiver(
                    x = lon,
                    y = lat,
                    u = vector_lon,
                    v = vector_lat,
                    transform = ccrs.PlateCarree(),
                    width = width,
                    scale = scale,
                    scale_units = scale_units,
                    zorder = zorder,
                    color = facecolour,
                    edgecolor = edgecolour,
                    alpha = alpha,
                    linewidth = linewidth,
                )
        
        return qu
        
    def plot_reconstruction(
            self,
            ax,
            _age: int, 
            coastlines_facecolour = "none",
            coastlines_edgecolour = "none",
            coastlines_linewidth = "none",
            plate_boundaries_linewidth = "none",
        ):
        """
        Function to plot reconstructed features: coastlines, plates and trenches.

        :param ax:                      axes object
        :type ax:                       matplotlib.axes.Axes
        :param _age:     the time for which to display the map
        :type _age:      int
        :param plotting_options:        options for plotting
        :type plotting_options:         dict
        :param coastlines:              whether or not to plot coastlines
        :type coastlines:               boolean
        :param plates:                  whether or not to plot plates
        :type plates:                   boolean
        :param trenches:                whether or not to plot trenches
        :type trenches:                 boolean
        :param default_frame:           whether or not to use the default reconstruction
        :type default_frame:            boolean

        :return:                        axes object with plotted features
        :rtype:                         matplotlib.axes.Axes
        """
        # Set gplot object
        gplot = _gplately.PlotTopologies(self.reconstruction, time=_age, coastlines=self.coastlines)

        # Set zorder for coastlines. They should be plotted under seafloor grids but above velocity grids.
        if coastlines_facecolour == "none":
            zorder_coastlines = 2
        else:
            zorder_coastlines = -5

        # Plot coastlines
        # NOTE: Some reconstructions on the GPlately DataServer do not have polygons for coastlines, that's why we need to catch the exception.
        try:
            gplot.plot_coastlines(
                ax,
                facecolor = coastlines_facecolour,
                edgecolor = coastlines_edgecolour,
                zorder = zorder_coastlines,
                lw = coastlines_linewidth
            )
        except:
            pass
        
        # Plot plates 
        if plate_boundaries_linewidth:
            gplot.plot_all_topologies(ax, lw=plate_boundaries_linewidth, zorder=4)
            gplot.plot_subduction_teeth(ax, zorder=4)
            
        return ax
