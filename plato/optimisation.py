# Standard libraries
import logging
import warnings
from typing import Optional, Union

# Third-party libraries
import numpy as _numpy
import matplotlib.pyplot as plt

# Plato libraries
from .settings import Settings
from .plates import Plates
from .slabs import Slabs
from .points import Points
from .grids import Grids
from .globe import Globe
from .plate_torques import PlateTorques

class Optimisation():
    """
    A class to optimise the slab pull coefficient and mantle viscosity to match plate motions.
    """    
    def __init__(
            self,
            settings: Optional[Settings] = None,
            plates: Optional[Plates] = None,
            slabs: Optional[Slabs] = None,
            points: Optional[Points] = None,
            grids: Optional[Grids] = None,
            globe: Optional[Globe] = None,
            plate_torques: Optional[PlateTorques] = None,
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
            self.plates = plate_torques.plates
            self.slabs = plate_torques.slabs
            self.points = plate_torques.points
            self.grids = plate_torques.grids
            self.globe = plate_torques.globe
        
    def minimise_residual_torque(
            self,
            opt_time,
            opt_case,
            plates_of_interest=None,
            grid_size=500,
            visc_range=[5e18, 5e20],
            plot=True,
            weight_by_area=True,
            minimum_plate_area=None
        ):
        """
        Function to find optimised coefficients to match plate motions using a grid search

        :param opt_time:                reconstruction time to optimise
        :type opt_time:                 int
        :param opt_case:                case to optimise
        :type opt_case:                 str
        :param plates_of_interest:      plate IDs to include in optimisation
        :type plates_of_interest:       list of integers or None
        :param grid_size:               size of the grid to find optimal viscosity and slab pull coefficient
        :type grid_size:                int
        :param plot:                    whether or not to plot the grid
        :type plot:                     boolean
        :param weight_by_area:          whether or not to weight the residual torque by plate area
        :type weight_by_area:           boolean

        :return:                        None
        """
        # Set range of viscosities
        viscs = _numpy.linspace(visc_range[0],visc_range[1],grid_size)

        # Set range of slab pull coefficients
        if self.options[opt_case]["Sediment subduction"]:
            # Range is smaller with sediment subduction
            sp_consts = _numpy.linspace(1e-5,0.25,grid_size)
        else:
            sp_consts = _numpy.linspace(1e-5,1.,grid_size)

        # Create grids from ranges of viscosities and slab pull coefficients
        visc_grid, sp_const_grid = _numpy.meshgrid(viscs, sp_consts)
        ones_grid = _numpy.ones_like(visc_grid)

        # Filter plates
        selected_plates = self.plates[opt_time][opt_case].copy()
        if plates_of_interest:
            selected_plates = selected_plates[selected_plates["plateID"].isin(plates_of_interest)]
            selected_plates = selected_plates.reset_index(drop=True)
        else:
            plates_of_interest = selected_plates["plateID"]

        # Filter plates by minimum area
        if minimum_plate_area is None:
            minimum_plate_area = self.options[opt_case]["Minimum plate area"]
        selected_plates = selected_plates[selected_plates["area"] > minimum_plate_area]
        selected_plates = selected_plates.reset_index(drop=True)
        plates_of_interest = selected_plates["plateID"]

        # Get total area
        total_area = selected_plates["area"].sum()

        # Initialise dictionaries and arrays to store driving and residual torques
        if opt_time not in self.driving_torque:
            self.driving_torque[opt_time] = {}
        if opt_time not in self.driving_torque_normalised:
            self.driving_torque_normalised[opt_time] = {}
        if opt_time not in self.residual_torque:
            self.residual_torque[opt_time] = {}
        if opt_time not in self.residual_torque_normalised:
            self.residual_torque_normalised[opt_time] = {}
            
        self.driving_torque[opt_time][opt_case] = _numpy.zeros_like(sp_const_grid); self.driving_torque_normalised[opt_time][opt_case] = _numpy.zeros_like(sp_const_grid)
        self.residual_torque[opt_time][opt_case] = _numpy.zeros_like(sp_const_grid); self.residual_torque_normalised[opt_time][opt_case] = _numpy.zeros_like(sp_const_grid)

        # Initialise dictionaries to store optimal coefficients
        if opt_time not in self.opt_i:
            self.opt_i[opt_time] = {}
        if opt_time not in self.opt_j:
            self.opt_j[opt_time] = {}
        if opt_time not in self.opt_sp_const:
            self.opt_sp_const[opt_time] = {}
        if opt_time not in self.opt_visc:
            self.opt_visc[opt_time] = {}

        # Get torques
        for k, _ in enumerate(plates_of_interest):
            residual_x = _numpy.zeros_like(sp_const_grid); residual_y = _numpy.zeros_like(sp_const_grid); residual_z = _numpy.zeros_like(sp_const_grid)
            if self.options[opt_case]["Slab pull torque"] and "slab_pull_torque_x" in selected_plates.columns:
                residual_x -= selected_plates.slab_pull_torque_x.iloc[k] * sp_const_grid
                residual_y -= selected_plates.slab_pull_torque_y.iloc[k] * sp_const_grid
                residual_z -= selected_plates.slab_pull_torque_z.iloc[k] * sp_const_grid

            # Add GPE torque
            if self.options[opt_case]["GPE torque"] and "GPE_torque_x" in selected_plates.columns:
                residual_x -= selected_plates.GPE_torque_x.iloc[k] * ones_grid
                residual_y -= selected_plates.GPE_torque_y.iloc[k] * ones_grid
                residual_z -= selected_plates.GPE_torque_z.iloc[k] * ones_grid
            
            # Compute magnitude of driving torque
            if weight_by_area:
                self.driving_torque[opt_time][opt_case] += _numpy.sqrt(residual_x**2 + residual_y**2 + residual_z**2) * selected_plates.area.iloc[k] / total_area
            else:
                self.driving_torque[opt_time][opt_case] += _numpy.sqrt(residual_x**2 + residual_y**2 + residual_z**2) / selected_plates.area.iloc[k]

            # Add slab bend torque
            if self.options[opt_case]["Slab bend torque"] and "slab_bend_torque_x" in selected_plates.columns:
                residual_x -= selected_plates.slab_bend_torque_x.iloc[k] * ones_grid
                residual_y -= selected_plates.slab_bend_torque_y.iloc[k] * ones_grid
                residual_z -= selected_plates.slab_bend_torque_z.iloc[k] * ones_grid

            # Add mantle drag torque
            if self.options[opt_case]["Mantle drag torque"] and "mantle_drag_torque_x" in selected_plates.columns:
                residual_x -= selected_plates.mantle_drag_torque_x.iloc[k] * visc_grid / self.mech.La
                residual_y -= selected_plates.mantle_drag_torque_y.iloc[k] * visc_grid / self.mech.La
                residual_z -= selected_plates.mantle_drag_torque_z.iloc[k] * visc_grid / self.mech.La

            # Compute magnitude of residual
            if weight_by_area:
                self.residual_torque[opt_time][opt_case] += _numpy.sqrt(residual_x**2 + residual_y**2 + residual_z**2) * selected_plates.area.iloc[k] / total_area
            else:
                self.residual_torque[opt_time][opt_case] += _numpy.sqrt(residual_x**2 + residual_y**2 + residual_z**2) / selected_plates.area.iloc[k]
    
            # Divide residual by driving torque
            self.residual_torque_normalised[opt_time][opt_case] = _numpy.log10(self.residual_torque[opt_time][opt_case] / self.driving_torque[opt_time][opt_case])

        # Find the indices of the minimum value directly using _numpy.argmin
        self.opt_i[opt_time][opt_case], self.opt_j[opt_time][opt_case] = _numpy.unravel_index(_numpy.argmin(self.residual_torque_normalised[opt_time][opt_case]), self.residual_torque_normalised[opt_time][opt_case].shape)
        self.opt_visc[opt_time][opt_case] = visc_grid[self.opt_i[opt_time][opt_case], self.opt_j[opt_time][opt_case]]
        self.opt_sp_const[opt_time][opt_case] = sp_const_grid[self.opt_i[opt_time][opt_case], self.opt_j[opt_time][opt_case]]

        # Plot
        if plot == True:
            fig, ax = plt.subplots(figsize=(15*self.constants.cm2in, 12*self.constants.cm2in))
            im = ax.imshow(self.residual_torque_normalised[opt_time][opt_case], cmap="cmc.lapaz_r", vmin=-1.5, vmax=1.5)
            ax.set_yticks(_numpy.linspace(0, grid_size - 1, 5))
            ax.set_xticks(_numpy.linspace(0, grid_size - 1, 5))
            ax.set_xticklabels(["{:.2e}".format(visc) for visc in _numpy.linspace(visc_range[0], visc_range[1], 5)])
            ax.set_yticklabels(["{:.2f}".format(sp_const) for sp_const in _numpy.linspace(sp_consts.min(), sp_consts.max(), 5)])
            ax.set_xlabel("Mantle viscosity [Pa s]")
            ax.set_ylabel("Slab pull reduction factor")
            ax.scatter(self.opt_j[opt_time][opt_case], self.opt_i[opt_time][opt_case], marker="*", facecolor="none", edgecolor="k", s=30)  # Adjust the marker style and size as needed
            fig.colorbar(im, label = "Log(residual torque/driving torque)")
            plt.show()

        # Print results
        print(f"Optimal coefficients for ", ", ".join(selected_plates.name.astype(str)), " plate(s), (PlateIDs: ", ", ".join(selected_plates.plateID.astype(str)), ")")
        print("Minimum residual torque: {:.2%} of driving torque".format(10**(_numpy.amin(self.residual_torque_normalised[opt_time][opt_case]))))
        print("Optimum viscosity [Pa s]: {:.2e}".format(self.opt_visc[opt_time][opt_case]))
        print("Optimum Drag Coefficient [Pa s/m]: {:.2e}".format(self.opt_visc[opt_time][opt_case] / self.mech.La))
        print("Optimum Slab Pull constant: {:.2%}".format(self.opt_sp_const[opt_time][opt_case]))

        return self.opt_sp_const[opt_time][opt_case], self.opt_visc[opt_time][opt_case], self.residual_torque_normalised[opt_time][opt_case]
    
    def find_slab_pull_coefficient(
            self,
            opt_time, 
            opt_case, 
            plates_of_interest=None, 
            grid_size=500, 
            viscosity=1e19, 
            plot=False, 
        ):
        """
        Function to find optimised slab pull coefficient for a given (set of) plates using a grid search.

        :param opt_time:                reconstruction time to optimise
        :type opt_time:                 int
        :param opt_case:                case to optimise
        :type opt_case:                 str
        :param plates_of_interest:      plate IDs to include in optimisation
        :type plates_of_interest:       list of integers or None
        :param grid_size:               size of the grid to find optimal slab pull coefficient
        :type grid_size:                int
        :param plot:                    whether or not to plot the grid
        :type plot:                     boolean
        :param weight_by_area:          whether or not to weight the residual torque by plate area
        :type weight_by_area:           boolean
        
        :return:                        The optimal slab pull coefficient
        :rtype:                         float
        """
        # Generate range of possible slab pull coefficients
        sp_consts = _numpy.linspace(1e-5,1,grid_size)
        ones = _numpy.ones_like(sp_consts)

        # Filter plates
        selected_plates = self.plates[opt_time][opt_case].copy()
        if plates_of_interest:
            selected_plates = selected_plates[selected_plates["plateID"].isin(plates_of_interest)]
            if selected_plates.empty:
                return _numpy.nan, _numpy.nan, _numpy.nan
            
            selected_plates = selected_plates.reset_index(drop=True)
        else:
            plates_of_interest = selected_plates["plateID"]

        # Initialise dictionary to store optimal slab pull coefficient per plate
        opt_sp_consts = {None for _ in plates_of_interest}
        
        # Loop through plates
        for k, plateID in enumerate(plates_of_interest):
            residual_x = _numpy.zeros_like(sp_consts)
            residual_y = _numpy.zeros_like(sp_consts)
            residual_z = _numpy.zeros_like(sp_consts)

            if self.options[opt_case]["Slab pull torque"] and "slab_pull_torque_x" in selected_plates.columns:
                residual_x -= selected_plates.slab_pull_torque_x.iloc[k] * sp_consts
                residual_y -= selected_plates.slab_pull_torque_y.iloc[k] * sp_consts
                residual_z -= selected_plates.slab_pull_torque_z.iloc[k] * sp_consts

            # Add GPE torque
            if self.options[opt_case]["GPE torque"] and "GPE_torque_x" in selected_plates.columns:
                residual_x -= selected_plates.GPE_torque_x.iloc[k] * ones
                residual_y -= selected_plates.GPE_torque_y.iloc[k] * ones
                residual_z -= selected_plates.GPE_torque_z.iloc[k] * ones

            # Compute magnitude of driving torque
            driving_mag = _numpy.sqrt(residual_x**2 + residual_y**2 + residual_z**2)
            
            # Add slab bend torque
            if self.options[opt_case]["Slab bend torque"] and "slab_bend_torque_x" in selected_plates.columns:
                residual_x -= selected_plates.slab_bend_torque_x.iloc[k] * ones
                residual_y -= selected_plates.slab_bend_torque_y.iloc[k] * ones
                residual_z -= selected_plates.slab_bend_torque_z.iloc[k] * ones

            # Add mantle drag torque
            if self.options[opt_case]["Mantle drag torque"] and "mantle_drag_torque_x" in selected_plates.columns:
                residual_x -= selected_plates.mantle_drag_torque_x.iloc[k] * viscosity / self.mech.La
                residual_y -= selected_plates.mantle_drag_torque_y.iloc[k] * viscosity / self.mech.La
                residual_z -= selected_plates.mantle_drag_torque_z.iloc[k] * viscosity / self.mech.La

            # Compute magnitude of residual
            residual_mag = _numpy.sqrt(residual_x**2 + residual_y**2 + residual_z**2)

            # Find minimum residual torque
            residual_mag_min = residual_mag[_numpy.argmin(_numpy.log10(residual_mag/driving_mag))]

            if plot:
                fig, ax = plt.subplots(figsize=(15*self.constants.cm2in, 12*self.constants.cm2in))
                p = ax.plot(residual_mag/driving_mag)
                ax.semilogy()
                ax.set_xticks(_numpy.linspace(0, grid_size - 1, 5))
                ax.set_xticklabels(["{:.2f}".format(sp_const) for sp_const in _numpy.linspace(sp_consts.min(), sp_consts.max(), 5)])
                ax.set_ylim([10**-1.5, 10**1.5])
                ax.set_xlim([0, grid_size])
                ax.set_ylabel("Normalised residual torque")
                ax.set_xlabel("Slab pull reduction factor")
                plt.show()

            # Find optimal driving torque
            driving_mag_min = driving_mag[_numpy.argmin(_numpy.log10(residual_mag/driving_mag))]

            # Find optimal slab pull coefficient
            opt_sp_const = sp_consts[_numpy.argmin(_numpy.log10(residual_mag/driving_mag))]

        return opt_sp_const, driving_mag_min, residual_mag_min
    
    def minimise_residual_velocity(self, opt_time, opt_case, plates_of_interest=None, grid_size=10, visc_range=[1e19, 5e20], plot=True, weight_by_area=True, ref_case=None):
        """
        Function to find optimised coefficients to match plate motions using a grid search.

        :param opt_time:                reconstruction time to optimise
        :type opt_time:                 int
        :param opt_case:                case to optimise
        :type opt_case:                 str
        :param plates_of_interest:      plate IDs to include in optimisation
        :type plates_of_interest:       list of integers or None
        :param grid_size:               size of the grid to find optimal viscosity and slab pull coefficient
        :type grid_size:                int
        :param plot:                    whether or not to plot the grid
        :type plot:                     boolean
        :param weight_by_area:          whether or not to weight the residual torque by plate area
        :type weight_by_area:           boolean
        
        :return:                        The optimal slab pull coefficient, the optimal viscosity, the residual plate velocity, and the residual slab velocity.
        :rtype:                         float, float, float, float
        """
        if self.options[opt_case]["Reconstructed motions"]:
            print("Optimisation method designed for synthetic plate velocities only!")
            return
        
        # Get "true" plate velocities
        true_slabs = self.slabs[opt_time][ref_case].copy()

        # Generate grid
        viscs = _numpy.linspace(visc_range[0],visc_range[1],grid_size)
        sp_consts = _numpy.linspace(1e-4,1,grid_size)
        v_upper_plate_residual = _numpy.zeros((grid_size, grid_size))
        v_lower_plate_residual = _numpy.zeros((grid_size, grid_size))
        v_convergence_residual = _numpy.zeros((grid_size, grid_size))

        # Filter plates and slabs
        selected_plates = self.plates[opt_time][opt_case].copy()
        selected_slabs = self.slabs[opt_time][opt_case].copy()
        selected_points = self.points[opt_time][opt_case].copy()

        if plates_of_interest:
            selected_plates = selected_plates[selected_plates["plateID"].isin(plates_of_interest)]
            selected_plates = selected_plates.reset_index(drop=True)
            selected_slabs = selected_slabs[selected_slabs["lower_plateID"].isin(plates_of_interest)]
            selected_slabs = selected_slabs.reset_index(drop=True)
            selected_points = selected_points[selected_points["plateID"].isin(plates_of_interest)]
            selected_points = selected_points.reset_index(drop=True)
            selected_options = self.options[opt_case].copy()
        else:
            plates_of_interest = selected_plates["plateID"]

        # Initialise starting old_plates, old_points, old_slabs by copying self.plates[reconstruction_time][key], self.points[reconstruction_time][key], self.slabs[reconstruction_time][key]
        old_plates = selected_plates.copy(); old_points = selected_points.copy(); old_slabs = selected_slabs.copy()

        # Delete self.slabs[reconstruction_time][key], self.points[reconstruction_time][key], self.plates[reconstruction_time][key]
        del selected_plates, selected_points, selected_slabs
        
        # Loop through plates and slabs and calculate residual velocity
        for i, visc in enumerate(viscs):
            for j, sp_const in enumerate(sp_consts):
                print(i, j)
                # Assign current visc and sp_const to options
                selected_options["Mantle viscosity"] = visc
                selected_options["Slab pull constant"] = sp_const

                # Optimise slab pull force
                [old_plates.update({"slab_pull_torque_opt_" + axis: old_plates["slab_pull_torque_" + axis] * selected_options["Slab pull constant"]}) for axis in ["x", "y", "z"]]

                for k in range(100):
                    # Delete new DataFrames
                    if k != 0:
                        del new_slabs, new_points, new_plates
                    else:
                        old_slabs["v_convergence_mag"] = 0

                    print(_numpy.mean(old_slabs["v_convergence_mag"].values))
                    # Compute interface shear force
                    if self.options[opt_case]["Interface shear torque"]:
                        new_slabs = utils_calc.compute_interface_shear_force(old_slabs, self.options[opt_case], self.mech, self.constants)
                    else:
                        new_slabs = old_slabs.copy()

                    # Compute interface shear torque
                    new_plates = utils_calc.compute_torque_on_plates(
                        old_plates,
                        new_slabs.lat,
                        new_slabs.lon,
                        new_slabs.lower_plateID,
                        new_slabs.interface_shear_force_lat,
                        new_slabs.interface_shear_force_lon,
                        new_slabs.trench_segment_length,
                        1,
                        self.constants,
                        torque_variable="interface_shear_torque"
                    )

                    # Compute mantle drag force
                    new_plates, new_points, new_slabs = utils_calc.compute_mantle_drag_force(old_plates, old_points, new_slabs, self.options[opt_case], self.mech, self.constants)

                    # Compute mantle drag torque
                    new_plates = utils_calc.compute_torque_on_plates(
                        new_plates, 
                        new_points.lat, 
                        new_points.lon, 
                        new_points.plateID, 
                        new_points.mantle_drag_force_lat, 
                        new_points.mantle_drag_force_lon,
                        new_points.segment_length_lat,
                        new_points.segment_length_lon,
                        self.constants,
                        torque_variable="mantle_drag_torque"
                    )

                    # Calculate convergence rates
                    v_convergence_lat = new_slabs["v_lower_plate_lat"].values - new_slabs["v_upper_plate_lat"].values
                    v_convergence_lon = new_slabs["v_lower_plate_lon"].values - new_slabs["v_upper_plate_lon"].values
                    v_convergence_mag = _numpy.sqrt(v_convergence_lat**2 + v_convergence_lon**2)

                    # Calculate convergence rates
                    v_convergence_lat = new_slabs["v_lower_plate_lat"].values - new_slabs["v_upper_plate_lat"].values
                    v_convergence_lon = new_slabs["v_lower_plate_lon"].values - new_slabs["v_upper_plate_lon"].values
                    v_convergence_mag = _numpy.sqrt(v_convergence_lat**2 + v_convergence_lon**2)

                    # Check convergence rates
                    if _numpy.max(abs(v_convergence_mag - old_slabs["v_convergence_mag"].values)) < 1e-2: # and _numpy.max(v_convergence_mag) < 25:
                        print(f"Convergence rates converged after {k} iterations")
                        break
                    else:
                        # Assign new values to latest slabs DataFrame
                        new_slabs["v_convergence_lat"], new_slabs["v_convergence_lon"] = utils_calc.mag_azi2lat_lon(v_convergence_mag, new_slabs.trench_normal_azimuth); new_slabs["v_convergence_mag"] = v_convergence_mag
                        
                        # Delecte old DataFrames
                        del old_plates, old_points, old_slabs
                        
                        # Overwrite DataFrames
                        old_plates = new_plates.copy(); old_points = new_points.copy(); old_slabs = new_slabs.copy()

                # Calculate residual of plate velocities
                v_upper_plate_residual[i,j] = _numpy.max(abs(new_slabs.v_upper_plate_mag - true_slabs.v_upper_plate_mag))
                print("upper_plate_residual: ", v_upper_plate_residual[i,j])
                v_lower_plate_residual[i,j] = _numpy.max(abs(new_slabs.v_lower_plate_mag - true_slabs.v_lower_plate_mag))
                print("lower_plate_residual: ", v_lower_plate_residual[i,j])
                v_convergence_residual[i,j] = _numpy.max(abs(new_slabs.v_convergence_mag - true_slabs.v_convergence_mag))
                print("convergence_rate_residual: ", v_convergence_residual[i,j])

        # Find the indices of the minimum value directly using _numpy.argmin
        opt_upper_plate_i, opt_upper_plate_j = _numpy.unravel_index(_numpy.argmin(v_upper_plate_residual), v_upper_plate_residual.shape)
        opt_upper_plate_visc = viscs[opt_upper_plate_i]
        opt_upper_plate_sp_const = sp_consts[opt_upper_plate_j]

        opt_lower_plate_i, opt_lower_plate_j = _numpy.unravel_index(_numpy.argmin(v_lower_plate_residual), v_lower_plate_residual.shape)
        opt_lower_plate_visc = viscs[opt_lower_plate_i]
        opt_lower_plate_sp_const = sp_consts[opt_lower_plate_j]

        opt_convergence_i, opt_convergence_j = _numpy.unravel_index(_numpy.argmin(v_convergence_residual), v_convergence_residual.shape)
        opt_convergence_visc = viscs[opt_convergence_i]
        opt_convergence_sp_const = sp_consts[opt_convergence_j]

        # Plot
        for i, j, visc, sp_const, residual in zip([opt_upper_plate_i, opt_lower_plate_i, opt_convergence_i], [opt_upper_plate_j, opt_lower_plate_j, opt_convergence_j], [opt_upper_plate_visc, opt_lower_plate_visc, opt_convergence_visc], [opt_upper_plate_sp_const, opt_lower_plate_sp_const, opt_convergence_sp_const], [v_upper_plate_residual, v_lower_plate_residual, v_convergence_residual]):
            if plot == True:
                fig, ax = plt.subplots(figsize=(15*self.constants.cm2in, 12*self.constants.cm2in))
                im = ax.imshow(residual, cmap="cmc.davos_r")#, vmin=-1.5, vmax=1.5)
                ax.set_yticks(_numpy.linspace(0, grid_size - 1, 5))
                ax.set_xticks(_numpy.linspace(0, grid_size - 1, 5))
                ax.set_xticklabels(["{:.2e}".format(visc) for visc in _numpy.linspace(visc_range[0], visc_range[1], 5)])
                ax.set_yticklabels(["{:.2f}".format(sp_const) for sp_const in _numpy.linspace(sp_consts.min(), sp_consts.max(), 5)])
                ax.set_xlabel("Mantle viscosity [Pa s]")
                ax.set_ylabel("Slab pull reduction factor")
                ax.scatter(j, i, marker="*", facecolor="none", edgecolor="k", s=30)
                fig.colorbar(im, label = "Residual velocity magnitude [cm/a]")
                plt.show()

            print(f"Optimal coefficients for ", ", ".join(new_plates.name.astype(str)), " plate(s), (PlateIDs: ", ", ".join(new_plates.plateID.astype(str)), ")")
            print("Minimum residual torque: {:.2e} cm/a".format(_numpy.amin(residual)))
            print("Optimum viscosity [Pa s]: {:.2e}".format(visc))
            print("Optimum Drag Coefficient [Pa s/m]: {:.2e}".format(visc / self.mech.La))
            print("Optimum Slab Pull constant: {:.2%}".format(sp_const))