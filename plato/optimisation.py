# Third-party libraries
import numpy as _numpy
import matplotlib.pyplot as plt

# Plato libraries
import plato.utils_data as utils_data
from .plate_torques import PlateTorques

class Optimisation():
    """
    A class to optimise the slab pull coefficient and mantle viscosity to match plate motions.

    :param plate_torques:   plate torques object
    :type plate_torques:    PlateTorques
    """    
    def __init__(
            self,
            plate_torques = None,
        ):
        """
        Constructor for the `Plot` class.
        """
        # Store the input data, if provided
        if isinstance(plate_torques, PlateTorques):
            self.settings = plate_torques.settings
            self.plates = plate_torques.plates
            self.slabs = plate_torques.slabs
            self.points = plate_torques.points
            self.grids = plate_torques.grids
            self.globe = plate_torques.globe
        
    def minimise_residual_torque(
            self,
            age = None, 
            case = None, 
            plateIDs = None, 
            grid_size = 500, 
            viscosity_range = [5e18, 5e20],
            plot = True,
            weight_by_area = True,
            minimum_plate_area = None
        ):
        """
        Function to find optimised coefficients to match plate motions using a grid search

        :param age:                     reconstruction age to optimise
        :type age:                      int, float
        :param case:                    case to optimise
        :type case:                     str
        :param plateIDs:                plate IDs to include in optimisation
        :type plateIDs:                 list of integers or None
        :param grid_size:               size of the grid to find optimal viscosity and slab pull coefficient
        :type grid_size:                int
        :param plot:                    whether or not to plot the grid
        :type plot:                     boolean
        :param weight_by_area:          whether or not to weight the residual torque by plate area
        :type weight_by_area:           boolean

        :return:                        The optimal slab pull coefficient, optimal viscosity, normalised residual torque, and indices of the optimal coefficients
        :rtype:                         float, float, numpy.ndarray, tuple
        """
        # Define age and case if not provided
        if age is None:
            age = self.settings.ages[0]

        if case is None:
            case = self.settings.cases[0]

        # Set range of viscosities
        viscs = _numpy.linspace(viscosity_range[0], viscosity_range[1], grid_size)

        # Set range of slab pull coefficients
        if self.settings.options[case]["Sediment subduction"]:
            # Range is smaller with sediment subduction
            sp_consts = _numpy.linspace(1e-5, 0.25, grid_size)
        else:
            sp_consts = _numpy.linspace(1e-5, 1., grid_size)

        # Create grids from ranges of viscosities and slab pull coefficients
        visc_grid, sp_const_grid = _numpy.meshgrid(viscs, sp_consts)
        ones_grid = _numpy.ones_like(visc_grid)

        # Filter plates
        _data = self.plates.data[age][case].copy()

        _plateIDs = utils_data.select_plateIDs(plateIDs, self.plates.data[age][case])

        if plateIDs is not None:
            _data = _data[_data["plateID"].isin(_plateIDs)]
            # _data = _data.reset _index(drop=True)

        print(f"_plateIDs length: {len(_plateIDs)}")
        print(f"_data indices: {_data.index}")
        # # Filter plates by minimum area
        # if minimum_plate_area is None:
        #     minimum_plate_area = self.settings.options[case]["Minimum plate area"]

        # _data = _data[_data["area"] > minimum_plate_area]
        # _data = _data.reset_index(drop=True)

        # Get total area
        total_area = _data["area"].sum()

        # Initialise dictionaries and arrays to store driving and residual torques
        # if age not in self.driving_torque:
        #     self.driving_torque[age] = {}
        # if age not in self.driving_torque_normalised:
        #     self.driving_torque_normalised[age] = {}
        # if age not in self.residual_torque:
        #     self.residual_torque[age] = {}
        # if age not in self.residual_torque_normalised:
        #     self.residual_torque_normalised[age] = {}
            
        driving_mag = _numpy.zeros_like(sp_const_grid); 
        # self.driving_torque_normalised[age][case] = _numpy.zeros_like(sp_const_grid)
        residual_mag = _numpy.zeros_like(sp_const_grid); 
        # self.residual_torque_normalised[age][case] = _numpy.zeros_like(sp_const_grid)

        # Initialise dictionaries to store optimal coefficients
        # if age not in self.opt_i:
        #     self.opt_i[age] = {}
        # if age not in self.opt_j:
        #     self.opt_j[age] = {}
        # if age not in self.opt_sp_const:
        #     self.opt_sp_const[age] = {}
        # if age not in self.opt_visc:
        #     self.opt_visc[age] = {}

        # Get torques
        for k, _plateID in enumerate(_plateIDs):
            if _plateID in _data.plateID.values:
                residual_x = _numpy.zeros_like(sp_const_grid); residual_y = _numpy.zeros_like(sp_const_grid); residual_z = _numpy.zeros_like(sp_const_grid)
                if self.settings.options[case]["Slab pull torque"] and "slab_pull_torque_x" in _data.columns:
                    residual_x -= _data.slab_pull_torque_x.iloc[k] * sp_const_grid / self.settings.options[case]["Slab pull constant"]
                    residual_y -= _data.slab_pull_torque_y.iloc[k] * sp_const_grid / self.settings.options[case]["Slab pull constant"]
                    residual_z -= _data.slab_pull_torque_z.iloc[k] * sp_const_grid / self.settings.options[case]["Slab pull constant"]

                # Add GPE torque
                if self.settings.options[case]["GPE torque"] and "GPE_torque_x" in _data.columns:
                    residual_x -= _data.GPE_torque_x.iloc[k] * ones_grid
                    residual_y -= _data.GPE_torque_y.iloc[k] * ones_grid
                    residual_z -= _data.GPE_torque_z.iloc[k] * ones_grid
                
                # Compute magnitude of driving torque
                if weight_by_area:
                    driving_mag += _numpy.sqrt(residual_x**2 + residual_y**2 + residual_z**2) * _data.area.iloc[k] / total_area
                else:
                    driving_mag += _numpy.sqrt(residual_x**2 + residual_y**2 + residual_z**2) / _data.area.iloc[k]

                # Add slab bend torque
                if self.settings.options[case]["Slab bend torque"] and "slab_bend_torque_x" in _data.columns:
                    residual_x -= _data.slab_bend_torque_x.iloc[k] * ones_grid
                    residual_y -= _data.slab_bend_torque_y.iloc[k] * ones_grid
                    residual_z -= _data.slab_bend_torque_z.iloc[k] * ones_grid

                # Add mantle drag torque
                if self.settings.options[case]["Mantle drag torque"] and "mantle_drag_torque_x" in _data.columns:
                    residual_x -= _data.mantle_drag_torque_x.iloc[k] * visc_grid / self.settings.options[case]["Mantle viscosity"]
                    residual_y -= _data.mantle_drag_torque_y.iloc[k] * visc_grid / self.settings.options[case]["Mantle viscosity"]
                    residual_z -= _data.mantle_drag_torque_z.iloc[k] * visc_grid / self.settings.options[case]["Mantle viscosity"]

                # Compute magnitude of residual
                if weight_by_area:
                    residual_mag += _numpy.sqrt(residual_x**2 + residual_y**2 + residual_z**2) * _data.area.iloc[k] / total_area
                else:
                    residual_mag += _numpy.sqrt(residual_x**2 + residual_y**2 + residual_z**2) / _data.area.iloc[k]
            
        # Divide residual by driving torque
        residual_mag_normalised = _numpy.log10(residual_mag / driving_mag)

        # Find the indices of the minimum value directly using _numpy.argmin
        opt_i, opt_j = _numpy.unravel_index(_numpy.argmin(residual_mag), residual_mag.shape)
        opt_visc = visc_grid[opt_i, opt_j]
        opt_sp_const = sp_const_grid[opt_i, opt_j]

        # Plot
        if plot == True:
            fig, ax = plt.subplots(figsize=(15, 12))
            im = ax.imshow(residual_mag_normalised, cmap="cmc.lapaz_r", vmin=-1.5, vmax=1.5)
            ax.set_yticks(_numpy.linspace(0, grid_size - 1, 5))
            ax.set_xticks(_numpy.linspace(0, grid_size - 1, 5))
            ax.set_xticklabels(["{:.2e}".format(visc) for visc in _numpy.linspace(viscosity_range[0], viscosity_range[1], 5)])
            ax.set_yticklabels(["{:.2f}".format(sp_const) for sp_const in _numpy.linspace(sp_consts.min(), sp_consts.max(), 5)])
            ax.set_xlabel("Mantle viscosity [Pa s]")
            ax.set_ylabel("Slab pull reduction factor")
            ax.scatter(opt_j, opt_i, marker="*", facecolor="none", edgecolor="k", s=30)  # Adjust the marker style and size as needed
            fig.colorbar(im, label = "Log(residual torque/driving torque)")
            plt.show()

        # Print results
        print(f"Optimal coefficients for ", ", ".join(_data.name.astype(str)), " plate(s), (PlateIDs: ", ", ".join(_data.plateID.astype(str)), ")")
        print("Minimum residual torque: {:.2%} of driving torque".format(10**(_numpy.amin(residual_mag_normalised))))
        print("Optimum viscosity [Pa s]: {:.2e}".format(opt_visc))
        print("Optimum Drag Coefficient [Pa s/m]: {:.2e}".format(opt_visc / self.settings.mech.La))
        print("Optimum Slab Pull constant: {:.2%}".format(opt_sp_const))

        # return self.opt_sp_const[age][case], self.opt_visc[age][case], self.residual_torque_normalised[age][case], (self.opt_i[age][case], self.opt_j[age][case])
    
    def find_slab_pull_coefficient(
            self,
            age = None, 
            case = None, 
            plateIDs = None, 
            grid_size = 500, 
            viscosity = 1e19, 
            plot = False, 
        ):
        """
        Function to find optimised slab pull coefficient for a given (set of) plates using a grid search.

        :param age:                     reconstruction age to optimise
        :type age:                      int, float
        :param case:                    case to optimise
        :type case:                     str
        :param plateIDs:                plate IDs to include in optimisation
        :type plateIDs:                 list of integers or None
        :param grid_size:               size of the grid to find optimal slab pull coefficient
        :type grid_size:                int
        :param plot:                    whether or not to plot the grid
        :type plot:                     boolean
        
        :return:                        The optimal slab pull coefficient
        :rtype:                         float
        """
        # Set age if not provided
        if age is None:
            age = self.settings.ages[0]

        # Set case if not provided
        if case is None:
            case = self.settings.cases[0]

        # Generate range of possible slab pull coefficients
        sp_consts = _numpy.linspace(1e-5,1,grid_size)
        ones = _numpy.ones_like(sp_consts)

        # Filter plates
        selected_plates = self.plates[age][case].copy()
        if plates_of_interest:
            selected_plates = selected_plates[selected_plates["plateID"].isin(plateIDs)]
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

            if self.settings.options[case]["Slab pull torque"] and "slab_pull_torque_x" in selected_plates.columns:
                residual_x -= selected_plates.slab_pull_torque_x.iloc[k] * sp_consts / self.settings.options[case]["Slab pull constant"]
                residual_y -= selected_plates.slab_pull_torque_y.iloc[k] * sp_consts / self.settings.options[case]["Slab pull constant"]
                residual_z -= selected_plates.slab_pull_torque_z.iloc[k] * sp_consts / self.settings.options[case]["Slab pull constant"]

            # Add GPE torque
            if self.settings.options[case]["GPE torque"] and "GPE_torque_x" in selected_plates.columns:
                residual_x -= selected_plates.GPE_torque_x.iloc[k] * ones
                residual_y -= selected_plates.GPE_torque_y.iloc[k] * ones
                residual_z -= selected_plates.GPE_torque_z.iloc[k] * ones

            # Compute magnitude of driving torque
            driving_mag = _numpy.sqrt(residual_x**2 + residual_y**2 + residual_z**2)
            
            # Add slab bend torque
            if self.settings.options[case]["Slab bend torque"] and "slab_bend_torque_x" in selected_plates.columns:
                residual_x -= selected_plates.slab_bend_torque_x.iloc[k] * ones
                residual_y -= selected_plates.slab_bend_torque_y.iloc[k] * ones
                residual_z -= selected_plates.slab_bend_torque_z.iloc[k] * ones

            # Add mantle drag torque
            if self.settings.options[case]["Mantle drag torque"] and "mantle_drag_torque_x" in selected_plates.columns:
                residual_x -= selected_plates.mantle_drag_torque_x.iloc[k] * viscosity / self.mech.La / self.settings.options[case]["Mantle viscosity"]
                residual_y -= selected_plates.mantle_drag_torque_y.iloc[k] * viscosity / self.mech.La / self.settings.options[case]["Mantle viscosity"]
                residual_z -= selected_plates.mantle_drag_torque_z.iloc[k] * viscosity / self.mech.La / self.settings.options[case]["Mantle viscosity"]

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