# Standard libraries
import logging
import warnings
from typing import Dict, List, Optional, Union

# Third-party libraries
import numpy as _numpy
import matplotlib.pyplot as plt
from tqdm import tqdm as _tqdm

# Plato libraries
from . import utils_data, utils_calc
from .plate_torques import PlateTorques

class Optimisation():
    """
    A class to optimise the slab pull coefficient and mantle viscosity to match plate motions.

    :param plate_torques:   plate torques object
    :type plate_torques:    PlateTorques
    """    
    def __init__(
            self,
            plate_torques: Optional[PlateTorques] = None,
        ):
        """
        Constructor for the `Optimisation` class.
        """
        # Store the input data, if provided
        if isinstance(plate_torques, PlateTorques):
            # Store plate torques object
            self.plate_torques = plate_torques

            # Create shortcuts to the settings, plates, slabs, points, grids, and globe objects, and the ages and cases
            self.settings = plate_torques.settings
            self.plates = plate_torques.plates
            self.slabs = plate_torques.slabs
            self.points = plate_torques.points
            self.grids = plate_torques.grids
            self.globe = plate_torques.globe

            # Set shortcut to ages, cases and options
            self.ages = self.settings.ages
            self.cases = self.settings.cases
            self.options = self.settings.options

    def minimise_residual_torque(
            self,
            ages: Union[int, float] = None, 
            cases: str = None, 
            plateIDs: Optional[Union[int, float, List[Union[int, float]], _numpy.ndarray]] = None,
            grid_size: int = 500, 
            viscosity_range: List[Union[int, float]] = [5e18, 5e20],
            plot: bool = True,
            weight_by_area: bool = True,
            minimum_plate_area: bool = None,
        ):
        """
        Function to find optimised coefficients to match plate motions using a grid search.
        The optimal coefficients are those that minimise the residual torque normalised by the driving torque.
        If more than one plate is provided, the normalised residual torque is weighted by the plate area.

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

        NOTE: Lateral viscosity variations need to be introduced along a third dimension.
        """
        # Define ages if not provided
        _ages = utils_data.select_ages(ages, self.settings.ages)
        
        # Define cases if not provided
        _cases = utils_data.select_cases(cases, self.settings.point_cases)

        # Set range of viscosities
        viscs = _numpy.linspace(viscosity_range[0], viscosity_range[1], grid_size)
        
        # Loop through ages
        for _age in _ages:
            for _case in _cases:

                # Set range of slab pull coefficients
                if self.settings.options[_case]["Sediment subduction"]:
                    # Range is smaller with sediment subduction
                    sp_consts = _numpy.linspace(1e-5, 0.25, grid_size)
                else:
                    sp_consts = _numpy.linspace(1e-5, 1., grid_size)

                # Create grids from ranges of viscosities and slab pull coefficients
                visc_grid, sp_const_grid = _numpy.meshgrid(viscs, sp_consts)
                ones_grid = _numpy.ones_like(visc_grid)

                # Filter plates
                _data = self.plates.data[_age][_case]

                _plateIDs = utils_data.select_plateIDs(plateIDs, self.plates.data[_age][_case].plateID)

                if plateIDs is not None:
                    _data = _data[_data["plateID"].isin(_plateIDs)]

                if minimum_plate_area is not None:
                    _data = _data[_data["area"] > minimum_plate_area]

                # Get total area
                total_area = _data["area"].sum()
                    
                driving_mag = _numpy.zeros_like(sp_const_grid)
                residual_mag = _numpy.zeros_like(sp_const_grid)

                # Get torques
                for k, _ in enumerate(_data.plateID):
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")

                        residual_x = _numpy.zeros_like(sp_const_grid); residual_y = _numpy.zeros_like(sp_const_grid); residual_z = _numpy.zeros_like(sp_const_grid)
                        if self.settings.options[_case]["Slab pull torque"] and "slab_pull_torque_x" in _data.columns:
                            residual_x -= _data.slab_pull_torque_x.iloc[k] * sp_const_grid / self.settings.options[_case]["Slab pull constant"]
                            residual_y -= _data.slab_pull_torque_y.iloc[k] * sp_const_grid / self.settings.options[_case]["Slab pull constant"]
                            residual_z -= _data.slab_pull_torque_z.iloc[k] * sp_const_grid / self.settings.options[_case]["Slab pull constant"]

                        # Add GPE torque
                        if self.settings.options[_case]["GPE torque"] and "GPE_torque_x" in _data.columns:
                            residual_x -= _data.GPE_torque_x.iloc[k] * ones_grid
                            residual_y -= _data.GPE_torque_y.iloc[k] * ones_grid
                            residual_z -= _data.GPE_torque_z.iloc[k] * ones_grid
                        
                        # Compute magnitude of driving torque
                        if weight_by_area:
                            driving_mag += _numpy.sqrt(residual_x**2 + residual_y**2 + residual_z**2) * _data.area.iloc[k] / total_area
                        else:
                            driving_mag += _numpy.sqrt(residual_x**2 + residual_y**2 + residual_z**2) / _data.area.iloc[k]

                        # Add slab bend torque
                        if self.settings.options[_case]["Slab bend torque"] and "slab_bend_torque_x" in _data.columns:
                            residual_x -= _data.slab_bend_torque_x.iloc[k] * ones_grid
                            residual_y -= _data.slab_bend_torque_y.iloc[k] * ones_grid
                            residual_z -= _data.slab_bend_torque_z.iloc[k] * ones_grid

                        # Add mantle drag torque
                        if self.settings.options[_case]["Mantle drag torque"] and "mantle_drag_torque_x" in _data.columns:
                            residual_x -= _data.mantle_drag_torque_x.iloc[k] * visc_grid / self.settings.options[_case]["Mantle viscosity"]
                            residual_y -= _data.mantle_drag_torque_y.iloc[k] * visc_grid / self.settings.options[_case]["Mantle viscosity"]
                            residual_z -= _data.mantle_drag_torque_z.iloc[k] * visc_grid / self.settings.options[_case]["Mantle viscosity"]

                        # Compute magnitude of residual
                        if weight_by_area:
                            residual_mag += _numpy.sqrt(residual_x**2 + residual_y**2 + residual_z**2) * _data.area.iloc[k] / total_area
                        else:
                            residual_mag += _numpy.sqrt(residual_x**2 + residual_y**2 + residual_z**2) / _data.area.iloc[k]
                
                # Divide residual by driving torque
                residual_mag_normalised = _numpy.log10(residual_mag / driving_mag)

                # Find the indices of the minimum value
                opt_i = _numpy.argmin(_numpy.min(residual_mag_normalised, axis=1))
                opt_j = _numpy.argmin(_numpy.min(residual_mag_normalised, axis=0))
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
    
    def minimise_residual_torque_v2(
            self,
            ages = None, 
            cases = None, 
            plateIDs = None, 
            grid_size = 500, 
            lateral_viscosity_variation_range = [1, 100],
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

        NOTE: Slab suction will be removed shortly and optimisation of lateral viscosity variation will be moved to the original `minimise_residual_torque` function.
        """
        # Define ages if not provided
        _ages = utils_data.select_ages(ages, self.settings.ages)
        
        # Define cases if not provided
        _cases = utils_data.select_cases(cases, self.settings.point_cases)

        # Set range of viscosities
        viscs = _numpy.linspace(lateral_viscosity_variation_range[0], lateral_viscosity_variation_range[1], grid_size)
        
        # Loop through ages
        for _age in _ages:
            for _case in _cases:

                # Set range of slab pull coefficients
                if self.settings.options[_case]["Sediment subduction"]:
                    # Range is smaller with sediment subduction
                    sp_consts = _numpy.linspace(1e-5, 0.25, grid_size)
                else:
                    sp_consts = _numpy.linspace(1e-5, 1., grid_size)

                # Create grids from ranges of viscosities and slab pull coefficients
                visc_grid, sp_const_grid = _numpy.meshgrid(viscs, sp_consts)
                ones_grid = _numpy.ones_like(visc_grid)

                # Filter plates
                _data = self.plates.data[_age][_case]

                _plateIDs = utils_data.select_plateIDs(plateIDs, self.plates.data[_age][_case].plateID)

                if plateIDs is not None:
                    _data = _data[_data["plateID"].isin(_plateIDs)]

                if minimum_plate_area is not None:
                    _data = _data[_data["area"] > minimum_plate_area]

                # Get total area
                total_area = _data["area"].sum()
                    
                driving_mag = _numpy.zeros_like(sp_const_grid)
                residual_mag = _numpy.zeros_like(sp_const_grid)

                # Get torques
                for k, _ in enumerate(_data.plateID):
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")

                        residual_x = _numpy.zeros_like(sp_const_grid); residual_y = _numpy.zeros_like(sp_const_grid); residual_z = _numpy.zeros_like(sp_const_grid)
                        if self.settings.options[_case]["Slab pull torque"] and "slab_pull_torque_x" in _data.columns:
                            residual_x -= _data.slab_pull_torque_x.iloc[k] * ones_grid
                            residual_y -= _data.slab_pull_torque_y.iloc[k] * ones_grid
                            residual_z -= _data.slab_pull_torque_z.iloc[k] * ones_grid

                        # Add slab suction torque
                        if self.settings.options[_case]["Slab suction torque"] and "slab_suction_torque_x" in _data.columns:
                            residual_x -= _data.slab_suction_torque_x.iloc[k] * sp_const_grid / self.settings.options[_case]["Slab suction constant"]
                            residual_y -= _data.slab_suction_torque_y.iloc[k] * sp_const_grid / self.settings.options[_case]["Slab suction constant"]
                            residual_z -= _data.slab_suction_torque_z.iloc[k] * sp_const_grid / self.settings.options[_case]["Slab suction constant"]

                        # Add GPE torque
                        if self.settings.options[_case]["GPE torque"] and "GPE_torque_x" in _data.columns:
                            residual_x -= _data.GPE_torque_x.iloc[k] * ones_grid
                            residual_y -= _data.GPE_torque_y.iloc[k] * ones_grid
                            residual_z -= _data.GPE_torque_z.iloc[k] * ones_grid
                        
                        # Compute magnitude of driving torque
                        if weight_by_area:
                            driving_mag += _numpy.sqrt(residual_x**2 + residual_y**2 + residual_z**2) * _data.area.iloc[k] / total_area
                        else:
                            driving_mag += _numpy.sqrt(residual_x**2 + residual_y**2 + residual_z**2) / _data.area.iloc[k]

                        # Add slab bend torque
                        if self.settings.options[_case]["Slab bend torque"] and "slab_bend_torque_x" in _data.columns:
                            residual_x -= _data.slab_bend_torque_x.iloc[k] * ones_grid
                            residual_y -= _data.slab_bend_torque_y.iloc[k] * ones_grid
                            residual_z -= _data.slab_bend_torque_z.iloc[k] * ones_grid

                        # Add mantle drag torque
                        if self.settings.options[_case]["Mantle drag torque"] and "mantle_drag_torque_x" in _data.columns:
                            residual_x -= _data.mantle_drag_torque_x.iloc[k] * _data.continental_fraction.iloc[k] * visc_grid / self.settings.options[_case]["Lateral viscosity variation"] + \
                                _data.mantle_drag_torque_x.iloc[k] * (1-_data.continental_fraction.iloc[k])
                            residual_y -= _data.mantle_drag_torque_y.iloc[k] * _data.continental_fraction.iloc[k] * visc_grid / self.settings.options[_case]["Lateral viscosity variation"] + \
                                _data.mantle_drag_torque_y.iloc[k] * (1-_data.continental_fraction.iloc[k])
                            residual_z -= _data.mantle_drag_torque_z.iloc[k] * _data.continental_fraction.iloc[k] * visc_grid / self.settings.options[_case]["Lateral viscosity variation"] + \
                                _data.mantle_drag_torque_z.iloc[k] * (1-_data.continental_fraction.iloc[k])
                        
                        # Compute magnitude of residual
                        if weight_by_area:
                            residual_mag += _numpy.sqrt(residual_x**2 + residual_y**2 + residual_z**2) * _data.area.iloc[k] / total_area
                        else:
                            residual_mag += _numpy.sqrt(residual_x**2 + residual_y**2 + residual_z**2) / _data.area.iloc[k]                        
                
                # Divide residual by driving torque
                residual_mag_normalised = _numpy.log10(residual_mag / driving_mag)

                # Find the indices of the minimum value directly using _numpy.argmin
                opt_i = _numpy.argmin(_numpy.min(residual_mag_normalised, axis=1))
                opt_j = _numpy.argmin(_numpy.min(residual_mag_normalised, axis=0))
                # opt_i, opt_j = _numpy.unravel_index(_numpy.argmin(residual_mag), residual_mag.shape)
                opt_visc = visc_grid[opt_i, opt_j]
                opt_sp_const = sp_const_grid[opt_i, opt_j]

                # Plot
                if plot == True:
                    fig, ax = plt.subplots(figsize=(15, 12))
                    im = ax.imshow(residual_mag_normalised, cmap="cmc.lapaz_r", vmin=-1.5, vmax=1.5)
                    ax.set_yticks(_numpy.linspace(0, grid_size - 1, 5))
                    ax.set_xticks(_numpy.linspace(0, grid_size - 1, 5))
                    ax.set_xticklabels(["{:.2e}".format(visc) for visc in _numpy.linspace(lateral_viscosity_variation_range[0], lateral_viscosity_variation_range[1], 5)])
                    ax.set_yticklabels(["{:.2f}".format(sp_const) for sp_const in _numpy.linspace(sp_consts.min(), sp_consts.max(), 5)])
                    ax.set_xlabel("Lateral viscosity variation")
                    ax.set_ylabel("Slab suction constant")
                    ax.scatter(opt_j, opt_i, marker="*", facecolor="none", edgecolor="k", s=30)  # Adjust the marker style and size as needed
                    fig.colorbar(im, label = "Log(residual torque/driving torque)")
                    plt.show()

                # Print results
                print(f"Optimal coefficients for ", ", ".join(_data.name.astype(str)), " plate(s), (PlateIDs: ", ", ".join(_data.plateID.astype(str)), ")")
                print("Minimum residual torque: {:.2%} of driving torque".format(10**(_numpy.amin(residual_mag_normalised))))
                print("Optimum lateral viscosity variation: {:.2e}".format(opt_visc))
                print("Optimum slab suction constant: {:.2%}".format(opt_sp_const))

        return residual_mag_normalised, (opt_visc, opt_sp_const), (opt_i, opt_j)

    # def minimise_residual_torque_v2(
    #         self,
    #         age = None, 
    #         case = None, 
    #         plateIDs = None, 
    #         grid_size = 500, 
    #         lvv_range = [1, 100],
    #         plot = True,
    #         weight_by_area = True,
    #         minimum_plate_area = None
    #     ):
    #     """
    #     Function to find optimised coefficients to match plate motions using a grid search

    #     :param age:                     reconstruction age to optimise
    #     :type age:                      int, float
    #     :param case:                    case to optimise
    #     :type case:                     str
    #     :param plateIDs:                plate IDs to include in optimisation
    #     :type plateIDs:                 list of integers or None
    #     :param grid_size:               size of the grid to find optimal viscosity and slab pull coefficient
    #     :type grid_size:                int
    #     :param plot:                    whether or not to plot the grid
    #     :type plot:                     boolean
    #     :param weight_by_area:          whether or not to weight the residual torque by plate area
    #     :type weight_by_area:           boolean

    #     :return:                        The optimal slab pull coefficient, optimal viscosity, normalised residual torque, and indices of the optimal coefficients
    #     :rtype:                         float, float, numpy.ndarray, tuple
    #     """
    #     # Define age and case if not provided
    #     if age is None:
    #         age = self.settings.ages[0]

    #     if case is None:
    #         case = self.settings.cases[0]

    #     # Set range of viscosities
    #     viscs = _numpy.linspace(viscosity_range[0], viscosity_range[1], grid_size)

    #     # Set range of slab pull constants
    #     if self.settings.options[case]["Sediment subduction"]:
    #         # Range is smaller with sediment subduction
    #         sp_consts = _numpy.linspace(1e-5, 0.25, grid_size)
    #     else:
    #         sp_consts = _numpy.linspace(1e-5, 1., grid_size)

    #     # Set range of slab suction constants
    #     ss_consts = _numpy.linspace(1e-5, 1., grid_size)

    #     # Set range of lateral viscosity variations
    #     lvv_range = _numpy.linspace(1e-5, 100., grid_size)

    #     # Create grids from ranges of viscosities and slab pull coefficients
    #     visc_grid, lvv_grid, sp_const_grid, ss_cont_grid = _numpy.meshgrid(viscs, lvv_range, sp_consts, ss_consts)
    #     ones_grid = _numpy.ones_like(visc_grid)

    #     # Select plateIDs
    #     _plateIDs = utils_data.select_plateIDs(plateIDs, self.plates.data[age][case].plateID)

    #     # Filter plates
    #     _data = self.plates.data[age][case]


    #     if plateIDs is not None:
    #         _data = _data[_data["plateID"].isin(_plateIDs)]

    #     if minimum_plate_area is not None:
    #         _data = _data[_data["area"] > minimum_plate_area]

    #     # Get total area
    #     total_area = _data["area"].sum()
            
    #     driving_mag = _numpy.zeros_like(sp_const_grid); 
    #     residual_mag = _numpy.zeros_like(sp_const_grid); 

    #     # Get torques
    #     for k, _ in enumerate(_data.plateID):
    #         with warnings.catch_warnings():
    #             warnings.simplefilter("ignore")

    #             residual_x = _numpy.zeros_like(sp_const_grid); residual_y = _numpy.zeros_like(sp_const_grid); residual_z = _numpy.zeros_like(sp_const_grid)

    #             # Add slab pull torque
    #             if self.settings.options[case]["Slab pull torque"] and "slab_pull_torque_x" in _data.columns:
    #                 residual_x -= _data.slab_pull_torque_x.iloc[k] * sp_const_grid / self.settings.options[case]["Slab pull constant"]
    #                 residual_y -= _data.slab_pull_torque_y.iloc[k] * sp_const_grid / self.settings.options[case]["Slab pull constant"]
    #                 residual_z -= _data.slab_pull_torque_z.iloc[k] * sp_const_grid / self.settings.options[case]["Slab pull constant"]

    #             # Add slab suction torque
    #             if self.settings.options[case]["Slab suction torque"] and "slab_suction_torque_x" in _data.columns:
    #                 # print("Adding slab suction torque")
    #                 residual_x -= _data.slab_suction_torque_x.iloc[k] * ss_cont_grid * sp_const_grid / (self.settings.options[case]["Slab suction constant"] * self.settings.options[case]["Slab pull constant"])
    #                 residual_y -= _data.slab_suction_torque_y.iloc[k] * ss_cont_grid * sp_const_grid / (self.settings.options[case]["Slab suction constant"] * self.settings.options[case]["Slab pull constant"])
    #                 residual_z -= _data.slab_suction_torque_z.iloc[k] * ss_cont_grid * sp_const_grid / (self.settings.options[case]["Slab suction constant"] * self.settings.options[case]["Slab pull constant"])

    #                 # print(_data.slab_suction_torque_x.iloc[k] * ss_cont_grid * sp_const_grid / (self.settings.options[case]["Slab suction constant"] * self.settings.options[case]["Slab pull constant"]))

    #             # Add GPE torque
    #             if self.settings.options[case]["GPE torque"] and "GPE_torque_x" in _data.columns:
    #                 residual_x -= _data.GPE_torque_x.iloc[k] * ones_grid
    #                 residual_y -= _data.GPE_torque_y.iloc[k] * ones_grid
    #                 residual_z -= _data.GPE_torque_z.iloc[k] * ones_grid
                
    #             # Compute magnitude of driving torque
    #             if weight_by_area:
    #                 driving_mag += _numpy.sqrt(residual_x**2 + residual_y**2 + residual_z**2) * _data.area.iloc[k] / total_area
    #             else:
    #                 driving_mag += _numpy.sqrt(residual_x**2 + residual_y**2 + residual_z**2) / _data.area.iloc[k]

    #             # Add slab bend torque
    #             if self.settings.options[case]["Slab bend torque"] and "slab_bend_torque_x" in _data.columns:
    #                 residual_x -= _data.slab_bend_torque_x.iloc[k] * ones_grid
    #                 residual_y -= _data.slab_bend_torque_y.iloc[k] * ones_grid
    #                 residual_z -= _data.slab_bend_torque_z.iloc[k] * ones_grid

    #             # Add mantle drag torque
    #             if self.settings.options[case]["Mantle drag torque"] and "mantle_drag_torque_x" in _data.columns:
    #                 if self.settings.options[case]["Continental keels"]:
    #                     # print("Adding mantle drag torque with continental keels")
    #                     # print(1 + lvv_grid * (_data.continental_fraction.iloc[k] - 1))
    #                     residual_x -= _data.mantle_drag_torque_x.iloc[k] * visc_grid * (1 + lvv_grid * (_data.continental_fraction.iloc[k] - 1)) / \
    #                         (self.settings.options[case]["Mantle viscosity"] * (1 + lvv_grid * (_data.continental_fraction.iloc[k] - 1)))
    #                     residual_y -= _data.mantle_drag_torque_y.iloc[k] * visc_grid * (1 + lvv_grid * (_data.continental_fraction.iloc[k] - 1)) / \
    #                         (self.settings.options[case]["Mantle viscosity"] * (1 + lvv_grid * (_data.continental_fraction.iloc[k] - 1)))
    #                     residual_z -= _data.mantle_drag_torque_z.iloc[k] * visc_grid * (1 + lvv_grid * (_data.continental_fraction.iloc[k] - 1)) / \
    #                         (self.settings.options[case]["Mantle viscosity"] * (1 + lvv_grid * (_data.continental_fraction.iloc[k] - 1)))
    #                 else:
    #                     residual_x -= _data.mantle_drag_torque_x.iloc[k] * visc_grid / self.settings.options[case]["Mantle viscosity"]
    #                     residual_y -= _data.mantle_drag_torque_y.iloc[k] * visc_grid / self.settings.options[case]["Mantle viscosity"]
    #                     residual_z -= _data.mantle_drag_torque_z.iloc[k] * visc_grid / self.settings.options[case]["Mantle viscosity"]

    #             # Compute magnitude of residual
    #             if weight_by_area:
    #                 residual_mag += _numpy.sqrt(residual_x**2 + residual_y**2 + residual_z**2) * _data.area.iloc[k] / total_area
    #             else:
    #                 residual_mag += _numpy.sqrt(residual_x**2 + residual_y**2 + residual_z**2) / _data.area.iloc[k]
        
    #     # Divide residual by driving torque
    #     residual_mag_normalised = _numpy.log10(residual_mag / driving_mag)

    #     # Find the indices of the minimum value directly using _numpy.argmin
    #     opt_indices = _numpy.unravel_index(_numpy.argmin(residual_mag), residual_mag.shape)
    #     opt_visc = visc_grid[opt_indices]; opt_lvv = lvv_grid[opt_indices]
    #     opt_sp_const = sp_const_grid[opt_indices]; opt_ss_cont = ss_cont_grid[opt_indices]

    #     # Assign optimal values to last entry in arrays
    #     # self.opt_sp_const[age][case][-1] = opt_sp_const
    #     # self.opt_visc[age][case][-1] = opt_visc

    #     # Plot
    #     if plot == True:
    #         fig, ax = plt.subplots(figsize=(15, 12))
    #         im = ax.imshow(residual_mag_normalised, cmap="cmc.lapaz_r", vmin=-1.5, vmax=1.5)
    #         ax.set_yticks(_numpy.linspace(0, grid_size - 1, 5))
    #         ax.set_xticks(_numpy.linspace(0, grid_size - 1, 5))
    #         ax.set_xticklabels(["{:.2e}".format(visc) for visc in _numpy.linspace(viscosity_range[0], viscosity_range[1], 5)])
    #         ax.set_yticklabels(["{:.2f}".format(sp_const) for sp_const in _numpy.linspace(sp_consts.min(), sp_consts.max(), 5)])
    #         ax.set_xlabel("Mantle viscosity [Pa s]")
    #         ax.set_ylabel("Slab pull reduction factor")
    #         ax.scatter(opt_j, opt_i, marker="*", facecolor="none", edgecolor="k", s=30)  # Adjust the marker style and size as needed
    #         fig.colorbar(im, label = "Log(residual torque/driving torque)")
    #         plt.show()

    #     # Print results
    #     print(f"Optimal coefficients for ", ", ".join(_data.name.astype(str)), " plate(s), (PlateIDs: ", ", ".join(_data.plateID.astype(str)), ")")
    #     print("Minimum residual torque: {:.2%} of driving torque".format(10**(_numpy.amin(residual_mag_normalised))))
    #     print("Optimum viscosity [Pa s]: {:.2e}".format(opt_visc))
    #     print("Optimum lateral viscosity Variation: {:.2f}".format(opt_lvv))
    #     print("Optimum drag coefficient [Pa s/m]: {:.2e}".format(opt_visc / self.settings.mech.La))
    #     print("Optimum slab pull constant: {:.2%}".format(opt_sp_const))
    #     print("Optimum slab suction constant: {:.2%}".format(opt_ss_cont))

    #     return residual_mag_normalised, (opt_visc, opt_lvv, opt_sp_const, opt_ss_cont), opt_indices
    
    def optimise_slab_pull_coefficient(
            self,
            age = None, 
            cases = None, 
            plateIDs = None, 
            grid_size = 500, 
            viscosity = 1.23e20, 
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
        # Select ages, if not provided
        _ages = utils_data.select_ages(age, self.settings.ages)

        # Select cases, if not provided
        _cases = utils_data.select_cases(cases, self.settings.reconstructed_cases)

        for _age in _tqdm(_ages, desc="Optimising slab pull coefficient"):
            with warnings.catch_warnings():

                warnings.simplefilter("ignore")
                for _case in _cases:
                    # Generate range of possible slab pull coefficients
                    sp_consts = _numpy.linspace(1e-5,1,grid_size)
                    ones = _numpy.ones_like(sp_consts)

                    # Filter plates
                    _data = self.plates.data[_age][_case].copy()

                    _plateIDs = utils_data.select_plateIDs(plateIDs, _data.plateID)

                    if plateIDs is not None:
                        _data = _data[_data["plateID"].isin(_plateIDs)]

                    if _data.empty:
                        return _numpy.nan, _numpy.nan, _numpy.nan
                
                    # Loop through plates
                    for k, _plateID in enumerate(_data.plateID):
                        residual_x = _numpy.zeros_like(sp_consts)
                        residual_y = _numpy.zeros_like(sp_consts)
                        residual_z = _numpy.zeros_like(sp_consts)

                        if self.settings.options[_case]["Slab pull torque"] and "slab_pull_torque_x" in _data.columns:
                            residual_x -= _data.slab_pull_torque_x.iloc[k] * sp_consts / self.settings.options[_case]["Slab pull constant"]
                            residual_y -= _data.slab_pull_torque_y.iloc[k] * sp_consts / self.settings.options[_case]["Slab pull constant"]
                            residual_z -= _data.slab_pull_torque_z.iloc[k] * sp_consts / self.settings.options[_case]["Slab pull constant"]

                        # Add GPE torque
                        if self.settings.options[_case]["GPE torque"] and "GPE_torque_x" in _data.columns:
                            residual_x -= _data.GPE_torque_x.iloc[k] * ones
                            residual_y -= _data.GPE_torque_y.iloc[k] * ones
                            residual_z -= _data.GPE_torque_z.iloc[k] * ones

                        # Compute magnitude of driving torque
                        driving_mag = _numpy.sqrt(residual_x**2 + residual_y**2 + residual_z**2)
                        
                        # Add slab bend torque
                        if self.settings.options[_case]["Slab bend torque"] and "slab_bend_torque_x" in _data.columns:
                            residual_x -= _data.slab_bend_torque_x.iloc[k] * ones
                            residual_y -= _data.slab_bend_torque_y.iloc[k] * ones
                            residual_z -= _data.slab_bend_torque_z.iloc[k] * ones

                        # Add mantle drag torque
                        if self.settings.options[_case]["Mantle drag torque"] and "mantle_drag_torque_x" in _data.columns:
                            residual_x -= _data.mantle_drag_torque_x.iloc[k] * viscosity / self.settings.options[_case]["Mantle viscosity"]
                            residual_y -= _data.mantle_drag_torque_y.iloc[k] * viscosity / self.settings.options[_case]["Mantle viscosity"]
                            residual_z -= _data.mantle_drag_torque_z.iloc[k] * viscosity / self.settings.options[_case]["Mantle viscosity"]

                        # Compute magnitude of residual
                        residual_mag = _numpy.sqrt(residual_x**2 + residual_y**2 + residual_z**2)

                        if plot:
                            fig, ax = plt.subplots(figsize=(10,10))
                            p = ax.plot(residual_mag/driving_mag)
                            ax.semilogy()
                            ax.set_xticks(_numpy.linspace(0, grid_size - 1, 5))
                            ax.set_xticklabels(["{:.2f}".format(sp_const) for sp_const in _numpy.linspace(sp_consts.min(), sp_consts.max(), 5)])
                            ax.set_ylim([10**-1.5, 10**1.5])
                            ax.set_xlim([0, grid_size])
                            ax.set_ylabel("Normalised residual torque")
                            ax.set_xlabel("Slab pull reduction factor")
                            plt.show()
                        
                        # Find optimal slab pull coefficient
                        opt_sp_const = sp_consts[_numpy.argmin(residual_mag/driving_mag)]

                        # Store optimal slab pull coefficient
                        mask = self.slabs.data[_age][_case]["lower_plateID"] == _plateID
                        self.slabs.data[_age][_case].loc[mask, "slab_pull_constant"] = opt_sp_const

        # Recalculate all the relevant torques
        self.plate_torques.calculate_slab_pull_torque(ages=_ages, cases=_cases, plateIDs=plateIDs, PROGRESS_BAR=False)
        self.plate_torques.calculate_driving_torque(ages=_ages, cases=_cases, plateIDs=plateIDs, PROGRESS_BAR=False)
        self.plate_torques.calculate_residual_torque(ages=_ages, cases=_cases, plateIDs=plateIDs, PROGRESS_BAR=False)

    def invert_residual_torque(
            self,
            ages = None, 
            cases = None, 
            plateIDs = None, 
            parameter = "Slab pull constant",
            vmin = -13.0, 
            vmax = -7.0,
            step = .1, 
            plot = False, 
        ):
        """
        Function to find optimised slab pull constant by projecting the residual torque onto the subduction zones.
        The function loops through all the unique combinations of ages and cases and optimises the slab pull coefficient for each plate.
        The model space for the slab pull constant is explored using an iterative approach with an adaptive step size.

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

        NOTE: This function should be optimised in several places.
        """
        # Raise error if parameter is not slab pull coefficient or mantle viscosity
        if parameter not in ["Slab pull constant", "Mantle viscosity"]:
            raise ValueError("Free parameter must be 'Slab pull constant' or 'Mantle viscosity")
        
        # Select ages, if not provided
        _ages = utils_data.select_ages(ages, self.settings.ages)

        # Select cases, if not provided
        _cases = utils_data.select_cases(cases, self.settings.reconstructed_cases)

        # Define constants
        constants = _numpy.arange(vmin, vmax, step)

        # Initialise dictionaries to store minimum residual torque and optimal constants
        driving_torque_opt_stack = {_age: {_case: None for _case in _cases} for _age in _ages}
        residual_torque_opt_stack = {_age: {_case: None for _case in _cases} for _age in _ages}
        minimum_residual_torque = {_age: {_case: None for _case in _cases} for _age in _ages}
        opt_constants = {_age: {_case: None for _case in _cases} for _age in _ages}

        # Initialise data and plateID dictionary
        _data = {_age: {_case: None for _case in _cases} for _age in _ages}
        _plateIDs = {_age: {_case: None for _case in _cases} for _age in _ages}

        # Loop through ages
        for j, _age in enumerate(_tqdm(_ages, desc="Inverting residual torque")):
            for _case in _cases:
                # Obtain plateIDs for all unique combinations of ages and cases
                _plateIDs[_age][_case] = utils_data.select_plateIDs(plateIDs, self.slabs.data[_age][_case]["lower_plateID"].unique())

                # Initialise entries for each plate ID in dictionaries
                driving_torque_opt_stack[_age][_case] = {_plateID: _numpy.zeros((len(constants))) for _plateID in _plateIDs[_age][_case]}
                residual_torque_opt_stack[_age][_case] = {_plateID: _numpy.zeros((len(constants))) for _plateID in _plateIDs[_age][_case]}
                minimum_residual_torque[_age][_case] = {_plateID: _numpy.nan for _plateID in _plateIDs[_age][_case]}
                opt_constants[_age][_case] = {_plateID: _numpy.nan for _plateID in _plateIDs[_age][_case]}

                # Loop through constants
                # for i, constant in enumerate(constants):
                for _plateID in _plateIDs[_age][_case]:
                    # Extract rotation pole
                    rotation_pole_lat = self.plates.extract_data_through_time(ages=_age, cases=_case, plateIDs=_plateID, var="pole_lat")
                    rotation_pole_lon = self.plates.extract_data_through_time(ages=_age, cases=_case, plateIDs=_plateID, var="pole_lon")
                    rotation_pole_mag = self.plates.extract_data_through_time(ages=_age, cases=_case, plateIDs=_plateID, var="pole_angle")
                    
                    # Convert rotation pole to Cartesian coordinates
                    rotation_pole_x, rotation_pole_y, rotation_pole_z = utils_calc.geocentric_spherical2cartesian(
                        rotation_pole_lat[_plateID].values[0], 
                        rotation_pole_lon[_plateID].values[0],
                        rotation_pole_mag[_plateID].values[0]
                    )

                    existing_values = []; existing_scores = []; dot_products = []
                    # for i in _numpy.arange(0, 1e2):
                    for i, constant in enumerate(constants):
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")

                            # # Propose initial value
                            # if i == 0:
                            #     # The inversion should start with a more general, grid-based exploration of the parameter space
                            #     # Only after ~20 iterations or so, the algorithm should start to adapt the step size
                            #     if j !=0 and _plateID in opt_constants[_ages[j-1]][_case]:
                            #         constant = opt_constants[_ages[j-1]][_case][_plateID]
                            #     else:
                            #         constant = 1e-10

                            # else:
                            #     constant = utils_calc.propose_value(existing_values, existing_scores, 0.3)


                            constant = 10**constant
                                                                    
                            existing_values.append(constant)

                            # Inform the user which constant is being optimised
                            # logging.info(f"Optimising for {constant}")

                            # Calculate the torques the normal way
                            self.plate_torques.calculate_slab_pull_torque(ages=_age, cases=_case, plateIDs=_plateID, PROGRESS_BAR=False)
                            self.plate_torques.calculate_driving_torque(ages=_age, cases=_case, plateIDs=_plateID, PROGRESS_BAR=False)
                            self.plate_torques.calculate_residual_torque(ages=_age, cases=_case, plateIDs=_plateID, PROGRESS_BAR=False)

                            # Select data
                            _data[_age][_case] = self.slabs.data[_age][_case].copy()

                            # Filter data, if necessary
                            if plateIDs is not None:
                                _data[_age][_case] = _data[_age][_case][_data[_age][_case]["lower_plateID"].isin(_plateIDs[_age][_case])]

                            # Skip if no data
                            if _data[_age][_case].empty:
                                continue

                            # Get the slab pull force magnitude
                            max_slab_pull_force_mag = _data[_age][_case]["slab_pull_force_mag"] / _data[_age][_case]["slab_pull_constant"]

                            # Modify the magnitude of the slab pull force using the 2D dot product of the residual force and the slab pull force and the constant
                            # This step should be performed in Cartesian coordinates.
                            _data[_age][_case]["slab_pull_force_mag"] -= (
                                _data[_age][_case]["residual_force_lat"] * _data[_age][_case]["slab_pull_force_lat"] + \
                                _data[_age][_case]["residual_force_lon"] * _data[_age][_case]["slab_pull_force_lon"]
                            ) * constant

                            # Ensure the slab pull force magnitude is positive and not larger than the original slab pull force magnitude
                            _data[_age][_case].loc[_data[_age][_case]["slab_pull_force_mag"] < 0, "slab_pull_force_mag"] = 0
                            _data[_age][_case].loc[_data[_age][_case]["slab_pull_force_mag"] > max_slab_pull_force_mag, "slab_pull_force_mag"] = max_slab_pull_force_mag[_data[_age][_case]["slab_pull_force_mag"] > max_slab_pull_force_mag]

                            # Decompose the slab pull force into latitudinal and longitudinal components using the trench normal azimuth
                            _data[_age][_case]["slab_pull_force_lat"] = _numpy.cos(_numpy.deg2rad(_data[_age][_case]["trench_normal_azimuth"])) * _data[_age][_case]["slab_pull_force_mag"]
                            _data[_age][_case]["slab_pull_force_lon"] = _numpy.sin(_numpy.deg2rad(_data[_age][_case]["trench_normal_azimuth"])) * _data[_age][_case]["slab_pull_force_mag"]
                        
                            # Calculate the torques with the modified slab pull forces
                            self.plates.calculate_torque_on_plates(_data, ages=_age, cases=_case, plateIDs=_plateID, torque_var="slab_pull", PROGRESS_BAR=False)

                            # This should be rewritten, as calclculating the residual vecter in spherical coordinates is time-consuming
                            # Instead, the dot product should be calculated in Cartesian coordinates
                            self.plate_torques.calculate_driving_torque(ages=_age, cases=_case, plateIDs=_plateID, PROGRESS_BAR=False)
                            self.plate_torques.calculate_residual_torque(ages=_age, cases=_case, plateIDs=_plateID, PROGRESS_BAR=False)

                            # Extract the driving and residual torques
                            _iter_driving_torque = self.plate_torques.extract_data_through_time(ages=_age, cases=_case, plateIDs=_plateID, var="driving_torque_mag")
                            _iter_residual_torque = self.plate_torques.extract_data_through_time(ages=_age, cases=_case, plateIDs=_plateID, var= "residual_torque_mag")

                            # Extract driving torque
                            _iter_driving_torque_x = self.plate_torques.extract_data_through_time(ages=_age, cases=_case, plateIDs=_plateID, var="driving_torque_x")
                            _iter_driving_torque_y = self.plate_torques.extract_data_through_time(ages=_age, cases=_case, plateIDs=_plateID, var="driving_torque_y")
                            _iter_driving_torque_z = self.plate_torques.extract_data_through_time(ages=_age, cases=_case, plateIDs=_plateID, var="driving_torque_z")

                            # Calculate dot product 
                            normalised_dot_product = _numpy.abs(
                                    _iter_driving_torque_x[_plateID].values[0] * rotation_pole_x + \
                                    _iter_driving_torque_y[_plateID].values[0] * rotation_pole_y + \
                                    _iter_driving_torque_z[_plateID].values[0] * rotation_pole_z
                                )
                            normalised_dot_product /= _numpy.abs(_iter_driving_torque[_plateID].values[0] * rotation_pole_mag[_plateID].values[0])

                            dot_products.append(1-normalised_dot_product)
                            # Calculate normalised residual torque
                            # Why is this only the magnitude? Another approach would be to calculate the difference of each component
                            normalised_residual_torque = _numpy.log10(_iter_residual_torque[_plateID].values[0] / _iter_driving_torque[_plateID].values[0])
                            
                            score = normalised_residual_torque
                            existing_scores.append(score)
                            # for _plateID in _plateIDs[_age][_case]:
                            #     driving_torque_opt_stack[_age][_case][_plateID][i] = _iter_driving_torque[_plateID].values[0]
                            #     residual_torque_opt_stack[_age][_case][_plateID][i] = _iter_residual_torque[_plateID].values[0]
 
                            # After 20 iterations, check the difference between the minimum and maximum values
                            # if i > 10:
                                # if (existing_scores[int(i)] - existing_scores[-1]) < 0.01:
                                #     break
                                    # if _numpy.nanmin(_numpy.asarray(existing_scores)) == existing_scores[-10]:
                                # if _numpy.nanmin(_numpy.asarray(existing_scores)) == existing_scores[-10]:
                                #     break
                                # elif _numpy.abs(existing_scores[-1] - existing_scores[-2]) < 0.01:
                                #     break

                        # Calculate the residual torque
                    # plt.scatter(_numpy.log10(_numpy.asarray(existing_values)), existing_scores, c=_numpy.arange(0, i+1))
                    # plt.ylim([-2, 1])
                    # plt.xlim([vmin, vmax])
                    # plt.scatter(_numpy.log10(_numpy.asarray(existing_values)), _numpy.log10(_numpy.asarray(dot_products)), c="r")
                    # plt.colorbar(label="Iteration")
                    # plt.title(f"{_plateID} ({_numpy.log10(_numpy.sum(max_slab_pull_force_mag))} N")
                    # plt.show()
                    # plt.plot(existing_scores)
                    # plt.ylim([-2, 1])
                    # plt.title(f"{_plateID} ({_numpy.log10(_numpy.sum(max_slab_pull_force_mag))} N")
                    # plt.show()

                    # This sometimes throws and error, so it is wrapped in a try-except block
                    try:        
                        minimum_residual_torque[_age][_case][_plateID] = _numpy.nanmin(_numpy.asarray(existing_scores))
                        opt_index = _numpy.nanargmin(_numpy.asarray(existing_scores))
                        opt_constants[_age][_case][_plateID] = _numpy.asarray(existing_values)[opt_index]
                    except:
                        minimum_residual_torque[_age][_case][_plateID] = _numpy.nan
                        opt_constants[_age][_case][_plateID] = _numpy.nan

                    # if not _data[_age][_case].empty:
                    #     minimum_residual_torque[_age][_case][_plateID] = _numpy.nanmin(residual_torque_opt_stack[_age][_case][_plateID]/driving_torque_opt_stack[_age][_case][_plateID])
                    #     opt_index = _numpy.nanargmin(residual_torque_opt_stack[_age][_case][_plateID]/driving_torque_opt_stack[_age][_case][_plateID])
                    #     opt_constants[_age][_case][_plateID] = constants[opt_index]

        for _age in _tqdm(_ages, desc="Optimising torques"):
            for _case in _cases:
                for _plateID in _plateIDs[_age][_case]:
                    # Recalculate all the relevant torques
                    self.plate_torques.calculate_slab_pull_torque(ages=_age, cases=_case, plateIDs=_plateID, PROGRESS_BAR=False)
                    self.plate_torques.calculate_residual_torque(ages=_age, cases=_case, plateIDs=_plateID, PROGRESS_BAR=False)

                    # Select data
                    _data = self.slabs.data[_age][_case].copy()
                    _data = _data[_data["lower_plateID"] == _plateID]

                    if not _data.empty:
                        # Get the old slab pull force magnitude
                        max_slab_pull_force_mag = _data["slab_pull_force_mag"].values / _data["slab_pull_constant"].values

                        # Calculate the slab pull force
                        _data.loc[_data.index, "slab_pull_force_mag"] -= (
                                _data["residual_force_lat"] * _data["slab_pull_force_lat"] + \
                                _data["residual_force_lon"] * _data["slab_pull_force_lon"]
                            ) * opt_constants[_age][_case][_plateID]
                        
                        # Make sure the slab pull force magnitude is positive and not larger than the original slab pull force magnitude
                        _data.loc[_data["slab_pull_force_mag"] < 0, "slab_pull_force_mag"] = 0
                        _data.loc[_data["slab_pull_force_mag"] > max_slab_pull_force_mag, "slab_pull_force_mag"] = max_slab_pull_force_mag[_data["slab_pull_force_mag"] > max_slab_pull_force_mag]

                        # Decompose the slab pull force into latitudinal and longitudinal components using the trench normal azimuth
                        _data.loc[_data.index, "slab_pull_force_lat"] = _numpy.cos(_numpy.deg2rad(_data["trench_normal_azimuth"])) * _data["slab_pull_force_mag"]
                        _data.loc[_data.index, "slab_pull_force_lon"] = _numpy.sin(_numpy.deg2rad(_data["trench_normal_azimuth"])) * _data["slab_pull_force_mag"]

                        # Calculate the slab pull constant
                        _data.loc[_data.index, "slab_pull_constant"] = _data.loc[_data.index, "slab_pull_force_mag"] / max_slab_pull_force_mag

                        # Make sure the slab pull constant is between 0 and 1
                        _data.loc[_data["slab_pull_constant"] < 0, "slab_pull_constant"] = 0
                        _data.loc[_data["slab_pull_constant"] > 1, "slab_pull_constant"] = 1

                        # Feed optimal values back into slab data
                        self.slabs.data[_age][_case].loc[_data.index, "slab_pull_force_mag"] = _data["slab_pull_force_mag"].values
                        self.slabs.data[_age][_case].loc[_data.index, "slab_pull_force_lat"] = _data["slab_pull_force_lat"].values
                        self.slabs.data[_age][_case].loc[_data.index, "slab_pull_force_lon"] = _data["slab_pull_force_lon"].values
                        self.slabs.data[_age][_case].loc[_data.index, "slab_pull_constant"] = _data["slab_pull_constant"].values

                # Recalculate all the relevant torques
                self.plates.calculate_torque_on_plates(self.slabs.data, ages=_age, cases=_case, plateIDs=_plateIDs[_age][_case], torque_var="slab_pull", PROGRESS_BAR=False)
                self.plate_torques.calculate_driving_torque(ages=_age, cases=_case, plateIDs=_plateIDs[_age][_case], PROGRESS_BAR=False)
                self.plate_torques.calculate_residual_torque(ages=_age, cases=_case, plateIDs=_plateIDs[_age][_case], PROGRESS_BAR=False)
    
    def invert_residual_torque_v2(
            self,
            age = None, 
            cases = None, 
            plateIDs = None, 
            parameter = "Slab pull constant",
            vmin = 8.5, 
            vmax = 13.5,
            step = .1, 
            plot = False, 
        ):
        """
        Function to find optimised slab pull coefficient or mantle viscosity by projecting the residual torque onto the subduction zones and grid points.

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

        NOTE: This function is still under development and may not work as intended.
        """
        # Raise error if parameter is not slab pull coefficient or mantle viscosity
        if parameter not in ["Slab pull constant", "Mantle viscosity"]:
            raise ValueError("Free parameter must be 'Slab pull constant' or 'Mantle viscosity")
        
        # Select ages, if not provided
        _ages = utils_data.select_ages(age, self.settings.ages)

        # Select cases, if not provided
        _cases = utils_data.select_cases(cases, self.settings.reconstructed_cases)

        # Define constants
        constants = _numpy.arange(vmin, vmax, step)

        # Initialise dictionaries to store minimum residual torque and optimal constants
        pole_lat_opt_stack = {_age: {_case: None for _case in _cases} for _age in _ages}
        pole_lon_opt_stack = {_age: {_case: None for _case in _cases} for _age in _ages}
        pole_angle_opt_stack = {_age: {_case: None for _case in _cases} for _age in _ages}
        opt_constants = {_age: {_case: None for _case in _cases} for _age in _ages}

        # Initialise data and plateID dictionary
        _data = {_age: {_case: None for _case in _cases} for _age in _ages}
        _plateIDs = {_age: {_case: None for _case in _cases} for _age in _ages}

        # Loop through ages
        for _age in _tqdm(_ages, desc="Inverting residual torque"):
            for _case in _cases:
                # Obtain plateIDs for all unique combinations of ages and cases
                _plateIDs[_age][_case] = utils_data.select_plateIDs(plateIDs, self.slabs.data[_age][_case]["lower_plateID"].unique())

                # Initialise entries for each plate ID in dictionaries
                pole_lat_opt_stack[_age][_case] = {_plateID: _numpy.zeros((len(constants))) for _plateID in _plateIDs[_age][_case]}
                pole_lon_opt_stack[_age][_case] = {_plateID: _numpy.zeros((len(constants))) for _plateID in _plateIDs[_age][_case]}
                pole_angle_opt_stack[_age][_case] = {_plateID: _numpy.zeros((len(constants))) for _plateID in _plateIDs[_age][_case]}
                opt_constants[_age][_case] = {_plateID: _numpy.nan for _plateID in _plateIDs[_age][_case]}

                # Store reconstructed pole of rotation
                reconstructed_pole_lat = self.plate_torques.extract_data_through_time(ages=_age, cases=_case, plateIDs=_plateIDs[_age][_case], var="pole_lat")
                reconstructed_pole_lon = self.plate_torques.extract_data_through_time(ages=_age, cases=_case, plateIDs=_plateIDs[_age][_case], var="pole_lon")
                reconstructed_pole_angle = self.plate_torques.extract_data_through_time(ages=_age, cases=_case, plateIDs=_plateIDs[_age][_case], var="pole_angle")

                for i, constant in enumerate(constants):
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")

                        # Inform the user which constant is being optimised
                        logging.info(f"Optimising for {constant}")

                        # Calculate the torques the normal way
                        self.plate_torques.calculate_slab_pull_torque(ages=_age, cases=_case, plateIDs=_plateIDs[_age][_case], PROGRESS_BAR=False)
                        self.plate_torques.calculate_driving_torque(ages=_age, cases=_case, plateIDs=_plateIDs[_age][_case], PROGRESS_BAR=False)
                        self.plate_torques.calculate_residual_torque(ages=_age, cases=_case, plateIDs=_plateIDs[_age][_case], PROGRESS_BAR=False)

                        # Select data
                        _data[_age][_case] = self.slabs.data[_age][_case].copy()

                        # Filter data, if necessary
                        if plateIDs is not None:
                            _data[_age][_case] = _data[_age][_case][_data[_age][_case]["lower_plateID"].isin(_plateIDs[_age][_case])]

                        # Skip if no data
                        if _data[_age][_case].empty:
                            continue

                        # Get the slab pull force magnitude
                        max_slab_pull_force_mag = _data[_age][_case]["slab_pull_force_mag"] / _data[_age][_case]["slab_pull_constant"]

                        # Modify the magnitude of the slab pull force using the 2D dot product of the residual force and the slab pull force and the constant
                        _data[_age][_case]["slab_pull_force_mag"] -= (
                            _data[_age][_case]["residual_force_lat"] * _data[_age][_case]["slab_pull_force_lat"] + \
                            _data[_age][_case]["residual_force_lon"] * _data[_age][_case]["slab_pull_force_lon"]
                        ) * 10**-constant

                        # Ensure the slab pull force magnitude is positive and not larger than the original slab pull force magnitude
                        _data[_age][_case].loc[_data[_age][_case]["slab_pull_force_mag"] < 0, "slab_pull_force_mag"] = 0
                        _data[_age][_case].loc[_data[_age][_case]["slab_pull_force_mag"] > max_slab_pull_force_mag, "slab_pull_force_mag"] = max_slab_pull_force_mag[_data[_age][_case]["slab_pull_force_mag"] > max_slab_pull_force_mag]

                        # Decompose the slab pull force into latitudinal and longitudinal components using the trench normal azimuth
                        _data[_age][_case]["slab_pull_force_lat"] = _numpy.cos(_numpy.deg2rad(_data[_age][_case]["trench_normal_azimuth"])) * _data[_age][_case]["slab_pull_force_mag"]
                        _data[_age][_case]["slab_pull_force_lon"] = _numpy.sin(_numpy.deg2rad(_data[_age][_case]["trench_normal_azimuth"])) * _data[_age][_case]["slab_pull_force_mag"]
                    
                        # Calculate the torques with the modified slab pull forces
                        self.plates.calculate_torque_on_plates(_data, ages=_age, cases=_case, plateIDs=_plateIDs[_age][_case], torque_var="slab_pull", PROGRESS_BAR=False)
                        self.plates.calculate_synthetic_velocity(ages=_age, cases=_case, plateIDs=_plateIDs[_age][_case], PROGRESS_BAR=False, RECONSTRUCTED_CASES=True)

                        # Extract the pole of rotation
                        _iter_pole_lat = self.plate_torques.extract_data_through_time(ages=_age, cases=_case, plateIDs=_plateIDs[_age][_case], var="pole_lon")
                        _iter_pole_lon = self.plate_torques.extract_data_through_time(ages=_age, cases=_case, plateIDs=_plateIDs[_age][_case], var="pole_lat")
                        _iter_pole_angle = self.plate_torques.extract_data_through_time(ages=_age, cases=_case, plateIDs=_plateIDs[_age][_case], var="pole_angle")

                        for _plateID in _plateIDs[_age][_case]:
                            # print(_age, _case, _plateID, constant, _iter_pole_lat[_plateID].values[0], _iter_pole_lon[_plateID].values[0], _iter_pole_angle[_plateID].values[0])
                            pole_lon_opt_stack[_age][_case][_plateID][i] = _iter_pole_lat[_plateID].values[0]
                            pole_lat_opt_stack[_age][_case][_plateID][i] = _iter_pole_lon[_plateID].values[0]
                            pole_angle_opt_stack[_age][_case][_plateID][i] = _iter_pole_angle[_plateID].values[0]
                            
                for _plateID in _plateIDs[_age][_case]:
                    # Calculate distance to the pole of rotation
                    distances = utils_calc.haversine(
                        pole_lon_opt_stack[_age][_case][_plateID],
                        pole_lat_opt_stack[_age][_case][_plateID],
                        _numpy.repeat(reconstructed_pole_lon[_plateID], len(constants)),
                        _numpy.repeat(reconstructed_pole_lon[_plateID], len(constants)),
                    )
                    plt.plot(constants, distances, label="Distance to pole of rotation")
                    plt.plot(constants, pole_angle_opt_stack[_age][_case][_plateID]-_numpy.repeat(reconstructed_pole_angle[_plateID], len(constants)), label="Ratio between synthetic and reconstructed pole angle")
                    plt.plot(constants, distances + _numpy.abs(pole_angle_opt_stack[_age][_case][_plateID]-_numpy.repeat(reconstructed_pole_angle[_plateID], len(constants))), label="Equally weighted components")
                    plt.legend()
                    plt.show()

                    opt_index = _numpy.nanargmin(distances + _numpy.abs(pole_angle_opt_stack[_age][_case][_plateID]-_numpy.repeat(reconstructed_pole_angle[_plateID], len(constants))))
                    opt_constants[_age][_case][_plateID] = constants[opt_index]

                    print("Optimal constant for plate", _plateID, f"for case {_case} at age {_age}", opt_constants[_age][_case][_plateID])

        for _age in _tqdm(_ages, desc="Optimising torques"):
            for _case in _cases:
                for _plateID in _plateIDs[_age][_case]:
                    # Recalculate all the relevant torques
                    self.plate_torques.calculate_slab_pull_torque(ages=_age, cases=_case, plateIDs=_plateID, PROGRESS_BAR=False)
                    self.plate_torques.calculate_residual_torque(ages=_age, cases=_case, plateIDs=_plateID, PROGRESS_BAR=False)

                    # Select data
                    _data = self.slabs.data[_age][_case].copy()
                    _data = _data[_data["lower_plateID"] == _plateID]

                    if not _data.empty:
                        # Get the old slab pull force magnitude
                        max_slab_pull_force_mag = _data["slab_pull_force_mag"].values / _data["slab_pull_constant"].values

                        # Calculate the slab pull force
                        _data.loc[_data.index, "slab_pull_force_mag"] -= (
                                _data["residual_force_lat"] * _data["slab_pull_force_lat"] + \
                                _data["residual_force_lon"] * _data["slab_pull_force_lon"]
                            ) * 10**-opt_constants[_age][_case][_plateID]
                        
                        # Make sure the slab pull force magnitude is positive and not larger than the original slab pull force magnitude
                        _data.loc[_data["slab_pull_force_mag"] < 0, "slab_pull_force_mag"] = 0
                        _data.loc[_data["slab_pull_force_mag"] > max_slab_pull_force_mag, "slab_pull_force_mag"] = max_slab_pull_force_mag[_data["slab_pull_force_mag"] > max_slab_pull_force_mag]

                        # Decompose the slab pull force into latitudinal and longitudinal components using the trench normal azimuth
                        _data.loc[_data.index, "slab_pull_force_lat"] = _numpy.cos(_numpy.deg2rad(_data["trench_normal_azimuth"])) * _data["slab_pull_force_mag"]
                        _data.loc[_data.index, "slab_pull_force_lon"] = _numpy.sin(_numpy.deg2rad(_data["trench_normal_azimuth"])) * _data["slab_pull_force_mag"]

                        # Calculate the slab pull constant
                        _data.loc[_data.index, "slab_pull_constant"] = _data.loc[_data.index, "slab_pull_force_mag"] / max_slab_pull_force_mag

                        # Make sure the slab pull constant is between 0 and 1
                        _data.loc[_data["slab_pull_constant"] < 0, "slab_pull_constant"] = 0
                        _data.loc[_data["slab_pull_constant"] > 1, "slab_pull_constant"] = 1

                        # Feed optimal values back into slab data
                        self.slabs.data[_age][_case].loc[_data.index, "slab_pull_force_mag"] = _data["slab_pull_force_mag"].values
                        self.slabs.data[_age][_case].loc[_data.index, "slab_pull_force_lat"] = _data["slab_pull_force_lat"].values
                        self.slabs.data[_age][_case].loc[_data.index, "slab_pull_force_lon"] = _data["slab_pull_force_lon"].values
                        self.slabs.data[_age][_case].loc[_data.index, "slab_pull_constant"] = _data["slab_pull_constant"].values

                # Recalculate all the relevant torques
                self.plates.calculate_torque_on_plates(self.slabs.data, ages=_age, cases=_case, plateIDs=_plateIDs[_age][_case], torque_var="slab_pull", PROGRESS_BAR=False)
                self.plate_torques.calculate_driving_torque(ages=_age, cases=_case, plateIDs=_plateIDs[_age][_case], PROGRESS_BAR=False)
                self.plate_torques.calculate_residual_torque(ages=_age, cases=_case, plateIDs=_plateIDs[_age][_case], PROGRESS_BAR=False)

    def invert_residual_torque_v3(
            self,
            ages = None, 
            cases = None, 
            plateIDs = None, 
            parameter = "Slab pull constant",
            vmin = -14.0, 
            vmax = -6.0,
            step = .25, 
            plot = False, 
        ):
        """
        Function to find optimised slab pull constant by projecting the residual torque onto the subduction zones.
        The function loops through all the unique combinations of ages and cases and optimises the slab pull coefficient for each plate.
        The model space for the slab pull constant is explored using an iterative approach with an adaptive step size.

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
        # Raise error if parameter is not slab pull coefficient or mantle viscosity
        if parameter not in ["Slab pull constant", "Mantle viscosity"]:
            raise ValueError("Free parameter must be 'Slab pull constant' or 'Mantle viscosity")
        
        # Select ages, if not provided
        _ages = utils_data.select_ages(ages, self.settings.ages)

        # Select cases, if not provided
        _cases = utils_data.select_cases(cases, self.settings.reconstructed_cases)

        # Define constants
        constants = _numpy.arange(vmin, vmax, step)

        # Initialise dictionaries to store minimum residual torque and optimal constants
        driving_torque_opt_stack = {_age: {_case: None for _case in _cases} for _age in _ages}
        residual_torque_opt_stack = {_age: {_case: None for _case in _cases} for _age in _ages}
        minimum_residual_torque = {_age: {_case: None for _case in _cases} for _age in _ages}
        opt_constants = {_age: {_case: None for _case in _cases} for _age in _ages}

        # Initialise data and plateID dictionary
        _plate_data = {_age: {_case: None for _case in _cases} for _age in _ages}
        _slab_data = {_age: {_case: None for _case in _cases} for _age in _ages}
        _plateIDs = {_age: {_case: None for _case in _cases} for _age in _ages}

        # Loop through ages
        for j, _age in enumerate(_tqdm(_ages, desc="Inverting residual torque")):
            for _case in _cases:
                # Obtain plateIDs for all unique combinations of ages and cases
                _plateIDs[_age][_case] = utils_data.select_plateIDs(plateIDs, self.slabs.data[_age][_case]["lower_plateID"].unique())

                # Initialise entries for each plate ID in dictionaries
                driving_torque_opt_stack[_age][_case] = {_plateID: _numpy.zeros((len(constants))) for _plateID in _plateIDs[_age][_case]}
                residual_torque_opt_stack[_age][_case] = {_plateID: _numpy.zeros((len(constants))) for _plateID in _plateIDs[_age][_case]}
                minimum_residual_torque[_age][_case] = {_plateID: _numpy.nan for _plateID in _plateIDs[_age][_case]}
                opt_constants[_age][_case] = {_plateID: _numpy.nan for _plateID in _plateIDs[_age][_case]}

                # Loop through plates
                for _plateID in _plateIDs[_age][_case]:
                    if _plateID not in self.plates.data[_age][_case]["plateID"].values:
                        continue
                    # # Extract rotation pole
                    # rotation_pole_lat = self.plates.extract_data_through_time(ages=_age, cases=_case, plateIDs=_plateID, var="pole_lat")
                    # rotation_pole_lon = self.plates.extract_data_through_time(ages=_age, cases=_case, plateIDs=_plateID, var="pole_lon")
                    # rotation_pole_mag = self.plates.extract_data_through_time(ages=_age, cases=_case, plateIDs=_plateID, var="pole_angle")
                    
                    # # Convert rotation pole to Cartesian coordinates
                    # rotation_pole_x, rotation_pole_y, rotation_pole_z = utils_calc.geocentric_spherical2cartesian(
                    #     rotation_pole_lat[_plateID].values[0], 
                    #     rotation_pole_lon[_plateID].values[0],
                    #     rotation_pole_mag[_plateID].values[0]
                    # )

                    existing_values = []; existing_scores = []; dot_products = []
                    for i in _numpy.arange(0, len(constants)+10):
                    # for i, constant in enumerate(constants):
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")

                            # Propose initial value
                            if i > len(constants)-1:
                                # The inversion should start with a more general, grid-based exploration of the parameter space
                                # Only after ~20 iterations or so, the algorithm should start to adapt the step size
                                constant = utils_calc.propose_value(existing_values, existing_scores, lower_bound=vmin, upper_bound=vmax)
                            else:
                                constant = 10**constants[i]

                            # else:
                            #     constant = utils_calc.propose_value(existing_values, existing_scores, 0.3)

                            # constant = 10**constant
                                                                    
                            existing_values.append(constant)

                            # Inform the user which constant is being optimised
                            # logging.info(f"Optimising for {constant}")

                            # Calculate the torques the normal way
                            # self.plate_torques.calculate_slab_pull_torque(ages=_age, cases=_case, plateIDs=_plateID, PROGRESS_BAR=False)
                            # self.plate_torques.calculate_driving_torque(ages=_age, cases=_case, plateIDs=_plateID, PROGRESS_BAR=False)
                            # self.plate_torques.calculate_residual_torque(ages=_age, cases=_case, plateIDs=_plateID, PROGRESS_BAR=False, CALCULATE_AT_POINTS=False)

                            # Select data
                            _plate_data = self.plates.data[_age][_case].copy()
                            _slab_data[_age][_case] = self.slabs.data[_age][_case].copy()

                            # Filter data, if necessary
                            if plateIDs is not None:
                                _plate_data = _plate_data[_plate_data["plateID"].isin(_plateIDs[_age][_case])]
                                _slab_data[_age][_case] = _slab_data[_age][_case][_slab_data[_age][_case]["lower_plateID"].isin(_plateIDs[_age][_case])]

                            # Skip if no data
                            if _slab_data[_age][_case].empty:
                                continue

                            print(_plate_data)

                            # Get the slab pull force magnitude
                            max_slab_pull_force_mag = _slab_data[_age][_case]["slab_pull_force_mag"] / _slab_data[_age][_case]["slab_pull_constant"]

                            # Modify the magnitude of the slab pull force using the 2D dot product of the residual force and the slab pull force and the constant
                            # This step should be performed in Cartesian coordinates.
                            _slab_data[_age][_case]["slab_pull_force_mag"] -= (
                                _slab_data[_age][_case]["residual_force_lat"] * _slab_data[_age][_case]["slab_pull_force_lat"] + \
                                _slab_data[_age][_case]["residual_force_lon"] * _slab_data[_age][_case]["slab_pull_force_lon"]
                            ) * constant

                            # Ensure the slab pull force magnitude is positive and not larger than the original slab pull force magnitude
                            _slab_data[_age][_case].loc[_slab_data[_age][_case]["slab_pull_force_mag"] < 0, "slab_pull_force_mag"] = 0
                            _slab_data[_age][_case].loc[_slab_data[_age][_case]["slab_pull_force_mag"] > max_slab_pull_force_mag, "slab_pull_force_mag"] = max_slab_pull_force_mag[_slab_data[_age][_case]["slab_pull_force_mag"] > max_slab_pull_force_mag]

                            # Decompose the slab pull force into latitudinal and longitudinal components using the trench normal azimuth
                            _slab_data[_age][_case]["slab_pull_force_lat"] = _numpy.cos(_numpy.deg2rad(_slab_data[_age][_case]["trench_normal_azimuth"])) * _slab_data[_age][_case]["slab_pull_force_mag"]
                            _slab_data[_age][_case]["slab_pull_force_lon"] = _numpy.sin(_numpy.deg2rad(_slab_data[_age][_case]["trench_normal_azimuth"])) * _slab_data[_age][_case]["slab_pull_force_mag"]
                        
                            # Calculate the torques with the modified slab pull forces
                            # Calculate torques
                            _iter_torques = utils_calc.compute_torque_on_plates(
                                _plate_data,
                                _slab_data[_age][_case].lat.values,
                                _slab_data[_age][_case].lon.values,
                                _slab_data[_age][_case].lower_plateID.values,
                                _slab_data[_age][_case].slab_pull_force_lat.values, 
                                _slab_data[_age][_case].slab_pull_force_lon.values,
                                _slab_data[_age][_case].trench_segment_length.values,
                                self.settings.constants,
                                torque_var = "slab_pull",
                            )
                            # self.plates.calculate_torque_on_plates(_data, ages=_age, cases=_case, plateIDs=_plateID, torque_var="slab_pull", PROGRESS_BAR=False)

                            # This should be rewritten, as calculating the residual vecter in spherical coordinates is time-consuming
                            # Instead, the dot product should be calculated in Cartesian coordinates
                            # driving_torques = utils_calc.compute_driving_torque(

                            # Calculate driving and residual torque
                            _iter_torques = utils_calc.sum_torque(_iter_torques, "driving", self.settings.constants)
                            _iter_torques = utils_calc.sum_torque(_iter_torques, "residual", self.settings.constants)

                            print(_iter_torques)
                            # self.plate_torques.calculate_driving_torque(ages=_age, cases=_case, plateIDs=_plateID, PROGRESS_BAR=False)
                            # self.plate_torques.calculate_residual_torque(ages=_age, cases=_case, plateIDs=_plateID, PROGRESS_BAR=False, CALCULATE_AT_POINTS=False)

                            # Extract the driving and residual torques
                            # _iter_driving_torque = self.plate_torques.extract_data_through_time(ages=_age, cases=_case, plateIDs=_plateID, var="driving_torque_mag")
                            # _iter_residual_torque = self.plate_torques.extract_data_through_time(ages=_age, cases=_case, plateIDs=_plateID, var= "residual_torque_mag")

                            _iter_driving_torque = _iter_torques[_iter_torques["plateID"] == _plateID]["driving_torque_mag"].values
                            _iter_residual_torque = _iter_torques[_iter_torques["plateID"] == _plateID]["residual_torque_mag"].values

                            # # Extract driving torque
                            # _iter_driving_torque_x = self.plate_torques.extract_data_through_time(ages=_age, cases=_case, plateIDs=_plateID, var="driving_torque_x")
                            # _iter_driving_torque_y = self.plate_torques.extract_data_through_time(ages=_age, cases=_case, plateIDs=_plateID, var="driving_torque_y")
                            # _iter_driving_torque_z = self.plate_torques.extract_data_through_time(ages=_age, cases=_case, plateIDs=_plateID, var="driving_torque_z")

                            # # Calculate dot product 
                            # normalised_dot_product = _numpy.abs(
                            #         _iter_driving_torque_x[_plateID].values[0] * rotation_pole_x + \
                            #         _iter_driving_torque_y[_plateID].values[0] * rotation_pole_y + \
                            #         _iter_driving_torque_z[_plateID].values[0] * rotation_pole_z
                            #     )
                            # normalised_dot_product /= _numpy.abs(_iter_driving_torque[_plateID].values[0] * rotation_pole_mag[_plateID].values[0])

                            # dot_products.append(1-normalised_dot_product)
                            # Calculate normalised residual torque
                            # Why is this only the magnitude? Another approach would be to calculate the difference of each component
                            normalised_residual_torque = _numpy.log10(_iter_residual_torque / _iter_driving_torque)
                            
                            score = normalised_residual_torque
                            existing_scores.append(score)

                    if plot == True or plot == _plateID:
                        # Plot the optimisation process
                        fig, axes = plt.subplots(1, 2)
                        axes[0].plot(existing_scores)
                        axes[0].set_xlabel("Iteration")
                        axes[0].set_ylabel("Score")
                        axes[0].set_ylim(-2, 1)

                        p = axes[1].scatter(_numpy.log10(_numpy.asarray(existing_values)), existing_scores, c=_numpy.arange(0, i+1))
                        axes[1].set_ylim(-2, 1)
                        axes[1].set_xlim(vmin, vmax)
                        axes[1].colorbar(p, label="iteration", orientation="horizontal")
                        axes[1].set_xlabel("Parameter value")
                        axes[1].set_yticklabels([])
                        fig.suptitle(f"Optimisation for plateID {_plateID}")
                    
                    # Find the minimum value
                    # NOTE: This sometimes throws and error, so it is wrapped in a try-except block
                    try:        
                        minimum_residual_torque[_age][_case][_plateID] = _numpy.nanmin(_numpy.asarray(existing_scores))
                        opt_index = _numpy.nanargmin(_numpy.asarray(existing_scores))
                        opt_constants[_age][_case][_plateID] = _numpy.asarray(existing_values)[opt_index]
                    except:
                        minimum_residual_torque[_age][_case][_plateID] = _numpy.nan
                        opt_constants[_age][_case][_plateID] = _numpy.nan

        for _age in _tqdm(_ages, desc="Optimising torques"):
            for _case in _cases:
                for _plateID in _plateIDs[_age][_case]:
                    # Recalculate all the relevant torques
                    self.plate_torques.calculate_slab_pull_torque(ages=_age, cases=_case, plateIDs=_plateID, PROGRESS_BAR=False)
                    self.plate_torques.calculate_residual_torque(ages=_age, cases=_case, plateIDs=_plateID, PROGRESS_BAR=False)

                    # Select data
                    _data = self.slabs.data[_age][_case].copy()
                    _data = _data[_data["lower_plateID"] == _plateID]

                    if not _data.empty:
                        # Get the old slab pull force magnitude
                        max_slab_pull_force_mag = _data["slab_pull_force_mag"].values / _data["slab_pull_constant"].values

                        # Calculate the slab pull force
                        _data.loc[_data.index, "slab_pull_force_mag"] -= (
                                _data["residual_force_lat"] * _data["slab_pull_force_lat"] + \
                                _data["residual_force_lon"] * _data["slab_pull_force_lon"]
                            ) * opt_constants[_age][_case][_plateID]
                        
                        # Make sure the slab pull force magnitude is positive and not larger than the original slab pull force magnitude
                        _data.loc[_data["slab_pull_force_mag"] < 0, "slab_pull_force_mag"] = 0
                        _data.loc[_data["slab_pull_force_mag"] > max_slab_pull_force_mag, "slab_pull_force_mag"] = max_slab_pull_force_mag[_data["slab_pull_force_mag"] > max_slab_pull_force_mag]

                        # Decompose the slab pull force into latitudinal and longitudinal components using the trench normal azimuth
                        _data.loc[_data.index, "slab_pull_force_lat"] = _numpy.cos(_numpy.deg2rad(_data["trench_normal_azimuth"])) * _data["slab_pull_force_mag"]
                        _data.loc[_data.index, "slab_pull_force_lon"] = _numpy.sin(_numpy.deg2rad(_data["trench_normal_azimuth"])) * _data["slab_pull_force_mag"]

                        # Calculate the slab pull constant
                        _data.loc[_data.index, "slab_pull_constant"] = _data.loc[_data.index, "slab_pull_force_mag"] / max_slab_pull_force_mag

                        # Make sure the slab pull constant is between 0 and 1
                        _data.loc[_data["slab_pull_constant"] < 0, "slab_pull_constant"] = 0
                        _data.loc[_data["slab_pull_constant"] > 1, "slab_pull_constant"] = 1

                        # Feed optimal values back into slab data
                        self.slabs.data[_age][_case].loc[_data.index, "slab_pull_force_mag"] = _data["slab_pull_force_mag"].values
                        self.slabs.data[_age][_case].loc[_data.index, "slab_pull_force_lat"] = _data["slab_pull_force_lat"].values
                        self.slabs.data[_age][_case].loc[_data.index, "slab_pull_force_lon"] = _data["slab_pull_force_lon"].values
                        self.slabs.data[_age][_case].loc[_data.index, "slab_pull_constant"] = _data["slab_pull_constant"].values

                # Recalculate all the relevant torques
                self.plates.calculate_torque_on_plates(self.slabs.data, ages=_age, cases=_case, plateIDs=_plateIDs[_age][_case], torque_var="slab_pull", PROGRESS_BAR=False)
                self.plate_torques.calculate_driving_torque(ages=_age, cases=_case, plateIDs=_plateIDs[_age][_case], PROGRESS_BAR=False)
                self.plate_torques.calculate_residual_torque(ages=_age, cases=_case, plateIDs=_plateIDs[_age][_case], PROGRESS_BAR=False)

    def optimise_torques(
            self,
            ages = None,
            cases = None,
            plateIDs = None,
            optimisation_age = None,
            optimisation_case = None,
            optimisation_plateID = None
        ):
        """
        Function to optimsie the torques in the PlateTorques object.

        :param ages:                    ages to optimise (default: None)
        :type ages:                     list of int, float
        :param cases:                   cases to optimise (default: None)
        :type cases:                    list of str
        :param plateIDs:                plate IDs to optimise (default: None)
        :type plateIDs:                 list of int
        :param optimisation_age:        age to optimise (default: None)
        :type optimisation_age:         int, float
        :param optimisation_case:       case to optimise (default: None)
        :type optimisation_case:        str
        :param optimisation_plateID:    plate ID of plate to optimise (default: None)
        :type optimisation_plateID:     int

        NOTE: This function is not written in the most efficient way and can be improved.
        """
        # Select ages if not provided
        _ages = utils_data.select_ages(ages, self.settings.ages)

        # Select cases if not provided
        _cases = utils_data.select_cases(cases, self.settings.cases)

        # Loop through ages and cases
        for _age in _tqdm(_ages, desc="Optimising torques"):
            for _case in _cases:
                # Select plateIDs
                _plateIDs = utils_data.select_plateIDs(plateIDs, self.plates.data[_age][_case].plateID)

                for k, _plateID in enumerate(self.plates.data[_age][_case].plateID):
                    if _plateID in _plateIDs:
                        # Filter data
                        _plate_data = self.plates.data[_age][_case][self.plates.data[_age][_case]["plateID"] == _plateID].copy()
                        _slab_data = self.slabs.data[_age][_case][self.slabs.data[_age][_case]["lower_plateID"] == _plateID].copy()
                        _point_data = self.points.data[_age][_case][self.points.data[_age][_case]["plateID"] == _plateID].copy()
                        
                        # Select optimal slab pull coefficient and viscosity
                        if optimisation_age or optimisation_case or optimisation_plateID is None:
                            opt_sp_const = self.opt_sp_const[_age][_case][k]
                            opt_visc = self.opt_visc[_age][_case][k]
                        else:
                            opt_sp_const = self.opt_sp_const[optimisation_age][optimisation_case][optimisation_plateID] 
                            opt_visc = self.opt_visc[optimisation_age][optimisation_case][optimisation_plateID]

                        # Optimise plate torque
                        for coord in ["x", "y", "z", "mag"]:
                            _plate_data.loc[:, f"slab_pull_torque_{coord}"] *= opt_sp_const / self.settings.options[_case]["Slab pull constant"]
                            self.plates.data[_age][_case].loc[_plate_data.index, f"slab_pull_torque_{coord}"] = _plate_data[f"slab_pull_torque_{coord}"].values[0]

                            _plate_data.loc[:, f"mantle_drag_torque_{coord}"] *= opt_visc / self.settings.options[_case]["Mantle viscosity"]
                            self.plates.data[_age][_case].loc[_plate_data.index, f"mantle_drag_torque_{coord}"] = _plate_data[f"mantle_drag_torque_{coord}"].values[0]

                        # Optimise slab pull force
                        for coord in ["lat", "lon", "mag"]:
                            _slab_data.loc[:, f"slab_pull_force_{coord}"] *= opt_sp_const / self.settings.options[_case]["Slab pull constant"]
                            self.slabs.data[_age][_case].loc[_slab_data.index, f"slab_pull_force_{coord}"] = _slab_data[f"slab_pull_force_{coord}"].values[0]

                        # Optimise mantle drag torque
                        for coord in ["lat", "lon", "mag"]:
                            _point_data.loc[:, f"mantle_drag_force_{coord}"] *= opt_visc / self.settings.options[_case]["Mantle viscosity"]
                            self.points.data[_age][_case].loc[_point_data.index, f"mantle_drag_force_{coord}"] = _point_data[f"mantle_drag_force_{coord}"].values[0]

        # Recalculate driving and residual torques
        self.plate_torques.calculate_driving_torque(_ages, _cases)
        self.plate_torques.calculate_residual_torque(_ages, _cases)

    def remove_net_rotation(
            self,
            ages: Optional[Union[int, float, List[Union[int, float]], _numpy.ndarray]] = None,
            cases: Optional[Union[str, List[str]]] = None,
            plateIDs: Optional[Union[int, float, List[Union[int, float]], _numpy.ndarray]] = None,
        ):
        """
        Remove net rotation 
        
        NOTE: This function is not yet implemented.
        """
        # Define ages if not provided
        _ages = utils_data.select_ages(ages, self.settings.ages)
        
        # Define cases if not provided
        _cases = utils_data.select_cases(cases, self.settings.cases)

        # Loop through plates
        for _age in _tqdm(_ages, desc="Removing net lithospheric rotation"):
            for _case in _cases:
                # Select plateIDs
                _plateIDs = utils_data.select_plateIDs(plateIDs, self.plates.data[_age][_case].plateID)

                # Select data
                _plate_data = self.plates.data[_age][_case].copy()
                _point_data = self.points.data[_age][_case].copy()

                # Filter plates
                if plateIDs is not None:
                    _plate_data = _plate_data[_plate_data["plateID"].isin(_plateIDs)]
                    _point_data = _point_data[_point_data["plateID"].isin(_plateIDs)]

                # Remove net rotation
                computed_data = utils_calc.compute_no_net_rotation(_plate_data, _point_data)

                # Feed data back into the object
                self.plates.data[_age][_case] = computed_data

        # Recalculate new velocities
        self.points.calculate_velocities(_ages, _cases, self.plates.data)
        self.slabs.calculate_velocities(_ages, _cases, self.plates.data)
        self.plate_torques.calculate_rms_velocity(_ages, _cases)

    def remove_net_trench_migration(
            self,
            ages: Optional[Union[int, float, List[Union[int, float]], _numpy.ndarray]] = None,
            cases: Optional[Union[str, List[str]]] = None,
            plateIDs: Optional[Union[int, float, List[Union[int, float]], _numpy.ndarray]] = None,
            type: str = None
        ):
        """
        Remove net rotation 
        
        NOTE: This function is not yet implemented.
        """
        # Define ages if not provided
        _ages = utils_data.select_ages(ages, self.settings.ages)
        
        # Define cases if not provided
        _cases = utils_data.select_cases(cases, self.settings.cases)

        # Loop through plates
        for _age in _tqdm(_ages, desc="Removing net trench migration"):
            for _case in _cases:
                # Select plateIDs
                _plateIDs = utils_data.select_plateIDs(plateIDs, self.plates.data[_age][_case].plateID)

                # Select data
                _plate_data = self.plates.data[_age][_case].copy()
                _slab_data = self.slabs.data[_age][_case].copy()

                # Filter plates
                if plateIDs is not None:
                    _plate_data = _plate_data[_plate_data["plateID"].isin(_plateIDs)]
                    _slab_data = _slab_data[_slab_data["plateID"].isin(_plateIDs)]

                # Remove net rotation
                computed_data = utils_calc.compute_no_net_trench_migration(
                    _plate_data,
                    _slab_data,
                    self.options[_case],
                    type
                )

                # Feed data back into the object
                self.plates.data[_age][_case] = computed_data

        # Recalculate new velocities
        self.points.calculate_velocities(_ages, _cases, self.plates.data)
        self.slabs.calculate_velocities(_ages, _cases, self.plates.data)
        self.plate_torques.calculate_rms_velocity(_ages, _cases)

    def extract_data_through_time(
            self,
            ages = None,
            cases = None,
            plateIDs = None,
            type = "plates",
            var = "residual_torque_mag",
        ):
        """
        Function to extract data through time.

        :param ages:        ages of interest (default: None)
        :type ages:         float, int, list, numpy.ndarray
        :param cases:       cases of interest (default: None)
        :type cases:        str, list
        :param plateIDs:    plateIDs of interest (default: None)
        :type plateIDs:     int, float, list, numpy.ndarray
        """
        return self.plate_torques.extract_data_through_time(ages, cases, plateIDs, type, var)