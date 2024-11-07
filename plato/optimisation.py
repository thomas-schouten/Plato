# Third-party libraries
import numpy as _numpy
import matplotlib.pyplot as plt
from tqdm import tqdm as _tqdm

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
            # Store plate torques object
            self.plate_torques = plate_torques

            # Create shortcuts to the settings, plates, slabs, points, grids, and globe objects
            self.settings = plate_torques.settings
            self.plates = plate_torques.plates
            self.slabs = plate_torques.slabs
            self.points = plate_torques.points
            self.grids = plate_torques.grids
            self.globe = plate_torques.globe

        # Organise dictionaries to store optimal values for slab pull coefficient and viscosity for each unique combination of age, case and plate
        self.opt_sp_const = {age: {case: None for case in self.settings.cases} for age in self.settings.ages}
        self.opt_visc = {age: {case: None for case in self.settings.cases} for age in self.settings.ages}
        for age in self.settings.ages:
            for case in self.settings.cases:
                self.opt_sp_const[age][case] = _numpy.ones_like(self.plates.data[age][case].plateID) * self.settings.options[case]["Slab pull constant"]
                self.opt_visc[age][case] = _numpy.ones_like(self.plates.data[age][case].plateID) * self.settings.options[case]["Mantle viscosity"]

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
        for k, _plateID in enumerate(_data._plateIDs):
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

        # Assign optimal values to dictionaries
        self.opt_sp_const[age][case] = opt_sp_const
        self.opt_visc[age][case] = opt_visc

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
        _data = self.plates.data[age][case].copy()

        _plateIDs = utils_data.select_plateIDs(plateIDs, self.plates.data[age][case])

        if plateIDs is not None:
            _data = _data[_data["plateID"].isin(_plateIDs)]

        if _data.empty:
            return _numpy.nan, _numpy.nan, _numpy.nan

        # Initialise dictionary to store optimal slab pull coefficient per plate
        opt_sp_consts = {None for _ in _plateIDs}
        
        # Loop through plates
        for k, plateID in enumerate(_data.plateID):
            residual_x = _numpy.zeros_like(sp_consts)
            residual_y = _numpy.zeros_like(sp_consts)
            residual_z = _numpy.zeros_like(sp_consts)

            if self.settings.options[case]["Slab pull torque"] and "slab_pull_torque_x" in _data.columns:
                residual_x -= _data.slab_pull_torque_x.iloc[k] * sp_consts / self.settings.options[case]["Slab pull constant"]
                residual_y -= _data.slab_pull_torque_y.iloc[k] * sp_consts / self.settings.options[case]["Slab pull constant"]
                residual_z -= _data.slab_pull_torque_z.iloc[k] * sp_consts / self.settings.options[case]["Slab pull constant"]

            # Add GPE torque
            if self.settings.options[case]["GPE torque"] and "GPE_torque_x" in _data.columns:
                residual_x -= _data.GPE_torque_x.iloc[k] * ones
                residual_y -= _data.GPE_torque_y.iloc[k] * ones
                residual_z -= _data.GPE_torque_z.iloc[k] * ones

            # Compute magnitude of driving torque
            driving_mag = _numpy.sqrt(residual_x**2 + residual_y**2 + residual_z**2)
            
            # Add slab bend torque
            if self.settings.options[case]["Slab bend torque"] and "slab_bend_torque_x" in _data.columns:
                residual_x -= _data.slab_bend_torque_x.iloc[k] * ones
                residual_y -= _data.slab_bend_torque_y.iloc[k] * ones
                residual_z -= _data.slab_bend_torque_z.iloc[k] * ones

            # Add mantle drag torque
            if self.settings.options[case]["Mantle drag torque"] and "mantle_drag_torque_x" in _data.columns:
                residual_x -= _data.mantle_drag_torque_x.iloc[k] * viscosity / self.settings.options[case]["Mantle viscosity"]
                residual_y -= _data.mantle_drag_torque_y.iloc[k] * viscosity / self.settings.options[case]["Mantle viscosity"]
                residual_z -= _data.mantle_drag_torque_z.iloc[k] * viscosity / self.settings.options[case]["Mantle viscosity"]

            # Compute magnitude of residual
            residual_mag = _numpy.sqrt(residual_x**2 + residual_y**2 + residual_z**2)

            # Find minimum residual torque
            residual_mag_min = residual_mag[_numpy.argmin(_numpy.log10(residual_mag/driving_mag))]

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

            # Find optimal driving torque
            driving_mag_min = driving_mag[_numpy.argmin(_numpy.log10(residual_mag/driving_mag))]

            # Find optimal slab pull coefficient
            opt_sp_const = sp_consts[_numpy.argmin(_numpy.log10(residual_mag/driving_mag))]

        return opt_sp_const, driving_mag_min, residual_mag_min
    
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
        """
        # Select ages if not provided
        _ages = utils_data.select_ages(ages, self.settings.ages)

        # Select cases if not provided
        _cases = utils_data.select_cases(cases, self.settings.cases)

        # Loop through ages and cases
        for _age in _tqdm(_ages, desc="Optimising torques"):
            for _case in _cases:
                # Select plateIDs
                _plateIDs = utils_data.select_plateIDs(plateIDs, self.plates.data[_age][_case])

                # Select plate data
                _plate_data = self.plates.data[_age][_case]
                if plateIDs is not None:
                    _plate_data = _plate_data[_plate_data["plateID"].isin(_plateIDs)]
                
                for _plateID in _plate_data.plateID:
                    # Select optimal slab pull coefficient and viscosity
                    if optimisation_age or optimisation_case or optimisation_plateID is None:
                        opt_sp_const = self.opt_sp_const[_age][_case]
                        opt_visc = self.opt_visc[_age][_case]
                    else:
                        opt_sp_const = self.opt_sp_const[optimisation_age][optimisation_case][optimisation_plateID] 
                        opt_visc = self.opt_visc[optimisation_age][optimisation_case][optimisation_plateID]

                    # Optimise plate torque
                    for coord in ["x", "y", "z", "mag"]:
                        _plate_data[f"slab_pull_torque_{coord}"] *= opt_sp_const / self.settings.options[_case]["Slab pull constant"]
                        _plate_data[f"mantle_drag_torque_{coord}"] *= opt_visc / self.settings.options[_case]["Mantle viscosity"]

                    # Select slab data
                    _slab_data = self.slabs.data[_age][_case]
                    if plateIDs is not None:
                        _slab_data = _slab_data[_slab_data["lower_plateID"].isin(_plateIDs)]

                    for _plateID in _slab_data.lower_plateID:
                        # Select optimal slab pull coefficient
                        if optimisation_age or optimisation_case or optimisation_plateID is None:
                            opt_sp_const = self.opt_sp_const[_age][_case]
                        else:
                            opt_sp_const = self.opt_sp_const[optimisation_age][optimisation_case][optimisation_plateID] 

                        # Optimise slab pull force
                        for coord in ["lat", "lon", "mag"]:
                            _slab_data[f"slab_pull_force_{coord}"] *= opt_sp_const / self.settings.options[_case]["Slab pull constant"]

                    # Select point data
                    _point_data = self.points.data[_age][_case]
                    if plateIDs is not None:
                        _point_data = _point_data[_point_data["plateID"].isin(_plateIDs)]

                    for _plateID in _point_data.plateID:
                        # Select optimal slab pull coefficient
                        if optimisation_age or optimisation_case or optimisation_plateID is None:
                            opt_visc = self.opt_visc[_age][_case]
                        else:
                            opt_visc = self.opt_visc[optimisation_age][optimisation_case][optimisation_plateID] 

                        # Optimise point torque
                        for coord in ["lat", "lon", "mag"]:
                            _point_data[f"mantle_drag_force_{coord}"] *= opt_visc / self.settings.options[_case]["Mantle viscosity"]

        # Recalculate driving and residual torques
        self.plate_torques.calculate_driving_torque(_ages, _cases)
        self.plate_torques.calculate_residual_torque(_ages, _cases)