# Standard libraries
import logging

# Third-party libraries
import numpy as _numpy
import matplotlib.pyplot as plt
from tqdm import tqdm as _tqdm

# Plato libraries
from . import utils_data
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

        # Organise dictionaries to store optimal values for slab pull coefficient, viscosity and normalised residual torque magnitude for each unique combination of age, case and plate
        # The index of each plate corresponds to the index in the plate data arrays.
        # The last entry in the arrays is for the global value.
        self.opt_sp_const = {age: {case: self.settings.options[case]["Slab pull constant"] for case in self.settings.cases} for age in self.settings.ages}
        self.opt_visc = {age: {case: self.settings.options[case]["Mantle viscosity"] for case in self.settings.cases} for age in self.settings.ages}

        # for age in self.settings.ages:
        #     for case in self.settings.cases:
        #         self.opt_sp_const[age][case] = _numpy.ones(len(self.plates.data[age][case].plateID) + 1) * self.settings.options[case]["Slab pull constant"]
        #         self.opt_visc[age][case] = _numpy.ones(len(self.plates.data[age][case].plateID) + 1) * self.settings.options[case]["Mantle viscosity"]

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
        _data = self.plates.data[age][case]

        _plateIDs = utils_data.select_plateIDs(plateIDs, self.plates.data[age][case].plateID)

        if plateIDs is not None:
            _data = _data[_data["plateID"].isin(_plateIDs)]

        if minimum_plate_area is not None:
            _data = _data[_data["area"] > minimum_plate_area]

        # Get total area
        total_area = _data["area"].sum()
            
        driving_mag = _numpy.zeros_like(sp_const_grid); 
        residual_mag = _numpy.zeros_like(sp_const_grid); 

        # Get torques
        for k, _ in enumerate(_data.plateID):
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

        # Assign optimal values to last entry in arrays
        self.opt_sp_const[age][case][-1] = opt_sp_const
        self.opt_visc[age][case][-1] = opt_visc

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

        return self.opt_sp_const[age][case][-1], self.opt_visc[age][case][-1], residual_mag_normalised, (opt_i, opt_j)
    
    def optimise_slab_pull_coefficient(
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
        # Select ages, if not provided
        _ages = utils_data.select_ages(age, self.settings.ages)

        # Select cases, if not provided
        _cases = utils_data.select_cases(case, self.settings.reconstructed_cases)

        for _age in _tqdm(_ages, desc="Optimising slab pull coefficient"):
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
                        # self.opt_sp_const[_age][_case][_data.index[k]] = opt_sp_const
    
    def invert_residual_torque(
            self,
            age = None, 
            case = None, 
            plateIDs = None, 
            parameter = "Slab pull constant",
            grid_size = 500, 
            viscosity = 1.23e20, 
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
        """
        # Raise error if parameter is not slab pull coefficient or mantle viscosity
        if parameter not in ["Slab pull constant", "Mantle viscosity"]:
            raise ValueError("Free parameter must be 'Slab pull constant' or 'Mantle viscosity")
        
        # Select ages, if not provided
        _ages = utils_data.select_ages(age, self.settings.ages)

        # Select cases, if not provided
        _cases = utils_data.select_cases(case, self.settings.reconstructed_cases)

        # Define plateIDs, if not provided
        _plateIDs = utils_data.select_plateIDs(
            plateIDs, 
            [101,   # North America
            201,    # South America
            301,    # Eurasia
            501,    # India
            801,    # Australia
            802,    # Antarctica
            901,    # Pacific
            902,    # Farallon
            911,    # Nazca
            919,    # Phoenix
            926,    # Izanagi
            ]
        )

        # Define constants
        constants = _numpy.arange(8.5, 13.5, .1)

        driving_torque_opt_stack = {_case: {_plateID: _numpy.zeros((len(constants), len(_ages))) for _plateID in _plateIDs} for _case in _cases}
        residual_torque_opt_stack = {_case: {_plateID: _numpy.zeros((len(constants), len(_ages))) for _plateID in _plateIDs} for _case in _cases}

        for i, constant in _tqdm(enumerate(constants), desc="Optimising slab pull coefficient"):
            logging.info(f"Optimising for {constant}")
            _data = {}

            # Calculate the torques the normal way
            self.plate_torques.calculate_slab_pull_torque(ages=_ages, cases=_cases, plateIDs=_plateIDs)
            self.plate_torques.calculate_driving_torque(ages=_ages, cases=_cases, plateIDs=_plateIDs)
            self.plate_torques.calculate_residual_torque(ages=_ages, cases=_cases, plateIDs=_plateIDs)

            # Loop through ages
            for _age in _ages:
                for _case in _cases:
                    _data[_age] = {}
                    _data[_age][_case] = self.plate_torques.slabs.data[_age][_case].copy()

                    # Modify the magnitude of the slab pull force using the 2D dot product of the residual force and the slab pull force and the constant
                    _data[_age][_case]["slab_pull_force_mag"] -= (
                        _data[_age][_case]["residual_force_lat"] * _data[_age][_case]["slab_pull_force_lat"] + \
                        _data[_age][_case]["residual_force_lon"] * _data[_age][_case]["slab_pull_force_lon"]
                    ) * 10**-constant

                    # Decompose the slab pull force into latitudinal and longitudinal components using the trench normal azimuth
                    _data[_age][_case]["slab_pull_force_lat"] = _numpy.cos(_numpy.deg2rad(_data[_age][_case]["trench_normal_azimuth"])) * _data[_age][_case]["slab_pull_force_mag"]
                    _data[_age][_case]["slab_pull_force_lon"] = _numpy.sin(_numpy.deg2rad(_data[_age][_case]["trench_normal_azimuth"])) * _data[_age][_case]["slab_pull_force_mag"]
                
            # Calculate the torques with the modified slab pull forces
            self.plate_torques.plates.calculate_torque_on_plates(_data, ages=_ages, cases=_cases, plateIDs=_plateIDs, torque_var="slab_pull", PROGRESS_BAR = False)
            self.plate_torques.calculate_driving_torque(ages=_ages, cases=_cases, plateIDs=_plateIDs, PROGRESS_BAR = False)
            self.plate_torques.calculate_residual_torque(ages=_ages, cases=_cases, plateIDs=_plateIDs, PROGRESS_BAR = False)

            # Extract the driving and residual torques
            _iter_driving_torque = self.plate_torques.extract_data_through_time(ages=_ages, cases=_cases, plateIDs=_plateIDs, var="driving_torque_mag")
            _iter_residual_torque = self.plate_torques.extract_data_through_time(ages=_ages, cases=_cases, plateIDs=_plateIDs, var= "residual_torque_mag")

            for _case in _cases:
                for _plateID in _plateIDs:
                    driving_torque_opt_stack[_case][_plateID][i, :] = _iter_driving_torque[_case][_plateID].values
                    residual_torque_opt_stack[_case][_plateID][i, :] = _iter_residual_torque[_case][_plateID].values

        # Get minimum residual torque for each unique combination of case and plate ID.
        minimum_residual_torque = {_case: {_plateID: None for _plateID in _plateIDs} for _case in _cases}
        opt_constant = {_case: {_plateID: None for _plateID in _plateIDs} for _case in _cases}
        for _case in _cases:
            for _plateID in _plateIDs:
                minimum_residual_torque[_case][_plateID] = _numpy.min(residual_torque_opt_stack[_plateID]/driving_torque_opt_stack[_plateID], axis = 0)
                opt_index = _numpy.argmin(residual_torque_opt_stack[_plateID]/driving_torque_opt_stack[_plateID], axis = 0)
                opt_constant[_case][_plateID] = constants[opt_index]

        # Calculate optimal slab pull constant and store in slab data
        for k, _age in enumerate(_ages):
            for _case in _cases:
                for _plateID in _plateIDs:
                    # Select data
                    _data = self.slabs.data[_age][_case]
                    _data = _data[_data["lower_plateID"] == _plateID]

                    # Calculate slab pull constant
                    _data["slab_pull_constant"] = (
                        _data["slab_pull_force_mag"] - (
                            _data[_age][_case]["residual_force_lat"] * _data[_age][_case]["slab_pull_force_lat"] + \
                            _data[_age][_case]["residual_force_lon"] * _data[_age][_case]["slab_pull_force_lon"]
                        ) * 10**-constant
                    ) / _data["slab_pull_force_mag"]

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