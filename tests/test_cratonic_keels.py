# %%
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import gplately
from gplately import pygplates

def reconstruct_raster(
        raster: xr.DataArray,
        reconstruction: gplately.PlateReconstruction,
        polygons: pygplates.FeatureCollection,
        reconstruction_time: int,
        target_variable: str = "z",
):
    """
    Function to reconstruct a raster using a plate reconstruction and a set of polygons.
    The raster is cut using the polygons and each polygon is then reconstructed using the plate reconstruction.
    More information on https://gplates.github.io/gplately/index.html#gplately.Raster

    :param raster:                  the raster to be reconstructed.
    :type raster:                   xarray.DataArray
    :param reconstruction:          the plate reconstruction to be used.
    :type reconstruction:           gplately.PlateReconstruction
    :param polygons:                the polygons to be used for cutting the raster.
    :type polygons:                 gplately.FeatureCollection or pygplates.FeatureCollection
    :param reconstruction_time:     the age to which to reconstruct the raster
    :type reconstruction_time:      int

    :return output_raster:          the reconstructed raster.
    :rtype output_raster:           xarray.DataArray
    """
    # Convert xarray.Dataset to gplately.Raster
    raster = gplately.Raster(
            plate_reconstruction=reconstruction,
            data=raster[target_variable].values,
            extent="global",    # equivalent to (-180, 180, -90, 90)
            origin="lower",     # or set extent to (-180, 180, -90, 90)

        )
    
    # Reconstruct the raster back to the reconstruction time
    reconstructed_raster = raster.reconstruct(time=reconstruction_time, partitioning_features=polygons)
    
    # Convert reconstructed raster back to xarray.Dataset
    output_raster = xr.Dataset(
        data_vars={target_variable: (["lat", "lon"], reconstructed_raster.data)},
        coords={"lat": (["lat"], reconstructed_raster.lats),
                "lon": (["lon"], reconstructed_raster.lons)}
    )

    return output_raster

# %%
seafloor_grids = {}
for age in np.arange(0, 151):
    seafloor_grids[age] = xr.open_dataset(f"/Users/thomas/Documents/_Plato/Plato/sample_data/M2016/seafloor_age_grids/M2016_SeafloorAgeGrid_{age}Ma.nc")
lithoref = xr.open_dataset("/Users/thomas/Documents/_Data/Lithosphere/Lithoref18.nc")

# %%
lithoref
# %%
plt.imshow(seafloor_grids[0].z, origin="lower")
plt.colorbar()
# %%
litho = lithoref.interp_like(seafloor_grids[0])
for var in litho.data_vars:
    litho[var].values = np.where(
        ~np.isnan(seafloor_grids[0].z.values),
        np.nan,
        litho[var].values
    )
# %%
plt.imshow(litho.LAB, origin="lower")
plt.colorbar()

# %%
# Interpolate back to the original resolution
litho_cut = litho.interp_like(lithoref)

# Modify the origin to "lower" by flipping the 'lat' axis
litho_cut = litho_cut.reindex(lat=list(reversed(litho_cut.lat)))

# %%
plt.imshow(litho_cut.LAB, origin="lower")
plt.colorbar()

# %%
litho_cut

# %%
# Load the reconstruction model
rotations = pygplates.RotationModel("/Users/thomas/Documents/_Plato/Plato/sample_data/M2016/gplates_files/M2016_rotations_Lr-Hb.rot")
topologies = pygplates.FeatureCollection("/Users/thomas/Documents/_Plato/Plato/sample_data/M2016/gplates_files/M2016_topologies.gpml")
polgyons = pygplates.FeatureCollection("/Users/thomas/Documents/_Plato/Plato/sample_data/M2016/gplates_files/M2016_polygons.gpml")
reconstruction_model = gplately.PlateReconstruction(rotations, topologies)

# %%
for age in [100]:#np.arange(0, 181):
    # if age == 100:
    #     continue

    reconstructed_da = {}
    for var, _var in zip(["LAB", "Elevation", "Moho"], ["LAB_depth", "elevation", "Moho_depth"]):
        reconstructed_litho = reconstruct_raster(litho_cut, reconstruction_model, polgyons, age, target_variable=var)
        reconstructed_da[_var] = np.abs(reconstructed_litho[var])
    
    reconstructed_ds = xr.Dataset(
        data_vars=reconstructed_da,
        coords={"lat": reconstructed_litho.lat, "lon": reconstructed_litho.lon}
    )

    # Interpolate to resolution of the seafloor grid
    reconstructed_ds = reconstructed_ds.interp_like(seafloor_grids[age])

    # Interpolate NaN values
    for var in reconstructed_ds.data_vars:
        reconstructed_ds[var] = reconstructed_ds[var].interpolate_na(dim="lat").interpolate_na(dim="lon")

    plt.imshow(reconstructed_ds.LAB_depth, origin="lower")
    plt.colorbar()
    plt.show()

    # Remove seafloor ages
    for var in reconstructed_ds.data_vars:
        reconstructed_ds[var].values = np.where(
            ~np.isnan(seafloor_grids[age].z.values),
            np.nan,
            reconstructed_ds[_var].values
        )

    # Interpolate back to the original resolution
    reconstructed_ds = reconstructed_ds.interp_like(lithoref)

    # Modify the origin to "lower" by flipping the 'lat' axis
    reconstructed_ds = reconstructed_ds.reindex(lat=list(reversed(reconstructed_ds.lat)))

    plt.imshow(reconstructed_ds.LAB_depth, origin="lower")
    plt.colorbar()
    plt.show()

    # reconstructed_ds.to_netcdf(f"/Users/thomas/Documents/_Plato/Plato/sample_data/M2016/continental_grids/M2016_ContinentalGrid_{age}Ma.nc")
    print(f"Reconstructed continental grids for {age} Ma")
# %%
reconstructed_ds
 #%%
seafloor_grid

# %%
age = 5
reconstructed_100ma = xr.open_dataset(f"/Users/thomas/Documents/_Plato/Plato/sample_data/M2016/continental_grids/M2016_ContinentalGrid_{age}Ma.nc")

plt.imshow(reconstructed_100ma.Moho, origin="lower")