# %%
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

# %%
litho = xr.open_dataset("/Users/thomas/Documents/_Data/Lithosphere/Lithoref18.nc")

# %%
plt.imshow(litho.LAB)
plt.colorbar()

# %%
litho