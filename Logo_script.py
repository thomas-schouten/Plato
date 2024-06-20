# %%
import numpy as np
import gplately
import cmcrameri as cmc
import cartopy.crs as ccrs
import matplotlib.pyplot as plt

# %%
time = 120

gdownload = gplately.DataServer("Muller2016")
rotations, topologies, polygons = gdownload.get_plate_reconstruction_files()
coastlines, continents, COBs = gdownload.get_topology_geometries()
age_grid = gdownload.get_age_grid(time=time)

model = gplately.PlateReconstruction(rotations, topologies, polygons)

gplot = gplately.PlotTopologies(model, coastlines=coastlines, continents=continents, COBs=COBs, time=time)

# %%
fig, ax = plt.subplots(subplot_kw={'projection': ccrs.Orthographic(central_longitude=-10)}, figsize=(10, 10))
gplot.plot_coastlines(ax, edgecolor='None', facecolor='grey', alpha=0.1)
gplot.plot_all_topologies(ax, linewidth=1.5, color='black')
age_grid.imshow(ax, cmap="cmc.lajolla_r", vmin=0, vmax=250)
# gplot.plot_plate_motion_vectors(ax, color='black', linewidth=0.5)
# gplot.plot_subduction_teeth(ax, color='black', linewidth=0.5)
# %%
