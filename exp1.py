#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 13:30:53 2024

@author: viviangrom
"""
#%% Import libraries
import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
from scipy.ndimage import maximum_filter
from landlab import RasterModelGrid, imshow_grid, imshowhs_grid
from landlab.components import (
    FlowAccumulator,
    StreamPowerEroder,
    PriorityFloodFlowRouter,
    ExponentialWeatherer,
    DepthDependentTaylorDiffuser,
    SpaceLargeScaleEroder,
    SteepnessFinder,
    ChiFinder,
    ChannelProfiler
    )
from landlab.plot.drainage_plot import drainage_plot
from libpysal.weights import lat2W
from esda.moran import Moran
from esda.geary import Geary
import xarray as xr
import numpy as np

from landlab.io.netcdf import write_netcdf, read_netcdf

from ent_sautocor_class import Shannon_Entropy
from ent_sautocor_class import Spatial_Autocorrelation

# other things
import os
import time, warnings, copy
warnings.filterwarnings('ignore')

# creating printing function
def print_model_progress(start_time,current_time,elapsed_time,total_t,i,ndt):
    
    runtime_per_step = (current_time - start_time)/i
    runtime_remaining = runtime_per_step * (ndt-i)
    runtime_remaining_mins = round(runtime_remaining/60,2)
   
    print("Years elapsed :   ", elapsed_time/1000, " ka (", round(elapsed_time*100/total_t,2), "%)")
    print("Runtime remaining :   ", runtime_remaining_mins, " min")
    print("--------------")
    
    return None


#%% Run a model
# Setting up our parameters
# grid parameters

n_rows = 100 			# number of rows in grid
n_colms = 100 			# number of columns in grid
node_spacing = 30		# spacing between nodes (m)

# time parameters
total_time = 200000		# (yr)
timestep = 10			# (yr)
ndt = int(total_time // timestep) # calculation for loop
print_output = 1000 	# time between printing

# uplift rate
uplift = 1e-3 			# (m/yr)
uplift_per_step=uplift*timestep

# PriorityFloodFLowRouter parameters
flow_metric = "D8"
phi_FR = 0.0

# ExponentalWeatherer parameters
w_0 = 3e-4
w_star = 0.44
# DepthDependentTaylorDiffuser parameters
D = 0.01
H_transport = 0.1

# SpaceLargeScaleEroder parameters
k_sed = 1e-4
k_br = 1e-5
phi_sp = 0.0
F_f_sp = 0.0
H_star = 1.0
v_s = 1
m_sp = 0.45
n_sp = 1.0


# SteepnessFinder parameters
reference_concavity = 0.5
min_drainage_area = 1000.0

# ChannelProfiler parameters
number_of_watersheds = 1
minimum_channel_threshold=100

# %% Creating our grid
# create the raster grid that is n_rows by n_colms with space between nodes as node_spacing and units of m
grid = RasterModelGrid((n_rows, n_colms), node_spacing, xy_axis_units=('m', 'm'))

# add field "topographic elevation", "soil depth", and "bedrock elevation" to the grid
z = grid.add_ones("topographic__elevation", at="node", units = ['m','m'])
soil = grid.add_zeros("soil__depth", at="node", units = ['m','m'])
bedrock = grid.add_zeros("bedrock__elevation", at="node", units = ['m','m'])
soil_production = grid.add_zeros("soil_production__rate", at="node")

# set constant random seed for consistent topographic roughness
np.random.seed(seed=5000)

# impose topography values on model grid
grid["node"]["topographic__elevation"]+= np.random.rand(grid.number_of_nodes) / 100
# set bedrock elevation equal to the topography
grid.at_node["bedrock__elevation"][:] = grid.at_node["topographic__elevation"]
# then update topography to have soil too (adding soil depth)
grid.at_node["topographic__elevation"][:] += grid.at_node["soil__depth"]
# then set outlet in lower lefthand corner


# boundary conditions - closing everything, besides node 1
grid.set_watershed_boundary_condition_outlet_id(1, z)


# %% Instantiating our components
# Flow routing
fr = PriorityFloodFlowRouter(
	grid,
	surface = "topographic__elevation",
	flow_metric = flow_metric,
	depression_handler = "fill",
	accumulate_flow = True,
	separate_hill_flow = True,
	accumulate_flow_hill = True,
	)
# weathering
ew = ExponentialWeatherer(
	grid,
	soil_production_maximum_rate = w_0,
	soil_production_decay_depth = w_star,
	)

# diffusion
ddTd = DepthDependentTaylorDiffuser(
	grid,
	soil_transport_decay_depth = H_transport,
	nterms = 2,
	soil_transport_velocity = D,
	dynamic_dt = True,
	if_unstable = "raise",
	courant_factor = 0.9,
	)

# metrics
# stream power
sp = SpaceLargeScaleEroder(
	grid,
	K_sed = k_sed,
	K_br = k_br,
	F_f = F_f_sp,
	phi = phi_sp,
	H_star = H_star,
	v_s = v_s,
	m_sp = m_sp,
	n_sp = n_sp,
	)

# %% plot initial grid
# set color maps for plotting
cmap_terrain = mpl.cm.get_cmap("terrain").copy()
cmap_soil = mpl.cm.get_cmap("YlOrBr").copy()

imshow_grid(grid,"topographic__elevation",var_name="Elevation",var_units="m",cmap=cmap_terrain)
plt.show()
imshow_grid(grid,"soil__depth",var_name="Soil Depth",var_units="m",cmap=cmap_soil)
plt.show()

# %% Run model until equilibrium
elapsed_time = 0 			# set elapsed model time to 0 years
start_time = time.time()	# set intial timestamp, used to print progress updates


for i in range(ndt+1):
	elapsed_time = i*timestep

	# add uplift
	grid.at_node['bedrock__elevation'][grid.core_nodes] += uplift_per_step
	grid.at_node['topographic__elevation'][:] = (grid.at_node["bedrock__elevation"] + grid.at_node["soil__depth"])

	# run components
	ew.run_one_step()
	ddTd.run_one_step(timestep)
	fr.run_one_step()
	sp.run_one_step(timestep)


	# Print progress and plot grids

	if i > 0 and i*timestep % print_output == 0:
		current_time = time.time()
		print_model_progress(start_time, current_time, elapsed_time, total_time, i, ndt)

# %% Run experiments

# time parameters
total_time = 50000		# (yr)
timestep = 10			# (yr)
ndt = int(total_time // timestep) # calculation for loop
times = 5
grid_snapshots_topo = []  # List to store mg1 state at each timestep
grid_snapshots_soil = []

for cycle in range (times):
    # iterate through!
    for i in range(ndt+1):
        
    	elapsed_time = i*timestep

    	# add uplift
    	grid.at_node['bedrock__elevation'][grid.core_nodes] += uplift_per_step
    	grid.at_node['topographic__elevation'][:] = (grid.at_node["bedrock__elevation"] + grid.at_node["soil__depth"])

    	# run components
    	ew.run_one_step()
    	ddTd.run_one_step(timestep)
    	fr.run_one_step()
    	sp.run_one_step(timestep)


    	# Print progress and plot grids

    	if i > 0 and i*timestep % print_output == 0:
    		current_time = time.time()
    		print_model_progress(start_time, current_time, elapsed_time, total_time, i, ndt)
            
    grid_snapshots_topo.append(grid.at_node['topographic__elevation'].reshape(grid.shape).copy())
    grid_snapshots_soil.append(grid.at_node['soil__depth'].reshape(grid.shape).copy())
    
    plt.figure()
    imshow_grid(grid,"topographic__elevation",var_name="Elevation",var_units="m",cmap=cmap_terrain)
    plt.show()

    plt.figure()
    imshow_grid(grid,"soil__depth",var_name="Soil depth",var_units="m",cmap=cmap_soil)
    plt.show()

# %% post-run plotting
plt.figure()
imshow_grid(grid,"topographic__elevation",var_name="Elevation",var_units="m",cmap=cmap_terrain)
plt.show()

plt.figure()
imshow_grid(grid,"soil__depth",var_name="Soil depth",var_units="m",cmap=cmap_soil)
plt.show()

for i in range(5):
    plt.figure()
    imshow_grid(grid,grid_snapshots_topo[i],var_name="Elevation",var_units="m",cmap=cmap_terrain, vmax=300)
    plt.show()
    
    plt.figure()
    imshow_grid(grid,grid_snapshots_soil[i],var_name="Soil depth",var_units="m",cmap=cmap_soil, vmax=35)
    plt.show()

#%% Test
# Example setup for the class
grid = grid  # Replace with your RasterModelGrid object
matrices = grid_snapshots_topo  # Example matrices, replace with your actual data
k = None  # or some integer value
pixel = None
pixel_stream = None
pixel_hillslope = None
stream_threshold = 20
slope_threshold = 0.1
filter_size = 2
buffer_scaling_factor = 0.0007
minimum_buffer_size = 0.05
time_steps = [250, 300, 350, 400, 450]

# Create an instance of Shannon_Entropy
entropy_calculator = Shannon_Entropy(
    grid=grid,
    matrices=matrices,
    k=k,
    pixel=pixel,
    pixel_stream=pixel_stream,
    pixel_hillslope=pixel_hillslope,
    stream_threshold=stream_threshold,
    slope_threshold=slope_threshold,
    filter_size=filter_size,
    buffer_scaling_factor=buffer_scaling_factor,
    minimum_buffer_size=minimum_buffer_size
)

# Call the method
entropy_matrix, probabilities = entropy_calculator.calculate_shannon_entropy(k=k)
entropies = entropy_calculator.pixel_entropy_across_experiments()
tot_stream_mask = entropy_calculator.differentiate_hillslopes_and_streams()
buffered_stream_mask = entropy_calculator.differentiate_hillslopes_and_streams_buffer()
buffered_stream_mask_scaled = entropy_calculator.differentiate_hillslopes_and_streams_scaled_buffer()

#This
entropies_stream_array, entropies_hillslope_array = entropy_calculator.pixel_entropy_across_experiments_mask(mask = tot_stream_mask)

mskd_ent_matrix_stream, mskd_ent_matrix_hillslope, _, _ = entropy_calculator.calculate_shannon_entropy_mask(stream_mask = tot_stream_mask)
mskd_ent_matrix_stream, mskd_ent_matrix_hillslope, _, _ = entropy_calculator.calculate_shannon_entropy_mask(stream_mask = buffered_stream_mask_scaled)



# Create an instance of Spatial_Autocorrelation 
spatial_autocorr_calculator = Spatial_Autocorrelation(
    grid=grid,
    matrices=matrices,    
    time_steps=time_steps
)

# Call the method    
spatial_autocorr_calculator.plot_morans_i_over_time()
spatial_autocorr_calculator.plot_gearys_c_over_time()


# Convert lists to arrays
topo_array = np.stack(grid_snapshots_topo)  # shape: (time, y, x)
soil_array = np.stack(grid_snapshots_soil)  # shape: (time, y, x)

# Grid info
nrows, ncols = n_rows, n_colms
node_spacing = 30  # meters

# Coordinates
times = time_steps  # could be integers or actual datetime values
y = np.arange(nrows) * node_spacing
x = np.arange(ncols) * node_spacing

# Create dataset
ds = xr.Dataset(
    {
        "topographic__elevation": (("time", "y", "x"), topo_array),
        "soil__depth": (("time", "y", "x"), soil_array),
    },
    coords={
        "time": times,
        "y": y,
        "x": x
    },
    attrs={
        "nrows": nrows,
        "ncols": ncols,
        "node_spacing": node_spacing,
        "xy_units": "m",
        "description": "Landlab RasterModelGrid snapshots"
    }
)

# Save to NetCDF
ds.to_netcdf("/Users/viviangrom/Documents/papers/paper2_entropy/nc_files/exp1.nc")

    
    
    
    
    
    
    
    
    
    
