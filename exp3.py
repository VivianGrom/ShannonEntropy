#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  3 16:03:25 2025

@author: viviangrom

In this experiment I changed the uplift values of Run Experiments 1 and 2 following the
table I included on the paper (but also below). I ran once for each experiment.

	1st run	- Range (m/yr)     - 2nd run - Range (m/yr)
A	500 kyr	- 0.001 to 0.0015  - 500 kyr - 0.0015 to 0.001
B	500 kyr	- 0.001 to 0.002   - 500 kyr - 0.002 to 0.001
C	500 kyr	- 0.001 to 0.0025  - 500 kyr - 0.0025 to 0.001
D	500 kyr	- 0.001 to 0.003   - 500 kyr - 0.003 to 0.001
E	500 kyr	- 0.0015 to 0.001  - 500 kyr - 0.0015 to 0.001
F	500 kyr	- 0.002 to 0.001   - 500 kyr - 0.002 to 0.001
G	500 kyr	- 0.001 to 0.0015  - 500 kyr - 0.001 to 0.0015
H	500 kyr	- 0.001 to 0.002   - 500 kyr - 0.001 to 0.002
I	500 kyr	- 0.001 to 0.0015  - 500 kyr - 0.001 to 0.002
J	500 kyr	- 0.0015 to 0.001  - 500 kyr - 0.002 to 0.001

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

# # uplift rate
# uplift = 1e-3 			# (m/yr)
# uplift_per_step=uplift*timestep

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


# Define a linear uplift gradient across the columns (west to east direction)
uplift_gradient = np.linspace(0.001, 0.001, n_colms)  # Linear gradient from 0.001 to 0.005 m/yr
uplift_grid = np.tile(uplift_gradient, n_rows)  # Extend to all rows

# Assign the uplift rate to a field on the grid
grid.add_field("node", "uplift_rate", uplift_grid)


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
	grid.at_node['bedrock__elevation'][grid.core_nodes] += grid.at_node['uplift_rate'][grid.core_nodes] * timestep
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

# uplift_gradient = np.linspace(0.001, 0.0015, n_colms)  # A: Reverse gradient from 1e-3 to 0
# uplift_gradient = np.linspace(0.001, 0.002, n_colms)  # B: Reverse gradient from 1e-3 to 0
# uplift_gradient = np.linspace(0.001, 0.0025, n_colms)  # C: Reverse gradient from 1e-3 to 0
# uplift_gradient = np.linspace(0.001, 0.003, n_colms)  # D: Reverse gradient from 1e-3 to 0
# uplift_gradient = np.linspace(0.0015, 0.001, n_colms)  # E: Reverse gradient from 1e-3 to 0
# uplift_gradient = np.linspace(0.002, 0.001, n_colms)  # F: Reverse gradient from 1e-3 to 0
# uplift_gradient = np.linspace(0.001, 0.0015, n_colms)  # G: Reverse gradient from 1e-3 to 0
# uplift_gradient = np.linspace(0.001, 0.002, n_colms)  # H: Reverse gradient from 1e-3 to 0
# uplift_gradient = np.linspace(0.001, 0.0015, n_colms)  # I: Reverse gradient from 1e-3 to 0
uplift_gradient = np.linspace(0.0015, 0.001, n_colms)  # J: Reverse gradient from 1e-3 to 0
uplift_grid = np.tile(uplift_gradient, n_rows)  # Extend to all rows
grid.at_node['uplift_rate'][:] = uplift_grid  # Update the grid field

# time parameters
total_time = 100000		# (yr)
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
    	grid.at_node['bedrock__elevation'][grid.core_nodes] += grid.at_node['uplift_rate'][grid.core_nodes] * timestep
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
    
# %% Run experiments

# uplift_gradient = np.linspace(0.0015, 0.001, n_colms)  # A: Reverse gradient from 1e-3 to 0
# uplift_gradient = np.linspace(0.002, 0.001, n_colms)  # B: Reverse gradient from 1e-3 to 0
# uplift_gradient = np.linspace(0.0025, 0.001, n_colms)  # C: Reverse gradient from 1e-3 to 0
# uplift_gradient = np.linspace(0.003, 0.001, n_colms)  # D: Reverse gradient from 1e-3 to 0
# uplift_gradient = np.linspace(0.0015, 0.001, n_colms)  # E: Reverse gradient from 1e-3 to 0
# uplift_gradient = np.linspace(0.002, 0.001, n_colms)  # F: Reverse gradient from 1e-3 to 0
# uplift_gradient = np.linspace(0.001, 0.0015, n_colms)  # G: Reverse gradient from 1e-3 to 0
# uplift_gradient = np.linspace(0.001, 0.002, n_colms)  # H: Reverse gradient from 1e-3 to 0
# uplift_gradient = np.linspace(0.001, 0.002, n_colms)  # I: Reverse gradient from 1e-3 to 0
uplift_gradient = np.linspace(0.002, 0.001, n_colms)  # J: Reverse gradient from 1e-3 to 0
uplift_grid = np.tile(uplift_gradient, n_rows)  # Extend to all rows
grid.at_node['uplift_rate'][:] = uplift_grid  # Update the grid field

# time parameters
total_time = 100000		# (yr)
timestep = 10			# (yr)
ndt = int(total_time // timestep) # calculation for loop
times = 5
# grid_snapshots_topo = []  # List to store mg1 state at each timestep
# grid_snapshots_soil = []

for cycle in range (times):
    # iterate through!
    for i in range(ndt+1):
        
    	elapsed_time = i*timestep

    	# add uplift
    	grid.at_node['bedrock__elevation'][grid.core_nodes] += grid.at_node['uplift_rate'][grid.core_nodes] * timestep
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

# # %% post-run plotting
# plt.figure()
# imshow_grid(grid,"topographic__elevation",var_name="Elevation",var_units="m",cmap=cmap_terrain)
# plt.show()

# plt.figure()
# imshow_grid(grid,"soil__depth",var_name="Soil depth",var_units="m",cmap=cmap_soil)
# plt.show()

# for i in range(5):
#     plt.figure()
#     imshow_grid(grid,grid_snapshots_topo[i],var_name="Elevation",var_units="m",cmap=cmap_terrain, vmax=300)
#     plt.show()
    
#     plt.figure()
#     imshow_grid(grid,grid_snapshots_soil[i],var_name="Soil depth",var_units="m",cmap=cmap_soil, vmax=35)
#     plt.show()

#%% Test
# Example setup for the class
grid = grid  # Replace with your RasterModelGrid object
matrices = grid_snapshots_soil  # Example matrices, replace with your actual data
k = None  # or some integer value
pixel = None
pixel_stream = None
pixel_hillslope = None
stream_threshold = 20
slope_threshold = 0.1
filter_size = 2
buffer_scaling_factor = 0.004
minimum_buffer_size = 0.05
time_steps = [700, 1200, 1700, 2200, 2700, 3200, 3700, 4200, 4700, 5200]

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
tot_stream_mask = entropy_calculator.differentiate_hillslopes_and_streams()

entropies = entropy_calculator.pixel_entropy_across_experiments()
buffered_stream_mask = entropy_calculator.differentiate_hillslopes_and_streams_buffer()
buffered_stream_mask_scaled = entropy_calculator.differentiate_hillslopes_and_streams_scaled_buffer()
mskd_ent_matrix_stream, mskd_ent_matrix_hillslope, _, _ = entropy_calculator.calculate_shannon_entropy_mask(stream_mask = tot_stream_mask)
entropies_stream_array, entropies_hillslope_array = entropy_calculator.pixel_entropy_across_experiments_mask(mask = tot_stream_mask)
  
# I ran for topo and soil for each experiment and then plotted
# Example setup for the class
grid = grid  # Replace with your RasterModelGrid object
matrices = grid_snapshots_topo  # Example matrices, replace with your actual data
k = None  # or some integer value
pixel = None
pixel_stream = None
pixel_hillslope = None
stream_threshold = 3
slope_threshold = 0.1
filter_size = 2
buffer_scaling_factor = 0.004
minimum_buffer_size = 0.05
time_steps = [700, 1200, 1700, 2200, 2700, 3200, 3700, 4200, 4700, 5200]


# Create an instance of Spatial_Autocorrelation 
spatial_autocorr_calculator = Spatial_Autocorrelation(
    grid=grid,
    matrices=matrices,    
    time_steps=time_steps
)

# Call the method    
moran_values, geary_values, moran_p_values, geary_p_values = spatial_autocorr_calculator.plot_morans_i_and_gearys_c()

plt.figure()
imshow_grid(grid,"uplift_rate",var_name="Uplift",var_units="m/yr",cmap=cmap_terrain, at="node")
plt.show()


# Calculate mean and range for both metrics
mean_moran = np.mean(moran_values)
max_moran = np.max(moran_values)
min_moran = np.min(moran_values)   # Max - Min
mean_geary = np.mean(geary_values)
max_geary = np.max(geary_values)
min_geary = np.min(geary_values)  # Max - Min

# Print results for reference
print(f"Topo: Moran's I - Mean: {mean_moran:.3f}, Max: {max_moran:.3f}, Min: {min_moran:.3f}")
print(f"Topo: Geary's C - Mean: {mean_geary:.3f}, Max: {max_geary:.3f}, Min: {min_geary:.3f}")

# Example setup for the class
grid = grid  # Replace with your RasterModelGrid object
matrices = grid_snapshots_soil  # Example matrices, replace with your actual data
k = None  # or some integer value
pixel = None
pixel_stream = None
pixel_hillslope = None
stream_threshold = 3
slope_threshold = 0.1
filter_size = 2
buffer_scaling_factor = 0.004
minimum_buffer_size = 0.05
time_steps = [700, 1200, 1700, 2200, 2700, 3200, 3700, 4200, 4700, 5200]


# Create an instance of Spatial_Autocorrelation 
spatial_autocorr_calculator = Spatial_Autocorrelation(
    grid=grid,
    matrices=matrices,    
    time_steps=time_steps
)

# Call the method    
moran_values, geary_values, moran_p_values, geary_p_values = spatial_autocorr_calculator.plot_morans_i_and_gearys_c()

plt.figure()
imshow_grid(grid,"uplift_rate",var_name="Uplift",var_units="m/yr",cmap=cmap_terrain, at="node")
plt.show()


# Calculate mean and range for both metrics
mean_moran = np.mean(moran_values)
max_moran = np.max(moran_values)
min_moran = np.min(moran_values)   # Max - Min
mean_geary = np.mean(geary_values)
max_geary = np.max(geary_values)
min_geary = np.min(geary_values)  # Max - Min

# Print results for reference
print(f"Soil: Moran's I - Mean: {mean_moran:.3f}, Max: {max_moran:.3f}, Min: {min_moran:.3f}")
print(f"Soil: Geary's C - Mean: {mean_geary:.3f}, Max: {max_geary:.3f}, Min: {min_geary:.3f}")

# =============================================================================
#%% Plotting the mean and range of Experiment 3 metrics
# Data for each experiment
# experiments = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']
# metrics = {
#     "Topo": {
#         "Moran's I Mean": [0.726, 0.725, 0.725, 0.728, 0.702, 0.693, 0.747, 0.762, 0.751],
#         "Moran's I Max": [0.751, 0.765, 0.774, 0.782, 0.711, 0.707, 0.762, 0.782, 0.774],
#         "Moran's I Min": [0.704, 0.679, 0.664, 0.664, 0.680, 0.670, 0.711, 0.722, 0.711],
#         "Geary's C Mean": [0.258, 0.259, 0.258, 0.255, 0.275, 0.282, 0.240, 0.226, 0.236],
#         "Geary's C Max": [0.279, 0.299, 0.310, 0.309, 0.293, 0.301, 0.267, 0.257, 0.267],
#         "Geary's C Min": [0.236, 0.224, 0.216, 0.210, 0.268, 0.270, 0.229, 0.210, 0.218],
#     },
#     "Soil": {
#         "Moran's I Mean": [0.249, 0.194, 0.186, 0.201, 0.323, 0.293, 0.262, 0.232, 0.254],
#         "Moran's I Max": [0.702, 0.629, 0.555, 0.474, 0.822, 0.687, 0.702, 0.629, 0.702],
#         "Moran's I Min": [0.138, 0.091, 0.115, 0.076, 0.021, 0.165, 0.158, 0.135, 0.095],
#         "Geary's C Mean": [0.753, 0.807, 0.817, 0.803, 0.680, 0.715, 0.738, 0.769, 0.746],
#         "Geary's C Max": [0.865, 0.910, 0.893, 0.923, 0.979, 0.845, 0.843, 0.865, 0.904],
#         "Geary's C Min": [0.300, 0.372, 0.444, 0.526, 0.183, 0.324, 0.300, 0.372, 0.300],
#     }
# }

# # Data for each experiment
# experiments = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
# metrics = {
#     "Topo": {
#         "Moran's I Mean": [0.806, 0.818, 0.827, 0.836, 0.779, 0.786, 0.823, 0.847, 0.827, 0.775],
#         "Moran's I Max": [0.832, 0.856, 0.876, 0.890, 0.834, 0.839, 0.865, 0.882, 0.874, 0.824],
#         "Moran's I Min": [0.736, 0.762, 0.786, 0.790, 0.700, 0.707, 0.736, 0.762, 0.736, 0.700],
#         "Geary's C Mean": [0.179, 0.169, 0.161, 0.152, 0.203, 0.197, 0.164, 0.143, 0.160, 0.207],
#         "Geary's C Max": [0.238, 0.215, 0.194, 0.196, 0.270, 0.264, 0.238, 0.215, 0.238, 0.270],
#         "Geary's C Min": [0.156, 0.134, 0.117, 0.104, 0.156, 0.151, 0.129, 0.114, 0.120, 0.165],
#     },
#     "Soil": {
#         "Moran's I Mean": [0.584, 0.555, 0.480, 0.411, 0.659, 0.663, 0.535, 0.472, 0.528, 0.652],
#         "Moran's I Max": [0.855, 0.828, 0.810, 0.787, 0.869, 0.858, 0.855, 0.828, 0.855, 0.869],
#         "Moran's I Min": [0.178, 0.249, 0.157, 0.141, 0.206, 0.194, 0.026, -0.009, 0.008, 0.203],
#         "Geary's C Mean": [0.415, 0.444, 0.520, 0.587, 0.340, 0.337, 0.464, 0.527, 0.471, 0.347],
#         "Geary's C Max": [0.820, 0.750, 0.842, 0.857, 0.792, 0.804, 0.972, 1.008, 0.990, 0.795],
#         "Geary's C Min": [0.143, 0.170, 0.192, 0.211, 0.129, 0.143, 0.143, 0.170, 0.143, 0.129],
#     }
# }


# # Plot 1: Topo
# fig1, ax1 = plt.subplots(figsize=(10, 6))
# ax2_1 = ax1.twinx()  # Create a twin y-axis for Geary's C

# # Plot Moran's I
# ax1.errorbar(experiments, metrics["Topo"]["Moran's I Mean"], 
#              yerr=[np.array(metrics["Topo"]["Moran's I Mean"]) - np.array(metrics["Topo"]["Moran's I Min"]), 
#                    np.array(metrics["Topo"]["Moran's I Max"]) - np.array(metrics["Topo"]["Moran's I Mean"])], 
#              fmt='o', color='black', capsize=5)

# # Plot Geary's C
# ax2_1.errorbar(experiments, metrics["Topo"]["Geary's C Mean"], 
#                yerr=[np.array(metrics["Topo"]["Geary's C Mean"]) - np.array(metrics["Topo"]["Geary's C Min"]), 
#                      np.array(metrics["Topo"]["Geary's C Max"]) - np.array(metrics["Topo"]["Geary's C Mean"])], 
#                fmt='o', color='grey', capsize=5)

# # Set labels and title for Topo
# ax1.set_title("Topography Metrics across Experiment 3: Moran's I and Geary's C", fontsize=17)
# ax1.set_ylabel("Moran's I", fontsize=18, color='blue')
# ax2_1.set_ylabel("Geary's C", fontsize=18, color='red')
# ax1.set_ylim([-1, 1])  # Moran's I axis range
# ax2_1.set_ylim([0, 2])  # Geary's C axis range

# # Color the ticks for each axis
# ax1.tick_params(axis='y', labelcolor='black')
# ax2_1.tick_params(axis='y', labelcolor='grey')
# ax1.tick_params(axis='x', labelsize=18)  # X and Y axis tick labels for ax1
# ax2_1.tick_params(axis='x', labelsize=18)  # X and Y axis tick labels for ax2_1


# # Plot 2: Soil
# fig2, ax2 = plt.subplots(figsize=(10, 6))
# ax2_2 = ax2.twinx()  # Create a twin y-axis for Geary's C

# # Plot Moran's I for Soil
# ax2.errorbar(experiments, metrics["Soil"]["Moran's I Mean"], 
#              yerr=[np.array(metrics["Soil"]["Moran's I Mean"]) - np.array(metrics["Soil"]["Moran's I Min"]), 
#                    np.array(metrics["Soil"]["Moran's I Max"]) - np.array(metrics["Soil"]["Moran's I Mean"])], 
#              fmt='o', color='black', label="Moran's I", capsize=5)

# # Plot Geary's C for Soil
# ax2_2.errorbar(experiments, metrics["Soil"]["Geary's C Mean"], 
#                yerr=[np.array(metrics["Soil"]["Geary's C Mean"]) - np.array(metrics["Soil"]["Geary's C Min"]), 
#                      np.array(metrics["Soil"]["Geary's C Max"]) - np.array(metrics["Soil"]["Geary's C Mean"])], 
#                fmt='o', color='grey', label="Geary's C", capsize=5)

# # Set labels and title for Soil
# ax2.set_title("Soil Metrics across Experiment 3: Moran's I and Geary's C", fontsize=17)
# ax2.set_ylabel("Moran's I", fontsize=18, color='black')
# ax2_2.set_ylabel("Geary's C", fontsize=18, color='grey')
# ax2.set_ylim([-1, 1])  # Moran's I axis range
# ax2_2.set_ylim([0, 2])  # Geary's C axis range


# # Color the ticks for each axis
# ax2.tick_params(axis='y', labelcolor='black')
# ax2_2.tick_params(axis='y', labelcolor='grey')
# ax2.tick_params(axis='x', labelsize=18)  # X and Y axis tick labels for ax1
# ax2_2.tick_params(axis='x', labelsize=18)  # X and Y axis tick labels for ax2_1


# plt.show()


# # Convert lists to arrays
# topo_array = np.stack(grid_snapshots_topo)  # shape: (time, y, x)
# soil_array = np.stack(grid_snapshots_soil)  # shape: (time, y, x)

# # Grid info
# nrows, ncols = n_rows, n_colms
# node_spacing = 30  # meters

# # Coordinates
# times = time_steps  # could be integers or actual datetime values
# y = np.arange(nrows) * node_spacing
# x = np.arange(ncols) * node_spacing

# # Create dataset
# ds = xr.Dataset(
#     {
#         "topographic__elevation": (("time", "y", "x"), topo_array),
#         "soil__depth": (("time", "y", "x"), soil_array),
#     },
#     coords={
#         "time": times,
#         "y": y,
#         "x": x
#     },
#     attrs={
#         "nrows": nrows,
#         "ncols": ncols,
#         "node_spacing": node_spacing,
#         "xy_units": "m",
#         "description": "Landlab RasterModelGrid snapshots"
#     }
# )

# # Save to NetCDF
# ds.to_netcdf("/Users/viviangrom/Documents/papers/paper2_entropy/nc_files/expA.nc")





    
    
    
    
    
    
    
    
    
