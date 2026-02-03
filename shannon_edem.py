#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 11:20:26 2024

@author: viviangrom
"""

#%% Import libraries

import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
from scipy.ndimage import maximum_filter
from landlab import RasterModelGrid, imshow_grid

from ent_sautocor_class import Shannon_Entropy
from ent_sautocor_class import Spatial_Autocorrelation


def downsample_array(array, factor):
    """
    Downsample a 2D (masked) array by an integer factor using block averaging.

    Parameters:
    - array: 2D numpy masked array
    - factor: integer (e.g., 2, 3, 5) defining how much to reduce resolution

    Returns:
    - downsampled masked array
    """
    # Handle masked array properly
    data = np.ma.filled(array, np.nan)  # convert mask to NaN for averaging

    # Trim edges so shape is divisible by factor
    new_shape = (data.shape[0] // factor, data.shape[1] // factor)
    trimmed = data[:new_shape[0]*factor, :new_shape[1]*factor]

    # Reshape and average
    downsampled = np.nanmean(trimmed.reshape(new_shape[0], factor, new_shape[1], factor), axis=(1, 3))

    # Restore mask
    downsampled = np.ma.masked_invalid(downsampled)

    return downsampled



# # %% Load data

# arra_2010 = np.flipud(np.loadtxt(r'/Users/viviangrom/Documents/papers/paper2_entropy/raster_analysis/dem_cut6/arra_2010_6_3_mkd.asc', skiprows=6))
# lpc_2016 = np.flipud(np.loadtxt(r'/Users/viviangrom/Documents/papers/paper2_entropy/raster_analysis/dem_cut6/lpc_2016_6_mkd.asc', skiprows=6))
# fema_2018 = np.flipud(np.loadtxt(r'/Users/viviangrom/Documents/papers/paper2_entropy/raster_analysis/dem_cut6/fema_2018_6_mkd.asc', skiprows=6))


# # Create a dictionary of grid names and data
# grids = {
#     'arra_2010': arra_2010,
#     'lpc_2016': lpc_2016,
#     'fema_2018': fema_2018
# }

# # Assuming the files are 2D arrays with the same dimensions
# nrows, ncols = arra_2010.shape
# grid = RasterModelGrid((nrows, ncols), xy_spacing=(1, 1))  # Adjust xy_spacing as necessary

# def plot_all_grids(grid, grids):
#     cmap_terrain = mpl.cm.get_cmap("terrain").copy()
    
#     for name, data in grids.items():
#         imshow_grid(grid, data, var_name="Elevation", var_units="m", cmap=cmap_terrain)
#         plt.title(name)
#         plt.show()

# # Example call to the function
# plot_all_grids(grid, grids)

# matrices = [arra_2010, lpc_2016]
# time_steps = [2010, 2016]

# pixel = None
# pixel_stream = None
# pixel_hillslope = None
# stream_threshold = 50
# slope_threshold = 0.1
# filter_size = 2
# buffer_scaling_factor = 0.0007
# minimum_buffer_size = 0.05

# # Create an instance of Shannon_Entropy
# entropy_calculator = Shannon_Entropy(
#     grid=grid,
#     matrices=matrices,
#     pixel=pixel,
#     pixel_stream=pixel_stream,
#     pixel_hillslope=pixel_hillslope,
#     stream_threshold=stream_threshold,
#     slope_threshold=slope_threshold,
#     filter_size=filter_size,
#     buffer_scaling_factor=buffer_scaling_factor,
#     minimum_buffer_size=minimum_buffer_size
# )

# # Call the method
# entropy_matrix, probabilities = entropy_calculator.calculate_shannon_entropy()
# entropies = entropy_calculator.pixel_entropy_across_experiments()

# tot_stream_mask = entropy_calculator.differentiate_hillslopes_and_streams()
# buffered_stream_mask = entropy_calculator.differentiate_hillslopes_and_streams_buffer()
# buffered_stream_mask_scaled = entropy_calculator.differentiate_hillslopes_and_streams_scaled_buffer()



#%%


def load_dem_masked(path):
    """Load an ASCII DEM and mask all -9999 values."""
    data = np.flipud(np.loadtxt(path, skiprows=6))
    return np.ma.masked_equal(data, -9999)

# Load DEMs as masked arrays
arra_2010 = load_dem_masked(r'/Users/viviangrom/Documents/papers/paper2_entropy/raster_analysis/dem_cut6/arra_2010_6_3_mkd.asc')
lpc_2016  = load_dem_masked(r'/Users/viviangrom/Documents/papers/paper2_entropy/raster_analysis/dem_cut6/lpc_2016_6_mkd.asc')
fema_2018 = load_dem_masked(r'/Users/viviangrom/Documents/papers/paper2_entropy/raster_analysis/dem_cut6/fema_2018_6_mkd.asc')


# Downsample to lower resolution
factor = 1   # change this as needed
arra_2010 = downsample_array(arra_2010, factor)
lpc_2016  = downsample_array(lpc_2016, factor)
fema_2018 = downsample_array(fema_2018, factor)


# Store in dictionary
grids = {
    'arra_2010': arra_2010,
    'lpc_2016': lpc_2016,
    'fema_2018': fema_2018
}

# Example: your masked DEM (masked -9999 cells)
# nrows, ncols = arra_2010.shape
# grid = RasterModelGrid((nrows, ncols), xy_spacing=(1, 1))

nrows, ncols = arra_2010.shape
grid = RasterModelGrid((nrows, ncols), xy_spacing=(1 * factor, 1 * factor))


# Make all outer edges closed
grid.set_closed_boundaries_at_grid_edges(True, True, True, True)

# Add the DEM as a field
grid.add_field("topographic__elevation", arra_2010, at="node", clobber=True)

# Now close any node where the DEM is masked
mask = arra_2010.mask  # True = masked
grid.status_at_node[mask.flatten()] = grid.BC_NODE_IS_CLOSED

print("Total closed nodes:", np.sum(grid.status_at_node == grid.BC_NODE_IS_CLOSED))
print("Active nodes:", np.sum(grid.status_at_node == grid.BC_NODE_IS_CORE))

plt.rcParams.update({
    'font.size': 14,          # Base font size
    'axes.titlesize': 8,     # Title font size
    'axes.labelsize': 14,     # Axis label font size
    'xtick.labelsize': 16,    # X-tick font size
    'ytick.labelsize': 16,    # Y-tick font size
})


def plot_all_grids(grid, grids):
    cmap_terrain = mpl.cm.get_cmap("terrain").copy()
    
    for name, data in grids.items():
        imshow_grid(grid, data, var_name="Elevation", var_units="m", cmap=cmap_terrain)
        plt.title(name)  # font size handled globally
        plt.show()
        
        print("DEM shape:", data.shape)
        print("Non-NaN count:", np.count_nonzero(~np.isnan(data)))
    

# Example call to the function
plot_all_grids(grid, grids)




# #%% User defined

# matrices = [arra_2010, lpc_2016, fema_2018]
# time_steps = [2010, 2016, 2018]


# # Create an instance of Spatial_Autocorrelation 
# spatial_autocorr_calculator = Spatial_Autocorrelation(
#     grid=grid,
#     matrices=matrices,    
#     time_steps=time_steps
# )

# # Call the method    
# # spatial_autocorr_calculator.plot_morans_i_over_time()
# # spatial_autocorr_calculator.plot_gearys_c_over_time()
# spatial_autocorr_calculator.plot_morans_i_and_gearys_c_dem()

#%%

factors = [1, 5, 10, 20, 30]

dem_paths = {
    'arra_2010': r'/Users/viviangrom/Documents/papers/paper2_entropy/raster_analysis/dem_cut6/arra_2010_6_3_mkd.asc',
    'lpc_2016': r'/Users/viviangrom/Documents/papers/paper2_entropy/raster_analysis/dem_cut6/lpc_2016_6_mkd.asc',
    'fema_2018': r'/Users/viviangrom/Documents/papers/paper2_entropy/raster_analysis/dem_cut6/fema_2018_6_mkd.asc'
}

def load_dem_masked(path):
    """Load an ASCII DEM and mask all -9999 values."""
    data = np.flipud(np.loadtxt(path, skiprows=6))
    return np.ma.masked_equal(data, -9999)

# Function to process DEMs for a given downsampling factor
def process_dems(factor):
    # Load DEMs
    grids = {name: load_dem_masked(path) for name, path in dem_paths.items()}

    # Downsample
    for name in grids:
        grids[name] = downsample_array(grids[name], factor)

    # Create RasterModelGrid
    nrows, ncols = grids['arra_2010'].shape
    grid = RasterModelGrid((nrows, ncols), xy_spacing=(1*factor, 1*factor))
    grid.set_closed_boundaries_at_grid_edges(True, True, True, True)

    # Add field and close masked nodes
    grid.add_field("topographic__elevation", grids['arra_2010'], at="node", clobber=True)
    mask = grids['arra_2010'].mask
    grid.status_at_node[mask.flatten()] = grid.BC_NODE_IS_CLOSED

    return grid, grids

time_steps = [2010, 2016, 2018]

    
#%%

# factors = [1, 5, 10, 20, 30]
# time_steps = [2010, 2016, 2018]

# # Initialize dictionaries to store results
# all_moran = {}
# all_geary = {}

# for factor in factors:
#     print(f"\nProcessing downsampling factor: {factor}")
    
#     arra = downsample_array(load_dem_masked(r'/Users/viviangrom/Documents/papers/paper2_entropy/raster_analysis/dem_cut6/arra_2010_6_3_mkd.asc'), factor)
#     lpc  = downsample_array(load_dem_masked(r'/Users/viviangrom/Documents/papers/paper2_entropy/raster_analysis/dem_cut6/lpc_2016_6_mkd.asc'), factor)
#     fema = downsample_array(load_dem_masked(r'/Users/viviangrom/Documents/papers/paper2_entropy/raster_analysis/dem_cut6/fema_2018_6_mkd.asc'), factor)
    
#     matrices = [arra, lpc, fema]

#     # Build grid (xy_spacing scaled by factor)
#     nrows, ncols = arra.shape
#     grid = RasterModelGrid((nrows, ncols), xy_spacing=(1*factor, 1*factor))
#     grid.set_closed_boundaries_at_grid_edges(True, True, True, True)
#     mask = arra.mask
#     grid.status_at_node[mask.flatten()] = grid.BC_NODE_IS_CLOSED

#     # Create autocorrelation instance
#     autocorr_calc = Spatial_Autocorrelation(grid=grid, matrices=matrices, time_steps=time_steps)
    
#     # Compute Moran's I and Geary's C
#     moran_vals, geary_vals, _, _ = autocorr_calc.plot_morans_i_and_gearys_c_dem()

#     # Store results
#     all_moran[factor] = moran_vals
#     all_geary[factor] = geary_vals

# # Print all values for easy plotting
# print("\n=== Moran's I Values ===")
# for factor, vals in all_moran.items():
#     print(f"Factor {factor}: {vals}")

# print("\n=== Geary's C Values ===")
# for factor, vals in all_geary.items():
#     print(f"Factor {factor}: {vals}")


#%%

# Time steps
time_steps = [2010, 2016, 2018]

# Data
moran_dict = {
    1: [0.9999238659386558, 0.9999269732134626, 0.9999472080223979],
    5: [0.9989785463121756, 0.998977541428962, 0.9991602902401744],
    10: [0.9964031305873083, 0.9964174908561678, 0.9970229007683941],
    20: [0.9877702926415434, 0.9878117067678087, 0.9894601131423832],
    30: [0.9759233847089617, 0.9759577530284601, 0.9789286569523117],
}

geary_dict = {
    1: [6.11701559745648e-05, 5.834002343959338e-05, 4.3057875299639136e-05],
    5: [0.0008267542562895337, 0.0008217634775334389, 0.0007008459395032206],
    10: [0.0029064567308516834, 0.0028937708684624165, 0.0025211877029918165],
    20: [0.00986936115815654, 0.009833544912361748, 0.008784447929154997],
    30: [0.019214835958248272, 0.019163323746332908, 0.01738166094734472],
}

# Plot settings
label_font_size = 22
title_font_size = 22
tick_font_size = 20
marker_size = 9
line_width = 2.5

# Choose markers and line styles for accessibility
marker_styles = ['o', 's', '^', 'D', 'v']  # circle, square, triangle, diamond, inverted triangle
line_styles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1))]  # solid, dashed, dash-dot, dotted, custom

fig, ax1 = plt.subplots(figsize=(10, 8))
ax2 = ax1.twinx()

# Plot Moran's I
for i, (factor, values) in enumerate(moran_dict.items()):
    ax1.plot(
        time_steps,
        values,
        marker=marker_styles[i % len(marker_styles)],
        linestyle=line_styles[i % len(line_styles)],
        color='black',  # black-only for accessibility
        markersize=marker_size,
        linewidth=line_width,
        label=f"Moran's I, Factor {factor}"
    )

ax1.set_xlabel("Time Step", fontsize=label_font_size)
ax1.set_ylabel("Moran's I", fontsize=label_font_size)
ax1.tick_params(axis='y', labelsize=tick_font_size)
ax1.tick_params(axis='x', labelsize=tick_font_size)
ax1.set_ylim(0.9, 1.02)  # adjust for your data

# Plot Geary's C
for i, (factor, values) in enumerate(geary_dict.items()):
    ax2.plot(
        time_steps,
        values,
        marker=marker_styles[i % len(marker_styles)],
        linestyle=line_styles[i % len(line_styles)],
        color='black',  # black-only
        markersize=marker_size,
        linewidth=line_width,
        label=f"Geary's C, Factor {factor}"
    )

ax2.set_ylabel("Geary's C", fontsize=label_font_size)
ax2.tick_params(axis='y', labelsize=tick_font_size)
ax2.set_ylim(-0.03, 0.1)  # adjust for your data

# Combine legends from both axes
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()

fig.legend(
    lines1 + lines2,
    labels1 + labels2,
    loc='lower center',
    bbox_to_anchor=(0.5, -0.15),
    ncol=2,
    frameon=False,
    fontsize=20
)

fig.tight_layout()
fig.subplots_adjust(bottom=0.28)  # make room for the legend

fig.suptitle(
    "Moran's I and Geary's C for Different Downsampling Factors",
    fontsize=title_font_size,
    y=1.05  # move above the axes
)

plt.show()





