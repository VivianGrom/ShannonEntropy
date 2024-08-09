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
from landlab import RasterModelGrid, imshow_grid, imshowhs_grid
from landlab.components import (
    FlowAccumulator,
    StreamPowerEroder,
)
from landlab.plot.drainage_plot import drainage_plot
from libpysal.weights import lat2W
from esda.moran import Moran
from esda.geary import Geary
import math


#%% functions

def calculate_shannon_entropy(grid, matrices, k=None):
    """
    Calculate the entropy based on the difference between matrices.

    Parameters:
    - grid: RasterModelGrid object
    - matrices: List of matrices (numpy arrays) to compare
    - k: Maximum modular difference (optional)

    Returns:
    - entropy_matrix: The calculated entropy matrix
    - probabilities: The calculated probabilities for each element
    """

    n = len(matrices)  # number of elements

    if k is None:
        R = np.zeros(grid.shape)

        # Calculate the modular difference for each corresponding element
        for i in range(R.shape[0]):
            for j in range(R.shape[1]):
                R[i, j] = max(abs(matrices[k][i, j] - matrices[(k + 1) % n][i, j]) for k in range(n))

        k = np.max(R)
        print("max k is: ", k)

        plt.figure()
        imshow_grid(grid, R, grid_units=("m", "m"), cmap='Blues', var_name="Modular Difference")
        plt.title("Modular Difference Matrix R")
        plt.show()

    # Calculate probabilities
    if n == 2:
        p1 = abs(matrices[0] - matrices[1]) / (k * n)
        probabilities = [p1, 1 - p1]
    else:
        probabilities = []
        for i in range(n):
            j = (i + 1) % n  # Wrap around to the beginning if at the end
            probabilities.append(abs(matrices[i] - matrices[j]) / (k * n))

    # Handle case where probability is zero
    probabilities = [np.where(p == 0, 1, p) for p in probabilities]

    # Calculate entropy
    entropy_matrix = np.zeros(grid.shape)

    for prob in probabilities:
        entropy_matrix -= prob * np.log(prob) / np.log(n)

    plt.figure()
    imshow_grid(grid, entropy_matrix, cmap='plasma', grid_units=("m", "m"), var_name="Entropy")
    plt.title("Entropy Matrix")
    plt.show()

    return entropy_matrix, probabilities

def pixel_entropy_across_experiments(grid, matrices, k_global=None, pixel=None):
    """
    Calculate the entropy based on the difference between matrices.

    Parameters:
    - grid: RasterModelGrid object
    - matrices: List of matrices (numpy arrays) to compare
    - k_global: Maximum modular difference (optional)
    - pixel: Tuple of (row, col) coordinates (optional)

    Returns:
    - entropies: The calculated Shannon entropies for each pair of experiments,
    with a local k derived from the absolute difference between the pair 
    of models
    """

    n = len(matrices)  # number of matrices
    
    
    R = np.zeros(grid.shape)
    
    # Calculate the modular difference for each corresponding element
    for i in range(R.shape[0]):
        for j in range(R.shape[1]):
            R[i, j] = max(abs(matrices[m][i, j] - matrices[(m + 1) % n][i, j]) for m in range(n))
    
    k_global = np.max(R)
    print("k global is: ", k_global)
    
    max_k_pixel = np.argmax(R)
    max_abs_pixel = np.unravel_index(max_k_pixel, grid.shape)
    print(f"Pixel with highest absolute value: {max_abs_pixel} -> Value: {R[max_abs_pixel]}")
    
    if pixel is None:
        row, col = max_abs_pixel
    else:
        row, col = pixel
        
    entropies = []
    for i in range(n-1):
        j = i + 1  # Compare with the next matrix
        
        # Calculate probabilities
        diff_matrix = abs(matrices[i] - matrices[j])
        k_local = np.max(diff_matrix)
        
        p1 = diff_matrix / (k_local * 2)
        
        if k_local > 0:
            p1 = diff_matrix / (k_local * 2)
            p1 = np.where(p1 == 0, 1e-10, p1)  # Avoid log(0)
            p2 = 1 - p1
            
            entropy = -1 * (p1 * np.log2(p1) + p2 * np.log2(p2))
        else:
            entropy = np.zeros(diff_matrix.shape)
        
        # Calculate entropy
        
        entropies.append(entropy)
        
    entropies_array = np.array(entropies)
    
    # Extract pixel values from each matrix
    pixel_values = entropies_array[:, row, col]

    # Plotting
    plt.figure(figsize=(15, 6))
    plt.plot(range(1, len(entropies_array) + 1), pixel_values, marker='o', linestyle='-', color='b')
    plt.xticks(np.arange(0, 271, 20))  # Show ticks every 15 time steps
    #plt.xticks(np.arange(0, n+1))
    plt.xlabel('Time Step')
    plt.ylabel(f'Entropy at Pixel ({row}, {col})')
    plt.ylim(0,1)
    plt.title(f'Entropy across pairs of matrices for pixel:({row}, {col})')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return entropies

def pixel_entropy_across_experiments_mask(grid, matrices, mask, pixel_stream=None, pixel_hillslope=None):
    """
    Calculate the entropy based on the difference between matrices.

    Parameters:
    - grid: RasterModelGrid object
    - matrices: List of matrices (numpy arrays) to compare
    - k_global: Maximum modular difference (optional)
    - pixel: Tuple of (row, col) coordinates (optional)

    Returns:
    - entropies: The calculated Shannon entropies for each pair of experiments,
    with a local k derived from the absolute difference between the pair 
    of models
    """

    n = len(matrices)  # number of matrices
    
    # Calculate the modular difference for the streams
    R_stream = np.zeros(grid.shape)
    for i in range(R_stream.shape[0]):
        for j in range(R_stream.shape[1]):
            if mask[i, j]:
                R_stream[i, j] = max(abs(matrices[m][i, j] - matrices[(m + 1) % n][i, j]) for m in range(n))
    k_stream = np.max(R_stream)
    print("k_stream is: ", k_stream)
    
    max_k_pixel_stream = np.argmax(R_stream)
    max_abs_pixel_stream = np.unravel_index(max_k_pixel_stream, grid.shape)
    print(f"Pixel with highest absolute value: {max_abs_pixel_stream} -> Value: {R_stream[max_abs_pixel_stream]}")

    # Calculate the modular difference for the hillslopes
    R_hillslope = np.zeros(grid.shape)
    for i in range(R_hillslope.shape[0]):
        for j in range(R_hillslope.shape[1]):
            if not mask[i, j]:
                R_hillslope[i, j] = max(abs(matrices[m][i, j] - matrices[(m + 1) % n][i, j]) for m in range(n))
    k_hillslope = np.max(R_hillslope)
    print("k_hillslope is: ", k_hillslope)
    
    max_k_pixel_hlope = np.argmax(R_hillslope)
    max_abs_pixel_hlope = np.unravel_index(max_k_pixel_hlope, grid.shape)
    print(f"Pixel with highest absolute value: {max_abs_pixel_hlope} -> Value: {R_hillslope[max_abs_pixel_hlope]}")
    
    # Assign pixel for stream values
    if pixel_stream is None:
        row_stream, col_stream = max_abs_pixel_stream
    else:
        row_stream, col_stream = pixel_stream

    # Assign pixel for hillslope values
    if pixel_hillslope is None:
        row_hillslope, col_hillslope = max_abs_pixel_hlope
    else:
        row_hillslope, col_hillslope = pixel_hillslope
              
    entropies_stream = []
    entropies_hillslope = []
    
    mod_stream = []
    mod_hillslope = []

    for i in range(n - 1):
        j = i + 1  # Compare with the next matrix

        # Calculate probabilities for stream region
        diff_matrix_stream = abs(matrices[i] - matrices[j]) * mask
        k_local_stream = np.max(diff_matrix_stream)
        if k_local_stream > 0:
            p1_stream = diff_matrix_stream / (k_local_stream * 2)
            p1_stream = np.where(p1_stream == 0, 1e-10, p1_stream)  # Avoid log(0)
            p2_stream = 1 - p1_stream
            entropy_stream = -1 * (p1_stream * np.log2(p1_stream) + p2_stream * np.log2(p2_stream))
        else:
            entropy_stream = np.zeros(diff_matrix_stream.shape)
        entropies_stream.append(entropy_stream)
        mod_stream.append(diff_matrix_stream)

        # Calculate probabilities for hillslope region
        diff_matrix_hillslope = abs(matrices[i] - matrices[j]) * (~mask)
        k_local_hillslope = np.max(diff_matrix_hillslope)
        if k_local_hillslope > 0:
            p1_hillslope = diff_matrix_hillslope / (k_local_hillslope * 2)
            p1_hillslope = np.where(p1_hillslope == 0, 1e-10, p1_hillslope)  # Avoid log(0)
            p2_hillslope = 1 - p1_hillslope
            entropy_hillslope = -1 * (p1_hillslope * np.log2(p1_hillslope) + p2_hillslope * np.log2(p2_hillslope))
        else:
            entropy_hillslope = np.zeros(diff_matrix_hillslope.shape)
        entropies_hillslope.append(entropy_hillslope)
        mod_hillslope.append(diff_matrix_hillslope)

    entropies_stream_array = np.array(entropies_stream)
    entropies_hillslope_array = np.array(entropies_hillslope)
    
    mod_stream_array = np.array(mod_stream)
    mod_hillslope_array = np.array(mod_hillslope)

    # Extract pixel values from each matrix
    pixel_values_stream = entropies_stream_array[:, row_stream, col_stream]
    pixel_values_hillslope = entropies_hillslope_array[:, row_hillslope, col_hillslope]
    
    mod_pixel_values_stream = mod_stream_array[:, row_stream, col_stream]
    mod_pixel_values_hillslope = mod_hillslope_array[:, row_hillslope, col_hillslope]
    
    # Plotting stream pixel entropy
    plt.figure(figsize=(15, 6))
    plt.plot(range(1, len(entropies_stream_array) + 1), pixel_values_stream, marker='o', linestyle='-', color='b')
    #plt.xticks(np.arange(0, 271, 20))
    #plt.xticks(np.arange(0, len(entropies_stream_array) + 1, 1))  # Adjust the range as needed
    plt.xlabel('Time Step')
    plt.ylabel(f'Entropy at Stream Pixel ({row_stream}, {col_stream})')
    plt.ylim(0, 1)
    plt.title(f'Entropy across pairs of matrices for Stream pixel: ({row_stream}, {col_stream})')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Plotting stream pixel module
    plt.figure(figsize=(15, 6))
    plt.plot(range(1, len(mod_stream_array) + 1), mod_pixel_values_stream, marker='o', linestyle='-', color='b')
    #plt.xticks(np.arange(0, 271, 20))
    #plt.xticks(np.arange(0, len(entropies_stream_array) + 1, 1))  # Adjust the range as needed
    plt.xlabel('Time Step')
    plt.ylabel(f'Modular Difference Evolution for Stream Pixel ({row_stream}, {col_stream})')
    #plt.ylim(0, 1)
    plt.title(f'Modular Difference across pairs of matrices for Stream pixel: ({row_stream}, {col_stream})')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Plotting hillslope pixel entropy
    plt.figure(figsize=(15, 6))
    plt.plot(range(1, len(entropies_hillslope_array) + 1), pixel_values_hillslope, marker='o', linestyle='-', color='r')
    #plt.xticks(np.arange(0, 271, 20))
    #plt.xticks(np.arange(0, len(entropies_hillslope_array) + 1, 1))  # Adjust the range as needed
    plt.xlabel('Time Step')
    plt.ylabel(f'Entropy at Hillslope Pixel ({row_hillslope}, {col_hillslope})')
    plt.ylim(0, 1)
    plt.title(f'Entropy across pairs of matrices for Hillslope pixel: ({row_hillslope}, {col_hillslope})')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    # Plotting stream pixel module
    plt.figure(figsize=(15, 6))
    plt.plot(range(1, len(mod_hillslope_array) + 1), mod_pixel_values_hillslope, marker='o', linestyle='-', color='b')
    #plt.xticks(np.arange(0, 271, 20))
    #plt.xticks(np.arange(0, len(entropies_stream_array) + 1, 1))  # Adjust the range as needed
    plt.xlabel('Time Step')
    plt.ylabel(f'Modular Difference Evolution for Hillslope Pixel ({row_stream}, {col_stream})')
    #plt.ylim(0, 1)
    plt.title(f'Modular Difference across pairs of matrices for Hillslope pixel: ({row_stream}, {col_stream})')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    # Highlight the pixels in the grid

    grid.add_zeros('node', 'pixel_loc', clobber=True)
    
    # Convert the 2D pixel coordinate to 1D node index
    node_id_stream = grid.grid_coords_to_node_id(row_stream, col_stream)
    node_id_hillslope = grid.grid_coords_to_node_id(row_hillslope, col_hillslope)
    
    # Add 1 to pixel loc
    grid.at_node['pixel_loc'][node_id_stream] += 1
    grid.at_node['pixel_loc'][node_id_hillslope] += 1
    
    pixel_loc = grid.at_node['pixel_loc']

    # Replace all 0 values with NaN
    pixel_loc['pixel_loc' == 0] = np.nan
    
    # plt.figure()
    # imshow_grid(grid, 'pixel_loc', grid_units=("m", "m"), var_name="Pixel Loc", cmap="Blues")
    # plt.title("Pixel Loc")
    # plt.show()
    
    plt.figure()
    topo = np.ma.masked_equal(matrices[-1], -9999)
    plt.figure(figsize=(10, 8))
    imshowhs_grid(grid, 
                  values=topo, 
                  plot_type='Drape1',
                  drape1='pixel_loc', cmap= 'Reds', 
                  alpha=0.5,
                  #color_for_closed='black', 
                  var_name='Pixel Loc', 
                  add_double_colorbar=True)
    plt.show()
    
    
    return entropies_stream_array, entropies_hillslope_array

def plot_morans_i_over_time(matrices, time_steps):
    """
    Calculates and plots Moran's I and p-values over multiple time steps.

    Parameters:
    - matrices: list of 2D numpy arrays representing different time steps.
    - time_steps: list of integers representing the time steps.
    """
    # Create the matrix of weights 
    w = lat2W(matrices[0].shape[0], matrices[0].shape[1])

    # Calculate Moran's I and p-values for each matrix
    moran_values = []
    p_values = []
    for matrix in matrices:
        mi = Moran(matrix, w)
        moran_values.append(mi.I)
        p_values.append(mi.p_norm)

    # Display Moran's I values and p-values
    for i, (mi_val, p_val) in enumerate(zip(moran_values, p_values)):
        print(f"Time Step {i+1} - Moran's I: {mi_val:.4f}, p-value: {p_val:.4f}")
    
    # Plot Moran's I vs. Time Steps
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot Moran's I values
    color = 'tab:blue'
    ax1.set_xlabel("Time Step")
    ax1.set_ylabel("Moran's I", color=color)
    ax1.plot(time_steps, moran_values, marker='o', color=color, label="Moran's I")
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.axhline(y=0, color='red', linestyle='--', label="Moran's I = 0")
    ax1.set_ylim(-1, 1)

    # Annotate each point with Moran's I value
    for i, mi_val in enumerate(moran_values):
        ax1.annotate(f'{mi_val:.3f}', (time_steps[i], mi_val), textcoords="offset points", xytext=(0,10), ha='center')

    # Create a second y-axis for p-values
    ax2 = ax1.twinx()
    color = 'tab:green'
    ax2.set_ylabel("p-value", color=color)
    ax2.plot(time_steps, p_values, marker='s', linestyle='--', color=color, label="p-value")
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim(-0.01, 0.1)

    # Annotate each point with p-value
    for i, p_val in enumerate(p_values):
        ax2.annotate(f'{p_val:.2f}', (time_steps[i], p_val), textcoords="offset points", xytext=(0,-15), ha='center')

    fig.tight_layout()  # Adjust layout to make room for annotations
    fig.suptitle("Moran's I and p-value over Time Steps", y=1)
    plt.show()
    
def plot_gearys_c_over_time(matrices, time_steps):
    """
    Calculates and plots Geary's C and p-values over multiple time steps.

    Parameters:
    - matrices: list of 2D numpy arrays representing different time steps.
    - time_steps: list of integers representing the time steps.
    """
    # Create the matrix of weights 
    w = lat2W(matrices[0].shape[0], matrices[0].shape[1])

    # Calculate Geary's C and p-values for each matrix
    geary_values = []
    p_values = []
    for matrix in matrices:
        gc = Geary(matrix, w)
        geary_values.append(gc.C)
        p_values.append(gc.p_norm)

    # Display Geary's C values and p-values
    for i, (gc_val, p_val) in enumerate(zip(geary_values, p_values)):
        print(f"Time Step {i+1} - Geary's C: {gc_val:.4f}, p-value: {p_val:.4f}")
    
    # Plot Geary's C vs. Time Steps
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot Geary's C values
    color = 'tab:red'
    ax1.set_xlabel("Time Step")
    ax1.set_ylabel("Geary's C", color=color)
    ax1.plot(time_steps, geary_values, marker='o', color=color, label="Geary's C")
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.axhline(y=1, color='blue', linestyle='--', label="Geary's C = 1")
    ax1.set_ylim(0, 2)

    # Annotate each point with Geary's C value
    for i, gc_val in enumerate(geary_values):
        ax1.annotate(f'{gc_val:.3f}', (time_steps[i], gc_val), textcoords="offset points", xytext=(0,10), ha='center')

    # Create a second y-axis for p-values
    ax2 = ax1.twinx()
    color = 'tab:green'
    ax2.set_ylabel("p-value", color=color)
    ax2.plot(time_steps, p_values, marker='s', linestyle='--', color=color, label="p-value")
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim(-0.01, 0.1)

    # Annotate each point with p-value
    for i, p_val in enumerate(p_values):
        ax2.annotate(f'{p_val:.2f}', (time_steps[i], p_val), textcoords="offset points", xytext=(0,-15), ha='center')

    fig.tight_layout()  # Adjust layout to make room for annotations
    fig.suptitle("Geary's C and p-value over Time Steps", y=1)
    plt.show()
    
def differentiate_hillslopes_and_streams(grid, elevations, stream_threshold=4, slope_threshold=None):
    """
    Function to differentiate hillslopes and streams based on drainage area and slope thresholds.
    
    Parameters:
    - grid (RasterModelGrid): The Landlab model grid.
    - elevations: List of elevation arrays at different time steps.
    - stream_threshold (float): Threshold for drainage area to classify as stream.
    - slope_threshold (float): Threshold for slope to classify as stream.

    Returns:
    - tot_stream_mask (np.ndarray): Total Boolean mask indicating stream nodes across all time steps.
    """
    stream_masks = []
    dxy = grid.dx

    for elevation in elevations:
        # Set the elevation data to the grid
        grid.at_node['topographic__elevation'] = elevation

        # Calculate drainage area
        fa = FlowAccumulator(grid, flow_director="FlowDirectorD8")
        fa.run_one_step()
        drainage_area = grid.at_node['drainage_area']

        # Calculate slope
        slope = grid.calc_slope_at_node()

        # Reshape for plotting and classification
        drainage_area = drainage_area.reshape(grid.shape)/(dxy**2)
        print(drainage_area)
        slope = slope.reshape(grid.shape)

        # Classify nodes
        stream_mask = (drainage_area > stream_threshold) | (slope > slope_threshold)
        stream_masks.append(stream_mask)

        # Plot hillslopes and streams
        plt.figure()
        imshow_grid(grid, stream_mask.astype(int), grid_units=("m", "m"), var_name="Stream Mask", cmap="Blues")
        plt.title("Streams (1) and Hillslopes (0)")
        plt.show()

        print("Stream Mask (flipped):")
        print(np.flipud(stream_mask))

        plt.figure()
        imshow_grid(grid, slope, grid_units=("m", "m"), var_name="Slopes", cmap="Blues")
        plt.title("Slopes")
        plt.show()

    # Sum the stream masks
    tot_stream_mask = np.any(stream_masks, axis=0)
    plt.figure()
    imshow_grid(grid, tot_stream_mask.astype(int), grid_units=("m", "m"), var_name="Total Stream Mask", cmap="Blues")
    plt.title("Total Streams Mask")
    plt.show()

    return tot_stream_mask
    
def diff_hslp_and_strm_buff(grid, elevations, stream_threshold=4, slope_threshold=0.1, filter_size=3):
    """
    Function to differentiate hillslopes and streams based on drainage area and slope thresholds with buffer.
    
    Parameters:
    - grid (RasterModelGrid): The Landlab model grid.
    - elevations: List of elevation arrays at different time steps.
    - stream_threshold (float): Threshold for drainage area to classify as stream.
    - slope_threshold (float): Threshold for slope to classify as stream.
    - filter_size (int): Size of the maximum filter to create buffer around streams.

    Returns:
    - tot_stream_mask (np.ndarray): Total Boolean mask indicating stream nodes across all time steps.
    """
    stream_masks = []
    dxy = grid.dx
    slopes = []

    for elevation in elevations:
        # Set the elevation data to the grid
        grid.at_node['topographic__elevation'] = elevation

        # Calculate drainage area
        fa = FlowAccumulator(grid, flow_director="FlowDirectorD8")
        fa.run_one_step()
        drainage_area = grid.at_node['drainage_area']

        # Calculate slope
        slope = grid.calc_slope_at_node()

        # Reshape for plotting and classification
        drainage_area = drainage_area.reshape(grid.shape)/(dxy**2)
        # print(drainage_area)
        slope = slope.reshape(grid.shape)
        slopes.append(slope)

        # Classify nodes
        stream_mask = (drainage_area > stream_threshold) #| (slope > slope_threshold)
        stream_masks.append(stream_mask)

        # Plot hillslopes and streams
        # plt.figure()
        # imshow_grid(grid, stream_mask.astype(int), grid_units=("m", "m"), var_name="Stream Mask", cmap="Blues")
        # plt.title("Streams (1) and Hillslopes (0)")
        # plt.show()

        # print("Stream Mask (flipped):")
        # print(np.flipud(stream_mask))

        # plt.figure()
        # imshow_grid(grid, slope, grid_units=("m", "m"), var_name="Slopes", cmap="Blues")
        # plt.title("Slopes")
        # plt.show()

    # Sum the stream masks
    tot_stream_mask = np.any(stream_masks, axis=0)
    buffered_stream_mask = maximum_filter(tot_stream_mask.astype(int), size=filter_size).astype(bool)

    plt.figure()
    imshow_grid(grid, buffered_stream_mask.astype(int), grid_units=("m", "m"), var_name="Buffered Stream Mask", cmap="Blues")
    plt.title("Buffered Streams Mask")
    plt.show()
    
    # plt.figure()
    # imshow_grid(grid, tot_stream_mask.astype(int), grid_units=("m", "m"), var_name="Total Stream Mask", cmap="Blues")
    # plt.title("Total Streams Mask")
    # plt.show()
    
    max_slope = np.max(slopes)
    print("max slope is: ", max_slope)

    return buffered_stream_mask


def diff_hslp_and_strm_scal_buff(grid, elevations, stream_threshold=4, slope_threshold=0.1, buffer_scaling_factor=0.00015, minimum_buffer_size=0.5):
    """
    Function to differentiate hillslopes and streams based on drainage area and slope thresholds with scaled buffer.
    
    Parameters:
    - grid (RasterModelGrid): The Landlab model grid.
    - elevations: List of elevation arrays at different time steps.
    - stream_threshold (float): Threshold for drainage area to classify as stream.
    - slope_threshold (float): Threshold for slope to classify as stream.
    - buffer_scaling_factor (float): Factor to scale the buffer size based on drainage area.

    Returns:
    - tot_stream_mask (np.ndarray): Total Boolean mask indicating stream nodes across all time steps.
    """
    stream_masks = []
    dxy = grid.dx
    slopes = []

    for elevation in elevations:
        # Set the elevation data to the grid
        grid.at_node['topographic__elevation'] = elevation

        # Calculate drainage area
        fa = FlowAccumulator(grid, flow_director="FlowDirectorD8")
        fa.run_one_step()
        drainage_area = grid.at_node['drainage_area']

        # Calculate slope
        slope = grid.calc_slope_at_node()

        # Reshape for plotting and classification
        drainage_area = drainage_area.reshape(grid.shape) / (dxy**2)
        slope = slope.reshape(grid.shape)
        slopes.append(slope)

        # Classify nodes
        stream_mask = (drainage_area > stream_threshold)
        stream_masks.append(stream_mask)

    # Sum the stream masks
    tot_stream_mask = np.any(stream_masks, axis=0)

    # Calculate buffer size dynamically based on drainage area
    buffer_size_array = (drainage_area * buffer_scaling_factor).astype(int)
    buffer_size_array[buffer_size_array < 1] = minimum_buffer_size  # Ensure minimum buffer size is 1

    # Initialize the buffered stream mask
    buffered_stream_mask = np.zeros_like(tot_stream_mask)

    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            if tot_stream_mask[i, j]:
                buffer_size = buffer_size_array[i, j]
                buffered_stream_mask[max(0, i-buffer_size):min(grid.shape[0], i+buffer_size+1),
                                     max(0, j-buffer_size):min(grid.shape[1], j+buffer_size+1)] = 1

    buffered_stream_mask = buffered_stream_mask.astype(bool)

    plt.figure()
    imshow_grid(grid, buffered_stream_mask.astype(int), grid_units=("m", "m"), var_name="Buffered Stream Mask", cmap="Blues")
    plt.title("Buffered Streams Mask - Scaled")
    plt.show()

    max_slope = np.max(slopes)
    print("max slope is: ", max_slope)

    return buffered_stream_mask

def calculate_shannon_entropy_mask(grid, matrices, stream_mask, k_stream=None, k_hillslope=None):
    """
    Calculate the entropy based on the difference between matrices,
    separately for stream and hillslope areas based on a mask.

    Parameters:
    - grid: RasterModelGrid object
    - matrices: List of matrices (numpy arrays) to compare
    - stream_mask: Boolean mask indicating stream nodes
    - k_stream: Maximum modular difference for stream region (optional)
    - k_hillslope: Maximum modular difference for hillslope region (optional)

    Returns:
    - entropy_matrix_stream: The calculated entropy matrix for stream regions
    - entropy_matrix_hillslope: The calculated entropy matrix for hillslope regions
    - probabilities_stream: The calculated probabilities for each element in stream regions
    - probabilities_hillslope: The calculated probabilities for each element in hillslope regions
    """
    n = len(matrices)  # number of elements

    # Calculate k for stream regions if not provided
    if k_stream is None:
        R_stream = np.zeros(grid.shape)
        for i in range(R_stream.shape[0]):
            for j in range(R_stream.shape[1]):
                if stream_mask[i, j]:
                    R_stream[i, j] = max(abs(matrices[m][i, j] - matrices[(m + 1) % n][i, j]) for m in range(n))
        k_stream = np.max(R_stream)
        print("k_stream is: ", k_stream)

    # Calculate k for hillslope regions if not provided
    if k_hillslope is None:
        R_hillslope = np.zeros(grid.shape)
        for i in range(R_hillslope.shape[0]):
            for j in range(R_hillslope.shape[1]):
                if not stream_mask[i, j]:
                    R_hillslope[i, j] = max(abs(matrices[m][i, j] - matrices[(m + 1) % n][i, j]) for m in range(n))
        k_hillslope = np.max(R_hillslope)
        print("k_hillslope is: ", k_hillslope)

    # Calculate probabilities
    probabilities_stream = []
    probabilities_hillslope = []
    for i in range(n):
        j = (i + 1) % n  # Wrap around to the beginning if at the end
        prob_stream = abs(matrices[i] - matrices[j]) / (k_stream * n)
        prob_hillslope = abs(matrices[i] - matrices[j]) / (k_hillslope * n)
        probabilities_stream.append(prob_stream)
        probabilities_hillslope.append(prob_hillslope)

    # Handle case where probability is zero
    probabilities_stream = [np.where(p == 0, 1, p) for p in probabilities_stream]
    probabilities_hillslope = [np.where(p == 0, 1, p) for p in probabilities_hillslope]

    # Initialize entropy matrices for stream and hillslope regions
    entropy_matrix_stream = np.zeros(grid.shape)
    entropy_matrix_hillslope = np.zeros(grid.shape)

    for prob_stream, prob_hillslope in zip(probabilities_stream, probabilities_hillslope):
        entropy_matrix_stream[stream_mask] -= prob_stream[stream_mask] * np.log(prob_stream[stream_mask]) / np.log(n)
        entropy_matrix_hillslope[~stream_mask] -= prob_hillslope[~stream_mask] * np.log(prob_hillslope[~stream_mask]) / np.log(n)

    # Set entropy values of non-relevant regions to a high value for visualization
    entropy_matrix_stream[~stream_mask] = np.nan
    entropy_matrix_hillslope[stream_mask] = np.nan
    
    # Create masked arrays to handle np.nan values
    masked_entropy_matrix_stream = np.ma.masked_invalid(entropy_matrix_stream)
    masked_entropy_matrix_hillslope = np.ma.masked_invalid(entropy_matrix_hillslope)


    # Plot entropy matrices for stream and hillslope regions separately
    plt.figure()
    imshow_grid(grid, masked_entropy_matrix_stream, cmap='plasma', grid_units=("m", "m"), var_name="Entropy (Stream Regions)", colorbar_label='Entropy')
    plt.title("Entropy Matrix (Stream Regions)")
    plt.show()

    plt.figure()
    imshow_grid(grid, masked_entropy_matrix_hillslope, cmap='plasma', grid_units=("m", "m"), var_name="Entropy (Hillslope Regions)", colorbar_label='Entropy')
    plt.title("Entropy Matrix (Hillslope Regions)")
    plt.show()

    return masked_entropy_matrix_stream, masked_entropy_matrix_hillslope, probabilities_stream, probabilities_hillslope




#%% Load Dan data

k_19 = np.loadtxt(r'/Users/viviangrom/Documents/Project/Kali/ASCII_2/k_19.asc', skiprows=6)
edem_15_0509 = np.loadtxt(r'/Users/viviangrom/Documents/Project/Kali/ASCII_2/edem_15_0509.asc', skiprows=6)
edem_15_0510 = np.loadtxt(r'/Users/viviangrom/Documents/Project/Kali/ASCII_2/edem_15_0510.asc', skiprows=6)
edem_17_0128 = np.loadtxt(r'/Users/viviangrom/Documents/Project/Kali/ASCII_2/edem_17_0128.asc', skiprows=6)
edem_19_0105 = np.loadtxt(r'/Users/viviangrom/Documents/Project/Kali/ASCII_2/edem_19_0105.asc', skiprows=6)
edem_19_0121 = np.loadtxt(r'/Users/viviangrom/Documents/Project/Kali/ASCII_2/edem_19_0121.asc', skiprows=6)

#%% Fix shape k_19, because whenever I exported as a ASCII file it came with one less column and row

# Metadata for k_19 (the incorrect grid)
ncols_k19 = 2377
nrows_k19 = 1221
xllcorner_k19 = 770540.96661844
yllcorner_k19 = 3191665.0930816
cellsize_k19 = 2  # Replace with actual cell size if different

# Metadata for the target grid (correct grid)
ncols_target = 2378
nrows_target = 1222
xllcorner_target = 770540
yllcorner_target = 3191664
cellsize_target = 2  # Assuming the cell size is the same

# Calculate the column and row offsets
col_offset = int((xllcorner_target - xllcorner_k19) / cellsize_k19)
row_offset = int((yllcorner_target - yllcorner_k19) / cellsize_k19)

print(f'Column offset: {col_offset}, Row offset: {row_offset}')

# Create an empty grid with the target dimensions, filled with NODATA values (-9999)
aligned_k_19 = np.full((nrows_target, ncols_target), -9999)

# Determine where to place the k_19 data within the aligned grid
start_row = max(0, row_offset)
start_col = max(0, col_offset)

# Calculate the ending indices for the insertion
end_row = start_row + k_19.shape[0]
end_col = start_col + k_19.shape[1]

# Place the k_19 data into the correct position within the aligned grid
aligned_k_19[start_row:end_row, start_col:end_col] = k_19[:end_row-start_row, :end_col-start_col]

# Verify the alignment by printing the shapes
print(f'Original k_19 shape: {k_19.shape}')
print(f'Aligned k_19 shape: {aligned_k_19.shape}')

# Assuming the files are 2D arrays with the same dimensions
grid = RasterModelGrid((nrows_target, ncols_target), xy_spacing=(2, 2))  # Adjust xy_spacing as necessary

# Visualize the result to ensure the alignment looks correct
cmap_terrain = mpl.cm.get_cmap("terrain").copy()
imshow_grid(grid, aligned_k_19, var_name="Elevation", var_units="m", cmap=cmap_terrain, at='node')
plt.title('Aligned k_19')
plt.show()

# Save k_19 with the new dimensions
file_path = (r'/Users/viviangrom/Documents/Project/Kali/ASCII_2/k_19.asc')
output_path = file_path.replace('.asc', '_masked.asc')

# Save the masked grid to a new ASCII file
with open(output_path, 'w') as f:
    # Write the header
    f.write(f"ncols        {ncols_target}\n")
    f.write(f"nrows        {nrows_target}\n")
    f.write(f"xllcorner    {xllcorner_target}\n")
    f.write(f"yllcorner    {yllcorner_target}\n")
    f.write(f"cellsize     {cellsize_target}\n")
    f.write(f"NODATA_value -9999\n")
    
    # Write the masked grid data
    np.savetxt(f, aligned_k_19, fmt='%f', delimiter=' ')


#%% Plot Dan data

# Create a dictionary of grid names and data
grids = {
    'k_19': aligned_k_19,
    'edem_15_0509': edem_15_0509,
    'edem_15_0510': edem_15_0510,
    'edem_17_0128': edem_17_0128,
    'edem_19_0105': edem_19_0105,
    'edem_19_0121': edem_19_0121
}

def plot_all_grids(grid, grids):
    cmap_terrain = mpl.cm.get_cmap("terrain").copy()
    
    for name, data in grids.items():
        imshow_grid(grid, data, var_name="Elevation", var_units="m", cmap=cmap_terrain, at='node')
        plt.title(name)
        plt.show()

# Call the function
plot_all_grids(grid, grids)

#%% Mask the other grids like k_19

# List of files to process
files_to_process = [
    r'/Users/viviangrom/Documents/Project/Kali/ASCII_2/edem_15_0509.asc',
    r'/Users/viviangrom/Documents/Project/Kali/ASCII_2/edem_15_0510.asc',
    r'/Users/viviangrom/Documents/Project/Kali/ASCII_2/edem_17_0128.asc',
    r'/Users/viviangrom/Documents/Project/Kali/ASCII_2/edem_19_0105.asc',
    r'/Users/viviangrom/Documents/Project/Kali/ASCII_2/edem_19_0121.asc',
]

mask = aligned_k_19 != -9999

# Dictionary to store masked grids
masked_grids = {}
masked_grids['aligned_k_19'] = aligned_k_19 # Add k_19

# Process each file
for file_path in files_to_process:
    # Load the grid data
    grid_data = np.loadtxt(file_path, skiprows=6)
    
    # Check if the grid data has the same dimensions as aligned_k_19
    if grid_data.shape != aligned_k_19.shape:
        raise ValueError(f"Dimensions of grid data from {file_path} do not match aligned_k_19 dimensions.")
    
    # Apply the mask
    masked_grid = np.where(mask, grid_data, -9999)
    
    # Save the masked grid in the dictionary with the _masked suffix
    masked_grids[file_path.split('/')[-1].replace('.asc', '_masked')] = masked_grid
    
    # Define the output path
    output_path = file_path.replace('.asc', '_masked.asc')
    
    # Save the masked grid to a new ASCII file
    with open(output_path, 'w') as f:
        # Write the header
        f.write(f"ncols        {ncols_target}\n")
        f.write(f"nrows        {nrows_target}\n")
        f.write(f"xllcorner    {xllcorner_target}\n")
        f.write(f"yllcorner    {yllcorner_target}\n")
        f.write(f"cellsize     {cellsize_target}\n")
        f.write(f"NODATA_value -9999\n")
        
        # Write the masked grid data
        np.savetxt(f, masked_grid, fmt='%f', delimiter=' ')
    
    print(f'Masked grid for {file_path} has been saved to {output_path}')

# Call the function
plot_all_grids(grid, masked_grids)

#%% Mask data to match 19_0121, which has a hole in the data
# Just in case we need to mask anything else, but I'm too lazy to run again the block above
def mask_and_save_grids(files_to_process, mask, suffix):
    
    masked_grids = {}
    
    for file_path in files_to_process:
        # Load the grid data
        grid_data = np.loadtxt(file_path, skiprows=6)
        
        # Check if the grid data has the same dimensions as aligned_k_19
        if grid_data.shape != aligned_k_19.shape:
            raise ValueError(f"Dimensions of grid data from {file_path} do not match aligned_k_19 dimensions.")
        
        # Apply the mask
        masked_grid = np.where(mask, grid_data, -9999)
        
        # Save the masked grid in the dictionary with the _masked suffix
        masked_grids[file_path.split('/')[-1].replace('.asc', suffix)] = masked_grid
        
        # Define the output path
        output_path = file_path.replace('.asc', suffix)
        
        # Save the masked grid to a new ASCII file
        with open(output_path, 'w') as f:
            # Write the header
            f.write(f"ncols        {ncols_target}\n")
            f.write(f"nrows        {nrows_target}\n")
            f.write(f"xllcorner    {xllcorner_target}\n")
            f.write(f"yllcorner    {yllcorner_target}\n")
            f.write(f"cellsize     {cellsize_target}\n")
            f.write(f"NODATA_value -9999\n")
            
            # Write the masked grid data
            np.savetxt(f, masked_grid, fmt='%f', delimiter=' ')
        
        print(f'Masked grid for {file_path} has been saved to {output_path}')
    
    return masked_grids

# List of files to process
files_to_process = [
    r'/Users/viviangrom/Documents/Project/Kali/ASCII_2/edem_15_0509_masked.asc',
    r'/Users/viviangrom/Documents/Project/Kali/ASCII_2/edem_15_0510_masked.asc',
    r'/Users/viviangrom/Documents/Project/Kali/ASCII_2/edem_17_0128_masked.asc',
    r'/Users/viviangrom/Documents/Project/Kali/ASCII_2/edem_19_0105_masked.asc',
    r'/Users/viviangrom/Documents/Project/Kali/ASCII_2/edem_19_0121_masked.asc',
    r'/Users/viviangrom/Documents/Project/Kali/ASCII_2/k_19_masked.asc'
]

mask_2 = masked_grids['edem_19_0121_masked'] != -9999

# Call the function
hole_grids = mask_and_save_grids(files_to_process, mask_2, suffix = '_hole.asc')

# Call the function
plot_all_grids(grid, hole_grids)

#%% User defined

# matrices = [hole_grids['edem_15_0509_masked_hole.asc'], 
#             hole_grids['edem_15_0510_masked_hole.asc'],
#             hole_grids['edem_17_0128_masked_hole.asc'], 
#             hole_grids['edem_19_0105_masked_hole.asc'],
#             hole_grids['edem_19_0121_masked_hole.asc'], 
#             hole_grids['k_19_masked_hole.asc'],
#             ]

matrices = [masked_grids['edem_15_0509_masked'], 
            masked_grids['edem_15_0510_masked'],
            masked_grids['edem_17_0128_masked'], 
            masked_grids['edem_19_0105_masked'],
            masked_grids['aligned_k_19']
            ]

# Convert dtype to float because k_19 was int64 
matrices = [np.array(matrix, dtype=np.float64) for matrix in matrices]

###### THE GRID IN THE ASCII FILE IS IN A DIFFERENT ORGANIZATION THAN LANDLAB STUFF, MY FRIEND
###### FIX THIS EARLIER IN THE CODE

# Flip all matrices in the list along the vertical axis
matrices = [np.flipud(matrix) for matrix in matrices]


entropy_matrices, _ = calculate_shannon_entropy(grid, matrices)

# stream_mask = diff_hslp_and_strm_scal_buff(grid, matrices, stream_threshold=300, slope_threshold=0.2, buffer_scaling_factor=0.00015)

stream_mask = diff_hslp_and_strm_buff(grid, matrices, stream_threshold=350, filter_size=3)

entropy_matrix_stream, entropy_matrix_hillslope, _, _ = calculate_shannon_entropy_mask(grid, matrices, stream_mask)

entropies_stream_array, entropies_hillslope_array = pixel_entropy_across_experiments_mask (grid, matrices, stream_mask)

#%% Plotting hillshade

# topo = np.ma.masked_equal(topo_mat_tot[0], -9999)


# # Plot using imshowhs_grid
# plt.figure(figsize=(10, 8))
# imshowhs_grid(grid, 
#               values=topo, 
#               plot_type='Drape1',
#               drape1=entropy_matrix_hillslope, cmap= 'plasma', 
#               alpha=0.8,
#               #color_for_closed='black', 
#               var_name='Entropy', 
#               add_double_colorbar=True)
# plt.show()

# # Plot using imshowhs_grid
# plt.figure(figsize=(10, 8))
# imshowhs_grid(grid, 
#               values=topo, 
#               plot_type='Drape1',
#               drape1=entropy_matrix_stream, cmap= 'plasma', 
#               alpha=0.8,
#               #color_for_closed='black', 
#               var_name='Entropy', 
#               add_double_colorbar=True)
# plt.show()




# plt.figure(figsize=(10, 8))
# imshowhs_grid(grid, topo, drape1=entropy_matrix_hillslope, cmap='plasma',
#               var_name='Entropy (Hillslope Regions)', var_units='Entropy', add_double_colorbar=True)
# plt.title("Topography with Hillshade")
# plt.show()


# plt.figure()
# imshow_grid(grid, entropy_matrix_hillslope, cmap='plasma', grid_units=("m", "m"), var_name="Entropy (Stream Regions)", colorbar_label='Entropy')
# plt.title("Entropy Matrix (Hillslope Regions)")
# plt.show()






