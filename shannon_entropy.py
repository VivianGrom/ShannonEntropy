#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 10:06:13 2024

@author: viviangrom
"""
#%% Import libraries

import numpy as np
from matplotlib import pyplot as plt
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
        print("k is: ", k)

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
    
def differentiate_hillslopes_and_streams(grid, elevations, stream_threshold=4, slope_threshold=0.1):
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

    return entropy_matrix_stream, entropy_matrix_hillslope, probabilities_stream, probabilities_hillslope

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
    topo = np.ma.masked_equal(matrices[0], -9999)
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



#%% Run a model

# Model grid
number_of_rows = 10  # number of raster cells in vertical direction (y)
number_of_columns = 10  # number of raster cells in horizontal direction (x)
dxy = 10  # side length of a raster model cell, or resolution [m]
mg1 = RasterModelGrid((number_of_rows, number_of_columns), dxy)
mg1.set_closed_boundaries_at_grid_edges(True, True, True, False) # Set boundary conditions

# Initialize landscape
np.random.seed(35)  # seed set so our figures are reproducible
mg1_noise = (np.random.rand(mg1.number_of_nodes) / 1000.0)  # initial noise on elevation grid
z1 = mg1.add_zeros("topographic__elevation", at="node") # set up the elevation on the grid
z1 += mg1_noise


# Timesteps
tmax = 6000  # time for the model to run [yr] (Original value was 5E5 yr)
dt = 500  # time step [yr] (Original value was 100 yr)
total_time = 0  # amount of time the landscape has evolved [yr]
t = np.arange(0, tmax, dt)  # each of the time steps that the code will run

# Original K_sp value is 1e-5
K_sp = 1.0e-3  # units vary depending on m_sp and n_sp
m_sp = 0.5  # exponent on drainage area in stream power equation
n_sp = 1.0  # exponent on slope in stream power equation

frr = FlowAccumulator(mg1, flow_director="FlowDirectorD8")  # initializing flow routing
spr = StreamPowerEroder(mg1, K_sp=K_sp, m_sp=m_sp, n_sp=n_sp, threshold_sp=0.0)  # initializing stream power incision
theta = m_sp / n_sp

uplift_rate = np.ones(mg1.number_of_nodes) * 0.0001

# Model runs

mg1_snapshots = []  # List to store mg1 state at each timestep
stream_masks = []



for ti in t:
    z1[mg1.core_nodes] += uplift_rate[mg1.core_nodes] * dt  # uplift the landscape
    frr.run_one_step()  # route flow
    spr.run_one_step(dt)  # fluvial incision
    total_time += dt  # update time keeper
    print(total_time)
    
    if ti > 3500:
        # Save a snapshot of mg1 state
        z1_rshpd = z1.reshape(mg1.shape)
        mg1_snapshots.append(z1_rshpd.copy())
        # Plots snapshots used for entropy calculation
        plt.figure()
        imshow_grid(mg1, z1, grid_units=("m", "m"),cmap='terrain', var_name="Elevation (m)", vmax = 0.8)
        plt.title(f"Elevation at time {total_time} yr")
        plt.show()
      
  
        max_elev = np.max(z1)
        print("Maximum elevation is ", np.max(z1))
        
        plt.figure()
        drainage_plot(mg1, 'topographic__elevation')
        plt.show()
        
        plt.figure()
        drainage_plot(mg1, 'drainage_area')
        


# drainage_area = mg1.at_node['drainage_area'].reshape(mg1.shape)
# print(np.flipud(drainage_area))


#%% User defined

matrices = mg1_snapshots  # Add more matrices as needed

# Define time steps
time_steps = [4500, 5000, 5500, 6000]

entropy_matrices, _ = calculate_shannon_entropy(mg1, matrices)

stream_mask = differentiate_hillslopes_and_streams(mg1, matrices)

entropy_matrix_stream, entropy_matrix_hillslope, _, _ = calculate_shannon_entropy_mask(mg1, matrices, stream_mask)

entropies_stream_array, entropies_hillslope_array = pixel_entropy_across_experiments_mask (mg1, matrices, stream_mask)











grid=mg1
mask=stream_mask
n = len(matrices)  # number of matrices

# Calculate the modular difference for the streams
R_stream = np.zeros(grid.shape)
for i in range(R_stream.shape[0]):
    for j in range(R_stream.shape[1]):
        if stream_mask[i, j]:
            R_stream[i, j] = max(abs(matrices[m][i, j] - matrices[(m + 1) % n][i, j]) for m in range(n))
k_stream = np.max(R_stream)
print("k_stream is: ", k_stream)

max_k_pixel_stream = np.argmax(R_stream)
max_abs_pixel_stream = np.unravel_index(max_k_pixel_stream, grid.shape)
print(f"Pixel with highest absolute value: {max_abs_pixel_stream} -> Value: {R_stream[max_abs_pixel_stream]}")

# plt.figure()
# imshow_grid(grid, R_stream, cmap='Blues', grid_units=("m", "m"), var_name="Modular Diff")
# plt.title("R Stream")
# plt.show()

# Calculate the modular difference for the hillslopes
R_hillslope = np.zeros(grid.shape)
for i in range(R_hillslope.shape[0]):
    for j in range(R_hillslope.shape[1]):
        if not stream_mask[i, j]:
            R_hillslope[i, j] = max(abs(matrices[m][i, j] - matrices[(m + 1) % n][i, j]) for m in range(n))
k_hillslope = np.max(R_hillslope)
print("k_hillslope is: ", k_hillslope)

max_k_pixel_hlope = np.argmax(R_hillslope)
max_abs_pixel_hlope = np.unravel_index(max_k_pixel_hlope, grid.shape)
print(f"Pixel with highest absolute value: {max_abs_pixel_hlope} -> Value: {R_hillslope[max_abs_pixel_hlope]}")

# plt.figure()
# imshow_grid(grid, R_hillslope, cmap='Blues', grid_units=("m", "m"), var_name="Modular Diff")
# plt.title("R Hillslope")
# plt.show()

# Assign pixel for stream values
row_stream, col_stream = max_abs_pixel_stream

# Assign pixel for hillslope values
row_hillslope, col_hillslope = max_abs_pixel_hlope
          
entropies_stream = []
entropies_hillslope = []

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

entropies_stream_array = np.array(entropies_stream)
entropies_hillslope_array = np.array(entropies_hillslope)

# Extract pixel values from each matrix
pixel_values_stream = entropies_stream_array[:, row_stream, col_stream]
pixel_values_hillslope = entropies_hillslope_array[:, row_hillslope, col_hillslope]

# diff_stream = matrices[:, row_stream, col_stream]
# diff_hillslope = matrices[:, row_hillslope, col_hillslope]

# # Plotting stream pixel entropy
# plt.figure(figsize=(15, 6))
# plt.plot(range(1, len(entropies_stream_array) + 1), diff_stream, marker='o', linestyle='-', color='b')
# plt.xticks(np.arange(0, 271, 20))
# #plt.xticks(np.arange(0, len(entropies_stream_array) + 1, 1))  # Adjust the range as needed
# plt.xlabel('Time Step')
# plt.ylabel(f'Entropy at Stream Pixel ({row_stream}, {col_stream})')
# #plt.ylim(0, 1)
# plt.title(f'Entropy across pairs of matrices for Stream pixel: ({row_stream}, {col_stream})')
# plt.grid(True)
# plt.tight_layout()
# plt.show()



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
topo = np.ma.masked_equal(matrices[0], -9999)
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




    
    

