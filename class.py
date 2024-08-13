#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 13:30:53 2024

@author: viviangrom
"""
#%% Import libraries
import numpy as np
from matplotlib import pyplot as plt
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

# Functions to add    
# def plot_morans_i_over_time(matrices, time_steps):
# def plot_gearys_c_over_time(matrices, time_steps):




#%% Class

class Shannon_Entropy:
    def __init__(self, grid, matrices, k = None, pixel = None, pixel_stream = None, 
                 pixel_hillslope = None, stream_threshold = 4, slope_threshold = 0.1, 
                 filter_size = 3, buffer_scaling_factor = 0.00015, minimum_buffer_size = 0.5):
    
        self.grid = grid
        self.matrices = matrices
        self.k = k
        self.pixel = pixel
        self.pixel_stream = pixel_stream
        self.pixel_hillslope = pixel_hillslope
        self.stream_threshold = stream_threshold
        self.slope_threshold = slope_threshold
        self.filter_size = filter_size
        self.buffer_scaling_factor = buffer_scaling_factor
        self.minimum_buffer_size = minimum_buffer_size
        self.n = len(self.matrices)
    
    def calculate_shannon_entropy(self, k = None):
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

        if self.k is None:
            R = np.zeros(self.grid.shape)

            # Calculate the modular difference for each corresponding element
            for i in range(R.shape[0]):
                for j in range(R.shape[1]):
                    R[i, j] = max(abs(self.matrices[k][i, j] - self.matrices[(k + 1) % self.n][i, j]) 
                                  for k in range(self.n))

            self.k = np.max(R)
            print("max k is: ", self.k)

            plt.figure()
            imshow_grid(self.grid, R, grid_units=("m", "m"), cmap='Blues', var_name="Modular Difference")
            plt.title("Modular Difference Matrix R")
            plt.show()

        # Calculate probabilities
        if self.n == 2:
            p1 = abs(self.matrices[0] - self.matrices[1]) / (k * self.n)
            probabilities = [p1, 1 - p1]
        else:
            probabilities = []
            for i in range(self.n):
                j = (i + 1) % self.n  # Wrap around to the beginning if at the end
                probabilities.append(abs(self.matrices[i] - self.matrices[j]) / (self.k * self.n))

        # Handle case where probability is zero
        probabilities = [np.where(p == 0, 1, p) for p in probabilities]

        # Calculate entropy
        entropy_matrix = np.zeros(self.grid.shape)

        for prob in probabilities:
            entropy_matrix -= prob * np.log(prob) / np.log(self.n)

        plt.figure()
        imshow_grid(self.grid, entropy_matrix, cmap='plasma', grid_units=("m", "m"), var_name="Entropy")
        plt.title("Entropy Matrix")
        plt.show()

        return entropy_matrix, probabilities
    
    def pixel_entropy_across_experiments(self):
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
        
        R = np.zeros(self.grid.shape)
        
        # Calculate the modular difference for each corresponding element
        for i in range(R.shape[0]):
            for j in range(R.shape[1]):
                R[i, j] = max(abs(self.matrices[m][i, j] - self.matrices[(m + 1) % self.n][i, j]) for m in range(self.n))
        
        self.k_global = np.max(R)
        print("k global is: ", self.k_global)
        
        max_k_pixel = np.argmax(R)
        max_abs_pixel = np.unravel_index(max_k_pixel, grid.shape)
        print(f"Pixel with highest absolute value: {max_abs_pixel} -> Value: {R[max_abs_pixel]}")
        
        if self.pixel is None:
            row, col = max_abs_pixel
        else:
            row, col = self.pixel
            
        entropies = []
        for i in range(self.n-1):
            j = i + 1  # Compare with the next matrix
            
            # Calculate probabilities
            diff_matrix = abs(self.matrices[i] - self.matrices[j])
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
        #plt.xticks(np.arange(0, 271, 20))  # Show ticks every 15 time steps
        plt.xticks(np.arange(0, self.n+1))
        plt.xlabel('Time Step')
        plt.ylabel(f'Entropy at Pixel ({row}, {col})')
        plt.ylim(0,1)
        plt.title(f'Entropy across pairs of matrices for pixel:({row}, {col})')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        return entropies
    
    def differentiate_hillslopes_and_streams(self):
        """
        Function to differentiate hillslopes and streams based on drainage area and slope thresholds.
        
        Parameters:
        - grid (RasterModelGrid): The Landlab model grid.
        - matrices: List of elevation arrays at different time steps.
        - stream_threshold (float): Threshold for drainage area to classify as stream.
        - slope_threshold (float): Threshold for slope to classify as stream.

        Returns:
        - tot_stream_mask (np.ndarray): Total Boolean mask indicating stream nodes across all time steps.
        """
        stream_masks = []
        dxy = self.grid.dx

        for matrix in self.matrices:
            # Set the elevation data to the grid
            self.grid.at_node['topographic__elevation'] = matrix

            # Calculate drainage area
            fa = FlowAccumulator(self.grid, flow_director="FlowDirectorD8")
            fa.run_one_step()
            drainage_area = self.grid.at_node['drainage_area']

            # Calculate slope
            slope = self.grid.calc_slope_at_node()

            # Reshape for plotting and classification
            drainage_area = drainage_area.reshape(self.grid.shape)/(dxy**2)
            # print(drainage_area)
            slope = slope.reshape(self.grid.shape)

            # Classify nodes
            stream_mask = (drainage_area > stream_threshold)
            stream_masks.append(stream_mask)

        # Sum the stream masks
        tot_stream_mask = np.any(stream_masks, axis=0)
        plt.figure()
        imshow_grid(self.grid, tot_stream_mask.astype(int), grid_units=("m", "m"), var_name="Total Stream Mask", cmap="Blues")
        plt.title("Total Streams Mask")
        plt.show()

        return tot_stream_mask
    
    def differentiate_hillslopes_and_streams_buffer(self):
        """
        Function to differentiate hillslopes and streams based on drainage area and slope thresholds with buffer.
        
        Parameters:
        - grid (RasterModelGrid): The Landlab model grid.
        - matrices: List of elevation arrays at different time steps.
        - stream_threshold (float): Threshold for drainage area to classify as stream.
        - slope_threshold (float): Threshold for slope to classify as stream.
        - filter_size (int): Size of the maximum filter to create buffer around streams.

        Returns:
        - tot_stream_mask (np.ndarray): Total Boolean mask indicating stream nodes across all time steps.
        """
        stream_masks = []
        dxy = self.grid.dx
        slopes = []

        for matrix in self.matrices:
            # Set the elevation data to the grid
            self.grid.at_node['topographic__elevation'] = matrix

            # Calculate drainage area
            fa = FlowAccumulator(self.grid, flow_director="FlowDirectorD8")
            fa.run_one_step()
            drainage_area = self.grid.at_node['drainage_area']

            # Calculate slope
            slope = self.grid.calc_slope_at_node()

            # Reshape for plotting and classification
            drainage_area = drainage_area.reshape(self.grid.shape)/(dxy**2)
            # print(drainage_area)
            slope = slope.reshape(self.grid.shape)
            slopes.append(slope)

            # Classify nodes
            stream_mask = (drainage_area > self.stream_threshold) #| (slope > self.slope_threshold)
            stream_masks.append(stream_mask)

        # Sum the stream masks
        tot_stream_mask = np.any(stream_masks, axis=0)
        buffered_stream_mask = maximum_filter(tot_stream_mask.astype(int), size=self.filter_size).astype(bool)

        plt.figure()
        imshow_grid(self.grid, buffered_stream_mask.astype(int), grid_units=("m", "m"), var_name="Buffered Stream Mask", cmap="Blues")
        plt.title("Buffered Streams Mask")
        plt.show()
        
        max_slope = np.max(slopes)
        print("max slope is: ", max_slope)

        return buffered_stream_mask


    def differentiate_hillslopes_and_streams_scaled_buffer(self):
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

        for matrix in self.matrices:
            # Set the elevation data to the grid
            self.grid.at_node['topographic__elevation'] = matrix

            # Calculate drainage area
            fa = FlowAccumulator(self.grid, flow_director="FlowDirectorD8")
            fa.run_one_step()
            drainage_area = self.grid.at_node['drainage_area']

            # Calculate slope
            slope = self.grid.calc_slope_at_node()

            # Reshape for plotting and classification
            drainage_area = drainage_area.reshape(self.grid.shape) / (dxy**2)
            slope = slope.reshape(self.grid.shape)
            slopes.append(slope)

            # Classify nodes
            stream_mask = (drainage_area > self.stream_threshold)
            stream_masks.append(stream_mask)

        # Sum the stream masks
        tot_stream_mask = np.any(stream_masks, axis=0)

        # Calculate buffer size dynamically based on drainage area
        buffer_size_array = (drainage_area * self.buffer_scaling_factor).astype(int)
        buffer_size_array[buffer_size_array < 1] = self.minimum_buffer_size  # Ensure minimum buffer size is 1

        # Initialize the buffered stream mask
        buffered_stream_mask = np.zeros_like(tot_stream_mask)

        for i in range(self.grid.shape[0]):
            for j in range(self.grid.shape[1]):
                if tot_stream_mask[i, j]:
                    buffer_size = buffer_size_array[i, j]
                    buffered_stream_mask[max(0, i-buffer_size):min(self.grid.shape[0], i+buffer_size+1),
                                         max(0, j-buffer_size):min(self.grid.shape[1], j+buffer_size+1)] = 1

        buffered_stream_mask_scaled = buffered_stream_mask.astype(bool)

        plt.figure()
        imshow_grid(self.grid, buffered_stream_mask.astype(int), grid_units=("m", "m"), var_name="Buffered Stream Mask", cmap="Blues")
        plt.title("Buffered Streams Mask - Scaled")
        plt.show()

        max_slope = np.max(slopes)
        print("max slope is: ", max_slope)

        return buffered_stream_mask_scaled
    
    def calculate_shannon_entropy_mask(self, stream_mask):
        """
        Calculate the entropy based on the difference between matrices,
        separately for stream and hillslope areas based on a mask.

        Parameters:
        - grid: RasterModelGrid object
        - matrices: List of matrices (numpy arrays) to compare
        - stream_mask: Boolean mask indicating stream nodes

        Returns:
        - entropy_matrix_stream: The calculated entropy matrix for stream regions
        - entropy_matrix_hillslope: The calculated entropy matrix for hillslope regions
        - probabilities_stream: The calculated probabilities for each element in stream regions
        - probabilities_hillslope: The calculated probabilities for each element in hillslope regions
        """

        # Calculate k for stream regions if not provided
        R_stream = np.zeros(self.grid.shape)
        for i in range(R_stream.shape[0]):
            for j in range(R_stream.shape[1]):
                if stream_mask[i, j]:
                    R_stream[i, j] = max(abs(self.matrices[m][i, j] - self.matrices[(m + 1) % self.n][i, j]) for m in range(self.n))
        k_stream = np.max(R_stream)
        print("k_stream is: ", k_stream)

        # Calculate k for hillslope regions if not provided
        R_hillslope = np.zeros(self.grid.shape)
        for i in range(R_hillslope.shape[0]):
            for j in range(R_hillslope.shape[1]):
                if not stream_mask[i, j]:
                    R_hillslope[i, j] = max(abs(self.matrices[m][i, j] - self.matrices[(m + 1) % self.n][i, j]) for m in range(self.n))
        k_hillslope = np.max(R_hillslope)
        print("k_hillslope is: ", k_hillslope)

        # Calculate probabilities
        probabilities_stream = []
        probabilities_hillslope = []
        for i in range(self.n):
            j = (i + 1) % self.n  # Wrap around to the beginning if at the end
            prob_stream = abs(self.matrices[i] - self.matrices[j]) / (k_stream * self.n)
            prob_hillslope = abs(self.matrices[i] - self.matrices[j]) / (k_hillslope * self.n)
            probabilities_stream.append(prob_stream)
            probabilities_hillslope.append(prob_hillslope)

        # Handle case where probability is zero
        probabilities_stream = [np.where(p == 0, 1, p) for p in probabilities_stream]
        probabilities_hillslope = [np.where(p == 0, 1, p) for p in probabilities_hillslope]

        # Initialize entropy matrices for stream and hillslope regions
        entropy_matrix_stream = np.zeros(self.grid.shape)
        entropy_matrix_hillslope = np.zeros(self.grid.shape)

        for prob_stream, prob_hillslope in zip(probabilities_stream, probabilities_hillslope):
            entropy_matrix_stream[stream_mask] -= prob_stream[stream_mask] * np.log(prob_stream[stream_mask]) / np.log(self.n)
            entropy_matrix_hillslope[~stream_mask] -= prob_hillslope[~stream_mask] * np.log(prob_hillslope[~stream_mask]) / np.log(self.n)

        # Set entropy values of non-relevant regions to a high value for visualization
        entropy_matrix_stream[~stream_mask] = np.nan
        entropy_matrix_hillslope[stream_mask] = np.nan
        
        # Create masked arrays to handle np.nan values
        masked_entropy_matrix_stream = np.ma.masked_invalid(entropy_matrix_stream)
        masked_entropy_matrix_hillslope = np.ma.masked_invalid(entropy_matrix_hillslope)


        # Plot entropy matrices for stream and hillslope regions separately
        plt.figure()
        imshow_grid(self.grid, masked_entropy_matrix_stream, cmap='plasma', grid_units=("m", "m"), var_name="Entropy (Stream Regions)", colorbar_label='Entropy')
        plt.title("Entropy Matrix (Stream Regions)")
        plt.show()

        plt.figure()
        imshow_grid(self.grid, masked_entropy_matrix_hillslope, cmap='plasma', grid_units=("m", "m"), var_name="Entropy (Hillslope Regions)", colorbar_label='Entropy')
        plt.title("Entropy Matrix (Hillslope Regions)")
        plt.show()

        return masked_entropy_matrix_stream, masked_entropy_matrix_hillslope, probabilities_stream, probabilities_hillslope
    
    def pixel_entropy_across_experiments_mask(self, mask):
        """
        Calculate the entropy based on the difference between matrices.

        Parameters:
        - grid: RasterModelGrid object
        - matrices: List of matrices (numpy arrays) to compare
        - pixel: Tuple of (row, col) coordinates (optional)

        Returns:
        - entropies: The calculated Shannon entropies for each pair of experiments,
        with a local k derived from the absolute difference between the pair 
        of models
        """
        
        # Calculate the modular difference for the streams
        R_stream = np.zeros(self.grid.shape)
        for i in range(R_stream.shape[0]):
            for j in range(R_stream.shape[1]):
                if mask[i, j]:
                    R_stream[i, j] = max(abs(self.matrices[m][i, j] - self.matrices[(m + 1) % self.n][i, j]) for m in range(self.n))
        k_stream = np.max(R_stream)
        print("k_stream is: ", k_stream)
        
        max_k_pixel_stream = np.argmax(R_stream)
        max_abs_pixel_stream = np.unravel_index(max_k_pixel_stream, self.grid.shape)
        print(f"Pixel with highest absolute value: {max_abs_pixel_stream} -> Value: {R_stream[max_abs_pixel_stream]}")

        # Calculate the modular difference for the hillslopes
        R_hillslope = np.zeros(self.grid.shape)
        for i in range(R_hillslope.shape[0]):
            for j in range(R_hillslope.shape[1]):
                if not mask[i, j]:
                    R_hillslope[i, j] = max(abs(self.matrices[m][i, j] - self.matrices[(m + 1) % self.n][i, j]) for m in range(self.n))
        k_hillslope = np.max(R_hillslope)
        print("k_hillslope is: ", k_hillslope)
        
        max_k_pixel_hlope = np.argmax(R_hillslope)
        max_abs_pixel_hlope = np.unravel_index(max_k_pixel_hlope, self.grid.shape)
        print(f"Pixel with highest absolute value: {max_abs_pixel_hlope} -> Value: {R_hillslope[max_abs_pixel_hlope]}")
        
        # Assign pixel for stream values
        if self.pixel_stream is None:
            row_stream, col_stream = max_abs_pixel_stream
        else:
            row_stream, col_stream = self.pixel_stream

        # Assign pixel for hillslope values
        if self.pixel_hillslope is None:
            row_hillslope, col_hillslope = max_abs_pixel_hlope
        else:
            row_hillslope, col_hillslope = self.pixel_hillslope
                  
        entropies_stream = []
        entropies_hillslope = []
        
        mod_stream = []
        mod_hillslope = []

        for i in range(self.n - 1):
            j = i + 1  # Compare with the next matrix

            # Calculate probabilities for stream region
            diff_matrix_stream = abs(self.matrices[i] - self.matrices[j]) * mask
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
            diff_matrix_hillslope = abs(self.matrices[i] - self.matrices[j]) * (~mask)
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
        
        plt.figure()
        topo = np.ma.masked_equal(self.matrices[0], -9999)
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
    
class Spatial_Autocorrelation:
    def __init__(self, grid, matrices, time_steps):
    
        self.grid = grid
        self.matrices = matrices
        self.time_steps = time_steps
        
    def plot_morans_i_over_time(self):
        """
        Calculates and plots Moran's I and p-values over multiple time steps.

        Parameters:
        - matrices: list of 2D numpy arrays representing different time steps.
        - time_steps: list of integers representing the time steps.
        """
        # Create the matrix of weights 
        w = lat2W(self.matrices[0].shape[0], self.matrices[0].shape[1])

        # Calculate Moran's I and p-values for each matrix
        moran_values = []
        p_values = []
        for matrix in self.matrices:
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
        ax1.plot(self.time_steps, moran_values, marker='o', color=color, label="Moran's I")
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.axhline(y=0, color='red', linestyle='--', label="Moran's I = 0")
        ax1.set_ylim(-1, 1)

        # Annotate each point with Moran's I value
        for i, mi_val in enumerate(moran_values):
            ax1.annotate(f'{mi_val:.3f}', (self.time_steps[i], mi_val), textcoords="offset points", xytext=(0,10), ha='center')

        # Create a second y-axis for p-values
        ax2 = ax1.twinx()
        color = 'tab:green'
        ax2.set_ylabel("p-value", color=color)
        ax2.plot(self.time_steps, p_values, marker='s', linestyle='--', color=color, label="p-value")
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.set_ylim(-0.01, 0.1)

        # Annotate each point with p-value
        for i, p_val in enumerate(p_values):
            ax2.annotate(f'{p_val:.2f}', (self.time_steps[i], p_val), textcoords="offset points", xytext=(0,-15), ha='center')

        fig.tight_layout()  # Adjust layout to make room for annotations
        fig.suptitle("Moran's I and p-value over Time Steps", y=1)
        plt.show()
        
    def plot_gearys_c_over_time(self):
        """
        Calculates and plots Geary's C and p-values over multiple time steps.

        Parameters:
        - matrices: list of 2D numpy arrays representing different time steps.
        - time_steps: list of integers representing the time steps.
        """
        # Create the matrix of weights 
        w = lat2W(self.matrices[0].shape[0], self.matrices[0].shape[1])

        # Calculate Geary's C and p-values for each matrix
        geary_values = []
        p_values = []
        for matrix in self.matrices:
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
        ax1.plot(self.time_steps, geary_values, marker='o', color=color, label="Geary's C")
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.axhline(y=1, color='blue', linestyle='--', label="Geary's C = 1")
        ax1.set_ylim(0, 2)

        # Annotate each point with Geary's C value
        for i, gc_val in enumerate(geary_values):
            ax1.annotate(f'{gc_val:.3f}', (self.time_steps[i], gc_val), textcoords="offset points", xytext=(0,10), ha='center')

        # Create a second y-axis for p-values
        ax2 = ax1.twinx()
        color = 'tab:green'
        ax2.set_ylabel("p-value", color=color)
        ax2.plot(self.time_steps, p_values, marker='s', linestyle='--', color=color, label="p-value")
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.set_ylim(-0.01, 0.1)

        # Annotate each point with p-value
        for i, p_val in enumerate(p_values):
            ax2.annotate(f'{p_val:.2f}', (self.time_steps[i], p_val), textcoords="offset points", xytext=(0,-15), ha='center')

        fig.tight_layout()  # Adjust layout to make room for annotations
        fig.suptitle("Geary's C and p-value over Time Steps", y=1)
        plt.show()


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

matrices = mg1_snapshots  # Add more matrices as needed

# Define time steps
time_steps = [4500, 5000, 5500, 6000]

#%% Test
# Example setup for the class
grid = mg1  # Replace with your RasterModelGrid object
matrices = matrices  # Example matrices, replace with your actual data
k = None  # or some integer value
pixel = None
pixel_stream = None
pixel_hillslope = None
stream_threshold = 4
slope_threshold = 0.1
filter_size = 2
buffer_scaling_factor = 0.03
minimum_buffer_size = 0.05

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
mskd_ent_matrix_stream, mskd_ent_matrix_hillslope, _, _ = entropy_calculator.calculate_shannon_entropy_mask(stream_mask = tot_stream_mask)
entropies_stream_array, entropies_hillslope_array = entropy_calculator.pixel_entropy_across_experiments_mask(mask = tot_stream_mask)
  

# Create an instance of Spatial_Autocorrelation 
spatial_autocorr_calculator = Spatial_Autocorrelation(
    grid=grid,
    matrices=matrices,    
    time_steps=time_steps
)

# Call the method    
spatial_autocorr_calculator.plot_morans_i_over_time()
spatial_autocorr_calculator.plot_gearys_c_over_time()
    
    
    
    
    
    
    
    
    
    
