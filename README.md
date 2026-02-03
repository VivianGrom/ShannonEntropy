# ShannonEntropy

This repository contains code and examples demonstrating how to apply Shannon Entropy, Moran's I, and Geary's C to analyze Landscape Evolution Model (LEM) outputs. The repository accompanies the manuscript: Methods for Quantifying Spatial and Temporal Variation in Landscape Evolution Model Outputs, Geomorphica, 2026 (under review).


*Files Overview*

Code and notebook:
- 'Comparison.ipynb' – Jupyter notebook demonstrating the computation of Shannon Entropy, Moran’s I, and Geary’s C, and illustrating their application to LEM outputs. This notebook reproduces the analyses used to generate figures in the manuscript.
- 'ent_sautocor_class.py' – Standalone entropy calculation functions that were later integrated into ent_sautocor_class.py.
- 'shannon_entropy.py' – A collection of entropy calculation functions, later integrated into a class.

Case Studies (These scripts generate the processed outputs used for the paper's figures)
- 'exp1.py' – A case study analyzing steady uplift.
- 'exp2.py' – A case study exploring periodic alternating uplift.
- 'exp3.py' – A case study investigating spatially variable uplift.
- 'shannon_edem.py' – An example demonstrating entropy calculations using alternative data grids.


This repository provides a structured approach to quantifying spatial patterns and variability within LEM outputs, facilitating comparative analysis across different uplift scenarios.
