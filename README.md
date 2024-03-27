# 1D Magnetotelluric Inversion using Physics-Informed CNN

### Author: R. Corseri

## Overview 

This repository contains code for performing 1D magnetotelluric (MT) inversion using a physics-informed convolutional neural network (CNN). The project aims to leverage deep learning techniques to perform 1D magnetotelluric inversion. Magnetotellurics is a geophysical method used for imaging electrical conductivity of the subsurface.


## Features

- Implementation of a multihead CNN architecture tailored for 1D MT inversion.
- Integration of physics-based constraints into the CNN model to improve inversion results.
- Input data preprocessing and normalization routines.
- Training and evaluation scripts for the CNN model.
- Utilities for data loading, visualization, and analysis.

## Project Structure

- models/ : Contains the definition of the CNN architecture used for 1D MT inversion. The directory also stores the trained models. 
- src/ : Includes source code files for data preprocessing, model training, evaluation, and utilities.
- data/ : Placeholder directory for storing input data (e.g., apparent resistivity and phase data).
- tests/ : Directory for storing the field data, outputs of the tests and visualization
- report/ : contains the report

## Reproducibility

- To train the models, plot the loss function and visualize the output for an example: run main.py
- To test the trained models on real data: run tests/test_model_on_field_data.py

## Dependencies

    Python (>=3.6)
    PyTorch (>=1.9.0)
    NumPy
    Pandas
    Matplotlib
    scikit-learn
 
