# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 10:44:42 2024

@author: romain
"""

#data loader
# data_loader.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torchvision import transforms
import mt as mt
import modem3d as modem
from scipy.constants import mu_0


def load_data():
    # Load input data
    df = pd.read_csv('./input/app_rho_data.csv')
    df2 = pd.read_csv('./input/phi_data.csv')

    apparent_resistivity_data = df.to_numpy()
    phase_data = df2.to_numpy()
    
    
    
# Common periods for all receivers
    period_list = np.array([0.00020004, 0.000667557, 0.0010016, 0.00138523, 0.00190876, 0.00267523,
                       0.00401284, 0.00552792, 0.0077101, 0.0108507, 0.016276, 0.0225378, 0.0308356,
                       0.0450653, 0.0651042, 0.0901713, 0.12335, 0.180278, 0.260417, 0.36062, 0.49334,
                       0.720981, 1.04167, 1.44238])

    # Load target data (1D resistivity models with 53 layers)
    depth_list = np.array([0, 50, 105, 165.5, 232.05, 305.255, 385.781, 474.359, 571.795, 678.974,
                          796.871, 926.558, 1069.21, 1226.14, 1398.75, 1588.62, 1797.49, 2027.24, 2279.96,
                          2557.96, 2863.75, 3200.13, 3570.14, 3977.15, 4424.87, 4917.35, 5459.09, 6055,
                          6710.5, 7431.55, 8224.7, 9097.17, 10056.9, 11112.6, 12273.8, 13551.2, 14956.3,
                          16502, 18202.2, 20072.4, 22129.6, 24392.6, 26881.8, 29620, 32632, 35945.2,
                          39589.8, 43598.7, 48008.6, 52859.5, 57710.5, 65471.5, 77889.5])

    df3 = pd.read_csv('./target/target_training.csv',)
    target_data = df3.to_numpy()

    input_tensor = torch.tensor(
        np.concatenate([apparent_resistivity_data[:, :, np.newaxis], phase_data[:, :, np.newaxis]], axis=2),
        dtype=torch.float32
    )

    target_tensor = torch.tensor(target_data, dtype=torch.float32)

    # Split the data into training and test sets (80% - 20%)
    input_train, input_test, target_train, target_test = train_test_split(
        input_tensor, target_tensor, test_size=0.2, random_state=42
    )

    # Define normalization parameters
    mean = input_train.mean(dim=(0, 1), keepdim=True)
    std = input_train.std(dim=(0, 1), keepdim=True)

    # Create a normalization transform
    normalize = transforms.Normalize(mean=mean, std=std)

    # Apply normalization to training and test sets
    input_train = normalize(input_train)
    input_test = normalize(input_test)

    return input_train, input_test, target_train, target_test
