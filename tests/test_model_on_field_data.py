# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 21:44:45 2024

@author: romain
"""


import sys
sys.path.append('./src')
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torchvision import transforms
from models.cnn.CNN_multihead import CNNmultihead, CNNmultihead_with_physics
from train import train, train_with_batches, train_with_physics, train_with_batches_physics, normalize_data
import mt as mt

# Set path to EDI files folder
edi_folder = './field_data/'

# Get list of EDI files in the folder
edi_files = [file for file in os.listdir(edi_folder) if file.endswith('.edi')]
depth_list = pd.read_csv('../data/processed/depth_list.csv').to_numpy().squeeze()
period_list = 1. / pd.read_csv('../data/processed/period_list.csv').to_numpy().squeeze()


# Load trained CNN model
model = CNNmultihead()
model.load_state_dict(torch.load('../models/cnn/cnn_with_physics.pth'))
model.eval()

# Loop through each EDI file
for edi_file in edi_files:
    # Read EDI file
    df4, _, _ = mt.readEDI(os.path.join(edi_folder, edi_file))
    ZR, ZI, _, _ = mt.Zdet(df4)
    freq = df4.to_numpy()[:, 0]
    rho_field = ((ZR ** 2 + ZI ** 2) * 0.2 / freq)
    phase_field = np.degrees(np.arctan2(ZI, ZR))

    # Normalize input data
    normalized_rho = normalize_data(rho_field.reshape(1, -1))[0]
    normalized_phi = normalize_data(phase_field.reshape(1, -1))[0]

    # Convert normalized data into tensors
    input_tensor_rho = torch.tensor(normalized_rho[:, :, np.newaxis], dtype=torch.float32)
    input_tensor_phi = torch.tensor(normalized_phi[:, :, np.newaxis], dtype=torch.float32)

    # Use the trained model for prediction
    outputs = model(input_tensor_rho, input_tensor_phi)

    # Plot predicted model and its response
    receiver_name = os.path.splitext(edi_file)[0]
    
    # Plot predicted model
    plt.figure()
    mt.plot_1D_model(outputs.detach().numpy().flatten(), depth_list, color='g', label='Predicted model')
    plt.title(f'Predicted Model for {receiver_name}')
    plt.savefig(f'{receiver_name}_predicted_model.png')
    plt.show()

    # Plot model and data responses
    plt.figure()
    rho, phi = mt.forward_1D_MT(outputs.detach().numpy().flatten(), depth_list, (1. / period_list))
    mt.plot_rho_phi(rho, phi, period_list, color='g', label='Model')
    mt.plot_rho_phi(rho_field, phase_field, 1. / freq, color='r', label='Data')
    plt.tight_layout()
    plt.title(f'Model and Data Responses for {receiver_name}')
    plt.savefig(f'{receiver_name}_model_and_data_responses.png')
    plt.show()
