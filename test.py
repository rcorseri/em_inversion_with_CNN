# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 10:59:37 2024

@author: romain
"""

import sys
sys.path.append('./src')


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


### Read field data
df4,b,c = mt.readEDI('./tests/field_data/BS1305_H5_14Rx036a.edi')
ZR,ZI,dd,ff=mt.Zdet(df4)
test2 = df4.to_numpy()
freq = test2[:,0]
rho_field = ((ZR**2+ZI**2)*0.2/(freq))
phase_field = np.degrees(np.arctan2(ZI,ZR))
depth_list = pd.read_csv('./data/processed/depth_list.csv').to_numpy()
period_list = pd.read_csv('./data/processed/period_list.csv').to_numpy()
depth_list =  np.squeeze(depth_list)
period_list = 1./(np.squeeze(period_list))


### Normalize input data
normalized_rho,mean_rho, std_rho = normalize_data(rho_field.reshape(-1,1))
normalized_phi,mean_phi, std_phi = normalize_data(phase_field.reshape(-1,1))


#test4=np.concatenate([rho[ np.newaxis,:,np.newaxis], phase[np.newaxis,:,np.newaxis]], axis=2)
test4 = normalized_rho[np.newaxis,:,np.newaxis]
#test5 = test4.transpose((0,2,1))



### Convert Field data to tensor
field_tensor_rho = torch.tensor(
 test4,
    dtype=torch.float32
)

field_tensor_rho = field_tensor_rho.permute(0, 2, 1)



test8 = normalize_phi[np.newaxis,:,np.newaxis]
test9=test8.transpose((0,2,1))


field_tensor_phase= torch.tensor(
 test5,
    dtype=torch.float32
)

field_tensor_phase = field_tensor_phase.permute(0, 2, 1)



###Load trained CNN
# Load the trained model
model = CNNmultihead()  # Instantiate the model
model.load_state_dict(torch.load('./models/cnn/cnn_no_physics.pth'))
model.eval()  # Set model to evaluation mode


# Use the trained model for prediction
outputs = model(input_tensor_rho, input_tensor_phi)


#Plot predicted model and its response
mt.plot_1D_model(outputs_example.detach().numpy().flatten(),depth_list,color='g',label='Predicted model')
plt.show()


rho, phi = mt.forward_1D_MT(outputs[example_index].detach().numpy().flatten(), depth_list, (1./period_list)) 
mt.plot_rho_phi(rho, phi, period_list, color='g', label='Model')
mt.plot_rho_phi(rho_field, phase_field, period_list, color='r', label='Data')
plt.tight_layout()
plt.show()






