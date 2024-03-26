
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
from models.cnn.CNN_multihead import CNNmultihead, ResNet_CNNmultihead
from train import train, train_with_batches, train_with_physics, train_with_batches_physics, normalize_data
import mt as mt


### Load input data
apparent_resistivity_data =pd.read_csv('./data/processed/app_rho_data.csv').to_numpy()
phase_data = pd.read_csv('./data/processed/phi_data.csv').to_numpy()
target_data = pd.read_csv('./data/target/target_training.csv').to_numpy()
depth_list = pd.read_csv('./data/processed/depth_list.csv').to_numpy()
period_list = pd.read_csv('./data/processed/period_list.csv').to_numpy()
depth_list =  np.squeeze(depth_list)
period_list = 1./(np.squeeze(period_list))

### Normalize input data
normalized_rho,mean_rho, std_rho = normalize_data(apparent_resistivity_data)
normalized_phi,mean_phi, std_phi = normalize_data(phase_data)
#normalized_target, mean_target, std_target = normalize_data(target_data)

### Convert normalized data into tensors
input_tensor_rho = torch.tensor(normalized_rho[:, :, np.newaxis], dtype=torch.float32)
input_tensor_phi = torch.tensor(normalized_phi[:, :, np.newaxis], dtype=torch.float32)
target_tensor = torch.tensor(target_data , dtype=torch.float32)

### Split data into training and test 
random_state = 42
input_train_rho, input_test_rho, target_train, target_test = train_test_split(input_tensor_rho, target_tensor,test_size=0.2,random_state=random_state)
input_train_phi, input_test_phi, target_train_p, target_test_p = train_test_split(input_tensor_phi, target_tensor,test_size=0.2,random_state=random_state)

### Instantiate the model, loss function, and optimizer
num_epochs = 20
learning_rate = 0.01
model = CNNmultihead()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

### Training CNN

#trained_model, train_loss_list, test_loss_list = train(model, criterion, optimizer, num_epochs, input_train_rho, input_train_phi, target_train, input_test_rho, input_test_phi, target_test)
#trained_model, train_loss_list, test_loss_list = train_with_physics(model, criterion, optimizer, num_epochs, input_train_rho, input_train_phi, target_train, input_test_rho, input_test_phi, target_test,depth_list,period_list)
trained_model, train_loss_list, test_loss_list, model_loss_list, total_loss_list = train_with_batches_physics(model, criterion, optimizer, num_epochs, input_train_rho, input_train_phi, target_train, input_test_rho, input_test_phi, target_test,depth_list,period_list)


### Plotting the loss function for training and test sets
plt.figure(figsize=(10, 6))
plt.semilogy(train_loss_list[:], label='Training Loss', color='blue')
plt.semilogy(test_loss_list[:], label='Test Loss', color='red')
plt.semilogy(model_loss_list[:], label='Model Loss', color='magenta')
plt.semilogy(total_loss_list[:], label='Total Loss', color='black')

plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title("Training and Test Loss Over Epochs")
plt.show()


#### Testing the model on a specific example
example_index = 190
outputs_example = trained_model(input_test_rho[example_index:example_index+1, :, :], input_test_phi[example_index:example_index+1, :, :])
mt.plot_1D_model(target_test[example_index],depth_list,color='r',label='Target model')
mt.plot_1D_model(outputs_example.detach().numpy().flatten(),depth_list,color='g',label='Predicted model')
plt.show()

rho, phi = mt.forward_1D_MT(outputs_example.detach().numpy().flatten(), depth_list, (1./period_list)) 
rho2, phi2 = mt.forward_1D_MT(target_test[example_index].numpy(), depth_list, (1./period_list)) 
mt.plot_rho_phi(rho, phi,period_list,color='g',label='Model')
mt.plot_rho_phi(rho2, phi2,period_list,color='r',label='Data')
plt.tight_layout()
plt.show()