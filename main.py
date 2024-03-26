
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
from mt import plot_1D_model 

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



# Load input data
apparent_resistivity_data =pd.read_csv('./data/processed/app_rho_data.csv').to_numpy()
phase_data = pd.read_csv('./data/processed/phi_data.csv').to_numpy()
target_data = pd.read_csv('./data/target/target_training.csv').to_numpy()

# Normalize input data
normalized_rho,mean_rho, std_rho = normalize_data(apparent_resistivity_data)
normalized_phi,mean_phi, std_phi = normalize_data(phase_data)
#normalized_target, mean_target, std_target = normalize_data(target_data)

# Convert normalized data into tensors
input_tensor_rho = torch.tensor(normalized_rho[:, :, np.newaxis], dtype=torch.float32)
input_tensor_phi = torch.tensor(normalized_phi[:, :, np.newaxis], dtype=torch.float32)
target_tensor = torch.tensor(target_data , dtype=torch.float32)

# Split data into training and test 
random_state = 42
input_train_rho, input_test_rho, target_train, target_test = train_test_split(input_tensor_rho, target_tensor,test_size=0.2,random_state=random_state)
input_train_phi, input_test_phi, target_train_p, target_test_p = train_test_split(input_tensor_phi, target_tensor,test_size=0.2,random_state=random_state)

# Instantiate the model, loss function, and optimizer
num_epochs = 200
learning_rate = 0.01
model = CNNmultihead()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

#Training
#trained_model, train_loss_list, test_loss_list = train(model, criterion, optimizer, num_epochs, input_train_rho, input_train_phi, target_train, input_test_rho, input_test_phi, target_test)
#trained_model, train_loss_list, test_loss_list = train_with_physics(model, criterion, optimizer, num_epochs, input_train_rho, input_train_phi, target_train, input_test_rho, input_test_phi, target_test,depth_list,period_list)
trained_model, train_loss_list, test_loss_list, model_loss_list, total_loss_list = train_with_batches_physics(model, criterion, optimizer, num_epochs, input_train_rho, input_train_phi, target_train, input_test_rho, input_test_phi, target_test,depth_list,period_list)


# Plotting the loss function for training and test sets
plt.figure(figsize=(10, 6))
plt.semilogy(train_loss_list[:50], label='Training Loss', color='blue')
plt.semilogy(test_loss_list[:50], label='Test Loss', color='red')
plt.semilogy(model_loss_list[:50], label='model Loss', color='magenta')
plt.semilogy(total_loss_list[:50], label='Total Loss', color='black')


plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title("Training and Test Loss Over Epochs")
plt.show()


#### Testing the model on a specific example
example_index = 1
outputs_example = trained_model(input_test_rho[example_index:example_index+1, :, :], input_test_phi[example_index:example_index+1, :, :])


# Visualizing the results for the example
plt.semilogx(depth_list, target_test[example_index], 'b-', label='Target 1D model')
plt.semilogx(depth_list, outputs_example.detach().numpy().flatten(), 'r-', label='Predicted 1D model')
plt.title(f"Example {example_index} - Target vs Predicted")
plt.xlabel('Depth')
plt.ylabel(' Resistivity')
plt.legend()
plt.show()



mt.plot_1D_model(target_test[example_index],depth_list)
