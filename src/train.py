import torch
import mt as mt
import numpy as np

def train(model, criterion, optimizer, num_epochs, input_train_rho, input_train_phi, target_train, input_test_rho, input_test_phi, target_test):
    train_loss_list = []
    test_loss_list = []

    for epoch in range(num_epochs):
        # Forward pass for training set
        outputs_train = model(input_train_rho, input_train_phi)
        loss_train = criterion(outputs_train, target_train)

        # Backward and optimize for training set
        optimizer.zero_grad()
        loss_train.backward()
        optimizer.step()

        # Store the train loss for this epoch
        train_loss_list.append(loss_train.item())

        # Forward pass for test set
        outputs_test = model(input_test_rho, input_test_phi)
        loss_test = criterion(outputs_test, target_test)

        # Store the test loss for this epoch
        test_loss_list.append(loss_test.item())

        # Print the train and test losses every 100 epochs
        if (epoch+1) % 100 == 0:
            print(f'Epoch: {epoch+1}, Train Loss: {loss_train.item():.4f}, Test Loss: {loss_test.item():.4f}')

    return model, train_loss_list, test_loss_list



def train_with_batches(model, criterion, optimizer, num_epochs, input_train_rho, input_train_phi, target_train, input_test_rho, input_test_phi, target_test):
    train_loss_list = []
    test_loss_list = []
    batch_size=128

    train_dataset = torch.utils.data.TensorDataset(input_train_rho, input_train_phi, target_train)
    test_dataset = torch.utils.data.TensorDataset(input_test_rho, input_test_phi, target_test)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    for epoch in range(num_epochs):
        train_epoch_loss = 0.0
        for input_train_rho, input_train_phi, target_train in train_loader:
            # Forward pass for training set
            outputs_train = model(input_train_rho, input_train_phi)
            loss_train = criterion(outputs_train, target_train)

            # Backward and optimize for training set
            optimizer.zero_grad()
            loss_train.backward()
            optimizer.step()

            # Update the train epoch loss
            train_epoch_loss += loss_train.item()

        # Store the average train loss for this epoch
        train_loss_list.append(train_epoch_loss / len(train_loader))

        # Compute the test loss for this epoch
        test_epoch_loss = 0.0
        with torch.no_grad():
            for input_test_rho, input_test_phi, target_test in test_loader:
                # Forward pass for test set
                outputs_test = model(input_test_rho, input_test_phi)
                loss_test = criterion(outputs_test, target_test)

                # Update the test epoch loss
                test_epoch_loss += loss_test.item()

        # Store the average test loss for this epoch
        test_loss_list.append(test_epoch_loss / len(test_loader))

        # Print the train and test losses every 100 epochs
        if (epoch+1) % 25 == 0:
            print(f'Epoch: {epoch+1}, Train Loss: {train_epoch_loss / len(train_loader):.4f}, Test Loss: {test_epoch_loss / len(test_loader):.4f}')

    return model, train_loss_list, test_loss_list

def train_with_physics(model, criterion, optimizer, num_epochs, input_train_rho, input_train_phi, target_train, input_test_rho, input_test_phi, target_test,depth_list,period_list):
    train_loss_list = []
    test_loss_list = []
    total_loss_list = []
    alpha = 0.5
    beta = 0.5


    for epoch in range(num_epochs):
        # Forward pass for training set
        outputs_train = model(input_train_rho, input_train_phi)
        loss_train = criterion(outputs_train, target_train)
        prediction = outputs_train.detach().numpy()
        output1=np.zeros((prediction.shape[0],24),dtype=float)
        output2=np.zeros((prediction.shape[0],24),dtype=float)
        
        
        for i in range(0,prediction.shape[0]-1):
            output1[i], output2[i] = mt.forward_1D_MT(prediction[i], depth_list, (1./period_list))     
      
        input_rho= (output1-np.mean(output1))/np.std(output1)     
        input_phi= (output2-np.mean(output2))/np.std(output2) 
        
        print(np.shape(input_rho))
        loss_rho = criterion(torch.tensor(input_rho[:, :, np.newaxis], dtype=torch.float32), input_train_rho)
        loss_phi = criterion(torch.tensor(input_phi[:, :, np.newaxis], dtype=torch.float32),input_train_phi)
        
        loss_train = criterion(outputs_train, target_train)
        train_loss_list.append(loss_train.item())
        total_loss = alpha*loss_train + beta*(loss_rho+loss_phi) 
        total_loss_list.append(total_loss.item())
        
        
        # Backward and optimize for training set
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # Store the train loss for this epoch
        total_loss_list.append(loss_train.item())

        # Store the train loss for this epoch
        train_loss_list.append(loss_train.item())

        # Forward pass for test set
        outputs_test = model(input_test_rho, input_test_phi)
        loss_test = criterion(outputs_test, target_test)

        # Store the test loss for this epoch
        test_loss_list.append(loss_test.item())

        # Print the train and test losses every 100 epochs
        if (epoch+1) % 1 == 0:
            print(f'Epoch: {epoch+1},Phi Loss: {loss_phi.item():.4f} Rho Loss: {loss_rho.item():.4f} Train Loss: {loss_train.item():.4f}, Test Loss: {loss_test.item():.4f}')


    return model, total_loss_list, test_loss_list


def train_with_batches_physics(model, criterion, optimizer, num_epochs, input_train_rho, input_train_phi, target_train, input_test_rho, input_test_phi, target_test,depth_list,period_list):
    train_loss_list = []
    model_loss_list = []
    test_loss_list = []
    batch_size=128
    total_loss_list = []
    alpha = 0.5
    beta = 0.5
    
    train_dataset = torch.utils.data.TensorDataset(input_train_rho, input_train_phi, target_train)
    test_dataset = torch.utils.data.TensorDataset(input_test_rho, input_test_phi, target_test)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    for epoch in range(num_epochs):
        train_epoch_loss = 0.0
        model_epoch_loss = 0.0
        total_epoch_loss = 0.0
        
        
        
        for input_train_rho, input_train_phi, target_train in train_loader:
            # Forward pass for training set
            outputs_train = model(input_train_rho, input_train_phi)
            loss_train = criterion(outputs_train, target_train)


            prediction = outputs_train.detach().numpy()
            output1=np.zeros((prediction.shape[0],24),dtype=float)
            output2=np.zeros((prediction.shape[0],24),dtype=float)
             
            for i in range(0,prediction.shape[0]-1):
                output1[i], output2[i] = mt.forward_1D_MT(prediction[i], depth_list, (1./period_list))     
          
            input_rho= (output1-np.mean(output1))/np.std(output1)  
            #input_rho= (output1-np.mean(input_train_rho))/np.std(input_train_rho)
            #input_phi= (output2-np.mean(output2))/np.std(output2) 
            
            loss_rho = criterion(torch.tensor(input_rho[:, :, np.newaxis], dtype=torch.float32), input_train_rho)
            #loss_phi = criterion(torch.tensor(input_phi[:, :, np.newaxis], dtype=torch.float32),input_train_phi)
            loss_train = criterion(outputs_train, target_train)
            loss_model = loss_rho #+ loss_rho
            
            total_loss = alpha*loss_train + beta*(loss_model) 

            # Backward and optimize for training set
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            

            # Update the train epoch loss
            train_epoch_loss += loss_train.item()
            model_epoch_loss += loss_model.item()
            total_epoch_loss += total_loss.item()
            
        train_loss_list.append(train_epoch_loss / len(train_loader))      
        model_loss_list.append(model_epoch_loss / len(train_loader))
        total_loss_list.append(total_epoch_loss / len(train_loader))
       
        # Compute the test loss for this epoch
        test_epoch_loss = 0.0
        with torch.no_grad():
            for input_test_rho, input_test_phi, target_test in test_loader:
                # Forward pass for test set
                outputs_test = model(input_test_rho, input_test_phi)
                loss_test = criterion(outputs_test, target_test)

                # Update the test epoch loss
                test_epoch_loss += loss_test.item()

        # Store the average test loss for this epoch
        test_loss_list.append(test_epoch_loss / len(test_loader))

        # Print the train and test losses every 100 epochs
        if (epoch+1) % 1 == 0:
            print(f'Epoch: {epoch+1}, Model loss:{model_epoch_loss / len(train_loader):.4f}, Data Loss: {train_epoch_loss / len(train_loader):.4f}, Test Loss: {test_epoch_loss / len(test_loader):.4f}, TOTAL Loss: {total_epoch_loss / len(train_loader):.4f}')

    return model, train_loss_list, test_loss_list, model_loss_list, total_loss_list

def normalize_data(data):
    """Normalize input data."""
    mean = np.mean(data, axis=(0, 1),keepdims=True)
    std = np.std(data, axis=(0, 1), keepdims=True)
    return (data - mean) / std, mean, std
