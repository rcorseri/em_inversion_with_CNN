# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 10:01:09 2024

@author: romain
"""

import torch
import torch.nn as nn


conv1_filters = 16
conv2_filters = 32
k = 3
dense_nodes = 250
min_val = -1
max_val = 4
threshold = 0.1


class CNNmultihead(nn.Module):
    def __init__(self):
        super(CNNmultihead, self).__init__()

        # Entry 1
        self.layer1 = nn.Conv1d(1, conv1_filters, kernel_size=k, stride=1, padding=0)
        self.relu1 = nn.ReLU()
        self.layer2 = nn.Conv1d(conv1_filters, conv1_filters, kernel_size=k, stride=1, padding=0)
        self.relu2 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.layer3 = nn.Conv1d(conv1_filters, conv2_filters, kernel_size=k, stride=1, padding=0)
        self.relu3 = nn.ReLU()
        self.layer4 = nn.Conv1d(conv2_filters, conv2_filters, kernel_size=k, stride=1, padding=0)
        self.relu4 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.flatten = nn.Flatten()

        # Entry 2
        self.layer1_2 = nn.Conv1d(1, conv1_filters, kernel_size=k, stride=1, padding=0)
        self.relu1_2 = nn.ReLU()
        self.layer2_2 = nn.Conv1d(conv1_filters, conv1_filters, kernel_size=k, stride=1, padding=0)
        self.relu2_2 = nn.ReLU()
        self.pool1_2 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.layer3_2 = nn.Conv1d(conv1_filters, conv2_filters, kernel_size=k, stride=1, padding=0)
        self.relu3_2 = nn.ReLU()
        self.layer4_2 = nn.Conv1d(conv2_filters, conv2_filters, kernel_size=k, stride=1, padding=0)
        self.relu4_2 = nn.ReLU()
        self.pool2_2 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.flatten_2 = nn.Flatten()

        # Dense layers
        self.fc1 = nn.Linear(192, dense_nodes)
        self.relu5 = nn.ReLU()
        self.fc2 = nn.Linear(dense_nodes, 53)
        
        #normalize output (soft constrain) to 0.01 - 1000 ohm.m
        self.sigmoid6 = nn.Sigmoid()
        self.softmax6 = nn.Softmax()
        self.relu6 = nn.ReLU()
        
    def forward(self, x, y):
        x = self.pool1(self.relu2(self.layer2(self.relu1(self.layer1(x.permute(0, 2, 1))))))
        x = self.pool2(self.relu4(self.layer4(self.relu3(self.layer3(x)))))

        y = self.pool1_2(self.relu2_2(self.layer2_2(self.relu1_2(self.layer1_2(y.permute(0, 2, 1))))))
        y = self.pool2_2(self.relu4_2(self.layer4_2(self.relu3_2(self.layer3_2(y)))))

        
        x = self.flatten(x)
        y = self.flatten_2(y)
      

        out = torch.cat((x, y), dim=1)  

        out = self.relu5(self.fc1(out))
        out = self.fc2(out)
       
        
        #out =  min_val + (max_val - min_val) * self.sigmoid6(out)
        out = self.relu6(out)
        
        return out
    
    

class CNNmultihead_with_physics(nn.Module):
    def __init__(self):
        super(CNNmultihead_with_physics, self).__init__()

        # Entry 1
        self.layer1 = nn.Conv1d(1, conv1_filters, kernel_size=k, stride=1, padding=0)
        self.relu1 = nn.ReLU()
        self.layer2 = nn.Conv1d(conv1_filters, conv1_filters, kernel_size=k, stride=1, padding=0)
        self.relu2 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.layer3 = nn.Conv1d(conv1_filters, conv2_filters, kernel_size=k, stride=1, padding=0)
        self.relu3 = nn.ReLU()
        self.layer4 = nn.Conv1d(conv2_filters, conv2_filters, kernel_size=k, stride=1, padding=0)
        self.relu4 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.flatten = nn.Flatten()

        # Entry 2
        self.layer1_2 = nn.Conv1d(1, conv1_filters, kernel_size=k, stride=1, padding=0)
        self.relu1_2 = nn.ReLU()
        self.layer2_2 = nn.Conv1d(conv1_filters, conv1_filters, kernel_size=k, stride=1, padding=0)
        self.relu2_2 = nn.ReLU()
        self.pool1_2 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.layer3_2 = nn.Conv1d(conv1_filters, conv2_filters, kernel_size=k, stride=1, padding=0)
        self.relu3_2 = nn.ReLU()
        self.layer4_2 = nn.Conv1d(conv2_filters, conv2_filters, kernel_size=k, stride=1, padding=0)
        self.relu4_2 = nn.ReLU()
        self.pool2_2 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.flatten_2 = nn.Flatten()

        # Dense layers
        self.fc1 = nn.Linear(192, dense_nodes)
        self.relu5 = nn.ReLU()
        self.fc2 = nn.Linear(dense_nodes, 53)
        
        #normalize output (soft constrain) to 0.01 - 1000 ohm.m
        self.sigmoid6 = nn.Sigmoid()
        self.softmax6 = nn.Softmax()
        self.relu6 = nn.ReLU()
        
    def forward(self, x, y):
        x = self.pool1(self.relu2(self.layer2(self.relu1(self.layer1(x.permute(0, 2, 1))))))
        x = self.pool2(self.relu4(self.layer4(self.relu3(self.layer3(x)))))

        y = self.pool1_2(self.relu2_2(self.layer2_2(self.relu1_2(self.layer1_2(y.permute(0, 2, 1))))))
        y = self.pool2_2(self.relu4_2(self.layer4_2(self.relu3_2(self.layer3_2(y)))))

        
        x = self.flatten(x)
        y = self.flatten_2(y)
      

        out = torch.cat((x, y), dim=1)  

        out = self.relu5(self.fc1(out))
        out = self.fc2(out)
       
        
        out =  min_val + (max_val - min_val) * self.sigmoid6(out)
     
        
        return out