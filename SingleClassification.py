import numpy as np
import torch
from sklearn.model_selection import train_test_split
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data


torch.manual_seed(1)

data = pd.read_excel('data.xlsx', usecols="B:M", skiprows=2)
np_data = np.array(data)

output = pd.read_excel('data.xlsx', usecols="N", skiprows=2)
np_output = np.array(output)

data_train, data_test, labels_train, labels_test = train_test_split(np_data, np_output, test_size=0.2, random_state=42)
data_train = torch.from_numpy(data_train)
data_test = torch.from_numpy(data_test)
labels_train = torch.from_numpy(labels_train)
labels_test = torch.from_numpy(labels_test)

#print("Numpy: ", np_data)
#print("Torch: ", data_train)

class LogisticRegression(torch.nn.Module):
     def __init__(self, input_dim, output_dim):
         super(LogisticRegression, self).__init__()
         self.linear = torch.nn.Linear(input_dim, output_dim)     
         
     def test_step(self, batch, batch_idx):
        # this is the test loop
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        test_loss = F.mse_loss(x_hat, x)
        self.log("test_loss", test_loss)

epochs = 30
# epochs = 200000
input_dim = 12 # Twelve Inputs
output_dim = 4 # Four Output Classes
learning_rate = 0.01

model = LogisticRegression(input_dim,output_dim)
criterion = torch.nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)







