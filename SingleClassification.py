import numpy as np
import torch


import pandas as pd
data = pd.read_excel('data.xlsx', usecols="B:M", skiprows=2)
np_data = np.array(data)
torch_data = torch.from_numpy(np_data)

output = pd.read_excel('data.xlsx', usecols="N", skiprows=2)
np_output = np.array(output)
torch_output = torch.from_numpy(np_output)

print("Numpy: ", np_data)
print("Torch: ", torch_data)

