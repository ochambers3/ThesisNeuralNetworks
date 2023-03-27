import numpy as np
import torch
from sklearn.model_selection import train_test_split
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data

torch.manual_seed(1)

data1 = pd.read_excel('MultiClassData.xlsx', usecols="B:M", skiprows=2)
np_data = np.array(data1)

output = pd.read_excel('MultiClassData.xlsx', usecols="N:Q", skiprows=2)
np_output = np.array(output)

data_train, data_test, labels_train, labels_test = train_test_split(np_data, np_output, test_size=0.2, random_state=42)
data_train = data_train.astype(np.float32)
data_test = data_test.astype(np.float32)
labels_train = labels_train.astype(np.float32)
labels_test = labels_test.astype(np.float32)

data_train = torch.from_numpy(data_train)
data_test = torch.from_numpy(data_test)
labels_train = torch.from_numpy(labels_train)
labels_test = torch.from_numpy(labels_test)

train = data.TensorDataset(data_train, labels_train)
train_dataloader = data.DataLoader(train, batch_size=4, shuffle=True)

test = data.TensorDataset(data_test, labels_test)
test_dataloader = data.DataLoader(test, batch_size=4, shuffle=True)

for X, y in test_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break

# Get cpu or gpu device for training.
device = "cpu"
print(f"Using {device} device")

# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            #  nn.Linear(12, 128),
            #  nn.ReLU(),
            #  nn.Linear(128, 32),
            #  nn.ReLU(),
            #  nn.Linear(32, 1)
            nn.Linear(12, 4),
            #nn.ReLU()
            nn.Sigmoid()
        )
        #self.linear_relu_stack = nn.Sequential(
        #    nn.Linear(28*28, 10),
        #    nn.ReLU(),
        #)

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device)
print(model)

#loss_fn = nn.CrossEntropyLoss()
loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        
        # Compute prediction error
        pred = model(X)
        # print("Train:")
        # print("Predicted: ", pred)
        # print("Actual: ", y)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 50 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            # print("Pred: ", pred)
            # print("True: ", y)
            #print("Loss: ", loss_fn(pred, y))

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct1, correct2, correct3, correct4, correct, tracker = 0, 0, 0, 0, 0, 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            predn = pred.numpy()
            yn = y.numpy()
            # print("Test:")
            # print("Pred Numpy: \n", predn)
            # # print("Pred[0]: ", predn[0])
            # # print("Pred[0][1]: ", pred[0][1])
            # print("Actual Numpy: \n", yn)

            for i, j in zip(predn, yn):
                tracker = 0
                for k in range(4):
                    # print("i[k]: ", i[k])
                    # print("j[k]: ", j[k])
                    # print("Tracker: ", tracker+1)
                    if tracker == 0:
                        if (i[k] > 0.5 and j[k] == 1) or (i[k] < 0.5 and j[k] == 0):
                            #print("One correct")
                            correct1 += 1
                    elif tracker == 1:
                        if (i[k] > 0.5 and j[k] == 1) or (i[k] < 0.5 and j[k] == 0):
                            #print("Two correct")
                            correct2 += 1
                    elif tracker == 2:
                        if (i[k] > 0.5 and j[k] == 1) or (i[k] < 0.5 and j[k] == 0):
                            #print("Three correct")
                            correct3 += 1 
                    else:
                        if (i[k] > 0.5 and j[k] == 1) or (i[k] < 0.5 and j[k] == 0):
                            #print("Four correct")
                            correct4 += 1 
                    tracker += 1
                # predOne = i[0]
                # yOne = j[0]

                #print("predicted, actual: ", predOne, " ", yOne)
            #print("pred.argmax(1): ", pred.argmax(1))
            #print("Correct: ", correct)
            #correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct = correct1 + correct2 + correct3 + correct4
    print("Class 1 Accuracy: ", correct1/size*100)
    print("Class 2 Accuracy: ", correct2/size*100)
    print("Class 3 Accuracy: ", correct3/size*100)
    print("Class 4 Accuracy: ", correct4/size*100)
    correct /= size*4
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

epochs = 400
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
for layer in model.modules():
   if isinstance(layer, nn.Linear):
        print(layer.weight)
        print(layer.bias)
print("Done!")