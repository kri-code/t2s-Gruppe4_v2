import xml.etree.ElementTree as ET
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from random import shuffle
import numpy as np

trainingsdaten = []

'''matterporttree = ET.parse('matterport3d.xml')
matterportroot = matterporttree.getroot()

for child in matterportroot:
    if child.tag == "QSLINK" and child.get('figure') != "unknown" and child.get('ground') != "unknown":
        if child.get('relType') == "NTTPc":
            trainingsdaten.append([int(child.get('toId')), int(child.get('fromId')), 'NTPP'])
        elif child.get('relType') == "TPPc":
            trainingsdaten.append([int(child.get('toId')), int(child.get('fromId')), 'NPP'])
        else:
            trainingsdaten.append([int(child.get('fromId')), int(child.get('toId')), child.get('relType')])'''

matterporttree = ET.parse('matterport3dhouse.xml')
matterportroot = matterporttree.getroot()

for child in matterportroot:
    if child.tag == "QSLINK" and child.get('figure') != "unknown" and child.get('ground') != "unknown":
        if child.get('relType') == "NTTPc":
            trainingsdaten.append([int(child.get('toId')), int(child.get('fromId')), 'NTPP'])
        elif child.get('relType') == "TPPc":
            trainingsdaten.append([int(child.get('toId')), int(child.get('fromId')), 'NPP'])
        else:
            trainingsdaten.append([int(child.get('fromId')), int(child.get('toId')), child.get('relType')])

for x in trainingsdaten:
    for child in matterportroot:
        if child.tag == "SPATIAL_ENTITY" and x[0] == child.get('label'):
            x[0] = int(child.get('objectId'))
        elif child.tag == "SPATIAL_ENTITY" and x[1] == child.get('label'):
            x[1] = int(child.get('objectId'))

    if x[2] == "PO":
        x[2] = 0            # 100 -> 0
    elif x[2] == "NTPP":
        x[2] = 1            # 101 -> 1
    elif x[2] == "EC":
        x[2] = 2            # 102 -> 2

# validationsize in percent rest will be used for validationset
val_size = 0.6
print(len(trainingsdaten))

# shuffles data
shuffle(trainingsdaten)

# create validationset
validierungsdaten = trainingsdaten[:int(len(trainingsdaten) * val_size)]
print(len(validierungsdaten))

# create trainingset
trainingsdaten = trainingsdaten[int(len(trainingsdaten) * val_size):]
print(len(trainingsdaten))


# sdata = open('trainingsdaten.txt', "w")
# sdata.write(str(trainingsdaten))
# sdata.close


############ umwandeln der trainings- und validierungsdaten #######################
X_train = []
X_val = []
y_train = []
y_val = []
for i in trainingsdaten:
    x = [i[0], i[1]]  # figure und object als Eingabe
    X_train.append(x)
    y = i[2]  # was am ende rauskommen soll, als die QSLinks
    y_train.append(y)
X_train = np.asarray(X_train)
X_train = (X_train / 100) #normalisieren
X_train = X_train.reshape(-1, X_train.shape[1]).astype('float32')
#X_train = torch.tensor(X_train)

y_train = np.asarray(y_train)
#y_train = torch.tensor(y_train).long()
print(X_train.shape, y_train.shape)

for i in validierungsdaten:
    x = [i[0], i[1]]  # figure und object als Eingabe
    X_val.append(x)
    y = i[2] # was am ende rauskommen soll, als die QSLinks
    y_val.append(y)
X_val = np.asarray(X_val)
X_val = (X_val / 100) #normalisieren
X_val = X_val.reshape(-1, X_val.shape[1]).astype('float32')
X_val = torch.tensor(X_val)

y_val = np.asarray(y_val)
y_val = torch.tensor(y_val).long()
print(X_val.shape, y_val.shape)

class Data(Dataset):
    def __init__(self):
        self.x = torch.from_numpy(X_train)
        self.y = (torch.from_numpy(y_train)).long()    # data type long for y_train
        self.len = self.x.shape[0]
    def __getitem__(self,index):
        return self.x[index], self.y[index]
    def __len__(self):
        return self.len

data_set = Data()
trainloader = DataLoader(dataset=data_set, batch_size=len(X_train))
#print(data_set.x[1:10])
#print(data_set.y[1:10])
#print(data_set.x.shape, data_set.y.shape)

#print(z[0:10])
############################# Neuronales Netz ###############################
class NeuralNet(nn.Module):

    def __init__(self):
        super(NeuralNet, self).__init__()
        # Schichten
        self.lin1 = nn.Linear(2, 6) #2-(6-6)-3
        self.lin2 = nn.Linear(6, 6)
        #output layer: [Klasse PO, Klasse NTPP, Klasse EC]
        self.oupt = nn.Linear(6, 3)

        nn.init.xavier_uniform_(self.lin1.weight)
        nn.init.zeros_(self.lin1.bias)
        nn.init.xavier_uniform_(self.lin2.weight)
        nn.init.zeros_(self.lin2.bias)
        nn.init.xavier_uniform_(self.oupt.weight)
        nn.init.zeros_(self.oupt.bias)

    def forward(self, x):
        z = torch.tanh(self.lin1(x))
        z = torch.tanh(self.lin2(z))
        z = self.oupt(z)  # no softmax: CrossEntropyLoss()

        return z

    def num_flat_features(self, x):
        size = x.size()[1:]
        num = 1
        for i in size:
            num *= i
        return num

model = NeuralNet()

criterion = nn.CrossEntropyLoss()
learning_rate = 0.01
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

loss_list = []
# epochs
for epoch in range(1000):
    for x, y in trainloader:
        # clear gradient
        optimizer.zero_grad()
        # make a prediction
        z = model(x)
        # calculate loss, da Cross Entropy benutzt wird Wahrscheinlichkeit pro Klasse
        # vorhergesagt. Das macht torch.max(y,1)[1])
        loss = criterion(z, y)
        # calculate gradients of parameters
        loss.backward()
        # update parameters
        optimizer.step()

        loss_list.append(loss.data)

        #print('epoch {}, loss {}'.format(epoch, loss.item()))
print(y[0:10])
print(z[0:10])


def accuracy(y_hat, y):
    pred = torch.argmax(y_hat, dim=1)
    return (pred == y).float().mean()

Y_pred_train = model(torch.tensor(X_train))
Y_pred_val = model(X_val)

accuracy_train = accuracy(Y_pred_train, torch.tensor(y_train).long())
accuracy_val = accuracy(Y_pred_val, y_val)

print("Training accuracy", accuracy_train)
print("Validation accuracy", accuracy_val)
