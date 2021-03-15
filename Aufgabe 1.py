import xml.etree.ElementTree as ET
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from random import shuffle
import numpy as np
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import time
import datetime
import os
import sys

#save datestamp for unique outputfile each time code gets executed
ts = time.time()
datestamp = datetime.datetime.fromtimestamp(ts).strftime('%d-%m-%Y-%H-%M-%S')

#write console output into html file
sys.stdout = open('Aufgabe1_performance/{}.txt'.format(datestamp), 'w')
#print('test2')

#prints starting time
print(datestamp)

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

# shuffles data
shuffle(trainingsdaten)

#balancierte trainingsdaten
po_data = []
ntpp_data = []
ec_data = []
counter_po = 0
counter_ntpp = 0
counter_ec = 0

for tarray in trainingsdaten:
    if tarray[2] == 0 and counter_po < 999:
        po_data.append(tarray)
        counter_po += 1
    elif tarray[2] == 1 and counter_ntpp < 999:
        ntpp_data.append(tarray)
        counter_ntpp += 1
    elif tarray[2] == 2 and counter_ec < 999:
        ec_data.append(tarray)
        counter_ec += 1

#artificially enlarge ec data since only 37 available
ec_data = 27*ec_data #37*27 = 999

#define new balanced trainingdata 
trainingsdaten = po_data + ntpp_data + ec_data


#print(len(trainingsdaten))
#print(ec_data)
#print(len(po_data), len(ntpp_data), len(ec_data))
        

# validationsize in percent rest will be used for trainingset
val_size = 0.6
print(val_size, "is relative validationsize rest will be used for trainingset")
print(len(trainingsdaten), "Total datasize")

# shuffles data
shuffle(trainingsdaten)

# create validationset
validierungsdaten = trainingsdaten[:int(len(trainingsdaten) * val_size)]
print(len(validierungsdaten), "validationsize")

# create trainingset
trainingsdaten = trainingsdaten[int(len(trainingsdaten) * val_size):]
print(len(trainingsdaten), "trainingsize")


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
#print(X_train.shape, y_train.shape)

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
#print(X_val.shape, y_val.shape)

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
        #self.lin1 = nn.ReLU() #2-(6-6)-3
        #self.lin2 = nn.Linear(6, 6)
        self.lin2 = nn.ReLU()
        #output layer: [Klasse PO, Klasse NTPP, Klasse EC]
        
        self.oupt = nn.Linear(6, 3)
        #self.oupt = nn.ReLU()

        nn.init.xavier_uniform_(self.lin1.weight)
        nn.init.zeros_(self.lin1.bias)
        
        #nn.init.xavier_uniform_(self.lin2.weight)
        #nn.init.zeros_(self.lin2.bias)
        
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
print(model)
criterion = nn.CrossEntropyLoss()
print(criterion)
learning_rate = 0.01
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
print(optimizer)

#calculate accuracy
def accuracy(y_hat, y):
    pred = torch.argmax(y_hat, dim=1)
    return (pred == y).float().mean()


accuracy_train_list = []
accuracy_val_list = []
loss_list = []
# epochs
for epoch in range(1500):
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
        
        Y_pred_train = model(torch.tensor(X_train))
        Y_pred_val = model(X_val)
        
        accuracy_train = accuracy(Y_pred_train, torch.tensor(y_train).long())
        accuracy_val = accuracy(Y_pred_val, y_val)
        
        accuracy_train_list.append(accuracy_train)
        accuracy_val_list.append(accuracy_val)
        

        #print('epoch {}, loss {}, Training accuracy {}, Validation accuracy {}'.format(epoch, loss.item(), accuracy_train, accuracy_val))
#print(y[0:10])
#print(z[0:10])

#print number of epochs for outputfile
print("Epochs:",epoch+1)


#transform data for classification report function
Y_pred_val = Y_pred_val.detach().numpy()    # tensor to array
prediction = []
for i in Y_pred_val:    # evaluate prediction
    if i[0] > i[1] and i[0] > i[2]:
        prediction.append(0)
    elif i[1] > i[0] and i[1] > i[2]:
        prediction.append(1)
    else:
        prediction.append(2)
#print(prediction[0:10])

y_true = y_val.tolist()    # tensor to python list
#print(y_true[0:10])

#classification report
#calculates f1 score for each class
target_names = ['PO', 'NTPP', 'EC']
print(classification_report(y_true, prediction, target_names=target_names, zero_division=0))

#save trained model
torch.save(model, "labelNet.pt")

#create plot
plt.title('train- und  validation-accuracy')
plt.plot(list(range(0,len(accuracy_train_list))),accuracy_train_list,label='Train accuracy')
plt.plot(list(range(0,len(accuracy_val_list))),accuracy_val_list,label='Validation accuracy')
plt.ylabel('Relative Rate')
plt.xlabel('Epochen')
plt.legend()
plt.savefig(os.path.join('Aufgabe1_performance','{}-Accuracyvsvalidationacc-{}-{}-{}-{}.png'.format(datestamp, (str(optimizer)).split()[0],val_size,criterion,epoch+1,)))
plt.show()

#prints elapsed time
print("elapsed time (min):", (time.time() - ts)/60)

#sys.stdout.close()
