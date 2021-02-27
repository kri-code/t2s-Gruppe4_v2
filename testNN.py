import xml.etree.ElementTree as ET
import torch
import torch.nn as nn 
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from random import shuffle
import numpy as np

########################### xml Datei auslesen #########################

matterporttree = ET.parse('matterport3d.xml')
matterportroot = matterporttree.getroot()
trainingsdaten = []
for child in matterportroot:
    if child.tag == "QSLINK" and child.get('figure') != "unknown" and child.get('ground') != "unknown":
        if child.get('relType') == "NTTPc":
             trainingsdaten.append([child.get('ground'), int(child.get('fromId')), 'NTPP'])
        elif child.get('relType') == "TPPc":
            trainingsdaten.append([child.get('ground'), int(child.get('fromId')), 'NPP'])
        else:
            trainingsdaten.append([int(child.get('fromId')), child.get('ground'), child.get('relType')])


for x in trainingsdaten:
    for child in matterportroot:
        if child.tag == "SPATIAL_ENTITY" and x[0] == child.get('label'):
            x[0] = int(child.get('objectId'))
        elif child.tag == "SPATIAL_ENTITY" and x[1] == child.get('label'):
            x[1] = int(child.get('objectId'))
    
    if x[2] == "PO":
        x[2] = 101
    elif x[2] == "NTPP":
        x[2] = 102
    elif x[2] == "EC":
        x[2] = 103

#validationsize in percent rest will be used for validationset
val_size = 0.5
print(len(trainingsdaten))

#shuffles data
shuffle(trainingsdaten)

#create validationset
validierungsdaten = trainingsdaten[:int(len(trainingsdaten)*val_size)]
print(len(validierungsdaten))

#create trainingset
trainingsdaten = trainingsdaten[int(len(trainingsdaten)*val_size):]
print(len(trainingsdaten))



#sdata = open('trainingsdaten.txt', "w")
#sdata.write(str(trainingsdaten))
#sdata.close  


############ umwandeln der trainings- und validierungsdaten #######################
X_train = []
X_val =[]
y_train = []
y_val = []
for i in trainingsdaten:
    x = [i[0], i[1]]  # figure und object als Eingabe
    X_train.append(x)
    y = [i[2]]  # was am ende rauskommen soll, als die QSLinks
    y_train.append(y)
X_train = np.asarray(X_train)
X_train = torch.tensor(X_train, dtype=torch.float32)

y_train = np.asarray(y_train)
y_train = torch.tensor(y_train, dtype=torch.float32)
print(X_train.shape, y_train.shape)

for i in validierungsdaten:
    x = [i[0], i[1]]  # figure und object als Eingabe
    X_val.append(x)
    y = [i[2]]  # was am ende rauskommen soll, als die QSLinks
    y_val.append(y)
X_val = np.asarray(X_val)
X_val = torch.tensor(X_val, dtype=torch.float32)

y_val = np.asarray(y_val)
y_val = torch.tensor(y_val, dtype=torch.float32)
print(X_val.shape, y_val.shape)
        
############################# Neuronales Netz ###############################
        
class NeuralNet(nn.Module):
    
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        #Schichten
        self.lin1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.lin2 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        out = self.lin1(x)
        out = self.relu(out)
        out = self.lin2(out)

        return out
    
    def num_flat_features(self,x):
        size = x.size()[1:]
        num = 1
        for i in size:
            num *= i
        return num
    
net = NeuralNet(2, 2, 1)

optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

net.train()
for epoch in range(8):  # loop over the dataset multiple times

    running_loss = 0.0
    for i in trainingsdaten:
        x = [i[0], i[1]]  # figure und object als Eingabe
        data = x
        input = Variable(torch.Tensor(x))

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        out = net(input)
        x = [i[2]]  # was am ende rauskommen soll, als die QSLinks
        target = Variable(torch.Tensor(x))

        criterion = nn.MSELoss()
        loss = criterion(out, target)
        print(loss)

        loss.backward()
        #net.zero_grad() #Fehler zurücksetzen, bruacht ihr das noch? war noch vom alten merge oder habt ihr das bewusst geloescht?
        optimizer.step()
        
        # print statistics
        running_loss += loss.item()
        print(epoch + 1, data, running_loss / len(trainingsdaten))
        running_loss = 0.0

print('Finished Training')


#calculate accuracy
correct = 0
total = 0

net.eval()
with torch.no_grad():
    for data in validierungsdaten:
        figure = data[0]
        ground = data[1]
        label = data[2]
        input = Variable(torch.Tensor([figure,ground]))
        #print(input)
        output = net(input)
        #print(label)
        pred = Variable(torch.Tensor([label]))
       
        #print(output)
        total += 1
      
        if (output - pred) <= 0.5 and (output - pred) >= -0.5:
            correct += 1

print('Accuracy of the network  %d %%' % (100 * correct / total))


print(net(Variable(torch.Tensor([1,2]))))
