import xml.etree.ElementTree as ET
import torch
import torch.nn as nn 
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from random import shuffle

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
        x[2] = 100
    elif x[2] == "NTPP":
        x[2] = 101
    elif x[2] == "EC":
        x[2] = 102

#validationsize in percent rest will be used for validationset
val_size = 0.6
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

for i in trainingsdaten: 
    x = [i[0],i[1]] #figure und object als Eingabe
    input = Variable(torch.Tensor(x))
    
    out = net(input)
    
    x = [i[2]] #was am ende rauskommen soll, als die QSLinks
    target = Variable(torch.Tensor(x))
    criterion = nn.MSELoss()
    loss = criterion(out,target)
    
    #print(loss)
    
    
    net.zero_grad() #Fehler zurücksetzen
    loss.backward()
    optimizer = optim.SGD(net.parameters(), lr = 0.1) #funktion anpassen
    optimizer.step() 

