
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

matterporttree = ET.parse('matterport3dhouse.xml')
matterportroot = matterporttree.getroot()

for child in matterportroot:
    if child.tag == "QSLINK" and child.get('figure') != "unknown" and child.get('ground') != "unknown":
        if child.get('relType') == "NTTPc":
            trainingsdaten.append([child.get('ground'), child.get('figure'), 'NTPP'])
        elif child.get('relType') == "TPPc":
            trainingsdaten.append([ child.get('ground'), child.get('figure'), 'NPP'])
        else:
            trainingsdaten.append([child.get('figure'), child.get('ground'), child.get('relType')])

print(trainingsdaten)

possibleoutputs = []

#possibleoutputs sppeichert alle möglichen object und QSLink Kombinationen, die für eine figure Eingabe vorhanden sind

for x in trainingsdaten:
    if [x[1],x[2]] in possibleoutputs:
        pass
    else:
        possibleoutputs.append([x[1],x[2]])

print(possibleoutputs)

possibleinputs = []

for x in trainingsdaten:
    if x[0] in possibleinputs:
        pass
    else:
        possibleinputs.append(x[0])

#anhand der possibleoutputs werden object und QsLink durch eine ID ersetzt die der Position der Möglichkeit in possible outputs ersetzt
i = 0
while i < len(possibleoutputs): 
    j = 0
    while j < len(trainingsdaten):
        if len(trainingsdaten[j]) > 2:
            x = trainingsdaten[j]
            if [x[1],x[2]] == possibleoutputs[i]:
                trainingsdaten[j] = [x[0],i]
        j = j+1
    i = i+1
    
#anhand possibleinputs wird figure durch ID ersetzt
i = 0
while i < len(possibleinputs):
    j = 0
    for x in trainingsdaten:
        if x[0] == possibleinputs[i]:
            x[0] = i
    i = i+1

print(trainingsdaten)
 
print(len(possibleoutputs))
print(possibleoutputs)
print(len(possibleinputs))
print(possibleinputs)

y_train_data = []
counter = 0
out_counter = 0
v = []
while counter < len(possibleinputs):
    for i in range(len(possibleoutputs)):
        if [counter, out_counter] in trainingsdaten:
            v.append(1)
            out_counter += 1
        else:
            v.append(0)
            out_counter += 1
    y_train_data.append(v)
    v = []
    counter += 1
    out_counter = 0
print(len(y_train_data[0]))
print(len(y_train_data))
