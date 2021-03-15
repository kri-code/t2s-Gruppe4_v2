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
objects = []
a = []
matterporttree = ET.parse('matterport3dhouse.xml')
matterportroot = matterporttree.getroot()

for child in matterportroot:
    if child.tag == "SPATIAL_ENTITY" and child.get('label') != "unknown":
        trainingsdaten.append([child.get('label'), child.get('objectId')])
for i in trainingsdaten:
    if i[0] not in a:
        objects.append(i)
        a.append(i[0])

# dictionary object:id
dict_objects = {}
for l2 in objects:
    dict_objects[l2[0]] = int(l2[1])
print(dict_objects)


class NeuralNet(nn.Module):

    def __init__(self):
        super(NeuralNet, self).__init__()
        # Schichten
        self.lin1 = nn.Linear(2, 6)  # 2-(6-6)-3
        # self.lin1 = nn.ReLU() #2-(6-6)-3
        # self.lin2 = nn.Linear(6, 6)
        self.lin2 = nn.ReLU()
        # output layer: [Klasse PO, Klasse NTPP, Klasse EC]

        self.oupt = nn.Linear(6, 3)
        # self.oupt = nn.ReLU()

        nn.init.xavier_uniform_(self.lin1.weight)
        nn.init.zeros_(self.lin1.bias)

        # nn.init.xavier_uniform_(self.lin2.weight)
        # nn.init.zeros_(self.lin2.bias)

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


# Load trained NeuralNet
model.load_state_dict(torch.load('labelNet.pt'))
model.eval()
