import xml.etree.ElementTree as ET
import torch
import torch.nn as nn
import torch.nn.functional as F
from spacy.lang.en import English
from spacy.tokenizer import Tokenizer
from torch.autograd import Variable
import torch.optim as optim
import numpy as np

import spacy
from spacy.lang.en import English

nlp = English()
doc = nlp("The old book is on the chair next to the table.")
l = []
for token in doc:
    l.append(token.text)
    print(token.text)

print(l)

# objects we trained net with
trained_objects = ["cabinet", "objects", "shower", "bathtub", "wall",
                "window", "ceiling", "towel", "counter", "lighting", "door", "mirror", "curtain", "sink",
                "floor", "picture", "toilet", "chair", "bed", "chest_of_drawers", "cushion", "stool", "void",
                "table", "tv_monitor", "plant", "shelving", "appliances", "misc", "sofa", "fireplace", "column",
                "beam", "railing", "stairs", "seating", "clothes", "furniture"]

# words contains all objects mentioned in text an that are in trained_objects
words = []
for i in l:
    if i in trained_objects:
        words.append(i)
print(words)
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
l = []
for i in objects:
    l.append(i[0])
print(l)
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
print(model)

# get IDs for input objects
inputs = [dict_objects.__getitem__(words[0]), dict_objects.__getitem__(words[1])]
print(inputs)
inputs = np.asarray(inputs)
inputs = (inputs/100)
inputs = inputs.reshape(-1, inputs.shape[0]).astype('float32')
inputs = torch.from_numpy(inputs)
print(inputs.shape)

# Load trained NeuralNet
model.load_state_dict(torch.load('labelNet.pt'))
model.eval()
output = model(inputs)

predicted = output.detach().numpy() #tensor to array
for pred in predicted:
    if pred[0] > pred[1] and pred[0] > pred[2]:
        print("predicted QSLink: PO")
    elif pred[1] > pred[0] and pred[1] > pred[2]:
        print("predicted QSLink: NTPP")
    else:
        print("predicted QSLink: EC")

