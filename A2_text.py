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
doc = nlp("The old book is on the bathtub")
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

possible_outputs = [['bathtub', 'PO'], ['wall', 'NTPP'], ['cabinet', 'NTPP'], ['ceiling', 'PO'],
                    ['counter', 'EC'], ['lighting', 'PO'], ['mirror', 'PO'],['towel', 'PO'], ['floor', 'PO'],
                    ['counter', 'PO'], ['objects', 'PO'], ['wall', 'PO'], ['shower', 'NTPP'], ['cabinet', 'PO'],
                    ['shower', 'PO'], ['window', 'PO'], ['door', 'PO'], ['picture', 'PO'], ['sink', 'PO'],
                    ['counter', 'NTPP'], ['cabinet', 'EC'], ['wall', 'EC'], ['toilet', 'PO'], ['ceiling', 'EC'],
                    ['stool', 'PO'], ['bed', 'PO'], ['cushion', 'PO'], ['chest_of_drawers', 'PO'], ['chair', 'PO'],
                    ['stairs', 'PO'], ['railing', 'PO'], ['shelving', 'PO'], ['curtain', 'PO'], ['door', 'NTPP'],
                    ['bed', 'NTPP'], ['curtain', 'EC'], ['cushion', 'NTPP'], ['void', 'PO'], ['picture', 'EC'],
                    ['plant', 'PO'], ['chair', 'NTPP'], ['objects', 'EC'], ['tv_monitor', 'PO'], ['table', 'PO'],
                    ['void', 'NTPP'], ['tv_monitor', 'EC'], ['shelving', 'NTPP'], ['appliances', 'PO'],
                    ['ceiling', 'NTPP'], ['floor', 'NTPP'], ['picture', 'NTPP'], ['toilet', 'EC'], ['bathtub', 'EC'],
                    ['chest_of_drawers', 'NTPP'], ['window', 'NTPP'], ['floor', 'EC'], ['mirror', 'EC'], ['misc', 'PO'],
                    ['objects', 'NTPP'], ['table', 'NTPP'], ['misc', 'NTPP'], ['window', 'EC'], ['chair', 'EC'],
                    ['appliances', 'EC'], ['sofa', 'PO'], ['sofa', 'NTPP'], ['curtain', 'NTPP'], ['column', 'PO'],
                    ['beam', 'PO'], ['fireplace', 'PO'], ['fireplace', 'NTPP'], ['railing', 'NTPP'], ['bathtub', 'NTPP'],
                    ['lighting', 'EC'], ['seating', 'PO'], ['stairs', 'NTPP'], ['clothes', 'PO'], ['furniture', 'PO']]

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 78),
        )

    def forward(self, input):
        return self.main(input)

classifier = Net()

inputs = [dict_objects.__getitem__(words[0])]
inputs = np.asarray(inputs)
inputs = inputs.reshape(-1, inputs.shape[0]).astype('float32')
inputs = torch.from_numpy(inputs)
inputs = Variable(inputs).view(1, -1)



# Load trained NeuralNet
classifier.load_state_dict(torch.load('classifierNet.pt'))
classifier.eval()

output = classifier(inputs)

output_calc = torch.sigmoid(output)
output_calc[output_calc >= 0.5] = 1
output_calc[output_calc < 0.5] = 0
output_calc_int = [int(i) for i in output_calc[0].tolist()]    # float to int
print(output_calc_int)

counter = 0
evals = []
for i in output_calc_int:
    if i == 1:
        evals.append(possible_outputs[counter])
    counter += 1
print(evals)

dictionary = ET.Element("ROOT")
for x in evals:
    eintrag = ET.SubElement(dictionary, "QSLINK", {"relType":x[1],"ground":x[0],"figure":words[0]})


ET.tostring(dictionary)

et = ET.ElementTree(dictionary)
et.write("file1.xml")


