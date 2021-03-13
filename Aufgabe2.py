
import xml.etree.ElementTree as ET
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from random import shuffle
import numpy as np
from sklearn.metrics import hamming_loss, f1_score, accuracy_score
from sklearn.neighbors import DistanceMetric

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

############################ preparing training data #########################################

possibleoutputs = []

#possibleoutputs speichert alle möglichen object und QSLink Kombinationen, die für eine figure Eingabe vorhanden sind

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

#list with ids for possibleinputs
num_possibleinputs = []
for i in range(38):
    num_possibleinputs.append([i])

final_trainingsdaten = []    #final data
for i in range(38):
    final_trainingsdaten.append([num_possibleinputs[i], y_train_data[i]])

################## splitting training and validation data ###############################

# validationsize in percent rest will be used for validationset
val_size = 0.6

# shuffles data
shuffle(final_trainingsdaten)

# create validationset
validierungsdaten = final_trainingsdaten[:int(len(final_trainingsdaten) * val_size)]
print(len(validierungsdaten))

# create trainingset
trainingsdaten = final_trainingsdaten[int(len(final_trainingsdaten) * val_size):]
print(len(trainingsdaten))

X_train = []
X_val = []
y_train = []
y_val = []
for i in final_trainingsdaten:
    x = [i[0]]  # objekt als Eingabe
    X_train.append(x)
    y = i[1]  # was am ende rauskommen soll
    y_train.append(y)
X_train = np.asarray(X_train)
X_train = X_train.reshape(-1, X_train.shape[1]).astype('float32')

y_train = np.asarray(y_train)

for i in validierungsdaten:
    x = [i[0]]  # objekt als eingabe
    X_val.append(x)
    y = i[1] # was am ende rauskommen soll
    y_val.append(y)
X_val = np.asarray(X_val)
X_val = X_val.reshape(-1, X_val.shape[1]).astype('float32')

y_val = np.asarray(y_val)

######################## Neural Net ###########################################

class Net(nn.Module):
    def __init__(self, nlabel):
        super(Net, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, nlabel),
        )

    def forward(self, input):
        return self.main(input)

nlabel = len(y_train_data[0]) # => 78 ( len(possibleoutputs))
classifier = Net(nlabel)

################## training #########################################

optimizer = optim.Adam(classifier.parameters())
criterion = nn.MultiLabelSoftMarginLoss()    # because multi label classification

epochs = 5
for epoch in range(epochs):
    losses = []
    hammingloss = []
    for i, sample in enumerate(X_train):
        inputv = torch.from_numpy(sample)    # array to tensor
        inputv = Variable(inputv).view(1, -1)    # for MultiLabelSoftMarginLoss()

        labelsv = torch.from_numpy(y_train[i]).long()    # array to tensor
        labelsv = Variable(labelsv).view(1, -1)    # for MultiLabelSoftMarginLoss()


        optimizer.zero_grad()

        output = classifier(inputv)
        loss = criterion(output, labelsv)

        loss.backward()
        optimizer.step()
        losses.append(loss.data.mean())
        
        # training accuracy, threshold 0.5
        output_train_acc = torch.sigmoid(output)
        output_train_acc[output_train_acc >= 0.5] = 1
        output_train_acc[output_train_acc < 0.5] = 0
        output_train_acc_int = [int(i) for i in output_train_acc[0].tolist()]    # float to int


        #hamming loss
        train_hamming = hamming_loss(labelsv.tolist(), output_train_acc.tolist())
        hammingloss.append(train_hamming)

        #hamming distance
        hamming_dist = DistanceMetric.get_metric('hamming')
        dist = hamming_dist.pairwise(labelsv.tolist(), output_train_acc.tolist())

        #hamming score
        acc_score = accuracy_score(labelsv.tolist()[0], output_train_acc_int, normalize=False)

        #f1 score
        score = f1_score(labelsv.tolist()[0], output_train_acc_int, average="macro", zero_division=1)

    print('epoch {}, loss {}, hamming loss {}, f1 score {}'.format(epoch, np.mean(losses), np.mean(hammingloss), score))
    print("Hamming Distance:", dist)
    print("correctly classified:", acc_score)

print("finished training", "\n")

################################# validation accuracy ############################################
print("Validation Data:")

with torch.no_grad():
    for i, sample in enumerate(X_val):
        inputv = torch.from_numpy(sample)  # array to tensor
        inputv = Variable(inputv).view(1, -1)

        labelsv = torch.from_numpy(y_val[i]).long()  # array to tensor
        labelsv = Variable(labelsv).view(1, -1)  # for MultiLabelSoftMarginLoss()

        output = classifier(inputv)

        # validation accuracy
        output_val_acc = torch.sigmoid(output)
        output_val_acc[output_val_acc >= 0.5] = 1
        output_val_acc[output_val_acc < 0.5] = 0
        output_val_acc_int = [int(i) for i in output_val_acc[0].tolist()]

        # hamming score
        acc_validation_score = accuracy_score(labelsv.tolist()[0], output_val_acc_int, normalize=False)
        print("Correct:", acc_validation_score,
              "Percentage:", round((100*acc_validation_score)/len(output_val_acc_int), 4))
