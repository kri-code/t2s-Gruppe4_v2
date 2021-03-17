
import xml.etree.ElementTree as ET
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from random import shuffle
import numpy as np
from sklearn.metrics import hamming_loss, f1_score, accuracy_score
from sklearn.neighbors import DistanceMetric
import matplotlib.pyplot as plt
import time
import datetime
import os
import sys

#save datestamp for unique outputfile each time code gets executed
ts = time.time()
datestamp = datetime.datetime.fromtimestamp(ts).strftime('%d-%m-%Y-%H-%M-%S')

#write console output into txt file
sys.stdout = open('Aufgabe2_performance/{}.txt'.format(datestamp), 'w')
#print('test2')

#prints starting time
print(datestamp)

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

#print(trainingsdaten)

############################ preparing training data #########################################

possibleoutputs = []

#possibleoutputs speichert alle möglichen object und QSLink Kombinationen, die für eine figure Eingabe vorhanden sind

for x in trainingsdaten:
    if [x[1],x[2]] in possibleoutputs:
        pass
    else:
        possibleoutputs.append([x[1],x[2]])

#print(possibleoutputs)

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

#print(trainingsdaten)
 
#print("Possible output legth:",len(possibleoutputs))
#print(possibleoutputs)
#print(len(possibleinputs))
#print(possibleinputs)

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
#print(len(y_train_data[0]))
#print(len(y_train_data))

#list with ids for possibleinputs
num_possibleinputs = []
for i in range(38):
    num_possibleinputs.append([i])

final_trainingsdaten = []    #final data
for j in range (10):
    for i in range(38):
        final_trainingsdaten.append([num_possibleinputs[i], y_train_data[i]])



################## splitting training and validation data ###############################

# validationsize in percent rest will be used for validationset
val_size = 0.5
print(val_size, "is relative validationsize rest will be used for trainingset")
print(len(trainingsdaten), "Total datasize")
print(len(final_trainingsdaten), "Final datasize")

# shuffles data
shuffle(final_trainingsdaten)

# create validationset
validierungsdaten = final_trainingsdaten[:int(len(final_trainingsdaten) * val_size)]
print(len(validierungsdaten), "validationsize")

# create trainingset
trainingsdaten = final_trainingsdaten[int(len(final_trainingsdaten) * val_size):]
print(len(trainingsdaten), "trainingsize")

X_train = []
X_val = []
y_train = []
y_val = []
for i in trainingsdaten:
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
print(classifier)

################## training #########################################

optimizer = optim.Adam(classifier.parameters())
print(optimizer)
criterion = nn.MultiLabelSoftMarginLoss()    # because multi label classification
print(criterion)

epochs = 100
print("epochs:",epochs)

hammingloss_mean = []
hammingloss_val_mean = []
score_mean = []
for epoch in range(epochs):
    losses = []
    hammingloss = []
    hammingloss_val = []
    score_list = []
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
        score_list.append(score)

    print('---epoch {}, loss {}, hamming loss {}, f1 score {}'.format(epoch, np.mean(losses), np.mean(hammingloss), np.mean(score_list)))
    print("Hamming Distance:", dist)
    print("correctly classified(hamming score):", acc_score)
    hammingloss_mean.append(np.mean(hammingloss))
    score_mean.append(np.mean(score_list))
    
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
            
            #hamming loss
            train_hamming = hamming_loss(labelsv.tolist(), output_val_acc.tolist())
            hammingloss_val.append(train_hamming)
        print('hamming loss val',np.mean(hammingloss_val))
        hammingloss_val_mean.append(np.mean(hammingloss_val))



print("finished training", "\n")

#save trained network
torch.save(classifier, "classifierNet.pt")

################################# validation accuracy ############################################
# print("Validation Data:")

# with torch.no_grad():
#     for i, sample in enumerate(X_val):
#         inputv = torch.from_numpy(sample)  # array to tensor
#         inputv = Variable(inputv).view(1, -1)

#         labelsv = torch.from_numpy(y_val[i]).long()  # array to tensor
#         labelsv = Variable(labelsv).view(1, -1)  # for MultiLabelSoftMarginLoss()

#         output = classifier(inputv)

#         # validation accuracy
#         output_val_acc = torch.sigmoid(output)
#         output_val_acc[output_val_acc >= 0.5] = 1
#         output_val_acc[output_val_acc < 0.5] = 0
#         output_val_acc_int = [int(i) for i in output_val_acc[0].tolist()]

#         # hamming score
#         acc_validation_score = accuracy_score(labelsv.tolist()[0], output_val_acc_int, normalize=False)
#         print("Correct:", acc_validation_score,
#               "Percentage:", round((100*acc_validation_score)/len(output_val_acc_int), 4),
#               "ObjectID:", int(sample[0]),
#               "Object:", possibleinputs[int(sample[0])])

################################# test accuracy per Object ############################################
print("accuracy per Object:")

with torch.no_grad():
    for i, sample in enumerate(np.array(num_possibleinputs, dtype="float32")):
        inputv = torch.from_numpy(sample)  # array to tensor
        inputv = Variable(inputv).view(1, -1)

        labelsv = torch.from_numpy(np.array(y_train_data[i], dtype="float32")).long()  # array to tensor
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
              "Percentage:", round((100*acc_validation_score)/len(output_val_acc_int), 4),
              "ObjectID:", int(sample[0]),
              "Object:", possibleinputs[int(sample[0])])



#create loss plot
plt.title('train- und  validation-loss')
plt.plot(list(range(0,len(hammingloss_mean))),hammingloss_mean,label='Train loss')
plt.plot(list(range(0,len(hammingloss_val_mean))),hammingloss_val_mean,label='Validation loss')
plt.ylabel('Hamming loss')
plt.xlabel('Epochen')
plt.legend()
plt.savefig(os.path.join('Aufgabe2_performance','{}-lossvsvalidationloss-{}-{}-{}-{}.png'.format(datestamp, (str(optimizer)).split()[0],val_size,criterion,epoch+1,)))
plt.show()

#create f1 plot
#f, ax = plt.subplots()
plt.title('F1-Score MLC')
plt.plot(list(range(0,len(score_mean))),score_mean)
plt.ylabel('F1-Score')
plt.xlabel('Epochen')
plt.legend()
plt.legend('',frameon=False)
plt.savefig(os.path.join('Aufgabe2_performance','{}-f1-score-{}-{}-{}-{}.png'.format(datestamp, (str(optimizer)).split()[0],val_size,criterion,epoch+1,)))
plt.show()



#prints elapsed time
print("elapsed time (min):", (time.time() - ts)/60)

#sys.stdout.close()
