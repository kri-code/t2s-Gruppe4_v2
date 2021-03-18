# t2s-Gruppe4_v2
Text2Scene Praktikum 20/21

AUFGABE 1
Dateien: 
Aufgabe 1.py // matterport3dhouse.xml
Ausführung:
Die xml Datei muss im gleichen Ordner wie Aufgabe 1.py liegen.
```shell
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
```
