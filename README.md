# t2s-Gruppe4_v2
Text2Scene Praktikum 20/21

## AUFGABE 1
<h2>Dateien:<h2> <br>
Aufgabe 1.py // matterport3dhouse.xml <br>
Ausführung: <br>
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
## AUFGABE 2
Dateien: <br>
Aufgabe2.py // matterport3dhouse.xml <br>
Ausführung: <br>
Die xml Datei muss im gleichen Ordner wie Aufgabe2.py liegen.
```shell
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
```
## AUFGABE 4
Dateien: <br>
A1_text.py // A2_text.py <br> matterport3dhouse.xml <br> labelNet.pt // classifierNet.pt <br>
Ausführung A1_text.py: <br>
Die xml Datei und labelNet.pt müssen im gleichen Ordner wie A1_text.py liegen.
```shell
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
```
Ausführung A2_text.py: <br>
Die xml Datei und classifierNet.pt müssen im gleichen Ordner wie A2_text.py liegen.
```shell
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
```
