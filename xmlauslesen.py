import xml.etree.ElementTree as ET

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

sdata = open('trainingsdaten.txt', "w")
sdata.write(str(trainingsdaten))
sdata.close       
        

        
        
    
