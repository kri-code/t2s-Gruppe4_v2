import xml.etree.ElementTree as ET

matterporttree = ET.parse('matterport3d.xml')
matterportroot = matterporttree.getroot()
trainingsdaten = []
for child in matterportroot:
    if child.tag == "QSLINK" and child.get('figure') != "unknown" and child.get('ground') != "unknown":
        if child.get('relType') == "NTTPc":
             trainingsdaten.append([child.get('ground'), child.get('figure'), 'NTTP'])
        elif child.get('relType') == "TPPc":
            trainingsdaten.append([child.get('ground'), child.get('figure'), 'NPP'])
        else:
            trainingsdaten.append([child.get('figure'), child.get('ground'), child.get('relType')])


sdata = open('trainingsdaten.txt', "w")
sdata.write(str(trainingsdaten))
sdata.close       
        

        
        
    