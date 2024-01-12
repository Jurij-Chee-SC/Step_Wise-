# -*- coding: mbcs -*-
#
# Abaqus/Viewer Release 6.14-1 replay file
# Internal Version: 2014_06_05-06.11.02 134264
# Run by Administrator on Thu Jun 29 15:55:28 2023
#

# from driverUtils import executeOnCaeGraphicsStartup
# executeOnCaeGraphicsStartup()
#: Executing "onCaeGraphicsStartup()" in the site directory ...
from abaqus import *
from abaqusConstants import *
from viewerModules import *
from driverUtils import executeOnCaeStartup
import os


def get_odb_filename():
    # Get the directory path where the Python file is located
    file_list = os.listdir(os.getcwd())

    # Find the file ending with '.odb'
    for filename in file_list:
        if filename.endswith('.odb'):
            return filename

    # If no file ending with '.odb' is found
    return None
executeOnCaeStartup()
o2 = session.openOdb(name=get_odb_filename())
session.viewports['Viewport: 1'].setValues(displayedObject=o2)
leaf = dgo.LeafFromNodeSets(nodeSets=(' ALL NODES', ))
session.viewports['Viewport: 1'].odbDisplay.displayGroup.replace(leaf=leaf)
odb = session.odbs[get_odb_filename()]
session.fieldReportOptions.setValues(printTotal=OFF, printMinMax=OFF)
# Get the last step
last_step_index = len(odb.steps) - 1
# Get the last step name
last_step_name = list(odb.steps.keys())[-1]
# Get the number of frames in the last step
last_frame_index = len(odb.steps[last_step_name].frames) - 1

session.writeFieldReport(fileName='moment.csv', append=OFF, 
    sortItem='Node Label', odb=odb, step=last_step_index, frame=last_frame_index ,
	outputPosition=NODAL, 
    variable=(('COORD', NODAL, 
    ((COMPONENT, 'COOR2'), )), ('U', NODAL, ((COMPONENT, 'U1'), )), ('SM', 
    INTEGRATION_POINT), ('SF', INTEGRATION_POINT)))
session.viewports['Viewport: 1'].odbDisplay.displayGroup.replace(leaf=leaf)
leaf = dgo.LeafFromElementSets(elementSets=('BEAM-1.BEAM_ELE', ))
session.viewports['Viewport: 1'].odbDisplay.displayGroup.replace(leaf=leaf)
session.writeFieldReport(fileName='stress.csv', append=OFF, 
    sortItem='Element Label', odb=odb, step=last_step_index, frame=last_frame_index , 
    outputPosition=INTEGRATION_POINT, variable=(('S', INTEGRATION_POINT, ((
    COMPONENT, 'S11'), )), ))

with open('stress.csv', 'r') as inp_file :
    out_stres = inp_file.readlines()
    out_stres = out_stres[21:-3]
    out_stress_word = ''
    for k in range(len(out_stres)):
        line_k =out_stres[k].strip().split()
        out_stress_word =out_stress_word+r'beam-1.'+line_k[0]+','+line_k[-1]+'\n'


with open('stress.csv', 'w') as out_stress_file :
    out_stress_file.write(out_stress_word)

line_index=-1
with open('moment.csv','r') as g:
    out = g.readlines()
for index in range(len(out)):
    line = out[index]
    if line.startswith(r'Field Output reported at nodes for region: BEAM-1.Region_2'):
        line_index = index
        break
out= [out[19]]+out[22:line_index-2]
print line_index
print out
with open('moment.csv','w') as f:
    f.writelines(out)