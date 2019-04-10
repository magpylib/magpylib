from magpylib import Collection, source
from numpy import arange
from magpylib import math

def returnCoilTurn(diameter,position): # Return turns for coils
    newTurn = source.current.Circular(curr=10,
                                      dim=diameter,
                                      pos=position)
    return newTurn

sizeOfCoil = 3 #mm
turnDistance = 0.1 #mm
diameter = 2 #mm

startMarker = [0,0,0]
endMarker = [0,0,sizeOfCoil] 

## Generate a coil structure
coilStructure = [returnCoilTurn(diameter,[0,0,i]) for i in arange(0,sizeOfCoil, turnDistance)]

## Group up the structure
col = Collection(coilStructure)
col.displaySystem(markers = [startMarker,
                             endMarker])

## Rotate the Structure
col.rotate(45,(0,1,1), anchor = [0,0,0])            ## Rotate all objects in collection relative to [0,0,0]
endMarker=math.rotatePosition(endMarker,45,(0,1,1)) ## Rotate Marker position
col.displaySystem(markers = [startMarker,
                             endMarker])