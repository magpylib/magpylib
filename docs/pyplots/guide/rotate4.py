from magpylib import source, Collection
from numpy import arange

def returnCoilTurn(diameter,position): # Return turns for coils
    newTurn = source.current.Circular(  curr=10,
                                        dim=diameter,
                                        pos=position)
    return newTurn

# Define Coil
sizeOfCoil = 3 #mm
turnDistance = 0.1 #mm
diameter = 5 #mm

startMarker = [0,0,0]
endMarker = [0,0,sizeOfCoil] 

## Generate a coil structure
coilStructure = [returnCoilTurn(diameter,[0,0,i]) for i in arange(0,sizeOfCoil, turnDistance)]

## Define Coil
coil = Collection(coilStructure)

## Generate Source Objects
box = source.magnet.Box( mag = [1,2,3],   
                         dim = [4,5,6],  
                         pos = [7,8,9])

sphere = source.magnet.Sphere( mag = [1,2,3],   
                               dim = 5,  
                               pos = [-7,-8,-9],)

superCollection = Collection(box,sphere,coil)  ## Make a mixed Collection
superCollection.displaySystem()

coil.rotate(45,(0,1,1),anchor=[0,0,0])
superCollection.displaySystem()
