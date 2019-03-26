import magpylib 
pointToCalculate = [-3,-2,-1] # Position Vector of the field point to calculate        

b = magpylib.source.magnet.Box( mag = [1,2,3],   
                                dim = [4,5,6],  
                                pos = [7,8,9],  
                                angle = 90,     
                                axis = (0,0,1))

col = magpylib.Collection(b) ## Make a Collection of 1 source object
print(col.getB(pointToCalculate)) ## Our GetB result


markerZero = [0,0,0]              ## A Marker On Zero Coordinates
markerPosition = pointToCalculate ## A Marker On the Field sample

col.displaySystem(markers=[ markerZero, 
                            markerPosition])