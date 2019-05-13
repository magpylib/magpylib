import magpylib 
pointToCalculate = [-3,-2,-1] # Position Vector of the field point to calculate        

box = magpylib.source.magnet.Box( mag = [1,2,3],   
                                  dim = [4,5,6],  
                                  pos = [7,8,9])

sphere = magpylib.source.magnet.Sphere( mag = [1,2,3],   
                                        dim = 5,  
                                        pos = [-7,-8,-9],)

col = magpylib.Collection(box,sphere)       ## Make a Collection of 2 source objects
sampleValue = col.getB(pointToCalculate)    ## Field Sample from the 2 item Collection
                                            ## Output: [0.02236098 0.02485036 0.02734824]

markerZero = [0,0,0] + ["Zero Mark"]                   ## A Marker On Zero Coordinates
markerSample = pointToCalculate + [str(sampleValue)]   ## A Marker On the Field sample

col.displaySystem(markers=[ markerZero, 
                            markerSample], direc=True) ## Direc kwarg shows magnetization
                                                       ## vector