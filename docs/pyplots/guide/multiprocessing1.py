from magpylib import source, Collection
from matplotlib import pyplot
from numpy import pi, sin,cos, arange
from multiprocessing import freeze_support

## First define code to be run in multiprocessing guard
def main():
    # Acquire Positions
    def x(angle,r):
        return r*sin(angle)

    def y(angle,r):
        return r*cos(angle)
    
    def markXY(angle,r):
        xPos = x(angle,r)
        yPos = y(angle,r)
        return [xPos,yPos,0]

    # Calculate marker positions of a circle around [0,0,0]
    markers = [markXY(angle,5) for angle in arange(0,2*pi,0.1)]

    # Define Source Object
    b = source.magnet.Sphere([0,-180,0],3,pos=[0,0,0])

    # Display Source Object
    col = Collection(b)
    col.displaySystem(markers=markers,direc=True)
    
    ## Calculate all marked points sequentially
    fields = b.getBsweep(markers,multiprocessing=False)

    ## Calculate all marked points in parallel, with multiple cores
    fields = b.getBsweep(markers,multiprocessing=True)
    print(fields)
    
    ## Plot Results
    fig = pyplot.figure()
    pyplot.plot(arange(0,2*pi,0.1),fields)
    pyplot.xlabel('Radians')
    pyplot.ylabel('miliTesla')
    fig.axes[0].lines
    pyplot.legend(  [fig.axes[0].lines[0], fig.axes[0].lines[1], fig.axes[0].lines[2]], 
                    ['X Measure', 'Y Measure', 'Z Measure'])

## Execute parallelized code safely for Windows OS
if __name__ == "__main__":
    freeze_support()
    main()