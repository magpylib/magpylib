import matplotlib.pyplot as plt
from numpy import sin,cos, linspace
from magpylib import source, Collection
from multiprocessing import freeze_support
import time

#main program
def main():
    #define spiral Positions
    spi1 = [(-t*cos(t),-t*sin(t),t) for t in linspace(-50,-10,200)]
    spi2 = [(10*cos(t),10*sin(t),t) for t in linspace(-9.8, 9.8,50)]
    spi3 = [( t*cos(t), t*sin(t),t) for t in linspace( 10, 50,200)]
    
    #create source + collection
    s = source.current.Line(curr=1,vertices=spi1+spi2+spi3)
    c = Collection(s)
    
    #create positions
    zs = linspace(-15,15,150)
    posis = [[x,0,z] for x in [7,8] for z in zs ]
    
    ##define figure with 2d and 3d axes
    fig = plt.figure(figsize=(9,4))
    ax1 = fig.add_subplot(121,projection='3d')
    ax2 = fig.add_subplot(122)
    
    #add displaySystem on ax1
    c.displaySystem(subplotAx=ax1)
    
    #calculate fields and check timings
    T0 = time.perf_counter()
    Bs = c.getBsweep(posis,multiprocessing=False).reshape(2,150,3)
    T1 = time.perf_counter()
    Bs = c.getBsweep(posis,multiprocessing=True).reshape(2,150,3)
    T2 = time.perf_counter()
    
    print('timing without multiprocessing: ' + str(T1-T0))
    print('timing with multiprocessing: ' + str(T2-T1))

    #plot fields
    for i in range(2):
        ax2.plot(zs,Bs[i])

#run main within multiprocessing guard
if __name__ == "__main__":
    freeze_support()
    main()