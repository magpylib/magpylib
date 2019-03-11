Examples
========

The following code executes a mixed simulation with most features.

.. code-block:: python

   # imports
   import numpy as np
   import matplotlib.pyplot as plt
   import magpylib as magpy

   # create magnets
   magnet1 = magpy.source.magnet.Box(mag=[0,0,600],dim=[3,3,3],pos=[-4,0,3])
   magnet2 = magpy.source.magnet.Cylinder(mag=[0,0,500], dim=[3,5], pos=[0,0,0])

   # manipulate magnets
   magnet1.rotate(45,[0,1,0],anchor=[0,0,0])
   magnet2.move([5,0,-4])

   # collect magnets
   pmc = magpy.Collection(magnet1,magnet2)

   # display system geometry
   pmc.displaySystem()

   # calculate B-fields on a grid
   xs = np.linspace(-10,10,20)
   zs = np.linspace(-10,10,20)
   Bs = np.array([[pmc.getB([x,0,z]) for x in xs] for z in zs])

   # display fields using matplotlib
   fig, ax = plt.subplots()
   X,Y = np.meshgrid(xs,zs)
   U,V = Bs[:,:,0], Bs[:,:,2]
   ax.streamplot(X, Y, U, V, color=np.log(U**2+V**2), density=1.5)
   plt.show()


.. image:: ../_static/examplePlot.jpg
