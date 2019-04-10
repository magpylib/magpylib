from magpylib import source, Collection
from matplotlib import pyplot

pyplot.ioff() # Enable interactivity suppression for figure exports

b = source.magnet.Box(mag=[1,2,3],
                      dim=[2,2,2],
                      angle=45,
                      axis=(0,0,1), # Rotate 45 in respect to Z
                      pos=[0,0,0])

col = Collection(b)

fig = col.displaySystem(suppress=True) # Suppress interactive images
fig.suptitle("Rotated 45° in (0,0,1)")
fig.set_size_inches(6, 6)

b.rotate(-90,(0,1,0))
fig2 = col.displaySystem(suppress=True) # Suppress interactive images
fig2.suptitle("Rotated -90° in (0,1,0) relative to previous change")
fig2.set_size_inches(6, 6)

pyplot.show() # Show images
