import matplotlib.pyplot as plt
import magpylib as magpy

# define Pyplot figure
fig, [ax1,ax2] = plt.subplots(1, 2, figsize=(10,5))
ax1 = plt.subplot(121,projection='3d')

# define Magpylib source
src = magpy.magnet.Sphere(magnetization=(500,0,0), diameter=1)

# define sensor and create a sensor path
sens1 = magpy.Sensor(position=(2,0,0))
sens1.rotate_from_angax([4]*90, 'y', anchor=0, increment=True)

# compute field at sensor
B = sens1.getB(src)

# define sensor with pixel
ts = (-.3,0,.3)
pix = [[(x,y,0) for x in ts] for y in ts]
sens2 = magpy.Sensor(position=(0,0,3), pixel=pix)

# compute and print field
print(sens2.getH(src))

# display system in ax1
magpy.display(src, sens1, sens2, axis=ax1, size_sensors=2)

# plot field in ax2
ax2.plot(B[:,0],'r')
ax2.plot(B[:,1],'g')
ax2.plot(B[:,2],'b')

plt.tight_layout()
plt.show()

# Output:
# [[[-0.57852203  0.01753097 -0.17530971]
#   [-0.60492715  0.          0.        ]
#   [-0.57852203 -0.01753097  0.17530971]]

#  [[-0.58695901  0.         -0.17968133]
#   [-0.6140237   0.          0.        ]
#   [-0.58695901  0.          0.17968133]]

#  [[-0.57852203 -0.01753097 -0.17530971]
#   [-0.60492715  0.          0.        ]
#   [-0.57852203  0.01753097  0.17530971]]]
