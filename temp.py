import magpylib3 as mag3

pm = mag3.magnet.Box(mag=(1,2,3),dim=(1,2,3))
pos_obs = (1,2,3)

print(mag3.getB([pm], pos_obs, sumupp=True))

