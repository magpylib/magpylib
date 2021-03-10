import magpylib as mag3

pm1 = mag3.magnet.Box((1,2,3),(1,2,3))
pm1.move_by((15,15,15),steps=5)

pm2 = mag3.magnet.Cylinder((1,2,3),(1,3))
pm2.move_by((15,15,15),steps=5)
pm2.rotate_from_angax(222,'z',anchor=0,steps=-5)

mag3.display(pm1,pm2, show_path=False, markers=[])
