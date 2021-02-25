import magpylib3 as mag3

pm = mag3.magnet.Box((10,10,10),(1,2,3))
pm.move_by((10,10,10),steps=100)
pm.rotate_from_angax(666,'z',anchor=0,steps=-100)
pm.display(show_path=True,direc=True)
