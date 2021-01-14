from magpylib._lib.classes.base import RCS
from magpylib.source.magnet import *
from IPython.display import HTML, Latex
import pandas as pd
import numpy as np

box = Box(mag=(1,2,3), dim=(1,2,3), pos=(200,0,0), angle=7, axis=(3,4,5))
repr_str = box.__repr__()

ll = np.array([[s for s in rs.strip().split(':')] for rs in repr_str.split('\n')])

df = pd.DataFrame(ll, columns=('property', 'value'))#.to_html(index=False)
#print(df)
print([box])