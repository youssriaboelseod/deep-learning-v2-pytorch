import numpy as np

from numpy import *
x = array([[3,2,3],[4,4,4]])
#y = set(x.flatten())
y = set(e for r in x
             for e in r)
set([2, 3, 4])
z = set(tuple(r) for r in x)
print(y)