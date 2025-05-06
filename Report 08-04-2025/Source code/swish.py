import numpy as np

def swish_function(x):
    return x/(1-np.exp(-x))

print(swish_function(-67), swish_function(4))  
# Output: (5.35e-28, 4.0746)
