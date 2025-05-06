import numpy as np

def tanh_function(x):
    z = (2/(1 + np.exp(-2*x))) -1
    return z

print(tanh_function(0.5), tanh_function(-1))  
# Output: (0.4621171572600098, -0.7615941559557646)
