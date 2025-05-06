import numpy as np

def sigmoid_function(x):
    return 1 / (1 + np.exp(-x))

print(sigmoid_function(7), sigmoid_function(-22))  
# Output: (0.9990889488055994,2.7894680920908113e-10)
