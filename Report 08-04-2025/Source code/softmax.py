import numpy as np

def softmax_function(x):
    z = np.exp(x)
    z_ = z/z.sum()
    return z_

print(softmax_function([0.8, 1.2, 3.1]))  
# Output: [0.0802, 0.1197, 0.8001]
