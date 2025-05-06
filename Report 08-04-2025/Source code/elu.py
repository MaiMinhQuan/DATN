import numpy as np
def elu_function(x, a):
    if x<0:
        return a*(np.exp(x)-1)
    else:
        return x

print(elu_function(5, 0.1), elu_function(-5, 0.1))  
# Output: (5, -0.09932620530009145)
