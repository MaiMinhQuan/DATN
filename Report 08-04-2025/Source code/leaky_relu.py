def leaky_relu_function(x, alpha=0.01):
    if x < 0:
        return alpha * x
    else:
        return x

print(leaky_relu_function(7), leaky_relu_function (-7))  
# Output: (7, -0.07)
