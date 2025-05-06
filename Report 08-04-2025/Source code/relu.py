def relu_function(x):
    if x<0:
        return 0
    else:
        return x

print(relu_function(7), relu_function(-7))  
# Output: (7, 0)
