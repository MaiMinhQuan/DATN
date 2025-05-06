import mnist_loader
import network2

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

# ----------------------
# - Ví dụ sử dụng network2.py:

'''
net = network2.Network([784, 30, 10], cost=network2.CrossEntropyCost)
net.SGD(training_data, 30, 10, 0.1, lmbda = 5.0, evaluation_data=validation_data,
    monitor_evaluation_accuracy=True)
'''

# Ví dụ overfitting - huấn luyện quá nhiều epoch trên tập dữ liệu nhỏ (1000 mẫu).
# Epoch 30, độ chính xác (accuracy) trên evaluation_data đạt 82% và không tăng thêm.
'''
net = network2.Network([784, 30, 10], cost=network2.CrossEntropyCost)
net.SGD(training_data[:1000], 400, 10, 0.5, evaluation_data=test_data,
    monitor_evaluation_accuracy=True,
    monitor_training_cost=True)
'''

# Ví dụ về L2 regularization (weight decay) với tập huấn luyện 1000 mẫu và 30 neuron ẩn
# Epoch 30, accuracy trên evaluation_data đạt 82%, epoch 50 - 83%, epoch 94 - 84%.
'''
net = network2.Network([784, 30, 10], cost=network2.CrossEntropyCost)
net.SGD(training_data[:1000], 400, 10, 0.5,
    evaluation_data=test_data,
    lmbda = 0.1, # tham số regularization
    monitor_evaluation_cost=True,
    monitor_evaluation_accuracy=True,
    monitor_training_cost=True,
    monitor_training_accuracy=True)
'''

# Ví dụ sử dụng Early stopping
# Dừng ở epoch 23 với độ chính xác 87%
'''
net = network2.Network([784, 30, 10], cost=network2.CrossEntropyCost)
net.SGD(training_data[:1000], 30, 10, 0.5,
    lmbda=5.0,
    evaluation_data=validation_data,
    monitor_evaluation_accuracy=True,
    monitor_training_cost=True,
    early_stopping_n=10)
'''


