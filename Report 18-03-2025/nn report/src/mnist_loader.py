"""
mnist_loader
~~~~~~~~~~~~

Cài đặt hàm để tải dữ liệu hình ảnh MNIST.
"""

#### Libraries
import pickle
import gzip
import numpy as np

def load_data():
    """
    Hàm trả về dữ liệu MNIST dưới dạng một bộ ba (tuple) chứa:
        +) Dữ liệu huấn luyện (training data)
        +) Dữ liệu kiểm định (validation data)
        +) Dữ liệu kiểm tra (test data)
        
    Cấu trúc của training_data: Training_data được trả về dưới dạng một tuple có hai phần:
        +) Phần thứ nhất: Chứa hình ảnh huấn luyện, được lưu dưới dạng numpy ndarray với 50.000 phần tử.
            Mỗi phần tử lại là một numpy ndarray gồm 784 giá trị (28 * 28 = 784 pixel), tương ứng với một hình ảnh MNIST.
        +) Phần thứ hai: Chứa nhãn số thực tế (0-9) của từng hình ảnh trong training_data, được lưu trong một numpy ndarray với 50.000 phần tử.
    """
    f = gzip.open('../data/mnist.pkl.gz', 'rb')
    u = pickle._Unpickler(f)
    u.encoding = 'latin1'
    training_data, validation_data, test_data = u.load()
    f.close()
    return (training_data, validation_data, test_data)

def load_data_wrapper():
    """
    Hàm trả về một tuple chứa (training_data, validation_data, test_data).
    Dữ liệu được lấy từ load_data, nhưng có định dạng thuận tiện hơn để sử dụng trong quá trình triển khai mạng nơ-ron.
    
    training_data là danh sách chứa 50.000 cặp (x, y), trong đó:
        +) x là một mảng numpy 784 chiều (numpy.ndarray), đại diện cho hình ảnh đầu vào (28 * 28 = 784 pixel).
        +) y là một mảng numpy 10 chiều (numpy.ndarray), đại diện cho chữ số thực tế trong hình ảnh x.
        
    validation_data và test_data là danh sách chứa 10.000 cặp (x, y), trong đó:
        +) x là một mảng numpy 784 chiều (numpy.ndarray), đại diện cho hình ảnh đầu vào.
        +) y là một số nguyên (0-9), tương ứng với chữ số thực tế trong hình ảnh x.
    """
    tr_d, va_d, te_d = load_data()
    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    training_results = [vectorized_result(y) for y in tr_d[1]]
    training_data = list(zip(training_inputs, training_results))
    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    validation_data = list(zip(validation_inputs, va_d[1]))
    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_data = list(zip(test_inputs, te_d[1]))
    return (training_data, validation_data, test_data)

def vectorized_result(j):
    """
    Trả về một vector đơn vị 10 chiều, trong đó giá trị 1.0 nằm ở vị trí thứ j, các vị trí còn lại là 0.
    Hàm được sử dụng để chuyển một chữ số (0...9) thành đầu ra mong muốn tương ứng của mạng nơ-ron.
    """
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e
