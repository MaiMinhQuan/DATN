"""
network.py


Module triển khai thuật toán Stochastic Gradient Descent (SGD) cho feedforward neural network. 
Gradient được tính toán bằng thuật toán backpropagation.
"""

#### Libraries
import random
import time
import numpy as np


class Network(object):

    def __init__(self, sizes): #sizes chứa số nơ-ron trong mỗi lớp của mạng nơ-ron.
        """
        sizes là 1 list chứa số lượng nơ-ron trong từng lớp của mạng nơ-ron.
        Ví dụ:
            Nếu sizes = [2, 3, 1], thì đây là một mạng nơ-ron ba lớp:
            Lớp thứ nhất (lớp đầu vào) có 2 nơ-ron.
            Lớp thứ hai (lớp ẩn) có 3 nơ-ron.
            Lớp thứ ba (lớp đầu ra) có 1 nơ-ron.
            weights và biases trong mạng được khởi tạo ngẫu nhiên 

            Lưu ý rằng lớp đầu tiên được coi là lớp đầu vào, và theo quy ước, không có bias nào được thiết lập cho lớp này, vì bias chỉ ảnh hưởng đến các lớp sau trong quá trình tính toán đầu ra.
        """
        self.num_layers = len(sizes) #Lưu lại số lớp của mạng nơ-ron.
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]] #Khởi tạo biases cho tất cả các lớp ngoại trừ lớp đầu vào
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])] #Khởi tạo trọng số (weights) giữa các lớp.
                                                                #sizes[:-1] lấy số nơ-ron của các lớp trước (lớp nguồn).
                                                                #sizes[1:] lấy số nơ-ron của các lớp sau (lớp đích).
    
    def feedforward(self, x):
        """
        Trả về đầu ra của mạng nơ-ron khi đầu vào là x, đầu ra là a.
        """
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, x)+b)
            x = a
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None):
        """
        Huấn luyện mạng nơ-ron bằng thuật toán Stochastic Gradient Descent với mini-batch
            training_data là một danh sách chứa các cặp (x, y), trong đó:
                x là đầu vào của mô hình.
                y là đầu ra mong muốn.
        
        Mạng sẽ được đánh giá trên test_data sau mỗi epoch, và đánh giá sẽ được hiển thị.
        """
        if test_data: n_test = len(test_data)
        n = len(training_data)
        for j in range(epochs):
            time1 = time.time()
            random.shuffle(training_data)
            
            mini_batches = [    #Tạo 1 list gồm các mini_batch
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)] 
            
            for mini_batch in mini_batches: #Cập nhật các trọng số weight và bias dựa trên mỗi mini-batch
                self.update_mini_batch(mini_batch, eta)
            
            time2 = time.time()
            if test_data:   #Đánh giá sau mỗi epoch
                print("Epoch {0}: {1} / {2}, took {3:.2f} seconds".format(
                    j, self.evaluate(test_data), n_test, time2-time1))
            else:
                print("Epoch {0} complete in {1:.2f} seconds".format(j, time2-time1))

    def update_mini_batch(self, mini_batch, eta):
        """
        Cập nhật trọng số weights và biases của mạng bằng cách áp dụng thuật toán Gradient Descent kết hợp Backpropagation
            mini_batch là danh sách chứa các cặp (x, y), trong đó:
                x là đầu vào của mô hình.
                y là đầu ra mong muốn.
            
            eta là learning rate
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)] # Gradient của C theo bias, delta_nabla_b: Gradient của bias cho một ví dụ trong mini-batch hiện tại, tính được từ hàm backprop(x, y)
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)] # Gradient của C theo weight, delta_nabla_w: Gradient của trọng số cho một ví dụ trong mini-batch hiện tại, tính được từ hàm backprop(x, y)
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        """
        Trả về một cặp (nabla_b, nabla_w) là gradient của Cx
            nabla_b: Gradient theo bias.
            nabla_w: Gradient theo weight.
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x      # Lưu đầu vào ban đầu của mạng.
        activations = [x]   # Danh sách chứa tất cả các đầu ra của mỗi lớp (bao gồm cả đầu vào).
        zs = []             # Danh sách chứa giá trị z trước khi áp dụng hàm sigmoid.
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * sigmoid_derivative(zs[-1])    # Lỗi của output layer (BP1)
        nabla_b[-1] = delta                                                              # BP3
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())                         # BP4
        
        for l in range(2, self.num_layers):
            z = zs[-l]
            sd = sigmoid_derivative(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sd                   # BP2
            nabla_b[-l] = delta                                                          # BP3
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())                   # BP4
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        """
        Trả về số lượng mẫu kiểm tra mà mạng nơ-ron dự đoán đúng.
        Output của mạng nơ-ron được giả định là chỉ số của nơ-ron trong lớp cuối cùng có giá trị kích hoạt cao nhất.
        """
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        """
        Trả về vector đạo hàm riêng của Cost function với các giá trị kích hoạt đầu ra
        """
        return (output_activations-y)

#### Miscellaneous functions
def sigmoid(z):
    """Hàm sigmoid"""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_derivative(z):
    """Đạo hàm của hàm sigmoid"""
    return sigmoid(z)*(1-sigmoid(z))
