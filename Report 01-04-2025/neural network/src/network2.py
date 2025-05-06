"""
network2.py
~~~~~~~~~~~~~~

Một phiên bản cải tiến của network.py, triển khai thuật toán
học bằng stochastic gradient descent (SGD) cho mạng nơ-ron feedforward. 
Bổ sung hàm cost cross-entropy, L2 regularization, early stopping
"""

# Thư viện 
import random
import numpy as np


# ĐỊNH NGHĨA COST FUNCTION
class CrossEntropyCost(object):
    """
    Lớp định nghĩa cost function dạng Cross-Entropy.
    """
    
    def fn(a, y):
        """
        Trả về cost function với output 'a' và nhãn mong muốn 'y'.
        Sử dụng công thức:
          cost = -sum( y*log(a) + (1-y)*log(1-a) )
        Hàm np.nan_to_num được dùng để tránh lỗi log(0).
        """
        return np.sum(np.nan_to_num(
            -y*np.log(a) - (1-y)*np.log(1-a)
        ))

    def delta(z, a, y):
        """
        Trả về đạo hàm của cost function theo z.
        """
        return (a-y)


# LỚP CHÍNH: NETWORK                     
class Network(object):
    def __init__(self, sizes, cost=CrossEntropyCost):
        """
        Khởi tạo mạng nơ-ron với cấu trúc cho trong 'sizes'.
        - sizes là 1 list chứa số lượng nơ-ron trong từng lớp của mạng nơ-ron.
        Ví dụ:
            Nếu sizes = [2, 3, 1], thì đây là một mạng nơ-ron ba lớp:
            Lớp thứ nhất (lớp đầu vào) có 2 nơ-ron.
            Lớp thứ hai (lớp ẩn) có 3 nơ-ron.
            Lớp thứ ba (lớp đầu ra) có 1 nơ-ron.
            
        - weights và biases trong mạng được khởi tạo ngẫu nhiên 
        
        - cost: Định nghĩa cost function sẽ dùng (mặc định CrossEntropyCost).
        """
        self.num_layers = len(sizes)  # Tổng số lớp
        self.sizes = sizes            # Lưu lại cấu trúc mạng
        self.cost = cost              # Hàm cost đang sử dụng
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [
            np.random.randn(y, x)
            for x, y in zip(self.sizes[:-1], self.sizes[1:])
        ]


    def feedforward(self, x):
        """
        Trả về đầu ra của mạng nếu đầu vào là 'x'.
        Thực hiện lần lượt mỗi lớp: z = w*x + b, a = sigmoid(z).
        """
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, x)+b)
            x = a
        return a
    

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            lmbda=0.0,
            evaluation_data=None,
            monitor_evaluation_cost=False,
            monitor_evaluation_accuracy=False,
            monitor_training_cost=False,
            monitor_training_accuracy=False,
            early_stopping_n=0):
        """
        Huấn luyện mạng bằng mini-batch stochastic gradient descent:
        - training_data: list (x, y) của dữ liệu huấn luyện.
        - epochs: số lần lặp toàn bộ dữ liệu.
        - mini_batch_size: kích thước mỗi batch.
        - eta: learning rate.
        - lmbda: tham số regularization (L2).
        - evaluation_data: data để đánh giá sau mỗi epoch.
        - monitor_*: các biến kiểm soát xem có in cost / accuracy
          của training_data và evaluation_data sau mỗi epoch hay không.
        - early_stopping_n: dừng sớm nếu sau n epoch liên tiếp
          không cải thiện độ chính xác.
        """

        # Chuyển đổi training_data thành list (nếu chưa phải list) 
        # và lấy số lượng mẫu.
        training_data = list(training_data)
        n = len(training_data)

        if evaluation_data:
            evaluation_data = list(evaluation_data)
            n_data = len(evaluation_data)

        # Các biến dùng cho Early Stopping
        best_accuracy = 0
        no_accuracy_change = 0

        # Các list lưu cost và accuracy để theo dõi cho từng epoch
        evaluation_cost, evaluation_accuracy = [], []
        training_cost, training_accuracy = [], []

        # Vòng lặp huấn luyện chính
        for j in range(epochs):
            # Trộn dữ liệu huấn luyện ngẫu nhiên mỗi epoch
            random.shuffle(training_data)

            # Chia training_data thành các mini_batch
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)
            ]

            # Huấn luyện trên từng mini_batch
            for mini_batch in mini_batches:
                self.update_mini_batch(
                    mini_batch, eta, lmbda, len(training_data)
                )

            # Thông báo khi hoàn thành 1 epoch
            print("Epoch %s training complete" % j)

            # Nếu bật monitor_training_cost => tính cost trên training_data
            if monitor_training_cost:
                cost_val = self.total_cost(training_data, lmbda)
                training_cost.append(cost_val)
                print("Cost on training data: {}".format(cost_val))

            # Nếu bật monitor_training_accuracy => tính accuracy trên training_data
            if monitor_training_accuracy:
                accuracy_val = self.accuracy(training_data, convert=True)
                training_accuracy.append(accuracy_val)
                print("Accuracy on training data: {} / {}".format(accuracy_val, n))

            # Nếu bật monitor_evaluation_cost => tính cost trên evaluation_data
            if monitor_evaluation_cost and evaluation_data:
                cost_val = self.total_cost(evaluation_data, lmbda, convert=True)
                evaluation_cost.append(cost_val)
                print("Cost on evaluation data: {}".format(cost_val))

            # Nếu bật monitor_evaluation_accuracy => tính accuracy trên evaluation_data
            if monitor_evaluation_accuracy and evaluation_data:
                accuracy_val = self.accuracy(evaluation_data)
                evaluation_accuracy.append(accuracy_val)
                print("Accuracy on evaluation data: {} / {}".format(accuracy_val, n_data))

                # Early Stopping
                if early_stopping_n > 0:
                    # Cập nhật best_accuracy nếu thấy accuracy mới tốt hơn
                    if accuracy_val > best_accuracy:
                        best_accuracy = accuracy_val
                        no_accuracy_change = 0
                    else:
                        no_accuracy_change += 1

                    # Nếu liên tiếp 'early_stopping_n' epoch không cải thiện
                    if no_accuracy_change == early_stopping_n:
                        return (evaluation_cost, evaluation_accuracy, training_cost, training_accuracy)

        # Trả về kết quả cuối cùng
        return (evaluation_cost, evaluation_accuracy, training_cost, training_accuracy)                

    def update_mini_batch(self, mini_batch, eta, lmbda, n):
        """
        Cập nhật weights và biases của mạng bằng backpropagation và gradient descent cho một mini_batch.
        - mini_batch: list (x, y)
        - eta: learning rate
        - lmbda: tham số L2 regularization
        - n: tổng số mẫu trong training_data (dùng tính trung bình cho regularization)
        """

        # Khởi tạo nabla_b, nabla_w chứa gradient của toàn batch, ban đầu = 0.
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # Tính và cộng dồn gradient từ từng (x, y) trong mini_batch
        for x, y in mini_batch:
            # Từ backprop ta được gradient cho (x, y)
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)

            # Cộng dồn gradient
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        # Cập nhật trọng số với L2 regularization
        # (1 - eta*(lmbda/n))*w => làm giảm w để chống overfitting
        # -(eta/len(mini_batch))*nw => áp dụng descent theo gradient trung bình của batch
        self.weights = [
            (1 - eta*(lmbda/n))*w - (eta/len(mini_batch))*nw
            for w, nw in zip(self.weights, nabla_w)
        ]

        # Cập nhật biases 
        self.biases = [
            b - (eta/len(mini_batch))*nb
            for b, nb in zip(self.biases, nabla_b)
        ]

    def backprop(self, x, y):
        """
        Thực hiện lan truyền ngược (backpropagation) để tính gradient cho cost function Cx với từng lớp.
        Trả về tuple (nabla_b, nabla_w) là list các gradient cho bias và weight.
        """
        # Khởi tạo list gradient = 0 cho mỗi lớp
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # Feedforward
        activation = x  # đầu ra (activation) của lớp hiện tại (đầu tiên = input)
        activations = [x]  # list lưu lại activation của tất cả lớp
        zs = []  # list lưu lại giá trị z = w.x + b trước khi qua sigmoid

        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        # Backward pass 
        # Tính lỗi ở lớp cuối 
        delta = (self.cost).delta(zs[-1], activations[-1], y)

        # Gradient cho bias lớp cuối
        nabla_b[-1] = delta                                                         # Công thức BP3
        # Gradient cho weight lớp cuối
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())                    # Công thức BP4

        # Tính lỗi cho các lớp ẩn 
        for l in range(2, self.num_layers):
            z = zs[-l]
            sd = sigmoid_derivative(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sd              # Công thức BP2
            nabla_b[-l] = delta                                                     # Công thức BP3       
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())              # Công thức BP4

        return (nabla_b, nabla_w)

    def accuracy(self, data, convert=False):
        """
        Trả về số lượng mẫu trong 'data' mà mạng dự đoán đúng.
        Mạng dự đoán 'label' = argmax(activation_layer_cuối).
        - convert=False: dùng cho data test/ validation (y là nhãn index).
        - convert=True : dùng cho training_data (y là vector one-hot => cần chuyển sang index).
        """
        if convert:
            # Nếu là dữ liệu training, y ở dạng vector one-hot => chuyển sang index
            results = [(np.argmax(self.feedforward(x)), np.argmax(y))
                       for (x, y) in data]
        else:
            # Dữ liệu test/ validation => y đã là index
            results = [(np.argmax(self.feedforward(x)), y)
                       for (x, y) in data]

        # Đếm số lượng (prediction == label) 
        result_accuracy = sum(int(x == y) for (x, y) in results)
        return result_accuracy

    def total_cost(self, data, lmbda, convert=False):
        """
        Tính tổng cost (bao gồm regularization) trên tập 'data'.
        - convert=False: cho training_data (y là vector one-hot).
        - convert=True: cho test/ validation (y là chỉ số lớp => cần vector one-hot).
        """
        cost = 0.0
        # Lặp qua từng (x, y) trong data
        for x, y in data:
            a = self.feedforward(x)
            # Nếu convert=True => y là chỉ số => chuyển thành vector one-hot
            if convert:
                y = vectorized_result(y)
            # Tính cost do hàm cost.fn(a, y) => chia cho len(data) để lấy trung bình
            cost += self.cost.fn(a, y)/len(data)
            # Thêm term regularization (L2) => 0.5*(lmbda/len(data))*sum(||w||^2)
            cost += 0.5 * (lmbda/len(data)) * sum(np.linalg.norm(w)**2 for w in self.weights)
        return cost




# HÀM HỖ TRỢ VECTOR HÓA KẾT QUẢ             
def vectorized_result(j):
    """
    Tạo ra một vector 10 chiều có giá trị 1.0 tại vị trí j (0..9),
    còn lại là 0. Dùng để chuyển nhãn (chỉ số lớp) sang vector one-hot
    trong bài toán MNIST (10 lớp).
    """
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e


# HÀM KÍCH HOẠT VÀ ĐẠO HÀM SIGMOID             
def sigmoid(z):
    """
    Hàm sigmoid: s(z) = 1/(1 + exp(-z)).
    """
    return 1.0/(1.0+np.exp(-z))

def sigmoid_derivative(z):
    """
    Đạo hàm của sigmoid: s'(z) = s(z)*(1 - s(z)).
    """
    return sigmoid(z)*(1-sigmoid(z))
