import numpy as np

# Hàm sigmoid tính toán giá trị đầu ra của hàm sigmoid
def sigmoid(x):
    return 1/(1+np.exp(-x))

# Hàm tính đạo hàm của hàm sigmoid
def sigmoid_derivative(x):
    return x*(1-x)

class NeuralNetwork:
    def __init__(self, layers, alpha=0.1):
        """
        Hàm khởi tạo lớp NeuralNetwork.
        layers: Danh sách chứa số lượng nơ-ron trong mỗi lớp của mạng.
        alpha: Tốc độ học (learning rate).
        """
        self.layers = layers  # Danh sách các lớp trong mạng
        self.alpha = alpha  # Tốc độ học
        
        self.W = []  # Danh sách chứa weights của các lớp
        self.b = []  # Danh sách chứa bias của các lớp
        
        # Khởi tạo trọng số và bias cho từng lớp
        for i in range(0, len(layers)-1):
            # Trọng số khởi tạo ngẫu nhiên, chia cho số lượng nơ-ron của lớp trước để làm chuẩn hóa
            w_ = np.random.randn(layers[i], layers[i+1])
            b_ = np.zeros((layers[i+1], 1))  # Bias khởi tạo bằng 0
            self.W.append(w_/layers[i])  # Lưu trọng số vào danh sách W
            self.b.append(b_)  # Lưu bias vào danh sách b

    def fit_partial(self, x, y):
        """
        Hàm huấn luyện 1 lần (partial) cho mạng nơ-ron, thực hiện forward propagation và backpropagation.
        x: Dữ liệu đầu vào.
        y: Nhãn thực tế (label).
        """
        A = [x]  # Danh sách lưu trữ đầu ra của từng lớp, bắt đầu với đầu vào x

        out = A[-1]
        # Feedforward
        for i in range(0, len(self.layers) - 1):
            out = sigmoid(np.dot(out, self.W[i]) + (self.b[i].T))  # Tính toán đầu ra cho lớp i
            A.append(out)  # Lưu trữ đầu ra lớp i

        y = y.reshape(-1, 1)  # Đảm bảo y có kích thước (n_samples, 1)
        
        # Tính toán sai số của đầu ra cuối cùng (dA)
        dA = [-(y/A[-1] - (1-y)/(1-A[-1]))]

        dW = []  # Lưu trữ gradient của trọng số
        db = []  # Lưu trữ gradient của bias
        for i in reversed(range(0, len(self.layers)-1)):
            # Tính gradient của trọng số và bias
            dw_ = np.dot((A[i]).T, dA[-1] * sigmoid_derivative(A[i+1]))
            db_ = (np.sum(dA[-1] * sigmoid_derivative(A[i+1]), 0)).reshape(-1,1)
            
            # Tính gradient ngược của dA
            dA_ = np.dot(dA[-1] * sigmoid_derivative(A[i+1]), self.W[i].T)
            
            dW.append(dw_)  # Lưu gradient của trọng số
            db.append(db_)  # Lưu gradient của bias
            dA.append(dA_)  # Lưu gradient của dA

        # Đảo ngược dW và db để khớp với thứ tự của các lớp
        dW = dW[::-1]
        db = db[::-1]
   
        # Cập nhật trọng số và bias bằng cách sử dụng gradient descent
        for i in range(0, len(self.layers)-1):
            self.W[i] = self.W[i] - self.alpha * dW[i]  # Cập nhật trọng số
            self.b[i] = self.b[i] - self.alpha * db[i]  # Cập nhật bias

    def fit(self, X, y, epochs=20, verbose=10):
        """
        Hàm huấn luyện mô hình cho nhiều epoch (số lần lặp).
        X: Dữ liệu đầu vào.
        y: Nhãn đầu ra.
        epochs: Số lần lặp huấn luyện.
        verbose: Cập nhật sau mỗi bao nhiêu epoch.
        """
        for epoch in range(0, epochs):
            self.fit_partial(X, y)  # Huấn luyện một phần (partial)
            if epoch % verbose == 0:
                # In ra loss sau mỗi verbose epoch
                loss = self.calculate_loss(X, y)
                print("Epoch {}, loss {}".format(epoch, loss))

    def predict(self, X):
        """
        Hàm dự đoán nhãn cho dữ liệu đầu vào X.
        X: Dữ liệu cần dự đoán.
        Trả về nhãn dự đoán.
        """
        for i in range(0, len(self.layers) - 1):
            Y = sigmoid(np.dot(X, self.W[i]) + (self.b[i].T))  # Lan truyền tiến qua các lớp
            X = Y
        return Y

    def calculate_loss(self, X, y):
        """
        Hàm tính toán loss (tổn thất) của mô hình.
        X: Dữ liệu đầu vào.
        y: Nhãn thực tế.
        """
        y_predict = self.predict(X)  # Dự đoán đầu ra từ mô hình
        # Tính loss theo công thức cross-entropy loss
        return -(np.sum(y*np.log(y_predict) + (1-y)*np.log(1-y_predict)))
