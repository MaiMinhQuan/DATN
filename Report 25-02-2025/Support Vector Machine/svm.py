import numpy as np

class SVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
        """
        Hàm khởi tạo cho mô hình SVM.
        learning_rate: Tốc độ học (learning rate) cho thuật toán Gradient Descent.
        lambda_param: Tham số điều chỉnh (regularization parameter).
        n_iters: Số lần lặp tối đa trong quá trình huấn luyện.
        """
        self.lr = learning_rate  # Tốc độ học
        self.lambda_param = lambda_param  # Tham số điều chỉnh
        self.n_iters = n_iters  # Số lần lặp
        self.w = None  
        self.b = None  

    def fit(self, X, y):
        """
        Hàm huấn luyện mô hình SVM.
        X: Dữ liệu huấn luyện (mỗi dòng là một mẫu dữ liệu, mỗi cột là một đặc trưng).
        y: Nhãn tương ứng với từng mẫu dữ liệu trong X.
        """
        n_samples, n_features = X.shape  # Lấy số mẫu và số đặc trưng từ dữ liệu huấn luyện

        # Chuyển nhãn về dạng +1 và -1 (SVM yêu cầu nhãn là +1 và -1)
        y_ = np.where(y <= 0, -1, 1)

        # Khởi tạo trọng số (w) và hệ số chặn (b)
        self.w = np.zeros(n_features)
        self.b = 0

        # Huấn luyện mô hình bằng cách sử dụng Gradient Descent
        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                # Kiểm tra điều kiện nếu mẫu thuộc lớp phân tách đúng
                condition = y_[idx] * (np.dot(x_i, self.w) - self.b) >= 1
                if condition:
                    # Nếu mẫu đã được phân tách đúng, cập nhật trọng số (w) 
                    self.w -= self.lr * (2 * self.lambda_param * self.w)
                else:
                    # Nếu mẫu không phân tách đúng, cập nhật cả trọng số (w) và bias (b)
                    self.w -= self.lr * (2 * self.lambda_param * self.w - np.dot(x_i, y_[idx]))
                    self.b -= self.lr * y_[idx]

    def predict(self, X):
        """
        Hàm dự đoán nhãn cho các mẫu dữ liệu mới.
        X: Dữ liệu cần dự đoán nhãn.
        """
        # Tính toán đầu ra tuyến tính (linear output) từ trọng số và bias
        linear_output = np.dot(X, self.w) - self.b
        # Trả về nhãn dự đoán (dấu của đầu ra tuyến tính)
        return np.sign(linear_output)
