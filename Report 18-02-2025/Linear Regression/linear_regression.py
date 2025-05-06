import numpy as np

class LinearRegression:
    def __init__(self, lr=0.001, n_iters=1000):
        """
        Hàm khởi tạo lớp LinearRegression.
        lr: Learning rate, tốc độ học của mô hình (mức độ điều chỉnh khi cập nhật tham số).
        n_iters: Số lần lặp tối đa khi huấn luyện mô hình.
        """
        self.lr = lr            # learning rate
        self.n_iters = n_iters  # số lần lặp tối đa
        self.weights = None     # trọng số ban đầu chưa được khởi tạo
        self.bias = None        # bias ban đầu chưa được khởi tạo
        
    def fit(self, X, y):        # X: training sample, y: nhãn (đầu ra)
        """
        Hàm huấn luyện mô hình Linear Regression sử dụng Gradient Descent.
        X: Dữ liệu huấn luyện, mỗi hàng là một mẫu dữ liệu, mỗi cột là một đặc trưng.
        y: Nhãn tương ứng với từng mẫu dữ liệu trong X.
        """
        # Khởi tạo các tham số ban đầu
        n_sample, n_feature = X.shape       # n_sample: số mẫu, n_feature: số đặc trưng
        self.weights = np.zeros(n_feature)  # Khởi tạo trọng số với giá trị 0
        self.bias = 0                       # Khởi tạo bias với giá trị 0
        
        # Thuật toán Gradient Descent để tối ưu hóa tham số
        for _ in range(self.n_iters):                               # Lặp qua số lần tối đa n_iters
            # Dự đoán giá trị đầu ra y_predicted
            y_predicted = np.dot(X, self.weights) + self.bias
            
            # Tính toán gradient đối với trọng số (dw)
            dw = (1 / n_sample) * np.dot(X.T, (y_predicted - y))    # Đạo hàm theo w
            # Tính toán gradient đối với bias (db)
            db = (1 / n_sample) * np.sum(y_predicted - y)           # Đạo hàm theo b
            
            # Cập nhật trọng số và bias 
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
    
    def predict(self, X):
        """
        Hàm dự đoán giá trị đầu ra cho các mẫu dữ liệu mới.
        X: Dữ liệu mới cần dự đoán.
        """
        # Dự đoán kết quả đầu ra y_predicted cho dữ liệu đầu vào X
        y_predicted = np.dot(X, self.weights) + self.bias
        return y_predicted  # Trả về giá trị dự đoán
