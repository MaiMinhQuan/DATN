import numpy as np

class LogisticRegression:
    def __init__(self, lr=0.001, n_iters=1000):
        """
        Hàm khởi tạo lớp LogisticRegression.
        lr: Tốc độ học (learning rate), kiểm soát mức độ thay đổi tham số trong mỗi lần cập nhật.
        n_iters: Số lần lặp (iterations) khi huấn luyện mô hình.
        """
        self.lr = lr  # Tốc độ học
        self.n_iters = n_iters  # Số lần lặp
        self.weights = None  
        self.bias = None  
    
    def fit(self, X, y):
        """
        Hàm huấn luyện mô hình Logistic Regression bằng phương pháp Gradient Descent.
        X: Dữ liệu huấn luyện, mỗi dòng là một mẫu, mỗi cột là một đặc trưng.
        y: Nhãn của các mẫu dữ liệu, 0 hoặc 1 trong bài toán phân loại nhị phân.
        """
        # Khởi tạo các tham số
        n_samples, n_features = X.shape         # Số mẫu và số đặc trưng trong dữ liệu huấn luyện
        self.weights = np.zeros(n_features)     # Khởi tạo trọng số bằng 0
        self.bias = 0                           # Khởi tạo bias bằng 0
        
        # Huấn luyện mô hình với Gradient Descent
        for _ in range(self.n_iters):  # Lặp lại quá trình tối ưu hóa n_iters lần
            # Tính toán đầu ra dự đoán từ mô hình tuyến tính
            linear_model = np.dot(X, self.weights) + self.bias
            # Áp dụng hàm sigmoid để chuyển đổi đầu ra tuyến tính thành xác suất
            y_predict = self._sigmoid(linear_model)
            
            # Tính toán gradient đối với trọng số (dw) và bias (db)
            dw = (1 / n_samples) * np.dot(X.T, (y_predict - y))     # Đạo hàm theo w
            db = (1 / n_samples) * np.sum(y_predict - y)            # Đạo hàm theo b
            
            # Cập nhật các tham số theo hướng ngược với gradient
            self.weights -= self.lr * dw    # Cập nhật trọng số
            self.bias -= self.lr * db       # Cập nhật bias
    
    def predict(self, X):
        """
        Hàm dự đoán nhãn cho các mẫu dữ liệu mới.
        X: Dữ liệu cần dự đoán.
        Trả về nhãn dự đoán (0 hoặc 1) cho từng mẫu trong X.
        """
        # Tính toán đầu ra dự đoán từ mô hình tuyến tính
        linear_model = np.dot(X, self.weights) + self.bias
        # Áp dụng hàm sigmoid để chuyển đổi đầu ra tuyến tính thành xác suất
        y_predict = self._sigmoid(linear_model)
        
        # Dự đoán nhãn: nếu xác suất > 0.5 thì dự đoán nhãn 1, ngược lại là 0
        y_predicted_class = [1 if i > 0.5 else 0 for i in y_predict]
        return y_predicted_class
        
    def _sigmoid(self, x):
        """
        Hàm sigmoid chuyển đầu vào thành xác suất trong khoảng (0, 1).
        x: Đầu vào của hàm sigmoid (giá trị tuyến tính từ mô hình).
        Trả về xác suất (0 đến 1).
        """
        return 1 / (1 + np.exp(-x))
