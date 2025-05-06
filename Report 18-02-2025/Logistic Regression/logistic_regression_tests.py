import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt
from logistic_regression import LogisticRegression

# Tải bộ dữ liệu Breast Cancer từ thư viện sklearn
bc = datasets.load_breast_cancer()

# Lấy dữ liệu (X) và nhãn (y) từ bộ dữ liệu Breast Cancer
X, y = bc.data, bc.target

# Chia dữ liệu thành 2 phần: 80% cho huấn luyện và 20% cho kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

# Hàm tính toán độ chính xác (accuracy) của mô hình
def accuracy(y_true, y_predict):
    accuracy = np.sum(y_true == y_predict) / len(y_true)  # Tính tỷ lệ mẫu dự đoán đúng
    return accuracy

# Khởi tạo mô hình Logistic Regression với learning rate là 0.0001 và số lần lặp là 1000
regressor = LogisticRegression(lr=0.0001, n_iters=1000)

# Huấn luyện mô hình với dữ liệu huấn luyện (X_train, y_train)
regressor.fit(X_train, y_train)

# Dự đoán nhãn cho dữ liệu kiểm tra (X_test)
predictions = regressor.predict(X_test)

# Tính toán và in ra độ chính xác của mô hình
print("LR classification accuracy", accuracy(y_test, predictions))
