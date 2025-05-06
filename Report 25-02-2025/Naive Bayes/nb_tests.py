import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
from nb import NaiveBayes  # Nhập lớp NaiveBayes từ module nb đã cài đặt trước

# Hàm tính toán độ chính xác (accuracy) của mô hình
def accuracy(y_true, y_pred):
    """
    Hàm này tính toán độ chính xác của mô hình.
    y_true: Nhãn thực tế của mẫu.
    y_pred: Nhãn dự đoán của mẫu.
    """
    accuracy = np.sum(y_true == y_pred) / len(y_true)  # Tính tỷ lệ mẫu dự đoán đúng
    return accuracy

# Tạo dữ liệu giả lập cho bài toán phân loại với 1000 mẫu, 10 đặc trưng và 2 lớp
X, y = datasets.make_classification(n_samples=1000, n_features=10, n_classes=2, random_state=123)

# Chia bộ dữ liệu thành 80% cho huấn luyện và 20% cho kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

# Khởi tạo đối tượng NaiveBayes
nb = NaiveBayes()

# Huấn luyện mô hình với dữ liệu huấn luyện (X_train, y_train)
nb.fit(X_train, y_train)

# Dự đoán nhãn cho dữ liệu kiểm tra (X_test)
predictions = nb.predict(X_test)

# Tính toán và in ra độ chính xác của mô hình
print("Naive Bayes classification accuracy", accuracy(y_test, predictions))
