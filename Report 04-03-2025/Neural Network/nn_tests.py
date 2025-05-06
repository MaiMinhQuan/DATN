import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from nn import NeuralNetwork  # Nhập lớp NeuralNetwork từ module nn đã cài đặt trước

# Tải bộ dữ liệu ung thư vú từ thư viện sklearn
bc = datasets.load_breast_cancer()
X, y = bc.data, bc.target  # X là các đặc trưng, y là nhãn (0: lành tính, 1: ác tính)

# Chia bộ dữ liệu thành 80% cho huấn luyện và 20% cho kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

# Chuẩn hóa các đặc trưng bằng StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # Chuẩn hóa dữ liệu huấn luyện
X_test_scaled = scaler.transform(X_test)  # Chuẩn hóa dữ liệu kiểm tra (dùng cùng tham số scaler)

# Định nghĩa hàm tính độ chính xác (accuracy)
def accuracy(y_true, y_predict):
    """
    Hàm tính độ chính xác của mô hình.
    y_true: Nhãn thực tế.
    y_predict: Nhãn dự đoán.
    """
    y_predict = (y_predict >= 0.5).astype(int)  # Chuyển đổi xác suất thành nhãn nhị phân (0 hoặc 1)
    return np.sum(y_true == y_predict) / len(y_true)  # Tính tỷ lệ mẫu dự đoán đúng

# Khởi tạo và huấn luyện mạng nơ-ron
nn_test = NeuralNetwork([X_train.shape[1], 10, 5, 1], alpha=0.001)  # Mạng có 3 lớp: đầu vào, lớp ẩn (10, 5 nơ-ron), đầu ra
nn_test.fit(X_train_scaled, y_train, epochs=2000, verbose=500)  # Huấn luyện trong 2000 epochs, in ra mỗi 500 epochs

# Dự đoán nhãn cho bộ dữ liệu kiểm tra
predictions_test = nn_test.predict(X_test_scaled)

# Tính toán và in độ chính xác
accuracy_test = accuracy(y_test, predictions_test)
print("Neural Network classification accuracy:", accuracy_test)
