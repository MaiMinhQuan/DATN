import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt
from linear_regression import LinearRegression

# Tạo dữ liệu giả lập cho bài toán hồi quy, với 100 mẫu, 1 đặc trưng và nhiễu là 20
X, y = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=4)

# Chia dữ liệu thành 2 phần: 80% cho huấn luyện (train) và 20% cho kiểm tra (test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

# Khởi tạo đối tượng LinearRegression với learning rate là 0.01
regressor = LinearRegression(lr=0.01)

# Huấn luyện mô hình với dữ liệu huấn luyện (X_train, y_train)
regressor.fit(X_train, y_train)

# Dự đoán giá trị đầu ra với dữ liệu kiểm tra (X_test)
predicted = regressor.predict(X_test)

# Hàm tính toán MSE (Mean Square Error) – hàm mất mát được sử dụng để đánh giá độ chính xác của mô hình
def mse(y_true, y_predicted):
    return np.mean((y_true - y_predicted) ** 2)

# Tính giá trị MSE giữa nhãn thật (y_test) và nhãn dự đoán (predicted)
mse_value = mse(y_test, predicted)

# In ra giá trị MSE
print(mse_value)

# Dự đoán cho toàn bộ dữ liệu X để vẽ đường hồi quy
y_predict_line = regressor.predict(X)

# Sử dụng colormap 'viridis' để tạo màu sắc cho đồ thị
cmap = plt.get_cmap("viridis")

# Tạo hình vẽ với kích thước 8x6
fig = plt.figure(figsize=(8,6))

# Vẽ các điểm kiểm tra (X_test, y_test) với màu đỏ
m2 = plt.scatter(X_test, y_test, color="red", s=10)

# Vẽ đường hồi quy (đường dự đoán) cho toàn bộ dữ liệu X với màu đen
plt.plot(X, y_predict_line, color="black", linewidth=2, label="Prediction")

# Hiển thị đồ thị
plt.show()
