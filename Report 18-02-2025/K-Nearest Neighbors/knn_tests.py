import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from matplotlib.colors import ListedColormap
from knn import KNN  

# Tải bộ dữ liệu Iris từ thư viện sklearn
iris = datasets.load_iris()

# Lấy dữ liệu và nhãn từ bộ dữ liệu Iris
X, y = iris.data, iris.target

# Chia dữ liệu thành 2 phần: 80% cho huấn luyện và 20% cho kiểm tra (test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

# Khởi tạo mô hình KNN với số lượng k=3 láng giềng gần nhất
clf = KNN(k=3)  # classifier

# Huấn luyện mô hình với dữ liệu huấn luyện (X_train, y_train)
clf.fit(X_train, y_train)

# Dự đoán nhãn cho dữ liệu kiểm tra (X_test)
predictions = clf.predict(X_test)

# Tính độ chính xác của mô hình bằng cách so sánh nhãn dự đoán với nhãn thật
accuracy = np.sum(predictions == y_test) / len(y_test)  # true predict / number of sample

# In ra độ chính xác của mô hình
print(accuracy)
