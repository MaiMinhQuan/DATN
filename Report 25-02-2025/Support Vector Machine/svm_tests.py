import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
from svm import SVM  # Nhập lớp SVM từ module svm đã cài đặt trước

# Tạo dữ liệu giả lập cho bài toán phân loại với 2 lớp, 50 mẫu và 2 đặc trưng
X, y = datasets.make_blobs(n_samples=50, n_features=2, centers=2, cluster_std=1.05, random_state=40)

# Chuyển nhãn về dạng -1 và 1 (SVM yêu cầu nhãn là +1 và -1)
y = np.where(y == 0, -1, 1)

# Khởi tạo mô hình SVM
clf = SVM()

# Huấn luyện mô hình SVM với dữ liệu (X, y)
clf.fit(X, y)

# Dự đoán nhãn cho dữ liệu (X)
predictions = clf.predict(X)

# In ra trọng số (w) và bias (b) của mô hình SVM sau khi huấn luyện
print(clf.w, clf.b)

# In ra nhãn dự đoán
print(predictions)

def visualize_svm():
    """
    Hàm này vẽ đồ thị để trực quan hóa siêu phẳng phân tách (hyperplane)
    trong không gian 2D cho mô hình SVM.
    """
    def get_hyperplane_value(x, w, b, offset):
        """
        Hàm tính toán giá trị y của siêu phẳng dựa trên phương trình của siêu phẳng.
        """
        return (-w[0] * x + b + offset) / w[1]

    # Tạo đồ thị
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    
    # Vẽ các điểm dữ liệu
    plt.scatter(X[:, 0], X[:, 1], marker="o", c=y)
    
    # Xác định các điểm x0_1 và x0_2 để vẽ đường phân tách (hyperplane)
    x0_1 = np.amin(X[:, 0])  # Giá trị min của đặc trưng 1
    x0_2 = np.amax(X[:, 0])  # Giá trị max của đặc trưng 1

    # Tính toán giá trị y của siêu phẳng phân tách chính (y = 0)
    x1_1 = get_hyperplane_value(x0_1, clf.w, clf.b, 0)
    x1_2 = get_hyperplane_value(x0_2, clf.w, clf.b, 0)

    # Tính toán giá trị y của siêu phẳng cho các đường biên (y = -1 và y = 1)
    x1_1_m = get_hyperplane_value(x0_1, clf.w, clf.b, -1)
    x1_2_m = get_hyperplane_value(x0_2, clf.w, clf.b, -1)

    x1_1_p = get_hyperplane_value(x0_1, clf.w, clf.b, 1)
    x1_2_p = get_hyperplane_value(x0_2, clf.w, clf.b, 1)

    # Vẽ các đường phân tách và đường biên (margin)
    ax.plot([x0_1, x0_2], [x1_1, x1_2], "y--")  # Đường phân tách (hyperplane)
    ax.plot([x0_1, x0_2], [x1_1_m, x1_2_m], "k")  # Đường biên âm (margin -1)
    ax.plot([x0_1, x0_2], [x1_1_p, x1_2_p], "k")  # Đường biên dương (margin +1)

    # Cài đặt giới hạn cho trục y để đồ thị không bị cắt
    x1_min = np.amin(X[:, 1])
    x1_max = np.amax(X[:, 1])
    ax.set_ylim([x1_min - 3, x1_max + 3])

    # Hiển thị đồ thị
    plt.show()

# Gọi hàm vẽ đồ thị để trực quan hóa siêu phẳng phân tách
visualize_svm()
