import numpy as np
from collections import Counter

# Hàm tính khoảng cách Euclidean giữa hai điểm x1 và x2
def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))
    
class KNN:
    def __init__(self, k = 3):
        """
        Hàm khởi tạo lớp KNN.
        k: số lượng láng giềng gần nhất (số lớp sẽ được dự đoán từ k láng giềng gần nhất).
        """
        self.k = k
    
    def fit(self, X, y):
        """
        Hàm huấn luyện mô hình KNN.
        X: Dữ liệu huấn luyện (mỗi dòng là một mẫu dữ liệu).
        y: Nhãn tương ứng với dữ liệu huấn luyện (một mảng 1 chiều).
        """
        self.X_train = X  # Lưu dữ liệu huấn luyện
        self.y_train = y  # Lưu nhãn huấn luyện
       
    def _predict(self, x):  # Dự đoán nhãn cho 1 mẫu dữ liệu x
        """
        Hàm này tính toán dự đoán cho 1 mẫu x dựa trên KNN.
        x: Mẫu dữ liệu cần dự đoán.
        """
        # Tính toán khoảng cách Euclidean từ mẫu x đến tất cả các mẫu trong dữ liệu huấn luyện
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
        
        # Lấy k chỉ số của các mẫu gần nhất (theo khoảng cách nhỏ nhất)
        k_indices = np.argsort(distances)[:self.k]  # sắp xếp khoảng cách và lấy k chỉ số đầu tiên
        
        # Lấy nhãn của k mẫu gần nhất
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        
        # Tính toán nhãn phổ biến nhất trong k nhãn gần nhất (bỏ phiếu)
        most_common = Counter(k_nearest_labels).most_common(1) # trả về nhãn xuất hiện nhiều nhất cùng tần suất
        
        # Trả về nhãn phổ biến nhất
        return most_common[0][0]  # nhãn được dự đoán
        
    def predict(self, X):   # Dự đoán nhãn cho nhiều mẫu dữ liệu X
        """
        Hàm này dự đoán nhãn cho nhiều mẫu dữ liệu X.
        X: Dữ liệu đầu vào (nhiều mẫu dữ liệu).
        """
        predicted_labels = [self._predict(x) for x in X]  # Dự đoán cho mỗi mẫu trong X
        return np.array(predicted_labels)  # Trả về mảng các nhãn đã dự đoán
