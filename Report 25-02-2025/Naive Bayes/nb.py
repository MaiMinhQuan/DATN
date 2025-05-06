import numpy as np

class NaiveBayes:
    def fit(self, X, y):
        """
        Hàm huấn luyện mô hình Naive Bayes.
        X: Dữ liệu huấn luyện (mỗi dòng là một mẫu dữ liệu, mỗi cột là một đặc trưng).
        y: Nhãn tương ứng với từng mẫu dữ liệu trong X.
        """
        n_samples, n_features = X.shape     # Lấy số lượng mẫu và số lượng đặc trưng từ dữ liệu huấn luyện
        self._classes = np.unique(y)        # Lấy các lớp (classes) khác nhau trong nhãn y
        n_classes = len(self._classes)      # Số lượng lớp

        # Khởi tạo các ma trận và mảng chứa mean, var và prior cho mỗi lớp
        self._mean = np.zeros((n_classes, n_features), dtype=np.float64)    # Mean cho mỗi lớp
        self._var = np.zeros((n_classes, n_features), dtype=np.float64)     # Variance cho mỗi lớp
        self._priors = np.zeros(n_classes, dtype=np.float64)                # Prior (xác suất lớp) cho mỗi lớp

        # Tính mean, variance và prior cho mỗi lớp
        for idx, c in enumerate(self._classes):
            X_c = X[y == c]                                         # Lọc các mẫu có nhãn là lớp c
            self._mean[idx, :] = X_c.mean(axis=0)                   # Tính trung bình (mean) của mỗi đặc trưng trong lớp c
            self._var[idx, :] = X_c.var(axis=0)                     # Tính phương sai (variance) của mỗi đặc trưng trong lớp c
            self._priors[idx] = X_c.shape[0] / float(n_samples)     # Tính prior (xác suất lớp) = số mẫu trong lớp c / tổng số mẫu

    def predict(self, X):
        """
        Hàm dự đoán nhãn cho một tập dữ liệu mới X.
        X: Dữ liệu đầu vào cần dự đoán nhãn.
        """
        # Dự đoán nhãn cho mỗi mẫu trong X
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)

    def _predict(self, x):
        """
        Hàm dự đoán nhãn cho một mẫu x duy nhất.
        x: Mẫu cần dự đoán.
        """
        posteriors = []  # Danh sách chứa xác suất hậu nghiệm (posterior) của mỗi lớp

        # Tính toán xác suất hậu nghiệm cho mỗi lớp
        for idx, c in enumerate(self._classes):
            prior = np.log(self._priors[idx])  # Tính log của prior (xác suất lớp)
            # Tính log của hàm mật độ xác suất (PDF) cho từng đặc trưng trong mẫu x
            posterior = np.sum(np.log(self._pdf(idx, x)))  
            posterior = prior + posterior  # Tổng log(prior) và log(PDF) cho mỗi lớp
            posteriors.append(posterior)

        # Trả về lớp có xác suất hậu nghiệm cao nhất
        return self._classes[np.argmax(posteriors)]

    def _pdf(self, class_idx, x):
        """
        Hàm tính toán hàm mật độ xác suất (PDF) cho một lớp cụ thể.
        class_idx: Chỉ số lớp.
        x: Mẫu cần tính xác suất.
        """
        mean = self._mean[class_idx]                        # Lấy trung bình của lớp class_idx
        var = self._var[class_idx]                          # Lấy phương sai của lớp class_idx
        # Tính toán hàm mật độ xác suất (Gaussian PDF)
        numerator = np.exp(-((x - mean) ** 2) / (2 * var))  # Tử số trong công thức Gaussian
        denominator = np.sqrt(2 * np.pi * var)              # Mẫu số trong công thức Gaussian
        return numerator / denominator                      # Trả về giá trị PDF cho lớp và mẫu x
