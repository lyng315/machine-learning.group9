import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from math import sqrt

# Đọc dữ liệu từ file CSV
url = "C:\\Users\\PC- MSI\\Desktop\\Machine learning\\datapower.csv"
df = pd.read_csv(url)

# Chia dữ liệu thành các biến đầu vào (X) và đầu ra (y)
X = df.drop(columns=['Power'])  # Giả sử cột 'Power' là biến phụ thuộc
y = df['Power']

# Chia dữ liệu thành tập huấn luyện (60%), tập validation (20%) và tập kiểm tra (20%)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Tiền xử lý dữ liệu
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Xây dựng mô hình hồi quy tuyến tính
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Dự đoán trên tập huấn luyện, validation và kiểm tra
y_train_pred = model.predict(X_train_scaled)
y_val_pred = model.predict(X_val_scaled)
y_test_pred = model.predict(X_test_scaled)

# Tính toán các chỉ số đánh giá cho các tập dữ liệu
metrics = {
    'Train': {
        'MAE': mean_absolute_error(y_train, y_train_pred),
        'MSE': mean_squared_error(y_train, y_train_pred),
        'R-squared': r2_score(y_train, y_train_pred),
        'RMSE': sqrt(mean_squared_error(y_train, y_train_pred))
    },
    'Validation': {
        'MAE': mean_absolute_error(y_val, y_val_pred),
        'MSE': mean_squared_error(y_val, y_val_pred),
        'R-squared': r2_score(y_val, y_val_pred),
        'RMSE': sqrt(mean_squared_error(y_val, y_val_pred))
    },
    'Test': {
        'MAE': mean_absolute_error(y_test, y_test_pred),
        'MSE': mean_squared_error(y_test, y_test_pred),
        'R-squared': r2_score(y_test, y_test_pred),
        'RMSE': sqrt(mean_squared_error(y_test, y_test_pred))
    }
}

# Tạo DataFrame chứa các chỉ số đánh giá
metrics_df = pd.DataFrame(metrics).T
print("\nBảng các chỉ số đánh giá trên 3 tập:")
print(metrics_df)

# Vẽ biểu đồ phân tán sai số (residuals)
plt.figure(figsize=(15, 5))

# Biểu đồ sai số trên tập huấn luyện
plt.subplot(1, 3, 1)
plt.scatter(y_train_pred, y_train - y_train_pred, color='blue')
plt.axhline(y=0, color='red', linestyle='--')
plt.title('Sai số trên tập Train')
plt.xlabel('Giá trị dự đoán')
plt.ylabel('Sai số')

# Biểu đồ sai số trên tập validation
plt.subplot(1, 3, 2)
plt.scatter(y_val_pred, y_val - y_val_pred, color='green')
plt.axhline(y=0, color='red', linestyle='--')
plt.title('Sai số trên tập Validation')
plt.xlabel('Giá trị dự đoán')

# Biểu đồ sai số trên tập kiểm tra
plt.subplot(1, 3, 3)
plt.scatter(y_test_pred, y_test - y_test_pred, color='orange')
plt.axhline(y=0, color='red', linestyle='--')
plt.title('Sai số trên tập Test')
plt.xlabel('Giá trị dự đoán')

plt.tight_layout()
plt.show()

# Tính toán loss function (MSE) trên từng tập
def calculate_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# Loss function sử dụng SGD
def sgd_loss(X, y, iterations=90, learning_rate=0.003):
    n_samples, n_features = X.shape
    w = np.zeros(n_features)
    losses = []
    
    for i in range(iterations):
        # Tính toán dự đoán
        predictions = X.dot(w)
        
        # Tính toán lỗi và gradient
        errors = predictions - y
        gradient = 2 * X.T.dot(errors) / n_samples
        
        # Cập nhật trọng số
        w -= learning_rate * gradient
        
        # Tính toán và lưu giá trị loss
        loss = calculate_loss(y, predictions)
        losses.append(loss)
    
    return losses

# Chạy SGD và tính toán loss function cho từng tập
iterations = 90
learning_rate = 0.0025

train_losses = sgd_loss(X_train_scaled, y_train, iterations, learning_rate)
val_losses = sgd_loss(X_val_scaled, y_val, iterations, learning_rate)
test_losses = sgd_loss(X_test_scaled, y_test, iterations, learning_rate)

# Vẽ đồ thị loss function theo số lần lặp
plt.figure(figsize=(10, 6))
plt.plot(range(iterations), train_losses, color='blue', label='Train Loss')
plt.plot(range(iterations), val_losses, color='green', label='Validation Loss')
plt.plot(range(iterations), test_losses, color='orange', label='Test Loss')
plt.xlabel('Số lần lặp')
plt.ylabel('Giá trị Loss')
plt.title('Đồ thị Loss Function theo số lần lặp')
plt.legend()
plt.grid(True)
plt.show()


# Nhập thông tin từ người dùng
def user_input():
    columns = ['Temp_2m', 'RelHum_2m', 'DP_2m', 'WS_10m', 'WS_100m', 'WD_10m', 'WD_100m', 'WG_10m']
    user_data = {}
    
    for column in columns:
        user_data[column] = [float(input(f"Nhập {column}: "))]

    return pd.DataFrame(user_data)

user_data = user_input()

# Chuẩn hóa dữ liệu đầu vào từ người dùng
user_data_scaled = scaler.transform(user_data)

# Dự đoán sản lượng điện dựa trên thông tin đã nhập
user_prediction = model.predict(user_data_scaled)

# In kết quả dự đoán
print(f"\nKết quả dự đoán sản lượng điện (Power) là: {user_prediction[0]:.2f}")
