import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

# Đọc dữ liệu
url = "C:\\Users\\BXT\\Downloads\\datapower.csv"
data = pd.read_csv(url)

# Giả định rằng dữ liệu có cột 'Power' là biến mục tiêu
X = data.drop(columns=['Power'])  # Các đặc trưng
y = data['Power']  # Biến mục tiêu

# Chia dữ liệu thành train, validation, và test
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Chuẩn hóa dữ liệu
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Bước 1: Huấn luyện các mô hình base
# Base-Models 1: Hồi quy tuyến tính
model_lr = LinearRegression()
model_lr.fit(X_train_scaled, y_train)

# Base-Models 2: Hồi quy Ridge
model_ridge = Ridge(alpha=1.0) 
model_ridge.fit(X_train_scaled, y_train)

# Bước 2: Dự đoán trên tập validation
y_pred_lr_val = model_lr.predict(X_val_scaled)
y_pred_ridge_val = model_ridge.predict(X_val_scaled)

# Tạo bảng meta-features cho mô hình tổng hợp
meta_features_val = np.column_stack((y_pred_lr_val, y_pred_ridge_val))

# Bước 3: Huấn luyện mô hình meta (Mạng nơ ron)
model_nn = MLPRegressor(hidden_layer_sizes=(100, 50), 
                        max_iter=1000,  
                        activation='relu', 
                        solver='adam',  
                        learning_rate_init=0.001,  
                        random_state=42)

# Huấn luyện mô hình từ bảng meta-fearute
model_nn.fit(meta_features_val, y_val)

# Dự đoán trên các tập dữ liệu
y_pred_train_meta = model_nn.predict(np.column_stack((model_lr.predict(X_train_scaled), model_ridge.predict(X_train_scaled))))
y_pred_val_meta = model_nn.predict(meta_features_val)
y_pred_test_meta = model_nn.predict(np.column_stack((model_lr.predict(X_test_scaled), model_ridge.predict(X_test_scaled))))

# Bước 4: Tính toán các chỉ số đánh giá trên từng tập
def calculate_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return mse, mae, r2

# Tính toán các chỉ số cho từng tập
metrics_train = calculate_metrics(y_train, y_pred_train_meta)
metrics_val = calculate_metrics(y_val, y_pred_val_meta)
metrics_test = calculate_metrics(y_test, y_pred_test_meta)

# Hiển thị các chỉ số đánh giá
print("Train Set Metrics:")
print(f"MSE: {metrics_train[0]:.4f}, MAE: {metrics_train[1]:.4f}, R2: {metrics_train[2]:.4f}")

print("\nValidation Set Metrics:")
print(f"MSE: {metrics_val[0]:.4f}, MAE: {metrics_val[1]:.4f}, R2: {metrics_val[2]:.4f}")

print("\nTest Set Metrics:")
print(f"MSE: {metrics_test[0]:.4f}, MAE: {metrics_test[1]:.4f}, R2: {metrics_test[2]:.4f}")

# Vẽ biểu đồ loss function cho từng tập
plt.figure(figsize=(10, 5))

# Loss cho tập train
plt.plot(range(1, len(model_nn.loss_curve_) + 1), model_nn.loss_curve_, label='Loss (Train)', color='blue')

# Title and labels
plt.title('Loss Function During Training (Neural Network)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid()
plt.show()
