import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, Callback
import matplotlib.pyplot as plt

# Đọc dữ liệu từ tệp CSV
df = pd.read_csv(r'datapower.csv')

# Tách dữ liệu thành đầu vào (X) và đầu ra (y)
X = df[['Temp_2m', 'RelHum_2m', 'DP_2m', 'WS_10m', 'WD_10m', 'WD_100m']]
y = df['Power']

# Chia dữ liệu thành tập huấn luyện, kiểm tra, và xác thực
X_train_full, X_temp, y_train_full, y_temp = train_test_split(X, y, test_size=0.30, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42)

# Chuẩn hóa dữ liệu
scaler = StandardScaler()
X_train_full = scaler.fit_transform(X_train_full)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# Định nghĩa mô hình
model = Sequential([
    Dense(128, input_dim=X_train_full.shape[1], activation='relu'),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(1)
])

# Biên dịch và huấn luyện mô hình
model.compile(optimizer='adam', loss='mean_squared_error')
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Callback tùy chỉnh để lưu trữ test loss
class TestLossHistory(Callback):
    def __init__(self, X_test, y_test):
        super().__init__()
        self.X_test = X_test
        self.y_test = y_test
        self.test_losses = []

    def on_epoch_end(self, epoch, logs=None):
        y_pred = self.model.predict(self.X_test)
        test_loss = mean_squared_error(self.y_test, y_pred)
        self.test_losses.append(test_loss)

test_loss_history = TestLossHistory(X_test, y_test)

# Huấn luyện mô hình
history = model.fit(X_train_full, y_train_full, epochs=100, batch_size=32, 
                    validation_data=(X_val, y_val), callbacks=[early_stopping, test_loss_history])

# Dự đoán và đánh giá
def evaluate_model(y_true, y_pred, label):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    print(f'Đánh giá mô hình trên tập {label}:')
    print(f'- MSE: {mse:.4f}')
    print(f'- RMSE: {rmse:.4f}')
    print(f'- MAE: {mae:.4f}')
    print(f'- R^2: {r2:.4f}')
    
  
    # Trả về kết quả
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2
    }

# Thực hiện đánh giá cho từng tập dữ liệu
train_results = evaluate_model(y_train_full, model.predict(X_train_full).flatten(), 'Train')
val_results = evaluate_model(y_val, model.predict(X_val).flatten(), 'Validation')
test_results = evaluate_model(y_test, model.predict(X_test).flatten(), 'Test')


# Vẽ biểu đồ hàm mất mát
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.plot(test_loss_history.test_losses, label='Test Loss', linestyle='--')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
