import pandas as pd  # thư viện pandas thao tác và phân tích dữ liệu dạng bảng
import numpy as np # thư viện numpy cung cấp các công cụ mạnh mẽ cho xử lý số học và toán học trên mảng và ma trận (matrix)
import random      # thư viện random cung cấp các hàm để tạo số ngẫu nhiên       
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import Ridge
from sklearn.linear_model import SGDRegressor
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Đường dẫn đến tệp CSV
data_path = r'C:\Users\admin\Downloads\datapower.csv'

# Đọc dữ liệu từ tệp CSV
try:
    df = pd.read_csv(data_path)   # đọc dl tv pandas
    print("Đọc dữ liệu thành công!")
except FileNotFoundError:   # kích hoạt nếu tệp csv ko tồn tại
    print(f"Lỗi: Tệp CSV không tồn tại ở đường dẫn: {data_path}")
    df = None
except pd.errors.EmptyDataError:
    print("Lỗi: Tệp CSV rỗng.")
    df = None
except pd.errors.ParserError:
    print("Lỗi: Không thể phân tích cú pháp tệp CSV.")
    df = None

# Kiểm tra nếu df đã được đọc thành công
if df is not None:
    # Tách dữ liệu thành đầu vào (X) và đầu ra (y)
    try:
        X = df[['Temp_2m', 'RelHum_2m', 'DP_2m', 'WS_10m','WS_100m', 'WD_10m', 'WD_100m','WG_10m']].values
        y = df['Power'].values
    except KeyError as e:
        print(f"Lỗi: Không tìm thấy cột trong dữ liệu. Chi tiết: {e}")
        X, y = None, None
    
    # Nếu việc tách dữ liệu không có lỗi, tiếp tục xử lý
    if X is not None and y is not None:
        # chia dữ liệu thành các tập huấn luyện, kiểm tra và xác thực
        def train_test_split_custom(X, y, train_size=0.7, test_size=0.15, val_size=0.15, random_state=None ):#không đặt số nn
            assert train_size + test_size + val_size == 1.0, "Tổng của train_size, test_size và val_size phải bằng 1.0"
            
            if random_state is not None: #cc gtri cụ thể
                random.seed(random_state)
                np.random.seed(random_state)
            
            X = np.array(X)   #dữ liệu X và y được chuyển đổi sang dạng numpy.ndarray
            y = np.array(y)
            
            total_samples = len(X)
            
            indices = list(range(total_samples))
            random.shuffle(indices)
            
            train_end = int(train_size * total_samples)
            test_end = train_end + int(test_size * total_samples)  #chia dữ liệu thành ba tập (train, validation, test) một cách ngẫu nhiên
            train_indices = indices[:train_end]              #theo tỷ lệ đã chỉ địnhNó đảm bảo rằng mỗi mẫu chỉ xuất hiện trong một tập 
            test_indices = indices[train_end:test_end]  #duy nhất giảm thiểu khả năng xảy ra tình trạng overfitting
            val_indices = indices[test_end:]
            
            X_train = X[train_indices]
            y_train = y[train_indices]
            X_test = X[test_indices]
            y_test = y[test_indices]
            X_val = X[val_indices]
            y_val = y[val_indices]
            
            return X_train, X_val, X_test, y_train, y_val, y_test

        # Chia dữ liệu bằng hàm tự viết
        X_train, X_val, X_test, y_train, y_val, y_test = train_test_split_custom(X, y, train_size=0.7, test_size=0.15, val_size=0.15, random_state=42)

        # Khởi tạo mô hình Ridge Regression
        ridge_model = Ridge(alpha=1.0)
        
        # Huấn luyện mô hình với dữ liệu train
        ridge_model.fit(X_train, y_train)
        
        # Đánh giá mô hình trên từng tập dữ liệu
        def evaluate_model(model, X, y, dataset_name="dataset"):
            y_pred = model.predict(X)
            mse = mean_squared_error(y, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y, y_pred)
            r2 = r2_score(y, y_pred)
            
            print(f'Đánh giá trên {dataset_name}:')
            print(f' - Mean Squared Error (MSE): {mse:.2f}')
            print(f' - Root Mean Squared Error (RMSE): {rmse:.2f}')
            print(f' - Mean Absolute Error (MAE): {mae:.2f}')
            print(f' - R^2 Score: {r2:.2f}\n')

        # Đánh giá mô hình trên từng tập
        evaluate_model(ridge_model, X_train, y_train, "tập huấn luyện (train)")
        evaluate_model(ridge_model, X_val, y_val, "tập xác thực (validation)")
        evaluate_model(ridge_model, X_test, y_test, "tập kiểm tra (test)")
        
        # Hàm dự đoán công suất dựa trên giá trị đầu vào
        def predict_power(model, input_features):
            # Dự đoán công suất dựa trên mô hình đã huấn luyện
            input_array = np.array(input_features).reshape(1, -1)
            predicted_power = model.predict(input_array)
            return predicted_power[0]

        # Nhập giá trị đầu vào từ người dùng
        input_values = []
        for feature in ['Temp_2m', 'RelHum_2m', 'DP_2m', 'WS_10m','WS_100m', 'WD_10m', 'WD_100m','WG_10m']:
            value = float(input(f'Nhập giá trị cho {feature}: '))
            input_values.append(value)

        # Dự đoán giá trị y
        predicted_power = predict_power(ridge_model, input_values)
        # Chuẩn hóa dữ liệu
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)

        # Khởi tạo mô hình Ridge Regression sử dụng SGD
        sgd_model = SGDRegressor(penalty='l2',     # Ridge Regression sử dụng L2 regularization
                                 alpha=0.01,       # Hệ số điều chuẩn
                                 learning_rate='invscaling',  # Learning rate giảm dần
                                 max_iter=1,       # Chỉ chạy một lần mỗi vòng lặp
                                 tol=None,         # Không dừng sớm
                                 eta0=0.01,        # Tốc độ học ban đầu
                                 warm_start=True,  # Tiếp tục từ các trọng số hiện tại
                                 random_state=42)
        
        n_epochs = 100
        train_losses = []
        val_losses = []
        test_losses = []
        
        # Huấn luyện mô hình qua nhiều epoch
        for epoch in range(n_epochs):
            sgd_model.fit(X_train, y_train)  # Huấn luyện một lần cho mỗi epoch
            
            # Dự đoán trên tập huấn luyện, xác thực và kiểm tra
            y_train_pred = sgd_model.predict(X_train)
            y_val_pred = sgd_model.predict(X_val)
            y_test_pred = sgd_model.predict(X_test)
            
            # Tính toán MSE cho mỗi epoch
            train_loss = mean_squared_error(y_train, y_train_pred)
            val_loss = mean_squared_error(y_val, y_val_pred)
            test_loss = mean_squared_error(y_test, y_test_pred)
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            test_losses.append(test_loss)
            
            # In ra thông tin mỗi 10 epoch
            if (epoch + 1) % 10 == 0:
                print(f'Epoch {epoch + 1}/{n_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Test Loss: {test_loss:.4f}')
        
        # Vẽ đồ thị loss function theo từng epoch với đường nối thay vì các điểm
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, n_epochs + 1), train_losses, label='Train Loss', linestyle='-', marker=None)  # Dùng đường liền
        plt.plot(range(1, n_epochs + 1), val_losses, label='Validation Loss', linestyle='-', marker=None)  # Dùng đường liền
        plt.plot(range(1, n_epochs + 1), test_losses, label='Test Loss', linestyle='-', marker=None)  # Dùng đường liền
        plt.xlabel('Epoch')
        plt.ylabel('Mean Squared Error (MSE)')
        plt.title('Loss Function vs. Epochs for Ridge Regression using SGD')
        plt.legend()
        plt.grid(True)
        plt.show()

        # In ra giá trị dự đoán
        print(f'Giá trị dự đoán cho Power là: {predicted_power:.2f}')
    else:
        print("Không thể tách dữ liệu, vui lòng kiểm tra lại cấu trúc tệp CSV.")
else:
    print("Không thể đọc dữ liệu, vui lòng kiểm tra lại đường dẫn hoặc tệp CSV.")
      
