import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Đọc dữ liệu từ file CSV hoặc DataFrame
df = pd.read_csv("C:\\Users\\BXT\\Downloads\\Train.csv\\Train.csv")






# Chuyển đổi cột 'Time' sang kiểu datetime
df['Time'] = pd.to_datetime(df['Time'], format='%d-%m-%Y %H:%M')


# Kiểm tra các giá trị null
print(df.isnull().sum())

# Kiểm tra các kiểu dữ liệu
print(df.dtypes)


# Chuyển đổi cột 'Time' sang kiểu datetime
df['Time'] = pd.to_datetime(df['Time'], format='%d-%m-%Y %H:%M')

# Loại bỏ các ký tự dấu chấm thập phân và chuyển đổi các cột thành số
cols_to_convert = ['Temp_2m', 'RelHum_2m', 'DP_2m', 'WS_10m', 'WS_100m', 'WD_10m', 'WD_100m', 'WG_10m']
df[cols_to_convert] = df[cols_to_convert].replace({'[.,]': ''}, regex=True).astype(float)

# Kiểm tra lại các giá trị sau khi chuyển đổi
#print(df.head())


# Tính toán Q1 (phân vị thứ 25%) và Q3 (phân vị thứ 75%) cho mỗi cột
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1

# Loại bỏ các giá trị ngoại lệ
df_clean = df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]

# Mô tả thống kê dữ liệu
print(df.describe())




# 1. Biểu đồ phân phối (Histogram + KDE)
plt.figure(figsize=(10, 6))
sns.histplot(df['Power'], kde=True, bins=30)  # Phân phối cho cột Power
plt.title('Biểu đồ phân phối Power')
plt.xlabel('Power')
plt.ylabel('Tần suất')
plt.show()

# 2. Biểu đồ Boxplot để xác định ngoại lệ
plt.figure(figsize=(10, 6))
sns.boxplot(x=df['Power'])
plt.title('Biểu đồ hộp cho Power')
plt.show()

# 3. Tính các chỉ số thống kê chi tiết: Mean, Median, Mode, Skewness, Kurtosis
mean_power = df['Power'].mean()
median_power = df['Power'].median()
mode_power = df['Power'].mode()[0]
std_power = df['Power'].std()
skewness_power = df['Power'].skew()
kurtosis_power = df['Power'].kurtosis()

# In kết quả
print(f"Trung bình (Mean) của Power: {mean_power}")
print(f"Trung vị (Median) của Power: {median_power}")
print(f"Mode của Power: {mode_power}")
print(f"Độ lệch chuẩn (Standard Deviation) của Power: {std_power}")
print(f"Độ lệch (Skewness) của Power: {skewness_power}")
print(f"Độ nhọn (Kurtosis) của Power: {kurtosis_power}")


# Ma trận tương quan
corr_matrix = df.corr()

# Vẽ heatmap cho ma trận tương quan
plt.figure(figsize=(12,8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Ma trận tương quan giữa các biến')
plt.show()


import numpy as np
df['Power_log'] = np.log1p(df['Power'])


# Xóa các cột không cần thiết
df = df.drop(columns=['unname', 'time', 'location'])

# Lưu tệp dữ liệu đã xử lý vào một tệp CSV mới
df.to_csv("C:\\Users\\BXT\\Downloads\\datapower.csv", index=False)
