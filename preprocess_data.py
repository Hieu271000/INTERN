import pandas as pd
import numpy as np

# --- CẤU HÌNH ---
INPUT_FILE = 'my_data.csv'  
TRAIN_FILE = 'dataset/MyData/train.csv' 
TEST_FILE = 'dataset/MyData/test.csv'   
TEST_LABEL_FILE = 'dataset/MyData/test_label.csv' 
TRAIN_RATIO = 0.8  # 80% dữ liệu cho việc huấn luyện

# Đảm bảo thư mục tồn tại
import os
os.makedirs('dataset/MyData', exist_ok=True)

# --- BẮT ĐẦU XỬ LÝ ---
print("Đang đọc dữ liệu từ file:", INPUT_FILE)
df = pd.read_csv(INPUT_FILE)

# Chuyển đổi cột 'date' sang định dạng datetime để sắp xếp
df['date'] = pd.to_datetime(df['date'])

# Dùng pivot để chuyển từ định dạng dài sang rộng
print("Đang chuyển đổi dữ liệu...")
df_pivot = df.pivot(index='date', columns='placeId', values='view')

# Sắp xếp lại theo thời gian để đảm bảo tính tuần tự
df_pivot = df_pivot.sort_index()

# Xử lý các giá trị bị thiếu (NaN) nếu có, ví dụ điền bằng 0 hoặc giá trị trước đó
df_pivot = df_pivot.fillna(0) 

print("Dữ liệu sau khi chuyển đổi có shape:", df_pivot.shape)
print("Ví dụ 5 dòng đầu tiên của dữ liệu đã chuyển đổi:")
print(df_pivot.head())

# Tách dữ liệu thành tập train và test
split_index = int(len(df_pivot) * TRAIN_RATIO)
train_df = df_pivot.iloc[:split_index]
test_df = df_pivot.iloc[split_index:]

print(f"\nTách dữ liệu:")
print(f"- Tập train: {len(train_df)} dòng")
print(f"- Tập test: {len(test_df)} dòng")

# Lưu các file CSV (không bao gồm index là cột date)
print("\nĐang lưu các file train.csv và test.csv...")
train_df.to_csv(TRAIN_FILE, index=False)
test_df.to_csv(TEST_FILE, index=False)
print("Đã lưu thành công!")


# --- Bước 2: Xử lý file Nhãn (Label) ---
# Dữ liệu của bạn chưa có nhãn bất thường.
# Để code có thể chạy, chúng ta cần tạo một file test_label.csv.
# GIẢ ĐỊNH: Ban đầu chúng ta chưa biết điểm nào bất thường, nên gán tất cả là 0 (bình thường).
# SAU NÀY: Bạn cần tự xác định các điểm bất thường trong tập test và thay đổi giá trị trong file này thành 1.
print("\nĐang tạo file test_label.csv giả định...")
num_test_samples = len(test_df)
# Nhãn cần có cùng số cột với dữ liệu (mỗi địa điểm có thể có nhãn riêng)
# Hoặc có thể chỉ có 1 cột nhãn chung cho cả hệ thống. Ở đây ta tạo 1 cột chung.
dummy_labels = np.zeros((num_test_samples, 1)) 
label_df = pd.DataFrame(dummy_labels, columns=['label'])
label_df.to_csv(TEST_LABEL_FILE, index=False)
print("Đã tạo file test_label.csv với tất cả các nhãn là 0.")