import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse

def visualize_results(location_index, data_path, result_path, win_size):
    """
    Hàm chính để tải, xử lý, và trực quan hóa kết quả phát hiện bất thường (chỉ dùng Matplotlib).
    """
    print("--- Bắt đầu quá trình trực quan hóa ---")

    # --- Bước 1: Xác định và kiểm tra các tệp cần thiết ---
    data_file = os.path.join(data_path, 'test.csv')
    score_file = os.path.join(result_path, 'anomaly_scores.npy')
    threshold_file = os.path.join(result_path, 'threshold.txt')

    required_files = [data_file, score_file, threshold_file]
    if not all(os.path.exists(f) for f in required_files):
        print("\n!!! LỖI: Thiếu tệp dữ liệu hoặc kết quả cần thiết.")
        for f in required_files:
            print(f"  - {f} {'(Tồn tại)' if os.path.exists(f) else '(KHÔNG TÌM THẤY)'}")
        print("\n>>> Gợi ý: Bạn đã chạy 'main.py' ở mode='test' và lưu threshold dạng .npy chưa?")
        return

    # --- Bước 2: Tải dữ liệu ---
    print("Đang tải dữ liệu và kết quả...")
    
    # Logic tải dữ liệu linh hoạt
    try:
        # Thử đọc theo định dạng đơn biến của bạn (date, value)
        test_df_full = pd.read_csv(data_file, skiprows=1, header=None, names=['date', 'value'], parse_dates=['date'], dayfirst=True)
        original_header = pd.read_csv(data_file, nrows=0).columns
        feature_name_from_header = original_header[1] if len(original_header) > 1 else "Feature"
    except Exception:
        # Nếu không được, đọc như file đa biến (chỉ có số)
        test_df_full = pd.read_csv(data_file)
        # Tạo index thời gian giả định
        test_df_full['date'] = pd.to_datetime('2023-01-01') + pd.to_timedelta(test_df_full.index, unit='d')
        feature_name_from_header = test_df_full.columns[location_index]
        
    test_df_full.set_index('date', inplace=True)

    # Tải kết quả từ mô hình
    anomaly_scores = np.load(score_file)
    with open(threshold_file, 'r') as f:
        threshold = float(f.read())
    
    # --- Bước 3: Xử lý và Căn chỉnh dữ liệu ---
    try:
        location_id = feature_name_from_header
        if 'value' in test_df_full.columns:
             location_series = test_df_full['value']
        else:
            location_series = test_df_full.iloc[:, location_index]
    except IndexError:
        print(f"!!! LỖI: Chỉ số địa điểm {location_index} không hợp lệ.")
        return

    alignment_offset = len(location_series) - len(anomaly_scores)
    if alignment_offset < 0:
        anomaly_scores = anomaly_scores[-len(location_series):]
    else:
        location_series = location_series.iloc[alignment_offset:]

    predicted_anomalies_indices = np.where(anomaly_scores > threshold)[0]
    anomaly_dates = location_series.index[predicted_anomalies_indices]
    anomaly_values = location_series.iloc[predicted_anomalies_indices]

    print(f"\n--- Phân tích cho địa điểm: {location_id} (chỉ số {location_index}) ---")
    print(f" - Ngưỡng phát hiện: {threshold:.4f}")
    print(f" - Tìm thấy {len(anomaly_dates)} ngày được dự đoán là bất thường.")

    # --- Bước 4: Trực quan hóa bằng Matplotlib ---
    print("Đang vẽ biểu đồ...")
    
    # Tạo một figure và một axes object, đây là cách làm chuẩn của Matplotlib
    fig, ax = plt.subplots(figsize=(20, 8))

    # Vẽ chuỗi thời gian gốc
    ax.plot(location_series.index, location_series.values, color='dodgerblue', linewidth=1.5, label='Lượt xem thực tế')
    
    # Đánh dấu các điểm bất thường bằng chấm đỏ
    ax.scatter(anomaly_dates, anomaly_values, color='red', s=80, zorder=5, label='Ngày bất thường')

    # Định dạng biểu đồ
    ax.set_title(f'Phát hiện Bất thường theo Ngày cho Địa điểm: {location_id}', fontsize=18)
    ax.set_xlabel('Ngày', fontsize=14)
    ax.set_ylabel('Số lượt xem (View Count)', fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.7) # Thêm lưới cho dễ nhìn
    
    fig.autofmt_xdate() # Tự động xoay ngày tháng cho đẹp
    fig.tight_layout()
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Trực quan hóa kết quả DCdetector.")
    
    parser.add_argument('--dataset', type=str, default='MyData',
                        help="Tên của dataset bạn muốn trực quan (ví dụ: 'MyData').")
    
    parser.add_argument('--location', type=int, default=0, 
                        help="Chỉ số của địa điểm (cột) cần phân tích (0 cho cột đầu tiên).")
                        
    parser.add_argument('--winsize', type=int, required=True,
                        help="(BẮT BUỘC) Kích thước cửa sổ (win_size) đã được sử dụng khi chạy 'test'.")

    args = parser.parse_args()

    data_directory = os.path.join('dataset', args.dataset)
    result_directory = os.path.join('result', args.dataset)

    visualize_results(
        location_index=args.location, 
        data_path=data_directory, 
        result_path=result_directory,
        win_size=args.winsize
    )