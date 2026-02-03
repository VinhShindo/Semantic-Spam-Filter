import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re

# Cấu hình phông chữ và thẩm mỹ cho biểu đồ
plt.rcParams['font.family'] = 'DejaVu Sans'
sns.set_theme(style="whitegrid")

def load_data(file_path):
    """Tải dữ liệu và tiền xử lý cơ bản."""
    try:
        df = pd.read_csv(file_path, encoding='latin-1')
        # Kiểm tra và đổi tên cột dựa trên cấu trúc thực tế của bạn
        if 'v1' in df.columns:
            df = df.rename(columns={'v1': 'label', 'v2': 'text'})
        elif 'sms' in df.columns:
            df = df.rename(columns={'sms': 'text', 'label': 'label'})
            
        df = df[['label', 'text']]
        df['text'] = df['text'].astype(str)
        return df
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file '{file_path}'")
        return None

def get_general_info(df):
    """Xác định quy mô, tên biến và kiểu dữ liệu."""
    print("\n--- 1. THÔNG TIN TỔNG QUAN ---")
    print(f"Số lượng quan sát: {df.shape[0]}")
    print(f"Số lượng biến: {df.shape[1]}")
    print("\nChi tiết kiểu dữ liệu:")
    print(df.info())

def analyze_quality(df):
    """Kiểm tra giá trị thiếu, lặp và dữ liệu không hợp lệ."""
    print("\n--- 2. KIỂM TRA CHẤT LƯỢNG DỮ LIỆU ---")
    print(f"Giá trị khuyết thiếu:\n{df.isnull().sum()}")
    
    duplicates = df.duplicated().sum()
    print(f"Số lượng hàng bị lặp lại: {duplicates} ({duplicates/len(df)*100:.2f}%)")
    
    empty_msgs = df[df['text'].str.strip() == ""].shape[0]
    print(f"Số lượng tin nhắn rỗng: {empty_msgs}")
    return duplicates

def clean_data_step(df):
    """Làm sạch sơ bộ để tránh sai số khi tính tương quan."""
    print("\n--- 3. LÀM SẠCH DỮ LIỆU (LOẠI BỎ TRÙNG LẶP) ---")
    df = df.drop_duplicates(keep='first').copy()
    print(f"Kích thước dữ liệu sau khi làm sạch: {df.shape}")
    return df

def analyze_statistics(df):
    """Thống kê mô tả cho biến số và biến phân loại."""
    # Tạo biến số: Độ dài tin nhắn
    df['length'] = df['text'].apply(len)
    
    print("\n--- 4. THỐNG KÊ MÔ TẢ ---")
    print("\n[Biến Phân Loại - Label]")
    counts = df['label'].value_counts()
    percent = df['label'].value_counts(normalize=True) * 100
    stat_df = pd.DataFrame({'Số lượng': counts, 'Tỷ lệ (%)': percent})
    print(stat_df)
    
    print("\n[Biến Số - Độ dài tin nhắn (Length)]")
    print(df['length'].describe())
    return df

def analyze_correlation(df):
    """Tính toán tương quan giữa độ dài và nhãn."""
    print("\n--- 5. PHÂN TÍCH TƯƠNG QUAN ---")
    
    # KIỂM TRA VÀ CHUẨN HÓA NHÃN:
    # Nếu label đã là số (0, 1), ta giữ nguyên. Nếu là chữ, ta mới map.
    if df['label'].dtype == 'object':
        df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})
    else:
        df['label_num'] = df['label'] # Đã là số sẵn rồi
    
    # Chuyển đổi chắc chắn về kiểu numeric để tránh lỗi NaN
    df['label_num'] = pd.to_numeric(df['label_num'], errors='coerce')
    
    corr_matrix = df[['length', 'label_num']].corr()
    print("Ma trận tương quan giữa Độ dài và Nhãn:")
    print(corr_matrix)

def visualize_target_relationship(df):
    """Phân tích mối quan hệ giữa độ dài và nhãn bằng biểu đồ."""
    print("\n--- 6. ĐANG TẠO BIỂU ĐỒ ---")
    plt.figure(figsize=(15, 5))

    # Biểu đồ 1: Phân phối nhãn
    plt.subplot(1, 3, 1)
    label_counts = df['label'].value_counts()
    plt.pie(label_counts, labels=label_counts.index, autopct='%1.1f%%', colors=['#66b3ff','#ff9999'])
    plt.title('Tỷ lệ nhãn')

    # Biểu đồ 2: Boxplot - Phân tích Outliers và Tứ phân vị
    plt.subplot(1, 3, 2)
    sns.boxplot(x='label', y='length', data=df, palette='Set2')
    plt.title('Phân bổ độ dài theo nhãn (Boxplot)')

    # Biểu đồ 3: Histogram - Mật độ phân phối
    plt.subplot(1, 3, 3)
    sns.histplot(data=df, x='length', hue='label', bins=50, kde=True, element="step")
    plt.title('Mật độ độ dài tin nhắn')

    plt.tight_layout()
    plt.show()

def main():
    # file_name = 'data/raw/spam.csv'
    file_name = 'data/raw/spam.csv'
    df = load_data(file_name)
    
    if df is not None:
        # Bước 1: Tổng quan
        get_general_info(df)
        
        # Bước 2: Chất lượng
        analyze_quality(df)
        
        # Bước 3: Làm sạch (Rất quan trọng để tránh nhiễu tương quan)
        # df = clean_data_step(df)
        
        # Bước 4: Thống kê mô tả
        df = analyze_statistics(df)
        
        # Bước 5: Phân tích tương quan (Đã sửa lỗi NaN)
        analyze_correlation(df)
        
        # Bước 6: Trực quan hóa
        visualize_target_relationship(df)

        print("\n=== QUÁ TRÌNH EDA HOÀN TẤT ===")

if __name__ == "__main__":
    main()