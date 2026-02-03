import pandas as pd
import re

def preprocess_text(text):
    """Làm sạch văn bản và giữ lại đặc trưng link_token."""
    if not isinstance(text, str): 
        return ""
    # 1. Chuyển về chữ thường
    text = text.lower()
    # 2. THAY THẾ link thay vì xóa để giữ đặc trưng nhận biết Spam
    # Đây là đặc trưng quan trọng vì spam thường chứa URL
    text = re.sub(r'http\S+|www\S+|https\S+', 'link_token', text)
    # 3. Loại bỏ ký tự đặc biệt nhưng giữ lại chữ cái và khoảng trắng
    text = re.sub(r'[^a-z\s]', '', text)
    # 4. Loại bỏ khoảng trắng thừa
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def run_preprocessing(input_file='data/processed/sms_balanced.csv', output_file='data/processed/spam_balanced_cleaned.csv'):
    print(f"--- Đang bắt đầu xử lý file: {input_file} ---")
    try:
        # Tải dữ liệu với encoding latin-1 (phổ biến cho dataset spam)
        df = pd.read_csv(input_file, encoding='latin-1')
        
        # SỬA LỖI Tên cột: Kiểm tra và đổi tên linh hoạt
        # Dataset gốc thường là 'v1','v2' hoặc 'label','sms'
        mapping = {}
        if 'v1' in df.columns: mapping['v1'] = 'label'
        if 'v2' in df.columns: mapping['v2'] = 'text'
        if 'sms' in df.columns: mapping['sms'] = 'text'
        
        if mapping:
            df = df.rename(columns=mapping)
            
        # Kiểm tra lại xem đã có cột 'text' chưa để tránh lỗi KeyError
        if 'text' not in df.columns:
            print(f"Lỗi: Không tìm thấy cột chứa nội dung tin nhắn. Các cột hiện có: {list(df.columns)}")
            return

        # 1. Loại bỏ hàng trùng lặp (EDA cho thấy có 7.23% hàng trùng)
        initial_count = len(df)
        df = df.drop_duplicates()
        print(f"-> Đã loại bỏ {initial_count - len(df)} hàng trùng lặp.")
        
        # 2. Giữ lại độ dài gốc làm đặc trưng bổ sung (Tương quan 0.38 trong EDA)
        df['original_length'] = df['text'].apply(lambda x: len(str(x)))
        
        # 3. Làm sạch văn bản (Giữ lại link_token)
        print("-> Đang làm sạch văn bản...")
        df['cleaned_text'] = df['text'].apply(preprocess_text)
        
        # 4. Mã hóa nhãn (0: ham, 1: spam)
        # Hỗ trợ cả trường hợp label là chữ (ham/spam) hoặc đã là số
        if df['label'].dtype == 'object':
            df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})
        else:
            df['label_num'] = df['label']
        
        # 5. Loại bỏ các hàng có dữ liệu trống sau khi xử lý
        df = df[['label_num', 'cleaned_text', 'original_length']].dropna()
        
        # Lưu file
        df.to_csv(output_file, index=False, encoding='utf-8')
        print(f"--- Hoàn thành! Dữ liệu sạch ({len(df)} dòng) đã lưu tại: {output_file} ---")
        
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file '{input_file}'. Hãy kiểm tra lại đường dẫn.")
    except Exception as e:
        print(f"Đã xảy ra lỗi hệ thống: {e}")

if __name__ == "__main__":
    run_preprocessing()