import pandas as pd
import numpy as np
import joblib
import time
import re
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
import seaborn as sns
from balance_data import balance_for_knn

# --- PHẦN 1: KNN TÙY CHỈNH ---
class CustomKNN:
    def __init__(self, k=5):
        self.k = k

    def fit(self, X, y):
        # L2 normalize để cosine = dot product
        self.X_train = normalize(np.array(X))
        self.y_train = np.array(y)

    def predict(self, X_test):
        X_test = normalize(np.array(X_test))

        # Cosine similarity = dot product vì đã normalize
        similarities = np.dot(X_test, self.X_train.T)

        # Lấy k điểm có similarity cao nhất
        k_indices = np.argsort(similarities, axis=1)[:, -self.k:]

        predictions = []
        for i in range(len(X_test)):
            k_labels = self.y_train[k_indices[i]]
            most_common = Counter(k_labels).most_common(1)[0][0]
            predictions.append(most_common)

        return np.array(predictions)

# --- PHẦN 2: HÀM ĐÁNH GIÁ & VẼ ĐỒ THỊ ---
def visualize_results(y_test, y_pred, name, filename="image/confusion_matrix.png"):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', 
                xticklabels=['Dự đoán Ham', 'Dự đoán Spam'],
                yticklabels=['Thực tế Ham', 'Thực tế Spam'])
    plt.title(f'Ma trận nhầm lẫn - {name}')
    plt.savefig(filename)
    print(f"--- Đã lưu ảnh ma trận nhầm lẫn vào: {filename} ---")
    plt.show()

def plot_elbow_curve(results_df, filename="image/knn_elbow_curve.png"):
    plt.figure(figsize=(10, 6))
    plt.plot(results_df['K_Value'], results_df['Accuracy'], marker='o', linestyle='-', color='b', label='Accuracy')
    plt.title('Đồ thị chọn K tối ưu (Elbow Method - Accuracy)')
    plt.xlabel('Giá trị K')
    plt.ylabel('Độ chính xác (Accuracy)')
    plt.xticks(results_df['K_Value'])
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.savefig(filename)
    print(f"--- Đã lưu đồ thị chọn K vào: {filename} ---")
    plt.show()

# --- PHẦN 3: LUỒNG CHẠY CHÍNH ---
def main():
    # 1. Tải dữ liệu
    input_file = 'data/processed/spam_cleaned.csv'
    print(f"--- Đang tải dữ liệu từ {input_file} ---")
    try:
        df = pd.read_csv(input_file).dropna()
    except FileNotFoundError:
        print("Lỗi: Không tìm thấy file dữ liệu sạch.")
        return

    # 2. Chia tập dữ liệu
    test_size = 0.2
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=42,
        stratify=df['label_num']
    )

    print("\n--- Phân phối train trước cân bằng ---")
    print(train_df['label_num'].value_counts())

    # 3. KIỂM TRA MẤT CÂN BẰNG
    ham_count = sum(train_df['label_num'] == 0)
    spam_count = sum(train_df['label_num'] == 1)

    imbalance_ratio = spam_count / ham_count

    print(f"\nSpam/Ham ratio (train): {imbalance_ratio:.3f}")

    # Nếu spam < 30% ham thì coi là mất cân bằng mạnh
    if imbalance_ratio < 0.3:
        print("\n⚠ Dataset mất cân bằng mạnh → Thực hiện cân bằng...")
        train_df = balance_for_knn(
            train_df,
            text_col="cleaned_text",
            label_col="label_num",
            target_ratio=0.5   # spam ≈ 50% ham
        )
    else:
        print("\nDataset đủ cân bằng → Không cần xử lý.")

    print("\n--- Phân phối train sau xử lý ---")
    print(train_df['label_num'].value_counts())

    # 5. VECTOR HÓA SAU CÂN BẰNG
    # ==========================================

    print("\n--- Đang tải model Embedding và tạo Vector ---")
    encoder = SentenceTransformer('all-MiniLM-L6-v2')

    X_train_embeddings = encoder.encode(
        train_df['cleaned_text'].tolist(),
        show_progress_bar=True
    )

    X_test_embeddings = encoder.encode(
        test_df['cleaned_text'].tolist(),
        show_progress_bar=True
    )

    y_train = train_df['label_num'].values
    y_test = test_df['label_num'].values

    X_train = X_train_embeddings
    X_test = X_test_embeddings

    # 4. CHẠY HUẤN LUYỆN VÀ TỐI ƯU HÓA K (2-10)
    optimization_results = []
    best_acc = 0
    best_model = None
    
    print(f"\n--- Bắt đầu thử nghiệm K từ 2 đến 10 ---")
    for k in range(2, 11):
        start_time = time.time()
        
        model_knn = CustomKNN(k=k)
        model_knn.fit(X_train, y_train)
        y_pred = model_knn.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        duration = time.time() - start_time
        
        optimization_results.append({
            'K_Value': k,
            'Accuracy': acc,
            'F1_Score': f1,
            'Train_Time_Sec': round(duration, 2)
        })
        
        print(f"K={k} hoàn tất | Accuracy: {acc:.4f} | Time: {duration:.2f}s")
        
        if acc > best_acc:
            best_acc = acc
            best_model = model_knn

    # 5. GHI THÔNG SỐ VÀ VẼ ĐỒ THỊ ELBOW
    results_df = pd.DataFrame(optimization_results)
    output_xlsx = 'log/knn_optimization_balanced_log.xlsx'
    results_df.to_excel(output_xlsx, index=False)
    plot_elbow_curve(results_df, filename="image/knn_elbow_curve.png")

    # 6. ĐÁNH GIÁ CHI TIẾT MODEL TỐT NHẤT
    y_pred_best = best_model.predict(X_test)
    print("\n" + "="*45)
    print(f"KẾT QUẢ ĐÁNH GIÁ TRÊN TẬP TEST (Best K={best_model.k})")
    print(classification_report(y_test, y_pred_best, target_names=['Ham', 'Spam']))

    # 7. Dự đoán thực tế
    sample_real = df.sample(3, random_state=10)
    real_texts = sample_real['cleaned_text'].tolist()
    real_labels = sample_real['label_num'].tolist()
    real_embeddings = encoder.encode(real_texts)
    real_preds = best_model.predict(real_embeddings)

    for i in range(len(real_texts)):
        actual = "Spam" if real_labels[i] == 1 else "Ham"
        predicted = "Spam" if real_preds[i] == 1 else "Ham"
        print(f"Mẫu {i+1}: Thực tế: {actual} | Dự đoán: {predicted}")

    # 8. LƯU CÁC FILE BỔ SUNG CHO WEB APP (PHẦN MỚI BỔ SUNG)
    print("\n--- Đang lưu trữ dữ liệu bổ trợ cho Dashboard và Explainable AI ---")
    
    # Lưu vector train và metadata để tính khoảng cách láng giềng trên Web
    joblib.dump(X_train, 'model/X_train_vectors.pkl')
    train_metadata = {
        'text': train_df['cleaned_text'].values,
        'label': y_train
    }
    joblib.dump(train_metadata, 'model/train_metadata.pkl')

    # Trích xuất từ khóa Spam phổ biến (Dùng CountVectorizer)
    spam_texts = df[df['label_num'] == 1]['cleaned_text']
    cv = CountVectorizer(stop_words='english', max_features=50) # Lấy top 50 từ khóa rác
    cv.fit(spam_texts)
    spam_keywords = cv.get_feature_names_out().tolist()

    # Thống kê tổng quan cho Dashboard
    data_stats = {
        'total_samples': len(df),
        'spam_count': int(sum(df['label_num'] == 1)),
        'ham_count': int(sum(df['label_num'] == 0)),
        'spam_keywords': spam_keywords
    }
    joblib.dump(data_stats, 'model/data_stats.pkl')

    # Lưu Model chính và Encoder
    joblib.dump(best_model, 'model/custom_knn_balanced_best_model.pkl')
    joblib.dump(encoder, 'model/sentence_encoder.pkl')
    
    visualize_results(y_test, y_pred_best, f"KNN (Best K={best_model.k})", filename="image/best_confusion_matrix.png")
    print("\nQuá trình hoàn tất. Tất cả các file đã sẵn sàng cho Web App.")

if __name__ == "__main__":
    main()