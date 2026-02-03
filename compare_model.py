import pandas as pd
import time
import seaborn as sns
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score, accuracy_score

def run_model_comparison():
    # 1. Chuẩn bị dữ liệu
    df = pd.read_csv('data/processed/spam_cleaned.csv').dropna()
    print("Đang khởi tạo SentenceTransformer...")
    model_st = SentenceTransformer('all-MiniLM-L6-v2')
    
    print("Đang tạo Embeddings...")
    X = model_st.encode(df['cleaned_text'].tolist(), show_progress_bar=True)
    y = df['label_num']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 2. Định nghĩa Model
    models = {
        'Naive_Bayes': GaussianNB(),
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'Decision_Tree': DecisionTreeClassifier(max_depth=10)
    }
    
    performance_metrics = []
    model_params = []

    # 3. Huấn luyện và Thu thập thông số
    for name, clf in models.items():
        print(f"--- Đang xử lý: {name} ---")
        
        # Đo thời gian huấn luyện
        start_train = time.time()
        clf.fit(X_train, y_train)
        train_time = time.time() - start_train
        
        # Đo thời gian dự đoán (Inference time)
        start_inf = time.time()
        y_pred = clf.predict(X_test)
        inf_time_total = time.time() - start_inf
        inf_ms_per_sample = (inf_time_total / len(X_test)) * 1000 # chuyển sang ms
        
        # Tính toán các chỉ số
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        # Lưu kết quả so sánh
        performance_metrics.append({
            'Model': name,
            'Accuracy': acc,
            'F1_score': f1,
            'Train_time_sec': round(train_time, 4),
            'Inference_ms_per_sample': round(inf_ms_per_sample, 4)
        })
        
        # Lưu thông số cấu hình của model
        params = clf.get_params()
        params['Model'] = name # Thêm tên model vào để dễ phân biệt
        model_params.append(params)

    # 4. Ghi tất cả vào file Excel
    with pd.ExcelWriter('log/model_comparison_balanced_detailed.xlsx', engine='openpyxl') as writer:
        # Sheet 1: Bảng so sánh các chỉ số (Yêu cầu chính)
        df_perf = pd.DataFrame(performance_metrics)
        df_perf.to_excel(writer, sheet_name='Performance_Metrics', index=False)
        
        # Sheet 2: Thông số huấn luyện của từng model
        df_params = pd.DataFrame(model_params)
        # Đưa cột 'Model' lên đầu cho dễ nhìn
        cols = ['Model'] + [c for c in df_params.columns if c != 'Model']
        df_params[cols].to_excel(writer, sheet_name='Model_Parameters', index=False)

    print("\n" + "="*30)
    print("HOÀN TẤT! Đã lưu mọi thông tin vào file: model_comparison_detailed.xlsx")
    print(df_perf)

    # 5. Trực quan hóa nhanh
    df_perf.plot(x='Model', y=['Accuracy', 'F1_score'], kind='bar', figsize=(10,5))
    plt.title("So sánh Accuracy và F1 Score")
    plt.ylim(0.8, 1.0)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig('image/comparison_balanced_metrics.png')
    plt.show()

if __name__ == "__main__":
    run_model_comparison()