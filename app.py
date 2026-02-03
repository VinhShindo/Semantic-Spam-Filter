from flask import Flask, request, jsonify, render_template, redirect, url_for
import joblib
import numpy as np
import re
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity

# --- PHẢI ĐỊNH NGHĨA LẠI CLASS ĐỂ JOBLIB LOAD ĐƯỢC ---
class CustomKNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = np.array(X)
        self.y_train = np.array(y)

    def _predict(self, x_test_sample):
        sims = cosine_similarity(
            [x_test_sample], self.X_train
        )[0]

        k_indices = np.argsort(sims)[-self.k:]

        k_labels = [self.y_train[i] for i in k_indices]
        most_common = Counter(k_labels).most_common(1)
        return most_common[0][0]

    def predict(self, X_test):
        X_test = np.array(X_test)
        return np.array([self._predict(x) for x in X_test])
    
app = Flask(__name__)

# --- TẢI CÁC FILE MÔ HÌNH VÀ DỮ LIỆU BỔ TRỢ ---
try:
    model_knn = joblib.load('model/custom_knn_balanced_best_model.pkl')
    model_encoder = joblib.load('model/sentence_encoder.pkl')
    X_train_vectors = joblib.load('model/X_train_vectors.pkl')
    train_metadata = joblib.load('model/train_metadata.pkl')
    data_stats = joblib.load('model/data_stats.pkl')
    print("--- Tất cả mô hình và dữ liệu đã tải thành công! ---")
except Exception as e:
    print(f"Lỗi khi tải file: {e}")
    print("Hãy đảm bảo bạn đã chạy file train.py mới nhất để tạo đủ các file .pkl")

def preprocess_text(text):
    if not isinstance(text, str): return ""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

@app.route('/')
def index():
    return render_template('index.html', stats=data_stats)

@app.route('/predict', methods=['POST'])
def predict():
    message = request.form.get('message', '')
    
    if not message: 
        return redirect(url_for('index'))

    # 1. Tiền xử lý & Vector hóa
    clean_msg = preprocess_text(message)
    vector = model_encoder.encode([clean_msg])[0]

    # 2. GIẢI THÍCH MÔ HÌNH: Tìm láng giềng và tính khoảng cách Euclid thực tế
    similarities = cosine_similarity(
        [vector], X_train_vectors
    )[0]
    
    # Lấy K láng giềng gần nhất (sử dụng giá trị K từ model đã huấn luyện)
    k_val = model_knn.k
    k_indices = np.argsort(similarities)[-k_val:]
    
    neighbor_details = []
    spam_votes = 0
    for idx in k_indices:
        label = int(train_metadata['label'][idx])
        sim_score = float(similarities[idx])
        if label == 1: 
            spam_votes += 1
        
        neighbor_details.append({
            'text': train_metadata['text'][idx][:120] + "...",
            'distance': round(sim_score, 4),
            'label': 'Spam' if label == 1 else 'Ham'
        })

    # 3. THỐNG KÊ TỪ KHÓA: Tìm từ khóa rác xuất hiện trong tin nhắn
    words_in_msg = set(clean_msg.split())
    found_keywords = [word for word in data_stats['spam_keywords'] if word in words_in_msg]

    # 4. TÍNH TOÁN KẾT QUẢ CUỐI CÙNG
    prediction_label = "Spam" if (spam_votes / k_val) >= 0.5 else "Ham"
    
    # Tính độ tin cậy dựa trên tỷ lệ phiếu bầu của láng giềng
    if prediction_label == "Spam":
        confidence = (spam_votes / k_val) * 100
    else:
        confidence = ((k_val - spam_votes) / k_val) * 100

    return render_template('index.html', 
                           prediction=prediction_label,
                           confidence=round(confidence, 1),
                           neighbors=neighbor_details,
                           keywords=found_keywords,
                           original_text=message,
                           stats=data_stats)

if __name__ == '__main__':
    app.run(debug=True)