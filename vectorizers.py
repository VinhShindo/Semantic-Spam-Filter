import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import StandardScaler
import joblib

def normalize_and_save_xlsx(
    input_csv='data/processed/spam_cleaned.csv',
    text_col='cleaned_text',
    label_col='label_num',
    embedding_model='all-MiniLM-L6-v2',
    output_prefix='spam_sentence_embedding'
):
    # 1. Load dữ liệu
    df = pd.read_csv(input_csv).dropna(subset=[text_col, label_col])
    texts = df[text_col].tolist()
    labels = df[label_col].values

    # 2. Load Sentence Transformer
    model = SentenceTransformer(embedding_model)

    print("Đang sinh Sentence Embeddings...")
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True
    )

    # 3. Chuẩn hóa embedding
    print("Đang chuẩn hóa embeddings...")
    scaler = StandardScaler()
    embeddings_norm = scaler.fit_transform(embeddings)

    # 4. Lưu numpy (dùng cho train)
    np.save(f"{output_prefix}_X.npy", embeddings_norm)
    np.save(f"{output_prefix}_y.npy", labels)
    joblib.dump(scaler, f"{output_prefix}_scaler.pkl")

    # 5. Tạo DataFrame để xuất Excel
    embed_cols = [f"emb_{i}" for i in range(embeddings_norm.shape[1])]
    df_xlsx = pd.DataFrame(embeddings_norm, columns=embed_cols)
    df_xlsx['label'] = labels

    # 6. Lưu file Excel
    xlsx_path = f"{output_prefix}.xlsx"
    df_xlsx.to_excel(xlsx_path, index=False)

    print("Hoàn tất.")
    print(f"- XLSX: {xlsx_path}")
    print(f"- Shape dữ liệu: {df_xlsx.shape}")

    return df_xlsx

if __name__ == "__main__":
    normalize_and_save_xlsx()
