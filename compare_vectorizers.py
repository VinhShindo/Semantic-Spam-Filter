import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sentence_transformers import SentenceTransformer


def evaluate_vectorizers_and_models():
    # =====================
    # Load data
    # =====================
    df = pd.read_csv('data/processed/spam_cleaned.csv').dropna()
    texts = df['cleaned_text'].tolist()
    y = df['label_num']

    # =====================
    # Vectorizers
    # =====================
    vectorizers = {
        'BoW': CountVectorizer(),
        'TF-IDF': TfidfVectorizer(),
        'Sentence Embedding': SentenceTransformer('all-MiniLM-L6-v2')
    }

    # =====================
    # Models
    # =====================
    models = {
        'Naive Bayes': GaussianNB(),
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'Decision Tree': DecisionTreeClassifier(
            max_depth=None,
            random_state=42
        )
    }

    results = []

    # =====================
    # Evaluation loop
    # =====================
    for vec_name, vec in vectorizers.items():
        print(f"\n===== Vectorizer: {vec_name} =====")

        if vec_name == 'Sentence Embedding':
            X = vec.encode(texts, show_progress_bar=True)
        else:
            X = vec.fit_transform(texts).toarray()

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        for model_name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)

            results.append({
                'Vectorizer': vec_name,
                'Model': model_name,
                'Accuracy': acc
            })

            print(f"{model_name:<15}: Accuracy = {acc:.4f}")

    # =====================
    # Results dataframe
    # =====================
    results_df = pd.DataFrame(results)
    print("\n===== BẢNG SO SÁNH KẾT QUẢ =====")
    print(results_df)

    # =====================
    # Visualization
    # =====================
    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=results_df,
        x='Vectorizer',
        y='Accuracy',
        hue='Model'
    )
    plt.title("So sánh Vector hóa & Mô hình phân loại")
    plt.ylim(0, 1)
    plt.legend(title="Model")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    evaluate_vectorizers_and_models()