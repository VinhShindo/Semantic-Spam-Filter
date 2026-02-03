import pandas as pd
import random
import re
import nltk
from nltk.corpus import wordnet
from googletrans import Translator

nltk.download("wordnet", quiet=True)

random.seed(42)
translator = Translator()

# ==============================
# 1. TEXT NORMALIZATION
# ==============================
def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ==============================
# 2. SAFE CONFIG
# ==============================
SAFE_WORDS = {
    "you", "your", "we", "us", "me", "please", "kindly"
}

SPAM_KEYWORDS = {
    "win", "won", "free", "prize", "reward",
    "claim", "urgent", "offer", "cash"
}


# ==============================
# 3. AUGMENTATION METHODS
# ==============================

def synonym_replacement(text, n=1):
    words = text.split()
    candidates = [
        w for w in words[1:-1]
        if w.isalpha()
        and w.lower() not in SAFE_WORDS
        and w.lower() not in SPAM_KEYWORDS
    ]

    if not candidates:
        return text

    word = random.choice(candidates)
    syns = wordnet.synsets(word)

    if syns:
        synonym = syns[0].lemmas()[0].name().replace("_", " ")
        words = [synonym if w == word else w for w in words]

    return " ".join(words)


def back_translate(text):
    try:
        vi = translator.translate(text, src="en", dest="vi").text
        en = translator.translate(vi, src="vi", dest="en").text
        return en
    except:
        return text


def slight_noise(text):
    """
    Thêm nhiễu nhỏ: thay đổi thứ tự nhẹ hoặc chèn từ phụ
    Giữ ngữ nghĩa nhưng làm embedding đa dạng hơn
    """
    words = text.split()
    if len(words) > 5 and random.random() < 0.3:
        i = random.randint(1, len(words)-2)
        words[i], words[i+1] = words[i+1], words[i]
    return " ".join(words)


# Weighted strategy (không dùng template thuần)
AUGMENT_METHODS = [
    (back_translate, 0.5),
    (synonym_replacement, 0.3),
    (slight_noise, 0.2),
]


def augment_spam(text):
    funcs, weights = zip(*AUGMENT_METHODS)
    func = random.choices(funcs, weights=weights, k=1)[0]
    return func(text)


# ==============================
# 4. MAIN BALANCE FUNCTION
# ==============================

def balance_for_knn(
    df,
    text_col="sms",
    label_col="label",
    target_spam_count=None,
    target_ratio=0.4,
    max_attempts=50000
):
    """
    Oversample spam (label=1) để đạt số lượng mong muốn.
    
    - Nếu truyền target_spam_count → ưu tiên dùng số này
    - Nếu không → spam = target_ratio * ham
    - Không bao giờ giảm dữ liệu
    """

    df = df.copy()

    spam_df = df[df[label_col] == 1]
    ham_df = df[df[label_col] == 0]

    ham_count = len(ham_df)
    spam_count = len(spam_df)

    # -------------------------
    # Xác định target spam
    # -------------------------
    if target_spam_count is None:
        target_spam_count = int(ham_count * target_ratio)

    if spam_count >= target_spam_count:
        print("Spam đã đạt hoặc vượt target. Không cần cân bằng.")
        return df.sample(frac=1, random_state=42).reset_index(drop=True)

    to_generate = target_spam_count - spam_count

    print("===== OVERSAMPLING SPAM =====")
    print(f"Ham  : {ham_count}")
    print(f"Spam : {spam_count}")
    print(f"Target Spam : {target_spam_count}")
    print(f"Generating  : {to_generate}")
    print("================================")

    existing_texts = set(
        normalize_text(t) for t in spam_df[text_col]
    )

    generated_texts = set()
    new_rows = []
    attempts = 0

    while len(new_rows) < to_generate and attempts < max_attempts:
        attempts += 1

        base_text = spam_df.sample(1)[text_col].values[0]
        new_text = augment_spam(base_text)
        norm_text = normalize_text(new_text)

        if norm_text in existing_texts or norm_text in generated_texts:
            continue

        generated_texts.add(norm_text)

        new_rows.append({
            text_col: new_text,
            label_col: 1
        })

    print(f"Generated {len(new_rows)} samples after {attempts} attempts")

    df_balanced = pd.concat(
        [df, pd.DataFrame(new_rows)],
        ignore_index=True
    )

    df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

    print("===== AFTER BALANCE =====")
    print(df_balanced[label_col].value_counts())

    return df_balanced