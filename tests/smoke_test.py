"""Smoke test — runs all implemented components end-to-end."""
import sys
sys.path.insert(0, '.')

import numpy as np

# ── Preprocessing ────────────────────────────────────────────────────────────
from utils.preprocessing import tokenize, normalize_text, tokenize_many, get_stopwords

assert tokenize("Hello, WORLD! Don't stop. 123") == ["hello", "world", "don", "t", "stop", "123"]
assert normalize_text("Red--BLUE!") == "red blue"
assert "i" in get_stopwords()
assert tokenize("I am happy", remove_stopwords=True) == ["happy"]
print("[OK] preprocessing")

# ── Data loader ──────────────────────────────────────────────────────────────
from utils.data_loader import load_emotion_splits

data = load_emotion_splits(seed=42)
assert (len(data.train), len(data.val), len(data.test)) == (12800, 1600, 1600)
assert data.num_classes == 6
print(f"[OK] data_loader — train/val/test: {len(data.train)}/{len(data.val)}/{len(data.test)}")
print(f"     classes: {data.class_names}")

# ── Evaluation (Part 2.3) ────────────────────────────────────────────────────
from utils.evaluation import (
    confusion_matrix_numpy,
    metrics_from_confusion_matrix_numpy,
    sklearn_evaluation,
)

y_true = np.array([0, 1, 1, 2, 2, 2, 0])
y_pred = np.array([0, 1, 0, 2, 2, 1, 0])
cm = confusion_matrix_numpy(y_true, y_pred, num_classes=3)
ref = sklearn_evaluation(y_true, y_pred, num_classes=3)
assert np.array_equal(cm, ref["confusion_matrix"]), f"CM mismatch:\n{cm}\nvs\n{ref['confusion_matrix']}"
m = metrics_from_confusion_matrix_numpy(cm)
assert np.allclose(m["macro_f1"], ref["macro_f1"], atol=1e-9)
print(f"[OK] evaluation — CM matches, macro_f1: {m['macro_f1']:.4f}")

# ── Naïve Bayes from scratch ─────────────────────────────────────────────────
train_tokens = [tokenize(t) for t in data.train.texts]
val_tokens   = [tokenize(t) for t in data.val.texts]

class _NB:
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, tokenized_texts, y):
        y = np.asarray(y, dtype=np.int64)
        self.classes_ = np.unique(y)
        K = len(self.classes_)
        word_set = {t for doc in tokenized_texts for t in doc}
        self.vocab_ = {w: i for i, w in enumerate(sorted(word_set))}
        V = len(self.vocab_)
        cc = np.bincount(y, minlength=K).astype(np.float64)
        self.class_log_prior_ = np.log(cc / len(tokenized_texts))
        tc = np.zeros((K, V), dtype=np.float64)
        for tokens, label in zip(tokenized_texts, y):
            for token in tokens:
                if token in self.vocab_:
                    tc[label, self.vocab_[token]] += 1
        tc += self.alpha
        self.feature_log_prob_ = np.log(tc / tc.sum(axis=1, keepdims=True))
        return self

    def predict(self, tokenized_texts):
        K = len(self.classes_)
        scores = np.broadcast_to(self.class_log_prior_, (len(tokenized_texts), K)).copy()
        for i, tokens in enumerate(tokenized_texts):
            for token in tokens:
                idx = self.vocab_.get(token)
                if idx is not None:
                    scores[i] += self.feature_log_prob_[:, idx]
        return self.classes_[np.argmax(scores, axis=1)]

model = _NB(alpha=1.0)
model.fit(train_tokens, data.train.labels)
preds = model.predict(val_tokens)
cm_val = confusion_matrix_numpy(data.val.labels, preds, data.num_classes)
m_val = metrics_from_confusion_matrix_numpy(cm_val)
print(f"[OK] NB scratch — val macro-F1: {m_val['macro_f1']:.4f}")

print("\nAll smoke tests passed!")
