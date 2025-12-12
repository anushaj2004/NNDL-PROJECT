import pandas as pd
import numpy as np
import re
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint

# =======================
# 1. LOAD CSV FILES
# =======================
pos = pd.read_csv("positive_reviews.csv")
neu = pd.read_csv("neutral_reviews.csv")
neg = pd.read_csv("negative_reviews.csv")

df = pd.concat([pos, neu, neg], ignore_index=True)

print("Dataset size:", len(df))
print(df.label.value_counts())


# =======================
# 2. CLEAN TEXT FUNCTION
# =======================
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

df["cleaned"] = df["review"].astype(str).apply(clean_text)


# =======================
# 3. LABEL ENCODING
# =======================
label_map = {"negative": 0, "neutral": 1, "positive": 2}
df["label_id"] = df["label"].map(label_map)


# =======================
# 4. TRAIN-TEST SPLIT
# =======================
X_train, X_val, y_train, y_val = train_test_split(
    df["cleaned"], df["label_id"],
    test_size=0.2,
    stratify=df["label_id"],
    random_state=42
)


# =======================
# 5. TOKENIZER
# =======================
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)

with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

X_train_seq = tokenizer.texts_to_sequences(X_train)
X_val_seq = tokenizer.texts_to_sequences(X_val)

MAX_LEN = 50
vocab_size = len(tokenizer.word_index) + 1

X_train_pad = pad_sequences(X_train_seq, maxlen=MAX_LEN)
X_val_pad = pad_sequences(X_val_seq, maxlen=MAX_LEN)


# =======================
# 6. BUILD MODEL
# =======================
model = Sequential([
    Embedding(vocab_size, 128, input_length=MAX_LEN),
    LSTM(128),
    Dropout(0.3),
    Dense(64, activation="relu"),
    Dropout(0.3),
    Dense(3, activation="softmax")
])

model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer="adam",
    metrics=["accuracy"]
)


# =======================
# 7. CHECKPOINT
# =======================
ckpt = ModelCheckpoint(
    "best_sentiment_model.keras",
    monitor="val_loss",
    save_best_only=True,
    verbose=1
)


# =======================
# 8. TRAIN
# =======================
history = model.fit(
    X_train_pad, y_train,
    validation_data=(X_val_pad, y_val),
    epochs=15,
    batch_size=4,
    callbacks=[ckpt]
)

model.save("sentiment_model.h5")
print("Training Complete!")


# =======================
# 9. CONFUSION MATRIX & CLASSIFICATION REPORT
# =======================
y_pred = model.predict(X_val_pad)
y_pred_classes = np.argmax(y_pred, axis=1)

print("\nClassification Report:")
print(classification_report(y_val, y_pred_classes, target_names=["Negative", "Neutral", "Positive"]))

cm = confusion_matrix(y_val, y_pred_classes)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=["Negative", "Neutral", "Positive"],
            yticklabels=["Negative", "Neutral", "Positive"])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


# =======================
# 10. TRAINING & VALIDATION ACCURACY CURVE
# =======================
plt.figure(figsize=(6, 5))
plt.plot(history.history['accuracy'], label="Train Accuracy")
plt.plot(history.history['val_accuracy'], label="Validation Accuracy")
plt.title("Accuracy Curve")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.grid()
plt.show()


# =======================
# 11. TRAINING & VALIDATION LOSS CURVE
# =======================
plt.figure(figsize=(6, 5))
plt.plot(history.history['loss'], label="Train Loss")
plt.plot(history.history['val_loss'], label="Validation Loss")
plt.title("Loss Curve")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.grid()
plt.show()
