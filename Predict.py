import re
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

MAX_LEN = 50

# ========================
# 1. TEXT CLEANING
# ========================
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ========================
# 2. LOAD TOKENIZER
# ========================
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# ========================
# 3. LOAD MODEL
# ========================
try:
    model = load_model("best_sentiment_model.keras")   # preferred
except:
    model = load_model("sentiment_model.h5")           # fallback

# ========================
# 4. LABEL MAPPING
# ========================
id_to_label = {0: "negative", 1: "neutral", 2: "positive"}

# ========================
# 5. PREDICT FUNCTION
# ========================
def predict_sentiment(text):
    cleaned = clean_text(text)
    seq = tokenizer.texts_to_sequences([cleaned])
    pad = pad_sequences(seq, maxlen=MAX_LEN)
    
    pred = model.predict(pad)[0]
    class_id = np.argmax(pred)
    confidence = pred[class_id]

    return id_to_label[class_id], float(confidence)

# ========================
# 6. TEST HERE
# ========================
if __name__ == "__main__":
    while True:
        text = input("\nEnter a review (or 'exit'): ")
        if text.lower() == "exit":
            break
        
        label, prob = predict_sentiment(text)
        print(f"Prediction: {label} (confidence: {prob:.2f})")
