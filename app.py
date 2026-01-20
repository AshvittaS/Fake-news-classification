# app.py

import re
import pickle
import numpy as np
import gradio as gr
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import nltk

# Download stopwords (needed if running first time)
nltk.download('stopwords')
with open("preprocess.pkl", "rb") as f:
    preprocess_data = pickle.load(f)

voc_size = preprocess_data["voc_size"]
sen_len = preprocess_data["sen_len"]
stop_words = preprocess_data["stop_words"]
ps = preprocess_data["stemmer"]

# -----------------------------
# Load trained model
# -----------------------------
model = load_model("fake_news_lstm.h5")

# -----------------------------
# Preprocessing function
# -----------------------------
def preprocess_text(text):
    review = re.sub('[^a-zA-Z]', ' ', text)   # Remove special chars
    review = review.lower()                    # Lowercase
    review = review.split()
    review = [ps.stem(word) for word in review if word not in stop_words]
    review = ' '.join(review)
    
    onehot = one_hot(review, voc_size)                  # One-hot encode
    padded = pad_sequences([onehot], maxlen=sen_len, padding='pre')
    
    return padded

# -----------------------------
# Prediction function
# -----------------------------
def predict_news(text):
    processed = preprocess_text(text)
    pred_prob = model.predict(processed)[0][0]
    
    label = "🟥 FAKE NEWS" if pred_prob >= 0.5 else "🟩 REAL NEWS"
    confidence = round(pred_prob*100, 2) if pred_prob >= 0.5 else round((1-pred_prob)*100,2)
    
    return f"{label} (Confidence: {confidence}%)"

# -----------------------------
# Gradio Interface
# -----------------------------
interface = gr.Interface(
    fn=predict_news,
    inputs=gr.Textbox(lines=6, placeholder="Paste news article text here..."),
    outputs=gr.Textbox(label="Prediction"),
    title="Fake News Detection using LSTM",
    description="Enter a news article and the model will predict whether it is Fake or Real."
)

if __name__ == "__main__":
    interface.launch(share=True)   # share=True gives public link (Colab / server)
