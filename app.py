import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import gradio as gr
import pickle
import re
import nltk
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from tensorflow.keras.preprocessing.text import one_hot

# Download stopwords (IMPORTANT for Docker)
nltk.download("stopwords")

# Load model & tokenizer
model = load_model("fake_news_lstm.h5")

with open("preprocess.pkl", "rb") as f:
        data = pickle.load(f)
        voc_size = data["voc_size"]
        sen_len = data["sen_len"]
        stop_words = data["stop_words"]
        ps = data["stemmer"]

def preprocess_text(text):
    review = re.sub("[^a-zA-Z]", " ", text)
    review = review.lower().split()
    review = [ps.stem(word) for word in review if word not in stop_words]
    review = " ".join(review)
    onehot_repr = one_hot(review, voc_size)
    padded = pad_sequences([onehot_repr], maxlen=sen_len, padding="pre")
    return padded

def predict_news(text):
    processed = preprocess_text(text)
    pred = model.predict(processed)[0][0]
    if pred >= 0.5:
        return f"🟢 REAL NEWS ({pred:.2f})"
    else:
        return f"🔴 FAKE NEWS ({pred:.2f})"

# Gradio UI
interface = gr.Interface(
    fn=predict_news,
    inputs=gr.Textbox(lines=6, placeholder="Paste news article here..."),
    outputs="text",
    title="Fake News Detection using LSTM",
    description="LSTM-based Fake News Classification"
)

# IMPORTANT for Docker
interface.launch(server_name="0.0.0.0", server_port=7860)
