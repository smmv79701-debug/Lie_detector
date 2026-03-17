import streamlit as st
import pickle
import re
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences

st.set_page_config(page_title="AI Lie Detector", layout="centered")
st.title("AI Lie Detector")
st.write("Enter text and check the predicted class.")

@st.cache_resource
def load_resources():
    model = Sequential([
        Embedding(10000, 128, input_shape=(50,)),
        LSTM(64),
        Dense(32, activation="relu"),
        Dense(6, activation="softmax")
    ])

    model.load_weights("model.weights.h5")

    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)

    with open("encoder.pkl", "rb") as f:
        encoder = pickle.load(f)

    return model, tokenizer, encoder

with st.spinner("Loading model..."):
    model, tokenizer, encoder = load_resources()

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\\S+", "", text)
    text = re.sub(r"[^a-zA-Z0-9\\s]", "", text)
    return text.strip()

user_input = st.text_area("Enter text here")

if st.button("Predict"):
    if not user_input.strip():
        st.warning("Please enter some text.")
    else:
        cleaned = clean_text(user_input)
        seq = tokenizer.texts_to_sequences([cleaned])
        padded = pad_sequences(seq, maxlen=50)
        pred = model.predict(padded)
        label = encoder.inverse_transform([pred.argmax(axis=1)[0]])[0]
        st.success(f"Prediction: {label}")
