
import streamlit as st
import pickle
import re
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load model (compile=False removes warning)
@st.cache_resource
def load_resources():
    model = load_model("lie_detector_model.h5", compile=False, safe_mode=False)
    tokenizer = pickle.load(open("tokenizer.pkl", "rb"))
    encoder = pickle.load(open("encoder.pkl", "rb"))
    return model, tokenizer, encoder
model, tokenizer, encoder = load_resources()
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

st.title("🧠 AI Lie Detector")
st.write("Enter a statement and the model will predict its truthfulness.")

user_input = st.text_area("Enter statement")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter a statement.")
    else:
        cleaned = clean_text(user_input)

        seq = tokenizer.texts_to_sequences([cleaned])
        padded = pad_sequences(seq, maxlen=50, padding="post", truncating="post")

        prediction = model.predict(padded, verbose=0)
        predicted_class = np.argmax(prediction, axis=1)[0]
        predicted_label = encoder.inverse_transform([predicted_class])[0]
        confidence = float(np.max(prediction)) * 100

        st.subheader("Prediction")
        st.success(predicted_label)
        st.write(f"Confidence: {confidence:.2f}%")
