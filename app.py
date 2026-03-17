import streamlit as st
import pickle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

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

model, tokenizer, encoder = load_resources()
