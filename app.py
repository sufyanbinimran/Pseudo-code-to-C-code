import streamlit as st
import pickle
import tensorflow as tf
import numpy as np
import os
from tensorflow import keras
from keras.saving import register_keras_serializable
from transformer import Transformer  # Import Transformer class

# Register Transformer for Keras serialization
register_keras_serializable()(Transformer)

# Load tokenizers
def load_tokenizers():
    try:
        with open("tokenizer_pseudo.pkl", "rb") as f:
            tokenizer_pseudo = pickle.load(f)
        with open("tokenizer_cpp.pkl", "rb") as f:
            tokenizer_cpp = pickle.load(f)
        return tokenizer_pseudo, tokenizer_cpp
    except Exception as e:
        st.error(f"Error loading tokenizers: {e}")
        return None, None

# Load model
def load_model():
    model_path = "transformer_pseudo_cpp.keras"
    if not os.path.exists(model_path):
        st.error("Model file not found. Please upload transformer_pseudo_cpp.keras")
        return None
    try:
        return keras.models.load_model(model_path)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

tokenizer_pseudo, tokenizer_cpp = load_tokenizers()
model = load_model()

# Function to preprocess input and generate output
def generate_cpp_code(pseudo_code):
    if model is None or tokenizer_pseudo is None or tokenizer_cpp is None:
        return "Error: Model or tokenizers not loaded."

    input_seq = tokenizer_pseudo.texts_to_sequences([pseudo_code])
    input_seq = tf.keras.preprocessing.sequence.pad_sequences(input_seq, maxlen=100, padding="post")

    prediction = model.predict(input_seq)
    predicted_indices = np.argmax(prediction, axis=-1)

    generated_code = tokenizer_cpp.sequences_to_texts(predicted_indices)[0]
    return generated_code

# Streamlit UI
st.title("Pseudo-Code to C++ Code Generator")

pseudo_input = st.text_area("Enter Pseudo-Code:", "")

if st.button("Generate C++ Code"):
    if pseudo_input.strip():
        cpp_code = generate_cpp_code(pseudo_input)
        st.code(cpp_code, language="cpp")
    else:
        st.warning("Please enter some pseudocode.")
