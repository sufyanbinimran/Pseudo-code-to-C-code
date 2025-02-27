import streamlit as st
import pickle
import tensorflow as tf
import numpy as np

# Load tokenizers
@st.cache_resource
def load_tokenizers():
    with open("tokenizer_pseudo.pkl", "rb") as f:
        tokenizer_pseudo = pickle.load(f)
    with open("tokenizer_cpp.pkl", "rb") as f:
        tokenizer_cpp = pickle.load(f)
    return tokenizer_pseudo, tokenizer_cpp

# Load model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("transformer_pseudo_cpp.keras")

tokenizer_pseudo, tokenizer_cpp = load_tokenizers()
model = load_model()

# Function to preprocess input and generate output
def generate_cpp_code(pseudo_code):
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
    if pseudo_input:
        cpp_code = generate_cpp_code(pseudo_input)
        st.code(cpp_code, language="cpp")
    else:
        st.warning("Please enter some pseudocode.")
