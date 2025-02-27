import streamlit as st
import pickle
import tensorflow as tf
import numpy as np
from tensorflow import keras
from transformer import Transformer  # Ensure 'transformer.py' contains the Transformer class

# Register the Transformer class
custom_objects = {"Transformer": Transformer}

# Load Tokenizers
with open("tokenizer_cpp.pkl", "rb") as f:
    tokenizer_cpp = pickle.load(f)

with open("tokenizer_pseudo_to_cpp.pkl", "rb") as f:
    tokenizer_pseudo = pickle.load(f)

# Load Transformer Model
model = keras.models.load_model("transformer_pseudo_cpp.keras", custom_objects=custom_objects)

# Function to Convert Pseudocode to C++
def predict_cpp(pseudo_code):
    input_seq = tokenizer_pseudo.texts_to_sequences([pseudo_code])
    input_seq = tf.keras.preprocessing.sequence.pad_sequences(input_seq, maxlen=100, padding="post")
    
    prediction = model.predict(input_seq)
    predicted_tokens = np.argmax(prediction, axis=-1)
    
    cpp_code = tokenizer_cpp.sequences_to_texts(predicted_tokens)[0]
    return cpp_code

# Streamlit UI
st.title("Pseudocode to C++ Converter")
pseudo_code = st.text_area("Enter Pseudocode:")

if st.button("Convert"):
    if pseudo_code:
        cpp_code = predict_cpp(pseudo_code)
        st.code(cpp_code, language="cpp")
    else:
        st.warning("Please enter some pseudocode.")
