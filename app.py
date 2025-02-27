import streamlit as st
import tensorflow as tf
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load model and tokenizers
@st.cache_resource
def load_resources():
    model = load_model("transformer_pseudo_cpp.keras", custom_objects={"Transformer": Transformer})
    
    with open("tokenizer_pseudo_to_cpp.pkl", "rb") as file:
        tokenizer_pseudo_to_cpp = pickle.load(file)

    with open("tokenizer_cpp.pkl", "rb") as file:
        tokenizer_cpp = pickle.load(file)
    
    return model, tokenizer_pseudo_to_cpp, tokenizer_cpp

model, tokenizer_pseudo_to_cpp, tokenizer_cpp = load_resources()

def generate_cpp(pseudocode, tokenizer_input, tokenizer_output, model, max_length=150):
    """ Converts pseudocode to C++ using the trained Transformer model. """
    input_seq = tokenizer_input.texts_to_sequences([pseudocode])
    input_seq = pad_sequences(input_seq, maxlen=max_length, padding="post")

    prediction = model.predict(input_seq)
    predicted_tokens = np.argmax(prediction, axis=-1)[0]
    generated_code = tokenizer_output.sequences_to_texts([predicted_tokens])[0]

    return generated_code

# Streamlit UI
st.title("Pseudocode to C++ Converter")
st.write("Enter pseudocode and get equivalent C++ code.")

pseudocode_input = st.text_area("Enter your pseudocode:")

if st.button("Convert"):
    if pseudocode_input.strip():
        cpp_output = generate_cpp(pseudocode_input, tokenizer_pseudo_to_cpp, tokenizer_cpp, model)
        st.subheader("Generated C++ Code:")
        st.code(cpp_output, language="cpp")
    else:
        st.warning("Please enter valid pseudocode.")
