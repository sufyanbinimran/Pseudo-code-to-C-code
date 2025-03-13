import streamlit as st
import tensorflow as tf
import pickle
import numpy as np

@st.cache_resource
def load_models():
    model1 = tf.keras.models.load_model("transformer_model1.keras")
    model2 = tf.keras.models.load_model("transformer_model2.keras")
    return model1, model2

@st.cache_resource
def load_tokenizers():
    with open("cpp_tokenizer1.pkl", "rb") as f:
        cpp_tokenizer1 = pickle.load(f)
    with open("pseudocode_tokenizer1.pkl", "rb") as f:
        pseudocode_tokenizer1 = pickle.load(f)
    with open("cpp_tokenizer2.pkl", "rb") as f:
        cpp_tokenizer2 = pickle.load(f)
    with open("pseudocode_tokenizer2.pkl", "rb") as f:
        pseudocode_tokenizer2 = pickle.load(f)
    return cpp_tokenizer1, pseudocode_tokenizer1, cpp_tokenizer2, pseudocode_tokenizer2

model1, model2 = load_models()
cpp_tokenizer1, pseudocode_tokenizer1, cpp_tokenizer2, pseudocode_tokenizer2 = load_tokenizers()

def convert_text(input_text, model, input_tokenizer, output_tokenizer):
    input_seq = input_tokenizer.texts_to_sequences([input_text])
    input_seq = tf.keras.preprocessing.sequence.pad_sequences(input_seq, maxlen=100, padding='post')
    prediction = model.predict(input_seq)
    predicted_seq = np.argmax(prediction, axis=-1)
    output_text = output_tokenizer.sequences_to_texts(predicted_seq)
    return output_text[0]

st.title("Code Converter (Transformer Model)")
option = st.selectbox("Select conversion type:", ["C++ to Pseudocode", "Pseudocode to C++"])
input_text = st.text_area("Enter your code:")

if st.button("Convert"):
    if option == "C++ to Pseudocode":
        output_text = convert_text(input_text, model1, cpp_tokenizer1, pseudocode_tokenizer1)
    else:
        output_text = convert_text(input_text, model2, pseudocode_tokenizer2, cpp_tokenizer2)
    st.text_area("Converted Code:", value=output_text, height=200)
