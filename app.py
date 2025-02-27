import streamlit as st
import tensorflow as tf
import pickle
import numpy as np
import keras

# ✅ Register Transformer class for Keras deserialization
@keras.saving.register_keras_serializable()
class Transformer(keras.Model):
    def __init__(self, vocab_size, seq_length, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.seq_length = seq_length
        # Define model layers (Replace with your actual Transformer structure)
        self.dummy_layer = keras.layers.Dense(vocab_size, activation="softmax")

    def call(self, inputs):
        return self.dummy_layer(inputs)  # Replace with actual forward pass

    def get_config(self):
        config = super().get_config()
        config.update({"vocab_size": self.vocab_size, "seq_length": self.seq_length})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

# ✅ Load model and tokenizers
@st.cache_resource
def load_resources():
    try:
        model = tf.keras.models.load_model("transformer_pseudo_cpp.keras", custom_objects={"Transformer": Transformer})

        with open("tokenizer_pseudo_to_cpp.pkl", "rb") as file:
            tokenizer_pseudo_to_cpp = pickle.load(file)
        with open("tokenizer_cpp.pkl", "rb") as file:
            tokenizer_cpp = pickle.load(file)

        return model, tokenizer_pseudo_to_cpp, tokenizer_cpp
    except Exception as e:
        st.error(f"Error loading model or tokenizers: {e}")
        return None, None, None

# ✅ Initialize resources
model, tokenizer_pseudo_to_cpp, tokenizer_cpp = load_resources()

# ✅ Define prediction function
def predict_cpp_code(pseudo_code):
    if model is None:
        return "Error: Model not loaded."
    
    input_seq = tokenizer_pseudo_to_cpp.texts_to_sequences([pseudo_code])
    input_seq = tf.keras.preprocessing.sequence.pad_sequences(input_seq, maxlen=150, padding="post")

    prediction = model.predict(input_seq)
    predicted_seq = np.argmax(prediction, axis=-1)

    cpp_code = tokenizer_cpp.sequences_to_texts(predicted_seq)[0]
    return cpp_code

# ✅ Streamlit UI
st.title("Pseudo-to-C++ Code Converter")

pseudo_code_input = st.text_area("Enter Pseudocode:")
if st.button("Convert to C++"):
    if pseudo_code_input.strip():
        cpp_output = predict_cpp_code(pseudo_code_input)
        st.subheader("Generated C++ Code:")
        st.code(cpp_output, language="cpp")
    else:
        st.warning("Please enter some pseudocode.")

