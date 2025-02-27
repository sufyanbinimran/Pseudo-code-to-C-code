import streamlit as st
import pickle
import tensorflow as tf
import numpy as np
import os
from tensorflow import keras
from keras.saving import register_keras_serializable

# Register Transformer class if it's custom-defined
@register_keras_serializable()
class Transformer(keras.Model):
    def __init__(self, vocab_size, seq_length, embed_dim=256, num_heads=8, ff_dim=512, **kwargs):
        super(Transformer, self).__init__(**kwargs)
        self.vocab_size = vocab_size
        self.seq_length = seq_length
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim

        self.embedding = keras.layers.Embedding(vocab_size, embed_dim, mask_zero=True)
        self.pos_encoding = keras.layers.Embedding(seq_length, embed_dim)
        self.encoder = keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential([
            keras.layers.Dense(ff_dim, activation='relu'),
            keras.layers.Dense(embed_dim)
        ])
        self.layernorm1 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.output_layer = keras.layers.Dense(vocab_size, activation="softmax")

    def call(self, inputs):
        x = self.embedding(inputs) + self.pos_encoding(tf.range(inputs.shape[1]))
        attn_output = self.encoder(x, x, x)
        out1 = self.layernorm1(x + attn_output)
        ffn_output = self.ffn(out1)
        out2 = self.layernorm2(out1 + ffn_output)
        return self.output_layer(out2)

    def get_config(self):
        config = super(Transformer, self).get_config()
        config.update({
            "vocab_size": self.vocab_size,
            "seq_length": self.seq_length,
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

# Load tokenizers
@st.cache_resource
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
@st.cache_resource
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

# Function to generate C++ code
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
