import streamlit as st
import tensorflow as tf
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, MultiHeadAttention, LayerNormalization, Dense
from tensorflow.keras.models import Model

# Define the Transformer class (Must match the architecture used in training)
class Transformer(Model):
    def __init__(self, vocab_size, seq_length, embed_dim=256, num_heads=8, ff_dim=512, **kwargs):
        super(Transformer, self).__init__(**kwargs)
        self.embedding = Embedding(vocab_size, embed_dim, mask_zero=True)
        self.pos_encoding = Embedding(seq_length, embed_dim)
        self.encoder = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([
            Dense(ff_dim, activation='relu'),
            Dense(embed_dim)
        ])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.output_layer = Dense(vocab_size, activation="softmax")

    def call(self, inputs):
        x = self.embedding(inputs) + self.pos_encoding(tf.range(inputs.shape[1]))
        attn_output = self.encoder(x, x, x)
        out1 = self.layernorm1(x + attn_output)
        ffn_output = self.ffn(out1)
        out2 = self.layernorm2(out1 + ffn_output)
        return self.output_layer(out2)

# Cache model and tokenizer loading for efficiency
@st.cache_resource
def load_resources():
    try:
        model = load_model("transformer_pseudo_cpp.keras", custom_objects={"Transformer": Transformer})
        with open("tokenizer_pseudo_to_cpp.pkl", "rb") as file:
            tokenizer_pseudo_to_cpp = pickle.load(file)
        with open("tokenizer_cpp.pkl", "rb") as file:
            tokenizer_cpp = pickle.load(file)
        return model, tokenizer_pseudo_to_cpp, tokenizer_cpp
    except Exception as e:
        st.error(f"Error loading model or tokenizers: {e}")
        return None, None, None

# Load resources
model, tokenizer_pseudo_to_cpp, tokenizer_cpp = load_resources()

# Function to generate C++ code
def generate_cpp(pseudocode, tokenizer_input, tokenizer_output, model, max_length=150):
    """Converts pseudocode to C++ using the trained Transformer model."""
    if not model:
        return "Error: Model not loaded."

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
