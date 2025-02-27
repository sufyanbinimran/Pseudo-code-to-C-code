import streamlit as st
import tensorflow as tf
import numpy as np
import pickle
from tensorflow.keras.utils import register_keras_serializable  # ✅ Correct Import

# Load Tokenizers
with open("tokenizer_pseudo.pkl", "rb") as f:
    tokenizer_pseudo = pickle.load(f)

with open("tokenizer_cpp.pkl", "rb") as f:
    tokenizer_cpp = pickle.load(f)

# Register Transformer Model
@register_keras_serializable()
class Transformer(tf.keras.Model):
    def __init__(self, vocab_size, seq_length, embed_dim=256, num_heads=8, ff_dim=512):
        super(Transformer, self).__init__()
        self.embedding = tf.keras.layers.Embedding(vocab_size, embed_dim, mask_zero=True)
        self.pos_encoding = tf.keras.layers.Embedding(seq_length, embed_dim)
        self.encoder = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(ff_dim, activation='relu'),
            tf.keras.layers.Dense(embed_dim)
        ])
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.output_layer = tf.keras.layers.Dense(vocab_size)

    def call(self, inputs):
        x = self.embedding(inputs) + self.pos_encoding(tf.range(inputs.shape[1]))
        attn_output = self.encoder(x, x)
        out1 = self.layernorm1(x + attn_output)
        ffn_output = self.ffn(out1)
        out2 = self.layernorm2(out1 + ffn_output)
        return self.output_layer(out2)

# Load Model
try:
    transformer_model = tf.keras.models.load_model("transformer_pseudo_cpp.keras", custom_objects={"Transformer": Transformer})
    st.success("Model Loaded Successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")

# Streamlit UI
st.title("Pseudocode to C++ Code Generator")

user_input = st.text_area("Enter Pseudocode:")

if st.button("Generate Code"):
    if user_input:
        # Tokenize and pad input
        input_seq = tokenizer_pseudo.texts_to_sequences([user_input])
        input_seq = tf.keras.preprocessing.sequence.pad_sequences(input_seq, maxlen=150, padding="post")

        # Generate Prediction
        prediction = transformer_model.predict(input_seq)
        predicted_indices = np.argmax(prediction, axis=-1)

        # Decode C++ Code
        cpp_code = " ".join([tokenizer_cpp.index_word.get(idx, "") for idx in predicted_indices[0]])

        st.subheader("Generated C++ Code:")
        st.code(cpp_code, language="cpp")
    else:
        st.warning("Please enter some pseudocode.")
