import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle

# Manually redefine Transformer class to match the trained model
class Transformer(tf.keras.Model):
    def __init__(self, vocab_size, seq_length, embed_dim=256, num_heads=8, ff_dim=512, **kwargs):
        super(Transformer, self).__init__(**kwargs)
        self.embedding = tf.keras.layers.Embedding(vocab_size, embed_dim, mask_zero=True)
        self.pos_encoding = tf.keras.layers.Embedding(seq_length, embed_dim)
        self.encoder = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(ff_dim, activation='relu'),
            tf.keras.layers.Dense(embed_dim)
        ])
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.output_layer = tf.keras.layers.Dense(vocab_size, activation="softmax")

    def call(self, inputs):
        x = self.embedding(inputs) + self.pos_encoding(tf.range(inputs.shape[1]))
        attn_output = self.encoder(x, x, x)
        out1 = self.layernorm1(x + attn_output)
        ffn_output = self.ffn(out1)
        out2 = self.layernorm2(out1 + ffn_output)
        return self.output_layer(out2)

# Define function to load model and tokenizers
@st.cache_resource
def load_resources():
    try:
        model = load_model(
            "transformer_pseudo_cpp.keras",
            custom_objects={"Transformer": Transformer}  # Ensures correct deserialization
        )
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

if model is None:
    st.error("Failed to load the model. Check logs for details.")
else:
    st.success("Model loaded successfully!")

# Example user input
user_input = st.text_area("Enter your pseudocode:")
if st.button("Convert to C++"):
    if user_input and model is not None:
        # Tokenize input & predict
        input_seq = tokenizer_pseudo_to_cpp.texts_to_sequences([user_input])
        input_padded = tf.keras.preprocessing.sequence.pad_sequences(input_seq, maxlen=150)
        prediction = model.predict(input_padded)
        
        # Convert prediction to text
        generated_code = tokenizer_cpp.sequences_to_texts([tf.argmax(prediction, axis=-1).numpy()[0]])
        st.code(generated_code[0], language="cpp")
    else:
        st.warning("Please enter pseudocode to convert.")
