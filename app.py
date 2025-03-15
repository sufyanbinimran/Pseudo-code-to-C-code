import tensorflow as tf
import streamlit as st
import pickle
import numpy as np

@tf.keras.utils.register_keras_serializable()
class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.depth = d_model // num_heads
        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)
        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, q, k, v):
        batch_size = tf.shape(q)[0]
        q = self.split_heads(self.wq(q), batch_size)
        k = self.split_heads(self.wk(k), batch_size)
        v = self.split_heads(self.wv(v), batch_size)
        # Scaled dot-product attention
        attn_output = tf.nn.softmax(
            tf.matmul(q, k, transpose_b=True) / tf.math.sqrt(tf.cast(self.depth, tf.float32))
        )
        attn_output = tf.matmul(attn_output, v)
        attn_output = tf.transpose(attn_output, perm=[0, 2, 1, 3])
        return self.dense(tf.reshape(attn_output, (batch_size, -1, self.d_model)))


@tf.keras.utils.register_keras_serializable()
class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(dff, activation='relu'),
            tf.keras.layers.Dense(d_model)
        ])
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def call(self, x):
        attn_output = self.attention(x, x, x)
        out1 = self.layernorm1(x + attn_output)
        ffn_output = self.ffn(out1)
        return self.layernorm2(out1 + ffn_output)


@tf.keras.utils.register_keras_serializable()
class Transformer(tf.keras.Model):
    def __init__(self, vocab_size, d_model, num_heads, dff, max_len):
        super(Transformer, self).__init__()
        # Removed deprecated input_length
        self.embedding = tf.keras.layers.Embedding(vocab_size, d_model)
        self.encoder = TransformerBlock(d_model, num_heads, dff)
        self.decoder = TransformerBlock(d_model, num_heads, dff)
        self.final_layer = tf.keras.layers.Dense(vocab_size, activation='softmax')

    def call(self, inputs):
        enc_output = self.encoder(self.embedding(inputs))
        dec_output = self.decoder(self.embedding(inputs))
        return self.final_layer(dec_output)


@st.cache_resource
def load_models():
    # Define custom objects for loading the models
    custom_objs = {
        "Transformer": Transformer,
        "TransformerBlock": TransformerBlock,
        "MultiHeadAttention": MultiHeadAttention
    }
    model1 = tf.keras.models.load_model("transformer_model1.keras", safe_mode=False, custom_objects=custom_objs)
    model2 = tf.keras.models.load_model("transformer_model2.keras", safe_mode=False, custom_objects=custom_objs)
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

# Load models and tokenizers
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
