#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import pickle

# Load trained model
model = load_model("poetry_model.h5")

# Load tokenizer
with open("tokenizer.pkl", "rb") as handle:
    tokenizer = pickle.load(handle)

# Load max sequence length
with open("max_seq_length.pkl", "rb") as handle:
    max_seq_length = pickle.load(handle)

# Poetry generation function
def generate_poetry(seed_text, next_words=10, temperature=1.0):
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_seq_length, padding='pre')

        predicted_probs = model.predict(token_list, verbose=0)[0]
        predicted_probs = np.log(predicted_probs + 1e-8) / temperature
        exp_preds = np.exp(predicted_probs)
        predicted_probs = exp_preds / np.sum(exp_preds)

        predicted_index = np.random.choice(len(predicted_probs), p=predicted_probs)
        output_word = tokenizer.index_word.get(predicted_index, None)

        if not output_word or predicted_index == 0:
            continue

        seed_text += " " + output_word

    return seed_text

# Streamlit UI
st.title(" SherAI – A blend of Sher  Roman Urdu Poetry Generator 🎤✨")
st.write("Enter a phrase, and the AI will generate poetry in Roman Urdu.")

# User input for starting phrase
user_input = st.text_input("Enter a starting phrase:", "")

# User input for number of words
num_words = st.number_input("Enter the number of words to generate:", min_value=1, max_value=100, value=10)

if st.button("Generate Poetry"):
    if user_input.strip():
        poetry = generate_poetry(user_input, next_words=num_words, temperature=0.7)
        st.subheader("Generated Poetry:")
        st.write(poetry)
    else:
        st.warning("Please enter a valid phrase to start!")

