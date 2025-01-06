import streamlit as st
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from model import load_model_and_tokenizer
from predictor import predict_emotion

# Streamlit app
st.set_page_config(page_title="Mental Health Predictor", layout="wide")
st.markdown("""
    <h2 style='text-align: center;'>\U0001F64F Mental Health Conditions Predictor</h2>
""", unsafe_allow_html=True)


# Sidebar
st.sidebar.title("About the App")
st.sidebar.info("""
This app leverages deep learning to classify statements into one of the following categories:
- Anxiety
- Bipolar
- Depression
- Normal
- Personality Disorder
- Stress
- Suicidal
""")
st.sidebar.info("Model trained on custom data for educational purposes.")

# Load model and tokenizer
MODEL_PATH = './model.h5'
TOKENIZER_PATH = './tokenizer.pkl'
model, tokenizer = load_model_and_tokenizer(MODEL_PATH, TOKENIZER_PATH)

# User input
st.markdown("""<h3 style='text-align: center;'>\U0001F4DD Enter Your Text</h3>""",unsafe_allow_html=True)
text = st.text_area("Type your statement below:", height=200)

if st.button('Predict Emotion'):
    if text.strip():
        with st.spinner("Analyzing..."):
            labels, probs = predict_emotion(text, model, tokenizer)

        # Highlight the top emotion
        top_emotion_idx = np.argmax(probs)
        percent = round(probs[top_emotion_idx]*100)
        st.markdown(f"### Top Emotion: **{labels[top_emotion_idx]}** ({percent}%)")
            

        # Display results
        st.subheader("Prediction Results")
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.barh(labels, probs, color='skyblue')
        ax.bar_label(bars, fmt='%.2f')
        ax.set_xlabel("Probability")
        ax.set_ylabel("Emotion")
        ax.set_title("Predicted Probabilities")
        st.pyplot(fig)


    else:
        st.error("Please enter some text to analyze.")

# Footer
st.markdown("""
    <div style='text-align: center;'>
        ---
        Developed by Aziz
        ---
    </div>
""", unsafe_allow_html=True)
