import streamlit as st
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def predict_emotion(text, model, tokenizer):
    sequences = tokenizer.texts_to_sequences([text])
    x = pad_sequences(sequences, maxlen=5359)
    prediction = model.predict([x, x])

    emotions = {0: 'Anxiety', 1: 'Bipolar', 2: 'Depression', 3: 'Normal',
                4: 'Personality disorder', 5: 'Stress', 6: 'Suicidal'}

    labels = list(emotions.values())
    probs = list(prediction[0])

    return labels, probs
