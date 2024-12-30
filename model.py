import streamlit as st
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def load_model_and_tokenizer(model_path, token_path):
    with open(token_path, 'rb') as tokenizer_file:
        tokenizer = pickle.load(tokenizer_file)
    model = tf.keras.models.load_model(model_path)
    return model, tokenizer