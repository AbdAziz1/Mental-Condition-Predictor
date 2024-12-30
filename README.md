# Mental Condition Predictor

This repository contains a machine learning project designed to classify text statements into various mental health conditions using deep learning techniques. The project leverages Python and TensorFlow to create a dual-branch Convolutional Neural Network (CNN) model, enabling robust and accurate classification.

## Features
* Preprocessing Pipeline: Text tokenization, sequence padding, and label encoding for preparing the dataset.
* Deep Learning Model: A dual-branch CNN architecture with Conv1D layers, Batch Normalization, and Dropout for regularization.
* Visualization: Real-time visualization of prediction probabilities and confusion matrix for analysis.
* Streamlit App: An interactive and user-friendly web application for emotion detection in text.
* Export and Reusability: Saved model and tokenizer for deployment or further development.

## Emotion Categories
The model predicts the following mental health conditions:

* Anxiety
* Bipolar
* Depression
* Normal
* Personality Disorder
* Stress
* Suicidal

## Getting Started
1. Clone the repository:
```
git clone https://github.com/AbdAziz1/Mental-Condition-Predictor.git
```
2. Install the required dependencies:
```
pip install -r requirements.txt
```
3. Run the Streamlit app:
```
streamlit run app.py
```

## Technologies Used
* Python: Core programming language
* TensorFlow/Keras: Model building and training
* Streamlit: Web application framework
* Matplotlib & Seaborn: Visualization tools
* Scikit-learn: Data preprocessing and metrics

## Dataset
https://www.kaggle.com/code/sahityasetu/sentiment-analysis-for-mental-health-monitoring/input
