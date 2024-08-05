import os
import re
import string
import json
import numpy as np
import cv2
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Input, Dense, Flatten, Attention, concatenate
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import streamlit as st


# Define sequence length and max features
sequence_length = 50
max_features = 50000

# Load the vectorization layer model
vectorization_model = tf.keras.models.load_model('vectorization_layer_model')
vectorization_layer = vectorization_model.layers[1]

# Preprocessing function for text
def preprocess_text(text: str) -> np.ndarray:
    try:
        # Convert to lowercase
        text = text.lower()

        # Remove non-word characters, digits, punctuation, URLs, and HTML tags
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d', '', text)
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        text = re.sub(r'<.*?>', '', text)

        # Remove stopwords
        STOPWORDS = set(stopwords.words('english'))
        text = " ".join([word for word in text.split() if word not in STOPWORDS])

        # Stem words
        stemmer = SnowballStemmer(language='english')
        text = " ".join([stemmer.stem(word) for word in text.split()])

        # Tokenize and pad the text
        sequences = vectorization_layer([text])
        return sequences
    except UnicodeDecodeError as e:
        st.error(f"UnicodeDecodeError in preprocess_text: {e}")
        return np.zeros((1, sequence_length))  # Return a zero array in case of error

# Preprocessing function for images
def preprocess_image(image) -> np.ndarray:
    img = cv2.imdecode(np.frombuffer(image.read(), np.uint8), cv2.IMREAD_COLOR)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    return img

# Load models
visual_attention_model = load_model('C:\\Kingsley\\multimodia-hatespeech\\visual_attention_model_v2\\best_modelversion22_v4.h5')
semantic_attention_model = load_model('C:\\Kingsley\\multimodia-hatespeech\\semantic_attention_model_v2\\best_modelversion22_v4.h5')
multimodal_attention_model = load_model('C:\\Kingsley\\multimodia-hatespeech\\multimodal_attention_model_v2\\best_modelversion22_v4.h5')

# Compile models (if needed)
semantic_attention_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
visual_attention_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
multimodal_attention_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Combine predictions function
def combine_predictions(predictions_list):
    num_samples = len(predictions_list[0])
    combined_predictions = []
    for i in range(num_samples):
        sample_predictions = [model_predictions[i] for model_predictions in predictions_list]
        num_ones = np.sum(sample_predictions)
        num_zeros = len(sample_predictions) - num_ones
        combined_predictions.append(1 if num_ones > num_zeros else 0)
    return combined_predictions

# Streamlit app
st.title("Bullying Detection Model")
st.write("Upload a JPG image and enter some text to predict if it's bullying content.")

# Upload Image
uploaded_image = st.file_uploader("Choose a JPG image...", type=["jpg"])
text_input = st.text_area("Enter text here...")

if st.button("Predict"):
    if uploaded_image and text_input:
        # Preprocess the inputs
        img = preprocess_image(uploaded_image)
        txt = preprocess_text(text_input)

        # Get model predictions
        vis_preds = visual_attention_model.predict(np.array([img]))
        sem_preds = semantic_attention_model.predict(txt)
        mm_preds = multimodal_attention_model.predict([np.array([img]), txt])

        # Combine predictions using majority voting
        final_preds = combine_predictions([vis_preds, sem_preds, mm_preds])

        st.write(f"The prediction is: {'Bullying' if final_preds[0] == 1 else 'Not Bullying'}")
    else:
        st.write("Please upload a JPG image and enter text for prediction.")
