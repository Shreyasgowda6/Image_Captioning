import streamlit as st
import os
import pandas as pd
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from transformers import BlipProcessor, BlipForConditionalGeneration

# Function to load the BLIP model and processor
def load_blip_model(model_dir):
    processor = BlipProcessor.from_pretrained(model_dir)
    model = BlipForConditionalGeneration.from_pretrained(model_dir)
    return processor, model

# Function to load the CNN-RNN model
def load_cnn_rnn_model(model_path):
    model = load_model(model_path)
    return model

# Function to get actual caption from CSV
def get_actual_caption(image_name, csv_file):
    if csv_file and os.path.exists(csv_file):
        data = pd.read_csv(csv_file)
        caption = data[data['name'] == image_name]['caption'].values
        if len(caption) > 0:
            return caption[0]
    return None

# Function to decode CNN-RNN model output into a human-readable caption
def decode_cnn_rnn_output(output, tokenizer):
    predicted_indices = tf.argmax(output, axis=-1).numpy()[0]
    predicted_caption = tokenizer.sequences_to_texts([predicted_indices])[0]
    return predicted_caption

# Streamlit app
st.title("Image Captioning On Medical Images")
st.write("Upload an image and get predicted captions.")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Path to CSV file containing actual captions
csv_file = st.text_input("Path to CSV file containing actual captions", "D:\\Internship\\1\\train\\radiologytraindata.csv")

# Paths to saved models
blip_model_path = r'D:\Internship\1\saved_models\blip-image-captioning'
cnn_rnn_model_path = r'D:\Internship\1\saved_models\cnn_rnn_model.h5'

# Load tokenizer for decoding CNN-RNN output
with open(r'D:\Internship\1\saved_models\tokenizer.json') as f:
    tokenizer_json = f.read()
tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(tokenizer_json)

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Get the image name
    image_name = uploaded_file.name

    # Get the actual caption if CSV file is provided
    actual_caption = get_actual_caption(image_name, csv_file)

    # Load BLIP model and generate predicted caption
    blip_processor, blip_model = load_blip_model(blip_model_path)
    inputs = blip_processor(images=image, return_tensors="pt")
    blip_out = blip_model.generate(**inputs)
    predicted_caption_blip = blip_processor.decode(blip_out[0], skip_special_tokens=True)

    # Load CNN-RNN model and generate predicted caption
    cnn_rnn_model = load_cnn_rnn_model(cnn_rnn_model_path)
    processed_image = tf.image.resize(image, (224, 224)) / 255.0
    processed_image = tf.expand_dims(processed_image, axis=0)
    cnn_rnn_out = cnn_rnn_model.predict([processed_image, tf.zeros((1, 99))])  # Example input for text (all zeros)

    # Decode CNN-RNN output
    predicted_caption_cnn_rnn = decode_cnn_rnn_output(cnn_rnn_out, tokenizer)

    # Display captions
    if actual_caption:
        st.write(f"**Actual Caption:** {actual_caption}")
    st.write(f"**Predicted Caption (BLIP):** {predicted_caption_blip}")
    st.write(f"**Predicted Caption (CNN-RNN):** {predicted_caption_cnn_rnn}")
    




# pip install git+https://github.com/huggingface/transformers
# pip install transformers --upgrade