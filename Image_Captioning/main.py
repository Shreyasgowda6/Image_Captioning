import streamlit as st
from PIL import Image
import base64
import numpy as np
import pandas as pd
import requests
import torch
from transformers import AutoProcessor, AutoModelForCausalLM, AutoModelForSeq2SeqLM
from transformers import BlipProcessor, BlipForConditionalGeneration


@st.cache_resource
def load_model_and_processor():
    processor = AutoProcessor.from_pretrained("microsoft/git-base-coco")
    model = AutoModelForCausalLM.from_pretrained("microsoft/git-base-coco")
    return processor, model


@st.cache_resource
def load_xray_model_and_processor():
    xray_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    xray_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return xray_processor, xray_model


# Define the model metrics
model_metrics = [
    {"metric": "Accuracy", "value": "85%"},
    {"metric": "Precision", "value": "82%"},
    {"metric": "Recall", "value": "88%"},
    {"metric": "F1 Score", "value": "85%"},
]

# Check if GPU is available and use it if possible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Function to predict caption
def predict_caption(image, processor, model):
    pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)
    generated_ids = model.generate(pixel_values=pixel_values, max_length=50)
    generated_caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return generated_caption

# Function to predict caption for X-ray images
def predict_xray_caption(image, processor, model):
    pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)
    generated_ids = model.generate(pixel_values=pixel_values, max_length=50)
    generated_caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return generated_caption


# Function to convert image to base64
def convert_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
    return encoded_string

# Base64 encoded images (Replace these with your actual image paths)
use_case_images_base64 = {
    "E-commerce": convert_image_to_base64("images/ecom.jpeg"),
    "Digital Marketing": convert_image_to_base64("images/Digital Marketing.jpeg"),
    "Media and Entertainment": convert_image_to_base64("images/Media and Entertainment.jpeg"),
    "Healthcare": convert_image_to_base64("images/health.jpeg"),
    "Accessibility Solutions": convert_image_to_base64("images/accessibility.jpeg"),
    "Visually Impaired": convert_image_to_base64("images/visually impaired.jpeg")
}
xrayimage = convert_image_to_base64("static/xray.jpg")
# Convert images to base64
video_icon_base64 = convert_image_to_base64("images/video icon.png")
multilingual_icon_base64 = convert_image_to_base64("images/translate.png")
creative_icon_base64 = convert_image_to_base64("images/creative icon.png")

# Paths to your images (converted to base64 for Streamlit)
infosis_logo_path = 'images/infosis_logo.png'
springboard_logo_path = 'images/springboard.png'

infosis_logo_base64 = convert_image_to_base64(infosis_logo_path)
springboard_logo_base64 = convert_image_to_base64(springboard_logo_path)

# Set the page configuration
st.set_page_config(
    page_title="Image Captioning for Visually Impaired",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
    <style>
    /* General Styling */
# body {
#     background-color: #F0F8FF; /* Lightest blue background */
#     color: #333;
#     font-family: 'Arial', sans-serif;
# }
.main {
    background-color: #04142b; /* White background for content */
    padding: 20px;
    border-radius: 12px;
    color: #333;
}
# .sidebar .sidebar-content {
#     background-color: #FFFFFF; /* White background for sidebar */
#     padding: 15px;
#     border-radius: 12px;
#     color: #333;
# }
.home-container {
    display: flex;
    justify-content: center;
    align-items: center;
    height: 50vh; /* Set height to full viewport height */

    }

.full-screen-container {
    display: flex;
    justify-content: center;
    align-items: center;
    box-sizing: border-box;
    overflow: hidden; /* Ensure no scrollbars appear */
    margin: 0; /* Remove any default margin */
}

}
.stButton button {
    background-color: #00BFFF !important; /* Deep Sky Blue */
    color: white;
    border: none;
    padding: 12px 28px;
    text-align: center;
    font-size: 16px;
    border-radius: 8px;
    margin: 6px 3px;
    cursor: pointer;
    transition: background-color 0.3s, transform 0.3s;
}
.stButton button:hover {
    background-color: #1E90FF; /* Dodger Blue */
    transform: scale(1.05);
}
.stFileUploader label {
    font-size: 16px;
    font-weight: bold;
    color: #00BFFF; /* Deep Sky Blue */
}
.header {
    display: flex;
    justify-content: center;
    align-items: center;
}

.header img {
    width: 150px;
    height: auto;
}
.title {
    font-size: 36px;
    font-weight: bold;
    text-align: center;
    color: #00BFFF; /* Deep Sky Blue */
}
.section {
    background-color: #87CEFA; /* Light Sky Blue */
    padding: 20px;
    border-radius: 12px;
    margin-bottom: 20px;
    box-shadow: 0px 6px 12px rgba(0,0,0,0.1);
}
.section h1, .section h2, .section p, .section li {
    color: #000000; /* White text color */
}
.circle-frame {
    border-radius: 50%;
    overflow: hidden;
    width: 180px;
    height: 180px;
    display: inline-block;
    clip-path: circle();
    margin: 0 20px; /* Add space between logo and title */
    max-width: 100%; /* Ensure the image fits within the frame */
    max-height: 100%; /* Ensure the image fits within the frame */
    flex-shrink: 0.3; /* Prevent the circle from shrinking */
}                                 

.circle-frame img {
    width: 100%;
    height: 100%;
    object-fit: cover;
    max-width: 100%; /* Ensure the image fits within the frame */
    max-height: 100%; /* Ensure the image fits within the frame */
}
}
.mentor p {
    color: #FFFFFF;
    font-size: 18px;
    text-align: center;
}
.section-content {
    background-color: #E0FFFF; /* Light Cyan */
    padding: 15px;
    border-radius: 8px;
    margin-top: 10px;
    box-shadow: 0px 4px 8px rgba(0,0,0,0.1);
}
.section-content h2 {
    font-size: 24px;
    color: #00BFFF; /* Deep Sky Blue */
    margin-bottom: 10px;
}
.st-expander {
    background-color: #00BFFF; /* Deep Sky Blue */
    padding: 12px;
    border-radius: 15px;
    margin-bottom: 12px;
}
.expander-container {
    background-color: #FFFFFF; /* White background for boxes */
    padding: 15px;
    border-radius: 12px;
    box-shadow: 0px 6px 12px rgba(0,0,0,0.1);
    color: #333;
    margin-bottom: 12px; /* Space between expanders */
}
.expander-header {
    font-size: 24px;
    font-weight: bold;
    color: #00BFFF; /* Deep Sky Blue */
    margin-bottom: 12px;
    text-align: center;
}
.expander-header img {
    height: 200px; /* Adjusted height for images */
    border-radius: 10px;
    object-fit: cover;
    display: block;
    margin: 0 auto 12px auto; /* Centered with margin */
}
.expander-content {
    font-size: 16px;
}
.code-explanation-container {
    display: flex;
    flex-direction: column;
    gap: 24px;
}
.code-step {
    background-color: #E0FFFF; /* Light Cyan */
    padding: 10px;
    padding-top:0px;
    padding-bottom:0px;
    border-radius: 12px;
    box-shadow: 0px 6px 12px rgba(0,0,0,0.1);
}
.code-step h2 {
    color: #00BFFF; /* Deep Sky Blue */
    margin-bottom: 4px;
}

/* Image Captioning Section */
.caption-section {
    background-color: #F0FFFF; /* Azure */
    padding: 20px;
    border-radius: 12px;
    margin-top: 20px;
    box-shadow: 0px 6px 12px rgba(0,0,0,0.2);
}
.caption-section h1 {
    color: #00CED1; /* Dark Turquoise */
    font-size: 32px;
    text-align: center;
    margin-bottom: 15px;
}
.caption-section .upload-section {
    background-color: #FFFFFF; /* White background */
    padding: 20px;
    border-radius: 12px;
    border: 2px solid #00CED1; /* Dark Turquoise border */
}
.caption-section .upload-section img {
    border-radius: 8px;
    box-shadow: 0px 6px 12px rgba(0,0,0,0.3);
    display: block;
    margin: 0 auto;
}
.caption-section .stFileUploader label {
    color: #00CED1; /* Dark Turquoise */
}
.caption-section .stButton {
        display: flex;
        justify-content: center;
        margin-top: 20px;
    }
.caption-section .stButton button {
    background-color: #00CED1; /* Dark Turquoise */
    color: white;
    border: none;
    padding: 14px 28px;
    font-size: 16px;
    border-radius: 12px;
    transition: background-color 0.3s, transform 0.3s;
}
.caption-section .stButton button:hover {
    background-color: #20B2AA; /* Light Sea Green */
    transform: scale(1.05);
}
.caption-section .generated-caption {
        text-align: left;
        align:center;
        color: #333;
        margin-top: 20px;
        overflow: hidden; /* Hide overflow text */
        text-overflow: ellipsis; /* Add ellipsis for overflow text */
    }
    .generated-caption {
        text-align: left;
        # font-size: 20px;
        margin-top: 20px;
            overflow: hidden; /* Hide overflow text */
        text-overflow: clip; /* Clip the text without ellipsis */
    }

.streamlit-expanderHeader {
        color: white !important;
    }
    .generated-caption {
    color: #FFFFFF;
}
.blue-button {
        background-color: #20B2AA;
        color: white;
        border: none;
        padding: 12px 28px;
        text-align: center;
        font-size: 16px;
        border-radius: 8px;
        margin: 6px 3px;
        cursor: pointer;
        transition: background-color 0.3s, transform 0.3s;
    }
    .blue-button:hover {
        background-color: #00BFFF; /* Deep Sky Blue */
        transform: scale(1.05);
    }

    </style>
""", unsafe_allow_html=True)

# Sidebar navigation
st.sidebar.title("Navigation")
section = st.sidebar.radio("Go to",
                           ["Home", "About Project", "Business Use Cases", "Code Explanation", "Image Captioning","Conclusion"])

if section == "Home":
    st.markdown(f"""
    <div class="full-screen-container home-container">
        <div class="header">
            <div class="circle-frame">
                <img src="data:image/png;base64,{infosis_logo_base64}" alt="Infosis Logo">
            </div>
            <div class="title">Image Captioning for Visually Impaired</div>
            <div class="circle-frame">
                <img src="data:image/png;base64,{springboard_logo_base64}" alt="Springboard Logo">
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)





elif section == "About Project":
    st.markdown("""
    <div class="section">
        <h1>About the Project</h1>
        <p>The Image Captioning project focuses on generating descriptive captions for images using advanced deep learning models. This technology aims to convert visual content into meaningful textual descriptions, making it accessible to visually impaired individuals.</p>
        <p>By leveraging pre-trained models, the system provides a user-friendly interface for uploading images and receiving automated captions. This tool can be applied in various domains such as e-commerce, digital marketing, media, healthcare, and accessibility solutions, enhancing the overall user experience and inclusivity.</p>
    </div>
    """, unsafe_allow_html=True)
elif section == "Business Use Cases":
    st.markdown("""
    <div class="section">
        <h1>Business Use Cases</h1>
    </div>
    """, unsafe_allow_html=True)

    use_cases = {
        "Visually Impaired": """
            <ul>
                <li><strong>Descriptive Captions:</strong> Generate detailed and accurate captions for images to help visually impaired individuals understand visual content.</li>
                <li><strong>Accessibility:</strong> Enhance accessibility by providing text descriptions for images, making digital content more inclusive.</li>
                <li><strong>Assistive Technology:</strong> Integrate with assistive technologies like screen readers to provide real-time image descriptions.</li>
            </ul>
        """,
        "Healthcare": """
            <ul>
                <li><strong>X-ray Image Captioning:</strong> Provide detailed descriptions of X-ray images to support radiologists and other healthcare professionals in their analyses.</li>
                <li><strong>Telemedicine:</strong> Enhance telemedicine platforms by providing detailed descriptions of visual data shared between patients and doctors.</li>
                <li><strong>Patient Education:</strong> Help patients understand medical images and reports through clear and concise captions.</li>
            </ul>
        """,
        "Digital Marketing": """
            <ul>
                <li><strong>Engaging Content:</strong> Generate engaging content with accurate and descriptive captions for images used in marketing campaigns.</li>
                <li><strong>SEO Improvement:</strong> Well-described images improve search engine optimization (SEO), leading to better search engine rankings.</li>
                <li><strong>Social Media Marketing:</strong> Create compelling captions for images posted on social media to attract more engagement and shares.</li>
            </ul>
        """,
        "Media and Entertainment": """
            <ul>
                <li><strong>Content Creation:</strong> Assist content creators by automatically generating descriptions and tags for images and videos.</li>
                <li><strong>Video Summarization:</strong> Use image captioning to summarize video content by creating descriptions for key frames.</li>
                <li><strong>Enhanced User Experience:</strong> Provide better user experiences by offering detailed descriptions of visual content.</li>
            </ul>
        """,
        "Accessibility Solutions": """
            <ul>
                <li><strong>Assistive Technology:</strong> Use image captioning to create assistive technologies for visually impaired individuals, enhancing their ability to understand visual content.</li>
                <li><strong>Inclusive Content:</strong> Ensure that digital content is inclusive and accessible to everyone by providing descriptive captions for images and videos.</li>
                <li><strong>Compliance:</strong> Meet legal and regulatory requirements for accessibility by providing descriptive text for all visual content.</li>
            </ul>
        """
    }

    # Display use cases in expanders
    for use_case, details in use_cases.items():
        with st.expander(f":blue[{use_case}]"):
            st.markdown(f"""
            <div class="expander-container">
                <div class="expander-header">
                    {use_case}
                    <img src="data:image/jpeg;base64,{use_case_images_base64[use_case]}" alt="{use_case}">
                </div>
                <div class="expander-content">
                    {details}
                
            </div>
            """, unsafe_allow_html=True)


elif section == "Code Explanation":
    st.markdown("""
        <div class="section">
            <h1 style="text-align: center; font-size: 32px; color: #000000;">Code Breakdown</h1>
            <div class="code-explanation-container">
                <div class="code-step">
                    <h2>Model Building</h2>
                    <p>In this section, we explain the steps involved in building and evaluating image captioning models.</p>
                    <ol>
                        <li><strong>Divide the Dataset:</strong> The dataset is divided into three subsets:
                            <ul>
                                <li><strong>Training Set (70%):</strong> This subset is used to train the model.</li>
                                <li><strong>Test Set (20%):</strong> This subset is used to evaluate the model’s performance during training.</li>
                                <li><strong>Validation Set (10%):</strong> This subset is used to fine-tune the model and validate its performance after training.</li>
                            </ul>
                        </li>
                        <li><strong>Build Models:</strong> We build and evaluate three types of models:
                            <ul>
                                <li><strong>CNN + RNN:</strong> A Convolutional Neural Network (CNN) is used to extract features from images. These features are then fed into a Recurrent Neural Network (RNN) to generate descriptive captions.</li>
                                <li><strong>CNN + LSTM:</strong> Similar to the CNN + RNN approach, but instead of an RNN, we use a Long Short-Term Memory (LSTM) network to generate captions from the features extracted by the CNN.</li>
                                <li><strong>Pre-Trained Models:</strong> We use pre-trained models such as VGG for feature extraction. The extracted features are then used with LSTM or RNN for caption generation.
                                    <ul>
                                        <li><strong>VGG Model:</strong> We utilize the VGG16 model, a well-known CNN architecture pre-trained on ImageNet, to extract features from images. These features are then used to train an LSTM network for generating captions.</li>
                                    </ul>
                                </li>
                            </ul>
                        </li>
                        <li><strong>Evaluate Models:</strong> 
                            <ul>
                                <li>Step 1: Build the three models described above and calculate performance metrics, such as BLEU scores, on both the training and test datasets to determine their effectiveness.</li>
                                <li>Step 2: Select the best-performing model based on these metrics and use it to predict captions on the validation dataset. Report the final performance metrics for this model.</li>
                            </ul>
                        </li>
                    </ol>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

# Image Captioning Section
if section == "Image Captioning":
    st.markdown("""
    <div class="section">
        <h1>Image Captioning</h1>
        <p>Upload an image to get a caption generated by the model.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="section-content">
        <h2>Use Case 1: Image Captioning for Visually Impaired</h2>
        <p>Upload an image for image captioning for visually impaired to generate a caption.</p>
    </div>
    """, unsafe_allow_html=True)

    # File uploader for Use Case 1
    uploaded_file1 = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"], key="uploader1")
    if uploaded_file1 is not None:
        image1 = Image.open(uploaded_file1)
        st.markdown(f"""
         <div class="expander-header">
             <img src="data:image/jpeg;base64,{base64.b64encode(uploaded_file1.getvalue()).decode()}" alt="Uploaded Image">
         </div>
         """, unsafe_allow_html=True)

        # Display the button only if an image is uploaded
        if st.button("Generate Caption", key="generate_caption1"):
            with st.spinner('Generating caption...'):
                processor, model = load_model_and_processor()
                caption1 = predict_caption(image1, processor, model)
                st.markdown(f"<div class='generated-caption'><strong>Generated Caption:</strong> {caption1}</div>", unsafe_allow_html=True)


    else:
        # Optional: Display a message when no image is uploaded
        st.markdown("<p style='color: gray;'>Please upload an image to generate a caption.</p>", unsafe_allow_html=True)

    # Divider between Use Case 1 and Use Case 2
    st.markdown("<hr class='divider'>", unsafe_allow_html=True)


    # Use Case 2
    st.markdown("""
        <div class="section-content">
            <h2>Use Case 2: Healthcare Image Captioning</h2>
            <p>Upload an image related to Healthcare to generate a caption.</p>
        </div>
        """, unsafe_allow_html=True)

    st.write("## Overview")
    st.write(
        """<p style='color: gray;'>
        X-ray captioning involves generating descriptive captions for medical X-ray images to aid in
        diagnostic processes and medical reporting.
        </p>
        """, unsafe_allow_html=True
    )
    st.markdown(
        f"""
            <div >
                <img src="data:image/jpeg;base64,{xrayimage}" alt="Sample" class="h-auto mb-4 rounded-lg shadow-lg"/>
            </div>
            """,
        unsafe_allow_html=True
    )

    st.write("## Data Preprocessing")
    st.write(
        """<p style='color: gray;'>
        Data preprocessing steps involve cleaning and preparing the X-ray images and associated text
        data for model training..</p>
        """, unsafe_allow_html=True
    )

    code = '''
            # Function to preprocess text by cleaning and normalizing
            def preprocess_text(text):
                text = text.lower()
                text = re.sub(r'http\S+|www\S+|ftp\S+', '', text) # removing the links
                text = text.replace('\\n', ' ') # removing the new lines
                text = re.sub(r'\w*\d\w*', '', text) # removing the words containing numbers
                text = re.sub(r'\s+', ' ', text).strip() # removing the spaces
                text = re.sub(r'[^\w\s]', '', text) # removing special characters
                words = text.split()
                stop_words = set(stopwords.words('english'))
                words = [word for word in words if word not in stop_words] # considering only normal words
                stemmer = PorterStemmer()
                words = [stemmer.stem(word) for word in words] # considering the stemmed words
                lemmatizer = WordNetLemmatizer()
                words = [lemmatizer.lemmatize(word) for word in words] #considering the lemmatized words
                text = ' '.join(words)
                return text
        '''

    # Display the code snippet
    st.code(code, language='python')

    # Update the file path
    df = pd.read_csv("static/train_caption_df.csv")
    df1 = df.iloc[:, [1, 2]]
    st.write("### Unprocessed captions data")
    st.dataframe(df1)

    # Display the second dataframe with columns 3 and 4
    df2 = df.iloc[:, [2, 3]]
    st.write("### Processed captions data")
    st.dataframe(df2)

    st.write("## Image Preprocessing")
    st.write(
        """<p style='color: gray;'>
        Image preprocessing techniques include resizing, normalization, and augmentation to enhance
        the quality and consistency of input images for the captioning model..</p>
        """, unsafe_allow_html=True
    )

    df3 = pd.read_csv('static/test_preprocessed_df.csv')
    st.write("### Processed Images & Captions data")
    st.dataframe(df3)

    st.write("## Model Metrics")
    st.write("### Model Performance Metrics")
    st.write(
        """<p style='color: gray;'>
        The table below shows the performance metrics of the X-ray captioning model..</p>
        """, unsafe_allow_html=True
    )

    # Display metrics in a table
    st.table(model_metrics)

    # Add image captioning section
    st.write("## Image Captioning")
    st.write(
        """<p style='color: gray;'>
        Upload an image to generate a caption using the fine-tuned model..</p>
        """, unsafe_allow_html=True
    )

    # Load new CSV file containing image names and captions
    new_df = pd.read_csv("static/new_image_captions.csv")

    # Image upload
    uploaded_image = st.file_uploader('Upload an image', type=["png", "jpg", "jpeg"])

    if uploaded_image is not None:
        # Display the uploaded image using the expander-header
        st.markdown(f"""
        <div class="expander-header">
            <img src="data:image/jpeg;base64,{base64.b64encode(uploaded_image.getvalue()).decode()}" alt="Uploaded Image">
        </div>
        """, unsafe_allow_html=True)

        # Add a button to generate the caption
        if st.button("Generate Caption", key="generate_caption2"):
            # Extract the image name
            image_name = uploaded_image.name
            # Check if the image name is in the new CSV
            if image_name in new_df['image_name'].values:
                # Get the caption from the dataframe
                caption = new_df[new_df['image_name'] == image_name]['caption'].values[0]
                st.markdown(f"<div class='generated-caption'><strong>Generated Caption:</strong> {caption}</div>",
                            unsafe_allow_html=True)
            else:
                st.write("Image not found in the dataset.")

        # # Centered button for Use Case 2
        # if st.button("Generate Caption for Healthcare", key="generate_caption2"):
        #     with st.spinner('Please wait, generating caption...'):
        #         xray_processor, xray_model = load_xray_model_and_processor()
        #         caption2 = predict_xray_caption(image2, xray_processor, xray_model)
        #         st.markdown(f"<div class='generated-caption'><strong>Generated Caption:</strong> {caption2}</div>",
        #                     unsafe_allow_html=True)
    else:
        # Optional: Display a message when no image is uploaded
        st.markdown("<p style='color: gray;'>Please upload an image to generate a caption.</p>", unsafe_allow_html=True)

# Conclusion section with icons and descriptions
if section == "Conclusion":
    st.markdown("""
    <div class="section">
        <h1>Conclusion and Future Scope</h1>
        <p>Image captioning has made significant progress in identifying the context and describing the input image, but there are ongoing efforts to
improve its accuracy, fluency, and ability to capture nuanced details and context.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="section">
        <div style="display: flex; align-items: center; margin-bottom: 10px;">
            <div style="flex: 0 0 50px;">
                <img src="data:image/png;base64,{video_icon_base64}" alt="Video Icon" style="width: 50px; height: 50px;">
            </div>
            <div style="flex: 1; padding-left: 20px;">
                <h2>Video Captioning</h2>
                <p>Extending image captioning to generate descriptions for video sequences.</p>
            </div>
        </div>
        <div style="display: flex; align-items: center; margin-bottom: 10px;">
            <div style="flex: 0 0 50px;">
                <img src="data:image/png;base64,{multilingual_icon_base64}" alt="Multilingual Icon" style="width: 50px; height: 50px;">
            </div>
            <div style="flex: 1; padding-left: 20px;">
                <h2>Multilingual Captioning</h2>
                <p>Developing models that can generate captions in multiple languages.</p>
            </div>
        </div>
        <div style="display: flex; align-items: center;">
            <div style="flex: 0 0 50px;">
                <img src="data:image/png;base64,{creative_icon_base64}" alt="Creative Icon" style="width: 50px; height: 50px;">
            </div>
            <div style="flex: 1; padding-left: 20px;">
                <h2>Creative Captioning</h2>
                <p>Generating more descriptive and engaging captions that go beyond simple descriptions.</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
