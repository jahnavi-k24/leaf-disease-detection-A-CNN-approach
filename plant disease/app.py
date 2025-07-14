# Imporiting Necessary Libraries
import streamlit as st
from PIL import Image
import io
import numpy as np
import tensorflow as tf
from utils import clean_image, get_prediction, make_results
st.set_page_config(page_title="Plant Disease Detection", page_icon=":seedling:")


# Loading the Model and saving to cache
# Loading the Model and saving to cache
@st.cache_resource
def load_model(path):
    
    # Xception Model
    xception_model = tf.keras.models.Sequential([
    tf.keras.applications.xception.Xception(include_top=False, weights='imagenet', input_shape=(512, 512, 3)),
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(4,activation='softmax')
    ])


    # DenseNet Model
    densenet_model = tf.keras.models.Sequential([
        tf.keras.applications.densenet.DenseNet121(include_top=False, weights='imagenet',input_shape=(512, 512, 3)),
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(4,activation='softmax')
    ])

    # Ensembling the Models
    inputs = tf.keras.Input(shape=(512, 512, 3))

    xception_output = xception_model(inputs)
    densenet_output = densenet_model(inputs)

    outputs = tf.keras.layers.average([densenet_output, xception_output])


    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    # Loading the Weights of Model
    model.load_weights(path)
    
    return model

# Removing Menu
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 

# Loading the Model
model = load_model('model.h5')
# Title and Description
st.title('Plant Disease Detector')
st.write("Upload a leaf image and receive an instant analysis of the plant's health.")




# Setting the files that can be uploaded
uploaded_file = st.file_uploader("Choose a Image file", type=["png", "jpg"])

# If there is a uploaded file, start making prediction
# If there is an uploaded file, start making prediction
if uploaded_file is not None:
    
    # Display progress and text
    progress = st.text("Crunching Image")
    my_bar = st.progress(0)
    
    # Reading the uploaded image
    image = Image.open(io.BytesIO(uploaded_file.read()))
    st.image(np.array(Image.fromarray(
        np.array(image)).resize((700, 400), Image.Resampling.LANCZOS)), width=None)
    my_bar.progress(40)
    
    # Cleaning the image
    image = clean_image(image)
    
    # Making the predictions
    predictions, predictions_arr = get_prediction(model, image)
    my_bar.progress(70)
    
    # Making the results
    result = make_results(predictions, predictions_arr)
    
    # Removing progress bar and text after prediction done
    my_bar.progress(100)
    progress.empty()
    my_bar.empty()
    
    # Professional and detailed output message with line breaks
    output_message = (
        f"**Diagnosis:** The plant {result['status']}\n\n"
        f"**Confidence Level:** This conclusion is based on a confidence level of {result['prediction']}.\n\n"
        f"**Recommended Action:** {result['remedy']}."
    )

    # Show the results with formatting
    st.markdown(output_message)
