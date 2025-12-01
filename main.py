import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Set page configuration
st.set_page_config(
    page_title="Plant Disease Recognition",
    page_icon="🌿",
    layout="wide"
)

# Add custom CSS to improve appearance
def add_custom_styling():
    st.markdown(
        """
        <style>
        .stApp {
            background-color: #262730;  /* Light green tinted background */
        }
        .main-header {
            color: #2e7d32;
            text-align: center;
            font-size: 2.5rem;
            margin-bottom: 1.5rem;
            padding-bottom: 1rem;
            border-bottom: 2px solid #a5d6a7;
            background-color: #e8f5e9;
            padding: 1rem;
            border-radius: 8px;
        }
        .sub-header {
            font-size: 1.5rem;
            margin-bottom: 1rem;
            color: #1b5e20;
            background-color: #c8e6c9;
            padding: 0.5rem;
            border-radius: 5px;
        }
        .healthy {
            color: #2e7d32;
            font-weight: bold;
            background-color: #d7f9d7;
            padding: 0.2rem 0.5rem;
            border-radius: 4px;
        }
        .diseased {
            color: #1b5e20;
            background-color: #e8f5e9;
            font-weight: bold;
            padding: 0.2rem 0.5rem;
            border-radius: 4px;
        }
        .stButton>button {
            background-color: #43a047;
            color: #ffffff;
            font-weight: bold;
        }
        .footer {
            text-align: center;
            margin-top: 1rem;
            color: #1b5e20;
            font-size: 0.8rem;
            background-color: #e8f5e9;
            padding: 0.5rem;
            border-radius: 5px;
        }
        .recommendation-title {
            color: #1b5e20;
            background-color: #43a047;
            color: white;
            padding: 0.5rem;
            border-radius: 5px 5px 0 0;
            margin-bottom: 0;
        }
        .recommendation-content {
            color: #1b5e20;
            background-color: #e8f5e9;
            border-left: 4px solid #43a047;
            padding: 1rem;
            margin-top: 0;
            border-radius: 0 0 5px 5px;
        }
        .info-text {
            background-color: #e3f2fd;
            color: #0d47a1;
            padding: 1rem;
            border-radius: 5px;
        }
        .error-text {
            background-color: #ffebee;
            color: #c62828;
            padding: 1rem;
            border-radius: 5px;
        }
        .spinner-text {
            background-color: #fff8e1;
            color: #f57f17;
            padding: 1rem;
            border-radius: 5px;
        }
        .upload-section {
            background-color: white;
            padding: 1.5rem;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        .results-section {
            background-color: white;
            padding: 1.5rem;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        img {
            max-width: 100%;
            max-height: 300px;
            object-fit: contain;
            border: 2px solid #a5d6a7;
            border-radius: 5px;
            padding: 5px;
            background-color: #f5f5f5;
        }
        /* Style for the description text */
        .description {
            color: #555;
            background-color: #f8f9fa;
            padding: 0.8rem;
            border-radius: 5px;
            border-left: 3px solid #43a047;
        }
        /* Style for confidence score */
        .confidence {
            color: #0d47a1;
            background-color: #e3f2fd;
            padding: 0.5rem;
            border-radius: 5px;
            display: inline-block;
            margin-top: 0.5rem;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

# TensorFlow Model Prediction
def model_prediction(test_image):
    try:
        # Load the model with .h5 extension
        model = tf.keras.models.load_model('trained_model.h5')
        
        # Open and process the image
        image = Image.open(test_image)
        image = image.resize((128, 128))
        
        # Convert to array and preprocess
        input_arr = tf.keras.preprocessing.image.img_to_array(image)
        input_arr = np.expand_dims(input_arr, axis=0)  # Convert single image to batch
        
        # Make prediction
        prediction = model.predict(input_arr)
        result_index = np.argmax(prediction)
        confidence = float(prediction[0][result_index]) * 100
        
        return result_index, confidence
    except Exception as e:
        st.error(f"Error in prediction: {e}")
        return None, None

# Define disease classes
class_name = [
    'Apple___Apple_scab',
    'Apple___Black_rot',
    'Apple___Cedar_apple_rust',
    'Apple___healthy',
    'Blueberry___healthy',
    'Cherry_(including_sour)___Powdery_mildew',
    'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight',
    'Corn_(maize)___healthy',
    'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)',
    'Peach___Bacterial_spot',
    'Peach___healthy',
    'Pepper,_bell___Bacterial_spot',
    'Pepper,_bell___healthy',
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy',
    'Raspberry___healthy',
    'Soybean___healthy',
    'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch',
    'Strawberry___healthy',
    'Tomato___Bacterial_spot',
    'Tomato___Early_blight',
    'Tomato___Late_blight',
    'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]

# Get treatment recommendations based on disease
def get_treatment_recommendation(disease_class):
    recommendations = {
        'Apple_scab': "Apply fungicide treatments early in the season. Prune affected branches and remove fallen leaves to reduce infection spread.",
        'Black_rot': "Remove mummified fruits, cankers and infected plant material. Apply fungicides and ensure good air circulation.",
        'Cedar_apple_rust': "Apply preventative fungicides and avoid planting near cedar trees which host the fungus.",
        'Powdery_mildew': "Apply sulfur or potassium bicarbonate sprays. Improve air circulation and avoid overhead watering.",
        'Cercospora_leaf_spot': "Rotate crops, remove crop debris, and apply appropriate fungicides when needed.",
        'Common_rust': "Plant resistant varieties and apply fungicides early in the season.",
        'Northern_Leaf_Blight': "Use crop rotation, plant resistant varieties, and apply fungicides when necessary.",
        'Esca': "Prune during dry weather and seal large pruning wounds. There's no effective chemical treatment.",
        'Leaf_blight': "Apply appropriate fungicides and ensure good vineyard sanitation.",
        'Haunglongbing': "Remove infected trees entirely as there is no cure. Control psyllid populations with insecticides.",
        'Bacterial_spot': "Apply copper-based bactericides and avoid overhead irrigation.",
        'Early_blight': "Practice crop rotation, remove infected leaves, and apply fungicides when necessary.",
        'Late_blight': "Apply preventative fungicides, ensure good air circulation, and avoid overhead watering.",
        'Leaf_Mold': "Improve air circulation, avoid overhead watering, and apply appropriate fungicides.",
        'Septoria_leaf_spot': "Remove infected leaves, practice crop rotation, and apply fungicides preventatively.",
        'Spider_mites': "Introduce predatory mites or apply insecticidal soap or neem oil.",
        'Target_Spot': "Improve air circulation, avoid overhead watering, and apply appropriate fungicides.",
        'Yellow_Leaf_Curl_Virus': "Control whitefly populations, use reflective mulches, and plant resistant varieties.",
        'Tomato_mosaic_virus': "Remove infected plants, control aphids, and wash hands and tools regularly.",
        'Leaf_scorch': "Ensure adequate irrigation and nutrients, improve soil drainage.",
        'Powdery_mildew_squash': "Apply neem oil or potassium bicarbonate sprays. Improve air circulation."
    }
    
    # Search for partial matches in the disease class name
    for key in recommendations:
        if key.lower() in disease_class.lower().replace('_', ' '):
            return recommendations[key]
    
    return "Maintain healthy growing practices including proper watering, fertilization, and pest management."

# Format disease name for display
def format_disease_name(class_name):
    parts = class_name.split('___')
    plant = parts[0].replace('_', ' ')
    condition = parts[1].replace('_', ' ')
    
    return f"{plant} - {condition}"

# Add styling
add_custom_styling()

# Main content
st.markdown("<h1 class='main-header'>🌿 Plant Disease Recognition System</h1>", unsafe_allow_html=True)

# Two column layout for better organization
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("<h2 class='sub-header'>Upload Plant Image</h2>", unsafe_allow_html=True)
    
    # Description
    st.markdown("<p class='description'>Upload a clear image of a plant leaf to detect potential diseases. For best results, use well-lit, in-focus images with the leaf centered in the frame.</p>", unsafe_allow_html=True)
    
    # File uploader
    test_image = st.file_uploader("Choose an image file:", type=["jpg", "jpeg", "png"])
    
    # Button row
    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        preview_button = st.button("Preview Image")
    with col_btn2:
        predict_button = st.button("Analyze Image")
        
    # Image preview section
    if test_image is not None and (preview_button or predict_button):
        st.image(test_image, caption="Uploaded Image")
    st.markdown("</section>", unsafe_allow_html=True)

with col2:
    st.markdown("<h2 class='sub-header'>Results & Analysis</h2>", unsafe_allow_html=True)
    
    # Show prediction results
    if test_image is not None and predict_button:
        with st.spinner("Analyzing image... Please wait"):
            # Get prediction
            result_index, confidence = model_prediction(test_image)
            
            if result_index is not None:
                result = class_name[result_index]
                formatted_result = format_disease_name(result)
                
                # Extract disease name for treatment recommendations
                disease_parts = result.split('___')[1]
                
                # Display result with clear styling
                if "healthy" in result.lower():
                    st.markdown(f"### Diagnosis: <span class='healthy'>HEALTHY</span>", unsafe_allow_html=True)
                    
                    # Recommendation for healthy plants
                    st.markdown("<h4 class='recommendation-title'>Recommendation:</h4>", unsafe_allow_html=True)
                    st.markdown("<p class='recommendation-content'>Continue with regular plant care and maintenance. Ensure proper watering, adequate sunlight, and regular fertilization.</p>", unsafe_allow_html=True)
                else:
                    st.markdown(f"### Diagnosis: <span class='diseased'>{formatted_result}</span>", unsafe_allow_html=True)
                    
                    # Get treatment recommendations
                    treatment = get_treatment_recommendation(disease_parts)
                    
                    # Display recommendations with clear styling
                    st.markdown("<h4 class='recommendation-title'>Treatment Recommendations:</h4>", unsafe_allow_html=True)
                    st.markdown(f"<p class='recommendation-content'>{treatment}</p>", unsafe_allow_html=True)
            else:
                st.markdown("<p class='error-text'>Unable to process the image. Please try uploading a different image.</p>", unsafe_allow_html=True)
    else:
        st.markdown("<p class='info-text'>Upload an image and click 'Analyze Image' to get disease diagnosis and recommendations.</p>", unsafe_allow_html=True)
    st.markdown("</section>", unsafe_allow_html=True)

st.markdown("<p class='footer'>Plant Disease Recognition System | © 2025</p>", unsafe_allow_html=True)
