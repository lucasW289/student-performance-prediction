# app.py

import streamlit as st
import joblib
import pandas as pd
import numpy as np
import base64 

# Project Name: EduPredict: Student Pass/Fail Predictor
st.set_page_config(page_title="Student Performance Predictor", layout="wide")
st.title('Project2_Group17\n🎓📊 EduPredict: Student Performance Prediction')

st.write("""
Enter the student's information to predict whether they will Pass or Fail.
""")

# Optional: Add a background image (You'll need an image file)
# def add_bg_from_local(image_file):
#     with open(image_file, "rb") as f:
#         img_bytes = f.read()
#     encoded_img = base64.b64encode(img_bytes).decode()
#     st.markdown(
#         f"""
#         <style>
#         .stApp {{
#             background-image: url("data:image/jpeg;base64,{encoded_img}");
#             background-size: cover;
#             background-attachment: fixed;
#             opacity: 0.8; # Adjust opacity as needed
#         }}
#         </style>
#         """,
#         unsafe_allow_html=True
#     )

# Try adding a background image - uncomment the next line and provide your image file name
# try:
#     add_bg_from_local('student_bg.jpg') # Replace 'student_bg.jpg' with your image file name
# except FileNotFoundError:
#     st.warning("Background image file not found. Skipping background image.")


# Load the trained pipeline, original feature names, and label encoder
# Make sure these files are in the same directory as your app.py file,
# or provide the correct path.
pipeline_filename = 'student_performance_pipeline.kpl'
original_features_filename = 'student_performance_original_features.kpl'
label_encoder_filename = 'student_performance_label_encoder.kpl'

try:
    model_pipeline = joblib.load(pipeline_filename)
    original_feature_names = joblib.load(original_features_filename)
    label_encoder = joblib.load(label_encoder_filename)
    
except FileNotFoundError:
    st.sidebar.error("Error: Model, features, or label encoder file(s) not found.")
    st.sidebar.info(f"Please ensure '{pipeline_filename}', '{original_features_filename}', and '{label_encoder_filename}' are in the same directory.")
    st.stop() # Stop the app if files are not found
except Exception as e:
    st.sidebar.error(f"An error occurred while loading the files: {e}")
    st.stop()


st.sidebar.header('Student Information Input')

# Create input fields for each original feature in the sidebar
input_data = {}

# Define possible values for categorical features based on your dataset sample
gender_options = ['Male', 'Female']
parental_education_options = ['High School', 'Bachelors', 'Masters', 'PhD']
internet_access_options = ['Yes', 'No']
extracurricular_options = ['Yes', 'No']

for feature in original_feature_names:
    if feature == 'Gender':
        input_data[feature] = st.sidebar.selectbox(f'{feature.replace("_", " ").title()}', gender_options)
    elif feature == 'Study_Hours_per_Week':
        # Use st.slider for numerical features with a typical range
        # Adjust min_value, max_value, and value based on your dataset's distribution
        input_data[feature] = st.sidebar.slider(f'{feature.replace("_", " ").title()}', 0.0, 50.0, 15.0, 0.5)
    elif feature == 'Attendance_Rate':
        # Attendance rate is a percentage
         input_data[feature] = st.sidebar.slider(f'{feature.replace("_", " ").title()} (%)', 0.0, 100.0, 80.0, 0.1)
    elif feature == 'Past_Exam_Scores':
         # Past exam scores have a range (e.g., 50-100)
         input_data[feature] = st.sidebar.slider(f'{feature.replace("_", " ").title()}', 0.0, 100.0, 75.0, 0.5)
    elif feature == 'Parental_Education_Level':
        input_data[feature] = st.sidebar.selectbox(f'{feature.replace("_", " ").title()}', parental_education_options)
    elif feature == 'Internet_Access_at_Home':
        input_data[feature] = st.sidebar.selectbox(f'{feature.replace("_", " ").title()}', internet_access_options)
    elif feature == 'Extracurricular_Activities':
        input_data[feature] = st.sidebar.selectbox(f'{feature.replace("_", " ").title()}', extracurricular_options)
    else:
        # Fallback for any unexpected features (shouldn't happen if features are loaded correctly)
        input_data[feature] = st.sidebar.text_input(f'{feature.replace("_", " ").title()}', value='')


# Create a button to make predictions
if st.button('Predict Pass/Fail Status'):
    # Prepare the input data for the pipeline
    # Create a DataFrame with the same column order as the original training data
    input_df = pd.DataFrame([input_data])

    # Ensure column order matches the training data (important for the pipeline)
    input_df = input_df[original_feature_names]

    # Make prediction using the pipeline (handles preprocessing and prediction)
    predicted_encoded = model_pipeline.predict(input_df)

    # Decode the numerical prediction back to the original class label ('Pass' or 'Fail')
    predicted_class = label_encoder.inverse_transform(predicted_encoded)

    # Display the prediction
    st.subheader('Prediction Result:')
    predicted_status = predicted_class[0]

    if predicted_status == 'Pass':
        st.success(f'The predicted student status is: **{predicted_status}** 🎉')
        st.balloons()
    else:
        st.error(f'The predicted student status is: **{predicted_status}** 😞')
        st.warning("This student is predicted to be at risk of failing based on the input information.")

    # Optional: Display prediction probabilities
    # st.subheader('Prediction Probabilities:')
    # prediction_proba = model_pipeline.predict_proba(input_df)
    # proba_df = pd.DataFrame(prediction_proba, columns=label_encoder.classes_)
    # proba_df = proba_df.transpose().reset_index()
    # proba_df.columns = ['Status', 'Probability']
    # st.dataframe(proba_df)
