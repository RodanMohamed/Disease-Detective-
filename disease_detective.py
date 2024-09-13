# -*- coding: utf-8 -*-
"""
Created on Sat Sep  7 15:59:16 2024

@author: Rodan Mohamed
"""

import os
import pickle
import streamlit as st
from streamlit_option_menu import option_menu


# Set page configuration
st.set_page_config(page_title="Disease Detective",
                   layout="wide",
                   page_icon="🩺",  # Medical emoji
                   initial_sidebar_state="expanded")

# Custom CSS for background and colors
st.markdown("""
    <style>
    body {
        background-image: url('https://image.shutterstock.com/image-photo/medical-background-doctor-holding-stethoscope-260nw-1938504977.jpg');
        background-size: cover;
        color: #ffffff;
    }
    .css-1d391kg {
        background-color: rgba(255, 255, 255, 0.7) !important;
        border-radius: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

# Loading the saved models
diabetes_model = pickle.load(open('Diabetes_trained_model.save', 'rb'))
heart_disease_model = pickle.load(open('heart_disease_model.sav', 'rb'))
parkinsons_model = pickle.load(open('parkinsons_model.sav', 'rb'))
breast_cancer_model = pickle.load(open('breast_cancer_model1.sav', 'rb'))
chronic_kidney_model = pickle.load(open('kidney_disease_model.sav', 'rb'))
Hepatitis_disease_model = pickle.load(open('Hepatitis_disease_model.sav', 'rb'))

# Sidebar navigation with emojis and medical icons
with st.sidebar:
    # Set the title with custom styling
    st.markdown("""
        <style>
        .custom-title {
            font-size: 33px;
            font-weight: bold;
            color: #008CBA;  /* Change to your desired color */
        }
        </style>
        """, unsafe_allow_html=True)

    st.markdown('<div class="custom-title">Disease Detective👾</div>', unsafe_allow_html=True)

    # Create the sidebar menu with the correct arguments
    selected = option_menu(
    menu_title=None,  # You can leave this as None or specify a title
    options=['🩺 Diabetes Prediction', '❤️ Heart Disease Prediction', '🧠 Parkinson’s Prediction', '🎀 Breast Cancer Prediction','🦠 Kidney Disease Prediction','🩸 Hepatitis Disease Prediction' ],
    icons=['activity', 'heart', 'person', 'ribbon','virus','droplet'],
    menu_icon='hospital',  # Sidebar icon
    default_index=0,
    styles={
        "container": {"background-color": "#00264d", "padding": "5px"},
        "icon": {"color": "orange", "font-size": "25px"},
        "nav-link": {"font-size": "20px", "text-align": "left", "color": "#ffffff"},
        "nav-link-selected": {"background-color": "#1c2e4a"},
    }
)

    # Add your name after the menu
    st.markdown("<h4 style='text-align: center; color: #008CBA;'>Developed by: Rodan Mohamed</h4>", unsafe_allow_html=True)

# Function to create a custom button style
def create_button_style():
    st.markdown("""
    <style>
    div.stButton > button:first-child {
        background-color: #008CBA;
        color: white;
        height: 3em;
        width: 100%;
        border-radius: 10px;
        font-size: 20px;
    }
    div.stButton > button:first-child:hover {
        background-color: #005f73;
        color: #ffffff;
    }
    </style>
    """, unsafe_allow_html=True)

# Header for each page
def page_header(title):
    st.markdown(f"<h1 style='text-align: center; color:  #ff5722;'>{title}</h1>", unsafe_allow_html=True)
# Diabetes Prediction Page
if selected == '🩺 Diabetes Prediction':
    page_header('🩺 Diabetes Prediction')
    col1, col2, col3 = st.columns(3)
    # (Rest of the code for Diabetes Prediction)

# Heart Disease Prediction Page
if selected == '❤️ Heart Disease Prediction':
    page_header('❤️ Heart Disease Prediction')
    col1, col2, col3 = st.columns(3)
    # (Rest of the code for Heart Disease Prediction)

# Parkinson's Prediction Page
if selected == "🧠 Parkinson’s Prediction":
    page_header("🧠 Parkinson’s Prediction")
    col1, col2, col3, col4, col5 = st.columns(5)
    # (Rest of the code for Parkinson’s Prediction)

# Breast Cancer Prediction Page
if selected == '🎀 Breast Cancer Prediction':
    page_header('🎀 Breast Cancer Prediction')
    col1, col2, col3, col4 = st.columns(4)
    # (Rest of the code for Breast Cancer Prediction)

# Kidney Disease Prediction Page
if selected == '🦠 Kidney Disease Prediction':
    page_header('🦠 Kidney Disease Prediction')
    col1, col2, col3, col4 = st.columns(4)
    # (Rest of the code for Kidney Disease Prediction)
# Function to display welcome message
def display_welcome_message():
    st.markdown("<h1 style='text-align: center; color: #008CBA; font-size: 40px; margin-top: -190px;'>Welcome to Disease Detective👾</h1>", unsafe_allow_html=True)
# Check if a prediction page is selected and if the welcome message hasn't been shown
if selected in ['🩺 Diabetes Prediction', '❤️ Heart Disease Prediction', '🧠 Parkinson’s Prediction', '🎀 Breast Cancer Prediction', '🦠 Kidney Disease Prediction']:
    if 'welcome_message_shown' not in st.session_state or not st.session_state.welcome_message_shown:
        st.session_state.welcome_message_shown = True
        display_welcome_message()  # Call the function to display the message

# Diabetes Prediction Page
if selected == '🩺 Diabetes Prediction':
    #page_header('🩺 Diabetes Prediction')
    col1, col2, col3 = st.columns(3)

    with col1:
        Pregnancies = st.text_input('🤰 Number of Pregnancies')
    with col2:
        Glucose = st.text_input('🍬 Glucose Level')
    with col3:
        BloodPressure = st.text_input('💉 Blood Pressure')
    with col1:
        SkinThickness = st.text_input('📏 Skin Thickness')
    with col2:
        Insulin = st.text_input('💉 Insulin Level')
    with col3:
        BMI = st.text_input('📊 BMI Value')
    with col1:
        DiabetesPedigreeFunction = st.text_input('📈 Diabetes Pedigree Function')
    with col2:
        Age = st.text_input('👵 Age')

    diab_diagnosis = ''
    create_button_style()

    if st.button('Get Diabetes Test Result'):
        # Check if all fields are filled
        if not all([Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]):
            st.warning("Please enter all the required information to help us provide a diagnosis.")
        else:
            try:
                # Prepare user input
                user_input = [float(Pregnancies), float(Glucose), float(BloodPressure), float(SkinThickness), float(Insulin), float(BMI), float(DiabetesPedigreeFunction), float(Age)]
                
                # Predict using the loaded model
                diab_prediction = diabetes_model.predict([user_input])
                
                # Display the diagnosis
                diab_diagnosis = 'The person is diabetic' if diab_prediction[0] == 1 else 'The person is not diabetic'
                st.success(diab_diagnosis)
                
            except Exception as e:
                st.error(f"Problem occurred: {e}")

# Heart Disease Prediction Page
if selected == '❤️ Heart Disease Prediction':
   # page_header('❤️ Heart Disease Prediction')
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.text_input('👵 Age')
    with col2:
        sex = st.text_input('⚤ Sex')
    with col3:
        cp = st.text_input('💢 Chest Pain Types')
    with col1:
        trestbps = st.text_input('💉 Resting Blood Pressure')
    with col2:
        chol = st.text_input('🍳 Serum Cholesterol (mg/dl)')
    with col3:
        fbs = st.text_input('🍬 Fasting Blood Sugar > 120 mg/dl (1 = True, 0 = False)')
    with col1:
        restecg = st.text_input('🧑‍⚕️ Resting ECG Results')
    with col2:
        thalach = st.text_input('🫀 Max Heart Rate Achieved')
    with col3:
        exang = st.text_input('🏋️‍♂️ Exercise Induced Angina (1 = Yes, 0 = No)')
    with col1:
        oldpeak = st.text_input('📉 ST Depression Induced by Exercise')
    with col2:
        slope = st.text_input('📈 Slope of the Peak Exercise ST Segment')
    with col3:
        ca = st.text_input('🩺 Major Vessels Colored by Fluoroscopy')
    with col1:
        thal = st.text_input('🧬 Thalassemia (0 = Normal, 1 = Fixed Defect, 2 = Reversible Defect)')

    heart_diagnosis = ''
    create_button_style()

    if st.button('Get Heart Disease Test Result'):
        # Check if all fields are filled
        if not all([age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]):
            st.warning("Please enter all the required information to help us provide a diagnosis.")
        else:
            user_input = [float(age), float(sex), float(cp), float(trestbps), float(chol), float(fbs), float(restecg), float(thalach), float(exang), float(oldpeak), float(slope), float(ca), float(thal)]

            heart_prediction = heart_disease_model.predict([user_input])
            heart_diagnosis = 'The person has heart disease' if heart_prediction[0] == 1 else 'The person does not have heart disease'
            st.success(heart_diagnosis)

# Parkinson's Prediction Page
# Parkinson's Prediction Page
if selected == "🧠 Parkinson’s Prediction":
   # st.title("🧠 Parkinson's Disease Prediction")
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        fo = st.text_input('🔹 MDVP:Fo(Hz)')
    with col2:
        fhi = st.text_input('🔹 MDVP:Fhi(Hz)')
    with col3:
        flo = st.text_input('🔹 MDVP:Flo(Hz)')
    with col4:
        Jitter_percent = st.text_input('🔹 MDVP:Jitter(%)')
    with col5:
        Jitter_Abs = st.text_input('🔹 MDVP:Jitter(Abs)')
    with col1:
        RAP = st.text_input('🔹 MDVP:RAP')
    with col2:
        PPQ = st.text_input('🔹 MDVP:PPQ')
    with col3:
        DDP = st.text_input('🔹 Jitter:DDP')
    with col4:
        Shimmer = st.text_input('🔹 MDVP:Shimmer')
    with col5:
        Shimmer_dB = st.text_input('🔹MDVP:Shimmer(dB)')
    with col1:
        APQ3 = st.text_input('🔹 Shimmer:APQ3')
    with col2:
        APQ5 = st.text_input('🔹 Shimmer:APQ5')
    with col3:
        APQ = st.text_input('🔹 MDVP:APQ')
    with col4:
        DDA = st.text_input('🔹 Shimmer:DDA')
    with col5:
        NHR = st.text_input('🔹 NHR')
    with col1:
        HNR = st.text_input('🔹 HNR')
    with col2:
        RPDE = st.text_input('🔹 RPDE')
    with col3:
        DFA = st.text_input('🔹 DFA')
    with col4:
        spread1 = st.text_input('🔹 Spread1')
    with col5:
        spread2 = st.text_input('🔹 Spread2')
    with col1:
        D2 = st.text_input('🔹 D2')
    with col2:
        PPE = st.text_input('🔹 PPE')

    parkinsons_diagnosis = ''
    create_button_style()

    if st.button("Get Parkinson's Test Result"):
        # Check if all fields are filled
        if not all([fo, fhi, flo, Jitter_percent, Jitter_Abs, RAP, PPQ, DDP, Shimmer, Shimmer_dB, APQ3, APQ5, APQ, DDA, NHR, HNR, RPDE, DFA, spread1, spread2, D2, PPE]):
            st.warning("Please enter all the required information to help us provide a diagnosis.")
        else:
            user_input = [float(fo), float(fhi), float(flo), float(Jitter_percent), float(Jitter_Abs), float(RAP), float(PPQ), float(DDP), float(Shimmer), float(Shimmer_dB), float(APQ3), float(APQ5), float(APQ), float(DDA), float(NHR), float(HNR), float(RPDE), float(DFA), float(spread1), float(spread2), float(D2), float(PPE)]

            parkinsons_prediction = parkinsons_model.predict([user_input])
            parkinsons_diagnosis = "The person has Parkinson's disease" if parkinsons_prediction[0] == 1 else "The person does not have Parkinson's disease"
            st.success(parkinsons_diagnosis)

import pickle

# Load the scaler, NCA, and KNN models from the file
with open('breast_cancer_model1.sav', 'rb') as file:
    loaded_scaler, loaded_nca, loaded_knn_nca = pickle.load(file)

# Define a function to make predictions using the loaded models
def predict_with_loaded_knn_nca(input_data):
    # Standardize the input data using the loaded scaler
    input_data_standardized = loaded_scaler.transform([input_data])

    # Transform the standardized input data using the loaded NCA
    input_data_nca = loaded_nca.transform(input_data_standardized)

    # Make prediction using the loaded KNN model
    prediction = loaded_knn_nca.predict(input_data_nca)

    # Return the prediction result
    return "The person has breast cancer" if prediction[0] == 1 else "The person does not have breast cancer"

# Breast Cancer Prediction Page
if selected == '🎀 Breast Cancer Prediction':
    #page_header('🎀 Breast Cancer Prediction')
    col1, col2, col3 = st.columns(3)

    # Initialize session state for inputs
    if 'input_values' not in st.session_state:
        st.session_state.input_values = {}

    # Fields dictionary for input management
    fields = {
        'Radius Mean': 'radius_mean', 'Texture Mean': 'texture_mean', 'Perimeter Mean': 'perimeter_mean',
        'Area Mean': 'area_mean', 'Smoothness Mean': 'smoothness_mean', 'Compactness Mean': 'compactness_mean',
        'Concavity Mean': 'concavity_mean', 'Concave Points Mean': 'concave_points_mean', 'Symmetry Mean': 'symmetry_mean',
        'Fractal Dimension Mean': 'fractal_dimension_mean', 'Radius SE': 'radius_se', 'Texture SE': 'texture_se',
        'Perimeter SE': 'perimeter_se', 'Area SE': 'area_se', 'Smoothness SE': 'smoothness_se',
        'Compactness SE': 'compactness_se', 'Concavity SE': 'concavity_se', 'Concave Points SE': 'concave_points_se',
        'Symmetry SE': 'symmetry_se', 'Fractal Dimension SE': 'fractal_dimension_se', 'Radius Worst': 'radius_worst',
        'Texture Worst': 'texture_worst', 'Perimeter Worst': 'perimeter_worst', 'Area Worst': 'area_worst',
        'Smoothness Worst': 'smoothness_worst', 'Compactness Worst': 'compactness_worst', 'Concavity Worst': 'concavity_worst',
        'Concave Points Worst': 'concave_points_worst', 'Symmetry Worst': 'symmetry_worst', 'Fractal Dimension Worst': 'fractal_dimension_worst'
    }

    # Populate input fields
    col_index = 0
    for label, key in fields.items():
        value = st.session_state.input_values.get(key, '')
        if col_index == 0:
            with col1:
                st.session_state.input_values[key] = st.text_input(label, value=value)
        elif col_index == 1:
            with col2:
                st.session_state.input_values[key] = st.text_input(label, value=value)
        else:
            with col3:
                st.session_state.input_values[key] = st.text_input(label, value=value)

        col_index = (col_index + 1) % 3

    cancer_diagnosis = ''
    create_button_style()

    if st.button('Get Breast Cancer Test Result'):
        # Check if all fields are filled
        if not all(st.session_state.input_values.values()):
            st.warning("Please enter all the required information to help us provide a diagnosis.")
        else:
            try:
                # Prepare user input
                user_input = [float(st.session_state.input_values[key]) for key in fields.values()]

                # Use the loaded models to predict
                cancer_diagnosis = predict_with_loaded_knn_nca(user_input)
                
                # Display the result
                st.success(cancer_diagnosis)
            except Exception as e:
                st.error(f"Problem occurred: {e}")
# Kidney Disease Prediction Page
import pickle
from keras.models import load_model
import streamlit as st
import numpy as np

# Function to load the model and scaler
def load_model_and_scaler(model_filename, scaler_filename):
    model = load_model(model_filename)  # Load the CNN model from H5
    with open(scaler_filename, 'rb') as file:
        scaler = pickle.load(file)  # Load the scaler using pickle
    return model, scaler

# Load the CNN model and scaler for kidney disease prediction
#kidney_disease_model, scaler = load_model_and_scaler('C:/Users\ALRWOAD LABTOB/Documents/Study/medical website/kidney_disease_model.sav', 'scaler.pkl')

# Kidney Disease Prediction Page
if selected == '🦠 Kidney Disease Prediction':
    #st.header('🦠 Kidney Disease Prediction')
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.text_input('👶 Age')
    with col2:
        bp = st.text_input('💉 Blood Pressure')
    with col3:
        sg = st.text_input('🔬 Specific Gravity')
    with col1:
        al = st.text_input('⚗️ Albumin')
    with col2:
        su = st.text_input('🍬 Sugar')
    with col3:
        rbc = st.text_input('🩸 Red Blood Cells')
    with col1:
        pc = st.text_input('🩺 Pus Cells')
    with col2:
        pcc = st.text_input('💧 Pus Cell Clumps')
    with col3:
        ba = st.text_input('🔬 Bacteria')
    with col1:
        bgr = st.text_input('🩸 Blood Glucose Random')
    with col2:
        bu = st.text_input('💉 Blood Urea')
    with col3:
        sc = st.text_input('⚗️ Serum Creatinine')
    with col1:
        sod = st.text_input('🔬 Sodium')
    with col2:
        pot = st.text_input('🔬 Potassium')
    with col3:
        hemo = st.text_input('🩸 Hemoglobin')
    with col1:
        pcv = st.text_input('🔬 Packed Cell Volume')
    with col2:
        wc = st.text_input('🩸 White Blood Cell Count')
    with col3:
        rc = st.text_input('🔬 Red Blood Cell Count')
    with col1:
        htn = st.text_input('💉 Hypertension (1 = Yes, 0 = No)')
    with col2:
        dm = st.text_input('🍬 Diabetes Mellitus (1 = Yes, 0 = No)')
    with col3:
        cad = st.text_input('❤️ Coronary Artery Disease (1 = Yes, 0 = No)')
    with col1:
        appet = st.text_input('🍽️ Appetite')
    with col2:
        pe = st.text_input('🏥 Pedal Edema (1 = Yes, 0 = No)')
    with col3:
        ane = st.text_input('🩸 Anemia (1 = Yes, 0 = No)')

    kidney_diagnosis = ''
    create_button_style()

    if st.button('Get Kidney Disease Test Result'):
        # Check if all fields are filled
        if not all([age, bp, sg, al, su, rbc, pc, pcc, ba, bgr, bu, sc, sod, pot, hemo, pcv, wc, rc, htn, dm, cad, appet, pe, ane]):
            st.warning("Please enter all the required information to help us provide a diagnosis.")
        else:
            try:
                # Convert user inputs to a list of floats
                user_input = [float(age), float(bp), float(sg), float(al), float(su), float(rbc), float(pc), float(pcc), float(ba), float(bgr), 
                              float(bu), float(sc), float(sod), float(pot), float(hemo), float(pcv), float(wc), float(rc), float(htn), 
                              float(dm), float(cad), float(appet), float(pe), float(ane)]
                
                # Reshape and scale the input data using the loaded scaler
                user_input_scaled = scaler.transform([user_input])

                # Make prediction
                kidney_prediction = kidney_disease_model.predict(user_input_scaled)
                kidney_diagnosis = 'The person has kidney disease' if kidney_prediction[0][0] > 0.5 else 'The person does not have kidney disease'
                st.success(kidney_diagnosis)
            
            except Exception as e:
                st.error(f"An error occurred: {e}")

