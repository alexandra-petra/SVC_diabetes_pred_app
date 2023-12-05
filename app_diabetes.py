import numpy as np
import pickle
import pandas as pd
import streamlit as st
import pipreqs
from PIL import Image


#loading save model
loaded_model = pickle.load(open('trained_model_ppg.pkl', 'rb'))

# st.title('Model')
# st.sidebar.header('PPG data')
# image = Image.open('PPG-signal-analysis-2.png')
# st.image(image, '')



def diabetesPrediction(input):
    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input)

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    if (prediction[0] == 0):
        return 'The person is not diabetic'
    else:
        return 'The person is diabetic'
    
    
def main():
    #title
    st.title('Diabetes Prediction Web App')
    image = Image.open('PPG-signal-analysis-2.png')
    st.image(image, '')
    #getting input from users
    Sex_M_F = st.text_input('Your Gender')
    Age_year = st.text_input('Your Age')
    Height_cm = st.text_input('Height, cm')
    Weight_kg = st.text_input('Weight')
    Systolic_Blood_Pressure_mmHg  = st.text_input('Systolic BP')
    Diastolic_Blood_Pressure_mmHg = st.text_input('Diastolic BP')
    Heart_Rate_b_m = st.text_input('Your Heart Rate Value')
    BMI_kg_m2 = st.text_input('BMI Value')
    Hypertension = st.text_input('Prehypertension - 1, Normal - 0, Stage 1 hypertension - 2, Stage 2 hypertension - 3')
    cerebral_infarction = st.text_input('cerebral infarction - 0, no cerebral infarction - 1')
    cerebrovascular_disease = st.text_input('Nothing - 2, insufficiency of cerebral blood supply - 1, cerebrovascular disease - 0')
    
    #code for prediction
    diagnosis = ''
    
    #creating a button for prediction
    if st.button('Diabetes Test Result'):
        diagnosis = diabetesPrediction([Sex_M_F, Age_year, Height_cm, Weight_kg, Systolic_Blood_Pressure_mmHg, Diastolic_Blood_Pressure_mmHg, Heart_Rate_b_m, BMI_kg_m2, Hypertension, cerebral_infarction, cerebrovascular_disease])
        
    st.success(diagnosis)
    
if __name__ == '__main__':
    main()
