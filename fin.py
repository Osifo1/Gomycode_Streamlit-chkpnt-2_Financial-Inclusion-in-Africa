import streamlit as st
import joblib
import numpy as np

# Load the trained model
model = joblib.load('Financial Inclusion_model.joblib')

# Function to make predictions
def predict(features):
    # Reshape the features if necessary
    features = np.array(features).reshape(1, -1)
    prediction = model.predict(features)
    return prediction

# Streamlit user interface
st.title("Financial Inclusion Prediction")
st.write("Enter the following features:")

# Input fields for all necessary features
country = st.selectbox("Country", ["Kenya", "Rwanda", "Tanzania", "Uganda"])
year = 2018  # Fixed as per your data sample

# Location type
location_type = st.selectbox("Location Type", ["Urban", "Rural"])
location_encoded = 1 if location_type == "Urban" else 0  # Encode as 1 for Urban, 0 for Rural

# Cellphone access
cellphone_access = st.selectbox("Cellphone Access", ["Yes", "No"])
cellphone_access_encoded = 1 if cellphone_access == "Yes" else 0

# Household size
household_size = st.number_input("Household Size", min_value=1, value=1)

# Age of respondent
age_of_respondent = st.number_input("Age of Respondent", min_value=0, max_value=100)

# Gender
gender_of_respondent = st.selectbox("Gender", ["Male", "Female"])
gender_encoded = 1 if gender_of_respondent == "Male" else 0

# Relationship with head of household
relationship_with_head = st.selectbox("Relationship with Head of Household", 
    ["Head of Household", "Other non-relatives", "Other relative", "Parent", "Spouse"])
relationship_encoded = {
    "Head of Household": 0,
    "Other non-relatives": 1,
    "Other relative": 2,
    "Parent": 3,
    "Spouse": 4
}.get(relationship_with_head)

# Marital status
marital_status = st.selectbox("Marital Status", 
    ["Dont know", "Married/Living together", "Single/Never Married", "Widowed"])
marital_status_encoded = {
    "Dont know": 0,
    "Married/Living together": 1,
    "Single/Never Married": 2,
    "Widowed": 3
}.get(marital_status)

# Education level
education_level = st.selectbox("Education Level", 
    ["Other/Dont know/RTA", "Primary education", "Secondary education", "Tertiary education", "Vocational/Specialised training"])
education_encoded = {
    "Other/Dont know/RTA": 0,
    "Primary education": 1,
    "Secondary education": 2,
    "Tertiary education": 3,
    "Vocational/Specialised training": 4
}.get(education_level)

# Job type
job_type = st.selectbox("Job Type", 
    ["Farming and Fishing", "Formally employed Government", "Formally employed Private", "Government Dependent", 
     "Informally employed", "No Income", "Other Income", "Remittance Dependent", "Self employed"])
job_type_encoded = {
    "Farming and Fishing": 0,
    "Formally employed Government": 1,
    "Formally employed Private": 2,
    "Government Dependent": 3,
    "Informally employed": 4,
    "No Income": 5,
    "Other Income": 6,
    "Remittance Dependent": 7,
    "Self employed": 8
}.get(job_type)

# Prepare features for the model
features = [
    year,  # Year is fixed as 2018
    location_encoded,
    cellphone_access_encoded,
    household_size,
    age_of_respondent,
    gender_encoded,
    relationship_encoded,
    marital_status_encoded,
    education_encoded,
    job_type_encoded,
]

# Ensure features is the correct length
# The encoded features should make up the total of 32
if len(features) < 32:
    # Fill in missing features with zeros or a suitable default value
    missing_features_count = 32 - len(features)
    features.extend([0] * missing_features_count)

# Button to make prediction
if st.button("Predict"):
    # Get the prediction
    prediction = predict(features)
    
    # Display the prediction
    if prediction[0]:
        st.success("Prediction: Financially Included")
    else:
        st.error("Prediction: Not Financially Included")
