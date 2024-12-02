import streamlit as st
import pandas as pd
import pickle
import os

# Set the path for the model file
model_path = "model L.pkl"  # Ensure this matches your actual file name and extension

# Debugging: Display the current working directory
st.write(f"Current Working Directory: {os.getcwd()}")

# Load the model
try:
    with open(model_path, "rb") as file:
        model = pickle.load(file)
    # Ensure the model has a predict method
    if not hasattr(model, "predict"):
        st.error("The loaded object is not a valid model. Please check the model file.")
except FileNotFoundError:
    st.error(f"Model file not found at: {model_path}. Please ensure it is in the correct directory.")
    st.stop()
except Exception as e:
    st.error(f"An error occurred while loading the model: {e}")
    st.stop()

# App title
st.title("Employee Performance Prediction")

# Sidebar for user input
st.sidebar.header("Input Parameters")

# Function to get user input for the dataset
def user_input_features():
    # Sidebar inputs for user features
    Age = st.sidebar.number_input("Age", min_value=18, max_value=65, step=1)
    Gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
    EducationBackground = st.sidebar.selectbox("Education Background", ["Science", "Commerce", "Arts", "Others"])
    MaritalStatus = st.sidebar.selectbox("Marital Status", ["Single", "Married", "Divorced", "Widowed"])
    EmpDepartment = st.sidebar.selectbox("Department", ["HR", "Finance", "R&D", "Sales", "IT"])
    EmpJobRole = st.sidebar.selectbox("Job Role", ["Manager", "Executive", "Analyst", "Technician", "Clerk"])
    BusinessTravelFrequency = st.sidebar.selectbox("Business Travel Frequency", ["Rarely", "Frequently", "Never"])
    DistanceFromHome = st.sidebar.number_input("Distance From Home (km)", min_value=0, max_value=100, step=1)
    EmpEducationLevel = st.sidebar.slider("Education Level (1-5)", min_value=1, max_value=5, step=1)
    EmpEnvironmentSatisfaction = st.sidebar.slider("Environment Satisfaction (1-5)", min_value=1, max_value=5, step=1)
    EmpHourlyRate = st.sidebar.number_input("Hourly Rate", min_value=10, max_value=100, step=1)
    EmpJobInvolvement = st.sidebar.slider("Job Involvement (1-5)", min_value=1, max_value=5, step=1)
    EmpJobLevel = st.sidebar.slider("Job Level (1-5)", min_value=1, max_value=5, step=1)
    EmpJobSatisfaction = st.sidebar.slider("Job Satisfaction (1-5)", min_value=1, max_value=5, step=1)
    NumCompaniesWorked = st.sidebar.number_input("Number of Companies Worked", min_value=0, max_value=10, step=1)
    OverTime = st.sidebar.selectbox("Overtime", ["Yes", "No"])
    EmpLastSalaryHikePercent = st.sidebar.number_input("Last Salary Hike Percent", min_value=0, max_value=100, step=1)
    EmpRelationshipSatisfaction = st.sidebar.slider("Relationship Satisfaction (1-5)", min_value=1, max_value=5, step=1)
    TotalWorkExperienceInYears = st.sidebar.number_input("Total Work Experience (Years)", min_value=0, max_value=50, step=1)
    TrainingTimesLastYear = st.sidebar.number_input("Training Times Last Year", min_value=0, max_value=10, step=1)
    EmpWorkLifeBalance = st.sidebar.slider("Work-Life Balance (1-5)", min_value=1, max_value=5, step=1)
    ExperienceYearsAtThisCompany = st.sidebar.number_input("Experience Years At Company", min_value=0, max_value=50, step=1)
    ExperienceYearsInCurrentRole = st.sidebar.number_input("Experience Years In Current Role", min_value=0, max_value=50, step=1)
    YearsSinceLastPromotion = st.sidebar.number_input("Years Since Last Promotion", min_value=0, max_value=50, step=1)
    YearsWithCurrManager = st.sidebar.number_input("Years With Current Manager", min_value=0, max_value=50, step=1)
    Attrition = st.sidebar.selectbox("Attrition", ["Yes", "No"])

    # Combine inputs into a dataframe
    data = {
        'Age': Age,
        'Gender': Gender,
        'EducationBackground': EducationBackground,
        'MaritalStatus': MaritalStatus,
        'EmpDepartment': EmpDepartment,
        'EmpJobRole': EmpJobRole,
        'BusinessTravelFrequency': BusinessTravelFrequency,
        'DistanceFromHome': DistanceFromHome,
        'EmpEducationLevel': EmpEducationLevel,
        'EmpEnvironmentSatisfaction': EmpEnvironmentSatisfaction,
        'EmpHourlyRate': EmpHourlyRate,
        'EmpJobInvolvement': EmpJobInvolvement,
        'EmpJobLevel': EmpJobLevel,
        'EmpJobSatisfaction': EmpJobSatisfaction,
        'NumCompaniesWorked': NumCompaniesWorked,
        'OverTime': OverTime,
        'EmpLastSalaryHikePercent': EmpLastSalaryHikePercent,
        'EmpRelationshipSatisfaction': EmpRelationshipSatisfaction,
        'TotalWorkExperienceInYears': TotalWorkExperienceInYears,
        'TrainingTimesLastYear': TrainingTimesLastYear,
        'EmpWorkLifeBalance': EmpWorkLifeBalance,
        'ExperienceYearsAtThisCompany': ExperienceYearsAtThisCompany,
        'ExperienceYearsInCurrentRole': ExperienceYearsInCurrentRole,
        'YearsSinceLastPromotion': YearsSinceLastPromotion,
        'YearsWithCurrManager': YearsWithCurrManager,
        'Attrition': Attrition
    }
    features = pd.DataFrame(data, index=[0])
    return features


# Load user input
input_features = user_input_features()

# Display input
st.subheader("User Input:")
st.write(input_features)

# Make prediction
if st.button("Predict"):
    try:
        prediction = model.predict(input_features)
        st.subheader("Prediction:")
        st.write(prediction[0])
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
