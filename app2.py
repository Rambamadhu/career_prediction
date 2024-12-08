import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the trained model
model = pickle.load(open('CAREER.pkl', 'rb'))

# Function to collect user input features
def user_input_features():
    Logical_quotient_rating = st.sidebar.slider("Logical Quotient Rating", 1, 10, 5)
    coding_skills_rating = st.sidebar.slider("Coding Skills Rating", 1, 10, 5)
    hackathons = st.sidebar.slider("Hackathons", 0, 10, 1)
    public_speaking_points = st.sidebar.slider("Public Speaking Points", 1, 10, 5)
    self_learning_capability = st.sidebar.selectbox("Self-Learning Capability?", ["yes", "no"])
    extra_courses_did = st.sidebar.selectbox("Extra Courses Did?", ["yes", "no"])
    inputs_from_seniors = st.sidebar.selectbox("Taken Inputs From Seniors?", ["yes", "no"])
    worked_in_teams = st.sidebar.selectbox("Worked in Teams?", ["yes", "no"])
    introvert = st.sidebar.selectbox("Introvert?", ["yes", "no"])
    reading_skills = st.sidebar.selectbox("Reading and Writing Skills", ["poor", "medium", "excellent"])
    memory_capability = st.sidebar.selectbox("Memory Capability Score", ["poor", "medium", "excellent"])

    # Feature dictionary matching model's expected input
    features = {
        'Logical quotient rating': Logical_quotient_rating,
        'coding skills rating': coding_skills_rating,
        'hackathons': hackathons,
        'public speaking points': public_speaking_points,
        'self-learning capability?': 1 if self_learning_capability == "yes" else 0,
        'Extra-courses did': 1 if extra_courses_did == "yes" else 0,
        'Taken inputs from seniors or elders': 1 if inputs_from_seniors == "yes" else 0,
        'worked in teams ever?': 1 if worked_in_teams == "yes" else 0,
        'Introvert': 1 if introvert == "yes" else 0,
        'reading and writing skills': {"poor": 0, "medium": 1, "excellent": 2}[reading_skills],
        'memory capability score': {"poor": 0, "medium": 1, "excellent": 2}[memory_capability],
        'B_hard worker': 0,  # Placeholder for binary-encoded features
        'B_smart worker': 0,
        'A_Management': 0,
        'A_Technical': 0,
        'Interested subjects_code': 0,
        'Interested Type of Books_code': 0,
        'certifications_code': 0,
        'workshops_code': 0,
        'Type of company want to settle in?_code': 0,
        'interested career area _code': 0
    }

    return pd.DataFrame([features])

# Streamlit UI
st.title("Career Recommendation System")
st.write("""
### Predict the best career path based on your skills and preferences.
""")

# Collect user input features
input_df = user_input_features()

# Display user input features
st.subheader("Your Input Features")
st.write(input_df)

# Make prediction
if st.button("Predict"):
    try:
        prediction = model.predict(input_df)
        prediction_proba = model.predict_proba(input_df)

        st.subheader("Prediction")
        st.write(f"Suggested Job Role: **{prediction[0]}**")

        st.subheader("Prediction Probabilities")
        st.write("Probabilities for all possible job roles:")
        st.write(prediction_proba)
    except ValueError as e:
        st.error(f"Error: {e}")
