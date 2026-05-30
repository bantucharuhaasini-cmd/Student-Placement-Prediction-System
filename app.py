import streamlit as st
import joblib
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Placement Prediction System",
    page_icon="🎓",
    layout="wide"
)

# Load model
model = joblib.load("model/placement_model.pkl")

# Sidebar
st.sidebar.title("📌 About Project")

st.sidebar.info(
    """
    This ML application predicts whether
    a student is likely to get placed
    based on academic and skill data.
    """
)

st.sidebar.success("Model: Random Forest Classifier")

# Main title
st.title("🎓 Student Placement Prediction System")

st.markdown("---")

st.subheader("📋 Enter Student Details")

# Create columns
col1, col2 = st.columns(2)

# Left column
with col1:

    cgpa = st.slider(
        "CGPA",
        0.0,
        10.0,
        7.0
    )

    internships = st.number_input(
        "Internships",
        min_value=0,
        max_value=10,
        value=1
    )

    projects = st.number_input(
        "Projects",
        min_value=0,
        max_value=20,
        value=2
    )

    workshops = st.number_input(
        "Workshops / Certifications",
        min_value=0,
        max_value=20,
        value=1
    )

# Right column
with col2:

    aptitude = st.slider(
        "Aptitude Test Score",
        0,
        100,
        50
    )

    soft_skills = st.slider(
        "Soft Skills Rating",
        0,
        100,
        50
    )

    ssc_marks = st.slider(
        "SSC Marks",
        0.0,
        100.0,
        75.0
    )

    eca = st.selectbox(
        "Extracurricular Activities",
        ["Yes", "No"]
    )

    training = st.selectbox(
        "Placement Training",
        ["Yes", "No"]
    )

# Encode values
eca_value = 1 if eca == "Yes" else 0
training_value = 1 if training == "Yes" else 0

st.markdown("---")

# Predict button
if st.button("🚀 Predict Placement"):

    # Progress bar animation
    progress = st.progress(0)

    for i in range(100):
        progress.progress(i + 1)

    # Prepare input
    input_data = np.array([[
        cgpa,
        internships,
        projects,
        workshops,
        aptitude,
        soft_skills,
        ssc_marks,
        eca_value,
        training_value
    ]])

    # Prediction
    prediction = model.predict(input_data)

    # Probability
    probability = model.predict_proba(input_data)

    confidence = round(probability[0][1] * 100, 2)

    st.markdown("---")

    st.subheader("📊 Prediction Result")

    # Result
    if prediction[0] == 1:

        st.success("✅ Student is likely to be PLACED")

        st.balloons()

    else:

        st.error("❌ Student is NOT likely to be placed")

    # Metrics
    col3, col4 = st.columns(2)

    with col3:
        st.metric(
            label="Placement Probability",
            value=f"{confidence}%"
        )

    with col4:
        st.metric(
            label="Model Accuracy",
            value="77.7%"
        )

st.markdown("---")

st.caption("Built using Python, Scikit-learn and Streamlit")