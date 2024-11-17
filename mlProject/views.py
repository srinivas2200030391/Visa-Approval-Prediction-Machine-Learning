import numpy as np
import pandas as pd
import tensorflow as tf
from django.shortcuts import render
from sklearn.preprocessing import LabelEncoder
import joblib

# Load pre-trained models
model_nn = tf.keras.models.load_model("mlProject/feed_forward_neural_networks.h5")
model_rf = joblib.load("mlProject/random_forest_model.pkl")
model_dt = joblib.load("mlProject/decision_tree_model.pkl")
model_lr = joblib.load("mlProject/logistic_regression_model.pkl")

# Define categorical columns and categories in the correct order
categorical_columns = [
    "country", "university", "education_level", "field_of_study",
    "requires_job_training", "job_offer_status", "financial_status",
    "scholarship_status"
]

all_categories = {
    "country": ['America', 'Dubai', 'Australia'],
    "university": ["University of Dubai",
                   "American University in Dubai",
                   "Zayed University", "University of Sydney",
                   "University of Melbourne",
                   "Australian National University",
                   "Harvard", "MIT", "Stanford"],
    "education_level": ['Undergraduate', 'Masters', 'PhD'],
    "field_of_study": ['Engineering', 'Computer Science', 'Medicine', 'Business', 'Arts'],
    "requires_job_training": ['True', 'False'],
    "job_offer_status": ['True', 'False'],
    "financial_status": ['Low', 'Medium', 'High'],
    "scholarship_status": ['True', 'False']
}

# Initialize label encoders
label_encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    le.fit(all_categories[col])
    label_encoders[col] = le


def predict_visa_status(request):
    if request.method != 'POST':
        return render(request, 'index.html')

    try:
        # Calculate student merit score
        print(request.POST)
        gpa = float(request.POST['gpa'])
        english_proficiency = float(request.POST['english_proficiency'])
        research_publications = int(request.POST['research_publications'])

        student_merit_score = float(gpa * 10 + english_proficiency * 0.3 + research_publications * 10)

        # Create input data in the exact order as training data
        input_data = {
            'country': request.POST['country'],
            'university': request.POST['university'],
            'student_merit_score': student_merit_score,
            'education_level': request.POST['education_level'],
            'field_of_study': request.POST['field_of_study'],
            'work_experience_years': float(request.POST['work_experience_years']),
            'requires_job_training': request.POST['requires_job_training'],
            'job_offer_status': request.POST['job_offer_status'],
            'financial_status': request.POST['financial_status'],
            'annual_income': float(request.POST['annual_income']),
            'scholarship_status': request.POST['scholarship_status'],
            'GPA': gpa,
            'research_publications': research_publications,
            'previous_visa_rejections': int(request.POST['previous_visa_rejections']),
            'english_proficiency': english_proficiency,
            'country_approval_rate': float(request.POST['country_approval_rate'])
        }

        # Convert to DataFrame and ensure column order
        columns_order = [
            'country', 'university', 'student_merit_score', 'education_level',
            'field_of_study', 'work_experience_years', 'requires_job_training',
            'job_offer_status', 'financial_status', 'annual_income',
            'scholarship_status', 'GPA', 'research_publications',
            'previous_visa_rejections', 'english_proficiency', 'country_approval_rate'
        ]

        df = pd.DataFrame([input_data])[columns_order]

        # Apply label encoding for categorical columns
        for col in categorical_columns:
            df[col] = label_encoders[col].transform(df[col].astype(str))

        # Convert boolean strings to numeric
        bool_columns = ['requires_job_training', 'job_offer_status', 'scholarship_status']
        for col in bool_columns:
            df[col] = (df[col] == 'True').astype(int)

        # Select model based on user input
        model_choice = request.POST['model']
        if model_choice == "neural_networks":
            model = model_nn
            prediction = model.predict(df)
            prediction = float(prediction[0][0])  # Convert to float for neural network
        elif model_choice == "random_forest":
            print(df)
            model = model_rf
            prediction = float(model.predict_proba(df)[0][1])  # Get probability of positive class
        elif model_choice == "decision_tree":
            model = model_dt
            prediction = float(model.predict_proba(df)[0][1])  # Get probability of positive class
        elif model_choice == "logistic_regression":
            model = model_lr
            prediction = float(model.predict_proba(df)[0][1])  # Get probability of positive class
        else:
            return render(request, 'index.html', {'error': "Invalid model choice"})

        # Calculate confidence score
        confidence = float(abs(prediction - 0.5) * 2 * 100)
        predicted_status = "Approved" if confidence > 0.7 else "Denied"
        print("Prediction (Neural Network):", model_nn.predict(df))
        print("Prediction (Random Forest):", model_rf.predict_proba(df)[0][1])
        print("Prediction (Decision Tree):", model_dt.predict_proba(df)[0][1])
        print("Prediction (Logistic Regression):", model_lr.predict_proba(df)[0][1])

        # Prepare context with formatted input data for display
        display_data = {
            'country': input_data['country'],
            'university': input_data['university'],
            'student_merit_score': f"{student_merit_score:.2f}",
            'education_level': input_data['education_level'],
            'field_of_study': input_data['field_of_study'],
            'work_experience_years': f"{input_data['work_experience_years']:.1f}",
            'requires_job_training': input_data['requires_job_training'],
            'job_offer_status': input_data['job_offer_status'],
            'financial_status': input_data['financial_status'],
            'annual_income': f"{input_data['annual_income']:.2f}",
            'scholarship_status': input_data['scholarship_status'],
            'GPA': f"{input_data['GPA']:.2f}",
            'research_publications': input_data['research_publications'],
            'previous_visa_rejections': input_data['previous_visa_rejections'],
            'english_proficiency': f"{input_data['english_proficiency']:.1f}",
            'country_approval_rate': f"{input_data['country_approval_rate']:.2f}"
        }
        print(display_data, prediction, predicted_status, confidence)
        return render(request, 'prediction_result.html', {
            'prediction': predicted_status,
            'confidence': f"{confidence:.1f}%",
            'input_data': display_data,
            'raw_prediction': f"{prediction:.4f}"
        })

    except Exception as e:
        print(e)
        return render(request, 'index.html', {'error': str(e)})


def index(request):
    return render(request, 'index.html')


def predict(request):
    if request.method == 'POST':
        return predict_visa_status(request)
    return render(request, 'index.html')
