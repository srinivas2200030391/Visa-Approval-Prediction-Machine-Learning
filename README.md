# Visa Approval Prediction System

This project implements a machine learning-based system for predicting visa application outcomes using multiple models including Neural Networks, Random Forest, Decision Tree, and Logistic Regression.

## 🌟 Features

- Multiple ML model support (Neural Networks, Random Forest, Decision Trees, Logistic Regression)
- Web-based interface for predictions
- Comprehensive input validation
- Confidence score calculation
- Detailed prediction results display

## 📋 Prerequisites

- Python 3.8+
- Django 3.2+
- TensorFlow 2.x
- scikit-learn
- NumPy
- Pandas

## 🛠️ Installation

1. Clone the repository:
```bash
git clone https://github.com/srinivas2200030391/Visa-Approval-Prediction-Machine-Learning.git
cd Visa-Approval-Prediction-Machine-Learning
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

## 📁 Project Structure

```
mlProject/
├── requirements.txt
├── manage.py
├── mlProject/
│   ├── __init__.py
│   ├── settings.py
│   ├── urls.py
|   ├── views.py
│   └── wsgi.py
├── templates/
│   ├── index.html
│   └── prediction_result.html
└── models/
    ├── feed_forward_neural_networks.h5
    ├── random_forest_model.pkl
    ├── decision_tree_model.pkl
    └── logistic_regression_model.pkl
```

## 🚀 Running the Application

1. Start the development server:
```bash
python manage.py runserver
```

2. Access the application at: `http://127.0.0.1:8000`

## 📝 Input Parameters

The system accepts the following parameters for prediction:

- Country
- University
- GPA (0-4 scale)
- Education Level
- Field of Study
- Work Experience (years)
- Job Training Requirement
- Job Offer Status
- Financial Status
- Annual Income
- Scholarship Status
- Research Publications
- Previous Visa Rejections
- English Proficiency Score
- Country Approval Rate

## 🎯 Models

The system includes four different machine learning models:

1. Neural Networks (Deep Learning)
2. Random Forest Classifier
3. Decision Tree Classifier
4. Logistic Regression

## 📊 Prediction Output

The system provides:
- Predicted visa status (Approved/Denied)
- Confidence score
- Detailed breakdown of input parameters
- Raw prediction probability

## 🔒 Important Note

The predictions made by this system are for educational purposes only and should not be used as a substitute for official visa application processes or legal advice.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👥 Authors

- Srinivas Kommirisetty
- Contributors welcome!

## 📬 Contact

For any queries or suggestions, please open an issue in the GitHub repository.
