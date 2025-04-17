# Disease Prediction and Medical Recommendation System 🩺

## Introduction
The Disease Prediction and Medical Recommendation System leverages machine learning techniques to predict diseases based on user-entered symptoms. It provides recommendations for medications, diets, and workouts tailored to specific diseases. The project uses a dataset from Kaggle comprising symptoms, diseases, medications, and other medical attributes to train and evaluate machine learning models.

## Project Overview
The project files are organized into the following directories:

### 1. kaggle_dataset
- `description.csv`: Descriptions of diseases.
- `diets.csv`: Recommended diets for diseases.
- `medications.csv`: Medications prescribed for diseases.
- `precautions_df.csv`: Precautions to be taken for diseases.
- `Symptom-severity.csv`: Severity of symptoms.
- `symptoms_df.csv`: Symptoms with corresponding disease labels.
- `Training.csv`: Dataset for training machine learning models.
- `workout_df.csv`: Recommended workouts for diseases.

### 2. model
- `RandomForest.pkl`: Trained Random Forest model for disease prediction.

### 3. templates
- `index.html`: Frontend interface for the Disease Prediction System.
- `signup.html`: Signup Page
- `login.html`: Login Page to let user login afer signup.

### 4. static
- `bgCover.jpg`, `img.png` : Images utilized in the frontend webpage.

### 5. screenshots
- Includes screenshots of the project.


## How to Run the Project
To run the Disease Prediction and Medical Recommendation System:

1. **Install required Python libraries:**
   ```bash
   pip install pandas scikit-learn flask ast numpy fuzzywuzzy pickle

2. **Navigate to the project directory:**
   ```bash
   cd ML_project

3. **Start the Flask application:**
   ```bash
   python3 main.py

4. **Access the web interface in your browser at:**
   ```bash
   http://localhost:5000

## Team Members

- Sachin Kumar Sharma - 2152055
- Ritesh Raj - 2152064
- Bikash Kumar Yadav - 2152058
- Md. Faisal Kalim - 2152060
