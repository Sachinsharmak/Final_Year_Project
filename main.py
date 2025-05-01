from flask import Flask, request, render_template, redirect, url_for, session, flash
from flask_pymongo import PyMongo
import numpy as np
import pandas as pd
from werkzeug.security import generate_password_hash, check_password_hash
import pickle
from fuzzywuzzy import process
import ast
import uuid
from datetime import datetime

app = Flask(__name__)
app.secret_key = "Generate a random secret key for your app"
app.config["MONGO_URI"] = "Enter the MongoDb Atlas URL"  
mongo = PyMongo(app)


users = mongo.db.users
predictions = mongo.db.predictions

DATA_DIR = "kaggle_dataset/" 
sym_des = pd.read_csv(DATA_DIR + "symptoms_df.csv")
precautions = pd.read_csv(DATA_DIR + "precautions_df.csv")
workout = pd.read_csv(DATA_DIR + "workout_df.csv")
description = pd.read_csv(DATA_DIR + "description.csv")
medications = pd.read_csv(DATA_DIR + 'medications.csv')
diets = pd.read_csv(DATA_DIR + "diets.csv")

dataset = pd.read_csv(DATA_DIR + "Training.csv")  

serious_diseases = ["Heart attack", "Stroke", "Paralysis", "Pneumonia", "Tuberculosis"]  

DEFAULT_PRECAUTIONS = "Please consult with a doctor immediately, it appears to be something critical. Do not ignore this."

doctors_database = [
    {
        "name": "Dr. Ritesh Raj",
        "disease": "Heart attack",
        "for_emergency": True,
        "hospital": "AIIMS Delhi",
        "contact": "+91-93239*****"
    },
    {
        "name": "Dr. Anjali Verma",
        "disease": "Stroke",
        "for_emergency": True,
        "hospital": "Fortis Hospital Mumbai",
        "contact": "+91-91234*****"
    },
    {
        "name": "Dr. Suresh Iyer",
        "disease": "Tuberculosis",
        "for_emergency": False,
        "hospital": "Apollo Hospital Chennai",
        "contact": "+91-99887*****"
    }
]




Random_Forest = pickle.load(open('model/RandomForest.pkl', 'rb'))

symptoms_dictionary = {symptom.replace('_', ' ').lower(): index for index, symptom in enumerate(dataset.columns[:-1])}


diseases_dict = {index: disease for index, disease in enumerate(sorted(dataset.prognosis.unique()))}

# Prediction functions
def predict_disease(patient_symptoms):
    symptom_vector = np.zeros(len(symptoms_dictionary))
    for symptom in patient_symptoms:
        try:
            symptom_vector[symptoms_dictionary[symptom]] = 1
        except KeyError:
            return None

    try:
        predicted_prognosis = Random_Forest.predict([symptom_vector])[0]
        return diseases_dict.get(predicted_prognosis, None)
    except Exception as e:
        print(f"Prediction error: {e}")
        return None

def correct_symptom_spelling(symptom, spell_threshold=80):
    try:
        closest_match = process.extractOne(symptom, symptoms_dictionary.keys())
        if closest_match:
            matched_symptom, score = closest_match
            return matched_symptom if score >= spell_threshold else None
        return None
    except Exception as e:
        print(f" Please Check the Spelling of the Symptoms: {e}")
        return None

def fetch_articles(disease, num_articles=3):
    try:
        from googlesearch import search
        search_query = f"{disease} medical condition treatment"
        articles = list(search(search_query, num_results=num_articles))
        return articles
    except ImportError:
        return ["Unable to load articles related to it. Please consult your healthcare provider for more information."]
    except Exception as e:
        print(f"Article not Found: {e}")
        return []

def personalized_workout(disease, age=None, fitness_level=None, other_conditions=None):
    if any(disease == d for d in serious_diseases):
        return f"For {disease}, it's crucial to consult your healthcare provider regarding exercise. Some options, upon medical approval, might include:"
    
    try:
        general_workout = workout[workout['disease'] == disease]['workout'].iloc[0] if not workout[workout['disease'] == disease].empty else None
        
        if general_workout:
            if other_conditions:
                additional_info = f"Note: Your other condition(s) ({', '.join(other_conditions)}) may impact exercise suitability."
                return f"{general_workout}\n\n{additional_info}"
            
            if age and age > 60:
                return f"{general_workout}\n\nNote: Since you're over 60, please make sure that these exercises are approved by your doctor."
                
            if fitness_level:
                return f"{general_workout}\n\nAdjust intensity based on your {fitness_level} fitness level."
                
            return general_workout
        else:
            return "No specific workout information is available for this condition. Please consult with your healthcare provider."
    except Exception as e:
        print(f"Workout advice error: {e}")
        return "Unable to provide workout recommendations. Please consult with your healthcare provider."

def get_medical_info(predicted_disease, age=None, patient_history=None, other_conditions=None):
    if not predicted_disease:
        return None, [], ["No disease prediction available."]
    
    serious_doctors = [doc for doc in doctors_database if 
                      (predicted_disease == doc.get('disease') and predicted_disease in serious_diseases) or 
                      (doc.get('for_emergency') and predicted_disease in serious_diseases)]
    
    flash_msg_list = []
    
    description_data = description[description['Disease'] == predicted_disease]
    precautions_data = precautions[precautions['Disease'] == predicted_disease]
    medications_data = medications[medications['Disease'] == predicted_disease]
    diets_data = diets[diets['Disease'] == predicted_disease]
    workout_data = workout[workout['disease'] == predicted_disease]
    
    try:
        related_articles = fetch_articles(predicted_disease)
        
        if not description_data.empty:
            info_dict = {
                'dis_des': description_data['Description'].iloc[0] if not description_data.empty else 
                          "Description Not available. Please consult a Doctor for an in-depth understanding.",
                
                'my_precautions': precautions_data[['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']].values.tolist()[0] 
                                if not precautions_data.empty else [DEFAULT_PRECAUTIONS],
                
                'medications': []
            }
            
            if not medications_data.empty:
                try:
                    med_data = medications_data['Medication'].iloc[0]
                    if isinstance(med_data, list):
                        info_dict['medications'] = med_data
                    elif isinstance(med_data, str):
                        info_dict['medications'] = ast.literal_eval(med_data)
                    else:
                        info_dict['medications'] = ["Medication information not available in the expected format."]
                except Exception as e:
                    info_dict['medications'] = ["Error processing medication data."]
            
            if not diets_data.empty:
                try:
                    diet_data = diets_data['Diet'].iloc[0]
                    if isinstance(diet_data, list):
                        info_dict['rec_diet'] = diet_data
                    elif isinstance(diet_data, str):
                        info_dict['rec_diet'] = ast.literal_eval(diet_data)
                    else:
                        info_dict['rec_diet'] = ["Diet information not available in the expected format."]
                except Exception as e:
                    info_dict['rec_diet'] = ["Error processing diet data."]
            else:
                info_dict['rec_diet'] = ["No specific diet recommendations available."]
            
            info_dict['workout'] = workout_data['workout'].iloc[0] if not workout_data.empty else "No specific workout information available."
            
            info_dict['serious_doctors'] = serious_doctors
            
            if any(disease_match == predicted_disease for disease_match in serious_diseases):
                flash_msg_list.append("You must immediately consult a doctor. Consider going to a nearby hospital or Doctor's clinic.")
            
            if patient_history and any(disease_match == predicted_disease for disease_match in serious_diseases):
                if other_conditions and any(predicted_disease in condition for condition in other_conditions):
                    if serious_doctors:
                        message = f"Dr. {serious_doctors[0].get('name')} recommends continuing previous treatments that worked well with your condition."
                        flash_msg_list.append(message)
            
            return info_dict, related_articles, flash_msg_list
        else:
            if patient_history and predicted_disease:
                return {'serious_doctors': [doctor for doctor in serious_doctors if doctor.get("for_emergency")]}, [], ["Patient data history not available. There may still be serious medical issues that need urgent review."]
            return None, [], ["No medical information available for the predicted condition."]
    
    except Exception as e:
        print(f"Medical info error: {e}")
        return None, [], [f"Error retrieving medical information: {str(e)}"]

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    if request.method == 'POST':
        symptoms = request.form.get('symptoms')
        if not symptoms:
            flash("Please enter symptoms.", 'error')
            return redirect(url_for('predict'))

        patient_symptoms = [s.strip() for s in symptoms.split(',')]
        corrected_symptoms = []
        
        for symptom in patient_symptoms:
            corrected_symptom = correct_symptom_spelling(symptom.lower())
            if corrected_symptom:
                corrected_symptoms.append(corrected_symptom)

        if not corrected_symptoms:
            message_for_invalid_symptoms = ["No valid symptoms entered. Please check your entries and try again."]
            user = users.find_one({'user_id': session['user_id']})
            return render_template('index.html', message=message_for_invalid_symptoms, username=user['username'])

        predicted_disease = predict_disease(corrected_symptoms)
        
        if predicted_disease is None:
            user = users.find_one({'user_id': session['user_id']})
            return render_template('index.html', 
                                  message=["No matching diseases found for submitted symptoms. Please provide additional details or consult a healthcare professional."],
                                  username=user['username'])

        user = users.find_one({'user_id': session['user_id']})
        past_history = None
        
        if request.form.get('past_history'):
            try:
                past_history_input = request.form.get('past_history').strip()
                if past_history_input:

                    past_history_clean = past_history_input.strip("[]").replace('"', '')
                    if past_history_clean:
                        past_history = [tuple(item.strip().split(':')) for item in past_history_clean.split(",")]
            except (ValueError, TypeError, AttributeError) as e:
                flash(f"Past history details improperly formatted. Please use format: ['treatment1:disease1', 'treatment2:disease2']", 'error')
                return redirect(url_for('predict'))
        
        other_conditions = None
        if request.form.get('other_conditions'):
            try:
                other_conditions_input = request.form.get('other_conditions').strip()
                if other_conditions_input:
                    other_conditions = [cond.strip() for cond in other_conditions_input.split(',')]
            except Exception as e:
                flash("Other conditions improperly formatted. Please enter as comma-separated values.", 'error')
                return redirect(url_for('predict'))
        
        age = None
        if request.form.get('age'):
            try:
                age = int(request.form.get('age'))
                if age <= 0 or age > 120:
                    flash("Please enter a valid age between 1 and 120.", 'error')
                    return redirect(url_for('predict'))
            except ValueError:
                flash("Age must be a number.", 'error')
                return redirect(url_for('predict'))
        
        fitness_level = request.form.get('fitness_level')
        
        info_dict, related_articles, flash_messages = get_medical_info(
            predicted_disease, 
            age=age, 
            patient_history=past_history,
            other_conditions=other_conditions
        )
        
        for msg in flash_messages:
            flash(msg, 'info')
        
        workout_advice = personalized_workout(
            predicted_disease,
            age=age,
            fitness_level=fitness_level,
            other_conditions=other_conditions
        )
        
        prediction_data = {
            'user_id': session['user_id'],
            'symptoms': corrected_symptoms,
            'disease': predicted_disease,
            'timestamp': datetime.now(),
            'age': age,
            'fitness_level': fitness_level,
            'past_history': past_history,
            'other_conditions': other_conditions
        }
        predictions.insert_one(prediction_data)
        
        return render_template(
            'results.html',
            username=user['username'],
            symptoms=corrected_symptoms,
            disease=predicted_disease,
            info=info_dict,
            articles=related_articles,
            workout_advice=workout_advice
        )
        
    user = users.find_one({'user_id': session['user_id']})
    return render_template('predict.html', username=user['username'])



@app.route('/history')
def history():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    user = users.find_one({'user_id': session['user_id']})
    if not user:
        return redirect(url_for('login'))
    
    user_predictions = list(predictions.find({'user_id': session['user_id']}))
    
    if user_predictions:
        print(f"First prediction document keys: {list(user_predictions[0].keys())}")
        print(f"First prediction document content: {user_predictions[0]}")
    else:
        print("No prediction documents found for this user")
    
    formatted_predictions = []
    for prediction in user_predictions:
        symptoms_str = ""
        if 'symptoms' in prediction:
            if isinstance(prediction['symptoms'], list):
                symptoms_str = ', '.join(prediction['symptoms'])
            else:
                symptoms_str = str(prediction['symptoms'])
        
        disease_value = prediction.get('disease', 
                      prediction.get('predicted_disease', 
                      prediction.get('prognosis', "Unknown")))
        
        timestamp = prediction.get('timestamp', datetime.now())
        
        formatted_predictions.append({
            'symptoms': symptoms_str,
            'predicted_disease': disease_value,
            'timestamp': timestamp
        })
    
    formatted_predictions.sort(key=lambda x: x['timestamp'], reverse=True)
    
    return render_template(
        'history.html', 
        username=user['username'], 
        predictions=formatted_predictions
    )
    
@app.route('/')
def index():
    if 'user_id' in session:
        user = users.find_one({'user_id': session['user_id']})
        if user:  
            return render_template('index.html', username=user['username'])
        else:
            session.clear() 
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        user = users.find_one({'username': username})
        
        if user and check_password_hash(user['password'], password):
            session['user_id'] = user['user_id']
            return redirect(url_for('index'))
        flash('Invalid username or password', 'error')
    
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        
        if password != confirm_password:
            flash('Passwords do not match', 'error')
            return render_template('register.html')
        
        if users.find_one({'username': username}):
            flash('Username already exists', 'error')
            return render_template('register.html')
        
        if users.find_one({'email': email}):
            flash('Email already in use', 'error')
            return render_template('register.html')
        
        hashed_password = generate_password_hash(password)
        user_id = str(uuid.uuid4())
        users.insert_one({
            'user_id': user_id,
            'username': username,
            'email': email,
            'password': hashed_password
        })
        
        flash('Registration successful! Please login to continue.', 'success')
        
        return redirect(url_for('login'))
    
    return render_template('register.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(debug=True)
