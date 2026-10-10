
import joblib

# Load model
model = joblib.load("lung_health_ai_model.pkl")

print("Lung Health AI Model Loaded Successfully!")
print("Number of features expected:", model.n_features_in_)

import joblib

# Load model
model = joblib.load("lung_health_ai_model.pkl")

print("===================================")
print("     LUNG HEALTH AI")
print("===================================")

# 1. Gender
gender = input("Enter gender (M/F): ").upper()
gender = 1 if gender == "M" else 0

# 2. Age
age = int(input("Enter age: "))

# 3-15: Symptoms / habits
smoking = int(input("Smoking (1=No, 2=Yes): "))
yellow_fingers = int(input("Yellow Fingers (1=No, 2=Yes): "))
anxiety = int(input("Anxiety (1=No, 2=Yes): "))
peer_pressure = int(input("Peer Pressure (1=No, 2=Yes): "))
chronic_disease = int(input("Chronic Disease (1=No, 2=Yes): "))
fatigue = int(input("Fatigue (1=No, 2=Yes): "))
allergy = int(input("Allergy (1=No, 2=Yes): "))
wheezing = int(input("Wheezing (1=No, 2=Yes): "))
alcohol_consuming = int(input("Alcohol Consuming (1=No, 2=Yes): "))
coughing = int(input("Coughing (1=No, 2=Yes): "))
shortness_of_breath = int(input("Shortness of Breath (1=No, 2=Yes): "))
swallowing_difficulty = int(input("Swallowing Difficulty (1=No, 2=Yes): "))
chest_pain = int(input("Chest Pain (1=No, 2=Yes): "))

# Create input
input_data = [[
    gender,
    age,
    smoking,
    yellow_fingers,
    anxiety,
    peer_pressure,
    chronic_disease,
    fatigue,
    allergy,
    wheezing,
    alcohol_consuming,
    coughing,
    shortness_of_breath,
    swallowing_difficulty,
    chest_pain
]]

# Prediction
prediction = model.predict(input_data)

print("\n===================================")

if prediction[0] == 1:
    print("Prediction: Higher risk detected")
else:
    print("Prediction: Lower risk detected")

print("===================================")
prediction = model.predict(input_data)

print("Prediction:", prediction)
prediction = model.predict(input_data)
prediction = model.predict(input_data)

