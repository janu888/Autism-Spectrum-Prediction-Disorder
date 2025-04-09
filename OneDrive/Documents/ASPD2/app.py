from flask import Flask, request, render_template
import pickle
import pandas as pd

app = Flask(__name__)

# Load model and accuracy
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("accuracy.txt", "r") as f:
    accuracy = f.read()

gender_map = {"Male": 0, "Female": 1}
jaundice_map = {"Yes": 1, "No": 0}
autism_map = {"Yes": 1, "No": 0}
prediction_map = {1: "Yes", 0: "No"}

features = [
    "A1_Score", "A2_Score", "A3_Score", "A4_Score", "A5_Score",
    "A6_Score", "A7_Score", "A8_Score", "A9_Score", "A10_Score",
    "gender", "jundice", "austim"
]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form
    try:
        gender = gender_map.get(data.get("gender"))
        jaundice = jaundice_map.get(data.get("jaundice"))
        autism = autism_map.get(data.get("autism"))

        if None in [gender, jaundice, autism]:
            return "Invalid input values"

        patient_data = pd.DataFrame([[ 
            int(data["A1_Score"]), int(data["A2_Score"]), int(data["A3_Score"]),
            int(data["A4_Score"]), int(data["A5_Score"]), int(data["A6_Score"]),
            int(data["A7_Score"]), int(data["A8_Score"]), int(data["A9_Score"]),
            int(data["A10_Score"]), gender, jaundice, autism
        ]], columns=features)

        prediction_numeric = model.predict(patient_data)[0]
        prediction_text = prediction_map[prediction_numeric]

        # Confidence score for individual user
        confidence_score = model.predict_proba(patient_data)[0][prediction_numeric] * 100
        confidence_score = f"{confidence_score:.2f}"

        return render_template('result.html',
                               prediction=prediction_text,
                               accuracy=accuracy,
                               confidence=confidence_score)

    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == "__main__":
    app.run(debug=True)
