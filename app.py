from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

# Load the trained model
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

# Initialize the Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get form data
        age = float(request.form['age'])
        passenger_id = int(request.form['passenger_id'])
        pclass = int(request.form['pclass'])
        
        # Convert "Male"/"Female" to 1/0
        sex = 1 if request.form['sex'] == "Male" else 0
        
        family_size = int(request.form['family_size'])
        fare = float(request.form['fare'])
        embarked = int(request.form['embarked'])

        # Scale input features (ensure they are scaled as per your training)
        input_features = np.array([[age, passenger_id, pclass, sex, family_size, fare, embarked]])

        # Make a prediction
        prediction = model.predict(input_features)

        # Render the prediction result on the page
        return render_template('index.html', prediction_text=f'Predicted Survival: {"Survived" if prediction[0] == 1 else "Did Not Survive"}')

if __name__ == "__main__":
    app.run(debug=True)