import numpy as np
import pickle
from flask import Flask, request, render_template

app = Flask(__name__)

# Load the pickle model
with open("LinearRegression_sales.pkl", "rb") as pickle_in:
    classifier = pickle.load(pickle_in)

@app.route('/')
def home():
    return render_template('index_1.html', prediction=None)

@app.route('/predict', methods=["POST"])
def predict_note_authentication():
    try:
        # Get form inputs
        TV = float(request.form['TV'])
        Radio = float(request.form['Radio'])
        Newspaper = float(request.form['Newspaper'])
        
        # Prepare input for the model
        input_data = np.array([[TV, Newspaper, Radio]])  # Ensure input is 2D
        prediction = classifier.predict(input_data)[0]  # Get the prediction
        
        return render_template('index_1.html', prediction=round(prediction, 2))
    except Exception as e:
        return render_template('index_1.html', prediction=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
