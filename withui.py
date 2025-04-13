from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

# Load the model
with open("LinearRegression_sales.pkl", "rb") as file:
    classifier = pickle.load(file)

@app.route('/')
def home():
    return render_template("index_2.html")

@app.route('/predict', methods=["POST"])
def predict_note_authentication():
    input_cols = ['TV', 'Newspaper', 'Radio']
    list1 = [float(request.form.get(i)) for i in input_cols]
    prediction = classifier.predict([list1])
    return render_template("index_2.html", prediction_text=f"Predicted Sales: {prediction[0]:.2f}")

if __name__ == '__main__':
    app.run(debug=True)
