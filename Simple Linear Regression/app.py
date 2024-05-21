from flask import Flask, render_template, request
import pickle
import numpy as np


app = Flask(__name__)

with open('Linear_Regression_pickled_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)
    
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    
    hours = float(request.form['hours'])
    
    predicted_score = model.predict(np.array([[hours]]))

    
    return render_template('result.html', hours=hours, predicted_score=predicted_score)


if __name__ == "__main__":
    app.run(debug=True)