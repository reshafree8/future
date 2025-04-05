from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load model
model = pickle.load(open('../backend/model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        cgpa = float(request.form['cgpa'])
        env = int(request.form['env'])
        exp = int(request.form['exp'])
        edu = int(request.form['edu'])
        risk = int(request.form['risk'])

        features = np.array([[cgpa, env, exp, edu, risk]])
        prediction = model.predict(features)[0]

        return render_template('index.html', prediction_text=f'Prediction: {prediction}')
    except:
        return render_template('index.html', prediction_text='Error in prediction. Check your input values.')

if __name__ == "__main__":
    app.run(debug=True)