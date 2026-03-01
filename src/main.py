from flask import Flask, request, jsonify, render_template
import numpy as np
import joblib

app = Flask(__name__, template_folder='templates', static_folder='statics')

model = joblib.load('model.joblib')
feature_names = joblib.load('feature_names.joblib')


@app.route('/')
def home():
    return "Welcome to the Heart Disease Prediction API!"


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return render_template('predict.html')
    try:
        data = request.form
        features = [float(data[f]) for f in feature_names]
        input_data = np.array(features).reshape(1, -1)
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0]
        return jsonify({
            'prediction': int(prediction),
            'result': 'Heart Disease Detected' if prediction == 1 else 'No Heart Disease Detected',
            'confidence': round(float(max(probability)) * 100, 1)
        })
    except Exception as e:
        return jsonify({'error': str(e)})


@app.route('/feature-importance')
def feature_importance():
    rf = model.named_steps['clf']
    ranked = sorted(
        zip(feature_names, rf.feature_importances_.tolist()),
        key=lambda x: x[1],
        reverse=True
    )
    return jsonify([{'feature': f, 'importance': round(i, 4)} for f, i in ranked])


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=4000)
