import numpy as np
from bottle import run, route, static_file
# from flask import Flask, request, jsonify, render_template
import pickle

# app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))
root_path = '/Users/swetha.tanamala/Desktop/Projects/Personal/Projects/'\
            'model_deployment_in_class1/predict_sales/templates'

@route('/')
def home():
    return static_file('index.html', root_path)


# @app.route('/predict', methods=['POST'])
# def predict():

#     int_features = [int(x) for x in request.form.values()]
#     final_features = [np.array(int_features)]
#     prediction = model.predict(final_features)

#     output = round(prediction[0], 2)

#     return static_file('index.html', prediction_text='Sales should be $ {}'.format(output))


# @route('/results', methods=['POST'])
# def results():

#     data = request.get_json(force=True)
#     prediction = model.predict([np.array(list(data.values()))])

#     output = prediction[0]
#     return jsonify(output)


if __name__ == "__main__":
    run(host='localhost', port=8000, debug=True)