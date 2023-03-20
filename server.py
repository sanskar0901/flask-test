# Import libraries
import numpy as np
from flask import Flask, request, jsonify
import pickle
from flask_cors import CORS
app = Flask(__name__)
CORS(app)
# Load the model

model = pickle.load(open('../Fertilizer-Recommendation-System/fertilizer.pkl', 'rb'))
modelCropPredict=pickle.load(open('../Crop-Recommendation-System/RandomForest.pkl','rb'))
@app.route('/', methods = ['GET'])
def home():
    return 'Hello World'

@app.route('/croppredict', methods = ['POST'])
def predictrop():
    # Get the data from the POST request.
    data = request.get_json(force = True)
    # Make prediction using model loaded from disk as per the data.
    prediction = modelCropPredict.predict(np.array([[data['N'],data['P'],data['K'],data['temp'],data['humid'],	data['ph'],	data['rain']]]))
    # Take the first value of prediction
    output = prediction[0]
    return jsonify(output)
@app.route('/fertilizer', methods = ['POST'])
def predictFertilizer():
    # Get the data from the POST request.
    # data = request.get_json(force = True)
    # Make prediction using model loaded from disk as per the data.
    prediction = model.predict(np.array([[29,	52,	45,	24,	0,	19,	1,	7]]))
    # Take the first value of prediction
    output = prediction[0]
    return jsonify(output)
if __name__ == '__main__':
    app.run(port = 5000, debug = True)
