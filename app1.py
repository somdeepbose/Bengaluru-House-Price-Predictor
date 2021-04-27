from flask import Flask, render_template, request
import pickle
import numpy as np
app1 = Flask(__name__)

area_type_encoder = pickle.load(open('models/area_type_encoder_1.pkl', 'rb'))
#availability_encoder = pickle.load(open('models/availability_encoder_1.pkl', 'rb'))

ohe = pickle.load(open('models/oh_encoder.pkl', 'rb'))
#model_lin = pickle.load(open('models/model_lin.pkl', 'rb'))
model_rfr = pickle.load(open('models/model_rfr.pkl', 'rb'))

#area_type = ['Super built-up  Area', 'Plot  Area', 'Built-up  Area',
 #      'Carpet  Area']
#availability = ['Available Soon', 'Available Now']

@app1.route('/')
def index():
    return render_template('index.html')#, area_type = area_type)#, availability = availability)

@app1.route('/predict', methods=['POST'])
def predict():
    area_type = request.form.get('area_type')
    #availability = request.form.get('availability')
    size = request.form.get('size')
    total_sqft = request.form.get('total_sqft')
    bath = request.form.get('bath')

    area_type = area_type_encoder.transform(np.array([area_type]))
   # availability = availability_encoder.transform(np.array([availability]))

    X = np.array([size, total_sqft, bath]).reshape(1, 3)
    X_trans = np.array([area_type])

    print(X)

    X_trans = ohe.transform(X_trans).toarray()

    X = np.hstack((X, X_trans))

    prediction = model_rfr.predict(X) *100000

    print(X_trans.shape)
    print(prediction)

    return render_template('index.html', prediction=prediction)
if __name__=="__main__":
    app1.run(debug=True, host='0.0.0.0' port=9696)