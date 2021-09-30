from flask import Flask, render_template, request
import pickle
import numpy as np
from sklearn.naive_bayes import GaussianNB


app = Flask(__name__)
model = pickle.load(open('MHP.pkl', 'rb'))




@app.route('/')
def man():
    return render_template('home.html')

@app.route('/Sahayak')
def home():
    return render_template('Sahayak.html')

@app.route('/predict', methods=['POST'])
def result():
    data1 = request.form['a']
    data2 = request.form['b']
    data3 = request.form['c']
    data4 = request.form['d']
    data5 = request.form['e']
    data6 = request.form['f']
    data7 = request.form['g']
    data8 = request.form['h']
    data9 = request.form['i']
    data10 = request.form['j']
    arr = np.array([[data1, data2, data3, data4,data5,data6,data7,data8,data9,data10]])
    pred = model.predict(arr)
    return render_template('after.html', data=pred)


if __name__ == '__main__':
    app.debug = True
    app.run()
    

