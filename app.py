from flask import Flask, render_template, request
import sklearn
import pickle
import numpy as np

app = Flask(__name__)

@app.route('/')
def hello():
    return render_template('index.html')

@app.route('/prediction', methods = ['GET','POST'])
def prediction():
    print('The scikit-learn version is {}.'.format(sklearn.__version__))
    if request.method == 'POST':
        category = request.form['category']
        style = request.form['style']
        saleschannel = request.form['saleschannel']
        month = request.form['month']
        qty = request.form['qty']
        encoder1 = pickle.load(open('encoder1.pkl', 'rb'))
        e_category =  encoder1.transform([category])
        encoder2 = pickle.load(open('encoder2.pkl', 'rb'))
        e_style =  encoder2.transform([style])
        encoder3 = pickle.load(open('encoder3.pkl', 'rb'))
        e_saleschannel =  encoder3.transform([saleschannel])
        encoder4 = pickle.load(open('encoder4.pkl', 'rb'))
        e_month =  encoder4.transform([month])
        scaler1 = pickle.load(open('scaler1.pkl','rb'))
        Xscaled = scaler1.transform([[category,style,saleschannel,month,qty]])
        model_rf = pickle.load(open('model_rf.pkl','rb'))
        sales_amount = model_rf.predict(Xscaled)
    return render_template('prediction.html', sales_amount = sales_amount)



if __name__ == '__main__':
    app.debug = True
    app.run(port=15000)
    
