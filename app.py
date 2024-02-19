from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

@app.route('/')
def hello():
    return render_template('index.html')

@app.route('/prediction', methods = ['GET','POST'])
def prediction():
    if request.method == 'POST':
        category = request.form['category']
        style = request.form['style']
        saleschannel = request.form['saleschannel']
        month = request.form['month']
        qty = request.form['qty']
        encoder = pickle.load(open('encoder.pkl', 'rb'))
        Xencoded =  encoder.fit_transform([category,style,saleschannel,month,qty])
        scaler = pickle.load(open('scaler.pkl','rb'))
        Xscaled = scaler.fit_transform([Xencoded])
        model_rf = pickle.load(open('model_rf.pkl','rb'))
        sales_amount = model_rf.predict([Xscaled])
    return render_template('prediction.html', sales_amount = sales_amount)


    


if __name__ == '__main__':
    app.debug = True
    app.run(port=15001)
    
