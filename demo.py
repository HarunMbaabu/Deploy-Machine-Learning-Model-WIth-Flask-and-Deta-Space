from flask import Flask, jsonify, request  
 
import joblib  

app = Flask(__name__)    



model = joblib.load("Credit_Card_model.joblib")    


@app.route("/") 
def index():
	return "this is the test route for prediction go to /api/predict" 

@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.get_json()
    data = scaler.transform([data])
    prediction = model.predict(data)
    return jsonify({'prediction': str(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True)


