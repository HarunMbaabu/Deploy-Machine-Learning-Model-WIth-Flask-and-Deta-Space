# **Deploy Machine Learning Model WIth Flask and Deta**

**A step-by-step guide to building a credit card fraud detection machine learning model using scikit-learn RandomForestClassifier, save, package, and deploy the model using Flask and deta.sh.** 


#### **1). Import the necessary libraries and load the data:**

```python
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

df = pd.read_csv('creditcard.csv') 
```

#### **2). Preprocess the data by scaling the features and splitting the data into training and testing sets:**

```python
from sklearn.preprocessing import StandardScaler

X = df.drop('Class', axis=1)
y = df['Class']

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 
``` 

#### **3). Build and train the RandomForestClassifier mode.**

```python
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train) 
``` 

#### **4).Evaluate the performance of the model on the test set:**

```python
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

y_pred = model.predict(X_test)

print("Accuracy Score:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

``` 

#### **5).Save the trained model using joblib:** 

```python
from joblib import dump

dump(model, 'Credit_Card_model.joblib') 
``` 

#### **6). Package the model into a Flask app:** 


```python 
from flask import Flask, request, jsonify
from joblib import load

app = Flask(__name__)
model = load('Credit_Card_model.joblib')

@app.route('/', methods=['POST'])
def predict():
    data = request.get_json()
    data = scaler.transform([data])
    prediction = model.predict(data)
    return jsonify({'prediction': str(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True)

``` 
Run the Flask app and test the API endpoint:

```python
import requests

data = {
    "Time": 1000.23,
    "V1": -3.30016,
    "V2": -0.67606,
    "V3": 1.98869,
    "V4": 0.30351,
    "V5": -2.34785,
    "V6": -0.27872,
    "V7": -0.64068,
    "V8": 1.17911,
    "V9": -0.28575,
    "V10": -2.17152,
    "V11": -0.69616,
    "V12": -0.05025,
    "V13": 0.85212,
    "V14": -0.2136,
    "V15": -0.83305,
    "V16": -0.05909,
    "V17": -1.12053,
    "V18": 0.23474,
    "V19": -0.31713,
    "V20": -0.10204,
    "V21": -0.3883,
    "V22": -0.80726,
    "V23": -0.21236,
    "V24": 0.53682,
    "V25": 0.47959,
    "V26": -0.32223,
    "V27": -0.01846,
    "V28":
```