# **Deploy Machine Learning Model WIth Flask and Deta**

**A step-by-step guide to building a credit card fraud detection machine learning model using scikit-learn RandomForestClassifier, save, package, and deploy the model using Flask and deta.sh.** 


#### **1). Import the necessary libraries and load the data:**

```python
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

df = pd.read_csv('creditcard.csv') 
```

#### **Preprocess the data by scaling the features and splitting the data into training and testing sets:**

```python
from sklearn.preprocessing import StandardScaler

X = df.drop('Class', axis=1)
y = df['Class']

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 
``` 

#### **Build and train the RandomForestClassifier mode.**

```python
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train) 
``` 


