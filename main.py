from app import load_dataset, explore_dataset

data = load_dataset("creditcard.csv")  


#Preprocess the data by scaling the features and splitting the data into training and testing sets:
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split 


X = data.drop('Class', axis=1)
y = data['Class']


scaler = StandardScaler()
X = scaler.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


 

#Build and train the RandomForestClassifier model:
from sklearn.ensemble import RandomForestClassifier  


model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)



#Evaluate the performance of the model on the test set:


from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

y_pred = model.predict(X_test)

print("Accuracy Score:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
 

#Save the trained model using joblib:

from joblib import dump
dump(model, 'Credit_Card_model.joblib')
