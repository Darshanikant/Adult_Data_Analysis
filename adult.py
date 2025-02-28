import pandas as pd



data=pd.read_csv("adult.csv")

# Dictionary of new column names with index positions
new_column_names = {
    0: "Age",
    1: "Work_Class",
    2: "fnlwgt",
    3: "Education",
    4: "Education_digit",
    5: "Marital_status",
    6: "Occupation",
    7: "Relation_Family",
    8: "Race",
    9: "Sex",
    10: "Capital_gain",
    11: "Capital_loss",
    12: "Work_Hr_week",
    13: "Country",
    14: "Salary"
}

# Rename columns using a loop
for index, new_name in new_column_names.items():
    data.columns.values[index] = new_name

# as we dont need the column of fnlwgt and education degit lets drop it
data.columns = data.columns.str.strip()  # Remove hidden spaces
data.drop(columns=["fnlwgt", "Education_digit"], inplace=True)

from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
categorical_cols = ["Work_Class", "Education", "Marital_status", "Occupation", 
                    "Relation_Family", "Race", "Sex", "Country", "Salary"]

for col in categorical_cols:
    data[col] = encoder.fit_transform(data[col])  # Convert to numeric


X = data.drop(columns=["Salary"])  # Features
y = data["Salary"]  # Target variable


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)



from sklearn.metrics import accuracy_score, confusion_matrix


import xgboost as xgb

# Train XGBoost model
xgb_model = xgb.XGBClassifier()
xgb_model.fit(X_train, y_train)

# Predict
y_pred_xgb = xgb_model.predict(X_test)

# Accuracy
xgb_accuracy = accuracy_score(y_test, y_pred_xgb)
print(f"XGBoost Model Accuracy: {xgb_accuracy*100:.2f}%")

new_data = [[35, 1, 10, 2, 1, 3, 1, 1,5000, 0, 40, 10]]  # Replace with actual values
prediction = xgb_model.predict(new_data)

if prediction[0] == 1:
    print("Predicted Income: >50K")
else:
    print("Predicted Income: <=50K")

# pickel the code
import pickle
filename='Salary prediction.pkl'
with open(filename,"wb") as file:
    pickle.dump(xgb_model,file)
print('Model has been picked saved in Salary prediction.pkl ')

