# Importing necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Load the dataset
mh_data = pd.read_csv("synthetic_depression_data.csv")

# Selecting relevant features and target variable
features = ['age', 'sex', 'income_t1', 'employment_t1','stress_t1', 'mh_score_t1']
target = 'depression_t1'
X = mh_data[features]
y = mh_data[target]

# Handling missing values (if any)
imputer = SimpleImputer(strategy='most_frequent')
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# Label Encoding for categorical variables
label_encoder = LabelEncoder()
X['sex'] = label_encoder.fit_transform(X['sex'])
X['employment_t1'] = label_encoder.fit_transform(X['employment_t1'])
X['stress_t1'] = label_encoder.fit_transform(X['stress_t1'])

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training the Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Making predictions on the test set
y_pred = model.predict(X_test)

# Evaluating the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Write predictions to a CSV file
predictions_df = pd.DataFrame({'Actual_Depression': y_test, 'Predicted_Depression': y_pred})
predictions_df.to_csv('predictions.csv', index=False)
print("Predictions written to 'predictions.csv' file successfully.")

