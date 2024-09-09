import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import streamlit as st

# Load the dataset
data = pd.read_csv('creditcard.csv')

# Separate legitimate and fraudulent transactions
legit = data[data.Class == 0]
fraud = data[data.Class == 1]

# Undersample legitimate transactions to balance the classes
legit_sample = legit.sample(n=len(fraud), random_state=2)
data_set = pd.concat([legit_sample, fraud], axis=0)

# Split data into features and target
X = data_set.drop(columns="Class", axis=1)
y = data_set["Class"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=2)

# Train the logistic regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Evaluate model performance on the training set
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(y_train, X_train_prediction)
print('Accuracy on Training data :', training_data_accuracy)
print('Training Classification Report:\n', classification_report(y_train, X_train_prediction))

# Evaluate model performance on the test set
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(y_test, X_test_prediction)
print('Accuracy on Test data :', test_data_accuracy)
print('Test Classification Report:\n', classification_report(y_test, X_test_prediction))

# Create the Streamlit app
st.title("Credit Card Fraud Detection Model")
st.write("Enter the following features to check if the transaction is legitimate or fraudulent:")

# Create input fields for user to enter feature values
input_df = st.text_input('Input All features (comma-separated)')
if input_df:
    input_df_lst = input_df.split(',')

    # Clean up input values: strip whitespace and remove quotes
    cleaned_input = [value.strip().replace('"', '').replace("'", "") for value in input_df_lst]

    try:
        # Convert input to numpy array and reshape it
        features = np.array(cleaned_input, dtype=np.float64).reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(features)
        
        # Display result
        if prediction[0] == 0:
            st.write("Legitimate transaction")
        else:
            st.write("Fraudulent transaction")
    except Exception as e:
        st.write("Error in input or prediction:", str(e))
