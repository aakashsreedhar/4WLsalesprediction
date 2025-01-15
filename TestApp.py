import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from math import sqrt

# Streamlit app
def main():
    st.title("Model Evaluation App")

    # Load pre-trained model from uploaded .py file
    st.header("Upload Pre-trained Model File")
    model_file = st.file_uploader("Upload a Python file with a trained model", type=["py"])

    if model_file is not None:
        exec(model_file.read(), globals())  # Load the model file dynamically
        st.success("Model file loaded successfully!")

    # Allow user to specify the dependent variable
    st.header("Specify Dependent Variable")
    dependent_var = st.text_input("Enter the name of the dependent variable (e.g., 'cnt')", "cnt")

    # Upload training data
    st.header("Upload Training Data")
    train_file = st.file_uploader("Upload a CSV file for training", type=["csv"])

    if train_file is not None:
        train_data = pd.read_csv(train_file)
        st.write("Training Data Preview:")
        st.write(train_data.head())

        if dependent_var in train_data.columns:
            y_train = train_data[dependent_var]
            X_train = train_data.drop(columns=[dependent_var, 'CustomerID'], errors='ignore')

            # Train the Linear Regression model
            model = LinearRegression()
            model.fit(X_train, y_train)

            # Calculate R^2 and RMSE
            predictions = model.predict(X_train)
            r2 = r2_score(y_train, predictions)
            rmse = sqrt(mean_squared_error(y_train, predictions))

            st.subheader("Training Results")
            st.write(f"R-squared: {r2:.4f}")
            st.write(f"RMSE: {rmse:.4f}")
        else:
            st.error(f"The training data must contain the '{dependent_var}' column as the dependent variable.")

    # Upload test data
    st.header("Upload Test Data")
    test_file = st.file_uploader("Upload a CSV file for testing", type=["csv"], key="test")

    if test_file is not None:
        test_data = pd.read_csv(test_file)
        st.write("Test Data Preview:")
        st.write(test_data.head())

        if dependent_var in test_data.columns:
            y_test = test_data[dependent_var]
            X_test = test_data.drop(columns=[dependent_var, 'CustomerID'], errors='ignore')

            # Ensure model is trained before testing
            if train_file is not None:
                # Predict and calculate RMSE for test data
                test_predictions = model.predict(X_test)
                test_rmse = sqrt(mean_squared_error(y_test, test_predictions))

                st.subheader("Test Results")
                st.write(f"Test RMSE: {test_rmse:.4f}")
            else:
                st.error("Please upload training data to train the model before testing.")
        else:
            st.error(f"The test data must contain the '{dependent_var}' column as the dependent variable.")

if __name__ == "__main__":
    main()