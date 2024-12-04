import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def manual_multiple_linear_regression(X, y):
    X = np.c_[np.ones(X.shape[0]), X]  # Add intercept term
    X_transpose = X.T
    beta = np.linalg.inv(X_transpose @ X) @ X_transpose @ y  # Calculate coefficients
    y_pred = X @ beta
    mse = np.mean((y - y_pred) ** 2)
    ss_total = np.sum((y - np.mean(y)) ** 2)
    ss_residual = np.sum((y - y_pred) ** 2)
    r_squared = 1 - (ss_residual / ss_total)
    return beta, mse, r_squared

# Load and preprocess the data
data = pd.read_csv('house_price_regression_dataset.csv')
data = data.fillna(data.mean())  # Fill missing values

# Select features and target
X = data[['Square_Footage', 'Num_Bedrooms', 'Num_Bathrooms', 'Year_Built', 
          'Lot_Size', 'Garage_Size', 'Neighborhood_Quality']].values
y = data['House_Price'].values

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model on the training set
beta, train_mse, train_r_squared = manual_multiple_linear_regression(X_train, y_train)

# Save model coefficients to a file
np.save("beta_coefficients.npy", beta)

# Display results
print("Model coefficients:", beta)
print("Model coefficients saved as 'beta_coefficients.npy'.")
print("Training Root Mean Squared Error:", np.sqrt(train_mse))
print("Training R^2 Score:", train_r_squared)
