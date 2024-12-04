import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Load the saved coefficients
beta = np.load("beta_coefficients.npy")

# Load and preprocess the data
data = pd.read_csv('house_price_regression_dataset.csv')
data = data.fillna(data.mean())  

# Select features and target
X = data[['Square_Footage', 'Num_Bedrooms', 'Num_Bathrooms', 'Year_Built', 
          'Lot_Size', 'Garage_Size', 'Neighborhood_Quality']].values
y = data['House_Price'].values

# Split data into training and testing sets
_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Add intercept term to the test set
X_test = np.c_[np.ones(X_test.shape[0]), X_test]

# Predict on the test set
y_test_pred = X_test @ beta

# Calculate evaluation metrics
test_mse = np.mean((y_test - y_test_pred) ** 2)
ss_total_test = np.sum((y_test - np.mean(y_test)) ** 2)
ss_residual_test = np.sum((y_test - y_test_pred) ** 2)
test_r_squared = 1 - (ss_residual_test / ss_total_test)

# Print evaluation metrics
print("Testing Root Mean Squared Error:", np.sqrt(test_mse))
print("Testing R^2 Score:", test_r_squared)

# Plot actual vs predicted prices
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_test_pred, alpha=0.5, color='b')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2) 
plt.xlabel('Actual House Prices')
plt.ylabel('Predicted House Prices')
plt.title('Actual vs Predicted House Prices (Test Set)')
plt.show()
