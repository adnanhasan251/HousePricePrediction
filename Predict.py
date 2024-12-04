import numpy as np

# Load the saved coefficients
beta = np.load("beta_coefficients.npy")

# Prompt the user to enter data for each feature
print("Please enter the following details to predict the house price:")

try:
    square_footage = float(input("Square Footage: "))
    num_bedrooms = int(input("Number of Bedrooms: "))
    num_bathrooms = int(input("Number of Bathrooms: "))
    year_built = int(input("Year Built: "))
    lot_size = float(input("Lot Size: "))
    garage_size = int(input("Garage Size: "))
    neighborhood_quality = int(input("Neighborhood Quality (e.g., rating from 1 to 10): "))

    # Construct the feature array with an intercept term
    X_user = np.array([1, square_footage, num_bedrooms, num_bathrooms, year_built, 
                       lot_size, garage_size, neighborhood_quality])

    # Make prediction
    predicted_price = X_user @ beta

    # Display prediction
    print(f"Predicted House Price: ${predicted_price:.2f}")

except ValueError:
    print("Invalid input! Please ensure all values are entered in the correct format.")
