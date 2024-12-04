# House Price Prediction using Multiple Linear Regression

This project demonstrates a complete pipeline for predicting house prices using multiple linear regression. The implementation includes:
- Manual training of a regression model.
- Prediction using saved model coefficients.
- Evaluation of model performance on a test dataset.

## Features
- **User Input Prediction**: Input house details and get a predicted price.
- **Model Training**: Manually computes regression coefficients using the Normal Equation.
- **Model Evaluation**: Evaluates performance with metrics like RMSE and R-squared.
- **Visualization**: Plots actual vs predicted house prices for the test set.

## Files and Directories
- `house_price_regression_dataset.csv`: Dataset for training and testing the model.
- `beta_coefficients.npy`: Saved model coefficients.
- `Predict.py`: Script for user input-based price prediction.
- `Test.py`: Script to evaluate model performance on the test set.
- `Train.py`: Script to train the model and save coefficients.
- `Requirements.txt`: List of required Python libraries.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/house-price-prediction.git
   ```
2. Navigate to the project directory:
   ```bash
   cd house-price-prediction
   ```
3. Install the required Python libraries:
   ```bash
   pip install -r Requirements.txt
   ```

## Usage

### 1. Train the Model
Run the `Train.py` script to train the regression model and save the coefficients:
```bash
python Train.py
```

### 2. Predict House Price
Run the `Predict.py` script and provide input values to predict the price:
```bash
python Predict.py
```

### 3. Evaluate the Model
Run the `Test.py` script to evaluate the model on the test set and visualize the results:
```bash
python Test.py
```

## Example
1. **Training**:
   Output includes the computed coefficients, RMSE, and R-squared values for the training set.

2. **Prediction**:
   Input example:
   ```
   Square Footage: 2000
   Number of Bedrooms: 3
   Number of Bathrooms: 2
   Year Built: 2010
   Lot Size: 0.5
   Garage Size: 2
   Neighborhood Quality (1-10): 8
   ```
   Output example:
   ```
   Predicted House Price: $350,000.00
   ```

3. **Evaluation**:
   Outputs testing RMSE, R-squared score, and displays a scatter plot of actual vs predicted house prices.

## Dependencies
- Python 3.x
- numpy
- pandas
- scikit-learn
- matplotlib

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgements
- Dataset: https://www.kaggle.com/datasets/prokshitha/home-value-insights/data
- Inspiration: Implementation of multiple linear regression using the Normal Equation.

