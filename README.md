# Linear Regression from Scratch

![Python](https://img.shields.io/badge/Python-3.x-blue.svg)

## Overview
This repository demonstrates the implementation of a linear regression model entirely from scratch in Python. It includes:

- **Custom Scaler**: Preprocesses the dataset by standardizing features.
- **Linear Regression Model**: Trains a regression model using gradient descent.
- **Configuration File**: Hyperparameters are stored in a modular configuration file.
- **Evaluation Metrics**: Evaluates performance using metrics like MSE, MAE, and R² Score.

## Getting Started

### Prerequisites
- Python 3.x installed on your machine.
- Basic understanding of machine learning concepts.

### Repository Structure

```plaintext
├── config.py               # Hyperparameter configuration file
├── custom_scaler.py        # Custom scaler for feature normalization
├── linear_regression_scratch.py  # Linear regression implementation
├── experimental_data.csv   # Example dataset for testing
├── train_linear_regression.ipynb # Jupyter Notebook for training and evaluation
```

### Installation
Clone this repository:
```bash
git clone https://github.com/username/linear-regression-from-scratch.git
cd linear-regression-from-scratch
```

## Usage

### 1. Configure Hyperparameters
Edit the `config.py` file to set the desired hyperparameters:
```python
config = {
    'learning_rate': 0.01,
    'epochs': 1000,
    'regularization': 'l2',  # Options: 'l1', 'l2', or None
    'reg_lambda': 0.1
}
```

### 2. Run the Jupyter Notebook
Launch the `train_linear_regression.ipynb` notebook to:

1. Load and preprocess the dataset (`experimental_data.csv`).
2. Train the linear regression model.
3. Evaluate the model using metrics like MSE, MAE, and R² Score.
4. Make predictions for new data points.

To start Jupyter Notebook:
```bash
jupyter notebook train_linear_regression.ipynb
```

### 3. Key Classes and Methods

#### `CustomScaler`
Preprocesses features by standardizing them to zero mean and unit variance.
- `fit`: Computes mean and standard deviation for each feature.
- `transform`: Scales features based on the computed values.
- `fit_transform`: Combines fitting and transforming.

#### `LinearRegressionScratch`
Implements linear regression using gradient descent.
- `fit`: Trains the model.
- `predict`: Makes predictions for new data.
- `mean_squared_error`, `r2_score`: Evaluates model performance.

### Example Code
```python
from linear_regression_scratch import LinearRegressionScratch
from custom_scaler import CustomScaler
from config import config

# Load and preprocess data
scaler = CustomScaler()
data = [[...], [...]]  # Example features
scaled_data = scaler.fit_transform(data)

# Train the model
model = LinearRegressionScratch(
    lr=config['learning_rate'],
    epochs=config['epochs'],
    regularization=config['regularization'],
    reg_lambda=config['reg_lambda']
)
model.fit(X_train, y_train)

# Evaluate the model
print("MSE:", model.mean_squared_error(y_test, model.predict(X_test)))
print("R² Score:", model.r2_score(y_test, model.predict(X_test)))

# Make predictions
predictions = model.predict([[...], [...]])  # Example new data
```
