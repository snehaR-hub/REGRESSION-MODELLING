'''Stepwise Regression Using Python
Stepwise Regression is a method of selecting the most significant variables for a regression model. It involves adding or removing predictors from the model based on statistical criteria, such as the Akaike Information Criterion (AIC), p-values, or Adjusted R². The goal is to identify the most important predictors for predicting the dependent variable.'''

'''1. Install Required Libraries
You need the following libraries for this task:

pandas for data manipulation.
numpy for numerical computations.
statsmodels for regression modeling.
sklearn for loading the dataset.
You can install these libraries using pip if you don't have them:

bash
Copy code
pip install pandas numpy statsmodels scikit-learn
2. Import Libraries and Load the Dataset'''

import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

# Step 1: Load the Boston dataset
boston = load_boston()

# Create a DataFrame
df = pd.DataFrame(boston.data, columns=boston.feature_names)
df['Price'] = boston.target

# Check the first few rows of the dataset
df.head()

'''3. Prepare the Data
We need to split the dataset into training and testing sets. In stepwise regression, we will try to identify the most significant predictors that are useful in predicting the target variable (Price).'''

# Step 2: Split the data into train and test sets
X = df.drop('Price', axis=1)
y = df['Price']

# Split into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Add a constant (intercept) to the model
X_train = sm.add_constant(X_train)
X_test = sm.add_constant(X_test)


