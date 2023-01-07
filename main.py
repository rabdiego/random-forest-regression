# Importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Data preprocessing
data = pd.read_csv('Position_Salaries.csv')
X = data.iloc[:, 1:-1].values
y = data.iloc[:, -1].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Training the model
regressor = RandomForestRegressor()
regressor.fit(X_train, y_train)

# Predicting results
X_range = np.arange(min(X_test), max(X_test), 0.1)
X_range = X_range.reshape(-1, 1)
y_pred = regressor.predict(X_range)

# Plotting results
plt.title('Position x Salary', c='m')
plt.xlabel('Position', c='m')
plt.ylabel('Salary', c='m')
plt.scatter(X_test, y_test, c='c')
plt.plot(X_range, y_pred, c='m')
plt.legend(['Real values', 'Predicted values'])
plt.savefig('plot.png')
