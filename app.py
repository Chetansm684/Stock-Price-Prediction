import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import quandl
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import accuracy_score
from sklearn import neighbors
import os

# Fetch data from Quandl
quandl.ApiConfig.api_key = os.getenv('QUANDL_API_KEY')
data = quandl.get("NSE/TATAGLOBAL")

# Display first 10 rows of data
print(data.head(10))

# Plot Closing Prices
plt.figure(figsize=(16, 8))
plt.plot(data['Close'], label='Closing Price')
plt.title('Closing Prices of Tata Global Beverages')
plt.xlabel('Date')
plt.ylabel('Closing Price')
plt.legend()
plt.show()

# Calculate features for classification
data['Open - Close'] = data['Open'] - data['Close']
data['High - Low'] = data['High'] - data['Low']
data.dropna(inplace=True)

# Classification target variable
Y = np.where(data['Close'].shift(-1) > data['Close'], 1, -1)

# Prepare features and target for classification
X = data[['Open - Close', 'High - Low']]

# Split data for classification
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=44)

# Define parameters for KNN classification
params = {'n_neighbors': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]}
knn = neighbors.KNeighborsClassifier()
model = GridSearchCV(knn, params, cv=5)

# Fit the classification model
model.fit(X_train, y_train)

# Accuracy scores for classification
accuracy_train = accuracy_score(y_train, model.predict(X_train))
accuracy_test = accuracy_score(y_test, model.predict(X_test))

print('Train_data Accuracy: %.2f' % accuracy_train)
print('Test_data Accuracy: %.2f' % accuracy_test)

# Predictions and classifications
predictions_classifications = model.predict(X_test)
actual_predicted_data = pd.DataFrame({'Actual Class': y_test, 'Predicted Class': predictions_classifications})
print(actual_predicted_data.head(10))

# Regression using KNN
y = data['Close']

# Split data for regression
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X, y, test_size=0.25, random_state=44)

# Define parameters for KNN regression
params_reg = {'n_neighbors': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]}
knn_reg = neighbors.KNeighborsRegressor()
model_reg = GridSearchCV(knn_reg, params_reg, cv=5)

# Fit the regression model
model_reg.fit(X_train_reg, y_train_reg)

# Make predictions for regression
predictions_reg = model_reg.predict(X_test_reg)
print(predictions_reg)

# Calculate RMSE
rms = np.sqrt(np.mean(np.power((np.array(y_test_reg) - np.array(predictions_reg)), 2)))
print('Root Mean Squared Error:', rms)

# Display predictions and actual values for regression
valid = pd.DataFrame({'Predictions Close value': predictions_reg, 'Actual Close': y_test_reg})
print(valid.head(10))
