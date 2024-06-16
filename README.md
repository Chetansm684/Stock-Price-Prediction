# Stock Price Analysis with K-Nearest Neighbors (KNN)

This Python script fetches historical stock price data from Quandl, performs K-Nearest Neighbors (KNN) classification and regression, and evaluates the model performance. It demonstrates how to use machine learning algorithms for stock price prediction and analysis.

<img width="600" alt="Figure_1.png" src="https://github.com/Chetansm684/Stock-Price-Prediction/image/Figure_1.png">

## Overview

The script `main.py` utilizes Quandl's API to fetch historical stock prices of Tata Global Beverages (TATAGLOBAL) from the National Stock Exchange (NSE). It then:
- Plots the closing prices of the stock.
- Computes additional features (`Open - Close` and `High - Low`) for classification and regression.
- Splits the data into training and testing sets.
- Trains a KNN classifier to predict stock price movements (up or down).
- Trains a KNN regressor to predict actual stock prices.
- Evaluates the models using accuracy scores for classification and RMSE for regression.

## Usage
After running main.py, the script will:

- Display the first 10 rows of the fetched data.
- Plot the closing prices of Tata Global Beverages.
- Train and evaluate a KNN classifier for predicting stock price movements.
- Train and evaluate a KNN regressor for predicting actual stock prices.

## Results
- The script outputs accuracy scores for the classification model (Train_data Accuracy and Test_data Accuracy).
- It calculates the RMSE (Root Mean Squared Error) for the regression model.
- Predictions and actual values are displayed for both classification and regression tasks.

## Dependencies
- numpy: For numerical operations.
- pandas: For data manipulation and analysis.
- matplotlib: For plotting graphs.
- quandl: For fetching financial data.
- scikit-learn: For machine learning models and evaluation metrics.
