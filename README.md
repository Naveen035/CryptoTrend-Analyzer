# **Bitcoin Price Predictor** 🚀

## **Overview**

The **Bitcoin Price Predictor** is a machine learning project aimed at forecasting the future price of Bitcoin based on historical data.

## **📚 Table of Contents**

- [Project Background](#project-background)
- [Problem Statement](#problem-statement)
- [Features](#features)
- [Data](#data)
- [Models Used](#models-used)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Conclusion](#conclusion)
- [License](#license)

## **Project Background** 🌍

Bitcoin, created in 2009 by an unknown person or group using the name Satoshi Nakamoto, is the first decentralized digital currency. It has grown significantly over the years, gaining popularity as an investment asset and a medium of exchange. This project aims to predict future Bitcoin prices using machine learning techniques, providing insights into its price trends.

## **Problem Statement** ❓

The volatility of Bitcoin prices makes it challenging for investors to make informed decisions. This project seeks to develop a predictive model that can help forecast Bitcoin prices based on historical data, aiding users in understanding potential future trends.

## **Features** ✨

- **Interactive User Interface**: Users can input a year and receive predictions for Bitcoin prices.
- **Data Visualization**: Graphs displaying historical trends in Bitcoin prices.
- **Model Evaluation**: Display of various models' performance metrics.

## **Data** 📊

The data for this project was sourced from **Kaggle.com**, containing historical Bitcoin prices, including columns like `Date`, `Open`, `High`, `Low`, `Close`, `Volume`, and `Market Cap`.

## **Models Used** 🛠️

The following models were utilized in this project:

- **Decision Tree Regressor**
- **Random Forest Regressor**
- **Gradient Boosting Regressor**
- **LightGBM Regressor**
- **XGBoost Regressor**

## **Installation** 💻

To run this project locally, follow these steps:

1. Clone the repository:
   ```bash
   git clone <repository-url>
## **Usage** 📈

To use the **Bitcoin Price Predictor**:

1. **Run the Streamlit app**:
   ```bash
   streamlit run app.py
## **Results** 📊

The following models were evaluated for predicting Bitcoin prices, and their mean squared error (MSE) results are as follows:

- **Decision Tree Regressor (Tuned)** - MSE: 18522311.82
- **Random Forest Regressor (Tuned)** - MSE: 18499370.65
- **Gradient Boosting Regressor** - MSE: 18522311.80
- **LightGBM Regressor (Tuned)** - MSE: 18521660.15
- **XGBoost Regressor (Tuned)** - MSE: 18521572.42

## **Conclusion** 🏁

This project provides a robust framework for predicting Bitcoin prices using machine learning techniques. By leveraging historical data and various regression models, users can gain insights into potential future price trends. The comparative analysis of different models helps in understanding their performance, enabling users to make informed decisions regarding Bitcoin investments.

The results indicate that the **Random Forest Regressor** performed the best in terms of minimizing the mean squared error, suggesting it may be the most reliable model for this particular dataset. With the visualizations included in the Streamlit app, users can better understand the historical price trends of Bitcoin and make predictions accordingly.
