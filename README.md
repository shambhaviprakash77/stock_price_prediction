# IBEX Stock Price Prediction using LSTM

## Overview
This project aims to predict the IBEX 35 stock index prices using Long Short-Term Memory (LSTM) neural networks. The dataset is retrieved using the `yfinance` library, preprocessed, and then used to train an LSTM model for forecasting future stock prices.

## Features
- **Data Collection**: Uses `yfinance` to fetch historical IBEX 35 stock prices.
- **Data Preprocessing**: Scales data using MinMaxScaler and splits it into training and test datasets.
- **LSTM Model**: Implements an LSTM-based neural network using TensorFlow/Keras.
- **Evaluation Metrics**: Compares LSTM predictions with a naive forecast using Mean Squared Error (MSE) and Median Absolute Error (MAE).
- **Data Visualization**: Plots stock prices, predictions, and statistical insights.

## Dependencies
Ensure you have the following Python packages installed:
```sh
pip install numpy pandas matplotlib seaborn scikit-learn yfinance tensorflow
```

## Project Structure
```
|-- stock_price_prediction/
    |-- main.py                # Main script for training and testing
    |-- README.md              # Project documentation
    |-- requirements.txt       # List of dependencies
```

## Usage
### 1. Clone the repository
```sh
git clone https://github.com/shambhaviprakash77/stock_price_prediction.git
cd stock_price_prediction
```

### 2. Run the Script
```sh
python main.py
```

## Step-by-Step Workflow
### 1. Download IBEX stock data
- Uses `yfinance` to fetch historical stock prices from 2016 to 2021.

### 2. Preprocess Data
- Normalize the 'Close' prices using MinMaxScaler.
- Split the dataset into training and testing sets.

### 3. Train LSTM Model
- Create and compile an LSTM-based neural network.
- Train the model using the training dataset.

### 4. Make Predictions
- Generate predictions using the trained LSTM model.
- Compare with a naive forecast.

### 5. Evaluate the Model
- Compute MSE and MAE for both LSTM and naive forecasts.
- Generate visualizations to compare actual vs. predicted stock prices.

### 6. Analyze Trends
- Compute daily percentage change.
- Generate histograms and autocorrelation plots for trend analysis.

## Model Performance
After training, the LSTM model is evaluated using the following metrics:
- **MSE (Mean Squared Error)**
- **MAE (Median Absolute Error)**

## Results
The project visualizes the stock price predictions, compares them against actual values, and analyzes trends in the IBEX stock market.

## Future Improvements
- Use more complex LSTM architectures for better predictions.
- Experiment with additional stock indicators.
- Implement hyperparameter tuning.

## License
This project is open-source and available under the MIT License.

## Author
**Shambhavi Prakash**  
GitHub: [shambhaviprakash77](https://github.com/shambhaviprakash77)

