
from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
from datetime import timedelta

app = Flask(__name__)

# Load stock data (example dataset)
def load_stock_data():
    try:
        # Example: Load a CSV file with historical stock data
        # Replace this with your actual dataset or API call
        data = {
            'Date': pd.date_range(start='2023-01-01', periods=100, freq='D'),
            'Price': np.random.rand(100) * 100 + 100  # Random prices for demonstration
        }
        df = pd.DataFrame(data)
        return df
    except Exception as e:
        print(f"Error loading stock data: {e}")
        return None

# Train a Random Forest Regressor model
def train_model(df):
    try:
        df['Days'] = (df['Date'] - df['Date'].min()).dt.days
        X = df[['Days']]
        y = df['Price']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train Random Forest Regressor
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        return model
    except Exception as e:
        print(f"Error training model: {e}")
        return None

# Predict future stock prices for each day
def predict_prices(model, days_ahead):
    try:
        future_days = np.arange(0, days_ahead + 1).reshape(-1, 1)
        predicted_prices = model.predict(future_days)
        return future_days, predicted_prices
    except Exception as e:
        print(f"Error predicting prices: {e}")
        return None

# Generate a plot of historical and predicted prices
def generate_plot(df, model, days_ahead):
    try:
        plt.figure(figsize=(10, 6))
        
        # Plot historical data
        plt.plot(df['Date'], df['Price'], label='Historical Prices', marker='o', color='blue')
        
        # Predict future prices
        future_days, predicted_prices = predict_prices(model, days_ahead)
        future_dates = [df['Date'].max() + timedelta(days=int(day)) for day in future_days]
        
        # Plot predicted data
        plt.plot(future_dates, predicted_prices, label='Predicted Prices', marker='o', color='red', linestyle='-')
        
        # Add text annotation for the last predicted point
        last_date = future_dates[-1]
        last_price = predicted_prices[-1]
        plt.annotate(
            f'${last_price:.2f}',
            xy=(last_date, last_price),
            xytext=(10, -20),
            textcoords='offset points',
            fontsize=12,
            color='red',
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='red')
        )
        # Add labels and title
        plt.title('Stock Price Prediction')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        
        # Save the plot
        plot_path = os.path.join('static', 'plot.png')
        plt.savefig(plot_path)
        plt.close()
        return plot_path
    except Exception as e:
        print(f"Error generating plot: {e}")
        return None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        stock_symbol = request.form['stock-symbol']
        days_ahead = int(request.form['days-ahead'])  # Number of days to predict ahead

        # Load data and train model
        df = load_stock_data()
        if df is None:
            return "Error loading stock data"
        
        model = train_model(df)
        if model is None:
            return "Error training model"

        # Generate plot
        plot_path = generate_plot(df, model, days_ahead)
        if plot_path is None:
            return "Error generating plot"

        return render_template('result.html', stock_symbol=stock_symbol, days_ahead=days_ahead, plot_path=plot_path)
    return redirect(url_for('home'))

if __name__ == '__main__':
    app.run(debug=True)
    
