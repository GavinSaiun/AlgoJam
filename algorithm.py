import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error

class Algorithm():
    def __init__(self, positions):
        self.data = {}
        self.positionLimits = {}
        self.day = 0
        self.positions = positions

    def get_current_price(self, instrument):
        return self.data[instrument][-1]

    def get_positions(self):
        currentPositions = self.positions
        positionLimits = self.positionLimits
        
        desiredPositions = {}
        for instrument, positionLimit in positionLimits.items():
            desiredPositions[instrument] = 0
            
            if instrument in ["Fintech Token", "Fun Drink", "Red Pens", "Thrifted jeans", 
 "UQ Dollar", "Coffee Beans", "Goober Eats", "Milk"]:
                forecasted_price = self.forecast_price(instrument, steps=1)
                current_price = self.get_current_price(instrument)

                if forecasted_price > current_price:
                    desiredPositions[instrument] = positionLimits[instrument]
                else:
                    desiredPositions[instrument] = -positionLimits[instrument]

        return desiredPositions

    def load_data(self, filename, instrument):
        df = pd.read_csv(filename, header=None, names=['Day', 'Price'])
        self.data[instrument] = df['Price'].values

    def forecast_price(self, instrument, steps=1):
        price_data = self.data[instrument]

        # Ensure price data is a pandas Series
        price_data = pd.Series(price_data)

        # Fit Exponential Smoothing model without seasonality if data is insufficient
        try:
            # Use a simple model if not enough data for seasonal components
            if len(price_data) < 2 * 365:  # Less than two full years
                model = ExponentialSmoothing(
                    price_data,
                    trend='add',  # Additive trend component
                    seasonal=None  # No seasonal component
                )
            else:
                model = ExponentialSmoothing(
                    price_data,
                    trend='add',  # Additive trend component
                    seasonal='add',  # Additive seasonal component
                    seasonal_periods=365  # Annual seasonality
                )
            
            model_fit = model.fit()
            
            # Forecast the next 'steps' days
            forecast = model_fit.forecast(steps=steps)
            
            return forecast.iloc[0]  # get the first element of the forecast series
        except Exception as e:
            print(f"Error forecasting price for {instrument}: {e}")
            return np.nan
