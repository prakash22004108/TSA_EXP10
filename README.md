# Exp.no: 10   IMPLEMENTATION OF SARIMA MODEL
### Date: 23/10/2025

### AIM:
To implement SARIMA model using python.
### ALGORITHM:
1. Explore the dataset
2. Check for stationarity of time series
3. Determine SARIMA models parameters p, q
4. Fit the SARIMA model
5. Make time series predictions and Auto-fit the SARIMA model
6. Evaluate model predictions
### PROGRAM:
Import necessary library:
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
```
Load and clean data:
```
data = pd.read_csv('GoogleStockPrices.csv')
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)
data_ts = data['Close']
```
Plot GDP Trend:
```
plt.figure(figsize=(12, 6))
plt.plot(data_ts)
plt.xlabel('Date')
plt.ylabel('Closing Price (USD)')
plt.title('Google Stock Price Time Series (Close)')
plt.grid(True)
plt.show()
```
 Check Stationarity :
```
def check_stationarity(timeseries):
    # Perform Dickey-Fuller test
    result = adfuller(timeseries)
    print('ADF Statistic:', result[0])
    print('p-value:', result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t{}: {}'.format(key, value))

print("\n--- Stationarity Check on Closing Prices ---")
check_stationarity(data_ts)

```

Plot ACF and PCF:
```
plt.figure(figsize=(12, 5))
plot_acf(data_ts, lags=50)
plt.title('Autocorrelation Function (ACF) - Google Close Price')
plt.show()
plt.figure(figsize=(12, 5))
plot_pacf(data_ts, lags=50)
plt.title('Partial Autocorrelation Function (PACF) - Google Close Price')
plt.show()

```
Split data:

```
train_size = int(len(data_ts) * 0.8)
train, test = data_ts[:train_size], data_ts[train_size:]

```
Fit SARIMA model:
```
print("\n--- Fitting SARIMA(1, 1, 1)x(1, 1, 1, 5) model ---")
sarima_model = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 5), enforce_stationarity=False, enforce_invertibility=False)
sarima_result = sarima_model.fit(disp=False)

```
Make predictions& Evaluate RMSE:

```
predictions = sarima_result.predict(start=len(train), end=len(data_ts) - 1) 

mse = mean_squared_error(test, predictions)
rmse = np.sqrt(mse)
print('RMSE:', rmse)
```

Plot Predictions:
```
plt.figure(figsize=(12, 6))
plt.plot(test.index, test, label='Actual')
plt.plot(test.index, predictions, color='red', label='Predicted')
plt.xlabel('Date')
plt.ylabel('Closing Price (USD)')
plt.title(f'SARIMA Model Predictions (RMSE: {rmse:.3f})')
plt.legend()
plt.grid(True)
plt.show()

```
### OUTPUT:
Original Data:
<img width="887" height="476" alt="image" src="https://github.com/user-attachments/assets/66f5046a-e57d-4ee0-8f6f-13b611816bd6" />

Autocorrelation:
<img width="742" height="541" alt="image" src="https://github.com/user-attachments/assets/8548a654-1791-4e3f-acc2-b592a67b59b7" />

Partial Autocorrelation:
<img width="719" height="540" alt="image" src="https://github.com/user-attachments/assets/29c68568-86aa-4d7f-8547-b287b8422f03" />

SARIMA Model:
<img width="892" height="478" alt="image" src="https://github.com/user-attachments/assets/80312f77-8ed3-48e9-a0f7-a5870462a97a" />


RMSE Value:

<img width="236" height="32" alt="image" src="https://github.com/user-attachments/assets/76116724-1c92-4805-9512-3a870b7600b7" />



### RESULT:
Thus the program run successfully based on the SARIMA model.
