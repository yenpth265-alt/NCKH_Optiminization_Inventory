import pandas as pd
import numpy as np

def rmsse(train, test, forecast):

    #Sort index để phòng bị sai thứ tự
    test = test.sort_index()
    forecast = forecast.sort_index()
    train = train.sort_index()

    #Tính toán RMSSE
    forecast_mse = np.mean((test - forecast)**2, axis=1)
    train_mse = train.apply(
        lambda row: np.mean(np.diff(np.trim_zeros(row.values)) ** 2)
        if len(np.trim_zeros(row.values)) > 1 else 0,
        axis=1
    )
    return np.sqrt(forecast_mse / train_mse)