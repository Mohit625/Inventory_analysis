from sklearn.linear_model import LinearRegression
import pandas as pd

def train_linear_model(df):
    df = df.copy()
    df['day'] = range(len(df))
    
    X = df[['day', 'price', 'promo']]
    y = df['y']
    
    model = LinearRegression()
    model.fit(X, y)
    
    return model

def predict_linear(model, df, future_days):
    df = df.copy()

    # Convert date to ordinal (numeric)
    df['day'] = pd.to_datetime(df['ds']).map(pd.Timestamp.toordinal)

    last_day = df['day'].iloc[-1]

    future_days_range = range(1, future_days + 1)

    future_data = pd.DataFrame({
        'day': [last_day + i for i in future_days_range],
        'price': df['price'].tail(7).mean(),
        'promo': 0
    })

    preds = model.predict(future_data)
    return preds