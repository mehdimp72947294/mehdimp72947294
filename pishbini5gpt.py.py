import pandas as pd
import numpy as np
import requests
from pymongo import MongoClient
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense
import smtplib, ssl
from email.mime.text import MIMEText
import json
import logging
from plyer import notification
import tensorflow as tf
from gym import spaces
import gym

# Load configuration parameters from a JSON file
def load_config(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

# Get cryptocurrency prices from coingecko API
def get_cryptocurrency_prices(api_key, cryptocurrency_symbol, start_date, end_date, interval):
    url = f"https://api.coingecko.com/api/v3/global/decentralized_finance_defi"
    response = requests.get(url)
    return response.json()

# Store cryptocurrency prices in MongoDB database
def store_cryptocurrency_prices(db_url, db_name, collection_name, cryptocurrency_prices):
    client = MongoClient(db_url)
    db = client[db_name]
    collection = db[collection_name]

    # Convert timestamp string to datetime object
    cryptocurrency_prices['timestamp'] = pd.to_datetime(cryptocurrency_prices['timestamp'])

    # Convert price string to float
    cryptocurrency_prices['close'] = cryptocurrency_prices['close'].astype(float)

    # Store data in MongoDB
    records = cryptocurrency_prices.to_dict(orient='records')
    collection.insert_many(records)

# Load cryptocurrency prices from MongoDB database
def load_cryptocurrency_prices(db_url, db_name, collection_name):
    client = MongoClient(db_url)
    db = client[db_name]
    collection = db[collection_name]

    # Load data from MongoDB into a Pandas DataFrame
    cursor = collection.find({})
    cryptocurrency_prices = pd.DataFrame(list(cursor))

    # Remove '_id' column
    cryptocurrency_prices.drop('_id', axis=1, inplace=True)

    return cryptocurrency_prices

# Create LSTM model for predicting cryptocurrency prices
def create_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))

    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2)) 

    model.add(LSTM(units=50))
    model.add(Dropout(0.2))

    model.add(Dense(units=1))

    model.compile(optimizer='adam', loss='mean_squared_error')

    return model

# Train LSTM model on cryptocurrency prices data
def train_lstm_model(model, X_train, y_train, batch_size, epochs):
    model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs)

# Make cryptocurrency price predictions using trained LSTM model
def make_lstm_predictions(model, X_test):
    return model.predict(X_test)

# Send email alert when cryptocurrency price reaches a certain threshold
def send_email_alert(sender_email, sender_password, recipient_email, subject, message):
    port = 465  # For SSL
    smtp_server = "smtp.gmail.com"
    sender_email = sender_email
    password = sender_password
    recipient_email = recipient_email
    msg = MIMEText(message)
    msg['Subject'] = subject
    msg['From'] = sender_email
    msg['To'] = recipient_email

    context = ssl.create_default_context()
    with smtplib.SMTP_SSL(smtp_server, port, context=context) as server:
        server.login(sender_email, password)
        server.sendmail(sender_email, recipient_email, msg.as_string())

# Log error messages to a file
def log_error(error_msg):
    logging.basicConfig(filename='crypto_trading.log', level=logging.ERROR)
    logging.error(f'{datetime.now()}: {error_msg}')

# Send desktop notification when cryptocurrency price reaches a certain threshold
def send_desktop_notification(title, message):
    notification.notify(title=title, message=message)

# Define a custom OpenAI Gym environment for crypto trading
class CryptoTradingEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    def __init__(self, api_key, cryptocurrency_symbol, start_date, end_date, interval, initial_balance):
        super(CryptoTradingEnv, self).__init__()

        self.api_key = api_key
        self.cryptocurrency_symbol = cryptocurrency_symbol
        self.start_date = start_date
        self.end_date = end_date
        self.interval = interval
        self.initial_balance = initial_balance

        # Define action and observation space
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(5,), dtype=np.float32)

        # Load cryptocurrency prices from coingecko API
        self.cryptocurrency_prices = get_cryptocurrency_prices(api_key=self.api_key,
                                                               cryptocurrency_symbol=self.cryptocurrency_symbol,
                                                               start_date=self.start_date,
                                                               end_date=self.end_date,
                                                               interval=self.interval)

        # Store cryptocurrency prices in MongoDB database
        store_cryptocurrency_prices(db_url='mongodb://localhost:27017/',
                                    db_name='crypto_trading',
                                    collection_name=self.cryptocurrency_symbol,
                                    cryptocurrency_prices=self.cryptocurrency_prices)

        # Load cryptocurrency prices from MongoDB database
        self.cryptocurrency_prices = load_cryptocurrency_prices(db_url='mongodb://localhost:27017/',
                                                                 db_name='crypto_trading',
                                                                 collection_name=self.cryptocurrency_symbol)

        self.current_step = 0
        self.balance = self.initial_balance
        self.crypto_owned = 0

    def reset(self):
        self.current_step = 0
        self.balance = self.initial_balance
        self.crypto_owned = 0
        return self._next_observation()

    def step(self, action):
        # Execute action
        if action == 0:
            # Buy cryptocurrency
            self._buy_cryptocurrency()
        elif action == 1:
            # Sell cryptocurrency
            self._sell_cryptocurrency()

        # Update current step index
        self.current_step += 1

        # Check if the simulation is over
        done = False
        if self.current_step == len(self.cryptocurrency_prices) - 1:
            done = True

        # Get observation, reward, and info for the current step
        obs = self._next_observation()
        reward = self._get_reward()
        info = {'balance': self.balance, 'crypto_owned': self.crypto_owned}

        return obs, reward, done, info

    def render(self, mode='human'):
        print(f'Step: {self.current_step}')
        print(f'Balance: {self.balance}')
        print(f'Crypto owned: {self.crypto_owned}')

    def _next_observation(self):
        # Get price data for the last 5 days
        close_prices = self.cryptocurrency_prices.iloc[self.current_step - 4:self.current_step + 1]['close'].values

        # Get technical indicators for the last 5 days
        sma_5 = np.mean(close_prices)
        sma_10 = np.mean(self.cryptocurrency_prices.iloc[self.current_step - 9:self.current_step + 1]['close'].values)
        rsi_5 = self._compute_rsi(self.cryptocurrency_prices.iloc[self.current_step - 4:self.current_step + 1]['close'].values)

        # Return observation as a numpy array
        obs = np.array([sma_5, sma_10, rsi_5, self.balance, self.crypto_owned])

        return obs

    def _get_reward(self):
        if self.current_step == 0:
            return 0

        current_price = self.cryptocurrency_prices.iloc[self.current_step]['close']
        prev_price = self.cryptocurrency_prices.iloc[self.current_step - 1]['close']

        # Calculate percentage change in cryptocurrency price
        pct_change = (current_price - prev_price) / prev_price * 100

        # Calculate total portfolio value (in USD)
        portfolio_value = self.balance + self.crypto_owned * current_price

        # Calculate reward as the percentage change in portfolio value
        reward = pct_change * portfolio_value / 100

        return reward

    def _buy_cryptocurrency(self):
        current_price = self.cryptocurrency_prices.iloc[self.current_step]['close']
        max_crypto_purchase = self.balance / current_price

        # Buy all available cryptocurrency if purchase amount is less than 10 USD
        if max_crypto_purchase * current_price < 10:
            crypto_to_purchase = max_crypto_purchase
        else:
            crypto_to_purchase = max_crypto_purchase / 2

        # Update balance and cryptocurrency owned
        self.balance -= crypto_to_purchase * current_price
        self.crypto_owned += crypto_to_purchase

    def _sell_cryptocurrency(self):
        current_price = self.cryptocurrency_prices.iloc[self.current_step]['close']

        # Sell all owned cryptocurrency
        crypto_to_sell = self.crypto_owned

        # Update balance and cryptocurrency owned
        self.balance += crypto_to_sell * current_price
        self.crypto_owned = 0

    def _compute_rsi(self, prices):
        delta = np.diff(prices)
        gain = delta.copy()
        loss = delta.copy()
        gain[gain < 0] = 0
        loss[loss > 0] = 0
        avg_gain = np.mean(gain)
        avg_loss = -np.mean(loss)
        if avg_loss == 0:
            return 100
        else:
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            return rsi

# Train and evaluate LSTM model for cryptocurrency price prediction
def train_and_evaluate_lstm_model(api_key, cryptocurrency_symbol, start_date, end_date, interval, batch_size, epochs):
    # Load and preprocess data
    cryptocurrency_prices = get_cryptocurrency_prices(api_key=api_key,
                                                       cryptocurrency_symbol=cryptocurrency_symbol,
                                                       start_date=start_date,
                                                       end_date=end_date,
                                                       interval=interval)
    store_cryptocurrency_prices(db_url='mongodb://localhost:27017/',
                                db_name='crypto_trading',
                                collection_name=cryptocurrency_symbol,
                                cryptocurrency_prices=cryptocurrency_prices)

    cryptocurrency_prices = load_cryptocurrency_prices(db_url='mongodb://localhost:27017/',
                                                        db_name='crypto_trading',
                                                        collection_name=cryptocurrency_symbol)

    # Scale the cryptocurrency prices data
    scaler = MinMaxScaler()
    scaled_cryptocurrency_prices = scaler.fit_transform(cryptocurrency_prices[['close']])

    # Split and process data
    X_train, y_train, X_test, y_test = preprocess_data(scaled_cryptocurrency_prices)

    # Create and train LSTM model
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = create_lstm_model(input_shape)
    train_lstm_model(model, X_train, y_train, batch_size=batch_size, epochs=epochs)

    # Evaluate model
    mse = evaluate_model(model, X_test, y_test)

    return mse

# Define main function to run the program
def main():
    try:
        # Load configuration parameters from a JSON file
        config = load_config('config.json')

        # Train and evaluate LSTM model for cryptocurrency price prediction
        mse = train_and_evaluate_lstm_model(api_key=config['nomics_api_key'],
                                            cryptocurrency_symbol=config['cryptocurrency_symbol'],
                                            start_date=config['start_date'],
                                            end_date=config['end_date'],
                                            interval=config['interval'],
                                            batch_size=config['batch_size'],
                                            epochs=config['epochs'])

        # Send email alert if mean squared error of predictions is too high
        if mse > config['mse_threshold']:
            send_email_alert(sender_email=config['sender_email'],
                            sender_password=config['sender_password'],
                            recipient_email=config['recipient_email'],
                            subject='Crypto Trading Alert',
                            message=f'Mean squared error of cryptocurrency price predictions is {mse}.')

        # Send desktop notification if cryptocurrency price reaches a certain threshold
        current_price = load_cryptocurrency_prices(db_url='mongodb://localhost:27017/',
                                                    db_name='crypto_trading',
                                                    collection_name=config['cryptocurrency_symbol'])['close'].iloc[-1]
        if current_price > config['price_threshold']:
            send_desktop_notification(title='Crypto Trading Alert',
                                    message=f'{config["cryptocurrency_symbol"]} price has reached {current_price} USD.')

        # Run simulation of cryptocurrency trading using custom Gym environment
        env = CryptoTradingEnv(api_key=config['nomics_api_key'],
                                cryptocurrency_symbol=config['cryptocurrency_symbol'],
                                start_date=config['start_date'],
                                end_date=config['end_date'],
                                interval=config['interval'],
                                initial_balance=config['initial_balance'])
        state = env.reset()
        while True:
            action = env.action_space.sample()
            next_state, reward, done, info = env.step(action)
            if done:
                break
            state = next_state

    except Exception as e:
        log_error(f"Error: {e}")

if __name__ == '__main__':
    main()
