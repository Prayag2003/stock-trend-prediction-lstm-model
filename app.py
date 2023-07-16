import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
import streamlit as st

start_date = '2010-01-01'
end_date = '2022-12-31'
st.set_page_config(page_title='Stock Trend Prediction', page_icon='📈',
                   layout="centered", initial_sidebar_state="expanded")
st.markdown("<center><h1 style='color: green;'>Stock Trends Prediction 📈</h1></center>",
            unsafe_allow_html=True)

st.markdown(
    '<center><h3>A Time Series analogy LSTM Deep Learning Model</h3></center>', unsafe_allow_html=True)

stock_ticker_ip = st.text_input('Enter a Stock Ticker ', 'MSFT')
data = yf.download(stock_ticker_ip, start=start_date, end=end_date)


# describing the model
st.markdown("---")
st.markdown('<h3>Data from 2010-2022</h3>',
            unsafe_allow_html=True)
st.dataframe(data.describe().style.set_table_styles([
    {'selector': 'th', 'props': [('background-color', 'white')]}
]))
# st.write(data.describe())


# visualisations
st.markdown("---")
st.markdown('<center><h3>Closing Price vs Time chart</h3></center>',
            unsafe_allow_html=True)
fig = plt.figure(figsize=(9, 5))
plt.plot(data.Close, 'g', label='Closing price')
plt.xlabel('Years', fontsize=13)
plt.ylabel('Closing Price', fontsize=13)
plt.legend()
st.pyplot(fig)
st.markdown("---")

st.markdown('<center><h3>100 days Moving Average</h3></center>',
            unsafe_allow_html=True)
moving_avg100 = data.Close.rolling(100).mean()
fig = plt.figure(figsize=(12, 7))
plt.plot(data.Close, 'b', label='Closing price')
plt.plot(moving_avg100, 'g', label='Moving Average 100 days')
plt.xlabel('Years', fontsize=14)
plt.ylabel('Closing Price', fontsize=14)
plt.legend()
st.pyplot(fig)
st.markdown("---")

st.markdown('<center><h3>100 and 200 days Moving Average</h3></center>',
            unsafe_allow_html=True)
moving_avg100 = data.Close.rolling(100).mean()
moving_avg200 = data.Close.rolling(200).mean()
fig = plt.figure(figsize=(12, 7))
plt.plot(moving_avg100, 'r', label='Moving Average 100 days')
plt.plot(moving_avg200, 'g', label='Moving Average 200 days')
plt.plot(data.Close, 'b', label='Closing price')
plt.legend()
plt.xlabel('Years', fontsize=14)
plt.ylabel('Price', fontsize=14)
st.pyplot(fig)
st.markdown("---")

data_training = pd.DataFrame(data['Close'][0: int(len(data) * 0.7)])
data_testing = pd.DataFrame(data['Close'][int(len(data)*0.7): int(len(data))])

scaler = MinMaxScaler(feature_range=(0, 1))

data_training_array = scaler.fit_transform(data_training)


# load the model
model = load_model('keras_stock_prediction_model.keras')

# testing
prev_100_days = data_training.tail(100)
new_df = pd.concat([prev_100_days, data_testing], ignore_index=True)
input_data = scaler.fit_transform(new_df)

x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100: i])
    y_test.append(input_data[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)
y_predicted = model.predict(x_test)

scale = scaler.scale_
factor = 1 / scale[0]
y_test = y_test * factor
y_predicted = y_predicted * factor


# Chart of 100 MA, 200MA and closing price
st.markdown('<center><h3>Predictions vs Original</h3></center>',
            unsafe_allow_html=True)
fig = plt.figure(figsize=(12, 6))
plt.title('Original vs Predicted Price Graph', fontsize=15)
plt.plot(y_test, 'b', label='Original price')
plt.plot(y_predicted, 'r', label='Predicted price')
plt.xlabel('Years', fontsize=14)
plt.ylabel('Price', fontsize=14)
plt.legend()
st.pyplot(fig)
st.markdown("---")

st.write("<center><h5><span style='color: white;'>~ Made with ❤️ by Prayag Bhatt</span><h5></center>",
         unsafe_allow_html=True)
