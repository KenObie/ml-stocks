# pip install streamlit fbprophet yfinance plotly
import streamlit as st
import webbrowser
from datetime import date

import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go


START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

home_link = '[Home](https://www.obie.codes)'
href = st.markdown(home_link, unsafe_allow_html=True)



st.title('ObieCodes - Stock Price Neural Network')

stocks = []
stock_input = st.text_input("Enter Name of Stock Ticker", "AMZN")
stock_input.capitalize()

if stock_input:
    stocks.append(stock_input)


# stocks = ('GOOG', 'AAPL', 'MSFT', 'FB', 'TSLA', 'AMD', 'SQ', 'UBER', 'T', 'TWTR', 'AMZN', 'GE', 'SNDL')
# selected_stock = st.selectbox('Select dataset for prediction', stocks)

# stock_input = st.text_input("Enter Name of Stock Ticker")

n_years = st.slider('Years of prediction:', 1, 4)
period = n_years * 365


@st.cache
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data


data_load_state = st.text('Loading data...')
data = load_data(stock_input)
data_load_state.text('Loading data... done!')

st.subheader('Raw data')
st.write(data.tail())


# Plot raw data
def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="stock_open"))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close"))
    fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)


plot_raw_data()

# Predict forecast with Prophet.
df_train = data[['Date', 'Close']]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

# Show and plot forecast
st.subheader('Forecast data')
st.write(forecast.tail())

st.write(f'Forecast plot for {n_years} years')
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)

st.write("Forecast components")
fig2 = m.plot_components(forecast)
st.write(fig2)