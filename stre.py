import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the trained model
with open(r"C:\Users\jayas\OneDrive\Desktop\New folder\Bitcoin_project\random_forest_model.pkl", 'rb') as file:
    model = pickle.load(file)

# Load Bitcoin historical data for graph
df = pd.read_csv(r"C:\Users\jayas\Downloads\BTC-USD.csv")
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# Streamlit app setup
st.set_page_config(page_title="💰 CryptoTrend Analyzer", layout="centered")

# CSS for background image
page_bg_img = '''
<style>
body {
    background-image: url("https://c4.wallpaperflare.com/wallpaper/905/666/494/background-render-fon-bitcoin-bitcoin-hd-wallpaper-preview.jpg");
    background-size: cover;
    background-repeat: no-repeat;
    background-attachment: fixed;
}
h1 {
    color: #669ae8;
    font-family: 'Helvetica', sans-serif;
    font-size: 36px;
}
p, label {
    color: #669ae8;
    font-size: 18px;
    font-family: 'Arial', sans-serif;
}
button {
    background-color: #FFFFFFF;
    color: #669ae8;
}
</style>
'''

st.markdown(page_bg_img, unsafe_allow_html=True)

# Title with emoji
st.title("💰 CryptoTrend Analyzer 📈")

# Subheader with emoji
st.subheader("🔍 Enter the year to predict the price of Bitcoin 🕵️‍♂️")

# Input for the year
year_input = st.number_input("Enter the year:", min_value=2009, max_value=2100, step=1, value=2024)

# Predict button
if st.button("🚀 Predict Bitcoin Price 💵"):
    # Reshape the input for prediction
    year_array = np.array([[year_input]])
    prediction = model.predict(year_array)[0]
    
    # Display the prediction
    st.success(f"✨ The predicted price of Bitcoin in {year_input} is approximately ${prediction:.2f} 🎉")

# Plotting the historical Bitcoin prices
st.subheader("📊 Historical Bitcoin Prices 📉")
plt.figure(figsize=(10, 5))
plt.plot(df.index, df['Adj Close'], label='Adjusted Close Price', color='orange')
plt.title('Bitcoin Price History')
plt.xlabel('Year')
plt.ylabel('Price in USD')
plt.xticks(rotation=45)
plt.legend()
plt.grid()
st.pyplot(plt)
