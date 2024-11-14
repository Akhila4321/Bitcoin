import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import OneHotEncoder

# Function to train the model
def train_model(data):
    # Load data
    df = pd.read_csv(data)
    
    # Preprocess data
    X, y = preprocess_data(df)

    # Data splitting
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Model training
    model = RandomForestRegressor()
    model.fit(X_train, y_train)

    return model, X_train, y_train

# Function to preprocess data
def preprocess_data(df):
    # Convert 'Date' column to datetime type
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Extract relevant features from the date
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    
    # Separate features and target variable
    X = df.drop(columns=["Date", "Close"])  # Exclude 'Date' and 'Adj Close' columns
    y = df['Close']  # Target variable
    
    return X, y

# Function to make predictions
def make_prediction(model, features):
    prediction = model.predict(features)
    return prediction

# Bitcoin price prediction Page
def home():
    st.title('Bitcoin Price Prediction App')
    st.write('This app predicts Bitcoin prices using machine learning algorithms.')
    st.image("download.jpeg", caption="Bitcoin Image")

# Prediction page
def prediction():
    st.title('Make a Prediction')
    uploaded_file = st.file_uploader("Upload your input CSV file", type=["csv"])
    
    if uploaded_file is not None:
        model, X_train, y_train = train_model(uploaded_file)
        
        if model is not None:
            sample_input = {}
            for column in X_train.columns:
                value = st.number_input(column, value=X_train.iloc[0][column])
                sample_input[column] = value
            
            if st.button('Predict'):
                input_features = pd.DataFrame([sample_input])
                prediction = make_prediction(model, input_features)
                st.success('Predicted Price: {:.2f}'.format(prediction[0]))

# About page
def about():
  st.markdown("## Bitcoin Price Prediction")
  st.write("Welcome to the Bitcoin Price Prediction App.")
  st.write("# **About Bitcoin Price Prediction**")
  st.write("This app predicts Bitcoin prices using machine learning algorithms.")
  st.write("Bitcoin has gained significant attention in recent years due to its high volatility and potential for substantial returns.")
  st.write("Understanding and predicting Bitcoin prices can provide valuable insights for investors and traders in the cryptocurrency market.")
  st.write("Our app utilizes advanced machine learning techniques to analyze historical data and forecast future Bitcoin prices.")
  st.write("Whether you're a seasoned trader or a newcomer to cryptocurrency, our prediction models aim to assist you in making informed decisions.")
  st.write("Stay ahead of the curve with accurate forecasts and real-time insights into the cryptocurrency market.")
  st.write("# **Project Overview**")
  st.write("Our project aims to forecast Bitcoin prices based on historical data and relevant features.")
  st.write("By analyzing market trends and historical price movements, we aim to develop accurate predictive models for Bitcoin prices.")
  st.write("This project combines data science, financial analysis, and domain expertise to provide actionable insights for cryptocurrency enthusiasts.")
  st.write("We continuously refine our models with the latest data and market developments to ensure the highest level of accuracy.")
  st.write("# **Key Objectives**")
  st.write("### **Predict Bitcoin Prices**")
  st.write("Our primary objective is to predict Bitcoin prices accurately using machine learning techniques.")
  st.write("### **Analyze Market Trends**")
  st.write("We analyze market trends and historical data to identify patterns that influence Bitcoin prices.")
  st.write("### **Provide Insights**")
  st.write("We provide insights and predictions to help users make informed decisions in the cryptocurrency market.")
  st.write("Our platform offers detailed analysis reports, price forecasts, and market sentiment indicators to empower users.")
  st.write("# **Get Involved**")
  st.write("Interested in the potential of machine learning in cryptocurrency analysis? Join us and explore the world of Bitcoin price prediction.")
  st.write("Subscribe to our newsletter for updates, market insights, and exclusive offers.")
  st.write("Connect with us on social media to engage with our community and stay informed about the latest developments.")
  st.write("# **Contact Us**")
  st.write("Have questions or feedback? Contact our team at bitcoinprediction@example.com.")



# Main function to run the app
def main():
    st.sidebar.title('Bitcoin Price Prediction')
    page = st.sidebar.radio("Go to", ["Home", "Prediction", "About"])

    if page == "Home":
        home()
    elif page == "Prediction":
        prediction()
    elif page == "About":
        about()

if __name__ == '__main__':
    main()
