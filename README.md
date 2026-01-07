# 🌤 Weather Prediction App  

## Overview  
The **Weather Prediction App** is a machine learning-powered web application that predicts temperature based on real-time weather factors such as humidity, wind speed, and precipitation. The app utilizes a trained Random Forest model to provide accurate temperature forecasts and supports features like data visualization, trend analysis, and database storage of predictions.

---

## Features  
- **Real-Time Weather Data**: Fetches live weather data using the OpenWeatherMap API.  
- **Machine Learning Predictions**: Predicts temperature using a Random Forest model trained on historical weather data.  
- **Interactive UI**: User-friendly interface built with Streamlit.  
- **Data Visualization**: Provides hourly, daily, and 5-day temperature trends with graphs.  
- **Database Integration**: Saves and retrieves predictions from a SQL Server database.  
- **Customizable Options**: Supports temperature units (Celsius and Fahrenheit).  

---

## Technologies Used  
- **Frontend**: [Streamlit](https://streamlit.io/)  
- **Backend**: Python, Machine Learning with Random Forest  
- **Database**: SQL Server  
- **APIs**: [OpenWeatherMap API](https://openweathermap.org/api)  
- **Libraries**:  
  - `pandas`  
  - `numpy`  
  - `matplotlib`  
  - `requests`  
  - `pyodbc`  

---

## Prerequisites  
1. **Python** (Version 3.7 or above)  
2. **Libraries**: Install dependencies using the command:  
   ```bash
   pip install -r requirements.txt

Create a database WeatherDB in SQL Server.
Run the following SQL script to create the weather_predictions table


CREATE TABLE weather_predictions (
    id INT IDENTITY(1,1) PRIMARY KEY,
    location NVARCHAR(100),
    humidity FLOAT,
    wind_speed FLOAT,
    precipitation FLOAT,
    predicted_temperature FLOAT,
    timestamp DATETIME DEFAULT GETDATE()
);




-->Set Up the API Key
Replace API_KEY in app.py with your OpenWeatherMap API key.

-->Run the Application
-bash

    streamlit run app.py

Application Structure

weather-prediction-app/
│
├── app.py                  # Main application script
├── random_forest_model.pkl # Pre-trained machine learning model
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation




Contact

Name: Muhammad Arslan Jameel,Raheel Anjum
Email: arslan.jameel8532@gmail.com
GitHub: https://github.com/Arslan8532

