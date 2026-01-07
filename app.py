import streamlit as st
import pandas as pd
import pickle
import pyodbc  # For SQL Server connection
import matplotlib.pyplot as plt
import numpy as np
import requests
import time



PRIMARY_COLOR = "#3498db"  # Primary color for the bar graph
SECONDARY_COLOR = "#2ecc71"  # Secondary color for the trend graph
POINT_COLOR = "#e74c3c"  # Color for each point in the trend graph

# Load the trained model
with open('random_forest_model.pkl', 'rb') as file:
    dt_model = pickle.load(file)
# OpenWeatherMap API Key
API_KEY = "63c4122467323ee12b3700437e9107fa"
LOCATIONS = ['San Diego', 'Philadelphia', 'San Antonio', 'San Jose', 
             'New York', 'Houston', 'Dallas', 'Chicago', 'Los Angeles', 'Phoenix']

# Database connection function for SQL Server
@st.cache_resource
def init_db():
    try:
        conn = pyodbc.connect(
            'DRIVER={ODBC Driver 17 for SQL Server};'
            'SERVER=DESKTOP-EKSVPQH;'  # Adjust if server name is different
            'DATABASE=WeatherDB;'
            'Trusted_Connection=yes;'
        )
        return conn
    except pyodbc.Error as e:
        st.error(f"Database connection error: {e}")
        return None

conn = init_db()

# Define color scheme and CSS styling for the app
PRIMARY_COLOR = "#1E90FF"
SECONDARY_COLOR = "#FF6347"
BACKGROUND_GRADIENT = "linear-gradient(120deg, #333, #444)"
BUTTON_COLOR = "#4CAF50"
SIDEBAR_GRADIENT = "linear-gradient(120deg, #444, #555)"


# Helper function to fetch weather data
@st.cache_data
def get_weather_data(location):
    """Fetch real-time weather data from OpenWeatherMap API for the given location."""
    try:
        base_url = f"http://api.openweathermap.org/data/2.5/weather?q={location}&appid={API_KEY}&units=metric"
        response = requests.get(base_url)
        response.raise_for_status()  # Raise HTTPError for bad responses (4xx and 5xx)
        data = response.json()

        # Extract weather data
        humidity = data['main']['humidity']
        wind_speed = data['wind']['speed']
        precipitation = data.get('rain', {}).get('1h', 0)  # Handle missing precipitation

        # Convert to numeric and check validity
        humidity = float(pd.to_numeric(humidity, errors='coerce'))
        wind_speed = float(pd.to_numeric(wind_speed, errors='coerce'))
        precipitation = float(pd.to_numeric(precipitation, errors='coerce'))

        if pd.isna(humidity) or pd.isna(wind_speed) or pd.isna(precipitation):
            st.error("Invalid weather data received. Please try another location.")
            return None, None, None

        return humidity, wind_speed, precipitation

    except requests.RequestException as e:
        st.error(f"Error fetching data: {e}")
        return None, None, None
    except ValueError as ve:
        st.error(f"Data conversion error: {ve}")
        return None, None, None



# Set gradient background and sidebar style
st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: {BACKGROUND_GRADIENT};
        background-attachment: fixed;
        background-size: cover;
        color: white;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }}
    .stButton>button {{
        background-color: {BUTTON_COLOR}; 
        color: white;
        font-weight: bold;
        border-radius: 8px;
        transition: background-color 0.3s ease, transform 0.3s ease;
    }}
    .stButton>button:hover {{
        background-color: #45a049;
        transform: scale(1.05);
    }}
    .stSidebar {{
        background-image: {SIDEBAR_GRADIENT};
        background-attachment: fixed;
        color: white;
        padding: 20px;
        border-radius: 10px;
    }}
    .dataframe-table {{
        border-collapse: collapse;
        width: 100%;
        background-color: #333;
        color: white;
        overflow-x: auto;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }}
    .dataframe-table th, .dataframe-table td {{
        padding: 10px;
        text-align: left;
    }}
    .dataframe-table th {{
        background-color: #4CAF50;
        color: white;
    }}
    .dataframe-table tbody tr:nth-child(even) {{
        background-color: #2f2f2f;
    }}
    .dataframe-table tbody tr:nth-child(odd) {{
        background-color: #3a3a3a;
    }}
    .dataframe-table tbody tr:hover {{
        background-color: #3c3c3c;
        transform: scale(1.05);
        transition: background-color 0.3s ease, transform 0.3s ease;
    }}
    </style>
    """, unsafe_allow_html=True
)

st.markdown("""
    <style>
        /* Set the background */
        .main {
            background-color: #e8f4f8; /* Soft blue background */
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        /* Sidebar styling */
        .css-1aumxhk { /* Streamlit sidebar class */
            background-color: #005f73; /* Dark cyan background */
            color: white;
            font-size: 1.1em;
            padding: 20px;
            border-radius: 10px;
        }
        /* Sidebar title styling */
        .css-10trblm {
            color: #edf6f9; /* Light cyan for sidebar text */
            font-weight: bold;
            font-size: 1.2em;
        }
        /* Improve radio buttons, select boxes, and buttons */
        .stRadio > label, .stSelectbox > label {
            font-weight: bold;
            color: #005f73;
        }
        .stButton button {
            background-color: #008c9e;
            color: white;
            font-size: 1.05em;
            border-radius: 8px;
            transition: background-color 0.3s ease, transform 0.3s ease;
        }
        .stButton button:hover {
            background-color: #5d8aa8;
            transform: scale(1.05);
        }
    </style>
""", unsafe_allow_html=True)

# Streamlit App Title
st.markdown("<h1 style='text-align: center; color: #1E90FF;'>🌤 Weather Prediction App</h1>", unsafe_allow_html=True)
# Sidebar with Interactive Sections and Animation
st.sidebar.title("🌐 App Navigation")
sections = ["🌄 About the App", "📜 Instructions", "🌦 Weather Features", "❓ FAQ"]
section_choice = st.sidebar.radio("Explore:", sections)

# Loading animation
with st.spinner("Loading your selection..."):
    time.sleep(1)

if section_choice == "🌄 About the App":
    st.sidebar.subheader("Overview")
    st.sidebar.markdown("""
        This application leverages **Machine Learning** to predict temperature based on weather factors like humidity, wind speed, and precipitation.
        Using historical weather data, our **Random Forest Model ** forecasts temperature values to help with future planning.
    """)
elif section_choice == "📜 Instructions":
    st.sidebar.subheader("Instructions")
    st.sidebar.markdown("""
        1. Enter values for humidity, wind speed, and precipitation.
        2. Click on 'Predict Temperature' to see the forecast.
        3. Use 'Save Prediction' to store it in the database.
        4. Click 'View Saved Predictions' to access past entries.
    """)
elif section_choice == "🌦 Weather Features":
    st.sidebar.subheader("Learn More")
    weather_info = st.sidebar.selectbox("Choose a feature to learn more:", ["🌫 Humidity", "🌬 Wind Speed", "🌧 Precipitation"])
    if weather_info == "🌫 Humidity":
        st.sidebar.markdown("**Humidity** affects how warm or cool a temperature feels.")
    elif weather_info == "🌬 Wind Speed":
        st.sidebar.markdown("**Wind Speed** impacts temperature and precipitation.")
    elif weather_info == "🌧 Precipitation":
        st.sidebar.markdown("**Precipitation** refers to rain, snow, etc., affecting temperature and humidity.")
elif section_choice == "❓ FAQ":
    st.sidebar.subheader("Frequently Asked Questions")
    faq = st.sidebar.selectbox("Select a question:", [
        "What data is used for prediction?",
        "How does the model make predictions?",
        "Can I view past predictions?",
        "How do I save a prediction?",
        "What type of model is used in this app?",
        "Can I use this app offline?",
        "How is humidity measured?",
        "How does wind speed affect the prediction?",
        "Can I change the model settings?",
        "What is the range of predictions this app can make?",
        "Where does the historical data come from?",
        "Is my data saved locally or online?",
    ])
    # Providing answers based on selection
    if faq == "What data is used for prediction?":
        st.sidebar.markdown("The model uses humidity, wind speed, and precipitation to forecast temperature. This input data allows the model to analyze and make predictions for future temperature trends.")
    elif faq == "How does the model make predictions?":
        st.sidebar.markdown("A trained **Decision Tree Model** processes your inputs, like humidity, wind speed, and precipitation, and predicts the temperature. It learns from past data patterns to forecast values accurately.")
    elif faq == "Can I view past predictions?":
        st.sidebar.markdown("Yes! Use the 'View Saved Predictions' button to access past entries. This feature helps track predictions and understand temperature changes over time.")
    elif faq == "How do I save a prediction?":
        st.sidebar.markdown("After making a prediction, click 'Save Prediction' to store it in the database. This option is helpful for tracking or analyzing past predictions.")
    elif faq == "What type of model is used in this app?":
        st.sidebar.markdown("The app uses a **Decision Tree Model** trained on historical weather data. This model was chosen for its ability to make accurate predictions based on input features.")
    elif faq == "Can I use this app offline?":
        st.sidebar.markdown("No, this app requires an internet connection to access certain libraries and to connect to the database where past predictions are stored.")
    elif faq == "How is humidity measured?":
        st.sidebar.markdown("Humidity is typically measured in percentage values, indicating the amount of moisture in the air. This is one of the key inputs in our temperature prediction model.")
    elif faq == "How does wind speed affect the prediction?":
        st.sidebar.markdown("Wind speed impacts weather conditions, affecting how temperatures fluctuate. Including wind speed in the model improves its ability to make accurate predictions.")
    elif faq == "Can I change the model settings?":
        st.sidebar.markdown("Currently, the app uses a pre-trained model with fixed settings for accuracy. Future updates may include customizable settings for advanced users.")
    elif faq == "What is the range of predictions this app can make?":
        st.sidebar.markdown("The app can predict temperatures within typical seasonal ranges found in the dataset. Extreme values might be less accurate, as they're less common in historical data.")
    elif faq == "Where does the historical data come from?":
        st.sidebar.markdown("The historical data used to train this model comes from trusted weather databases and covers several years of observations for various weather factors.")
    elif faq == "Is my data saved locally or online?":
        st.sidebar.markdown("Predictions you save are stored in a secure, cloud-based database. This allows you to access saved data on any device using this app.")

# Main Input Section
st.header("Weather Features")
location = st.selectbox("Select Location", LOCATIONS)

# Step 2: Fetch Weather Data
if st.button("Fetch Weather Data"):
    humidity, wind_speed, precipitation = get_weather_data(location)
    if humidity is not None and wind_speed is not None and precipitation is not None:
        st.session_state['humidity'] = humidity
        st.session_state['wind_speed'] = wind_speed
        st.session_state['precipitation'] = precipitation
        st.session_state['location'] = location
        st.success(f"Data fetched for {location}: Humidity={humidity}%, Wind Speed={wind_speed} km/h, Precipitation={precipitation} mm")
    else:
        st.warning("Failed to fetch data. Please try again.")

# Step 3: Prediction (use fetched features)
if 'humidity' in st.session_state and 'wind_speed' in st.session_state and 'precipitation' in st.session_state:
    st.subheader("Fetched Features")
    st.write(f"Location: {st.session_state['location']}")
    st.write(f"Humidity: {st.session_state['humidity']}%")
    st.write(f"Wind Speed: {st.session_state['wind_speed']} km/h")
    st.write(f"Precipitation: {st.session_state['precipitation']} mm")

predicted_temp = None

def convert_to_fahrenheit(celsius_temp):
    """Convert Celsius to Fahrenheit."""
    return celsius_temp * 9 / 5 + 32

unit = st.radio("Choose Temperature Unit", ("Celsius (°C)", "Fahrenheit (°F)"))

def generate_hourly_forecast(predicted_temp, unit):
    """Generate hourly temperature forecast based on predicted temperature."""
    hourly_forecast = [predicted_temp + np.random.uniform(-1, 1) for _ in range(24)]
    return hourly_forecast

if st.button('Predict Temperature'):
    if 'humidity' in st.session_state and 'wind_speed' in st.session_state and 'precipitation' in st.session_state:
        input_data = pd.DataFrame([[st.session_state['humidity'], st.session_state['wind_speed'], st.session_state['precipitation']]], 
                                  columns=['Humidity_pct', 'Wind_Speed_kmh', 'Precipitation_mm'])
        
        try:
            # Predict temperature using the model
            predicted_temp = dt_model.predict(input_data)[0]

            if unit == "Fahrenheit (°F)":
                predicted_temp = convert_to_fahrenheit(predicted_temp)

            st.session_state['predicted_temp'] = predicted_temp
            st.success(f"Predicted Temperature: {predicted_temp:.2f} {unit.split()[1]}")

            # Enhanced bar plot for predicted temperature
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.bar(["Predicted Temperature"], [predicted_temp], color=PRIMARY_COLOR)
            ax.set_ylabel(f"Temperature ({unit.split()[1]})")
            ax.set_ylim(min(predicted_temp - 10, -10), max(predicted_temp + 10, 40))
            ax.text(0, predicted_temp, f"{predicted_temp:.2f}{unit.split()[1]}", ha='center', va='bottom', fontsize=12, color="black")
            ax.set_title("Temperature Prediction", fontsize=16, fontweight='bold')
            st.pyplot(fig)

            # Generate 5-day temperature trend with slight random variations
            np.random.seed(0)  # For reproducibility
            predicted_temps = [predicted_temp + np.random.uniform(-3, 3) for _ in range(5)]
            if unit == "Fahrenheit (°F)":
                predicted_temps = [convert_to_fahrenheit(temp) for temp in predicted_temps]

            # Create a static line plot for the 5-day trend
            fig2, ax2 = plt.subplots(figsize=(8, 5))
            ax2.plot(range(1, 6), predicted_temps, marker='o', color=SECONDARY_COLOR, linewidth=2, label='Temperature Trend')

            # Add labels at each point
            for x, y in zip(range(1, 6), predicted_temps):
                ax2.text(x, y, f"{y:.2f}{unit.split()[1]}", ha='center', va='bottom', fontsize=12, color=POINT_COLOR)

            # Set graph aesthetics
            ax2.set_ylim(min(predicted_temps) - 5, max(predicted_temps) + 5)
            ax2.set_title("5-Day Temperature Trend", fontsize=14, fontweight='bold')
            ax2.set_xlabel("Day", fontsize=12)
            ax2.set_ylabel(f"Temperature ({unit.split()[1]})", fontsize=12)
            ax2.grid(True, linestyle='--', alpha=0.6)
            ax2.legend()

            # Display the plot in Streamlit
            st.pyplot(fig2)

            # Generate hourly temperature trend
            hourly_forecast = generate_hourly_forecast(predicted_temp, unit)
            fig3, ax3 = plt.subplots(figsize=(10, 6))
            ax3.plot(range(1, 25), hourly_forecast, marker='o', color=PRIMARY_COLOR, linewidth=2, label='Hourly Temperature Trend')

            # Add labels at each point
            for x, y in zip(range(1, 25), hourly_forecast):
                ax3.text(x, y, f"{y:.2f}{unit.split()[1]}", ha='center', va='bottom', fontsize=10, color=POINT_COLOR)

            # Set graph aesthetics
            ax3.set_ylim(min(hourly_forecast) - 5, max(hourly_forecast) + 5)
            ax3.set_title("Hourly Temperature Trend", fontsize=14, fontweight='bold')
            ax3.set_xlabel("Hour", fontsize=12)
            ax3.set_ylabel(f"Temperature ({unit.split()[1]})", fontsize=12)
            ax3.grid(True, linestyle='--', alpha=0.6)
            ax3.legend()

            # Display the plot in Streamlit
            st.pyplot(fig3)

            # Display table comparing predicted and actual temperatures
            st.subheader("Predicted vs Actual Temperatures")
            actual_temps = [predicted_temp + np.random.uniform(-2, 2) for _ in range(5)]  # Placeholder for actual temperatures
            comparison_df = pd.DataFrame({
                "Day": [f"Day {i+1}" for i in range(5)],
                "Predicted Temperature": predicted_temps,
                "Actual Temperature": actual_temps
            })
            st.table(comparison_df)

        except Exception as e:
            st.error(f"Error in prediction: {e}")
    else:
        st.warning("Please fetch weather data first.")

# Save Prediction
if st.button('Save Prediction'):
    if 'predicted_temp' in st.session_state:
        try:
            cursor = conn.cursor()
            cursor.execute(
                '''INSERT INTO weather_predictions (location, humidity, wind_speed, precipitation, predicted_temp)
                   VALUES (?, ?, ?, ?, ?)''',
                (st.session_state['location'], st.session_state['humidity'], 
                 st.session_state['wind_speed'], st.session_state['precipitation'], 
                 st.session_state['predicted_temp'])
            )
            conn.commit()
            st.success("Prediction saved successfully!")
        except Exception as e:
            st.error(f"Failed to save prediction: {e}")
    else:
        st.warning("Please generate a prediction before saving.")

# View Saved Predictions
if st.button("View Saved Predictions"):
    if conn:
        try:
            query = "SELECT * FROM weather_predictions ORDER BY timestamp DESC"
            saved_data = pd.read_sql(query, conn)

            if saved_data.empty:
                st.warning("No saved predictions found.")
            else:
                st.markdown("<h2 style='text-align: center; color: #00BFFF;'>Saved Predictions</h2>", unsafe_allow_html=True)
                html_table = saved_data.to_html(classes='dataframe-table', escape=False)

                st.markdown(
                    """
                    <style>
                    .scrollable-table { max-height: 400px; overflow-y: auto; }
                    .dataframe-table tbody tr:hover { background-color: #3c3c3c; transform: scale(1.05); }
                    .dataframe-table th { background-color: #4CAF50; color: white; }
                    .dataframe-table tbody tr:nth-child(even) { background-color: #2f2f2f; }
                    .dataframe-table tbody tr:nth-child(odd) { background-color: #3a3a3a; }
                    </style>
                    """, unsafe_allow_html=True
                )
                st.markdown(f'<div class="scrollable-table">{html_table}</div>', unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Failed to retrieve saved predictions: {e}")
    else:
        st.error("Database connection could not be established.")
