CREATE TABLE weather_predictions (
    id INT IDENTITY(1,1) PRIMARY KEY,
    location VARCHAR(255),
    humidity FLOAT,
    wind_speed FLOAT,
    precipitation FLOAT,
    predicted_temp FLOAT,
    timestamp DATETIME DEFAULT GETDATE()
);
