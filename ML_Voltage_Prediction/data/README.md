# Data Directory

This directory contains processed datasets for machine learning training.

## Generated Files

After running the data preparation pipeline, you'll find:

- **features.csv** - Extracted features from voltage time series (input data)
- **targets.csv** - Target variables for prediction (output data)  
- **metadata.csv** - Experiment and electrode information

## Data Description

### Features (X)
- `mean_voltage`: Average voltage in input window
- `max_voltage`: Maximum voltage reached
- `min_voltage`: Minimum voltage value
- `std_voltage`: Voltage standard deviation
- `final_voltage`: Last voltage measurement
- `initial_voltage`: First voltage measurement
- `voltage_trend`: Overall voltage change
- `peak_location`: Relative position of peak voltage
- `positive_hours`: Hours with positive voltage
- `voltage_range`: Voltage range (max - min)
- `avg_positive_voltage`: Average of positive voltages
- `experiment_*`: One-hot encoded experiment indicators

### Targets (y)
Same feature set extracted from the prediction window (next 24 hours)

### Metadata
- `experiment`: Source experiment (exp1, exp2, exp3)
- `electrode`: Electrode type/name
- `start_hour`: Starting hour of the prediction window

## Data Processing Pipeline

1. **Raw Data**: Original CSV files from experiments
2. **Cleaning**: Handle missing values, convert formats
3. **Windowing**: Create 24-hour input/output windows
4. **Feature Extraction**: Calculate statistical features
5. **Standardization**: Scale features for ML training

This processed data enables training ML models to predict MFC performance from early-stage measurements.