# MFC Voltage Prediction using Machine Learning

## Project Overview
This project implements machine learning models to predict MFC (Microbial Fuel Cell) voltage evolution based on early-stage performance data.

## Objective
Predict MFC system performance after 24-48 hours instead of waiting for complete 200+ hour experiments.

## Data Sources
- Experiment 1: Artificial dairy wastewater data
- Experiment 2: 302-hour comparative study (CB-SSM, Toray, SSM, Carbon Paper)
- Experiment 3: 233-hour fish farm wastewater study (Modified vs SSM)

## Project Structure
```
ML_Voltage_Prediction/
├── data/                    # Processed data for ML
├── models/                  # Trained ML models
├── notebooks/              # Jupyter notebooks for analysis
├── results/                # Model outputs and predictions
├── src/                    # Source code
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## Key Features
- LSTM-based voltage prediction
- Early performance classification
- Treatment efficiency correlation
- Industrial application insights

## Results Expected
- Predict peak voltage timing with >90% accuracy
- Identify high/low performers within 24 hours
- Reduce experimental time by 80-90%

## Academic Value
- Novel application of ML to bioelectrochemical systems
- Practical engineering solution for MFC optimization
- Strong thesis contribution for sustainable technology