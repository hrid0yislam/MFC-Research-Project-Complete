"""
MFC Voltage Prediction - Source Code Package

This package contains the core machine learning implementation for predicting
MFC (Microbial Fuel Cell) voltage evolution based on early-stage performance data.

Modules:
- data_preparation: Load and process experimental data for ML
- voltage_prediction_model: Train and evaluate ML models
- prediction_application: Practical application for making predictions

Author: MFC Research Team
Purpose: Academic research and industrial MFC optimization
"""

__version__ = "1.0.0"
__author__ = "MFC Research Team"

# Make key classes available at package level
from .data_preparation import MFCDataProcessor
from .voltage_prediction_model import MFCVoltagePredictor
from .prediction_application import MFCPredictionApp

__all__ = [
    "MFCDataProcessor",
    "MFCVoltagePredictor", 
    "MFCPredictionApp"
]