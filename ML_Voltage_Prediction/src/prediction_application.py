#!/usr/bin/env python3
"""
MFC Voltage Prediction Application
Practical application for making predictions on new MFC data
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import joblib
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class MFCPredictionApp:
    def __init__(self, models_dir="../models"):
        self.models_dir = Path(models_dir)
        self.models = {}
        self.scalers = {}
        self.load_trained_models()
    
    def load_trained_models(self):
        """Load all trained models and scalers"""
        print("Loading trained models...")
        
        try:
            # Load traditional models
            model_files = {
                'Random Forest': 'random_forest_model.pkl',
                'Linear Regression': 'linear_regression_model.pkl',
                'SVR': 'svr_model.pkl'
            }
            
            for name, filename in model_files.items():
                model_path = self.models_dir / filename
                if model_path.exists():
                    self.models[name] = joblib.load(model_path)
                    print(f"✓ {name} loaded")
            
            # Load LSTM model
            lstm_path = self.models_dir / "lstm_model.h5"
            if lstm_path.exists():
                self.models['LSTM'] = tf.keras.models.load_model(lstm_path)
                print("✓ LSTM loaded")
            
            # Load scalers
            self.scalers['X'] = joblib.load(self.models_dir / "scaler_X.pkl")
            self.scalers['y'] = joblib.load(self.models_dir / "scaler_y.pkl")
            print("✓ Scalers loaded")
            
            print(f"Total models loaded: {len(self.models)}")
            
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            print("Please run voltage_prediction_model.py first!")
    
    def extract_features_from_voltage(self, voltage_data):
        """Extract features from voltage time series"""
        features = {
            'mean_voltage': np.mean(voltage_data),
            'max_voltage': np.max(voltage_data),
            'min_voltage': np.min(voltage_data),
            'std_voltage': np.std(voltage_data),
            'final_voltage': voltage_data[-1],
            'initial_voltage': voltage_data[0],
            'voltage_trend': voltage_data[-1] - voltage_data[0],
            'peak_location': np.argmax(voltage_data) / len(voltage_data),
            'positive_hours': np.sum(voltage_data > 0),
            'voltage_range': np.max(voltage_data) - np.min(voltage_data),
            'avg_positive_voltage': np.mean(voltage_data[voltage_data > 0]) if np.any(voltage_data > 0) else 0,
            # Experiment dummy variables (assume new data is similar to exp3)
            'experiment_exp1': 0,
            'experiment_exp2': 0,
            'experiment_exp3': 1
        }
        return features
    
    def predict_voltage_performance(self, voltage_window, electrode_name="Unknown"):
        """Predict voltage performance from early-stage data"""
        print(f"\nPredicting performance for electrode: {electrode_name}")
        print(f"Input window: {len(voltage_window)} hours")
        print(f"Voltage range: {np.min(voltage_window):.2f}V to {np.max(voltage_window):.2f}V")
        
        # Extract features
        features = self.extract_features_from_voltage(voltage_window)
        features_df = pd.DataFrame([features])
        
        # Scale features
        features_scaled = self.scalers['X'].transform(features_df)
        
        # Make predictions with all models
        predictions = {}
        
        for model_name, model in self.models.items():
            if model_name == 'LSTM':
                # For LSTM, we need sequence data
                # Create a simple sequence by repeating the features
                seq_length = 5
                features_seq = np.repeat(features_scaled, seq_length, axis=0).reshape(1, seq_length, -1)
                pred_scaled = model.predict(features_seq, verbose=0)
            else:
                pred_scaled = model.predict(features_scaled)
            
            # Inverse transform prediction
            pred_original = self.scalers['y'].inverse_transform(pred_scaled.reshape(-1, 1))[0, 0]
            predictions[model_name] = pred_original
        
        # Calculate ensemble prediction (average of all models)
        ensemble_pred = np.mean(list(predictions.values()))
        predictions['Ensemble'] = ensemble_pred
        
        return predictions
    
    def classify_electrode_performance(self, predictions):
        """Classify electrode performance based on predictions"""
        ensemble_pred = predictions['Ensemble']
        
        # Define performance categories based on historical data
        if ensemble_pred > 140:
            category = "Excellent"
            confidence = "High"
        elif ensemble_pred > 100:
            category = "Good" 
            confidence = "Medium"
        elif ensemble_pred > 60:
            category = "Fair"
            confidence = "Medium"
        else:
            category = "Poor"
            confidence = "High"
        
        return category, confidence
    
    def generate_prediction_report(self, voltage_window, electrode_name, predictions):
        """Generate a comprehensive prediction report"""
        category, confidence = self.classify_electrode_performance(predictions)
        
        report = []
        report.append("="*60)
        report.append("MFC VOLTAGE PREDICTION REPORT")
        report.append("="*60)
        report.append(f"Electrode: {electrode_name}")
        report.append(f"Input Data: {len(voltage_window)} hours")
        report.append(f"Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Current performance
        current_max = np.max(voltage_window)
        current_final = voltage_window[-1]
        current_trend = voltage_window[-1] - voltage_window[0]
        
        report.append("CURRENT PERFORMANCE (Input Window):")
        report.append("-" * 40)
        report.append(f"Maximum Voltage: {current_max:.2f} V")
        report.append(f"Final Voltage: {current_final:.2f} V")
        report.append(f"Voltage Trend: {current_trend:+.2f} V")
        report.append(f"Positive Hours: {np.sum(voltage_window > 0)}/{len(voltage_window)}")
        report.append("")
        
        # Predictions
        report.append("PREDICTED MAXIMUM VOLTAGE:")
        report.append("-" * 40)
        for model_name, pred in predictions.items():
            if model_name == 'Ensemble':
                report.append(f"🎯 {model_name}: {pred:.2f} V")
            else:
                report.append(f"   {model_name}: {pred:.2f} V")
        report.append("")
        
        # Performance classification
        report.append("PERFORMANCE CLASSIFICATION:")
        report.append("-" * 40)
        report.append(f"Category: {category}")
        report.append(f"Confidence: {confidence}")
        report.append("")
        
        # Recommendations
        report.append("RECOMMENDATIONS:")
        report.append("-" * 40)
        
        if category == "Excellent":
            report.append("✓ Continue current operation - excellent performance expected")
            report.append("✓ This electrode configuration is optimal")
            report.append("✓ Consider scaling up for industrial application")
        elif category == "Good":
            report.append("✓ Good performance expected - continue monitoring")
            report.append("• Monitor for any performance decline")
            report.append("• Consider minor optimizations")
        elif category == "Fair":
            report.append("⚠ Fair performance - optimization recommended")
            report.append("• Check electrode preparation and biofilm development")
            report.append("• Consider operational parameter adjustments")
        else:
            report.append("❌ Poor performance expected")
            report.append("• Recommend electrode replacement or modification")
            report.append("• Review experimental conditions")
            report.append("• Consider alternative electrode materials")
        
        report.append("")
        report.append("INDUSTRIAL IMPLICATIONS:")
        report.append("-" * 40)
        predicted_improvement = ((predictions['Ensemble'] - current_max) / current_max) * 100
        report.append(f"Expected performance improvement: {predicted_improvement:+.1f}%")
        
        if predicted_improvement > 20:
            report.append("• High potential for continued improvement")
        elif predicted_improvement > 0:
            report.append("• Moderate potential for improvement")
        else:
            report.append("• Performance may have peaked")
        
        return report
    
    def visualize_prediction(self, voltage_window, predictions, electrode_name):
        """Create prediction visualization"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'MFC Prediction Analysis: {electrode_name}', fontsize=16, fontweight='bold')
        
        # Plot 1: Input voltage evolution
        hours = range(len(voltage_window))
        ax1.plot(hours, voltage_window, linewidth=3, color='#2E86AB', marker='o', markersize=4)
        ax1.set_xlabel('Time (Hours)')
        ax1.set_ylabel('Voltage (V)')
        ax1.set_title('Input Voltage Evolution')
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        
        # Add trend line
        z = np.polyfit(hours, voltage_window, 1)
        p = np.poly1d(z)
        ax1.plot(hours, p(hours), "r--", alpha=0.8, linewidth=2, label=f'Trend: {z[0]:+.2f}V/h')
        ax1.legend()
        
        # Plot 2: Model predictions comparison
        model_names = list(predictions.keys())
        pred_values = list(predictions.values())
        
        colors = ['#2E86AB', '#A23B72', '#F18F01', '#8E44AD', '#E74C3C']
        bars = ax2.bar(model_names, pred_values, color=colors[:len(model_names)])
        ax2.set_ylabel('Predicted Max Voltage (V)')
        ax2.set_title('Model Predictions Comparison')
        ax2.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, pred_values):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{value:.1f}V', ha='center', va='bottom', fontweight='bold')
        
        # Rotate x-axis labels for better readability
        ax2.tick_params(axis='x', rotation=45)
        
        # Plot 3: Performance classification
        ensemble_pred = predictions['Ensemble']
        current_max = np.max(voltage_window)
        
        categories = ['Poor\n(0-60V)', 'Fair\n(60-100V)', 'Good\n(100-140V)', 'Excellent\n(140V+)']
        category_ranges = [30, 80, 120, 160]  # Midpoints for visualization
        category_colors = ['#E74C3C', '#F39C12', '#27AE60', '#2E86AB']
        
        # Show all categories with current prediction highlighted
        bars_cat = ax3.bar(categories, [60, 40, 40, 40], color=category_colors, alpha=0.3)
        
        # Highlight the predicted category
        if ensemble_pred > 140:
            idx = 3
        elif ensemble_pred > 100:
            idx = 2
        elif ensemble_pred > 60:
            idx = 1
        else:
            idx = 0
        
        bars_cat[idx].set_alpha(1.0)
        
        ax3.axhline(y=ensemble_pred, color='red', linewidth=3, linestyle='-', 
                   label=f'Prediction: {ensemble_pred:.1f}V')
        ax3.axhline(y=current_max, color='blue', linewidth=3, linestyle='--', 
                   label=f'Current Max: {current_max:.1f}V')
        
        ax3.set_ylabel('Voltage (V)')
        ax3.set_title('Performance Classification')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Prediction confidence intervals
        # Simulate confidence intervals based on model agreement
        pred_std = np.std(list(predictions.values())[:-1])  # Exclude ensemble
        ensemble_pred = predictions['Ensemble']
        
        confidence_levels = [0.68, 0.95]  # 1σ and 2σ
        colors_conf = ['#3498DB', '#5DADE2']
        
        for i, (conf, color) in enumerate(zip(confidence_levels, colors_conf)):
            lower = ensemble_pred - (i+1) * pred_std
            upper = ensemble_pred + (i+1) * pred_std
            ax4.fill_between([0, 1], [lower, lower], [upper, upper], 
                           alpha=0.3, color=color, label=f'{conf*100:.0f}% Confidence')
        
        ax4.axhline(y=ensemble_pred, color='red', linewidth=3, label=f'Prediction: {ensemble_pred:.1f}V')
        ax4.axhline(y=current_max, color='blue', linewidth=2, linestyle='--', label=f'Current: {current_max:.1f}V')
        
        ax4.set_xlim(0, 1)
        ax4.set_ylabel('Voltage (V)')
        ax4.set_title('Prediction Confidence')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.set_xticks([])
        
        plt.tight_layout()
        
        # Save plot
        results_dir = Path("../results")
        results_dir.mkdir(exist_ok=True)
        filename = f"prediction_{electrode_name.lower().replace(' ', '_')}.png"
        plt.savefig(results_dir / filename, dpi=300, bbox_inches='tight')
        
        print(f"✓ Prediction plot saved: {filename}")
        plt.show()

def demo_prediction():
    """Demonstrate prediction with sample data"""
    print("="*60)
    print("MFC VOLTAGE PREDICTION - DEMO APPLICATION")
    print("="*60)
    
    # Initialize app
    app = MFCPredictionApp()
    
    if not app.models:
        print("❌ No models found. Please train models first!")
        return
    
    # Create sample voltage data (first 24 hours)
    # Simulating a typical voltage evolution pattern
    hours = np.arange(24)
    
    # Sample 1: Good performing electrode
    voltage_good = np.array([
        -20, -15, -10, -5, 0, 5, 15, 25, 35, 45, 55, 65,
        75, 85, 95, 100, 105, 110, 115, 118, 120, 122, 125, 128
    ])
    
    # Sample 2: Poor performing electrode  
    voltage_poor = np.array([
        -30, -25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25,
        28, 30, 32, 35, 37, 38, 40, 41, 42, 43, 44, 45
    ])
    
    # Make predictions for both samples
    samples = [
        ("Good_Electrode_Demo", voltage_good),
        ("Poor_Electrode_Demo", voltage_poor)
    ]
    
    for electrode_name, voltage_data in samples:
        print(f"\n{'='*60}")
        print(f"ANALYZING: {electrode_name}")
        print('='*60)
        
        # Make prediction
        predictions = app.predict_voltage_performance(voltage_data, electrode_name)
        
        # Generate report
        report = app.generate_prediction_report(voltage_data, electrode_name, predictions)
        
        # Print report
        for line in report:
            print(line)
        
        # Save report
        results_dir = Path("../results")
        results_dir.mkdir(exist_ok=True)
        
        report_filename = f"prediction_report_{electrode_name.lower()}.txt"
        with open(results_dir / report_filename, 'w') as f:
            f.write('\n'.join(report))
        
        print(f"\n✓ Report saved: {report_filename}")
        
        # Create visualization
        app.visualize_prediction(voltage_data, predictions, electrode_name)
    
    print(f"\n{'='*60}")
    print("DEMO COMPLETE!")
    print("="*60)
    print("Use this application with your own 24-hour voltage data")
    print("to predict MFC performance!")

if __name__ == "__main__":
    demo_prediction()