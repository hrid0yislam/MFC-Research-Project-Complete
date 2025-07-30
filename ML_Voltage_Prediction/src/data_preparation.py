#!/usr/bin/env python3
"""
Data Preparation for MFC Voltage Prediction
Load and process experimental data for machine learning
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class MFCDataProcessor:
    def __init__(self, project_root="../"):
        self.project_root = Path(project_root)
        self.data = {}
        self.processed_data = {}
        
    def load_experimental_data(self):
        """Load all experimental data"""
        print("Loading MFC experimental data...")
        
        # Experiment 1 - Load from EXP 1 folder
        try:
            exp1_path = self.project_root / "EXP 1" / "Final DATA EXP 1.csv"
            self.data['exp1'] = pd.read_csv(exp1_path)
            print(f"✓ Experiment 1 loaded: {len(self.data['exp1'])} data points")
        except Exception as e:
            print(f"✗ Could not load Experiment 1: {e}")
            
        # Experiment 2 - Load from EXP 2 folder
        try:
            exp2_path = self.project_root / "EXP 2" / "Final data_EXP_2.csv"
            self.data['exp2'] = pd.read_csv(exp2_path)
            print(f"✓ Experiment 2 loaded: {len(self.data['exp2'])} data points")
        except Exception as e:
            print(f"✗ Could not load Experiment 2: {e}")
            
        # Experiment 3 - Load from Exp 3 folder
        try:
            exp3_path = self.project_root / "Exp 3 Mod vs ssm" / "MOD vs SSM.csv"
            self.data['exp3'] = pd.read_csv(exp3_path)
            print(f"✓ Experiment 3 loaded: {len(self.data['exp3'])} data points")
        except Exception as e:
            print(f"✗ Could not load Experiment 3: {e}")
            
        return self.data
    
    def clean_and_standardize_data(self):
        """Clean and standardize all experimental data"""
        print("\nCleaning and standardizing data...")
        
        for exp_name, df in self.data.items():
            print(f"\nProcessing {exp_name}...")
            
            # Basic cleaning
            df_clean = df.copy()
            
            # Create time column (hours)
            df_clean['Hours'] = range(len(df_clean))
            
            # Identify voltage columns (exclude time columns)
            voltage_columns = [col for col in df_clean.columns 
                             if col not in ['Hours', 'Time', 'Time (Hours)', 'Unnamed: 0']]
            
            print(f"  Voltage columns found: {voltage_columns}")
            
            # Clean voltage data
            for col in voltage_columns:
                if col in df_clean.columns:
                    # Convert to numeric, replacing any non-numeric values with NaN
                    df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
                    
                    # Fill NaN values with interpolation
                    df_clean[col] = df_clean[col].interpolate(method='linear')
                    
            # Store cleaned data
            self.processed_data[exp_name] = {
                'data': df_clean,
                'voltage_columns': voltage_columns,
                'duration_hours': len(df_clean)
            }
            
            print(f"  ✓ Cleaned {len(df_clean)} data points")
            print(f"  ✓ Duration: {len(df_clean)} hours")
            
    def create_prediction_windows(self, input_hours=24, prediction_hours=24):
        """Create sliding windows for prediction"""
        print(f"\nCreating prediction windows:")
        print(f"Input window: {input_hours} hours")
        print(f"Prediction window: {prediction_hours} hours")
        
        training_data = []
        
        for exp_name, exp_data in self.processed_data.items():
            df = exp_data['data']
            voltage_cols = exp_data['voltage_columns']
            
            print(f"\nProcessing {exp_name}...")
            
            for col in voltage_cols:
                voltage_series = df[col].values
                
                # Create sliding windows
                for i in range(len(voltage_series) - input_hours - prediction_hours + 1):
                    # Input window (features)
                    X = voltage_series[i:i + input_hours]
                    
                    # Prediction window (target)
                    y = voltage_series[i + input_hours:i + input_hours + prediction_hours]
                    
                    # Store with metadata
                    training_data.append({
                        'experiment': exp_name,
                        'electrode': col,
                        'start_hour': i,
                        'X': X,  # Input features
                        'y': y,  # Target values
                        'peak_voltage_in_X': np.max(X),
                        'final_voltage_in_X': X[-1],
                        'trend_in_X': X[-1] - X[0],
                        'max_voltage_in_y': np.max(y),
                        'final_voltage_in_y': y[-1]
                    })
            
            print(f"  ✓ Created {len([d for d in training_data if d['experiment'] == exp_name])} windows")
        
        print(f"\nTotal training samples: {len(training_data)}")
        return training_data
    
    def extract_features(self, voltage_window):
        """Extract statistical features from voltage window"""
        features = {
            'mean_voltage': np.mean(voltage_window),
            'max_voltage': np.max(voltage_window),
            'min_voltage': np.min(voltage_window),
            'std_voltage': np.std(voltage_window),
            'final_voltage': voltage_window[-1],
            'initial_voltage': voltage_window[0],
            'voltage_trend': voltage_window[-1] - voltage_window[0],
            'peak_location': np.argmax(voltage_window) / len(voltage_window),
            'positive_hours': np.sum(voltage_window > 0),
            'voltage_range': np.max(voltage_window) - np.min(voltage_window),
            'avg_positive_voltage': np.mean(voltage_window[voltage_window > 0]) if np.any(voltage_window > 0) else 0
        }
        return features
    
    def prepare_ml_dataset(self, training_data):
        """Prepare final dataset for ML training"""
        print("\nPreparing ML dataset...")
        
        # Extract features and targets
        feature_list = []
        target_list = []
        metadata_list = []
        
        for sample in training_data:
            # Extract features from input window
            features = self.extract_features(sample['X'])
            
            # Add metadata as features
            features['experiment_exp1'] = 1 if sample['experiment'] == 'exp1' else 0
            features['experiment_exp2'] = 1 if sample['experiment'] == 'exp2' else 0
            features['experiment_exp3'] = 1 if sample['experiment'] == 'exp3' else 0
            
            feature_list.append(features)
            
            # Target: next window statistics
            target_stats = self.extract_features(sample['y'])
            target_list.append(target_stats)
            
            # Metadata
            metadata_list.append({
                'experiment': sample['experiment'],
                'electrode': sample['electrode'],
                'start_hour': sample['start_hour']
            })
        
        # Convert to DataFrames
        X_df = pd.DataFrame(feature_list)
        y_df = pd.DataFrame(target_list)
        metadata_df = pd.DataFrame(metadata_list)
        
        print(f"✓ Features shape: {X_df.shape}")
        print(f"✓ Targets shape: {y_df.shape}")
        print(f"✓ Feature columns: {list(X_df.columns)}")
        
        return X_df, y_df, metadata_df
    
    def save_processed_data(self, X_df, y_df, metadata_df):
        """Save processed data for ML training"""
        print("\nSaving processed data...")
        
        # Create data directory
        data_dir = Path("../data")
        data_dir.mkdir(exist_ok=True)
        
        # Save datasets
        X_df.to_csv(data_dir / "features.csv", index=False)
        y_df.to_csv(data_dir / "targets.csv", index=False)
        metadata_df.to_csv(data_dir / "metadata.csv", index=False)
        
        print(f"✓ Saved to {data_dir}/")
        print("  - features.csv")
        print("  - targets.csv") 
        print("  - metadata.csv")
        
    def visualize_data_overview(self):
        """Create overview visualizations"""
        print("\nCreating data overview visualizations...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('MFC Experimental Data Overview', fontsize=16, fontweight='bold')
        
        # Plot 1: Duration comparison
        exp_names = list(self.processed_data.keys())
        durations = [self.processed_data[exp]['duration_hours'] for exp in exp_names]
        
        axes[0,0].bar(exp_names, durations, color=['#2E86AB', '#A23B72', '#F18F01'])
        axes[0,0].set_title('Experiment Duration')
        axes[0,0].set_ylabel('Hours')
        axes[0,0].grid(True, alpha=0.3)
        
        # Plot 2: Electrode count per experiment
        electrode_counts = [len(self.processed_data[exp]['voltage_columns']) for exp in exp_names]
        
        axes[0,1].bar(exp_names, electrode_counts, color=['#2E86AB', '#A23B72', '#F18F01'])
        axes[0,1].set_title('Number of Electrodes per Experiment')
        axes[0,1].set_ylabel('Electrode Count')
        axes[0,1].grid(True, alpha=0.3)
        
        # Plot 3: Sample voltage evolution from each experiment
        for i, (exp_name, exp_data) in enumerate(self.processed_data.items()):
            df = exp_data['data']
            first_col = exp_data['voltage_columns'][0]
            
            axes[1,0].plot(df['Hours'], df[first_col], 
                          label=f'{exp_name} - {first_col}', linewidth=2)
        
        axes[1,0].set_title('Sample Voltage Evolution')
        axes[1,0].set_xlabel('Time (Hours)')
        axes[1,0].set_ylabel('Voltage (V)')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)
        
        # Plot 4: Data availability matrix
        availability_data = []
        for exp_name, exp_data in self.processed_data.items():
            df = exp_data['data']
            for col in exp_data['voltage_columns']:
                availability_data.append({
                    'Experiment': exp_name,
                    'Electrode': col,
                    'Data_Points': len(df[col].dropna()),
                    'Coverage': len(df[col].dropna()) / len(df) * 100
                })
        
        avail_df = pd.DataFrame(availability_data)
        
        # Create heatmap-style visualization
        axes[1,1].scatter(avail_df['Experiment'], avail_df['Electrode'], 
                         s=avail_df['Data_Points'], c=avail_df['Coverage'],
                         cmap='viridis', alpha=0.7)
        axes[1,1].set_title('Data Availability (Size=Points, Color=Coverage%)')
        axes[1,1].set_xlabel('Experiment')
        axes[1,1].set_ylabel('Electrode')
        
        plt.tight_layout()
        
        # Save plot
        results_dir = Path("../results")
        results_dir.mkdir(exist_ok=True)
        plt.savefig(results_dir / "data_overview.png", dpi=300, bbox_inches='tight')
        plt.savefig(results_dir / "data_overview.pdf", dpi=300, bbox_inches='tight')
        
        print(f"✓ Saved data overview plots to {results_dir}/")
        plt.show()

def main():
    """Main data preparation pipeline"""
    print("="*60)
    print("MFC VOLTAGE PREDICTION - DATA PREPARATION")
    print("="*60)
    
    # Initialize processor
    processor = MFCDataProcessor()
    
    # Load experimental data
    data = processor.load_experimental_data()
    
    if not data:
        print("❌ No data loaded. Please check file paths.")
        return
    
    # Clean and standardize
    processor.clean_and_standardize_data()
    
    # Create prediction windows
    training_data = processor.create_prediction_windows(
        input_hours=24,      # Use first 24 hours
        prediction_hours=24  # Predict next 24 hours
    )
    
    # Prepare ML dataset
    X_df, y_df, metadata_df = processor.prepare_ml_dataset(training_data)
    
    # Save processed data
    processor.save_processed_data(X_df, y_df, metadata_df)
    
    # Create visualizations
    processor.visualize_data_overview()
    
    print("\n" + "="*60)
    print("DATA PREPARATION COMPLETE!")
    print("="*60)
    print("Next step: Run voltage_prediction_model.py")

if __name__ == "__main__":
    main()