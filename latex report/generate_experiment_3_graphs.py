#!/usr/bin/env python3
"""
Graph generation script for MFC Experiment 3: Modified vs SSM Electrode Comparison
BRIDGE Project - UiT Narvik
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style for publication-quality plots
try:
    plt.style.use('seaborn-v0_8')
except:
    try:
        plt.style.use('seaborn')
    except:
        plt.style.use('default')
sns.set_palette("husl")

def load_data():
    """Load and prepare experiment 3 data"""
    try:
        # Load the voltage data
        data_path = Path("../Exp 3 Mod vs ssm/MOD vs SSM.csv")
        df = pd.read_csv(data_path)
        
        # Clean column names
        df.columns = ['Time', 'Modified_10pct_CB_SSM', 'SSM']
        
        # Convert time to hours (assuming it's already in hours)
        # Extract numeric hours from time string
        df['Hours'] = df.index  # Use index as hours since data is hourly
        
        print(f"Voltage data loaded successfully: {len(df)} data points")
        print(f"Time range: 0 to {len(df)-1} hours")
        print(f"Modified electrode voltage range: {df['Modified_10pct_CB_SSM'].min():.2f}V to {df['Modified_10pct_CB_SSM'].max():.2f}V")
        print(f"SSM electrode voltage range: {df['SSM'].min():.2f}V to {df['SSM'].max():.2f}V")
        
        return df
        
    except Exception as e:
        print(f"Error loading voltage data: {e}")
        return None

def load_treatment_data():
    """Load and prepare treatment efficiency data"""
    try:
        # Load the fish farm wastewater treatment data
        treatment_path = Path("../fish_farm_wastewater.csv")
        treatment_df = pd.read_csv(treatment_path, index_col=0)
        
        print(f"Treatment data loaded successfully")
        print(f"Parameters: {list(treatment_df.index)}")
        print(f"Conditions: {list(treatment_df.columns)}")
        
        return treatment_df
        
    except Exception as e:
        print(f"Error loading treatment data: {e}")
        return None

def create_voltage_evolution_plot(df):
    """Create voltage evolution comparison plot"""
    plt.figure(figsize=(14, 8))
    
    # Plot both electrodes
    plt.plot(df['Hours'], df['Modified_10pct_CB_SSM'], 
             linewidth=2.5, label='Modified (10% CB+SSM)', color='#2E86AB', alpha=0.8)
    plt.plot(df['Hours'], df['SSM'], 
             linewidth=2.5, label='SSM (Pristine)', color='#A23B72', alpha=0.8)
    
    # Add phase annotations
    plt.axvspan(0, 24, alpha=0.1, color='red', label='Initial Phase')
    plt.axvspan(24, 72, alpha=0.1, color='orange', label='Establishment Phase')  
    plt.axvspan(72, 150, alpha=0.1, color='green', label='Growth/Peak Phase')
    plt.axvspan(150, 233, alpha=0.1, color='blue', label='Stability Phase')
    
    # Formatting
    plt.xlabel('Time (Hours)', fontsize=12, fontweight='bold')
    plt.ylabel('Voltage (V)', fontsize=12, fontweight='bold')
    plt.title('Experiment 3: Voltage Evolution Comparison\nModified (10% CB+SSM) vs. Pristine SSM Electrodes', 
              fontsize=14, fontweight='bold', pad=20)
    
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper left', fontsize=10, frameon=True)
    
    # Add statistics box
    max_mod = df['Modified_10pct_CB_SSM'].max()
    max_ssm = df['SSM'].max()
    enhancement = ((max_mod - max_ssm) / max_ssm) * 100
    
    stats_text = f'Peak Performance:\nModified: {max_mod:.1f}V\nSSM: {max_ssm:.1f}V\nEnhancement: {enhancement:.1f}%\n\nFish Farm Wastewater\nTreatment Study'
    plt.text(0.98, 0.02, stats_text, transform=plt.gca().transAxes, 
             verticalalignment='bottom', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
             fontsize=9)
    
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('experiment_3_voltage_evolution.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('experiment_3_voltage_evolution.png', dpi=300, bbox_inches='tight')
    print("Voltage evolution plot saved")
    plt.show()

def create_performance_comparison_plot(df):
    """Create performance comparison analysis"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Peak performance comparison (bar chart)
    electrodes = ['Modified\n(10% CB+SSM)', 'SSM\n(Pristine)']
    peak_voltages = [df['Modified_10pct_CB_SSM'].max(), df['SSM'].max()]
    colors = ['#2E86AB', '#A23B72']
    
    bars = ax1.bar(electrodes, peak_voltages, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax1.set_ylabel('Peak Voltage (V)', fontweight='bold')
    ax1.set_title('Peak Performance Comparison', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, voltage in zip(bars, peak_voltages):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 2,
                f'{voltage:.1f}V', ha='center', va='bottom', fontweight='bold')
    
    # 2. Performance phases analysis
    phases = ['0-24h\n(Initial)', '24-72h\n(Establishment)', '72-150h\n(Growth/Peak)', '150-233h\n(Stability)']
    
    # Calculate average voltages for each phase
    phase_ranges = [(0, 24), (24, 72), (72, 150), (150, 233)]
    mod_averages = []
    ssm_averages = []
    
    for start, end in phase_ranges:
        mod_avg = df.iloc[start:end]['Modified_10pct_CB_SSM'].mean()
        ssm_avg = df.iloc[start:end]['SSM'].mean()
        mod_averages.append(mod_avg)
        ssm_averages.append(ssm_avg)
    
    x = np.arange(len(phases))
    width = 0.35
    
    bars1 = ax2.bar(x - width/2, mod_averages, width, label='Modified (10% CB+SSM)', 
                    color='#2E86AB', alpha=0.8, edgecolor='black')
    bars2 = ax2.bar(x + width/2, ssm_averages, width, label='SSM (Pristine)', 
                    color='#A23B72', alpha=0.8, edgecolor='black')
    
    ax2.set_ylabel('Average Voltage (V)', fontweight='bold')
    ax2.set_title('Performance by Operational Phase', fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(phases)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Performance enhancement over time
    enhancement_ratio = (df['Modified_10pct_CB_SSM'] / df['SSM']) * 100
    # Handle division by zero or negative values
    enhancement_ratio = enhancement_ratio.replace([np.inf, -np.inf], np.nan)
    
    # Use moving average to smooth the data
    window_size = 12  # 12-hour moving average
    if len(enhancement_ratio) > window_size:
        enhancement_smooth = enhancement_ratio.rolling(window=window_size, center=True).mean()
    else:
        enhancement_smooth = enhancement_ratio
    
    ax3.plot(df['Hours'], enhancement_smooth, color='#F18F01', linewidth=2.5)
    ax3.axhline(y=100, color='black', linestyle='--', alpha=0.7, label='Equal Performance')
    ax3.set_xlabel('Time (Hours)', fontweight='bold')
    ax3.set_ylabel('Modified/SSM Performance Ratio (%)', fontweight='bold')
    ax3.set_title('Relative Performance Enhancement Over Time', fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # 4. Voltage distribution comparison (box plot)
    voltage_data = [df['Modified_10pct_CB_SSM'].dropna(), df['SSM'].dropna()]
    
    box_plot = ax4.boxplot(voltage_data, labels=['Modified\n(10% CB+SSM)', 'SSM\n(Pristine)'],
                          patch_artist=True, notch=True)
    
    # Color the boxes
    for patch, color in zip(box_plot['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)
    
    ax4.set_ylabel('Voltage (V)', fontweight='bold')
    ax4.set_title('Voltage Distribution Comparison', fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('experiment_3_performance_comparison.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('experiment_3_performance_comparison.png', dpi=300, bbox_inches='tight')
    print("Performance comparison plot saved")
    plt.show()

def create_statistical_summary_plot(df):
    """Create statistical summary and analysis plots"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Cumulative performance comparison
    mod_cumsum = df['Modified_10pct_CB_SSM'].cumsum()
    ssm_cumsum = df['SSM'].cumsum()
    
    ax1.plot(df['Hours'], mod_cumsum, label='Modified (10% CB+SSM)', color='#2E86AB', linewidth=2.5)
    ax1.plot(df['Hours'], ssm_cumsum, label='SSM (Pristine)', color='#A23B72', linewidth=2.5)
    ax1.set_xlabel('Time (Hours)', fontweight='bold')
    ax1.set_ylabel('Cumulative Voltage (V·h)', fontweight='bold')
    ax1.set_title('Cumulative Energy Generation Potential', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Performance stability analysis (rolling standard deviation)
    window = 24  # 24-hour rolling window
    mod_rolling_std = df['Modified_10pct_CB_SSM'].rolling(window=window).std()
    ssm_rolling_std = df['SSM'].rolling(window=window).std()
    
    ax2.plot(df['Hours'], mod_rolling_std, label='Modified (10% CB+SSM)', color='#2E86AB', linewidth=2)
    ax2.plot(df['Hours'], ssm_rolling_std, label='SSM (Pristine)', color='#A23B72', linewidth=2)
    ax2.set_xlabel('Time (Hours)', fontweight='bold')
    ax2.set_ylabel('24h Rolling Standard Deviation (V)', fontweight='bold')
    ax2.set_title('Performance Stability Analysis', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Enhancement factor over time
    # Calculate enhancement factor more carefully
    valid_mask = (df['SSM'] != 0) & (df['SSM'].notna()) & (df['Modified_10pct_CB_SSM'].notna())
    enhancement_factor = np.where(valid_mask, 
                                 ((df['Modified_10pct_CB_SSM'] - df['SSM']) / df['SSM'].abs()) * 100,
                                 np.nan)
    
    # Smooth the enhancement factor
    enhancement_df = pd.Series(enhancement_factor)
    enhancement_smooth = enhancement_df.rolling(window=12, center=True).mean()
    
    ax3.plot(df['Hours'], enhancement_smooth, color='#F18F01', linewidth=2.5)
    ax3.axhline(y=0, color='black', linestyle='--', alpha=0.7, label='No Enhancement')
    ax3.set_xlabel('Time (Hours)', fontweight='bold')
    ax3.set_ylabel('Enhancement Factor (%)', fontweight='bold')
    ax3.set_title('Carbon Black Enhancement Effect Over Time', fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Performance statistics summary
    stats_data = {
        'Electrode': ['Modified (10% CB+SSM)', 'SSM (Pristine)'],
        'Mean (V)': [df['Modified_10pct_CB_SSM'].mean(), df['SSM'].mean()],
        'Max (V)': [df['Modified_10pct_CB_SSM'].max(), df['SSM'].max()],
        'Min (V)': [df['Modified_10pct_CB_SSM'].min(), df['SSM'].min()],
        'Std (V)': [df['Modified_10pct_CB_SSM'].std(), df['SSM'].std()]
    }
    
    stats_df = pd.DataFrame(stats_data)
    
    # Create a table
    ax4.axis('tight')
    ax4.axis('off')
    
    table = ax4.table(cellText=[[f'{val:.2f}' if isinstance(val, (int, float)) else val 
                                for val in row] for row in stats_df.values],
                     colLabels=['Electrode Type', 'Mean (V)', 'Max (V)', 'Min (V)', 'Std Dev (V)'],
                     cellLoc='center',
                     loc='center')
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    # Style the table
    for (i, j), cell in table.get_celld().items():
        if i == 0:
            cell.set_text_props(weight='bold')
            cell.set_facecolor('#E8E8E8')
        elif j == 0:
            cell.set_text_props(weight='bold')
            cell.set_facecolor('#F5F5F5')
        else:
            cell.set_facecolor('#FFFFFF')
    
    ax4.set_title('Statistical Summary', fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('experiment_3_statistical_summary.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('experiment_3_statistical_summary.png', dpi=300, bbox_inches='tight')
    print("Statistical summary plot saved")
    plt.show()

def create_timeline_analysis_plot(df):
    """Create detailed timeline analysis"""
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 14))
    
    # 1. Detailed voltage evolution with key events
    ax1.plot(df['Hours'], df['Modified_10pct_CB_SSM'], 
             linewidth=2.5, label='Modified (10% CB+SSM)', color='#2E86AB', alpha=0.8)
    ax1.plot(df['Hours'], df['SSM'], 
             linewidth=2.5, label='SSM (Pristine)', color='#A23B72', alpha=0.8)
    
    # Mark key events
    mod_max_idx = df['Modified_10pct_CB_SSM'].idxmax()
    ssm_max_idx = df['SSM'].idxmax()
    
    ax1.scatter(mod_max_idx, df.loc[mod_max_idx, 'Modified_10pct_CB_SSM'], 
               color='#2E86AB', s=100, marker='o', edgecolors='black', linewidth=2,
               label=f'Modified Peak: {df.loc[mod_max_idx, "Modified_10pct_CB_SSM"]:.1f}V at {mod_max_idx}h')
    
    ax1.scatter(ssm_max_idx, df.loc[ssm_max_idx, 'SSM'], 
               color='#A23B72', s=100, marker='s', edgecolors='black', linewidth=2,
               label=f'SSM Peak: {df.loc[ssm_max_idx, "SSM"]:.1f}V at {ssm_max_idx}h')
    
    ax1.set_ylabel('Voltage (V)', fontweight='bold')
    ax1.set_title('Experiment 3: Detailed Timeline Analysis\nCarbon Black Enhancement Performance Validation', 
                  fontweight='bold', fontsize=14)
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # 2. Performance difference over time
    voltage_diff = df['Modified_10pct_CB_SSM'] - df['SSM']
    ax2.plot(df['Hours'], voltage_diff, color='#F18F01', linewidth=2.5)
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.7, label='Equal Performance')
    ax2.fill_between(df['Hours'], voltage_diff, 0, where=(voltage_diff > 0), 
                     color='#F18F01', alpha=0.3, label='Modified Advantage')
    ax2.fill_between(df['Hours'], voltage_diff, 0, where=(voltage_diff < 0), 
                     color='red', alpha=0.3, label='SSM Advantage')
    
    ax2.set_ylabel('Voltage Difference (V)\n[Modified - SSM]', fontweight='bold')
    ax2.set_title('Performance Advantage Analysis', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Performance ratio analysis
    # Calculate ratio more safely
    ratio = np.where((df['SSM'] != 0) & (df['SSM'].notna()), 
                     df['Modified_10pct_CB_SSM'] / df['SSM'], np.nan)
    
    ax3.plot(df['Hours'], ratio, color='#8E44AD', linewidth=2.5)
    ax3.axhline(y=1, color='black', linestyle='--', alpha=0.7, label='Equal Performance (Ratio = 1)')
    ax3.fill_between(df['Hours'], ratio, 1, where=(ratio > 1), 
                     color='green', alpha=0.3, label='Modified Superior')
    ax3.fill_between(df['Hours'], ratio, 1, where=(ratio < 1), 
                     color='red', alpha=0.3, label='SSM Superior')
    
    ax3.set_xlabel('Time (Hours)', fontweight='bold')
    ax3.set_ylabel('Performance Ratio\n[Modified / SSM]', fontweight='bold')
    ax3.set_title('Relative Performance Ratio Analysis', fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('experiment_3_timeline_analysis.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('experiment_3_timeline_analysis.png', dpi=300, bbox_inches='tight')
    print("Timeline analysis plot saved")
    plt.show()

def create_treatment_efficiency_plot(treatment_df):
    """Create treatment efficiency analysis plots"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. COD Removal Efficiency Comparison
    initial_cod = treatment_df.loc['COD', 'Initial ']
    mod_final_cod = treatment_df.loc['COD', 'Mod']
    ssm_final_cod = treatment_df.loc['COD', 'SSM']
    
    mod_removal = ((initial_cod - mod_final_cod) / initial_cod) * 100
    ssm_removal = ((initial_cod - ssm_final_cod) / initial_cod) * 100
    
    electrodes = ['Modified\n(10% CB+SSM)', 'SSM\n(Pristine)']
    cod_removals = [mod_removal, ssm_removal]
    colors = ['#2E86AB', '#A23B72']
    
    bars = ax1.bar(electrodes, cod_removals, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax1.set_ylabel('COD Removal Efficiency (%)', fontweight='bold')
    ax1.set_title('Fish Farm Wastewater COD Removal Comparison', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 70)
    
    # Add value labels on bars
    for bar, removal in zip(bars, cod_removals):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{removal:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # 2. Comprehensive Water Quality Improvement
    parameters = ['pH', 'TDS', 'Conductivity', 'COD']
    initial_values = [treatment_df.loc['pH', 'Initial '], 
                     treatment_df.loc['TDS', 'Initial '], 
                     treatment_df.loc['Conductivity', 'Initial '], 
                     treatment_df.loc['COD', 'Initial ']]
    
    mod_values = [treatment_df.loc['pH', 'Mod'], 
                  treatment_df.loc['TDS', 'Mod'], 
                  treatment_df.loc['Conductivity', 'Mod'], 
                  treatment_df.loc['COD', 'Mod']]
    
    ssm_values = [treatment_df.loc['pH', 'SSM'], 
                  treatment_df.loc['TDS', 'SSM'], 
                  treatment_df.loc['Conductivity', 'SSM'], 
                  treatment_df.loc['COD', 'SSM']]
    
    # Normalize values for comparison (except pH)
    normalized_initial = []
    normalized_mod = []
    normalized_ssm = []
    
    for i, param in enumerate(parameters):
        if param == 'pH':
            # For pH, show actual values
            normalized_initial.append(initial_values[i])
            normalized_mod.append(mod_values[i])
            normalized_ssm.append(ssm_values[i])
        else:
            # For others, normalize to percentage of initial
            normalized_initial.append(100)
            normalized_mod.append((mod_values[i] / initial_values[i]) * 100)
            normalized_ssm.append((ssm_values[i] / initial_values[i]) * 100)
    
    x = np.arange(len(parameters))
    width = 0.25
    
    bars1 = ax2.bar(x - width, normalized_initial, width, label='Initial', 
                    color='gray', alpha=0.7, edgecolor='black')
    bars2 = ax2.bar(x, normalized_mod, width, label='Modified (10% CB+SSM)', 
                    color='#2E86AB', alpha=0.8, edgecolor='black')
    bars3 = ax2.bar(x + width, normalized_ssm, width, label='SSM (Pristine)', 
                    color='#A23B72', alpha=0.8, edgecolor='black')
    
    ax2.set_ylabel('Normalized Values (% of Initial or Actual pH)', fontweight='bold')
    ax2.set_title('Water Quality Parameters Comparison', fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(parameters)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Removal Efficiency Comparison for all parameters
    removal_efficiencies_mod = []
    removal_efficiencies_ssm = []
    removal_params = ['TDS', 'Conductivity', 'COD']
    
    for param in removal_params:
        initial = treatment_df.loc[param, 'Initial ']
        mod_final = treatment_df.loc[param, 'Mod']
        ssm_final = treatment_df.loc[param, 'SSM']
        
        mod_removal = ((initial - mod_final) / initial) * 100
        ssm_removal = ((initial - ssm_final) / initial) * 100
        
        removal_efficiencies_mod.append(mod_removal)
        removal_efficiencies_ssm.append(ssm_removal)
    
    x = np.arange(len(removal_params))
    width = 0.35
    
    bars1 = ax3.bar(x - width/2, removal_efficiencies_mod, width, 
                    label='Modified (10% CB+SSM)', color='#2E86AB', alpha=0.8, edgecolor='black')
    bars2 = ax3.bar(x + width/2, removal_efficiencies_ssm, width, 
                    label='SSM (Pristine)', color='#A23B72', alpha=0.8, edgecolor='black')
    
    ax3.set_ylabel('Removal Efficiency (%)', fontweight='bold')
    ax3.set_title('Treatment Efficiency Comparison by Parameter', fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(removal_params)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{height:.1f}%', ha='center', va='bottom', fontsize=9)
    
    # 4. Summary table
    ax4.axis('tight')
    ax4.axis('off')
    
    # Create summary data
    summary_data = [
        ['Parameter', 'Initial', 'Modified Final', 'SSM Final', 'Mod Removal %', 'SSM Removal %'],
        ['pH', f'{treatment_df.loc["pH", "Initial "]:.1f}', 
         f'{treatment_df.loc["pH", "Mod"]:.1f}', 
         f'{treatment_df.loc["pH", "SSM"]:.1f}', 'N/A', 'N/A'],
        ['TDS (mg/L)', f'{treatment_df.loc["TDS", "Initial "]:,.0f}', 
         f'{treatment_df.loc["TDS", "Mod"]:,.0f}', 
         f'{treatment_df.loc["TDS", "SSM"]:,.0f}', 
         f'{removal_efficiencies_mod[0]:.1f}%', f'{removal_efficiencies_ssm[0]:.1f}%'],
        ['Conductivity (mS/cm)', f'{treatment_df.loc["Conductivity", "Initial "]:.2f}', 
         f'{treatment_df.loc["Conductivity", "Mod"]:.2f}', 
         f'{treatment_df.loc["Conductivity", "SSM"]:.2f}', 
         f'{removal_efficiencies_mod[1]:.1f}%', f'{removal_efficiencies_ssm[1]:.1f}%'],
        ['COD (mg/L)', f'{treatment_df.loc["COD", "Initial "]:,.0f}', 
         f'{treatment_df.loc["COD", "Mod"]:,.0f}', 
         f'{treatment_df.loc["COD", "SSM"]:,.0f}', 
         f'{removal_efficiencies_mod[2]:.1f}%', f'{removal_efficiencies_ssm[2]:.1f}%']
    ]
    
    table = ax4.table(cellText=summary_data[1:], colLabels=summary_data[0],
                     cellLoc='center', loc='center')
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.8)
    
    # Style the table
    for (i, j), cell in table.get_celld().items():
        if i == 0:
            cell.set_text_props(weight='bold')
            cell.set_facecolor('#E8E8E8')
        elif j == 0:
            cell.set_text_props(weight='bold')
            cell.set_facecolor('#F5F5F5')
        else:
            cell.set_facecolor('#FFFFFF')
    
    ax4.set_title('Fish Farm Wastewater Treatment Summary', fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('experiment_3_treatment_efficiency.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('experiment_3_treatment_efficiency.png', dpi=300, bbox_inches='tight')
    print("Treatment efficiency plot saved")
    plt.show()

def print_summary_statistics(df, treatment_df=None):
    """Print comprehensive summary statistics"""
    print("\n" + "="*80)
    print("EXPERIMENT 3 PERFORMANCE SUMMARY")
    print("Fish Farm Wastewater Treatment Study")
    print("="*80)
    
    # Basic statistics
    print(f"\nDuration: {len(df)} hours ({len(df)/24:.1f} days)")
    print(f"Data points: {len(df)}")
    
    print(f"\nMODIFIED ELECTRODE (10% CB+SSM):")
    print(f"  Peak voltage: {df['Modified_10pct_CB_SSM'].max():.2f}V (at hour {df['Modified_10pct_CB_SSM'].idxmax()})")
    print(f"  Average voltage: {df['Modified_10pct_CB_SSM'].mean():.2f}V")
    print(f"  Minimum voltage: {df['Modified_10pct_CB_SSM'].min():.2f}V")
    print(f"  Standard deviation: {df['Modified_10pct_CB_SSM'].std():.2f}V")
    print(f"  Final voltage: {df['Modified_10pct_CB_SSM'].iloc[-1]:.2f}V")
    
    print(f"\nSSM ELECTRODE (PRISTINE):")
    print(f"  Peak voltage: {df['SSM'].max():.2f}V (at hour {df['SSM'].idxmax()})")
    print(f"  Average voltage: {df['SSM'].mean():.2f}V")
    print(f"  Minimum voltage: {df['SSM'].min():.2f}V")
    print(f"  Standard deviation: {df['SSM'].std():.2f}V")
    print(f"  Final voltage: {df['SSM'].iloc[-1]:.2f}V")
    
    # Enhancement calculations
    peak_enhancement = ((df['Modified_10pct_CB_SSM'].max() - df['SSM'].max()) / df['SSM'].max()) * 100
    avg_enhancement = ((df['Modified_10pct_CB_SSM'].mean() - df['SSM'].mean()) / df['SSM'].mean()) * 100
    final_enhancement = ((df['Modified_10pct_CB_SSM'].iloc[-1] - df['SSM'].iloc[-1]) / df['SSM'].iloc[-1]) * 100
    
    print(f"\nENHANCEMENT ANALYSIS:")
    print(f"  Peak performance enhancement: {peak_enhancement:.1f}%")
    print(f"  Average performance enhancement: {avg_enhancement:.1f}%")
    print(f"  Final performance enhancement: {final_enhancement:.1f}%")
    
    # Stability analysis
    mod_peak_retention = (df['Modified_10pct_CB_SSM'].iloc[-1] / df['Modified_10pct_CB_SSM'].max()) * 100
    ssm_peak_retention = (df['SSM'].iloc[-1] / df['SSM'].max()) * 100
    
    print(f"\nSTABILITY ANALYSIS:")
    print(f"  Modified electrode peak retention: {mod_peak_retention:.1f}%")
    print(f"  SSM electrode peak retention: {ssm_peak_retention:.1f}%")
    print(f"  Stability advantage: {mod_peak_retention - ssm_peak_retention:.1f} percentage points")
    
    # Treatment efficiency analysis if treatment data is available
    if treatment_df is not None:
        print(f"\nTREATMENT EFFICIENCY ANALYSIS:")
        
        # COD Analysis
        initial_cod = treatment_df.loc['COD', 'Initial ']
        mod_cod = treatment_df.loc['COD', 'Mod']
        ssm_cod = treatment_df.loc['COD', 'SSM']
        mod_cod_removal = ((initial_cod - mod_cod) / initial_cod) * 100
        ssm_cod_removal = ((initial_cod - ssm_cod) / initial_cod) * 100
        
        print(f"  COD Treatment (Initial: {initial_cod:,.0f} mg/L):")
        print(f"    Modified electrode: {mod_cod:,.0f} mg/L final ({mod_cod_removal:.1f}% removal)")
        print(f"    SSM electrode: {ssm_cod:,.0f} mg/L final ({ssm_cod_removal:.1f}% removal)")
        print(f"    Treatment advantage: {mod_cod_removal - ssm_cod_removal:.1f} percentage points")
        
        # TDS Analysis
        initial_tds = treatment_df.loc['TDS', 'Initial ']
        mod_tds = treatment_df.loc['TDS', 'Mod']
        ssm_tds = treatment_df.loc['TDS', 'SSM']
        mod_tds_removal = ((initial_tds - mod_tds) / initial_tds) * 100
        ssm_tds_removal = ((initial_tds - ssm_tds) / initial_tds) * 100
        
        print(f"  TDS Treatment (Initial: {initial_tds:,.0f} mg/L):")
        print(f"    Modified electrode: {mod_tds:,.0f} mg/L final ({mod_tds_removal:.1f}% removal)")
        print(f"    SSM electrode: {ssm_tds:,.0f} mg/L final ({ssm_tds_removal:.1f}% removal)")
        print(f"    Treatment advantage: {mod_tds_removal - ssm_tds_removal:.1f} percentage points")
        
        # pH Analysis
        initial_ph = treatment_df.loc['pH', 'Initial ']
        mod_ph = treatment_df.loc['pH', 'Mod']
        ssm_ph = treatment_df.loc['pH', 'SSM']
        
        print(f"  pH Neutralization (Initial: {initial_ph:.1f}):")
        print(f"    Modified electrode: {mod_ph:.1f} final")
        print(f"    SSM electrode: {ssm_ph:.1f} final")
        
        # Conductivity Analysis
        initial_cond = treatment_df.loc['Conductivity', 'Initial ']
        mod_cond = treatment_df.loc['Conductivity', 'Mod']
        ssm_cond = treatment_df.loc['Conductivity', 'SSM']
        mod_cond_removal = ((initial_cond - mod_cond) / initial_cond) * 100
        ssm_cond_removal = ((initial_cond - ssm_cond) / initial_cond) * 100
        
        print(f"  Conductivity Reduction (Initial: {initial_cond:.2f} mS/cm):")
        print(f"    Modified electrode: {mod_cond:.2f} mS/cm final ({mod_cond_removal:.1f}% reduction)")
        print(f"    SSM electrode: {ssm_cond:.2f} mS/cm final ({ssm_cond_removal:.1f}% reduction)")
        print(f"    Treatment advantage: {mod_cond_removal - ssm_cond_removal:.1f} percentage points")
    
    # Phase analysis
    phases = [
        ("Initial Phase", 0, 24),
        ("Establishment Phase", 24, 72), 
        ("Growth/Peak Phase", 72, 150),
        ("Stability Phase", 150, len(df))
    ]
    
    print(f"\nELECTROCHEMICAL PHASE ANALYSIS:")
    for phase_name, start, end in phases:
        if end > len(df):
            end = len(df)
        mod_avg = df.iloc[start:end]['Modified_10pct_CB_SSM'].mean()
        ssm_avg = df.iloc[start:end]['SSM'].mean()
        enhancement = ((mod_avg - ssm_avg) / ssm_avg) * 100 if ssm_avg != 0 else 0
        print(f"  {phase_name} ({start}-{end-1}h):")
        print(f"    Modified average: {mod_avg:.1f}V")
        print(f"    SSM average: {ssm_avg:.1f}V") 
        print(f"    Enhancement: {enhancement:.1f}%")
    
    print("\n" + "="*80)

def main():
    """Main execution function"""
    print("MFC Experiment 3 Analysis: Fish Farm Wastewater Treatment Study")
    print("="*70)
    
    # Load voltage data
    df = load_data()
    if df is None:
        print("Failed to load voltage data. Exiting.")
        return
    
    # Load treatment data
    treatment_df = load_treatment_data()
    if treatment_df is None:
        print("Failed to load treatment data. Continuing with voltage analysis only.")
    
    # Print summary statistics
    print_summary_statistics(df, treatment_df)
    
    # Generate all plots
    print("\nGenerating plots...")
    
    try:
        create_voltage_evolution_plot(df)
        create_performance_comparison_plot(df)
        create_statistical_summary_plot(df)
        create_timeline_analysis_plot(df)
        
        # Generate treatment efficiency plot if data is available
        if treatment_df is not None:
            create_treatment_efficiency_plot(treatment_df)
        
        print("\nAll plots generated successfully!")
        print("Files saved:")
        print("- experiment_3_voltage_evolution.pdf/png")
        print("- experiment_3_performance_comparison.pdf/png")  
        print("- experiment_3_statistical_summary.pdf/png")
        print("- experiment_3_timeline_analysis.pdf/png")
        if treatment_df is not None:
            print("- experiment_3_treatment_efficiency.pdf/png")
        
    except Exception as e:
        print(f"Error generating plots: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()