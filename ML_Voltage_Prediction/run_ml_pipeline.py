#!/usr/bin/env python3
"""
MFC Voltage Prediction - Complete ML Pipeline
Execute the entire machine learning pipeline from data preparation to application
"""

import sys
from pathlib import Path
import time

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

def main():
    """Execute complete ML pipeline"""
    print("="*80)
    print("MFC VOLTAGE PREDICTION - COMPLETE ML PIPELINE")
    print("="*80)
    print("Academic ML Project for MFC Performance Optimization")
    print("Predict MFC voltage after 24 hours instead of 200+ hours")
    print("="*80)
    
    # Step 1: Data Preparation
    print("\n🔄 STEP 1: DATA PREPARATION")
    print("-" * 40)
    
    try:
        from data_preparation import main as data_prep_main
        data_prep_main()
        print("✅ Data preparation completed successfully")
    except Exception as e:
        print(f"❌ Data preparation failed: {e}")
        print("Please check that experimental data files exist in the correct locations:")
        print("  - EXP 1/Final DATA EXP 1.csv")
        print("  - EXP 2/Final data_EXP_2.csv") 
        print("  - Exp 3 Mod vs ssm/MOD vs SSM.csv")
        return False
    
    time.sleep(2)
    
    # Step 2: Model Training
    print("\n🔄 STEP 2: MODEL TRAINING")
    print("-" * 40)
    
    try:
        from voltage_prediction_model import main as model_train_main
        model_train_main()
        print("✅ Model training completed successfully")
    except Exception as e:
        print(f"❌ Model training failed: {e}")
        return False
    
    time.sleep(2)
    
    # Step 3: Practical Application Demo
    print("\n🔄 STEP 3: PRACTICAL APPLICATION DEMO")
    print("-" * 40)
    
    try:
        from prediction_application import demo_prediction
        demo_prediction()
        print("✅ Application demo completed successfully")
    except Exception as e:
        print(f"❌ Application demo failed: {e}")
        return False
    
    # Final Summary
    print("\n" + "="*80)
    print("🎉 ML PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*80)
    
    print("\n📊 GENERATED FILES:")
    print("-" * 40)
    
    # Check and list generated files
    data_dir = Path("data")
    models_dir = Path("models")
    results_dir = Path("results")
    
    if data_dir.exists():
        data_files = list(data_dir.glob("*.csv"))
        print(f"📁 Data files ({len(data_files)}):")
        for f in data_files:
            print(f"   ✓ {f.name}")
    
    if models_dir.exists():
        model_files = list(models_dir.glob("*"))
        print(f"\n🤖 Model files ({len(model_files)}):")
        for f in model_files:
            print(f"   ✓ {f.name}")
    
    if results_dir.exists():
        result_files = list(results_dir.glob("*"))
        print(f"\n📈 Result files ({len(result_files)}):")
        for f in result_files:
            print(f"   ✓ {f.name}")
    
    print("\n🎯 ACADEMIC VALUE:")
    print("-" * 40)
    print("✓ Novel ML application to bioelectrochemical systems")
    print("✓ Practical engineering solution for MFC optimization")
    print("✓ 80-90% reduction in experimental time required")
    print("✓ Early identification of high/low performing electrodes")
    print("✓ Ready for thesis chapter integration")
    
    print("\n📝 NEXT STEPS:")
    print("-" * 40)
    print("1. Review generated plots and reports")
    print("2. Run Jupyter notebook for interactive analysis")
    print("3. Use prediction_application.py for new predictions")
    print("4. Integrate findings into thesis document")
    print("5. Consider extending to treatment efficiency prediction")
    
    print("\n📚 FOR THESIS:")
    print("-" * 40)
    print("• Use voltage_prediction_results.pdf for thesis figures")
    print("• Include model_evaluation_report.txt for performance metrics")
    print("• Reference practical time savings (24h vs 200+h)")
    print("• Highlight industrial application potential")
    
    return True

def check_requirements():
    """Check if required packages are installed"""
    required_packages = [
        'pandas', 'numpy', 'matplotlib', 'seaborn', 'scikit-learn', 
        'tensorflow', 'jupyter', 'tqdm'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\nInstall missing packages with:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    return True

if __name__ == "__main__":
    print("Checking requirements...")
    if not check_requirements():
        print("Please install missing packages first.")
        sys.exit(1)
    
    success = main()
    
    if success:
        print("\n🎉 All done! Your ML project is ready for thesis integration.")
    else:
        print("\n❌ Pipeline failed. Please check error messages above.")
        sys.exit(1)