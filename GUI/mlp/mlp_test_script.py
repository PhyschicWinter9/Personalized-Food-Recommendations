# test_mlp_system.py
# Simple test script to debug MLP food recommendation system issues

import sys
import os

def test_dependencies():
    """Test if all required dependencies are installed"""
    print("Testing dependencies...")
    
    required_packages = [
        'tkinter',
        'pandas', 
        'numpy',
        'matplotlib',
        'sklearn',
        'scipy',
        'seaborn'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'tkinter':
                import tkinter
                print(f"✓ {package} - OK")
            elif package == 'pandas':
                import pandas
                print(f"✓ {package} - OK")
            elif package == 'numpy':
                import numpy
                print(f"✓ {package} - OK")
            elif package == 'matplotlib':
                import matplotlib
                print(f"✓ {package} - OK")
            elif package == 'sklearn':
                import sklearn
                print(f"✓ {package} - OK")
            elif package == 'scipy':
                import scipy
                print(f"✓ {package} - OK")
            elif package == 'seaborn':
                import seaborn
                print(f"✓ {package} - OK")
        except ImportError:
            print(f"✗ {package} - MISSING")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\nMissing packages: {', '.join(missing_packages)}")
        print("Install them with: pip install " + " ".join(missing_packages))
        return False
    else:
        print("\nAll dependencies are installed!")
        return True

def test_data_files():
    """Test if CSV data files are available"""
    print("\nTesting data files...")
    
    # Check for datasets folder
    datasets_folder = './datasets'
    current_dir = '.'
    
    csv_files_found = []
    
    # Check datasets folder
    if os.path.exists(datasets_folder):
        csv_files = [f for f in os.listdir(datasets_folder) if f.endswith('.csv')]
        if csv_files:
            print(f"✓ Found {len(csv_files)} CSV files in {datasets_folder}/")
            for csv_file in csv_files:
                print(f"  - {csv_file}")
                csv_files_found.extend(csv_files)
    
    # Check current directory
    csv_files = [f for f in os.listdir(current_dir) if f.endswith('.csv')]
    if csv_files:
        print(f"✓ Found {len(csv_files)} CSV files in current directory")
        for csv_file in csv_files:
            print(f"  - {csv_file}")
            csv_files_found.extend(csv_files)
    
    if not csv_files_found:
        print("✗ No CSV files found!")
        print("Expected files: Drinking.csv, Fruit.csv, meat.csv, etc.")
        print("Make sure CSV files are in ./datasets/ folder or current directory")
        return False
    else:
        print(f"\nFound {len(set(csv_files_found))} unique CSV files")
        return True

def test_tkinter():
    """Test if tkinter GUI can be created"""
    print("\nTesting tkinter GUI...")
    
    try:
        import tkinter as tk
        
        # Create a simple test window
        root = tk.Tk()
        root.title("Test Window")
        root.geometry("300x200")
        
        label = tk.Label(root, text="If you see this, tkinter is working!")
        label.pack(pady=50)
        
        # Show window briefly
        root.update()
        print("✓ tkinter GUI test successful")
        
        # Close after 2 seconds
        root.after(2000, root.destroy)
        root.mainloop()
        
        return True
        
    except Exception as e:
        print(f"✗ tkinter GUI test failed: {e}")
        return False

def test_minimal_mlp():
    """Test minimal MLP functionality"""
    print("\nTesting MLP functionality...")
    
    try:
        from sklearn.neural_network import MLPRegressor
        import numpy as np
        
        # Create sample data
        X = np.random.rand(100, 5)
        y = np.random.rand(100)
        
        # Create and train MLP
        mlp = MLPRegressor(hidden_layer_sizes=(10,), max_iter=100, random_state=42)
        mlp.fit(X, y)
        
        # Make prediction
        prediction = mlp.predict(X[:1])
        
        print("✓ MLP neural network test successful")
        return True
        
    except Exception as e:
        print(f"✗ MLP test failed: {e}")
        return False

def main():
    print("MLP Food Recommendation System - Diagnostic Test")
    print("=" * 50)
    
    all_tests_passed = True
    
    # Run tests
    all_tests_passed &= test_dependencies()
    all_tests_passed &= test_data_files()
    all_tests_passed &= test_tkinter()
    all_tests_passed &= test_minimal_mlp()
    
    print("\n" + "=" * 50)
    if all_tests_passed:
        print("✓ All tests passed! The MLP system should work.")
        print("\nTry running the main script now:")
        print("python food_recommender_system_ncd_v1.py")
    else:
        print("✗ Some tests failed. Please fix the issues above.")
        print("\nCommon solutions:")
        print("1. Install missing packages: pip install pandas numpy matplotlib scikit-learn scipy seaborn")
        print("2. Make sure CSV files are in ./datasets/ folder")
        print("3. Check if you're in the correct directory")

if __name__ == "__main__":
    main()