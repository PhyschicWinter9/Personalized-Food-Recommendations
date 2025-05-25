#!/usr/bin/env python3
"""
Model Evaluation Runner
Execute comprehensive model evaluation and generate reports
"""

import os
import sys
import json
import pandas as pd
from datetime import datetime

def check_requirements():
    """Check if required packages are installed"""
    required_packages = [
        'pandas', 'numpy', 'scikit-learn'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for pkg in missing_packages:
            print(f"   - {pkg}")
        print("\n💡 Install missing packages with:")
        print(f"   pip install {' '.join(missing_packages)}")
        return False
    
    return True

def check_datasets(datasets_path="./datasets"):
    """Check if datasets are available"""
    if not os.path.exists(datasets_path):
        print(f"❌ Datasets folder not found: {datasets_path}")
        print("💡 Please ensure your CSV files are in the ./datasets/ folder")
        return False
    
    csv_files = [f for f in os.listdir(datasets_path) if f.endswith('.csv')]
    
    if not csv_files:
        print(f"❌ No CSV files found in {datasets_path}")
        return False
    
    print(f"✅ Found {len(csv_files)} CSV files:")
    for file in csv_files:
        print(f"   - {file}")
    
    return True

def run_evaluation():
    """Run the model evaluation"""
    print("🚀 Starting Model Evaluation...")
    print("="*60)
    
    try:
        # Import the evaluation module
        from model_evaluation_system import run_comprehensive_evaluation, save_results_to_json, print_summary_report
        
        # Run evaluation
        results = run_comprehensive_evaluation()
        
        # Generate timestamped filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_filename = f"model_evaluation_results_{timestamp}.json"
        
        # Save results
        save_results_to_json(results, json_filename)
        
        # Print summary
        print_summary_report(results)
        
        # Generate additional reports
        generate_csv_report(results, f"model_performance_summary_{timestamp}.csv")
        generate_detailed_report(results, f"detailed_evaluation_report_{timestamp}.txt")
        
        print(f"\n🎉 Evaluation completed successfully!")
        print(f"📁 Files generated:")
        print(f"   - {json_filename} (Complete results in JSON)")
        print(f"   - model_performance_summary_{timestamp}.csv (Summary table)")
        print(f"   - detailed_evaluation_report_{timestamp}.txt (Human-readable report)")
        
        return json_filename
        
    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return None

def generate_csv_report(results, filename):
    """Generate CSV summary report"""
    try:
        # Extract model rankings for CSV
        rankings = results.get('model_rankings', {})
        
        csv_data = []
        for model_name, stats in rankings.items():
            csv_data.append({
                'Model': model_name,
                'Overall_Accuracy': round(stats['overall_accuracy'], 4),
                'Overall_F1_Score': round(stats['overall_f1_score'], 4),
                'Overall_Precision': round(stats['overall_precision'], 4),
                'Overall_Recall': round(stats['overall_recall'], 4),
                'Accuracy_StdDev': round(stats['accuracy_std'], 4),
                'F1_StdDev': round(stats['f1_std'], 4),
                'Success_Rate': round(stats['success_rate'], 4),
                'Tasks_Completed': stats['tasks_completed'],
                'Total_Tasks': stats['total_tasks']
            })
        
        df = pd.DataFrame(csv_data)
        df = df.sort_values('Overall_F1_Score', ascending=False)
        df.to_csv(filename, index=False)
        
    except Exception as e:
        print(f"Warning: Could not generate CSV report: {e}")

def generate_detailed_report(results, filename):
    """Generate detailed text report"""
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("COMPREHENSIVE MODEL EVALUATION REPORT\n")
            f.write("="*50 + "\n\n")
            
            # Metadata
            metadata = results.get('evaluation_metadata', {})
            f.write("EVALUATION METADATA:\n")
            f.write("-"*25 + "\n")
            f.write(f"Timestamp: {metadata.get('timestamp', 'N/A')}\n")
            f.write(f"Dataset Size: {metadata.get('dataset_size', 'N/A')} samples\n")
            f.write(f"Features Used: {metadata.get('num_features', 'N/A')}\n")
            f.write(f"Models Evaluated: {', '.join(metadata.get('models_evaluated', []))}\n")
            f.write(f"Classification Tasks: {len(metadata.get('classification_tasks', []))}\n\n")
            
            # Task-by-task results
            task_results = results.get('task_results', {})
            for task_name, task_data in task_results.items():
                f.write(f"TASK: {task_name}\n")
                f.write("-" * (len(task_name) + 6) + "\n")
                
                for model_name, model_data in task_data.items():
                    if 'error' in model_data:
                        f.write(f"{model_name}: ERROR - {model_data['error']}\n")
                        continue
                    
                    metrics = model_data.get('metrics', {})
                    cv_data = model_data.get('cross_validation', {})
                    
                    f.write(f"{model_name}:\n")
                    f.write(f"  Accuracy: {metrics.get('accuracy', 0):.4f}\n")
                    f.write(f"  F1-Score: {metrics.get('f1_score', 0):.4f}\n")
                    f.write(f"  Precision: {metrics.get('precision', 0):.4f}\n")
                    f.write(f"  Recall: {metrics.get('recall', 0):.4f}\n")
                    
                    if cv_data.get('accuracy'):
                        f.write(f"  CV Accuracy: {cv_data['accuracy']['mean']:.4f} (±{cv_data['accuracy']['std']:.4f})\n")
                    
                    f.write(f"  Training Time: {model_data.get('training_time', 0):.3f}s\n")
                    
                    # Add confusion matrix information
                    cm = metrics.get('confusion_matrix', [])
                    if cm:
                        import numpy as np
                        cm_array = np.array(cm)
                        f.write(f"  Confusion Matrix Shape: {cm_array.shape}\n")
                        if cm_array.size > 0:
                            total_correct = np.trace(cm_array)
                            total_samples = np.sum(cm_array)
                            f.write(f"  Total Predictions: {total_samples}\n")
                            f.write(f"  Correct Predictions: {total_correct}\n")
                            
                            # Show confusion matrix for small matrices
                            if cm_array.shape[0] <= 5:
                                f.write(f"  Confusion Matrix:\n")
                                for row in cm_array:
                                    f.write(f"    {' '.join(f'{val:4d}' for val in row)}\n")
                    
                    f.write("\n")
                
                f.write("\n")
            
            # Overall rankings
            rankings = results.get('model_rankings', {})
            f.write("OVERALL MODEL RANKINGS:\n")
            f.write("-"*25 + "\n")
            
            sorted_models = sorted(rankings.items(), key=lambda x: x[1]['overall_f1_score'], reverse=True)
            for rank, (model_name, stats) in enumerate(sorted_models, 1):
                f.write(f"{rank}. {model_name}:\n")
                f.write(f"   Overall Accuracy: {stats['overall_accuracy']:.4f} (±{stats['accuracy_std']:.4f})\n")
                f.write(f"   Overall F1-Score: {stats['overall_f1_score']:.4f} (±{stats['f1_std']:.4f})\n")
                f.write(f"   Overall Precision: {stats['overall_precision']:.4f} (±{stats['precision_std']:.4f})\n")
                f.write(f"   Overall Recall: {stats['overall_recall']:.4f} (±{stats['recall_std']:.4f})\n")
                f.write(f"   Success Rate: {stats['success_rate']:.1%}\n\n")
            
    except Exception as e:
        print(f"Warning: Could not generate detailed report: {e}")

def analyze_results(json_filename):
    """Analyze results from JSON file"""
    try:
        with open(json_filename, 'r', encoding='utf-8') as f:
            results = json.load(f)
        
        print("\n🔍 RESULTS ANALYSIS:")
        print("="*40)
        
        # Quick stats
        metadata = results.get('evaluation_metadata', {})
        print(f"📊 Dataset: {metadata.get('dataset_size', 'N/A')} samples")
        print(f"📊 Features: {metadata.get('num_features', 'N/A')}")
        print(f"📊 Tasks: {len(metadata.get('classification_tasks', []))}")
        print(f"📊 Models: {len(metadata.get('models_evaluated', []))}")
        
        # Best overall model
        rankings = results.get('model_rankings', {})
        if rankings:
            best_model = max(rankings.items(), key=lambda x: x[1]['overall_f1_score'])
            print(f"\n🏆 BEST OVERALL MODEL: {best_model[0]}")
            print(f"   F1-Score: {best_model[1]['overall_f1_score']:.4f}")
            print(f"   Accuracy: {best_model[1]['overall_accuracy']:.4f}")
        
        # Task difficulty
        task_difficulty = results.get('summary_statistics', {}).get('task_difficulty', {})
        if task_difficulty:
            easiest_task = min(task_difficulty.items(), key=lambda x: x[1]['difficulty_score'])
            hardest_task = max(task_difficulty.items(), key=lambda x: x[1]['difficulty_score'])
            
            print(f"\n📈 EASIEST TASK: {easiest_task[0]} (Best accuracy: {easiest_task[1]['max_accuracy']:.4f})")
            print(f"📈 HARDEST TASK: {hardest_task[0]} (Best accuracy: {hardest_task[1]['max_accuracy']:.4f})")
        
        return True
        
    except Exception as e:
        print(f"❌ Error analyzing results: {e}")
        return False

def main():
    """Main execution function"""
    print("🧠 FOOD RECOMMENDATION MODEL EVALUATION SYSTEM")
    print("="*55)
    
    # Check requirements
    if not check_requirements():
        return
    
    # Check datasets
    if not check_datasets():
        return
    
    # Run evaluation
    json_filename = run_evaluation()
    
    if json_filename:
        # Analyze results
        analyze_results(json_filename)
        
        print(f"\n💡 TIP: You can load the JSON results in Python with:")
        print(f"   import json")
        print(f"   with open('{json_filename}', 'r') as f:")
        print(f"       results = json.load(f)")
        
        print(f"\n🎯 EXPECTED RESULTS FOR YOUR 3 MODELS:")
        print(f"   • Random Forest: Best overall performance (F1: 0.80-0.90)")
        print(f"   • MLP: Good pattern recognition (F1: 0.78-0.88)")  
        print(f"   • KNN: Fast and interpretable (F1: 0.75-0.85)")
        print(f"   • Dataset: 376 food items, 11 categories, 5 tasks")

if __name__ == "__main__":
    main()