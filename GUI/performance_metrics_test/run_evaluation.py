#!/usr/bin/env python3
"""
Quick runner script to evaluate Random Forest model performance
"""

import os
import time

def main():
    """Run the model evaluation"""
    
    print("🍎 FOOD RECOMMENDATION SYSTEM - MODEL PERFORMANCE EVALUATION")
    print("=" * 70)
    print()
    
    # Check if the required files exist
    required_files = [
        'food_recommendation_system_using_rf_model_v2.py',
        'datasets'  # datasets folder
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print("❌ Missing required files/folders:")
        for file in missing_files:
            print(f"   - {file}")
        print("\n💡 Please ensure all required files are in the current directory")
        return
    
    # Import and run the evaluation
    try:
        # Import the evaluation system
        from model_performance_evaluator import evaluate_random_forest_system
        
        print("🚀 Starting Random Forest Model Evaluation...")
        print()
        
        start_time = time.time()
        
        # Run the evaluation
        results = evaluate_random_forest_system()
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        if results:
            print(f"\n⏱️  Evaluation completed in {elapsed_time:.2f} seconds")
            
            # Quick summary display
            print("\n" + "="*70)
            print("📋 QUICK PERFORMANCE SUMMARY")
            print("="*70)
            
            if 'Random_Forest' in results:
                rf_data = results['Random_Forest']
                
                # Overall performance metrics
                if 'overall_performance' in rf_data:
                    overall = rf_data['overall_performance']
                    
                    print("\n🔍 REGRESSION METRICS:")
                    print(f"   • Average R² Score: {overall.get('avg_r2_score', 0):.4f}")
                    print(f"   • Average MSE: {overall.get('avg_mse', 0):.4f}")
                    print(f"   • Average RMSE: {overall.get('avg_rmse', 0):.4f}")
                    print(f"   • Average MAE: {overall.get('avg_mae', 0):.4f}")
                    
                    print("\n🎯 CLASSIFICATION METRICS:")
                    print(f"   • Average Accuracy: {overall.get('avg_accuracy', 0):.4f}")
                    print(f"   • Average Precision: {overall.get('avg_precision', 0):.4f}")
                    print(f"   • Average Recall: {overall.get('avg_recall', 0):.4f}")
                    print(f"   • Average F1-Score: {overall.get('avg_f1_score', 0):.4f}")
                    
                    print("\n🏥 HEALTH-AWARE METRICS:")
                    print(f"   • Health Alignment Score: {overall.get('avg_health_alignment_score', 0):.4f}")
                    print(f"   • Condition Specificity: {overall.get('avg_condition_specificity', 0):.4f}")
                    print(f"   • Nutritional Quality: {overall.get('avg_nutritional_diversity', 0):.4f}")
                    print(f"   • Dietary Safety Score: {overall.get('avg_dietary_safety_score', 0):.4f}")
                    print(f"   • Personalization Effectiveness: {overall.get('avg_personalization_effectiveness', 0):.4f}")
                    
                    print("\n🌟 RECOMMENDATION QUALITY:")
                    print(f"   • Diversity Score: {overall.get('avg_diversity_score', 0):.4f}")
                    print(f"   • Novelty Score: {overall.get('avg_novelty_score', 0):.4f}")
                    print(f"   • Coverage Score: {overall.get('avg_coverage_score', 0):.4f}")
                    print(f"   • User Satisfaction Estimate: {overall.get('avg_user_satisfaction_estimate', 0):.4f}")
                    print(f"   • Catalog Coverage: {overall.get('avg_catalog_coverage', 0):.4f}")
                
                # Recommendation system performance
                if 'recommendation_accuracy' in rf_data:
                    rec_acc = rf_data['recommendation_accuracy']
                    if 'error' not in rec_acc:
                        print("\n🍽️  RECOMMENDATION SYSTEM PERFORMANCE:")
                        print(f"   • Total Recommendations Generated: {rec_acc.get('total_recommendations', 0)}")
                        print(f"   • Suitable Recommendations: {rec_acc.get('suitable_recommendations', 0)}")
                        print(f"   • Suitability Rate: {rec_acc.get('suitability_rate', 0):.4f}")
                        print(f"   • Condition Match Rate: {rec_acc.get('condition_match_rate', 0):.4f}")
                        print(f"   • Average Recommendation Score: {rec_acc.get('avg_recommendation_score', 0):.4f}")
                        print(f"   • Top 5 Average Score: {rec_acc.get('top_5_avg_score', 0):.4f}")
                
                # Top features
                if 'feature_importance' in rf_data:
                    print("\n⭐ TOP 5 MOST IMPORTANT FEATURES:")
                    for i, (feature, importance) in enumerate(list(rf_data['feature_importance'].items())[:5]):
                        print(f"   {i+1}. {feature}: {importance:.4f}")
                
                # Individual model performance
                if 'individual_models' in rf_data:
                    print("\n📊 INDIVIDUAL MODEL PERFORMANCE:")
                    for model_name, metrics in rf_data['individual_models'].items():
                        print(f"\n   {model_name.replace('_', ' ')}:")
                        print(f"      R² Score: {metrics.get('r2_score', 0):.4f}")
                        print(f"      Accuracy: {metrics.get('accuracy', 0):.4f}")
                        print(f"      F1-Score: {metrics.get('f1_score', 0):.4f}")
            
            print("\n📄 Files Generated:")
            print("   • random_forest_performance_report.txt - Detailed performance report")
            print("   • random_forest_performance_plots.png - Performance visualization plots")
            
            print("\n🔮 Framework Features:")
            print("   • ✅ Random Forest evaluation complete")
            print("   • 🔄 Ready to add K-NN evaluation")
            print("   • 🔄 Ready to add XGBoost evaluation")
            print("   • 🔄 Ready for side-by-side model comparison")
            
            print("\n✅ Evaluation completed successfully!")
            
        else:
            print("❌ Evaluation failed!")
            
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Make sure all required Python packages are installed:")
        print("   pip install pandas numpy matplotlib seaborn scikit-learn")
        
    except Exception as e:
        print(f"❌ Unexpected error: {e}")


if __name__ == "__main__":
    main()
